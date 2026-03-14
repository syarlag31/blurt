"""Audio chunker — splits raw audio streams into processable chunks.

Handles:
- Fixed-size chunking with configurable overlap for continuity
- Silence detection to split on natural boundaries
- Streaming and batch modes
- Multiple audio format inputs
"""

from __future__ import annotations

import logging
import math
import struct
from collections.abc import AsyncIterator
from typing import Optional

from blurt.audio.models import AudioChunk, AudioEncoding, AudioFormat

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_DURATION_MS = 5000  # 5 seconds
DEFAULT_OVERLAP_MS = 500  # 500ms overlap between chunks
DEFAULT_SILENCE_THRESHOLD_DB = -40.0  # dBFS
DEFAULT_SILENCE_MIN_MS = 800  # Minimum silence duration to split on
MAX_AUDIO_DURATION_S = 300  # 5 minute max per blurt


class AudioChunker:
    """Splits raw audio data into chunks suitable for Gemini API processing.

    Supports two chunking strategies:
    1. Fixed-size: Chunks of uniform duration with overlap for continuity
    2. Silence-aware: Splits on detected silence boundaries (preferred)

    The chunker is stateless for batch mode and stateful for streaming.
    """

    def __init__(
        self,
        audio_format: AudioFormat | None = None,
        chunk_duration_ms: int = DEFAULT_CHUNK_DURATION_MS,
        overlap_ms: int = DEFAULT_OVERLAP_MS,
        silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
        silence_min_ms: int = DEFAULT_SILENCE_MIN_MS,
        use_silence_detection: bool = True,
    ) -> None:
        self.format = audio_format or AudioFormat()
        self.chunk_duration_ms = chunk_duration_ms
        self.overlap_ms = overlap_ms
        self.silence_threshold_db = silence_threshold_db
        self.silence_min_ms = silence_min_ms
        self.use_silence_detection = use_silence_detection

        # Derived byte sizes
        self._bytes_per_ms = self.format.bytes_per_second / 1000.0
        self._chunk_size = int(self._bytes_per_ms * chunk_duration_ms)
        self._overlap_size = int(self._bytes_per_ms * overlap_ms)
        self._silence_min_bytes = int(self._bytes_per_ms * silence_min_ms)
        self._max_bytes = int(self.format.bytes_per_second * MAX_AUDIO_DURATION_S)

        # Streaming state
        self._buffer = bytearray()
        self._sequence = 0
        self._total_ms = 0

    def chunk_bytes(self, audio_data: bytes) -> list[AudioChunk]:
        """Split a complete audio byte buffer into chunks.

        Args:
            audio_data: Complete raw audio bytes.

        Returns:
            List of AudioChunk objects in order.

        Raises:
            ValueError: If audio data exceeds maximum duration.
        """
        if len(audio_data) > self._max_bytes:
            raise ValueError(
                f"Audio data ({len(audio_data)} bytes) exceeds maximum "
                f"duration of {MAX_AUDIO_DURATION_S}s "
                f"({self._max_bytes} bytes)"
            )

        if not audio_data:
            return []

        if self.use_silence_detection and self.format.encoding == AudioEncoding.LINEAR16:
            return self._chunk_with_silence_detection(audio_data)
        return self._chunk_fixed_size(audio_data)

    def _chunk_fixed_size(self, audio_data: bytes) -> list[AudioChunk]:
        """Split audio into fixed-size chunks with overlap."""
        chunks: list[AudioChunk] = []
        data_len = len(audio_data)
        step = self._chunk_size - self._overlap_size
        offset = 0
        seq = 0

        while offset < data_len:
            end = min(offset + self._chunk_size, data_len)
            chunk_data = audio_data[offset:end]
            chunk_duration_ms = int(len(chunk_data) / self._bytes_per_ms)
            timestamp_ms = int(offset / self._bytes_per_ms)
            is_final = end >= data_len

            chunks.append(
                AudioChunk(
                    data=bytes(chunk_data),
                    sequence_number=seq,
                    timestamp_ms=timestamp_ms,
                    duration_ms=chunk_duration_ms,
                    is_final=is_final,
                    format=self.format,
                )
            )
            seq += 1
            offset += step

            # Prevent infinite loop on very small audio
            if step <= 0:
                break

        return chunks

    def _chunk_with_silence_detection(self, audio_data: bytes) -> list[AudioChunk]:
        """Split audio on silence boundaries, falling back to fixed-size.

        Finds silence regions in the audio and splits at those points.
        If no silence is found within a chunk window, falls back to
        fixed-size splitting.
        """
        silence_points = self._find_silence_boundaries(audio_data)

        if not silence_points:
            return self._chunk_fixed_size(audio_data)

        chunks: list[AudioChunk] = []
        seq = 0
        start = 0

        for silence_start, silence_end in silence_points:
            # Split at the midpoint of silence
            split_point = (silence_start + silence_end) // 2
            # Align to sample boundary
            sample_size = self.format.sample_width_bytes * self.format.channels
            split_point = (split_point // sample_size) * sample_size

            if split_point <= start:
                continue

            chunk_data = audio_data[start:split_point]
            if not chunk_data:
                continue

            chunk_duration_ms = int(len(chunk_data) / self._bytes_per_ms)
            timestamp_ms = int(start / self._bytes_per_ms)

            chunks.append(
                AudioChunk(
                    data=bytes(chunk_data),
                    sequence_number=seq,
                    timestamp_ms=timestamp_ms,
                    duration_ms=chunk_duration_ms,
                    is_final=False,
                    format=self.format,
                )
            )
            seq += 1
            start = split_point

        # Final chunk: remaining audio after last silence
        if start < len(audio_data):
            remaining = audio_data[start:]
            chunk_duration_ms = int(len(remaining) / self._bytes_per_ms)
            timestamp_ms = int(start / self._bytes_per_ms)

            chunks.append(
                AudioChunk(
                    data=bytes(remaining),
                    sequence_number=seq,
                    timestamp_ms=timestamp_ms,
                    duration_ms=chunk_duration_ms,
                    is_final=True,
                    format=self.format,
                )
            )
        elif chunks:
            # Mark last chunk as final
            last = chunks[-1]
            chunks[-1] = last.model_copy(update={"is_final": True})

        return chunks

    def _find_silence_boundaries(
        self, audio_data: bytes
    ) -> list[tuple[int, int]]:
        """Detect silence regions in LINEAR16 audio.

        Returns list of (start_byte, end_byte) tuples for silence regions
        that are at least silence_min_ms long.
        """
        sample_size = self.format.sample_width_bytes
        threshold_amplitude = self._db_to_amplitude(self.silence_threshold_db)

        # Use a sliding window to compute RMS energy
        window_samples = int(self.format.sample_rate_hz * 0.02)  # 20ms window
        window_bytes = window_samples * sample_size * self.format.channels
        step_bytes = window_bytes  # Non-overlapping windows

        silence_regions: list[tuple[int, int]] = []
        in_silence = False
        silence_start = 0

        offset = 0
        while offset + window_bytes <= len(audio_data):
            window = audio_data[offset : offset + window_bytes]
            rms = self._compute_rms(window, sample_size)

            if rms < threshold_amplitude:
                if not in_silence:
                    in_silence = True
                    silence_start = offset
            else:
                if in_silence:
                    silence_end = offset
                    silence_duration_bytes = silence_end - silence_start
                    if silence_duration_bytes >= self._silence_min_bytes:
                        silence_regions.append((silence_start, silence_end))
                    in_silence = False

            offset += step_bytes

        # Handle trailing silence
        if in_silence:
            silence_end = len(audio_data)
            silence_duration_bytes = silence_end - silence_start
            if silence_duration_bytes >= self._silence_min_bytes:
                silence_regions.append((silence_start, silence_end))

        return silence_regions

    @staticmethod
    def _compute_rms(window: bytes, sample_width: int) -> float:
        """Compute RMS amplitude of a window of LINEAR16 samples."""
        if not window:
            return 0.0

        if sample_width != 2:
            return 0.0

        # Unpack 16-bit signed samples
        num_samples = len(window) // 2
        if num_samples == 0:
            return 0.0

        try:
            samples = struct.unpack(f"<{num_samples}h", window[:num_samples * 2])
        except struct.error:
            return 0.0

        # RMS calculation
        sum_squares = sum(s * s for s in samples)
        rms = math.sqrt(sum_squares / num_samples)
        return rms

    @staticmethod
    def _db_to_amplitude(db: float) -> float:
        """Convert dBFS to amplitude for 16-bit audio."""
        # 0 dBFS = 32767 (max 16-bit signed)
        max_amplitude = 32767.0
        return max_amplitude * (10.0 ** (db / 20.0))

    # --- Streaming API ---

    def feed(self, data: bytes) -> list[AudioChunk]:
        """Feed streaming audio data and get back any complete chunks.

        Call this repeatedly with incoming audio data. Returns chunks
        whenever enough data has accumulated. Call flush() when the
        stream ends to get the final chunk.
        """
        self._buffer.extend(data)
        chunks: list[AudioChunk] = []

        while len(self._buffer) >= self._chunk_size:
            chunk_data = bytes(self._buffer[: self._chunk_size])
            # Keep overlap in buffer
            self._buffer = self._buffer[self._chunk_size - self._overlap_size :]

            timestamp_ms = self._total_ms
            chunk_duration = int(len(chunk_data) / self._bytes_per_ms)

            chunks.append(
                AudioChunk(
                    data=chunk_data,
                    sequence_number=self._sequence,
                    timestamp_ms=timestamp_ms,
                    duration_ms=chunk_duration,
                    is_final=False,
                    format=self.format,
                )
            )
            self._sequence += 1
            self._total_ms += chunk_duration - self.overlap_ms

        return chunks

    def flush(self) -> Optional[AudioChunk]:
        """Flush remaining buffered audio as the final chunk.

        Call this when the audio stream ends. Returns None if
        the buffer is empty.
        """
        if not self._buffer:
            return None

        chunk_data = bytes(self._buffer)
        chunk_duration = int(len(chunk_data) / self._bytes_per_ms)
        timestamp_ms = self._total_ms

        chunk = AudioChunk(
            data=chunk_data,
            sequence_number=self._sequence,
            timestamp_ms=timestamp_ms,
            duration_ms=chunk_duration,
            is_final=True,
            format=self.format,
        )

        # Reset state
        self._buffer.clear()
        self._sequence += 1
        self._total_ms += chunk_duration

        return chunk

    def reset(self) -> None:
        """Reset streaming state for a new stream."""
        self._buffer.clear()
        self._sequence = 0
        self._total_ms = 0

    async def chunk_stream(
        self, stream: AsyncIterator[bytes]
    ) -> AsyncIterator[AudioChunk]:
        """Process an async stream of audio bytes into chunks.

        Yields AudioChunk objects as they become available.
        """
        self.reset()

        async for data in stream:
            chunks = self.feed(data)
            for chunk in chunks:
                yield chunk

        # Yield final chunk
        final = self.flush()
        if final:
            yield final
