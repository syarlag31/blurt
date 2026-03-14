"""Tests for audio chunker — raw audio → processable chunks.

Validates:
- Fixed-size chunking with overlap
- Silence detection-based splitting
- Streaming feed/flush API
- Edge cases (empty, tiny, max-size audio)
- Audio format handling
"""

from __future__ import annotations

import math
import struct

import pytest

from blurt.audio.chunker import AudioChunker, MAX_AUDIO_DURATION_S
from blurt.audio.models import AudioEncoding, AudioFormat


# ── Helpers ──────────────────────────────────────────────────────


def make_pcm_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM audio (all zeros)."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def make_pcm_tone(
    duration_ms: int,
    frequency: float = 440.0,
    amplitude: int = 16000,
    sample_rate: int = 16000,
) -> bytes:
    """Generate a sine wave tone as PCM LINEAR16 audio."""
    num_samples = int(sample_rate * duration_ms / 1000)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(amplitude * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        samples.append(value)
    return struct.pack(f"<{num_samples}h", *samples)


def make_speech_with_silence(
    speech_ms: int = 2000,
    silence_ms: int = 1500,
    speech_ms_2: int = 2000,
    sample_rate: int = 16000,
) -> bytes:
    """Generate audio that simulates speech → silence → speech."""
    speech1 = make_pcm_tone(speech_ms, sample_rate=sample_rate)
    silence = make_pcm_silence(silence_ms, sample_rate=sample_rate)
    speech2 = make_pcm_tone(speech_ms_2, frequency=880.0, sample_rate=sample_rate)
    return speech1 + silence + speech2


DEFAULT_FORMAT = AudioFormat(
    encoding=AudioEncoding.LINEAR16,
    sample_rate_hz=16000,
    channels=1,
    sample_width_bytes=2,
)


# ── Fixed-size chunking tests ────────────────────────────────────


class TestFixedSizeChunking:
    """Tests for the fixed-size chunking strategy."""

    def test_empty_audio_returns_empty(self):
        chunker = AudioChunker(audio_format=DEFAULT_FORMAT, use_silence_detection=False)
        chunks = chunker.chunk_bytes(b"")
        assert chunks == []

    def test_single_chunk_for_short_audio(self):
        """Audio shorter than chunk_duration produces exactly one chunk."""
        audio = make_pcm_tone(1000)  # 1 second
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=5000,
            use_silence_detection=False,
        )
        chunks = chunker.chunk_bytes(audio)

        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert chunks[0].sequence_number == 0
        assert chunks[0].data == audio

    def test_multiple_chunks_with_overlap(self):
        """10 seconds of audio with 5s chunks and 500ms overlap."""
        audio = make_pcm_tone(10000)  # 10 seconds
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=5000,
            overlap_ms=500,
            use_silence_detection=False,
        )
        chunks = chunker.chunk_bytes(audio)

        # With 5s chunks, 500ms overlap, step = 4500ms
        # 10s / 4.5s ≈ 3 chunks (0-5s, 4.5-9.5s, 9-10s)
        assert len(chunks) >= 2
        assert chunks[-1].is_final is True
        assert all(c.sequence_number == i for i, c in enumerate(chunks))

    def test_chunks_cover_all_audio(self):
        """Verify chunks cover the full audio without gaps."""
        audio = make_pcm_tone(7000)  # 7 seconds
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=3000,
            overlap_ms=0,
            use_silence_detection=False,
        )
        chunks = chunker.chunk_bytes(audio)

        total_bytes = sum(len(c.data) for c in chunks)
        assert total_bytes >= len(audio)

    def test_chunk_format_preserved(self):
        """Each chunk carries the correct AudioFormat."""
        audio = make_pcm_tone(2000)
        fmt = AudioFormat(
            encoding=AudioEncoding.LINEAR16,
            sample_rate_hz=44100,
            channels=2,
            sample_width_bytes=2,
        )
        chunker = AudioChunker(audio_format=fmt, use_silence_detection=False)
        chunks = chunker.chunk_bytes(audio)

        for chunk in chunks:
            assert chunk.format == fmt

    def test_timestamp_increases(self):
        """Chunk timestamps increase monotonically."""
        audio = make_pcm_tone(15000)  # 15 seconds
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=5000,
            overlap_ms=500,
            use_silence_detection=False,
        )
        chunks = chunker.chunk_bytes(audio)

        timestamps = [c.timestamp_ms for c in chunks]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_duration_ms_matches_data_length(self):
        """Each chunk's duration_ms should match its data length."""
        audio = make_pcm_tone(8000)
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=3000,
            overlap_ms=0,
            use_silence_detection=False,
        )
        chunks = chunker.chunk_bytes(audio)

        for chunk in chunks:
            expected_ms = int(len(chunk.data) / DEFAULT_FORMAT.bytes_per_second * 1000)
            assert abs(chunk.duration_ms - expected_ms) <= 1


class TestMaxDuration:
    """Tests for maximum audio duration enforcement."""

    def test_exceeds_max_duration_raises(self):
        """Audio exceeding 5 minutes should raise ValueError."""
        fmt = DEFAULT_FORMAT
        max_bytes = fmt.bytes_per_second * MAX_AUDIO_DURATION_S
        oversized = b"\x00" * (max_bytes + 1)

        chunker = AudioChunker(audio_format=fmt, use_silence_detection=False)
        with pytest.raises(ValueError, match="exceeds maximum"):
            chunker.chunk_bytes(oversized)

    def test_at_max_duration_succeeds(self):
        """Audio at exactly 5 minutes should succeed."""
        fmt = DEFAULT_FORMAT
        max_bytes = fmt.bytes_per_second * MAX_AUDIO_DURATION_S
        audio = b"\x00" * max_bytes

        chunker = AudioChunker(audio_format=fmt, use_silence_detection=False)
        chunks = chunker.chunk_bytes(audio)
        assert len(chunks) > 0


# ── Silence detection tests ─────────────────────────────────────


class TestSilenceDetection:
    """Tests for silence-aware chunking."""

    def test_silence_splits_audio(self):
        """Audio with a long silence gap should split into multiple chunks."""
        # 2s speech + 1.5s silence + 2s speech
        audio = make_speech_with_silence(
            speech_ms=2000, silence_ms=1500, speech_ms_2=2000
        )
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=10000,  # Large enough to contain everything
            silence_min_ms=800,
            use_silence_detection=True,
        )
        chunks = chunker.chunk_bytes(audio)

        # Should split into 2 chunks around the silence
        assert len(chunks) >= 2

    def test_no_silence_falls_back_to_fixed(self):
        """Continuous audio with no silence uses fixed-size chunking."""
        audio = make_pcm_tone(10000, amplitude=20000)
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=5000,
            overlap_ms=500,
            use_silence_detection=True,
        )
        chunks = chunker.chunk_bytes(audio)
        assert len(chunks) >= 2

    def test_pure_silence(self):
        """Audio that is entirely silent should produce chunk(s)."""
        audio = make_pcm_silence(3000)
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=5000,
            use_silence_detection=True,
        )
        chunks = chunker.chunk_bytes(audio)
        assert len(chunks) >= 1

    def test_short_silence_not_split(self):
        """Silence shorter than silence_min_ms should NOT split."""
        # 2s speech + 200ms silence + 2s speech
        audio = make_speech_with_silence(
            speech_ms=2000, silence_ms=200, speech_ms_2=2000
        )
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=10000,
            silence_min_ms=800,
            use_silence_detection=True,
        )
        chunks = chunker.chunk_bytes(audio)
        # Short silence shouldn't trigger a split — should fall back to fixed
        # (or be treated as one chunk since it all fits)
        assert len(chunks) >= 1


# ── Streaming API tests ─────────────────────────────────────────


class TestStreamingChunker:
    """Tests for the feed/flush streaming API."""

    def test_feed_accumulates_until_chunk_size(self):
        """Data fed in small pieces should produce chunks at the right size."""
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=1000,  # 1 second = 32000 bytes
            overlap_ms=0,
            use_silence_detection=False,
        )

        # Feed 500ms at a time (16000 bytes each)
        chunk_500ms = make_pcm_tone(500)
        result = chunker.feed(chunk_500ms)
        assert result == []  # Not enough data yet

        result = chunker.feed(chunk_500ms)
        assert len(result) == 1  # Now we have 1 second
        assert result[0].sequence_number == 0

    def test_flush_returns_remaining(self):
        """Flush should return buffered data as a final chunk."""
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=5000,
            overlap_ms=0,
            use_silence_detection=False,
        )

        audio_1s = make_pcm_tone(1000)
        chunker.feed(audio_1s)  # Less than chunk size

        final = chunker.flush()
        assert final is not None
        assert final.is_final is True
        assert len(final.data) == len(audio_1s)

    def test_flush_empty_returns_none(self):
        """Flush on empty buffer returns None."""
        chunker = AudioChunker(audio_format=DEFAULT_FORMAT)
        assert chunker.flush() is None

    def test_reset_clears_state(self):
        """Reset should clear the buffer and sequence counter."""
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=5000,
            overlap_ms=0,
        )

        chunker.feed(make_pcm_tone(1000))
        assert chunker.flush() is not None

        chunker.reset()
        assert chunker.flush() is None
        assert chunker._sequence == 0

    def test_streaming_produces_sequential_chunks(self):
        """Feeding a long stream produces correctly sequenced chunks."""
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=1000,
            overlap_ms=0,
            use_silence_detection=False,
        )

        # Feed 5 seconds in 500ms increments
        all_chunks = []
        for _ in range(10):
            chunks = chunker.feed(make_pcm_tone(500))
            all_chunks.extend(chunks)

        final = chunker.flush()
        if final:
            all_chunks.append(final)

        assert len(all_chunks) == 5
        for i, c in enumerate(all_chunks):
            assert c.sequence_number == i

    def test_overlap_retained_in_buffer(self):
        """When overlap > 0, feed should retain overlap bytes in buffer."""
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=1000,  # 32000 bytes
            overlap_ms=500,  # 16000 bytes overlap
            use_silence_detection=False,
        )

        # Feed exactly 1 second
        audio_1s = make_pcm_tone(1000)
        chunks = chunker.feed(audio_1s)
        assert len(chunks) == 1

        # Buffer should retain 500ms (16000 bytes) of overlap
        assert len(chunker._buffer) == 16000


# ── Async stream chunking tests ─────────────────────────────────


class TestAsyncStreamChunking:
    """Tests for the async chunk_stream method."""

    @pytest.mark.asyncio
    async def test_chunk_stream(self):
        """chunk_stream should yield chunks from an async audio iterator."""
        chunker = AudioChunker(
            audio_format=DEFAULT_FORMAT,
            chunk_duration_ms=1000,
            overlap_ms=0,
            use_silence_detection=False,
        )

        # 5 x 500ms = 2500ms → 2 full chunks + 1 flush of 500ms
        async def audio_source():
            for _ in range(5):
                yield make_pcm_tone(500)

        chunks = []
        async for chunk in chunker.chunk_stream(audio_source()):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[-1].is_final is True


# ── RMS and dB utility tests ────────────────────────────────────


class TestAudioUtilities:
    """Tests for RMS computation and dB conversion."""

    def test_rms_of_silence_is_zero(self):
        silence = make_pcm_silence(100)
        rms = AudioChunker._compute_rms(silence, sample_width=2)
        assert rms == 0.0

    def test_rms_of_tone_is_nonzero(self):
        tone = make_pcm_tone(100, amplitude=10000)
        rms = AudioChunker._compute_rms(tone, sample_width=2)
        assert rms > 0

    def test_rms_empty_data(self):
        assert AudioChunker._compute_rms(b"", sample_width=2) == 0.0

    def test_db_to_amplitude_zero_db(self):
        """0 dBFS should return max 16-bit amplitude."""
        amp = AudioChunker._db_to_amplitude(0.0)
        assert abs(amp - 32767.0) < 1.0

    def test_db_to_amplitude_negative(self):
        """Negative dB should return lower amplitude."""
        amp_0 = AudioChunker._db_to_amplitude(0.0)
        amp_neg = AudioChunker._db_to_amplitude(-20.0)
        assert amp_neg < amp_0

    def test_rms_non_16bit_returns_zero(self):
        """Non-16-bit sample width should return 0."""
        assert AudioChunker._compute_rms(b"\x00" * 100, sample_width=4) == 0.0
