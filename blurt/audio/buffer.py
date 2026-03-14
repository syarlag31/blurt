"""Audio buffer for accumulating streaming audio chunks.

Each WebSocket session has its own AudioBuffer that collects incoming
audio chunks until an utterance boundary is detected (client sends
audio.commit or silence detection triggers). The buffer then yields
the complete audio for processing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class AudioBuffer:
    """Thread-safe buffer for accumulating streaming audio data.

    Attributes:
        max_bytes: Maximum bytes to buffer before rejecting new chunks.
        chunks: Ordered list of raw audio byte chunks.
        total_bytes: Running count of bytes in the buffer.
        chunk_count: Number of chunks received.
        started_at: Timestamp of the first chunk in the current utterance.
    """

    max_bytes: int = 10 * 1024 * 1024  # 10MB default
    chunks: list[bytes] = field(default_factory=list)
    total_bytes: int = 0
    chunk_count: int = 0
    started_at: float | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def append(self, data: bytes) -> None:
        """Append an audio chunk to the buffer.

        Raises:
            BufferOverflowError: If adding this chunk would exceed max_bytes.
            ValueError: If data is empty.
        """
        if not data:
            raise ValueError("Cannot append empty audio data")

        async with self._lock:
            if self.total_bytes + len(data) > self.max_bytes:
                raise BufferOverflowError(
                    f"Buffer would exceed {self.max_bytes} bytes "
                    f"(current: {self.total_bytes}, incoming: {len(data)})"
                )
            if self.started_at is None:
                self.started_at = time.monotonic()
            self.chunks.append(data)
            self.total_bytes += len(data)
            self.chunk_count += 1

    async def commit(self) -> bytes:
        """Flush the buffer and return all accumulated audio as a single bytes object.

        Returns:
            Concatenated audio bytes from all chunks.

        Raises:
            BufferEmptyError: If the buffer has no data.
        """
        async with self._lock:
            if not self.chunks:
                raise BufferEmptyError("No audio data in buffer to commit")
            audio = b"".join(self.chunks)
            self._reset()
            return audio

    async def discard(self) -> int:
        """Discard all buffered audio, returning the number of bytes discarded."""
        async with self._lock:
            discarded = self.total_bytes
            self._reset()
            return discarded

    def _reset(self) -> None:
        """Reset internal state (caller must hold lock)."""
        self.chunks.clear()
        self.total_bytes = 0
        self.chunk_count = 0
        self.started_at = None

    @property
    def duration_seconds(self) -> float | None:
        """Approximate duration of buffered audio since first chunk."""
        if self.started_at is None:
            return None
        return time.monotonic() - self.started_at

    @property
    def is_empty(self) -> bool:
        """Whether the buffer has any data."""
        return self.total_bytes == 0

    @property
    def utilization(self) -> float:
        """Buffer utilization as a fraction (0.0 to 1.0)."""
        return self.total_bytes / self.max_bytes if self.max_bytes > 0 else 0.0


class BufferOverflowError(Exception):
    """Raised when an audio chunk would exceed the buffer's max capacity."""


class BufferEmptyError(Exception):
    """Raised when attempting to commit an empty buffer."""
