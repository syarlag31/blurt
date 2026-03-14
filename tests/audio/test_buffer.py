"""Tests for AudioBuffer."""

import pytest

from blurt.audio.buffer import AudioBuffer, BufferEmptyError, BufferOverflowError


@pytest.fixture
def buffer():
    return AudioBuffer(max_bytes=1024)


class TestAudioBuffer:
    async def test_append_single_chunk(self, buffer):
        await buffer.append(b"\x00" * 100)
        assert buffer.total_bytes == 100
        assert buffer.chunk_count == 1
        assert not buffer.is_empty

    async def test_append_multiple_chunks(self, buffer):
        await buffer.append(b"\x00" * 100)
        await buffer.append(b"\x01" * 200)
        assert buffer.total_bytes == 300
        assert buffer.chunk_count == 2

    async def test_append_empty_raises(self, buffer):
        with pytest.raises(ValueError, match="empty"):
            await buffer.append(b"")

    async def test_append_overflow_raises(self, buffer):
        await buffer.append(b"\x00" * 1000)
        with pytest.raises(BufferOverflowError):
            await buffer.append(b"\x00" * 100)

    async def test_commit_returns_concatenated_audio(self, buffer):
        await buffer.append(b"\x00" * 50)
        await buffer.append(b"\x01" * 50)
        result = await buffer.commit()
        assert result == b"\x00" * 50 + b"\x01" * 50
        assert buffer.is_empty

    async def test_commit_empty_raises(self, buffer):
        with pytest.raises(BufferEmptyError):
            await buffer.commit()

    async def test_commit_resets_state(self, buffer):
        await buffer.append(b"\x00" * 100)
        await buffer.commit()
        assert buffer.total_bytes == 0
        assert buffer.chunk_count == 0
        assert buffer.started_at is None

    async def test_discard(self, buffer):
        await buffer.append(b"\x00" * 500)
        discarded = await buffer.discard()
        assert discarded == 500
        assert buffer.is_empty

    async def test_utilization(self, buffer):
        await buffer.append(b"\x00" * 512)
        assert buffer.utilization == pytest.approx(0.5, abs=0.01)

    async def test_duration_seconds_tracks_time(self, buffer):
        assert buffer.duration_seconds is None
        await buffer.append(b"\x00" * 10)
        assert buffer.duration_seconds is not None
        assert buffer.duration_seconds >= 0.0

    async def test_reuse_after_commit(self, buffer):
        """Buffer can be reused after committing."""
        await buffer.append(b"\x00" * 100)
        await buffer.commit()
        await buffer.append(b"\x01" * 200)
        result = await buffer.commit()
        assert result == b"\x01" * 200
