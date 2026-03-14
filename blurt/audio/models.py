"""Audio processing data models.

Defines the data structures flowing through the audio pipeline:
raw audio → chunks → Gemini API → processed results.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AudioEncoding(str, Enum):
    """Supported audio encodings for input."""

    LINEAR16 = "LINEAR16"  # PCM signed 16-bit little-endian
    FLAC = "FLAC"
    MULAW = "MULAW"
    OGG_OPUS = "OGG_OPUS"
    WAV = "WAV"
    WEBM = "WEBM"


class AudioFormat(BaseModel):
    """Audio format specification."""

    encoding: AudioEncoding = AudioEncoding.LINEAR16
    sample_rate_hz: int = 16000
    channels: int = 1
    sample_width_bytes: int = 2  # 16-bit = 2 bytes

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate_hz * self.channels * self.sample_width_bytes

    @property
    def mime_type(self) -> str:
        """Return MIME type for Gemini API."""
        mime_map = {
            AudioEncoding.LINEAR16: "audio/l16",
            AudioEncoding.FLAC: "audio/flac",
            AudioEncoding.MULAW: "audio/basic",
            AudioEncoding.OGG_OPUS: "audio/ogg",
            AudioEncoding.WAV: "audio/wav",
            AudioEncoding.WEBM: "audio/webm",
        }
        return mime_map.get(self.encoding, "audio/l16")


class AudioChunk(BaseModel):
    """A single chunk of audio data ready for processing.

    Chunks are produced by the AudioChunker and consumed by the
    Gemini audio client. Each chunk carries enough metadata for
    the API call and for reassembly of the full transcript.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data: bytes  # Raw audio bytes
    sequence_number: int  # Order within the stream
    timestamp_ms: int  # Offset from stream start in milliseconds
    duration_ms: int  # Duration of this chunk in milliseconds
    is_final: bool = False  # True for the last chunk in a stream
    format: AudioFormat = Field(default_factory=AudioFormat)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def size_bytes(self) -> int:
        return len(self.data)


class TranscriptionSegment(BaseModel):
    """A segment of transcribed text with timing info."""

    text: str
    start_ms: int  # Offset from audio start
    end_ms: int
    confidence: float = 0.0  # 0.0–1.0
    speaker_id: Optional[str] = None  # For multi-speaker scenarios


class AudioProcessingResult(BaseModel):
    """Result of processing audio through the Gemini API.

    Contains the full transcript plus any multimodal understanding
    that Gemini extracted directly from the audio (tone, emphasis,
    emotional cues).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transcript: str  # Full transcribed text
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    language: str = "en"
    audio_duration_ms: int = 0

    # Multimodal audio understanding from Gemini
    raw_audio_analysis: Optional[str] = None  # Free-form Gemini analysis
    detected_tone: Optional[str] = None  # e.g., "anxious", "excited", "calm"
    detected_emphasis: list[str] = Field(default_factory=list)  # Words/phrases emphasized

    # Processing metadata
    model_used: str = ""
    processing_time_ms: int = 0
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_empty(self) -> bool:
        return not self.transcript.strip()


class StreamState(str, Enum):
    """State of an audio processing stream."""

    IDLE = "idle"
    RECEIVING = "receiving"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class AudioStreamSession(BaseModel):
    """Tracks the state of an active audio stream session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    state: StreamState = StreamState.IDLE
    format: AudioFormat = Field(default_factory=AudioFormat)
    chunks_received: int = 0
    total_bytes: int = 0
    total_duration_ms: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Accumulated results
    partial_transcript: str = ""
    result: Optional[AudioProcessingResult] = None
