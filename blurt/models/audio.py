"""Audio session and streaming models."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AudioEncoding(str, enum.Enum):
    """Supported audio encodings for streaming input."""

    PCM_S16LE = "pcm_s16le"  # 16-bit signed little-endian PCM (default)
    PCM_F32LE = "pcm_f32le"  # 32-bit float little-endian PCM
    OPUS = "opus"
    FLAC = "flac"
    MULAW = "mulaw"


class AudioConfig(BaseModel):
    """Audio stream configuration sent during session init."""

    encoding: AudioEncoding = AudioEncoding.PCM_S16LE
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    language: str = Field(default="en-US", max_length=10)


class SessionState(str, enum.Enum):
    """WebSocket session lifecycle states."""

    CONNECTING = "connecting"
    ACTIVE = "active"
    LISTENING = "listening"  # Actively receiving audio
    PROCESSING = "processing"  # Processing a completed utterance
    PAUSED = "paused"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


class ClientMessageType(str, enum.Enum):
    """Message types the client can send over WebSocket."""

    SESSION_INIT = "session.init"
    AUDIO_CHUNK = "audio.chunk"
    AUDIO_COMMIT = "audio.commit"  # Client signals end of utterance
    TEXT_INPUT = "text.input"  # Text fallback for edits/corrections
    SESSION_PAUSE = "session.pause"
    SESSION_RESUME = "session.resume"
    SESSION_END = "session.end"
    PING = "ping"


class ServerMessageType(str, enum.Enum):
    """Message types the server sends over WebSocket."""

    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    SESSION_ENDED = "session.ended"
    AUDIO_ACK = "audio.ack"  # Acknowledge audio chunk receipt
    TRANSCRIPT_PARTIAL = "transcript.partial"  # Streaming partial transcript
    TRANSCRIPT_FINAL = "transcript.final"  # Final transcript for utterance
    BLURT_CREATED = "blurt.created"  # Blurt processed and stored
    RESPONSE_AUDIO = "response.audio"  # Audio response chunk (TTS)
    RESPONSE_TEXT = "response.text"  # Text response (acknowledgment)
    ERROR = "error"
    PONG = "pong"


class ClientMessage(BaseModel):
    """Envelope for all client-to-server WebSocket messages."""

    type: ClientMessageType
    sequence: int = Field(ge=0, description="Monotonically increasing sequence number")
    payload: dict | None = None
    # For audio.chunk, raw binary follows the JSON header as a separate WS frame


class SessionInitPayload(BaseModel):
    """Payload for session.init message."""

    user_id: str = Field(min_length=1, max_length=128)
    audio_config: AudioConfig = Field(default_factory=AudioConfig)
    device_id: str | None = Field(default=None, max_length=128)
    timezone: str = Field(default="UTC", max_length=64)
    encryption_key_id: str | None = Field(
        default=None,
        description="Client-side encryption key identifier for E2E encryption",
    )


class TextInputPayload(BaseModel):
    """Payload for text.input message (edits/corrections)."""

    text: str = Field(min_length=1, max_length=5000)


class ServerMessage(BaseModel):
    """Envelope for all server-to-client WebSocket messages."""

    type: ServerMessageType
    session_id: str
    sequence: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict | None = None


class AudioSession(BaseModel):
    """Tracks state for an active WebSocket audio session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    state: SessionState = SessionState.CONNECTING
    audio_config: AudioConfig = Field(default_factory=AudioConfig)
    device_id: str | None = None
    timezone: str = "UTC"
    encryption_key_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    bytes_received: int = 0
    chunks_received: int = 0
    utterances_processed: int = 0
    server_sequence: int = 0

    def next_sequence(self) -> int:
        """Get and increment the server-side sequence counter."""
        seq = self.server_sequence
        self.server_sequence += 1
        return seq

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)
