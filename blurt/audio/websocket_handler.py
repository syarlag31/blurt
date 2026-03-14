"""WebSocket handler for full-duplex audio streaming.

This is the primary voice input endpoint for Blurt. It accepts raw audio
chunks over a persistent WebSocket connection, buffers them into complete
utterances, and returns acknowledgments, transcripts, and processing results
bidirectionally.

Protocol flow:
    1. Client connects to ws://host/ws/audio
    2. Client sends session.init with user_id and audio config
    3. Server responds with session.created
    4. Client streams audio.chunk messages (binary frames)
    5. Client sends audio.commit when done speaking
    6. Server processes audio → returns transcript.final + blurt.created
    7. Server may send response.text or response.audio at any time
    8. Client sends session.end to close gracefully

The connection supports full-duplex: the server can send messages at any
time (partial transcripts, acknowledgments) while receiving audio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from blurt.audio.buffer import AudioBuffer, BufferEmptyError, BufferOverflowError
from blurt.audio.session_manager import SessionLimitError, SessionManager
from blurt.config.settings import WebSocketConfig
from blurt.models.audio import (
    AudioConfig,
    AudioSession,
    ClientMessageType,
    ServerMessage,
    ServerMessageType,
    SessionInitPayload,
    SessionState,
    TextInputPayload,
)

logger = logging.getLogger(__name__)


class AudioWebSocketHandler:
    """Handles a single WebSocket connection for full-duplex audio streaming.

    Each instance manages one client connection through its lifecycle:
    init → streaming → processing → response → close.
    """

    def __init__(
        self,
        websocket: WebSocket,
        session_manager: SessionManager,
        config: WebSocketConfig,
    ) -> None:
        self._ws = websocket
        self._session_manager = session_manager
        self._config = config
        self._session: AudioSession | None = None
        self._buffer: AudioBuffer | None = None
        self._client_sequence: int = -1
        self._closed = False

    @property
    def session_id(self) -> str | None:
        """The current session ID, if initialized."""
        return self._session.session_id if self._session else None

    async def handle(self) -> None:
        """Main handler loop — run after WebSocket accept.

        Receives messages, dispatches to handlers, and manages
        the session lifecycle. Handles all error cases gracefully.
        """
        try:
            await self._ws.accept()
            logger.info("WebSocket connection accepted")

            # Wait for session.init as the first message
            await self._wait_for_init()

            # Main message loop
            await self._message_loop()

        except WebSocketDisconnect as e:
            logger.info(
                "Client disconnected: session=%s code=%s",
                self.session_id,
                e.code,
            )
        except asyncio.CancelledError:
            logger.info("Handler cancelled: session=%s", self.session_id)
        except Exception:
            logger.exception("Unexpected error in WebSocket handler: session=%s", self.session_id)
            await self._send_error("internal_error", "An unexpected error occurred")
        finally:
            await self._cleanup()

    async def _wait_for_init(self) -> None:
        """Wait for and process the session.init message.

        The first message MUST be session.init. Any other message
        type results in an error and connection close.
        """
        try:
            raw = await asyncio.wait_for(
                self._ws.receive_text(),
                timeout=self._config.ping_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error("timeout", "Timed out waiting for session.init")
            raise WebSocketDisconnect(code=4000)

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error("invalid_json", "First message must be valid JSON")
            raise WebSocketDisconnect(code=4001)

        msg_type = msg.get("type")
        if msg_type != ClientMessageType.SESSION_INIT.value:
            await self._send_error(
                "protocol_error",
                f"First message must be session.init, got: {msg_type}",
            )
            raise WebSocketDisconnect(code=4002)

        await self._handle_session_init(msg)

    async def _message_loop(self) -> None:
        """Main message receive loop — processes all message types."""
        while not self._closed:
            try:
                message = await asyncio.wait_for(
                    self._ws.receive(),
                    timeout=self._config.idle_timeout,
                )
            except asyncio.TimeoutError:
                logger.info("Session idle timeout: %s", self.session_id)
                await self._send_error("idle_timeout", "Session timed out due to inactivity")
                break

            # Handle different WebSocket frame types
            if "text" in message:
                await self._handle_text_frame(message["text"])
            elif "bytes" in message:
                await self._handle_binary_frame(message["bytes"])
            elif message.get("type") == "websocket.disconnect":
                break
            else:
                logger.warning("Unknown WebSocket frame type: %s", message)

    async def _handle_text_frame(self, raw: str) -> None:
        """Parse and dispatch a JSON text frame."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error("invalid_json", "Message is not valid JSON")
            return

        msg_type = msg.get("type")
        sequence = msg.get("sequence", -1)

        # Validate sequence ordering
        if sequence <= self._client_sequence and msg_type != ClientMessageType.PING.value:
            await self._send_error(
                "sequence_error",
                f"Expected sequence > {self._client_sequence}, got {sequence}",
            )
            return
        if msg_type != ClientMessageType.PING.value:
            self._client_sequence = sequence

        # Dispatch by message type
        handlers = {
            ClientMessageType.AUDIO_COMMIT.value: self._handle_audio_commit,
            ClientMessageType.TEXT_INPUT.value: self._handle_text_input,
            ClientMessageType.SESSION_PAUSE.value: self._handle_session_pause,
            ClientMessageType.SESSION_RESUME.value: self._handle_session_resume,
            ClientMessageType.SESSION_END.value: self._handle_session_end,
            ClientMessageType.PING.value: self._handle_ping,
        }

        handler = handlers.get(msg_type)
        if handler:
            await handler(msg)
        else:
            await self._send_error(
                "unknown_message_type",
                f"Unknown message type: {msg_type}",
            )

    async def _handle_binary_frame(self, data: bytes) -> None:
        """Handle a binary WebSocket frame (audio chunk)."""
        if not self._session or not self._buffer:
            await self._send_error("no_session", "No active session — send session.init first")
            return

        if self._session.state == SessionState.PAUSED:
            # Silently drop audio while paused
            return

        if self._session.state not in (SessionState.ACTIVE, SessionState.LISTENING):
            await self._send_error(
                "invalid_state",
                f"Cannot receive audio in state: {self._session.state.value}",
            )
            return

        # Validate chunk size
        if len(data) > self._config.audio_chunk_max_bytes:
            await self._send_error(
                "chunk_too_large",
                f"Audio chunk exceeds max size ({len(data)} > {self._config.audio_chunk_max_bytes})",
            )
            return

        # Transition to listening state on first audio chunk
        if self._session.state == SessionState.ACTIVE:
            self._session.state = SessionState.LISTENING

        try:
            await self._buffer.append(data)
        except BufferOverflowError as e:
            await self._send_error("buffer_overflow", str(e))
            return

        self._session.chunks_received += 1
        self._session.bytes_received += len(data)
        self._session.touch()

        # Send periodic acknowledgments (every 10 chunks)
        if self._session.chunks_received % 10 == 0:
            await self._send_message(
                ServerMessageType.AUDIO_ACK,
                {
                    "chunks_received": self._session.chunks_received,
                    "bytes_received": self._session.bytes_received,
                    "buffer_utilization": self._buffer.utilization,
                },
            )

    async def _handle_session_init(self, msg: dict[str, Any]) -> None:
        """Process session.init — create session and audio buffer."""
        # Track sequence from init message
        sequence = msg.get("sequence", 0)
        self._client_sequence = sequence

        payload_data = msg.get("payload", {})

        try:
            init_payload = SessionInitPayload(**payload_data)
        except ValidationError as e:
            await self._send_error("invalid_payload", f"Invalid session.init payload: {e}")
            raise WebSocketDisconnect(code=4003)

        try:
            self._session = await self._session_manager.create_session(
                user_id=init_payload.user_id,
                audio_config=init_payload.audio_config,
                device_id=init_payload.device_id,
                timezone=init_payload.timezone,
                encryption_key_id=init_payload.encryption_key_id,
            )
        except SessionLimitError as e:
            await self._send_error("session_limit", str(e))
            raise WebSocketDisconnect(code=4004)

        self._buffer = await self._session_manager.get_buffer(self._session.session_id)

        await self._send_message(
            ServerMessageType.SESSION_CREATED,
            {
                "user_id": init_payload.user_id,
                "audio_config": init_payload.audio_config.model_dump(),
                "device_id": init_payload.device_id,
                "timezone": init_payload.timezone,
            },
        )

        logger.info(
            "Session initialized: %s user=%s encoding=%s rate=%d",
            self._session.session_id,
            init_payload.user_id,
            init_payload.audio_config.encoding.value,
            init_payload.audio_config.sample_rate,
        )

    async def _handle_audio_commit(self, msg: dict[str, Any]) -> None:
        """Handle audio.commit — process the buffered utterance."""
        if not self._session or not self._buffer:
            await self._send_error("no_session", "No active session")
            return

        if self._buffer.is_empty:
            await self._send_error("empty_buffer", "No audio data to commit")
            return

        self._session.state = SessionState.PROCESSING

        try:
            audio_data = await self._buffer.commit()
        except BufferEmptyError:
            await self._send_error("empty_buffer", "No audio data to commit")
            self._session.state = SessionState.ACTIVE
            return

        self._session.utterances_processed += 1

        # For now, send a placeholder transcript.final and blurt.created
        # The actual Gemini processing pipeline will be plugged in here
        await self._send_message(
            ServerMessageType.TRANSCRIPT_FINAL,
            {
                "utterance_id": self._session.utterances_processed,
                "audio_bytes": len(audio_data),
                "audio_duration_ms": _estimate_duration_ms(
                    audio_data,
                    self._session.audio_config,
                ),
                "transcript": None,  # Will be filled by Gemini pipeline
                "status": "pending_processing",
            },
        )

        await self._send_message(
            ServerMessageType.RESPONSE_TEXT,
            {
                "utterance_id": self._session.utterances_processed,
                "text": "Got it.",
                "type": "acknowledgment",
            },
        )

        # Return to active state, ready for next utterance
        self._session.state = SessionState.ACTIVE

    async def _handle_text_input(self, msg: dict[str, Any]) -> None:
        """Handle text.input — process text as a blurt (edits/corrections)."""
        if not self._session:
            await self._send_error("no_session", "No active session")
            return

        payload_data = msg.get("payload", {})
        try:
            text_payload = TextInputPayload(**payload_data)
        except ValidationError as e:
            await self._send_error("invalid_payload", f"Invalid text.input payload: {e}")
            return

        self._session.touch()

        # Acknowledge text receipt — pipeline processing will be added later
        await self._send_message(
            ServerMessageType.RESPONSE_TEXT,
            {
                "text": "Got it.",
                "type": "acknowledgment",
                "input_text": text_payload.text,
            },
        )

    async def _handle_session_pause(self, msg: dict[str, Any]) -> None:
        """Handle session.pause — pause audio processing."""
        if not self._session:
            await self._send_error("no_session", "No active session")
            return

        self._session.state = SessionState.PAUSED
        self._session.touch()
        await self._send_message(
            ServerMessageType.SESSION_UPDATED,
            {"state": SessionState.PAUSED.value},
        )

    async def _handle_session_resume(self, msg: dict[str, Any]) -> None:
        """Handle session.resume — resume audio processing."""
        if not self._session:
            await self._send_error("no_session", "No active session")
            return

        self._session.state = SessionState.ACTIVE
        self._session.touch()
        await self._send_message(
            ServerMessageType.SESSION_UPDATED,
            {"state": SessionState.ACTIVE.value},
        )

    async def _handle_session_end(self, msg: dict[str, Any]) -> None:
        """Handle session.end — graceful shutdown."""
        if self._session:
            # Commit any remaining audio before closing
            if self._buffer and not self._buffer.is_empty:
                await self._handle_audio_commit(msg)

            await self._send_message(
                ServerMessageType.SESSION_ENDED,
                {
                    "chunks_received": self._session.chunks_received,
                    "bytes_received": self._session.bytes_received,
                    "utterances_processed": self._session.utterances_processed,
                },
            )

        self._closed = True

    async def _handle_ping(self, msg: dict[str, Any]) -> None:
        """Handle ping — respond with pong."""
        session_id = self._session.session_id if self._session else "none"
        await self._ws.send_text(
            ServerMessage(
                type=ServerMessageType.PONG,
                session_id=session_id,
                sequence=0,
                payload={"server_time": time.time()},
            ).model_dump_json()
        )

    async def _send_message(
        self,
        msg_type: ServerMessageType,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Send a typed server message to the client."""
        if self._closed:
            return

        session_id = self._session.session_id if self._session else "none"
        sequence = self._session.next_sequence() if self._session else 0

        message = ServerMessage(
            type=msg_type,
            session_id=session_id,
            sequence=sequence,
            payload=payload,
        )

        try:
            await self._ws.send_text(message.model_dump_json())
        except Exception:
            logger.exception("Failed to send message: type=%s", msg_type.value)
            self._closed = True

    async def _send_error(
        self,
        code: str,
        message: str,
    ) -> None:
        """Send an error message to the client."""
        await self._send_message(
            ServerMessageType.ERROR,
            {"code": code, "message": message},
        )

    async def _cleanup(self) -> None:
        """Clean up session state on disconnect."""
        if self._session:
            await self._session_manager.end_session(self._session.session_id)
            self._session = None
            self._buffer = None


def _estimate_duration_ms(audio_data: bytes, config: AudioConfig) -> int:
    """Estimate audio duration in milliseconds from raw PCM data.

    This is an approximation based on encoding, sample rate, and channels.
    """
    from blurt.models.audio import AudioEncoding

    bytes_per_sample = {
        AudioEncoding.PCM_S16LE: 2,
        AudioEncoding.PCM_F32LE: 4,
        AudioEncoding.MULAW: 1,
    }

    bps = bytes_per_sample.get(config.encoding)
    if bps is None:
        # For compressed formats, we can't easily estimate
        return 0

    num_samples = len(audio_data) // (bps * config.channels)
    return int((num_samples / config.sample_rate) * 1000)
