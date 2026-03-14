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
from blurt.audio.models import AudioFormat, AudioEncoding as PipelineAudioEncoding
from blurt.audio.pipeline import AudioPipeline
from blurt.audio.session_manager import SessionLimitError, SessionManager
from blurt.classification.models import ClassificationResult, ClassificationStatus
from blurt.config.settings import WebSocketConfig
from blurt.models.audio import (
    AudioConfig,
    AudioEncoding,
    AudioSession,
    ClientMessageType,
    ServerMessage,
    ServerMessageType,
    SessionInitPayload,
    SessionState,
    TextInputPayload,
)
from blurt.models.emotions import EmotionResult, EmotionScore, PrimaryEmotion
from blurt.models.intents import BlurtIntent
from blurt.services.acknowledgment import AcknowledgmentService

logger = logging.getLogger(__name__)


class AudioWebSocketHandler:
    """Handles a persistent WebSocket connection for full-duplex audio streaming.

    The connection stays alive across multiple audio capture sessions.
    A single WebSocket supports:
      - Multiple session.init → session.end cycles (audio capture)
      - Server-push messages (task nudges) at any time
      - Keepalive pings between capture sessions

    Lifecycle: connect → (init → stream → end)* → disconnect
    """

    def __init__(
        self,
        websocket: WebSocket,
        session_manager: SessionManager,
        config: WebSocketConfig,
        pipeline: AudioPipeline | None = None,
        ack_service: AcknowledgmentService | None = None,
    ) -> None:
        self._ws = websocket
        self._session_manager = session_manager
        self._config = config
        self._pipeline = pipeline
        self._ack_service = ack_service or AcknowledgmentService()
        self._session: AudioSession | None = None
        self._buffer: AudioBuffer | None = None
        self._client_sequence: int = -1
        self._closed = False
        # Queue for server-initiated push messages (task nudges, etc.)
        self._push_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        # User ID is set on first session.init and persists across sessions
        self._user_id: str | None = None

    @property
    def session_id(self) -> str | None:
        """The current session ID, if initialized."""
        return self._session.session_id if self._session else None

    @property
    def user_id(self) -> str | None:
        """The user ID from the first session.init."""
        return self._user_id

    async def push_message(self, msg_type: ServerMessageType, payload: dict[str, Any]) -> None:
        """Push a server-initiated message to the client (e.g. task nudge).

        Safe to call from any coroutine. Messages are delivered via the
        push writer task running alongside the message loop.
        """
        if self._closed:
            return
        await self._push_queue.put({"type": msg_type, "payload": payload})

    async def send_task_nudge(self, nudge_payload: dict[str, Any]) -> None:
        """Convenience method to send a task.nudge event to this client.

        Args:
            nudge_payload: Dict with task_id, content, intent, reason,
                           priority, due_at, entity_names, etc.
        """
        await self.push_message(ServerMessageType.TASK_NUDGE, nudge_payload)

    async def handle(self) -> None:
        """Main handler loop — persistent across multiple capture sessions.

        The WebSocket stays open after session.end, waiting for the next
        session.init. Only an explicit disconnect or error closes it.
        """
        try:
            await self._ws.accept()
            logger.info("WebSocket connection accepted (persistent mode)")

            # Wait for first session.init
            await self._wait_for_init()

            # Run message loop and push writer concurrently
            push_task = asyncio.create_task(self._push_writer())
            try:
                await self._message_loop()
            finally:
                push_task.cancel()
                try:
                    await push_task
                except asyncio.CancelledError:
                    pass

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

    async def _push_writer(self) -> None:
        """Drains the push queue and sends server-initiated messages.

        Runs as a background task alongside _message_loop so that
        server-push (task nudges) can be sent at any time.
        """
        try:
            while not self._closed:
                msg = await self._push_queue.get()
                msg_type = msg["type"]
                payload = msg["payload"]
                await self._send_message(msg_type, payload)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in push writer")

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
        """Persistent message receive loop — survives across capture sessions.

        When a session ends, the loop continues waiting for the next
        session.init or keepalive pings. The connection only closes on
        disconnect, cancellation, or idle timeout.
        """
        while not self._closed:
            try:
                message = await asyncio.wait_for(
                    self._ws.receive(),
                    timeout=self._config.idle_timeout,
                )
            except asyncio.TimeoutError:
                logger.info(
                    "Connection idle timeout: session=%s user=%s",
                    self.session_id,
                    self._user_id,
                )
                await self._send_error("idle_timeout", "Connection timed out due to inactivity")
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

        # Dispatch by message type — session.init allowed for re-initialization
        handlers = {
            ClientMessageType.SESSION_INIT.value: self._handle_session_init,
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
        """Process session.init — create session and audio buffer.

        Supports re-initialization: if an old session exists, it is
        cleaned up before creating a new one. This allows the persistent
        WebSocket to cycle through multiple capture sessions.
        """
        # Clean up any existing session before re-initializing
        if self._session:
            logger.info(
                "Re-initializing session on persistent connection: old=%s",
                self._session.session_id,
            )
            await self._session_manager.end_session(self._session.session_id)
            self._session = None
            self._buffer = None

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

        # Persist user_id across session cycles on this connection
        self._user_id = init_payload.user_id

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
            "Session initialized: %s user=%s encoding=%s rate=%d (persistent connection)",
            self._session.session_id,
            init_payload.user_id,
            init_payload.audio_config.encoding.value,
            init_payload.audio_config.sample_rate,
        )

    async def _handle_audio_commit(self, msg: dict[str, Any]) -> None:
        """Handle audio.commit — process the buffered utterance through Gemini."""
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
        utterance_id = self._session.utterances_processed

        # Convert WebSocket audio config to pipeline AudioFormat
        audio_format = _ws_config_to_audio_format(self._session.audio_config)

        if self._pipeline is not None:
            try:
                result = await self._pipeline.process_audio(
                    audio_data,
                    audio_format,
                    user_id=self._session.user_id,
                )

                # Use pipeline duration or fall back to local estimate
                duration_ms = result.audio_duration_ms or _estimate_duration_ms(
                    audio_data, self._session.audio_config,
                )

                # Send the final transcript
                await self._send_message(
                    ServerMessageType.TRANSCRIPT_FINAL,
                    {
                        "utterance_id": utterance_id,
                        "audio_bytes": len(audio_data),
                        "audio_duration_ms": duration_ms,
                        "transcript": result.transcript,
                        "status": "complete" if result.pipeline_complete else "partial",
                        "intent": result.intent,
                        "confidence": result.confidence,
                        "processing_time_ms": result.processing_time_ms,
                    },
                )

                # Generate emotion-aware acknowledgment via AcknowledgmentService
                ack = self._build_acknowledgment(result)

                if ack.is_silent and ack.answer:
                    # For questions, send the answer
                    await self._send_message(
                        ServerMessageType.RESPONSE_TEXT,
                        {
                            "utterance_id": utterance_id,
                            "text": ack.answer,
                            "type": "answer",
                        },
                    )
                elif not ack.is_silent and ack.text:
                    await self._send_message(
                        ServerMessageType.RESPONSE_TEXT,
                        {
                            "utterance_id": utterance_id,
                            "text": ack.text,
                            "type": "acknowledgment",
                            "tone": ack.tone.value,
                        },
                    )

                # Send blurt.created with full classification data (only on success)
                if result.pipeline_complete and not result.error:
                    await self._send_message(
                        ServerMessageType.BLURT_CREATED,
                        {
                            "utterance_id": utterance_id,
                            "blurt_id": result.id,
                            "transcript": result.transcript,
                            "intent": result.intent,
                            "confidence": result.confidence,
                            "entities": result.entities,
                            "emotion": result.emotion,
                            "pipeline_complete": result.pipeline_complete,
                        },
                    )

            except Exception:
                logger.exception(
                    "Pipeline processing failed: session=%s utterance=%d",
                    self.session_id,
                    utterance_id,
                )
                # Graceful degradation: still acknowledge receipt
                await self._send_message(
                    ServerMessageType.TRANSCRIPT_FINAL,
                    {
                        "utterance_id": utterance_id,
                        "audio_bytes": len(audio_data),
                        "audio_duration_ms": _estimate_duration_ms(
                            audio_data, self._session.audio_config,
                        ),
                        "transcript": None,
                        "status": "error",
                    },
                )
                from blurt.services.acknowledgment import generate_acknowledgment_for_error
                error_ack = generate_acknowledgment_for_error()
                await self._send_message(
                    ServerMessageType.RESPONSE_TEXT,
                    {
                        "utterance_id": utterance_id,
                        "text": error_ack.text,
                        "type": "acknowledgment",
                    },
                )
        else:
            # No pipeline available — send placeholder (development fallback)
            await self._send_message(
                ServerMessageType.TRANSCRIPT_FINAL,
                {
                    "utterance_id": utterance_id,
                    "audio_bytes": len(audio_data),
                    "audio_duration_ms": _estimate_duration_ms(
                        audio_data, self._session.audio_config,
                    ),
                    "transcript": None,
                    "status": "no_pipeline",
                },
            )
            await self._send_message(
                ServerMessageType.RESPONSE_TEXT,
                {
                    "utterance_id": utterance_id,
                    "text": "Got it.",
                    "type": "acknowledgment",
                },
            )

        # Return to active state, ready for next utterance
        self._session.state = SessionState.ACTIVE

    async def _handle_text_input(self, msg: dict[str, Any]) -> None:
        """Handle text.input — classify text and generate acknowledgment.

        Works both during and between capture sessions on the persistent
        connection. Text input does not require an active audio session.
        """

        payload_data = msg.get("payload", {})
        try:
            text_payload = TextInputPayload(**payload_data)
        except ValidationError as e:
            await self._send_error("invalid_payload", f"Invalid text.input payload: {e}")
            return

        if self._session:
            self._session.touch()

        if self._pipeline is not None:
            try:
                # Use Gemini to classify the text input
                classification_data = await self._pipeline._gemini.classify_transcript(
                    text_payload.text
                )

                intent_str = classification_data.get("intent", "journal")
                confidence = classification_data.get("confidence", 0.0)
                emotion_data = classification_data.get("emotion", {})
                entities = classification_data.get("entities", [])

                # Build ClassificationResult for the AcknowledgmentService
                try:
                    intent = BlurtIntent(intent_str)
                except ValueError:
                    intent = BlurtIntent.JOURNAL

                status = (
                    ClassificationStatus.CONFIDENT
                    if confidence >= 0.85
                    else ClassificationStatus.LOW_CONFIDENCE
                )
                cr = ClassificationResult(
                    input_text=text_payload.text,
                    primary_intent=intent,
                    confidence=confidence,
                    status=status,
                )

                # Build EmotionResult if available
                emotion_result = _parse_emotion_data(emotion_data)

                ack = self._ack_service.acknowledge(cr, emotion_result)

                await self._send_message(
                    ServerMessageType.RESPONSE_TEXT,
                    {
                        "text": ack.text if not ack.is_silent else (ack.answer or ""),
                        "type": "answer" if ack.is_silent else "acknowledgment",
                        "input_text": text_payload.text,
                        "intent": intent_str,
                        "confidence": confidence,
                        "entities": entities,
                    },
                )

                await self._send_message(
                    ServerMessageType.BLURT_CREATED,
                    {
                        "transcript": text_payload.text,
                        "intent": intent_str,
                        "confidence": confidence,
                        "entities": entities,
                        "emotion": emotion_data,
                        "input_source": "text",
                    },
                )

            except Exception:
                logger.exception("Text classification failed: session=%s", self.session_id)
                await self._send_message(
                    ServerMessageType.RESPONSE_TEXT,
                    {
                        "text": "Got it.",
                        "type": "acknowledgment",
                        "input_text": text_payload.text,
                    },
                )
        else:
            # No pipeline — fallback
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
        """Handle session.end — end the capture session but keep connection alive.

        The WebSocket stays open for:
          - Starting a new capture session (session.init)
          - Receiving server-push messages (task nudges)
          - Keepalive pings
        """
        if self._session:
            # Commit any remaining audio before ending
            if self._buffer and not self._buffer.is_empty:
                await self._handle_audio_commit(msg)

            await self._send_message(
                ServerMessageType.SESSION_ENDED,
                {
                    "chunks_received": self._session.chunks_received,
                    "bytes_received": self._session.bytes_received,
                    "utterances_processed": self._session.utterances_processed,
                    "persistent": True,
                },
            )

            # Clean up session state, but keep the connection open
            await self._session_manager.end_session(self._session.session_id)
            logger.info(
                "Session ended (connection persists): %s user=%s",
                self._session.session_id,
                self._user_id,
            )
            self._session = None
            self._buffer = None
            # Note: self._closed is NOT set — connection stays alive

    async def _handle_ping(self, msg: dict[str, Any]) -> None:
        """Handle ping — respond with pong.

        Works both during and between capture sessions to keep
        the persistent connection alive.
        """
        session_id = self._session.session_id if self._session else "none"
        await self._ws.send_text(
            ServerMessage(
                type=ServerMessageType.PONG,
                session_id=session_id,
                sequence=0,
                payload={
                    "server_time": time.time(),
                    "has_session": self._session is not None,
                },
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

    def _build_acknowledgment(self, result: Any) -> Any:
        """Build an acknowledgment from a PipelineResult using the AcknowledgmentService."""
        from blurt.services.acknowledgment import generate_acknowledgment_for_error

        intent_str = result.intent or "journal"
        try:
            intent = BlurtIntent(intent_str)
        except ValueError:
            intent = BlurtIntent.JOURNAL

        confidence = result.confidence or 0.0
        status = (
            ClassificationStatus.CONFIDENT
            if confidence >= 0.85
            else ClassificationStatus.LOW_CONFIDENCE
        )

        if result.error:
            return generate_acknowledgment_for_error()

        cr = ClassificationResult(
            input_text=result.transcript,
            primary_intent=intent,
            confidence=confidence,
            status=status,
        )

        emotion_result = _parse_emotion_data(result.emotion)

        # Use the acknowledgment from Gemini classification if available,
        # otherwise fall back to the AcknowledgmentService
        if result.acknowledgment:
            from blurt.services.acknowledgment import Acknowledgment, _select_tone
            tone = _select_tone(emotion_result)
            return Acknowledgment(
                text=result.acknowledgment,
                tone=tone,
                intent=intent,
                is_silent=(intent == BlurtIntent.QUESTION),
            )

        return self._ack_service.acknowledge(cr, emotion_result)

    async def _cleanup(self) -> None:
        """Clean up session state on final disconnect.

        Called when the WebSocket connection itself closes (not on session.end).
        """
        self._closed = True
        if self._session:
            await self._session_manager.end_session(self._session.session_id)
            self._session = None
            self._buffer = None
        logger.info(
            "Persistent WebSocket connection closed: user=%s",
            self._user_id,
        )


def _parse_emotion_data(emotion_data: dict[str, Any] | None) -> EmotionResult | None:
    """Parse emotion dict from Gemini classification into an EmotionResult."""
    if not emotion_data:
        return None
    try:
        primary_str = emotion_data.get("primary", "trust")
        try:
            primary_emotion = PrimaryEmotion(primary_str)
        except ValueError:
            primary_emotion = PrimaryEmotion.TRUST

        intensity = float(emotion_data.get("intensity", 0.3))
        # Clamp to valid range
        intensity = max(0.0, min(1.0, intensity))

        valence = float(emotion_data.get("valence", 0.0))
        valence = max(-1.0, min(1.0, valence))

        arousal = float(emotion_data.get("arousal", 0.5))
        arousal = max(0.0, min(1.0, arousal))

        return EmotionResult(
            primary=EmotionScore(emotion=primary_emotion, intensity=intensity),
            valence=valence,
            arousal=arousal,
            confidence=0.8,
        )
    except (TypeError, KeyError, ValueError):
        return None


def _ws_config_to_audio_format(config: AudioConfig) -> AudioFormat:
    """Convert WebSocket AudioConfig to pipeline AudioFormat.

    Maps the WebSocket-layer encoding enum to the pipeline-layer format.
    """
    encoding_map = {
        AudioEncoding.PCM_S16LE: PipelineAudioEncoding.LINEAR16,
        AudioEncoding.PCM_F32LE: PipelineAudioEncoding.LINEAR16,  # Best-effort
        AudioEncoding.OPUS: PipelineAudioEncoding.OGG_OPUS,
        AudioEncoding.FLAC: PipelineAudioEncoding.FLAC,
        AudioEncoding.MULAW: PipelineAudioEncoding.MULAW,
    }

    pipeline_encoding = encoding_map.get(config.encoding, PipelineAudioEncoding.LINEAR16)

    sample_width = 2  # default for 16-bit
    if config.encoding == AudioEncoding.PCM_F32LE:
        sample_width = 4
    elif config.encoding == AudioEncoding.MULAW:
        sample_width = 1

    return AudioFormat(
        encoding=pipeline_encoding,
        sample_rate_hz=config.sample_rate,
        channels=config.channels,
        sample_width_bytes=sample_width,
    )


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
