"""FastAPI application factory for Blurt backend.

Creates and configures the FastAPI application with WebSocket
audio streaming endpoint and health checks.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket

from blurt.audio.session_manager import SessionManager
from blurt.audio.websocket_handler import AudioWebSocketHandler
from blurt.config.settings import BlurtConfig, DeploymentMode
from blurt.middleware.egress_guard import install_egress_guards

logger = logging.getLogger(__name__)


async def _idle_cleanup_loop(
    manager: SessionManager,
    idle_timeout: float,
    interval: float = 60.0,
) -> None:
    """Periodically clean up idle sessions."""
    while True:
        try:
            await asyncio.sleep(interval)
            removed = await manager.cleanup_idle(idle_timeout)
            if removed:
                logger.info("Cleaned up %d idle sessions", len(removed))
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Error in idle cleanup loop")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — init and teardown."""
    config: BlurtConfig = app.state.config
    ws_cfg = config.websocket

    app.state.ws_config = ws_cfg
    app.state.session_manager = SessionManager(
        max_concurrent=ws_cfg.max_concurrent_sessions,
        max_audio_buffer_bytes=ws_cfg.max_audio_buffer_bytes,
    )

    # Start idle session cleanup task
    app.state.cleanup_task = asyncio.create_task(
        _idle_cleanup_loop(app.state.session_manager, ws_cfg.idle_timeout)
    )

    logger.info(
        "Blurt backend started (mode=%s, ws_max_sessions=%d)",
        config.mode.value,
        ws_cfg.max_concurrent_sessions,
    )

    yield

    # Shutdown
    if hasattr(app.state, "cleanup_task") and app.state.cleanup_task:
        app.state.cleanup_task.cancel()
        try:
            await app.state.cleanup_task
        except asyncio.CancelledError:
            pass

    # Clean up remaining sessions
    if hasattr(app.state, "session_manager"):
        for sid in list(app.state.session_manager.session_ids):
            await app.state.session_manager.end_session(sid)

    logger.info("Blurt backend shutdown complete")


def create_app(config: BlurtConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional BlurtConfig. If not provided, loads from environment.

    Returns:
        Configured FastAPI application with WebSocket endpoint.
    """
    if config is None:
        config = BlurtConfig.from_env()

    app = FastAPI(
        title="Blurt API",
        description="AI second brain with full-duplex voice interaction",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store config on app state for lifespan access
    app.state.config = config

    # Install egress guards (activated in local-only mode)
    install_egress_guards(app, local_mode=config.mode == DeploymentMode.LOCAL)

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    from blurt.api.capture import router as capture_router
    from blurt.api.classify import router as classify_router
    from blurt.api.clear_state import router as clear_state_router
    from blurt.api.episodes import router as episodes_router
    from blurt.api.google_calendar_routes import router as google_calendar_router
    from blurt.api.patterns import router as patterns_router
    from blurt.api.question import router as question_router
    from blurt.api.recall import router as recall_router
    from blurt.api.rhythms import router as rhythms_router
    from blurt.api.task_feedback import router as feedback_router
    from blurt.api.task_surfacing import router as surfacing_router
    from blurt.api.temporal_activity import router as temporal_activity_router

    app.include_router(capture_router)
    app.include_router(classify_router)
    app.include_router(clear_state_router)
    app.include_router(episodes_router)
    app.include_router(google_calendar_router)
    app.include_router(patterns_router)
    app.include_router(question_router)
    app.include_router(recall_router)
    app.include_router(feedback_router)
    app.include_router(surfacing_router)
    app.include_router(rhythms_router)
    app.include_router(temporal_activity_router)

    @app.get("/health")
    async def health() -> dict:
        """Health check endpoint."""
        manager = getattr(app.state, "session_manager", None)
        return {
            "status": "ok",
            "version": "0.1.0",
            "active_sessions": manager.active_count if manager else 0,
        }

    @app.websocket("/ws/audio")
    async def websocket_audio(websocket: WebSocket) -> None:
        """Full-duplex audio streaming WebSocket endpoint.

        Protocol:
            1. Connect → ws://host/ws/audio
            2. Send session.init (JSON) with user_id and audio config
            3. Stream audio.chunk (binary frames)
            4. Send audio.commit (JSON) to end an utterance
            5. Receive transcript.final + blurt.created
            6. Send session.end (JSON) to close
        """
        session_manager = getattr(app.state, "session_manager", None)
        ws_config = getattr(app.state, "ws_config", None)

        if session_manager is None or ws_config is None:
            await websocket.close(code=1013, reason="Server not ready")
            return

        handler = AudioWebSocketHandler(
            websocket=websocket,
            session_manager=session_manager,
            config=ws_config,
        )
        await handler.handle()
