"""Connection registry for tracking active WebSocket handlers.

Provides a centralized way to push server-initiated messages (task nudges,
notifications) to connected clients by user_id or broadcast to all.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from blurt.models.audio import ServerMessageType

logger = logging.getLogger(__name__)


@runtime_checkable
class WebSocketHandler(Protocol):
    """Protocol for WebSocket handlers that can receive push messages.

    Both ``WebSocketHandler`` and test fakes satisfy this interface.
    """

    @property
    def user_id(self) -> str | None:
        """The authenticated user id for this connection."""
        ...

    async def push_message(
        self, msg_type: ServerMessageType, payload: dict[str, Any]
    ) -> None:
        """Push a server-initiated message to the connected client."""
        ...


@dataclass
class TaskNudgePayload:
    """Structured payload for a task.nudge server-push message.

    Attributes:
        task_id: Unique identifier of the task being nudged.
        content: Human-readable task description (shown in UI).
        intent: Task type — task, event, or reminder.
        due_at: Optional ISO 8601 due date string.
        reason: Why the nudge is being sent (e.g. "due_soon", "context_match").
        priority: Numeric priority 0.0–1.0 (higher = more urgent).
        entity_names: Related entities (people, projects, places).
        surface_count: How many times this task has been surfaced.
    """

    task_id: str
    content: str
    intent: str = "task"
    due_at: str | None = None
    reason: str = "scheduled"
    priority: float = 0.5
    entity_names: list[str] = field(default_factory=list)
    surface_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for WebSocket JSON payload."""
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "content": self.content,
            "intent": self.intent,
            "reason": self.reason,
            "priority": self.priority,
            "nudge_ts": time.time(),
        }
        if self.due_at is not None:
            payload["due_at"] = self.due_at
        if self.entity_names:
            payload["entity_names"] = self.entity_names
        if self.surface_count > 0:
            payload["surface_count"] = self.surface_count
        return payload


class ConnectionRegistry:
    """Thread-safe registry of active WebSocket handlers.

    Used by the task surfacing engine and scheduled nudge loops to push
    messages to connected clients without those clients polling.

    Usage::

        registry = ConnectionRegistry()
        # In WebSocket endpoint:
        registry.register(handler)
        try:
            await handler.handle()
        finally:
            registry.unregister(handler)

        # From any coroutine:
        await registry.send_task_nudge("user_123", nudge_payload)
    """

    def __init__(self) -> None:
        self._handlers: set[WebSocketHandler] = set()
        self._lock = asyncio.Lock()

    @property
    def connection_count(self) -> int:
        """Number of currently connected handlers."""
        return len(self._handlers)

    def register(self, handler: WebSocketHandler) -> None:
        """Register a handler when a WebSocket connects."""
        self._handlers.add(handler)
        logger.debug(
            "Handler registered: user=%s connections=%d",
            handler.user_id,
            len(self._handlers),
        )

    def unregister(self, handler: WebSocketHandler) -> None:
        """Unregister a handler when a WebSocket disconnects."""
        self._handlers.discard(handler)
        logger.debug(
            "Handler unregistered: user=%s connections=%d",
            handler.user_id,
            len(self._handlers),
        )

    def get_handlers_for_user(self, user_id: str) -> list[WebSocketHandler]:
        """Get all active handlers for a specific user."""
        return [h for h in self._handlers if h.user_id == user_id]

    async def send_task_nudge(
        self,
        user_id: str,
        nudge: TaskNudgePayload,
    ) -> int:
        """Push a task nudge to all connections for a specific user.

        Args:
            user_id: Target user ID.
            nudge: Structured task nudge payload.

        Returns:
            Number of handlers the nudge was sent to.
        """
        handlers = self.get_handlers_for_user(user_id)
        if not handlers:
            logger.debug("No connected handlers for user=%s, nudge dropped", user_id)
            return 0

        payload = nudge.to_dict()
        sent = 0
        for handler in handlers:
            try:
                await handler.push_message(ServerMessageType.TASK_NUDGE, payload)
                sent += 1
            except Exception:
                logger.exception(
                    "Failed to push task nudge to handler: user=%s task=%s",
                    user_id,
                    nudge.task_id,
                )
        logger.info(
            "Task nudge sent: user=%s task=%s reason=%s handlers=%d",
            user_id,
            nudge.task_id,
            nudge.reason,
            sent,
        )
        return sent

    async def broadcast_task_nudge(self, nudge: TaskNudgePayload) -> int:
        """Broadcast a task nudge to ALL connected clients.

        Returns:
            Number of handlers the nudge was sent to.
        """
        payload = nudge.to_dict()
        sent = 0
        for handler in list(self._handlers):
            try:
                await handler.push_message(ServerMessageType.TASK_NUDGE, payload)
                sent += 1
            except Exception:
                logger.exception(
                    "Failed to broadcast nudge to handler: task=%s",
                    nudge.task_id,
                )
        return sent

    async def push_to_user(
        self,
        user_id: str,
        msg_type: ServerMessageType,
        payload: dict[str, Any],
    ) -> int:
        """Push an arbitrary server message to all connections for a user.

        Lower-level than send_task_nudge — use this for non-nudge push messages.

        Returns:
            Number of handlers the message was sent to.
        """
        handlers = self.get_handlers_for_user(user_id)
        sent = 0
        for handler in handlers:
            try:
                await handler.push_message(msg_type, payload)
                sent += 1
            except Exception:
                logger.exception(
                    "Failed to push message to handler: user=%s type=%s",
                    user_id,
                    msg_type.value,
                )
        return sent
