"""Background polling service for Google Calendar change detection.

Periodically polls Google Calendar for changes and invokes callbacks
when events are created, updated, or deleted externally.

Supports exponential backoff on errors and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from blurt.integrations.google_calendar.reader import GoogleCalendarReader
from blurt.integrations.google_calendar.sync_state import PollResult, SyncState

logger = logging.getLogger(__name__)


class CalendarPoller:
    """Background polling service for calendar change detection.

    Wraps GoogleCalendarReader.poll_changes() in an async loop with
    configurable interval, error backoff, and change callbacks.

    Usage:
        reader = GoogleCalendarReader(auth)
        state = SyncState(user_id="u1")
        poller = CalendarPoller(reader, state, on_changes=handle_changes)
        await poller.start()
        # ... later ...
        await poller.stop()
    """

    def __init__(
        self,
        reader: GoogleCalendarReader,
        sync_state: SyncState,
        on_changes: Callable[[PollResult], Any] | None = None,
        error_backoff_base: int = 60,
        max_backoff: int = 3600,
    ):
        self._reader = reader
        self._sync_state = sync_state
        self._on_changes = on_changes
        self._error_backoff_base = error_backoff_base
        self._max_backoff = max_backoff
        self._running = False
        self._task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def sync_state(self) -> SyncState:
        return self._sync_state

    async def start(self) -> None:
        """Start the background polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Calendar poller started for user=%s calendar=%s interval=%ds",
            self._sync_state.user_id,
            self._sync_state.calendar_id,
            self._sync_state.poll_interval_seconds,
        )

    async def stop(self) -> None:
        """Stop the background polling loop gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info(
            "Calendar poller stopped for user=%s calendar=%s",
            self._sync_state.user_id,
            self._sync_state.calendar_id,
        )

    def poll_once(self) -> PollResult:
        """Execute a single synchronous poll and return results.

        Useful for on-demand sync or testing.
        """
        result = self._reader.poll_changes(self._sync_state)
        if result.has_changes and self._on_changes:
            self._on_changes(result)
        return result

    async def _poll_loop(self) -> None:
        """Internal polling loop with error handling and backoff."""
        while self._running:
            try:
                # Run the synchronous poll in a thread to avoid blocking
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, self._reader.poll_changes, self._sync_state
                )

                if result.has_changes:
                    logger.info(
                        "Calendar changes detected: created=%d updated=%d deleted=%d",
                        result.created_count,
                        result.updated_count,
                        result.deleted_count,
                    )
                    if self._on_changes:
                        cb_result = self._on_changes(result)
                        # Support async callbacks
                        if asyncio.iscoroutine(cb_result):
                            await cb_result

                interval = self._sync_state.poll_interval_seconds

            except asyncio.CancelledError:
                raise
            except Exception:
                self._sync_state.consecutive_errors += 1
                backoff = min(
                    self._error_backoff_base * (2 ** (self._sync_state.consecutive_errors - 1)),
                    self._max_backoff,
                )
                logger.exception(
                    "Calendar poll error (attempt %d), backing off %ds",
                    self._sync_state.consecutive_errors,
                    backoff,
                )
                interval = backoff

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
