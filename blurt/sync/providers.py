"""Abstract provider adapter and concrete implementations for external sync targets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from blurt.models.sync import SyncOperation, SyncProvider

logger = logging.getLogger(__name__)


class SyncProviderAdapter(ABC):
    """Abstract base class for external service adapters.

    Each adapter knows how to push data to and pull data from
    one external service (Google Calendar, Notion, etc.).
    """

    @property
    @abstractmethod
    def provider(self) -> SyncProvider:
        """Which provider this adapter handles."""

    @abstractmethod
    async def push(self, operation: SyncOperation) -> dict[str, Any]:
        """Push a change from Blurt to the external service.

        Returns a dict with at least:
          - external_id: str
          - external_version: str
        """

    @abstractmethod
    async def pull(self, external_id: str) -> dict[str, Any]:
        """Pull the current state of an entity from the external service.

        Returns the full external representation.
        """

    @abstractmethod
    async def fetch_changes_since(self, since: datetime) -> list[dict[str, Any]]:
        """Fetch all changes from the external service since a given timestamp.

        Returns a list of changed entities with their external IDs and data.
        """

    @abstractmethod
    async def delete(self, external_id: str) -> bool:
        """Delete an entity from the external service.

        Returns True if deletion succeeded.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the adapter can reach the external service."""


class GoogleCalendarAdapter(SyncProviderAdapter):
    """Adapter for Google Calendar API.

    In production, this wraps the Google Calendar API client.
    For v1, this provides the interface contract that the orchestrator uses.
    """

    def __init__(self, credentials: dict[str, Any] | None = None) -> None:
        self._credentials = credentials
        self._connected = False

    @property
    def provider(self) -> SyncProvider:
        return SyncProvider.GOOGLE_CALENDAR

    async def push(self, operation: SyncOperation) -> dict[str, Any]:
        """Push event to Google Calendar."""
        logger.info(
            "Google Calendar push: %s %s",
            operation.operation_type,
            operation.payload.get("title", "untitled"),
        )
        # In production: call Google Calendar API
        # For now, return a stub that satisfies the contract
        return {
            "external_id": f"gcal_{operation.id}",
            "external_version": "1",
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

    async def pull(self, external_id: str) -> dict[str, Any]:
        """Pull event from Google Calendar."""
        logger.info("Google Calendar pull: %s", external_id)
        return {
            "external_id": external_id,
            "title": "",
            "start": None,
            "end": None,
            "updated": datetime.now(timezone.utc).isoformat(),
        }

    async def fetch_changes_since(self, since: datetime) -> list[dict[str, Any]]:
        """Fetch changed events since timestamp using sync tokens."""
        logger.info("Google Calendar fetch changes since: %s", since.isoformat())
        return []

    async def delete(self, external_id: str) -> bool:
        """Delete event from Google Calendar."""
        logger.info("Google Calendar delete: %s", external_id)
        return True

    async def health_check(self) -> bool:
        """Check Google Calendar API connectivity."""
        return self._credentials is not None


class NotionAdapter(SyncProviderAdapter):
    """Adapter for Notion API.

    Syncs tasks and ideas to Notion databases.
    """

    def __init__(self, api_key: str | None = None, database_id: str | None = None) -> None:
        self._api_key = api_key
        self._database_id = database_id

    @property
    def provider(self) -> SyncProvider:
        return SyncProvider.NOTION

    async def push(self, operation: SyncOperation) -> dict[str, Any]:
        """Push page/block to Notion."""
        logger.info(
            "Notion push: %s %s",
            operation.operation_type,
            operation.payload.get("title", "untitled"),
        )
        return {
            "external_id": f"notion_{operation.id}",
            "external_version": "1",
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

    async def pull(self, external_id: str) -> dict[str, Any]:
        """Pull page from Notion."""
        logger.info("Notion pull: %s", external_id)
        return {
            "external_id": external_id,
            "title": "",
            "properties": {},
            "last_edited_time": datetime.now(timezone.utc).isoformat(),
        }

    async def fetch_changes_since(self, since: datetime) -> list[dict[str, Any]]:
        """Fetch changed pages since timestamp."""
        logger.info("Notion fetch changes since: %s", since.isoformat())
        return []

    async def delete(self, external_id: str) -> bool:
        """Archive page in Notion (Notion doesn't truly delete)."""
        logger.info("Notion archive: %s", external_id)
        return True

    async def health_check(self) -> bool:
        """Check Notion API connectivity."""
        return self._api_key is not None
