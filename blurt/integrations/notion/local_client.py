"""Local Notion client for local-only mode.

Provides a file-backed implementation that mirrors the NotionAPIClient
interface. Pages are stored locally as JSON, ensuring full feature
parity with no data leakage to Notion.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LocalNotionClient:
    """File-backed local Notion client for local-only mode.

    Stores pages as JSON files in the data directory. Implements the
    same interface as NotionAPIClient so all sync logic works unchanged.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = (data_dir or Path.home() / ".blurt") / "notion"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._connected = False

    @property
    def _databases_file(self) -> Path:
        return self._data_dir / "databases.json"

    def _load_databases(self) -> dict[str, dict[str, Any]]:
        """Load all database data."""
        if self._databases_file.exists():
            return json.loads(self._databases_file.read_text())
        return {}

    def _save_databases(self, databases: dict[str, dict[str, Any]]) -> None:
        """Save all database data."""
        self._databases_file.write_text(json.dumps(databases, indent=2, default=str))

    async def connect(self) -> None:
        """Initialize (no-op for local mode)."""
        self._connected = True
        logger.info("Local Notion client connected (no external API)")

    async def close(self) -> None:
        """Shut down (no-op for local mode)."""
        self._connected = False
        logger.info("Local Notion client closed")

    async def __aenter__(self) -> LocalNotionClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def create_page(
        self,
        database_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a page in a local 'database'.

        Args:
            database_id: The database ID (creates if doesn't exist).
            properties: Page properties.

        Returns:
            The created page object with a local ID.
        """
        databases = self._load_databases()

        if database_id not in databases:
            databases[database_id] = {"pages": {}}

        page_id = f"local_page_{uuid.uuid4().hex[:12]}"
        page = {
            "id": page_id,
            "parent": {"database_id": database_id},
            "properties": properties,
            "created_time": datetime.now(timezone.utc).isoformat(),
            "last_edited_time": datetime.now(timezone.utc).isoformat(),
        }

        databases[database_id]["pages"][page_id] = page
        self._save_databases(databases)

        logger.info("Created local Notion page: %s in database %s", page_id, database_id)
        return page

    async def update_page(
        self,
        page_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Update an existing local page.

        Args:
            page_id: The page ID.
            properties: Updated properties.

        Returns:
            The updated page object.
        """
        databases = self._load_databases()

        # Find the page across all databases
        for db_id, db_data in databases.items():
            pages = db_data.get("pages", {})
            if page_id in pages:
                pages[page_id]["properties"].update(properties)
                pages[page_id]["last_edited_time"] = datetime.now(timezone.utc).isoformat()
                self._save_databases(databases)
                logger.info("Updated local Notion page: %s", page_id)
                return pages[page_id]

        raise KeyError(f"Page not found: {page_id}")

    async def query_database(
        self,
        database_id: str,
        filter_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query a local database.

        Args:
            database_id: The database ID.
            filter_params: Ignored in local mode (returns all pages).

        Returns:
            List of pages in the database.
        """
        databases = self._load_databases()

        if database_id not in databases:
            return []

        pages = databases[database_id].get("pages", {})
        return list(pages.values())
