"""Notion API client for Blurt — cloud mode.

Wraps the Notion API to sync pages, tasks, and notes from Blurt
to Notion databases. Uses the official Notion API v1.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_API_VERSION = "2022-06-28"


class NotionAPIError(Exception):
    """Base exception for Notion API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class NotionAPIClient:
    """Async client for the Notion API.

    Provides create, update, and query operations for Notion pages
    and databases.
    """

    def __init__(
        self,
        api_token: str,
        base_url: str = NOTION_API_BASE,
    ) -> None:
        self._api_token = api_token
        self._base_url = base_url
        self._http: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Initialize the HTTP client."""
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_token}",
                "Content-Type": "application/json",
                "Notion-Version": NOTION_API_VERSION,
            },
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
        )
        logger.info("Notion API client connected")

    async def close(self) -> None:
        """Shut down the HTTP client."""
        if self._http:
            await self._http.aclose()
            self._http = None
        logger.info("Notion API client closed")

    async def __aenter__(self) -> NotionAPIClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def create_page(
        self,
        database_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a page in a Notion database.

        Args:
            database_id: The ID of the database to add the page to.
            properties: Page properties in Notion API format.

        Returns:
            The created page object from Notion.
        """
        if not self._http:
            raise RuntimeError("Client not connected — call connect() first")

        body = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }

        response = await self._http.post("/pages", json=body)
        if response.status_code != 200:
            raise NotionAPIError(
                f"Failed to create page: {response.text}",
                status_code=response.status_code,
            )
        return response.json()

    async def update_page(
        self,
        page_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Update an existing Notion page.

        Args:
            page_id: The ID of the page to update.
            properties: Updated properties.

        Returns:
            The updated page object.
        """
        if not self._http:
            raise RuntimeError("Client not connected — call connect() first")

        response = await self._http.patch(
            f"/pages/{page_id}",
            json={"properties": properties},
        )
        if response.status_code != 200:
            raise NotionAPIError(
                f"Failed to update page: {response.text}",
                status_code=response.status_code,
            )
        return response.json()

    async def query_database(
        self,
        database_id: str,
        filter_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query a Notion database.

        Args:
            database_id: The database ID to query.
            filter_params: Optional Notion filter object.

        Returns:
            List of matching page objects.
        """
        if not self._http:
            raise RuntimeError("Client not connected — call connect() first")

        body: dict[str, Any] = {}
        if filter_params:
            body["filter"] = filter_params

        response = await self._http.post(
            f"/databases/{database_id}/query",
            json=body,
        )
        if response.status_code != 200:
            raise NotionAPIError(
                f"Failed to query database: {response.text}",
                status_code=response.status_code,
            )
        data = response.json()
        return data.get("results", [])
