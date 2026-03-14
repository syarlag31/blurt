"""API routes for Google Calendar OAuth2 authentication.

Provides endpoints for:
- Initiating the OAuth2 consent flow
- Handling the OAuth2 callback
- Checking connection status
- Disconnecting (revoking) the integration
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from blurt.integrations.google_calendar.auth import GoogleCalendarAuth
from blurt.integrations.google_calendar.models import AuthStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth/google", tags=["google-calendar-auth"])

# Module-level auth service instance (initialized by app startup)
_auth_service: GoogleCalendarAuth | None = None


def get_auth_service() -> GoogleCalendarAuth:
    """Get the auth service, creating if needed."""
    global _auth_service
    if _auth_service is None:
        _auth_service = GoogleCalendarAuth()
    return _auth_service


def set_auth_service(service: GoogleCalendarAuth) -> None:
    """Set the auth service (for dependency injection in tests)."""
    global _auth_service
    _auth_service = service


# --- Request/Response models ---


class ConnectResponse(BaseModel):
    """Response for the connect endpoint."""

    authorization_url: str
    state: str


class CallbackRequest(BaseModel):
    """Request body for the OAuth callback."""

    code: str
    state: str | None = None


class StatusResponse(BaseModel):
    """Response for the status endpoint."""

    user_id: str
    status: AuthStatus
    connected_email: str | None = None
    last_error: str | None = None


class DisconnectResponse(BaseModel):
    """Response for the disconnect endpoint."""

    success: bool
    message: str


# --- Routes ---


@router.get("/connect", response_model=ConnectResponse)
async def connect_google_calendar(
    user_id: str = Query(..., description="The Blurt user ID"),
) -> ConnectResponse:
    """Initiate Google Calendar OAuth2 connection.

    Returns an authorization URL that the client should open
    in the user's browser for consent.
    """
    auth = get_auth_service()
    try:
        import secrets

        state = secrets.token_urlsafe(32)
        auth_url = auth.get_authorization_url(user_id, state=state)
        return ConnectResponse(authorization_url=auth_url, state=state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/callback", response_model=StatusResponse)
async def handle_oauth_callback(
    user_id: str = Query(..., description="The Blurt user ID"),
    code: str = Query(..., description="Authorization code from Google"),
) -> StatusResponse:
    """Handle the OAuth2 callback after user consent.

    Exchanges the authorization code for tokens and stores them
    encrypted. Returns the connection status.
    """
    auth = get_auth_service()
    try:
        await auth.exchange_code(user_id, code)
        status = auth.get_auth_status(user_id)
        return StatusResponse(
            user_id=status.user_id,
            status=status.status,
            connected_email=status.connected_email,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/status", response_model=StatusResponse)
async def get_connection_status(
    user_id: str = Query(..., description="The Blurt user ID"),
) -> StatusResponse:
    """Check the current Google Calendar connection status."""
    auth = get_auth_service()
    status = auth.get_auth_status(user_id)
    return StatusResponse(
        user_id=status.user_id,
        status=status.status,
        connected_email=status.connected_email,
        last_error=status.last_error,
    )


@router.post("/disconnect", response_model=DisconnectResponse)
async def disconnect_google_calendar(
    user_id: str = Query(..., description="The Blurt user ID"),
) -> DisconnectResponse:
    """Disconnect Google Calendar integration.

    Revokes the OAuth token at Google and deletes the local encrypted
    token file.
    """
    auth = get_auth_service()
    try:
        success = await auth.revoke_token(user_id)
        return DisconnectResponse(
            success=success,
            message="Google Calendar disconnected successfully."
            if success
            else "Disconnection completed with warnings.",
        )
    except Exception as e:
        logger.error("Error disconnecting user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=str(e)) from e
