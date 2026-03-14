"""Data models for Google Calendar OAuth2 credentials and tokens."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class AuthStatus(str, Enum):
    """OAuth2 authentication status."""

    NOT_CONFIGURED = "not_configured"
    AWAITING_AUTH = "awaiting_auth"
    AUTHENTICATED = "authenticated"
    TOKEN_EXPIRED = "token_expired"
    REFRESH_FAILED = "refresh_failed"
    REVOKED = "revoked"


class OAuthToken(BaseModel):
    """Google OAuth2 token data."""

    access_token: str
    refresh_token: str | None = None
    token_type: str = "Bearer"
    expires_at: datetime | None = None
    scopes: list[str] = Field(default_factory=list)
    id_token: str | None = None

    @property
    def is_expired(self) -> bool:
        """Check if the access token has expired."""
        if self.expires_at is None:
            return True
        now = datetime.now(timezone.utc)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return now >= expires

    def to_storage_dict(self) -> dict:
        """Convert to dictionary for encrypted storage."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scopes": self.scopes,
            "id_token": self.id_token,
        }

    @classmethod
    def from_storage_dict(cls, data: dict) -> OAuthToken:
        """Reconstruct from storage dictionary."""
        expires_at = data.get("expires_at")
        if expires_at and isinstance(expires_at, str):
            data["expires_at"] = datetime.fromisoformat(expires_at)
        return cls(**data)


class OAuthClientConfig(BaseModel):
    """Google OAuth2 client configuration."""

    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: list[str]

    @classmethod
    def from_credentials_file(cls, file_path: str) -> OAuthClientConfig:
        """Load client config from a Google OAuth client secrets JSON file."""
        import json
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Credentials file not found: {file_path}")

        with open(path) as f:
            data = json.load(f)

        # Google's client_secrets.json has either "web" or "installed" key
        if "web" in data:
            client_data = data["web"]
        elif "installed" in data:
            client_data = data["installed"]
        else:
            raise ValueError("Invalid Google credentials file format")

        return cls(
            client_id=client_data["client_id"],
            client_secret=client_data["client_secret"],
            redirect_uri=client_data.get("redirect_uris", [""])[0],
            scopes=[
                "https://www.googleapis.com/auth/calendar",
                "https://www.googleapis.com/auth/calendar.events",
            ],
        )


class AuthState(BaseModel):
    """Current authentication state for a user's Google Calendar connection."""

    user_id: str
    status: AuthStatus = AuthStatus.NOT_CONFIGURED
    auth_url: str | None = None
    last_refresh: datetime | None = None
    last_error: str | None = None
    connected_email: str | None = None
