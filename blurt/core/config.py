"""Blurt configuration management.

Centralizes all configuration with environment variable overrides.
Supports both cloud and local-only deployment modes.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class DeploymentMode(str, Enum):
    """Deployment mode - cloud-first or local-only."""

    CLOUD = "cloud"
    LOCAL = "local"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Deployment
    deployment_mode: DeploymentMode = DeploymentMode.CLOUD

    # Data storage
    data_dir: Path = Field(
        default_factory=lambda: Path(os.environ.get("BLURT_DATA_DIR", "~/.blurt")).expanduser()
    )

    # Encryption
    encryption_key_path: Path | None = None
    encryption_enabled: bool = True

    # Google Calendar OAuth2
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8080/auth/google/callback"
    google_scopes: list[str] = Field(
        default=[
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/calendar.events",
        ]
    )
    google_credentials_file: str = ""  # Path to OAuth client secrets JSON

    # API Server
    api_host: str = "127.0.0.1"
    api_port: int = 8080

    model_config = {
        "env_prefix": "BLURT_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def get_credentials_dir(self) -> Path:
        """Get the directory for storing encrypted credentials."""
        creds_dir = self.data_dir / "credentials"
        creds_dir.mkdir(parents=True, exist_ok=True)
        return creds_dir

    def get_tokens_dir(self) -> Path:
        """Get the directory for storing encrypted OAuth tokens."""
        tokens_dir = self.data_dir / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        return tokens_dir


# Singleton settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
