"""Tests for Google Calendar OAuth2 authentication and token management.

Tests cover:
- OAuth2 authorization URL generation
- Authorization code exchange
- Encrypted token storage and retrieval
- Token refresh logic (proactive pre-expiry refresh)
- Token revocation and cleanup
- Auth status tracking
- Edge cases (missing tokens, expired tokens, no refresh token)
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blurt.core.config import Settings
from blurt.core.encryption import CredentialEncryptor
from blurt.integrations.google_calendar.auth import (
    GoogleCalendarAuth,
)
from blurt.integrations.google_calendar.models import (
    AuthStatus,
    OAuthClientConfig,
    OAuthToken,
)


# --- Fixtures ---


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def test_settings(tmp_dir: Path) -> Settings:
    """Create test settings with a temp directory."""
    return Settings(
        data_dir=tmp_dir,
        google_client_id="test-client-id",
        google_client_secret="test-client-secret",
        google_redirect_uri="http://localhost:8080/auth/google/callback",
        encryption_key_path=tmp_dir / "test.key",
    )


@pytest.fixture
def test_encryptor() -> CredentialEncryptor:
    """Create a test encryptor with an ephemeral key."""
    return CredentialEncryptor(master_key=b"test-master-key-32-bytes-long!!")


@pytest.fixture
def test_client_config() -> OAuthClientConfig:
    """Create a test OAuth client config."""
    return OAuthClientConfig(
        client_id="test-client-id",
        client_secret="test-client-secret",
        redirect_uri="http://localhost:8080/auth/google/callback",
        scopes=[
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/calendar.events",
        ],
    )


@pytest.fixture
def auth_service(
    test_settings: Settings,
    test_encryptor: CredentialEncryptor,
    test_client_config: OAuthClientConfig,
) -> GoogleCalendarAuth:
    """Create a fully configured auth service for testing."""
    return GoogleCalendarAuth(
        settings=test_settings,
        encryptor=test_encryptor,
        client_config=test_client_config,
    )


@pytest.fixture
def sample_token() -> OAuthToken:
    """Create a sample non-expired token."""
    return OAuthToken(
        access_token="ya29.test-access-token",
        refresh_token="1//test-refresh-token",
        token_type="Bearer",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        scopes=[
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/calendar.events",
        ],
    )


@pytest.fixture
def expired_token() -> OAuthToken:
    """Create an expired token with a refresh token."""
    return OAuthToken(
        access_token="ya29.expired-access-token",
        refresh_token="1//test-refresh-token",
        token_type="Bearer",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        scopes=[
            "https://www.googleapis.com/auth/calendar",
        ],
    )


# --- OAuthToken model tests ---


class TestOAuthToken:
    """Tests for the OAuthToken model."""

    def test_is_expired_when_no_expiry(self):
        """Token with no expiry is considered expired."""
        token = OAuthToken(access_token="test", expires_at=None)
        assert token.is_expired is True

    def test_is_expired_when_past(self):
        """Token with past expiry is expired."""
        token = OAuthToken(
            access_token="test",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert token.is_expired is True

    def test_is_not_expired_when_future(self):
        """Token with future expiry is not expired."""
        token = OAuthToken(
            access_token="test",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert token.is_expired is False

    def test_to_storage_dict(self, sample_token: OAuthToken):
        """Conversion to storage dict includes all fields."""
        d = sample_token.to_storage_dict()
        assert d["access_token"] == sample_token.access_token
        assert d["refresh_token"] == sample_token.refresh_token
        assert d["token_type"] == "Bearer"
        assert d["scopes"] == sample_token.scopes
        assert d["expires_at"] is not None

    def test_roundtrip_storage(self, sample_token: OAuthToken):
        """Token survives to_storage_dict -> from_storage_dict roundtrip."""
        d = sample_token.to_storage_dict()
        restored = OAuthToken.from_storage_dict(d)
        assert restored.access_token == sample_token.access_token
        assert restored.refresh_token == sample_token.refresh_token
        assert restored.scopes == sample_token.scopes


# --- OAuthClientConfig tests ---


class TestOAuthClientConfig:
    """Tests for OAuthClientConfig."""

    def test_from_credentials_file_web(self, tmp_dir: Path):
        """Load config from a 'web' format credentials file."""
        creds_file = tmp_dir / "creds.json"
        creds_file.write_text(
            json.dumps(
                {
                    "web": {
                        "client_id": "web-client-id",
                        "client_secret": "web-secret",
                        "redirect_uris": ["http://localhost:8080/callback"],
                    }
                }
            )
        )
        config = OAuthClientConfig.from_credentials_file(str(creds_file))
        assert config.client_id == "web-client-id"
        assert config.client_secret == "web-secret"

    def test_from_credentials_file_installed(self, tmp_dir: Path):
        """Load config from an 'installed' format credentials file."""
        creds_file = tmp_dir / "creds.json"
        creds_file.write_text(
            json.dumps(
                {
                    "installed": {
                        "client_id": "installed-client-id",
                        "client_secret": "installed-secret",
                        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
                    }
                }
            )
        )
        config = OAuthClientConfig.from_credentials_file(str(creds_file))
        assert config.client_id == "installed-client-id"

    def test_from_credentials_file_not_found(self):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            OAuthClientConfig.from_credentials_file("/nonexistent/creds.json")

    def test_from_credentials_file_invalid_format(self, tmp_dir: Path):
        """Raises ValueError for invalid format."""
        creds_file = tmp_dir / "bad.json"
        creds_file.write_text(json.dumps({"invalid": {}}))
        with pytest.raises(ValueError, match="Invalid Google credentials"):
            OAuthClientConfig.from_credentials_file(str(creds_file))


# --- Encrypted token storage tests ---


class TestEncryptedTokenStorage:
    """Tests for encrypted token persistence."""

    @pytest.mark.asyncio
    async def test_store_and_load_token(
        self,
        auth_service: GoogleCalendarAuth,
        sample_token: OAuthToken,
    ):
        """Token is encrypted to disk and can be decrypted back."""
        user_id = "user-123"
        await auth_service._store_token(user_id, sample_token)

        loaded = await auth_service._load_token(user_id)
        assert loaded.access_token == sample_token.access_token
        assert loaded.refresh_token == sample_token.refresh_token

    @pytest.mark.asyncio
    async def test_stored_token_is_encrypted(
        self,
        auth_service: GoogleCalendarAuth,
        sample_token: OAuthToken,
    ):
        """Stored token file contains encrypted data, not plaintext."""
        user_id = "user-enc-check"
        await auth_service._store_token(user_id, sample_token)

        token_path = auth_service._get_token_path(user_id)
        raw_bytes = token_path.read_bytes()

        # The raw bytes should NOT contain the access token in plaintext
        assert sample_token.access_token.encode() not in raw_bytes
        assert sample_token.refresh_token is not None
        assert sample_token.refresh_token.encode() not in raw_bytes

    @pytest.mark.asyncio
    async def test_load_nonexistent_token_raises(
        self,
        auth_service: GoogleCalendarAuth,
    ):
        """Loading a token for a user with no stored token raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No stored token"):
            await auth_service._load_token("nonexistent-user")

    @pytest.mark.asyncio
    async def test_token_file_permissions(
        self,
        auth_service: GoogleCalendarAuth,
        sample_token: OAuthToken,
    ):
        """Stored token file has restrictive permissions (0o600)."""
        user_id = "user-perms"
        await auth_service._store_token(user_id, sample_token)

        token_path = auth_service._get_token_path(user_id)
        mode = oct(token_path.stat().st_mode & 0o777)
        assert mode == "0o600"

    @pytest.mark.asyncio
    async def test_delete_token_file(
        self,
        auth_service: GoogleCalendarAuth,
        sample_token: OAuthToken,
    ):
        """Token file is deleted from disk."""
        user_id = "user-delete"
        await auth_service._store_token(user_id, sample_token)

        token_path = auth_service._get_token_path(user_id)
        assert token_path.exists()

        auth_service._delete_token_file(user_id)
        assert not token_path.exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_token_is_safe(
        self,
        auth_service: GoogleCalendarAuth,
    ):
        """Deleting a token that doesn't exist doesn't raise."""
        auth_service._delete_token_file("nonexistent-user")  # Should not raise

    @pytest.mark.asyncio
    async def test_different_users_have_separate_tokens(
        self,
        auth_service: GoogleCalendarAuth,
    ):
        """Each user has their own encrypted token file."""
        token_a = OAuthToken(
            access_token="token-a", refresh_token="refresh-a",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        token_b = OAuthToken(
            access_token="token-b", refresh_token="refresh-b",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        await auth_service._store_token("user-a", token_a)
        await auth_service._store_token("user-b", token_b)

        loaded_a = await auth_service._load_token("user-a")
        loaded_b = await auth_service._load_token("user-b")

        assert loaded_a.access_token == "token-a"
        assert loaded_b.access_token == "token-b"


# --- Authorization URL tests ---


class TestGetAuthorizationUrl:
    """Tests for authorization URL generation."""

    def test_generates_url(self, auth_service: GoogleCalendarAuth):
        """Generates a valid Google authorization URL."""
        url = auth_service.get_authorization_url("user-123")
        assert "accounts.google.com" in url
        assert "client_id=test-client-id" in url
        assert "access_type=offline" in url

    def test_uses_provided_state(self, auth_service: GoogleCalendarAuth):
        """Uses the provided state parameter."""
        url = auth_service.get_authorization_url("user-123", state="my-csrf-state")
        assert "state=my-csrf-state" in url

    def test_auto_generates_state(self, auth_service: GoogleCalendarAuth):
        """Generates a random state when not provided."""
        url = auth_service.get_authorization_url("user-123")
        assert "state=" in url

    def test_sets_awaiting_auth_status(self, auth_service: GoogleCalendarAuth):
        """Sets the auth state to AWAITING_AUTH after URL generation."""
        auth_service.get_authorization_url("user-123")
        status = auth_service.get_auth_status("user-123")
        assert status.status == AuthStatus.AWAITING_AUTH
        assert status.auth_url is not None

    def test_requests_offline_access(self, auth_service: GoogleCalendarAuth):
        """Requests offline access to get a refresh token."""
        url = auth_service.get_authorization_url("user-123")
        assert "access_type=offline" in url

    def test_forces_consent_prompt(self, auth_service: GoogleCalendarAuth):
        """Forces consent prompt to ensure refresh token is returned."""
        url = auth_service.get_authorization_url("user-123")
        assert "prompt=consent" in url


# --- Auth status tests ---


class TestGetAuthStatus:
    """Tests for auth status checking."""

    def test_not_configured_by_default(self, auth_service: GoogleCalendarAuth):
        """New users have NOT_CONFIGURED status."""
        status = auth_service.get_auth_status("new-user")
        assert status.status == AuthStatus.NOT_CONFIGURED
        assert status.user_id == "new-user"

    @pytest.mark.asyncio
    async def test_authenticated_with_stored_token(
        self,
        auth_service: GoogleCalendarAuth,
        sample_token: OAuthToken,
    ):
        """Users with stored tokens show AUTHENTICATED."""
        await auth_service._store_token("stored-user", sample_token)
        status = auth_service.get_auth_status("stored-user")
        assert status.status == AuthStatus.AUTHENTICATED

    def test_has_stored_token(
        self,
        auth_service: GoogleCalendarAuth,
    ):
        """has_stored_token returns False for new users."""
        assert auth_service.has_stored_token("new-user") is False


# --- Token refresh tests ---


class TestTokenRefresh:
    """Tests for automatic token refresh."""

    @pytest.mark.asyncio
    async def test_get_valid_token_returns_fresh_token(
        self,
        auth_service: GoogleCalendarAuth,
        sample_token: OAuthToken,
    ):
        """get_valid_token returns the token when it's still valid."""
        user_id = "user-fresh"
        await auth_service._store_token(user_id, sample_token)

        result = await auth_service.get_valid_token(user_id)
        assert result.access_token == sample_token.access_token

    @pytest.mark.asyncio
    async def test_get_valid_token_refreshes_expired(
        self,
        auth_service: GoogleCalendarAuth,
        expired_token: OAuthToken,
    ):
        """get_valid_token automatically refreshes expired tokens."""
        user_id = "user-expired"
        await auth_service._store_token(user_id, expired_token)

        # Mock the HTTP refresh call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "ya29.new-access-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await auth_service.get_valid_token(user_id)
            assert result.access_token == "ya29.new-access-token"
            assert result.refresh_token == expired_token.refresh_token

    @pytest.mark.asyncio
    async def test_proactive_refresh_before_expiry(
        self,
        auth_service: GoogleCalendarAuth,
    ):
        """Refreshes token that's within the refresh buffer window."""
        # Token expires in 2 minutes (within the 5-minute buffer)
        soon_expiring_token = OAuthToken(
            access_token="ya29.soon-expiring",
            refresh_token="1//refresh",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
        )
        user_id = "user-soon-expiry"
        await auth_service._store_token(user_id, soon_expiring_token)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "ya29.refreshed",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await auth_service.get_valid_token(user_id)
            assert result.access_token == "ya29.refreshed"

    @pytest.mark.asyncio
    async def test_refresh_without_refresh_token_raises(
        self,
        auth_service: GoogleCalendarAuth,
    ):
        """Refresh fails if no refresh_token is available."""
        token_no_refresh = OAuthToken(
            access_token="ya29.no-refresh",
            refresh_token=None,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        user_id = "user-no-refresh"
        await auth_service._store_token(user_id, token_no_refresh)

        with pytest.raises(ValueError, match="No refresh token"):
            await auth_service.get_valid_token(user_id)

        status = auth_service.get_auth_status(user_id)
        assert status.status == AuthStatus.REFRESH_FAILED

    @pytest.mark.asyncio
    async def test_refresh_preserves_refresh_token_when_not_returned(
        self,
        auth_service: GoogleCalendarAuth,
        expired_token: OAuthToken,
    ):
        """Keeps existing refresh_token when Google doesn't return a new one."""
        user_id = "user-keep-refresh"
        await auth_service._store_token(user_id, expired_token)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "ya29.new",
            "expires_in": 3600,
            # No refresh_token in response
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await auth_service.get_valid_token(user_id)
            # Original refresh token should be preserved
            assert result.refresh_token == expired_token.refresh_token


# --- Token revocation tests ---


class TestTokenRevocation:
    """Tests for token revocation."""

    @pytest.mark.asyncio
    async def test_revoke_existing_token(
        self,
        auth_service: GoogleCalendarAuth,
        sample_token: OAuthToken,
    ):
        """Revoking an existing token deletes the file and calls Google."""
        user_id = "user-revoke"
        await auth_service._store_token(user_id, sample_token)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await auth_service.revoke_token(user_id)

        assert result is True
        assert not auth_service.has_stored_token(user_id)

        status = auth_service.get_auth_status(user_id)
        assert status.status == AuthStatus.REVOKED

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_token(
        self,
        auth_service: GoogleCalendarAuth,
    ):
        """Revoking a nonexistent token succeeds gracefully."""
        result = await auth_service.revoke_token("no-such-user")
        assert result is True

        status = auth_service.get_auth_status("no-such-user")
        assert status.status == AuthStatus.NOT_CONFIGURED

    @pytest.mark.asyncio
    async def test_revoke_cleans_up_even_on_network_error(
        self,
        auth_service: GoogleCalendarAuth,
        sample_token: OAuthToken,
    ):
        """Token file is deleted even if Google revocation fails."""
        user_id = "user-net-fail"
        await auth_service._store_token(user_id, sample_token)

        import httpx as httpx_mod

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=httpx_mod.RequestError("Connection failed")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await auth_service.revoke_token(user_id)

        # Should still succeed and clean up locally
        assert result is True
        assert not auth_service.has_stored_token(user_id)


# --- Google Credentials conversion tests ---


class TestGoogleCredentials:
    """Tests for converting to Google API credentials."""

    def test_converts_to_google_credentials(
        self,
        auth_service: GoogleCalendarAuth,
        sample_token: OAuthToken,
    ):
        """Converts OAuthToken to Google API Credentials object."""
        creds = auth_service.get_google_credentials(sample_token)
        assert creds.token == sample_token.access_token
        assert creds.refresh_token == sample_token.refresh_token
        assert creds.client_id == "test-client-id"
        assert creds.client_secret == "test-client-secret"


# --- Client config loading tests ---


class TestClientConfigLoading:
    """Tests for lazy client config loading."""

    def test_raises_when_not_configured(self):
        """Raises ValueError when no Google credentials are configured."""
        settings = Settings(
            data_dir=Path("/tmp/blurt-test"),
            google_client_id="",
            google_client_secret="",
        )
        auth = GoogleCalendarAuth(
            settings=settings,
            encryptor=CredentialEncryptor(),
            client_config=None,
        )
        with pytest.raises(ValueError, match="Google OAuth2 not configured"):
            _ = auth.client_config

    def test_loads_from_settings(self, test_settings: Settings):
        """Loads client config from settings when available."""
        auth = GoogleCalendarAuth(
            settings=test_settings,
            encryptor=CredentialEncryptor(),
        )
        config = auth.client_config
        assert config.client_id == "test-client-id"
        assert config.client_secret == "test-client-secret"


# --- Encryption integration tests ---


class TestEncryptionIntegration:
    """Integration tests for the encryption layer."""

    def test_encryptor_roundtrip(self):
        """CredentialEncryptor encrypts and decrypts data correctly."""
        encryptor = CredentialEncryptor(master_key=b"test-key-for-roundtrip-test!!!!!")
        original = {
            "access_token": "secret-token",
            "refresh_token": "secret-refresh",
            "scopes": ["scope1", "scope2"],
        }
        encrypted = encryptor.encrypt(original)
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == original

    def test_different_keys_cannot_decrypt(self):
        """Data encrypted with one key cannot be decrypted with another."""
        enc1 = CredentialEncryptor(master_key=b"key-one-32-bytes-long-padding!!!")
        enc2 = CredentialEncryptor(master_key=b"key-two-32-bytes-long-padding!!!")

        encrypted = enc1.encrypt({"secret": "data"})
        with pytest.raises(ValueError, match="Decryption failed"):
            enc2.decrypt(encrypted)

    def test_encrypt_to_file_and_back(self, tmp_dir: Path):
        """File-based encrypt/decrypt roundtrip works."""
        encryptor = CredentialEncryptor(master_key=b"file-test-key-32-bytes-padding!!")
        data = {"token": "my-secret-token"}
        file_path = tmp_dir / "test.enc"

        encryptor.encrypt_to_file(data, file_path)
        assert file_path.exists()

        # Verify file is encrypted (doesn't contain plaintext)
        raw = file_path.read_bytes()
        assert b"my-secret-token" not in raw

        decrypted = encryptor.decrypt_from_file(file_path)
        assert decrypted == data

    def test_key_generation_and_persistence(self, tmp_dir: Path):
        """Master key is generated and persisted correctly."""
        key_path = tmp_dir / "master.key"
        enc1 = CredentialEncryptor(key_path=key_path)
        assert key_path.exists()

        # Same key path should load the same key
        enc2 = CredentialEncryptor(key_path=key_path)

        data = {"test": "value"}
        enc1.encrypt(data)
        # Different salt means we can't decrypt with enc2.decrypt directly,
        # but encrypt_to_file + decrypt_from_file should work
        file_path = tmp_dir / "roundtrip.enc"
        enc1.encrypt_to_file(data, file_path)
        decrypted = enc2.decrypt_from_file(file_path)
        assert decrypted == data
