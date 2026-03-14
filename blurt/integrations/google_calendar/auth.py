"""Google Calendar OAuth2 authentication and token management.

Handles the full OAuth2 lifecycle:
1. Generate authorization URL for user consent
2. Exchange authorization code for tokens
3. Securely store encrypted tokens (E2E encryption at rest)
4. Automatic token refresh before expiry
5. Token revocation on disconnect

All tokens are encrypted at rest using the core encryption module.
Works identically in cloud and local-only deployment modes.
No data leakage vectors — encryption is always on by default.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from google.oauth2.credentials import Credentials as GoogleCredentials
from google_auth_oauthlib.flow import Flow

from blurt.core.config import Settings, get_settings
from blurt.core.encryption import CredentialEncryptor
from blurt.integrations.google_calendar.models import (
    AuthState,
    AuthStatus,
    OAuthClientConfig,
    OAuthToken,
)

logger = logging.getLogger(__name__)

# Token refresh buffer — refresh 5 minutes before actual expiry
TOKEN_REFRESH_BUFFER = timedelta(minutes=5)

# Google OAuth2 endpoints
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# Default scopes
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
]


class GoogleCalendarAuth:
    """Manages Google Calendar OAuth2 authentication lifecycle.

    Provides:
    - Authorization URL generation for user consent (web OAuth2 flow)
    - Authorization code exchange for access + refresh tokens
    - Encrypted per-user token persistence
    - Automatic token refresh with proactive pre-expiry renewal
    - Token revocation and cleanup
    - Auth state tracking per user

    All tokens are encrypted at rest using Fernet (AES-128-CBC) via
    the CredentialEncryptor. Supports both cloud and local-only modes
    with identical security guarantees.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        encryptor: CredentialEncryptor | None = None,
        client_config: OAuthClientConfig | None = None,
    ):
        """Initialize the auth service.

        Args:
            settings: App settings. Uses global settings if None.
            encryptor: Credential encryptor. Creates one from settings if None.
            client_config: OAuth client config. Loaded from settings if None.
        """
        self._settings = settings or get_settings()
        self._encryptor = encryptor or CredentialEncryptor(
            key_path=self._settings.encryption_key_path
            or (self._settings.data_dir / "master.key")
        )
        self._client_config = client_config
        self._auth_states: dict[str, AuthState] = {}

    @property
    def client_config(self) -> OAuthClientConfig:
        """Get the OAuth client configuration, loading lazily.

        Raises:
            ValueError: If Google OAuth2 is not configured.
        """
        if self._client_config is None:
            if self._settings.google_credentials_file:
                self._client_config = OAuthClientConfig.from_credentials_file(
                    self._settings.google_credentials_file
                )
            elif self._settings.google_client_id and self._settings.google_client_secret:
                self._client_config = OAuthClientConfig(
                    client_id=self._settings.google_client_id,
                    client_secret=self._settings.google_client_secret,
                    redirect_uri=self._settings.google_redirect_uri,
                    scopes=self._settings.google_scopes,
                )
            else:
                raise ValueError(
                    "Google OAuth2 not configured. Set BLURT_GOOGLE_CLIENT_ID and "
                    "BLURT_GOOGLE_CLIENT_SECRET, or provide a credentials file via "
                    "BLURT_GOOGLE_CREDENTIALS_FILE."
                )
        return self._client_config

    def _get_token_path(self, user_id: str) -> Path:
        """Get the encrypted token file path for a user."""
        return self._settings.get_tokens_dir() / f"google_calendar_{user_id}.enc"

    def _build_flow(self, state: str | None = None) -> Flow:
        """Build a Google OAuth2 web flow from client config.

        Args:
            state: Optional state parameter for CSRF protection.

        Returns:
            Configured OAuth2 Flow instance.
        """
        config = self.client_config
        client_config = {
            "web": {
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": GOOGLE_TOKEN_URL,
                "redirect_uris": [config.redirect_uri],
            }
        }
        flow = Flow.from_client_config(
            client_config,
            scopes=config.scopes,
            state=state,
        )
        flow.redirect_uri = config.redirect_uri
        return flow

    def get_authorization_url(self, user_id: str, state: str | None = None) -> str:
        """Generate the Google OAuth2 authorization URL.

        Creates a consent URL that the user opens in their browser to grant
        Blurt access to their Google Calendar. Uses the web OAuth2 flow
        (not installed app flow) for server-side compatibility.

        Args:
            user_id: The Blurt user ID initiating the connection.
            state: Optional CSRF state token. Generated if not provided.

        Returns:
            Authorization URL to redirect the user to.
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        flow = self._build_flow(state=state)
        auth_url, _ = flow.authorization_url(
            access_type="offline",  # Request refresh token
            include_granted_scopes="true",
            prompt="consent",  # Always prompt to ensure we get refresh token
        )

        # Track auth state
        self._auth_states[user_id] = AuthState(
            user_id=user_id,
            status=AuthStatus.AWAITING_AUTH,
            auth_url=auth_url,
        )

        logger.info("Generated authorization URL for user %s", user_id)
        return auth_url

    async def exchange_code(self, user_id: str, authorization_code: str) -> OAuthToken:
        """Exchange an authorization code for OAuth2 access and refresh tokens.

        This is called after the user completes the consent flow and Google
        redirects back with an authorization code.

        Args:
            user_id: The Blurt user ID.
            authorization_code: The authorization code from Google's callback.

        Returns:
            The obtained OAuth token with access_token and refresh_token.

        Raises:
            ValueError: If the exchange fails.
        """
        flow = self._build_flow()

        try:
            flow.fetch_token(code=authorization_code)
        except Exception as e:
            self._auth_states[user_id] = AuthState(
                user_id=user_id,
                status=AuthStatus.REFRESH_FAILED,
                last_error=str(e),
            )
            raise ValueError(f"Failed to exchange authorization code: {e}") from e

        credentials = flow.credentials
        token = OAuthToken(
            access_token=credentials.token or "",
            refresh_token=credentials.refresh_token,
            token_type="Bearer",
            expires_at=credentials.expiry.replace(tzinfo=timezone.utc)
            if credentials.expiry
            else None,
            scopes=list(credentials.scopes or []),
            id_token=getattr(credentials, "id_token", None),
        )

        # Encrypt and persist the token
        await self._store_token(user_id, token)

        # Fetch connected email for status display
        connected_email = await self._fetch_user_email(token.access_token)

        self._auth_states[user_id] = AuthState(
            user_id=user_id,
            status=AuthStatus.AUTHENTICATED,
            last_refresh=datetime.now(timezone.utc),
            connected_email=connected_email,
        )

        logger.info("Successfully authenticated user %s (email: %s)", user_id, connected_email)
        return token

    async def get_valid_token(self, user_id: str) -> OAuthToken:
        """Get a valid (non-expired) OAuth token for a user.

        Automatically refreshes the token if it's expired or about to expire
        within the TOKEN_REFRESH_BUFFER window (5 minutes).

        Args:
            user_id: The Blurt user ID.

        Returns:
            A valid OAuth token ready for API calls.

        Raises:
            FileNotFoundError: If no token exists for the user.
            ValueError: If token refresh fails.
        """
        token = await self._load_token(user_id)

        # Proactively refresh before expiry
        if token.expires_at is not None:
            now = datetime.now(timezone.utc)
            # Ensure expires_at is timezone-aware for comparison
            expires_at = token.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            refresh_threshold = now + TOKEN_REFRESH_BUFFER
            if expires_at <= refresh_threshold:
                logger.info("Token for user %s is expiring, refreshing...", user_id)
                token = await self._refresh_token(user_id, token)
        elif token.is_expired:
            token = await self._refresh_token(user_id, token)

        return token

    async def _refresh_token(self, user_id: str, token: OAuthToken) -> OAuthToken:
        """Refresh an expired OAuth token using the refresh_token grant.

        Args:
            user_id: The Blurt user ID.
            token: The current (expired) token containing a refresh_token.

        Returns:
            The refreshed OAuth token with a new access_token.

        Raises:
            ValueError: If refresh fails (no refresh token, revoked, network error).
        """
        if not token.refresh_token:
            self._auth_states[user_id] = AuthState(
                user_id=user_id,
                status=AuthStatus.REFRESH_FAILED,
                last_error="No refresh token available",
            )
            raise ValueError(
                "No refresh token available. User must re-authenticate via "
                "get_authorization_url()."
            )

        config = self.client_config

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    GOOGLE_TOKEN_URL,
                    data={
                        "client_id": config.client_id,
                        "client_secret": config.client_secret,
                        "refresh_token": token.refresh_token,
                        "grant_type": "refresh_token",
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                error_body = ""
                try:
                    error_body = e.response.text
                except Exception:
                    pass
                error_msg = (
                    f"Token refresh failed (HTTP {e.response.status_code}): {error_body}"
                )
                logger.error("%s for user %s", error_msg, user_id)
                self._auth_states[user_id] = AuthState(
                    user_id=user_id,
                    status=AuthStatus.REFRESH_FAILED,
                    last_error=error_msg,
                )
                raise ValueError(error_msg) from e
            except httpx.RequestError as e:
                error_msg = f"Token refresh network error: {e}"
                logger.error("%s for user %s", error_msg, user_id)
                self._auth_states[user_id] = AuthState(
                    user_id=user_id,
                    status=AuthStatus.REFRESH_FAILED,
                    last_error=error_msg,
                )
                raise ValueError(error_msg) from e

        # Google may not return a new refresh_token; keep the original
        new_token = OAuthToken(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", token.refresh_token),
            token_type=data.get("token_type", "Bearer"),
            expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=data.get("expires_in", 3600)),
            scopes=data.get("scope", "").split() if data.get("scope") else token.scopes,
            id_token=data.get("id_token", token.id_token),
        )

        # Persist the refreshed token (encrypted)
        await self._store_token(user_id, new_token)

        previous_state = self._auth_states.get(user_id, AuthState(user_id=user_id))
        self._auth_states[user_id] = AuthState(
            user_id=user_id,
            status=AuthStatus.AUTHENTICATED,
            last_refresh=datetime.now(timezone.utc),
            connected_email=previous_state.connected_email,
        )

        logger.info("Successfully refreshed token for user %s", user_id)
        return new_token

    async def revoke_token(self, user_id: str) -> bool:
        """Revoke a user's Google Calendar access and delete stored tokens.

        Calls Google's revocation endpoint and then deletes the local
        encrypted token file regardless of whether the remote revocation
        succeeded (defensive cleanup).

        Args:
            user_id: The Blurt user ID.

        Returns:
            True if revocation completed (even if remote revocation failed).
        """
        try:
            token = await self._load_token(user_id)
        except FileNotFoundError:
            logger.warning("No token to revoke for user %s", user_id)
            self._auth_states[user_id] = AuthState(
                user_id=user_id,
                status=AuthStatus.NOT_CONFIGURED,
            )
            return True

        # Revoke at Google's endpoint
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    GOOGLE_REVOKE_URL,
                    params={"token": token.access_token},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30.0,
                )
                if response.status_code not in (200, 400):
                    # 400 means token was already invalid — that's fine
                    logger.warning(
                        "Token revocation returned HTTP %s for user %s",
                        response.status_code,
                        user_id,
                    )
            except httpx.RequestError as e:
                logger.warning(
                    "Token revocation request failed for user %s: %s", user_id, e
                )

        # Always delete local token — defensive cleanup
        self._delete_token_file(user_id)

        self._auth_states[user_id] = AuthState(
            user_id=user_id,
            status=AuthStatus.REVOKED,
        )

        logger.info("Revoked Google Calendar access for user %s", user_id)
        return True

    def get_auth_status(self, user_id: str) -> AuthState:
        """Get the current authentication status for a user.

        Checks in-memory state first, then falls back to checking for
        a stored token file on disk.

        Args:
            user_id: The Blurt user ID.

        Returns:
            Current authentication state.
        """
        if user_id in self._auth_states:
            return self._auth_states[user_id]

        # Check if we have a stored (encrypted) token
        token_path = self._get_token_path(user_id)
        if token_path.exists():
            return AuthState(
                user_id=user_id,
                status=AuthStatus.AUTHENTICATED,
            )

        return AuthState(
            user_id=user_id,
            status=AuthStatus.NOT_CONFIGURED,
        )

    def get_credentials(self) -> GoogleCredentials:
        """Get Google API credentials using application default credentials.

        Used by clients that do not have a specific user context.

        Returns:
            Google API credentials object.
        """
        import google.auth  # type: ignore[import-untyped]

        credentials, _ = google.auth.default(scopes=SCOPES)
        return credentials

    def get_google_credentials(self, token: OAuthToken) -> GoogleCredentials:
        """Convert an OAuthToken to Google API client credentials.

        This bridges our token model to the google-api-python-client
        Credentials format needed for building API service objects.

        Args:
            token: The Blurt OAuth token.

        Returns:
            Google API credentials object.
        """
        config = self.client_config
        return GoogleCredentials(
            token=token.access_token,
            refresh_token=token.refresh_token,
            token_uri=GOOGLE_TOKEN_URL,
            client_id=config.client_id,
            client_secret=config.client_secret,
            scopes=token.scopes,
        )

    def has_stored_token(self, user_id: str) -> bool:
        """Check if a user has a stored (encrypted) token file.

        Args:
            user_id: The Blurt user ID.

        Returns:
            True if a token file exists for this user.
        """
        return self._get_token_path(user_id).exists()

    # --- Private storage methods ---

    async def _store_token(self, user_id: str, token: OAuthToken) -> None:
        """Encrypt and store a token to disk.

        Uses CredentialEncryptor for AES-128-CBC encryption with
        per-encryption random salt. File permissions set to 0o600.
        """
        token_path = self._get_token_path(user_id)
        self._encryptor.encrypt_to_file(token.to_storage_dict(), token_path)
        logger.debug("Stored encrypted token for user %s at %s", user_id, token_path)

    async def _load_token(self, user_id: str) -> OAuthToken:
        """Load and decrypt a token from disk.

        Args:
            user_id: The Blurt user ID.

        Returns:
            The decrypted OAuthToken.

        Raises:
            FileNotFoundError: If no token exists for the user.
            ValueError: If decryption fails (corrupted file or wrong key).
        """
        token_path = self._get_token_path(user_id)
        if not token_path.exists():
            raise FileNotFoundError(
                f"No stored token for user {user_id}. "
                "User must authenticate via get_authorization_url()."
            )

        data = self._encryptor.decrypt_from_file(token_path)
        return OAuthToken.from_storage_dict(data)

    def _delete_token_file(self, user_id: str) -> None:
        """Delete the stored encrypted token file for a user."""
        token_path = self._get_token_path(user_id)
        if token_path.exists():
            token_path.unlink()
            logger.debug("Deleted token file for user %s", user_id)

    @staticmethod
    async def _fetch_user_email(access_token: str) -> str | None:
        """Fetch the authenticated user's email from Google's userinfo endpoint.

        Used to display which Google account is connected. Non-critical —
        returns None on any failure.

        Args:
            access_token: A valid Google access token.

        Returns:
            The user's email address or None.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10.0,
                )
                response.raise_for_status()
                return response.json().get("email")
            except Exception as e:
                logger.warning("Failed to fetch user email: %s", e)
                return None
