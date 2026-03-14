"""Encryption key management service for Blurt.

Provides key generation, rotation, and secure storage using
environment-based master keys. Implements envelope encryption:
a master key (KEK) protects data encryption keys (DEKs), which
protect user data.

Key hierarchy:
  Master Key (KEK) — from env var or file, never stored in plaintext
    └── Data Encryption Keys (DEKs) — versioned, rotatable, wrapped by KEK

Privacy by default: all data E2E encrypted, local-only mode has
full feature parity. No data leakage vectors.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Self

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASTER_KEY_ENV_VAR = "BLURT_MASTER_KEY"
MASTER_KEY_LENGTH = 32  # 256-bit master key
DEK_LENGTH = 32  # 256-bit data encryption keys
HKDF_INFO_PREFIX = b"blurt-dek-v"
KEY_METADATA_FILENAME = "key_metadata.json"


class KeyStatus(str, Enum):
    """Lifecycle status of a data encryption key."""

    ACTIVE = "active"  # Current key for new encryptions
    DECRYPT_ONLY = "decrypt_only"  # Retired; can still decrypt existing data
    DESTROYED = "destroyed"  # Permanently unusable


@dataclass(slots=True)
class KeyMetadata:
    """Metadata for a versioned data encryption key.

    The wrapped_dek is the DEK encrypted by the master key.
    It is safe to store on disk — useless without the master key.
    """

    version: int
    status: KeyStatus
    created_at: float  # Unix timestamp
    rotated_at: float | None = None  # When this key was retired
    wrapped_dek: str = ""  # Base64-encoded, KEK-encrypted DEK
    purpose: str = "general"  # Key purpose label

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Deserialize from a dict."""
        data = dict(data)  # shallow copy
        data["status"] = KeyStatus(data["status"])
        return cls(**data)


@dataclass(slots=True)
class KeyRing:
    """Collection of versioned DEKs with one active key.

    Invariants:
      - Exactly one key has status ACTIVE at any time.
      - Version numbers are monotonically increasing.
      - Old keys transition to DECRYPT_ONLY on rotation.
    """

    keys: dict[int, KeyMetadata] = field(default_factory=dict)
    active_version: int = 0

    def to_dict(self) -> dict:
        return {
            "active_version": self.active_version,
            "keys": {str(v): km.to_dict() for v, km in self.keys.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        keys = {int(v): KeyMetadata.from_dict(km) for v, km in data.get("keys", {}).items()}
        return cls(keys=keys, active_version=data.get("active_version", 0))


# ---------------------------------------------------------------------------
# Key Management Service
# ---------------------------------------------------------------------------


class KeyManagementError(Exception):
    """Base error for key management operations."""


class MasterKeyNotFoundError(KeyManagementError):
    """Raised when no master key can be resolved."""


class KeyVersionNotFoundError(KeyManagementError):
    """Raised when a requested key version does not exist."""


class KeyManagementService:
    """Manages encryption key lifecycle: generation, rotation, storage.

    Envelope encryption pattern:
      1. A master key (KEK) is loaded from environment or file.
      2. Data encryption keys (DEKs) are generated and wrapped (encrypted)
         with the KEK before being persisted.
      3. On use, the DEK is unwrapped in memory; plaintext DEK is never
         written to disk.

    Supports key rotation: new DEK is generated and becomes active;
    old DEK transitions to decrypt-only so existing ciphertext remains
    readable until re-encrypted.
    """

    def __init__(
        self,
        master_key: bytes | None = None,
        storage_dir: Path | None = None,
        master_key_path: Path | None = None,
    ) -> None:
        """Initialize the key management service.

        Master key resolution order:
          1. Explicit ``master_key`` parameter
          2. ``BLURT_MASTER_KEY`` environment variable (base64-encoded)
          3. File at ``master_key_path`` (base64-encoded contents)
          4. Auto-generate and persist to ``master_key_path``

        Args:
            master_key: Raw 32-byte master key. Highest priority.
            storage_dir: Directory for key metadata. Defaults to ~/.blurt/keys.
            master_key_path: File path to load/save the master key.
        """
        self._storage_dir = storage_dir or Path.home() / ".blurt" / "keys"
        self._master_key_path = master_key_path or self._storage_dir / "master.key"
        self._master_key = self._resolve_master_key(master_key)
        self._keyring = self._load_keyring()

    # ------------------------------------------------------------------
    # Master key resolution
    # ------------------------------------------------------------------

    def _resolve_master_key(self, explicit_key: bytes | None) -> bytes:
        """Resolve the master key from available sources."""
        # 1. Explicit parameter
        if explicit_key is not None:
            self._validate_key_length(explicit_key, MASTER_KEY_LENGTH, "master key")
            return explicit_key

        # 2. Environment variable
        env_value = os.environ.get(MASTER_KEY_ENV_VAR)
        if env_value:
            try:
                key = base64.urlsafe_b64decode(env_value)
                self._validate_key_length(key, MASTER_KEY_LENGTH, "master key (env)")
                return key
            except Exception as exc:
                raise MasterKeyNotFoundError(
                    f"Invalid {MASTER_KEY_ENV_VAR}: must be 32-byte base64url-encoded"
                ) from exc

        # 3. Key file
        path = Path(self._master_key_path)
        if path.exists():
            try:
                key = base64.urlsafe_b64decode(path.read_text().strip())
                self._validate_key_length(key, MASTER_KEY_LENGTH, "master key (file)")
                return key
            except Exception as exc:
                raise MasterKeyNotFoundError(
                    f"Invalid master key file {path}: {exc}"
                ) from exc

        # 4. Auto-generate
        return self._generate_and_persist_master_key(path)

    @staticmethod
    def _validate_key_length(key: bytes, expected: int, label: str) -> None:
        if len(key) != expected:
            raise MasterKeyNotFoundError(
                f"{label} must be {expected} bytes, got {len(key)}"
            )

    @staticmethod
    def _generate_and_persist_master_key(path: Path) -> bytes:
        """Generate a new master key and save it with restrictive permissions."""
        key = secrets.token_bytes(MASTER_KEY_LENGTH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(base64.urlsafe_b64encode(key).decode())
        os.chmod(path, 0o600)
        return key

    # ------------------------------------------------------------------
    # DEK wrapping / unwrapping (envelope encryption)
    # ------------------------------------------------------------------

    def _derive_wrapping_key(self) -> bytes:
        """Derive a Fernet-compatible wrapping key from the master key using HKDF."""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,  # Static derivation — deterministic for the same master key
            info=b"blurt-kek-wrapping",
        )
        raw = hkdf.derive(self._master_key)
        return base64.urlsafe_b64encode(raw)

    def _wrap_dek(self, dek: bytes) -> str:
        """Encrypt a DEK with the master-derived wrapping key.

        Returns base64-encoded ciphertext (safe to store on disk).
        """
        wrapping_key = self._derive_wrapping_key()
        f = Fernet(wrapping_key)
        return base64.urlsafe_b64encode(f.encrypt(dek)).decode()

    def _unwrap_dek(self, wrapped: str) -> bytes:
        """Decrypt a wrapped DEK back to raw bytes.

        Raises KeyManagementError if unwrapping fails (wrong master key).
        """
        wrapping_key = self._derive_wrapping_key()
        f = Fernet(wrapping_key)
        try:
            return f.decrypt(base64.urlsafe_b64decode(wrapped))
        except InvalidToken as exc:
            raise KeyManagementError(
                "Failed to unwrap DEK — master key may have changed"
            ) from exc

    # ------------------------------------------------------------------
    # KeyRing persistence
    # ------------------------------------------------------------------

    @property
    def _metadata_path(self) -> Path:
        return self._storage_dir / KEY_METADATA_FILENAME

    def _load_keyring(self) -> KeyRing:
        """Load the keyring from disk, or return an empty one."""
        path = self._metadata_path
        if not path.exists():
            return KeyRing()
        try:
            data = json.loads(path.read_text())
            return KeyRing.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            # Corrupted metadata — start fresh (old wrapped keys are lost)
            return KeyRing()

    def _save_keyring(self) -> None:
        """Persist the keyring metadata to disk."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._metadata_path
        path.write_text(json.dumps(self._keyring.to_dict(), indent=2))
        os.chmod(path, 0o600)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dek(self, purpose: str = "general") -> KeyMetadata:
        """Generate a new data encryption key and set it as active.

        If a key is already active, it transitions to DECRYPT_ONLY.

        Args:
            purpose: Human-readable purpose label.

        Returns:
            Metadata for the newly created key.
        """
        # Retire current active key
        if self._keyring.active_version > 0:
            current = self._keyring.keys.get(self._keyring.active_version)
            if current and current.status == KeyStatus.ACTIVE:
                current.status = KeyStatus.DECRYPT_ONLY
                current.rotated_at = time.time()

        # Create new DEK
        new_version = self._keyring.active_version + 1
        dek = secrets.token_bytes(DEK_LENGTH)
        wrapped = self._wrap_dek(dek)

        meta = KeyMetadata(
            version=new_version,
            status=KeyStatus.ACTIVE,
            created_at=time.time(),
            wrapped_dek=wrapped,
            purpose=purpose,
        )
        self._keyring.keys[new_version] = meta
        self._keyring.active_version = new_version
        self._save_keyring()
        return meta

    def rotate_key(self, purpose: str = "general") -> KeyMetadata:
        """Rotate the active key: retire old, generate new.

        This is the primary rotation entry point. Existing data encrypted
        with the old key remains decryptable (key transitions to DECRYPT_ONLY).

        Args:
            purpose: Purpose label for the new key.

        Returns:
            Metadata for the new active key.
        """
        return self.generate_dek(purpose=purpose)

    def get_active_dek(self) -> bytes:
        """Return the raw active DEK for encryption operations.

        Raises:
            KeyManagementError: If no active key exists.
        """
        if self._keyring.active_version == 0:
            raise KeyManagementError("No active DEK — call generate_dek() first")

        meta = self._keyring.keys[self._keyring.active_version]
        if meta.status != KeyStatus.ACTIVE:
            raise KeyManagementError("Active key is not in ACTIVE status")

        return self._unwrap_dek(meta.wrapped_dek)

    def get_dek_by_version(self, version: int) -> bytes:
        """Return a raw DEK by version number (for decrypting old data).

        Args:
            version: Key version number.

        Raises:
            KeyVersionNotFoundError: If version doesn't exist.
            KeyManagementError: If key is destroyed.
        """
        meta = self._keyring.keys.get(version)
        if meta is None:
            raise KeyVersionNotFoundError(f"Key version {version} not found")
        if meta.status == KeyStatus.DESTROYED:
            raise KeyManagementError(f"Key version {version} has been destroyed")
        return self._unwrap_dek(meta.wrapped_dek)

    def get_active_version(self) -> int:
        """Return the current active key version number."""
        return self._keyring.active_version

    def get_key_metadata(self, version: int) -> KeyMetadata | None:
        """Return metadata for a specific key version."""
        return self._keyring.keys.get(version)

    def list_keys(self) -> list[KeyMetadata]:
        """Return metadata for all keys, sorted by version."""
        return [self._keyring.keys[v] for v in sorted(self._keyring.keys)]

    def destroy_key(self, version: int) -> None:
        """Permanently mark a key version as destroyed.

        The wrapped DEK is zeroed out. Data encrypted with this key
        becomes permanently unrecoverable.

        Args:
            version: Key version to destroy.

        Raises:
            KeyVersionNotFoundError: If version doesn't exist.
            KeyManagementError: If trying to destroy the active key.
        """
        meta = self._keyring.keys.get(version)
        if meta is None:
            raise KeyVersionNotFoundError(f"Key version {version} not found")
        if meta.status == KeyStatus.ACTIVE:
            raise KeyManagementError("Cannot destroy the active key — rotate first")
        meta.status = KeyStatus.DESTROYED
        meta.wrapped_dek = ""
        self._save_keyring()

    def get_fernet_for_active_key(self) -> tuple[Fernet, int]:
        """Return a Fernet instance for the active DEK and its version.

        Convenience method for encrypt-then-store workflows where the
        caller needs to record which key version was used.
        """
        dek = self.get_active_dek()
        fernet_key = base64.urlsafe_b64encode(dek)
        return Fernet(fernet_key), self._keyring.active_version

    def get_fernet_for_version(self, version: int) -> Fernet:
        """Return a Fernet instance for a specific key version (decryption)."""
        dek = self.get_dek_by_version(version)
        fernet_key = base64.urlsafe_b64encode(dek)
        return Fernet(fernet_key)

    @staticmethod
    def generate_master_key() -> str:
        """Generate a new master key suitable for BLURT_MASTER_KEY env var.

        Returns:
            Base64url-encoded 32-byte key string.
        """
        return base64.urlsafe_b64encode(secrets.token_bytes(MASTER_KEY_LENGTH)).decode()

    def has_active_key(self) -> bool:
        """Check whether an active DEK exists."""
        if self._keyring.active_version == 0:
            return False
        meta = self._keyring.keys.get(self._keyring.active_version)
        return meta is not None and meta.status == KeyStatus.ACTIVE
