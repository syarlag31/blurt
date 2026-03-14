"""End-to-end encryption utilities for Blurt data at rest.

Provides AES-256-GCM authenticated encryption for all data before cloud storage.
Supports both cloud and local-only modes with identical encryption guarantees.
No data leakage vectors — encryption is always on by default.

Wire format (per encrypted blob):
    [version: 1 byte][salt: 16 bytes][nonce: 12 bytes][ciphertext+tag: variable]

- version: format version byte (currently 0x01)
- salt: random salt for PBKDF2 key derivation
- nonce: random nonce for AES-256-GCM (must never repeat per key)
- ciphertext: AES-256-GCM encrypted data
- tag: 16-byte GCM authentication tag (appended by AESGCM)

Key derivation uses PBKDF2-HMAC-SHA256 with 600,000 iterations (OWASP 2024
recommendation) to derive a 256-bit key from the master key + random salt.

Also retains backward-compatible CredentialEncryptor (Fernet/AES-128-CBC)
for existing credential/token storage.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import struct
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FORMAT_VERSION = 0x01
_SALT_LENGTH = 16  # 128-bit salt for KDF
_NONCE_LENGTH = 12  # 96-bit nonce (standard for GCM)
_KEY_LENGTH = 32  # 256-bit key for AES-256
_KDF_ITERATIONS = 600_000  # OWASP 2024 recommendation for PBKDF2-SHA256
_HEADER_LENGTH = 1 + _SALT_LENGTH + _NONCE_LENGTH  # version + salt + nonce


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""


class InvalidKeyError(EncryptionError):
    """Raised when the master key is invalid or missing."""


class DecryptionError(EncryptionError):
    """Raised when decryption fails (wrong key, tampered data, etc.)."""


class CorruptedDataError(EncryptionError):
    """Raised when encrypted data is malformed or too short."""


# ---------------------------------------------------------------------------
# Key Management
# ---------------------------------------------------------------------------


def generate_master_key() -> bytes:
    """Generate a cryptographically secure 256-bit master key.

    Returns:
        32 random bytes suitable for use as a master key.
    """
    return secrets.token_bytes(_KEY_LENGTH)


def save_master_key(key: bytes, path: Path) -> None:
    """Save a master key to a file with restrictive permissions (0o600).

    Args:
        key: 32-byte master key.
        path: Destination file path.

    Raises:
        InvalidKeyError: If key is not 32 bytes.
    """
    if len(key) != _KEY_LENGTH:
        raise InvalidKeyError(f"Master key must be {_KEY_LENGTH} bytes, got {len(key)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(base64.urlsafe_b64encode(key).decode("ascii"))
    os.chmod(path, 0o600)


def load_master_key(path: Path) -> bytes:
    """Load a master key from a file.

    Args:
        path: Path to the key file.

    Returns:
        32-byte master key.

    Raises:
        FileNotFoundError: If the key file doesn't exist.
        InvalidKeyError: If the key file contains invalid data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Master key file not found: {path}")

    try:
        raw = base64.urlsafe_b64decode(path.read_text().strip())
    except Exception as e:
        raise InvalidKeyError(f"Failed to decode master key: {e}") from e

    if len(raw) != _KEY_LENGTH:
        raise InvalidKeyError(f"Master key must be {_KEY_LENGTH} bytes, got {len(raw)}")

    return raw


def load_or_generate_master_key(path: Path) -> bytes:
    """Load master key from file, generating a new one if it doesn't exist.

    Args:
        path: Path to the key file.

    Returns:
        32-byte master key.
    """
    path = Path(path)
    if path.exists():
        return load_master_key(path)

    key = generate_master_key()
    save_master_key(key, path)
    return key


# ---------------------------------------------------------------------------
# Key Derivation
# ---------------------------------------------------------------------------


def _derive_key(master_key: bytes, salt: bytes) -> bytes:
    """Derive AES-256 key from master key using PBKDF2-HMAC-SHA256.

    Args:
        master_key: 32-byte master key.
        salt: 16-byte random salt.

    Returns:
        32-byte derived key for AES-256-GCM.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_KEY_LENGTH,
        salt=salt,
        iterations=_KDF_ITERATIONS,
    )
    return kdf.derive(master_key)


# ---------------------------------------------------------------------------
# AES-256-GCM Encrypt / Decrypt
# ---------------------------------------------------------------------------


def encrypt(plaintext: bytes, master_key: bytes, aad: bytes | None = None) -> bytes:
    """Encrypt plaintext using AES-256-GCM with authenticated encryption.

    Args:
        plaintext: Raw bytes to encrypt.
        master_key: 32-byte master key.
        aad: Optional additional authenticated data (authenticated but not
            encrypted). Use for binding ciphertext to context (e.g., user ID).

    Returns:
        Encrypted blob: [version][salt][nonce][ciphertext+tag]

    Raises:
        InvalidKeyError: If master key is invalid.
        EncryptionError: If encryption fails.
    """
    if len(master_key) != _KEY_LENGTH:
        raise InvalidKeyError(f"Master key must be {_KEY_LENGTH} bytes, got {len(master_key)}")

    try:
        salt = secrets.token_bytes(_SALT_LENGTH)
        nonce = secrets.token_bytes(_NONCE_LENGTH)
        derived_key = _derive_key(master_key, salt)

        aesgcm = AESGCM(derived_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

        # Pack: version + salt + nonce + ciphertext (includes GCM tag)
        return struct.pack("B", _FORMAT_VERSION) + salt + nonce + ciphertext
    except InvalidKeyError:
        raise
    except Exception as e:
        raise EncryptionError(f"Encryption failed: {e}") from e


def decrypt(encrypted_data: bytes, master_key: bytes, aad: bytes | None = None) -> bytes:
    """Decrypt an AES-256-GCM encrypted blob.

    Args:
        encrypted_data: Encrypted blob produced by encrypt().
        master_key: 32-byte master key (same key used to encrypt).
        aad: Optional additional authenticated data (must match what was
            used during encryption).

    Returns:
        Decrypted plaintext bytes.

    Raises:
        InvalidKeyError: If master key is invalid.
        CorruptedDataError: If encrypted data is malformed.
        DecryptionError: If decryption/authentication fails.
    """
    if len(master_key) != _KEY_LENGTH:
        raise InvalidKeyError(f"Master key must be {_KEY_LENGTH} bytes, got {len(master_key)}")

    if len(encrypted_data) < _HEADER_LENGTH + 16:  # minimum: header + GCM tag
        raise CorruptedDataError(
            f"Encrypted data too short: {len(encrypted_data)} bytes "
            f"(minimum {_HEADER_LENGTH + 16})"
        )

    # Unpack header
    version = encrypted_data[0]
    if version != _FORMAT_VERSION:
        raise CorruptedDataError(f"Unsupported format version: {version} (expected {_FORMAT_VERSION})")

    salt = encrypted_data[1 : 1 + _SALT_LENGTH]
    nonce = encrypted_data[1 + _SALT_LENGTH : 1 + _SALT_LENGTH + _NONCE_LENGTH]
    ciphertext = encrypted_data[_HEADER_LENGTH:]

    try:
        derived_key = _derive_key(master_key, salt)
        aesgcm = AESGCM(derived_key)
        return aesgcm.decrypt(nonce, ciphertext, aad)
    except CorruptedDataError:
        raise
    except Exception as e:
        raise DecryptionError(f"Decryption failed: {e}") from e


# ---------------------------------------------------------------------------
# JSON Convenience Functions
# ---------------------------------------------------------------------------


def encrypt_json(data: Any, master_key: bytes, aad: bytes | None = None) -> bytes:
    """Encrypt a JSON-serializable object using AES-256-GCM.

    Args:
        data: Any JSON-serializable Python object.
        master_key: 32-byte master key.
        aad: Optional additional authenticated data.

    Returns:
        Encrypted blob bytes.
    """
    plaintext = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
    return encrypt(plaintext, master_key, aad)


def decrypt_json(encrypted_data: bytes, master_key: bytes, aad: bytes | None = None) -> Any:
    """Decrypt an AES-256-GCM encrypted blob back to a JSON object.

    Args:
        encrypted_data: Encrypted blob produced by encrypt_json().
        master_key: 32-byte master key.
        aad: Optional additional authenticated data.

    Returns:
        Deserialized Python object.
    """
    plaintext = decrypt(encrypted_data, master_key, aad)
    return json.loads(plaintext.decode("utf-8"))


# ---------------------------------------------------------------------------
# File I/O Convenience
# ---------------------------------------------------------------------------


def encrypt_to_file(
    data: bytes, master_key: bytes, path: Path, aad: bytes | None = None
) -> None:
    """Encrypt data and write to a file with restrictive permissions.

    Args:
        data: Raw bytes to encrypt.
        master_key: 32-byte master key.
        path: Destination file path.
        aad: Optional additional authenticated data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encrypted = encrypt(data, master_key, aad)
    path.write_bytes(encrypted)
    os.chmod(path, 0o600)


def decrypt_from_file(
    path: Path, master_key: bytes, aad: bytes | None = None
) -> bytes:
    """Read and decrypt data from a file.

    Args:
        path: Source file path.
        master_key: 32-byte master key.
        aad: Optional additional authenticated data.

    Returns:
        Decrypted bytes.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Encrypted file not found: {path}")
    encrypted = path.read_bytes()
    return decrypt(encrypted, master_key, aad)


def encrypt_json_to_file(
    data: Any, master_key: bytes, path: Path, aad: bytes | None = None
) -> None:
    """Encrypt a JSON object and write to a file.

    Args:
        data: JSON-serializable object.
        master_key: 32-byte master key.
        path: Destination file path.
        aad: Optional additional authenticated data.
    """
    plaintext = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
    encrypt_to_file(plaintext, master_key, path, aad)


def decrypt_json_from_file(
    path: Path, master_key: bytes, aad: bytes | None = None
) -> Any:
    """Read, decrypt, and deserialize JSON from a file.

    Args:
        path: Source file path.
        master_key: 32-byte master key.
        aad: Optional additional authenticated data.

    Returns:
        Deserialized Python object.
    """
    plaintext = decrypt_from_file(path, master_key, aad)
    return json.loads(plaintext.decode("utf-8"))


# ---------------------------------------------------------------------------
# DataEncryptor — stateful convenience wrapper
# ---------------------------------------------------------------------------


class DataEncryptor:
    """Stateful AES-256-GCM encryptor for data at rest.

    Holds a master key and provides convenient encrypt/decrypt methods
    for bytes, JSON, and file I/O. Designed for encrypting blurt data
    (episodes, memories, knowledge graph nodes) before cloud storage.

    Example:
        >>> enc = DataEncryptor()  # ephemeral key for testing
        >>> blob = enc.encrypt(b"hello world")
        >>> enc.decrypt(blob)
        b'hello world'
    """

    def __init__(
        self,
        master_key: bytes | None = None,
        key_path: Path | None = None,
    ):
        """Initialize with a master key or key file path.

        Args:
            master_key: 32-byte master key. If None, loads from key_path.
            key_path: Path to load/generate master key file.
                If both are None, generates an ephemeral key (testing only).
        """
        if master_key is not None:
            if len(master_key) != _KEY_LENGTH:
                raise InvalidKeyError(
                    f"Master key must be {_KEY_LENGTH} bytes, got {len(master_key)}"
                )
            self._master_key = master_key
        elif key_path is not None:
            self._master_key = load_or_generate_master_key(key_path)
        else:
            self._master_key = generate_master_key()

    def encrypt(self, plaintext: bytes, aad: bytes | None = None) -> bytes:
        """Encrypt plaintext bytes with AES-256-GCM."""
        return encrypt(plaintext, self._master_key, aad)

    def decrypt(self, encrypted_data: bytes, aad: bytes | None = None) -> bytes:
        """Decrypt AES-256-GCM encrypted data."""
        return decrypt(encrypted_data, self._master_key, aad)

    def encrypt_json(self, data: Any, aad: bytes | None = None) -> bytes:
        """Encrypt a JSON-serializable object."""
        return encrypt_json(data, self._master_key, aad)

    def decrypt_json(self, encrypted_data: bytes, aad: bytes | None = None) -> Any:
        """Decrypt to a JSON object."""
        return decrypt_json(encrypted_data, self._master_key, aad)

    def encrypt_to_file(
        self, data: bytes, path: Path, aad: bytes | None = None
    ) -> None:
        """Encrypt and write to file."""
        encrypt_to_file(data, self._master_key, path, aad)

    def decrypt_from_file(self, path: Path, aad: bytes | None = None) -> bytes:
        """Read and decrypt from file."""
        return decrypt_from_file(path, self._master_key, aad)

    def encrypt_json_to_file(
        self, data: Any, path: Path, aad: bytes | None = None
    ) -> None:
        """Encrypt JSON and write to file."""
        encrypt_json_to_file(data, self._master_key, path, aad)

    def decrypt_json_from_file(
        self, path: Path, aad: bytes | None = None
    ) -> Any:
        """Read, decrypt, and deserialize JSON from file."""
        return decrypt_json_from_file(path, self._master_key, aad)


# ---------------------------------------------------------------------------
# Legacy CredentialEncryptor (Fernet / AES-128-CBC)
# ---------------------------------------------------------------------------


class CredentialEncryptor:
    """Encrypts and decrypts sensitive credential data using Fernet (AES-128-CBC).

    Retained for backward compatibility with existing credential/token storage.
    New code should prefer DataEncryptor (AES-256-GCM) instead.

    Key derivation uses PBKDF2 with SHA256 and 480,000 iterations.
    Each encryption includes a random salt for key derivation uniqueness.
    """

    SALT_LENGTH = 16
    KDF_ITERATIONS = 480_000

    def __init__(self, master_key: bytes | None = None, key_path: Path | None = None):
        if master_key is not None:
            self._master_key = master_key
        elif key_path is not None:
            self._master_key = load_or_generate_master_key(key_path)
        else:
            self._master_key = secrets.token_bytes(32)

    def _derive_fernet_key(self, salt: bytes) -> bytes:
        """Derive a Fernet key from the master key using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.KDF_ITERATIONS,
        )
        return base64.urlsafe_b64encode(kdf.derive(self._master_key))

    def encrypt(self, data: dict) -> bytes:
        salt = secrets.token_bytes(self.SALT_LENGTH)
        fernet_key = self._derive_fernet_key(salt)
        f = Fernet(fernet_key)
        plaintext = json.dumps(data, default=str).encode("utf-8")
        ciphertext = f.encrypt(plaintext)
        return salt + ciphertext

    def decrypt(self, encrypted_data: bytes) -> dict:
        if len(encrypted_data) < self.SALT_LENGTH:
            raise ValueError("Encrypted data is too short")

        salt = encrypted_data[: self.SALT_LENGTH]
        ciphertext = encrypted_data[self.SALT_LENGTH :]
        fernet_key = self._derive_fernet_key(salt)
        f = Fernet(fernet_key)

        try:
            plaintext = f.decrypt(ciphertext)
            return json.loads(plaintext.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}") from e

    def encrypt_to_file(self, data: dict, file_path: Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        encrypted = self.encrypt(data)
        file_path.write_bytes(encrypted)
        os.chmod(file_path, 0o600)

    def decrypt_from_file(self, file_path: Path) -> dict:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {file_path}")
        encrypted = file_path.read_bytes()
        return self.decrypt(encrypted)
