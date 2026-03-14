"""Tests for the encryption key management service.

Covers key generation, rotation, wrapping/unwrapping, persistence,
environment-based master keys, and error handling.
"""

from __future__ import annotations

import base64
import secrets
from pathlib import Path

import pytest

from blurt.core.key_management import (
    DEK_LENGTH,
    KEY_METADATA_FILENAME,
    MASTER_KEY_ENV_VAR,
    MASTER_KEY_LENGTH,
    KeyManagementError,
    KeyManagementService,
    KeyMetadata,
    KeyRing,
    KeyStatus,
    KeyVersionNotFoundError,
    MasterKeyNotFoundError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_keys_dir(tmp_path: Path) -> Path:
    """Temporary directory for key storage."""
    d = tmp_path / "keys"
    d.mkdir()
    return d


@pytest.fixture
def master_key() -> bytes:
    """A fixed 32-byte master key for deterministic tests."""
    return secrets.token_bytes(MASTER_KEY_LENGTH)


@pytest.fixture
def kms(master_key: bytes, tmp_keys_dir: Path) -> KeyManagementService:
    """A fresh KMS instance with explicit master key and temp storage."""
    return KeyManagementService(
        master_key=master_key,
        storage_dir=tmp_keys_dir,
    )


# ---------------------------------------------------------------------------
# Master key resolution
# ---------------------------------------------------------------------------


class TestMasterKeyResolution:
    """Test master key loading from various sources."""

    def test_explicit_master_key(self, master_key: bytes, tmp_keys_dir: Path) -> None:
        kms = KeyManagementService(master_key=master_key, storage_dir=tmp_keys_dir)
        # Should not raise; internal key is set
        kms.generate_dek()
        assert kms.has_active_key()

    def test_master_key_from_env(self, tmp_keys_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        key = secrets.token_bytes(MASTER_KEY_LENGTH)
        encoded = base64.urlsafe_b64encode(key).decode()
        monkeypatch.setenv(MASTER_KEY_ENV_VAR, encoded)

        kms = KeyManagementService(storage_dir=tmp_keys_dir)
        kms.generate_dek()
        assert kms.has_active_key()

    def test_invalid_env_key_raises(
        self, tmp_keys_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(MASTER_KEY_ENV_VAR, "not-valid-base64!!!")
        # Remove key file so it doesn't fall through
        master_path = tmp_keys_dir / "master.key"
        if master_path.exists():
            master_path.unlink()
        with pytest.raises(MasterKeyNotFoundError):
            KeyManagementService(storage_dir=tmp_keys_dir, master_key_path=tmp_keys_dir / "nope.key")

    def test_master_key_from_file(self, tmp_keys_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MASTER_KEY_ENV_VAR, raising=False)
        key = secrets.token_bytes(MASTER_KEY_LENGTH)
        key_path = tmp_keys_dir / "master.key"
        key_path.write_text(base64.urlsafe_b64encode(key).decode())

        kms = KeyManagementService(storage_dir=tmp_keys_dir, master_key_path=key_path)
        kms.generate_dek()
        assert kms.has_active_key()

    def test_auto_generates_master_key(
        self, tmp_keys_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(MASTER_KEY_ENV_VAR, raising=False)
        key_path = tmp_keys_dir / "auto_master.key"
        assert not key_path.exists()

        kms = KeyManagementService(storage_dir=tmp_keys_dir, master_key_path=key_path)
        assert key_path.exists()
        # File should have restrictive permissions
        assert oct(key_path.stat().st_mode)[-3:] == "600"

        kms.generate_dek()
        assert kms.has_active_key()

    def test_wrong_length_master_key_raises(self, tmp_keys_dir: Path) -> None:
        with pytest.raises(MasterKeyNotFoundError, match="must be 32 bytes"):
            KeyManagementService(master_key=b"too-short", storage_dir=tmp_keys_dir)


# ---------------------------------------------------------------------------
# DEK generation
# ---------------------------------------------------------------------------


class TestDEKGeneration:
    def test_generate_first_dek(self, kms: KeyManagementService) -> None:
        meta = kms.generate_dek()
        assert meta.version == 1
        assert meta.status == KeyStatus.ACTIVE
        assert meta.wrapped_dek != ""
        assert meta.created_at > 0
        assert meta.rotated_at is None
        assert kms.get_active_version() == 1

    def test_generate_sets_active(self, kms: KeyManagementService) -> None:
        kms.generate_dek()
        assert kms.has_active_key()

    def test_generate_with_purpose(self, kms: KeyManagementService) -> None:
        meta = kms.generate_dek(purpose="user-data")
        assert meta.purpose == "user-data"

    def test_active_dek_is_correct_length(self, kms: KeyManagementService) -> None:
        kms.generate_dek()
        dek = kms.get_active_dek()
        assert len(dek) == DEK_LENGTH

    def test_no_active_dek_raises(self, kms: KeyManagementService) -> None:
        with pytest.raises(KeyManagementError, match="No active DEK"):
            kms.get_active_dek()


# ---------------------------------------------------------------------------
# Key rotation
# ---------------------------------------------------------------------------


class TestKeyRotation:
    def test_rotate_creates_new_active(self, kms: KeyManagementService) -> None:
        kms.generate_dek()
        old_dek = kms.get_active_dek()

        new_meta = kms.rotate_key()
        assert new_meta.version == 2
        assert new_meta.status == KeyStatus.ACTIVE

        new_dek = kms.get_active_dek()
        assert new_dek != old_dek

    def test_rotate_retires_old_key(self, kms: KeyManagementService) -> None:
        kms.generate_dek()
        kms.rotate_key()

        old_meta = kms.get_key_metadata(1)
        assert old_meta is not None
        assert old_meta.status == KeyStatus.DECRYPT_ONLY
        assert old_meta.rotated_at is not None

    def test_old_key_still_decryptable(self, kms: KeyManagementService) -> None:
        kms.generate_dek()
        old_dek = kms.get_active_dek()

        kms.rotate_key()

        # Old key should still be retrievable for decryption
        recovered = kms.get_dek_by_version(1)
        assert recovered == old_dek

    def test_multiple_rotations(self, kms: KeyManagementService) -> None:
        kms.generate_dek()
        kms.rotate_key()
        kms.rotate_key()
        kms.rotate_key()

        assert kms.get_active_version() == 4
        keys = kms.list_keys()
        assert len(keys) == 4

        # Only last key should be active
        active = [k for k in keys if k.status == KeyStatus.ACTIVE]
        decrypt_only = [k for k in keys if k.status == KeyStatus.DECRYPT_ONLY]
        assert len(active) == 1
        assert len(decrypt_only) == 3

    def test_fernet_encrypt_decrypt_across_rotation(self, kms: KeyManagementService) -> None:
        """Data encrypted with old key is decryptable after rotation."""
        kms.generate_dek()
        fernet_v1, v1 = kms.get_fernet_for_active_key()
        ciphertext = fernet_v1.encrypt(b"secret data v1")

        # Rotate
        kms.rotate_key()
        assert kms.get_active_version() == 2

        # Decrypt with old version
        fernet_old = kms.get_fernet_for_version(v1)
        assert fernet_old.decrypt(ciphertext) == b"secret data v1"


# ---------------------------------------------------------------------------
# Key destruction
# ---------------------------------------------------------------------------


class TestKeyDestruction:
    def test_destroy_retired_key(self, kms: KeyManagementService) -> None:
        kms.generate_dek()
        kms.rotate_key()
        kms.destroy_key(1)

        meta = kms.get_key_metadata(1)
        assert meta is not None
        assert meta.status == KeyStatus.DESTROYED
        assert meta.wrapped_dek == ""

    def test_cannot_destroy_active_key(self, kms: KeyManagementService) -> None:
        kms.generate_dek()
        with pytest.raises(KeyManagementError, match="Cannot destroy the active key"):
            kms.destroy_key(1)

    def test_destroyed_key_not_decryptable(self, kms: KeyManagementService) -> None:
        kms.generate_dek()
        kms.rotate_key()
        kms.destroy_key(1)
        with pytest.raises(KeyManagementError):
            kms.get_dek_by_version(1)

    def test_destroy_nonexistent_raises(self, kms: KeyManagementService) -> None:
        with pytest.raises(KeyVersionNotFoundError):
            kms.destroy_key(999)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_keyring_persists_to_disk(
        self, master_key: bytes, tmp_keys_dir: Path
    ) -> None:
        kms = KeyManagementService(master_key=master_key, storage_dir=tmp_keys_dir)
        kms.generate_dek()
        kms.rotate_key()

        # Load fresh instance from same storage
        kms2 = KeyManagementService(master_key=master_key, storage_dir=tmp_keys_dir)
        assert kms2.get_active_version() == 2
        assert len(kms2.list_keys()) == 2

    def test_dek_survives_restart(
        self, master_key: bytes, tmp_keys_dir: Path
    ) -> None:
        kms = KeyManagementService(master_key=master_key, storage_dir=tmp_keys_dir)
        kms.generate_dek()
        original_dek = kms.get_active_dek()

        kms2 = KeyManagementService(master_key=master_key, storage_dir=tmp_keys_dir)
        recovered_dek = kms2.get_active_dek()
        assert recovered_dek == original_dek

    def test_metadata_file_permissions(
        self, master_key: bytes, tmp_keys_dir: Path
    ) -> None:
        kms = KeyManagementService(master_key=master_key, storage_dir=tmp_keys_dir)
        kms.generate_dek()

        meta_path = tmp_keys_dir / KEY_METADATA_FILENAME
        assert meta_path.exists()
        assert oct(meta_path.stat().st_mode)[-3:] == "600"

    def test_corrupted_metadata_starts_fresh(
        self, master_key: bytes, tmp_keys_dir: Path
    ) -> None:
        meta_path = tmp_keys_dir / KEY_METADATA_FILENAME
        meta_path.write_text("{invalid json!!!")

        kms = KeyManagementService(master_key=master_key, storage_dir=tmp_keys_dir)
        assert not kms.has_active_key()
        assert kms.get_active_version() == 0

    def test_wrong_master_key_fails_unwrap(self, tmp_keys_dir: Path) -> None:
        key1 = secrets.token_bytes(MASTER_KEY_LENGTH)
        key2 = secrets.token_bytes(MASTER_KEY_LENGTH)

        kms1 = KeyManagementService(master_key=key1, storage_dir=tmp_keys_dir)
        kms1.generate_dek()

        kms2 = KeyManagementService(master_key=key2, storage_dir=tmp_keys_dir)
        with pytest.raises(KeyManagementError, match="master key may have changed"):
            kms2.get_active_dek()


# ---------------------------------------------------------------------------
# KeyRing / KeyMetadata serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_key_metadata_roundtrip(self) -> None:
        meta = KeyMetadata(
            version=3,
            status=KeyStatus.DECRYPT_ONLY,
            created_at=1700000000.0,
            rotated_at=1700001000.0,
            wrapped_dek="abc123",
            purpose="user-data",
        )
        d = meta.to_dict()
        recovered = KeyMetadata.from_dict(d)
        assert recovered.version == 3
        assert recovered.status == KeyStatus.DECRYPT_ONLY
        assert recovered.rotated_at == 1700001000.0
        assert recovered.purpose == "user-data"

    def test_keyring_roundtrip(self) -> None:
        kr = KeyRing()
        kr.active_version = 2
        kr.keys[1] = KeyMetadata(
            version=1, status=KeyStatus.DECRYPT_ONLY, created_at=1.0
        )
        kr.keys[2] = KeyMetadata(
            version=2, status=KeyStatus.ACTIVE, created_at=2.0
        )
        d = kr.to_dict()
        kr2 = KeyRing.from_dict(d)
        assert kr2.active_version == 2
        assert len(kr2.keys) == 2
        assert kr2.keys[1].status == KeyStatus.DECRYPT_ONLY


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_generate_master_key_format(self) -> None:
        encoded = KeyManagementService.generate_master_key()
        raw = base64.urlsafe_b64decode(encoded)
        assert len(raw) == MASTER_KEY_LENGTH

    def test_list_keys_sorted(self, kms: KeyManagementService) -> None:
        kms.generate_dek()
        kms.rotate_key()
        kms.rotate_key()
        keys = kms.list_keys()
        versions = [k.version for k in keys]
        assert versions == [1, 2, 3]

    def test_get_nonexistent_version_raises(self, kms: KeyManagementService) -> None:
        with pytest.raises(KeyVersionNotFoundError):
            kms.get_dek_by_version(42)

    def test_get_key_metadata_nonexistent(self, kms: KeyManagementService) -> None:
        assert kms.get_key_metadata(99) is None

    def test_has_active_key_false_initially(self, kms: KeyManagementService) -> None:
        assert not kms.has_active_key()

    def test_has_active_key_true_after_generate(self, kms: KeyManagementService) -> None:
        kms.generate_dek()
        assert kms.has_active_key()
