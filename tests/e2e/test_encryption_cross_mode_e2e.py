"""E2E Scenario: Encryption works identically across cloud and local modes.

Validates that:
- AES-256-GCM encryption/decryption works in cloud mode
- Encryption/decryption works identically in local mode
- Ciphertext wire format (version, salt, nonce, tag) is consistent across modes
- DataEncryptor and CredentialEncryptor produce interoperable output
- Encrypted data created in one mode can be decrypted in the other
- Wire format invariants hold regardless of deployment configuration
- AAD (Additional Authenticated Data) binding is preserved across modes
- JSON convenience functions maintain format consistency
- File-based encryption round-trips work in both modes
- Key management (generate, save, load) is mode-agnostic

All tests are self-contained with per-test setup/teardown.
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from blurt.config.settings import BlurtConfig, DeploymentMode
from blurt.core.app import create_app
from blurt.core.encryption import (
    CorruptedDataError,
    CredentialEncryptor,
    DataEncryptor,
    DecryptionError,
    _HEADER_LENGTH,
    _KEY_LENGTH,
    _NONCE_LENGTH,
    _SALT_LENGTH,
    decrypt,
    decrypt_json,
    encrypt,
    encrypt_json,
    generate_master_key,
    load_master_key,
    save_master_key,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FORMAT_VERSION = 0x01


def _parse_wire_format(blob: bytes) -> dict[str, Any]:
    """Parse the AES-256-GCM wire format into its components."""
    version = blob[0]
    salt = blob[1 : 1 + _SALT_LENGTH]
    nonce = blob[1 + _SALT_LENGTH : 1 + _SALT_LENGTH + _NONCE_LENGTH]
    ciphertext_with_tag = blob[_HEADER_LENGTH:]
    # GCM tag is always the last 16 bytes of the ciphertext
    tag = ciphertext_with_tag[-16:]
    ciphertext_body = ciphertext_with_tag[:-16]
    return {
        "version": version,
        "salt": salt,
        "nonce": nonce,
        "ciphertext_body": ciphertext_body,
        "tag": tag,
        "full_blob": blob,
    }


def _make_app(mode: DeploymentMode) -> Any:
    """Create a FastAPI app in the specified deployment mode."""
    config = BlurtConfig(mode=mode, debug=True)
    return create_app(config)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestWireFormatConsistency:
    """Wire format structure is identical regardless of deployment mode."""

    async def test_wire_format_version_byte(self):
        """Version byte is 0x01 in both cloud and local mode."""
        key = generate_master_key()
        plaintext = b"wire format version test"

        for mode in (DeploymentMode.CLOUD, DeploymentMode.LOCAL):
            # Create app to ensure mode is active
            _make_app(mode)
            blob = encrypt(plaintext, key)
            parsed = _parse_wire_format(blob)
            assert parsed["version"] == _FORMAT_VERSION, (
                f"Version byte mismatch in {mode.value} mode"
            )

    async def test_wire_format_field_lengths(self):
        """Salt (16B), nonce (12B), and tag (16B) sizes are constant."""
        key = generate_master_key()
        plaintext = b"field length consistency"

        for mode in (DeploymentMode.CLOUD, DeploymentMode.LOCAL):
            _make_app(mode)
            blob = encrypt(plaintext, key)
            parsed = _parse_wire_format(blob)

            assert len(parsed["salt"]) == _SALT_LENGTH == 16, (
                f"Salt length wrong in {mode.value}"
            )
            assert len(parsed["nonce"]) == _NONCE_LENGTH == 12, (
                f"Nonce length wrong in {mode.value}"
            )
            assert len(parsed["tag"]) == 16, (
                f"GCM tag length wrong in {mode.value}"
            )

    async def test_total_overhead_consistent(self):
        """Total overhead (header + tag) is 1 + 16 + 12 + 16 = 45 bytes."""
        key = generate_master_key()
        plaintext = b"overhead test"
        expected_overhead = 1 + _SALT_LENGTH + _NONCE_LENGTH + 16  # 45

        for mode in (DeploymentMode.CLOUD, DeploymentMode.LOCAL):
            _make_app(mode)
            blob = encrypt(plaintext, key)
            actual_overhead = len(blob) - len(plaintext)
            assert actual_overhead == expected_overhead, (
                f"Overhead {actual_overhead} != {expected_overhead} in {mode.value}"
            )

    async def test_header_layout_matches_struct(self):
        """First byte is version, followed by salt and nonce contiguously."""
        key = generate_master_key()
        plaintext = b"header layout check"

        blob = encrypt(plaintext, key)

        # version is a single byte
        (version,) = struct.unpack("B", blob[:1])
        assert version == _FORMAT_VERSION

        # Header is exactly _HEADER_LENGTH bytes
        header = blob[:_HEADER_LENGTH]
        assert len(header) == 1 + _SALT_LENGTH + _NONCE_LENGTH


class TestCrossModEncryptDecrypt:
    """Encrypt in one mode, decrypt in the other — must be identical."""

    async def test_cloud_encrypt_local_decrypt(self):
        """Data encrypted with cloud config decrypts correctly in local mode."""
        key = generate_master_key()
        plaintext = b"encrypted in cloud, decrypted in local"

        # Encrypt under cloud mode
        _make_app(DeploymentMode.CLOUD)
        blob = encrypt(plaintext, key)

        # Decrypt under local mode
        _make_app(DeploymentMode.LOCAL)
        result = decrypt(blob, key)
        assert result == plaintext

    async def test_local_encrypt_cloud_decrypt(self):
        """Data encrypted in local mode decrypts correctly under cloud config."""
        key = generate_master_key()
        plaintext = b"encrypted in local, decrypted in cloud"

        _make_app(DeploymentMode.LOCAL)
        blob = encrypt(plaintext, key)

        _make_app(DeploymentMode.CLOUD)
        result = decrypt(blob, key)
        assert result == plaintext

    async def test_json_cross_mode_roundtrip(self):
        """JSON encrypt/decrypt is consistent across modes."""
        key = generate_master_key()
        data = {
            "user": "alice",
            "episodes": [1, 2, 3],
            "metadata": {"mode": "cross-mode-test"},
        }

        # Encrypt JSON in cloud mode
        _make_app(DeploymentMode.CLOUD)
        blob = encrypt_json(data, key)

        # Decrypt JSON in local mode
        _make_app(DeploymentMode.LOCAL)
        result = decrypt_json(blob, key)
        assert result == data

    async def test_cross_mode_with_aad(self):
        """AAD binding works across modes — same AAD decrypts, different fails."""
        key = generate_master_key()
        plaintext = b"aad-bound data"
        aad = b"user-id:e2e-test-user"

        # Encrypt in cloud with AAD
        _make_app(DeploymentMode.CLOUD)
        blob = encrypt(plaintext, key, aad=aad)

        # Decrypt in local with matching AAD
        _make_app(DeploymentMode.LOCAL)
        result = decrypt(blob, key, aad=aad)
        assert result == plaintext

        # Wrong AAD must fail
        with pytest.raises(DecryptionError):
            decrypt(blob, key, aad=b"wrong-user")

    async def test_data_encryptor_cross_mode(self):
        """DataEncryptor instances with same key produce interoperable blobs."""
        key = generate_master_key()
        plaintext = b"DataEncryptor cross-mode interop"

        # Encrypt via DataEncryptor in cloud mode
        _make_app(DeploymentMode.CLOUD)
        enc_cloud = DataEncryptor(master_key=key)
        blob = enc_cloud.encrypt(plaintext)

        # Decrypt via DataEncryptor in local mode
        _make_app(DeploymentMode.LOCAL)
        enc_local = DataEncryptor(master_key=key)
        result = enc_local.decrypt(blob)
        assert result == plaintext


class TestEncryptionBothModes:
    """Encryption/decryption works correctly in each mode independently."""

    @pytest.mark.parametrize("mode", [DeploymentMode.CLOUD, DeploymentMode.LOCAL])
    async def test_bytes_roundtrip(self, mode: DeploymentMode):
        """Raw bytes encrypt → decrypt roundtrip in both modes."""
        _make_app(mode)
        key = generate_master_key()
        plaintext = f"roundtrip in {mode.value}".encode()
        blob = encrypt(plaintext, key)
        assert decrypt(blob, key) == plaintext

    @pytest.mark.parametrize("mode", [DeploymentMode.CLOUD, DeploymentMode.LOCAL])
    async def test_json_roundtrip(self, mode: DeploymentMode):
        """JSON object encrypt → decrypt roundtrip in both modes."""
        _make_app(mode)
        key = generate_master_key()
        data = {"mode": mode.value, "items": [1, "two", 3.0, None, True]}
        blob = encrypt_json(data, key)
        assert decrypt_json(blob, key) == data

    @pytest.mark.parametrize("mode", [DeploymentMode.CLOUD, DeploymentMode.LOCAL])
    async def test_empty_plaintext(self, mode: DeploymentMode):
        """Empty bytes encrypt/decrypt correctly."""
        _make_app(mode)
        key = generate_master_key()
        blob = encrypt(b"", key)
        assert decrypt(blob, key) == b""

    @pytest.mark.parametrize("mode", [DeploymentMode.CLOUD, DeploymentMode.LOCAL])
    async def test_large_payload(self, mode: DeploymentMode):
        """Large payloads (1MB) encrypt/decrypt without issues."""
        _make_app(mode)
        key = generate_master_key()
        plaintext = b"x" * (1024 * 1024)  # 1 MB
        blob = encrypt(plaintext, key)
        assert decrypt(blob, key) == plaintext

    @pytest.mark.parametrize("mode", [DeploymentMode.CLOUD, DeploymentMode.LOCAL])
    async def test_wrong_key_fails(self, mode: DeploymentMode):
        """Decryption with wrong key raises DecryptionError in both modes."""
        _make_app(mode)
        key1 = generate_master_key()
        key2 = generate_master_key()
        blob = encrypt(b"secret", key1)
        with pytest.raises(DecryptionError):
            decrypt(blob, key2)


class TestCredentialEncryptorCrossMode:
    """Legacy CredentialEncryptor (Fernet) also works across modes."""

    async def test_credential_roundtrip_cloud(self):
        """CredentialEncryptor roundtrip in cloud mode."""
        _make_app(DeploymentMode.CLOUD)
        key = generate_master_key()
        enc = CredentialEncryptor(master_key=key)
        data = {"token": "abc123", "refresh": "xyz789"}
        blob = enc.encrypt(data)
        assert enc.decrypt(blob) == data

    async def test_credential_roundtrip_local(self):
        """CredentialEncryptor roundtrip in local mode."""
        _make_app(DeploymentMode.LOCAL)
        key = generate_master_key()
        enc = CredentialEncryptor(master_key=key)
        data = {"token": "local-tok", "secret": "local-sec"}
        blob = enc.encrypt(data)
        assert enc.decrypt(blob) == data

    async def test_credential_cross_mode_interop(self):
        """Credentials encrypted in cloud mode decrypt in local mode."""
        key = generate_master_key()
        data = {"api_key": "cross-mode-key", "user": "alice"}

        _make_app(DeploymentMode.CLOUD)
        enc_cloud = CredentialEncryptor(master_key=key)
        blob = enc_cloud.encrypt(data)

        _make_app(DeploymentMode.LOCAL)
        enc_local = CredentialEncryptor(master_key=key)
        result = enc_local.decrypt(blob)
        assert result == data


class TestKeyManagementCrossMode:
    """Key generation, save, and load are mode-independent."""

    async def test_key_save_load_roundtrip(self, tmp_path: Path):
        """Master key persists and reloads correctly."""
        key = generate_master_key()
        assert len(key) == _KEY_LENGTH

        key_file = tmp_path / "master.key"
        save_master_key(key, key_file)
        loaded = load_master_key(key_file)
        assert loaded == key

    async def test_key_from_file_encrypts_identically(self, tmp_path: Path):
        """DataEncryptor loaded from key file produces same decryption as raw key."""
        key = generate_master_key()
        key_file = tmp_path / "test.key"
        save_master_key(key, key_file)

        enc_raw = DataEncryptor(master_key=key)
        enc_file = DataEncryptor(key_path=key_file)

        plaintext = b"key file interop"
        blob = enc_raw.encrypt(plaintext)
        assert enc_file.decrypt(blob) == plaintext

    async def test_key_file_permissions(self, tmp_path: Path):
        """Key file is saved with 0o600 permissions."""
        key = generate_master_key()
        key_file = tmp_path / "perms.key"
        save_master_key(key, key_file)
        mode = key_file.stat().st_mode & 0o777
        assert mode == 0o600


class TestFileIOCrossMode:
    """File-based encrypt/decrypt works identically in both modes."""

    async def test_encrypt_to_file_decrypt_cross_mode(self, tmp_path: Path):
        """Encrypt to file in cloud mode, decrypt from file in local mode."""
        key = generate_master_key()
        plaintext = b"file-based cross-mode data"
        filepath = tmp_path / "encrypted.bin"

        _make_app(DeploymentMode.CLOUD)
        enc = DataEncryptor(master_key=key)
        enc.encrypt_to_file(plaintext, filepath)

        _make_app(DeploymentMode.LOCAL)
        dec = DataEncryptor(master_key=key)
        result = dec.decrypt_from_file(filepath)
        assert result == plaintext

    async def test_json_file_cross_mode(self, tmp_path: Path):
        """JSON encrypt-to-file in local, decrypt-from-file in cloud."""
        key = generate_master_key()
        data = {"episodes": [{"id": 1, "text": "hello"}], "count": 1}
        filepath = tmp_path / "data.enc"

        _make_app(DeploymentMode.LOCAL)
        enc = DataEncryptor(master_key=key)
        enc.encrypt_json_to_file(data, filepath)

        _make_app(DeploymentMode.CLOUD)
        dec = DataEncryptor(master_key=key)
        result = dec.decrypt_json_from_file(filepath)
        assert result == data

    async def test_encrypted_file_has_correct_wire_format(self, tmp_path: Path):
        """File contents follow the same wire format as in-memory blobs."""
        key = generate_master_key()
        plaintext = b"wire format file check"
        filepath = tmp_path / "wire.bin"

        enc = DataEncryptor(master_key=key)
        enc.encrypt_to_file(plaintext, filepath)

        raw = filepath.read_bytes()
        parsed = _parse_wire_format(raw)
        assert parsed["version"] == _FORMAT_VERSION
        assert len(parsed["salt"]) == _SALT_LENGTH
        assert len(parsed["nonce"]) == _NONCE_LENGTH
        assert len(parsed["tag"]) == 16


class TestCorruptionDetection:
    """Tampered ciphertext is detected in both modes."""

    @pytest.mark.parametrize("mode", [DeploymentMode.CLOUD, DeploymentMode.LOCAL])
    async def test_tampered_ciphertext_detected(self, mode: DeploymentMode):
        """Flipping a byte in ciphertext raises DecryptionError."""
        _make_app(mode)
        key = generate_master_key()
        blob = encrypt(b"tamper test", key)

        # Flip a byte in the ciphertext (after header)
        tampered = bytearray(blob)
        tampered[_HEADER_LENGTH + 1] ^= 0xFF
        with pytest.raises(DecryptionError):
            decrypt(bytes(tampered), key)

    @pytest.mark.parametrize("mode", [DeploymentMode.CLOUD, DeploymentMode.LOCAL])
    async def test_truncated_blob_detected(self, mode: DeploymentMode):
        """Truncated blob raises CorruptedDataError."""
        _make_app(mode)
        key = generate_master_key()
        blob = encrypt(b"truncate test", key)

        with pytest.raises(CorruptedDataError):
            decrypt(blob[:10], key)

    @pytest.mark.parametrize("mode", [DeploymentMode.CLOUD, DeploymentMode.LOCAL])
    async def test_wrong_version_byte_rejected(self, mode: DeploymentMode):
        """Invalid version byte raises CorruptedDataError."""
        _make_app(mode)
        key = generate_master_key()
        blob = encrypt(b"version test", key)

        # Set version to 0xFF
        tampered = bytearray(blob)
        tampered[0] = 0xFF
        with pytest.raises(CorruptedDataError):
            decrypt(bytes(tampered), key)


class TestAADBoundaryBehavior:
    """AAD (Additional Authenticated Data) behaves identically across modes."""

    async def test_aad_none_vs_empty_bytes(self):
        """aad=None and aad omitted produce blobs decodable without AAD."""
        key = generate_master_key()
        plaintext = b"aad boundary"

        blob_none = encrypt(plaintext, key, aad=None)
        assert decrypt(blob_none, key, aad=None) == plaintext

    async def test_aad_mismatch_fails_both_modes(self):
        """Mismatched AAD causes decryption failure in both modes."""
        key = generate_master_key()
        plaintext = b"aad mismatch test"
        aad = b"correct-user"

        for mode in (DeploymentMode.CLOUD, DeploymentMode.LOCAL):
            _make_app(mode)
            blob = encrypt(plaintext, key, aad=aad)
            # Correct AAD works
            assert decrypt(blob, key, aad=aad) == plaintext
            # Wrong AAD fails
            with pytest.raises(DecryptionError):
                decrypt(blob, key, aad=b"wrong-user")

    async def test_aad_user_binding_cross_mode(self):
        """User-ID as AAD binds ciphertext to user across modes."""
        key = generate_master_key()
        user_aad = b"user:e2e-test-user"
        episode_data = json.dumps({"text": "secret note", "intent": "journal"}).encode()

        _make_app(DeploymentMode.CLOUD)
        blob = encrypt(episode_data, key, aad=user_aad)

        _make_app(DeploymentMode.LOCAL)
        result = decrypt(blob, key, aad=user_aad)
        assert json.loads(result) == {"text": "secret note", "intent": "journal"}


class TestEncryptionWithAppLifecycle:
    """Encryption works correctly when driven through the FastAPI app lifecycle."""

    async def test_encrypt_decrypt_with_cloud_app(self, client: httpx.AsyncClient):
        """Encryption works alongside the cloud-mode app."""
        key = generate_master_key()
        enc = DataEncryptor(master_key=key)

        # Verify app is functional
        resp = await client.get("/health")
        assert resp.status_code == 200

        # Encrypt/decrypt still works
        data = {"captured_via": "cloud-app", "test": True}
        blob = enc.encrypt_json(data)
        assert enc.decrypt_json(blob) == data

    async def test_encrypt_episode_data_structure(self, client: httpx.AsyncClient):
        """Encrypt a realistic episode payload and verify roundtrip."""
        key = generate_master_key()
        enc = DataEncryptor(master_key=key)

        episode = {
            "id": "ep-001",
            "user_id": "e2e-test-user",
            "raw_text": "I need to buy groceries",
            "intent": "task",
            "intent_confidence": 0.92,
            "emotion": {
                "primary": "trust",
                "intensity": 0.5,
                "valence": 0.0,
                "arousal": 0.2,
            },
            "entities": [{"name": "groceries", "type": "item"}],
        }

        blob = enc.encrypt_json(episode, aad=b"user:e2e-test-user")
        parsed = _parse_wire_format(blob)

        # Verify wire format invariants
        assert parsed["version"] == _FORMAT_VERSION
        assert len(parsed["salt"]) == 16
        assert len(parsed["nonce"]) == 12
        assert len(parsed["tag"]) == 16

        # Verify decryption roundtrip
        result = enc.decrypt_json(blob, aad=b"user:e2e-test-user")
        assert result == episode
