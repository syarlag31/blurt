"""Tests for E2E encryption: data stored encrypted, correct key decrypts, wrong key fails.

Covers both:
- AES-256-GCM (DataEncryptor) for data at rest before cloud storage
- Legacy CredentialEncryptor (Fernet/AES-128-CBC) for backward compatibility
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import pytest

from blurt.core.encryption import (
    CorruptedDataError,
    CredentialEncryptor,
    DataEncryptor,
    DecryptionError,
    InvalidKeyError,
    decrypt,
    decrypt_from_file,
    decrypt_json,
    decrypt_json_from_file,
    encrypt,
    encrypt_json,
    encrypt_json_to_file,
    encrypt_to_file,
    generate_master_key,
    load_master_key,
    load_or_generate_master_key,
    save_master_key,
    _FORMAT_VERSION,
    _HEADER_LENGTH,
    _KEY_LENGTH,
)


# ---------------------------------------------------------------------------
# AES-256-GCM Key Management
# ---------------------------------------------------------------------------


class TestAES256KeyManagement:
    def test_generate_master_key_length(self):
        key = generate_master_key()
        assert len(key) == _KEY_LENGTH == 32

    def test_generate_master_key_randomness(self):
        assert generate_master_key() != generate_master_key()

    def test_save_and_load_master_key(self, tmp_path: Path):
        key = generate_master_key()
        key_file = tmp_path / "master.key"
        save_master_key(key, key_file)
        assert load_master_key(key_file) == key

    def test_save_sets_600_permissions(self, tmp_path: Path):
        save_master_key(generate_master_key(), tmp_path / "k.key")
        assert oct(os.stat(tmp_path / "k.key").st_mode & 0o777) == "0o600"

    def test_save_invalid_key_length_raises(self, tmp_path: Path):
        with pytest.raises(InvalidKeyError, match="32 bytes"):
            save_master_key(b"short", tmp_path / "bad.key")

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_master_key(tmp_path / "nonexistent.key")

    def test_load_or_generate_creates_new(self, tmp_path: Path):
        key_file = tmp_path / "auto.key"
        key = load_or_generate_master_key(key_file)
        assert len(key) == 32 and key_file.exists()

    def test_load_or_generate_loads_existing(self, tmp_path: Path):
        key_file = tmp_path / "auto.key"
        original = generate_master_key()
        save_master_key(original, key_file)
        assert load_or_generate_master_key(key_file) == original


# ---------------------------------------------------------------------------
# AES-256-GCM Encrypt / Decrypt (functional API)
# ---------------------------------------------------------------------------


class TestAES256GCMEncryptDecrypt:
    @pytest.fixture()
    def key(self) -> bytes:
        return generate_master_key()

    def test_roundtrip_bytes(self, key: bytes):
        plaintext = b"Hello, Blurt!"
        assert decrypt(encrypt(plaintext, key), key) == plaintext

    def test_roundtrip_empty_bytes(self, key: bytes):
        assert decrypt(encrypt(b"", key), key) == b""

    def test_roundtrip_large_payload(self, key: bytes):
        data = os.urandom(1024 * 1024)  # 1 MB
        assert decrypt(encrypt(data, key), key) == data

    def test_ciphertext_differs_each_call(self, key: bytes):
        """Random salt + nonce produce unique ciphertext."""
        pt = b"same input"
        assert encrypt(pt, key) != encrypt(pt, key)

    def test_format_version_in_header(self, key: bytes):
        assert encrypt(b"x", key)[0] == _FORMAT_VERSION

    def test_output_length(self, key: bytes):
        pt = b"hello"
        ct = encrypt(pt, key)
        # version(1) + salt(16) + nonce(12) + ciphertext(5) + GCM tag(16) = 50
        assert len(ct) == _HEADER_LENGTH + len(pt) + 16

    def test_no_plaintext_in_ciphertext(self, key: bytes):
        pt = b"super-secret-plaintext-value-12345"
        ct = encrypt(pt, key)
        assert pt not in ct

    def test_wrong_key_fails(self, key: bytes):
        ct = encrypt(b"secret", key)
        with pytest.raises(DecryptionError):
            decrypt(ct, generate_master_key())

    def test_tampered_ciphertext_fails(self, key: bytes):
        ct = bytearray(encrypt(b"secret", key))
        ct[-5] ^= 0xFF
        with pytest.raises(DecryptionError):
            decrypt(bytes(ct), key)

    def test_tampered_nonce_fails(self, key: bytes):
        ct = bytearray(encrypt(b"secret", key))
        ct[1 + 16] ^= 0xFF  # nonce starts after version+salt
        with pytest.raises(DecryptionError):
            decrypt(bytes(ct), key)

    def test_truncated_data_raises_corrupted(self, key: bytes):
        with pytest.raises(CorruptedDataError, match="too short"):
            decrypt(b"\x01" + b"\x00" * 10, key)

    def test_wrong_version_raises_corrupted(self, key: bytes):
        ct = bytearray(encrypt(b"test", key))
        ct[0] = 0xFF
        with pytest.raises(CorruptedDataError, match="Unsupported format version"):
            decrypt(bytes(ct), key)

    def test_invalid_key_length_encrypt(self):
        with pytest.raises(InvalidKeyError):
            encrypt(b"test", b"short_key")

    def test_invalid_key_length_decrypt(self):
        with pytest.raises(InvalidKeyError):
            decrypt(b"\x01" + b"\x00" * 50, b"short_key")


# ---------------------------------------------------------------------------
# Additional Authenticated Data (AAD)
# ---------------------------------------------------------------------------


class TestAAD:
    @pytest.fixture()
    def key(self) -> bytes:
        return generate_master_key()

    def test_roundtrip_with_aad(self, key: bytes):
        aad = b"user:abc123"
        pt = b"sensitive data"
        assert decrypt(encrypt(pt, key, aad=aad), key, aad=aad) == pt

    def test_wrong_aad_fails(self, key: bytes):
        ct = encrypt(b"data", key, aad=b"user:abc123")
        with pytest.raises(DecryptionError):
            decrypt(ct, key, aad=b"user:wrong")

    def test_missing_aad_fails(self, key: bytes):
        ct = encrypt(b"data", key, aad=b"context")
        with pytest.raises(DecryptionError):
            decrypt(ct, key)  # no AAD

    def test_unexpected_aad_fails(self, key: bytes):
        ct = encrypt(b"data", key)  # no AAD
        with pytest.raises(DecryptionError):
            decrypt(ct, key, aad=b"unexpected")


# ---------------------------------------------------------------------------
# JSON Convenience (AES-256-GCM)
# ---------------------------------------------------------------------------


class TestJSONEncryption:
    @pytest.fixture()
    def key(self) -> bytes:
        return generate_master_key()

    def test_roundtrip_dict(self, key: bytes):
        data = {"name": "Blurt", "version": 1, "nested": {"k": "v"}}
        assert decrypt_json(encrypt_json(data, key), key) == data

    def test_roundtrip_list(self, key: bytes):
        data = [1, "two", {"three": 3}]
        assert decrypt_json(encrypt_json(data, key), key) == data

    def test_unicode_data(self, key: bytes):
        data = {"emoji": "🧠", "jp": "こんにちは"}
        assert decrypt_json(encrypt_json(data, key), key) == data

    def test_json_with_aad(self, key: bytes):
        data = {"secret": "value"}
        aad = b"episode:xyz"
        assert decrypt_json(encrypt_json(data, key, aad=aad), key, aad=aad) == data


# ---------------------------------------------------------------------------
# File I/O (AES-256-GCM)
# ---------------------------------------------------------------------------


class TestAES256FileIO:
    @pytest.fixture()
    def key(self) -> bytes:
        return generate_master_key()

    def test_encrypt_decrypt_file_bytes(self, key: bytes, tmp_path: Path):
        pt = b"file content"
        fp = tmp_path / "enc.bin"
        encrypt_to_file(pt, key, fp)
        assert decrypt_from_file(fp, key) == pt

    def test_file_permissions(self, key: bytes, tmp_path: Path):
        fp = tmp_path / "secure.bin"
        encrypt_to_file(b"d", key, fp)
        assert oct(os.stat(fp).st_mode & 0o777) == "0o600"

    def test_decrypt_missing_file(self, key: bytes, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            decrypt_from_file(tmp_path / "nope.bin", key)

    def test_encrypt_decrypt_json_file(self, key: bytes, tmp_path: Path):
        data = {"episodes": [1, 2, 3]}
        fp = tmp_path / "data.enc"
        encrypt_json_to_file(data, key, fp)
        assert decrypt_json_from_file(fp, key) == data

    def test_creates_parent_dirs(self, key: bytes, tmp_path: Path):
        fp = tmp_path / "deep" / "nested" / "file.enc"
        encrypt_to_file(b"data", key, fp)
        assert fp.exists()

    def test_file_with_aad(self, key: bytes, tmp_path: Path):
        fp = tmp_path / "aad.bin"
        aad = b"ctx"
        encrypt_to_file(b"payload", key, fp, aad=aad)
        assert decrypt_from_file(fp, key, aad=aad) == b"payload"


# ---------------------------------------------------------------------------
# DataEncryptor (stateful AES-256-GCM wrapper)
# ---------------------------------------------------------------------------


class TestDataEncryptor:
    def test_ephemeral_key(self):
        enc = DataEncryptor()
        assert enc.decrypt(enc.encrypt(b"test")) == b"test"

    def test_explicit_key(self):
        key = generate_master_key()
        enc = DataEncryptor(master_key=key)
        assert enc.decrypt(enc.encrypt(b"hello")) == b"hello"

    def test_key_from_file(self, tmp_path: Path):
        kf = tmp_path / "enc.key"
        enc1 = DataEncryptor(key_path=kf)
        ct = enc1.encrypt(b"persist")
        enc2 = DataEncryptor(key_path=kf)
        assert enc2.decrypt(ct) == b"persist"

    def test_invalid_key_length(self):
        with pytest.raises(InvalidKeyError):
            DataEncryptor(master_key=b"too_short")

    def test_json_methods(self):
        enc = DataEncryptor()
        data = {"key": "value", "num": 42}
        assert enc.decrypt_json(enc.encrypt_json(data)) == data

    def test_file_methods(self, tmp_path: Path):
        enc = DataEncryptor()
        fp = tmp_path / "test.enc"
        enc.encrypt_to_file(b"file data", fp)
        assert enc.decrypt_from_file(fp) == b"file data"

    def test_json_file_methods(self, tmp_path: Path):
        enc = DataEncryptor()
        fp = tmp_path / "json.enc"
        data = {"list": [1, 2, 3]}
        enc.encrypt_json_to_file(data, fp)
        assert enc.decrypt_json_from_file(fp) == data

    def test_aad_support(self):
        enc = DataEncryptor()
        aad = b"user:123"
        assert enc.decrypt(enc.encrypt(b"bound", aad=aad), aad=aad) == b"bound"

    def test_cross_instance_same_key(self):
        key = generate_master_key()
        enc1 = DataEncryptor(master_key=key)
        enc2 = DataEncryptor(master_key=key)
        assert enc2.decrypt(enc1.encrypt(b"cross")) == b"cross"

    def test_wrong_key_fails(self):
        enc1 = DataEncryptor()
        enc2 = DataEncryptor()
        ct = enc1.encrypt(b"secret")
        with pytest.raises(DecryptionError):
            enc2.decrypt(ct)


# ===========================================================================
# Legacy CredentialEncryptor (Fernet / AES-128-CBC) Tests
# ===========================================================================


class TestEncryptedStorage:
    """Verify that data is actually stored in encrypted form (not plaintext)."""

    def test_encrypted_bytes_differ_from_plaintext(self):
        """Encrypted output must not contain the plaintext JSON."""
        encryptor = CredentialEncryptor(master_key=secrets.token_bytes(32))
        data = {"access_token": "super-secret-token-12345", "refresh_token": "refresh-abc"}

        encrypted = encryptor.encrypt(data)

        # The encrypted bytes must not contain any plaintext values
        assert b"super-secret-token-12345" not in encrypted
        assert b"refresh-abc" not in encrypted
        assert b"access_token" not in encrypted

    def test_encrypted_file_contains_no_plaintext(self, tmp_path: Path):
        """Data written to file must be encrypted — no plaintext leakage."""
        encryptor = CredentialEncryptor(master_key=secrets.token_bytes(32))
        data = {"secret": "my-private-value", "api_key": "key-999"}
        file_path = tmp_path / "creds.enc"

        encryptor.encrypt_to_file(data, file_path)

        raw_bytes = file_path.read_bytes()
        assert b"my-private-value" not in raw_bytes
        assert b"key-999" not in raw_bytes
        assert b"secret" not in raw_bytes

    def test_encrypted_output_starts_with_salt(self):
        """Encrypted output should begin with the 16-byte salt."""
        encryptor = CredentialEncryptor(master_key=secrets.token_bytes(32))
        data = {"key": "value"}

        encrypted = encryptor.encrypt(data)

        # Must be at least salt (16) + some ciphertext
        assert len(encrypted) > CredentialEncryptor.SALT_LENGTH

    def test_same_data_produces_different_ciphertext(self):
        """Each encryption uses a unique salt, so ciphertext must differ."""
        encryptor = CredentialEncryptor(master_key=secrets.token_bytes(32))
        data = {"token": "same-value"}

        encrypted1 = encryptor.encrypt(data)
        encrypted2 = encryptor.encrypt(data)

        assert encrypted1 != encrypted2


class TestDecryptionWithCorrectKey:
    """Verify that decryption with the correct key recovers original data."""

    def test_round_trip_in_memory(self):
        """Encrypt then decrypt in-memory returns identical data."""
        key = secrets.token_bytes(32)
        encryptor = CredentialEncryptor(master_key=key)
        data = {
            "access_token": "ya29.a0ARrdaM...",
            "refresh_token": "1//0dx...",
            "token_uri": "https://oauth2.googleapis.com/token",
            "expiry": "2026-03-13T12:00:00Z",
        }

        encrypted = encryptor.encrypt(data)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == data

    def test_round_trip_via_file(self, tmp_path: Path):
        """Encrypt to file then decrypt from file returns identical data."""
        key = secrets.token_bytes(32)
        encryptor = CredentialEncryptor(master_key=key)
        data = {"notion_token": "secret_abc123", "workspace_id": "ws-456"}
        file_path = tmp_path / "tokens.enc"

        encryptor.encrypt_to_file(data, file_path)
        decrypted = encryptor.decrypt_from_file(file_path)

        assert decrypted == data

    def test_same_key_different_encryptor_instance(self):
        """A new CredentialEncryptor with the same key can decrypt data."""
        key = secrets.token_bytes(32)
        encryptor1 = CredentialEncryptor(master_key=key)
        encryptor2 = CredentialEncryptor(master_key=key)

        data = {"credential": "sensitive-data"}
        encrypted = encryptor1.encrypt(data)
        decrypted = encryptor2.decrypt(encrypted)

        assert decrypted == data

    def test_key_loaded_from_file_decrypts(self, tmp_path: Path):
        """Key persisted to file can be reloaded and used to decrypt."""
        key_path = tmp_path / "master.key"
        encryptor1 = CredentialEncryptor(key_path=key_path)
        data = {"session": "abc-session-token"}

        encrypted = encryptor1.encrypt(data)

        # New instance loads the same key from file
        encryptor2 = CredentialEncryptor(key_path=key_path)
        decrypted = encryptor2.decrypt(encrypted)

        assert decrypted == data

    def test_decryption_preserves_types(self):
        """JSON types (str, int, float, bool, null, list, nested dict) survive round-trip."""
        encryptor = CredentialEncryptor(master_key=secrets.token_bytes(32))
        data = {
            "string": "hello",
            "integer": 42,
            "float_val": 3.14,
            "boolean": True,
            "null_val": None,
            "list_val": [1, 2, 3],
            "nested": {"inner": "value"},
        }

        encrypted = encryptor.encrypt(data)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == data

    def test_empty_dict_round_trip(self):
        """Empty dictionary encrypts and decrypts correctly."""
        encryptor = CredentialEncryptor(master_key=secrets.token_bytes(32))
        data: dict = {}

        encrypted = encryptor.encrypt(data)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == data


class TestDecryptionWithWrongKey:
    """Verify that decryption with wrong keys fails cleanly."""

    def test_wrong_master_key_raises_value_error(self):
        """Decryption with a different master key must fail."""
        key1 = secrets.token_bytes(32)
        key2 = secrets.token_bytes(32)
        assert key1 != key2  # Sanity check

        encryptor1 = CredentialEncryptor(master_key=key1)
        encryptor2 = CredentialEncryptor(master_key=key2)

        data = {"token": "secret-value"}
        encrypted = encryptor1.encrypt(data)

        with pytest.raises(ValueError, match="Decryption failed"):
            encryptor2.decrypt(encrypted)

    def test_wrong_key_file_raises_value_error(self, tmp_path: Path):
        """Decryption with a key from a different key file must fail."""
        key_path1 = tmp_path / "key1.key"
        key_path2 = tmp_path / "key2.key"

        encryptor1 = CredentialEncryptor(key_path=key_path1)
        encryptor2 = CredentialEncryptor(key_path=key_path2)

        data = {"credential": "private"}
        encrypted = encryptor1.encrypt(data)

        with pytest.raises(ValueError, match="Decryption failed"):
            encryptor2.decrypt(encrypted)

    def test_wrong_key_on_file_stored_data(self, tmp_path: Path):
        """Data encrypted to file cannot be read with a different key."""
        key1 = secrets.token_bytes(32)
        key2 = secrets.token_bytes(32)

        encryptor1 = CredentialEncryptor(master_key=key1)
        encryptor2 = CredentialEncryptor(master_key=key2)

        data = {"api_secret": "top-secret"}
        file_path = tmp_path / "encrypted.dat"

        encryptor1.encrypt_to_file(data, file_path)

        with pytest.raises(ValueError, match="Decryption failed"):
            encryptor2.decrypt_from_file(file_path)

    def test_corrupted_ciphertext_raises_value_error(self):
        """Tampered ciphertext must fail decryption."""
        encryptor = CredentialEncryptor(master_key=secrets.token_bytes(32))
        data = {"token": "value"}

        encrypted = encryptor.encrypt(data)

        # Flip some bytes in the ciphertext portion (after the salt)
        corrupted = bytearray(encrypted)
        mid = len(corrupted) // 2
        corrupted[mid] ^= 0xFF
        corrupted[mid + 1] ^= 0xFF

        with pytest.raises(ValueError, match="Decryption failed"):
            encryptor.decrypt(bytes(corrupted))

    def test_truncated_data_raises_value_error(self):
        """Data shorter than the salt length must fail."""
        encryptor = CredentialEncryptor(master_key=secrets.token_bytes(32))

        with pytest.raises(ValueError, match="too short"):
            encryptor.decrypt(b"short")

    def test_empty_ciphertext_after_salt_raises_value_error(self):
        """Salt-only data (no ciphertext) must fail decryption."""
        encryptor = CredentialEncryptor(master_key=secrets.token_bytes(32))
        # Exactly 16 bytes of salt, no ciphertext
        fake_data = secrets.token_bytes(CredentialEncryptor.SALT_LENGTH)

        with pytest.raises(ValueError, match="Decryption failed"):
            encryptor.decrypt(fake_data)

    def test_wrong_key_never_returns_data(self):
        """Run multiple attempts — wrong key must never accidentally succeed."""
        data = {"token": "important-secret"}

        for _ in range(10):
            correct_key = secrets.token_bytes(32)
            wrong_key = secrets.token_bytes(32)

            encryptor_correct = CredentialEncryptor(master_key=correct_key)
            encryptor_wrong = CredentialEncryptor(master_key=wrong_key)

            encrypted = encryptor_correct.encrypt(data)

            with pytest.raises(ValueError):
                encryptor_wrong.decrypt(encrypted)


class TestFilePermissions:
    """Verify encrypted files have restrictive permissions."""

    def test_encrypted_file_has_600_permissions(self, tmp_path: Path):
        """Encrypted files should only be readable by the owner."""
        encryptor = CredentialEncryptor(master_key=secrets.token_bytes(32))
        file_path = tmp_path / "secure.enc"

        encryptor.encrypt_to_file({"key": "value"}, file_path)

        mode = file_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_key_file_has_600_permissions(self, tmp_path: Path):
        """Generated key files should only be readable by the owner."""
        key_path = tmp_path / "master.key"
        CredentialEncryptor(key_path=key_path)

        mode = key_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"
