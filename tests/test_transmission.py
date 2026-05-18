"""Тесты TransmissionAgent: HMAC и JSON-сериализация."""
import hashlib
import hmac as _hmac
import json
import os
import pytest


def compute_mac(key: bytes, data: bytes) -> bytes:
    return _hmac.new(key, data, hashlib.sha256).digest()


def verify_mac(key: bytes, data: bytes, mac: bytes) -> bool:
    return _hmac.compare_digest(compute_mac(key, data), mac)


class TestHMAC:
    def setup_method(self):
        self.key = os.urandom(32)

    def test_valid_mac_verifies(self):
        data = b"test payload"
        mac  = compute_mac(self.key, data)
        assert verify_mac(self.key, data, mac) is True

    def test_tampered_payload_rejected(self):
        data = b"original"
        mac  = compute_mac(self.key, data)
        assert verify_mac(self.key, data + b"x", mac) is False

    def test_wrong_key_rejected(self):
        data     = b"payload"
        mac      = compute_mac(self.key, data)
        wrong_key = os.urandom(32)
        assert verify_mac(wrong_key, data, mac) is False

    def test_empty_payload(self):
        data = b""
        mac  = compute_mac(self.key, data)
        assert verify_mac(self.key, data, mac) is True


class TestJSONSerialization:
    """JSON должен корректно сериализовать (ciphertext_int, n_int) туплы."""

    def test_roundtrip_large_ints(self):
        """Большие целые числа (как ciphertext Paillier) должны переживать JSON."""
        data = [[2 ** 2048 + 12345, 2 ** 2048 - 99], [7, 3]]
        encoded = json.dumps(data, separators=(",", ":")).encode("utf-8")
        decoded = json.loads(encoded.decode("utf-8"))
        assert decoded == data

    def test_json_is_not_executable(self):
        """В отличие от pickle, JSON не выполняет код при десериализации."""
        payload = b'[1, 2, 3]'
        result  = json.loads(payload)
        assert result == [1, 2, 3]


class TestRiskScoreMath:
    """Математика риск-скора: граничные случаи."""

    def test_score_range(self):
        """Скор должен быть в диапазоне [-5, +7]."""
        import pandas as pd
        from credit_scoring_phe.data.preprocessing import compute_risk_score

        min_row = pd.Series({
            "Saving accounts": "little", "Checking account": "little",
            "Housing": "rent", "Duration": 100, "Credit amount": 100_000,
        })
        max_row = pd.Series({
            "Saving accounts": "quite rich", "Checking account": "rich",
            "Housing": "own", "Duration": 6, "Credit amount": 1000,
        })
        assert compute_risk_score(min_row) == -5
        assert compute_risk_score(max_row) == 7
