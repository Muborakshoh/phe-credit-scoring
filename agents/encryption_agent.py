"""Агент шифрования: Paillier PHE + сохранение/загрузка модели."""
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import mesa
from phe import paillier

logger = logging.getLogger(__name__)


class EncryptionAgent(mesa.Agent):
    """Шифрование Paillier с поддержкой персистентности ключей и весов модели.

    Args:
        key_bits: длина модуля n. 2048 → ~112-bit, 3072 → ~128-bit (продакшен).
    """

    def __init__(self, model, key_bits: int = 2048):
        super().__init__(model)
        self.key_bits      = key_bits
        self.encrypt_calls = 0

        logger.info("Генерация ключей Paillier (%d бит)...", key_bits)
        t0 = time.perf_counter()
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=key_bits
        )
        elapsed = time.perf_counter() - t0
        logger.info("Ключи сгенерированы за %.2fs", elapsed)

    # ── Шифрование / расшифровка ─────────────────────────────────────────────

    def encrypt_vector(self, x: np.ndarray) -> list:
        """Шифрует вектор признаков: c_i = E_PK(x_i).

        Paillier автоматически добавляет случайный r → вероятностное шифрование.
        """
        self.encrypt_calls += 1
        logger.debug("Шифрование вектора длиной %d", len(x))
        return [self.public_key.encrypt(float(v)) for v in x]

    def decrypt_value(self, ciphertext) -> float:
        """Расшифровывает одно значение на стороне клиента."""
        return self.private_key.decrypt(ciphertext)

    # ── Сохранение / загрузка артефактов ─────────────────────────────────────

    def save_model(self, weights: np.ndarray, bias: float,
                   scaler, model_dir: str = "artifacts") -> Path:
        """Сохраняет веса, смещение и scaler на диск (joblib).

        Args:
            weights:   веса логистической регрессии
            bias:      смещение
            scaler:    обученный StandardScaler
            model_dir: директория для артефактов

        Returns:
            Путь к сохранённому файлу
        """
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        path = Path(model_dir) / "model.pkl"
        artifact = {
            "weights":  weights,
            "bias":     bias,
            "scaler":   scaler,
            "key_bits": self.key_bits,
        }
        joblib.dump(artifact, path)
        logger.info("Модель сохранена → %s", path)
        return path

    @staticmethod
    def load_model(model_dir: str = "artifacts") -> dict:
        """Загружает артефакты с диска.

        Returns:
            dict с ключами: weights, bias, scaler, key_bits
        """
        path = Path(model_dir) / "model.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Артефакт не найден: {path}")
        artifact = joblib.load(path)
        logger.info("Модель загружена ← %s", path)
        return artifact

    def step(self):
        pass
