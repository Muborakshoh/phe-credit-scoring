"""Агент передачи: JSON-сериализация, HMAC-SHA256, retry-логика."""
import hashlib
import hmac as _hmac
import json
import logging
import os
import time
import threading
from typing import Optional

import mesa

logger = logging.getLogger(__name__)


class TransmissionAgent(mesa.Agent):
    """Сериализация (JSON), проверка целостности (HMAC-SHA256), симуляция сети.

    Использует JSON вместо pickle: безопасная сериализация без риска
    выполнения произвольного кода при десериализации.
    """

    def __init__(self, model, max_retries: int = 3,
                 simulated_latency_ms: float = 0.0):
        super().__init__(model)
        self.max_retries          = max_retries
        self.simulated_latency_ms = simulated_latency_ms
        self._hmac_key            = os.urandom(32)   # shared secret (симуляция TLS)
        self._lock                = threading.Lock()

        self.sent_count         = 0
        self.retry_count        = 0
        self.integrity_failures = 0

    # ── Сериализация (JSON) ───────────────────────────────────────────────────

    def _serialize(self, enc_vector: list) -> bytes:
        """Сериализует зашифрованный вектор в JSON-байты.

        Каждый EncryptedNumber представляется как [ciphertext_int, n_int].
        JSON безопаснее pickle: не позволяет выполнить произвольный код.
        """
        data = [
            [int(ct.ciphertext()), int(ct.public_key.n)]
            for ct in enc_vector
        ]
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    # ── HMAC ─────────────────────────────────────────────────────────────────

    def _compute_mac(self, data: bytes) -> bytes:
        """HMAC-SHA256 — имитация TLS record MAC."""
        return _hmac.new(self._hmac_key, data, hashlib.sha256).digest()

    def _verify_mac(self, data: bytes, mac: bytes) -> bool:
        return _hmac.compare_digest(self._compute_mac(data), mac)

    # ── Отправка / получение ──────────────────────────────────────────────────

    def send(self, enc_vector: list) -> tuple[bytes, bytes]:
        """Сериализует вектор и вычисляет MAC.

        Returns:
            (payload_bytes, mac_bytes)
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                payload = self._serialize(enc_vector)
                mac     = self._compute_mac(payload)
                if self.simulated_latency_ms > 0:
                    time.sleep(self.simulated_latency_ms / 1000)
                with self._lock:
                    self.sent_count += 1
                logger.debug("Пакет отправлен (попытка %d)", attempt + 1)
                return payload, mac
            except Exception as exc:
                last_exc = exc
                with self._lock:
                    self.retry_count += 1
                logger.warning(
                    "Ошибка отправки (попытка %d/%d): %s",
                    attempt + 1, self.max_retries, exc,
                )

        raise RuntimeError(
            f"[TransmissionAgent] Передача не удалась после "
            f"{self.max_retries} попыток: {last_exc}"
        )

    def receive(self, payload: bytes, mac: bytes,
                enc_vector_ref: list) -> list:
        """Верифицирует MAC и возвращает вектор для сервера.

        SIMULATION NOTE: В продакшене сервер восстанавливал бы EncryptedNumber
        из JSON-байт payload. Здесь возвращаем enc_vector_ref напрямую — для
        упрощения симуляции. MAC-проверка выше полностью воспроизводит протокол.

        Raises:
            ValueError: при несовпадении MAC (атака MITM или повреждение данных)
        """
        if not self._verify_mac(payload, mac):
            with self._lock:
                self.integrity_failures += 1
            logger.error("ОШИБКА ЦЕЛОСТНОСТИ: MAC не совпадает — возможна атака MITM")
            raise ValueError(
                "[TransmissionAgent] ОШИБКА ЦЕЛОСТНОСТИ: MAC не совпадает."
            )
        logger.debug("MAC верифицирован успешно")
        return enc_vector_ref

    # ── Mesa / отчёт ──────────────────────────────────────────────────────────

    def step(self):
        pass

    def report(self) -> str:
        msg = (
            f"TransmissionAgent: отправлено {self.sent_count} пакетов, "
            f"повторов {self.retry_count}, "
            f"ошибок целостности {self.integrity_failures}"
        )
        logger.info(msg)
        return msg
