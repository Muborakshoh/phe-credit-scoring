"""Mesa-пайплайн кредитного скоринга с параллельным инференсом (threads)."""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import mesa
from tqdm import tqdm

from credit_scoring_phe.agents.monitoring_agent import MonitoringAgent
from credit_scoring_phe.agents.encryption_agent import EncryptionAgent
from credit_scoring_phe.agents.transmission_agent import TransmissionAgent
from credit_scoring_phe.agents.analysis_agent import AnalysisAgent

logger = logging.getLogger(__name__)


class CreditScoringMesaModel(mesa.Model):
    """Mesa-модель агентного пайплайна кредитного скоринга.

    Пайплайн:
        MonitoringAgent → EncryptionAgent → TransmissionAgent → AnalysisAgent

    Поддерживает параллельный батч-инференс через ThreadPoolExecutor.
    Потоки безопасны: счётчики агентов защищены threading.Lock внутри агентов.
    """

    def __init__(self, weights: np.ndarray, bias: float,
                 key_bits: int = 2048, activation: str = "sigmoid"):
        super().__init__()
        self.weights    = weights
        self.bias       = bias
        self.activation = activation

        self.monitoring   = MonitoringAgent(self)
        self.encryption   = EncryptionAgent(self, key_bits=key_bits)
        self.transmission = TransmissionAgent(self)
        self.analysis     = AnalysisAgent(self)

        logger.info(
            "CreditScoringMesaModel инициализирована | key_bits=%d | activation=%s",
            key_bits, activation,
        )

    # ── Активация ─────────────────────────────────────────────────────────────

    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_approx(self, z: float) -> float:
        """Линейная аппроксимация Тейлора 1-го порядка: σ(z) ≈ 0.5 + 0.25z."""
        return float(np.clip(0.5 + 0.25 * z, 0.0, 1.0))

    # ── Инференс для одного сэмпла ────────────────────────────────────────────

    def predict_one(self, raw_row: dict, encoded_x: np.ndarray) -> dict:
        """Полный агентный пайплайн для одного сэмпла.

        Returns:
            dict: prob, blocked, error, latency_ms, stage_latencies
        """
        stage_times: dict = {}
        t_total = time.perf_counter()

        # 1. Мониторинг
        t0 = time.perf_counter()
        is_valid, error_msg = self.monitoring.validate(raw_row)
        stage_times["monitoring_ms"] = (time.perf_counter() - t0) * 1000

        if not is_valid:
            return {
                "prob": None, "blocked": True, "error": error_msg,
                "latency_ms": (time.perf_counter() - t_total) * 1000,
                "stage_latencies": stage_times,
            }

        # 2. Шифрование
        t0 = time.perf_counter()
        enc_x = self.encryption.encrypt_vector(encoded_x)
        stage_times["encryption_ms"] = (time.perf_counter() - t0) * 1000

        # 3. Передача (HMAC)
        t0 = time.perf_counter()
        payload, mac        = self.transmission.send(enc_x)
        enc_x_received      = self.transmission.receive(payload, mac, enc_x)
        stage_times["transmission_ms"] = (time.perf_counter() - t0) * 1000

        # 4. Гомоморфный анализ (сервер)
        t0 = time.perf_counter()
        C_final = self.analysis.homomorphic_linear(
            enc_x_received, self.weights, self.bias
        )
        stage_times["analysis_ms"] = (time.perf_counter() - t0) * 1000

        # 5. Расшифровка + активация (клиент)
        t0 = time.perf_counter()
        z    = self.encryption.decrypt_value(C_final)
        prob = (self._sigmoid(z) if self.activation == "sigmoid"
                else self._sigmoid_approx(z))
        stage_times["decrypt_activate_ms"] = (time.perf_counter() - t0) * 1000

        return {
            "prob": prob, "blocked": False, "error": None,
            "latency_ms": (time.perf_counter() - t_total) * 1000,
            "stage_latencies": stage_times,
        }

    # ── Батч-инференс ─────────────────────────────────────────────────────────

    def run_batch(self, raw_rows: list, encoded_xs: np.ndarray,
                  max_workers: int = 1) -> list:
        """Запускает пайплайн на батче сэмплов с прогресс-баром.

        Args:
            raw_rows:    список сырых записей (dict) для MonitoringAgent
            encoded_xs:  нормализованная матрица признаков (N × D)
            max_workers: число потоков (1 = последовательно, >1 = параллельно)
                         Paillier-операции освобождают GIL → потоки ускоряют работу.

        Returns:
            list[dict] — результаты в исходном порядке
        """
        n = len(raw_rows)
        logger.info(
            "Батч-инференс: %d сэмплов | потоков: %d", n, max_workers
        )

        if max_workers <= 1:
            results = []
            for row, x in tqdm(zip(raw_rows, encoded_xs),
                                total=n, desc="PHE inference", unit="sample"):
                results.append(self.predict_one(row, x))
            return results

        # Параллельный режим
        results = [None] * n
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self.predict_one, row, x): i
                for i, (row, x) in enumerate(zip(raw_rows, encoded_xs))
            }
            with tqdm(total=n, desc="PHE inference (parallel)",
                      unit="sample") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    pbar.update(1)

        logger.info("Батч завершён")
        return results

    def print_agent_report(self):
        self.monitoring.report()
        self.transmission.report()
        self.analysis.report()
