"""Агент анализа: серверный гомоморфный инференс."""
import logging
import threading

import numpy as np
import mesa

logger = logging.getLogger(__name__)


class AnalysisAgent(mesa.Agent):
    """Вычисляет E(Σ w_i·x_i + b) гомоморфно, без расшифровки данных."""

    def __init__(self, model):
        super().__init__(model)
        self.inference_count = 0
        self._lock = threading.Lock()

    def homomorphic_linear(self, enc_x: list, weights: np.ndarray,
                           bias: float) -> object:
        """Гомоморфная линейная комбинация над зашифрованным вектором.

        Используемые свойства Paillier:
            E(x_i) * w_i = E(w_i · x_i)   — умножение на открытый скаляр
            E(a)  + E(b) = E(a + b)         — гомоморфное сложение

        Returns:
            C_final = E(Σ w_i·x_i + b)
        """
        logger.debug("Гомоморфное вычисление для вектора длиной %d", len(enc_x))

        C_res = None
        for enc_xi, wi in zip(enc_x, weights):
            term  = enc_xi * float(wi)
            C_res = term if C_res is None else C_res + term

        C_final = C_res + float(bias)

        with self._lock:
            self.inference_count += 1
        return C_final

    def step(self):
        pass

    def report(self) -> str:
        msg = f"AnalysisAgent: выполнено {self.inference_count} гомоморфных вычислений"
        logger.info(msg)
        return msg
