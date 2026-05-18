"""Агент мониторинга: валидация и обнаружение дрейфа данных."""
import logging
import threading
from typing import Optional

import numpy as np
import mesa

logger = logging.getLogger(__name__)


class MonitoringAgent(mesa.Agent):
    """Валидирует сырые входные данные и отслеживает дрейф распределения (PSI).

    PSI (Population Stability Index) — стандартная метрика сдвига распределения.
    PSI < 0.1  → стабильно
    PSI < 0.25 → небольшой сдвиг
    PSI >= 0.25 → значительный дрейф, требует переобучения
    """

    NUMERIC_BOUNDS = {
        "Age":           (18, 100),
        "Job":           (0, 3),
        "Credit amount": (100, 200_000),
        "Duration":      (1, 120),
    }
    CATEGORICAL_VALUES = {
        "Sex":              {"male", "female"},
        "Housing":          {"own", "free", "rent"},
        "Saving accounts":  {"none", "little", "moderate", "rich", "quite rich"},
        "Checking account": {"none", "little", "moderate", "rich"},
        "Purpose": {
            "car", "furniture/equipment", "radio/TV", "domestic appliances",
            "repairs", "education", "business", "vacation/others",
        },
    }
    PSI_THRESHOLD = 0.25

    def __init__(self, model):
        super().__init__(model)
        self.blocked_count = 0
        self.passed_count  = 0
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()

        # Для PSI: храним референсное распределение (задаётся при fit)
        self._ref_distributions: dict = {}

    # ── Валидация ────────────────────────────────────────────────────────────

    def validate(self, row: dict) -> tuple[bool, Optional[str]]:
        """Проверяет одну запись на корректность.

        Returns:
            (True, None)        — данные корректны
            (False, error_msg)  — данные заблокированы
        """
        for col, (lo, hi) in self.NUMERIC_BOUNDS.items():
            if col not in row:
                continue
            val = row[col]
            if not isinstance(val, (int, float, np.integer, np.floating)):
                return False, self._block(
                    f"[MonitoringAgent] Некорректный тип: {col}={val!r}"
                )
            if not (lo <= val <= hi):
                return False, self._block(
                    f"[MonitoringAgent] Значение вне диапазона: {col}={val} "
                    f"(допустимо [{lo}, {hi}])"
                )

        for col, allowed in self.CATEGORICAL_VALUES.items():
            if col not in row:
                continue
            if str(row[col]) not in allowed:
                return False, self._block(
                    f"[MonitoringAgent] Недопустимое значение: {col}={row[col]!r}"
                )

        for col in self.NUMERIC_BOUNDS:
            if col in row and (
                row[col] is None
                or (isinstance(row[col], float) and np.isnan(row[col]))
            ):
                return False, self._block(
                    f"[MonitoringAgent] NaN в числовом признаке: {col}"
                )

        with self._lock:
            self.passed_count += 1
        logger.debug("Запись прошла валидацию")
        return True, None

    def _block(self, msg: str) -> str:
        with self._lock:
            self.blocked_count += 1
            self.last_error = msg
        logger.warning(msg)
        return msg

    # ── Обнаружение дрейфа (PSI) ─────────────────────────────────────────────

    def fit_reference(self, df_ref, numeric_cols: Optional[list] = None):
        """Сохраняет референсное распределение обучающей выборки для PSI.

        Args:
            df_ref:       DataFrame с референсными данными (обычно train)
            numeric_cols: список числовых колонок; если None — берёт из NUMERIC_BOUNDS
        """
        cols = numeric_cols or list(self.NUMERIC_BOUNDS.keys())
        self._ref_distributions = {}
        for col in cols:
            if col in df_ref.columns:
                self._ref_distributions[col] = df_ref[col].dropna().values
        logger.info(
            "Референсные распределения сохранены для %d признаков",
            len(self._ref_distributions),
        )

    def compute_psi(self, df_current, n_bins: int = 10) -> dict:
        """Вычисляет PSI для каждого числового признака.

        Args:
            df_current: DataFrame с текущими данными (test / production)
            n_bins:     количество бинов для гистограммы

        Returns:
            dict {col: psi_value}
        """
        if not self._ref_distributions:
            logger.warning(
                "Референсные данные не заданы. Вызовите fit_reference() сначала."
            )
            return {}

        results = {}
        for col, ref_vals in self._ref_distributions.items():
            if col not in df_current.columns:
                continue
            cur_vals = df_current[col].dropna().values
            psi = self._psi(ref_vals, cur_vals, n_bins)
            results[col] = psi

            level = (
                "STABLE"  if psi < 0.10 else
                "WARNING" if psi < 0.25 else
                "DRIFT"
            )
            log_fn = logger.info if psi < 0.10 else (
                logger.warning if psi < 0.25 else logger.error
            )
            log_fn("PSI %s=%s: %.4f [%s]", col, "", psi, level)

        return results

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int) -> float:
        """Population Stability Index между двумя массивами."""
        eps = 1e-6
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)

        e_counts, _ = np.histogram(expected, bins=bins)
        a_counts, _ = np.histogram(actual,   bins=bins)

        e_pct = (e_counts / len(expected)) + eps
        a_pct = (a_counts / len(actual))   + eps

        return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

    # ── Mesa / отчёт ──────────────────────────────────────────────────────────

    def step(self):
        pass

    def report(self) -> str:
        total = self.passed_count + self.blocked_count
        msg = (
            f"MonitoringAgent: обработано {total}, "
            f"пропущено {self.passed_count}, "
            f"заблокировано {self.blocked_count}"
        )
        logger.info(msg)
        return msg
