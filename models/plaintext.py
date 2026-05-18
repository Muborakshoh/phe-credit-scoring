"""Базовая plaintext-модель логистической регрессии."""
import logging
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from credit_scoring_phe.config import SEED, N_PHE

logger = logging.getLogger(__name__)


def train_plaintext_model(X_train: np.ndarray,
                          y_train: np.ndarray) -> LogisticRegression:
    """Обучает логистическую регрессию на открытых данных."""
    logger.info("Обучение plaintext-модели (LogisticRegression)...")
    model = LogisticRegression(max_iter=2000, random_state=SEED)
    model.fit(X_train, y_train)
    logger.info("Модель обучена. Классов: %s", model.classes_.tolist())
    return model


def evaluate_plaintext(model: LogisticRegression,
                       X_test: np.ndarray,
                       y_test: np.ndarray) -> dict:
    """Вычисляет метрики и латентность для plaintext-модели.

    Returns:
        dict: y_prob_full, y_pred_full, y_prob_sub, y_pred_sub,
              y_sub, latencies_ms, weights, bias
    """
    y_prob_full = model.predict_proba(X_test)[:, 1]
    y_pred_full = (y_prob_full > 0.5).astype(int)

    y_prob_sub = y_prob_full[:N_PHE]
    y_pred_sub = y_pred_full[:N_PHE]
    y_sub      = y_test[:N_PHE]

    latencies = []
    for x in X_test[:N_PHE]:
        t0 = time.perf_counter()
        model.predict_proba(x.reshape(1, -1))
        latencies.append((time.perf_counter() - t0) * 1000)

    acc_full = accuracy_score(y_test, y_pred_full)
    auc_full = roc_auc_score(y_test, y_prob_full)
    acc_sub  = accuracy_score(y_sub, y_pred_sub)
    lat_mean = np.mean(latencies)
    lat_std  = np.std(latencies)

    logger.info(
        "Plaintext | Accuracy(full)=%.3f | AUC(full)=%.3f | "
        "Accuracy(sub n=%d)=%.3f | Latency=%.3f±%.3f ms",
        acc_full, auc_full, N_PHE, acc_sub, lat_mean, lat_std,
    )

    return {
        "y_prob_full": y_prob_full,
        "y_pred_full": y_pred_full,
        "y_prob_sub":  y_prob_sub,
        "y_pred_sub":  y_pred_sub,
        "y_sub":       y_sub,
        "latencies_ms": latencies,
        "weights": model.coef_[0],
        "bias":    model.intercept_[0],
    }
