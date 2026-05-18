"""Метрики, сводная таблица и MLflow-трекинг."""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from credit_scoring_phe.config import N_PHE

logger = logging.getLogger(__name__)

STAGE_KEYS = [
    "monitoring_ms", "encryption_ms", "transmission_ms",
    "analysis_ms",   "decrypt_activate_ms",
]

# MLflow — опциональная зависимость
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning("mlflow не установлен — трекинг отключён. pip install mlflow")


def extract_mesa_results(results: list, y_sub: np.ndarray) -> dict:
    """Извлекает метрики и латентности из результатов Mesa-пайплайна."""
    passed_indices = [i for i, r in enumerate(results) if not r["blocked"]]

    y_prob = np.array([results[i]["prob"] for i in passed_indices])
    y_pred = (y_prob > 0.5).astype(int)
    y_true = y_sub[passed_indices]
    latencies = [results[i]["latency_ms"] for i in passed_indices]

    stage_means = {
        k: np.mean([results[i]["stage_latencies"].get(k, 0) for i in passed_indices])
        for k in STAGE_KEYS
    }

    n = len(passed_indices)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    lat_mean = np.mean(latencies)
    lat_std  = np.std(latencies)

    logger.info(
        "Mesa PHE | n=%d | Accuracy=%.3f | AUC=%.3f | Latency=%.1f±%.1f ms",
        n, acc, auc, lat_mean, lat_std,
    )
    logger.info("Поэтапные задержки: %s",
                {k: f"{v:.2f}ms" for k, v in stage_means.items()})

    return {
        "y_prob": y_prob, "y_pred": y_pred, "y_true": y_true,
        "latencies_ms": latencies, "stage_means": stage_means,
        "passed_indices": passed_indices,
    }


def build_comparison_table(plain: dict, mesa: dict, fed: dict,
                            y_sub: np.ndarray,
                            fed_training_time_s: float,
                            n_rounds: int,
                            mlflow_run=None) -> pd.DataFrame:
    """Строит сводную таблицу и логирует метрики в MLflow."""
    n = N_PHE
    table = pd.DataFrame({
        "Method": ["Plaintext", "PHE via Mesa Agents", "Federated (PHE agg.)"],
        f"Accuracy (n={n})": [
            round(accuracy_score(y_sub, plain["y_pred_sub"]), 3),
            round(accuracy_score(mesa["y_true"], mesa["y_pred"]), 3),
            round(accuracy_score(y_sub, fed["y_pred_sub"]), 3),
        ],
        f"ROC-AUC (n={n})": [
            round(roc_auc_score(y_sub, plain["y_prob_sub"]), 3),
            round(roc_auc_score(mesa["y_true"], mesa["y_prob"]), 3),
            round(roc_auc_score(y_sub, fed["y_prob_sub"]), 3),
        ],
        "Avg Inference Latency (ms)": [
            f"{np.mean(plain['latencies_ms']):.2f}",
            f"{np.mean(mesa['latencies_ms']):.1f}",
            f"{np.mean(fed['latencies_ms']):.3f}",
        ],
    })

    logger.info(
        "Federated training: %.1fs total (%d rounds)", fed_training_time_s, n_rounds
    )
    logger.info("\n%s", table.to_string(index=False))

    # ── MLflow ────────────────────────────────────────────────────────────────
    if _MLFLOW_AVAILABLE:
        _log_to_mlflow(plain, mesa, fed, y_sub, fed_training_time_s, n)

    return table


def _log_to_mlflow(plain, mesa, fed, y_sub, fed_training_time_s, n):
    """Логирует метрики в активный MLflow run (если он есть)."""
    try:
        mlflow.log_metrics({
            "plain_accuracy":      accuracy_score(y_sub, plain["y_pred_sub"]),
            "plain_roc_auc":       roc_auc_score(y_sub, plain["y_prob_sub"]),
            "plain_latency_ms":    float(np.mean(plain["latencies_ms"])),

            "phe_accuracy":        accuracy_score(mesa["y_true"], mesa["y_pred"]),
            "phe_roc_auc":         roc_auc_score(mesa["y_true"], mesa["y_prob"]),
            "phe_latency_ms":      float(np.mean(mesa["latencies_ms"])),
            "phe_encrypt_ms":      mesa["stage_means"].get("encryption_ms", 0),
            "phe_analysis_ms":     mesa["stage_means"].get("analysis_ms", 0),
            "phe_blocked_samples": float(n - len(mesa["passed_indices"])),

            "fed_accuracy":        accuracy_score(y_sub, fed["y_pred_sub"]),
            "fed_roc_auc":         roc_auc_score(y_sub, fed["y_prob_sub"]),
            "fed_latency_ms":      float(np.mean(fed["latencies_ms"])),
            "fed_training_time_s": fed_training_time_s,
        })
        logger.info("Метрики залогированы в MLflow")
    except Exception as exc:
        logger.warning("Ошибка MLflow logging: %s", exc)
