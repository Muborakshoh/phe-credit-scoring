"""Федеративное обучение с PHE secure aggregation и чекпоинтингом."""
import logging
import time
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def federated_training(X: np.ndarray, y: np.ndarray,
                       public_key, private_key,
                       n_clients: int = 5, n_rounds: int = 5,
                       lr: float = 0.05,
                       checkpoint_dir: str = "artifacts/checkpoints") -> tuple:
    """Федеративное обучение с PHE-агрегацией градиентов и чекпоинтингом.

    После каждого раунда сохраняет состояние модели — если процесс
    прерывается, следующий запуск продолжит с последнего чекпоинта.

    Схема:
        Каждый клиент: вычисляет grad_w, grad_b → шифрует → передаёт серверу
        Сервер: агрегирует E(grad) по всем клиентам → расшифровывает → обновляет w, b

    Returns:
        (w, b) — итоговые веса и смещение
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_path / "fed_checkpoint.pkl"

    # ── Восстановление из чекпоинта ──────────────────────────────────────────
    start_round = 0
    w = np.zeros(X.shape[1])
    b = 0.0

    if ckpt_file.exists():
        ckpt = joblib.load(ckpt_file)
        w, b, start_round = ckpt["w"], ckpt["b"], ckpt["round"] + 1
        logger.info(
            "Чекпоинт найден: продолжаем с раунда %d/%d", start_round + 1, n_rounds
        )
    else:
        logger.info("Чекпоинт не найден, обучение с нуля")

    if start_round >= n_rounds:
        logger.info("Обучение уже завершено по чекпоинту")
        return w, b

    # ── Разбивка данных по клиентам ──────────────────────────────────────────
    X_split = np.array_split(X, n_clients)
    y_split = np.array_split(y, n_clients)

    # ── Раунды обучения ───────────────────────────────────────────────────────
    for round_idx in range(start_round, n_rounds):
        t0 = time.perf_counter()
        enc_grad_w_agg = None
        enc_grad_b_agg = None

        for client_idx, (Xc, yc) in enumerate(zip(X_split, y_split)):
            pred   = 1.0 / (1.0 + np.exp(-(Xc @ w + b)))
            err    = pred - yc
            grad_w = Xc.T @ err / len(yc)
            grad_b = err.mean()

            enc_gw = [public_key.encrypt(float(v)) for v in grad_w]
            enc_gb = public_key.encrypt(float(grad_b))

            if enc_grad_w_agg is None:
                enc_grad_w_agg = enc_gw
                enc_grad_b_agg = enc_gb
            else:
                enc_grad_w_agg = [g1 + g2 for g1, g2 in zip(enc_grad_w_agg, enc_gw)]
                enc_grad_b_agg += enc_gb

            logger.debug(
                "Раунд %d/%d, клиент %d/%d — градиент зашифрован",
                round_idx + 1, n_rounds, client_idx + 1, n_clients,
            )

        agg_grad_w = np.array([private_key.decrypt(g) for g in enc_grad_w_agg])
        agg_grad_b = private_key.decrypt(enc_grad_b_agg)

        w -= lr * agg_grad_w
        b -= lr * agg_grad_b

        elapsed = time.perf_counter() - t0
        logger.info("Раунд %d/%d — %.1fs", round_idx + 1, n_rounds, elapsed)

        # Чекпоинт после каждого раунда
        joblib.dump({"w": w, "b": b, "round": round_idx}, ckpt_file)
        logger.debug("Чекпоинт сохранён → %s (раунд %d)", ckpt_file, round_idx + 1)

    # Удаляем чекпоинт после успешного завершения
    ckpt_file.unlink(missing_ok=True)
    logger.info("Федеративное обучение завершено, чекпоинт удалён")
    return w, b


def evaluate_federated(w: np.ndarray, b: float,
                       X_test: np.ndarray, y_test: np.ndarray,
                       n_phe: int) -> dict:
    """Метрики и латентность инференса федеративной модели."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    y_prob_full  = 1.0 / (1.0 + np.exp(-(X_test @ w + b)))
    y_pred_full  = (y_prob_full > 0.5).astype(int)
    y_prob_sub   = y_prob_full[:n_phe]
    y_pred_sub   = y_pred_full[:n_phe]
    y_sub        = y_test[:n_phe]

    latencies = []
    for x in X_test[:n_phe]:
        t0 = time.perf_counter()
        _ = 1.0 / (1.0 + np.exp(-(x @ w + b)))
        latencies.append((time.perf_counter() - t0) * 1000)

    logger.info(
        "Federated | Accuracy(full)=%.3f | AUC(full)=%.3f | Accuracy(sub n=%d)=%.3f",
        accuracy_score(y_test, y_pred_full),
        roc_auc_score(y_test, y_prob_full),
        n_phe,
        accuracy_score(y_sub, y_pred_sub),
    )

    return {
        "y_prob_full": y_prob_full,
        "y_pred_full": y_pred_full,
        "y_prob_sub":  y_prob_sub,
        "y_pred_sub":  y_pred_sub,
        "latencies_ms": latencies,
    }
