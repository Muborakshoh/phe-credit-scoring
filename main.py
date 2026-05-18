"""Точка входа: полный эксперимент кредитного скоринга с PHE + Mesa.

Использование:
    python -m credit_scoring_phe.main --csv german_credit_data.csv

Env-переменные:
    PHE_KEY_BITS=3072   — размер ключа Paillier (default: 2048)
    N_PHE=100           — число сэмплов для PHE-сравнения (default: 50)
    N_WORKERS=4         — число потоков для параллельного инференса (default: 1)
    MLFLOW_TRACKING_URI — URI MLflow-сервера (default: ./mlruns локально)
"""
import argparse
import logging
import logging.config
import os
import time
from pathlib import Path

from credit_scoring_phe.config import (
    N_PHE, KEY_BITS, ACTIVATION, N_CLIENTS, N_ROUNDS, LR, MODEL_DIR
)
from credit_scoring_phe.data.loader import load_data
from credit_scoring_phe.data.preprocessing import add_target, encode_and_split
from credit_scoring_phe.models.plaintext import train_plaintext_model, evaluate_plaintext
from credit_scoring_phe.models.pipeline import CreditScoringMesaModel
from credit_scoring_phe.models.federated import federated_training, evaluate_federated
from credit_scoring_phe.evaluation.metrics import (
    extract_mesa_results, build_comparison_table,
)
from credit_scoring_phe.evaluation.visualization import (
    plot_roc_curves, plot_stage_latencies, plot_latency_comparison,
)
from credit_scoring_phe.evaluation.reporter import (
    trace_one_sample, print_agent_report, print_predictions_report,
    print_defense_summary,
)

# ── Логирование ───────────────────────────────────────────────────────────────
logging.config.dictConfig({
    "version": 1,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "experiment.log",
            "formatter": "detailed",
            "level": "DEBUG",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    # Убираем шум от сторонних библиотек
    "loggers": {
        "matplotlib": {"level": "WARNING"},
        "PIL":        {"level": "WARNING"},
    },
})

logger = logging.getLogger(__name__)

# MLflow — опционально
try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

N_WORKERS = int(os.getenv("N_WORKERS", 1))


def main(csv_path: str):
    logger.info("=" * 60)
    logger.info("Эксперимент: PHE Credit Scoring + Mesa Agents")
    logger.info("KEY_BITS=%d | N_PHE=%d | N_WORKERS=%d", KEY_BITS, N_PHE, N_WORKERS)
    logger.info("=" * 60)

    # ── MLflow run ────────────────────────────────────────────────────────────
    ctx = mlflow.start_run(run_name="phe_credit_scoring") if _MLFLOW else _NullCtx()

    with ctx as run:
        if _MLFLOW and run:
            mlflow.log_params({
                "key_bits":   KEY_BITS,
                "n_phe":      N_PHE,
                "n_clients":  N_CLIENTS,
                "n_rounds":   N_ROUNDS,
                "lr":         LR,
                "activation": ACTIVATION,
                "n_workers":  N_WORKERS,
            })
            logger.info("MLflow run_id: %s", run.info.run_id)

        # 1. Данные
        df = load_data(csv_path)
        df = add_target(df)
        X_train, X_test, y_train, y_test, df_test_raw, scaler = encode_and_split(df)

        # 2. Plaintext
        logger.info("--- Plaintext модель ---")
        plain_model = train_plaintext_model(X_train, y_train)
        plain       = evaluate_plaintext(plain_model, X_test, y_test)
        weights, bias = plain["weights"], plain["bias"]

        # 3. Mesa PHE пайплайн
        logger.info("--- Инициализация Mesa агентной системы ---")
        mesa_model = CreditScoringMesaModel(
            weights=weights, bias=bias,
            key_bits=KEY_BITS, activation=ACTIVATION,
        )

        # Сохранение модели
        mesa_model.encryption.save_model(weights, bias, scaler, MODEL_DIR)

        # Настройка PSI (drift detection)
        import pandas as pd
        df_train_raw = df.drop(columns=["target"]).iloc[:len(y_train)].copy()
        df_train_raw["Saving accounts"]  = df_train_raw["Saving accounts"].fillna("none")
        df_train_raw["Checking account"] = df_train_raw["Checking account"].fillna("none")
        mesa_model.monitoring.fit_reference(df_train_raw)

        # PHE инференс
        logger.info("--- PHE-инференс на %d сэмплах (потоков: %d) ---",
                    N_PHE, N_WORKERS)
        raw_rows   = [df_test_raw.iloc[i].to_dict() for i in range(N_PHE)]
        encoded_xs = X_test[:N_PHE]

        # Детальный трейс одного сэмпла до батча — показывает как каждый агент работает
        logger.info("--- Трейс агентов (сэмпл #0) ---")
        trace_one_sample(mesa_model, raw_rows[0], encoded_xs[0], sample_idx=0)

        t0 = time.perf_counter()
        results = mesa_model.run_batch(raw_rows, encoded_xs, max_workers=N_WORKERS)
        batch_time = time.perf_counter() - t0
        logger.info("Время батча: %.1fs", batch_time)

        # Подробный отчёт по агентам после батча
        print_agent_report(mesa_model, results, batch_time)

        # PSI-проверка дрейфа на тестовых данных
        logger.info("--- PSI (drift detection) ---")
        mesa_model.monitoring.compute_psi(df_test_raw)

        # Таблица предсказаний: prob / pred / real / ✓✗
        print_predictions_report(results, plain["y_sub"], n_show=15)

        mesa_res = extract_mesa_results(results, plain["y_sub"])

        # 4. Федеративное обучение
        logger.info("--- Федеративное обучение ---")
        t0 = time.perf_counter()
        fed_w, fed_b = federated_training(
            X_train, y_train,
            public_key=mesa_model.encryption.public_key,
            private_key=mesa_model.encryption.private_key,
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, lr=LR,
        )
        fed_training_time = time.perf_counter() - t0

        fed_res = evaluate_federated(fed_w, fed_b, X_test, y_test, n_phe=N_PHE)

        # 5. Сводная таблица
        logger.info("--- Сводная таблица ---")
        build_comparison_table(
            plain, mesa_res, fed_res, plain["y_sub"],
            fed_training_time, N_ROUNDS,
        )

        # 6. Визуализация
        plot_roc_curves(plain, mesa_res, fed_res, plain["y_sub"])
        plot_stage_latencies(mesa_res["stage_means"])
        plot_latency_comparison(plain["latencies_ms"], mesa_res["latencies_ms"])

        # 7. Итоговая таблица для защиты дипломной работы
        print_defense_summary(
            plain=plain,
            mesa_res=mesa_res,
            fed_res=fed_res,
            y_sub=plain["y_sub"],
            mesa_model=mesa_model,
            results=results,
            batch_time_s=batch_time,
            fed_training_time_s=fed_training_time,
            key_bits=KEY_BITS,
            n_phe=N_PHE,
            n_clients=N_CLIENTS,
            n_rounds=N_ROUNDS,
        )

    logger.info("Эксперимент завершён. Лог: experiment.log")


class _NullCtx:
    """Заглушка для MLflow context manager когда mlflow не установлен."""
    def __enter__(self): return None
    def __exit__(self, *_): pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHE Credit Scoring + Mesa Agents")
    parser.add_argument("--csv", required=True, help="Путь к german_credit_data.csv")
    args = parser.parse_args()
    main(args.csv)
