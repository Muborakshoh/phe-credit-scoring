"""Предобработка: синтетическая метка, кодирование, нормализация."""
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from credit_scoring_phe.config import SEED

logger = logging.getLogger(__name__)


def compute_risk_score(row: pd.Series) -> int:
    """Взвешенный риск-скор для формирования синтетической целевой переменной."""
    saving_map  = {"little": -1, "moderate": 1, "rich": 2, "quite rich": 2}
    checking_map= {"little": -1, "moderate": 1, "rich": 2}
    housing_map = {"own": 1, "free": 0, "rent": -1}

    score  = saving_map.get(row["Saving accounts"], -1)
    score += checking_map.get(row["Checking account"], -1)
    score += housing_map.get(row["Housing"], 0)
    score += -1 if row["Duration"] > 24 else 1
    score += -1 if row["Credit amount"] > 5000 else 1
    return score


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет бинарную метку target.

    Если в датасете есть колонка 'Risk' (оригинальная метка Kaggle/UCI):
        'good' → 1,  'bad' → 0
    Иначе использует синтетический риск-скор по 5 признакам.
    """
    df = df.copy()

    if "Risk" in df.columns:
        # Оригинальная метка из датасета с Risk-колонкой
        df["target"] = (df["Risk"].str.strip().str.lower() == "good").astype(int)
        df = df.drop(columns=["Risk"])
        logger.info(
            "Используется оригинальная метка Risk | "
            "Good: %d | Bad: %d | Ratio: %.1f%%",
            df["target"].sum(),
            (df["target"] == 0).sum(),
            df["target"].mean() * 100,
        )
    else:
        # Синтетический риск-скор (если Risk-колонки нет)
        df["risk_score"] = df.apply(compute_risk_score, axis=1)
        threshold        = df["risk_score"].quantile(0.30)
        df["target"]     = (df["risk_score"] >= threshold).astype(int)
        df               = df.drop(columns=["risk_score"])
        logger.info(
            "Используется синтетическая метка | Good credit ratio: %.1f%%",
            df["target"].mean() * 100,
        )

    return df


def encode_and_split(df: pd.DataFrame, test_size: float = 0.2):
    """One-hot кодирование + разбивка + нормализация без data leakage.

    Scaler обучается только на тренировочной части.

    Returns:
        X_train, X_test, y_train, y_test, df_test_raw, scaler
    """
    y     = df["target"].values
    X_raw = df.drop(columns=["target"]).copy()
    df_raw = X_raw.copy()

    X_raw["Saving accounts"]  = X_raw["Saving accounts"].fillna("none")
    X_raw["Checking account"] = X_raw["Checking account"].fillna("none")

    X_encoded = pd.get_dummies(
        X_raw,
        columns=["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"],
    )

    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_encoded.values, y, test_size=test_size, random_state=SEED
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_tr_raw)   # fit только на train
    X_test  = scaler.transform(X_te_raw)        # transform без утечки

    df_raw_filled = df_raw.copy()
    df_raw_filled["Saving accounts"]  = df_raw_filled["Saving accounts"].fillna("none")
    df_raw_filled["Checking account"] = df_raw_filled["Checking account"].fillna("none")

    _, df_test_raw, _, _ = train_test_split(
        df_raw_filled, y, test_size=test_size, random_state=SEED
    )
    df_test_raw = df_test_raw.reset_index(drop=True)

    logger.info(
        "Признаков: %d | Train: %d | Test: %d",
        X_train.shape[1], X_train.shape[0], X_test.shape[0],
    )
    return X_train, X_test, y_train, y_test, df_test_raw, scaler
