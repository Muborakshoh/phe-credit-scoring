"""Загрузка датасета."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(csv_path: str) -> pd.DataFrame:
    """Загружает датасет German Credit из CSV.

    Args:
        csv_path: путь к german_credit_data.csv

    Returns:
        DataFrame с исходными признаками
    """
    logger.info("Загрузка данных из %s", csv_path)
    df = pd.read_csv(csv_path, index_col=0)
    logger.info("Shape: %s | Columns: %s", df.shape, list(df.columns))
    return df
