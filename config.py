"""Конфигурация эксперимента.

Все параметры можно переопределить через переменные окружения:
    PHE_KEY_BITS=3072 N_PHE=100 python -m credit_scoring_phe.main --csv data.csv
"""
import os
import numpy as np

SEED     = int(os.getenv("SEED",        42))
N_PHE    = int(os.getenv("N_PHE",       50))
KEY_BITS = int(os.getenv("PHE_KEY_BITS", 2048))
ACTIVATION = os.getenv("ACTIVATION", "sigmoid")   # "sigmoid" | "approx"

# Федеративное обучение
N_CLIENTS = int(os.getenv("N_CLIENTS", 5))
N_ROUNDS  = int(os.getenv("N_ROUNDS",  5))
LR        = float(os.getenv("LR",      0.05))

# Пути к артефактам
MODEL_DIR = os.getenv("MODEL_DIR", "artifacts")

np.random.seed(SEED)
