# 🔐 PHE Credit Scoring with Mesa Agents

> 🇷🇺 [Читать на русском](#-phe-кредитный-скоринг-с-mesa-агентами) &nbsp;·&nbsp; 🇬🇧 English below

---

A privacy-preserving credit scoring system using **Paillier Partial Homomorphic Encryption (PHE)** and a **multi-agent architecture** built with Mesa 3.x. The bank computes a credit score directly on encrypted client data — raw personal data is never exposed during inference.

## How It Works

```
Client data  →  MonitoringAgent  →  EncryptionAgent   →  TransmissionAgent  →  AnalysisAgent  →  Decision
                 (validate, PSI)    (Paillier 2048-bit)    (HMAC-SHA256)         (w·x_enc → sigmoid)
```

1. **MonitoringAgent** — validates incoming features and runs a PSI drift check against the training distribution.
2. **EncryptionAgent** — one-hot encodes categoricals, normalizes numerics (StandardScaler), encrypts the feature vector with a 2048-bit Paillier public key.
3. **TransmissionAgent** — signs the encrypted payload with HMAC-SHA256 and forwards it.
4. **AnalysisAgent** — computes `w · x_enc` homomorphically, decrypts only the final scalar, applies sigmoid, returns a credit approval probability.

The model weights are public; the client's feature values are **never seen in plaintext** by the server.

## Features

| Feature | Details |
|---|---|
| PHE scheme | Paillier (additive homomorphic), 2048-bit key |
| Agent framework | Mesa 3.x |
| Federated learning | 5 clients × 5 rounds, PHE gradient aggregation |
| Drift detection | Population Stability Index (PSI) |
| Integrity check | HMAC-SHA256 per request |
| Experiment tracking | MLflow |
| Parallel inference | ThreadPoolExecutor |
| Tests | pytest |
| Container | Docker (python:3.11-slim) |

## Project Structure

```
credit_scoring_phe/
├── agents/
│   ├── monitoring_agent.py      # PSI drift detection + input validation
│   ├── encryption_agent.py      # Paillier key gen + encryption
│   ├── transmission_agent.py    # HMAC-SHA256 integrity check
│   └── analysis_agent.py        # Homomorphic dot product + decrypt + sigmoid
├── models/
│   ├── plaintext.py             # Logistic regression baseline
│   ├── pipeline.py              # CreditScoringMesaModel (batch + parallel)
│   └── federated.py             # Federated learning with PHE aggregation
├── data/
│   ├── loader.py                # CSV loading
│   └── preprocessing.py         # OHE + StandardScaler (no data leakage)
├── evaluation/
│   ├── metrics.py               # Accuracy, AUC, MLflow logging
│   ├── reporter.py              # Terminal reports
│   └── visualization.py         # ROC curves, latency charts
├── tests/
│   ├── test_monitoring.py
│   ├── test_preprocessing.py
│   └── test_transmission.py
├── config.py                    # All parameters (env-var overridable)
├── main.py                      # Entry point
├── Dockerfile
└── requirements.txt
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/phe-credit-scoring.git
cd phe-credit-scoring
pip install -r requirements.txt

# 2. Download dataset → save as german_credit_data.csv
# https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk

# 3. Run
python -m credit_scoring_phe.main --csv german_credit_data.csv
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PHE_KEY_BITS` | `2048` | Paillier key size (bits) |
| `N_PHE` | `50` | Samples for PHE evaluation |
| `N_CLIENTS` | `5` | Federated learning clients |
| `N_ROUNDS` | `5` | Federated learning rounds |
| `LR` | `0.05` | Learning rate |
| `N_WORKERS` | `1` | Parallel inference threads |
| `SEED` | `42` | Random seed |
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow server URI |

```bash
PHE_KEY_BITS=3072 N_WORKERS=4 python -m credit_scoring_phe.main --csv german_credit_data.csv
```

## Docker

```bash
docker build -t phe-credit-scoring .
docker run --rm \
  -v $(pwd)/german_credit_data.csv:/data/german_credit_data.csv \
  phe-credit-scoring --csv /data/german_credit_data.csv
```

## Results

> Apple M3 Pro · macOS Tahoe · Python 3.11 · 1000 records · 26 features after OHE · 80/20 split

| Method | Accuracy | ROC-AUC | Avg latency |
|---|---|---|---|
| Plaintext (LR) | ~0.72 | ~0.715 | ~0.038 ms |
| **PHE via Mesa Agents** | ~0.72 | ~0.715 | ~4060 ms |
| Federated (PHE aggr.) | ~0.66 | ~0.721 | ~0.002 ms |

PHE adds **zero accuracy loss**. Latency breakdown:

| Stage | Agent | Avg latency |
|---|---|---|
| Validation + PSI | MonitoringAgent | ~2 ms |
| Feature encryption | EncryptionAgent | ~3500–3800 ms |
| HMAC sign + forward | TransmissionAgent | ~5 ms |
| Homomorphic dot product | AnalysisAgent | ~250–300 ms |
| Decrypt + sigmoid | AnalysisAgent | ~20–30 ms |

## Tests & MLflow

```bash
pytest tests/ -v

mlflow ui   # → http://localhost:5000
```

## License

MIT

---
---

# 🔐 PHE Кредитный скоринг с Mesa Агентами

> 🇬🇧 [Read in English](#-phe-credit-scoring-with-mesa-agents) &nbsp;·&nbsp; 🇷🇺 Русский ниже

---

Система кредитного скоринга с защитой конфиденциальности на основе **частичного гомоморфного шифрования Paillier (PHE)** и **мультиагентной архитектуры** Mesa 3.x. Банк вычисляет кредитный балл непосредственно на зашифрованных данных клиента — персональные данные в открытом виде никогда не раскрываются в процессе инференса.

## Как это работает

```
Данные клиента  →  MonitoringAgent  →  EncryptionAgent    →  TransmissionAgent  →  AnalysisAgent  →  Решение
                    (валидация, PSI)    (Paillier 2048 бит)    (HMAC-SHA256)         (w·x_enc → сигмоида)
```

1. **MonitoringAgent** — проверяет входные признаки и вычисляет PSI для обнаружения дрейфа данных.
2. **EncryptionAgent** — кодирует категориальные признаки (OHE), нормализует числовые (StandardScaler) и шифрует вектор признаков 2048-битным ключом Paillier.
3. **TransmissionAgent** — подписывает зашифрованный пакет через HMAC-SHA256 и передаёт его на анализ.
4. **AnalysisAgent** — гомоморфно вычисляет `w · x_enc`, расшифровывает только итоговый скаляр, применяет сигмоиду и возвращает вероятность одобрения кредита.

Веса модели публичны; значения признаков клиента **никогда не видны в открытом виде** серверу скоринга.

## Возможности

| Компонент | Детали |
|---|---|
| Схема PHE | Paillier (аддитивный гомоморфизм), ключ 2048 бит |
| Агентный фреймворк | Mesa 3.x |
| Федеративное обучение | 5 клиентов × 5 раундов, PHE-агрегация градиентов |
| Обнаружение дрейфа | Population Stability Index (PSI) |
| Целостность данных | HMAC-SHA256 на каждый запрос |
| Трекинг экспериментов | MLflow |
| Параллельный инференс | ThreadPoolExecutor |
| Тестирование | pytest |
| Контейнеризация | Docker (python:3.11-slim) |

## Структура проекта

```
credit_scoring_phe/
├── agents/
│   ├── monitoring_agent.py      # PSI-детектор дрейфа + валидация
│   ├── encryption_agent.py      # Генерация ключей Paillier + шифрование
│   ├── transmission_agent.py    # Верификация целостности HMAC-SHA256
│   └── analysis_agent.py        # Гомоморфное скал. произв. + дешифровка + сигмоида
├── models/
│   ├── plaintext.py             # Логистическая регрессия (базовая линия)
│   ├── pipeline.py              # CreditScoringMesaModel (батч + параллельный режим)
│   └── federated.py             # Федеративное обучение с PHE-агрегацией
├── data/
│   ├── loader.py                # Загрузка CSV
│   └── preprocessing.py         # OHE + StandardScaler (без data leakage)
├── evaluation/
│   ├── metrics.py               # Accuracy, AUC, MLflow-логирование
│   ├── reporter.py              # Терминальные отчёты
│   └── visualization.py         # ROC-кривые, графики задержек
├── tests/
│   ├── test_monitoring.py
│   ├── test_preprocessing.py
│   └── test_transmission.py
├── config.py                    # Все параметры (через env-переменные)
├── main.py                      # Точка входа
├── Dockerfile
└── requirements.txt
```

## Быстрый старт

```bash
# 1. Клонировать
git clone https://github.com/YOUR_USERNAME/phe-credit-scoring.git
cd phe-credit-scoring
pip install -r requirements.txt

# 2. Скачать датасет → сохранить как german_credit_data.csv
# https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk

# 3. Запустить
python -m credit_scoring_phe.main --csv german_credit_data.csv
```

## Конфигурация

| Переменная | По умолчанию | Описание |
|---|---|---|
| `PHE_KEY_BITS` | `2048` | Размер ключа Paillier (бит) |
| `N_PHE` | `50` | Число сэмплов для PHE-оценки |
| `N_CLIENTS` | `5` | Число клиентов в федеративном обучении |
| `N_ROUNDS` | `5` | Число раундов федеративного обучения |
| `LR` | `0.05` | Скорость обучения |
| `N_WORKERS` | `1` | Число потоков параллельного инференса |
| `SEED` | `42` | Случайное начальное число |
| `MLFLOW_TRACKING_URI` | `./mlruns` | URI MLflow-сервера |

```bash
PHE_KEY_BITS=3072 N_WORKERS=4 python -m credit_scoring_phe.main --csv german_credit_data.csv
```

## Docker

```bash
docker build -t phe-credit-scoring .
docker run --rm \
  -v $(pwd)/german_credit_data.csv:/data/german_credit_data.csv \
  phe-credit-scoring --csv /data/german_credit_data.csv
```

## Результаты экспериментов

> Apple M3 Pro · macOS Tahoe · Python 3.11 · 1000 записей · 26 признаков после OHE · разбивка 80/20

| Метод | Accuracy | ROC-AUC | Средняя задержка |
|---|---|---|---|
| Plaintext (LR) | ~0.72 | ~0.715 | ~0.038 мс |
| **PHE via Mesa Agents** | ~0.72 | ~0.715 | ~4060 мс |
| Federated (PHE aggr.) | ~0.66 | ~0.721 | ~0.002 мс |

PHE **не снижает точность модели**. Поэтапная задержка:

| Этап | Агент | Средняя задержка |
|---|---|---|
| Валидация + PSI | MonitoringAgent | ~2 мс |
| Шифрование признаков | EncryptionAgent | ~3500–3800 мс |
| HMAC-подпись + передача | TransmissionAgent | ~5 мс |
| Гомоморфное скалярное произведение | AnalysisAgent | ~250–300 мс |
| Дешифровка + сигмоида | AnalysisAgent | ~20–30 мс |

Основная задержка — шифрование (операции возведения в степень в Z*_{n²} при n=2048 бит на чистом Python). В production снижается до ~200–500 мс с помощью C-расширений (gmpy2) или аппаратных HSM.

## Тесты и MLflow

```bash
pytest tests/ -v

mlflow ui   # → http://localhost:5000
```

## Лицензия

MIT
