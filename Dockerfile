FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для graphviz и phe (OpenSSL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        graphviz \
        libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# Зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код как подпакет credit_scoring_phe/
# (build context = папка credit_scoring_phe/, поэтому кладём в подпапку)
COPY . ./credit_scoring_phe/

# Папка для артефактов и чекпоинтов
RUN mkdir -p artifacts/checkpoints mlruns

ENTRYPOINT ["python", "-m", "credit_scoring_phe.main"]
CMD ["--help"]
