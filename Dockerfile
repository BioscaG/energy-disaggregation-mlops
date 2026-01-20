FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt requirements_dev.txt ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY app/ ./app/

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

VOLUME ["/app/data", "/app/models"]

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
