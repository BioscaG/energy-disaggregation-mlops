FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt requirements_dev.txt README.md ./
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

ENTRYPOINT ["uvicorn", "energy_dissagregation_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
