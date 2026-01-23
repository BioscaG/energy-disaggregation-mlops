# CPU-friendly base; switch to a CUDA base only if you will request GPU on Vertex AI
FROM pytorch/pytorch:2.0.1-cpu

WORKDIR /app

# System deps: git is useful for editable installs; curl for debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy minimal files first to leverage Docker layer caching
COPY pyproject.toml requirements.txt ./

# Install python deps + DVC GCS plugin
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir dvc-gs

# Copy source
COPY src/ ./src/

# Install your package
RUN pip install --no-cache-dir -e .

# Vertex typically mounts /gcs and /tmp; keep outputs in /app/models unless you change train.py to write to GCS
ENV PYTHONUNBUFFERED=1

# Pull data via DVC, then run training
# (Assumes your DVC remote is already set to gcsremote in .dvc/config which is in the repo)
ENTRYPOINT ["bash", "-lc", "dvc pull -v && python -m energy_dissagregation_mlops.train"]
