FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app
RUN apt-get update && apt-get install -y build-essential
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/

RUN pip install -r requirements.txt && pip install -e .

VOLUME ["/app/data", "/app/models"]

ENV CUDA_VISIBLE_DEVICES=0
ENTRYPOINT ["python", "-m", "energy_dissagregation_mlops.train"]
