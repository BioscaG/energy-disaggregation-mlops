import uuid
from pathlib import Path
from typing import List, Union

import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from energy_dissagregation_mlops.data_collection import (
    get_collector,
    get_drift_monitor,
)
from energy_dissagregation_mlops.model import Model

_model: Model | None = None
_onnx_session: ort.InferenceSession | None = None
_device = "cpu"
_collector = None
_reference_dist = None

# FastAPI
app = FastAPI(title="Energy Disaggregation API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for local dev / course
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    x: Union[List[float], List[List[float]]]


class PredictResponse(BaseModel):
    y: Union[List[float], List[List[float]]]
    t: int
    batch_size: int


# loads the model
@app.on_event("startup")
def startup():
    global _model, _onnx_session, _collector, _reference_dist

    model_path = Path("models/best.pt")
    if not model_path.exists():
        logger.warning("models/best.pt not found – PyTorch inference disabled")
        return

    base_model = Model(window_size=1024).to(_device)
    ckpt = torch.load(model_path, map_location=_device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        base_model.load_state_dict(ckpt["model_state"])
    else:
        base_model.load_state_dict(ckpt)

    base_model.eval()
    _model = base_model

    # Load ONNX model if available
    onnx_path = Path("models/model.onnx")
    if onnx_path.exists():
        _onnx_session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

    # Initialize data collector
    _collector = get_collector()
    logger.info("Data collector initialized")

    # Load reference distribution if available
    ref_dist_path = Path("models/reference_distribution.npy")
    if ref_dist_path.exists():
        _reference_dist = np.load(ref_dist_path)
        logger.info(f"Reference distribution loaded: {_reference_dist.shape}")
    else:
        logger.warning("Reference distribution not found at models/reference_distribution.npy")


# gets the health
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": _device,
        "model_loaded": _model is not None,
        "onnx_loaded": _onnx_session is not None,
    }


# the predict using model
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not req.x:
        raise HTTPException(status_code=400, detail="Input x is empty.")

    prediction_id = str(uuid.uuid4())[:8]
    is_batch = not isinstance(req.x[0], (int, float))

    if is_batch:
        x = torch.tensor(req.x, dtype=torch.float32).unsqueeze(1)
    else:
        x = torch.tensor(req.x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        y = _model(x)

    y = y.squeeze(1).cpu().numpy()

    # Collect data for monitoring
    if _collector is not None:
        try:
            x_np = x.squeeze(1).cpu().numpy()
            _collector.record_prediction(
                input_data=x_np,
                output_data=y,
                prediction_id=prediction_id,
                metadata={"endpoint": "/predict", "is_batch": is_batch},
            )
        except Exception as e:
            logger.warning(f"Failed to collect prediction data: {e}")

    return {
        "y": y.tolist() if is_batch else y[0].tolist(),
        "t": y.shape[-1],
        "batch_size": y.shape[0],
    }


# ONNX
@app.post("/predict/onnx", response_model=PredictResponse)
def predict_onnx(req: PredictRequest):
    if _onnx_session is None:
        raise HTTPException(status_code=503, detail="ONNX model not loaded.")

    if not req.x:
        raise HTTPException(status_code=400, detail="Input x is empty.")

    is_batch = not isinstance(req.x[0], (int, float))

    if is_batch:
        x = np.array(req.x, dtype="float32")[:, None, :]
    else:
        x = np.array(req.x, dtype="float32")[None, None, :]

    y = _onnx_session.run(None, {"x": x})[0]
    y = y.squeeze(1)

    # Collect data for monitoring
    if _collector is not None:
        try:
            _collector.record_prediction(
                input_data=x.squeeze(1),
                output_data=y,
                prediction_id=str(uuid.uuid4())[:8],
                metadata={"endpoint": "/predict/onnx", "is_batch": is_batch},
            )
        except Exception as e:
            logger.warning(f"Failed to collect prediction data: {e}")

    return {
        "y": y.tolist() if is_batch else y[0].tolist(),
        "t": y.shape[-1],
        "batch_size": y.shape[0],
    }


# === DATA COLLECTION & MONITORING ENDPOINTS ===


@app.get("/api/v1/collector/stats")
def get_collector_stats():
    """Get statistics about collected data."""
    if _collector is None:
        return {"status": "collector_not_initialized"}
    return _collector.get_statistics()


@app.get("/api/v1/collector/recent")
def get_recent_predictions(n: int = 50):
    """Get recent predictions."""
    if _collector is None:
        return {"status": "collector_not_initialized"}
    return _collector.get_recent_data(n)


@app.post("/api/v1/collector/save")
def save_collected_data(filename: str = None):
    """Save collected data to disk."""
    if _collector is None:
        return {"status": "collector_not_initialized"}
    filepath = _collector.save_batch(filename)
    return {"status": "saved", "filepath": str(filepath)}


@app.delete("/api/v1/collector/clear")
def clear_collector():
    """Clear collected data from memory."""
    if _collector is None:
        return {"status": "collector_not_initialized"}
    _collector.clear()
    return {"status": "cleared"}


@app.get("/api/v1/drift/status")
def get_drift_status():
    """Check for data drift in collected data."""
    if _collector is None:
        return {"status": "collector_not_initialized"}
    if _reference_dist is None:
        return {"status": "reference_distribution_not_loaded"}

    monitor = get_drift_monitor()
    result = monitor.analyze_drift(_reference_dist)
    return result


@app.get("/api/v1/drift/performance")
def get_performance_metrics():
    """Get performance metrics from collected data."""
    if _collector is None:
        return {"status": "collector_not_initialized"}

    monitor = get_drift_monitor()
    return monitor.get_performance_metrics()


@app.get("/api/v1/hourly/summary")
def get_hourly_summary():
    """Get hourly summary of predictions."""
    if _collector is None:
        return {"status": "collector_not_initialized"}
    return _collector.get_hourly_summary()
