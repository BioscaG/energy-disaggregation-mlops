from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from energy_dissagregation_mlops.model import Model


_model: Model | None = None
_onnx_session: ort.InferenceSession | None = None
_device = "cpu"

# FastAPI
app = FastAPI(title="Energy Disaggregation API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # OK for local dev / course
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
    global _model, _onnx_session

    model_path = Path("models/best.pt")
    if not model_path.exists():
        print("WARNING: models/best.pt not found – PyTorch inference disabled")
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


#gets the health
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

    is_batch = not isinstance(req.x[0], (int, float))

    if is_batch:
        x = torch.tensor(req.x, dtype=torch.float32).unsqueeze(1)
    else:
        x = torch.tensor(req.x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        y = _model(x)

    y = y.squeeze(1).cpu().numpy()

    return {
        "y": y.tolist() if is_batch else y[0].tolist(),
        "t": y.shape[-1],
        "batch_size": y.shape[0],
    }


#ONNX
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

    return {
        "y": y.tolist() if is_batch else y[0].tolist(),
        "t": y.shape[-1],
        "batch_size": y.shape[0],
    }
