from pathlib import Path

import torch

from energy_dissagregation_mlops.model import Model

MODEL_PATH = Path("models/best.pt")
ONNX_PATH = Path("models/model.onnx")

device = "cpu"

model = Model(window_size=1024).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)

# Your checkpoint stores multiple fields
if "model_state" in ckpt:
    model.load_state_dict(ckpt["model_state"])
else:
    model.load_state_dict(ckpt)

model.eval()

dummy_input = torch.randn(1, 1, 1024)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["x"],
    output_names=["y"],
    dynamic_axes={
        "x": {0: "batch_size", 2: "time"},
        "y": {0: "batch_size", 2: "time"},
    },
    opset_version=17,
)

print(f"Exported ONNX model to {ONNX_PATH}")
