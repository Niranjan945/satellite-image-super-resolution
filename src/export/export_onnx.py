"""
Stage 5: Model Export
Converts the pretrained PyTorch Real-ESRGAN model into ONNX format.

Why: ONNX is a shared, framework-neutral file format for trained models -
like a PDF for neural networks. Once exported, the model can be loaded
and run in C++ (our next stage) without needing Python or PyTorch installed.

Reads:  models/pretrained/RealESRGAN_x4plus.pth
Writes: models/onnx/RealESRGAN_x4plus.onnx
"""

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

WEIGHTS_PATH = "models/pretrained/RealESRGAN_x4plus.pth"
ONNX_OUT_PATH = "models/onnx/RealESRGAN_x4plus.onnx"

# Must match the architecture used when the pretrained weights were trained
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# Load the pretrained weights. The Real-ESRGAN checkpoint stores them
# under the 'params_ema' key (the "exponential moving average" weights,
# which is the version actually meant for inference/deployment).
checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["params_ema"], strict=True)
model.eval()  # inference mode: disables dropout/batchnorm training behaviour

# ONNX export needs one example input to trace the model's operations.
# Shape: (batch_size=1, channels=3, height=64, width=64) - matches our
# pipeline's low-res patch size, but dynamic_axes below lets the exported
# model accept other sizes too, not just exactly 64x64.
dummy_input = torch.randn(1, 3, 64, 64)

print("Exporting model to ONNX... (this can take a minute)")
torch.onnx.export(
    model,
    dummy_input,
    ONNX_OUT_PATH,
    export_params=True,          # store the trained weights inside the .onnx file
    opset_version=17,            # ONNX operator-set version, 17 is well-supported
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {2: "height", 3: "width"},    # allow variable image sizes
        "output": {2: "height", 3: "width"},
    },
    dynamo=False,   # use PyTorch's stable TorchScript-based exporter
                     # (the newer "dynamo" exporter needs the extra
                     # 'onnxscript' package; this avoids that dependency)
)
print(f"Done. Saved to: {ONNX_OUT_PATH}")