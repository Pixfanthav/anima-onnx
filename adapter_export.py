#!/usr/bin/env python
# Export the LLMAdapter (dit.llm_adapter): (qwen3_hidden, t5xxl_ids) -> adapted context [B,L,1024]
import sys, os, traceback
ROOT="D:/AnimaPort"; CU=ROOT+"/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv=["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu=True
import torch

def _rms_norm_decomposed(x, normalized_shape, weight=None, eps=None):
    if eps is None: eps = 1e-6
    dims = tuple(range(-len(normalized_shape), 0))
    var = x.pow(2).mean(dim=dims, keepdim=True)
    xn = x * torch.rsqrt(var + eps)
    if weight is not None: xn = xn * weight
    return xn
torch.nn.functional.rms_norm = _rms_norm_decomposed
print("patched rms_norm")

import comfy.sd, comfy.utils
DIT = CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors"
mp = comfy.sd.load_diffusion_model(DIT)
adapter = mp.model.diffusion_model.llm_adapter
adapter.eval()
for m in adapter.modules():
    if hasattr(m, "comfy_cast_weights"): m.comfy_cast_weights = False
adapter.float()
print("adapter loaded:", type(adapter).__name__)

src = torch.randn(1, 16, 1024)                       # qwen3 hidden states
ids = torch.randint(0, 32000, (1, 20), dtype=torch.long)  # t5xxl token ids

class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, src, ids): return s.m(src, ids)
w = Wrap(adapter).eval()

print(">>> trial forward ...")
try:
    with torch.no_grad():
        out = w(src, ids)
    print("forward OK, out:", tuple(out.shape))
except Exception:
    print("FORWARD FAILED:\n" + "\n".join(traceback.format_exc().splitlines()[-20:])); sys.exit(1)

print(">>> ONNX export ...")
try:
    torch.onnx.export(
        w, (src, ids), ROOT + "/onnx/llm_adapter.onnx",
        input_names=["source_hidden", "target_ids"], output_names=["adapted"],
        dynamic_axes={"source_hidden": {0: "b", 1: "src"}, "target_ids": {0: "b", 1: "tgt"},
                      "adapted": {0: "b", 1: "tgt"}},
        opset_version=17)
    print("ADAPTER ONNX OK ->", os.path.getsize(ROOT + "/onnx/llm_adapter.onnx")/1e6, "MB")
except Exception:
    print("EXPORT FAILED tail:\n" + "\n".join(l[:220] for l in traceback.format_exc().splitlines()[-16:]))
