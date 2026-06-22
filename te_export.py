#!/usr/bin/env python
# Qwen3-0.6B text encoder export attempt (uses same rms_norm patch as DiT).
import sys, os, traceback
ROOT="D:/AnimaPort"; CU=ROOT+"/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv=["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu=True
import torch

# rms_norm decomposed (covers comfy.rmsnorm -> F.rms_norm)
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
TE = CU + "/models/text_encoders/qwen_3_06b_base.safetensors"
clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
cm = clip.cond_stage_model
qwen = cm.qwen3_06b
tr = qwen.transformer
print("transformer class:", type(tr).__name__)
for m in tr.modules():
    if hasattr(m, "comfy_cast_weights"): m.comfy_cast_weights = False
tr.float(); tr.eval()

import inspect
print("forward sig:", str(inspect.signature(tr.forward)))

ids = torch.randint(0, 1000, (1, 16), dtype=torch.long)
print(">>> trial forward ...")
try:
    with torch.no_grad():
        out = tr(ids)
    if isinstance(out, (tuple, list)):
        print("forward OK, outputs:", [tuple(o.shape) if hasattr(o,'shape') else type(o) for o in out])
        main = out[0]
    else:
        print("forward OK, out:", tuple(out.shape)); main = out
except Exception:
    print("FORWARD FAILED:\n" + "\n".join(traceback.format_exc().splitlines()[-20:])); sys.exit(1)

class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, ids):
        o = s.m(ids)
        return o[0] if isinstance(o, (tuple, list)) else o

print(">>> ONNX export ...")
os.makedirs(ROOT + "/onnx/te", exist_ok=True)
try:
    torch.onnx.export(
        Wrap(tr).eval(), (ids,), ROOT + "/onnx/te/qwen3_te.onnx",
        input_names=["input_ids"], output_names=["hidden"],
        dynamic_axes={"input_ids": {0: "b", 1: "seq"}, "hidden": {0: "b", 1: "seq"}},
        opset_version=17)
    print("TE ONNX OK ->", os.path.getsize(ROOT + "/onnx/te/qwen3_te.onnx")/1e6, "MB")
except Exception:
    print("EXPORT FAILED tail:\n" + "\n".join(l[:220] for l in traceback.format_exc().splitlines()[-16:]))
