#!/usr/bin/env python
# Validate qwen3_te.onnx numerically matches the PyTorch Qwen3 text encoder.
import sys, os, numpy as np
ROOT="D:/AnimaPort"; CU=ROOT+"/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv=["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu=True
import torch

def _rms(x, ns, weight=None, eps=None):
    if eps is None: eps=1e-6
    d=tuple(range(-len(ns),0)); v=x.pow(2).mean(dim=d,keepdim=True); xn=x*torch.rsqrt(v+eps)
    return xn*weight if weight is not None else xn
torch.nn.functional.rms_norm=_rms

import comfy.sd
TE=CU+"/models/text_encoders/qwen_3_06b_base.safetensors"
clip=comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
tr=clip.cond_stage_model.qwen3_06b.transformer
for m in tr.modules():
    if hasattr(m,"comfy_cast_weights"): m.comfy_cast_weights=False
tr.float(); tr.eval()

torch.manual_seed(42)
ids=torch.randint(0,1000,(1,16),dtype=torch.long)
with torch.no_grad():
    o=tr(ids); ref=(o[0] if isinstance(o,(tuple,list)) else o).numpy()
print("pytorch out:", ref.shape, "mean", float(ref.mean()), "std", float(ref.std()))

import onnxruntime as ort
sess=ort.InferenceSession(ROOT+"/onnx/te/qwen3_te.onnx", providers=["CPUExecutionProvider"])
names=[i.name for i in sess.get_inputs()]
print("onnx inputs:", names)
out=sess.run(None, {names[0]: ids.numpy()})[0]
print("onnx out:", out.shape, "mean", float(out.mean()), "std", float(out.std()))

diff=np.abs(out-ref); rel=diff.max()/(np.abs(ref).max()+1e-8)
print(f"MAX abs diff = {diff.max():.3e} | mean abs diff = {diff.mean():.3e} | rel = {rel:.3e}")
print("VERDICT:", "MATCH (export correct)" if diff.max()<1e-2 else "MISMATCH - investigate")
