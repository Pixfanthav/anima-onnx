#!/usr/bin/env python
# Validate llm_adapter.onnx numerically matches the PyTorch LLMAdapter.
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
DIT=CU+"/models/diffusion_models/kiwimixAnima_v1.safetensors"
adapter=comfy.sd.load_diffusion_model(DIT).model.diffusion_model.llm_adapter
adapter.eval()
for m in adapter.modules():
    if hasattr(m,"comfy_cast_weights"): m.comfy_cast_weights=False
adapter.float()

torch.manual_seed(42)
src=torch.randn(1,16,1024)
ids=torch.randint(0,32000,(1,20),dtype=torch.long)
with torch.no_grad():
    ref=adapter(src,ids).numpy()
print("pytorch out:", ref.shape, "mean", float(ref.mean()), "std", float(ref.std()))

import onnxruntime as ort
sess=ort.InferenceSession(ROOT+"/onnx/llm_adapter.onnx", providers=["CPUExecutionProvider"])
names=[i.name for i in sess.get_inputs()]
print("onnx inputs:", names)
out=sess.run(None, {names[0]: src.numpy(), names[1]: ids.numpy()})[0]
print("onnx out:", out.shape, "mean", float(out.mean()), "std", float(out.std()))

diff=np.abs(out-ref); rel=diff.max()/(np.abs(ref).max()+1e-8)
print(f"MAX abs diff = {diff.max():.3e} | mean abs diff = {diff.mean():.3e} | rel = {rel:.3e}")
print("VERDICT:", "MATCH (export correct)" if diff.max()<1e-2 else "MISMATCH - investigate")
