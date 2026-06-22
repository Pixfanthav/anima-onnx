#!/usr/bin/env python
# Validate dit.onnx (onnxruntime) numerically matches the PyTorch DiT.
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
import comfy.quant_ops as _qo
def _rope(q,k,rope):
    def f(t,fr):
        t_=t.reshape(*t.shape[:-1],2,-1).movedim(-2,-1).unsqueeze(-2)
        return (fr[...,0]*t_[...,0]+fr[...,1]*t_[...,1]).movedim(-1,-2).reshape(t.shape)
    return f(q,rope),f(k,rope)
_qo.ck.apply_rope_split_half=_rope

import comfy.sd
DIT=CU+"/models/diffusion_models/kiwimixAnima_v1.safetensors"
dit=comfy.sd.load_diffusion_model(DIT).model.diffusion_model; dit.eval()
for m in dit.modules():
    if hasattr(m,"comfy_cast_weights"): m.comfy_cast_weights=False
dit.float()

torch.manual_seed(42)
x=torch.randn(1,16,1,32,32); t=torch.ones(1,1); c=torch.randn(1,512,1024)
with torch.no_grad():
    ref=dit(x,t,c).numpy()
print("pytorch out:", ref.shape, "mean", float(ref.mean()), "std", float(ref.std()))

import onnxruntime as ort
so=ort.SessionOptions()
sess=ort.InferenceSession(ROOT+"/onnx/dit.onnx", so, providers=["CPUExecutionProvider"])
inp={i.name:None for i in sess.get_inputs()}
names=[i.name for i in sess.get_inputs()]
print("onnx inputs:", names)
feed={names[0]:x.numpy(), names[1]:t.numpy(), names[2]:c.numpy()}
out=sess.run(None, feed)[0]
print("onnx out:", out.shape, "mean", float(out.mean()), "std", float(out.std()))

diff=np.abs(out-ref)
rel=diff.max()/ (np.abs(ref).max()+1e-8)
print(f"MAX abs diff = {diff.max():.3e} | mean abs diff = {diff.mean():.3e} | rel = {rel:.3e}")
print("VERDICT:", "MATCH (export correct)" if diff.max()<1e-2 else "MISMATCH - investigate")
