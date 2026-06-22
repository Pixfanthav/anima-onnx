#!/usr/bin/env python
# Does dit.onnx generalize to resolutions other than the 32x32 trace size?
# Compare ONNX vs PyTorch DiT at a NON-trace latent size.
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
        B,S,H,D=t.shape
        t_=t.reshape(B,S,H,2,D//2).transpose(3,4).unsqueeze(4)
        out=fr[...,0]*t_[...,0]+fr[...,1]*t_[...,1]
        return out.transpose(3,4).reshape(B,S,H,D)
    return f(q,rope),f(k,rope)
_qo.ck.apply_rope_split_half=_rope

import comfy.sd
DIT=CU+"/models/diffusion_models/kiwimixAnima_v1.safetensors"
dit=comfy.sd.load_diffusion_model(DIT).model.diffusion_model; dit.eval()
for m in dit.modules():
    if hasattr(m,"comfy_cast_weights"): m.comfy_cast_weights=False
dit.float()

import onnxruntime as ort
sess=ort.InferenceSession(ROOT+"/onnx/dit.onnx", providers=["CPUExecutionProvider"])
names=[i.name for i in sess.get_inputs()]

def trial(H,W,tok):
    torch.manual_seed(0)
    x=torch.randn(1,16,1,H,W); t=torch.ones(1,1); c=torch.randn(1,tok,1024)
    with torch.no_grad(): ref=dit(x,t,c).numpy()
    try:
        out=sess.run(None,{names[0]:x.numpy(),names[1]:t.numpy(),names[2]:c.numpy()})[0]
    except Exception as e:
        print(f"  [{H}x{W} tok{tok}] ONNX RUN FAILED: {str(e)[:160]}"); return
    if out.shape!=ref.shape:
        print(f"  [{H}x{W} tok{tok}] SHAPE MISMATCH onnx{out.shape} vs torch{ref.shape}"); return
    d=np.abs(out-ref); rel=d.max()/(np.abs(ref).max()+1e-8)
    print(f"  [{H}x{W} tok{tok}] MAX {d.max():.3e} mean {d.mean():.3e} rel {rel:.3e} -> {'MATCH' if d.max()<1e-2 else 'MISMATCH'}")

print("trace size 32x32 (sanity):"); trial(32,32,512)
print("non-trace 48x48:");          trial(48,48,512)
print("non-trace 40x56:");          trial(40,56,300)
print("full-res latent 144x112:");  trial(144,112,512)
