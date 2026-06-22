#!/usr/bin/env python
# Capture real (q,k,rope_emb,out) tensors from ck.apply_rope_split_half during a real DiT forward.
import sys, os, traceback
ROOT="D:/AnimaPort"; CU=ROOT+"/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv=["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu=True
import torch
import comfy.quant_ops as qo
import comfy.sd, comfy.utils

orig = qo.ck.apply_rope_split_half
captured = {}
def wrap(q, k, rope_emb, *a, **kw):
    out = orig(q, k, rope_emb, *a, **kw)
    if "q" not in captured:
        captured["q"] = q.detach().clone()
        captured["k"] = k.detach().clone()
        captured["rope"] = rope_emb.detach().clone()
        captured["oq"] = out[0].detach().clone()
        captured["ok"] = out[1].detach().clone()
        print("CAPTURED shapes: q", tuple(q.shape), "rope", tuple(rope_emb.shape), "out_q", tuple(out[0].shape))
    return out
qo.ck.apply_rope_split_half = wrap

DIT = CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors"
mp = comfy.sd.load_diffusion_model(DIT)
dit = mp.model.diffusion_model; dit.eval()
for m in dit.modules():
    if hasattr(m, "comfy_cast_weights"): m.comfy_cast_weights = False
dit.float()

torch.manual_seed(0)
x = torch.randn(1, 16, 1, 16, 16)        # random latent -> nonzero q,k
t = torch.ones(1, 1)
c = torch.randn(1, 512, 1024)
with torch.no_grad():
    dit(x, t, c)

torch.save(captured, ROOT + "/rope_capture.pt")
print("saved rope_capture.pt with keys:", list(captured.keys()))
