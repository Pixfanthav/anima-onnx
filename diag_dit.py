#!/usr/bin/env python
# Diagnose why onnx DiT produces noise during sampling though STAGE R (t=1) matched.
# Compares torch vs onnx DiT across timestep values and context token counts.
import sys, os, numpy as np
ROOT = "D:/AnimaPort"; CU = ROOT + "/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv = ["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu = True
import torch, onnxruntime as ort
import comfy.sd

mp = comfy.sd.load_diffusion_model(CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors")
dit = mp.model.diffusion_model; dit.eval()
for m in dit.modules():
    if hasattr(m, "comfy_cast_weights"): m.comfy_cast_weights = False
dit.float()
sess = ort.InferenceSession(ROOT + "/onnx/dit.onnx", providers=["CPUExecutionProvider"])
IN = [i.name for i in sess.get_inputs()]
print("onnx inputs:", IN, flush=True)

def cmp(x, t, c, tag):
    with torch.no_grad(): r = dit(x, t, c).numpy()
    o = sess.run(None, {IN[0]: x.numpy().astype(np.float32),
                        IN[1]: t.numpy().astype(np.float32),
                        IN[2]: c.numpy().astype(np.float32)})[0]
    d = float(np.abs(o - r).max())
    print(f"  {tag}: maxdiff={d:.3e}  torch[mean={r.mean():+.4f} std={r.std():.4f}]  onnx[mean={o.mean():+.4f} std={o.std():.4f}]  {'MATCH' if d<1e-2 else 'DIVERGE'}", flush=True)

torch.manual_seed(0)
x32 = torch.randn(1, 16, 1, 32, 32); c512 = torch.randn(1, 512, 1024)

print("=== A) timestep sweep (latent 32x32, ctx 512) ===", flush=True)
for tv in [1.0, 0.7, 0.3, 0.1, 0.01]:
    cmp(x32, torch.full((1, 1), tv), c512, f"t={tv}")

print("=== B) context token-count sweep (t=0.5) ===", flush=True)
for tok in [77, 128, 256, 512]:
    cmp(x32, torch.full((1, 1), 0.5), torch.randn(1, tok, 1024), f"tok={tok}")

print("=== C) resolution sweep (t=0.5, ctx 512) ===", flush=True)
for (h, w) in [(32, 32), (64, 64), (144, 112)]:
    cmp(torch.randn(1, 16, 1, h, w), torch.full((1, 1), 0.5), c512, f"{h}x{w}")
print("=== DONE ===", flush=True)
