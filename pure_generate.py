#!/usr/bin/env python
"""
pure_generate.py — fully ComfyUI-free Anima text-to-image (numpy + onnxruntime only).

This is the PC reference for the Android port: NO ComfyUI / no torch model. Pipeline:
  prompt --pure_te--> context512 (qwen3_te.onnx + llm_adapter.onnx)
         --pure_sampler--> raw latent (dit.onnx + numpy er_sde)
         --process_out (Wan21 latents_mean/std)--> vae_decoder.onnx --> image

Only external deps: numpy, onnxruntime, transformers (tokenizers), PIL.
(qwen25/t5 tokenizer folders are data under ComfyUI dir but no ComfyUI code is imported.)

Usage: pure_generate.py [W] [H] [seed]   (default 256 256, seed 807882066116956)
"""
import sys, os, time, numpy as np, onnxruntime as ort
ROOT = "D:/AnimaPort"
sys.path.insert(0, ROOT)
import pure_sampler as ps
import pure_te

W = int(sys.argv[1]) if len(sys.argv) > 1 else 256
H = int(sys.argv[2]) if len(sys.argv) > 2 else 256
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 807882066116956
STEPS, CFG = 30, 4.0
# use the EXACT workflow prompts (same as fix_e2e/baseline) so results are comparable
import json as _json
_wf = _json.load(open(ROOT + "/anima_comparison.json", encoding="utf-8"))
_nd = {n["id"]: n for n in _wf["nodes"]}
POS = _nd[4]["widgets_values"][0]
NEG = _nd[3]["widgets_values"][0]

# Wan21 latent format: process_out(x) = x * std + mean  (scale_factor=1.0)
MEAN = np.array([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
                dtype=np.float64).reshape(1, 16, 1, 1, 1)
STD = np.array([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160],
               dtype=np.float64).reshape(1, 16, 1, 1, 1)

def p(*a): print(*a, flush=True)

# ---- onnx sessions (DiT, VAE). adapter+TE handled inside pure_te. ----
dit_sess = ort.InferenceSession(ROOT + "/onnx/dit.onnx", providers=["CPUExecutionProvider"])
DI = [i.name for i in dit_sess.get_inputs()]
def dit_run(x, t, c):
    return dit_sess.run(None, {DI[0]: x.astype(np.float32), DI[1]: t.astype(np.float32), DI[2]: c.astype(np.float32)})[0]
vae_sess = ort.InferenceSession(ROOT + "/onnx/vae_decoder.onnx", providers=["CPUExecutionProvider"])
VI = vae_sess.get_inputs()[0].name

p("=== text encode (qwen3 + t5 + llm_adapter) ===")
t0 = time.time()
ctx_pos = pure_te.encode(POS).astype(np.float64)
ctx_neg = pure_te.encode(NEG).astype(np.float64)
p("  context", ctx_pos.shape, round(time.time() - t0), "s")

p(f"=== sample {W}x{H} steps={STEPS} cfg={CFG} seed={SEED} ===")
t0 = time.time()
shape = (1, 16, 1, H // 8, W // 8)
cache = ROOT + f"/pure_latent_{W}x{H}_{SEED}.npy"
if os.path.exists(cache):
    latent = np.load(cache); p("  latent (cached)", latent.shape)
else:
    latent = ps.generate_latent(dit_run, ctx_pos, ctx_neg, shape, SEED, STEPS, CFG)  # raw
    np.save(cache, latent); p("  latent", latent.shape, round(time.time() - t0), "s")

p("=== vae decode ===")
latent_proc = (latent.astype(np.float64) * STD + MEAN).astype(np.float32)   # Wan21 process_out
img = vae_sess.run(None, {VI: latent_proc})[0]                              # first_stage decode, [-1,1]
img = np.asarray(img)
if img.ndim == 5: img = img[:, :, 0]          # drop temporal
arr = (((img[0].transpose(1, 2, 0) + 1.0) / 2.0) * 255).clip(0, 255).astype(np.uint8)  # [-1,1]->[0,255]
from PIL import Image
out = ROOT + f"/pure_gen_{W}x{H}.png"
Image.fromarray(arr).save(out)
p(f"=== saved {out} mean {arr.mean():.1f} ===")
