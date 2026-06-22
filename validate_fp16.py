#!/usr/bin/env python
# Validate fp16 DiT vs fp32 DiT across timesteps/resolution. fp16 has limited mantissa so
# expect larger diffs than onnx-vs-torch; what matters is whether the sampled image stays clean.
import sys, os, numpy as np
ROOT = "D:/AnimaPort"; CU = ROOT + "/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv = ["main.py"]
import onnxruntime as ort

# fp16 graph has leftover Casts (cast-cleanup was no-op'd); ORT fusion chokes on them, so
# disable graph optimization to load. (Mobile uses pre-optimized models; fine for validation.)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
f32 = ort.InferenceSession(ROOT + "/onnx/dit.onnx", so, providers=["CPUExecutionProvider"])
f16 = ort.InferenceSession(ROOT + "/onnx_fp16/dit.onnx", so, providers=["CPUExecutionProvider"])
I32 = [i.name for i in f32.get_inputs()]
I16 = [i.name for i in f16.get_inputs()]
print("fp16 inputs:", [(i.name, i.type) for i in f16.get_inputs()], flush=True)

def run(sess, names, x, t, c, dt):
    return sess.run(None, {names[0]: x.astype(dt), names[1]: t.astype(dt), names[2]: c.astype(dt)})[0]

rng = np.random.default_rng(0)
for (h, w) in [(32, 32), (144, 112)]:
    x = rng.standard_normal((1, 16, 1, h, w)).astype(np.float32)
    c = rng.standard_normal((1, 512, 1024)).astype(np.float32)
    for tv in [0.999, 0.5, 0.05]:
        t = np.full((1, 1), tv, dtype=np.float32)
        r32 = run(f32, I32, x, t, c, np.float32)
        r16 = run(f16, I16, x, t, c, np.float16)
        d = np.abs(r32.astype(np.float64) - r16.astype(np.float64))
        rel = d.mean() / (np.abs(r32).mean() + 1e-9)
        print(f"  {h}x{w} t={tv}: MAE={d.mean():.4e} MAX={d.max():.4e} rel={rel:.2%}", flush=True)
print("=== done. small rel% => fp16 safe ===", flush=True)
