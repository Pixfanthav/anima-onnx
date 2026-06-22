#!/usr/bin/env python
# Run surgered fp16 DiT on DML. Compare forward speed CPU vs DML to confirm GPU is actually used.
import onnxruntime as ort, numpy as np, time, sys
ROOT = "D:/AnimaPort"
MODEL = ROOT + "/onnx_fp16_dml/dit.onnx"

def mk():
    return (np.random.randn(1, 16, 1, 32, 32).astype(np.float16),
            np.array([[0.5]], dtype=np.float16),
            np.random.randn(1, 512, 1024).astype(np.float16))

def bench(providers, label):
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(MODEL, so, providers=providers)
    DI = [i.name for i in s.get_inputs()]
    x, t, c = mk()
    feeds = {DI[0]: x, DI[1]: t, DI[2]: c}
    s.run(None, feeds)  # warmup
    t0 = time.time()
    for _ in range(3): y = s.run(None, feeds)
    dt = (time.time() - t0) / 3
    print(f"  {label}: 32x32 {dt:.2f}s/fwd  out={y[0].shape}  EP={s.get_providers()[0]}", flush=True)
    # full-res latent
    x2 = np.random.randn(1, 16, 1, 144, 112).astype(np.float16)
    f2 = {DI[0]: x2, DI[1]: t, DI[2]: c}
    s.run(None, f2)
    t0 = time.time(); y2 = s.run(None, f2)
    print(f"  {label}: 144x112 {time.time()-t0:.2f}s/fwd out={y2[0].shape}", flush=True)

print("=== DML (GPU) ===", flush=True)
try:
    bench(["DmlExecutionProvider", "CPUExecutionProvider"], "DML")
except Exception as e:
    print("  DML FAIL:", str(e)[:160], flush=True)
print("=== CPU baseline ===", flush=True)
bench(["CPUExecutionProvider"], "CPU")
