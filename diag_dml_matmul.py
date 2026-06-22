#!/usr/bin/env python
# Does RX580 DirectML support fp16 MatMul at all? Build tiny fp16 MatMul graphs and run on DML.
import numpy as np, onnx
from onnx import helper, TensorProto
import onnxruntime as ort

def test(M, K, N, dtype=TensorProto.FLOAT16, label=""):
    npdt = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
    A = helper.make_tensor_value_info('A', dtype, [1, M, K])
    Y = helper.make_tensor_value_info('Y', dtype, [1, M, N])
    Bw = (np.random.randn(K, N) * 0.05).astype(npdt)
    Bt = helper.make_tensor('B', dtype, [K, N], Bw.tobytes(), raw=True)
    node = helper.make_node('MatMul', ['A', 'B'], ['Y'])
    g = helper.make_graph([node], 't', [A], [Y], [Bt])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])
    path = f'D:/AnimaPort/mm_{label}.onnx'
    onnx.save(m, path)
    try:
        so = ort.SessionOptions(); so.log_severity_level = 3
        sess = ort.InferenceSession(path, so, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        a = (np.random.randn(1, M, K) * 0.05).astype(npdt)
        y = sess.run(None, {'A': a})[0]
        print(f"  {label} [{M}x{K}@{K}x{N}] {npdt.__name__}: OK shape={y.shape} EP={sess.get_providers()[0]}", flush=True)
        return True
    except Exception as e:
        print(f"  {label} [{M}x{K}@{K}x{N}] {npdt.__name__}: FAIL {str(e)[:120]}", flush=True)
        return False

print("DML providers:", ort.get_available_providers(), flush=True)
print("=== fp16 MatMul sizes ===", flush=True)
test(64, 64, 64, TensorProto.FLOAT16, "fp16_small")
test(1024, 1024, 1024, TensorProto.FLOAT16, "fp16_1k")
# x_embedder proj.1 is roughly: patches x (16*1*2*2=64? or model_dim). try a tall-K case
test(1024, 4096, 2048, TensorProto.FLOAT16, "fp16_bigK")
print("=== fp32 baseline (same sizes) ===", flush=True)
test(1024, 1024, 1024, TensorProto.FLOAT, "fp32_1k")
