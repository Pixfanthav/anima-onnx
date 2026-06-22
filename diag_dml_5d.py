#!/usr/bin/env python
# Confirm: fp16 5D MatMul fails on DML, but reshape->3D->MatMul->reshape->5D works.
import numpy as np, onnx
from onnx import helper, TensorProto
import onnxruntime as ort
ROOT = "D:/AnimaPort"

def run(path, feeds, label):
    try:
        so = ort.SessionOptions(); so.log_severity_level = 3
        s = ort.InferenceSession(path, so, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        y = s.run(None, feeds)[0]
        print(f"  {label}: OK shape={y.shape} EP={s.get_providers()[0]}", flush=True)
    except Exception as e:
        print(f"  {label}: FAIL {str(e)[:100]}", flush=True)

B, T, H, W, K, N = 1, 1, 32, 32, 512, 2048
A = (np.random.randn(B, T, H, W, K) * 0.05).astype(np.float16)
Bw = (np.random.randn(K, N) * 0.05).astype(np.float16)
Bt = helper.make_tensor('B', TensorProto.FLOAT16, [K, N], Bw.tobytes(), raw=True)

# 1) direct 5D MatMul
ai = helper.make_tensor_value_info('A', TensorProto.FLOAT16, [B, T, H, W, K])
yo = helper.make_tensor_value_info('Y', TensorProto.FLOAT16, [B, T, H, W, N])
g1 = helper.make_graph([helper.make_node('MatMul', ['A', 'B'], ['Y'])], 'd', [ai], [yo], [Bt])
onnx.save(helper.make_model(g1, opset_imports=[helper.make_opsetid('', 17)]), ROOT + "/mm5_direct.onnx")

# 2) reshape -> 2D MatMul -> reshape back (dynamic batch via Shape/Slice/Concat)
nodes = [
    helper.make_node('Shape', ['A'], ['s']),                                   # [5]
    helper.make_node('Constant', [], ['c4'], value=helper.make_tensor('c4', TensorProto.INT64, [1], [4])),
    helper.make_node('Constant', [], ['c0'], value=helper.make_tensor('c0', TensorProto.INT64, [1], [0])),
    helper.make_node('Constant', [], ['cN'], value=helper.make_tensor('cN', TensorProto.INT64, [1], [N])),
    helper.make_node('Constant', [], ['neg1K'], value=helper.make_tensor('neg1K', TensorProto.INT64, [2], [-1, K])),
    helper.make_node('Slice', ['s', 'c0', 'c4'], ['batch4']),                   # first 4 dims
    helper.make_node('Reshape', ['A', 'neg1K'], ['a2d']),                       # [-1, K]
    helper.make_node('MatMul', ['a2d', 'B'], ['y2d']),                          # [-1, N]
    helper.make_node('Concat', ['batch4', 'cN'], ['outshape'], axis=0),         # [b,t,h,w,N]
    helper.make_node('Reshape', ['y2d', 'outshape'], ['Y']),
]
g2 = helper.make_graph(nodes, 'r', [ai], [yo], [Bt])
onnx.save(helper.make_model(g2, opset_imports=[helper.make_opsetid('', 17)]), ROOT + "/mm5_reshape.onnx")

print("DML 5D MatMul test:", flush=True)
run(ROOT + "/mm5_direct.onnx", {'A': A}, "direct 5D")
run(ROOT + "/mm5_reshape.onnx", {'A': A}, "reshape 2D wrap")
