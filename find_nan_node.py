#!/usr/bin/env python
# Find the first node where DML fp16 produces NaN (= the buggy DML op). Expose all node outputs,
# run on DML, report first NaN in topological order. Compare with CPU to be sure it's DML-specific.
import os, time, numpy as np, onnx
from onnx import helper
import onnxruntime as ort
ROOT = "D:/AnimaPort"
SRC = ROOT + "/onnx_fp16_dml/dit.onnx"
DBG = ROOT + "/onnx_dbg/dit.onnx"
os.makedirs(ROOT + "/onnx_dbg", exist_ok=True)

print("infer types (graph only)...", flush=True); t0 = time.time()
mi = onnx.shape_inference.infer_shapes(onnx.load(SRC, load_external_data=False))
tmap = {v.name: v.type.tensor_type.elem_type for v in mi.graph.value_info}
for vi in list(mi.graph.input) + list(mi.graph.output):
    tmap[vi.name] = vi.type.tensor_type.elem_type
print(f"  type map {len(tmap)} in {round(time.time()-t0)}s", flush=True)

print("load + expose all node outputs...", flush=True); t0 = time.time()
m = onnx.load(SRC, load_external_data=True)
existing = {o.name for o in m.graph.output}
order = []  # topological order of exposed tensors
for n in m.graph.node:
    for o in n.output:
        if o and o not in existing:
            et = tmap.get(o, onnx.TensorProto.FLOAT16)
            if et == 0: et = onnx.TensorProto.FLOAT16
            m.graph.output.append(helper.make_tensor_value_info(o, et, None))
            order.append((o, n.op_type, n.name))
print(f"  exposed {len(order)} tensors in {round(time.time()-t0)}s", flush=True)
onnx.save(m, DBG, save_as_external_data=True, all_tensors_to_one_file=True, location="dit.weights")
print("  saved dbg model", flush=True)

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
np.random.seed(0)
x = np.random.randn(1, 16, 1, 32, 32).astype(np.float16)
t = np.array([[1.0]], dtype=np.float16)
c = np.random.randn(1, 512, 1024).astype(np.float16)

def first_nan(providers, label):
    s = ort.InferenceSession(DBG, so, providers=providers)
    DI = [i.name for i in s.get_inputs()]
    names = [o.name for o in s.get_outputs()]
    res = dict(zip(names, s.run(None, {DI[0]: x, DI[1]: t, DI[2]: c})))
    meta = {o: (op, nm) for o, op, nm in order}
    for o, op, nm in order:
        if o in res:
            a = np.asarray(res[o]).astype(np.float32)
            if a.size and (np.isnan(a).any() or np.isinf(a).any()):
                print(f"  [{label}] FIRST NaN/Inf: op={op} node={nm} out={o} shape={a.shape}", flush=True)
                return nm, op
    print(f"  [{label}] no NaN found", flush=True)
    return None, None

print("=== DML ===", flush=True)
first_nan(["DmlExecutionProvider", "CPUExecutionProvider"], "DML")
