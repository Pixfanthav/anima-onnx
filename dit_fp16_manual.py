#!/usr/bin/env python
"""
Manual fp16 conversion (bypass onnxconverter_common, which left mixed types / hit Windows
temp PermissionError on this external-data model). Convert ALL float tensors consistently:
initializers, Constant node values, Cast targets, graph IO. -> uniform fp16 graph, loadable.
"""
import os, time, numpy as np
import onnx
from onnx import numpy_helper, TensorProto
ROOT = "D:/AnimaPort"
SRC = ROOT + "/onnx/dit.onnx"
DST = ROOT + "/onnx_fp16/dit.onnx"
os.makedirs(ROOT + "/onnx_fp16", exist_ok=True)
F, F16 = TensorProto.FLOAT, TensorProto.FLOAT16

print("=== loading fp32 (external) ===", flush=True)
t0 = time.time()
m = onnx.load(SRC, load_external_data=True)
print("  loaded", round(time.time() - t0), "s", flush=True)

def conv(t):
    arr = numpy_helper.to_array(t).astype(np.float16)
    t.CopyFrom(numpy_helper.from_array(arr, t.name))

ni = 0
for init in m.graph.initializer:
    if init.data_type == F:
        conv(init); ni += 1
nc = ncast = 0
for node in m.graph.node:
    if node.op_type == "Constant":
        for a in node.attribute:
            if a.name == "value" and a.t.data_type == F:
                conv(a.t); nc += 1
    elif node.op_type == "Cast":
        for a in node.attribute:
            if a.name == "to" and a.i == F:
                a.i = F16; ncast += 1
nio = 0
for vi in list(m.graph.input) + list(m.graph.output):
    if vi.type.tensor_type.elem_type == F:
        vi.type.tensor_type.elem_type = F16; nio += 1
del m.graph.value_info[:]  # drop stale fp32 value_info; ORT re-infers
print(f"=== converted: init={ni} const={nc} cast={ncast} io={nio} ===", flush=True)

print("=== saving fp16 ===", flush=True)
t0 = time.time()
onnx.save(m, DST, save_as_external_data=True, all_tensors_to_one_file=True,
          location="dit.weights", convert_attribute=True)
sz = sum(os.path.getsize(os.path.join(ROOT + "/onnx_fp16", f)) for f in os.listdir(ROOT + "/onnx_fp16")) / 1e9
print(f"  saved {round(time.time()-t0)}s -> {DST}  (~{sz:.2f} GB)", flush=True)
