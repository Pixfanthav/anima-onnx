#!/usr/bin/env python
"""
fp16 conversion, value_info-guided. The plain converter (in-memory / model_path / manual) all
hit float/float16 mixes because, without per-tensor type info, boundaries between fp32-output
ops (ConstantOfShape, padding_mask gen, ...) and fp16 tensors were left un-cast.

Fix: run disk-based shape inference first (handles the 7.7GB external-data model), which fills
graph.value_info with every tensor's type. Then onnxconverter_common, given that value_info,
inserts the boundary Casts correctly. RUN AFTER E2E idle (loads ~7.7GB).
"""
import os, time
import onnx
from onnx import shape_inference
from onnxconverter_common import float16
# onnxconverter_common 1.16 bug: remove_unnecessary_cast_node crashes with node_block_list.
# It's only a Cast-cleanup optimization; no-op it (a few extra Casts remain, correctness intact).
float16.remove_unnecessary_cast_node = lambda graph: None
ROOT = "D:/AnimaPort"
SRC = ROOT + "/onnx/dit.onnx"
INF = ROOT + "/onnx/dit_inf.onnx"          # same dir as SRC so it can reference dit's external data
DST = ROOT + "/onnx_fp16/dit.onnx"
os.makedirs(ROOT + "/onnx_fp16", exist_ok=True)

if not os.path.exists(INF):
    print("=== disk-based shape inference ===", flush=True)
    t0 = time.time()
    shape_inference.infer_shapes_path(SRC, INF)
    print("  inferred", round(time.time() - t0), "s", flush=True)
else:
    print("=== reuse existing", INF, "===", flush=True)

print("=== load inferred (with value_info) ===", flush=True)
t0 = time.time()
model = onnx.load(INF)
print("  loaded", round(time.time() - t0), "s, value_info:", len(model.graph.value_info), flush=True)

# Keep position/timestep embedders in fp32: they compute coords (int->float, sinusoidal/RoPE)
# that are precision-sensitive and produce fp32 outputs the converter can't safely halve.
# node_block_list keeps them fp32 AND auto-inserts boundary Casts to the fp16 stream.
# Keep fp32: embedders (precision-sensitive coords), Casts (explicit types), and norm ops
# (rms_norm's x^2 / mean / rsqrt overflow fp16 on large residuals -> NaN). MatMul weights stay fp16.
NORM_OPS = {"Pow", "ReduceMean", "Sqrt", "Reciprocal", "Softmax", "ReduceSum"}
NODE_BLOCK = [n.name for n in model.graph.node
              if "pos_embedder" in n.name or "t_embedder" in n.name
              or n.op_type == "Cast" or n.op_type in NORM_OPS]
print(f"=== node_block_list: {len(NODE_BLOCK)} nodes kept fp32 (embedders + Casts + norm ops) ===", flush=True)

print("=== convert to fp16 (value_info-guided, auto boundary Casts) ===", flush=True)
t0 = time.time()
model16 = float16.convert_float_to_float16(model, keep_io_types=False, disable_shape_infer=True,
                                           node_block_list=NODE_BLOCK)
print("  converted", round(time.time() - t0), "s", flush=True)

print("=== save fp16 ===", flush=True)
t0 = time.time()
onnx.save(model16, DST, save_as_external_data=True, all_tensors_to_one_file=True,
          location="dit.weights", convert_attribute=True)
sz = sum(os.path.getsize(os.path.join(ROOT + "/onnx_fp16", f)) for f in os.listdir(ROOT + "/onnx_fp16")) / 1e9
print(f"=== DONE  saved {round(time.time()-t0)}s -> {DST}  (~{sz:.2f} GB) ===", flush=True)
