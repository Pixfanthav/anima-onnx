#!/usr/bin/env python
"""
Wrap every 5D MatMul (x_embedder + all MLP layers) as reshape->2D->MatMul->reshape->5D, so
DirectML runs them on GPU instead of falling back to CPU. 5D MatMul names come from the
shape-inferred graph; surgery is applied to the fp16 model.
  in0[b,t,h,w,K] @ w[K,N]  ->  Reshape(in0,[-1,K]) @ w -> Reshape(out,[b,t,h,w,N])
"""
import os, sys, time, onnx
from onnx import helper, TensorProto
ROOT = "D:/AnimaPort"
FP32 = "fp32" in sys.argv
SRC = (ROOT + "/onnx/dit.onnx") if FP32 else (ROOT + "/onnx_fp16/dit.onnx")
OUTDIR = (ROOT + "/onnx_fp32_dml") if FP32 else (ROOT + "/onnx_fp16_dml")
DST = OUTDIR + "/dit.onnx"
os.makedirs(OUTDIR, exist_ok=True)
print("SRC:", SRC, flush=True)

# 1) 5D MatMul node names from shape-inferred graph
inf = onnx.load(ROOT + "/onnx/dit_inf.onnx", load_external_data=False)
vi = {v.name: v for v in inf.graph.value_info}
def rank(n): return len(vi[n].type.tensor_type.shape.dim) if n in vi else -1
MM5 = set(n.name for n in inf.graph.node if n.op_type == "MatMul" and rank(n.input[0]) == 5)
print(f"5D MatMul to wrap: {len(MM5)}", flush=True)

# 2) load fp16 model
print("loading fp16...", flush=True); t0 = time.time()
m = onnx.load(SRC, load_external_data=True)
ini = {i.name: i for i in m.graph.initializer}
print("  loaded", round(time.time() - t0), "s", flush=True)

# shared constants
extra_init = [
    helper.make_tensor("s5d_c0", TensorProto.INT64, [1], [0]),
    helper.make_tensor("s5d_c4", TensorProto.INT64, [1], [4]),
]
new_nodes = []
wrapped = 0
for node in m.graph.node:
    if node.name in MM5 and node.op_type == "MatMul":
        in0, w = node.input[0], node.input[1]
        out = node.output[0]
        wt = ini[w]; K, N = int(wt.dims[0]), int(wt.dims[1])
        p = "s5d_" + node.name.replace("/", "_").replace(".", "_")
        extra_init.append(helper.make_tensor(p + "_neg1K", TensorProto.INT64, [2], [-1, K]))
        extra_init.append(helper.make_tensor(p + "_cN", TensorProto.INT64, [1], [N]))
        new_nodes += [
            helper.make_node("Shape", [in0], [p + "_s"]),
            helper.make_node("Slice", [p + "_s", "s5d_c0", "s5d_c4"], [p + "_batch"]),
            helper.make_node("Reshape", [in0, p + "_neg1K"], [p + "_a2d"]),
            helper.make_node("MatMul", [p + "_a2d", w], [p + "_y2d"]),
            helper.make_node("Concat", [p + "_batch", p + "_cN"], [p + "_oshape"], axis=0),
            helper.make_node("Reshape", [p + "_y2d", p + "_oshape"], [out]),
        ]
        wrapped += 1
    else:
        new_nodes.append(node)
print(f"wrapped {wrapped} nodes", flush=True)

m.graph.ClearField("node"); m.graph.node.extend(new_nodes)
m.graph.initializer.extend(extra_init)

print("saving...", flush=True); t0 = time.time()
onnx.save(m, DST, save_as_external_data=True, all_tensors_to_one_file=True,
          location="dit.weights", convert_attribute=True)
sz = sum(os.path.getsize(os.path.join(OUTDIR, f)) for f in os.listdir(OUTDIR)) / 1e9
print(f"  saved {round(time.time()-t0)}s -> {DST} (~{sz:.2f} GB)", flush=True)
