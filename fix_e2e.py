#!/usr/bin/env python
# Fixed E2E: insert llm_adapter.onnx (qwen_hidden + t5xxl_ids -> adapted -> pad512) before dit.onnx.
import sys, os, json, numpy as np
ROOT = "D:/AnimaPort"; CU = ROOT + "/ComfyUI"
W = int(sys.argv[1]) if len(sys.argv) > 1 else 256
H = int(sys.argv[2]) if len(sys.argv) > 2 else 256
FP16 = "fp16" in sys.argv
DML = "dml" in sys.argv
os.chdir(CU); sys.path.insert(0, CU); sys.argv = ["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu = True
import torch, onnxruntime as ort
import comfy.sd, nodes
from comfy_extras import nodes_model_advanced as nma

PROV = (["DmlExecutionProvider", "CPUExecutionProvider"] if DML else ["CPUExecutionProvider"])
print("providers:", PROV, "| fp16:", FP16, flush=True)
if FP16:
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # DML needs the 5D-MatMul-wrapped (surgered) model; CPU can use the plain fp16
    fp16_path = ROOT + "/onnx_fp16_dml/dit.onnx" if DML else ROOT + "/onnx_fp16/dit.onnx"
    dit_sess = ort.InferenceSession(fp16_path, so, providers=PROV)
    DITDT = np.float16
else:
    dit_sess = ort.InferenceSession(ROOT + "/onnx/dit.onnx", providers=PROV)
    DITDT = np.float32
DI = [i.name for i in dit_sess.get_inputs()]
print("dit EP:", dit_sess.get_providers(), flush=True)
ad_sess = ort.InferenceSession(ROOT + "/onnx/llm_adapter.onnx", providers=PROV)
AI = [i.name for i in ad_sess.get_inputs()]
print("dit inputs", DI, "| adapter inputs", AI, flush=True)

mp = comfy.sd.load_diffusion_model(CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors")
dit = mp.model.diffusion_model; dit.eval()
for m in dit.modules():
    if hasattr(m, "comfy_cast_weights"): m.comfy_cast_weights = False
dit.float()

def shim(x, timesteps, context, *a, **kw):
    t = timesteps
    if t.ndim == 1: t = t.unsqueeze(1)
    ctx = context.detach().cpu().float().numpy().astype(np.float32)
    t5 = kw.get("t5xxl_ids")
    if t5 is not None:                                   # non-inference path: adapt here
        t5n = t5.detach().cpu().numpy().astype(np.int64)
        if t5n.ndim == 1: t5n = t5n[None]
        adapted = ad_sess.run(None, {AI[0]: ctx, AI[1]: t5n})[0]
        w = kw.get("t5xxl_weights")
        if w is not None:
            adapted = adapted * w.detach().cpu().float().numpy()
        L = adapted.shape[1]
        if L < 512:
            adapted = np.pad(adapted, ((0, 0), (0, 512 - L), (0, 0)))
        ctx = adapted.astype(np.float32)
    out = dit_sess.run(None, {DI[0]: x.detach().cpu().numpy().astype(DITDT),
                              DI[1]: t.detach().cpu().numpy().astype(DITDT), DI[2]: ctx.astype(DITDT)})[0]
    return torch.from_numpy(out.astype(np.float32)).to(x.device).to(x.dtype)
dit.forward = shim
print("=== shim installed (llm_adapter + dit) ===", flush=True)

wf = json.load(open(ROOT + "/anima_comparison.json", encoding="utf-8"))
nd = {n["id"]: n for n in wf["nodes"]}
POS = nd[4]["widgets_values"][0]; NEG = nd[3]["widgets_values"][0]
model = nma.ModelSamplingAuraFlow().patch_aura(mp, 3.0)[0]
clip = nodes.CLIPLoader().load_clip("qwen_3_06b_base.safetensors", "stable_diffusion")[0]
pos = nodes.CLIPTextEncode().encode(clip, POS)[0]
neg = nodes.CLIPTextEncode().encode(clip, NEG)[0]
vae = nodes.VAELoader().load_vae("qwen_image_vae.safetensors")[0]
lat = nodes.EmptyLatentImage().generate(W, H, 1)[0]
samples = nodes.KSampler().sample(model, 807882066116956, 30, 4.0, "er_sde", "simple", pos, neg, lat, 1.0)[0]
img = nodes.VAEDecode().decode(vae, samples)[0]
arr = (img[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
from PIL import Image
out_png = ROOT + f"/onnx_fix_{'fp16_' if FP16 else ''}{W}x{H}.png"
Image.fromarray(arr).save(out_png)
print(f"=== saved {out_png} mean {float(arr.mean()):.2f} ===", flush=True)

# compare to baseline (same seed/prompt, native torch DiT) if shapes match
base_path = ROOT + "/ComfyUI/output/anima_baseline_00001_.png"
if os.path.exists(base_path):
    base = np.asarray(Image.open(base_path).convert("RGB")).astype(np.float32)
    gen = arr.astype(np.float32)
    if base.shape == gen.shape:
        print(f"=== MAE vs baseline = {float(np.abs(base - gen).mean()):.3f} (0=identical) ===", flush=True)
    else:
        print(f"=== baseline shape {base.shape} != gen {gen.shape}; skip MAE ===", flush=True)
