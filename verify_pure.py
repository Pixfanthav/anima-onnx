#!/usr/bin/env python
"""
verify_pure.py — prove pure_sampler.py (and pure_te.py) match ComfyUI on the FIXED pipeline.

Both paths share the SAME onnx DiT + onnx llm_adapter, so differences isolate to the pure
numpy sampler / tokenizer re-implementations.

  context512 = llm_adapter.onnx(raw_qwen, t5xxl_ids) -> pad512   (ComfyUI tokens, shared)
  reference  : ComfyUI KSampler with dit.forward = shim(llm_adapter.onnx + dit.onnx)
  candidate  : pure_sampler.generate_latent(dit.onnx, context512)
  TE check   : pure_te.encode(text) vs context512

RUN ONLY AFTER full-res fix_e2e finishes (shared onnx + heavy RAM). Default 256x256.
"""
import sys, os, json, time, numpy as np
ROOT = "D:/AnimaPort"; CU = ROOT + "/ComfyUI"
W = int(sys.argv[1]) if len(sys.argv) > 1 else 256
H = int(sys.argv[2]) if len(sys.argv) > 2 else 256
STEPS = 30; CFG = 4.0; SEED = 807882066116956

os.chdir(CU); sys.path.insert(0, CU); sys.argv = ["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu = True
import torch, onnxruntime as ort
sys.path.insert(0, ROOT)
import pure_sampler as ps, pure_te

def p(*a): print(*a, flush=True)

dit_sess = ort.InferenceSession(ROOT + "/onnx/dit.onnx", providers=["CPUExecutionProvider"])
DI = [i.name for i in dit_sess.get_inputs()]
ad_sess = ort.InferenceSession(ROOT + "/onnx/llm_adapter.onnx", providers=["CPUExecutionProvider"])
AI = [i.name for i in ad_sess.get_inputs()]
def dit_run(x, t, c):
    return dit_sess.run(None, {DI[0]: x.astype(np.float32), DI[1]: t.astype(np.float32), DI[2]: c.astype(np.float32)})[0]

import comfy.sd, nodes
from comfy_extras import nodes_model_advanced as nma
mp = comfy.sd.load_diffusion_model(CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors")
dit = mp.model.diffusion_model; dit.eval()
for m in dit.modules():
    if hasattr(m, "comfy_cast_weights"): m.comfy_cast_weights = False
dit.float()

def adapt_pad(qhidden_np, t5_ids_np):
    """onnx llm_adapter + pad512 (shared by both paths)."""
    if t5_ids_np.ndim == 1: t5_ids_np = t5_ids_np[None]
    adapted = ad_sess.run(None, {AI[0]: qhidden_np.astype(np.float32), AI[1]: t5_ids_np.astype(np.int64)})[0]
    L = adapted.shape[1]
    if L < 512: adapted = np.pad(adapted, ((0, 0), (0, 512 - L), (0, 0)))
    return adapted.astype(np.float32)

def shim(x, timesteps, context, *a, **kw):
    t = timesteps
    if t.ndim == 1: t = t.unsqueeze(1)
    ctx = context.detach().cpu().float().numpy()
    t5 = kw.get("t5xxl_ids")
    if t5 is not None:
        ctx = adapt_pad(ctx, t5.detach().cpu().numpy())
    out = dit_run(x.detach().cpu().float().numpy(), t.detach().cpu().float().numpy(), ctx)
    return torch.from_numpy(out).to(x.device).to(x.dtype)
dit.forward = shim
p("=== shim installed (onnx llm_adapter + dit) ===")

wf = json.load(open(ROOT + "/anima_comparison.json", encoding="utf-8"))
nd = {n["id"]: n for n in wf["nodes"]}
POS = nd[4]["widgets_values"][0]; NEG = nd[3]["widgets_values"][0]
model = nma.ModelSamplingAuraFlow().patch_aura(mp, 3.0)[0]
clip = nodes.CLIPLoader().load_clip("qwen_3_06b_base.safetensors", "stable_diffusion")[0]
pos = nodes.CLIPTextEncode().encode(clip, POS)[0]
neg = nodes.CLIPTextEncode().encode(clip, NEG)[0]
vae = nodes.VAELoader().load_vae("qwen_image_vae.safetensors")[0]

# shared context512 from ComfyUI tokens (raw qwen + t5xxl_ids) via onnx adapter
def ctx512_from(cond):
    qh = cond[0][0].detach().cpu().float().numpy()
    t5 = cond[0][1]["t5xxl_ids"].detach().cpu().numpy()
    return adapt_pad(qh, t5)
cpos = ctx512_from(pos).astype(np.float64)
cneg = ctx512_from(neg).astype(np.float64)
p("context512:", cpos.shape)

# ---- TE check: pure_te vs ComfyUI context512 ----
try:
    pte = pure_te.encode(POS).astype(np.float64)
    if pte.shape == cpos.shape:
        p(f"  >>> pure_te context MAE = {float(np.abs(pte - cpos).mean()):.6g}")
    else:
        p(f"  pure_te shape {pte.shape} != comfy {cpos.shape}")
except Exception as e:
    p("pure_te check skipped:", repr(e))

# ---- reference: ComfyUI KSampler (shim), SAME noise as pure (apples-to-apples) ----
import comfy.samplers
p(f"=== REFERENCE KSampler {W}x{H} ===")
t0 = time.time()
shape = (1, 16, 1, H // 8, W // 8)
noise_t = torch.from_numpy(ps.torch_initial_noise(shape, SEED))
lat_t = torch.zeros(shape)
ks = comfy.samplers.KSampler(model, STEPS, "cpu", sampler="er_sde", scheduler="simple", denoise=1.0, model_options=model.model_options)
ref_lat = ks.sample(noise_t, pos, neg, CFG, latent_image=lat_t, seed=SEED, disable_pbar=True).detach().cpu().float().numpy().astype(np.float64)
p("  ref", ref_lat.shape, round(time.time() - t0), "s")

# ---- candidate: pure numpy sampler (same onnx dit + shared context512) ----
p(f"=== PURE sampler {W}x{H} ===")
t0 = time.time()
pure_lat = ps.generate_latent(dit_run, cpos, cneg, tuple(ref_lat.shape), SEED, STEPS, CFG)
# KSampler.sample() applies latent_format.process_out on return; pure is raw -> apply it too
lf = model.get_model_object("latent_format")
pure_lat = lf.process_out(torch.from_numpy(pure_lat.astype(np.float32))).numpy().astype(np.float64)
p("  pure", pure_lat.shape, round(time.time() - t0), "s")

lat_mae = float(np.abs(ref_lat - pure_lat).mean()); lat_max = float(np.abs(ref_lat - pure_lat).max())
p(f"  >>> sampler latent MAE={lat_mae:.6g} MAX={lat_max:.6g}")

# decode both to images — chaotic SDE divergence shows as same content, different micro-noise
vae = nodes.VAELoader().load_vae("qwen_image_vae.safetensors")[0]
def dec(latnp):
    img = nodes.VAEDecode().decode(vae, {"samples": torch.from_numpy(latnp.astype(np.float32))})[0]
    return (img[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
ref_img = dec(ref_lat); pure_img = dec(pure_lat)
from PIL import Image
Image.fromarray(ref_img).save(ROOT + "/verify_ref.png")
Image.fromarray(pure_img).save(ROOT + "/verify_pure_img.png")
img_mae = float(np.abs(ref_img.astype(np.float64) - pure_img.astype(np.float64)).mean())
p(f"  >>> image MAE={img_mae:.3f} (saved verify_ref.png / verify_pure_img.png)")

open(ROOT + "/STATUS_PURE.md", "w", encoding="utf-8").write(
    "# Pure pipeline verification\n\n```\n" + json.dumps({
        "res": f"{W}x{H}", "latent_MAE": lat_mae, "latent_MAX": lat_max,
        "verdict": "MATCH" if lat_max < 1e-2 else ("CLOSE" if lat_mae < 1e-2 else "DIVERGE"),
    }, indent=2) + "\n```\n")
p("=== DONE -> STATUS_PURE.md ===")
