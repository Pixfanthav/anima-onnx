#!/usr/bin/env python
# Step-by-step compare: ComfyUI er_sde vs pure er_sde. Find where x diverges and why.
import sys, os, json, numpy as np
ROOT = "D:/AnimaPort"; CU = ROOT + "/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv = ["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu = True
import torch, onnxruntime as ort
sys.path.insert(0, ROOT)
import pure_sampler as ps

dit_sess = ort.InferenceSession(ROOT + "/onnx/dit.onnx", providers=["CPUExecutionProvider"])
DI = [i.name for i in dit_sess.get_inputs()]
ad_sess = ort.InferenceSession(ROOT + "/onnx/llm_adapter.onnx", providers=["CPUExecutionProvider"])
AI = [i.name for i in ad_sess.get_inputs()]
def dit_run(x, t, c):
    return dit_sess.run(None, {DI[0]: x.astype(np.float32), DI[1]: t.astype(np.float32), DI[2]: c.astype(np.float32)})[0]
def adapt_pad(q, t5):
    if t5.ndim == 1: t5 = t5[None]
    a = ad_sess.run(None, {AI[0]: q.astype(np.float32), AI[1]: t5.astype(np.int64)})[0]
    L = a.shape[1]
    if L < 512: a = np.pad(a, ((0, 0), (0, 512 - L), (0, 0)))
    return a.astype(np.float32)

import comfy.sd, nodes, comfy.sample, comfy.samplers
from comfy_extras import nodes_model_advanced as nma
mp = comfy.sd.load_diffusion_model(CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors")
dit = mp.model.diffusion_model; dit.eval()
for m in dit.modules():
    if hasattr(m, "comfy_cast_weights"): m.comfy_cast_weights = False
dit.float()
def shim(x, timesteps, context, *a, **kw):
    t = timesteps
    if t.ndim == 1: t = t.unsqueeze(1)
    ctx = context.detach().cpu().float().numpy()
    t5 = kw.get("t5xxl_ids")
    if t5 is not None: ctx = adapt_pad(ctx, t5.detach().cpu().numpy())
    out = dit_run(x.detach().cpu().float().numpy(), t.detach().cpu().float().numpy(), ctx)
    return torch.from_numpy(out).to(x.device).to(x.dtype)
dit.forward = shim

wf = json.load(open(ROOT + "/anima_comparison.json", encoding="utf-8"))
nd = {n["id"]: n for n in wf["nodes"]}
POS = nd[4]["widgets_values"][0]; NEG = nd[3]["widgets_values"][0]
model = nma.ModelSamplingAuraFlow().patch_aura(mp, 3.0)[0]
clip = nodes.CLIPLoader().load_clip("qwen_3_06b_base.safetensors", "stable_diffusion")[0]
pos = nodes.CLIPTextEncode().encode(clip, POS)[0]
neg = nodes.CLIPTextEncode().encode(clip, NEG)[0]
cpos = adapt_pad(pos[0][0].detach().cpu().float().numpy(), pos[0][1]["t5xxl_ids"].detach().cpu().numpy())
cneg = adapt_pad(neg[0][0].detach().cpu().float().numpy(), neg[0][1]["t5xxl_ids"].detach().cpu().numpy())

STEPS = 30; SEED = 807882066116956
lat_t = torch.zeros((1, 16, 1, 32, 32))
noise = comfy.sample.prepare_noise(lat_t, SEED)
ks = comfy.samplers.KSampler(model, steps=STEPS, device="cpu", sampler="er_sde", scheduler="simple", denoise=1.0, model_options=model.model_options)
comfy_steps = []
def cb_c(step, denoised, x, total):
    comfy_steps.append((int(step), x.detach().cpu().numpy().astype(np.float32).copy(), denoised.detach().cpu().numpy().astype(np.float32).copy()))
ks.sample(noise, pos, neg, cfg=4.0, latent_image=lat_t, callback=cb_c, seed=SEED, disable_pbar=True)

pure_steps = []
def cb_p(i, x, denoised):
    pure_steps.append((int(i), x.copy(), denoised.copy()))
ps.generate_latent(dit_run, cpos.astype(np.float32), cneg.astype(np.float32), (1, 16, 1, 32, 32), SEED, STEPS, 4.0, callback=cb_p)

print("step :  x_MAE      denoised_MAE", flush=True)
for (ic, xc, dc), (ip, xp, dp) in zip(comfy_steps, pure_steps):
    print(f"{ic:3d}  : {np.abs(xc-xp).mean():.4e}  {np.abs(dc-dp).mean():.4e}", flush=True)
