#!/usr/bin/env python
# Log what args ComfyUI's sampler actually passes to DiT.forward (fps? padding_mask?).
import sys, os, json
ROOT = "D:/AnimaPort"; CU = ROOT + "/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv = ["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu = True
import torch
import comfy.sd, nodes
from comfy_extras import nodes_model_advanced as nma

mp = comfy.sd.load_diffusion_model(CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors")
dit = mp.model.diffusion_model; dit.eval()
for m in dit.modules():
    if hasattr(m, "comfy_cast_weights"): m.comfy_cast_weights = False
dit.float()

orig_fwd = dit.forward
CAP = {}
def logfwd(x, timesteps, context=None, *a, **kw):
    def desc(v):
        if v is None: return None
        if torch.is_tensor(v): return {"shape": list(v.shape), "dtype": str(v.dtype), "min": float(v.min()), "max": float(v.max())}
        if isinstance(v, dict): return "dict(" + ",".join(v.keys()) + ")"
        return repr(v)[:80]
    CAP["x"] = desc(x); CAP["timesteps"] = desc(timesteps); CAP["context"] = desc(context)
    CAP["positional_extra"] = [desc(z) for z in a]
    CAP["kwargs"] = {k: desc(v) for k, v in kw.items()}
    print(json.dumps(CAP, indent=2), flush=True)
    # isolate which kwarg changes the native output
    base = orig_fwd(x, timesteps, context).detach()
    def diff(**extra):
        o = orig_fwd(x, timesteps, context, **extra).detach()
        return float((base - o).abs().max())
    to = kw.get("transformer_options", {})
    print(f"  +control            : {diff(control=kw.get('control')):.4e}", flush=True)
    print(f"  +transformer_options: {diff(transformer_options=to):.4e}", flush=True)
    if "t5xxl_ids" in kw:
        print(f"  +t5xxl_ids/weights  : {diff(t5xxl_ids=kw['t5xxl_ids'], t5xxl_weights=kw['t5xxl_weights']):.4e}", flush=True)
    print(f"  +ALL                : {diff(**kw):.4e}", flush=True)
    raise SystemExit(0)
dit.forward = logfwd

wf = json.load(open(ROOT + "/anima_comparison.json", encoding="utf-8"))
nd = {n["id"]: n for n in wf["nodes"]}
POS = nd[4]["widgets_values"][0]; NEG = nd[3]["widgets_values"][0]
model = nma.ModelSamplingAuraFlow().patch_aura(mp, 3.0)[0]
clip = nodes.CLIPLoader().load_clip("qwen_3_06b_base.safetensors", "stable_diffusion")[0]
pos = nodes.CLIPTextEncode().encode(clip, POS)[0]
neg = nodes.CLIPTextEncode().encode(clip, NEG)[0]
lat = nodes.EmptyLatentImage().generate(256, 256, 1)[0]
try:
    nodes.KSampler().sample(model, 807882066116956, 30, 4.0, "er_sde", "simple", pos, neg, lat, 1.0)
except SystemExit:
    print("=== captured first DiT call; stopped ===", flush=True)
