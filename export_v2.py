#!/usr/bin/env python
# ONNX export stage 1: fix CPU init, inspect all 3 components, export VAE decoder.
import sys, os, json, struct, traceback, inspect
ROOT = "D:/AnimaPort"
CU = ROOT + "/ComfyUI"
os.chdir(CU)
sys.path.insert(0, CU)

# --- force CPU before any comfy import (fixes torch.cuda.current_device crash) ---
sys.argv = ["main.py"]
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
import comfy.sd, comfy.utils  # noqa

DIT  = CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors"
TE   = CU + "/models/text_encoders/qwen_3_06b_base.safetensors"
VAE  = CU + "/models/vae/qwen_image_vae.safetensors"

print("==== STAGE1: env ====")
print("torch", torch.__version__, "| args.cpu =", comfy.cli_args.args.cpu)

# ---------- VAE ----------
print("\n==== VAE inspect + export ====")
try:
    sd = comfy.utils.load_torch_file(VAE)
    vae = comfy.sd.VAE(sd=sd)
    fsm = vae.first_stage_model
    fsm.eval()
    print("VAE first_stage_model class:", type(fsm).__name__)
    print("latent_channels:", getattr(vae, "latent_channels", "?"),
          "| downscale:", getattr(vae, "downscale_ratio", "?"))
    print("decode methods:", [m for m in dir(fsm) if "decode" in m.lower()])
    lc = getattr(vae, "latent_channels", 16)
    z = torch.zeros(1, lc, 64, 64, dtype=torch.float32)
    class Dec(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x): return s.m.decode(x)
    with torch.no_grad():
        y = Dec(fsm)(z)
    print("decode dummy OK, output shape:", tuple(y.shape))
    torch.onnx.export(Dec(fsm), z, ROOT + "/onnx/vae_decoder.onnx",
                      input_names=["latent"], output_names=["image"],
                      dynamic_axes={"latent": {2: "h", 3: "w"}, "image": {2: "H", 3: "W"}},
                      opset_version=17)
    print("VAE decoder ONNX export OK -> onnx/vae_decoder.onnx")
except Exception:
    print("VAE FAILED:\n" + traceback.format_exc())

# ---------- DiT ----------
print("\n==== DiT inspect ====")
try:
    mp = comfy.sd.load_diffusion_model(DIT)
    bm = mp.model
    dit = bm.diffusion_model
    print("BaseModel class:", type(bm).__name__, "| model_type:", getattr(bm, "model_type", "?"))
    print("diffusion_model class:", type(dit).__name__)
    try:
        print("forward signature:", str(inspect.signature(dit.forward)))
    except Exception as e:
        print("sig err:", e)
    # count params
    n = sum(p.numel() for p in dit.parameters())
    print(f"DiT params: {n/1e9:.2f}B | dtype sample:", next(dit.parameters()).dtype)
except Exception:
    print("DiT inspect FAILED:\n" + traceback.format_exc())

# ---------- TE ----------
print("\n==== TE inspect ====")
try:
    clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
    print("CLIP wrapper:", type(clip).__name__)
    cm = clip.cond_stage_model
    print("cond_stage_model class:", type(cm).__name__)
    print("submodules:", [k for k, _ in cm.named_children()][:10])
except Exception:
    print("TE inspect FAILED:\n" + traceback.format_exc())

print("\n==== STAGE1 DONE ====")
