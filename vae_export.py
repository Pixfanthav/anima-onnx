#!/usr/bin/env python
# Focused VAE decoder export — WanVAE needs 5D input [B,C,T,H,W]
import sys, os, traceback
ROOT = "D:/AnimaPort"; CU = ROOT + "/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU)
sys.argv = ["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu = True
import torch, comfy.sd, comfy.utils

VAE = CU + "/models/vae/qwen_image_vae.safetensors"
sd = comfy.utils.load_torch_file(VAE)
vae = comfy.sd.VAE(sd=sd)
fsm = vae.first_stage_model; fsm.eval()
print("VAE:", type(fsm).__name__, "latent_channels:", vae.latent_channels)

# --- disable ComfyUI runtime weight-casting so conv kernels have static shape for ONNX ---
n_patched = 0
for mod in fsm.modules():
    if hasattr(mod, "comfy_cast_weights"):
        mod.comfy_cast_weights = False
        n_patched += 1
fsm.float()  # force all params to fp32 (no-op cast)
print(f"patched comfy_cast_weights=False on {n_patched} modules; dtype now fp32")

# 5D latent: [batch, 16ch, T=1, H/8, W/8]
z = torch.zeros(1, 16, 1, 64, 64, dtype=torch.float32)

class Dec(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m.decode(x)

with torch.no_grad():
    y = Dec(fsm)(z)
print("decode OK, output shape:", tuple(y.shape))

try:
    torch.onnx.export(
        Dec(fsm), z, ROOT + "/onnx/vae_decoder.onnx",
        input_names=["latent"], output_names=["image"],
        dynamic_axes={"latent": {0: "b", 2: "t", 3: "h", 4: "w"},
                      "image":  {0: "b", 2: "T", 3: "H", 4: "W"}},
        opset_version=17)
    print("ONNX export OK -> onnx/vae_decoder.onnx")
    print("size:", os.path.getsize(ROOT + "/onnx/vae_decoder.onnx")/1e6, "MB")
except Exception:
    print("ONNX export FAILED:\n" + traceback.format_exc())
