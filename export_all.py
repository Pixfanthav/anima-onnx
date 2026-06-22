#!/usr/bin/env python
# Best-effort ONNX export recon for the 3 Anima components. Never fatal; logs everything.
import sys, os, struct, json, traceback
ROOT = "D:/AnimaPort"
CU = ROOT + "/ComfyUI"
sys.argv = ["main.py", "--cpu"]
os.chdir(CU)
sys.path.insert(0, CU)

COMPONENTS = {
    "dit":  CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors",
    "te":   CU + "/models/text_encoders/qwen_3_06b_base.safetensors",
    "vae":  CU + "/models/vae/qwen_image_vae.safetensors",
}

def header_recon(path):
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        keys = [k for k in h if k != "__metadata__"]
        dtypes = {}
        for k in keys:
            dtypes[h[k]["dtype"]] = dtypes.get(h[k]["dtype"], 0) + 1
        print(f"  tensors={len(keys)} dtypes={dtypes}")
        print(f"  sample keys: {keys[:3]}")
        return True
    except Exception:
        print("  HEADER RECON FAILED:\n" + traceback.format_exc())
        return False

print("==== EXPORT RECON ====")
for name, p in COMPONENTS.items():
    print(f"[{name}] {p}  exists={os.path.exists(p)}")
    if os.path.exists(p):
        header_recon(p)

# Best-effort: load VAE via ComfyUI and try ONNX export of the decoder.
print("\n==== VAE ONNX EXPORT ATTEMPT ====")
try:
    import torch
    import comfy.sd, comfy.utils
    sd = comfy.utils.load_torch_file(COMPONENTS["vae"])
    vae = comfy.sd.VAE(sd=sd)
    print("  VAE loaded:", type(vae).__name__, "| latent_channels:", getattr(vae, "latent_channels", "?"))
    fs = vae.first_stage_model
    fs.eval()
    # probe decode dims via a dummy latent
    lc = getattr(vae, "latent_channels", 16)
    dummy = torch.zeros(1, lc, 112, 144)  # 896/8, 1152/8 ~ guess
    print("  attempting decoder onnx export with dummy", tuple(dummy.shape))
    class Dec(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, z): return s.m.decode(z)
    torch.onnx.export(Dec(fs), dummy, ROOT + "/onnx_vae_decoder.onnx",
                      input_names=["latent"], output_names=["image"],
                      dynamic_axes={"latent": {2: "h", 3: "w"}}, opset_version=17)
    print("  VAE decoder ONNX export OK -> onnx_vae_decoder.onnx")
except Exception:
    print("  VAE export failed (expected, will fix when back):\n" + traceback.format_exc())

# Recon: load DiT via ComfyUI UNETLoader to capture model class/config for later export.
print("\n==== DiT STRUCTURE RECON ====")
try:
    import comfy.sd, comfy.utils
    m = comfy.sd.load_diffusion_model(COMPONENTS["dit"])
    print("  DiT model object:", type(m.model).__name__)
    print("  model_config:", type(getattr(m.model, "model_config", None)).__name__)
    inner = getattr(m.model, "diffusion_model", None)
    print("  diffusion_model class:", type(inner).__name__ if inner is not None else None)
except Exception:
    print("  DiT recon failed:\n" + traceback.format_exc())

print("\n==== EXPORT RECON DONE ====")
