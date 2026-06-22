#!/usr/bin/env python
# VAE decoder export attempt #2: modern TorchDynamo-based ONNX exporter
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
for mod in fsm.modules():
    if hasattr(mod, "comfy_cast_weights"):
        mod.comfy_cast_weights = False
fsm.float()

z = torch.zeros(1, 16, 1, 64, 64, dtype=torch.float32)
class Dec(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m.decode(x)
dec = Dec(fsm).eval()
with torch.no_grad():
    print("decode OK:", tuple(dec(z).shape))

print(">>> trying torch.onnx.dynamo_export ...")
try:
    with torch.no_grad():
        onnx_program = torch.onnx.dynamo_export(dec, z)
    onnx_program.save(ROOT + "/onnx/vae_decoder.onnx")
    print("DYNAMO EXPORT OK ->", os.path.getsize(ROOT + "/onnx/vae_decoder.onnx")/1e6, "MB")
except Exception:
    print("DYNAMO EXPORT FAILED:")
    tb = traceback.format_exc()
    # print only concise tail (avoid giant graph dumps)
    print("\n".join(tb.splitlines()[-25:]))
