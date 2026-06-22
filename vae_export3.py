#!/usr/bin/env python
# VAE export #3: monkeypatch CausalConv3d to an export-clean (static) causal conv, then legacy ONNX export.
import sys, os, traceback
ROOT = "D:/AnimaPort"; CU = ROOT + "/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU)
sys.argv = ["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu = True
import torch, torch.nn.functional as F
import comfy.sd, comfy.utils
import comfy.ldm.wan.vae as wanvae

# --- export-clean replacement for CausalConv3d.forward (no autopad kwarg, static slicing) ---
def clean_causal_forward(self, x, cache_x=None, cache_list=None, cache_idx=None):
    T = x.shape[2]
    if T == 1 and self._padding > 0:
        # original fast path: use last temporal slice of kernel (static for T=1)
        w = self.weight[:, :, -1:, :, :]
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
    if self._padding > 0:
        x = F.pad(x, (0, 0, 0, 0, self._padding, 0))  # causal zero-pad on front of time dim
    return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

wanvae.CausalConv3d.forward = clean_causal_forward
print("monkeypatched CausalConv3d.forward")

VAE = CU + "/models/vae/qwen_image_vae.safetensors"
sd = comfy.utils.load_torch_file(VAE)
vae = comfy.sd.VAE(sd=sd)
fsm = vae.first_stage_model; fsm.eval()
n_up = 0
for mod in fsm.modules():
    if hasattr(mod, "comfy_cast_weights"):
        mod.comfy_cast_weights = False
    if isinstance(mod, torch.nn.Upsample) and getattr(mod, "mode", "") == "nearest-exact":
        mod.mode = "nearest"  # ONNX-exportable; identical for integer 2x scale
        n_up += 1
fsm.float()
print(f"patched {n_up} nearest-exact upsamples -> nearest")

z = torch.zeros(1, 16, 1, 64, 64, dtype=torch.float32)
class Dec(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m.decode(x)
dec = Dec(fsm).eval()
with torch.no_grad():
    print("decode OK:", tuple(dec(z).shape))

try:
    torch.onnx.export(
        dec, z, ROOT + "/onnx/vae_decoder.onnx",
        input_names=["latent"], output_names=["image"],
        dynamic_axes={"latent": {0: "b", 3: "h", 4: "w"}, "image": {0: "b", 3: "H", 4: "W"}},
        opset_version=17)
    print("LEGACY EXPORT OK ->", os.path.getsize(ROOT + "/onnx/vae_decoder.onnx")/1e6, "MB")
except Exception:
    tb = traceback.format_exc().splitlines()
    print("EXPORT FAILED tail:\n" + "\n".join(l[:200] for l in tb[-15:]))
