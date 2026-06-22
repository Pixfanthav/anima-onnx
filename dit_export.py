#!/usr/bin/env python
# DiT (Anima / MiniTrainDIT) export attempt — auto-detect dims, patch comfy ops, try ONNX export.
import sys, os, traceback
ROOT = "D:/AnimaPort"; CU = ROOT + "/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU)
sys.argv = ["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu = True
import torch
import comfy.sd, comfy.utils

# --- ONNX-friendly RMSNorm: decompose aten::rms_norm into basic ops ---
def _rms_norm_decomposed(x, normalized_shape, weight=None, eps=None):
    if eps is None:
        eps = 1e-6
    dims = tuple(range(-len(normalized_shape), 0))
    var = x.pow(2).mean(dim=dims, keepdim=True)
    xn = x * torch.rsqrt(var + eps)
    if weight is not None:
        xn = xn * weight
    return xn
torch.nn.functional.rms_norm = _rms_norm_decomposed
print("patched torch.nn.functional.rms_norm -> decomposed")

# --- ONNX-friendly RoPE: verified (diff=0) pure-python replacement for comfy_kitchen op ---
import comfy.quant_ops as _qo
def _apply_rope_split_half(q, k, rope):
    def f(t, freqs):
        B, S, H, D = t.shape
        # positive-index transpose (ONNX Transpose perm must be non-negative)
        t_ = t.reshape(B, S, H, 2, D // 2).transpose(3, 4).unsqueeze(4)  # [B,S,H,D/2,1,2]
        out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]      # [B,S,H,D/2,2]
        return out.transpose(3, 4).reshape(B, S, H, D)
    return f(q, rope), f(k, rope)
_qo.ck.apply_rope_split_half = _apply_rope_split_half
print("patched ck.apply_rope_split_half -> pure python (verified)")

DIT = CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors"
mp = comfy.sd.load_diffusion_model(DIT)
dit = mp.model.diffusion_model
dit.eval()

# disable comfy runtime weight casting (export-clean), force fp32
npatch = 0
for mod in dit.modules():
    if hasattr(mod, "comfy_cast_weights"):
        mod.comfy_cast_weights = False; npatch += 1
dit.float()
print(f"patched comfy_cast_weights on {npatch} modules")

# --- auto-detect dims ---
ic = getattr(dit, "in_channels", "?")
mc = getattr(dit, "model_channels", "?")
ps = getattr(dit, "patch_spatial", "?")
pt = getattr(dit, "patch_temporal", "?")
print("in_channels:", ic, "model_channels:", mc, "patch_spatial:", ps, "patch_temporal:", pt)
# find context (crossattn) dim by scanning for a cross-attn k/v projection
ctx_dim = None
for name, p in dit.named_parameters():
    nl = name.lower()
    if "cross" in nl and (("k_proj" in nl and "weight" in nl) or (".to_k." in nl) or ("kv" in nl and "weight" in nl)):
        ctx_dim = p.shape[1]
        print("ctx dim from", name, "->", tuple(p.shape))
        break
if ctx_dim is None:
    ctx_dim = 1024
    print("ctx dim not found, defaulting 1024")

# --- build dummy inputs (small latent for speed) ---
in_lat = ic if isinstance(ic, int) else 16
x = torch.zeros(1, in_lat, 1, 32, 32, dtype=torch.float32)     # [B, C, T, H, W]
timesteps = torch.ones(1, 1, dtype=torch.float32)              # [B, T]
context = torch.zeros(1, 512, ctx_dim, dtype=torch.float32)    # [B, tokens, ctx_dim]
print("dummy shapes:", tuple(x.shape), tuple(timesteps.shape), tuple(context.shape))

class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x, t, c): return s.m(x, t, c)
w = Wrap(dit).eval()

print(">>> trial forward ...")
try:
    with torch.no_grad():
        y = w(x, timesteps, context)
    print("FORWARD OK, out:", tuple(y.shape) if hasattr(y, "shape") else type(y))
except Exception:
    print("FORWARD FAILED:\n" + "\n".join(traceback.format_exc().splitlines()[-20:]))
    sys.exit(1)

print(">>> ONNX export ...")
try:
    torch.onnx.export(
        w, (x, timesteps, context), ROOT + "/onnx/dit.onnx",
        input_names=["latent", "timesteps", "context"], output_names=["noise_pred"],
        dynamic_axes={"latent": {0: "b", 3: "h", 4: "w"}, "context": {0: "b", 1: "tok"}},
        opset_version=17)
    print("DiT ONNX OK ->", os.path.getsize(ROOT + "/onnx/dit.onnx")/1e6, "MB")
except Exception:
    print("EXPORT FAILED tail:\n" + "\n".join(l[:220] for l in traceback.format_exc().splitlines()[-18:]))
