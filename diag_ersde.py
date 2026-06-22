#!/usr/bin/env python
# Isolate the er_sde formula: run comfy.sample_er_sde vs pure sample_er_sde with the SAME
# deterministic fake denoise + same noise. Any diff => bug in pure_sampler's er_sde port.
import sys, os, numpy as np
ROOT = "D:/AnimaPort"; CU = ROOT + "/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv = ["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu = True
import torch
import comfy.k_diffusion.sampling as kss
import comfy.sd
from comfy_extras import nodes_model_advanced as nma
sys.path.insert(0, ROOT)
import pure_sampler as ps

mp = comfy.sd.load_diffusion_model(CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors")
model = nma.ModelSamplingAuraFlow().patch_aura(mp, 3.0)[0]
ms = model.get_model_object("model_sampling")

shape = (1, 16, 1, 32, 32); seed = 807882066116956
sig = ps.simple_scheduler(30)
x0_np = ps.noise_scaling(float(sig[0]), ps.torch_initial_noise(shape, seed), np.zeros(shape))

def fake(x_np, sigma):
    return x_np * (1.0 - sigma * 0.05) + 0.01  # arbitrary deterministic denoise

# ---- pure ----
ns_pure = ps.torch_step_noise_sampler(shape, seed)
out_pure = ps.sample_er_sde(lambda x, s: fake(x, s), sig.copy(), x0_np.copy(), ns_pure)

# ---- comfy ----
class MP:
    def get_model_object(self, n): return ms
class IM:
    model_patcher = MP()
class Model:
    inner_model = IM()
    def __call__(self, x, sigma, **kw):
        s = float(sigma.reshape(-1)[0])
        return torch.from_numpy(fake(x.cpu().numpy().astype(np.float64), s)).to(x)

ns_comfy = kss.default_noise_sampler(torch.zeros(shape), seed=seed)
x0_t = torch.from_numpy(x0_np)
sig_t = torch.from_numpy(sig.astype(np.float64))
out_comfy = kss.sample_er_sde(Model(), x0_t, sig_t, extra_args={"seed": seed}, noise_sampler=ns_comfy, disable=True).cpu().numpy()

d = np.abs(out_comfy - out_pure)
print(f"er_sde formula  MAE={d.mean():.3e}  MAX={d.max():.3e}  -> {'FORMULA OK' if d.max()<1e-4 else 'FORMULA BUG'}", flush=True)
