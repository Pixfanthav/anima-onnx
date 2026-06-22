#!/usr/bin/env python
# Isolate noise reproduction: does pure_sampler's torch RNG match ComfyUI's?
import sys, os, numpy as np
ROOT = "D:/AnimaPort"; CU = ROOT + "/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv = ["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu = True
import torch, comfy.sample
sys.path.insert(0, ROOT)
import pure_sampler as ps

shape = (1, 16, 1, 32, 32); seed = 807882066116956

# 1) initial noise
n_comfy = comfy.sample.prepare_noise(torch.zeros(shape), seed).numpy()
n_pure = ps.torch_initial_noise(shape, seed)
print("initial noise  max|diff| =", float(np.abs(n_comfy - n_pure).max()))

# 2) er_sde per-step noise sampler (comfy default_noise_sampler vs ours)
from comfy.k_diffusion.sampling import default_noise_sampler
x = torch.zeros(shape)
ns_comfy = default_noise_sampler(x, seed=seed)
ns_pure = ps.torch_step_noise_sampler(shape, seed)
for i in range(3):
    a = ns_comfy(None, None).numpy()
    b = ns_pure(0.0, 0.0)
    print(f"step-noise[{i}] max|diff| =", float(np.abs(a - b).max()))

# 3) sigma schedule vs ComfyUI
import comfy.samplers
from comfy_extras import nodes_model_advanced as nma
import comfy.sd
mp = comfy.sd.load_diffusion_model(CU + "/models/diffusion_models/kiwimixAnima_v1.safetensors")
model = nma.ModelSamplingAuraFlow().patch_aura(mp, 3.0)[0]
ms = model.get_model_object("model_sampling")
sig_comfy = comfy.samplers.simple_scheduler(ms, 30).numpy()
sig_pure = ps.simple_scheduler(30)
print("sigma schedule max|diff| =", float(np.abs(sig_comfy - sig_pure).max()))
print("comfy sig[:3]", sig_comfy[:3], "pure", sig_pure[:3])
print("comfy sig[-3:]", sig_comfy[-3:], "pure", sig_pure[-3:])
