#!/usr/bin/env python
"""
pure_sampler.py — ComfyUI-free numpy reference sampler for Anima (Cosmos-Predict2, flow-matching).

This re-implements, in pure numpy, exactly what ComfyUI does for the Anima workflow:
  ModelSamplingAuraFlow(shift=3.0)  ->  ModelSamplingDiscreteFlow + CONST, multiplier=1.0
  KSampler(steps=30, cfg=4.0, sampler="er_sde", scheduler="simple")

Why: ComfyUI is GPL-3.0 and cannot ship in the Android app. This module depends only on
numpy + onnxruntime (DiT, VAE). It is the direct source for the C++/Kotlin native port.

The math is copied 1:1 from comfy/model_sampling.py, comfy/samplers.py,
comfy/k_diffusion/sampling.py (sample_er_sde) for the CONST / flow-matching branch.

Noise (initial + er_sde per-step) is generated with torch's RNG ONLY so we can reproduce
ComfyUI's seed bit-for-bit during PC verification. On-device this is replaced by any
gaussian source; seed-identity is a verification convenience, not a correctness requirement.
"""
import numpy as np

# Match ComfyUI's compute precision (torch float32 on CPU). Using float64 diverges over the
# stochastic er_sde trajectory (chaotic amplification), so we mirror float32 exactly.
DTYPE = np.float32

# ----------------------------------------------------------------------------- model sampling
SHIFT = 3.0
MULTIPLIER = 1.0
TIMESTEPS = 1000

def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)

def sigma_of_t(t):
    # ModelSamplingDiscreteFlow.sigma: time_snr_shift(shift, t/multiplier)
    return time_snr_shift(SHIFT, t / MULTIPLIER)

# precomputed sigma buffer (1000 entries), monotonically increasing
_T = (np.arange(1, TIMESTEPS + 1, dtype=np.float64) / TIMESTEPS) * MULTIPLIER
SIGMAS_BUF = sigma_of_t(_T)          # SIGMAS_BUF[-1] == sigma_max == 1.0
SIGMA_MAX = float(SIGMAS_BUF[-1])
SIGMA_MIN = float(SIGMAS_BUF[0])

def timestep_of_sigma(sigma):
    # ModelSamplingDiscreteFlow.timestep: sigma * multiplier  (== sigma here)
    return sigma * MULTIPLIER

def percent_to_sigma(percent):
    if percent <= 0.0:
        return 1.0
    if percent >= 1.0:
        return 0.0
    return time_snr_shift(SHIFT, 1.0 - percent)

# ----------------------------------------------------------------------------- scheduler
def simple_scheduler(steps):
    """comfy.samplers.simple_scheduler for our sigma buffer. Returns steps+1 sigmas ending in 0."""
    n = len(SIGMAS_BUF)
    ss = n / steps
    sigs = [float(SIGMAS_BUF[-(1 + int(x * ss))]) for x in range(steps)]
    sigs.append(0.0)
    return np.array(sigs, dtype=DTYPE)

# ----------------------------------------------------------------------------- CONST input/output
def calculate_input(sigma, x):
    # CONST.calculate_input: identity
    return x

def calculate_denoised(sigma, model_output, x):
    # CONST.calculate_denoised: x - v * sigma
    return x - model_output * sigma

def noise_scaling(sigma, noise, latent_image, noise_scale=1.0):
    # CONST.noise_scaling: sigma*(s*noise) + (1-sigma)*latent_image
    return sigma * (noise_scale * noise) + (1.0 - sigma) * latent_image

def inverse_noise_scaling(sigma, latent):
    # CONST.inverse_noise_scaling: latent / (1 - sigma); sigma_last == 0 -> no-op
    if sigma == 0.0:
        return latent
    return latent / (1.0 - sigma)

# ----------------------------------------------------------------------------- er_sde helpers (CONST branch)
def sigma_to_half_log_snr_const(sigma):
    # CONST: sigma.logit().neg() = -log(sigma/(1-sigma)) = log((1-sigma)/sigma)
    return -(np.log(sigma) - np.log1p(-sigma))

def offset_first_sigma_for_snr(sigmas, percent_offset=1e-4):
    # CONST: if sigmas[0] >= 1, replace with percent_to_sigma(1e-4) to keep logit finite
    sigmas = sigmas.copy()
    if len(sigmas) > 1 and sigmas[0] >= 1.0:
        sigmas[0] = percent_to_sigma(percent_offset)
    return sigmas

def default_er_sde_noise_scaler(x):
    return x * (np.exp(x ** 0.3) + 10.0)

# ----------------------------------------------------------------------------- torch RNG (verification only)
def torch_initial_noise(shape, seed):
    """Reproduce comfy.sample.prepare_noise: torch.manual_seed(seed); randn(shape) on CPU."""
    import torch
    g = torch.manual_seed(seed)
    return torch.randn(list(shape), dtype=torch.float32, generator=g, device="cpu").numpy().astype(DTYPE)

def torch_step_noise_sampler(shape, seed):
    """Reproduce comfy default_noise_sampler on CPU: seed+1, fresh randn each call."""
    import torch
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)  # CPU path adds 1
    def sampler(_s_cur, _s_next):
        return torch.randn(list(shape), dtype=torch.float32, generator=g, device="cpu").numpy().astype(DTYPE)
    return sampler

# ----------------------------------------------------------------------------- main loop
def sample_er_sde(denoise_fn, sigmas, x, noise_sampler, s_noise=1.0, max_stage=3, callback=None):
    """
    Direct numpy port of comfy.k_diffusion.sampling.sample_er_sde for the CONST branch.
      denoise_fn(x, sigma) -> denoised (x0 estimate)
      sigmas: np.array length steps+1 (ends with 0)
      x: initial latent (already noise-scaled)
      noise_sampler(sigma_cur, sigma_next) -> gaussian like x
    """
    num_integration_points = 200.0
    point_indice = np.arange(0, num_integration_points, dtype=DTYPE)

    sigmas = offset_first_sigma_for_snr(sigmas)
    half_log_snrs = sigma_to_half_log_snr_const(sigmas)
    er_lambdas = np.exp(-half_log_snrs)  # sigma_t / alpha_t

    old_denoised = None
    old_denoised_d = None
    ns = default_er_sde_noise_scaler

    for i in range(len(sigmas) - 1):
        denoised = denoise_fn(x, float(sigmas[i]))
        if callback is not None:
            callback(i, x, denoised)
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = ns(er_lambda_t) / ns(er_lambda_s)

            # Stage 1 Euler
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = ns(lambda_pos)

                # Stage 2
                s = np.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * ns(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u = np.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * ns(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                step_n = noise_sampler(float(sigmas[i]), float(sigmas[i + 1]))
                coef = np.sqrt(np.nan_to_num(er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2, nan=0.0))
                x = x + alpha_t * step_n * s_noise * coef
        old_denoised = denoised
    return x

# ----------------------------------------------------------------------------- end-to-end driver
def build_denoise_fn(dit_run, ctx_pos, ctx_neg, cfg=4.0):
    """
    dit_run(x[B,16,1,h,w] f32, t[B,1] f32, c[B,T,1024] f32) -> v prediction (numpy)
    Returns denoise_fn(x_np_f64, sigma_float) -> denoised f64.
    """
    def denoise_fn(x, sigma):
        xc = calculate_input(sigma, x).astype(np.float32)
        t = np.full((x.shape[0], 1), timestep_of_sigma(sigma), dtype=np.float32)
        v_pos = dit_run(xc, t, ctx_pos.astype(np.float32)).astype(DTYPE)
        v_neg = dit_run(xc, t, ctx_neg.astype(np.float32)).astype(DTYPE)
        v = v_neg + (v_pos - v_neg) * cfg
        return calculate_denoised(sigma, v, x)
    return denoise_fn

def generate_latent(dit_run, ctx_pos, ctx_neg, latent_shape, seed, steps=30, cfg=4.0, callback=None):
    """
    Full sampling: empty latent -> noise -> er_sde -> final latent (pre-VAE).
    latent_shape: (B,16,1,H//8,W//8)
    """
    sigmas = simple_scheduler(steps)
    noise = torch_initial_noise(latent_shape, seed)
    latent_image = np.zeros(latent_shape, dtype=DTYPE)
    # KSampler max_denoise=True at first step; CONST ignores max_denoise flag
    x = noise_scaling(float(sigmas[0]), noise, latent_image)
    denoise_fn = build_denoise_fn(dit_run, ctx_pos, ctx_neg, cfg)
    noise_sampler = torch_step_noise_sampler(latent_shape, seed)
    x = sample_er_sde(denoise_fn, sigmas, x, noise_sampler, callback=callback)
    x = inverse_noise_scaling(float(sigmas[-1]), x)
    return x

if __name__ == "__main__":
    # quick self-check of the schedule (no model needed)
    print("sigma_max", SIGMA_MAX, "sigma_min", SIGMA_MIN)
    s = simple_scheduler(30)
    print("scheduler len", len(s), "first", s[0], "last", s[-1])
    print("first 5:", s[:5])
