# Anima ONNX — runtime-independent inference for a Cosmos-Predict2 anime model

Run **Anima** (`kiwimixAnima`, a NVIDIA **Cosmos-Predict2-2B** DiT merge) for text-to-image
**without ComfyUI or PyTorch at inference time** — only `onnxruntime` + `numpy`.

Most on-device / lightweight diffusion stacks are locked to SD1.5 (e.g. Local Dream).
This is, as far as I can tell, the first attempt to take a **Cosmos-Predict2 / Anima**
model out of ComfyUI into a portable ONNX pipeline (target: phone NPU, but the ONNX
pipeline runs anywhere).

> ⚠️ **No model weights here.** Anima is non-commercial licensed; bring your own model.
> Export needs ComfyUI (GPL-3.0) installed separately — only my scripts are in this repo.

## What this is

Anima is **not** a plain single-encoder model. Reverse-engineering the noise bug revealed:

- **Dual text encoder**: Qwen3-0.6B (`qwen3_te`) **+ t5xxl token ids**, fused by a
  **`llm_adapter`** (cross-attention: t5xxl ids as query, qwen hidden as key/value),
  then zero-padded to 512 tokens → that is the real DiT context.
- DiT = `comfy/ldm/anima/model.py` `Anima(MiniTrainDIT)`, AdaLN, RoPE, flow-matching.
- Sampler: `ModelSamplingAuraFlow(shift=3.0)` (CONST / flow-matching) + `er_sde` + `simple`.
- Latent: **Wan21** format (`process_out = raw * std + mean`).

## Pipeline (ComfyUI-free)

```
prompt
  ├─ qwen tokenizer ─► qwen3_te.onnx ─┐
  └─ t5  tokenizer ──────────────────┤
                       llm_adapter.onnx ─► pad512 ─► context [1,512,1024]
empty latent ─► numpy er_sde loop ( dit.onnx, CFG=4, 30 steps ) ─► raw latent
raw latent ─► Wan21 process_out ─► vae_decoder.onnx ─► (img+1)/2 ─► image
```

All glue (er_sde, AuraFlow sigma schedule, CFG, latent format) is reimplemented in pure
numpy in `pure_sampler.py` — this is the reference for a native (C++/Kotlin) port.

## Files

| file | role |
|---|---|
| `dit_export.py` `vae_export.py` `te_export.py` `adapter_export.py` | export 4 ONNX components from ComfyUI |
| `dit_fp16.py` | DiT → fp16 (~3.95GB; embedders/norm/Cast kept fp32) |
| `surgery_5d.py` | wrap 5D MatMuls as reshape→2D→reshape (DirectML/NPU don't do 5D MatMul) |
| `pure_te.py` | ComfyUI-free text encoder (qwen3 + t5 + llm_adapter, onnx) |
| `pure_sampler.py` | ComfyUI-free numpy er_sde sampler |
| **`pure_generate.py`** | **full ComfyUI-free text→image (onnx + numpy only)** |
| `verify_pure.py` | validate pure pipeline vs ComfyUI |

## Usage

```bash
# 1) export ONNX (needs ComfyUI + the model + venv) — one time
python dit_export.py && python vae_export.py && python te_export.py && python adapter_export.py
# 2) generate — ComfyUI not needed
python pure_generate.py 896 1152
```

## Export recipe — key patches

DiT/VAE don't trace to clean ONNX out of the box. The working patches:

- `torch.nn.functional.rms_norm` → decomposed `x * rsqrt(mean(x²)+eps) * w`
- `apply_rope_split_half` → verified pure-python split-half RoPE (diff=0 vs original)
- `comfy_cast_weights=False` + `.float()` on all modules
- VAE: `CausalConv3d` → static causal padding; `nearest-exact` → `nearest`
- external-data models: shape-infer first, then fp16 (else float/float16 type mixes)

## Validation

- ONNX DiT vs PyTorch: MAX abs diff ~7e-6 (per-component MATCH)
- Full onnx pipeline vs native baseline: **image MAE 0.073** (effectively identical)
- Pure numpy sampler vs ComfyUI KSampler: **image MAE 1.82**

## Notes on mobile / GPU

- 5D MatMul (x_embedder + all MLP) is **not supported by DirectML** (and likely needs the
  reshape wrap for QNN/Hexagon too) — see `surgery_5d.py`.
- fp16 overflows the large residual stream → keep norm/embedder ops fp32 (`dit_fp16.py`).
  (CPU onnxruntime accumulates in fp32 and is fine; pure-fp16 backends are not.)
- Phone NPU port (ORT QNN EP) is WIP.

## Licensing

- **This repo's scripts**: MIT (see `LICENSE`). Scripts only — no weights, no ComfyUI.
- **Model weights**: not included. Anima/kiwimixAnima is non-commercial — bring your own.
- **ComfyUI**: GPL-3.0, not included; install separately for the export step. The export
  scripts import/monkeypatch ComfyUI at runtime — see `NOTICE`.
- **NVIDIA Cosmos-Predict2**: subject to NVIDIA's license.
