#!/usr/bin/env bash
# Anima port - unattended setup: ComfyUI deps + component downloads
set -u
ROOT=/d/AnimaPort
CU=$ROOT/ComfyUI
LOG=$ROOT/setup.log
exec > >(tee -a "$LOG") 2>&1
echo "==== SETUP START $(date) ===="

cd "$CU" || { echo "no ComfyUI dir"; exit 1; }

# 1) venv
if [ ! -f "$CU/venv/Scripts/python.exe" ]; then
  echo "[venv] creating"
  py -3.11 -m venv venv
fi
PY="$CU/venv/Scripts/python.exe"
"$PY" -m pip install --upgrade pip wheel

# 2) torch-directml first (pins torch 2.4.1+cpu for AMD RX580)
echo "[pip] torch-directml"
"$PY" -m pip install torch-directml
echo "[pip] matching torchvision/torchaudio/torchsde"
"$PY" -m pip install torchvision==0.19.1 torchaudio==2.4.1 torchsde

# 3) ComfyUI requirements minus torch lines (avoid clobbering pinned torch)
grep -viE '^(torch|torchvision|torchaudio|torchsde)([ =<>!~]|$)' requirements.txt > req_notorch.txt
echo "[pip] ComfyUI requirements (no torch)"
"$PY" -m pip install -r req_notorch.txt

# 4) onnx tooling for later export
"$PY" -m pip install onnx onnxruntime onnxscript || true

# 5) model components
mkdir -p models/diffusion_models models/text_encoders models/vae
BASE=https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files
echo "[dl] qwen_3_06b_base text encoder"
curl -fL -C - -o models/text_encoders/qwen_3_06b_base.safetensors \
  "$BASE/text_encoders/qwen_3_06b_base.safetensors"
echo "[dl] qwen_image_vae"
curl -fL -C - -o models/vae/qwen_image_vae.safetensors \
  "$BASE/vae/qwen_image_vae.safetensors"

# 6) DiT (use the user's merged checkpoint already on disk)
SRC="/c/Users/ksh/OneDrive/Desktop/kiwimixAnima_v1.safetensors"
if [ -f "$SRC" ] && [ ! -f models/diffusion_models/kiwimixAnima_v1.safetensors ]; then
  echo "[cp] kiwimixAnima -> diffusion_models"
  cp "$SRC" models/diffusion_models/
fi

echo "==== SETUP DONE $(date) ===="
echo "FILES:"; ls -la models/text_encoders models/vae models/diffusion_models
