#!/usr/bin/env bash
# Master unattended orchestrator: wait for setup -> baseline gen -> export recon -> STATUS.md
set -u
ROOT=/d/AnimaPort
CU=$ROOT/ComfyUI
PY="$CU/venv/Scripts/python.exe"
LOG=$ROOT/run_all.log
STATUS=$ROOT/STATUS.md
exec > >(tee -a "$LOG") 2>&1
echo "==== RUN_ALL START $(date) ===="

baseline_result="not reached"
export_result="not reached"

# 1) wait for setup.sh (deps install + downloads) up to ~120 min
echo "[wait] waiting for setup.sh completion..."
ok=0
for i in $(seq 1 240); do
  if grep -q "SETUP DONE" "$ROOT/setup.log" 2>/dev/null \
     && [ -f "$CU/models/text_encoders/qwen_3_06b_base.safetensors" ] \
     && [ -f "$CU/models/vae/qwen_image_vae.safetensors" ] \
     && [ -f "$CU/models/diffusion_models/kiwimixAnima_v1.safetensors" ]; then
     ok=1; echo "[wait] setup complete after ~$((i*30))s"; break
  fi
  sleep 30
done
if [ "$ok" != "1" ]; then echo "[wait] setup did not finish in time; aborting"; fi

# sanity: torch import
"$PY" -c "import torch,torchvision;print('[env] torch',torch.__version__)" || echo "[env] torch import FAILED"

if [ "$ok" = "1" ]; then
  # 2) start ComfyUI server (CPU, headless)
  echo "[server] starting ComfyUI --cpu on :8188"
  ( cd "$CU" && "$PY" main.py --cpu --port 8188 ) > "$ROOT/comfy_server.log" 2>&1 &
  SVPID=$!
  echo "[server] pid=$SVPID"
  up=0
  for i in $(seq 1 90); do
    if curl -fs http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then up=1; echo "[server] ready"; break; fi
    sleep 5
  done
  if [ "$up" = "1" ]; then
    # 3) baseline generation
    echo "[baseline] submitting prompt..."
    if "$PY" "$ROOT/gen_baseline.py"; then baseline_result="SUCCESS (see ComfyUI/output/anima_baseline*.png)"; else baseline_result="FAILED rc=$? (see run_all.log)"; fi
  else
    baseline_result="server never came up (see comfy_server.log)"
  fi
  # stop server before export recon (free RAM)
  kill $SVPID 2>/dev/null; sleep 5

  # 4) export recon (best effort)
  echo "[export] recon..."
  if "$PY" "$ROOT/export_all.py"; then export_result="recon ran (see run_all.log; onnx_vae_decoder.onnx if succeeded)"; else export_result="recon script error rc=$?"; fi
fi

# 5) write STATUS.md
{
  echo "# Anima Port — Unattended Run Status"
  echo ""
  echo "Generated: $(date)"
  echo ""
  echo "## Results"
  echo "- Setup (deps+downloads): $([ "$ok" = "1" ] && echo OK || echo INCOMPLETE)"
  echo "- Baseline generation: $baseline_result"
  echo "- Export recon: $export_result"
  echo ""
  echo "## Artifacts"
  echo '```'
  ls -la "$CU/output" 2>/dev/null | tail -n 15
  echo "--- onnx ---"
  ls -la "$ROOT"/*.onnx 2>/dev/null
  echo '```'
  echo ""
  echo "## Logs"
  echo "- D:/AnimaPort/run_all.log (this run)"
  echo "- D:/AnimaPort/comfy_server.log (server)"
  echo "- D:/AnimaPort/setup.log (install+downloads)"
} > "$STATUS"

echo "==== RUN_ALL DONE $(date) ===="
echo "STATUS written to $STATUS"
