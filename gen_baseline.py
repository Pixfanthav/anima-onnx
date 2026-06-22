#!/usr/bin/env python
# Headless Anima baseline generation via ComfyUI /prompt API (server must be running on :8188)
import json, urllib.request, urllib.error, time, sys
ROOT = "D:/AnimaPort"
SERVER = "http://127.0.0.1:8188"

# pull exact prompts from the reference workflow (node 4 = positive, node 3 = negative)
wf = json.load(open(ROOT + "/anima_comparison.json", encoding="utf-8"))
nd = {n["id"]: n for n in wf["nodes"]}
POS = nd[4]["widgets_values"][0]
NEG = nd[3]["widgets_values"][0]
print("[gen] POS prompt:", POS[:80], "...")
print("[gen] NEG prompt:", NEG[:80])

graph = {
  "1":  {"class_type": "UNETLoader",            "inputs": {"unet_name": "kiwimixAnima_v1.safetensors", "weight_dtype": "default"}},
  "2":  {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
  "3":  {"class_type": "CLIPLoader",            "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion"}},
  "4":  {"class_type": "CLIPTextEncode",        "inputs": {"clip": ["3", 0], "text": POS}},
  "5":  {"class_type": "CLIPTextEncode",        "inputs": {"clip": ["3", 0], "text": NEG}},
  "6":  {"class_type": "VAELoader",             "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
  "7":  {"class_type": "EmptyLatentImage",      "inputs": {"width": 896, "height": 1152, "batch_size": 1}},
  "8":  {"class_type": "KSampler",              "inputs": {"model": ["2", 0], "seed": 807882066116956,
            "steps": 30, "cfg": 4.0, "sampler_name": "er_sde", "scheduler": "simple",
            "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0], "denoise": 1.0}},
  "9":  {"class_type": "VAEDecode",             "inputs": {"samples": ["8", 0], "vae": ["6", 0]}},
  "10": {"class_type": "SaveImage",             "inputs": {"images": ["9", 0], "filename_prefix": "anima_baseline"}},
}

def post():
    data = json.dumps({"prompt": graph, "client_id": "animaport"}).encode()
    req = urllib.request.Request(SERVER + "/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        print("[gen] HTTPError submitting prompt:", e.code)
        print(e.read().decode(errors="replace"))
        sys.exit(2)

r = post()
pid = r.get("prompt_id")
print("[gen] prompt_id:", pid, "| node_errors:", json.dumps(r.get("node_errors", {}))[:500])
if not pid:
    print("[gen] no prompt_id -> rejected"); sys.exit(2)

# poll history up to ~2.5h (CPU is slow)
for i in range(900):
    time.sleep(10)
    try:
        h = json.load(urllib.request.urlopen(SERVER + f"/history/{pid}", timeout=30))
    except Exception:
        continue
    if pid in h:
        st = h[pid].get("status", {})
        print("[gen] STATUS:", json.dumps(st))
        outs = h[pid].get("outputs", {})
        print("[gen] OUTPUTS:", json.dumps(outs))
        imgs = []
        for v in outs.values():
            for im in v.get("images", []):
                imgs.append(im.get("filename"))
        print("[gen] SAVED IMAGES:", imgs)
        sys.exit(0 if imgs else 3)
    if i % 6 == 0:
        print(f"[gen] still running... ({i*10}s)")
print("[gen] timed out waiting for completion"); sys.exit(4)
