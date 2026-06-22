#!/usr/bin/env python
# Unattended end-to-end proof: real ComfyUI pipeline with DiT.forward swapped to ONNX.
# Self-contained: (R) resolution-generalization test -> (G) generate -> (C) compare -> STATUS_E2E.md
import sys, os, time, traceback, json, numpy as np
ROOT="D:/AnimaPort"; CU=ROOT+"/ComfyUI"
LOG=open(ROOT+"/overnight.log","w",encoding="utf-8")
def p(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); LOG.write(s+"\n"); LOG.flush()
STATUS={"started":time.strftime("%Y-%m-%d %H:%M:%S"),"stages":{}}
def save_status():
    open(ROOT+"/STATUS_E2E.md","w",encoding="utf-8").write(
        "# Anima E2E (ONNX-injected pipeline) — overnight run\n\n```\n"+json.dumps(STATUS,indent=2)+"\n```\n")
def fail(stage,e):
    STATUS["stages"][stage]={"ok":False,"error":str(e)[:400]}; save_status()
    p(f"[{stage}] FAILED:", str(e)[:300]); p(traceback.format_exc())

os.chdir(CU); sys.path.insert(0, CU); sys.argv=["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu=True
import torch
import onnxruntime as ort

p("=== loading onnx DiT session ===")
sess=ort.InferenceSession(ROOT+"/onnx/dit.onnx", providers=["CPUExecutionProvider"])
IN=[i.name for i in sess.get_inputs()]
def onnx_dit(x,t,c):
    return sess.run(None,{IN[0]:x.astype(np.float32),IN[1]:t.astype(np.float32),IN[2]:c.astype(np.float32)})[0]

# ---------- load model patcher (single DiT load, reused) ----------
p("=== loading DiT (kiwimixAnima) ===")
import comfy.sd
DITPATH=CU+"/models/diffusion_models/kiwimixAnima_v1.safetensors"
mp=comfy.sd.load_diffusion_model(DITPATH)
dit=mp.model.diffusion_model; dit.eval()
for m in dit.modules():
    if hasattr(m,"comfy_cast_weights"): m.comfy_cast_weights=False
dit.float()

# ---------- STAGE R: resolution generalization (onnx vs native torch dit) ----------
p("=== STAGE R: resolution test ===")
orig_forward=dit.forward
def torch_ref(H,W,tok):
    torch.manual_seed(0)
    x=torch.randn(1,16,1,H,W); t=torch.ones(1,1); c=torch.randn(1,tok,1024)
    with torch.no_grad(): r=orig_forward(x,t,c).numpy()
    try: o=onnx_dit(x.numpy(),t.numpy(),c.numpy())
    except Exception as e: return ("RUNERR",str(e)[:160],None)
    if o.shape!=r.shape: return ("SHAPE",f"onnx{o.shape} vs torch{r.shape}",None)
    d=float(np.abs(o-r).max()); return ("OK",d,d<1e-2)
res={}
for tag,(H,W,tok) in {"32x32":(32,32,512),"144x112":(144,112,512)}.items():
    st,info,ok=torch_ref(H,W,tok); res[tag]={"status":st,"info":info,"match":ok}
    p(f"  {tag}: {st} {info} match={ok}")
STATUS["stages"]["R_res_test"]={"ok":True,"detail":res}; save_status()
FULL_OK = res.get("144x112",{}).get("match") is True
p("  => full-res onnx usable:", FULL_OK)

# ---------- swap DiT.forward -> ONNX ----------
def shim(x, timesteps, context=None, *a, **kw):
    t=timesteps
    if t.ndim==1: t=t.unsqueeze(1)
    out=onnx_dit(x.detach().cpu().float().numpy(), t.detach().cpu().float().numpy(),
                 context.detach().cpu().float().numpy())
    return torch.from_numpy(out).to(x.device).to(x.dtype)
dit.forward=shim
p("=== DiT.forward swapped to ONNX ===")

# ---------- build the rest of the pipeline via real ComfyUI nodes ----------
try:
    import nodes
    from comfy_extras import nodes_model_advanced as nma
    wf=json.load(open(ROOT+"/anima_comparison.json",encoding="utf-8"))
    nd={n["id"]:n for n in wf["nodes"]}
    POS=nd[4]["widgets_values"][0]; NEG=nd[3]["widgets_values"][0]
    p("POS:",POS[:70]); p("NEG:",NEG[:70])

    model=nma.ModelSamplingAuraFlow().patch(mp,3.0)[0]
    clip=nodes.CLIPLoader().load_clip("qwen_3_06b_base.safetensors","stable_diffusion")[0]
    pos=nodes.CLIPTextEncode().encode(clip,POS)[0]
    neg=nodes.CLIPTextEncode().encode(clip,NEG)[0]
    vae=nodes.VAELoader().load_vae("qwen_image_vae.safetensors")[0]
    STATUS["stages"]["G_setup"]={"ok":True}; save_status()
except Exception as e:
    fail("G_setup",e); LOG.close(); sys.exit(1)

def generate(W,H,steps,tag):
    p(f"=== GENERATE {tag}: {W}x{H} steps={steps} (seed 807882066116956) ===")
    t0=time.time()
    lat=nodes.EmptyLatentImage().generate(W,H,1)[0]
    samples=nodes.KSampler().sample(model,807882066116956,steps,4.0,"er_sde","simple",pos,neg,lat,1.0)[0]
    img=nodes.VAEDecode().decode(vae,samples)[0]
    arr=(img[0].detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(arr).save(ROOT+f"/onnx_e2e_{tag}.png")
    except Exception as e: p("  PIL save warn:",e)
    dt=time.time()-t0
    p(f"  done {tag} in {dt:.0f}s -> onnx_e2e_{tag}.png  mean={arr.mean():.1f}")
    return {"ok":True,"sec":round(dt),"file":f"onnx_e2e_{tag}.png","img_mean":float(arr.mean())}

# Always do the safe 256x256 (latent 32x32 = trace res) proof first.
try:
    STATUS["stages"]["G_256"]=generate(256,256,30,"256"); save_status()
except Exception as e:
    fail("G_256",e)

# If onnx generalizes, also do the real full-res baseline reproduction.
if FULL_OK:
    try:
        STATUS["stages"]["G_full"]=generate(896,1152,30,"full"); save_status()
        # compare to existing baseline png
        try:
            from PIL import Image
            base=np.asarray(Image.open(ROOT+"/ComfyUI/output/anima_baseline_00001_.png").convert("RGB")).astype(np.float32)
            gen=np.asarray(Image.open(ROOT+"/onnx_e2e_full.png").convert("RGB")).astype(np.float32)
            if base.shape==gen.shape:
                mae=float(np.abs(base-gen).mean())
                STATUS["stages"]["C_compare"]={"ok":True,"mae_vs_baseline":mae,"note":"0=identical,<5 very close"}
            else:
                STATUS["stages"]["C_compare"]={"ok":True,"note":f"shape diff base{base.shape} gen{gen.shape}"}
            save_status()
        except Exception as e: fail("C_compare",e)
    except Exception as e:
        fail("G_full",e)
else:
    STATUS["stages"]["G_full"]={"ok":False,"note":"skipped: onnx DiT baked to 32x32 trace res; full-res needs dynamic re-export of rope/pos-emb"}
    save_status()

STATUS["finished"]=time.strftime("%Y-%m-%d %H:%M:%S")
save_status()
p("=== ALL DONE ===")
LOG.close()
