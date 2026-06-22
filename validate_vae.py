#!/usr/bin/env python
# Validate vae_decoder.onnx numerically matches the PyTorch Qwen-Image VAE decode.
import sys, os, numpy as np
ROOT="D:/AnimaPort"; CU=ROOT+"/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv=["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu=True
import torch, torch.nn.functional as F
import comfy.sd, comfy.utils
import comfy.ldm.wan.vae as wanvae

def clean_causal_forward(self, x, cache_x=None, cache_list=None, cache_idx=None):
    T=x.shape[2]
    if T==1 and self._padding>0:
        w=self.weight[:,:,-1:,:,:]
        return F.conv3d(x,w,self.bias,self.stride,self.padding,self.dilation,self.groups)
    if self._padding>0:
        x=F.pad(x,(0,0,0,0,self._padding,0))
    return F.conv3d(x,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)
wanvae.CausalConv3d.forward=clean_causal_forward

VAE=CU+"/models/vae/qwen_image_vae.safetensors"
sd=comfy.utils.load_torch_file(VAE)
vae=comfy.sd.VAE(sd=sd)
fsm=vae.first_stage_model; fsm.eval()
for mod in fsm.modules():
    if hasattr(mod,"comfy_cast_weights"): mod.comfy_cast_weights=False
    if isinstance(mod,torch.nn.Upsample) and getattr(mod,"mode","")=="nearest-exact": mod.mode="nearest"
fsm.float()

torch.manual_seed(42)
z=torch.randn(1,16,1,64,64,dtype=torch.float32)
with torch.no_grad():
    ref=fsm.decode(z).numpy()
print("pytorch out:", ref.shape, "mean", float(ref.mean()), "std", float(ref.std()))

import onnxruntime as ort
sess=ort.InferenceSession(ROOT+"/onnx/vae_decoder.onnx", providers=["CPUExecutionProvider"])
names=[i.name for i in sess.get_inputs()]
print("onnx inputs:", names)
out=sess.run(None, {names[0]: z.numpy()})[0]
print("onnx out:", out.shape, "mean", float(out.mean()), "std", float(out.std()))

diff=np.abs(out-ref); rel=diff.max()/(np.abs(ref).max()+1e-8)
print(f"MAX abs diff = {diff.max():.3e} | mean abs diff = {diff.mean():.3e} | rel = {rel:.3e}")
print("VERDICT:", "MATCH (export correct)" if diff.max()<1e-2 else "MISMATCH - investigate")
