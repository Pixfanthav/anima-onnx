#!/usr/bin/env python
# Determine exact convention of comfy_kitchen apply_rope_split_half by numeric comparison.
import sys, os
ROOT="D:/AnimaPort"; CU=ROOT+"/ComfyUI"
os.chdir(CU); sys.path.insert(0, CU); sys.argv=["main.py"]
import comfy.cli_args; comfy.cli_args.args.cpu=True
import torch
import comfy.quant_ops as q
ck = q.ck
print("ck:", ck)

L, H, Dh = 8, 4, 16
half = Dh//2
torch.manual_seed(0)
qd = torch.randn(1, L, H, Dh)
kd = torch.randn(1, L, H, Dh)
rope = torch.randn(L, half, 2, 2)

qo, ko = ck.apply_rope_split_half(qd.clone(), kd.clone(), rope)
print("ck q out:", tuple(qo.shape))

def rope_a_b_c_d(rope):
    a = rope[..., 0, 0][None, :, None, :]
    b = rope[..., 0, 1][None, :, None, :]
    c = rope[..., 1, 0][None, :, None, :]
    d = rope[..., 1, 1][None, :, None, :]
    return a, b, c, d

def split_half(t, rope):
    h = t.shape[-1]//2
    t0, t1 = t[..., :h], t[..., h:]
    a, b, c, d = rope_a_b_c_d(rope)
    return torch.cat([a*t0 + b*t1, c*t0 + d*t1], dim=-1)

def interleaved(t, rope):
    h = t.shape[-1]//2
    t_ = t.reshape(*t.shape[:-1], h, 2)
    t0, t1 = t_[..., 0], t_[..., 1]
    a, b, c, d = rope_a_b_c_d(rope)
    return torch.stack([a*t0 + b*t1, c*t0 + d*t1], dim=-1).reshape(t.shape)

for name, fn in [("split_half", split_half), ("interleaved", interleaved)]:
    try:
        dq = (fn(qd, rope) - qo).abs().max().item()
        dk = (fn(kd, rope) - ko).abs().max().item()
        print(f"{name:12s} q_maxdiff={dq:.3e} k_maxdiff={dk:.3e}  {'<-- MATCH' if max(dq,dk)<1e-4 else ''}")
    except Exception as e:
        print(f"{name:12s} ERROR {e}")
