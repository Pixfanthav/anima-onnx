import torch
cap = torch.load("D:/AnimaPort/rope_capture.pt")
q, k, rope, oq, ok = cap['q'], cap['k'], cap['rope'], cap['oq'], cap['ok']
print("q", tuple(q.shape), "rope", tuple(rope.shape), "oq", tuple(oq.shape))

def implA(t, freqs):
    # consecutive-pair reshape (matches ck traceback: reshape ..., -1, 1, 2)
    t_ = t.reshape(*t.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    return out.reshape(t.shape)

def implB(t, freqs):
    # split-half reshape
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
    out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    return out.movedim(-1, -2).reshape(t.shape)

for name, fn in [("A_consecutive", implA), ("B_splithalf", implB)]:
    try:
        dq = (fn(q, rope) - oq).abs().max().item()
        dk = (fn(k, rope) - ok).abs().max().item()
        print(f"{name:16s} q_diff={dq:.3e} k_diff={dk:.3e}  {'<== MATCH' if max(dq,dk)<1e-4 else ''}")
    except Exception as e:
        print(f"{name:16s} ERROR: {e}")
