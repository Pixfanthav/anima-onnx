#!/usr/bin/env python
"""
pure_te.py — ComfyUI-free text encoder for Anima (qwen3 + t5xxl dual + llm_adapter).

Anima conditioning is NOT just qwen3 hidden. The real DiT context is:
  qwen_hidden = qwen3_te.onnx(qwen_ids)                         [1, Lq, 1024]
  adapted     = llm_adapter.onnx(qwen_hidden, t5xxl_ids)        [1, Lt5, 1024]
  context     = pad(adapted * t5xxl_weights, to 512 tokens)     [1, 512, 1024]
(see comfy/ldm/anima/model.py: Anima.preprocess_text_embeds + LLMAdapter)

Tokenizers:
  qwen3  : qwen25_tokenizer (Qwen2 BPE), no start/end, pad 151643, weights 1.0
  t5xxl  : t5_tokenizer (T5TokenizerFast)
Both replaced by tokenizers-cpp / hand-rolled BPE on Android.

PC verification: compare encode(text) against ComfyUI dit.preprocess_text_embeds output (verify_te.py).
"""
import os, numpy as np

ROOT = "D:/AnimaPort"
QWEN_TOK_DIR = ROOT + "/ComfyUI/comfy/text_encoders/qwen25_tokenizer"
T5_TOK_DIR = ROOT + "/ComfyUI/comfy/text_encoders/t5_tokenizer"
TE_ONNX = ROOT + "/onnx/te/qwen3_te.onnx"
ADAPTER_ONNX = ROOT + "/onnx/llm_adapter.onnx"
PAD_TOKEN = 151643
CONTEXT_LEN = 512

_qtok = _t5tok = _te = _ad = None

def _load_qtok():
    global _qtok
    if _qtok is None:
        from transformers import Qwen2Tokenizer
        _qtok = Qwen2Tokenizer.from_pretrained(QWEN_TOK_DIR)
    return _qtok

def _load_t5tok():
    global _t5tok
    if _t5tok is None:
        from transformers import T5TokenizerFast
        _t5tok = T5TokenizerFast.from_pretrained(T5_TOK_DIR)
    return _t5tok

def _load_te():
    global _te
    if _te is None:
        import onnxruntime as ort
        _te = ort.InferenceSession(TE_ONNX, providers=["CPUExecutionProvider"])
    return _te

def _load_ad():
    global _ad
    if _ad is None:
        import onnxruntime as ort
        _ad = ort.InferenceSession(ADAPTER_ONNX, providers=["CPUExecutionProvider"])
    return _ad

def tokenize_qwen(text):
    """qwen3: raw BPE ids, no special tokens. min_length 1."""
    ids = _load_qtok().encode(text, add_special_tokens=False)
    return ids if ids else [PAD_TOKEN]

def tokenize_t5(text):
    """t5xxl ids (T5TokenizerFast). add_special_tokens=True keeps the </s> eos like SDTokenizer."""
    ids = _load_t5tok().encode(text, add_special_tokens=True)
    return ids if ids else [1]

def encode(text):
    """text -> DiT context [1, 512, 1024] float32 (qwen3 -> llm_adapter -> pad512)."""
    te, ad = _load_te(), _load_ad()
    qids = np.array([tokenize_qwen(text)], dtype=np.int64)
    qhidden = te.run(None, {te.get_inputs()[0].name: qids})[0].astype(np.float32)  # [1,Lq,1024]
    t5ids = np.array([tokenize_t5(text)], dtype=np.int64)
    ain = {ad.get_inputs()[0].name: qhidden, ad.get_inputs()[1].name: t5ids}
    adapted = ad.run(None, ain)[0].astype(np.float32)                              # [1,Lt5,1024]
    # t5xxl_weights are all 1.0 in CLIPTextEncode (no emphasis) -> skip multiply
    L = adapted.shape[1]
    if L < CONTEXT_LEN:
        adapted = np.pad(adapted, ((0, 0), (0, CONTEXT_LEN - L), (0, 0)))
    elif L > CONTEXT_LEN:
        adapted = adapted[:, :CONTEXT_LEN]
    return adapted

if __name__ == "__main__":
    import sys
    txt = "masterpiece, best quality, 1girl, fern \\(sousou no frieren\\)"
    print("qwen tokens:", len(tokenize_qwen(txt)))
    print("t5 tokens:", len(tokenize_t5(txt)))
    if "--full" in sys.argv:  # loads TE + adapter onnx; run only when E2E idle
        ctx = encode(txt)
        print("context:", ctx.shape, ctx.dtype, "mean", float(ctx.mean()))
