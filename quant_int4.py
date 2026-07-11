"""
quant_int4.py — RTN W4A16 quantization of ShapeGPT via torchao.

Quantization method : RTN (Round-To-Nearest), symmetric, per-group
Kernel              : torchao _weight_int4pack_mm (fused, never materialises W_fp16)
Group size          : 128
Skipped layers      : shape_proj (in=16, not divisible by 128)
                      lm_head    (out=4099, not divisible by 16)

Usage (Colab cells):
    import quant_int4
    engine = quant_int4.build_int4_engine(cfg)       # quantize or load from cache
"""

import os
import json
import torch
import torch.nn as nn


# ── Layer filter ──────────────────────────────────────────────────────────────

def _torchao_filter(mod: nn.Module, fqn: str) -> bool:
    """
    Return True for nn.Linear layers that torchao can safely quantize.

    Requirements of _weight_int4pack_mm:
      - in_features  must be divisible by group_size (128)
      - out_features must be divisible by 16

    Skipped by this filter:
      - shape_proj  : in_features=16  → 16 % 128 ≠ 0
      - lm_head     : out_features=4099 → 4099 % 16 ≠ 0
    All other linears (including CLIP text encoder) pass.
    """
    return (
        isinstance(mod, nn.Linear)
        and mod.in_features  % 128 == 0
        and mod.out_features % 16  == 0
    )


# ── Layer counting helper ─────────────────────────────────────────────────────

def _count_quantized(model: nn.Module) -> int:
    """Count nn.Linear layers whose weight was replaced by torchao AffineQuantizedTensor."""
    try:
        from torchao.dtypes import AffineQuantizedTensor
    except ImportError:
        return 0
    return sum(
        1 for _, m in model.named_modules()
        if isinstance(m, nn.Linear) and isinstance(m.weight, AffineQuantizedTensor)
    )


# ── BF16 engine loader ────────────────────────────────────────────────────────

def load_bf16_engine(cfg: dict):
    """
    Load EngineFast with BF16 weights from Drive.
    ~30 s I/O for 7.17 GB safetensors.
    EngineFast uses flash attention + KV cache + CUDA graph.
    """
    from cube3d.inference.engine import EngineFast

    print("[quant_int4] Loading BF16 engine ...")
    engine = EngineFast(
        config_path     = cfg["config_path"],
        gpt_ckpt_path   = cfg["gpt_ckpt"],
        shape_ckpt_path = cfg["tok_ckpt"],
        device          = cfg["device"],
    )
    n_lin = sum(1 for _, m in engine.gpt_model.named_modules() if isinstance(m, nn.Linear))
    print(f"[quant_int4] BF16 engine ready. gpt_model has {n_lin} nn.Linear layers.")
    return engine


# ── Quantization ──────────────────────────────────────────────────────────────

def quantize_rtn_w4a16(engine, cfg: dict):
    """
    Apply torchao int4_weight_only (RTN) to engine.gpt_model in-place.
    After quantization, re-captures the EngineFast CUDA graph.

    Notes:
      - quantize_() replaces matching nn.Linear modules with torchao INT4 modules.
      - The CUDA graph captured at EngineFast.__init__ used BF16 ops; it must be
        re-captured after quantization so the decode loop uses _weight_int4pack_mm.
      - If CUDA graph re-capture raises (rare: torch.compile conflict), catch and
        continue — inference will still work, just without graph replay.
    """
    from torchao.quantization import int4_weight_only, quantize_

    group_size = cfg["group_size"]
    print(f"[quant_int4] Quantizing gpt_model  (W4A16 RTN, group_size={group_size}) ...")
    print(f"[quant_int4] Skipping: shape_proj (in=16), lm_head (out=4099)")

    # Cast to bfloat16 first: EngineFast loads in float32, but torchao 0.10.0 requires
    # consistent dtypes for scale and zero_point (both must be the same dtype as the weight).
    engine.gpt_model = engine.gpt_model.to(torch.bfloat16)

    quantize_(
        engine.gpt_model,
        int4_weight_only(group_size=group_size),
        filter_fn=_torchao_filter,
    )
    engine.gpt_model.eval()

    n_q      = _count_quantized(engine.gpt_model)
    n_lin    = sum(1 for _, m in engine.gpt_model.named_modules() if isinstance(m, nn.Linear))
    n_skipped = n_lin - n_q
    print(f"[quant_int4] INT4 layers : {n_q} / {n_lin}  |  Skipped (fp32/bf16) : {n_skipped}")

    _recapture_cuda_graph(engine)
    return engine


def _recapture_cuda_graph(engine):
    """
    Re-capture EngineFast CUDA graph with the current (INT4) ops.

    Two issues to handle after .to(bfloat16) + quantize_():
      1. engine.graph already owns a captured graph from __init__; reset it first.
      2. run_clip calls encode_text with float32 CLIP output, but encode_text now
         has bfloat16 weights → dtype error.  Patch encode_text to cast its input.
    """
    print("[quant_int4] Re-capturing CUDA graph ...")

    # Fix 1: reset the CUDAGraph instance so _warmup_and_capture_graph can re-capture.
    engine.graph = torch.cuda.CUDAGraph()

    # Fix 2: encode_text now has bfloat16 weights (after .to(bfloat16)), but run_clip
    # calls it with float32 CLIP output (autocast is explicitly disabled for text_model).
    # Wrap encode_text to cast float32 → bfloat16 before the linear.
    _orig_encode_text = engine.gpt_model.encode_text
    engine.gpt_model.encode_text = (
        lambda x: _orig_encode_text(x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x)
    )

    try:
        engine._warmup_and_capture_graph()
        print("[quant_int4] CUDA graph captured  ✓")
    except Exception as e:
        print(f"[quant_int4] CUDA graph capture failed: {e}")
        print("[quant_int4] Inference will proceed without CUDA graph (slower but correct.)")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_int4_weights(engine, cfg: dict) -> str:
    """
    Save torchao INT4 gpt_model state dict to Drive.
    Expected size: ~1.8 GB  (vs 7.17 GB BF16).
    """
    os.makedirs(cfg["int4_dir"], exist_ok=True)
    path = cfg["int4_weights"]
    print(f"[quant_int4] Saving INT4 state dict → {path} ...")
    torch.save(engine.gpt_model.state_dict(), path)
    size_gb = os.path.getsize(path) / 1e9
    print(f"[quant_int4] Saved. Size: {size_gb:.2f} GB  (BF16 was 7.17 GB, {7.17/size_gb:.1f}× reduction)")

    with open(cfg["int4_config"], "w") as f:
        json.dump(
            {
                "w_bit":      4,
                "group_size": cfg["group_size"],
                "method":     "RTN",
                "kernel":     "torchao_int4_weight_only (_weight_int4pack_mm)",
                "skipped":    ["shape_proj (in=16)", "lm_head (out=4099)"],
            },
            f, indent=2,
        )
    return path


# ── Load from saved weights ───────────────────────────────────────────────────

def load_int4_engine(cfg: dict):
    """
    Load INT4 engine from previously saved Drive weights.

    Sequence:
      1. Load BF16 engine   (~30 s — needed to build module structure)
      2. quantize_()        (replaces nn.Linear with _Int4WeightLinear, runs RTN)
      3. load_state_dict()  (overwrites RTN weights with saved GPTQ/RTN weights)
      4. Re-capture CUDA graph

    Step 2 runs a fast RTN pass — its output is immediately discarded in step 3.
    The overhead of step 2 is ~5–10 s (no calibration, just weight packing).
    """
    from torchao.quantization import int4_weight_only, quantize_

    assert os.path.exists(cfg["int4_weights"]), (
        f"INT4 weights not found: {cfg['int4_weights']}\n"
        "Run build_int4_engine(cfg) first to quantize and save."
    )

    engine = load_bf16_engine(cfg)

    # Cast to bfloat16 first (same requirement as the quantize path).
    engine.gpt_model = engine.gpt_model.to(torch.bfloat16)

    # Patch encode_text to accept float32 CLIP input with bfloat16 weights.
    _orig = engine.gpt_model.encode_text
    engine.gpt_model.encode_text = (
        lambda x: _orig(x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x)
    )

    print("[quant_int4] Setting up INT4 module structure (fast RTN pass, discarded) ...")
    quantize_(
        engine.gpt_model,
        int4_weight_only(group_size=cfg["group_size"]),
        filter_fn=_torchao_filter,
    )

    print(f"[quant_int4] Loading saved INT4 weights from Drive ...")
    state = torch.load(cfg["int4_weights"], map_location="cpu", weights_only=False)
    missing, unexpected = engine.gpt_model.load_state_dict(state, strict=False)
    if missing:
        print(f"[quant_int4] WARNING — missing keys  : {missing[:5]}")
    if unexpected:
        print(f"[quant_int4] WARNING — unexpected keys: {unexpected[:5]}")

    engine.gpt_model.eval()
    _recapture_cuda_graph(engine)
    print("[quant_int4] INT4 engine ready  ✓")
    return engine


# ── One-shot builder (main entry point) ───────────────────────────────────────

def build_int4_engine(cfg: dict):
    """
    One-shot builder.

    - If saved INT4 weights exist on Drive → load them (avoids re-quantization).
    - Otherwise → load BF16, quantize with RTN, save weights, return engine.

    Call this from Colab:
        engine = quant_int4.build_int4_engine(cfg)
    """
    if os.path.exists(cfg["int4_weights"]):
        size_gb = os.path.getsize(cfg["int4_weights"]) / 1e9
        if size_gb > 3.0:
            print(f"[quant_int4] WARNING — saved file is {size_gb:.2f} GB "
                  f"(expected ~1.3 GB for INT4 g128). File is likely a bad BF16 dump. "
                  f"Deleting and re-quantizing ...")
            os.remove(cfg["int4_weights"])
        else:
            print(f"[quant_int4] Found saved INT4 weights ({size_gb:.2f} GB). "
                  f"Loading (skipping re-quantization).")
            return load_int4_engine(cfg)

    print("[quant_int4] No saved INT4 weights found — quantizing from BF16.")
    engine = load_bf16_engine(cfg)
    engine = quantize_rtn_w4a16(engine, cfg)
    save_int4_weights(engine, cfg)
    return engine


# ── Memory report helper ──────────────────────────────────────────────────────

def vram_report():
    """Print current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved()  / 1e9
    print(f"[quant_int4] VRAM allocated: {allocated:.2f} GB  |  reserved: {reserved:.2f} GB")
