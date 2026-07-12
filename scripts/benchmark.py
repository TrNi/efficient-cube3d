"""
benchmark.py — Inference and Chamfer Distance evaluation for Cube3D INT4.

All output files are written under cfg["compare_dir"]:
  {label}_{idx:03d}.npz       — per-prompt mesh (vertices, faces)
  {label}_metrics.json        — latency + VRAM per prompt
  {label}_cd_results.csv      — Chamfer Distance per prompt
  {label}_visualization.html  — side-by-side BF16 vs INT4 (1 per category)

Usage (Colab cells):
    import benchmark
    benchmark.run_full_benchmark(engine, cfg, label="rtn_int4_w4a16")

    # Or step by step:
    benchmark.single_inference(engine, "A wooden chair", cfg)
    metrics = benchmark.run_inference_suite(engine, cfg, label="rtn_int4_w4a16")
    results = benchmark.compute_chamfer_distances(cfg, label="rtn_int4_w4a16")
    benchmark.print_summary(results)
"""

import os
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import KDTree


# ── Constants ─────────────────────────────────────────────────────────────────

SEED         = 42
N_CD_SAMPLES = 30_000   # standard in 3DGen literature
CD_SEED      = 42


# ── Determinism ────────────────────────────────────────────────────────────────

def set_deterministic(seed: int = SEED):
    """Set all RNG sources for bit-exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Chamfer Distance helpers ──────────────────────────────────────────────────

def _sample_surface(verts: np.ndarray, faces: np.ndarray, n: int = N_CD_SAMPLES) -> np.ndarray:
    """
    Area-weighted uniform surface sampling (trimesh).
    Weights each triangle by area → uniform distribution on surface.
    """
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float64)


def _normalize_unit_sphere(pts: np.ndarray) -> np.ndarray:
    """
    Translate centroid → origin, scale so max L2 norm = 1.
    Applied per-shape so CD is scale-invariant across prompts.
    """
    pts = pts - pts.mean(axis=0)
    r   = np.linalg.norm(pts, axis=1).max()
    return pts / (r + 1e-10)


def _chamfer_l2(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Bidirectional L2 Chamfer Distance (non-squared, symmetric sum-of-means).
    CD(A, B) = mean_a min_b ||a-b|| + mean_b min_a ||b-a||
    Reference: Fan et al., CVPR 2017.
    Returns (cd_total, cd_fwd, cd_bwd).
    """
    tree_b = KDTree(b)
    tree_a = KDTree(a)
    d_ab = tree_b.query(a, k=1, workers=-1)[0]
    d_ba = tree_a.query(b, k=1, workers=-1)[0]
    return float(d_ab.mean()) + float(d_ba.mean()), float(d_ab.mean()), float(d_ba.mean())


# ── Prompt suite loader ───────────────────────────────────────────────────────

def _load_suite(cfg: dict) -> tuple:
    """Load benchmark prompts and categories from master_suite.json."""
    assert os.path.exists(cfg["prompts_path"]), (
        f"Prompt suite not found: {cfg['prompts_path']}"
    )
    with open(cfg["prompts_path"]) as f:
        suite = json.load(f)
    return suite["prompts"], suite["categories"]


# ── Single inference ──────────────────────────────────────────────────────────

def single_inference(
    engine,
    prompt: str,
    cfg: dict,
    resolution_base: float = 8.0,
    visualize: bool = True,
):
    """
    Run one prompt through engine, print stats, optionally show plotly mesh.
    Returns (vertices, faces) as numpy arrays.
    """
    set_deterministic()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with torch.inference_mode():
        out = engine.t2s(
            [prompt],
            use_kv_cache    = True,
            resolution_base = resolution_base,
            top_p           = None,
        )

    elapsed   = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    v, f      = out[0][0], out[0][1]

    print(f"Prompt   : {prompt}")
    print(f"Latency  : {elapsed:.2f} s")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Verts    : {v.shape[0]:,}   Faces: {f.shape[0]:,}")

    if visualize:
        _show_mesh(v, f, title=prompt)

    return v, f


def _show_mesh(verts, faces, title: str = ""):
    """Interactive plotly 3D mesh (works in Colab)."""
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.85, color="lightblue",
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
        )])
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=30),
            height=500,
        )
        fig.show()
    except Exception as e:
        print(f"[benchmark] Plotly visualization skipped: {e}")


# ── Full inference suite ───────────────────────────────────────────────────────

def run_inference_suite(engine, cfg: dict, label: str) -> list:
    """
    Run all 140 benchmark prompts under deterministic greedy decoding.
    Saves each mesh as {compare_dir}/{label}_{idx:03d}.npz.
    Skips already-saved .npz files (resumable after crash).
    Returns list of metric dicts.
    """
    prompts, categories = _load_suite(cfg)
    out_dir = cfg["compare_dir"]
    os.makedirs(out_dir, exist_ok=True)

    set_deterministic(SEED)
    metrics = []

    print(f"\n[benchmark] Running {len(prompts)} prompts  label={label}")
    print(f"{'idx':>4}  {'category':24}  {'s':>7}  {'verts':>7}  {'vram':>6}  prompt")
    print("─" * 94)

    for idx, prompt in enumerate(prompts):
        npz_path = Path(out_dir) / f"{label}_{idx:03d}.npz"

        if npz_path.exists():
            d = np.load(npz_path)
            metrics.append(dict(
                idx=idx, category=categories[idx], prompt=prompt,
                latency_s=None,
                n_verts=int(d["vertices"].shape[0]),
                n_faces=int(d["faces"].shape[0]),
                peak_vram_gb=None,
            ))
            print(f"  {idx:03d}  {categories[idx]:24}  {'SKIP':>7}  "
                  f"{d['vertices'].shape[0]:>7,}  {'—':>6}  {prompt[:35]}")
            continue

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                out = engine.t2s(
                    [prompt],
                    use_kv_cache    = True,
                    resolution_base = 8.0,
                    top_p           = None,
                )
            elapsed   = time.perf_counter() - t0
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
            v, f      = out[0][0], out[0][1]

            np.savez_compressed(npz_path, vertices=v, faces=f)
            metrics.append(dict(
                idx=idx, category=categories[idx], prompt=prompt,
                latency_s=round(elapsed, 2),
                n_verts=int(v.shape[0]),
                n_faces=int(f.shape[0]),
                peak_vram_gb=round(peak_vram, 2),
            ))
            print(f"  {idx:03d}  {categories[idx]:24}  {elapsed:>6.1f}s  "
                  f"{v.shape[0]:>7,}  {peak_vram:>5.1f}G  {prompt[:35]}")

        except Exception as e:
            print(f"  {idx:03d}  ERROR: {e}")
            metrics.append(dict(
                idx=idx, category=categories[idx], prompt=prompt,
                latency_s=None, n_verts=0, n_faces=0, peak_vram_gb=None,
            ))

    metrics_path = Path(out_dir) / f"{label}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    _print_latency_summary(metrics)
    print(f"\n[benchmark] Metrics saved → {metrics_path}")
    return metrics


def _print_latency_summary(metrics: list):
    valid = [m for m in metrics if m["latency_s"] is not None]
    if not valid:
        return
    times = [m["latency_s"] for m in valid]
    vrams = [m["peak_vram_gb"] for m in valid]
    print(f"\n── Latency  mean={np.mean(times):.1f}s  median={np.median(times):.1f}s  "
          f"p95={np.percentile(times, 95):.1f}s  (n={len(valid)})")
    print(f"── VRAM     mean={np.mean(vrams):.2f} GB  max={np.max(vrams):.2f} GB")


# ── BF16 baseline generation ──────────────────────────────────────────────────

def generate_baselines(engine, cfg: dict, use_kv_cache: bool = True) -> None:
    """
    Generate BF16 baseline .npz files for any prompt index that does not yet
    have a file in baseline_dir.  Existing files are skipped — safe to re-run.

    Saves: {baseline_dir}/bf16_{idx:03d}.npz   (vertices, faces)

    Typical use: run once after adding a new category to master_suite.json so
    that the new prompts (e.g. indices 140–169) get their BF16 reference meshes
    before running compute_chamfer_distances() on any quantized model.

    Parameters
    ----------
    engine       : BF16 engine (Engine or EngineFast)
    cfg          : config dict from setup.bootstrap()
    use_kv_cache : True for EngineFast, False for Engine
    """
    prompts, categories = _load_suite(cfg)
    baseline_dir = cfg["baseline_dir"]
    os.makedirs(baseline_dir, exist_ok=True)

    missing = [
        idx for idx in range(len(prompts))
        if not (Path(baseline_dir) / f"bf16_{idx:03d}.npz").exists()
    ]

    if not missing:
        print(f"[benchmark] All {len(prompts)} BF16 baselines already exist. Nothing to do.")
        return

    print(f"[benchmark] Generating {len(missing)} missing BF16 baselines "
          f"(out of {len(prompts)} total) ...")
    print(f"  Saving to: {baseline_dir}")
    print(f"  Indices  : {missing[0]}–{missing[-1]}")

    set_deterministic(SEED)
    n_done = n_err = 0

    for idx in missing:
        prompt   = prompts[idx]
        npz_path = Path(baseline_dir) / f"bf16_{idx:03d}.npz"

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                out = engine.t2s(
                    [prompt],
                    use_kv_cache    = use_kv_cache,
                    resolution_base = 8.0,
                    top_p           = None,
                )
            elapsed   = time.perf_counter() - t0
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
            v, f      = out[0][0], out[0][1]
            np.savez_compressed(npz_path, vertices=v, faces=f)
            n_done += 1
            print(f"  [{idx:03d}] {elapsed:.1f}s  {v.shape[0]:,}v  {peak_vram:.1f}GB  "
                  f"[{categories[idx]}]  {prompt[:50]}")
        except Exception as e:
            n_err += 1
            print(f"  [{idx:03d}] ERROR: {e}  —  {prompt[:50]}")

    print(f"\n[benchmark] Done.  Generated: {n_done}  Errors: {n_err}")


# ── Chamfer Distance ──────────────────────────────────────────────────────────

def compute_chamfer_distances(cfg: dict, label: str) -> list:
    """
    Load BF16 baseline .npz and label .npz for each prompt index.
    Compute bidirectional L2 Chamfer Distance.
    Returns list of result dicts.
    """
    prompts, categories = _load_suite(cfg)
    np.random.seed(CD_SEED)

    results = []
    print(f"\n[benchmark] Computing Chamfer Distances  (N={N_CD_SAMPLES:,}, label={label})")
    print(f"{'idx':>4}  {'category':24}  {'CD×10⁻³':>10}  {'BF16 v':>8}  {'INT4 v':>8}")
    print("─" * 68)

    for idx in range(len(prompts)):
        bf16_npz = Path(cfg["baseline_dir"]) / f"bf16_{idx:03d}.npz"
        int4_npz = Path(cfg["compare_dir"])  / f"{label}_{idx:03d}.npz"

        row = dict(
            idx=idx, category=categories[idx], prompt=prompts[idx],
            cd=None, cd_fwd=None, cd_bwd=None, bv=0, qv=0,
        )

        if not bf16_npz.exists():
            print(f"  {idx:03d}  MISSING BF16 baseline ({bf16_npz.name})")
            results.append(row)
            continue
        if not int4_npz.exists():
            print(f"  {idx:03d}  MISSING {label} result")
            results.append(row)
            continue

        b = np.load(bf16_npz)
        q = np.load(int4_npz)
        bv = int(b["vertices"].shape[0])
        qv = int(q["vertices"].shape[0])
        row["bv"] = bv
        row["qv"] = qv

        try:
            pts_b = _normalize_unit_sphere(_sample_surface(b["vertices"], b["faces"]))
            pts_q = _normalize_unit_sphere(_sample_surface(q["vertices"], q["faces"]))
            cd, cd_fwd, cd_bwd = _chamfer_l2(pts_b, pts_q)
            row.update(cd=cd, cd_fwd=cd_fwd, cd_bwd=cd_bwd)
            print(f"  {idx:03d}  {categories[idx]:24}  {cd*1e3:>10.4f}  "
                  f"{bv:>8,}  {qv:>8,}")
        except Exception as e:
            print(f"  {idx:03d}  CD error: {e}")

        results.append(row)

    return results


# ── Summary reporting ─────────────────────────────────────────────────────────

def print_summary(results: list, label: str = ""):
    """Print per-category and overall Chamfer Distance statistics to stdout."""
    valid = [r for r in results if r["cd"] is not None]
    if not valid:
        print("[benchmark] No valid CD results to summarise.")
        return

    cds = np.array([r["cd"] for r in valid])

    print(f"\n{'═'*60}")
    print(f"  Chamfer Distance Summary  {label}")
    print(f"{'═'*60}")
    print(f"  Prompts evaluated : {len(valid)} / {len(results)}")
    print(f"  Mean   CD × 10⁻³ : {cds.mean()*1e3:.4f}")
    print(f"  Median CD × 10⁻³ : {np.median(cds)*1e3:.4f}")
    print(f"  Std    CD × 10⁻³ : {cds.std()*1e3:.4f}")
    print(f"  P75    CD × 10⁻³ : {np.percentile(cds, 75)*1e3:.4f}")
    print(f"  P95    CD × 10⁻³ : {np.percentile(cds, 95)*1e3:.4f}")
    print(f"  Max    CD × 10⁻³ : {cds.max()*1e3:.4f}  "
          f"(idx={valid[int(cds.argmax())]['idx']:03d})")

    print(f"\n── Per-category ──")
    cats = {}
    for r in valid:
        cats.setdefault(r["category"], []).append(r["cd"])
    for cat, vals in sorted(cats.items()):
        v = np.array(vals)
        print(f"  {cat:28}  {v.mean()*1e3:.4f} ± {v.std()*1e3:.4f}  (n={len(v)})")


def save_summary_csv(results: list, cfg: dict, label: str):
    """Save full per-prompt results as CSV."""
    import pandas as pd
    df   = pd.DataFrame(results)
    path = Path(cfg["compare_dir"]) / f"{label}_cd_results.csv"
    df.to_csv(path, index=False)
    print(f"[benchmark] CD results saved → {path}")


# ── Plotly HTML visualization ─────────────────────────────────────────────────

def save_visualization(cfg: dict, label: str):
    """
    Save a side-by-side BF16 (blue) vs INT4 (salmon) plotly HTML.
    One mesh per category (14 rows × 2 columns = 28 subplots).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from collections import OrderedDict

    prompts, categories = _load_suite(cfg)

    seen = OrderedDict()
    for i, c in enumerate(categories):
        if c not in seen:
            seen[c] = i
    vis_idx = list(seen.values())

    titles = []
    for idx in vis_idx:
        short = prompts[idx][:38] + ("…" if len(prompts[idx]) > 38 else "")
        titles += [f"BF16 [{idx:03d}] {short}", f"{label} [{idx:03d}]"]

    n   = len(vis_idx)
    fig = make_subplots(
        rows=n, cols=2, subplot_titles=titles,
        specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}]] * n,
        vertical_spacing=0.03, horizontal_spacing=0.02,
    )

    for row_i, idx in enumerate(vis_idx, start=1):
        bf16_npz = Path(cfg["baseline_dir"]) / f"bf16_{idx:03d}.npz"
        int4_npz = Path(cfg["compare_dir"])  / f"{label}_{idx:03d}.npz"
        for col, color, npz_path in [
            (1, "lightblue",   bf16_npz),
            (2, "lightsalmon", int4_npz),
        ]:
            if not npz_path.exists():
                continue
            d = np.load(npz_path)
            v, f = d["vertices"], d["faces"]
            fig.add_trace(
                go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2],
                    i=f[:, 0], j=f[:, 1], k=f[:, 2],
                    color=color, opacity=0.85,
                    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
                    showscale=False,
                ),
                row=row_i, col=col,
            )

    fig.update_layout(
        height=420 * n,
        title_text=f"BF16 (blue) vs {label} (salmon) — 1 per category",
        showlegend=False,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    html_path = Path(cfg["compare_dir"]) / f"{label}_visualization.html"
    fig.write_html(str(html_path))
    print(f"[benchmark] Visualization saved → {html_path}")


# ── One-shot entry point ──────────────────────────────────────────────────────

def run_full_benchmark(engine, cfg: dict, label: str = "rtn_int4_w4a16"):
    """
    One-shot: inference suite → Chamfer Distances → summary → CSV → HTML.
    Existing .npz results are reused automatically (resumable).

    Call from Colab:
        benchmark.run_full_benchmark(engine, cfg, label="rtn_int4_w4a16")
    """
    metrics = run_inference_suite(engine, cfg, label)
    results = compute_chamfer_distances(cfg, label)
    print_summary(results, label)
    save_summary_csv(results, cfg, label)
    save_visualization(cfg, label)
    print(f"\n[benchmark] All outputs → {cfg['compare_dir']}/")


# ── Four-way engine comparison ────────────────────────────────────────────────

def benchmark_config(
    cfg: dict,
    label: str,
    quantized: bool = False,
    use_fast: bool = True,
    n_bench: int = None,
) -> dict:
    """
    Load one engine configuration, time setup + inference, save metrics, free VRAM.

    Parameters
    ----------
    cfg       : config dict from setup.bootstrap()
    label     : output filename stem, e.g. "bf16_engine"
    quantized : False → BF16 weights;  True → torchao INT4
    use_fast  : True  → EngineFast (flash attn + KV cache + CUDA graph)
                False → Engine     (standard attn, no KV cache, no graph)
    n_bench   : number of prompts to time (takes first prompt per category → 14 max)

    INT4 shortcut path (quantized=True and int4_weights file present):
        Calls load_int4_engine / _build_int4_gpt_model directly, bypassing the
        BF16 load entirely. Peak VRAM ~8 GB vs ~22 GB for the legacy inline path.

    Run from Colab (one cell per config):
        results = benchmark.benchmark_config(cfg, label="bf16_enginefast",
                                             quantized=False, use_fast=True)
    """
    import gc
    from cube3d.inference.engine import Engine, EngineFast

    EngineClass = EngineFast if use_fast else Engine

    print(f"\n{'═'*64}")
    print(f"  Benchmarking: {label}  "
          f"({'INT4' if quantized else 'BF16'}, "
          f"{'EngineFast' if use_fast else 'Engine'})")
    print(f"{'═'*64}")

    # ── Model size on disk ────────────────────────────────────────────────────
    if quantized and os.path.exists(cfg["int4_weights"]):
        model_size_gb = os.path.getsize(cfg["int4_weights"]) / 1e9
    else:
        model_size_gb = os.path.getsize(cfg["gpt_ckpt"]) / 1e9
    print(f"  Model on disk : {model_size_gb:.3f} GB")

    # ── Load engine (weights + CUDA graph if EngineFast) ─────────────────────
    # INT4 shortcut: if saved weights exist, build via meta-device scaffold so the
    # 7.17 GB BF16 checkpoint never touches GPU. Falls back to the inline BF16 path
    # when INT4 weights are absent (first-time quantization scenario).
    int4_shortcut = quantized and os.path.exists(cfg["int4_weights"])

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_load_start = time.perf_counter()

    if int4_shortcut:
        import quant_int4
        if use_fast:
            engine = quant_int4.load_int4_engine(cfg)
        else:
            from cube3d.inference.engine import Engine
            gpt_model = quant_int4._build_int4_gpt_model(cfg)
            engine = Engine(
                config_path     = cfg["config_path"],
                gpt_ckpt_path   = cfg["gpt_ckpt"],
                shape_ckpt_path = cfg["tok_ckpt"],
                device          = cfg["device"],
                _gpt_model      = gpt_model,
            )
    else:
        engine = EngineClass(
            config_path     = cfg["config_path"],
            gpt_ckpt_path   = cfg["gpt_ckpt"],
            shape_ckpt_path = cfg["tok_ckpt"],
            device          = cfg["device"],
        )

        if quantized:
            import torch.nn as nn
            from torchao.quantization import int4_weight_only, quantize_

            engine.gpt_model = engine.gpt_model.to(torch.bfloat16)
            _orig_et = engine.gpt_model.encode_text
            engine.gpt_model.encode_text = (
                lambda x: _orig_et(x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x)
            )

            def _filter(mod, fqn):
                return (isinstance(mod, nn.Linear)
                        and mod.in_features  % 128 == 0
                        and mod.out_features % 16  == 0)

            quantize_(engine.gpt_model, int4_weight_only(group_size=cfg["group_size"]),
                      filter_fn=_filter)
            engine.gpt_model.eval()
            if use_fast:
                engine.graph = torch.cuda.CUDAGraph()
                engine._warmup_and_capture_graph()

    torch.cuda.synchronize()
    weight_load_time_s = time.perf_counter() - t_load_start
    vram_after_load_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Load time     : {weight_load_time_s:.2f} s")
    print(f"  VRAM after load: {vram_after_load_gb:.3f} GB")

    # ── Separate CUDA graph capture timing (EngineFast only) ─────────────────
    cuda_graph_time_s = 0.0
    if use_fast:
        torch.cuda.synchronize()
        t_graph = time.perf_counter()
        engine.graph = torch.cuda.CUDAGraph()
        engine._warmup_and_capture_graph()
        torch.cuda.synchronize()
        cuda_graph_time_s = time.perf_counter() - t_graph
        print(f"  CUDA graph    : {cuda_graph_time_s:.2f} s  (re-captured to isolate timing)")

    # ── Select benchmark prompts (first from each category) ──────────────────
    prompts, categories = _load_suite(cfg)
    seen = {}
    for i, c in enumerate(categories):
        if c not in seen:
            seen[c] = i
    all_pairs   = list(seen.items())             # one entry per category, insertion order
    bench_pairs = all_pairs if n_bench is None else all_pairs[:n_bench]

    # ── Warmup inference (Engine only — no CUDA graph pre-warms kernels) ─────
    if not use_fast:
        print(f"\n  Warmup inference (1×, untimed) ...")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            engine.t2s([prompts[bench_pairs[0][1]]], use_kv_cache=False,
                       resolution_base=4.0, top_p=None)
        print(f"  Warmup done.")

    # ── Timed inference loop ──────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    set_deterministic(SEED)
    latencies_s  = []
    n_verts_list = []
    use_kv       = use_fast   # KV cache only available in EngineFast

    print(f"\n  {'#':>3}  {'category':24}  {'lat (s)':>8}  {'verts':>8}  prompt")
    print(f"  {'─'*80}")

    for rank, (cat, idx) in enumerate(bench_pairs, 1):
        prompt = prompts[idx]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = engine.t2s([prompt], use_kv_cache=use_kv,
                             resolution_base=8.0, top_p=None)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        v = out[0][0]
        latencies_s.append(elapsed)
        n_verts_list.append(int(v.shape[0]))
        print(f"  {rank:>3}  {cat:24}  {elapsed:>8.2f}  {v.shape[0]:>8,}  {prompt[:35]}")

    vram_peak_inference_gb = torch.cuda.max_memory_allocated() / 1e9
    total_bench_time_s = weight_load_time_s + cuda_graph_time_s + sum(latencies_s)

    # ── Results dict ──────────────────────────────────────────────────────────
    results = dict(
        label                     = label,
        quantized                 = quantized,
        use_fast                  = use_fast,
        model_size_gb             = round(model_size_gb, 3),
        weight_load_time_s        = round(weight_load_time_s, 2),
        cuda_graph_capture_time_s = round(cuda_graph_time_s, 2),
        total_setup_time_s        = round(weight_load_time_s + cuda_graph_time_s, 2),
        vram_after_load_gb        = round(vram_after_load_gb, 3),
        vram_peak_inference_gb    = round(vram_peak_inference_gb, 3),
        int4_shortcut_path        = int4_shortcut,
        n_prompts                 = len(latencies_s),
        latencies_s               = [round(x, 3) for x in latencies_s],
        mean_latency_s            = round(float(np.mean(latencies_s)), 2),
        median_latency_s          = round(float(np.median(latencies_s)), 2),
        p95_latency_s             = round(float(np.percentile(latencies_s, 95)), 2),
        min_latency_s             = round(float(np.min(latencies_s)), 2),
        max_latency_s             = round(float(np.max(latencies_s)), 2),
        mean_verts                = round(float(np.mean(n_verts_list)), 0),
        total_bench_time_s        = round(total_bench_time_s, 2),
    )

    out_path = Path(cfg["compare_dir"]) / f"{label}_timing.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Mean latency  : {results['mean_latency_s']:.2f} s")
    print(f"  Median latency: {results['median_latency_s']:.2f} s")
    print(f"  Peak VRAM     : {vram_peak_inference_gb:.3f} GB")
    print(f"  Total time    : {total_bench_time_s:.1f} s  (load + graph + inference)")
    print(f"  Saved → {out_path}")

    # ── Free VRAM ─────────────────────────────────────────────────────────────
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    vram_after_cleanup = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after cleanup: {vram_after_cleanup:.3f} GB")

    return results


def compare_configs(
    cfg: dict,
    labels: list = ("bf16_engine", "bf16_enginefast", "int4_engine", "int4_enginefast"),
) -> None:
    """
    Load saved {label}_timing.json for each label and print a side-by-side table.

    Call from Colab:
        benchmark.compare_configs(cfg)
    """
    data = []
    for label in labels:
        path = Path(cfg["compare_dir"]) / f"{label}_timing.json"
        if path.exists():
            with open(path) as f:
                data.append(json.load(f))
        else:
            print(f"[compare] Missing: {path.name}  (run benchmark_config first)")

    if not data:
        print("[compare] No results to compare.")
        return

    bf16_fast = next((d for d in data if d["label"] == "bf16_enginefast"), None)
    bf16_std  = next((d for d in data if d["label"] == "bf16_engine"),     None)

    cols = [
        ("label",                     "Config",          20),
        ("model_size_gb",             "Model(GB)",       10),
        ("weight_load_time_s",        "Load(s)",          8),
        ("cuda_graph_capture_time_s", "Graph(s)",         8),
        ("total_setup_time_s",        "Setup(s)",         9),
        ("vram_after_load_gb",        "VRAM load",        9),
        ("vram_peak_inference_gb",    "VRAM peak",        9),
        ("mean_latency_s",            "Mean(s)",          8),
        ("median_latency_s",          "Median(s)",        9),
        ("p95_latency_s",             "95%ile(s)",        9),
    ]

    sep   = "  "
    hdr   = sep.join(f"{h:{w}}" for _, h, w in cols) + "  [Peak VRAM Reduction, Mean Latency Speedup, Loading Time Speedup]"
    ruler = "─" * len(hdr)

    # second header row: annotate the latency block and improvements suffix
    _pos = 0
    _lat_start = _lat_end = 0
    for _i, (_k, _h, _w) in enumerate(cols):
        if _i > 0:
            _pos += len(sep)
        if _k == "mean_latency_s":
            _lat_start = _pos
        _pos += _w
        if _k == "p95_latency_s":
            _lat_end = _pos
    _lat_label = "\u2190 per-inference latency (s) \u2192"
    _lat_span  = _lat_end - _lat_start
    subhdr = " " * _lat_start + f"{_lat_label:^{_lat_span}}"
    # " " * (len(hdr) - _lat_end)
    print(f"\n{'═'*len(hdr)}")
    print(f"  Engine Configuration Comparison  ({len(data)} configs, {data[0]['n_prompts']} prompts each)")
    print(f"{'═'*len(hdr)}")
    print(subhdr)
    print(hdr)
    print(ruler)

    # independent references per metric
    lat_ref  = max(data, key=lambda d: d["mean_latency_s"]        or 0)
    vram_ref = max(data, key=lambda d: d["vram_peak_inference_gb"] or 0)
    load_ref = max(data, key=lambda d: d["total_setup_time_s"]     or 0)

    def _pct(ref_val, val):
        return (ref_val - val) / ref_val * 100

    for d in data:
        row = sep.join(f"{str(d.get(k, '—')):{w}}" for k, _, w in cols)
        vram_s = "VRAM:1×" if d["label"] == vram_ref["label"] else \
                 f"VRAM:{_pct(vram_ref['vram_peak_inference_gb'], d['vram_peak_inference_gb']):.0f}%"
        lat_s  = "Lat:1×"  if d["label"] == lat_ref["label"]  else \
                 f"Lat:{_pct(lat_ref['mean_latency_s'], d['mean_latency_s']):.0f}%"
        load_s = "Load:1×" if d["label"] == load_ref["label"] else \
                 f"Load:{_pct(load_ref['total_setup_time_s'], d['total_setup_time_s']):.0f}%"
        print(row + f"  [{vram_s},  {lat_s},  {load_s}]")

    print(ruler)

    # ── Key observations ──────────────────────────────────────────────────────
    print(f"\n── Key numbers ──")
    for d in data:
        if d["mean_latency_s"] and lat_ref["mean_latency_s"]:
            if d["label"] == lat_ref["label"]:
                lat_note = "lat: 1× (slowest)"
            else:
                pct = (lat_ref["mean_latency_s"] - d["mean_latency_s"]) / lat_ref["mean_latency_s"] * 100
                lat_note = f"{pct:.0f}% faster"
        else:
            lat_note = "lat: n/a"
        if d["vram_peak_inference_gb"] and vram_ref["vram_peak_inference_gb"]:
            if d["label"] == vram_ref["label"]:
                vram_note = "vram: 1× (most VRAM)"
            else:
                pct = _pct(vram_ref["vram_peak_inference_gb"], d["vram_peak_inference_gb"])
                vram_note = f"{pct:.0f}% peak VRAM"
        else:
            vram_note = "vram: n/a"
        if d["total_setup_time_s"] and load_ref["total_setup_time_s"]:
            if d["label"] == load_ref["label"]:
                load_note = "load: 1× (slowest)"
            else:
                pct = _pct(load_ref["total_setup_time_s"], d["total_setup_time_s"])
                load_note = f"{pct:.0f}% faster load"
        else:
            load_note = "load: n/a"
        print(f"  {d['label']:22}  {lat_note}  |  mean={d['mean_latency_s']:.2f}s  |  "
              f"{vram_note}  |  peak={d['vram_peak_inference_gb']:.2f} GB  |  {load_note}  |  setup={d['total_setup_time_s']:.1f}s")
