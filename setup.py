"""
setup.py — Colab session bootstrap for Cube3D INT4 experiments.

Usage (Colab notebooks — two cells in order):

  Cell 0 — torch version pin (may require runtime restart):
    import sys; sys.path.insert(0, "/content/drive/MyDrive/trials/efficiency/cube3d/scripts")
    import setup; setup.ensure_torch()          # installs correct torch, restarts if needed

  Cell 1 — rest of session setup (run after restart, skip Cell 0):
    from google.colab import drive; drive.mount('/content/drive')
    GDRIVE_ROOT = "/content/drive/MyDrive/trials/efficiency/cube3d"
    import sys; sys.path.insert(0, f"{GDRIVE_ROOT}/scripts")
    import setup; cfg = setup.bootstrap(GDRIVE_ROOT)
"""

import os
import sys
import subprocess


# ── Torch version pin ────────────────────────────────────────────────────────

TORCH_TARGET        = "2.10.0+cu128"
TORCHVISION_TARGET  = "0.25.0+cu128"
TORCHAUDIO_TARGET   = "2.10.0"
TORCH_INDEX_URL     = "https://download.pytorch.org/whl/cu128"


def ensure_torch():
    """
    Verify that the pinned torch version is installed.
    If not, install it and force a Colab runtime restart.

    Run this in Cell 0 BEFORE mounting Drive or importing anything else.
    After restart, skip Cell 0 and start from Cell 1.
    """
    try:
        import torch
        if torch.__version__ == TORCH_TARGET:
            print(f"[setup] torch {torch.__version__}  ✓  (correct version, no restart needed)")
            return
        print(f"[setup] torch {torch.__version__} detected — need {TORCH_TARGET}. Reinstalling ...")
    except ImportError:
        print(f"[setup] torch not found — installing {TORCH_TARGET} ...")

    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            f"torch=={TORCH_TARGET}",
            f"torchvision=={TORCHVISION_TARGET}",
            "--index-url", TORCH_INDEX_URL,
            "--no-deps",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            f"torchaudio=={TORCHAUDIO_TARGET}",
            "--index-url", TORCH_INDEX_URL,
            "--no-deps",
        ],
        check=True,
    )
    print("\n" + "═" * 60)
    print("  ⚠️  RESTART REQUIRED")
    print("  torch was (re)installed — the running process still uses the")
    print("  old version. Restarting Colab runtime now ...")
    print("  After restart: skip Cell 0, run from Cell 1 onward.")
    print("═" * 60)
    import os
    os.kill(os.getpid(), 9)   # SIGKILL — Colab auto-restarts the runtime


# ── Public entry point ────────────────────────────────────────────────────────

def bootstrap(gdrive_root: str) -> dict:
    """
    Install dependencies, configure sys.path, verify GPU, build config dict.
    Safe to call multiple times — all steps are idempotent.

    Returns:
        cfg (dict): paths + device used by all other scripts.
    """
    _ensure_packages()
    _ensure_cube3d(gdrive_root)
    device = _check_gpu()

    cfg = _build_cfg(gdrive_root, device)
    _verify_weights(cfg)
    _make_dirs(cfg)

    print("[setup] Bootstrap complete. cfg ready.\n")
    return cfg


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_packages():
    """Install required packages if not already present."""
    packages = {
        "torchao":         "torchao",
        "safetensors":     "safetensors",
        "trimesh":         "trimesh",
        "scipy":           "scipy",
        "plotly":          "plotly",
        "pandas":          "pandas",
        "huggingface_hub": "huggingface_hub",
    }
    # torch itself is NOT in this list — it is pinned by ensure_torch() in Cell 0.
    to_install = []
    for module, pip_name in packages.items():
        try:
            __import__(module)
        except ImportError:
            to_install.append(pip_name)

    if to_install:
        print(f"[setup] Installing: {' '.join(to_install)} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + to_install,
            check=True,
        )
        print("[setup] Packages installed.")
    else:
        print("[setup] All packages already available.")

    import torchao
    import torch
    print(f"[setup] torch   : {torch.__version__}")
    print(f"[setup] torchao : {getattr(torchao, '__version__', 'bundled')}")


def _ensure_cube3d(gdrive_root: str):
    """Clone and install cube3d if not already importable."""
    try:
        import cube3d  # noqa: F401
        print("[setup] cube3d already installed.")
        return
    except ImportError:
        pass

    cube_dir = f"{gdrive_root}/cube"
    if not os.path.exists(cube_dir):
        print(f"[setup] Cloning Roblox/cube → {cube_dir} ...")
        subprocess.run(
            ["git", "clone", "https://github.com/Roblox/cube.git", cube_dir],
            check=True,
        )
        print("[setup] Clone complete.")

    print("[setup] Installing cube3d (editable) ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", cube_dir],
        check=True,
    )
    print("[setup] cube3d installed.")


def _check_gpu() -> "torch.device":
    import torch
    assert torch.cuda.is_available(), (
        "No GPU detected — switch Colab runtime to A100 (Runtime → Change runtime type)."
    )
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1e9
    print(f"[setup] GPU  : {props.name}")
    print(f"[setup] VRAM : {vram:.1f} GB")
    assert vram > 20, f"A100 (40 GB) required; detected {vram:.1f} GB."
    return torch.device("cuda")


def _build_cfg(gdrive_root: str, device) -> dict:
    cube_dir = f"{gdrive_root}/cube"
    return dict(
        gdrive_root   = gdrive_root,
        cube_dir      = cube_dir,
        config_path   = f"{cube_dir}/cube3d/configs/open_model_v0.5.yaml",
        gpt_ckpt      = f"{gdrive_root}/weights/bf16/shape_gpt.safetensors",
        tok_ckpt      = f"{gdrive_root}/weights/bf16/shape_tokenizer.safetensors",
        int4_dir      = f"{gdrive_root}/weights/torchao_int4",
        int4_weights  = f"{gdrive_root}/weights/torchao_int4/shape_gpt_rtn_int4_g128.pt",
        int4_config   = f"{gdrive_root}/weights/torchao_int4/quant_config.json",
        baseline_dir  = f"{gdrive_root}/baseline_meshes",
        compare_dir   = f"{gdrive_root}/comparison_rtn_int4",
        prompts_path  = f"{gdrive_root}/benchmark_prompts/master_suite.json",
        device        = device,
        group_size    = 128,
    )


def _verify_weights(cfg: dict):
    for key, label in [("gpt_ckpt", "GPT BF16"), ("tok_ckpt", "Tokenizer BF16")]:
        path = cfg[key]
        assert os.path.exists(path), (
            f"{label} weights not found: {path}\n"
            f"Download from: huggingface.co/Roblox/cube3d-v0.5"
        )
        print(f"[setup] {label:14}: {os.path.getsize(path)/1e9:.2f} GB  ✓")

    if os.path.exists(cfg["int4_weights"]):
        print(f"[setup] INT4 weights : {os.path.getsize(cfg['int4_weights'])/1e9:.2f} GB  ✓  "
              f"(will skip re-quantization)")
    else:
        print(f"[setup] INT4 weights : not yet saved  (will quantize on next step)")


def _make_dirs(cfg: dict):
    for key in ("int4_dir", "compare_dir"):
        os.makedirs(cfg[key], exist_ok=True)
