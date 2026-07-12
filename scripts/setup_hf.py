"""
setup_hf.py — Public bootstrap for Cube3D INT4 inference.

Clones cube3d from GitHub, downloads weights from
HuggingFace, and builds the cfg dict needed by quant_int4.py.

All paths are parameters with sensible cross-platform defaults: nothing is
hardcoded to a specific runtime environment.

Usage:

  # Minimal - paths default to sub-directories of the current working directory:
  import setup_hf
  cfg = setup_hf.bootstrap_hf()

  # Explicit paths (Colab, RunPod, local, etc.):
  cfg = setup_hf.bootstrap_hf(
      cube_dir  = "/workspace/cube",          # where to clone/find cube3d
      work_dir  = "/workspace/cube3d_output", # for any local output files
  )

  # Torch version pin (only needed if torch is not yet at the pinned version):
  setup_hf.ensure_torch()
"""

import os
import sys
import subprocess


# ── Torch version pin ────────────────────────────────────────────────────────
TORCH_TARGET        = "2.10.0+cu128"
TORCHVISION_TARGET  = "0.25.0+cu128"
TORCHAUDIO_TARGET   = "2.10.0"
TORCH_INDEX_URL     = "https://download.pytorch.org/whl/cu128"
_DEFAULT_HF_REPO      = "TrNi/efficient-cube3d" # you may specify custom ones in bootstrap_hf()
_DEFAULT_CUBE3D_REPO  = "TrNi/cube"
_DEFAULT_CUBE3D_GIT   = "https://github.com/TrNi/cube.git"

def ensure_torch():
    """
    Verify that the pinned torch version is installed.
    If not, install it and force a Colab runtime restart.

    Run this in Cell 0 BEFORE importing anything else.
    After restart, skip Cell 0 and start from Cell 1.
    """
    try:
        import torch
        if torch.__version__ == TORCH_TARGET:
            print(f"[setup_hf] torch {torch.__version__}  ✓  (correct version, no restart needed)")
            return
        print(f"[setup_hf] torch {torch.__version__} detected — need {TORCH_TARGET}. Reinstalling ...")
    except ImportError:
        print(f"[setup_hf] torch not found — installing {TORCH_TARGET} ...")

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
    os.kill(os.getpid(), 9)   # SIGKILL — Colab auto-restarts the runtime


# ── Public entry point ────────────────────────────────────────────────────────
def bootstrap_hf(
    cube_dir: str = "/content/efficient-cube3d/cube",
    work_dir: str = "/content/cube3d_output",
    hf_repo: str = _DEFAULT_HF_REPO,
    cube3d_repo: str = _DEFAULT_CUBE3D_REPO,
    cube3d_git_url: str = _DEFAULT_CUBE3D_GIT,
) -> dict:
    """
    Install dependencies, clone cube3d, download weights from HuggingFace,
    verify GPU, and build config dict.

    Safe to call multiple times — all steps are idempotent.

    Args:
        cube_dir (str): Directory where the cube3d repo is (or will be) cloned.
            Defaults to <cwd>/cube.
        work_dir (str): Directory for local output files (e.g. int4_dir).
            Defaults to <cwd>/cube3d_output.
        hf_repo (str): HuggingFace repo ID containing INT4 weights and config.            
        cube3d_repo (str): HuggingFace repo ID for the Roblox cube3d model weights.            
        cube3d_git_url (str): Git URL for the cube3d source repo to clone.            

    Returns:
        cfg (dict): paths + device used by quant_int4 and other scripts.
    """
    if cube_dir is None:
        cube_dir = os.path.join(os.getcwd(), "cube")
    if work_dir is None:
        work_dir = os.path.join(os.getcwd(), "cube3d_output")

    _ensure_packages()
    _ensure_cube3d(cube_dir, cube3d_git_url)
    device = _check_gpu()

    cfg = _download_weights(device, work_dir, hf_repo, cube3d_repo)
    _make_dirs(cfg)

    print("[setup_hf] Bootstrap complete. cfg ready.\n")
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
        print(f"[setup_hf] Installing: {' '.join(to_install)} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + to_install,
            check=True,
        )
        print("[setup_hf] Packages installed.")
    else:
        print("[setup_hf] All packages already available.")

    import torchao
    import torch
    print(f"[setup_hf] torch   : {torch.__version__}")
    print(f"[setup_hf] torchao : {getattr(torchao, '__version__', 'bundled')}")


def _ensure_cube3d(cube_dir: str, git_url: str):
    """Clone and install cube3d from git if not already importable."""
    # pyproject.toml package discovery is misconfigured (looks for cube/ inside cube3d/),
    # so pip install -e doesn't register cube3d in site-packages.
    # Adding cube_dir to sys.path makes `import cube3d` work (cube3d/ has __init__.py).
    # This must run BEFORE the import check — cube3d may be importable now via CWD
    # but fail later when CWD changes.
    if cube_dir not in sys.path:
        sys.path.insert(0, cube_dir)

    try:
        import cube3d  # noqa: F401
        print("[setup_hf] cube3d already installed.")
        return
    except ImportError:
        pass

    if not os.path.exists(cube_dir):
        print(f"[setup_hf] Cloning {git_url} → {cube_dir} ...")
        subprocess.run(
            ["git", "clone", git_url, cube_dir],
            check=True,
        )
        print("[setup_hf] Clone complete.")

    print("[setup_hf] Installing cube3d (editable) ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", cube_dir],
        check=True,
    )
    print("[setup_hf] cube3d installed.")


def _check_gpu() -> "torch.device":
    import torch
    assert torch.cuda.is_available(), (
        "No CUDA GPU detected. Attach a GPU with at least 16 GB VRAM (e.g. L4, A10, A100) "
        "before running this script."
    )
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1e9
    print(f"[setup_hf] GPU  : {props.name}")
    print(f"[setup_hf] VRAM : {vram:.1f} GB")
    assert vram > 15, (
        f"Minimum 16 GB VRAM required; detected {vram:.1f} GB. "
        f"Use an L4 (16 GB) or A100 (40 GB) runtime."
    )
    return torch.device("cuda")


def _download_weights(device, work_dir: str, hf_repo: str, cube3d_repo: str) -> dict:
    """Download all weights from HuggingFace and build cfg dict."""
    from huggingface_hub import hf_hub_download

    print(f"[setup_hf] Downloading weights (hf_repo={hf_repo}, cube3d_repo={cube3d_repo}) ...")

    print("  INT4 GPT weights (1.26 GB) ...")
    int4_weights = hf_hub_download(hf_repo, "shape_gpt_rtn_int4_g128.pt")

    # BF16 GPT weights (7.17 GB) are NOT downloaded here. The meta-device load path in
    # load_int4_engine builds the model structure from config alone and overwrites it
    # with the INT4 state dict — BF16 weights are never needed on 16 GB GPUs.
    # First-time quantization (A100 only) is a separate offline step.
    gpt_ckpt = ""   # not used by load_int4_engine; kept in cfg for interface consistency

    print("  VQ-VAE tokenizer (1.10 GB) ...")
    tok_ckpt = hf_hub_download(hf_repo, "shape_tokenizer.safetensors")

    print("  Model config ...")
    config_path = hf_hub_download(hf_repo, "open_model_v0.5.yaml")
    quant_config_path = hf_hub_download(hf_repo, "quant_config.json")

    print("[setup_hf] All weights downloaded.  ✓")

    return dict(
        config_path  = config_path,
        gpt_ckpt     = gpt_ckpt,
        tok_ckpt     = tok_ckpt,
        int4_weights = int4_weights,
        int4_dir     = work_dir,
        int4_config  = quant_config_path,
        device       = device,
        group_size   = 128,
    )


def _make_dirs(cfg: dict):
    os.makedirs(cfg["int4_dir"], exist_ok=True)
    print(f"[setup_hf] Output directory: {cfg['int4_dir']}")
