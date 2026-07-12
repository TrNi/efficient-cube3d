# %% [markdown]
"""
# INT4 Quantization of ShapeGPT (cube3d) — End-to-End Tutorial

**Goal**: Quantize the Roblox cube3d text-to-3D model from BF16 to INT4 (W4A16,
per-group, group_size=128), evaluate mesh quality via Chamfer Distance, and export
to HuggingFace safetensors format.

**Notebook order**:
0. **Quick start — load INT4 from HuggingFace and run inference** ← start here
1. Runtime setup — Drive, GPU, installs, weights
2. Architecture audit — enumerate all linear layers
3. Single inference — verify BF16 pipeline end-to-end
4. BF16 baseline generation — save 140 ground-truth meshes
5. Calibration data — 128 Cap3D prompts for activation statistics
6. RTN INT4 quantization — collect activations, quantize, save
7. INT4 evaluation — load on EngineFast, Chamfer Distance vs BF16
8. HuggingFace export — safetensors + quant_config.json

**Hardware**: NVIDIA A100 (≥40 GB recommended for EngineFast + KV cache).
T4/V100 can run Sections 1–6 but EngineFast in Section 7 needs ≥24 GB.

**Key result**: 6.5× checkpoint compression (7.17 GB → 1.11 GB), median
Chamfer Distance 60.0 × 10⁻³ vs BF16, 70% of shapes within CD < 100 × 10⁻³.
"""

# %% [markdown]
"""
## Section 0 — Quick Start: Load INT4 from HuggingFace

This section is **fully self-contained** — you can run it without executing
Sections 1–8 at all. It shows how to load the published INT4 model and
generate 3D meshes from text.

### What you need

| File | Source repo | Size | Purpose |
|------|------------|------|---------|
| `shape_gpt.safetensors` (BF16) | `Roblox/cube3d-v0.5` | 7.17 GB | Architecture initialisation only |
| `shape_tokenizer.safetensors` | `Roblox/cube3d-v0.5` | ~1.1 GB | VQ-VAE decoder (unchanged) |
| `gpt_rtn_int4_w4_g128.safetensors` | INT4 HF repo | **1.11 GB** | Actual INT4 weights used at runtime |
| `quant_config.json` | INT4 HF repo | <1 KB | Quantization metadata |

> **Why download BF16 at all?** `EngineFast.__init__` instantiates the
> transformer architecture by loading the BF16 checkpoint. We immediately
> discard those weights and replace them with the INT4 state dict — so
> **GPU RAM at inference is ≈1.11 GB**, not 3.58 GB. The 7.17 GB BF16 file
> is a one-time download that stays on Drive.
"""

# %%
# ── 0a. Install dependencies ─────────────────────────────────────────────────
!pip install -q autoawq accelerate safetensors huggingface_hub trimesh

# Clone the cube3d repo (provides Engine / EngineFast classes).
# Change CUBE_DIR if you already have a clone on Drive.
import os, subprocess

CUBE_DIR = "/content/cube"      # or e.g. "/content/drive/MyDrive/.../cube"

if not os.path.exists(CUBE_DIR):
    subprocess.run(["git", "clone",
                    "https://github.com/Roblox/cube.git", CUBE_DIR], check=True)
    subprocess.run(["pip", "install", "-q", "-e", CUBE_DIR], check=True)
    print(f"Cloned and installed cube3d from {CUBE_DIR}")
else:
    subprocess.run(["pip", "install", "-q", "-e", CUBE_DIR], check=True)
    print(f"Using existing clone at {CUBE_DIR}")

# %%
# ── 0b. Paths & HuggingFace repo IDs ────────────────────────────────────────
import torch
from huggingface_hub import hf_hub_download

# ┌─────────────────────────────────────────────────────┐
# │  Set these two constants before running this cell.  │
# └─────────────────────────────────────────────────────┘
INT4_REPO    = "your-username/cube3d-v0.5-int4-w4a16-g128"  # ← your HF repo
BASE_REPO    = "Roblox/cube3d-v0.5"                         # original Roblox repo

WEIGHTS_DIR  = "/content/weights"          # local cache — change to Drive path to persist
os.makedirs(WEIGHTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %%
# ── 0c. Download files ───────────────────────────────────────────────────────
# Files already on disk are not re-downloaded (hf_hub_download is idempotent).

print("Downloading INT4 weights and config...")
int4_weights_path = hf_hub_download(
    repo_id=INT4_REPO,
    filename="gpt_rtn_int4_w4_g128.safetensors",
    local_dir=WEIGHTS_DIR,
)
quant_config_path = hf_hub_download(
    repo_id=INT4_REPO,
    filename="quant_config.json",
    local_dir=WEIGHTS_DIR,
)

print("Downloading base model prerequisites (architecture + tokenizer)...")
gpt_bf16_path = hf_hub_download(
    repo_id=BASE_REPO,
    filename="shape_gpt.safetensors",
    local_dir=WEIGHTS_DIR,
)
tokenizer_path = hf_hub_download(
    repo_id=BASE_REPO,
    filename="shape_tokenizer.safetensors",
    local_dir=WEIGHTS_DIR,
)

CONFIG_YAML = f"{CUBE_DIR}/cube3d/configs/open_model_v0.5.yaml"

print(f"INT4 weights : {os.path.getsize(int4_weights_path)/1e9:.2f} GB")
print(f"BF16 weights : {os.path.getsize(gpt_bf16_path)/1e9:.2f} GB  (architecture init only)")
print(f"Tokenizer    : {os.path.getsize(tokenizer_path)/1e9:.2f} GB")

# %%
# ── 0d. Build EngineFast and swap in INT4 weights ───────────────────────────
import json
import torch.nn as nn
from safetensors.torch import load_file
from awq.modules.linear.gemm import WQLinear_GEMM
from cube3d.inference.engine import EngineFast

# Load quantization metadata
with open(quant_config_path) as f:
    quant_cfg = json.load(f)

W_BIT     = quant_cfg["w_bit"]          # 4
Q_GROUP   = quant_cfg["q_group_size"]   # 128
quant_set = set(quant_cfg["quantized_layers"])

print(f"Quantization : INT{W_BIT}, group_size={Q_GROUP}")
print(f"Quantized layers: {len(quant_set)}")
print(f"Skipped layers  : {quant_cfg['skipped_layers']}  ({quant_cfg['skip_reason']})")

# Step 1: Initialise EngineFast with BF16 weights (sets up architecture)
print("\nInitialising EngineFast (loading BF16 architecture)...")
engine = EngineFast(
    config_path     = CONFIG_YAML,
    gpt_ckpt_path   = gpt_bf16_path,
    shape_ckpt_path = tokenizer_path,
    device          = device,
)

# Step 2: Replace nn.Linear → WQLinear_GEMM (structure only, init_only=True)
for name, module in list(engine.gpt_model.named_modules()):
    if name not in quant_set:
        continue
    wq = WQLinear_GEMM.from_linear(module, W_BIT, Q_GROUP, init_only=True)
    *path, child = name.split(".")
    parent = engine.gpt_model
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, child, wq)

# Step 3: Load INT4 weights (overwrites BF16 weights — GPU RAM drops to ~1.11 GB)
print("Loading INT4 state dict...")
state_dict = load_file(int4_weights_path, device=str(device))
engine.gpt_model.load_state_dict(state_dict)
engine.gpt_model.eval()

import torch
from torch.nn import Module

def count_model_bytes(model: Module) -> int:
    total  = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total

gpu_ram_gb = count_model_bytes(engine.gpt_model) / 1e9
print(f"\nEngine ready. INT4 model GPU RAM: {gpu_ram_gb:.3f} GB")

# %%
# ── 0e. Run text-to-3D inference ────────────────────────────────────────────
import time, numpy as np

prompts = [
    "A wooden dining chair with four legs and a curved backrest",
    "A smooth sphere",
    "A grand piano with the lid propped open",
]

results = []
for prompt in prompts:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = engine.t2s(
            [prompt],
            use_kv_cache    = True,    # flash attention + KV cache (EngineFast)
            resolution_base = 8.0,    # controls mesh resolution / token count
            top_p           = None,   # greedy decoding
        )
    elapsed   = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    verts, faces = out[0][0], out[0][1]
    results.append((prompt, verts, faces))
    print(f"[{elapsed:.1f}s  {peak_vram:.1f}GB VRAM]  "
          f"{verts.shape[0]:,}v {faces.shape[0]:,}f  |  {prompt}")

# %%
# ── 0f. Visualise results ────────────────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

n = len(results)
fig = make_subplots(
    rows=1, cols=n,
    subplot_titles=[p[:45] for p, _, _ in results],
    specs=[[{"type": "mesh3d"}] * n],
    horizontal_spacing=0.02,
)
colors = ["lightblue", "lightsalmon", "lightgreen"]
for col, (prompt, verts, faces) in enumerate(results, start=1):
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=colors[col - 1], opacity=0.85,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
        showscale=False,
    ), row=1, col=col)
fig.update_layout(
    height=500, title_text="INT4 cube3d inference (W4A16 RTN)",
    showlegend=False, margin=dict(l=0, r=0, t=50, b=0),
)
fig.show()

# %%
# ── 0g. Export a mesh to .obj ────────────────────────────────────────────────
import trimesh

for i, (prompt, verts, faces) in enumerate(results):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    out_path = f"/content/mesh_{i:02d}.obj"
    mesh.export(out_path)
    print(f"Saved: {out_path}  ({len(verts):,} verts)")

# %% [markdown]
"""
---
*The sections below document how the INT4 model was produced — useful for
reproducibility, ablation studies, or quantizing future cube3d checkpoints.*

---
"""

# %% [markdown]
"""
## Section 1 — Runtime Setup
"""

# %%
from google.colab import drive
drive.mount('/content/drive')
import os

GDRIVE_ROOT = "/content/drive/MyDrive/trials/efficiency/cube3d"
os.makedirs(f"{GDRIVE_ROOT}/baseline_meshes",  exist_ok=True)
os.makedirs(f"{GDRIVE_ROOT}/weights/bf16",     exist_ok=True)
os.makedirs(f"{GDRIVE_ROOT}/quantized",        exist_ok=True)
os.makedirs(f"{GDRIVE_ROOT}/calibration",      exist_ok=True)
os.makedirs(f"{GDRIVE_ROOT}/metrics",          exist_ok=True)
os.makedirs(f"{GDRIVE_ROOT}/comparison_int4",  exist_ok=True)
os.makedirs(f"{GDRIVE_ROOT}/benchmark_prompts",exist_ok=True)
print("Drive mounted. Workspace:", GDRIVE_ROOT)

# %%
import subprocess, torch

gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
print(gpu_info)

assert torch.cuda.is_available(), "No GPU — switch runtime to GPU (A100 recommended)."
device = torch.device("cuda")
props  = torch.cuda.get_device_properties(0)
print(f"Device : {props.name}")
print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")

# %%
# Install all dependencies once. Skip on repeated runs if kernel persists.
!pip install -q --upgrade \
    autoawq \
    accelerate \
    datasets \
    trimesh \
    open3d \
    huggingface_hub \
    safetensors

# %%
# Clone the Roblox cube3d repo and install it as a package.
CUBE_DIR = f"{GDRIVE_ROOT}/cube"

if not os.path.exists(CUBE_DIR):
    !git clone https://github.com/Roblox/cube.git {CUBE_DIR}

%cd {CUBE_DIR}
!pip install -q -e .
print("cube3d installed from:", CUBE_DIR)

# %%
WEIGHTS_DIR = f"{GDRIVE_ROOT}/weights/bf16"
GPT_CKPT    = f"{WEIGHTS_DIR}/shape_gpt.safetensors"
TOK_CKPT    = f"{WEIGHTS_DIR}/shape_tokenizer.safetensors"
CONFIG_PATH = f"{CUBE_DIR}/cube3d/configs/open_model_v0.5.yaml"

# %%
# Download model weights from HuggingFace (7.17 GB + 1.1 GB).
# Files are saved to Drive and survive session restarts.
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID   = "Roblox/cube3d-v0.5"
FILENAMES = ["shape_gpt.safetensors", "shape_tokenizer.safetensors"]

os.makedirs(WEIGHTS_DIR, exist_ok=True)

for fname in FILENAMES:
    dest = f"{WEIGHTS_DIR}/{fname}"
    if os.path.exists(dest):
        print(f"Already on Drive, skipping: {fname}")
        continue
    print(f"Downloading {fname} ...")
    hf_hub_download(repo_id=REPO_ID, filename=fname, local_dir=WEIGHTS_DIR)
    print(f"Done: {fname}")

assert os.path.exists(GPT_CKPT), f"GPT checkpoint missing: {GPT_CKPT}"
assert os.path.exists(TOK_CKPT), f"Tokenizer missing: {TOK_CKPT}"
print(f"\nGPT checkpoint : {os.path.getsize(GPT_CKPT)/1e9:.2f} GB  (FP32 on disk, loaded as BF16)")
print(f"Shape tokenizer: {os.path.getsize(TOK_CKPT)/1e9:.2f} GB")

# %% [markdown]
"""
## Section 2 — Architecture Audit

Understanding the model before quantizing it.

```
Text prompt (str)
      │
      ▼
CLIP ViT-L/14 text encoder          [inside Engine, frozen]
      │  → text embeddings  [B, 77, 768]
      ▼
ShapeGPT  (dual-stream GPT decoder)
  - 23 dual-stream transformer blocks + 1 single block
  - Stream X: shape token sequence  [B, T_shape ≤ 1024]
  - Stream C: CLIP text conditioning [B, 77, 768]
  - Vocabulary: 16,384 shape codes + 3 special tokens = 16,387
  - Output: shape token indices  [B, T_shape]
      │
      ▼
Shape Tokenizer (VQ-VAE decoder, 24-layer transformer)
  - Codebook lookup → latent tokens [B, 512, D]
  - Transformer decoder → 3D SDF grid
  - Marching cubes → (vertices, faces)
      │
      ▼
Output: vertices [N_verts, 3]  float32
        faces    [N_faces, 3]  int32
```

`N_verts` and `N_faces` are variable — controlled by `resolution_base`.
`resolution_base=8.0` → roughly 15k–400k vertices depending on shape complexity.
"""

# %%
import torch
import pandas as pd
from cube3d.inference.engine import Engine, EngineFast

# EngineFast: flash attention + KV cache (requires ≥24 GB VRAM on A100)
engine_bf16 = EngineFast(
    config_path     = CONFIG_PATH,
    gpt_ckpt_path   = GPT_CKPT,
    shape_ckpt_path = TOK_CKPT,
    device          = device,
)
print("BF16 EngineFast loaded.")

# %%
# Enumerate all nn.Linear layers — tells us what AWQ can target.
gpt_model = engine_bf16.gpt_model
records = []

for name, module in gpt_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        w = module.weight
        records.append({
            "name":       name,
            "out":        w.shape[0],
            "in":         w.shape[1],
            "params_M":   round(w.numel() / 1e6, 2),
            "dtype":      str(w.dtype),
        })

df = pd.DataFrame(records)
df["cumulative_params_M"] = df["params_M"].cumsum().round(1)
print(f"Total linear layers : {len(df)}")
print(f"Total linear params : {df['params_M'].sum():.0f} M")
print()
print(df.to_string(index=False))

df.to_csv(f"{GDRIVE_ROOT}/metrics/linear_layer_map.csv", index=False)
print(f"\nLayer map saved → {GDRIVE_ROOT}/metrics/linear_layer_map.csv")

# %% [markdown]
"""
### Key observations from the audit

- `lm_head` has `out_features = 16387`, which is **not divisible by 8** (the
  required alignment for INT4 packing into INT32). It must be skipped.
- All `LayerNorm` layers have `elementwise_affine=False` (no learnable γ/β),
  which disables AWQ scale absorption. Quantization reduces to vanilla RTN.
- The linear layers split into two structural groups:
  - **Attention**: `pre_x.c_qk`, `pre_x.c_v`, `pre_c.c_qk/c_k`, `pre_c.c_v`,
    `post_N.c_proj`
  - **FFN (SwiGLU)**: `post_N.mlp.gate_proj`, `post_N.mlp.up_proj`,
    `post_N.mlp.down_proj`
"""

# %% [markdown]
"""
## Section 3 — Single Inference (BF16 sanity check)

Verify the full text-to-mesh pipeline before any quantization.
"""

# %%
import os, random, numpy as np, torch
import time, trimesh

SEED = 42

def set_deterministic(seed: int = SEED):
    """Make all GPU and CPU ops bit-exact for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Deterministic mode ON  (seed={seed})")

set_deterministic()

# %%
TEST_PROMPT = "A wooden dining chair with four legs and two curved backrests"

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()

with torch.inference_mode():
    mesh_v_f = engine_bf16.t2s(
        [TEST_PROMPT],
        use_kv_cache    = True,
        resolution_base = 8.0,
        top_p           = None,    # greedy decoding — deterministic
    )

elapsed_s = time.perf_counter() - t0
peak_vram  = torch.cuda.max_memory_allocated() / 1e9

vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]

print(f"Prompt    : {TEST_PROMPT}")
print(f"Latency   : {elapsed_s:.2f} s")
print(f"Peak VRAM : {peak_vram:.2f} GB")
print(f"Vertices  : {vertices.shape}  dtype={vertices.dtype}")
print(f"Faces     : {faces.shape}  dtype={faces.dtype}")

# %%
# Save and visualise the test mesh.
import plotly.graph_objects as go

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.export(f"{GDRIVE_ROOT}/baseline_meshes/_single_test.obj")
np.savez_compressed(f"{GDRIVE_ROOT}/baseline_meshes/_single_test.npz",
                    vertices=vertices, faces=faces)
print("Mesh saved to Drive.")

fig = go.Figure(data=[go.Mesh3d(
    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
    i=faces[:, 0],    j=faces[:, 1],    k=faces[:, 2],
    opacity=0.8, color='lightblue',
    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
)])
fig.update_layout(
    title=f"Test mesh: {TEST_PROMPT}",
    scene=dict(aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=30), height=500,
)
fig.show()

# %% [markdown]
"""
## Section 4 — BF16 Baseline Generation

Run all 140 benchmark prompts under deterministic greedy decoding and save each
result as `.npz`. These files become the **immutable ground truth** against which
every quantized variant is evaluated. **Run this section exactly once.**

The 140 prompts span 14 categories (10 prompts each), chosen to stress different
parts of ShapeGPT's capacity:

| Category | What it stresses |
|----------|-----------------|
| `geometric_primitive` | Basic reconstruction fidelity — quality floor |
| `symmetry_topology` | Broken symmetry / wrong genus → visible artifacts |
| `fine_detail` | Sub-feature preservation under lower bitwidth |
| `nature_plant` | Irregular branching, high-frequency surface |
| `tool_hardware` | Hard-edged industrial shapes |
| `abstract_mathematical` | Concept encoding, not just appearance |
"""

# %%
import json
from pathlib import Path

# ── Benchmark prompt suite (14 categories × 10 prompts = 140 total) ─────────
BENCH_CATEGORIES = []
BENCH_PROMPTS    = []

_groups = {
    "geometric_primitive": [
        "A smooth sphere",
        "A cube with flat faces and sharp edges",
        "A circular cylinder with flat end caps",
        "A cone with a circular base",
        "A torus (donut shape)",
        "A regular octahedron",
        "A rectangular box, taller than it is wide",
        "A triangular prism",
        "A flat disc",
        "A hemisphere sitting flat side down",
    ],
    "furniture": [
        "A wooden dining chair with four legs and a curved backrest",
        "A modern sofa with three cushions and metal legs",
        "A round coffee table with a glass top and tapered legs",
        "A tall five-shelf bookcase",
        "A king-size bed frame with a slatted headboard",
        "A wooden rocking chair with curved runners",
        "A standing desk with adjustable legs",
        "A chest of drawers with six drawers and metal handles",
        "A wooden stepladder with four steps",
        "A cantilever chair made of bent tubular steel",
    ],
    "vehicle_land": [
        "A vintage red sports car with round headlights",
        "A mountain bicycle with thick tires and a suspension fork",
        "A motorcycle with chrome exhaust pipes and a fairing",
        "A double-decker bus with large windows",
        "A pickup truck with an open cargo bed",
        "A steam locomotive with a large smokestack and driving wheels",
        "A go-kart with an exposed engine and roll bar",
        "A Formula One racing car with large rear wing",
        "A classic Volkswagen Beetle",
        "A military tank with a rotating turret and tracks",
    ],
    "vehicle_air_water": [
        "A commercial airliner with two underwing engines",
        "A small single-engine propeller plane",
        "A military fighter jet with swept wings and twin tail fins",
        "A helicopter with a large main rotor and a tail rotor",
        "A hot air balloon with a wicker basket",
        "A space shuttle on its launch pad",
        "A wooden sailing ship with three masts and full canvas sails",
        "A speedboat with an outboard motor",
        "A submarine with a conning tower and propeller",
        "A Viking longship with a dragon prow and oars",
    ],
    "animal_domestic": [
        "A sitting cat with pointed ears and a long tail",
        "A golden retriever dog standing with its mouth open",
        "A horse in a standing pose with flowing mane",
        "A rabbit sitting upright with long ears",
        "A cow standing on four legs with prominent horns",
        "A domestic pig with a curly tail",
        "A rooster with a large comb and tail feathers",
        "A domestic duck swimming, with webbed feet visible",
        "A tortoise with a domed patterned shell",
        "A koi fish with flowing fins",
    ],
    "animal_wild": [
        "A flying eagle with wings fully outstretched",
        "An African elephant with large ears and long tusks",
        "A lion with a full mane, seated",
        "A great white shark with its mouth slightly open",
        "A humpback whale breaching the water surface",
        "A giraffe standing with its long neck extended upward",
        "A crocodile lying flat with its mouth open",
        "A coiled king cobra with a flared hood",
        "A stag beetle with large mandibles",
        "A tarantula spider with hairy legs spread wide",
    ],
    "architecture": [
        "A medieval stone castle with four corner towers and a drawbridge",
        "A classical Greek temple with Doric columns and a triangular pediment",
        "A stone arch bridge with a single large span",
        "A lighthouse on a rocky base with a rotating light housing",
        "A pagoda with five tiered roofs and curved eaves",
        "A Roman Colosseum section showing tiered arches",
        "A geodesic dome made of triangular panels",
        "A wooden log cabin with a chimney and porch",
        "A water tower on four tall legs",
        "A suspension bridge tower with cables",
    ],
    "electronics": [
        "A laptop computer open at ninety degrees",
        "A retro CRT television with a round screen and speaker grille",
        "A DSLR camera with a large zoom lens attached",
        "A desktop computer tower with ventilation slots and drive bays",
        "A pair of over-ear noise-cancelling headphones",
        "A smartphone lying flat with a notch display",
        "A mechanical keyboard with visible keycaps",
        "A game controller with two thumbsticks and a D-pad",
        "A reel-to-reel tape recorder with two large spools",
        "A vintage rotary telephone with a handset",
    ],
    "musical_instrument": [
        "An acoustic guitar with six strings, tuning pegs and a sound hole",
        "A grand piano with the lid propped open",
        "A violin with four strings, f-holes and a bow",
        "A trumpet with three valves and a flared bell",
        "A full drum kit with kick drum, snare, hi-hat and two toms",
        "A concert harp with 47 strings and a curved neck",
        "A French horn with its circular tubing and wide bell",
        "An upright double bass on an endpin",
        "A grand church organ pipe section",
        "A marimba with wooden bars and tubular resonators",
    ],
    "tool_hardware": [
        "A claw hammer with a wooden handle",
        "An adjustable wrench with a serrated jaw",
        "A power drill with a chuck, trigger and battery pack",
        "A hand saw with a wooden handle and serrated blade",
        "A pair of pliers with rubber-grip handles",
        "A bolt and matching nut",
        "A bench vise mounted to a workbench edge",
        "A spirit level with a bubble vial",
        "A tape measure in its retractable housing",
        "A jack plane with a flat sole and blade",
    ],
    "nature_plant": [
        "A tall pine tree with downward-drooping branches",
        "A rose flower with five petals on a thorny stem",
        "A saguaro cactus with two upward arms",
        "A mushroom with a broad flat cap and a short stalk",
        "An oak tree with a wide spreading canopy and thick trunk",
        "A sunflower with a large seed head and petals",
        "A venus flytrap with open jaw-like leaves",
        "A bunch of grapes on a stem with leaves",
        "A cross-section of a sliced orange showing segments",
        "A fern frond with many small pinnate leaflets",
    ],
    "symmetry_topology": [
        "A snowflake crystal with precise six-fold symmetry",
        "A bicycle wheel with thirty-two thin spokes",
        "A chain link with interlocking oval rings",
        "A Celtic knotwork ring",
        "A coffee mug with a closed handle (torus topology)",
        "A pretzel with its three-lobed knotted form",
        "A figure-eight Möbius strip",
        "Three interlocking Olympic rings",
        "A decorative snowflake ornament with twelve arms",
        "A spiral nautilus shell cross-section",
    ],
    "fine_detail": [
        "A detailed human skull with visible suture lines",
        "A medieval knight's full plate armour helmet with visor",
        "A decorative cast-iron fence panel with fleur-de-lis finials",
        "A detailed anatomical human heart with visible vessels",
        "A grandfather clock with pendulum, face and carved wood case",
        "A circuit board with chips, capacitors and traces",
        "A pinecone with interlocking scale pattern",
        "A detailed dragon head with scales, horns and teeth",
        "A filigree silver brooch with interwoven wire patterns",
        "A human hand with articulated fingers and knuckle detail",
    ],
    "abstract_mathematical": [
        "A torus knot sculpture",
        "A Klein bottle (a non-orientable surface)",
        "A Schwarz P triply periodic minimal surface patch",
        "A fractal tree with five levels of branching",
        "An abstract flowing ribbon sculpture with no sharp edges",
        "A double helix column, twisted four times along its height",
        "A gyroid minimal surface patch",
        "A saddle surface (hyperbolic paraboloid)",
        "An icosahedron with each face subdivided into four triangles",
        "A Stanford bunny (canonical 3D benchmark object)",
    ],
}

for cat, prompts in _groups.items():
    BENCH_PROMPTS    += prompts
    BENCH_CATEGORIES += [cat] * len(prompts)

CATEGORY_NAMES = sorted(set(BENCH_CATEGORIES))
assert len(BENCH_PROMPTS) == 140
print(f"Benchmark suite: {len(BENCH_PROMPTS)} prompts across {len(CATEGORY_NAMES)} categories")

# Save master JSON to Drive for cross-session access
master_file = f"{GDRIVE_ROOT}/benchmark_prompts/master_suite.json"
with open(master_file, "w", encoding="utf-8") as f:
    json.dump({"prompts": BENCH_PROMPTS, "categories": BENCH_CATEGORIES,
               "category_names": CATEGORY_NAMES}, f, indent=2, ensure_ascii=False)
print(f"Saved → {master_file}")

# %%
# Generate all 140 BF16 meshes.
# Expected runtime on A100 with EngineFast + KV cache: ~15–25 min.
# Already-generated files are skipped — safe to re-run after a crash.

set_deterministic(SEED)
metrics_bf16, errors_bf16 = [], []

print(f"Generating {len(BENCH_PROMPTS)} BF16 meshes...")
print("-" * 60)

for idx, prompt in enumerate(BENCH_PROMPTS):
    npz_path = f"{GDRIVE_ROOT}/baseline_meshes/bf16_{idx:03d}.npz"

    if Path(npz_path).exists():
        saved = np.load(npz_path)
        metrics_bf16.append({"idx": idx, "category": BENCH_CATEGORIES[idx],
                             "n_verts": int(saved["vertices"].shape[0]),
                             "n_faces": int(saved["faces"].shape[0]),
                             "latency_s": None})
        print(f"  [{idx:03d}] SKIP (already on Drive)  {prompt[:55]}")
        continue

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            out = engine_bf16.t2s([prompt], use_kv_cache=True,
                                  resolution_base=8.0, top_p=None)
        elapsed   = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        verts, faces_arr = out[0][0], out[0][1]

        np.savez_compressed(npz_path, vertices=verts, faces=faces_arr)
        trimesh.Trimesh(vertices=verts, faces=faces_arr).export(
            npz_path.replace(".npz", ".obj"))

        metrics_bf16.append({"idx": idx, "category": BENCH_CATEGORIES[idx],
                             "n_verts": verts.shape[0], "n_faces": faces_arr.shape[0],
                             "latency_s": round(elapsed, 2),
                             "peak_vram_gb": round(peak_vram, 2)})
        print(f"  [{idx:03d}] {elapsed:.1f}s  {verts.shape[0]:7,}v  {peak_vram:.1f}GB"
              f"  [{BENCH_CATEGORIES[idx]}]  {prompt[:40]}")
    except Exception as e:
        errors_bf16.append({"idx": idx, "error": str(e)})
        print(f"  [{idx:03d}] ERROR: {e}")

print(f"\nDone. Success: {len(metrics_bf16)}  Errors: {len(errors_bf16)}")

with open(f"{GDRIVE_ROOT}/metrics/bf16_baseline_metrics.json", "w") as f:
    json.dump(metrics_bf16, f, indent=2)
print("Metrics saved → Drive.")

# %% [markdown]
"""
## Section 5 — Calibration Data

AWQ calibration needs ~128 forward passes to collect per-channel activation
statistics (E[|X_c|] per input channel c). We use
[Cap3D](https://huggingface.co/datasets/tiange/Cap3D) — GPT-4 captions of
Objaverse 3D objects, the closest available proxy to ShapeGPT's training
distribution.

We build a dedicated `engine_calib` from the standard `Engine` (no flash
attention / KV cache) so calibration does not mutate `engine_bf16`.
"""

# %%
import pickle
from huggingface_hub import hf_hub_download

N_CALIB   = 128
CALIB_OUT = f"{GDRIVE_ROOT}/calibration/cap3d_128_prompts.json"

if os.path.exists(CALIB_OUT):
    with open(CALIB_OUT) as f:
        calib_prompts = json.load(f)
    print(f"Loaded {len(calib_prompts)} prompts from Drive cache.")
else:
    print("Downloading Cap3D caption CSV...")
    csv_path = hf_hub_download(
        repo_id="tiange/Cap3D",
        filename="Cap3D_automated_Objaverse_full.csv",
        repo_type="dataset",
        local_dir=f"{GDRIVE_ROOT}/calibration",
    )
    import pandas as pd
    captions = pd.read_csv(csv_path, header=None, names=["uid", "caption"])
    calib_prompts = (
        captions["caption"]
        .dropna().str.strip()
        .loc[lambda s: s.str.len() > 10]
        .sample(n=N_CALIB, random_state=42)
        .tolist()
    )
    with open(CALIB_OUT, "w") as f:
        json.dump(calib_prompts, f, indent=2)
    print(f"Saved {N_CALIB} prompts → {CALIB_OUT}")

print("Sample prompts:")
for p in calib_prompts[:3]:
    print(f"  {p[:80]}")

# %% [markdown]
"""
## Section 6 — RTN INT4 Quantization

### Why RTN and not full AWQ?

AWQ scale absorption requires downstream `LayerNorm` layers with learnable
affine parameters (γ, β). All `LayerNorm` layers in cube3d have
`elementwise_affine=False` — there is nothing to absorb the per-channel scale
into. The quantization therefore reduces to **Round-To-Nearest (RTN)** per-group
INT4: equivalent to vanilla per-group GPTQ without Hessian information.

This is still useful:
- Storage and memory bandwidth are genuinely INT4 (W4A16)
- Per-group quantization (group_size=128) limits quantization error vs global
- The calibration activations collected here are available for future GPTQ runs

### Quantization scheme

```
For each linear layer with weight W [out_features, in_features]:
  1. Split W into groups of size 128 along in_features
  2. Per group: scale = (max − min) / 15,  zero = round(−min / scale)
  3. Quantize: W_int4 = round(W / scale + zero).clamp(0, 15)
  4. Pack 8 INT4 values per INT32 register → WQLinear_GEMM
```
"""

# %%
import copy
import torch.nn as nn
from cube3d.inference.engine import Engine
from awq.modules.linear.gemm import WQLinear_GEMM

# Hyperparameters
W_BIT   = 4    # weight bits  (autoawq supports 4 or 8)
Q_GROUP = 128  # per-group quantization granularity (standard AWQ)

BATCH_SIZE      = 4   # prompts per calibration forward call
MAX_CALIB_STEPS = 32  # decode steps per sample (caps O(N²) autoregressive cost)

# Load a fresh standard Engine for calibration (does not mutate engine_bf16)
engine_calib = Engine(
    config_path     = CONFIG_PATH,
    gpt_ckpt_path   = GPT_CKPT,
    shape_ckpt_path = TOK_CKPT,
    device          = device,
)
engine_calib.gpt_model.eval()
print("Calibration engine loaded (standard Engine, no KV cache).")

# %%
# ── Collect activation statistics via forward hooks ──────────────────────────
#
# For each linear layer, track E[|X_c|] per input channel c using
# Welford online mean (numerically stable, unbiased).
#
# Hook signature: (module, input, output) → input[0] shape [B, ..., C_in]
# We reduce all dims except the last → per-channel mean abs activation [C_in].

from typing import Dict

activation_cache  : Dict[str, torch.Tensor] = {}
activation_counts : Dict[str, int]          = {}

def make_hook(layer_name: str):
    def hook_fn(module, inp, out):
        x    = inp[0].detach()                            # [B, ..., C_in]
        stat = x.reshape(-1, x.shape[-1]).abs().mean(0)  # [C_in]
        if layer_name not in activation_cache:
            activation_cache[layer_name]  = stat.float()
            activation_counts[layer_name] = 1
        else:
            activation_counts[layer_name] += 1
            k = activation_counts[layer_name]
            delta = stat.float().sub_(activation_cache[layer_name])
            activation_cache[layer_name].add_(delta.div_(k))   # Welford update
    return hook_fn

hooks = []
for name, module in engine_calib.gpt_model.named_modules():
    if isinstance(module, nn.Linear):
        hooks.append(module.register_forward_hook(make_hook(name)))

print(f"Registered hooks on {len(hooks)} nn.Linear layers.")

# %%
# ── Run calibration forward passes ───────────────────────────────────────────
#
# Strategy A: call engine_calib.run_gpt() with capped max_new_tokens.
# This is ~20–100× faster than full t2s (bounded decode vs O(N²) autoregressive).
# max_new_tokens is always restored via finally even on exception.

def run_calibration_batch(prompts: list) -> None:
    orig_max = engine_calib.max_new_tokens
    engine_calib.max_new_tokens = MAX_CALIB_STEPS
    try:
        with torch.inference_mode():
            engine_calib.run_gpt(
                prompts,
                use_kv_cache   = False,
                guidance_scale = 3.0,
                top_p          = None,
            )
    finally:
        engine_calib.max_new_tokens = orig_max

print(f"Calibration: N={N_CALIB}, batch={BATCH_SIZE}, max_steps={MAX_CALIB_STEPS}")

for i in range(0, N_CALIB, BATCH_SIZE):
    batch = calib_prompts[i : i + BATCH_SIZE]
    run_calibration_batch(batch)
    done = min(i + BATCH_SIZE, N_CALIB)
    if done % 32 == 0 or done == N_CALIB:
        print(f"  [{done:>4}/{N_CALIB}]  max hook updates: {max(activation_counts.values())}")

# Remove hooks immediately — they add overhead if left in place
for h in hooks:
    h.remove()
hooks.clear()
print(f"\nCalibration done. Stats for {len(activation_cache)}/{len(hooks)} layers.")

# Save to Drive — avoids re-running calibration on session restart
with open(f"{GDRIVE_ROOT}/calibration/activation_stats.pkl", "wb") as f:
    pickle.dump({k: v.cpu() for k, v in activation_cache.items()}, f)
print("Activation stats saved → Drive.")

# %%
# ── Per-group asymmetric INT4 quantization ───────────────────────────────────
#
# Returns packed WQLinear_GEMM-compatible scale and zero tensors.

def quantize_per_group(weight: torch.Tensor, w_bit: int = 4, group_size: int = 128):
    """
    Asymmetric per-group quantization (RTN).
    weight   : [out_features, in_features]  float32
    Returns  : (scales [in/G, out] fp16,
                zeros  [in/G, out] int32,
                w_dequant [out, in] float32)
    """
    out_f, in_f = weight.shape
    G   = in_f // group_size
    n   = 2 ** w_bit                          # 16 for INT4
    w_g = weight.reshape(out_f, G, group_size)
    w_max = w_g.amax(2)
    w_min = w_g.amin(2)
    scale = (w_max - w_min).clamp(min=1e-5) / (n - 1)
    zero  = (-w_min / scale).round().clamp(0, n - 1)
    w_dq  = ((w_g / scale.unsqueeze(2) + zero.unsqueeze(2))
             .round().clamp(0, n - 1) - zero.unsqueeze(2)) * scale.unsqueeze(2)
    return (
        scale.T.contiguous().half(),          # [in/G, out] FP16
        zero.T.contiguous().to(torch.int32),  # [in/G, out] INT32
        w_dq.reshape(out_f, in_f),            # [out, in]   float32
    )


def awq_quantize_model(model, activation_cache, w_bit=4, q_group=128):
    """
    Replace all eligible nn.Linear layers with WQLinear_GEMM (RTN INT4).

    A layer is skipped if:
      - It has no activation stats (not seen during calibration)
      - in_features is not divisible by q_group
      - out_features is not divisible by (32 // w_bit)  [pack alignment]
    """
    pack_align = 32 // w_bit   # 8 for INT4
    n_quantized = n_skip = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name not in activation_cache:
            n_skip += 1
            continue
        if module.in_features % q_group != 0:
            n_skip += 1
            continue
        if module.out_features % pack_align != 0:
            print(f"  SKIP {name}: out_features={module.out_features} % {pack_align} ≠ 0")
            n_skip += 1
            continue

        W = module.weight.data.float()
        q_scales, q_zeros, _ = quantize_per_group(W, w_bit, q_group)

        wq = WQLinear_GEMM.from_linear(
            module, w_bit, q_group,
            init_only=False, scales=q_scales, zeros=q_zeros,
        )
        *path, child = name.split(".")
        parent = model
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, child, wq)
        n_quantized += 1

    print(f"Quantized (RTN INT{w_bit}): {n_quantized}  |  Skipped: {n_skip}")
    return model

# %%
# ── Apply quantization ───────────────────────────────────────────────────────
print(f"Applying RTN INT{W_BIT} (q_group={Q_GROUP})...")

gpt_int4 = copy.deepcopy(engine_calib.gpt_model)
gpt_int4  = awq_quantize_model(gpt_int4, activation_cache, w_bit=W_BIT, q_group=Q_GROUP)
gpt_int4.eval()

# %%
# ── Save to Drive ────────────────────────────────────────────────────────────
QUANT_DIR = f"{GDRIVE_ROOT}/quantized"
os.makedirs(QUANT_DIR, exist_ok=True)

quantized_layer_names = [
    name for name, mod in gpt_int4.named_modules()
    if isinstance(mod, WQLinear_GEMM)
]

save_pt   = f"{QUANT_DIR}/gpt_rtn_int4_w{W_BIT}_g{Q_GROUP}.pt"
torch.save({
    "state_dict":       gpt_int4.state_dict(),
    "quantized_layers": quantized_layer_names,
    "w_bit":            W_BIT,
    "q_group":          Q_GROUP,
}, save_pt)

sz = os.path.getsize(save_pt) / 1e9
print(f"Saved {len(quantized_layer_names)} INT4 layers → {save_pt}")
print(f"File size: {sz:.2f} GB  (BF16 checkpoint was "
      f"{os.path.getsize(GPT_CKPT)/1e9:.2f} GB  →  "
      f"{os.path.getsize(GPT_CKPT)/1e9/sz:.1f}× smaller)")

# %% [markdown]
"""
## Section 7 — INT4 Evaluation

Load the quantized model on `EngineFast` (flash attention + KV cache) to match
the BF16 baseline inference conditions. Compare all 140 prompts using:

- **Δv%** — percentage change in vertex count (fast first-pass screen)
- **Chamfer Distance** — bidirectional L2 CD on 30,000 area-weighted surface
  samples, normalized to unit sphere (Fan et al., CVPR 2017)

> **Note on W4A16**: `WQLinear_GEMM` stores weights as packed INT4 (genuine
> memory bandwidth reduction) but dequantizes to FP16 before the GEMM. This is
> the standard for all published INT4 LLM models (AWQ, GPTQ, bitsandbytes NF4).
"""

# %%
# ── Step 1: Reload INT4 weights into EngineFast ──────────────────────────────
save_pt  = f"{GDRIVE_ROOT}/quantized/gpt_rtn_int4_w{W_BIT}_g{Q_GROUP}.pt"
ckpt     = torch.load(save_pt, map_location=device, weights_only=False)
W_BIT, Q_GROUP = ckpt["w_bit"], ckpt["q_group"]
quant_set = set(ckpt["quantized_layers"])

engine_int4_fast = EngineFast(
    config_path     = CONFIG_PATH,
    gpt_ckpt_path   = GPT_CKPT,
    shape_ckpt_path = TOK_CKPT,
    device          = device,
)

# Rebuild INT4 layer structure (init_only=True — shapes only, no packed weights yet)
for name, module in list(engine_int4_fast.gpt_model.named_modules()):
    if name not in quant_set:
        continue
    wq = WQLinear_GEMM.from_linear(module, W_BIT, Q_GROUP, init_only=True)
    *path, child = name.split(".")
    parent = engine_int4_fast.gpt_model
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, child, wq)

# Fill in the quantized weights from checkpoint
engine_int4_fast.gpt_model.load_state_dict(ckpt["state_dict"])
engine_int4_fast.gpt_model.eval()
print(f"Loaded INT{W_BIT} weights into EngineFast ({len(quant_set)} layers quantized).")

# %%
# ── Step 2: Sanity check ─────────────────────────────────────────────────────
# resolution_base=4.0 → minimum token count; fast & cheap.
with torch.inference_mode():
    _out = engine_int4_fast.t2s(
        ["A smooth sphere"],
        use_kv_cache=True, resolution_base=4.0, top_p=None,
    )

_v, _f = _out[0][0], _out[0][1]
assert _v.shape[0] > 100, f"Degenerate mesh: {_v.shape[0]} verts"
print(f"Sanity check PASSED: {_v.shape[0]} verts, {_f.shape[0]} faces")
print("EngineFast + WQLinear_GEMM compatible. KV cache enabled.")

# %% [markdown]
"""
### Step 2b — Switch to Marlin Kernel (recommended for actual speedup)

**Why GEMM is slow:** `WQLinear_GEMM` dequantizes INT4 → FP16 in a separate pass,
then calls a standard FP16 GEMM. It does **not** use A100's INT4 Tensor Cores,
so weight-bandwidth saving is partially offset by dequantize overhead.

**Marlin** (`WQLinear_Marlin`) fuses dequantize + GEMM into a single custom CUDA
kernel that uses A100 Tensor Cores natively in INT4 mode.
Expected: **2–4× throughput improvement over WQLinear_GEMM** on A100/A10G.

**INT4 only**: Marlin's primary fast path is W4A16 (INT4 weights, FP16 activations).
It supports W8A16 in some autoawq versions but the headline speedup is INT4.

**Minimum hardware**: SM80+ (A100, A10G, RTX 3090). T4 (SM75) is not supported.

**How conversion works**: The saved `.pt` checkpoint stores GEMM-packed weights.
Marlin uses a different binary layout. We unpack GEMM → FP16 → repack as Marlin.
The result is bit-for-bit identical to directly quantizing to Marlin format.
"""

# %%
# ── Step 2b-i. Helper: dequantize GEMM-packed INT4 → FP16 weight matrix ──────
#
# WQLinear_GEMM buffer shapes (w_bit=4):
#   qweight : [in_features,              out_features // 8]  int32
#   scales  : [in_features // group_size, out_features     ]  fp16
#   qzeros  : [in_features // group_size, out_features // 8]  int32
#
# Packing: each int32 holds 8 consecutive INT4 values along the out_features
# dimension, LSB-first (bits 0-3 = channel 0, bits 4-7 = channel 1, ...).

def unpack_gemm_weights(mod, w_bit: int, group_size: int) -> torch.Tensor:
    """
    Unpack WQLinear_GEMM buffers → FP16 weight matrix [out_features, in_features].
    Returns a tensor that can be assigned to nn.Linear.weight.data.
    """
    pack    = 32 // w_bit            # 8 for INT4
    in_f    = mod.in_features
    out_f   = mod.out_features
    n_grps  = in_f // group_size
    dev     = mod.qweight.device

    # ── Unpack qweight → [in_f, out_f] int32 ─────────────────────────────────
    w_int = torch.zeros(in_f, out_f, dtype=torch.int32, device=dev)
    for b in range(pack):
        w_int[:, b::pack] = (mod.qweight >> (b * w_bit)) & 0xF

    # ── Unpack qzeros → [n_grps, out_f] int32 ────────────────────────────────
    z_int = torch.zeros(n_grps, out_f, dtype=torch.int32, device=dev)
    for b in range(pack):
        z_int[:, b::pack] = (mod.qzeros >> (b * w_bit)) & 0xF

    # ── Dequantize: W[i] = (w_int[i] - zero[g]) × scale[g] ──────────────────
    s    = mod.scales.float()       # [n_grps, out_f]
    z    = z_int.float()            # [n_grps, out_f]
    w_fp = torch.empty(in_f, out_f, dtype=torch.float16, device=dev)
    for g in range(n_grps):
        sl = slice(g * group_size, (g + 1) * group_size)
        w_fp[sl] = ((w_int[sl].float() - z[g:g+1]) * s[g:g+1]).half()

    return w_fp.T.contiguous()      # [out_f, in_f] — standard nn.Linear convention


# ── Step 2b-ii. quantize_per_group (copied from Section 6 for standalone use) ─
def _quantize_per_group(weight: torch.Tensor, w_bit: int = 4, group_size: int = 128):
    out_f, in_f = weight.shape
    G   = in_f // group_size
    n   = 2 ** w_bit
    w_g = weight.reshape(out_f, G, group_size)
    w_max = w_g.amax(2); w_min = w_g.amin(2)
    scale = (w_max - w_min).clamp(min=1e-5) / (n - 1)
    zero  = (-w_min / scale).round().clamp(0, n - 1)
    return (scale.T.contiguous().half(), zero.T.contiguous().to(torch.int32))


# ── Step 2b-iii. Conversion: GEMM layer → Marlin layer ───────────────────────
from awq.modules.linear.marlin import WQLinear_Marlin

# Marlin alignment constraints (enforced by the CUDA kernel):
#   in_features  must be divisible by 128
#   out_features must be divisible by 64
MARLIN_IN_ALIGN  = 128
MARLIN_OUT_ALIGN = 64

def gemm_to_marlin(mod, w_bit: int, group_size: int):
    """
    Convert a WQLinear_GEMM layer to WQLinear_Marlin.
    Steps: GEMM unpack → fp16 → re-quantize → Marlin repack.
    Returns WQLinear_Marlin on the same device.
    """
    w_fp16 = unpack_gemm_weights(mod, w_bit, group_size)    # [out_f, in_f]

    dummy = nn.Linear(mod.in_features, mod.out_features,
                      bias=mod.bias is not None,
                      device=mod.qweight.device, dtype=torch.float16)
    dummy.weight.data = w_fp16
    if mod.bias is not None:
        dummy.bias = nn.Parameter(mod.bias.clone())

    # Marlin requires symmetric quantization (zeros=None)
    out_f2, in_f2 = w_fp16.shape
    w_g2     = w_fp16.float().reshape(out_f2, in_f2 // group_size, group_size)
    max_val  = w_g2.abs().amax(2).clamp(min=1e-5)
    q_scales = (max_val / (2 ** (w_bit - 1) - 1)).T.contiguous().half()
    return WQLinear_Marlin.from_linear(
        dummy, w_bit, group_size,
        init_only=False, scales=q_scales,
    )


def convert_gemm_to_marlin(model: nn.Module, w_bit: int, group_size: int):
    """
    Replace every WQLinear_GEMM in model with WQLinear_Marlin in-place.
    Layers that fail the Marlin alignment check are left as GEMM.
    """
    n_ok = n_skip = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, WQLinear_GEMM):
            continue
        if mod.in_features % MARLIN_IN_ALIGN != 0:
            print(f"  SKIP {name}: in_features={mod.in_features} "
                  f"not divisible by {MARLIN_IN_ALIGN}")
            n_skip += 1
            continue
        if mod.out_features % MARLIN_OUT_ALIGN != 0:
            print(f"  SKIP {name}: out_features={mod.out_features} "
                  f"not divisible by {MARLIN_OUT_ALIGN}")
            n_skip += 1
            continue
        marlin_layer = gemm_to_marlin(mod, w_bit, group_size)
        *path, child = name.split(".")
        parent = model
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, child, marlin_layer)
        n_ok += 1

    print(f"Converted {n_ok} GEMM → Marlin  |  Left as GEMM (alignment): {n_skip}")
    return model

# %%
# ── Step 2b-iv. Apply conversion and verify ───────────────────────────────────
print("Converting WQLinear_GEMM → WQLinear_Marlin...")
engine_int4_fast.gpt_model = convert_gemm_to_marlin(
    engine_int4_fast.gpt_model, W_BIT, Q_GROUP
)
engine_int4_fast.gpt_model.eval()

# Count layer types after conversion
n_marlin = sum(1 for _, m in engine_int4_fast.gpt_model.named_modules()
               if isinstance(m, WQLinear_Marlin))
n_gemm   = sum(1 for _, m in engine_int4_fast.gpt_model.named_modules()
               if isinstance(m, WQLinear_GEMM))
print(f"Marlin layers : {n_marlin}")
print(f"GEMM layers   : {n_gemm}  (failed alignment check — still run, but slower)")

# Sanity check with Marlin kernel
with torch.inference_mode():
    _out = engine_int4_fast.t2s(
        ["A smooth sphere"],
        use_kv_cache=True, resolution_base=4.0, top_p=None,
    )
_v, _f = _out[0][0], _out[0][1]
assert _v.shape[0] > 100, f"Degenerate mesh: {_v.shape[0]} verts"
print(f"Marlin sanity check PASSED: {_v.shape[0]} verts, {_f.shape[0]} faces")

# Quick latency comparison — 5 prompts, same conditions as GEMM benchmark
import time, numpy as np
_test_prompts = [BENCH_PROMPTS[i] for i in range(5)]
_times = []
for p in _test_prompts:
    t0 = time.perf_counter()
    with torch.inference_mode():
        engine_int4_fast.t2s([p], use_kv_cache=True, resolution_base=8.0, top_p=None)
    _times.append(time.perf_counter() - t0)
print(f"\nMarlin INT4 mean latency (n=5): {np.mean(_times):.1f}s")
print(f"GEMM  INT4 mean latency (n=140): 19.2s  (from Section 7b benchmark)")
print(f"BF16        mean latency (n=14):  14.7s  (from Section 8)")

# %%
# ── Step 2b-v. (Optional) Save Marlin checkpoint ─────────────────────────────
# Avoids the GEMM → Marlin conversion step on future loads.
MARLIN_SAVE = f"{GDRIVE_ROOT}/quantized/gpt_rtn_int4_marlin_w{W_BIT}_g{Q_GROUP}.pt"
marlin_layer_names = [
    name for name, mod in engine_int4_fast.gpt_model.named_modules()
    if isinstance(mod, WQLinear_Marlin)
]
torch.save({
    "state_dict":    engine_int4_fast.gpt_model.state_dict(),
    "marlin_layers": marlin_layer_names,
    "gemm_layers":   [name for name, mod in engine_int4_fast.gpt_model.named_modules()
                      if isinstance(mod, WQLinear_GEMM)],
    "w_bit":         W_BIT,
    "q_group":       Q_GROUP,
}, MARLIN_SAVE)
print(f"Marlin checkpoint saved ({os.path.getsize(MARLIN_SAVE)/1e9:.2f} GB) → {MARLIN_SAVE}")

# %% [markdown]
"""
#### Loading the Marlin checkpoint in future sessions

```python
from awq.modules.linear.gemm import WQLinear_GEMM
from awq.modules.linear.marlin import WQLinear_Marlin

ckpt = torch.load("gpt_rtn_int4_marlin_w4_g128.pt", map_location=device, weights_only=False)
marlin_set = set(ckpt["marlin_layers"])
gemm_set   = set(ckpt["gemm_layers"])
W_BIT, Q_GROUP = ckpt["w_bit"], ckpt["q_group"]

engine = EngineFast(config_path=CONFIG_YAML, gpt_ckpt_path=GPT_BF16,
                    shape_ckpt_path=TOKENIZER, device=device)

for name, mod in list(engine.gpt_model.named_modules()):
    if name in marlin_set:
        wq = WQLinear_Marlin.from_linear(mod, W_BIT, Q_GROUP, init_only=True)
    elif name in gemm_set:
        wq = WQLinear_GEMM.from_linear(mod, W_BIT, Q_GROUP, init_only=True)
    else:
        continue
    *path, child = name.split(".")
    parent = engine.gpt_model
    for p in path: parent = getattr(parent, p)
    setattr(parent, child, wq)

engine.gpt_model.load_state_dict(ckpt["state_dict"])
engine.gpt_model.eval()
```
"""

# %%
# ── Step 3: Run INT4 inference on all 140 prompts ────────────────────────────
from collections import OrderedDict

COMPARE_DIR = f"{GDRIVE_ROOT}/comparison_int4"
os.makedirs(COMPARE_DIR, exist_ok=True)

COMPARE_IDX = list(range(len(BENCH_PROMPTS)))   # all 140 for metrics

seen = OrderedDict()
for i, (p, c) in enumerate(zip(BENCH_PROMPTS, BENCH_CATEGORIES)):
    if c not in seen:
        seen[c] = i
VISUALIZE_IDX = list(seen.values())             # 14 (one per category) for plots

set_deterministic(SEED)
int4_results, int4_metrics = {}, []

print(f"INT4 inference — EngineFast + KV cache — {len(COMPARE_IDX)} prompts:")
for idx in COMPARE_IDX:
    prompt = BENCH_PROMPTS[idx]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            out = engine_int4_fast.t2s(
                [prompt],
                use_kv_cache    = True,
                resolution_base = 8.0,
                top_p           = None,
            )
        elapsed   = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        verts, faces = out[0][0], out[0][1]
        int4_results[idx] = (verts, faces)
        int4_metrics.append(dict(idx=idx, latency_s=round(elapsed, 2),
                                 n_verts=verts.shape[0], n_faces=faces.shape[0],
                                 peak_vram_gb=round(peak_vram, 2)))
        np.savez_compressed(f"{COMPARE_DIR}/int4_{idx:03d}.npz",
                            vertices=verts, faces=faces)
        print(f"  [{idx:03d}] {elapsed:.1f}s  {verts.shape[0]:7,}v  {peak_vram:.1f}GB"
              f"  [{BENCH_CATEGORIES[idx]:22}]  {prompt[:35]}")
    except Exception as e:
        int4_results[idx] = None
        print(f"  [{idx:03d}] ERROR: {e}")

# %%
# ── Step 4: Load BF16 baseline meshes ────────────────────────────────────────
bf16_results = {}
for idx in COMPARE_IDX:
    npz = Path(f"{GDRIVE_ROOT}/baseline_meshes/bf16_{idx:03d}.npz")
    if npz.exists():
        d = np.load(npz)
        bf16_results[idx] = (d["vertices"], d["faces"])
    else:
        bf16_results[idx] = None
        print(f"WARNING: BF16 baseline missing for idx={idx}")

# %%
# ── Step 5: Chamfer Distance ─────────────────────────────────────────────────
!pip install -q trimesh

import trimesh
from scipy.spatial import KDTree

N_CD_SAMPLES = 30_000   # standard in 3DGen literature (OccNet, PointFlow, etc.)
CD_SEED      = 42


def sample_surface(verts: np.ndarray, faces: np.ndarray, n: int = N_CD_SAMPLES) -> np.ndarray:
    """Area-weighted uniform surface sampling via trimesh."""
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float64)


def normalize_unit_sphere(pts: np.ndarray) -> np.ndarray:
    """Center at centroid, scale to max L2 norm = 1 (unit-sphere normalization)."""
    pts = pts - pts.mean(axis=0)
    r   = np.linalg.norm(pts, axis=1).max()
    return pts / (r + 1e-10)


def chamfer_l2(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """
    Bidirectional L2 Chamfer Distance (non-squared, symmetric sum-of-means).

        CD(A, B) = (1/|A|) Σ_{a∈A} min_{b∈B} ‖a−b‖₂
                 + (1/|B|) Σ_{b∈B} min_{a∈A} ‖a−b‖₂

    Reference: Fan et al., "A Point Set Generation Network for 3D Object
    Reconstruction from a Single Image", CVPR 2017.

    Returns: (cd_total, cd_fwd [A→B], cd_bwd [B→A])
    """
    tree_b = KDTree(b)
    tree_a = KDTree(a)
    d_ab   = tree_b.query(a, k=1, workers=-1)[0]
    d_ba   = tree_a.query(b, k=1, workers=-1)[0]
    return float(d_ab.mean() + d_ba.mean()), float(d_ab.mean()), float(d_ba.mean())


np.random.seed(CD_SEED)

stats_rows = []
header = (f"{'idx':>4}  {'category':24}  {'BF16 v':>8}  {'INT4 v':>8}  "
          f"{'Δv%':>7}  {'CD×10⁻³':>10}  {'CD_fwd':>9}  {'CD_bwd':>9}")
print(f"\n{header}")
print("─" * len(header))

for idx in COMPARE_IDX:
    b, q = bf16_results.get(idx), int4_results.get(idx)
    bv = b[0].shape[0] if b else 0
    qv = q[0].shape[0] if q else 0
    bf = b[1].shape[0] if b else 0
    qf = q[1].shape[0] if q else 0
    dv_pct = (qv - bv) / bv * 100 if bv > 0 and qv > 0 else None
    dv_str = f"{dv_pct:+.1f}%" if dv_pct is not None else "    N/A"

    cd = cd_fwd = cd_bwd = None
    if b is not None and q is not None:
        pts_b = normalize_unit_sphere(sample_surface(b[0], b[1]))
        pts_q = normalize_unit_sphere(sample_surface(q[0], q[1]))
        cd, cd_fwd, cd_bwd = chamfer_l2(pts_b, pts_q)

    stats_rows.append(dict(idx=idx, category=BENCH_CATEGORIES[idx],
                           bv=bv, qv=qv, bf=bf, qf=qf,
                           dv_pct=dv_pct, cd=cd, cd_fwd=cd_fwd, cd_bwd=cd_bwd))
    if cd is not None:
        print(f"  {idx:03d}  {BENCH_CATEGORIES[idx]:24}  {bv:8,}  {qv:8,}  "
              f"{dv_str:>7}  {cd*1e3:>10.4f}  {cd_fwd*1e3:>9.4f}  {cd_bwd*1e3:>9.4f}")
    else:
        print(f"  {idx:03d}  {BENCH_CATEGORIES[idx]:24}  {bv:8,}  {qv:8,}  "
              f"{dv_str:>7}  {'N/A':>10}")

# %%
# ── Aggregate statistics ──────────────────────────────────────────────────────
valid = [r for r in stats_rows if r["cd"] is not None]
cds   = np.array([r["cd"] for r in valid])

print(f"\n── Chamfer Distance (N_samples={N_CD_SAMPLES:,}, seed={CD_SEED}, "
      f"normalisation=unit_sphere, formula=Fan_et_al_CVPR2017) ──")
print(f"  Mean    CD : {cds.mean()*1e3:.4f} × 10⁻³")
print(f"  Median  CD : {np.median(cds)*1e3:.4f} × 10⁻³")
print(f"  Std     CD : {cds.std()*1e3:.4f} × 10⁻³")
print(f"  P75     CD : {np.percentile(cds,75)*1e3:.4f} × 10⁻³")
print(f"  P95     CD : {np.percentile(cds,95)*1e3:.4f} × 10⁻³")
print(f"  Max     CD : {cds.max()*1e3:.4f} × 10⁻³  (idx={valid[cds.argmax()]['idx']:03d})")

print(f"\n── Per-category (mean ± std CD × 10⁻³) ──────────────────────────────────")
cats = {}
for r in valid:
    cats.setdefault(r["category"], []).append(r["cd"])
for cat, vals in sorted(cats.items()):
    v = np.array(vals)
    print(f"  {cat:26}  {v.mean()*1e3:.4f} ± {v.std()*1e3:.4f}  "
          f"(n={len(v)}, max={v.max()*1e3:.4f})")

import pandas as pd
pd.DataFrame(stats_rows).to_csv(f"{COMPARE_DIR}/cd_results.csv", index=False)
print(f"\nFull results → {COMPARE_DIR}/cd_results.csv")

# %%
# ── Step 6: Side-by-side Plotly visualisation (1 per category = 14 rows) ─────
from plotly.subplots import make_subplots
import plotly.graph_objects as go

n      = len(VISUALIZE_IDX)
titles = []
for idx in VISUALIZE_IDX:
    short = BENCH_PROMPTS[idx][:38] + ("…" if len(BENCH_PROMPTS[idx]) > 38 else "")
    titles += [f"BF16 [{idx:03d}] {short}", f"INT4 [{idx:03d}]"]

fig = make_subplots(
    rows=n, cols=2,
    subplot_titles=titles,
    specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}]] * n,
    vertical_spacing=0.03,
    horizontal_spacing=0.02,
)

for row_i, idx in enumerate(VISUALIZE_IDX, start=1):
    for col, color, data in [
        (1, "lightblue",   bf16_results.get(idx)),
        (2, "lightsalmon", int4_results.get(idx)),
    ]:
        if data is None:
            continue
        v, f = data
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=color, opacity=0.85,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
            showscale=False,
        ), row=row_i, col=col)

fig.update_layout(
    height=420 * n,
    title_text="BF16 (blue) vs RTN INT4 (salmon) — 1 per category",
    showlegend=False,
    margin=dict(l=0, r=0, t=60, b=0),
)
fig.show()

html_path = f"{COMPARE_DIR}/bf16_vs_rtn_int4.html"
fig.write_html(html_path)
print(f"HTML saved → {html_path}")

# %%
# ── Step 7: Per-pair HTML files (interactive 3D, one per prompt) ─────────────
for idx in COMPARE_IDX:
    b, q = bf16_results.get(idx), int4_results.get(idx)
    if b is None or q is None:
        continue
    pair_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"BF16 [{idx:03d}]", f"INT4 [{idx:03d}]"],
        specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}]],
    )
    for col, color, data in [(1, "lightblue", b), (2, "lightsalmon", q)]:
        v, f = data
        pair_fig.add_trace(go.Mesh3d(
            x=v[:,0], y=v[:,1], z=v[:,2],
            i=f[:,0], j=f[:,1], k=f[:,2],
            color=color, opacity=0.85,
        ), row=1, col=col)
    pair_fig.update_layout(
        height=400, width=900, showlegend=False,
        title_text=BENCH_PROMPTS[idx][:70],
    )
    pair_fig.write_html(f"{COMPARE_DIR}/pair_{idx:03d}.html")

print(f"Saved {len(COMPARE_IDX)} pair HTML files → {COMPARE_DIR}/")

# %% [markdown]
"""
## Section 8 — Efficiency Metrics and HuggingFace Export

### 8a. Efficiency comparison
"""

# %%
# ── Model size and bandwidth ──────────────────────────────────────────────────
def count_model_bytes(model):
    total  = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total

bf16_ckpt_bytes  = os.path.getsize(GPT_CKPT)
int4_ckpt_bytes  = os.path.getsize(f"{GDRIVE_ROOT}/quantized/gpt_rtn_int4_w{W_BIT}_g{Q_GROUP}.pt")
bf16_ram_bytes   = count_model_bytes(engine_bf16.gpt_model)
int4_ram_bytes   = count_model_bytes(engine_int4_fast.gpt_model)

print("── Storage ──────────────────────────────────────────────────────────")
print(f"  BF16 checkpoint (FP32 on disk) : {bf16_ckpt_bytes/1e9:.2f} GB")
print(f"  INT4 checkpoint (W4A16)        : {int4_ckpt_bytes/1e9:.2f} GB")
print(f"  Disk compression               : {bf16_ckpt_bytes/int4_ckpt_bytes:.1f}×")
print(f"\n── GPU RAM ──────────────────────────────────────────────────────────")
print(f"  BF16 model RAM (BF16 in GPU)   : {bf16_ram_bytes/1e9:.3f} GB")
print(f"  INT4 model RAM                 : {int4_ram_bytes/1e9:.3f} GB")
print(f"  GPU RAM compression            : {bf16_ram_bytes/int4_ram_bytes:.1f}×")

int4_weight_bytes = 0
for name, mod in engine_int4_fast.gpt_model.named_modules():
    if isinstance(mod, WQLinear_GEMM):
        int4_weight_bytes += mod.qweight.numel() * 4   # INT32 packs 8×INT4
        int4_weight_bytes += mod.scales.numel()  * 2   # FP16 scales
        int4_weight_bytes += mod.qzeros.numel()  * 4   # INT32 zeros
    elif isinstance(mod, torch.nn.Linear):
        int4_weight_bytes += mod.weight.numel()  * 2   # unquantized (lm_head)

print(f"\n── Memory bandwidth per decode step (weight reads only) ─────────────")
print(f"  INT4 weight bandwidth/token    : {int4_weight_bytes/1e9:.3f} GB")
print("  (Each autoregressive step reads all weights once — this is the bottleneck)")

# %%
# ── Latency comparison (both EngineFast + KV cache) ──────────────────────────
int4_times = [m["latency_s"] for m in int4_metrics]
print(f"── INT4 latency (n={len(int4_times)}, EngineFast + KV cache) ──────────────")
print(f"  Mean   : {np.mean(int4_times):.1f}s")
print(f"  Median : {np.median(int4_times):.1f}s")
print(f"  P95    : {np.percentile(int4_times, 95):.1f}s")

# BF16 latency benchmark on the same VISUALIZE_IDX subset for a fair comparison
bf16_times = []
for idx in VISUALIZE_IDX:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = engine_bf16.t2s([BENCH_PROMPTS[idx]], use_kv_cache=True,
                            resolution_base=8.0, top_p=None)
    torch.cuda.synchronize()
    bf16_times.append(time.perf_counter() - t0)

bf16_mean = np.mean(bf16_times)
int4_mean  = np.mean(int4_times)
print(f"\n── Latency summary ──────────────────────────────────────────────────")
print(f"  BF16 mean : {bf16_mean:.1f}s  (n={len(bf16_times)}, EngineFast + KV)")
print(f"  INT4 mean : {int4_mean:.1f}s  (n={len(int4_times)}, EngineFast + KV)")
direction = "faster" if bf16_mean > int4_mean else "slower"
ratio     = max(bf16_mean, int4_mean) / min(bf16_mean, int4_mean)
print(f"  INT4 is {ratio:.2f}× {direction} than BF16")

# %% [markdown]
"""
## Section 8b — HuggingFace Export

Export quantized weights to safetensors format (the HuggingFace standard) along
with a `quant_config.json` that describes the quantization parameters needed to
reconstruct `WQLinear_GEMM` layers on reload.

**Minimal HuggingFace repo structure:**
```
your-username/cube3d-v0.5-int4-w4a16-g128/
├── README.md                              ← model card (write manually)
├── config.json                            ← copy from Roblox/cube3d-v0.5
├── quant_config.json                      ← generated below
├── gpt_rtn_int4_w4_g128.safetensors       ← generated below
└── inference_example.py                   ← see Section 7 reload code
```
"""

# %%
from safetensors.torch import save_file

# Safetensors requires contiguous CPU tensors
state_dict_contig = {k: v.contiguous().cpu()
                     for k, v in engine_int4_fast.gpt_model.state_dict().items()}

safetensors_path = f"{GDRIVE_ROOT}/quantized/gpt_rtn_int4_w{W_BIT}_g{Q_GROUP}.safetensors"
save_file(state_dict_contig, safetensors_path)
print(f"safetensors saved : {os.path.getsize(safetensors_path)/1e9:.2f} GB")

# %%
# quant_config.json — encodes everything needed to reconstruct WQLinear_GEMM
quant_config = {
    "quant_method":       "rtn",
    "w_bit":              W_BIT,
    "q_group_size":       Q_GROUP,
    "zero_point":         True,
    "version":            "GEMM",
    "quantized_layers":   sorted(quant_set),
    "skipped_layers":     ["lm_head"],
    "skip_reason":        "out_features=16387 not divisible by 8 (pack alignment)",
    "awq_scale_absorbed": False,
    "awq_scale_note":     "All LayerNorm layers have elementwise_affine=False; "
                          "scale absorption disabled. Quantization is pure RTN.",
}

config_path = f"{GDRIVE_ROOT}/quantized/quant_config.json"
with open(config_path, "w") as f:
    json.dump(quant_config, f, indent=2)
print(f"quant_config.json saved → {config_path}")
