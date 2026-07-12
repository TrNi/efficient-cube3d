#!/usr/bin/env python3
"""
Generate a Roblox-style rotating GIF comparing Cube Model BF16 vs INT4 outputs.

- Layout: 2 rows × 4 columns per category slide.
  - Top row: Cube Model BFloat16 meshes.
  - Bottom row: INT4 Quantized meshes.
- Each slide shows the first 4 prompts in that category (only if both BF16 and
  INT4 meshes exist) and rotates all meshes about the Y axis in the same
  direction.
- Color: vertex norm → custom turbo colormap clipped to the middle 90% of the
  original turbo colors (5th–95th percentile). Values are untouched; only the
  colormap is trimmed before use so that (0,0,0) still maps to a real color.
- Duration: 4 seconds per category (configurable) with smooth rotation.
- Output: GIF saved under scripts/visuals by default.

Usage (from repo root):
    python scripts/visuals/make_category_gif.py \
        --drive_root . \
        --output scripts/visuals/category_grid.gif \
        --fps 8 --seconds-per-category 4
"""

import argparse
import gc
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")  # only for colormap
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image, ImageDraw, ImageFont


RNG = np.random.default_rng(0)


@dataclass
class MeshView:
    vertices: np.ndarray
    faces: np.ndarray
    face_colors: np.ndarray
    prompt: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Create BF16 vs INT4 rotating GIF")
    parser.add_argument(
        "--drive_root",
        type=Path,
        default=repo_root,
        help="Path to repo root containing baseline_meshes, comparison_int4, benchmark_prompts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "scripts" / "visuals" / "category_grid.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "scripts" / "visuals" / "category_gifs",
        help="Directory for per-category GIFs (used when splitting)",
    )
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the GIF")
    parser.add_argument(
        "--seconds-per-category", type=float, default=4.0, help="Seconds to show each category slide"
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="y",
        choices=["x", "y", "z"],
        help="Rotation axis (same for all meshes)",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=1.35,
        help="Multiply face colors by this factor (clipped to 0-1)",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=20000,
        help="Maximum faces per mesh to render (downsamples if exceeded; 0 disables)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.6,
        help="Zoom factor for mesh view (1.0 = fit, 2.0 = 2x zoom)",
    )
    parser.add_argument(
        "--snapshots-only",
        action="store_true",
        help="Render snapshots (PNGs) only; skip GIF writing",
    )
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=repo_root / "scripts" / "visuals" / "category_pngs",
        help="Directory to save per-angle PNG snapshots",
    )
    parser.add_argument(
        "--snapshots-views",
        type=int,
        default=8,
        help="Number of evenly spaced rotation views to snapshot when snapshots-only is set",
    )
    parser.add_argument(
        "--split-per-category",
        dest="split_per_category",
        action="store_true",
        help="Write one GIF per category instead of a single combined GIF (default)",
    )
    parser.add_argument(
        "--single-gif",
        dest="split_per_category",
        action="store_false",
        help="Combine all categories into one GIF at --output",
    )
    parser.set_defaults(split_per_category=True)
    return parser.parse_args()


def load_metadata(master_suite: Path) -> Tuple[List[str], List[str], List[str]]:
    with master_suite.open("r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data["prompts"]
    categories = data["categories"]
    category_names = data.get("category_names", sorted(set(categories)))
    return prompts, categories, category_names


def clipped_turbo() -> LinearSegmentedColormap:
    base = cm.get_cmap("turbo")
    # Use only the middle 90% of the colormap (5th–95th percentile of colors)
    samples = base(np.linspace(0.05, 0.95, 256))
    return LinearSegmentedColormap.from_list("turbo_clipped", samples)


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    center = vertices.mean(axis=0)
    span = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    scale = span if span > 0 else 1.0
    return (vertices - center) / scale


def compute_face_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    cmap: LinearSegmentedColormap,
    brightness: float,
) -> np.ndarray:
    norms = np.linalg.norm(vertices, axis=1)
    max_norm = float(norms.max()) if norms.size else 1.0
    normalized = np.clip(norms / max_norm, 0.0, 1.0)
    vertex_colors = cmap(normalized)
    face_colors = vertex_colors[faces].mean(axis=1)
    if brightness != 1.0:
        face_colors = face_colors.copy()
        face_colors[:, :3] = np.clip(face_colors[:, :3] * brightness, 0.0, 1.0)
    return face_colors


def slugify(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.lower())
    safe = "_".join(filter(None, safe.split("_")))
    return safe or "category"


def format_prompt_title(prompt: str, width: int = 26*2.2, max_lines: int = 2) -> str:
    lines = textwrap.wrap(prompt, width=width)
    if not lines:
        return ""
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1] + " …"]
    return "\n".join(lines)


def maybe_downsample_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if max_faces and faces.shape[0] > max_faces:
        stride = math.ceil(faces.shape[0] / max_faces)
        return faces[::stride][:max_faces]
    return faces


def load_mesh(
    npz_path: Path,
    prompt: str,
    cmap: LinearSegmentedColormap,
    max_faces: int,
    brightness: float,
) -> MeshView:
    data = np.load(npz_path)
    vertices = data["vertices"].astype(np.float32)
    faces = data["faces"].astype(np.int32)
    faces = maybe_downsample_faces(faces, max_faces)
    vertices = normalize_vertices(vertices)
    face_colors = compute_face_colors(vertices, faces, cmap, brightness=brightness)
    return MeshView(vertices=vertices, faces=faces, face_colors=face_colors, prompt=prompt)


def rotate_vertices(vertices: np.ndarray, angle_rad: float, axis: str = "y") -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    if axis == "x":
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    elif axis == "y":
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    else:  # axis == "z"
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return vertices @ rot.T


def project_vertices(vertices: np.ndarray, elev: float = 20.0, azim: float = -35.0) -> Tuple[np.ndarray, np.ndarray]:
    """Orthographic projection. Returns (screen_xy, depth)."""
    er, ar = math.radians(elev), math.radians(azim)
    ce, se = math.cos(er), math.sin(er)
    ca, sa = math.cos(ar), math.sin(ar)
    x = vertices[:, 0] * ca - vertices[:, 1] * sa
    y = vertices[:, 0] * sa + vertices[:, 1] * ca
    z = vertices[:, 2]
    screen_x = x
    screen_y = y * ce - z * se
    depth = y * se + z * ce
    return np.column_stack([screen_x, screen_y]), depth


def _load_font(size: int = 14) -> ImageFont.FreeTypeFont:
    for p in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


_FONTS: dict = {}


def _font(size: int = 14) -> ImageFont.FreeTypeFont:
    if size not in _FONTS:
        _FONTS[size] = _load_font(size)
    return _FONTS[size]


def render_mesh_to_array(mesh: MeshView, angle_rad: float, axis: str, size: int = 350, zoom: float = 1.6) -> np.ndarray:
    rotated = rotate_vertices(mesh.vertices, angle_rad, axis)
    xy, z = project_vertices(rotated)

    cx = float((xy[:, 0].max() + xy[:, 0].min()) / 2)
    cy = float((xy[:, 1].max() + xy[:, 1].min()) / 2)
    rx = float((xy[:, 0].max() - xy[:, 0].min()) / 2)
    ry = float((xy[:, 1].max() - xy[:, 1].min()) / 2)
    r = max(rx, ry, 1e-6) / zoom

    margin = 5
    scale = (size - 2 * margin) / (2 * r)
    px = (xy[:, 0] - (cx - r)) * scale + margin
    py = size - ((xy[:, 1] - (cy - r)) * scale + margin)

    face_px = np.column_stack([px, py])[mesh.faces]
    face_z = z[mesh.faces].mean(axis=1)
    order = np.argsort(-face_z)

    # Pre-convert to Python lists for fast iteration (avoids numpy indexing overhead)
    polys = face_px[order].tolist()
    cols = (mesh.face_colors[order, :3] * 255).astype(np.uint8).tolist()

    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for tri, c in zip(polys, cols):
        draw.polygon(
            [(tri[0][0], tri[0][1]), (tri[1][0], tri[1][1]), (tri[2][0], tri[2][1])],
            fill=(c[0], c[1], c[2]),
        )

    return np.array(img)


def category_indices(
    categories: Sequence[str], category_names: Sequence[str], baseline_dir: Path, int4_dir: Path
) -> Dict[str, List[int]]:
    cat_to_idxs: Dict[str, List[int]] = {c: [] for c in category_names}
    for idx, cat in enumerate(categories):
        if cat in cat_to_idxs:
            bf = baseline_dir / f"bf16_{idx:03d}.npz"
            q = int4_dir / f"int4_{idx:03d}.npz"
            if bf.exists() and q.exists():
                cat_to_idxs[cat].append(idx)
    return {c: idxs[:4] for c, idxs in cat_to_idxs.items() if len(idxs) >= 4}


def render_frame(
    meshes: List[Tuple[MeshView, MeshView]],
    category_name: str,
    angle_rad: float,
    axis: str,
    fps: int,
    zoom: float = 1.6,
) -> np.ndarray:
    cell_size = 350
    row_label_h = 24
    prompt_h = 36
    col_gap = 6
    side_w = 100
    bottom_label_h = 30

    n_cols = len(meshes)
    top_h = row_label_h + cell_size
    mid_h = prompt_h
    bot_h = row_label_h + cell_size
    total_w = side_w + n_cols * cell_size + (n_cols - 1) * col_gap + side_w
    total_h = top_h + mid_h + bot_h + bottom_label_h

    img = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    f_title = _font(13)
    f_label = _font(16)
    f_cat = _font(16)
    f_prompt = _font(14)

    x0 = side_w
    for col, (bf_mesh, int4_mesh) in enumerate(meshes):
        bf_arr = render_mesh_to_array(bf_mesh, angle_rad, axis, cell_size, zoom)
        img.paste(Image.fromarray(bf_arr), (x0, row_label_h))

        int4_arr = render_mesh_to_array(int4_mesh, angle_rad, axis, cell_size, zoom)
        y_int4 = top_h + mid_h + row_label_h
        img.paste(Image.fromarray(int4_arr), (x0, y_int4))

        x0 += cell_size + col_gap

    draw.text((5, 5), "Cube Model BFloat16", fill=(255, 255, 255), font=f_label)
    draw.text((5, top_h + mid_h + 5), "INT4 Quantized", fill=(255, 255, 255), font=f_label)

    # Prompts between the two rows
    x0 = side_w
    for col, (bf_mesh, _) in enumerate(meshes):
        for li, line in enumerate(format_prompt_title(bf_mesh.prompt).split("\n")):
            draw.text((x0 + 2, top_h + 4 + li * 16), line, fill=(217, 217, 217), font=f_prompt)
        x0 += cell_size + col_gap

    cat_text = f"Category: {category_name}"
    cat_w = draw.textlength(cat_text, font=f_cat)
    draw.text((int((total_w - cat_w) / 2), total_h - bottom_label_h + 5), cat_text, fill=(187, 187, 187), font=f_cat)

    return np.array(img)


def main():
    args = parse_args()
    drive_root = args.drive_root.resolve()

    baseline_dir = drive_root / "baseline_meshes"
    int4_dir = drive_root / "comparison_int4"
    master_suite = drive_root / "benchmark_prompts" / "master_suite.json"

    if not master_suite.exists():
        raise FileNotFoundError(f"Master suite JSON not found: {master_suite}")
    if not baseline_dir.exists() or not int4_dir.exists():
        raise FileNotFoundError("Expected baseline_meshes and comparison_int4 under drive_root")

    prompts, categories, category_names = load_metadata(master_suite)
    selection = category_indices(categories, category_names, baseline_dir, int4_dir)
    if not selection:
        raise RuntimeError("No category has 4 overlapping BF16/INT4 meshes to visualize")

    cmap = clipped_turbo()
    frames_per_category = max(1, int(args.fps * args.seconds_per_category))
    angle_step = 2 * math.pi / frames_per_category

    if args.snapshots_only:
        args.snapshots_dir.mkdir(parents=True, exist_ok=True)
        views = max(1, args.snapshots_views)
        angle_step_snap = 2 * math.pi / views
        print(f"Writing snapshots to {args.snapshots_dir} ...")

        for category_name in category_names:
            idxs = selection.get(category_name)
            if not idxs:
                continue

            mesh_pairs: List[Tuple[MeshView, MeshView]] = []
            for idx in idxs:
                prompt = prompts[idx]
                bf_path = baseline_dir / f"bf16_{idx:03d}.npz"
                int4_path = int4_dir / f"int4_{idx:03d}.npz"
                bf_mesh = load_mesh(bf_path, prompt, cmap, args.max_faces, args.brightness)
                int4_mesh = load_mesh(int4_path, prompt, cmap, args.max_faces, args.brightness)
                mesh_pairs.append((bf_mesh, int4_mesh))

            slug = slugify(category_name)
            for view_idx in range(views):
                angle_rad = view_idx * angle_step_snap
                print(f"  [{slug}] rendering view {view_idx + 1}/{views} ...", end=" ", flush=True)
                frame = render_frame(mesh_pairs, category_name, angle_rad, args.axis, args.fps, args.zoom)
                out_path = args.snapshots_dir / f"{slug}_view{view_idx:02d}.png"
                imageio.imwrite(out_path, frame)
                gc.collect()
                print("saved")
            print(f"  Saved {views} views for {slug}")

        print("Done. Snapshots ready for stitching.")

    elif args.split_per_category:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing per-category GIFs to {args.output_dir} ...")

        for category_name in category_names:
            idxs = selection.get(category_name)
            if not idxs:
                continue

            mesh_pairs: List[Tuple[MeshView, MeshView]] = []
            for idx in idxs:
                prompt = prompts[idx]
                bf_path = baseline_dir / f"bf16_{idx:03d}.npz"
                int4_path = int4_dir / f"int4_{idx:03d}.npz"
                bf_mesh = load_mesh(bf_path, prompt, cmap, args.max_faces, args.brightness)
                int4_mesh = load_mesh(int4_path, prompt, cmap, args.max_faces, args.brightness)
                mesh_pairs.append((bf_mesh, int4_mesh))

            output_path = args.output_dir / f"{slugify(category_name)}.gif"
            with imageio.get_writer(output_path, mode="I", duration=1 / args.fps, loop=0) as writer:
                for frame_idx in range(frames_per_category):
                    angle_rad = frame_idx * angle_step
                    frame = render_frame(mesh_pairs, category_name, angle_rad, args.axis, args.fps, args.zoom)
                    writer.append_data(frame)
            print(f"  Saved {output_path.name}")

        print("Done. Open the GIFs to preview the rotating slides.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing combined GIF to {args.output} ...")

        with imageio.get_writer(args.output, mode="I", duration=1 / args.fps, loop=0) as writer:
            for category_name in category_names:
                idxs = selection.get(category_name)
                if not idxs:
                    continue

                mesh_pairs: List[Tuple[MeshView, MeshView]] = []
                for idx in idxs:
                    prompt = prompts[idx]
                    bf_path = baseline_dir / f"bf16_{idx:03d}.npz"
                    int4_path = int4_dir / f"int4_{idx:03d}.npz"
                    bf_mesh = load_mesh(bf_path, prompt, cmap, args.max_faces, args.brightness)
                    int4_mesh = load_mesh(int4_path, prompt, cmap, args.max_faces, args.brightness)
                    mesh_pairs.append((bf_mesh, int4_mesh))

                for frame_idx in range(frames_per_category):
                    angle_rad = frame_idx * angle_step
                    frame = render_frame(mesh_pairs, category_name, angle_rad, args.axis, args.fps, args.zoom)
                    writer.append_data(frame)

        print("Done. Open the GIF to preview the rotating slides.")


if __name__ == "__main__":
    main()
