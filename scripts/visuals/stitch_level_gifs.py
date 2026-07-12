#!/usr/bin/env python3
"""
Stitch per-category PNG snapshots into 3 GIFs (easy, medium, complex).

Levels are defined in HF/README.md based on Chamfer Distance:
  Easy   (CD < 75):     vehicle_land, geometric_primitive, animal_wild,
                        animal_domestic, tool_hardware, furniture, musical_instrument
  Medium (CD 75-100):   vehicle_air_water, fine_detail, electronics,
                        architecture, nature_plant
  Complex (CD > 100):   abstract_mathematical, symmetry_topology

Usage:
    python scripts/visuals/stitch_level_gifs.py \
        --snapshots-dir scripts/visuals/category_pngs \
        --output-dir scripts/visuals/level_gifs \
        --fps 2
"""

import argparse
from pathlib import Path

from PIL import Image

LEVELS = {
    "easy": [
        "vehicle_land",
        "geometric_primitive",
        "animal_wild",
        "animal_domestic",
        "tool_hardware",
        "furniture",
        "musical_instrument",
    ],
    "medium": [
        "vehicle_air_water",
        "fine_detail",
        "electronics",
        "architecture",
        "nature_plant",
    ],
    "complex": [
        "abstract_mathematical",
        "symmetry_topology",
    ],
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Stitch category PNGs into per-level GIFs")
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=repo_root / "scripts" / "visuals" / "category_pngs",
        help="Directory containing the per-category view PNGs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "scripts" / "visuals" / "level_gifs",
        help="Directory to save the 3 level GIFs",
    )
    parser.add_argument("--duration", type=float, default=1.0, help="Seconds per frame (e.g. 2.0 = 2s per frame)")
    parser.add_argument("--views", type=int, default=16, help="Number of views per category")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for level_name, categories in LEVELS.items():
        frames = []
        for cat in categories:
            for v in range(args.views):
                p = args.snapshots_dir / f"{cat}_view{v:02d}.png"
                if p.exists():
                    frames.append(Image.open(p).convert("RGB"))
                else:
                    print(f"  WARNING: missing {p.name}")

        if not frames:
            print(f"  No frames for {level_name}, skipping.")
            continue

        out_path = args.output_dir / f"{level_name}.gif"
        duration_cs = int(args.duration * 100)
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_cs,
            loop=0,
            disposal=2,
        )

        print(f"  {level_name}: {len(frames)} frames, {args.duration}s each -> {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
