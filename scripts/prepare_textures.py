#!/usr/bin/env python3
"""Make the model's textures from CC0 texture sets (ambientCG.com, CC0 1.0).

  prepare_textures.py SOURCE_DIR      (unzipped 1K-JPG sets, one folder per asset)

Writes textures/<material>_color.jpg and textures/<material>_normal.jpg (512 px), used by
blender/common.py for the GLBs (CRAMERA) and renders:

- "photo" materials use the set's colour image, scaled to the model's average colour
  (grass, gravel, concrete);
- "tint" materials keep the model's colour (blender/common.py COLORS) and take only the
  set's fine detail: its brightness variation, scaled down, times the colour. So a
  painted gearbox stays gear_paint, with the scratches and relief of real paint.
"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "blender"))
OUT = os.path.join(ROOT, "textures")
SIZE = 512

# material -> (ambientCG asset, mode, detail strength for "tint")
MATERIALS = {
    "grass": ("Grass004", "photo", None),
    "gravel": ("Gravel023", "photo", None),
    "concrete": ("Concrete034", "photo", None),
    "grout": ("Concrete034", "tint", 0.5),
    "grating": ("MetalWalkway011", "tint", 1.0),
    "steel_grey": ("Metal027", "tint", 0.35),
    "pipe_steel": ("Metal027", "tint", 0.35),
    "gear_paint": ("PaintedMetal004", "tint", 0.08),
    "generator_paint": ("PaintedMetal004", "tint", 0.08),
    "tower_paint": ("Plastic010", "tint", 0.25),
    "cabinet_grey": ("Plastic010", "tint", 0.25),
    "hatch_yellow": ("PaintedMetal004", "tint", 0.15),
    "grp_white": ("Plastic010", "tint", 0.3),
    "blade_white": ("Plastic010", "tint", 0.25),
}
ATTRIBUTION = """Textures made from CC0 1.0 material sets by ambientCG (https://ambientcg.com):
{assets}
See scripts/prepare_textures.py for how they are processed.
"""


def colours():
    """COLORS from blender/common.py without importing bpy."""
    src = open(os.path.join(ROOT, "blender", "common.py")).read()
    start = src.index("COLORS = {")
    return eval(src[start + len("COLORS = "):src.index("}", start) + 1])


def find(folder, suffix):
    return next(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(suffix))


def main():
    source = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    palette = colours()
    for name, (asset, mode, strength) in MATERIALS.items():
        folder = os.path.join(source, asset)
        colour = Image.open(find(folder, "_Color.jpg")).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
        if mode == "photo":        # the photo's pattern and hue, at the model's average colour
            rgb = np.asarray(colour, dtype=np.float32) / 255.0
            base = np.array(palette[name][:3], dtype=np.float32)
            rgb = np.clip(rgb * (base / np.maximum(rgb.reshape(-1, 3).mean(0), 1e-3)), 0.0, 1.0)
            colour = Image.fromarray((rgb * 255).astype(np.uint8))
        if mode == "tint":
            lum = np.asarray(colour.convert("L"), dtype=np.float32) / 255.0
            detail = 1.0 + strength * (lum / max(lum.mean(), 1e-3) - 1.0)
            detail = detail / max(detail.mean(), 1e-3)                        # average stays the model's colour
            base = np.array(palette[name][:3], dtype=np.float32)
            rgb = np.clip(detail[..., None] * base[None, None, :], 0.0, 1.0)
            colour = Image.fromarray((rgb * 255).astype(np.uint8))
        colour.save(os.path.join(OUT, f"{name}_color.jpg"), quality=85)
        normal = Image.open(find(folder, "_NormalGL.jpg")).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
        normal.save(os.path.join(OUT, f"{name}_normal.jpg"), quality=90)
        print(f"{name:16s} {asset:18s} {mode}")
    assets = sorted({a for a, _, _ in MATERIALS.values()})
    with open(os.path.join(OUT, "ATTRIBUTION.txt"), "w") as f:
        f.write(ATTRIBUTION.format(assets="\n".join(f"- {a}: https://ambientcg.com/view?id={a}" for a in assets)))


if __name__ == "__main__":
    main()
