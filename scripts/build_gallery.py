#!/usr/bin/env python3
"""Rebuild the picture-tour page (gallery/template.html) from the current model.

  python3 scripts/build_gallery.py [--skip-renders] [--skip-screenshots]

Writes preview/site/ (index.html + img/*.jpg + files.json):
- Blender renders of the healthy turbine and the fault scenarios, one image per
  inspection point that has a fault in them (scripts/render_preview.py);
- CRAMERA screenshots of the turbine scenes and the G1 recordings (headless
  Chromium; needs the viewer on localhost:8711, see view.sh);
- the fault browser data and counts, stamped with the date and git commit.

Publish the result to the page's existing URL (see README, "Picture tour").
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

import yaml
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from turbine import parts  # noqa: E402

OUT = os.path.join(ROOT, "preview", "site")
RENDERS = os.path.join(ROOT, "preview", "renders")
SHOTS = os.path.join(ROOT, "preview")
SCENARIOS = {"H": "windturbine", "FN": "scenarios/nacelle_all_faults", "FO": "scenarios/outside_all_faults"}
CRAMERA_SCENES = {   # screenshot file -> CRAMERA scene
    "windturbine": "windturbine", "windturbine_outside_ground": "windturbine_outside_ground",
    "windturbine_nacelle_all": "windturbine_nacelle_all_faults", "g1_t1": "windturbine_g1_t1", "g1_t2": "windturbine_g1_t2", "g1_t3": "windturbine_g1_t3",
}
VIEWER = "http://localhost:8711/?scene="
PANEL = (17, 192, 791, 875)          # the 3D panel in a 1600x1000 CRAMERA screenshot


def render_all():
    for tag, urdf in SCENARIOS.items():
        target = os.path.join(RENDERS, tag)
        shutil.rmtree(target, ignore_errors=True)
        subprocess.run(["blender", "-b", "--factory-startup", "-P", os.path.join(ROOT, "scripts", "render_preview.py"),
                        "--", os.path.join(ROOT, "urdf", urdf + ".urdf"), target],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def screenshot_all():
    profile = os.path.expanduser("~/snap/chromium/common/wt-profile")
    for name, scene in CRAMERA_SCENES.items():
        subprocess.run(["timeout", "200", "chromium", "--headless=new", "--no-sandbox", f"--user-data-dir={profile}",
                        "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--window-size=1600,1000",
                        "--virtual-time-budget=60000", f"--screenshot={os.path.join(SHOTS, name + '.png')}", VIEWER + scene],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def to_jpeg(src, name, size, crop=None):
    im = Image.open(src).convert("RGBA")
    ground = Image.new("RGBA", im.size, (14, 20, 27, 255))
    ground.alpha_composite(im)
    im = ground.convert("RGB")
    if crop:
        im = im.crop(crop)
    im.thumbnail(size)
    im.save(os.path.join(OUT, "img", name + ".jpg"), quality=82, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-renders", action="store_true")
    ap.add_argument("--skip-screenshots", action="store_true")
    args = ap.parse_args()
    if not args.skip_renders:
        render_all()
    if not args.skip_screenshots:
        screenshot_all()

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "img"))
    for n in ("overview", "tower_base", "rotor", "nacelle"):
        to_jpeg(os.path.join(RENDERS, "H", n + ".png"), "render_" + n, (1400, 900))
    to_jpeg(os.path.join(RENDERS, "FN", "nacelle.png"), "render_nacelle_faults", (1400, 900))
    for name in CRAMERA_SCENES:
        to_jpeg(os.path.join(SHOTS, name + ".png"), "cramera_" + name, (1400, 900), crop=PANEL)

    active = set()
    for tag in ("FN", "FO"):
        with open(os.path.join(ROOT, "urdf", SCENARIOS[tag] + "_ground_truth.yaml")) as f:
            active |= {x["id"] for x in yaml.safe_load(f)["faults"]}
    faults = {}
    total = 0
    for part in parts.ALL:
        total += len(part.FAULTS)
        for fid, f in part.FAULTS.items():
            if fid in active:
                faults.setdefault(f["inspection_point"], []).append({"id": fid, "severity": f["severity"], "text": f["description"]})
    points = []
    for part in parts.ALL:
        for p in part.INSPECTION_POINTS:
            name, tag = p["name"], "FO" if p["outside"] else "FN"
            if name not in faults or not os.path.exists(os.path.join(RENDERS, tag, name + ".png")):
                continue
            to_jpeg(os.path.join(RENDERS, "H", name + ".png"), "ok_" + name, (800, 600))
            to_jpeg(os.path.join(RENDERS, tag, name + ".png"), "bad_" + name, (800, 600))
            points.append({"name": name, "what": p["what"], "zone": "outside" if p["outside"] else "nacelle", "faults": faults[name]})

    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(ROOT, "gallery", "template.html")) as f:
        html = f.read()
    html = (html.replace("__POINTS__", json.dumps(points)).replace("__FAULTS__", str(total))
            .replace("__POINT_COUNT__", str(len(points))).replace("__COMMIT__", commit or "uncommitted")
            .replace("__UPDATED__", datetime.date.today().isoformat()))
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(html)
    with open(os.path.join(OUT, "files.json"), "w") as f:
        json.dump({f"img/{n}": f"img/{n}" for n in sorted(os.listdir(os.path.join(OUT, "img")))}, f)
    print(f"built {OUT}: {len(points)} inspection points, {total} faults, commit {commit}")


if __name__ == "__main__":
    main()
