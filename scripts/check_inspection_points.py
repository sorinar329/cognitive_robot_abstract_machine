#!/usr/bin/env python3
"""Check that the G1 can take each nacelle inspection view: the camera position
(inspection point + distance * view_from) must lie over a walkway or the free
floor behind the generator, at G1 camera height.

  python3 scripts/check_inspection_points.py [urdf]
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from turbine import dims as d  # noqa: E402
from turbine import parts  # noqa: E402

CAMERA_Z = (1.0, 1.75)        # G1 D435 at 1.27-1.60 m, some slack for waist pitch and crouching
FREE_REAR_X = d.GEN_X[0]      # behind the generator the floor is free across the width (drivetrain x)


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def transforms(urdf_path):
    root = ET.parse(urdf_path).getroot()
    parent = {}
    for j in root.iter("joint"):
        o = j.find("origin")
        t = np.eye(4)
        t[:3, :3] = rpy_matrix(*[float(v) for v in o.get("rpy").split()])
        t[:3, 3] = [float(v) for v in o.get("xyz").split()]
        parent[j.find("child").get("link")] = (j.find("parent").get("link"), t)

    def tf(link):
        if link not in parent:
            return np.eye(4)
        p, t = parent[link]
        return tf(p) @ t
    return tf


def main():
    urdf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "urdf", "windturbine.urdf")
    tf = transforms(urdf)
    to_nacelle = np.linalg.inv(tf("nacelle"))
    rear_x = d.drivetrain_to_nacelle(FREE_REAR_X)[0]
    bad = 0
    for part in parts.ALL:
        for p in part.INSPECTION_POINTS:
            if p["outside"]:
                continue
            target = (to_nacelle @ tf(p["name"]))[:3, 3]
            rot = (to_nacelle @ tf(p["parent"]))[:3, :3]
            v = rot @ np.array(p["view_from"], dtype=float)
            cam = target + p["distance"] * v / np.linalg.norm(v)
            over_floor = abs(cam[1]) >= d.WALKWAY_Y - 0.05 or cam[0] <= rear_x
            inside = d.INNER_X[0] < cam[0] < d.INNER_X[1] and abs(cam[1]) < d.INNER_Y[1]
            height_ok = CAMERA_Z[0] <= cam[2] <= CAMERA_Z[1]
            ok = over_floor and inside and height_ok
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {p['name']:40s} camera x={cam[0]:6.2f} y={cam[1]:5.2f} z={cam[2]:4.2f}"
                  + ("" if over_floor else "  not over a walkway") + ("" if inside else "  outside the nacelle")
                  + ("" if height_ok else "  camera height"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
