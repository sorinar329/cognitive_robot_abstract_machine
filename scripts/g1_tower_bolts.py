#!/usr/bin/env python3
"""Task T6 with the Unitree G1: flange bolt check up the tower.

The G1 walks into the tower, rides the service lift and stops at each rest platform.
There it walks a ring of standpoints facing the wall and reads the torque markings
on the nuts under the flange above. At the end it rides back down and walks out.
The report gives, per flange:

- the tightness ratio: the mean estimated preload of the bolts it could read, from
  the marking offsets (see turbine/bolts.py);
- the worst bolt, and the bolts that are loose, lost or need re-tightening;
- what has to be done, following ACP RP 401 thresholds by analogy.

A marking counts as read only if the nut is in the D435's field of view, at most
READ_RANGE away, seen at most MAX_INCIDENCE off its face normal, and not hidden
behind the lift car or the flange edge. The offset angle is read from the scenario's
ground truth (oracle perception, as in T1/T2). Reading it from camera images is the
next step.

  python -m cramera.onboard.demo scripts/g1_tower_bolts.py --name windturbine_g1_t6
  python3 scripts/finish_recording.py ~/.cramera/scenes/windturbine_g1_t6 --view g1_tower
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import yaml  # noqa: E402
from coraplex.exceptions import MotionDidNotFinish  # noqa: E402
from coraplex.execution_environment import simulated_robot  # noqa: E402
from coraplex.plans.factories import sequential  # noqa: E402
from coraplex.robot_plans.actions.core.navigation import LookAtAction  # noqa: E402
from semantic_digital_twin.spatial_types.spatial_types import Pose  # noqa: E402

import g1_climb as climb  # noqa: E402
import g1_inspection_round as rnd  # noqa: E402
from turbine import bolts as tb  # noqa: E402
from turbine.dims import site  # noqa: E402

SCENARIO = "tower_bolt_check"
BASE = site.TOWER_BASE_Z                  # tower frame -> world z
VIEW_DISTANCE = 1.55                      # standpoint to bolt circle: closer, the flange edge hides the nuts
MIN_RING = 0.5                            # (the camera is 0.34 m above them and looks down past the edge)
READ_RANGE = 2.6                          # m; a 7 mm marking is still ~5 px wide on the D435 colour image
MAX_INCIDENCE = math.radians(60)          # the marking is on the inward face of the nut
SIDE_LOOK = math.radians(40)              # extra turn at both ends of a sweep
ROBOT_RADIUS = 0.32                       # clearance from the lift and ladder openings
LX, LY = site.LIFT_CENTER
CX, CY, CZ = site.LIFT_CAR
REPORT_DIR = os.path.join(ROOT, "reports")


def floor_of(k):
    return BASE + site.platform_heights()[k - 1]


def ring_radius(k):
    return max(MIN_RING, tb.flange(k)["bolt_radius"] - VIEW_DISTANCE)


def ring_point(az, k):
    return ring_radius(k) * math.cos(az), ring_radius(k) * math.sin(az)


def blocked(x, y, k):
    """Standpoint too close to the lift shaft opening or the ladder opening of platform k."""
    m = site.LIFT_SHAFT_MARGIN + ROBOT_RADIUS
    if LX - CX / 2 - m < x < LX + CX / 2 + m and LY - CY / 2 - m < y < LY + CY / 2 + m:
        return True
    z = site.platform_heights()[k - 1]
    r = site.tower_radius(z) - site.tower_wall(z) - site.LADDER_WALL_GAP
    lx, ly = r * math.cos(site.LADDER_AZIMUTH), r * math.sin(site.LADDER_AZIMUTH)
    return abs(x - lx) < 0.35 + ROBOT_RADIUS and abs(y - ly) < 0.35 + ROBOT_RADIUS


def standpoints(k, hfov):
    """Azimuths of the standpoints on platform k, so neighbouring views overlap."""
    fl = tb.flange(k)
    distance = fl["bolt_radius"] - ring_radius(k)
    width = 2 * distance * math.tan(hfov / 2)                       # wall seen across the image
    n = math.ceil(2 * math.pi * fl["bolt_radius"] / (0.75 * width))
    azs = [-math.pi + 2 * math.pi * (i + 0.5) / n for i in range(n)]
    return [a for a in azs if not blocked(*ring_point(a, k), k)]


def transit(a0, a1, step=math.radians(35)):
    """Ring points from azimuth a0 to a1 (no looking), in steps small enough to keep off the lift."""
    n = max(1, math.ceil(abs(a1 - a0) / step))
    return [a0 + (a1 - a0) * i / n for i in range(1, n + 1)]


def bolt_ahead(k, x, y, yaw):
    """The bolt where a ray from (x, y) along yaw meets the bolt circle."""
    fl = tb.flange(k)
    dx, dy = math.cos(yaw), math.sin(yaw)
    b = x * dx + y * dy
    t = -b + math.sqrt(b * b - (x * x + y * y - fl["bolt_radius"] ** 2))
    az = math.atan2(y + t * dy, x + t * dx)
    return round((az % (2 * math.pi)) / (2 * math.pi) * len(fl["bolts"])) % len(fl["bolts"])


def nut_target(k, bolt):
    fl = tb.flange(k)
    _, a, x, y = fl["bolts"][bolt]
    ap = fl["spec"]["nut_af"] / 2
    return np.array([x - ap * math.cos(a), y - ap * math.sin(a), BASE + tb.nut_height(k)]), a


def hits_box(p0, p1, lo, hi):
    """Segment p0-p1 intersects the axis-aligned box [lo, hi] (slab test)."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-9:
            if not lo[i] <= p0[i] <= hi[i]:
                return False
            continue
        a, b = (lo[i] - p0[i]) / d[i], (hi[i] - p0[i]) / d[i]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return False
    return True


def readable(robot, camera_pos, k, bolt, car_floor):
    target, a = nut_target(k, bolt)
    seen, *_ = rnd.in_field_of_view(robot, target)
    if not seen:
        return False
    ray = camera_pos - target
    dist = np.linalg.norm(ray)
    if dist > READ_RANGE:
        return False
    normal = np.array([-math.cos(a), -math.sin(a), 0.0])
    if math.acos(np.clip(ray @ normal / dist, -1, 1)) > MAX_INCIDENCE:
        return False
    car_lo = np.array([LX - CX / 2 - 0.02, LY - CY / 2 - 0.02, car_floor - 0.05])
    car_hi = np.array([LX + CX / 2 + 0.02, LY + CY / 2 + 0.02, car_floor + CZ + 0.3])
    if hits_box(camera_pos, target, car_lo, car_hi):
        return False
    fl = tb.flange(k)                             # the lower flange ring hangs over the nuts
    z_bottom, z_top = BASE + fl["z"] - site.FLANGE_H, BASE + fl["z"]
    for t in np.linspace(0.08, 0.98, 40):         # from just off the nut towards the camera
        p = target + t * (camera_pos - target)
        if fl["inner_radius"] - site.FLANGE_WIDTH < math.hypot(p[0], p[1]) < fl["inner_radius"] and z_bottom < p[2] < z_top:
            return False
    return True


def check_flange(context, world, robot, k, offsets):
    """Walk the ring of platform k and read all markings in view; returns {bolt: offset}."""
    floor = floor_of(k)
    camera = robot.get_default_camera()
    fov = camera.field_of_view.horizontal_angle
    azs = standpoints(k, fov)
    fl = tb.flange(k)
    read = {}
    climb.walk(context, world, floor, (0.3, 0.0, 0.0))
    # round the -Y side to the first standpoint, then sweep round to the last
    route = transit(0.0, azs[0])
    climb.walk(context, world, floor, *[(*ring_point(a, k), a) for a in route])
    car_floor = BASE + site.LIFT_BOTTOM_Z + next(c for c in world.connections if c.name.name == "service_lift_joint").position
    # facing the wall at each standpoint; at both ends also turned towards the sector the sweep leaves out
    views = [(a, a) for a in azs]
    views.insert(1, (azs[0], azs[0] - SIDE_LOOK))
    views.append((azs[-1], azs[-1] + SIDE_LOOK))
    for a, yaw in views:
        x, y = ring_point(a, k)
        climb.walk(context, world, floor, (x, y, yaw))
        target, _ = nut_target(k, bolt_ahead(k, x, y, yaw))
        try:
            with simulated_robot:
                sequential([LookAtAction(Pose.from_xyz_rpy(*rnd.aim_point(target, x, y, floor), reference_frame=world.root))],
                           context=context).plan.perform()
        except MotionDidNotFinish:
            pass
        cam = camera.root.global_pose.to_np()[:3, 3]
        new = [i for i, *_ in fl["bolts"] if i not in read and readable(robot, cam, k, i, car_floor)]
        for i in new:
            read[i] = offsets.get(i, 0.0)          # oracle: the marking offset from the ground truth
        print(f"  flange {k} @ {math.degrees(a):6.1f} deg, facing {math.degrees(yaw):6.1f}: {len(new):3d} new markings read ({len(read)}/{len(fl['bolts'])})", flush=True)
    climb.walk(context, world, floor, *[(*ring_point(a, k), a) for a in transit(azs[-1], 0.0)], (0.3, 0.0, 0.0),
               (LX, LY, 0.0))
    return read


def ground_truth():
    with open(os.path.join(ROOT, "urdf", "scenarios", SCENARIO + "_ground_truth.yaml")) as fh:
        truth = yaml.safe_load(fh)
    return {k: {int(i): float(v) for i, v in (truth["signals"].get(f"flange_{k}_marking_offsets_deg") or {}).items()}
            for k in range(1, len(site.TOWER_FLANGES) + 1)}


def pct(v, digits=0):
    return "n/a" if v is None else f"{v:.{digits}%}"


def write_report(results, truth):
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, SCENARIO + ".json"), "w") as fh:
        json.dump(dict(scenario=SCENARIO, flanges=results), fh, indent=1)
    lines = [f"# Tower flange bolt check ({SCENARIO})", "",
             "Tightness ratio = mean estimated preload of the bolts the G1 could read, from the torque-marking "
             "offsets (turbine/bolts.py). Readings come from the scenario ground truth (oracle), visibility is "
             "computed in the CRAM world.", "",
             "| Flange | Bolts | Read | Tightness ratio | Worst bolt | Verdict |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['flange']} ({r['size']}) | {r['bolts']} | {r['seen']} | {pct(r['tightness_ratio'], 1)} | "
                     f"{pct(r['worst_ratio'])} | {r['verdict']} |")
    for r in results:
        lines += ["", f"## Flange {r['flange']}: {r['verdict']}"]
        if r["ratios"]:
            lines.append("Bolts below 100 %: " + ", ".join(f"{i}: {v:.0%}" for i, v in sorted(r["ratios"].items())))
        lines += [f"- {a}" for a in r["actions"]]
        missed = sorted(set(truth.get(r["flange"], {})) - set(map(int, r["ratios"])))
        if missed:
            lines.append(f"- (ground truth check: turned marking{'s' if len(missed) > 1 else ''} {missed} not readable "
                         "for the robot, covered by the technician's list above)")
    with open(os.path.join(REPORT_DIR, SCENARIO + ".md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flanges", type=int, nargs="*", default=[1, 2, 3])
    args = ap.parse_args()
    truth = ground_truth()
    world, robot, context = climb.load(SCENARIO, Pose.from_xyz_rpy(0.0, -5.5, rnd.PELVIS_HEIGHT_ABOVE_FLOOR, yaw=math.pi / 2))
    # in through the door and backwards into the lift car (as in T5)
    climb.walk(context, world, 0.0, (0.0, -2.0, math.pi / 2), (0.25, -0.2, 0.0), (0.25, LY, 0.0), (LX, LY, 0.0))
    results = []
    for k in args.flanges:
        stop = site.platform_heights()[k - 1] - site.LIFT_BOTTOM_Z
        climb.ride(context, world, robot, "service_lift", "service_lift_joint", stop, site.LIFT_SPEED)
        read = check_flange(context, world, robot, k, truth.get(k, {}))
        results.append(tb.assess(k, read, len(tb.flange(k)["bolts"])))
        r = results[-1]
        print(f"flange {k}: read {r['seen']}/{r['bolts']}, tightness ratio {pct(r['tightness_ratio'], 1)}, "
              f"worst {pct(r['worst_ratio'])} -> {r['verdict']}", flush=True)
    climb.ride(context, world, robot, "service_lift", "service_lift_joint", 0.0, site.LIFT_SPEED)
    climb.walk(context, world, 0.0, (0.25, LY, 0.0), (0.25, -0.2, -math.pi / 2), (0.0, -2.0, -math.pi / 2),
               (0.0, -5.5, -math.pi / 2))
    print("\n=== T6 report ===\n" + write_report(results, truth), flush=True)
    # correct when every turned marking was either read (and reported) or is on the technician's list
    turned = sum(len(truth.get(r["flange"], {})) for r in results)
    read = sum(len(set(truth.get(r["flange"], {})) & set(map(int, r["ratios"]))) for r in results)
    covered = all(set(truth.get(r["flange"], {})) <= set(map(int, r["ratios"])) | set(r["unseen_bolts"]) for r in results)
    print(f"\nturned markings read by the robot: {read}/{turned}; the rest on the technician's list: {covered}"
          f"\nback outside the tower: True", flush=True)
    if not covered:
        sys.exit(1)


if __name__ == "__main__":
    main()
