#!/usr/bin/env python3
"""Make the G1 walk in a CRAMERA recording (visual only).

In CRAM the G1's base is planar: the recording slides the pelvis over the floor with
straight legs. This adds a walking gait to the recorded trajectory, derived from the
recorded base motion, so CRAMERA and the videos show steps. The plan and its
results are unchanged.

  add_gait.py ~/.cramera/scenes/windturbine_g1_t6

- The gait phase advances with the distance walked (no-slip: one step covers the
  stance foot's sweep), or at a fixed cadence when turning on the spot.
- Hip pitch swings with forward/backward speed, hip roll with sideways speed, and hip
  yaw with turning. The swing leg lifts its foot with the knee. The ankles keep
  both soles level.
- The pelvis dips by the height the straight stance leg loses when it swings, so the
  stance foot stays on the floor.

The original trajectory is kept as trajectory_raw.json; running again starts from it.
Base heights of lift and hoist rides are untouched (no planar motion, no steps).
"""
import argparse
import json
import math
import os
import shutil

import numpy as np

LEG = 0.66                # hip joint to sole with straight legs (m)
PITCH_PER_SPEED = 0.6     # hip pitch amplitude per m/s
PITCH_MAX = 0.42
ROLL_PER_SPEED = 0.35
ROLL_MAX = 0.18
YAW_PER_RATE = 0.35       # hip yaw amplitude per rad/s of turning
YAW_MAX = 0.3
KNEE_LIFT = 0.75          # knee bend at mid-swing (rad) at full activity
KNEE_STANCE = 0.05
TURN_CADENCE = 0.9        # gait cycles per second when turning on the spot
ACTIVE_SPEED = 0.08       # m/s (or equivalent turning) for full stepping
SMOOTH_S = 0.4            # velocity smoothing window (s)


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def smooth(a, n):
    if n < 2:
        return a
    k = np.ones(n) / n
    pad = np.pad(a, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def gait(base, fps):
    """Per frame: leg joint angles {side_joint: angle} and the pelvis dip (m)."""
    base = np.array(base, dtype=float)
    dt = 1.0 / fps
    xy = base[:, :2]
    yaw = np.unwrap([yaw_of(q) for q in base[:, 3:7]])
    vel = np.gradient(xy, dt, axis=0)
    omega = np.gradient(yaw, dt)
    n = max(1, round(SMOOTH_S * fps))
    vx = smooth(np.cos(yaw) * vel[:, 0] + np.sin(yaw) * vel[:, 1], n)       # robot frame
    vy = smooth(-np.sin(yaw) * vel[:, 0] + np.cos(yaw) * vel[:, 1], n)
    omega = smooth(omega, n)
    speed = np.hypot(vx, vy)
    active = np.clip((speed + 0.3 * np.abs(omega)) / ACTIVE_SPEED, 0.0, 1.0)
    active = smooth(active, n)
    phase, out = 0.0, []
    for i in range(len(base)):
        a = float(np.clip(PITCH_PER_SPEED * vx[i], -PITCH_MAX, PITCH_MAX))
        b = float(np.clip(ROLL_PER_SPEED * vy[i], -ROLL_MAX, ROLL_MAX))
        c = float(np.clip(YAW_PER_RATE * abs(omega[i]), 0.0, YAW_MAX))
        # no-slip: a gait cycle (two steps) covers twice the stance sweep 2 L sin(amplitude)
        sweep = 2 * LEG * math.sin(max(math.hypot(a, b), 1e-3))
        rate = speed[i] / (2 * sweep) if math.hypot(a, b) > 0.02 else 0.0
        rate = max(rate, TURN_CADENCE * min(1.0, abs(omega[i]) / 0.3))
        phase += 2 * math.pi * rate * dt * (active[i] > 0.02)
        angles, dip = {}, 0.0
        for side, offset in (("left", 0.0), ("right", math.pi)):
            psi = phase + offset
            swing = max(0.0, -math.sin(psi)) * active[i]          # stance for sin >= 0, swing for sin < 0
            hip = -a * math.cos(psi) - 0.45 * KNEE_LIFT * swing
            knee = KNEE_STANCE * active[i] + KNEE_LIFT * swing
            roll = b * math.cos(psi)
            angles[f"{side}_hip_pitch"] = hip
            angles[f"{side}_knee"] = knee
            angles[f"{side}_ankle_pitch"] = -(hip + knee)
            angles[f"{side}_hip_roll"] = roll
            angles[f"{side}_ankle_roll"] = -roll
            angles[f"{side}_hip_yaw"] = c * math.cos(psi) * (1 if omega[i] >= 0 else -1) * (1 if side == "left" else -1)
            if math.sin(psi) >= 0:                                 # stance leg carries the pelvis
                dip = max(dip, LEG * (1 - math.cos(hip) * math.cos(roll)) + 0.02 * KNEE_STANCE * active[i])
        out.append((angles, dip))
    return out


def add_gait(bundle):
    bundle = os.path.expanduser(bundle)
    path, raw = os.path.join(bundle, "trajectory.json"), os.path.join(bundle, "trajectory_raw.json")
    with open(path) as f:
        fresh = not json.load(f).get("gait")
    if fresh or not os.path.exists(raw):          # a new recording (or first run): keep it as the raw one
        shutil.copyfile(path, raw)
    with open(raw) as f:
        traj = json.load(f)
    with open(os.path.join(bundle, "scene.json")) as f:
        scene = json.load(f)
    robot = next(m for m in scene["models"] if m.get("robot"))
    prefix = robot.get("prefix") or ""
    fps = traj.get("framesPerSecond") or scene.get("framesPerSecond") or 25
    steps = gait(traj["base"], fps)
    for frame, base, (angles, dip) in zip(traj["frames"], traj["base"], steps):
        for name, value in angles.items():
            frame[f"{prefix}/{name}_joint" if prefix else f"{name}_joint"] = round(value, 5)
        base[2] = round(base[2] - dip, 5)
    traj["gait"] = True
    with open(path, "w") as f:
        json.dump(traj, f)
    walking = sum(1 for a, _ in steps if abs(a["left_knee"]) > 0.1 or abs(a["right_knee"]) > 0.1)
    print(f"gait added to {bundle}: {len(steps)} frames, {walking} with a foot in the air (robot prefix {prefix!r})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", nargs="+")
    for bundle in ap.parse_args().bundle:
        add_gait(bundle)


if __name__ == "__main__":
    main()
