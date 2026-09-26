#!/usr/bin/env python3
"""Replay a CRAMERA recording of the G1 in MuJoCo physics.

Physics replay, as in CRAMERA's laboratory: the CRAM plan is not changed, its
recorded joint motion becomes the target of force-limited servos, and MuJoCo
measures what physically happens:

- the turbine: the collision boxes and cylinders of the scenario URDF as static
  geometry; links whose joint moves during the run (lift car, hoist platform,
  doors) follow the recording kinematically;
- the G1: rigid bodies with the URDF's masses and collision meshes. The pelvis
  is held on the recorded path by a stiff weld (no balance controller: it cannot
  fall over). Every joint is a position servo limited to the URDF effort (e.g.
  25 Nm arms, 1.4 Nm fingers);
- free objects (e.g. the torque tool case): a box with mass and friction, lying
  where the recording spawned it. No attachment: it is held only by the fingers.

  physics_replay.py ~/.cramera/scenes/windturbine_g1_t3 [--mass torque_tool_case=3.0]

Writes <bundle>_physics (the same scene, trajectory from the simulation) and
<bundle>_physics/physics_report.json. Run with the cramera-port Python (mujoco).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

TIMESTEP = 0.002
DEFAULT_MASS = 3.0            # kg for free objects (a torque tool in its case)
FRICTION = 1.0
SERVO_KP_PER_NM = 10.0        # servo stiffness (Nm/rad) per Nm of effort limit
SERVO_DAMPING = 0.08          # joint damping per Nm of effort limit (Nms/rad)
CONTACT_REPORT_N = 5.0        # robot-turbine contacts above this force are reported
FEET = ("left_ankle_roll_link", "right_ankle_roll_link", "left_ankle_pitch_link", "right_ankle_pitch_link")


# %% URDF helpers
def origin(el):
    o = el.find("origin") if el is not None else None
    xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    m[:3, 3] = xyz
    return m


def quat_wxyz(rot):
    x, y, z, w = Rotation.from_matrix(rot).as_quat()
    return [w, x, y, z]


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


class Urdf:
    def __init__(self, path):
        self.path = path
        self.root = ET.parse(path).getroot()
        self.links = {l.get("name"): l for l in self.root.iter("link")}
        self.joints = {}          # child link -> joint element
        for j in self.root.iter("joint"):
            self.joints[j.find("child").get("link")] = j
        self.children = {}
        for child, j in self.joints.items():
            self.children.setdefault(j.find("parent").get("link"), []).append(child)

    def root_links(self):
        return [l for l in self.links if l not in self.joints]

    def joint_motion(self, j, q):
        t = j.get("type")
        axis = np.array([float(v) for v in (j.find("axis").get("xyz") if j.find("axis") is not None else "1 0 0").split()])
        m = np.eye(4)
        if t in ("revolute", "continuous"):
            m[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
        elif t == "prismatic":
            m[:3, 3] = axis * q
        return m

    def fk(self, values, prefix=""):
        """World transform of every link for joint values {prefix/joint: q}."""
        out = {}

        def walk(link, parent_tf):
            out[link] = parent_tf
            for child in self.children.get(link, []):
                j = self.joints[child]
                q = values.get(f"{prefix}/{j.get('name')}" if prefix else j.get("name"), 0.0)
                walk(child, parent_tf @ origin(j) @ self.joint_motion(j, q))
        for r in self.root_links():
            walk(r, np.eye(4))
        return out


# %% scene building
class SceneBuilder:
    def __init__(self, bundle):
        self.bundle = bundle
        self.meshes = {}
        self.assets = []
        self.world = []           # xml lines inside <worldbody>
        self.actuators = []
        self.equalities = []

    def geom_xml(self, col, name, contype, conaffinity, extra=""):
        g = col.find("geometry")[0]
        tf = origin(col)
        dims = [float(v) for v in (g.get("size") or "").split()] + [float(g.get(a)) for a in ("radius", "length") if g.get(a)]
        if any(d <= 1e-6 for d in dims):
            print(f"  skipped degenerate collision {name} ({g.tag} {dims})")
            return ""
        pose = f'pos="{fmt(tf[:3, 3])}" quat="{fmt(quat_wxyz(tf[:3, :3]))}"'
        common = f'contype="{contype}" conaffinity="{conaffinity}" friction="{FRICTION} 0.02 0.001" {extra}'
        if g.tag == "box":
            size = np.array([float(v) for v in g.get("size").split()]) / 2
            return f'<geom name="{name}" type="box" size="{fmt(size)}" {pose} {common}/>'
        if g.tag == "cylinder":
            return f'<geom name="{name}" type="cylinder" size="{float(g.get("radius")):.6g} {float(g.get("length")) / 2:.6g}" {pose} {common}/>'
        if g.tag == "sphere":
            return f'<geom name="{name}" type="sphere" size="{float(g.get("radius")):.6g}" {pose} {common}/>'
        if g.tag == "mesh":
            path = os.path.join(self.bundle, g.get("filename"))
            if not os.path.exists(path):
                return ""
            scale = g.get("scale", "1 1 1")
            key = (path, scale)
            if key not in self.meshes:
                self.meshes[key] = f"mesh{len(self.meshes)}"
                self.assets.append(f'<mesh name="{self.meshes[key]}" file="{path}" scale="{scale}"/>')
            return f'<geom name="{name}" type="mesh" mesh="{self.meshes[key]}" {pose} {common}/>'
        return ""

    def add_environment(self, urdf, prefix, values0, moving_links):
        """Static geoms in the world; links in ``moving_links`` as mocap bodies."""
        fk = urdf.fk(values0, prefix)
        n = 0
        for name, link in urdf.links.items():
            cols = link.findall("collision")
            if not cols:
                continue
            tf = fk[name]
            geoms = [self.geom_xml(c, f"env_{name}_{i}", 0, 1) for i, c in enumerate(cols)]
            geoms = [g for g in geoms if g]
            n += len(geoms)
            if name in moving_links:
                self.world.append(f'<body name="env_{name}" mocap="true" pos="{fmt(tf[:3, 3])}" quat="{fmt(quat_wxyz(tf[:3, :3]))}">'
                                  + "".join(geoms) + "</body>")
            else:
                self.world.append(f'<body name="env_{name}" pos="{fmt(tf[:3, 3])}" quat="{fmt(quat_wxyz(tf[:3, :3]))}">'
                                  + "".join(geoms) + "</body>")
        return n

    def robot_body(self, urdf, link, tf, joint=None):
        el = urdf.links[link]
        parts = [f'<body name="rb_{link}" pos="{fmt(tf[:3, 3])}" quat="{fmt(quat_wxyz(tf[:3, :3]))}">']
        if joint is None:
            parts.append('<freejoint name="root"/>')
        inertial = el.find("inertial")
        if inertial is not None and float(inertial.find("mass").get("value")) > 1e-6:
            it = origin(inertial)
            i = inertial.find("inertia")
            full = np.array([[float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                             [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                             [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))]])
            full = it[:3, :3] @ full @ it[:3, :3].T
            full = np.maximum(full, 0) * np.eye(3) + full * (1 - np.eye(3))
            diag = np.maximum(np.diag(full), 1e-6)
            parts.append(f'<inertial pos="{fmt(it[:3, 3])}" mass="{float(inertial.find("mass").get("value")):.6g}" '
                         f'fullinertia="{fmt([diag[0], diag[1], diag[2], full[0, 1], full[0, 2], full[1, 2]])}"/>')
        elif joint is not None and joint.get("type") != "fixed":
            parts.append('<inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>')
        if joint is not None and joint.get("type") in ("revolute", "continuous", "prismatic"):
            name = joint.get("name")
            axis = joint.find("axis").get("xyz") if joint.find("axis") is not None else "1 0 0"
            lim = joint.find("limit")
            effort = float(lim.get("effort", 50)) if lim is not None else 50.0
            rng = ""
            if lim is not None and joint.get("type") != "continuous" and lim.get("lower") is not None:
                rng = f'limited="true" range="{float(lim.get("lower")):.6g} {float(lim.get("upper")):.6g}"'
            kind = "slide" if joint.get("type") == "prismatic" else "hinge"
            parts.append(f'<joint name="{name}" type="{kind}" axis="{axis}" {rng} damping="{SERVO_DAMPING * effort:.4g}" armature="0.01"/>')
            self.actuators.append(f'<position name="act_{name}" joint="{name}" kp="{SERVO_KP_PER_NM * effort:.4g}" '
                                  f'forcelimited="true" forcerange="{-effort:.4g} {effort:.4g}"/>')
        for i, c in enumerate(el.findall("collision")):
            parts.append(self.geom_xml(c, f"rb_{link}_{i}", 1, 0))
        for child in urdf.children.get(link, []):
            j = urdf.joints[child]
            parts.append(self.robot_body(urdf, child, origin(j), j))
        parts.append("</body>")
        return "".join(parts)

    def add_robot(self, urdf, base, base_pose):
        tf = np.eye(4)
        tf[:3, 3] = base_pose[:3]
        tf[:3, :3] = Rotation.from_quat(base_pose[3:7]).as_matrix()
        self.world.append(self.robot_body(urdf, base, tf))
        self.world.append(f'<body name="base_target" mocap="true" pos="{fmt(tf[:3, 3])}" quat="{fmt(quat_wxyz(tf[:3, :3]))}"/>')
        self.equalities.append(f'<weld body1="base_target" body2="rb_{base}" solref="0.004 1" solimp="0.95 0.99 0.001"/>')

    def add_object(self, key, parts, pose, mass):
        """A free body from boxes [(centre, size)] in its own frame; the mass is spread by volume."""
        volumes = [float(np.prod(s)) for _, s in parts]
        geoms = "".join(
            f'<geom name="obj_{key}_{i}" type="box" pos="{fmt(c)}" size="{fmt(np.array(s) / 2)}" '
            f'mass="{mass * v / sum(volumes):.6g}" contype="1" conaffinity="1" friction="{FRICTION} 0.02 0.001" '
            f'rgba="0.85 0.15 0.1 1"/>' for i, ((c, s), v) in enumerate(zip(parts, volumes)))
        self.world.append(f'<body name="obj_{key}" pos="{fmt(pose[:3])}" quat="{fmt([pose[6], pose[3], pose[4], pose[5]])}">'
                          f'<freejoint name="obj_{key}"/>{geoms}</body>')

    def xml(self):
        return (f'<mujoco model="windturbine_physics"><compiler angle="radian" autolimits="true"/>'
                f'<option timestep="{TIMESTEP}" integrator="implicitfast" cone="elliptic" impratio="10"/>'
                f'<size memory="400M"/>'
                f'<asset>{"".join(self.assets)}</asset><worldbody>{"".join(self.world)}</worldbody>'
                f'<equality>{"".join(self.equalities)}</equality><actuator>{"".join(self.actuators)}</actuator></mujoco>')


# %% replay
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle")
    ap.add_argument("--mass", action="append", default=[], help="object=kg")
    ap.add_argument("--out", help="output bundle (default: <bundle>_physics)")
    ap.add_argument("--with-gait", action="store_true", help="replay the visual walking gait too (legs swing into things)")
    args = ap.parse_args()
    bundle = os.path.expanduser(args.bundle).rstrip("/")
    out_dir = args.out or bundle + "_physics"
    masses = {k: float(v) for k, v in (m.split("=") for m in args.mass)}

    with open(os.path.join(bundle, "scene.json")) as f:
        scene = json.load(f)
    # the CRAM motion itself: without the visual walking gait (scripts/add_gait.py), if one was added
    raw = os.path.join(bundle, "trajectory_raw.json")
    source = raw if os.path.exists(raw) and not args.with_gait else os.path.join(bundle, "trajectory.json")
    with open(source) as f:
        traj = json.load(f)
    frames, bases, objects = traj["frames"], traj["base"], traj.get("objects") or []
    fps = traj.get("framesPerSecond") or scene.get("framesPerSecond") or 10
    robot_spec = next(m for m in scene["models"] if m.get("robot"))
    env_spec = next(m for m in scene["models"] if not m.get("robot"))
    rprefix, eprefix = robot_spec.get("prefix") or "", env_spec.get("prefix") or ""
    robot = Urdf(os.path.join(bundle, robot_spec["urdf"]))
    env = Urdf(os.path.join(bundle, env_spec["urdf"]))
    base_body = scene["robot"]["baseBody"]

    # environment joints that move during the recording
    env_joint_keys = {j.get("name"): f"{eprefix}/{j.get('name')}" for j in env.joints.values() if j.get("type") != "fixed"}
    moving_joints = {n for n, k in env_joint_keys.items()
                     if max(abs(fr.get(k, 0.0) - frames[0].get(k, 0.0)) for fr in frames) > 1e-5}
    moving_links = set()
    for child, j in env.joints.items():
        if j.get("name") in moving_joints:
            stack = [child]
            while stack:
                link = stack.pop()
                moving_links.add(link)
                stack += env.children.get(link, [])
    moving_links = {l for l in moving_links if env.links[l].findall("collision")}

    b = SceneBuilder(bundle)
    n_env = b.add_environment(env, eprefix, frames[0], moving_links)
    b.add_robot(robot, base_body, bases[0])
    for spec in scene.get("objects") or []:
        parts = [(p["centre"], p["size"]) for p in spec["parts"]] if spec.get("parts") else \
            [((0, 0, 0), spec["box"])] if "box" in spec else None
        if parts:
            b.add_object(spec["key"], parts, spec["spawn"], masses.get(spec["key"], spec.get("mass", DEFAULT_MASS)))
    xml = b.xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    print(f"MuJoCo scene: {n_env} turbine geoms ({len(moving_links)} moving links), {model.nbody} bodies, "
          f"{model.nu} servos, {model.ngeom} geoms", flush=True)

    # indices
    jnames = [model.joint(i).name for i in range(model.njnt)]
    act_joint = [(a, model.joint(model.actuator(a).trnid[0]).name) for a in range(model.nu)]
    qadr = {n: model.jnt_qposadr[i] for i, n in enumerate(jnames)}
    base_mocap = model.body("base_target").mocapid[0]
    env_mocap = {l: model.body(f"env_{l}").mocapid[0] for l in moving_links}
    obj_keys = [s["key"] for s in scene.get("objects") or [] if "box" in s or s.get("parts")]
    geom_body = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
    robot_geoms = np.array([n.startswith("rb_") for n in geom_body])
    env_geoms = np.array([n.startswith("env_") for n in geom_body])

    # initial state: robot joints at frame 0, settle
    for a, jn in act_joint:
        q0 = frames[0].get(f"{rprefix}/{jn}", 0.0)
        data.qpos[qadr[jn]] = q0
        data.ctrl[a] = q0
    data.qpos[qadr["root"]:qadr["root"] + 3] = bases[0][:3]
    data.qpos[qadr["root"] + 3:qadr["root"] + 7] = [bases[0][6], *bases[0][3:6]]
    mujoco.mj_forward(model, data)

    base_rot = Rotation.from_quat([b_[3:7] for b_ in bases])
    env_fk = [env.fk(fr, eprefix) for fr in frames] if moving_links else None
    sub = max(1, round(1.0 / fps / TIMESTEP))
    out_frames, out_base, out_objects = [], [], []
    contacts = {}              # (robot body, env body) -> [max force, frames]
    foot_max = 0.0
    obj_hist = {k: [] for k in obj_keys}
    palms = [n for n in ("left_hand_palm_link", "right_hand_palm_link") if f"rb_{n}" in [model.body(b).name for b in range(model.nbody)]]
    palm_hist = {p: [] for p in palms}
    t0 = time.time()
    for i in range(len(frames)):
        j = min(i + 1, len(frames) - 1)
        slerp = Slerp([0, 1], Rotation.concatenate([base_rot[i], base_rot[j]]))
        for s in range(sub):
            w = s / sub
            for a, jn in act_joint:
                k = f"{rprefix}/{jn}"
                data.ctrl[a] = (1 - w) * frames[i].get(k, 0.0) + w * frames[j].get(k, 0.0)
            data.mocap_pos[base_mocap] = (1 - w) * np.array(bases[i][:3]) + w * np.array(bases[j][:3])
            x, y, z, qw = slerp(w).as_quat()
            data.mocap_quat[base_mocap] = [qw, x, y, z]
            for link, mid in env_mocap.items():
                ti, tj = env_fk[i][link], env_fk[j][link]
                data.mocap_pos[mid] = (1 - w) * ti[:3, 3] + w * tj[:3, 3]
                data.mocap_quat[mid] = quat_wxyz(ti[:3, :3])
            mujoco.mj_step(model, data)
            if data.ncon:
                forces = np.zeros(6)
                for ci in range(data.ncon):
                    c = data.contact[ci]
                    g1, g2 = c.geom1, c.geom2
                    if not ((robot_geoms[g1] and env_geoms[g2]) or (robot_geoms[g2] and env_geoms[g1])):
                        continue
                    mujoco.mj_contactForce(model, data, ci, forces)
                    rg, eg = (g1, g2) if robot_geoms[g1] else (g2, g1)
                    rb, eb = geom_body[rg][3:], geom_body[eg][4:]
                    if rb in FEET:
                        foot_max = max(foot_max, forces[0])
                        continue
                    rec = contacts.setdefault((rb, eb), [0.0, set(), None])
                    if forces[0] > rec[0]:
                        rec[0], rec[2] = forces[0], np.array(c.pos)
                    rec[1].add(i)
        # record this frame
        fr = dict(frames[i])
        for jn in jnames:
            if jn in ("root",) or jn.startswith("obj_"):
                continue
            fr[f"{rprefix}/{jn}"] = round(float(data.qpos[qadr[jn]]), 5)
        out_frames.append(fr)
        pel = data.body(f"rb_{base_body}")
        w_, x_, y_, z_ = pel.xquat
        out_base.append([*map(float, np.round(pel.xpos, 5)), float(x_), float(y_), float(z_), float(w_)])
        objs = {}
        for k in obj_keys:
            o = data.body(f"obj_{k}")
            w_, x_, y_, z_ = o.xquat
            objs[k] = [*map(float, np.round(o.xpos, 5)), float(x_), float(y_), float(z_), float(w_)]
            obj_hist[k].append(np.array(o.xpos))
        out_objects.append(objs)
        for p in palms:
            palm_hist[p].append(np.array(data.body(f"rb_{p}").xpos))
        if i % 200 == 0:
            print(f"  frame {i}/{len(frames)}  sim {data.time:6.1f} s  wall {time.time() - t0:5.1f} s", flush=True)

    # %% report
    report = dict(bundle=os.path.basename(bundle), source=os.path.basename(source), frames=len(frames), sim_seconds=round(data.time, 2),
                  foot_max_force_n=round(foot_max, 1), robot_turbine_contacts=[], objects=[])
    for (rb, eb), (fmax, fs, where) in sorted(contacts.items(), key=lambda kv: -kv[1][0]):
        if fmax >= CONTACT_REPORT_N:
            report["robot_turbine_contacts"].append(dict(robot_link=rb, turbine_link=eb, max_force_n=round(fmax, 1),
                                                         frames=len(fs), first_frame=min(fs),
                                                         at=[round(float(v), 3) for v in where]))
    for seg in scene.get("segments") or []:
        key = seg.get("picks")
        if key not in obj_keys:
            continue
        hist = np.array(obj_hist[key])
        attach, detach = seg.get("attach"), seg.get("detach")
        start_z = hist[0][2]
        lifted = float(hist[attach:detach, 2].max() - start_z) if attach is not None else 0.0
        # held: the case followed the recorded (kinematically attached) path during transport
        ref = np.array([o[key][:3] for o in objects[attach:detach]]) if objects and attach is not None else None
        dev = np.linalg.norm(hist[attach:detach] - ref, axis=1) if ref is not None else np.array([np.nan])
        place = np.array(seg.get("place") or objects[-1][key][:3])
        final = hist[-1]
        # held: the case stays at the grasping hand (the palm nearest to it at the grasp)
        hand = min(palms, key=lambda p: np.linalg.norm(palm_hist[p][attach] - hist[attach])) if palms and attach is not None else None
        to_hand = (np.linalg.norm(np.array(palm_hist[hand][attach + 20:detach]) - hist[attach + 20:detach], axis=1)
                   if hand else np.array([np.inf]))
        lost_at = next((attach + 20 + n for n, d in enumerate(to_hand) if d > 0.20), None)
        report["objects"].append(dict(
            object=key, mass_kg=masses.get(key, next((s.get("mass", DEFAULT_MASS) for s in scene["objects"] if s["key"] == key))), lifted_m=round(lifted, 3),
            max_deviation_from_plan_m=round(float(np.nanmax(dev)), 3), dropped_at_frame=lost_at,
            max_distance_to_hand_m=round(float(np.max(to_hand)), 3) if hand else None,
            final_position=[round(float(v), 3) for v in final],
            placement_error_m=round(float(np.linalg.norm(final[:2] - place[:2])), 3),
            height_above_target_m=round(float(final[2] - place[2]), 3),
            held_through_transport=bool(lost_at is None and lifted > 0.02),
            placed=bool(np.linalg.norm(final[:2] - place[:2]) < 0.06 and abs(final[2] - place[2]) < 0.03)))

    # %% physics bundle
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(bundle, out_dir, ignore=shutil.ignore_patterns("trajectory_raw.json"))
    phys = dict(traj, frames=out_frames, base=out_base, objects=out_objects, physics=True)
    phys.pop("gait", None)
    with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
        json.dump(phys, f)
    scene["name"] = os.path.basename(out_dir)
    if isinstance(scene.get("task"), str):
        scene["task"] += " (MuJoCo physics replay)"
    with open(os.path.join(out_dir, "scene.json"), "w") as f:
        json.dump(scene, f, indent=1)
    with open(os.path.join(out_dir, "physics_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(out_dir, "physics_scene.xml"), "w") as f:
        f.write(xml)
    print(json.dumps(report, indent=1))
    print(f"wrote {out_dir} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
