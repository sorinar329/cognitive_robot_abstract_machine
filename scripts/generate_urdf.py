#!/usr/bin/env python3
"""Generate the nacelle URDF, healthy or with injected faults.

  generate_urdf.py                              healthy: urdf/windturbine.urdf
  generate_urdf.py --scenario scenarios/x.yaml  urdf/scenarios/x.urdf + x_ground_truth.yaml
  generate_urdf.py --random 2 --seed 7          urdf/scenarios/random_7.urdf + ground truth
  generate_urdf.py --list                       list all known faults

Always writes urdf/inspection_points.yaml as well.
"""
import argparse
import os
import random
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from turbine import parts, urdf  # noqa: E402


def collect():
    links, points, faults, signals = [], [], {}, {}
    for part in parts.ALL:
        links += part.LINKS
        points += part.INSPECTION_POINTS
        faults.update(part.FAULTS)
        signals.update(getattr(part, "SIGNALS", {}))
    return links, points, faults, signals


def extras(names):
    """Link specs for a scenario's ``extras`` (e.g. a technician)."""
    available = {k: v for part in parts.ALL for k, v in getattr(part, "EXTRAS", {}).items()}
    unknown = set(names) - set(available)
    if unknown:
        sys.exit(f"unknown extras {unknown}, available: {sorted(available)}")
    return [spec for n in names for spec in available[n]]


def build(fault_ids, links, points, faults, signals):
    variants = {l["name"]: l["variants"] for l in links if l["variants"]}
    states, extra, active, joint_states = {}, [], [], {}
    signals = dict(signals)
    for fid in fault_ids:
        if fid not in faults:
            sys.exit(f"unknown fault '{fid}', see --list")
        f = faults[fid]
        for name, state in f["set_variants"].items():
            if state not in variants.get(name, {}):
                sys.exit(f"fault {fid}: link {name} has no variant '{state}'")
            states[name] = state
        extra += f["overlays"]
        signals.update(f["signals"])
        joint_states.update(f["joint_states"])
        active.append(dict(id=fid, **{k: f[k] for k in ("part", "component", "description", "severity",
                                                           "inspection_point", "observable_by")}))
    frames = [urdf.link(p["name"], p["parent"], xyz=p["xyz"]) for p in points]
    all_links = links + extra + frames
    names = [l["name"] for l in all_links]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        sys.exit(f"duplicate link names: {dupes}")
    joints = {urdf.joint_name(l) for l in all_links}
    unknown = set(joint_states) - joints
    if unknown:
        sys.exit(f"unknown joints in joint_states: {unknown}")
    return all_links, states, dict(faults=active, initial_joint_states=joint_states, signals=signals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--scenario", help="YAML file with 'faults: [ids]'")
    g.add_argument("--random", type=int, metavar="N", help="inject N random faults")
    g.add_argument("--list", action="store_true", help="list known faults")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    links, points, faults, signals = collect()
    if args.list:
        for fid, f in faults.items():
            print(f"{fid:40s} [{f['severity']:6s}] {f['description']}")
        return

    with open(os.path.join(ROOT, "urdf", "inspection_points.yaml"), "w") as fh:
        yaml.safe_dump({"inspection_points": [dict(p, xyz=list(p["xyz"]), view_from=list(p["view_from"]))
                                              for p in points]}, fh, sort_keys=False)

    if args.scenario:
        with open(args.scenario) as fh:
            scenario = yaml.safe_load(fh)
        name = os.path.splitext(os.path.basename(args.scenario))[0]
        fault_ids = scenario.get("faults") or []
        links = links + extras(scenario.get("extras") or [])
    elif args.random is not None:
        name = f"random_{args.seed}"
        fault_ids = sorted(random.Random(args.seed).sample(sorted(faults), min(args.random, len(faults))))
    else:
        name, fault_ids = None, []

    all_links, states, truth = build(fault_ids, links, points, faults, signals)
    if name is None:
        out = os.path.join(ROOT, "urdf", "windturbine.urdf")
    else:
        os.makedirs(os.path.join(ROOT, "urdf", "scenarios"), exist_ok=True)
        out = os.path.join(ROOT, "urdf", "scenarios", name + ".urdf")
        with open(os.path.join(ROOT, "urdf", "scenarios", name + "_ground_truth.yaml"), "w") as fh:
            yaml.safe_dump(dict(scenario=name, **truth), fh, sort_keys=False)
    urdf.write(out, "windturbine", all_links, states)
    print(f"wrote {out} ({len(fault_ids)} faults)")


if __name__ == "__main__":
    main()
