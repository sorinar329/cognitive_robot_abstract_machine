#!/usr/bin/env python3
"""Parse URDFs with CRAM's semantic_digital_twin and check that every mesh exists.

~/cram/cram_venv/bin/python scripts/check_cram.py [urdf ...]   (default: all generated URDFs)
"""
import glob
import os
import re
import sys

import numpy as np
import yaml
from semantic_digital_twin.adapters.urdf import URDFParser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
paths = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "urdf", "*.urdf")) +
                               glob.glob(os.path.join(ROOT, "urdf", "scenarios", "*.urdf")))
ok = True
for path in paths:
    text = open(path).read()
    missing = [m for m in re.findall(r'package://windturbine_model/([^"]+)', text)
               if not os.path.exists(os.path.join(ROOT, m))]
    world = URDFParser(urdf=text, package_resolver={"windturbine_model": ROOT}).parse()
    world.validate()
    kinds = {}
    for conn in world.connections:
        kinds[type(conn).__name__] = kinds.get(type(conn).__name__, 0) + 1
    # apply the scenario's initial joint states, as a CRAM loader would
    truth = path.replace(".urdf", "_ground_truth.yaml")
    applied = 0
    if os.path.exists(truth):
        by_name = {conn.name.name: conn for conn in world.connections}
        for joint, pos in (yaml.safe_load(open(truth)).get("initial_joint_states") or {}).items():
            by_name[joint].position = pos
            assert np.isclose(by_name[joint].position, pos), joint
            applied += 1
    status = ("OK" if not missing else f"MISSING {missing}") + (f", {applied} joint states set" if applied else "")
    ok &= not missing
    print(f"{os.path.relpath(path, ROOT):45s} bodies={len(world.kinematic_structure_entities):3d} {kinds} {status}")
sys.exit(0 if ok else 1)
