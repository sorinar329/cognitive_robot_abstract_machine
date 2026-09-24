#!/usr/bin/env python3
"""Load the nacelle into a CRAM semantic world and show it in RViz.

Uses CRAM's own TFPublisher + VizMarkerPublisher, i.e. what you see is the
world as CRAM holds it (after applying the scenario's initial joint states).

  source /opt/ros/jazzy/setup.bash
  ~/cram/cram_venv/bin/python scripts/show_in_cram.py [urdf] [--with-cover] [--no-rviz]

The nacelle cover is left out by default so you can see inside.
"""
import os
import subprocess
import sys
import time

import rclpy
import yaml
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world_description.geometry import Color, FileMesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
urdf_path = os.path.abspath(args[0] if args else os.path.join(ROOT, "urdf", "windturbine.urdf"))

with open(urdf_path) as f:
    world = URDFParser(urdf=f.read(), package_resolver={"windturbine_model": ROOT}).parse()

# a transparent marker colour makes RViz use the colours from the .mtl files
for body in world.bodies:
    if "--with-cover" not in sys.argv:
        body.visual.shapes[:] = [s for s in body.visual.shapes
                                 if not (isinstance(s, FileMesh) and s.filename.endswith("nacelle/cover.obj"))]
    for shape in body.visual.shapes:
        if isinstance(shape, FileMesh):
            shape.color = Color(0.0, 0.0, 0.0, 0.0)

truth_path = urdf_path.replace(".urdf", "_ground_truth.yaml")
if os.path.exists(truth_path):
    with open(truth_path) as f:
        truth = yaml.safe_load(f)
    by_name = {c.name.name: c for c in world.connections}
    for joint, pos in (truth.get("initial_joint_states") or {}).items():
        by_name[joint].position = pos
    print("active faults:", flush=True)
    for fault in truth["faults"]:
        print(f"  [{fault['severity']:6s}] {fault['id']}: {fault['description']}  (look at {fault['inspection_point']})")

rclpy.init()
node = rclpy.create_node("windturbine_cram_viz")
tf = TFPublisher(node=node, world=world)
viz = VizMarkerPublisher(world=world, node=node)
root = str(world.root.name)

rviz = None
if "--no-rviz" not in sys.argv:
    cfg = os.path.join(ROOT, "rviz", "cram.rviz")
    with open(cfg) as f:
        text = f.read().replace("__ROOT_FRAME__", root)
    tmp = os.path.join("/tmp", f"windturbine_cram_{os.getpid()}.rviz")
    with open(tmp, "w") as f:
        f.write(text)
    rviz = subprocess.Popen(["rviz2", "-d", tmp])

print(f"publishing {len(world.bodies)} bodies, fixed frame '{root}', markers on {viz.topic_name}; Ctrl+C to stop", flush=True)
try:
    tick = 0
    while rviz is None or rviz.poll() is None:
        tf._notify()          # keep the (non-static) TF tree alive for RViz
        if tick % 6 == 0:     # re-send markers so RViz re-resolves them once TF is there
            viz._notify()
        tick += 1
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.4)
except KeyboardInterrupt:
    pass
finally:
    if rviz and rviz.poll() is None:
        rviz.terminate()
    node.destroy_node()
    rclpy.try_shutdown()
