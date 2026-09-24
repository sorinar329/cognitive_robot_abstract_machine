#!/usr/bin/env python3
"""Task T0: the Unitree G1 walks from its start to the standpoint in front of the
tower door and aims its D435 camera at the door.

Done when the door's inspection point lies inside the camera's field of view.

Run with a CRAM environment that has coraplex, giskardpy and the G1 model
(e.g. ~/.virtualenvs/cram2-env), with ROS and the workspaces sourced:

  source /opt/ros/jazzy/setup.bash
  source ~/workspace/segmind_ws/install/setup.bash      # iai_offis_g1_description
  source ~/workspace/windturbine_model/install/setup.bash
  python scripts/g1_t0_look_at_door.py [--scenario outside_ground]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from coraplex.datastructures.dataclasses import Context  # noqa: E402
from coraplex.datastructures.enums import Arms  # noqa: E402
from coraplex.execution_environment import simulated_robot  # noqa: E402
from coraplex.plans.factories import sequential  # noqa: E402
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction  # noqa: E402
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction  # noqa: E402
from semantic_digital_twin.api import RobotSpecification, WorldSpecification  # noqa: E402
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1  # noqa: E402
from semantic_digital_twin.spatial_types.spatial_types import Pose  # noqa: E402

from turbine import parts  # noqa: E402

PELVIS_HEIGHT_ABOVE_FLOOR = 0.7923
"""The G1's pelvis height with all leg joints at zero (from the G1 warehouse demo)."""
START = (6.0, -14.0, 0.0)
"""Start on the gravel apron south-east of the tower: x, y, yaw."""
TARGET = "tower_inspect_door"


def inspection_point(name):
    return next(p for part in parts.ALL for p in part.INSPECTION_POINTS if p["name"] == name)


def world_pose(world, body_name):
    return world.get_body_by_name(body_name).global_pose.to_np()


def standpoint(world, point):
    """Where the robot stands: below the camera position the inspection point asks
    for, facing the target."""
    target = world_pose(world, point["name"])[:3, 3]
    parent = world_pose(world, point["parent"])[:3, :3]
    direction = parent @ np.array(point["view_from"], dtype=float)
    direction /= np.linalg.norm(direction)
    camera = target + point["distance"] * direction
    yaw = math.atan2(target[1] - camera[1], target[0] - camera[0])
    return camera[0], camera[1], yaw, target


def in_field_of_view(world, robot, target):
    camera = robot.get_default_camera()
    t = np.linalg.inv(camera.root.global_pose.to_np()) @ np.append(target, 1.0)
    forward = t[2]                                        # D435: forward axis is Z
    horizontal = math.degrees(math.atan2(abs(t[0]), forward))
    vertical = math.degrees(math.atan2(abs(t[1]), forward))
    fov = camera.field_of_view
    ok = forward > 0 and horizontal < math.degrees(fov.horizontal_angle) / 2 and vertical < math.degrees(fov.vertical_angle) / 2
    return ok, forward, horizontal, vertical


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", help="scenario name under urdf/scenarios (default: healthy turbine)")
    args = ap.parse_args()
    urdf = os.path.join(ROOT, "urdf", "scenarios", args.scenario + ".urdf") if args.scenario else os.path.join(ROOT, "urdf", "windturbine.urdf")

    start = Pose.from_xyz_rpy(START[0], START[1], PELVIS_HEIGHT_ABOVE_FLOOR, yaw=START[2])
    world = WorldSpecification.from_urdf(
        urdf, robots=[RobotSpecification(semantic_annotation_type=UnitreeG1, world_T_odom=start.to_homogeneous_matrix())]
    ).to_domain_object()
    robot = world.get_semantic_annotations_by_type(UnitreeG1)[0]

    point = inspection_point(TARGET)
    x, y, yaw, target = standpoint(world, point)
    print(f"standpoint for {TARGET}: x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.1f} deg, target {np.round(target, 2)}")

    context = Context(world=world, robot=robot, evaluate_conditions=False)
    stand = Pose.from_xyz_rpy(x, y, PELVIS_HEIGHT_ABOVE_FLOOR, yaw=yaw, reference_frame=world.root)
    look = Pose.from_xyz_rpy(*target, reference_frame=world.root)
    plan = sequential([ParkArmsAction(Arms.BOTH), NavigateAction(stand), LookAtAction(look)], context=context).plan
    with simulated_robot:
        plan.perform()

    ok, forward, h, v = in_field_of_view(world, robot, target)
    base = world_pose(world, robot.root.name.name)[:3, 3]
    print(f"robot at {np.round(base[:2], 2)}; target {forward:.2f} m ahead, {h:.1f} deg off horizontally, {v:.1f} deg vertically")
    print("T0 DONE: door is in the camera image" if ok else "T0 FAILED: door not in the camera image")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
