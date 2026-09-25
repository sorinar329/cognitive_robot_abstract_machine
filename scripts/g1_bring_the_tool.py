#!/usr/bin/env python3
"""Task T3 with the Unitree G1: find the loose bolt, then bring the tool.

1. From the -Y walkway the G1 checks the shrink disc's torque markings and finds the
   bolt that backed out (scenario ``bring_the_tool``).
2. It walks to the tool rack at the rear of the nacelle and picks up the torque tool
   case.
3. It carries the case along the walkway and puts it on the tool tray next to the
   technician at the front main bearing.

Done when the loose bolt was seen and the case rests on the tray.

Run in the full CRAM stack (cramera-port .venv) with ROS and the workspaces sourced;
record it for CRAMERA with the onboarder:

  python -m cramera.onboard.demo scripts/g1_bring_the_tool.py --name windturbine_g1_t3
  python3 scripts/finish_recording.py ~/.cramera/scenes/windturbine_g1_t3 --view g1_nacelle
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from coraplex.datastructures.dataclasses import Context  # noqa: E402
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment  # noqa: E402
from coraplex.datastructures.grasp import GraspDescription  # noqa: E402
from coraplex.exceptions import MotionDidNotFinish  # noqa: E402
from coraplex.execution_environment import simulated_robot  # noqa: E402
from coraplex.plans.factories import sequential  # noqa: E402
from coraplex.robot_plans.actions.core.navigation import LookAtAction  # noqa: E402
from coraplex.robot_plans.actions.core.pick_up import PickUpAction  # noqa: E402
from coraplex.robot_plans.actions.core.placing import PlaceAction  # noqa: E402
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction  # noqa: E402
from coraplex.view_manager import ViewManager  # noqa: E402
from semantic_digital_twin.api import BodySpecification, RobotSpecification, WorldSpecification  # noqa: E402
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1  # noqa: E402
from semantic_digital_twin.spatial_types.spatial_types import Pose  # noqa: E402
from semantic_digital_twin.world_description.geometry import Color, Scale  # noqa: E402

import g1_inspection_round as rnd  # noqa: E402
from turbine import dims as d  # noqa: E402
from turbine.dims import site  # noqa: E402

SCENARIO = "bring_the_tool"
FLOOR = site.NACELLE_ORIGIN_Z               # nacelle yaw 0: nacelle frame = world frame shifted up
PELVIS = FLOOR + rnd.PELVIS_HEIGHT_ABOVE_FLOOR
WALKWAY_Y = -1.7
STANDING_DISTANCE = 0.6
"""How far the robot stands from an object it grasps or places (as in the G1 warehouse demo)."""
PLACE_TOLERANCE = 0.06
"""How far the case may end up from the tray centre, in metres."""

CASE_SIZE = d.TOOL_CASE
CASE_START = Pose.from_xyz_rpy(sum(d.TOOL_RACK_X) / 2, d.TOOL_RACK_Y[1] - 0.1, FLOOR + d.TOOL_RACK_TOP + CASE_SIZE[2] / 2,
                               yaw=-math.pi / 2)
"""On the rack shelf, near its front lip; yaw = the direction the robot reaches in (towards -Y)."""
CASE_TARGET = (sum(d.TOOL_TRAY_X) / 2, sum(d.TOOL_TRAY_Y) / 2, FLOOR + d.TOOL_TRAY_TOP + CASE_SIZE[2] / 2 + 0.005)
"""Centre of the tool tray next to the technician; the robot reaches towards +Y."""


def standing_pose(x, y, yaw, world):
    """The base pose STANDING_DISTANCE behind (x, y), facing along yaw."""
    return Pose.from_xyz_rpy(x - STANDING_DISTANCE * math.cos(yaw), y - STANDING_DISTANCE * math.sin(yaw), PELVIS,
                             yaw=yaw, reference_frame=world.root)


def walk(context, world, *waypoints):
    """Straight base moves along the walkway (see g1_inspection_round.StraightMove)."""
    with simulated_robot:
        sequential([rnd.move_to(Pose.from_xyz_rpy(x, y, PELVIS, yaw=yaw, reference_frame=world.root), "nacelle")
                    for x, y, yaw in waypoints], context=context).plan.perform()


def main():
    urdf = os.path.join(ROOT, "urdf", "scenarios", SCENARIO + ".urdf")
    start = Pose.from_xyz_rpy(-0.6, WALKWAY_Y, PELVIS)
    world = WorldSpecification.from_urdf(
        urdf,
        robots=[RobotSpecification(semantic_annotation_type=UnitreeG1, world_T_odom=start.to_homogeneous_matrix())],
        objects=[BodySpecification.box("torque_tool_case", Scale(*CASE_SIZE), color=Color(0.85, 0.15, 0.1),
                                       parent_T_self=CASE_START.to_homogeneous_matrix())],
    ).to_domain_object()
    robot = world.get_semantic_annotations_by_type(UnitreeG1)[0]
    rnd.use_optical_axis(robot)
    context = Context(world=world, robot=robot, evaluate_conditions=False)
    case = world.get_body_by_name("torque_tool_case")

    with simulated_robot:
        sequential([ParkArmsAction(Arms.BOTH)], context=context).plan.perform()

    # 1. check the shrink disc's torque markings
    point = rnd.inspection_point("main_shaft_inspect_shrink_disc")
    x, y, yaw, target = rnd.standpoint(world, point)
    walk(context, world, (x, y, yaw))
    try:
        with simulated_robot:
            sequential([LookAtAction(Pose.from_xyz_rpy(*rnd.aim_point(target, x, y, FLOOR), reference_frame=world.root))],
                       context=context).plan.perform()
    except MotionDidNotFinish:
        pass
    seen, *_ = rnd.in_field_of_view(robot, target)
    found = rnd.findings(SCENARIO, point["name"])
    print(f"[{'seen' if seen else 'MISSED'}] shrink disc: " + (", ".join(f["id"] for f in found) or "markings aligned"), flush=True)

    # 2. fetch the torque tool case from the rack at the rear
    grasp = GraspDescription(ApproachDirection.FRONT, VerticalAlignment.NoAlignment,
                             ViewManager.get_end_effector_view(Arms.LEFT, robot))
    rack_stand = standing_pose(CASE_START.x, CASE_START.y, -math.pi / 2, world)
    walk(context, world, (-7.6, WALKWAY_Y, math.pi), (rack_stand.x, rack_stand.y, -math.pi / 2))
    with simulated_robot:
        sequential([PickUpAction(case, Arms.LEFT, grasp), ParkArmsAction(Arms.BOTH)], context=context).plan.perform()
    print("picked up the torque tool case", flush=True)

    # 3. carry it to the technician's tray
    tray_stand = standing_pose(CASE_TARGET[0], CASE_TARGET[1], math.pi / 2, world)
    walk(context, world, (-7.6, WALKWAY_Y, 0.0), (tray_stand.x, tray_stand.y, math.pi / 2))
    place = Pose.from_xyz_rpy(*CASE_TARGET, yaw=math.pi / 2, reference_frame=world.root)
    with simulated_robot:
        sequential([PlaceAction(case, place, Arms.LEFT), ParkArmsAction(Arms.BOTH)], context=context).plan.perform()

    final = case.global_pose.to_np()[:3, 3]
    off = float(np.linalg.norm(final[:2] - np.array(CASE_TARGET[:2])))
    on_tray = off < PLACE_TOLERANCE and final[2] > FLOOR + d.TOOL_TRAY_TOP
    print(f"case at {np.round(final - [0, 0, FLOOR], 3)} (nacelle frame), {off * 100:.1f} cm from the tray centre")
    print(f"\n=== T3 report ===\nloose bolt found: {bool(seen and found)}\ntool delivered to the technician: {on_tray}")
    if not (seen and found and on_tray):     # no exit on success, so cramera-onboard can finish recording
        sys.exit(1)


if __name__ == "__main__":
    main()
