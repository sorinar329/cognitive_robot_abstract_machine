#!/usr/bin/env python3
"""Inspection round with the Unitree G1 (tasks T1 and T2).

The G1 starts inside a zone, walks to the standpoint of each inspection point
in turn, aims its D435 camera at it, checks that the point is inside the camera
image, and reports what it finds there.

  --zone ground    T1: foundation and tower base, on the flat ground around the tower
                   (T0 is the door alone: --zone ground --points tower_inspect_door)
  --zone nacelle   T2: gearbox, main shaft and brake checks from the -Y walkway

Findings are read from the scenario's ground truth (an "oracle" perception);
recognising the faults in rendered camera images is a later step, so the report
labels them as such.

Run with a CRAM environment that has coraplex, giskardpy and the G1 model, with
ROS and the workspaces sourced (see README.md). To record the
run for CRAMERA, wrap it with the onboarder:

  python -m cramera.onboard.demo scripts/g1_inspection_round.py --name windturbine_g1_t1 -- --zone ground --scenario outside_ground
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from coraplex.datastructures.dataclasses import Context  # noqa: E402
from coraplex.datastructures.enums import Arms  # noqa: E402
from coraplex.exceptions import MotionDidNotFinish  # noqa: E402
from coraplex.execution_environment import simulated_robot  # noqa: E402
from coraplex.plans.factories import sequential  # noqa: E402
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction  # noqa: E402
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction  # noqa: E402
from semantic_digital_twin.api import RobotSpecification, WorldSpecification  # noqa: E402
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1  # noqa: E402
from semantic_digital_twin.spatial_types import Vector3  # noqa: E402
from semantic_digital_twin.spatial_types.spatial_types import Pose  # noqa: E402

try:   # coraplex with route planning (cramera-port); older stacks drive straight anyway
    from coraplex.robot_plans.motions.navigation import MoveMotion  # noqa: E402
    from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose  # noqa: E402
except ImportError:
    MoveMotion = None

from turbine import parts  # noqa: E402
from turbine import dims as d  # noqa: E402
from turbine.dims import site  # noqa: E402

PELVIS_HEIGHT_ABOVE_FLOOR = 0.7923 + 0.005
"""The G1's pelvis height with all leg joints at zero (0.7923 m, from the G1 warehouse demo),
plus 5 mm so its feet do not touch the floor's collision box (giskard treats contact
as a collision violation)."""
WALKWAY_STAND = (1.5, 1.75)
"""|y| of the G1's pelvis on a nacelle walkway (walkway 1.35-2.08 m from the axis)."""
WALKWAY_LANE = (d.WALKWAY_Y + d.INNER_Y[1]) / 2   # walkway centre, 1.715 m
"""|y| of the lane the G1 walks along between standpoints, facing along the walkway: its parked
hands (0.35 m ahead) then point along the walkway instead of into the drivetrain."""
RACK_CLEARANCE = 0.3
"""Standpoints keep this distance (x) from the tool rack at the -Y walkway's rear end."""
CAMERA_HEIGHT_ABOVE_FLOOR = 1.27
"""D435 height with the G1 standing upright."""
MIN_AIM_DEPRESSION = math.radians(19.0)
PREFERRED_AIM_DEPRESSION = math.radians(24.0)
TARGET_ABOVE_AXIS = math.radians(19.0)
"""The optical axis points 48 deg down; with the waist pitched back to its 0.52 rad limit it
still points ~18 deg down. Shallow targets are aimed at from below: the camera points
19-24 deg down along the same bearing and the target lands in the upper image (at most
19 deg above the axis; the half vertical field of view is 21.5 deg)."""


def use_optical_axis(robot):
    """Aim with the D435's x axis.

    The G1 model mounts ``d435_link`` pitched 0.83 rad on the torso. By ROS link
    convention the camera looks along the link's x axis, i.e. about 48 deg downwards,
    as on the real G1. CRAM's ``UnitreeG1`` annotation declares z as the forward axis,
    which points about 42 deg upwards; this corrects it for this run only.
    """
    camera = robot.get_default_camera()
    camera.forward_facing_axis = Vector3(1, 0, 0, reference_frame=camera.root)
    return camera

ZONES = {
    # zone: (floor height in the world, start x, y, yaw, inspection points in visiting order)
    "ground": (0.0, 6.0, -14.0, 0.0, [
        "tower_inspect_base_coating", "foundation_inspect_plinth", "foundation_inspect_grout",
        "foundation_inspect_earthing", "tower_inspect_anchor_nuts", "tower_inspect_door",
    ]),
    # nacelle yaw 0: nacelle frame = world frame shifted up to the nacelle floor
    # only points at or below the G1's camera height (scripts/check_inspection_points.py)
    "nacelle": (site.NACELLE_ORIGIN_Z, -0.6, -1.7, 0.0, [
        "main_shaft_inspect_shrink_disc", "main_bearing_inspect_grease_collector",
        "gearbox_inspect_input_seal", "gearbox_inspect_bushing_left", "gearbox_inspect_sight_glass",
        "generator_inspect_slip_ring",
    ]),
}


if MoveMotion is not None:
    class StraightMove(MoveMotion):
        """Drive the base straight to the target, without route planning.

        The route planner reserves the robot's axis-aligned footprint plus clearance
        and wants room to turn; a 0.69 m nacelle walkway offers neither. On the walkway
        the G1 moves sideways between standpoints, as the real robot would.
        """

        @property
        def _motion_chart(self):
            return CartesianPose(root_link=self.world.root, tip_link=self.robot.root, goal_pose=self.target)


def walkway_route(world, robot, stand, floor):
    """Waypoints to a nacelle standpoint along the walkway lane: turn along the walkway, walk down
    its middle, then turn to the standpoint (none if not both on the same walkway)."""
    here = robot.root.global_pose.to_np()[:3, 3]
    axis_y = world_pose(world, "nacelle")[1, 3]
    dy0, dy1 = here[1] - axis_y, stand.y - axis_y
    if abs(dy0) < 1.0 or abs(dy1) < 1.0 or dy0 * dy1 < 0 or abs(stand.x - here[0]) < 0.3:
        return []
    lane = axis_y + math.copysign(WALKWAY_LANE, dy1)
    heading = 0.0 if stand.x > here[0] else math.pi
    z = floor + PELVIS_HEIGHT_ABOVE_FLOOR
    return [Pose.from_xyz_rpy(here[0], lane, z, yaw=heading, reference_frame=world.root),
            Pose.from_xyz_rpy(stand.x, lane, z, yaw=heading, reference_frame=world.root)]


def move_to(stand, zone):
    """The motion that brings the base to a standpoint in this zone."""
    if zone == "nacelle" and MoveMotion is not None:
        return StraightMove(stand)
    return NavigateAction(stand)


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
    axis_y = world_pose(world, "nacelle")[1, 3]
    if point.get("zone") == "nacelle" and abs(camera[1] - axis_y) > 1.0:
        # on a walkway: stay inside it, clear of the bedplate girders, with room behind for the waist to lean
        # back (the MuJoCo replay found the hips on the girder and the head at the wall otherwise)
        y = min(max(abs(camera[1] - axis_y), WALKWAY_STAND[0]), WALKWAY_STAND[1])
        camera[1] = axis_y + math.copysign(y, camera[1] - axis_y)
        rack_x = world_pose(world, "nacelle")[0, 3] + np.array(d.TOOL_RACK_X)
        if camera[1] < axis_y and rack_x[0] - RACK_CLEARANCE < camera[0] < rack_x[1] + RACK_CLEARANCE:
            camera[0] = rack_x[1] + RACK_CLEARANCE
    yaw = math.atan2(target[1] - camera[1], target[0] - camera[0])
    return camera[0], camera[1], yaw, target


def aim_point(target, x, y, floor):
    """Where to point the optical axis so the target is in the image: the target itself,
    or for shallow targets a point below it on the same bearing."""
    horizontal = math.hypot(target[0] - x, target[1] - y)
    drop = floor + CAMERA_HEIGHT_ABOVE_FLOOR - target[2]
    target_depression = math.atan2(drop, horizontal)
    if target_depression >= PREFERRED_AIM_DEPRESSION:
        return target
    aim = max(MIN_AIM_DEPRESSION, min(PREFERRED_AIM_DEPRESSION, target_depression + TARGET_ABOVE_AXIS))
    return np.array([target[0], target[1], floor + CAMERA_HEIGHT_ABOVE_FLOOR - horizontal * math.tan(aim)])


def in_field_of_view(robot, target):
    camera = robot.get_default_camera()
    t = np.linalg.inv(camera.root.global_pose.to_np()) @ np.append(target, 1.0)
    forward = t[0]                                        # optical axis x, image left y, image up z
    horizontal = math.degrees(math.atan2(abs(t[1]), forward))
    vertical = math.degrees(math.atan2(abs(t[2]), forward))
    fov = camera.field_of_view
    ok = (forward > 0 and horizontal < math.degrees(fov.horizontal_angle) / 2
          and vertical < math.degrees(fov.vertical_angle) / 2)
    return ok, forward, horizontal, vertical


def findings(scenario, point_name):
    """Faults the ground truth places at this inspection point (oracle perception)."""
    if not scenario:
        return []
    path = os.path.join(ROOT, "urdf", "scenarios", scenario + "_ground_truth.yaml")
    with open(path) as f:
        truth = yaml.safe_load(f)
    return [f for f in truth["faults"] if f["inspection_point"] == point_name]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zone", choices=sorted(ZONES), default="ground")
    ap.add_argument("--scenario", help="scenario name under urdf/scenarios (default: healthy turbine)")
    ap.add_argument("--points", nargs="*", help="only these inspection points")
    args = ap.parse_args()
    urdf = (os.path.join(ROOT, "urdf", "scenarios", args.scenario + ".urdf") if args.scenario
            else os.path.join(ROOT, "urdf", "windturbine.urdf"))
    floor, sx, sy, syaw, names = ZONES[args.zone]
    names = args.points or names

    start = Pose.from_xyz_rpy(sx, sy, floor + PELVIS_HEIGHT_ABOVE_FLOOR, yaw=syaw)
    world = WorldSpecification.from_urdf(
        urdf, robots=[RobotSpecification(semantic_annotation_type=UnitreeG1, world_T_odom=start.to_homogeneous_matrix())]
    ).to_domain_object()
    robot = world.get_semantic_annotations_by_type(UnitreeG1)[0]
    use_optical_axis(robot)
    context = Context(world=world, robot=robot, evaluate_conditions=False)

    with simulated_robot:
        sequential([ParkArmsAction(Arms.BOTH)], context=context).plan.perform()

    report = []
    for name in names:
        point = inspection_point(name)
        x, y, yaw, target = standpoint(world, point)
        stand = Pose.from_xyz_rpy(x, y, floor + PELVIS_HEIGHT_ABOVE_FLOOR, yaw=yaw, reference_frame=world.root)
        aim = aim_point(target, x, y, floor)
        look = Pose.from_xyz_rpy(*aim, reference_frame=world.root)
        route = walkway_route(world, robot, stand, floor) if args.zone == "nacelle" else []
        with simulated_robot:
            sequential([move_to(p, args.zone) for p in route + [stand]], context=context).plan.perform()
        converged = True
        try:
            with simulated_robot:
                sequential([LookAtAction(look)], context=context).plan.perform()
        except MotionDidNotFinish:      # waist at its limits: what counts is whether the target is in the image
            converged = False
        seen, forward, h, v = in_field_of_view(robot, target)
        found = findings(args.scenario, name)
        report.append((name, point["what"], seen, found))
        print(f"[{'seen' if seen else 'MISSED'}] {name}: {forward:.2f} m ahead, {h:.1f}/{v:.1f} deg off centre"
              + ("" if converged else " (aim stopped at the waist limits)") + "; "
              + (", ".join(f["id"] for f in found) if found else "no finding"), flush=True)

    print("\n=== inspection report (findings from the ground truth, not yet from camera images) ===")
    for name, what, seen, found in report:
        state = "NOT SEEN" if not seen else ("FAULT" if found else "ok")
        print(f"{state:8s} {name:42s} {what}")
        for f in found:
            print(f"         -> [{f['severity']}] {f['description']}")
    missed = [r for r in report if not r[2]]
    print(f"\n{len(report) - len(missed)}/{len(report)} inspection points seen")
    if missed:     # no exit on success, so a recorder wrapping this script (cramera-onboard) can finish
        sys.exit(1)


if __name__ == "__main__":
    main()
