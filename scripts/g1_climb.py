#!/usr/bin/env python3
"""Tasks T4 and T5 with the Unitree G1: get up the turbine.

The G1 walks on flat floors only (planar base), so it rides:

T4 ``--route hoist`` (scenario ``hoist_access``): the nacelle's service crane has
   lowered the lifting platform through the open rear floor hatch to the ground. The
   G1 walks from the tower door round to the platform, steps on, is hoisted 108 m,
   steps off onto the nacelle floor, checks the controller cabinet and walks to
   the walkway. Done when it stands on the nacelle walkway.
T5 ``--route lift`` (scenario ``tower_climb``): the G1 walks in through the tower door,
   into the service lift car, rides it to the yaw deck under the tower top and checks
   the cable loop. Done when it stands on the yaw deck and has seen the cable loop.

A ride: the robot's odom frame is re-parented to the moving body (lifting platform or
lift car), then one joint motion drives the carrier's joint; afterwards the odom goes
back to the world root. The recorded trajectory has the carrier's joint, so CRAMERA
replays the ride.

  python -m cramera.onboard.demo scripts/g1_climb.py --name windturbine_g1_t4 -- --route hoist
  python3 scripts/finish_recording.py ~/.cramera/scenes/windturbine_g1_t4 --view g1_hoist
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import yaml  # noqa: E402
from coraplex.datastructures.dataclasses import Context  # noqa: E402
from coraplex.datastructures.enums import Arms  # noqa: E402
from coraplex.exceptions import MotionDidNotFinish  # noqa: E402
from coraplex.execution_environment import simulated_robot  # noqa: E402
from coraplex.plans.factories import sequential  # noqa: E402
from coraplex.robot_plans.actions.core.navigation import LookAtAction  # noqa: E402
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction  # noqa: E402
from coraplex.robot_plans.motions.base import BaseMotion  # noqa: E402
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState  # noqa: E402
from semantic_digital_twin.api import RobotSpecification, WorldSpecification  # noqa: E402
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1  # noqa: E402
from semantic_digital_twin.spatial_types.spatial_types import Pose  # noqa: E402
from semantic_digital_twin.world_description.world_entity import Connection  # noqa: E402

import g1_inspection_round as rnd  # noqa: E402
from turbine import dims as d  # noqa: E402
from turbine.dims import site  # noqa: E402

GROUND = 0.0
NACELLE = site.NACELLE_ORIGIN_Z           # nacelle yaw 0: nacelle frame = world frame shifted up
YAW_DECK = site.TOWER_BASE_Z + site.platform_heights()[-1]
HOOK = d.CRANE_TRAVEL[0]                  # trolley home: the platform hangs at (HOOK, 0)
LIFT_X, LIFT_Y = site.LIFT_CENTER


@dataclass
class RideMotion(BaseMotion):
    """Drive one joint of the environment (the carrier the robot stands on)."""

    connection: Connection
    position: float
    speed: float

    def perform(self):
        return

    @property
    def _motion_chart(self):
        return JointPositionList(goal_state=JointState.from_mapping({self.connection: self.position}),
                                 max_velocity=self.speed)


def odom_of(robot):
    return robot.root.parent_connection.parent


def move_odom(world, odom, parent):
    """Re-parent the robot's odom frame, keeping its world pose (the world-to-odom joint is
    a 6-DoF joint, whose new DoFs must be created inside a modification block)."""
    world.update_forward_kinematics()
    with world.modify_world():
        world.move_branch(odom, parent, enable_unsafe_inside_world_block=True)


def ride(context, world, robot, carrier, joint_name, position, speed):
    """Stand still on ``carrier`` while its joint moves to ``position``; then back to the world."""
    connection = next(c for c in world.connections if c.name.name == joint_name)
    connection.has_hardware_interface = True
    odom = odom_of(robot)
    move_odom(world, odom, world.get_body_by_name(carrier))
    try:
        with simulated_robot:
            sequential([RideMotion(connection, position, speed)], context=context).plan.perform()
    finally:
        print(f"{carrier}: {joint_name} at {connection.position:.3f} (goal {position:.3f})", flush=True)
        move_odom(world, odom, world.root)
        connection.has_hardware_interface = False


def walk(context, world, floor, *waypoints):
    """Straight base moves (see g1_inspection_round.StraightMove) at this floor height."""
    pelvis = floor + rnd.PELVIS_HEIGHT_ABOVE_FLOOR
    with simulated_robot:
        sequential([rnd.move_to(Pose.from_xyz_rpy(x, y, pelvis, yaw=yaw, reference_frame=world.root), "nacelle")
                    for x, y, yaw in waypoints], context=context).plan.perform()


def inspect(context, world, robot, scenario, point_name, floor):
    """Walk to the inspection point's standpoint, look, report."""
    point = rnd.inspection_point(point_name)
    x, y, yaw, target = rnd.standpoint(world, point)
    walk(context, world, floor, (x, y, yaw))
    try:
        with simulated_robot:
            sequential([LookAtAction(Pose.from_xyz_rpy(*rnd.aim_point(target, x, y, floor), reference_frame=world.root))],
                       context=context).plan.perform()
    except MotionDidNotFinish:
        pass
    seen, *_ = rnd.in_field_of_view(robot, target)
    found = rnd.findings(scenario, point_name)
    print(f"[{'seen' if seen else 'MISSED'}] {point['what']}: " + (", ".join(f["id"] for f in found) or "ok"), flush=True)
    return seen, found


def load(scenario, start):
    urdf = os.path.join(ROOT, "urdf", "scenarios", scenario + ".urdf")
    world = WorldSpecification.from_urdf(
        urdf, robots=[RobotSpecification(semantic_annotation_type=UnitreeG1, world_T_odom=start.to_homogeneous_matrix())],
    ).to_domain_object()
    with open(urdf.replace(".urdf", "_ground_truth.yaml")) as fh:
        states = yaml.safe_load(fh).get("initial_joint_states") or {}
    by_name = {c.name.name: c for c in world.connections}
    for joint, pos in states.items():            # the scenario's hatch, hook, door positions
        by_name[joint].position = pos
    robot = world.get_semantic_annotations_by_type(UnitreeG1)[0]
    rnd.use_optical_axis(robot)
    context = Context(world=world, robot=robot, evaluate_conditions=False)
    with simulated_robot:
        sequential([ParkArmsAction(Arms.BOTH)], context=context).plan.perform()
    return world, robot, context


def pelvis_z(robot):
    return float(robot.root.global_pose.to_np()[2, 3])


def hoist():
    """T4: tower door -> lifting platform -> hoisted up -> nacelle floor -> walkway."""
    scenario = "hoist_access"
    world, robot, context = load(scenario, Pose.from_xyz_rpy(0.0, -5.5, GROUND + rnd.PELVIS_HEIGHT_ABOVE_FLOOR, yaw=math.pi))
    # round the tower to the platform; it is open at its +-X edges (toe boards on +-Y)
    walk(context, world, GROUND, (HOOK - 1.6, -5.5, math.pi / 2), (HOOK - 1.6, 0.0, 0.0), (HOOK, 0.0, 0.0))
    print("on the lifting platform", flush=True)
    ride(context, world, robot, "lifting_platform", "crane_hook_joint", d.HOOK_AT_FLOOR, d.HOOK_SPEED)
    up = pelvis_z(robot) - rnd.PELVIS_HEIGHT_ABOVE_FLOOR - NACELLE
    # off the platform, first check: the controller cabinet by the hatch
    seen, found = inspect(context, world, robot, scenario, "controller_inspect_door", NACELLE)
    # behind the slip ring, past the tool rack onto the -Y walkway
    walk(context, world, NACELLE, (-7.95, -0.4, 0.0), (-7.95, -1.25, 0.0), (-7.3, -1.7, 0.0), (-3.0, -1.7, 0.0))
    at = robot.root.global_pose.to_np()[:3, 3]
    on_walkway = abs(at[2] - NACELLE - rnd.PELVIS_HEIGHT_ABOVE_FLOOR) < 0.05 and at[1] < -d.WALKWAY_Y + 0.1
    print(f"\n=== T4 report ===\nhoisted to the nacelle floor: {abs(up) < 0.05} (floor offset {up:+.3f} m)"
          f"\ncontroller cabinet checked: {bool(seen)}\non the nacelle walkway: {on_walkway}", flush=True)
    if not (abs(up) < 0.05 and seen and on_walkway):
        sys.exit(1)


def lift():
    """T5: tower door -> lift car -> yaw deck -> cable loop."""
    scenario = "tower_climb"
    world, robot, context = load(scenario, Pose.from_xyz_rpy(0.0, -5.5, GROUND + rnd.PELVIS_HEIGHT_ABOVE_FLOOR, yaw=math.pi / 2))
    # in through the door, turn, and back into the car so the G1 faces the car door (+X)
    walk(context, world, GROUND, (0.0, -2.0, math.pi / 2), (0.25, -0.2, 0.0), (0.25, LIFT_Y, 0.0), (LIFT_X, LIFT_Y, 0.0))
    print("in the lift car", flush=True)
    ride(context, world, robot, "service_lift", "service_lift_joint", site.LIFT_TRAVEL, site.LIFT_SPEED)
    up = pelvis_z(robot) - rnd.PELVIS_HEIGHT_ABOVE_FLOOR - YAW_DECK
    walk(context, world, YAW_DECK, (0.2, -0.65, 0.0), (0.8, 0.1, math.pi / 2))     # round the cable opening
    seen, found = inspect(context, world, robot, scenario, "tower_inspect_cable_loop", YAW_DECK)
    print(f"\n=== T5 report ===\nlift at the yaw deck: {abs(up) < 0.05} (offset {up:+.3f} m)"
          f"\ncable loop checked: {bool(seen)} ({', '.join(f['id'] for f in found) or 'ok'})", flush=True)
    if not (abs(up) < 0.05 and seen):
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", choices=("hoist", "lift"), default="hoist")
    {"hoist": hoist, "lift": lift}[ap.parse_args().route]()
