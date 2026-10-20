"""
Smoke check for the i40lab MJCF scene: parse the world, verify free bodies exist, every
robot body carries collision geometry, and the contact tuning resolves hard and
uniformly.

Run from repository root:

    python coraplex/demos/i40lab/test.py

Exits 0 on success, non-zero on failure. The contact tuning check needs mujoco
through ``mujoco_demo`` and is skipped without it; everything else needs no
external dependencies (ROS, physics).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.contact import ContactParameters

SCENE = Path(__file__).parent / "mujocoscene" / "scene.xml"
EXPECTED_FREE_BODIES = {"cylinder", "Box", "T_block"}
EXPECTED_COLLISION_BODIES = {
    "ur5e_base",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "flange",
    "mount_2fg7",
    "mount_3fg15",
}
"""
Bodies of the robot assembly that must carry collision geometry.

The TCP debug frames stay collision-free on purpose: they sit at the grasp points, where
contact would break grasping.
"""


def main() -> int:
    if not SCENE.exists():
        print(f"ERROR: scene not found: {SCENE}")
        return 1

    world = MJCFParser.from_file(str(SCENE)).parse()
    free_bodies = {
        body.name.name
        for body in world.bodies
        if isinstance(body.parent_connection, Connection6DoF)
    }

    missing = EXPECTED_FREE_BODIES - free_bodies
    if missing:
        print(f"ERROR: missing expected free bodies: {sorted(missing)}")
        print(f"  found free bodies: {sorted(free_bodies)}")
        return 1

    bodies = {body.name.name: body for body in world.bodies}
    bare = sorted(
        name
        for name in EXPECTED_COLLISION_BODIES
        if name not in bodies or not bodies[name].collision.shapes
    )
    if bare:
        print(f"ERROR: robot bodies without collision geometry: {bare}")
        return 1

    soft = check_rigid_contact_tuning()
    if soft:
        print(f"ERROR: contacts not tuned rigid: {soft}")
        return 1

    headless_errors = check_headless_override()
    if headless_errors:
        print(f"ERROR: headless override broken: {headless_errors}")
        return 1

    grasp_errors = check_raised_grasp()
    if grasp_errors:
        print(f"ERROR: raised grasp broken: {grasp_errors}")
        return 1

    print(f"OK: parsed {SCENE}")
    print(f"  total bodies: {len(world.bodies)}")
    print(f"  free bodies: {sorted(free_bodies)}")
    return 0


def check_rigid_contact_tuning() -> list[str]:
    """
    Every collision shape resolves with the demo's rigid stiffness and impedance,
    keeping the grasp friction the T-block already carries.

    Runs against the world ``demo.build_world`` assembles — the same world the demo
    carries out its plan in — rather than the bare parse above, so the grasp friction is
    present to preserve. Needs ``mujoco_demo`` and is skipped without it, so the rest of
    this check keeps running wherever mujoco is unavailable.

    :return: Descriptions of every shape violating the tuning, if any.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import demo as i40lab_demo
        import mujoco_demo
    except ImportError:
        print("  SKIP contact tuning check (mujoco_demo unavailable)")
        return []
    finally:
        sys.path.remove(str(Path(__file__).parent))

    world, _ = i40lab_demo.build_world()
    mujoco_demo._make_every_contact_rigid(world)
    grasped = ContactParameters.create_for_grasped_object()
    violations = []
    for body in world.bodies_with_collision:
        for shape in body.collision.shapes:
            contact = shape.get_simulator_property_of_type(ContactParameters)
            if contact is None:
                violations.append(f"{body.name.name}: no contact parameters")
            elif (
                contact.stiffness != mujoco_demo.RIGID_CONTACT_STIFFNESS
                or contact.impedance != mujoco_demo.RIGID_CONTACT_IMPEDANCE
            ):
                violations.append(f"{body.name.name}: soft contact parameters")
            elif (
                body.name.name == i40lab_demo.CYLINDER_NAME
                and contact.friction != grasped.friction
            ):
                violations.append(f"{body.name.name}: grasp friction not preserved")
    return violations


def check_headless_override() -> list[str]:
    """
    The viewer switch defaults to on and follows the environment variable.

    Needs ``mujoco_demo`` and is skipped without it, like the tuning check.

    :return: Descriptions of every violated expectation, if any.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import mujoco_demo
    except ImportError:
        print("  SKIP headless check (mujoco_demo unavailable)")
        return []
    finally:
        sys.path.remove(str(Path(__file__).parent))

    previous = os.environ.get(mujoco_demo.MujocoDemoOption.HEADLESS)
    try:
        os.environ.pop(mujoco_demo.MujocoDemoOption.HEADLESS, None)
        if mujoco_demo.headless_from_environment() is not mujoco_demo.HEADLESS:
            return ["unset variable does not keep the module default"]
        os.environ[mujoco_demo.MujocoDemoOption.HEADLESS] = "1"
        if mujoco_demo.headless_from_environment() is not True:
            return ["set variable does not request headless"]
        return []
    finally:
        if previous is None:
            os.environ.pop(mujoco_demo.MujocoDemoOption.HEADLESS, None)
        else:
            os.environ[mujoco_demo.MujocoDemoOption.HEADLESS] = previous


def check_raised_grasp() -> list[str]:
    """
    The plan grasps above the cylinder origin, so the claw wraps the upper wall instead
    of reaching near the table.

    Needs the plan builders and is skipped without them, like the other demo-dependent
    checks.

    :return: Descriptions of every violated expectation, if any.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import demo as i40lab_demo
        from coraplex.datastructures.dataclasses import Context
        from coraplex.plans.plan_node import ActionNode
        from coraplex.robot_plans.actions.core.pick_up import ReachAction
    except ImportError:
        print("  SKIP grasp check (plan builders unavailable)")
        return []
    finally:
        sys.path.remove(str(Path(__file__).parent))

    world, robot = i40lab_demo.build_world()
    context = Context(
        world=world,
        robot=robot,
        evaluate_conditions=False,
        update_world_model_attachment=False,
    )
    plan = i40lab_demo.build_motion_plan(context)
    reaches = [
        node.designator
        for node in plan.root.children
        if isinstance(node, ActionNode) and isinstance(node.designator, ReachAction)
    ]
    if not reaches:
        return ["no reach action in plan"]
    origin_z = float(
        world.get_body_by_name(i40lab_demo.CYLINDER_NAME).global_pose.to_position().z
    )
    violations = []
    for reach in reaches:
        height = float(reach.target_pose.to_position().z)
        if (
            abs(height - (origin_z + i40lab_demo.RAISED_GRASP_OFFSET)) > 1e-9
            or reach.arm.name != "RIGHT"
        ):
            violations.append(f"grasp target off: {height}")
    return violations


if __name__ == "__main__":
    raise SystemExit(main())
