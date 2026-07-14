"""
Demonstration of TIAGo++ arm movement in MuJoCo.

Loads the TIAGo++ velocity-actuated scene, places the robot in its home
position, then drives the left arm through a reach-and-return motion using
the velocity actuators defined in the model.

Run with::

    python -m segmind.demos.tiago_arm_demo
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import mujoco
import numpy

from physics_simulators.mujoco_simulator import MujocoSimulator

SCENE_FILE = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "resources",
        "tiago_episodes",
        "models",
        "assets",
        "mjcf",
        "scene_velocity.xml",
    )
)

_HOME_JOINT_ANGLES: dict[str, float] = {
    "arm_left_1_joint":  0.27,
    "arm_left_2_joint": -1.07,
    "arm_left_3_joint":  1.5,
    "arm_left_4_joint":  1.96,
    "arm_left_5_joint": -2.0,
    "arm_left_6_joint":  1.2,
    "arm_left_7_joint":  0.5,
    "arm_right_1_joint":  0.27,
    "arm_right_2_joint": -1.07,
    "arm_right_3_joint":  1.5,
    "arm_right_4_joint":  1.96,
    "arm_right_5_joint": -2.0,
    "arm_right_6_joint":  1.2,
    "arm_right_7_joint":  0.5,
}

_LEFT_ARM_ACTUATOR_NAMES: list[str] = [
    "arm_left_1_joint_velocity",
    "arm_left_2_joint_velocity",
    "arm_left_3_joint_velocity",
    "arm_left_4_joint_velocity",
    "arm_left_5_joint_velocity",
    "arm_left_6_joint_velocity",
    "arm_left_7_joint_velocity",
]

_LEFT_ARM_JOINT_NAMES: list[str] = [
    "arm_left_1_joint",
    "arm_left_2_joint",
    "arm_left_3_joint",
    "arm_left_4_joint",
    "arm_left_5_joint",
    "arm_left_6_joint",
    "arm_left_7_joint",
]


@dataclass
class ArmTarget:
    """A set of target joint angles and the velocity used to drive toward them.

    Joints not listed in :attr:`joint_targets` have their ctrl zeroed.
    """

    joint_targets: dict[str, float]
    """Mapping from joint name to target angle in radians."""

    velocity: float = 0.3
    """Magnitude of velocity (rad/s) applied to each joint being moved."""

    position_tolerance: float = 0.05
    """Tolerance in radians for declaring a joint to have reached its target."""


@dataclass
class TiagoArmDemo:
    """Controls the TIAGo++ left arm through a reach-and-return trajectory.

    Uses velocity actuators: setting ``ctrl[i]`` to ±:attr:`velocity`
    drives joint *i* toward its target; zeroing it holds the joint in place.

    .. note::
        The home position is set once in :meth:`initialise_home_pose` because
        the scene file has no keyframe.
    """

    simulator: MujocoSimulator
    """The MuJoCo simulator driving the scene."""

    _actuator_indices: dict[str, int] = field(init=False, default_factory=dict)
    """Mapping from actuator name to its index in ``mj_data.ctrl``."""

    _joint_qpos_addrs: dict[str, int] = field(init=False, default_factory=dict)
    """Mapping from joint name to its ``qpos`` address."""

    def __post_init__(self) -> None:
        model = self.simulator._mj_model
        for actuator_name in _LEFT_ARM_ACTUATOR_NAMES:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            self._actuator_indices[actuator_name] = actuator_id
        all_joint_names = _LEFT_ARM_JOINT_NAMES + list(_HOME_JOINT_ANGLES.keys())
        for joint_name in set(all_joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self._joint_qpos_addrs[joint_name] = model.jnt_qposadr[joint_id]

    def initialise_home_pose(self) -> None:
        """Write the home joint angles into ``qpos`` and propagate kinematics.

        Called once after :meth:`~physics_simulators.mujoco_simulator.MujocoSimulator.start`
        because the scene file contains no keyframe.
        """
        for joint_name, angle in _HOME_JOINT_ANGLES.items():
            addr = self._joint_qpos_addrs[joint_name]
            self.simulator._mj_data.qpos[addr] = angle
        mujoco.mj_forward(self.simulator._mj_model, self.simulator._mj_data)

    def joint_angle(self, joint_name: str) -> float:
        """Return the current angle of the named joint in radians."""
        return float(self.simulator._mj_data.qpos[self._joint_qpos_addrs[joint_name]])

    def apply_velocities(self, target: ArmTarget) -> None:
        """Set ctrl entries to drive joints toward their targets.

        Each joint in :attr:`ArmTarget.joint_targets` gets ±:attr:`ArmTarget.velocity`
        depending on the sign of the remaining error. All other left-arm joints
        have their ctrl set to zero.
        """
        ctrl = self.simulator._mj_data.ctrl
        for actuator_name, joint_name in zip(_LEFT_ARM_ACTUATOR_NAMES, _LEFT_ARM_JOINT_NAMES):
            index = self._actuator_indices[actuator_name]
            if joint_name in target.joint_targets:
                error = target.joint_targets[joint_name] - self.joint_angle(joint_name)
                ctrl[index] = float(numpy.sign(error)) * target.velocity
            else:
                ctrl[index] = 0.0

    def target_reached(self, target: ArmTarget) -> bool:
        """Return ``True`` when all joints in *target* are within tolerance of their goals."""
        return all(
            abs(target.joint_targets[joint] - self.joint_angle(joint)) <= target.position_tolerance
            for joint in target.joint_targets
        )

    def hold(self) -> None:
        """Zero the ctrl entries for all left arm actuators."""
        ctrl = self.simulator._mj_data.ctrl
        for actuator_name in _LEFT_ARM_ACTUATOR_NAMES:
            ctrl[self._actuator_indices[actuator_name]] = 0.0


def _step(simulator: MujocoSimulator, step_delay: float) -> None:
    with simulator.renderer.lock():
        simulator.step()
    simulator.renderer.sync()
    time.sleep(step_delay)


if __name__ == "__main__":
    _SETTLE_STEPS = 200
    _HOLD_STEPS = 300
    _MAX_MOTION_STEPS = 3000
    _STEP_DELAY = 0.004

    _simulator = MujocoSimulator(_headless=False, file_path=SCENE_FILE)
    _simulator.start(simulate_in_thread=False, render_in_thread=False)
    _demo = TiagoArmDemo(simulator=_simulator)
    _demo.initialise_home_pose()

    _reach_target = ArmTarget(
        joint_targets={
            "arm_left_2_joint": 0.5,
            "arm_left_4_joint": 0.9,
        },
        velocity=0.3,
    )

    _home_target = ArmTarget(
        joint_targets={
            "arm_left_2_joint": -1.07,
            "arm_left_4_joint":  1.96,
        },
        velocity=0.3,
    )

    # Phase 1 – settle at home position.
    _demo.hold()
    for _ in range(_SETTLE_STEPS):
        _step(_simulator, _STEP_DELAY)

    # Phase 2 – reach: raise shoulder and extend elbow.
    for _ in range(_MAX_MOTION_STEPS):
        with _simulator.renderer.lock():
            _demo.apply_velocities(_reach_target)
            _simulator.step()
        _simulator.renderer.sync()
        time.sleep(_STEP_DELAY)
        if _demo.target_reached(_reach_target):
            break

    # Phase 3 – hold at reach position.
    _demo.hold()
    for _ in range(_HOLD_STEPS):
        _step(_simulator, _STEP_DELAY)

    # Phase 4 – return to home.
    for _ in range(_MAX_MOTION_STEPS):
        with _simulator.renderer.lock():
            _demo.apply_velocities(_home_target)
            _simulator.step()
        _simulator.renderer.sync()
        time.sleep(_STEP_DELAY)
        if _demo.target_reached(_home_target):
            break

    # Phase 5 – hold at home until viewer is closed.
    _demo.hold()
    while _simulator.renderer.is_running():
        _step(_simulator, _STEP_DELAY)

    _simulator.stop()
