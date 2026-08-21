from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Dict, List, Optional, Self

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from coraplex.exceptions import NoSteppingSimulatorForServoTargets
from coraplex.robot_plans.motions.base import BaseMotion
from semantic_digital_twin.adapters.multi_sim import (
    MujocoSynchronizer,
    _MultiSimStateCallback,
)
from semantic_digital_twin.robots.unitree_go2 import (
    STANDING_CONFIGURATION,
    UnitreeGo2Joint,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose, Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% commanding position servos


@dataclass
class JointServoTargets:
    """
    Setpoints of a running simulation's position-servo actuators, addressed by the
    joint each one drives.

    Commanding a setpoint leaves the joint itself to the physics engine, so the servo
    has to drive the joint there against gravity and ground contact. Writing a joint's
    world state instead places the joint directly (the world -> simulator sync writes
    ``qpos``), which produces none of the contact forces a robot walks on.
    """

    control_array: np.ndarray
    """The simulation's actuator setpoints, written in place."""

    control_index_by_joint_name: Dict[str, int]
    """Which entry of :attr:`control_array` drives which joint."""

    @classmethod
    def of_stepping_simulation(cls, world: World) -> Self:
        """
        Address the actuators of the simulation that is stepping ``world``'s physics.

        A world can be mirrored into more than one simulation at a time - recording a
        video adds a second, unstepped mirror - so the one whose physics actually runs
        is selected by having a simulation thread.

        :raises NoSteppingSimulatorForServoTargets: If no such simulation is attached.
        """
        simulators = [
            callback.synchronizer.simulator
            for callback in world.state.state_change_callbacks
            if isinstance(callback, _MultiSimStateCallback)
            and isinstance(callback.synchronizer, MujocoSynchronizer)
            and callback.synchronizer.simulator.simulation_thread is not None
        ]
        if not simulators:
            raise NoSteppingSimulatorForServoTargets(world=world)
        simulator = simulators[0]

        model = simulator._mj_model
        # An actuator carries the index of the joint it drives, so the mapping needs no
        # assumption about how the two are named relative to each other.
        control_index_by_joint_name = {
            model.joint(model.actuator(control_index).trnid[0]).name: control_index
            for control_index in range(model.nu)
        }
        return cls(
            control_array=simulator._mj_data.ctrl,
            control_index_by_joint_name=control_index_by_joint_name,
        )

    def command(self, joint_name: str, target: float) -> None:
        """
        Set the setpoint the servo driving ``joint_name`` pulls towards.
        """
        self.control_array[self.control_index_by_joint_name[joint_name]] = target


# %% gait


@dataclass
class QuadrupedLeg:
    """
    One leg of a trotting quadruped: the joints it is driven through, when in the gait
    cycle it swings, and which side of the body it is on.
    """

    hip_joint_name: str
    """Abduction joint, held at its stance angle."""

    thigh_joint_name: str
    """Swings the foot forward and back."""

    calf_joint_name: str
    """Lowers the foot onto the ground and lifts it off again."""

    swing_phase_offset: float
    """0.0 or 0.5 gait cycles -- legs sharing a value form a trotting diagonal pair."""

    is_on_left: bool
    """Whether this leg is on the robot's left, which decides how steering scales it."""

    hip_stance: float
    thigh_stance: float
    calf_stance: float
    """
    The standing joint angles this leg's gait oscillates around.
    """


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(eq=False, repr=False)
class TrotGait(MotionStatechartNode):
    """
    An open-loop trot walking the robot towards a target position.

    Diagonal leg pairs alternate: while one pair sweeps its feet backwards along the
    ground, the other lifts and swings forwards. Feet push the floor, the floor pushes
    back, and the base travels - its motion is never commanded, only the twelve leg
    servos are. Steering lengthens the stride on the outside of the turn and shortens
    it on the inside, which is what turns a quadruped; the base's heading is not
    written either.

    ..note:: The thigh drives the foot backwards as its angle grows and the calf
        lowers it as its angle grows (both joints turn about the same axis, a quarter
        cycle apart), so the foot traces an ellipse: down and back through stance, up
        and forward through swing.
    """

    legs: List[QuadrupedLeg] = field(kw_only=True)
    """The four legs, in any order."""

    servo_targets: JointServoTargets = field(kw_only=True)
    """Where the gait's joint commands are written."""

    robot_root: KinematicStructureEntity = field(kw_only=True)
    """The robot's floating base, read every tick to track its progress."""

    target_x: float = field(kw_only=True)
    target_y: float = field(kw_only=True)
    """
    Where the base should end up, in the world frame. Only the position is aimed for:
    the gait steers towards it and stops there, whatever the base's final heading.
    """

    stride_frequency: float = field(kw_only=True, default=1.0)
    """
    Gait cycles per second. The servo setpoints are only refreshed once per control
    cycle, so a faster gait is described by coarser steps in the setpoint.
    """

    thigh_amplitude: float = field(kw_only=True, default=0.3)
    """Radians the thigh sweeps forward/back from its stance angle."""

    calf_lift_amplitude: float = field(kw_only=True, default=0.4)
    """Radians the calf bends at mid-swing, lifting that foot clear of the floor."""

    steering_gain: float = field(kw_only=True, default=0.6)
    """
    How strongly a heading error (radians) scales the stride difference between the
    left and right legs.
    """

    max_steering: float = field(kw_only=True, default=0.8)
    """
    Largest stride scaling the steering may apply, as a fraction of stride length.
    Kept below 1: striding one side backwards to spin on the spot twists the robot off
    its two supporting feet and topples it.
    """

    position_tolerance: float = field(kw_only=True, default=0.25)
    """Planar distance to the target at which the gait stops."""

    timeout_seconds: float = field(kw_only=True, default=45.0)
    """
    Safety cutoff so a gait that never converges doesn't hang the plan forever. Has to
    cover turning on the spot as well as walking, which a trot does slowly.
    """

    _elapsed_seconds: float = field(init=False, default=0.0)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        self._elapsed_seconds += context.qp_controller_config.control_dt

        base_pose = context.world.compute_forward_kinematics_np(
            context.world.root, self.robot_root
        )
        base_x, base_y = base_pose[0, 3], base_pose[1, 3]
        remaining_distance = float(
            np.hypot(self.target_x - base_x, self.target_y - base_y)
        )

        if (
            remaining_distance <= self.position_tolerance
            or self._elapsed_seconds >= self.timeout_seconds
        ):
            self._stand(context)
            return ObservationStateValues.TRUE

        heading_error = _wrap_to_pi(
            float(np.arctan2(self.target_y - base_y, self.target_x - base_x))
            - float(np.arctan2(base_pose[1, 0], base_pose[0, 0]))
        )
        steering = float(
            np.clip(
                self.steering_gain * heading_error,
                -self.max_steering,
                self.max_steering,
            )
        )

        # Turning shortens the stride on the inside of the turn and leaves the outside
        # at its nominal length, rather than trading length from one side to the other.
        # Lengthening the outside instead over-extends those legs, which drives their
        # feet into the floor hard enough to throw the simulation off.
        left_stride_scale = 1.0 - max(0.0, steering)
        right_stride_scale = 1.0 - max(0.0, -steering)

        phase = 2 * math.pi * self._elapsed_seconds * self.stride_frequency
        for leg in self.legs:
            stride_scale = left_stride_scale if leg.is_on_left else right_stride_scale
            angle = phase + 2 * math.pi * leg.swing_phase_offset
            # Sweeping the thigh alone moves the foot almost horizontally, so the calf
            # is only bent to shorten the leg over the half cycle it swings forwards,
            # and left at its stance length over the half it drives backwards. Pushing
            # off by extending it through stance instead shoves the body off the two
            # feet carrying it, and the robot tips onto its belly.
            lift = self.calf_lift_amplitude * max(0.0, -math.cos(angle))
            self.servo_targets.command(leg.hip_joint_name, leg.hip_stance)
            self.servo_targets.command(
                leg.thigh_joint_name,
                leg.thigh_stance
                + stride_scale * self.thigh_amplitude * math.sin(angle),
            )
            self.servo_targets.command(leg.calf_joint_name, leg.calf_stance - lift)
        return ObservationStateValues.FALSE

    def _stand(self, context: MotionStatechartContext) -> None:
        """
        Return the legs to the stance they started in, so the robot comes to rest on
        all four feet instead of holding whatever the gait cycle stopped on.
        """
        for leg in self.legs:
            self.servo_targets.command(leg.hip_joint_name, leg.hip_stance)
            self.servo_targets.command(leg.thigh_joint_name, leg.thigh_stance)
            self.servo_targets.command(leg.calf_joint_name, leg.calf_stance)


@dataclass
class WalkMotion(BaseMotion):
    """
    Walks the robot to a target location by gaiting its legs, rather than commanding
    its base pose directly.
    """

    target: Pose
    """Location to walk to. Only its position is aimed for, not its orientation."""

    def perform(self):
        return

    @property
    def _motion_chart(self) -> TrotGait:
        target = Pose2D.from_pose(self.world.transform(self.target, self.world.root))
        return TrotGait(
            legs=self._legs(),
            servo_targets=JointServoTargets.of_stepping_simulation(self.world),
            robot_root=self.robot.root,
            target_x=float(target.x),
            target_y=float(target.y),
        )

    def _legs(self) -> List[QuadrupedLeg]:
        """
        The Go2's four legs, paired into the two diagonals of a trot.
        """

        def leg(
            hip: UnitreeGo2Joint,
            thigh: UnitreeGo2Joint,
            calf: UnitreeGo2Joint,
            swing_phase_offset: float,
            is_on_left: bool,
        ) -> QuadrupedLeg:
            return QuadrupedLeg(
                hip_joint_name=hip,
                thigh_joint_name=thigh,
                calf_joint_name=calf,
                swing_phase_offset=swing_phase_offset,
                is_on_left=is_on_left,
                hip_stance=STANDING_CONFIGURATION[hip],
                thigh_stance=STANDING_CONFIGURATION[thigh],
                calf_stance=STANDING_CONFIGURATION[calf],
            )

        return [
            leg(
                UnitreeGo2Joint.FRONT_LEFT_HIP,
                UnitreeGo2Joint.FRONT_LEFT_THIGH,
                UnitreeGo2Joint.FRONT_LEFT_CALF,
                swing_phase_offset=0.0,
                is_on_left=True,
            ),
            leg(
                UnitreeGo2Joint.REAR_RIGHT_HIP,
                UnitreeGo2Joint.REAR_RIGHT_THIGH,
                UnitreeGo2Joint.REAR_RIGHT_CALF,
                swing_phase_offset=0.0,
                is_on_left=False,
            ),
            leg(
                UnitreeGo2Joint.FRONT_RIGHT_HIP,
                UnitreeGo2Joint.FRONT_RIGHT_THIGH,
                UnitreeGo2Joint.FRONT_RIGHT_CALF,
                swing_phase_offset=0.5,
                is_on_left=False,
            ),
            leg(
                UnitreeGo2Joint.REAR_LEFT_HIP,
                UnitreeGo2Joint.REAR_LEFT_THIGH,
                UnitreeGo2Joint.REAR_LEFT_CALF,
                swing_phase_offset=0.5,
                is_on_left=True,
            ),
        ]
