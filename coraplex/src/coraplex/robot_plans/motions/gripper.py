from dataclasses import dataclass
from typing import Optional, List

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from coraplex.robot_plans.motions.base import BaseMotion
from coraplex.datastructures.enums import (
    Arms,
    MovementType,
    WaypointsMovementType,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.view_manager import ViewManager
from coraplex.utils import translate_pose_along_local_axis


TOOL_ORIENTATION_THRESHOLD = 0.02
"""
Orientation tolerance in rad (~1.1 degrees) for tool-center-point poses.

The position ``threshold`` of 0.005 doubles as a rotation tolerance in
:class:`CartesianPose` unless overridden -- 0.005 rad (0.29 degrees) is
unreachable for a physically simulated arm, whose PD-tracked joints settle
with a residual orientation error of ~0.008 rad, so such a task never
registers as done and stalls the rest of the plan behind it.
"""


@dataclass
class ReachMotion(BaseMotion):
    """ """

    object_designator: Body
    """
    Object designator_description describing the object that should be picked up
    """
    arm: Arms
    """
    The arm that should be used for pick up
    """
    grasp_description: GraspDescription
    """
    The grasp description that should be used for picking up the object
    """
    movement_type: MovementType = MovementType.CARTESIAN
    """
    The type of movement that should be performed.
    """
    reverse_pose_sequence: bool = False
    """
    Reverses the sequence of poses, i.e., moves away from the object instead of towards it. Used for placing objects.
    """

    def _calculate_pose_sequence(self) -> List[Pose]:
        end_effector = ViewManager.get_end_effector_view(self.arm, self.robot_view)

        target_pose = GraspDescription.get_grasp_pose(
            self.grasp_description, end_effector, self.object_designator
        )
        target_pose.rotate_by_quaternion(
            GraspDescription.calculate_grasp_orientation(
                self.grasp_description,
                end_effector.front_facing_orientation.to_np(),
            )
        )
        target_pre_pose = translate_pose_along_local_axis(
            target_pose,
            end_effector.front_facing_axis.to_np()[:3],
            -0.05,  # TODO: Maybe put these values in the semantic annotates
        )

        pose = self.world.transform(target_pre_pose, self.world.root)

        sequence = [target_pre_pose, pose]
        return sequence.reverse() if self.reverse_pose_sequence else sequence

    def perform(self):
        pass

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot_view).tool_frame
        nodes = [
            CartesianPose(
                root_link=self.robot_view.root,
                tip_link=tip,
                goal_pose=pose,
                threshold=0.005,
                orientation_threshold=TOOL_ORIENTATION_THRESHOLD,
                name="Reach",
            )
            for pose in self._calculate_pose_sequence()
        ]
        return Sequence(nodes=nodes)


@dataclass
class MoveGripperMotion(BaseMotion):
    """
    Opens or closes the gripper
    """

    motion: GripperState
    """
    Motion that should be performed, either 'open' or 'close'
    """
    gripper: Arms
    """
    Name of the gripper that should be moved
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper is allowed to collide with something
    """
    target_opening: Optional[float] = None
    """
    Explicit finger opening (in meters) to command instead of the position
    the GripperState would resolve to.

    Targets the same finger connections, so the sim synchronizer still
    translates it into the actuator ctrl setpoint the same way -- a smaller
    opening simply commands a tighter squeeze. Used to grasp with a specific
    force without changing the robot's shared OPEN/CLOSE states.
    """

    finger_velocity: Optional[float] = None
    """
    Explicit finger joint velocity (in m/s) to command instead of
    :attr:`JointPositionList`'s default. ``None`` keeps the default velocity.
    """

    def perform(self):
        return

    def _goal_state(self, end_effector: EndEffector) -> JointState:
        """
        The finger joint state this motion commands: the GripperState's own
        state, or -- when ``target_opening`` is set -- the same finger
        connections remapped to that explicit opening.
        """
        state = end_effector.get_joint_state_by_type(self.motion)
        if self.target_opening is None:
            return state
        return JointState.from_mapping(
            mapping={
                connection: self.target_opening for connection in state.connections
            },
        )

    @property
    def _motion_chart(self):
        arm = ViewManager().get_end_effector_view(self.gripper, self.robot)

        kwargs = {}
        if self.finger_velocity is not None:
            kwargs["max_velocity"] = self.finger_velocity

        return JointPositionList(
            goal_state=self._goal_state(arm),
            name=(
                "OpenGripper" if self.motion == GripperState.OPEN else "CloseGripper"
            ),
            # Closing must not fail just because a grasped object physically
            # stops the fingers before their nominal fully-closed target;
            # opening stalling early is a real problem worth surfacing.
            tolerate_stall=self.motion == GripperState.CLOSE,
            **kwargs,
        )


@dataclass
class MoveToolCenterPointMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    target: Pose
    """
    Target pose to which the TCP should be moved
    """
    arm: Arms
    """
    Arm with the TCP that should be moved to the target
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something
    """
    movement_type: Optional[MovementType] = MovementType.CARTESIAN
    """
    The type of movement that should be performed.
    """

    reference_linear_velocity: Optional[float] = None
    """
    Explicit linear reference velocity (in m/s) to command instead of the
    task's default. ``None`` keeps the default velocity.
    """

    reference_angular_velocity: Optional[float] = None
    """
    Explicit angular reference velocity (in rad/s) to command instead of the
    task's default. Only used for :class:`CartesianPose` (the orientation
    part). ``None`` keeps the default velocity.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        root = (
            self.world.root
            if isinstance(self.robot, HasMobileBase)
            and self.robot.mobile_base.full_body_controlled
            else self.robot.root
        )
        task = None
        if self.movement_type == MovementType.TRANSLATION:
            kwargs = {}
            if self.reference_linear_velocity is not None:
                kwargs["reference_velocity"] = self.reference_linear_velocity
            task = CartesianPosition(
                root_link=root,
                tip_link=tip,
                goal_point=self.target.to_position(),
                name="MoveTCP",
                weight=DefaultWeights.WEIGHT_BELOW_CA,
                threshold=0.005,
                **kwargs,
            )
        else:
            kwargs = {}
            if self.reference_linear_velocity is not None:
                kwargs["reference_linear_velocity"] = self.reference_linear_velocity
            if self.reference_angular_velocity is not None:
                kwargs["reference_angular_velocity"] = self.reference_angular_velocity
            task = CartesianPose(
                root_link=root,
                tip_link=tip,
                goal_pose=self.target,
                name="MoveTCP",
                weight=DefaultWeights.WEIGHT_BELOW_CA,
                threshold=0.005,
                orientation_threshold=TOOL_ORIENTATION_THRESHOLD,
                **kwargs,
            )
        return task


@dataclass
class MoveTCPWaypointsMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    waypoints: List[Pose]
    """
    Waypoints the TCP should move along 
    """
    arm: Arms
    """
    Arm with the TCP that should be moved to the target
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something
    """
    movement_type: WaypointsMovementType = (
        WaypointsMovementType.ENFORCE_ORIENTATION_FINAL_POINT
    )
    """
    The type of movement that should be performed.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        root = (
            self.world.root
            if isinstance(self.robot, HasMobileBase)
            and self.robot.mobile_base.full_body_controlled
            else self.robot.root
        )
        nodes = [
            CartesianPose(
                root_link=root,
                tip_link=tip,
                goal_pose=pose,
                # threshold=0.005,
            )
            for pose in self.waypoints
        ]
        return Sequence(nodes=nodes)


@dataclass
class MoveManipulatorMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    target: Pose
    """
    Target pose to which the TCP should be moved
    """

    end_effector: EndEffector
    """
    The end effector to move to the target pose
    """

    allow_gripper_collision: bool = False
    """
    If the gripper can collide with something
    """

    @property
    def _motion_chart(self):
        robot = self.robot
        full_body_controlled = (
            robot.mobile_base.full_body_controlled
            if isinstance(robot, HasMobileBase)
            else False
        )
        root = self.world.root if full_body_controlled else robot.root
        task = CartesianPose(
            root_link=root,
            tip_link=self.end_effector.tool_frame,
            goal_pose=self.target,
            threshold=0.005,
            orientation_threshold=TOOL_ORIENTATION_THRESHOLD,
            binding_policy=GoalBindingPolicy.Bind_on_start,
            name=self.__class__.__name__,
        )
        return task
