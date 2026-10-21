from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import List, Self, Dict, Dict

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.onrobot_2fg7 import OnRobot2FG7Gripper
from semantic_digital_twin.robots.onrobot_3fg15 import OnRobot3FG15Gripper
from semantic_digital_twin.robots.robot_part_mixins import HasLeftRightArm, HasOneArm
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    EndEffector,
)
from semantic_digital_twin.robots.ur10e_arm import UR10eArm
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% constants

PARK_JOINT_POSITIONS = (0.0, -1.2, 1.5, -1.0, -1.5708, 0.0)
"""
The park pose of the arm's six joints, ordered from shoulder to wrist.
"""

MAXIMUM_JOINT_VELOCITY = 0.2
"""
The fastest any joint of these arms may move, in radians per second, so demo motions
stay controllable.
"""

# %% shared helper


def _extend_default_collision_rules_for_ur5e(world, robot) -> None:
    """
    Add the default UR5e collision rules (external + self).
    """
    world.collision_manager.extend_default_rules(
        [
            AvoidExternalCollisions(
                buffer_zone_distance=0.05, violated_distance=0.0, robot=robot
            ),
            AvoidSelfCollisions(
                buffer_zone_distance=0.03, violated_distance=0.0, robot=robot
            ),
        ]
    )


# %% tool


@dataclass(eq=False)
class UR5eTool(EndEffector):
    """
    The bare flange of a UR5e, without a gripper.

    The tool frame is the ``tool0`` body the description carries at the flange.
    """

    def setup_hardware_interfaces(self):
        """
        Mark the tool's driven connections, of which a bare flange has none.
        """
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        """
        Named states of the tool; a bare flange holds no joints and declares none.
        """
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Build the tool from the flange and ``tool0`` bodies below the robot root.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "flange"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "tool0"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


# %% arm


@dataclass(eq=False)
class UR5eArm(UR10eArm[UR5eTool]):
    """
    A Universal Robots UR5e arm.

    The URDF joint names are identical across the UR family, so the drive tables of
    :class:`~semantic_digital_twin.robots.ur10e_arm.UR10eArm` apply unchanged; the
    torque limits tuned for the larger UR10e are kept as-is pending UR5e-specific
    tuning.
    """

    def setup_hardware_interfaces(self):
        """
        Mark every active connection of the arm as hardware-driven.
        """
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        """
        Named states of the arm.
        """
        connections = self.active_connections
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(zip(connections, PARK_JOINT_POSITIONS, strict=True)),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Build the arm from the base link to the third wrist link below the robot root.

        The park pose tucks the arm with the elbow bent and the wrist pointing down, a
        starting configuration for tabletop demos rather than a manufacturer value.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "wrist_3_link"
            ),
        )


# %% robot


@dataclass(eq=False)
class UR5e(AbstractRobot, HasOneArm[UR5eArm]):
    """
    A fixed-mount Universal Robots UR5e arm.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        """
        The xacro description of the arm.

        ..note:: The xacro needs its ``name`` and ``ur_type`` arguments set to the UR5e,
            so parse it with :meth:`URDFParser.from_xacro` and explicit mappings rather
            than the mapping-free :meth:`RobotSpecification.spawn`.
        """
        return "package://ur_description/urdf/ur.urdf.xacro"

    @classmethod
    def _get_root_body_name(cls) -> str:
        """
        The body the arm is bolted down with.
        """
        return "base_link"

    def _setup_collision_rules(self):
        """
        Keep the arm clear of the world and of itself.
        """
        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.03, violated_distance=0.0, robot=self
                ),
            ]
        )

    def _setup_velocity_limits(self):
        """
        Slow the arm down to ``MAXIMUM_JOINT_VELOCITY`` at its fastest joint, keeping
        the joints' proportions, so demo motions stay controllable.
        """
        self.tighten_dof_velocity_limits_of_1dof_connections(
            {
                connection: MAXIMUM_JOINT_VELOCITY
                for connection in self._one_dof_connections
            }
        )


# %% mujoco-dialect joints


class UR5eMujocoJoint(StrEnum):
    """
    The joints of the UR5e as the MuJoCo scene names them: the URDF names without their
    ``_joint`` suffix.
    """

    SHOULDER_PAN = "shoulder_pan"
    SHOULDER_LIFT = "shoulder_lift"
    ELBOW = "elbow"
    WRIST_1 = "wrist_1"
    WRIST_2 = "wrist_2"
    WRIST_3 = "wrist_3"


# %% mujoco-dialect tool


@dataclass(eq=False)
class UR5eMujocoTool(UR5eTool):
    """
    The UR5e's flange carrying the static dual gripper; no ``tool0`` body, so the tool
    frame is the flange itself.
    """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "flange"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "flange"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


# %% mujoco-dialect arm


@dataclass(eq=False)
class UR5eMujocoArm(UR10eArm[UR5eMujocoTool]):
    """
    The UR5e arm as the MuJoCo scene names it: rooted at ``ur5e_base``, joints keyed by
    :class:`UR5eMujocoJoint`, park pose same as :class:`UR5eArm`, tool frame on the
    flange because the scene carries no ``tool0`` body.
    """

    @property
    def drives_by_joint(self) -> Dict[str, UR10eJointDrive]:
        urdf_drives = super().drives_by_joint
        return {joint: urdf_drives[f"{joint.value}_joint"] for joint in UR5eMujocoJoint}

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, PARK_JOINT_POSITIONS, strict=True),
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "ur5e_base"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "wrist_3_link"
            ),
        )


# %% mujoco-dialect robot


@dataclass(eq=False)
class UR5eMujoco(AbstractRobot, HasOneArm[UR5eMujocoArm]):
    """A UR5e annotated in a MuJoCo scene that names it differently from the URDF:
    root body is ``ur5e_base``, joint names drop the ``_joint`` suffix, and the tool
    frame sits on the flange because the scene's flange carries no ``tool0`` body."""

    @classmethod
    def get_ros_file_path(cls) -> str:
        raise NotImplementedError(
            "The description for this variant lives in the demo's MJCF scene; "
            "there is no URDF file of its own."
        )

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "ur5e_base"

    def _setup_collision_rules(self):
        _extend_default_collision_rules_for_ur5e(self._world, self)

    def _setup_velocity_limits(self):
        self.tighten_dof_velocity_limits_of_1dof_connections(
            {
                connection: MAXIMUM_JOINT_VELOCITY
                for connection in self._one_dof_connections
            }
        )


# %% mujoco-dialect dual-gripper arms


@dataclass(eq=False)
class UR5eMujoco2FG7Arm(UR10eArm[OnRobot2FG7Gripper]):
    """
    The UR5e arm as the MuJoCo scene names it, carrying the 2FG7: same chain, park pose
    and drives as :class:`UR5eMujocoArm`, with the 2FG7 as its end effector.
    """

    @property
    def drives_by_joint(self) -> Dict[str, UR10eJointDrive]:
        urdf_drives = super().drives_by_joint
        return {joint: urdf_drives[f"{joint.value}_joint"] for joint in UR5eMujocoJoint}

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, PARK_JOINT_POSITIONS, strict=True),
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "ur5e_base"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "wrist_3_link"
            ),
        )


@dataclass(eq=False)
class UR5eMujoco3FG15Arm(UR10eArm[OnRobot3FG15Gripper]):
    """
    The UR5e arm as the MuJoCo scene names it, carrying the 3FG15: same chain, park pose
    and drives as :class:`UR5eMujocoArm`, with the 3FG15 as its end effector.
    """

    @property
    def drives_by_joint(self) -> Dict[str, UR10eJointDrive]:
        urdf_drives = super().drives_by_joint
        return {joint: urdf_drives[f"{joint.value}_joint"] for joint in UR5eMujocoJoint}

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, PARK_JOINT_POSITIONS, strict=True),
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "ur5e_base"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "wrist_3_link"
            ),
        )


# %% mujoco-dialect dual-gripper robot


@dataclass(eq=False)
class UR5eMujocoDualGripper(
    AbstractRobot, HasLeftRightArm[UR5eMujoco2FG7Arm, UR5eMujoco3FG15Arm]
):
    """
    A UR5e in a MuJoCo scene, carrying a 2FG7 and a 3FG15 on its single physical arm.

    Both arm annotations span the same six joints; they differ only in their end
    effector. ``LEFT`` addresses the 2FG7 view, ``RIGHT`` the 3FG15 view, so each
    gripper moves independently through :class:`MoveGripperMotion <coraplex.robot_plans.motions.gripper.MoveGripperMotion>`.
    """

    @cached_property
    def left_arm(self) -> UR5eMujoco2FG7Arm:
        """
        The arm view carrying the 2FG7.
        """
        return self._arm_carrying(OnRobot2FG7Gripper)

    @cached_property
    def right_arm(self) -> UR5eMujoco3FG15Arm:
        """
        The arm view carrying the 3FG15.
        """
        return self._arm_carrying(OnRobot3FG15Gripper)

    def _arm_carrying(self, tool: type[EndEffector]) -> Arm:
        """
        Select the arm view by its end effector type.

        The geometric assignment of the mixin cannot tell the two views apart: both
        span the same joints, so their positions coincide and its strict relations
        hold for neither.

        :param tool: The end effector type the wanted arm view carries.
        :return: The arm view carrying that end effector type.
        """
        [arm] = [arm for arm in self.get_arms() if isinstance(arm.end_effector, tool)]
        return arm

    @classmethod
    def get_ros_file_path(cls) -> str:
        raise NotImplementedError(
            "The description for this variant lives in the demo's MJCF scene; "
            "there is no URDF file of its own."
        )

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "ur5e_base"

    def _setup_collision_rules(self):
        _extend_default_collision_rules_for_ur5e(self._world, self)

    def _setup_velocity_limits(self):
        """
        Slow every joint down to ``MAXIMUM_JOINT_VELOCITY``, so demo motions stay
        controllable.
        """
        self.tighten_dof_velocity_limits_of_1dof_connections(
            {
                connection: MAXIMUM_JOINT_VELOCITY
                for connection in self._one_dof_connections
            }
        )
