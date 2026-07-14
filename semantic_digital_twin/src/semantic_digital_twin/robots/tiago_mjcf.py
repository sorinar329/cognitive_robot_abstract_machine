from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Self, List

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasLeftRightArm,
    HasMobileBase,
    HasNeck,
    HasTorso,
    HasTwoFingers,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    EndEffector,
    Finger,
    MobileBase,
    Neck,
    Torso,
)
from semantic_digital_twin.robots.tiago import TiagoCamera
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class TiagoMjcfLeftThumb(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_left_inner_finger"
            ),
        )


@dataclass(eq=False)
class TiagoMjcfLeftIndexFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_right_inner_finger"
            ),
        )


@dataclass(eq=False)
class TiagoMjcfRightThumb(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_left_inner_finger"
            ),
        )


@dataclass(eq=False)
class TiagoMjcfRightIndexFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_right_inner_finger"
            ),
        )


@dataclass(eq=False)
class TiagoMjcfLeftGripper(
    EndEffector, HasTwoFingers[TiagoMjcfLeftThumb, TiagoMjcfLeftIndexFinger]
):
    """
    Robotiq 2F-140 gripper mounted on the left arm.

    ..note:: The inner-finger and inner-knuckle joints of each side are only
        kept mechanically consistent with their driver joint by MuJoCo's
        ``equality`` constraints at simulation time; the digital twin's
        kinematic model does not encode that four-bar linkage. Only the two
        driver joints (mechanically coupled to each other via a shared
        degree of freedom) are exposed as hardware interfaces and controlled
        by :meth:`setup_joint_states`.
    """

    def setup_hardware_interfaces(self):
        controlled_joints = [
            "gripper_left_left_driver_joint",
            "gripper_left_right_driver_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        driver_connections = [
            connection
            for connection in self.active_connections
            if connection.name.name.endswith("_driver_joint")
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(driver_connections, [0.0] * len(driver_connections))),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(driver_connections, [0.7] * len(driver_connections))),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_tool_frame"
            ),
            front_facing_orientation=Quaternion(0, -0.70710678, 0, 0.70710678),
        )


@dataclass(eq=False)
class TiagoMjcfRightGripper(
    EndEffector, HasTwoFingers[TiagoMjcfRightThumb, TiagoMjcfRightIndexFinger]
):
    """
    Robotiq 2F-140 gripper mounted on the right arm.

    ..note:: See :class:`TiagoMjcfLeftGripper` for why only the driver
        joints are exposed as hardware interfaces.
    """

    def setup_hardware_interfaces(self):
        controlled_joints = [
            "gripper_right_left_driver_joint",
            "gripper_right_right_driver_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        driver_connections = [
            connection
            for connection in self.active_connections
            if connection.name.name.endswith("_driver_joint")
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(driver_connections, [0.0] * len(driver_connections))),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(driver_connections, [0.7] * len(driver_connections))),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_tool_frame"
            ),
            front_facing_orientation=Quaternion(0, -0.70710678, 0, 0.70710678),
        )


@dataclass(eq=False)
class TiagoMjcfLeftArm(Arm[TiagoMjcfLeftGripper]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            "arm_left_1_joint",
            "arm_left_2_joint",
            "arm_left_3_joint",
            "arm_left_4_joint",
            "arm_left_5_joint",
            "arm_left_6_joint",
            "arm_left_7_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_left_tool_link"
            ),
        )


@dataclass(eq=False)
class TiagoMjcfRightArm(Arm[TiagoMjcfRightGripper]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            "arm_right_1_joint",
            "arm_right_2_joint",
            "arm_right_3_joint",
            "arm_right_4_joint",
            "arm_right_5_joint",
            "arm_right_6_joint",
            "arm_right_7_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_right_tool_link"
            ),
        )


@dataclass(eq=False)
class TiagoMjcfNeck(Neck[TiagoCamera]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            "head_1_joint",
            "head_2_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "head_2_link"),
        )


@dataclass(eq=False)
class TiagoMjcfTorso(
    Torso, HasLeftRightArm[TiagoMjcfLeftArm, TiagoMjcfRightArm], HasNeck[TiagoMjcfNeck]
):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            "torso_lift_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        torso_joint = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.15])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.35])),
            state_type=TorsoState.HIGH,
        )

        return [torso_low, torso_mid, torso_high]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_fixed_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
        )


@dataclass(eq=False)
class TiagoMjcfMobileBase(MobileBase, HasTorso[TiagoMjcfTorso]):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
            forward_axis=Vector3.X(),
            full_body_controlled=True,
        )


@dataclass(eq=False)
class TiagoMjcf(AbstractRobot, HasMobileBase[TiagoMjcfMobileBase]):
    """
    The TIAGo++ robot as modeled by the ``iai_tiago.xml`` MuJoCo asset used for the
    segmind episodes, with Robotiq 2F-140 grippers instead of the PAL parallel
    grippers modeled by :class:`~semantic_digital_twin.robots.tiago.Tiago`.

    ..note:: ``base_footprint`` carries a raw 6-DoF MuJoCo ``<freejoint/>`` in the
        parsed asset, not a :class:`~semantic_digital_twin.world_description.connections.WheeledDrive`
        connection. :attr:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.drive`
        will be ``None``, and Coraplex's ``NavigateAction`` will not work, until the
        caller replaces that connection (for example with a
        :class:`~semantic_digital_twin.world_description.connections.DifferentialDrive`
        from the scene's ``odom`` body to ``base_footprint``) before calling
        :meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.from_world`.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        raise NotImplementedError(
            "TiagoMjcf has no URDF description; it is parsed directly from "
            "segmind's iai_tiago_velocity.xml MJCF asset via MJCFParser."
        )

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def _setup_collision_rules(self):
        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.03,
                    violated_distance=0.0,
                    robot=self,
                ),
            ]
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)
