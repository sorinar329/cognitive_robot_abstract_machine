import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Self

from importlib.resources import files
from pathlib import Path

from semantic_digital_twin.robots.abstract_robot import (
    AbstractRobot,
    Arm,
    Finger,
    ParallelGripper,
    FieldOfView,
    Base,
)
from semantic_digital_twin.robots.robot_mixins import HasArms
from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    ActiveConnection,
)


@dataclass(eq=False)
class OpenArmBimanual(AbstractRobot, HasArms):
    """
    Class that describes the OpenArm bimanual robot — a fixed-base torso with
    two 7-DOF arms (left and right), each equipped with a parallel two-finger gripper.

    URDF: openarm_bimanual.urdf
    Root link: openarm_body_link0 (fixed to world)
    Left arm:  openarm_left_link0  → … → openarm_left_link7  → openarm_left_hand  → openarm_left_hand_tcp
    Right arm: openarm_right_link0 → … → openarm_right_link7 → openarm_right_hand → openarm_right_hand_tcp
    """

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName("openarm_bimanual", prefix=world.name),
            root=world.get_body_by_name("openarm_body_link0"),
            _world=world,
        )

    @property
    def left_arm(self) -> Arm:
        """Convenience accessor for the left arm (arms[0])."""
        return self.arms[0]

    @property
    def right_arm(self) -> Arm:
        """Convenience accessor for the right arm (arms[1])."""
        return self.arms[1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_gripper(self, side: str) -> ParallelGripper:
        """
        Build a ParallelGripper for the given side ('left' or 'right').

        Finger layout in the URDF:
          - openarm_{side}_left_finger  (prismatic along +Y, joint2, mimic of joint1)
          - openarm_{side}_right_finger (prismatic along -Y, joint1, the driven joint)
        We treat right_finger as the 'thumb' (driven) and left_finger as 'finger' (mimic).
        """
        p = f"openarm_{side}"

        right_finger = Finger(
            name=PrefixedName(f"{side}_right_finger", prefix=self.name.name),
            root=self._world.get_body_by_name(f"{p}_right_finger"),
            tip=self._world.get_body_by_name(f"{p}_right_finger"),
            _world=self._world,
        )

        left_finger = Finger(
            name=PrefixedName(f"{side}_left_finger", prefix=self.name.name),
            root=self._world.get_body_by_name(f"{p}_left_finger"),
            tip=self._world.get_body_by_name(f"{p}_left_finger"),
            _world=self._world,
        )

        return ParallelGripper(
            name=PrefixedName(f"{side}_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name(f"{p}_hand"),
            tool_frame=self._world.get_body_by_name(f"{p}_hand_tcp"),
            thumb=right_finger,
            finger=left_finger,
            # The hand palm faces along +Z in the hand frame; the TCP is offset in +Z.
            front_facing_axis=Vector3(0, 0, 1),
            front_facing_orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            _world=self._world,
        )

    def _make_arm(self, side: str) -> Arm:
        """Build an Arm for the given side ('left' or 'right')."""
        p = f"openarm_{side}"
        gripper = self._make_gripper(side)

        return Arm(
            name=PrefixedName(f"{side}_arm", prefix=self.name.name),
            root=self._world.get_body_by_name(f"{p}_link0"),
            tip=self._world.get_body_by_name(f"{p}_hand"),
            manipulator=gripper,
            sensors=[],
            _world=self._world,
        )

    # ------------------------------------------------------------------
    # AbstractRobot interface
    # ------------------------------------------------------------------

    def _setup_semantic_annotations(self):
        # Left arm
        left_arm = self._make_arm("left")
        self.add_arm(left_arm)

        # Right arm
        right_arm = self._make_arm("right")
        self.add_arm(right_arm)

        self.full_body_controlled = True

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "openarm_bimanual.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )

        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05, violated_distance=0.0, robot=self
            )
        )

        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.1,
                violated_distance=0.03,
                robot=self,
                body_subset={self._world.get_body_by_name("openarm_body_link0")},
            )
        )

        self._world.collision_manager.add_default_rule(
            AvoidSelfCollisions(
                buffer_zone_distance=0.03,
                violated_distance=0.0,
                robot=self,
            )
        )

        self._world.collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(
                2,
                bodies={self._world.get_body_by_name("openarm_body_link0")},
            )
        )

        # Wrist links for both arms — allow more simultaneous avoidances near the EE
        for side in ("left", "right"):
            self._world.collision_manager.max_avoided_bodies_rules.append(
                MaxAvoidedCollisionsOverride(
                    4,
                    bodies=set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_by_name(f"openarm_{side}_link7")
                        )
                    ),
                )
            )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_hardware_interfaces(self):
        controlled_joints = [
            # Left arm
            "openarm_left_joint1",
            "openarm_left_joint2",
            "openarm_left_joint3",
            "openarm_left_joint4",
            "openarm_left_joint5",
            "openarm_left_joint6",
            "openarm_left_joint7",
            # Right arm
            "openarm_right_joint1",
            "openarm_right_joint2",
            "openarm_right_joint3",
            "openarm_right_joint4",
            "openarm_right_joint5",
            "openarm_right_joint6",
            "openarm_right_joint7",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def _setup_joint_states(self):
        # ----------------------------------------------------------------
        # Arm park states
        # ----------------------------------------------------------------
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            arm_park = JointState.from_mapping(
                name=PrefixedName(f"{side}_arm_park", prefix=self.name.name),
                mapping=dict(
                    zip(
                        [c for c in arm.connections if type(c) != FixedConnection],
                        # Neutral upright pose: all joints at zero except joint4
                        # (elbow, 0 → 2.44 range) set to a comfortable mid-value.
                        [0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0],
                    )
                ),
                state_type=StaticJointState.PARK,
            )
            arm.add_joint_state(arm_park)

        # ----------------------------------------------------------------
        # Gripper states (open / close) — both arms share the same pattern
        # ----------------------------------------------------------------
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            p = f"openarm_{side}"
            # Only the driven joint (finger_joint1) needs to be commanded;
            # finger_joint2 is a mimic joint.
            gripper_joints = [
                self._world.get_connection_by_name(f"{p}_finger_joint1"),
            ]

            gripper_open = JointState.from_mapping(
                name=PrefixedName(f"{side}_gripper_open", prefix=self.name.name),
                mapping=dict(zip(gripper_joints, [0.044])),  # upper limit = fully open
                state_type=GripperState.OPEN,
            )

            gripper_close = JointState.from_mapping(
                name=PrefixedName(f"{side}_gripper_close", prefix=self.name.name),
                mapping=dict(zip(gripper_joints, [0.0])),  # lower limit = fully closed
                state_type=GripperState.CLOSE,
            )

            arm.manipulator.add_joint_state(gripper_open)
            arm.manipulator.add_joint_state(gripper_close)