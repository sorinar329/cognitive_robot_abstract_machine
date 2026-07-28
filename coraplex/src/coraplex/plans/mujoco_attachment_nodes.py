from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.plans.attachment_nodes import AttachNode, DetachNode
from coraplex.plans.mujoco_executables import (
    MujocoAttachExecutable,
    MujocoDetachExecutable,
    MujocoGraspMode,
)


@dataclass
class MujocoAttachNode(AttachNode):
    """
    Attaches a grasped body to the gripper in the world model and, depending on the
    grasp mode, rigidly binds it in the MuJoCo world.
    """

    grasp_mode: MujocoGraspMode = field(kw_only=True)
    """
    The grasp mode deciding whether the MuJoCo world gets a rigid attachment.
    """

    def parse(self) -> MujocoAttachExecutable:
        return MujocoAttachExecutable(
            context=self.plan.context,
            body=self.body,
            new_parent=self.new_parent,
            grasp_mode=self.grasp_mode,
        )


@dataclass
class MujocoDetachNode(DetachNode):
    """
    Detaches a body from its gripper in the world model and, depending on the grasp
    mode, releases it in the MuJoCo world.
    """

    grasp_mode: MujocoGraspMode = field(kw_only=True)
    """
    The grasp mode deciding whether the MuJoCo world attachment is released.
    """

    def parse(self) -> MujocoDetachExecutable:
        return MujocoDetachExecutable(
            context=self.plan.context,
            body=self.body,
            new_parent=self.new_parent,
            grasp_mode=self.grasp_mode,
        )
