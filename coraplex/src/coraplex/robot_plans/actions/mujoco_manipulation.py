from __future__ import annotations

from dataclasses import dataclass

from coraplex.plans.mujoco_attachment_nodes import (
    MujocoAttachNode,
    MujocoDetachNode,
)
from coraplex.plans.mujoco_executables import MujocoGraspMode
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class MujocoPickUpAction(PickUpAction):
    """
    Picks up an object like :class:`PickUpAction`, but also realizes the grasp in the
    MuJoCo world according to :attr:`grasp_mode`.
    """

    grasp_mode: MujocoGraspMode = MujocoGraspMode.ATTACHMENT
    """
    Whether the grasp is held by contact friction or by a rigid attachment in MuJoCo.
    """

    def _make_attach_node(self, body: Body, new_parent: Body) -> MujocoAttachNode:
        return MujocoAttachNode(
            body=body, new_parent=new_parent, grasp_mode=self.grasp_mode
        )


@dataclass
class MujocoPlaceAction(PlaceAction):
    """
    Places an object like :class:`PlaceAction`, but also releases the grasp in the
    MuJoCo world according to :attr:`grasp_mode`.
    """

    grasp_mode: MujocoGraspMode = MujocoGraspMode.ATTACHMENT
    """
    Whether the grasp was held by contact friction or by a rigid attachment in MuJoCo;
    must match the mode the object was picked up with.
    """

    def _make_detach_node(self, body: Body, new_parent: Body) -> MujocoDetachNode:
        return MujocoDetachNode(
            body=body, new_parent=new_parent, grasp_mode=self.grasp_mode
        )
