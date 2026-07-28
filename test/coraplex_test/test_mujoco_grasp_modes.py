from dataclasses import dataclass, field

import pytest
from typing_extensions import List, Tuple

from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.exceptions import NoGraspAttachmentBackend
from coraplex.plans.mujoco_attachment_nodes import MujocoAttachNode, MujocoDetachNode
from coraplex.plans.mujoco_executables import (
    MujocoAttachExecutable,
    MujocoDetachExecutable,
    MujocoGraspMode,
)
from coraplex.robot_plans.actions.mujoco_manipulation import (
    MujocoPickUpAction,
    MujocoPlaceAction,
)
from semantic_digital_twin.adapters.grasp_attachment import GraspAttachmentBackend
from semantic_digital_twin.callbacks.callback import ModelChangeCallback
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% recording backend mimic


@dataclass(eq=False)
class RecordingGraspAttachmentBackend(ModelChangeCallback, GraspAttachmentBackend):
    """
    Grasp attachment backend that records the attach and release calls it receives,
    standing in for a real simulator synchronizer.
    """

    attached: List[Tuple[Body, Body]] = field(default_factory=list)
    """
    The ``(grasped_body, gripper_body)`` pairs passed to :meth:`attach_grasped_body`.
    """

    released: List[Body] = field(default_factory=list)
    """
    The bodies passed to :meth:`release_grasped_body`.
    """

    def on_model_change(self, **kwargs) -> None:
        pass

    def attach_grasped_body(self, grasped_body: Body, gripper_body: Body) -> None:
        self.attached.append((grasped_body, gripper_body))

    def release_grasped_body(self, grasped_body: Body) -> None:
        self.released.append(grasped_body)


# %% executable dispatch


def test_friction_attach_does_not_touch_backend(mutable_model_world):
    """
    In friction mode the attach only re-parents the body in the world model and never
    calls the attachment backend.
    """
    world, _, context = mutable_model_world
    milk = world.get_body_by_name("milk.stl")
    backend = RecordingGraspAttachmentBackend(_world=world)

    MujocoAttachExecutable(
        context=context,
        body=milk,
        new_parent=world.root,
        grasp_mode=MujocoGraspMode.FRICTION,
    ).execute()

    assert isinstance(milk.parent_connection, FixedConnection)
    assert backend.attached == []


def test_friction_attach_needs_no_backend(mutable_model_world):
    """
    Friction mode works even when no attachment backend is registered on the world.
    """
    world, _, context = mutable_model_world
    milk = world.get_body_by_name("milk.stl")

    MujocoAttachExecutable(
        context=context,
        body=milk,
        new_parent=world.root,
        grasp_mode=MujocoGraspMode.FRICTION,
    ).execute()

    assert isinstance(milk.parent_connection, FixedConnection)


def test_attachment_attach_binds_body_in_backend(mutable_model_world):
    """
    In attachment mode the attach re-parents the body in the world model and binds it to
    the gripper in the backend.
    """
    world, _, context = mutable_model_world
    milk = world.get_body_by_name("milk.stl")
    backend = RecordingGraspAttachmentBackend(_world=world)

    MujocoAttachExecutable(
        context=context,
        body=milk,
        new_parent=world.root,
        grasp_mode=MujocoGraspMode.ATTACHMENT,
    ).execute()

    assert isinstance(milk.parent_connection, FixedConnection)
    assert backend.attached == [(milk, world.root)]


def test_attachment_detach_releases_body_in_backend(mutable_model_world):
    """
    In attachment mode the detach releases the body in the backend and re-parents it in
    the world model.
    """
    world, _, context = mutable_model_world
    milk = world.get_body_by_name("milk.stl")
    backend = RecordingGraspAttachmentBackend(_world=world)

    MujocoDetachExecutable(
        context=context,
        body=milk,
        new_parent=world.root,
        grasp_mode=MujocoGraspMode.ATTACHMENT,
    ).execute()

    assert backend.released == [milk]


def test_friction_detach_does_not_touch_backend(mutable_model_world):
    """
    In friction mode the detach only re-parents the body and never releases anything in
    the backend.
    """
    world, _, context = mutable_model_world
    milk = world.get_body_by_name("milk.stl")
    backend = RecordingGraspAttachmentBackend(_world=world)

    MujocoDetachExecutable(
        context=context,
        body=milk,
        new_parent=world.root,
        grasp_mode=MujocoGraspMode.FRICTION,
    ).execute()

    assert backend.released == []


def test_attachment_without_backend_raises(mutable_model_world):
    """
    Attachment mode requires a backend; without one the attach raises.
    """
    world, _, context = mutable_model_world
    milk = world.get_body_by_name("milk.stl")

    with pytest.raises(NoGraspAttachmentBackend):
        MujocoAttachExecutable(
            context=context,
            body=milk,
            new_parent=world.root,
            grasp_mode=MujocoGraspMode.ATTACHMENT,
        ).execute()


# %% action and node wiring


def _grasp_description(robot) -> GraspDescription:
    return GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        robot.get_arms()[0].end_effector,
    )


def test_pick_up_action_defaults_to_attachment(mutable_model_world):
    """
    The MuJoCo pick-up action defaults to the attachment grasp mode.
    """
    world, robot, _ = mutable_model_world
    action = MujocoPickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        _grasp_description(robot),
    )

    assert action.grasp_mode is MujocoGraspMode.ATTACHMENT


def test_pick_up_action_builds_mujoco_attach_node(mutable_model_world):
    """
    The MuJoCo pick-up action builds a MuJoCo attach node carrying its grasp mode.
    """
    world, robot, _ = mutable_model_world
    milk = world.get_body_by_name("milk.stl")
    action = MujocoPickUpAction(
        milk,
        Arms.LEFT,
        _grasp_description(robot),
        grasp_mode=MujocoGraspMode.FRICTION,
    )

    node = action._make_attach_node(body=milk, new_parent=world.root)

    assert isinstance(node, MujocoAttachNode)
    assert node.grasp_mode is MujocoGraspMode.FRICTION


def test_place_action_builds_mujoco_detach_node(mutable_model_world):
    """
    The MuJoCo place action builds a MuJoCo detach node carrying its grasp mode.
    """
    world, _, _ = mutable_model_world
    milk = world.get_body_by_name("milk.stl")
    action = MujocoPlaceAction(
        milk,
        milk.global_pose,
        Arms.LEFT,
        grasp_mode=MujocoGraspMode.FRICTION,
    )

    node = action._make_detach_node(body=milk, new_parent=world.root)

    assert isinstance(node, MujocoDetachNode)
    assert node.grasp_mode is MujocoGraspMode.FRICTION
