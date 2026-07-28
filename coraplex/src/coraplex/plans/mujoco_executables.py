from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from coraplex.exceptions import NoGraspAttachmentBackend
from coraplex.plans.executables import ModelChangeExecutable
from semantic_digital_twin.adapters.grasp_attachment import GraspAttachmentBackend

# %% grasp mode


class MujocoGraspMode(Enum):
    """
    How a grasp is physically realized in the MuJoCo world once the object is in the
    gripper.
    """

    FRICTION = auto()
    """
    The grasped object stays a free body and is held purely by the contact friction of
    the closed gripper. No rigid constraint is added, so the grip is only as strong as
    the physics allow.
    """

    ATTACHMENT = auto()
    """
    The grasped object is rigidly bound to the gripper in MuJoCo (re-parented under the
    gripper body) for the duration of the grasp, so it follows the gripper regardless of
    contact forces.
    """


# %% mujoco-backed model changes


@dataclass
class MujocoGraspExecutable(ModelChangeExecutable):
    """
    Base for model changes that also drive the physical grasp in the MuJoCo world,
    according to a :class:`MujocoGraspMode`.
    """

    grasp_mode: MujocoGraspMode = field(kw_only=True)
    """
    The grasp mode deciding whether the MuJoCo world gets a rigid attachment.
    """

    def _grasp_attachment_backend(self) -> GraspAttachmentBackend:
        """
        Return the grasp attachment backend registered on the world.

        :raises NoGraspAttachmentBackend: If no backend is registered.
        """
        callbacks = self.context.world.get_world_model_manager().model_change_callbacks
        for callback in callbacks:
            if isinstance(callback, GraspAttachmentBackend):
                return callback
        raise NoGraspAttachmentBackend(self.body)


@dataclass
class MujocoAttachExecutable(MujocoGraspExecutable):
    """
    Re-attaches a grasped body to the gripper in the world model and, in
    :attr:`MujocoGraspMode.ATTACHMENT` mode, rigidly binds it in the MuJoCo world.
    """

    def execute(self) -> None:
        """
        Re-parent the body to the gripper in the world model and, in attachment mode,
        additionally bind it to the gripper in MuJoCo.
        """
        super().execute()
        if self.grasp_mode is MujocoGraspMode.FRICTION:
            return
        self._grasp_attachment_backend().attach_grasped_body(self.body, self.new_parent)


@dataclass
class MujocoDetachExecutable(MujocoGraspExecutable):
    """
    Detaches a body from its gripper in the world model and, in
    :attr:`MujocoGraspMode.ATTACHMENT` mode, releases it in the MuJoCo world first so it
    resumes free-body physics.
    """

    def execute(self) -> None:
        """
        Release the body in MuJoCo (attachment mode only) and then re-parent it to the
        new parent in the world model.
        """
        if self.grasp_mode is MujocoGraspMode.ATTACHMENT:
            self._grasp_attachment_backend().release_grasped_body(self.body)
        super().execute()
