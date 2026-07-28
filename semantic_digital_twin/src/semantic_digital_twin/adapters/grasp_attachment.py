from __future__ import annotations

from abc import ABC, abstractmethod

from semantic_digital_twin.world_description.world_entity import Body


class GraspAttachmentBackend(ABC):
    """
    Interface for a simulator backend that can rigidly bind a grasped body to a gripper
    and release it again.

    Implemented by the simulator synchronizer (for example
    :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSynchronizer`) so that
    plan code can drive the physical grasp attachment without importing a concrete
    simulator. The plan locates the backend among the world's model-change callbacks
    by this type.
    """

    @abstractmethod
    def attach_grasped_body(self, grasped_body: Body, gripper_body: Body) -> None:
        """
        Rigidly bind ``grasped_body`` to ``gripper_body`` in the backing simulator so
        the grasped body follows the gripper.

        :param grasped_body: The body that was grasped.
        :param gripper_body: The gripper body the grasped body is bound to.
        """
        raise NotImplementedError

    @abstractmethod
    def release_grasped_body(self, grasped_body: Body) -> None:
        """
        Release ``grasped_body`` from its gripper in the backing simulator so it becomes
        a free body again.

        :param grasped_body: The body to release.
        """
        raise NotImplementedError
