"""
The run of a plan as an episode: its timestamped joint states and the poses of the
bodies that move freely, kept with the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from typing_extensions import Self

from coraplex.plans.plan import Plan
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import PlanNode
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF

# %% episodes


@dataclass
class EpisodeFrame:
    """
    The world at one moment of an episode.
    """

    time: datetime
    """
    The moment the frame shows.
    """

    joint_positions: np.ndarray
    """
    The position of every degree of freedom, in the order of
    :attr:`Episode.degree_of_freedom_names`.
    """

    free_body_poses: np.ndarray
    """
    The global pose of every free body, as a homogeneous matrix, in the order of
    :attr:`Episode.free_body_names`.
    """


@dataclass
class Episode:
    """
    A performed plan and the frames of the world while it ran.
    """

    plan: Plan
    """
    The plan whose run the episode shows.
    """

    degree_of_freedom_names: list[PrefixedName]
    """
    The degrees of freedom whose positions a frame holds.
    """

    free_body_names: list[PrefixedName]
    """
    The bodies whose poses a frame holds: those that moved freely in the world the plan
    started in.
    """

    frames: list[EpisodeFrame] = field(default_factory=list)
    """
    The frames, in the order of their time.
    """

    @classmethod
    def starting_in(cls, plan: Plan) -> Self:
        """
        :return: An episode of ``plan`` without frames, for the world it starts in.
        """
        world = plan.world
        return cls(
            plan=plan,
            degree_of_freedom_names=list(world.state.to_position_dict()),
            free_body_names=[
                body.name
                for body in world.bodies
                if isinstance(body.parent_connection, Connection6DoF)
            ],
        )

    def record_frame(self, time: datetime) -> None:
        """
        Add a frame of the plan's world as it is now.

        :param time: The moment the frame is taken at.
        """
        world = self.plan.world
        positions = world.state.to_position_dict()
        self.frames.append(
            EpisodeFrame(
                time=time,
                joint_positions=np.array(
                    [positions[name] for name in self.degree_of_freedom_names]
                ),
                free_body_poses=np.array(
                    [
                        world.get_body_by_name(name).global_pose.to_np()
                        for name in self.free_body_names
                    ]
                ),
            )
        )

    def show_frame(self, frame: EpisodeFrame, world: World) -> None:
        """
        Put ``world`` where ``frame`` shows it: every degree of freedom it has a position
        of, then every free body at its pose.

        :param frame: A frame of this episode.
        :param world: A world holding the bodies and degrees of freedom of the world the
            plan started in.
        """
        positions = dict(zip(self.degree_of_freedom_names, frame.joint_positions))
        with world._world_lock:
            for degree_of_freedom in world.degrees_of_freedom:
                if degree_of_freedom.name in positions:
                    world.state[degree_of_freedom.id].position = positions[
                        degree_of_freedom.name
                    ]
            world.notify_state_change()
            for name, pose in zip(self.free_body_names, frame.free_body_poses):
                world.get_body_by_name(name).parent_connection.origin = (
                    HomogeneousTransformationMatrix(pose, reference_frame=world.root)
                )


# %% recording


@dataclass(eq=False)
class EpisodeFrameSampler(StateChangeCallback):
    """
    Adds a frame to an episode whenever the world changed, at most once per
    :attr:`sample_period`.
    """

    episode: Episode = field(kw_only=True)
    """
    The episode the frames are added to.
    """

    sample_period: timedelta = field(kw_only=True)
    """
    The least time between two frames.
    """

    def on_state_change(self, **kwargs) -> None:
        now = datetime.now()
        if now - self.episode.frames[-1].time >= self.sample_period:
            self.episode.record_frame(now)


@dataclass(eq=False)
class EpisodeRecorder(PlanCallback):
    """
    Records the run of a plan as an episode: entering it has the plan report to it,
    and :attr:`episode` holds the frames from the plan's start to its end.
    """

    sample_period: timedelta = field(
        kw_only=True, default_factory=lambda: timedelta(seconds=0.1)
    )
    """
    The least time between two frames while the plan runs.
    """

    episode: Episode | None = field(init=False, default=None)
    """
    The episode of the plan, once the plan started.
    """

    sampler: EpisodeFrameSampler | None = field(init=False, default=None)
    """
    What adds frames while the plan runs.
    """

    def __enter__(self) -> Self:
        self.plan.node_callbacks.append(self)
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.plan.node_callbacks.remove(self)
        if self.sampler is not None:
            self.sampler.stop()

    def on_start(self, node: PlanNode) -> None:
        """
        Start the episode with a frame of the world the plan starts in.
        """
        if node is not self.plan.root:
            return
        self.episode = Episode.starting_in(self.plan)
        self.episode.record_frame(node.start_time)
        self.sampler = EpisodeFrameSampler(
            _world=self.plan.world,
            episode=self.episode,
            sample_period=self.sample_period,
        )

    def on_end(self, node: PlanNode) -> None:
        """
        End the episode with a frame of the world the plan leaves.
        """
        if node is not self.plan.root:
            return
        self.sampler.stop()
        self.episode.record_frame(node.end_time)
