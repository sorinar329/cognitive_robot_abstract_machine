"""
SegMind annotating a recorded episode on demand: replaying its frames in the world its
plan started in, and detecting events there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from typing_extensions import Type

from coraplex.episodes import Episode
from segmind.datastructures.events import DetectionEvent
from segmind.detectors.base import AbstractDetector
from segmind.event_segmentation import Segmind
from segmind.exceptions import EpisodeWithoutFrames
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

# %% annotations


@dataclass
class EpisodeAnnotation:
    """
    What SegMind detected in an episode.
    """

    episode: Episode
    """
    The episode annotated.
    """

    detector_names: list[str]
    """
    The kinds of detector that watched the episode.
    """

    annotated_at: datetime
    """
    When the episode was annotated.
    """

    events: list[DetectionEvent]
    """
    The events detected, each carrying the moment of the episode it was seen at.
    """


# %% annotating


@dataclass
class EpisodeAnnotator:
    """
    Annotates episodes with the events SegMind detects in them.

    The frames are replayed in the world the episode's plan started in. The detectors
    tick at the first frame, then at the first frame a tick period after the previous
    tick, and at the last frame, each at the moment its frame was recorded. The world
    is then held at the last frame for as long as the detectors need to see it at rest.
    """

    detector_types: list[Type[AbstractDetector]] = field(default_factory=list)
    """
    The kinds of detector asked for; every kind SegMind defines when none.
    """

    def annotate(self, episode: Episode) -> EpisodeAnnotation:
        """
        :return: The events SegMind detects on the bodies that move freely in the
            episode's world and can touch others, other than the robots' own.
        :raises EpisodeWithoutFrames: If the episode holds no frame.
        """
        if not episode.frames:
            raise EpisodeWithoutFrames(episode=episode)
        world = episode.plan.initial_world
        segmind = Segmind.create_for_bodies(
            world, self.free_bodies_of(world), self.detector_types
        )
        tick_period = timedelta(seconds=segmind.tick_period)
        last_tick = None
        for frame in episode.frames:
            if (
                last_tick is not None
                and frame.time - last_tick < tick_period
                and frame is not episode.frames[-1]
            ):
                continue
            episode.show_frame(frame, world)
            segmind.tick(at=frame.time)
            last_tick = frame.time
        for tick in range(1, segmind.ticks_to_see_the_world_at_rest):
            segmind.tick(at=last_tick + tick * tick_period)
        return EpisodeAnnotation(
            episode=episode,
            detector_names=segmind.detector_names,
            annotated_at=datetime.now(),
            events=segmind.event_logger.get_events(),
        )

    @staticmethod
    def free_bodies_of(world: World) -> list[Body]:
        """
        :return: The bodies of ``world`` that move freely and can touch others, other
            than the robots' own.
        """
        robot_bodies = {
            body
            for robot in world.get_semantic_annotations_by_type(AbstractRobot)
            for body in robot.bodies
        }
        return [
            body
            for body in world.bodies_with_collision
            if isinstance(body.parent_connection, Connection6DoF)
            and body not in robot_bodies
        ]
