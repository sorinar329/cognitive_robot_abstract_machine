import datetime
import os
from pathlib import Path

import pytest
import rclpy

import segmind
from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.detectors.base import SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.players.csv_player import CSVEpisodePlayer
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.package_resolver import FileUriResolver
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture(scope="function")
def test_csv_player_context():
    scene_path = "/home/sorin/dev/cognitive_robot_abstract_machine/segmind/resources/tiago_episodes/models/assets/mjcf/iai_tiago_velocity_in_apartment_with_multiverse.xml"
    world = MJCFParser(scene_path).parse()

    #multiverse_episodes_dir = (
    #    f"{Path(segmind.__file__).parent.parent.parent}/resources/multiverse_episodes"
    #)
    #
    file_player = CSVEpisodePlayer(
        file_path="/home/sorin/dev/cognitive_robot_abstract_machine/segmind/resources/tiago_episodes/data/data.csv",
        world=world,
        time_between_frames=datetime.timedelta(milliseconds=0.01),
    )
    context = MotionStatechartContext(world=world)
    episode_executor = EpisodeSegmenterExecutor(
        context=context,
        player=file_player,
        ignored_objects=["iCub"],
        fixed_objects=["scene"],
    )

    return {
        "world": world,
        "context": context,
        "file_player": file_player,
        "episode_executor": episode_executor,
    }

def test_segmind_demo(test_csv_player_context):
    world = test_csv_player_context["world"]
    episode_executor = test_csv_player_context["episode_executor"]
    file_player = test_csv_player_context["file_player"]

    rclpy.init()
    node = rclpy.create_node("test_csv_player")
    viz_marker_publisher = VizMarkerPublisher(_world=world, node=node)
    viz_marker_publisher.with_tf_publisher()

    file_player.start()
    # statechart = SegmindStatechart().build_statechart()
    # segmind_context = episode_executor.context.require_extension(SegmindContext)
    # episode_executor.compile(statechart)
    #
    #
    # try:
    #     while episode_executor.player.is_alive():
    #         episode_executor.tick()
    # finally:
    #     print(segmind_context.logger.get_events())
