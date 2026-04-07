import datetime
import os
from pathlib import Path
from unittest import TestCase
from os.path import dirname

import rclpy

import pycram.ros
from segmind.detectors.base import SegmindContext
from segmind.event_logger import EventLogger

from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Cup, Food
from semantic_digital_twin.world import World


from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.players.json_player import JSONPlayer


from semantic_digital_twin.world_description.world_entity import Body

Multiverse = None
try:
    from pycram.worlds.multiverse import Multiverse
except ImportError:
    pass


class TestFileEpisodeSegmenter(TestCase):
    world: World
    file_player: JSONPlayer
    episode_segmenter: EpisodeSegmenterExecutor
    viz_marker_publisher: VizMarkerPublisher

    @classmethod
    def setUpClass(cls):
        json_file = f"{dirname(__file__)}/../resources/fame_episodes/alessandro_with_ycp_objects_in_max_room_2/refined_poses.json"
        cls.world = World()
        cls.logger = EventLogger()
        cls.context = SegmindContext(world=cls.world, logger=cls.logger)
        root = Body(name=PrefixedName(name="root", prefix="world"))
        with cls.world.modify_world():
            cls.world.add_kinematic_structure_entity(root)

        obj_id_to_name = {1: "obj_000001", 3: "obj_000003", 4: "obj_000004", 6: "obj_000006"}
        cls.file_player = JSONPlayer(json_file, world=cls.world,
                                            time_between_frames=datetime.timedelta(milliseconds=50),
                                            obj_id_to_name=obj_id_to_name)

        cls.episode_segmenter = EpisodeSegmenterExecutor(player=cls.file_player, context=cls.context)
        cls.file_player.transform_to_stl("/home/sorin/dev/Segmind/resources/fame_episodes/alessandro_sliding_bueno/models")
        rclpy.init()
        cls.node = rclpy.create_node("test_node")
        cls.viz_marker_publisher = VizMarkerPublisher(node=cls.node, _world=cls.world)

        cls.viz_marker_publisher.with_tf_publisher()
        cls.episode_segmenter.spawn_scene(models_dir="/home/sorin/dev/Segmind/resources/fame_episodes/alessandro_sliding_bueno/models/")


    def test_replay_episode(self):
        self.episode_segmenter.start()

