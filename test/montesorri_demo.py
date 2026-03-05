import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from unittest import TestCase

import rclpy


from segmind.detectors.atomic_event_detectors_nodes import SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.players.csv_player import CSVEpisodePlayer
from segmind.players.data_player import DataPlayer
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


class TestMultiverseEpisodeSegmenter(TestCase):
    world: World
    file_player: CSVEpisodePlayer

    viz_marker_publisher: VizMarkerPublisher

    @classmethod
    def setUpClass(cls):
        multiverse_episodes_dir = (
            "/home/sorin/dev/Segmind/resources/multiverse_episodes"
        )
        selected_episode = "icub_montessori_no_hands"
        episode_dir = os.path.join(multiverse_episodes_dir, selected_episode)
        csv_file = os.path.join(episode_dir, f"data.csv")
        print(csv_file)
        models_dir = os.path.join(episode_dir, "models")
        cls.world = World()
        root = Body(name=PrefixedName(name="root", prefix="world"))
        with cls.world.modify_world():
            cls.world.add_kinematic_structure_entity(root)
        cls.spawn_objects(models_dir)
        rclpy.init()
        cls.node = rclpy.create_node("test_node")
        cls.viz_marker_publisher = VizMarkerPublisher(node=cls.node, _world=cls.world)
        cls.viz_marker_publisher.with_tf_publisher()
        cls.file_player = CSVEpisodePlayer(
            file_path=csv_file,
            world=cls.world,
            time_between_frames=datetime.timedelta(milliseconds=4),
            position_shift=Vector3(0, 0, 0),
        )

    @classmethod
    def spawn_objects(cls, models_dir):
        # cls.copy_model_files_to_world_data_dir(models_dir)
        directory = Path(models_dir)
        urdf_files = [f.name for f in directory.glob("*.urdf")]
        for file in urdf_files:
            file_path = (
                "/home/sorin/dev/Segmind/resources/multiverse_episodes/icub_montessori_no_hands/models/"
                + file
            )
            obj_name = Path(file).stem

            if obj_name == "iCub":
                continue
            try:
                if obj_name == "scene":
                    obj_world = URDFParser.from_file(file_path).parse()
                    world_C_scene = FixedConnection(
                        parent=cls.world.root, child=obj_world.root
                    )
                    with cls.world.modify_world():

                        cls.world.merge_world(obj_world, world_C_scene)
                else:
                    obj_world = URDFParser.from_file(file_path).parse()
                    with cls.world.modify_world():
                        cls.world.merge_world(obj_world)

            except Exception as e:
                # import pdb
                # pdb.set_trace()

                continue


    @classmethod
    def tearDownClass(cls):
       pass

    def tearDown(self):
        self.file_player.reset()


    def test_replay_episode(self):
        self.file_player.start()