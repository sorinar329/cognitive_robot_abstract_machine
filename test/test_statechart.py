import time
from collections import defaultdict
from typing import Dict, DefaultDict, Set

import rclpy


from krrood.symbolic_math.symbolic_math import trinary_logic_or
from segmind.datastructures.events import (
    ContactEvent,
    LossOfContactEvent,
    SupportEvent,
    LossOfSupportEvent, LossOfContainmentEvent, ContainmentEvent, InsertionEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    LossOfContactDetector,
    SegmindContext,
)

from segmind.detectors.spatial_relation_detector_nodes import (
    SupportDetector,
    LossOfSupportDetector, ContainmentDetector, LossOfContainmentDetector
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.event_logger import EventLogger

from giskardpy.executor import Executor

from giskardpy.motion_statechart.goals.templates import Sequence
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body
from test import setup_contact_world, setup_support_world


class TestMotionStatechart:


    #ToDo: We need to add some constraints on ticking, how many time do we tick per update?
    def test_contact_detector(self):

        world = setup_contact_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        cylinder = world.get_body_by_name("cylinder_body")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        assert (
            len(
                [
                    i
                    for i in self.context.logger.get_events()
                    if isinstance(i, ContactEvent)
                ]
            )
            == 0
        )


        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4)
        )
        self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 2


        self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 2

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=2)
        )

        self.segmind_executor.tick()
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 2
        )

        self.segmind_executor.tick()
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 2
        )

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4)
        )
        self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 4

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=2)
        )

        self.segmind_executor.tick()
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 4
        )

        rclpy.shutdown()

    def test_support_detector(self):
        world = setup_support_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        cylinder = world.get_body_by_name("cylinder_body")
        table = world.get_body_by_name("table_body")
        cabinet = world.get_body_by_name("cabinet")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 0
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfSupportEvent)])
            == 0
        )

        self.segmind_executor.tick()

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=table.global_pose.x,
                y=table.global_pose.y,
                z=table.global_pose.z + 0.2,
            )
        )
        self.segmind_executor.tick()

        self.segmind_executor.tick()
        assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 1

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=cabinet.global_pose.x,
                y=cabinet.global_pose.y,
                z=cabinet.global_pose.z,
            )
        )
        self.segmind_executor.tick()
        self.segmind_executor.tick()

        assert (len([i for i in logger.get_events() if isinstance(i, LossOfSupportEvent)]) == 1)

        rclpy.shutdown()

    def test_containment_detector(self):
        world = setup_support_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        cylinder = world.get_body_by_name("cylinder_body")
        cabinet = world.get_body_by_name("cabinet")
        table = world.get_body_by_name("table_body")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 0
        assert len([i for i in logger.get_events() if isinstance(i, ContainmentEvent)]) == 0


        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=cabinet.global_pose.x,
                y=cabinet.global_pose.y,
                z=cabinet.global_pose.z,
            )
        )

        self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, ContainmentEvent)]) == 1

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=table.global_pose.x,
                y=table.global_pose.y,
                z=table.global_pose.z + 0.2,
            )
        )
        self.segmind_executor.tick()


        assert len([i for i in logger.get_events() if isinstance(i, LossOfContainmentEvent)]) == 1


        rclpy.shutdown()




    def test_insertion_detector(self):
        world = setup_support_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        hole = world.get_body_by_name("hole_body")
        cylinder = world.get_body_by_name("cylinder_body")
        cabinet = world.get_body_by_name("cabinet")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)


        assert len(self.context.holes) == 1
        assert len([i for i in logger.get_events() if  isinstance(i, InsertionEvent)]) == 0
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=hole.global_pose.x,
                y=hole.global_pose.y - 0.03,
                z=hole.global_pose.z,
            )
        )
        self.segmind_executor.tick()


        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=cabinet.global_pose.x,
                y=cabinet.global_pose.y,
                z=cabinet.global_pose.z,
            )
        )

        self.segmind_executor.tick()

        contact_events = [i for i in self.context.logger.get_events() if isinstance(i, ContactEvent)]
        contact_events_with_holes = [i for i in contact_events if i.with_object in self.context.holes]

        assert len([i for i in logger.get_events() if isinstance(i, ContainmentEvent)]) == 1
        assert len(contact_events_with_holes) == 1
        assert len([i for i in logger.get_events() if isinstance(i, InsertionEvent)]) == 1

    def test_pickup(self):
        pass

    def test_placing(self):
        pass

    def test_translation(self):
        pass

    def test_stop_translation(self):
        pass

    def visualize(self, world):
        rclpy.init()
        node = rclpy.create_node("test_node")
        viz = VizMarkerPublisher(_world=world, node=node)
        viz.with_tf_publisher()
