from copy import deepcopy

import pytest

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import (
    PickUpEvent,
    SupportEvent,
    TranslationEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import TranslationDetector
from segmind.detectors.base import SegmindContext
from segmind.detectors.coarse_event_detector_nodes import PickUpDetector
from segmind.detectors.spatial_relation_detector_nodes import (
    LossOfSupportDetector,
    SupportDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% standing in for what a robot does to an object it grasps


def carry_body(world: World, body: Body, carrier: Body) -> None:
    """
    Re-attach a body to the body carrying it, keeping its global pose.

    This is what coraplex does to an object when a pick-up grasps it, down to the
    connection type (see ``ModelChangeExecutable.execute``).

    :param world: The world both bodies live in.
    :param body: The body being carried.
    :param carrier: The body carrying it.
    """
    carrier_transform_body = world.compute_forward_kinematics(carrier, body)
    with world.modify_world():
        world.remove_connection(body.parent_connection)
        world.add_connection(
            FixedConnection(
                parent=carrier,
                child=body,
                parent_T_connection_expression=carrier_transform_body,
            )
        )


def world_with_carrier(apartment: World) -> World:
    """
    :param apartment: The apartment to copy.
    :return: A copy of the apartment holding an extra body able to carry objects.
    """
    world = deepcopy(apartment)
    carrier = Body(
        name=PrefixedName("carrier"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
        visual=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
    )
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                parent=world.root, child=carrier, world=world
            )
        )
    carrier.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0, 0, 1.8, reference_frame=world.root
    )
    return world


def events_of(segmind_context: SegmindContext, event_type: type) -> list:
    """
    :param segmind_context: The context whose logger holds the timeline.
    :param event_type: The type of event to select.
    :return: Every logged event of that type.
    """
    return [
        event
        for event in segmind_context.logger.get_events()
        if isinstance(event, event_type)
    ]


# %% an object has to stay segmented once something picks it up


@pytest.mark.xfail(
    strict=True,
    reason="A detector only tracks bodies whose parent connection is a Connection6DoF, "
    "so an object leaves the tracked set as soon as a grasp re-attaches it to the "
    "body carrying it. Its motion is no longer detected and no pick-up is derived.",
)
def test_object_is_segmented_while_it_is_carried(_simple_apartment_setup):
    world = world_with_carrier(_simple_apartment_setup)
    milk = world.get_body_by_name("milk.stl")
    surface = world.get_body_by_name("box_2")
    carrier = world.get_body_by_name("carrier")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=surface.global_pose.x,
        y=surface.global_pose.y,
        z=surface.global_pose.z + 0.56,
        reference_frame=milk.parent_connection.parent,
    )

    segmind_executor = EpisodeSegmenterExecutor(
        context=MotionStatechartContext(world=world)
    )
    segmind_context = segmind_executor.context.require_extension(SegmindContext)
    segmind_executor.compile(
        SegmindStatechart().build_statechart(
            [
                SupportDetector(),
                LossOfSupportDetector(),
                TranslationDetector(),
                PickUpDetector(),
            ]
        )
    )
    segmind_executor.tick()

    assert [
        event.tracked_object for event in events_of(segmind_context, SupportEvent)
    ] == [milk]

    carry_body(world, milk, carrier)
    for step in range(6):
        carrier.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            0, 0, 1.8 + (step + 1) * 0.1, reference_frame=world.root
        )
        segmind_executor.tick()

    assert [
        event.tracked_object for event in events_of(segmind_context, TranslationEvent)
    ] == [milk]
    [pick_up] = events_of(segmind_context, PickUpEvent)
    assert pick_up.tracked_object is milk
    assert pick_up.with_object is surface
