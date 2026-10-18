"""
The HSR performing RoboCasa's composite tasks in a RoboCasa kitchen, simulated in
MuJoCo, as SegMind sees it.
"""

import numpy as np
import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

from experiments.hsr_robocasa.composite import (
    AlignSilverware,
    BreadAndCheese,
    DefrostByCategory,
    RecycleBottlesByType,
)
from experiments.hsr_robocasa.pick_place import HSRPickPlaceTask, TaskObject
from semantic_digital_twin.semantic_annotations.semantic_annotations import Sink
from segmind.datastructures.events import LossOfGraspEvent, PlacingEvent

from .test_pick_place import HSRLimitation, runs_in_ci

# %% where the objects are set down


def test_each_object_is_set_down_on_a_spot_of_the_board_of_its_own():
    experiment = BreadAndCheese(headless=True)
    experiment.build_world()

    footprints = []
    for moved in experiment.moved_objects:
        experiment.moving = moved
        center = experiment.release_point[:2]
        half_size = (
            np.array(
                [
                    experiment.extent(experiment.food.root).width,
                    experiment.extent(experiment.food.root).depth,
                ]
            )
            / 2
        )
        footprints.append((center - half_size, center + half_size))

    (first_low, first_high), (second_low, second_high) = footprints
    overlap = np.minimum(first_high, second_high) - np.maximum(first_low, second_low)
    assert np.any(overlap < 0.0)


# %% where each object goes


def test_fruit_goes_into_the_sink_and_vegetables_into_the_bowl():
    experiment = DefrostByCategory(headless=True)
    experiment.build_world()
    sink = experiment.world.get_semantic_annotations_by_type(Sink)[0].root
    bowl = experiment.world.get_body_by_name(TaskObject.CONTAINER)

    receptacles = {}
    for moved in experiment.moved_objects:
        experiment.moving = moved
        receptacles[moved] = experiment.receptacle

    assert [receptacles[fruit] for fruit in experiment.fruits] == [sink, sink]
    assert [receptacles[vegetable] for vegetable in experiment.vegetables] == [
        bowl,
        bowl,
    ]


def test_each_middle_bottle_is_set_down_next_to_bottles_of_its_kind():
    """
    Where a bottle is let go lies within RoboCasa's cluster threshold of a bottle of
    its kind, and outside every other bottle's footprint grown by its own radius and
    the hand's half width.
    """
    experiment = RecycleBottlesByType(headless=True)
    experiment.build_world()

    for moved in experiment.moved_objects:
        experiment.moving = moved
        spot = experiment.release_point[:2]
        own_kind = [
            experiment.extent(experiment.world.get_body_by_name(sample)).center.to_np()[
                :2
            ]
            for sample in experiment.samples_of_its_kind
        ]
        assert (
            min(np.linalg.norm(spot - sample) for sample in own_kind)
            <= RecycleBottlesByType.cluster_threshold
        )
        for other in experiment.task.manipulated_objects:
            if other.name.name == moved:
                continue
            extent = experiment.extent(other)
            clearance = experiment.bottle_radius + experiment.hand_half_width
            off_centre = np.abs(spot - extent.center.to_np()[:2])
            half_sizes = np.array([extent.width, extent.depth]) / 2
            # To a millimetre: the free space grows each of the bottle's shapes, not
            # the box around all of them.
            assert np.any(off_centre >= half_sizes + clearance - 0.001)


def placed_beside_the_plate(experiment: AlignSilverware, middle: np.ndarray) -> bool:
    """
    :return: Whether ``middle`` lies on the side of the plate the object being moved
        belongs on, within RoboCasa's thresholds, in the seat's frame.
    """
    plate = experiment.extent(
        experiment.world.get_body_by_name(experiment.plate)
    ).center.to_np()[:2]
    lateral, forward = experiment.in_seat_frame(middle - plate)
    return (
        np.sign(lateral) == experiment.side
        and abs(lateral) <= AlignSilverware.lateral_threshold
        and abs(forward) <= AlignSilverware.forward_threshold
    )


def test_the_fork_is_set_down_left_of_the_plate_and_the_spoon_right():
    experiment = AlignSilverware(headless=True)
    experiment.build_world()

    for moved in experiment.moved_objects:
        experiment.moving = moved
        assert placed_beside_the_plate(experiment, experiment.release_point[:2])


# %% the run


def objects_put_down_in_their_receptacles(experiment: HSRPickPlaceTask) -> bool:
    """
    :return: Whether SegMind saw each of the task's moved objects put down in its
        receptacle the last time it was let go of.
    """
    for moved in experiment.moved_objects:
        experiment.moving = moved
        body = experiment.world.get_body_by_name(moved)
        events = [e for e in experiment.detected_events if e.tracked_object is body]
        last_release = max(
            e.timestamp for e in events if isinstance(e, LossOfGraspEvent)
        )
        put_down_on = {
            e.with_object
            for e in events
            if isinstance(e, PlacingEvent) and e.timestamp >= last_release
        }
        if not put_down_on & set(
            experiment.colliding_bodies_below(experiment.receptacle)
        ):
            return False
    return True


@runs_in_ci
@pytest.mark.xfail(strict=True, reason=HSRLimitation.ARM_MARGIN)
def test_segmind_sees_both_objects_put_down_on_the_board():
    experiment = BreadAndCheese(headless=True)
    experiment.build_world()

    experiment.run()

    assert objects_put_down_in_their_receptacles(experiment)


@runs_in_ci
@pytest.mark.xfail(strict=True, reason=HSRLimitation.ARM_MARGIN)
def test_segmind_sees_fruit_put_into_the_sink_and_vegetables_into_the_bowl():
    experiment = DefrostByCategory(headless=True)
    experiment.build_world()

    experiment.run()

    assert objects_put_down_in_their_receptacles(experiment)


@runs_in_ci
@pytest.mark.xfail(strict=True, reason=HSRLimitation.OBJECTS_TOO_CLOSE)
def test_segmind_sees_each_middle_bottle_put_down_next_to_its_kind():
    experiment = RecycleBottlesByType(headless=True)
    experiment.build_world()

    experiment.run()

    assert objects_put_down_in_their_receptacles(experiment)
    for moved in experiment.moved_objects:
        experiment.moving = moved
        bottle = experiment.extent(experiment.world.get_body_by_name(moved))
        assert (
            min(
                np.linalg.norm(
                    bottle.center.to_np()[:2]
                    - experiment.extent(
                        experiment.world.get_body_by_name(sample)
                    ).center.to_np()[:2]
                )
                for sample in experiment.samples_of_its_kind
            )
            <= RecycleBottlesByType.cluster_threshold
        )


@runs_in_ci
@pytest.mark.xfail(strict=True, reason=HSRLimitation.PLACING_HEADING_OUT_OF_REACH)
def test_segmind_sees_the_fork_and_spoon_put_down_beside_the_plate():
    experiment = AlignSilverware(headless=True)
    experiment.build_world()

    experiment.run()

    assert objects_put_down_in_their_receptacles(experiment)
    for moved in experiment.moved_objects:
        experiment.moving = moved
        middle = experiment.extent(experiment.world.get_body_by_name(moved)).center
        assert placed_beside_the_plate(experiment, middle.to_np()[:2])
