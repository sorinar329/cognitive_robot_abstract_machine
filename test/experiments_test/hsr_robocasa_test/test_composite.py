"""
The HSR performing RoboCasa's composite tasks in a RoboCasa kitchen, simulated in
MuJoCo, as SegMind sees it.
"""

import numpy as np
import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

from experiments.hsr_robocasa.composite import BreadAndCheese, DefrostByCategory
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
