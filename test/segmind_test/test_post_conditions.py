"""
Tests for SegMind judging, action by action, whether a plan achieved what its actions
are for, on Tracy sorting the Montessori pieces against MuJoCo physics.
"""

from __future__ import annotations

from typing_extensions import Any

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from segmind.datastructures.events import GraspEvent, LossOfSupportEvent
from segmind.detectors.agent_event_detector_nodes import GraspDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from segmind.event_segmentation import Segmind
from segmind.post_conditions.judge import (
    JudgedBy,
    PostConditionJudge,
    PostConditionVerdict,
)
from semantic_digital_twin.robots.tracy import Tracy

# %% Tracy sorting the pieces


@pytest.fixture(scope="module")
def sorting_run(tracy_montessori_demo) -> Any:
    """
    The demo carried out once against MuJoCo physics, without a viewer, with SegMind
    watching the pieces.

    :return: The runner, holding the scene and the verdicts of the run.
    """
    runner = tracy_montessori_demo.mujoco_demo.MujocoRunner(headless=True)
    runner.run()
    return runner


def verdicts_on(
    verdicts: list[PostConditionVerdict], action_type: type[ActionDescription]
) -> list[PostConditionVerdict]:
    """
    :return: The verdicts on the actions of ``action_type``, in the order the actions
        started.
    """
    return [
        verdict
        for verdict in verdicts
        if type(verdict.action_node.action) is action_type
    ]


# %% picking up and placing


def test_a_pick_up_holds_when_segmind_sees_the_piece_taken_hold_of_and_lifted(
    sorting_run,
):
    pick_ups = verdicts_on(sorting_run.verdicts, PickUpAction)

    assert [
        (verdict.holds, verdict.judged_by, {type(event) for event in verdict.evidence})
        for verdict in pick_ups
    ] == [(True, JudgedBy.SEGMIND, {GraspEvent, LossOfSupportEvent})] * len(
        sorting_run.scene.holes_of_pieces
    )


def test_a_placing_holds_when_the_piece_comes_to_rest_in_the_board(sorting_run):
    """
    A piece that drops through its hole comes to rest on a drawer's floor; one that
    wedges in its hole only touches the board.
    """
    scene = sorting_run.scene
    lid_top = (
        scene.board.global_pose.to_np()[2, 3]
        + scene.shape_sorting_board.geometry.lid_top
    )
    placings = verdicts_on(sorting_run.verdicts, PlaceAction)

    assert [(verdict.holds, verdict.judged_by) for verdict in placings] == [
        (bool(piece.root.global_pose.to_np()[2, 3] < lid_top), JudgedBy.SEGMIND)
        for piece in scene.holes_of_pieces
    ]


# %% what segmind does not judge


def test_an_action_segmind_has_no_expectation_of_is_judged_on_the_world(sorting_run):
    parkings = verdicts_on(sorting_run.verdicts, ParkArmsAction)

    assert {(verdict.holds, verdict.judged_by) for verdict in parkings} == {
        (True, JudgedBy.WORLD)
    }


def test_an_action_on_a_piece_segmind_does_not_watch_is_judged_on_the_world(
    tracy_montessori_demo,
):
    """
    Which piece is watched decides the verdict before anything is detected, so the
    robot is moved kinematically.
    """
    scene = tracy_montessori_demo.demo.MontessoriScene()
    world = scene.build_world()
    [tracy] = world.get_semantic_annotations_by_type(Tracy)
    picked, watched = list(scene.holes_of_pieces)[:2]
    [grasp] = picked.grasp_candidates()
    plan = sequential(
        [PickUpAction(grasp, scene.sorting_arm(tracy))],
        context=Context(world=world, robot=tracy, evaluate_conditions=False),
    ).plan
    segmind = Segmind.create_for_bodies(
        world, [watched.root], [GraspDetector, SupportDetector]
    )
    judge = PostConditionJudge(plan=plan, segmind=segmind)

    with segmind, judge, kinematically_simulated_robot:
        plan.perform()

    assert [
        verdict.judged_by for verdict in verdicts_on(judge.verdicts, PickUpAction)
    ] == [JudgedBy.NOT_WATCHED]


# %% failures


def test_no_action_of_a_run_that_succeeded_is_marked_failed(sorting_run):
    assert {verdict.action_failed for verdict in sorting_run.verdicts} == {False}
