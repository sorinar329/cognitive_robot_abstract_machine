"""
Tests for SegMind judging, action by action, whether a plan achieved what its actions
are for, on Tracy sorting the Montessori pieces and on the HSR carrying the milk, both
against MuJoCo physics.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from segmind.datastructures.events import (
    GraspEvent,
    LossOfGraspEvent,
    LossOfSupportEvent,
    SupportEvent,
)
from segmind.detectors.agent_event_detector_nodes import GraspDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from segmind.event_segmentation import Segmind
from segmind.post_conditions.judge import (
    JudgedBy,
    PostConditionJudge,
    PostConditionVerdict,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.world import World

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


# %% the HSR carrying the milk


@dataclass
class MilkTransportRun:
    """
    The HSR MuJoCo demo carried out once, with SegMind watching the milk.
    """

    world: World
    """
    The world the milk was carried in.
    """

    verdicts: list[PostConditionVerdict]
    """
    The verdicts on the actions of the run, in the order the actions started.
    """


@pytest.fixture(scope="module")
def milk_transport_run(hsr_mujoco_demo) -> MilkTransportRun:
    """
    The demo carried out once against MuJoCo physics, without a viewer, judged as it
    runs.
    """
    demonstration = hsr_mujoco_demo.MilkTransportDemonstration(headless=True)
    try:
        world = demonstration.acquire_world()
        demonstration.start_simulation(world)
        plan = demonstration.build_plan(demonstration.build_context(world))
        [milk] = world.get_semantic_annotations_by_type(Milk)
        segmind = Segmind.create_for_bodies(
            world, [milk.root], [GraspDetector, SupportDetector]
        )
        judge = PostConditionJudge(plan=plan, segmind=segmind)
        with segmind, judge, demonstration.execution_environment():
            plan.perform()
    finally:
        demonstration.tear_down()
    return MilkTransportRun(world=world, verdicts=judge.verdicts)


def test_the_hsrs_pick_up_holds_when_segmind_sees_the_milk_taken_hold_of_and_lifted(
    milk_transport_run,
):
    pick_ups = verdicts_on(milk_transport_run.verdicts, PickUpAction)

    assert [
        (verdict.holds, verdict.judged_by, {type(event) for event in verdict.evidence})
        for verdict in pick_ups
    ] == [(True, JudgedBy.SEGMIND, {GraspEvent, LossOfSupportEvent})]


def test_the_hsrs_placing_holds_when_segmind_sees_the_milk_let_go_of_onto_the_table(
    milk_transport_run, hsr_mujoco_demo
):
    world = milk_transport_run.world
    [hsrb] = world.get_semantic_annotations_by_type(HSRB)
    [arm] = hsrb.all_arms
    placings = verdicts_on(milk_transport_run.verdicts, PlaceAction)

    assert [
        (
            verdict.holds,
            verdict.judged_by,
            [(type(event), event.with_object) for event in verdict.evidence],
        )
        for verdict in placings
    ] == [
        (
            True,
            JudgedBy.SEGMIND,
            [
                (LossOfGraspEvent, arm.end_effector.tool_frame),
                (
                    SupportEvent,
                    world.get_body_by_name(hsr_mujoco_demo.SceneBody.TABLE),
                ),
            ],
        )
    ]


def test_the_hsrs_driving_and_posing_are_judged_on_the_world(milk_transport_run):
    moves = [
        verdict
        for action_type in (MoveTorsoAction, ParkArmsAction, NavigateAction)
        for verdict in verdicts_on(milk_transport_run.verdicts, action_type)
    ]

    assert {(verdict.holds, verdict.judged_by) for verdict in moves} == {
        (True, JudgedBy.WORLD)
    }
