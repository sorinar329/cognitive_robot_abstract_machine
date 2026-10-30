"""
Tests for SegMind judging, while a plan runs, whether each action that acts on an object
caused the events it is expected to: on the HSR carrying a milk carton, and on Tracy
sorting the pieces of a shape-sorting board, both in MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import ModuleType

import pytest

from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from cramph.executor import ExecutorExtension
from segmind.action_judge import ActionJudge
from segmind.datastructures.events import (
    GraspEvent,
    LossOfGraspEvent,
    LossOfSupportEvent,
    SupportEvent,
)
from segmind.detectors.agent_event_detector_nodes import GraspDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from segmind.event_segmentation import Segmind
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    ShapeSortingPiece,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World

from ..demo_folders import DemoFolder
from ..pytest_environment import runs_in_continuous_integration

pytestmark = [
    pytest.mark.skipif(
        not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
    ),
    pytest.mark.skipif(
        not HSRB.is_description_installed() or not Tracy.is_description_installed(),
        reason="The descriptions of the HSR and Tracy are not installed",
    ),
]

# %% the HSR demo, judged


@pytest.fixture(scope="module")
def hsr_demo() -> ModuleType:
    """
    The HSR demo's ``demo`` module.
    """
    return DemoFolder.HSR_MUJOCO.load_module("demo")


def judged(milk_transport_type: type) -> type:
    """
    :param milk_transport_type: The HSR demo's milk transport.
    :return: The milk transport, with SegMind watching the milk and an
        :class:`ActionJudge` judging every action that acts on it.
    """

    @dataclass
    class MilkTransportWatchedByAJudge(milk_transport_type):
        """
        The milk transport, its pick-up aimed :attr:`grasp_offset` to the side of the
        grasp it would take.
        """

        grasp_offset: float = 0.0
        """
        How far to the side of the milk the hand is sent, in metres.
        """

        segmind: Segmind | None = field(init=False, default=None)
        """
        SegMind watching the milk, once the executor is created.
        """

        judge: ActionJudge | None = field(init=False, default=None)
        """
        The judge of the run, once the executor is created.
        """

        def executor_extensions(self, world: World) -> list[ExecutorExtension]:
            self.segmind = Segmind.create_for_semantic_annotation_types(
                world, [Milk], detector_types=(GraspDetector, SupportDetector)
            )
            self.judge = ActionJudge(segmind=self.segmind)
            return [*super().executor_extensions(world), self.judge]

        def segment_events(self, world: World) -> Segmind:
            return self.segmind

        def _grasp_from(self, standing_pose: Pose, milk: Milk) -> GraspCandidate:
            grasp = super()._grasp_from(standing_pose, milk)
            beside = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=self.grasp_offset, reference_frame=milk.root
                )
                @ grasp.grasp_pose.homogeneous_matrix
            )
            return replace(grasp, grasp_pose=beside.pose)

    return MilkTransportWatchedByAJudge


@pytest.fixture(scope="module")
def judged_transport(hsr_demo) -> ActionJudge:
    """
    The judge of a milk transport that carries the milk to the table.
    """
    demonstration = judged(hsr_demo.MilkTransportDemonstration)(headless=True)
    demonstration.run()
    return demonstration.judge


# %% verdicts


def test_only_the_actions_that_act_on_the_milk_are_judged(judged_transport):
    assert [type(verdict.action) for verdict in judged_transport.verdicts] == [
        PickUpAction,
        PlaceAction,
    ]


def test_the_pick_up_holds_once_the_milk_is_held_and_off_the_counter(
    judged_transport,
):
    [pick_up] = [
        verdict
        for verdict in judged_transport.verdicts
        if isinstance(verdict.action, PickUpAction)
    ]

    assert pick_up.holds
    assert {type(event) for event in pick_up.evidence} == {
        GraspEvent,
        LossOfSupportEvent,
    }


def test_the_placing_holds_once_the_milk_is_let_go_of_onto_the_table(
    judged_transport,
):
    [place] = [
        verdict
        for verdict in judged_transport.verdicts
        if isinstance(verdict.action, PlaceAction)
    ]

    assert place.holds
    assert {type(event) for event in place.evidence} == {
        LossOfGraspEvent,
        SupportEvent,
    }


def test_a_pick_up_that_closes_the_hand_beside_the_milk_does_not_hold(hsr_demo):
    demonstration = judged(hsr_demo.MilkTransportDemonstration)(
        headless=True, grasp_offset=0.1
    )

    demonstration.run()

    [pick_up] = [
        verdict
        for verdict in demonstration.judge.verdicts
        if isinstance(verdict.action, PickUpAction)
    ]
    assert not pick_up.holds


# %% Tracy sorting the pieces, judged


@pytest.fixture(scope="module")
def tracy_runner_module() -> ModuleType:
    """
    The Tracy Montessori demo's ``mujoco_demo`` module.
    """
    return DemoFolder.TRACY_MONTESSORI.load_module("mujoco_demo")


@pytest.fixture(scope="module")
def judged_sorting(tracy_runner_module):
    """
    A run of Tracy sorting the pieces, with SegMind watching them and an
    :class:`ActionJudge` judging every action that acts on one.
    """

    @dataclass
    class SortingWatchedByAJudge(tracy_runner_module.MujocoRunner):
        """
        The sorting, watched and judged.
        """

        world: World | None = field(init=False, default=None)
        """
        The world the pieces were sorted in, once the executor is created.
        """

        segmind: Segmind | None = field(init=False, default=None)
        """
        SegMind watching the pieces, once the executor is created.
        """

        judge: ActionJudge | None = field(init=False, default=None)
        """
        The judge of the run, once the executor is created.
        """

        def executor_extensions(self, world: World) -> list[ExecutorExtension]:
            self.world = world
            self.segmind = Segmind.create_for_semantic_annotation_types(
                world,
                [ShapeSortingPiece],
                detector_types=(GraspDetector, SupportDetector),
            )
            self.judge = ActionJudge(segmind=self.segmind)
            return [self.judge]

        def segment_events(self, world: World) -> Segmind:
            return self.segmind

    runner = SortingWatchedByAJudge(headless=True)
    runner.run()
    return runner


def test_every_piece_is_judged_picked_up_and_then_placed(judged_sorting):
    assert [
        (type(verdict.action), verdict.expected[0].tracked_object)
        for verdict in judged_sorting.judge.verdicts
    ] == [
        (action_type, piece.root)
        for piece in judged_sorting.scene.holes_of_pieces
        for action_type in (PickUpAction, PlaceAction)
    ]


def test_every_pick_up_of_a_piece_off_the_table_holds(judged_sorting):
    """
    The table Tracy is bolted to is part of the robot, yet nothing the robot drives
    moves it, so SegMind sees each piece lose its support when it is lifted off it.
    """
    pick_ups = [
        verdict
        for verdict in judged_sorting.judge.verdicts
        if isinstance(verdict.action, PickUpAction)
    ]

    assert [verdict.holds for verdict in pick_ups] == [True] * len(pick_ups)


def test_the_placing_of_every_piece_that_dropped_into_the_board_holds(
    judged_sorting,
):
    world = judged_sorting.world
    scene = judged_sorting.scene
    lid_top = (
        world.compute_forward_kinematics_np(world.root, scene.board)[2, 3]
        + scene.shape_sorting_board.geometry.lid_top
    )
    dropped_in = {
        piece.root
        for piece in scene.holes_of_pieces
        if world.compute_forward_kinematics_np(world.root, piece.root)[2, 3] < lid_top
    }
    placings_of_dropped_in = [
        verdict
        for verdict in judged_sorting.judge.verdicts
        if isinstance(verdict.action, PlaceAction)
        and verdict.expected[0].tracked_object in dropped_in
    ]

    assert dropped_in
    assert [verdict.holds for verdict in placings_of_dropped_in] == [True] * len(
        dropped_in
    )
