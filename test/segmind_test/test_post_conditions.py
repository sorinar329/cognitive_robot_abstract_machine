from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest


from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
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
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from ..conftest import SAMPLING_SEED
from ..coraplex_test.conftest import left_or_only_arm
from ..coraplex_test.test_designator.test_multi_robot_action_designator import (
    ALTERNATIVE_MOTION_MAPPINGS,
    stand_facing,
)

# %% the robots in the apartment


@dataclass
class RobotSetup:
    """
    A robot the post-conditions are judged for, and the fixture its world comes from.
    """

    robot_type: type[AbstractRobot]
    """
    The type of the robot.
    """

    world_fixture: str
    """
    The name of the session fixture that parses the robot's world.
    """


@dataclass
class RobotInTheApartment:
    """
    One robot standing in its own copy of the apartment, and the context it acts in.
    """

    world: World
    """
    The apartment with the robot.
    """

    robot: AbstractRobot
    """
    The robot.
    """

    context: Context
    """
    The context the robot's plans are built in.
    """

    @property
    def milk(self) -> Body:
        """
        The milk carton of the apartment.
        """
        return self.world.get_body_by_name("milk.stl")

    @property
    def milk_annotation(self) -> Milk:
        """
        The milk carton's annotation, which the robot takes hold of by.
        """
        return self.world.get_semantic_annotations_by_type(Milk)[0]

    def set_down(self, body: Body, supporter: Body) -> None:
        """
        Lower or raise ``body`` until the bottom of its collision touches the top of
        ``supporter``'s.
        """
        body_box, supporter_box = [
            entity.collision.as_bounding_box_collection_in_frame(
                self.world.root
            ).bounding_box()
            for entity in (body, supporter)
        ]
        origin = body.parent_connection.origin.to_np()
        origin[2, 3] += float(
            (supporter_box.center.z + supporter_box.height / 2)
            - (body_box.center.z - body_box.height / 2)
        )
        body.parent_connection.origin = HomogeneousTransformationMatrix(
            origin, reference_frame=body.parent_connection.parent
        )
        self.world.notify_state_change()


@pytest.fixture(
    params=[
        RobotSetup(PR2, "_pr2_world_setup"),
        RobotSetup(Tiago, "_tiago_world_setup"),
        RobotSetup(HSRB, "_hsr_world_setup"),
    ],
    ids=lambda setup: setup.robot_type.__name__,
)
def robot_in_the_apartment(request, _apartment_world_setup) -> RobotInTheApartment:
    """
    A robot in its own copy of the apartment, by the island counter.
    """
    world = deepcopy(_apartment_world_setup)
    world.merge_world(deepcopy(request.getfixturevalue(request.param.world_fixture)))
    robot = world.get_semantic_annotations_by_type(request.param.robot_type)[0]
    robot.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.5, 2, 0
    )
    world.notify_state_change()
    return RobotInTheApartment(
        world=world,
        robot=robot,
        context=Context(
            world,
            robot,
            alternative_motion_mappings=ALTERNATIVE_MOTION_MAPPINGS,
            sampling_seed=SAMPLING_SEED,
            evaluate_conditions=False,
        ),
    )


@pytest.fixture
def milk_on_the_counter(robot_in_the_apartment) -> RobotInTheApartment:
    """
    The robot facing the milk, which rests on the island counter rather than hovering
    above it as the apartment is built.
    """
    scene = robot_in_the_apartment
    scene.set_down(scene.milk, scene.world.get_body_by_name("island_countertop"))
    scene.robot.root.parent_connection.origin = stand_facing(
        scene.robot,
        (1.6, 2, 0),
        scene.milk.global_pose.position.to_np()[:3],
        scene.world,
    )
    scene.world.notify_state_change()
    return scene


def judged(plan: PlanNode, segmind: Segmind) -> list[PostConditionVerdict]:
    """
    Perform ``plan`` kinematically while ``segmind`` watches it and judges its actions.

    :return: The verdict on each action of the plan.
    """
    judge = PostConditionJudge(plan=plan.plan, segmind=segmind)
    with segmind, judge, kinematically_simulated_robot:
        plan.perform()
    return judge.verdicts


def verdict_on(
    verdicts: list[PostConditionVerdict], action_type: type[ActionDescription]
) -> PostConditionVerdict:
    """
    :return: The one verdict on an action of ``action_type``.
    """
    [verdict] = [
        verdict
        for verdict in verdicts
        if type(verdict.action_node.action) is action_type
    ]
    return verdict


# %% picking up and placing


def pick_and_place_plan(scene: RobotInTheApartment, target: np.ndarray) -> PlanNode:
    """
    :return: A plan that picks the milk up and places it at ``target``.
    """
    return sequential(
        [
            ParkArmsAction(scene.robot.all_arms),
            PickUpAction(
                scene.milk_annotation.grasp_candidates()[0],
                left_or_only_arm(scene.robot),
            ),
            PlaceAction(
                scene.milk_annotation,
                Pose(Point3.from_iterable(target), reference_frame=scene.world.root),
            ),
        ],
        scene.context,
    )


def watching_the_milk(scene: RobotInTheApartment) -> Segmind:
    """
    :return: SegMind watching the milk be taken hold of and supported.
    """
    return Segmind.create_for_bodies(
        scene.world, [scene.milk], [GraspDetector, SupportDetector]
    )


def test_a_pick_up_holds_when_segmind_sees_the_object_taken_hold_of_and_lifted(
    milk_on_the_counter,
):
    scene = milk_on_the_counter
    beside = scene.milk.global_pose.position.to_np()[:3] + np.array([0, -0.15, 0])

    verdict = verdict_on(
        judged(pick_and_place_plan(scene, beside), watching_the_milk(scene)),
        PickUpAction,
    )

    assert (verdict.holds, verdict.judged_by) == (True, JudgedBy.SEGMIND)
    assert {type(event) for event in verdict.evidence} == {
        GraspEvent,
        LossOfSupportEvent,
    }


def test_a_placing_holds_when_segmind_sees_the_object_let_go_of_and_supported(
    milk_on_the_counter,
):
    scene = milk_on_the_counter
    beside = scene.milk.global_pose.position.to_np()[:3] + np.array([0, -0.15, 0])

    verdict = verdict_on(
        judged(pick_and_place_plan(scene, beside), watching_the_milk(scene)),
        PlaceAction,
    )

    assert (verdict.holds, verdict.judged_by) == (True, JudgedBy.SEGMIND)
    assert {type(event) for event in verdict.evidence} == {
        LossOfGraspEvent,
        SupportEvent,
    }


def test_a_placing_in_mid_air_does_not_hold(milk_on_the_counter):
    scene = milk_on_the_counter
    above = scene.milk.global_pose.position.to_np()[:3] + np.array([0, -0.15, 0.15])

    verdict = verdict_on(
        judged(pick_and_place_plan(scene, above), watching_the_milk(scene)),
        PlaceAction,
    )

    assert (verdict.holds, verdict.judged_by) == (False, JudgedBy.SEGMIND)


def test_an_action_segmind_has_no_expectation_of_is_judged_by_the_world(
    milk_on_the_counter,
):
    scene = milk_on_the_counter
    beside = scene.milk.global_pose.position.to_np()[:3] + np.array([0, -0.15, 0])

    verdict = verdict_on(
        judged(pick_and_place_plan(scene, beside), watching_the_milk(scene)),
        ParkArmsAction,
    )

    assert (verdict.holds, verdict.judged_by) == (True, JudgedBy.WORLD)


def test_an_action_on_an_object_segmind_does_not_watch_is_judged_by_the_world(
    milk_on_the_counter,
):
    scene = milk_on_the_counter
    beside = scene.milk.global_pose.position.to_np()[:3] + np.array([0, -0.15, 0])
    watching_the_spoon = Segmind.create_for_bodies(
        scene.world,
        [scene.world.get_body_by_name("spoon.stl")],
        [GraspDetector, SupportDetector],
    )

    verdict = verdict_on(
        judged(pick_and_place_plan(scene, beside), watching_the_spoon),
        PickUpAction,
    )

    assert verdict.judged_by == JudgedBy.NOT_WATCHED


def test_a_verdict_of_an_action_that_succeeded_does_not_mark_it_failed(
    milk_on_the_counter,
):
    scene = milk_on_the_counter
    beside = scene.milk.global_pose.position.to_np()[:3] + np.array([0, -0.15, 0])

    verdicts = judged(pick_and_place_plan(scene, beside), watching_the_milk(scene))

    assert [verdict.action_failed for verdict in verdicts] == [False] * len(verdicts)
