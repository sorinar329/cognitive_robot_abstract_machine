"""
Tests for the REAL/SIMULATED branch of ``GiskardExecutable.motion_state_chart`` (see
``coraplex/src/coraplex/plans/executables.py``).

On the real robot, tasks are wrapped in a single ``Sequence`` + ``EndMotion``; in
simulation, tasks are added individually and get pause/interrupt monitors and pre-/post-
condition monitors wired in.
"""

import pytest

from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import CancelMotion, EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import (
    ThreadedPredicateMonitor,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment, ExecutionType
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import real_robot, simulated_robot, ExecutionEnvironment
from coraplex.plans.condition_nodes import PlanNodeStatusMonitor
from coraplex.plans.executables import ModelChangeExecutable
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.pick_up import ReachAction


@pytest.fixture
def reach_action_executable(immutable_model_world):
    """
    A real, 2-motion ``GiskardExecutable`` with pre-/post-conditions, built the same way
    ``test_merge_motions`` in ``test_graph_parsing.py`` does.
    """
    world, view, context = immutable_model_world
    world.get_body_by_name("milk.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(2, 1.5, 0.7, 0, 0, 0)
    )
    plan = execute_single(
        ReachAction(
            Pose.from_xyz_rpy(2, 1.5, 0.7, reference_frame=world.root),
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                view.right_arm.end_effector,
            ),
            world.get_body_by_name("milk.stl"),
        ),
        context=context,
    )
    plan.notify()
    return plan.parse()


def test_motion_state_chart_simulated_execution_adds_tasks_directly(
    reach_action_executable,
):
    tasks = list(reach_action_executable.motion_mappings.values())

    with simulated_robot:
        chart = reach_action_executable.motion_state_chart

    assert chart.get_nodes_by_type(Sequence) == []
    for task in tasks:
        assert task in chart.nodes


def test_motion_state_chart_real_execution_wraps_tasks_in_sequence(
    reach_action_executable,
):
    tasks = list(reach_action_executable.motion_mappings.values())

    with real_robot:
        chart = reach_action_executable.motion_state_chart

    sequences = chart.get_nodes_by_type(Sequence)
    assert len(sequences) == 1
    assert sequences[0].nodes == tasks
    assert len(chart.get_nodes_by_type(EndMotion)) == 1
    # simulation-only machinery must not be present on the real-robot path
    for task in tasks:
        assert task not in chart.nodes


def test_motion_state_chart_simulated_execution_adds_condition_and_pause_interrupt_monitors(
    reach_action_executable,
):
    task_count = len(reach_action_executable.motion_mappings)
    assert reach_action_executable.pre_condition_node
    assert reach_action_executable.post_condition_node

    with simulated_robot:
        chart = reach_action_executable.motion_state_chart

    # one pause + one interrupt monitor per task
    assert len(chart.get_nodes_by_type(PlanNodeStatusMonitor)) == 2 * task_count
    # pre- and post-condition monitors
    assert len(chart.get_nodes_by_type(ThreadedPredicateMonitor)) == 2
    # abort paths for pre- and post-condition failing
    assert len(chart.get_nodes_by_type(CancelMotion)) == 2


def test_model_change_executable_reparents_the_body(mutable_model_world):
    """
    ModelChangeExecutable.execute() must re-parent the body to its new parent
    regardless of execution type: nothing else in coraplex represents "what
    is currently held by the gripper" other than this kinematic re-parenting
    (e.g. PlaceAction's pre-condition checks it via is_gripper_holding_something/
    GripperIsFree), including in purely-kinematic SIMULATED runs with no
    physics backend attached at all -- so this must not be skipped for
    either execution type.
    """
    world, view, context = mutable_model_world
    body = world.get_body_by_name("milk.stl")
    original_parent_connection = body.parent_connection

    executable = ModelChangeExecutable(context=context, body=body, new_parent=world.root)

    with simulated_robot:
        executable.execute()

    assert body.parent_connection is not original_parent_connection
    assert body.parent_connection.parent is world.root


def _capture_pacer_real_time_factors(monkeypatch) -> list:
    """
    Patches coraplex.plans.executables.SimulationPacer to record the
    real_time_factor it is constructed with on every call, while still
    constructing a real, functional pacer.
    """
    import coraplex.plans.executables as executables_module

    captured = []
    original_pacer_cls = executables_module.SimulationPacer

    def capturing_pacer(*args, real_time_factor=None, **kwargs):
        captured.append(real_time_factor)
        return original_pacer_cls(*args, real_time_factor=real_time_factor, **kwargs)

    monkeypatch.setattr(executables_module, "SimulationPacer", capturing_pacer)
    return captured


def test_real_time_pacing_defaults_to_an_unpaced_pacer(
    reach_action_executable, monkeypatch
):
    """
    GiskardExecutable.real_time_pacing defaults to False, so
    _execute_simulation's Ros2Executor must be constructed with
    SimulationPacer(real_time_factor=None) (never sleeps, see
    SimulationPacer.sleep) -- this must stay true for every existing caller
    (dozens of plan.perform() calls across the test suite) that never
    touches the flag, so they stay exactly as fast/unpaced as they are today.
    """
    captured = _capture_pacer_real_time_factors(monkeypatch)

    with simulated_robot:
        reach_action_executable.execute()

    assert captured == [None], (
        f"real_time_pacing=False (the default) should construct the pacer "
        f"with real_time_factor=None (never sleeps), got {captured}."
    )


def test_real_time_pacing_enabled_wires_a_real_time_factor(
    reach_action_executable, monkeypatch
):
    """
    Enabling ExecutionEnvironment(real_time_pacing=True) must construct
    _execute_simulation's Ros2Executor with SimulationPacer(real_time_factor=1.0),
    pacing the tick loop towards the QP controller's own target_frequency
    (see QPControllerConfig(target_frequency=50) in _execute_simulation) --
    otherwise a physically-simulated DOF's actual position could race ahead
    of Giskard's belief of where it is, regardless of sim<->world sync rate.
    """
    captured = _capture_pacer_real_time_factors(monkeypatch)

    with ExecutionEnvironment(
        execution_type=ExecutionType.SIMULATED, real_time_pacing=True
    ):
        reach_action_executable.execute()

    assert captured == [1.0], (
        f"real_time_pacing=True should construct the pacer with "
        f"real_time_factor=1.0 (paces towards target_frequency), got {captured}."
    )
