from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.episodic_memory import (
    NEEM,
    Clock,
    NEEMRecorder,
    NEEMSource,
    SimulationClock,
    TimeSource,
    WorkspaceVersion,
)
from semantic_digital_twin.exceptions import NoFrameRecordedAt
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_state import WorldStateTrajectory

# %% a world to record


@dataclass
class HandSetClock(Clock):
    """
    A clock that shows whatever time a test sets it to.
    """

    seconds: float = 0.0
    """
    The time the clock shows.
    """

    @classproperty
    def time_source(cls) -> TimeSource:
        return TimeSource.WALL_CLOCK

    def now(self) -> float:
        return self.seconds


@dataclass
class SlidingBoxWorld:
    """
    A world of a box that slides along one axis.
    """

    world: World = field(init=False)
    """
    The world holding the box.
    """

    slide: PrismaticConnection = field(init=False)
    """
    The connection the box slides along.
    """

    def __post_init__(self):
        self.world = World()
        with self.world.modify_world():
            root = Body(name=PrefixedName("root"))
            self.world.add_kinematic_structure_entity(root)
            box = Body(name=PrefixedName("box"))
            self.slide = PrismaticConnection.create_with_dofs(
                world=self.world,
                parent=root,
                child=box,
                axis=Vector3.X(reference_frame=root),
            )
            self.world.add_connection(self.slide)

    def add_lid(self) -> Body:
        """
        :return: A lid, newly fixed onto the box.
        """
        lid = Body(name=PrefixedName("lid"))
        with self.world.modify_world():
            self.world.add_connection(
                FixedConnection(parent=self.slide.child, child=lid)
            )
        return lid


@pytest.fixture
def sliding_box() -> SlidingBoxWorld:
    return SlidingBoxWorld()


def recorder_of(world: World, clock: Clock) -> NEEMRecorder:
    """
    :return: A recorder of the world that samples every tenth of a second.
    """
    return NEEMRecorder(
        world=world,
        clock=clock,
        source=NEEMSource.KINEMATIC_SIMULATION,
        creators=["a test"],
        sample_period=0.1,
    )


def recorded_times(neem: NEEM) -> list[float]:
    """
    :return: The time of every frame of the NEEM, in order.
    """
    return [
        time
        for model_version in neem.model_versions
        for time in model_version.trajectory.times.tolist()
    ]


# %% sampling the state


def test_frames_are_timed_in_seconds_since_the_recording_started(sliding_box):
    clock = HandSetClock(seconds=10.0)
    recorder = recorder_of(sliding_box.world, clock)

    with recorder:
        clock.seconds = 10.2
        sliding_box.slide.position = 0.2
        clock.seconds = 10.3

    assert recorded_times(recorder.neem) == pytest.approx([0.0, 0.2, 0.3])


def test_a_state_closer_to_the_last_frame_than_the_sample_period_is_not_recorded(
    sliding_box,
):
    clock = HandSetClock()
    recorder = recorder_of(sliding_box.world, clock)

    with recorder:
        clock.seconds = 0.05
        sliding_box.slide.position = 0.05
        clock.seconds = 0.15
        sliding_box.slide.position = 0.15

    assert recorded_times(recorder.neem) == [0.0, 0.15]


def test_the_recording_ends_with_the_state_the_world_is_left_in(sliding_box):
    clock = HandSetClock()
    recorder = recorder_of(sliding_box.world, clock)

    with recorder:
        clock.seconds = 0.15
        sliding_box.slide.position = 0.15
        clock.seconds = 0.2
        sliding_box.slide.position = 0.2

    end = recorder.neem.world_at(0.2)
    assert end.state[sliding_box.slide.raw_dof.id].position == 0.2


# %% changes of the model


def test_a_model_change_starts_the_recording_of_a_new_model_version(sliding_box):
    clock = HandSetClock()
    recorder = recorder_of(sliding_box.world, clock)

    with recorder:
        clock.seconds = 0.5
        sliding_box.add_lid()
        clock.seconds = 0.7

    first, second = recorder.neem.model_versions
    assert first.trajectory.times.tolist() == [0.0]
    assert second.trajectory.times.tolist() == [0.5, 0.7]
    assert second.history_length == len(sliding_box.world.modification_history)


def test_the_world_at_a_time_has_the_model_of_that_time(sliding_box):
    clock = HandSetClock()
    recorder = recorder_of(sliding_box.world, clock)

    with recorder:
        clock.seconds = 0.5
        lid = sliding_box.add_lid()
        clock.seconds = 0.7

    before = recorder.neem.world_at(0.4)
    after = recorder.neem.world_at(0.6)
    assert lid.name not in [body.name for body in before.bodies]
    assert lid.name in [body.name for body in after.bodies]


def test_the_world_at_a_time_has_the_state_of_the_last_frame_up_to_then(
    sliding_box,
):
    clock = HandSetClock()
    recorder = recorder_of(sliding_box.world, clock)

    with recorder:
        clock.seconds = 0.2
        sliding_box.slide.position = 0.2
        clock.seconds = 0.4
        sliding_box.slide.position = 0.4

    world = recorder.neem.world_at(0.3)
    assert world.state[sliding_box.slide.raw_dof.id].position == 0.2


def test_there_is_no_world_before_the_recording_started(sliding_box):
    recorder = recorder_of(sliding_box.world, HandSetClock())

    with recorder:
        pass

    with pytest.raises(NoFrameRecordedAt):
        recorder.neem.world_at(-1.0)


# %% metadata


def test_the_metadata_says_which_clock_timed_the_recording(sliding_box):
    clock = HandSetClock()
    recorder = recorder_of(sliding_box.world, clock)

    with recorder:
        pass

    assert recorder.neem.recording_metadata.time_source == clock.time_source
    assert recorder.neem.recording_metadata.created_at.tzinfo == timezone.utc


def test_the_workspace_version_is_the_commit_the_code_runs_from():
    repository = Path(__file__).parents[2]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert WorkspaceVersion.of_installed_code().commit == head


# %% the database


def through_the_database[Stored](stored: Stored, session_maker: sessionmaker) -> Stored:
    """
    :return: What a new session reads back after ``stored`` was written in another one.
    """
    data_access_object = to_dao(stored)
    with session_maker() as session:
        session.add(data_access_object)
        session.commit()
    with session_maker() as session:
        return session.scalar(select(type(data_access_object))).from_dao()


def test_a_world_state_trajectory_keeps_its_data_in_the_database(
    sliding_box, in_memory_session_maker
):
    trajectory = WorldStateTrajectory.from_world_state(sliding_box.world.state, 0.0)
    sliding_box.slide.position = 0.3
    trajectory.append(sliding_box.world.state, 1.0)

    restored = through_the_database(trajectory, in_memory_session_maker)

    assert restored.times.tolist() == trajectory.times.tolist()
    np.testing.assert_array_equal(restored.data, trajectory.data)


def test_a_neem_keeps_its_recording_in_the_database(
    sliding_box, in_memory_session_maker
):
    clock = HandSetClock()
    recorder = recorder_of(sliding_box.world, clock)
    with recorder:
        clock.seconds = 0.2
        sliding_box.slide.position = 0.2
        clock.seconds = 0.5
        lid = sliding_box.add_lid()
        clock.seconds = 0.7
    neem = recorder.neem

    restored = through_the_database(neem, in_memory_session_maker)

    assert recorded_times(restored) == recorded_times(neem)
    assert [
        model_version.history_length for model_version in restored.model_versions
    ] == [model_version.history_length for model_version in neem.model_versions]
    assert lid.name in [body.name for body in restored.world_at(0.6).bodies]
    assert restored.recording_metadata.workspace == neem.recording_metadata.workspace
    assert (
        restored.recording_metadata.created_at.replace(tzinfo=timezone.utc)
        == neem.recording_metadata.created_at
    )


# %% a recording timed by a simulation


@pytest.fixture
def falling_box_world() -> World:
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_kinematic_structure_entity(root)
        box = Body(name=PrefixedName("box"))
        geometry = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        reference_frame=box
                    ),
                    scale=Scale(0.1, 0.1, 0.1),
                )
            ],
            reference_frame=box,
        )
        box.collision, box.visual = geometry, geometry
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=1.0, reference_frame=root
                ),
            )
        )
    return world


@pytest.mark.skipif(
    os.environ.get("CI", "false").lower() != "true",
    reason="Simulates physics, which runs in CI only.",
)
def test_a_recording_of_a_simulation_is_timed_in_simulated_seconds(
    falling_box_world,
):
    simulation = MujocoSim(world=falling_box_world, headless=True)
    simulation.start_stepped_simulation()
    try:
        recorder = recorder_of(
            falling_box_world, SimulationClock(simulator=simulation.simulator)
        )
        with recorder:
            simulation.step_simulation(timedelta(milliseconds=500))
    finally:
        simulation.stop_simulation()

    times = recorded_times(recorder.neem)
    assert recorder.neem.recording_metadata.time_source == TimeSource.SIMULATION
    assert times[-1] == pytest.approx(0.5, abs=simulation.simulator.step_size)
    assert np.diff(times[:-1]).min() >= 0.1
