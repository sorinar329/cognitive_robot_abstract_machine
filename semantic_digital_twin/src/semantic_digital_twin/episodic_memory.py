"""
Narrative-enabled episodic memories (NEEMs): what happened in a world over time, recorded
so that it can be replayed, stored and annotated afterwards.
"""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, StrEnum
from pathlib import Path
from time import monotonic
from types import TracebackType
from uuid import UUID, uuid4

from typing_extensions import Self

import semantic_digital_twin
from krrood.ormatic.utils import classproperty
from physics_simulators.base_simulator import BaseSimulator
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_modification import (
    WorldModelModificationBlock,
)
from semantic_digital_twin.world_description.world_state import WorldStateTrajectory

# %% time


class TimeSource(StrEnum):
    """
    What the times of a recording were read from.
    """

    SIMULATION = "simulation"
    """
    The time of the simulation that moved the world.
    """

    WALL_CLOCK = "wall_clock"
    """
    The time that passed in the real world.
    """


@dataclass
class Clock(ABC):
    """
    Tells a recording what time it is.
    """

    @classproperty
    @abstractmethod
    def time_source(cls) -> TimeSource:
        """
        :return: What the clock reads its time from.
        """

    @abstractmethod
    def now(self) -> float:
        """
        :return: The current time in seconds, from an arbitrary but fixed origin.
        """


@dataclass
class WallClock(Clock):
    """
    The time that passes in the real world, which never jumps back.
    """

    @classproperty
    def time_source(cls) -> TimeSource:
        return TimeSource.WALL_CLOCK

    def now(self) -> float:
        return monotonic()


@dataclass
class SimulationClock(Clock):
    """
    The time of a simulation, which passes only while the simulation steps.
    """

    simulator: BaseSimulator
    """
    The simulator whose time is told.
    """

    @classproperty
    def time_source(cls) -> TimeSource:
        return TimeSource.SIMULATION

    def now(self) -> float:
        return self.simulator.current_simulation_time


# %% where a recording comes from


class NEEMSource(StrEnum):
    """
    What produced the behaviour a NEEM records.
    """

    KINEMATIC_SIMULATION = "kinematic_simulation"
    """
    A robot in a world whose bodies move only where they are put.
    """

    PHYSICS_SIMULATION = "physics_simulation"
    """
    A robot in a world moved by simulated physics.
    """

    REAL_ROBOT = "real_robot"
    """
    A robot in the real world.
    """

    VIRTUAL_REALITY = "virtual_reality"
    """
    A person acting in virtual reality.
    """

    HUMAN_DEMONSTRATION = "human_demonstration"
    """
    A person acting in the real world.
    """


class GitQuery(tuple, Enum):
    """
    A question git answers about the checkout a directory lies in.
    """

    REPOSITORY_ROOT = ("rev-parse", "--show-toplevel")
    """
    The top directory of the checkout.
    """

    COMMIT = ("rev-parse", "HEAD")
    """
    The commit the checkout is at.
    """

    UNCOMMITTED_CHANGES = ("status", "--porcelain")
    """
    The files that differ from that commit, one per line.
    """

    def answer_in(self, directory: Path) -> str | None:
        """
        :param directory: A directory inside the checkout.
        :return: Git's answer, or None if git is not installed or the directory is in
            no checkout.
        """
        if shutil.which("git") is None:
            return None
        result = subprocess.run(
            ["git", *self.value], cwd=directory, capture_output=True, text=True
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()


class WorkspaceFile(StrEnum):
    """
    A file at the top of the checkout that describes the code.
    """

    LOCK_FILE = "uv.lock"
    """
    The lock file that pins every dependency.
    """


@dataclass
class WorkspaceVersion:
    """
    The version of the code a recording was made with.
    """

    commit: str | None
    """
    The commit of the checkout the code ran from, or None if it ran from no checkout.
    """

    has_uncommitted_changes: bool
    """
    Whether the checkout differed from :attr:`commit`.
    """

    lock_file_content: str | None
    """
    The lock file that pinned every dependency, or None if the checkout has none.
    """

    @classmethod
    def of_installed_code(cls) -> Self:
        """
        :return: The version of the checkout this package is installed from.
        """
        package_directory = Path(semantic_digital_twin.__file__).parent
        repository_root = GitQuery.REPOSITORY_ROOT.answer_in(package_directory)
        if repository_root is None:
            return cls(
                commit=None, has_uncommitted_changes=False, lock_file_content=None
            )
        lock_file = Path(repository_root) / WorkspaceFile.LOCK_FILE
        return cls(
            commit=GitQuery.COMMIT.answer_in(package_directory),
            has_uncommitted_changes=bool(
                GitQuery.UNCOMMITTED_CHANGES.answer_in(package_directory)
            ),
            lock_file_content=lock_file.read_text() if lock_file.exists() else None,
        )


@dataclass
class RecordingMetadata:
    """
    What a NEEM records, who made it, when and with which code.
    """

    creators: list[str]
    """
    Who made the recording.
    """

    created_at: datetime
    """
    When the recording started, in UTC.
    """

    source: NEEMSource
    """
    What produced the recorded behaviour.
    """

    description: str
    """
    What the recording shows, in words.
    """

    time_source: TimeSource
    """
    What the times of the recording were read from.
    """

    workspace: WorkspaceVersion
    """
    The version of the code the recording was made with.
    """

    thumbnail_png: bytes | None = None
    """
    A picture of the recording as a PNG image, if one was taken.
    """


# %% the recording


@dataclass
class ModelVersionRecording:
    """
    The recording of the time the world kept one model.
    """

    history_length: int
    """
    How many blocks of the world's modification history made up the model.
    """

    trajectory: WorldStateTrajectory
    """
    The states the world went through meanwhile, timed in seconds since the recording
    started.
    """


@dataclass
class NEEM:
    """
    A narrative-enabled episodic memory: the states and models a world went through
    while something happened in it.
    """

    model_versions: list[ModelVersionRecording]
    """
    The recording of each model the world had, in the order it had them.
    """

    world_history: list[WorldModelModificationBlock]
    """
    The modification history the world had when the recording ended, oldest block
    first.
    """

    recording_metadata: RecordingMetadata
    """
    What the recording is and how it was made.
    """

    id: UUID = field(default_factory=uuid4)
    """
    The identifier of the NEEM.
    """

    def world_at(self, time: float) -> World:
        """
        :param time: Seconds since the recording started.
        :return: A new world with the model the recorded world had at ``time`` and the
            state of the last frame taken up to then.
        :raises NoFrameRecordedAt: If ``time`` lies before the recording started.
        """
        model_version = next(
            (
                model_version
                for model_version in reversed(self.model_versions)
                if model_version.trajectory.times[0] <= time
            ),
            self.model_versions[0],
        )
        state = model_version.trajectory.state_at(time)
        world = World()
        with world.modify_world():
            for block in self.world_history[: model_version.history_length]:
                block.update_references_for_world_and_apply(world=world)
            world.state.merge_state(state)
        return world


# %% recording


@dataclass
class NEEMRecorder:
    """
    Records a NEEM of a world while it is used as a context manager.

    Every time in the NEEM is read from :attr:`clock`, in seconds since the recording
    started.
    """

    world: World
    """
    The world to record.
    """

    clock: Clock
    """
    What the times of the recording are read from.
    """

    source: NEEMSource
    """
    What produces the behaviour to record.
    """

    creators: list[str]
    """
    Who makes the recording.
    """

    description: str = ""
    """
    What the recording shows, in words.
    """

    sample_period: float = 0.1
    """
    The least time between two recorded states of one model, in seconds.
    """

    neem: NEEM = field(init=False)
    """
    The recording, complete once the context is left.
    """

    _started_at: float = field(init=False)
    """
    The time of :attr:`clock` the recording started at.
    """

    _callbacks: list[StateSampler | ModelChangeLog] = field(
        init=False, default_factory=list
    )
    """
    What tells the recorder about changes of the world while it records.
    """

    def __enter__(self) -> Self:
        self._started_at = self.clock.now()
        self.neem = NEEM(
            model_versions=[],
            world_history=[],
            recording_metadata=RecordingMetadata(
                creators=self.creators,
                created_at=datetime.now(timezone.utc),
                source=self.source,
                description=self.description,
                time_source=self.clock.time_source,
                workspace=WorkspaceVersion.of_installed_code(),
            ),
        )
        self.record_model_change()
        self._callbacks = [
            StateSampler(_world=self.world, recorder=self),
            ModelChangeLog(_world=self.world, recorder=self),
        ]
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for callback in self._callbacks:
            callback.stop()
        self.take_frame(least_time_since_last_frame=0.0)
        self.neem.world_history = list(self.world.modification_history)

    def take_frame(self, least_time_since_last_frame: float) -> None:
        """
        Record the world's state, unless the world is being modified or the last frame
        is too recent.

        :param least_time_since_last_frame: How much time must have passed since the
            last frame for this one to be taken.
        """
        if self.world.world_is_being_modified:
            return
        time = self.clock.now() - self._started_at
        trajectory = self.neem.model_versions[-1].trajectory
        last_time = float(trajectory.times[-1])
        if time <= last_time or time - last_time < least_time_since_last_frame:
            return
        trajectory.append(self.world.state, time)

    def record_model_change(self) -> None:
        """
        Start recording the world's current model with its current state.
        """
        time = self.clock.now() - self._started_at
        self.neem.model_versions.append(
            ModelVersionRecording(
                history_length=len(self.world.modification_history),
                trajectory=WorldStateTrajectory.from_world_state(
                    self.world.state, time
                ),
            )
        )


@dataclass(eq=False)
class StateSampler(StateChangeCallback):
    """
    Has a recorder take a frame whenever the world's state changes, at most once per
    sample period.
    """

    recorder: NEEMRecorder = field(kw_only=True)
    """
    The recorder to take the frames.
    """

    def on_state_change(self, **kwargs) -> None:
        self.recorder.take_frame(
            least_time_since_last_frame=self.recorder.sample_period
        )


@dataclass(eq=False)
class ModelChangeLog(ModelChangeCallback):
    """
    Has a recorder start recording the new model whenever the world's model changes.
    """

    recorder: NEEMRecorder = field(kw_only=True)
    """
    The recorder to record the new model.
    """

    def on_model_change(self, **kwargs) -> None:
        self.recorder.record_model_change()
