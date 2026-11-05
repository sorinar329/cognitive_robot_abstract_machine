"""
SegMind watching a run from a process of its own, on a copy of the run's world kept in
step with the states and model changes the run publishes, so neither process waits for
the other while the run goes on.
"""

from __future__ import annotations

import enum
import multiprocessing
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event
from uuid import UUID

import numpy as np
from krrood.adapters.json_serializer import from_json, to_json
from typing_extensions import Any, Self, Sequence

from segmind.datastructures.events import DetectionEvent
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from segmind.detector_selection import DetectorSelection
from segmind.detectors.base import AbstractDetector
from segmind.event_logger import EventLogger
from segmind.event_segmentation import RunSegmentation, Segmind
from segmind.exceptions import SegmindProcessDidNotStart, SegmindProcessFailed
from segmind.shared_world_state import SharedWorldState
from semantic_digital_twin.adapters.ros.messages import WorldModelSnapshot
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


# %% what passes between the two processes


@dataclass
class PublishedState:
    """
    Where the state of the run's world is published, and how it is laid out.
    """

    memory_name: str
    """
    The name of the shared memory the state is kept in.
    """

    degree_of_freedom_ids: list[UUID]
    """
    The degree of freedom of each column of the state, in order.
    """

    shape: tuple[int, int]
    """
    The shape of the state: derivatives by degrees of freedom.
    """

    def columns_in(self, world: World) -> np.ndarray:
        """
        :param world: A copy of the run's world.
        :return: For each degree of freedom of ``world``, in its order, the column of
            the published state holding it.
        """
        column_of = {
            degree_of_freedom_id: column
            for column, degree_of_freedom_id in enumerate(self.degree_of_freedom_ids)
        }
        return np.array([column_of[identifier] for identifier in world.state._ids])


@dataclass
class ModelChange:
    """
    A change of the run's world model, with where the state is published from then on.
    """

    modification_block: dict[str, Any]
    """
    The modifications the world model went through, as JSON.
    """

    state: PublishedState
    """
    Where the state is published once the model has changed.
    """


@dataclass
class WatchRequest:
    """
    Everything SegMind's process needs to start watching a run.
    """

    world_snapshot: dict[str, Any]
    """
    The run's world as the modifications that built it, as JSON.
    """

    state: PublishedState
    """
    Where the state of the run's world is published.
    """

    watched_body_ids: list[UUID]
    """
    The bodies the detectors watch.
    """

    detector_selection: DetectorSelection
    """
    The kinds of detector ticked.
    """

    tick_period: float
    """
    The time, in seconds, from the start of one tick to the start of the next.
    """


class WatchingProgress(enum.StrEnum):
    """
    What SegMind's process tells the run about its watching, besides the events it
    detects.
    """

    STARTED = "started"
    """
    The process is watching the run.
    """

    ENDED = "ended"
    """
    The process has stopped watching, and sends nothing more.
    """


# %% the run's side


@dataclass(eq=False)
class WorldStatePublisher(StateChangeCallback, ModelChangeCallback):
    """
    Publishes every state of a world to shared memory, and every change of its model,
    with the memory the state is published to from then on, through a pipe.
    """

    changes: Connection = field(kw_only=True)
    """
    The pipe the model changes are sent through.
    """

    shared_state: SharedWorldState = field(init=False)
    """
    The memory the current state is published to.
    """

    replaced_states: list[SharedWorldState] = field(init=False, default_factory=list)
    """
    The memory the state was published to before each model change.
    """

    writing: threading.Lock = field(init=False, default_factory=threading.Lock)
    """
    Held while the state is written, since the world may change from several threads.
    """

    def __post_init__(self) -> None:
        """
        Register with the world for both kinds of change, and publish the current state.
        """
        # Both bases register the callback, and neither calls the other.
        StateChangeCallback.__post_init__(self)
        ModelChangeCallback.__post_init__(self)
        self.shared_state = self._publish_to_new_memory()

    @property
    def published_state(self) -> PublishedState:
        """
        Where the state is published now, and how it is laid out.
        """
        return PublishedState(
            memory_name=self.shared_state.name,
            degree_of_freedom_ids=list(self._world.state._ids),
            shape=self.shared_state.shape,
        )

    def on_state_change(self, **kwargs) -> None:
        # A state of another shape waits for its model change to publish it.
        if self._world.state._data.shape != self.shared_state.shape:
            return
        with self.writing:
            self.shared_state.write(self._world.state._data)

    def on_model_change(self, **kwargs) -> None:
        with self.writing:
            replaced = self.shared_state
            self.shared_state = self._publish_to_new_memory()
            # Marked before sending: a reader waits for the mark before it reads the pipe.
            replaced.mark_replaced()
            self.replaced_states.append(replaced)
            self.changes.send(
                ModelChange(
                    modification_block=to_json(
                        self._world.get_world_model_manager().model_modification_blocks[
                            -1
                        ]
                    ),
                    state=self.published_state,
                )
            )

    def release(self) -> None:
        """
        Stop publishing and free every memory published to.
        """
        self.stop()
        for state in [*self.replaced_states, self.shared_state]:
            state.close()
            state.unlink()

    def _publish_to_new_memory(self) -> SharedWorldState:
        """
        :return: New memory, holding the current state.
        """
        state = SharedWorldState.create(self._world.state._data.shape)
        state.write(self._world.state._data)
        return state


@dataclass(eq=False)
class SegmindProcess(RunSegmentation):
    """
    SegMind watching a run from a process of its own, so the run never waits for the
    watching. Starting raises :class:`SegmindProcessDidNotStart` or
    :class:`SegmindProcessFailed` if the process never watches, and stopping raises the
    latter if it failed.
    """

    bodies: list[Body]
    """
    The bodies the detectors watch.
    """

    detector_selection: DetectorSelection
    """
    The kinds of detector ticked.
    """

    tick_period: float = field(kw_only=True, default=0.01)
    """
    The time, in seconds, from the start of one tick to the start of the next.
    """

    startup_timeout: float = field(kw_only=True, default=120.0)
    """
    How long, in seconds, the run waits for the process to start watching.
    """

    detected_events: EventLogger = field(init=False, default_factory=EventLogger)
    """
    Every event the process detected.
    """

    tracker_registry: ObjectTrackerFactory = field(
        init=False, default_factory=ObjectTrackerFactory
    )
    """
    The trackers of the bodies the events are about.
    """

    publisher: WorldStatePublisher | None = field(init=False, default=None)
    """
    Publishes the run's world to the process, while it watches.
    """

    process: BaseProcess | None = field(init=False, default=None)
    """
    The process watching, once started.
    """

    events: Connection | None = field(init=False, default=None)
    """
    The pipe the process sends what it detected through.
    """

    stopping: Event | None = field(init=False, default=None)
    """
    Set when the process is to stop watching.
    """

    receiver: threading.Thread | None = field(init=False, default=None)
    """
    Logs the events the process sends, while it watches.
    """

    @classmethod
    def create_for_bodies(
        cls,
        world: World,
        bodies: Sequence[Body],
        detector_types: Sequence[type[AbstractDetector]] = (),
    ) -> Self:
        return cls(
            world=world,
            bodies=list(bodies),
            detector_selection=DetectorSelection.for_run(detector_types),
        )

    @property
    def watched_bodies(self) -> list[Body]:
        return list(dict.fromkeys(self.bodies))

    @property
    def detector_names(self) -> list[str]:
        return [
            detector_type.__name__
            for detector_type in self.detector_selection.detector_types
        ]

    @property
    def event_logger(self) -> EventLogger:
        return self.detected_events

    def start(self) -> None:
        context = multiprocessing.get_context("spawn")
        changes_received, changes_sent = context.Pipe(duplex=False)
        self.events, events_sent = context.Pipe(duplex=False)
        self.stopping = context.Event()
        self.publisher = WorldStatePublisher(_world=self.world, changes=changes_sent)
        self.process = context.Process(
            target=MirroringSegmind.watch,
            args=(self._watch_request(), changes_received, events_sent, self.stopping),
            daemon=True,
        )
        self.process.start()
        changes_received.close()
        events_sent.close()
        self._wait_until_watching()
        self.receiver = threading.Thread(
            target=self._log_events_until_watching_ended, daemon=True
        )
        self.receiver.start()

    def stop(self) -> None:
        self.stopping.set()
        self.receiver.join()
        self.process.join()
        self.publisher.release()
        self.events.close()
        if self.process.exitcode != 0:
            raise SegmindProcessFailed(self.process.exitcode)

    def _watch_request(self) -> WatchRequest:
        """
        :return: What the process needs to watch the run, as it stands now.
        """
        snapshot = WorldModelSnapshot(
            modifications=list(
                self.world.get_world_model_manager().model_modification_blocks
            ),
            ids=list(self.world.state.keys()),
            states=list(self.world.state.positions),
        )
        return WatchRequest(
            world_snapshot=snapshot.to_json(),
            state=self.publisher.published_state,
            watched_body_ids=[body.id for body in self.bodies],
            detector_selection=self.detector_selection,
            tick_period=self.tick_period,
        )

    def _wait_until_watching(self) -> None:
        """
        Wait until the process says it is watching, logging what it detects in the
        world as it stood, and stop publishing if it never will watch.
        """
        deadline = time.monotonic() + self.startup_timeout
        while wait([self.events], timeout=max(deadline - time.monotonic(), 0.0)):
            message = self.events.recv()
            if message is WatchingProgress.STARTED:
                return
            if message is WatchingProgress.ENDED:
                self.process.join()
                self.publisher.release()
                raise SegmindProcessFailed(self.process.exitcode)
            self._log_event(message)
        self.process.terminate()
        self.publisher.release()
        raise SegmindProcessDidNotStart(self.startup_timeout)

    def _log_events_until_watching_ended(self) -> None:
        """
        Log every event the process sends until it says it has stopped watching.
        """
        while (message := self.events.recv()) is not WatchingProgress.ENDED:
            self._log_event(message)

    def _log_event(self, event_json: dict[str, Any]) -> None:
        """
        Log an event the process sent, naming the bodies of :attr:`world`.
        """
        self.detected_events.log_event(
            from_json(
                event_json,
                **WorldEntityWithIDKwargsTracker.from_world(self.world).create_kwargs(),
            ),
            self.tracker_registry,
        )


# %% SegMind's side


@dataclass(eq=False)
class MirroredWorld:
    """
    A copy of the run's world, kept in step with the state and the model changes the
    run publishes.
    """

    world: World
    """
    The copy.
    """

    changes: Connection
    """
    The pipe the run sends its model changes through.
    """

    shared_state: SharedWorldState
    """
    The memory the run publishes its state to.
    """

    columns: np.ndarray
    """
    For each degree of freedom of :attr:`world`, the column of the published state
    holding it.
    """

    @classmethod
    def from_request(cls, request: WatchRequest, changes: Connection) -> Self:
        """
        :param request: What the run sent to have itself watched.
        :param changes: The pipe the run sends its model changes through.
        :return: A copy of the run's world, as the run stood when it asked.
        """
        world = World()
        WorldModelSnapshot.apply_to_json_snapshot_to_world(
            world,
            request.world_snapshot,
            **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
        )
        return cls(
            world=world,
            changes=changes,
            shared_state=SharedWorldState.attach(
                request.state.memory_name, request.state.shape
            ),
            columns=request.state.columns_in(world),
        )

    def copy_latest_state(self) -> None:
        """
        Bring the copy to the state the run published last, after every model change
        that came before it.
        """
        while self.shared_state.is_replaced:
            self._apply(self.changes.recv())
        self.world.state._data[:] = self.shared_state.read()[:, self.columns]
        self.world.notify_state_change(publish_changes=False)

    def _apply(self, change: ModelChange) -> None:
        """
        Change the copy's model as the run's changed, and read the state from where it
        is published from then on.
        """
        with self.world.modify_world(publish_changes=False):
            from_json(
                change.modification_block,
                **WorldEntityWithIDKwargsTracker.from_world(self.world).create_kwargs(),
            ).apply(self.world)
        self.shared_state.close()
        self.shared_state = SharedWorldState.attach(
            change.state.memory_name, change.state.shape
        )
        self.columns = change.state.columns_in(self.world)


@dataclass(eq=False)
class MirroringSegmind(Segmind):
    """
    SegMind ticking its detectors on a copy of the run's world, which it brings up to
    date before every tick.
    """

    mirror: MirroredWorld = field(kw_only=True)
    """
    The copy of the run's world the detectors watch.
    """

    @classmethod
    def for_request(cls, request: WatchRequest, changes: Connection) -> Self:
        """
        :param request: What the run sent to have itself watched.
        :param changes: The pipe the run sends its model changes through.
        :return: SegMind watching a copy of the run's world, as the run asked.
        """
        mirror = MirroredWorld.from_request(request, changes)
        return cls(
            world=mirror.world,
            detectors=request.detector_selection.detectors_for(
                [
                    mirror.world.find_world_entity_with_id(body_id)
                    for body_id in request.watched_body_ids
                ]
            ),
            tick_period=request.tick_period,
            mirror=mirror,
        )

    @classmethod
    def watch(
        cls,
        request: WatchRequest,
        changes: Connection,
        events: Connection,
        stopping: Event,
    ) -> None:
        """
        Watch a run from this process, from the world as it stood when the run asked
        until told to stop, sending every event back as soon as it is detected.

        :param request: What the run sent to have itself watched.
        :param changes: The pipe the run sends its model changes through.
        :param events: The pipe the events and the progress of the watching go through.
        :param stopping: Set when the watching is to stop.
        """
        try:
            segmind = cls.for_request(request, changes)
            segmind.event_logger.add_callback(
                DetectionEvent, lambda event: events.send(to_json(event))
            )
            threading.Thread(
                target=segmind.stop_ticking_once_set, args=(stopping,), daemon=True
            ).start()
            segmind.tick()
            events.send(WatchingProgress.STARTED)
            segmind.tick_until_stopped()
        finally:
            events.send(WatchingProgress.ENDED)

    def stop_ticking_once_set(self, run_stopping: Event) -> None:
        """
        Stop ticking, after seeing the world at rest, once the run sets
        ``run_stopping``.
        """
        run_stopping.wait()
        self.stopping.set()

    def tick(self) -> None:
        """
        Like :meth:`Segmind.tick`, on the state the run published last.
        """
        self.mirror.copy_latest_state()
        super().tick()
