"""
A stepped MuJoCo simulation that another thread can pause, resume, stop and change
while a control loop drives it.

Every change is carried out on the thread that steps the simulation, between two steps,
so the control loop never sees the world change halfway through one of its cycles.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum

from typing_extensions import Dict, List, Optional

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    BodyNotSimulatedError,
    BodyWithoutContactError,
    MasslessBodyError,
    NegativeFrictionError,
    NonPositiveMassError,
    SimulationStoppedError,
    UnmovableBodyError,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.contact import ContactFriction
from semantic_digital_twin.world_description.world_entity import Body

# %% run state


class SimulationRunState(StrEnum):
    """
    Whether a controlled simulation advances when its control loop steps it.
    """

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


# %% one simulated body


class Placement(StrEnum):
    """
    How a body is held in its world, which decides whether it can be put elsewhere.
    """

    LOOSE = "loose"
    """
    Hangs off a connection with six degrees of freedom: the physics moves it, and it
    can be put anywhere.
    """

    FIXED = "fixed"
    """
    Held by fixed connections all the way up to the root of its world, so nothing but
    being put elsewhere moves it.
    """

    UNMOVABLE = "unmovable"
    """
    The root of its world, or carried by a joint, so it cannot be put elsewhere on its
    own.
    """

    @classmethod
    def of_body(cls, body: Body) -> Placement:
        """
        :param body: A body of a world.
        :return: How ``body`` is held in its world.
        """
        connection = body.parent_connection
        if isinstance(connection, Connection6DoF):
            return cls.LOOSE
        if connection is None:
            return cls.UNMOVABLE
        while isinstance(connection, FixedConnection):
            connection = connection.parent.parent_connection
        return cls.FIXED if connection is None else cls.UNMOVABLE


@dataclass(frozen=True)
class SimulatedBody:
    """
    What the simulation makes of one body, as far as it can be changed while running.
    """

    body: Body
    """
    The body described.
    """

    mass: float
    """
    The mass the simulation gives the body, in kilograms; zero for a body it does not
    move, such as the world itself.
    """

    friction: Optional[ContactFriction]
    """
    The friction of the body's surface, or None if no shape of it collides.
    """

    placement: Placement
    """
    How the body is held in its world.
    """

    pose: Optional[HomogeneousTransformationMatrix]
    """
    Where a :attr:`Placement.FIXED` body stands in its world; None for any other body,
    whose pose either changes as it moves or cannot be changed.
    """

    @classmethod
    def of_body(cls, body: Body, simulation: MujocoSim) -> SimulatedBody:
        """
        :param body: A body ``simulation`` simulates.
        :param simulation: The simulation to read the body from.
        :return: What ``simulation`` currently makes of ``body``.
        """
        placement = Placement.of_body(body)
        return cls(
            body=body,
            mass=simulation.body_mass(body),
            friction=simulation.body_friction(body),
            placement=placement,
            pose=body.global_pose if placement is Placement.FIXED else None,
        )


# %% changes to a running simulation


@dataclass
class SimulationEdit(ABC):
    """
    One change to a body of a running simulation.
    """

    body: Body
    """
    The body to change.
    """

    @abstractmethod
    def check(self, simulated_body: SimulatedBody) -> None:
        """
        Raise if this change cannot be made to the body as the simulation has it.

        :param simulated_body: What the simulation makes of :attr:`body`.
        :raises SimulationEditError: If the change cannot be made.
        """

    @abstractmethod
    def apply(self, simulation: MujocoSim) -> None:
        """
        Make this change.

        :param simulation: The simulation to change.
        """


@dataclass
class BodyPoseChange(SimulationEdit):
    """
    Puts a loose or fixed body somewhere else.
    """

    pose: HomogeneousTransformationMatrix
    """
    Where the body is put; it has to carry the frame it is expressed in.
    """

    def check(self, simulated_body: SimulatedBody) -> None:
        if simulated_body.placement is Placement.UNMOVABLE:
            raise UnmovableBodyError(self.body.name)

    def apply(self, simulation: MujocoSim) -> None:
        if Placement.of_body(self.body) is Placement.LOOSE:
            self.body.parent_connection.origin = self.pose
            return
        connection = self.body.parent_connection
        parent_T_body = simulation.world.transform(self.pose, connection.parent)
        with simulation.world.modify_world():
            connection.parent_T_connection_expression = (
                parent_T_body @ connection.connection_T_child_expression.inverse()
            )
        simulation.set_fixed_body_pose(self.body, connection.origin)


@dataclass
class BodyMassChange(SimulationEdit):
    """
    Gives a body a new mass, keeping its mass distributed as before.
    """

    mass: float
    """
    The new mass, in kilograms.
    """

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise NonPositiveMassError(self.body.name, self.mass)

    def check(self, simulated_body: SimulatedBody) -> None:
        if simulated_body.mass == 0:
            raise MasslessBodyError(self.body.name)

    def apply(self, simulation: MujocoSim) -> None:
        simulation.set_body_mass(self.body, self.mass)


@dataclass
class BodyFrictionChange(SimulationEdit):
    """
    Gives every colliding shape of a body a new friction.
    """

    friction: ContactFriction
    """
    The new friction.
    """

    def __post_init__(self) -> None:
        coefficients = self.friction.to_list()
        if min(coefficients) < 0:
            raise NegativeFrictionError(self.body.name, coefficients)

    def check(self, simulated_body: SimulatedBody) -> None:
        if simulated_body.friction is None:
            raise BodyWithoutContactError(self.body.name)

    def apply(self, simulation: MujocoSim) -> None:
        simulation.set_body_friction(self.body, self.friction)


# %% the controlled simulation


@dataclass(frozen=True)
class SteppingInstruction:
    """
    What the stepping thread does next, taken from the shared state at one instant.
    """

    state: SimulationRunState
    """
    Whether the step advances, waits or ends the control loop.
    """

    edits: List[SimulationEdit]
    """
    The changes to make before that, oldest first.
    """


@dataclass
class ControlledSimulation:
    """
    A stepped simulation that any thread can pause, resume, stop and change, while a
    control loop steps it through :meth:`step_simulation`.

    While paused, the stepping thread waits inside :meth:`step_simulation`, so the
    control loop and the physics stop together and carry on from the same instant.
    Changes are made on the stepping thread: right away while paused, else before the
    next step.
    """

    simulation: MujocoSim
    """
    The simulation stepped and changed; it has to be started with
    :meth:`~semantic_digital_twin.adapters.multi_sim.MujocoSim.start_stepped_simulation`.
    """

    _state: SimulationRunState = field(default=SimulationRunState.RUNNING, init=False)
    """
    Whether the next step advances, waits or ends the control loop.
    """

    _condition: threading.Condition = field(
        default_factory=threading.Condition, init=False, repr=False
    )
    """
    Guards the run state, the pending changes and the simulated bodies, and wakes the stepping
    thread when either of the first two changes.
    """

    _pending_edits: List[SimulationEdit] = field(default_factory=list, init=False)
    """
    Changes submitted but not yet made, oldest first.
    """

    _simulated_bodies: Dict[PrefixedName, SimulatedBody] = field(
        default_factory=dict, init=False
    )
    """
    Every simulated body by its name, as of the last change.
    """

    def __post_init__(self) -> None:
        self._simulated_bodies = self._read_simulated_bodies()

    @property
    def state(self) -> SimulationRunState:
        """
        Whether the simulation is running, paused or stopped.
        """
        with self._condition:
            return self._state

    def simulated_bodies(self) -> List[SimulatedBody]:
        """
        :return: Every simulated body, as of the last change; safe to read from any
            thread.
        """
        with self._condition:
            return list(self._simulated_bodies.values())

    def simulated_body(self, body: Body) -> SimulatedBody:
        """
        :param body: A simulated body.
        :return: What the simulation makes of ``body``, as of the last change.
        :raises BodyNotSimulatedError: If the simulation does not simulate ``body``.
        """
        with self._condition:
            return self._recorded_simulated_body(body)

    # %% control from any thread

    def pause(self) -> None:
        """
        Hold the stepping thread at its next step until :meth:`resume`.

        :raises SimulationStoppedError: If the simulation was stopped.
        """
        self._change_state(SimulationRunState.PAUSED)

    def resume(self) -> None:
        """
        Let a paused stepping thread carry on.

        :raises SimulationStoppedError: If the simulation was stopped.
        """
        self._change_state(SimulationRunState.RUNNING)

    def stop(self) -> None:
        """
        End the control loop: its next step, or the one it is paused in, raises
        :class:`~semantic_digital_twin.exceptions.SimulationStoppedError`.
        """
        with self._condition:
            self._state = SimulationRunState.STOPPED
            self._pending_edits.clear()
            self._condition.notify_all()

    def submit(self, edit: SimulationEdit) -> None:
        """
        Have ``edit`` made on the stepping thread: right away while paused, else before
        the next step.

        :param edit: The change to make.
        :raises SimulationStoppedError: If the simulation was stopped.
        :raises SimulationEditError: If the change cannot be made to its body.
        """
        with self._condition:
            self._raise_if_stopped()
            edit.check(self._recorded_simulated_body(edit.body))
            self._pending_edits.append(edit)
            self._condition.notify_all()

    # %% the stepping thread

    def step_simulation(self, duration: timedelta) -> None:
        """
        Make the pending changes, wait while paused, then advance the physics.

        :param duration: How much simulated time to advance.
        :raises SimulationStoppedError: If the simulation was stopped.
        """
        while True:
            instruction = self._next_instruction()
            if instruction.state is SimulationRunState.STOPPED:
                raise SimulationStoppedError(self.simulation.world.root.name.name)
            self._apply(instruction.edits)
            if instruction.state is SimulationRunState.RUNNING:
                break
        self.simulation.step_simulation(duration)

    def _next_instruction(self) -> SteppingInstruction:
        """
        Wait while paused with nothing to change.

        :return: The run state, and the changes to make now.
        """
        with self._condition:
            while self._state is SimulationRunState.PAUSED and not self._pending_edits:
                self._condition.wait()
            edits, self._pending_edits = self._pending_edits, []
            return SteppingInstruction(self._state, edits)

    def _apply(self, edits: List[SimulationEdit]) -> None:
        """
        Make ``edits`` and record the simulated bodies they leave behind.

        :param edits: The changes to make, oldest first.
        """
        if not edits:
            return
        for edit in edits:
            edit.apply(self.simulation)
        simulated_bodies = self._read_simulated_bodies()
        with self._condition:
            self._simulated_bodies = simulated_bodies

    # %% helpers

    def _change_state(self, state: SimulationRunState) -> None:
        """
        Switch between running and paused, and wake the stepping thread.

        :param state: The state to switch to.
        :raises SimulationStoppedError: If the simulation was stopped.
        """
        with self._condition:
            self._raise_if_stopped()
            self._state = state
            self._condition.notify_all()

    def _raise_if_stopped(self) -> None:
        """
        :raises SimulationStoppedError: If the simulation was stopped.
        """
        if self._state is SimulationRunState.STOPPED:
            raise SimulationStoppedError(self.simulation.world.root.name.name)

    def _recorded_simulated_body(self, body: Body) -> SimulatedBody:
        """
        :return: What the simulation made of ``body`` at the last change; the caller
            holds the condition.
        :raises BodyNotSimulatedError: If the simulation does not simulate ``body``.
        """
        simulated_body = self._simulated_bodies.get(body.name)
        if simulated_body is None:
            raise BodyNotSimulatedError(body.name)
        return simulated_body

    def _read_simulated_bodies(self) -> Dict[PrefixedName, SimulatedBody]:
        """
        :return: Every simulated body, read from the simulation.
        """
        return {
            body.name: SimulatedBody.of_body(body, self.simulation)
            for body in self.simulation.world.bodies
            if self.simulation.simulates(body)
        }
