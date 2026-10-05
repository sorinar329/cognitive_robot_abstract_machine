"""
The opt-in control of a running simulation: the scene graph the viewer shows, and the
requests that pause, resume, stop and change the simulation.

A demo opts in by attaching a
:class:`~semantic_digital_twin.adapters.controlled_simulation.ControlledSimulation` to
its visualization; without one, the scene graph reports itself unavailable and every
control request is refused.

::

    GET  /simulation         {available, state, bodies: [{name, parent, connection,
                              placement, pose, mass, friction}]}
    POST /simulation/pause   {ok, state}
    POST /simulation/resume  {ok, state}
    POST /simulation/stop    {ok, state}
    POST /simulation/edit    {body, change: pose|mass|friction, value} -> {ok}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import StrEnum

from typing_extensions import Any, Dict, List, Optional

from semantic_digital_twin.adapters.controlled_simulation import (
    BodyFrictionChange,
    BodyMassChange,
    BodyPoseChange,
    ControlledSimulation,
    SimulatedBody,
    SimulationEdit,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.contact import ContactFriction

from cramera.body_geometry import POSE_PRECISION, NumericPose
from cramera.live.transforms import ConnectionActivity, TransformSnapshot

# %% vocabulary


class SimulationRoute(StrEnum):
    """
    The bridge endpoints of the simulation control.
    """

    SCENE_GRAPH = "/simulation"
    PAUSE = "/simulation/pause"
    RESUME = "/simulation/resume"
    STOP = "/simulation/stop"
    EDIT = "/simulation/edit"


class SceneGraphField(StrEnum):
    """
    Keys of the scene graph payload and of its body rows.
    """

    AVAILABLE = "available"
    STATE = "state"
    BODIES = "bodies"
    NAME = "name"
    PARENT = "parent"
    CONNECTION = "connection"
    PLACEMENT = "placement"
    POSE = "pose"
    MASS = "mass"
    FRICTION = "friction"


class SimulationEditField(StrEnum):
    """
    Keys of an edit request.
    """

    BODY = "body"
    CHANGE = "change"
    VALUE = "value"


class SimulationChange(StrEnum):
    """
    What an edit request changes about its body.
    """

    POSE = "pose"
    MASS = "mass"
    FRICTION = "friction"


class PoseField(StrEnum):
    """
    Keys of a requested pose: a position in metres and roll, pitch and yaw in radians,
    in the frame of the world's root.
    """

    X = "x"
    Y = "y"
    Z = "z"
    ROLL = "roll"
    PITCH = "pitch"
    YAW = "yaw"


# %% errors


class NoSimulationAttached(Exception):
    """
    Raised when a control request reaches a bridge that no simulation was attached to.
    """

    def __init__(self) -> None:
        super().__init__("no controllable simulation is attached to this bridge")


class UnknownSimulatedBody(Exception):
    """
    Raised when an edit request names a body the simulation does not simulate.
    """

    def __init__(self, name: str) -> None:
        """
        :param name: The name the request gave.
        """
        super().__init__("the simulation simulates no body named %r" % name)


class InvalidSimulationEditRequest(ValueError):
    """
    Raised when an edit request is not shaped as the endpoint expects.
    """


# %% the scene graph


@dataclass(frozen=True)
class SceneGraphBody:
    """
    One row of the scene graph: a simulated body, the connection holding it and where
    it stands.
    """

    simulated_body: SimulatedBody
    """
    What the simulation makes of the body.
    """

    connection: Optional[ConnectionActivity]
    """
    The connection holding the body, or None for the root of the world.
    """

    pose: Optional[List[float]]
    """
    Where the body stands, as ``[x, y, z, qx, qy, qz, qw]`` in the world's frame, or
    None if that is not published.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The row in the JSON shape the viewer reads.
        """
        friction = self.simulated_body.friction
        return {
            SceneGraphField.NAME: str(self.simulated_body.body.name),
            SceneGraphField.PARENT: self.connection.parent if self.connection else None,
            SceneGraphField.CONNECTION: (
                self.connection.kind if self.connection else None
            ),
            SceneGraphField.PLACEMENT: self.simulated_body.placement,
            SceneGraphField.POSE: self.pose,
            SceneGraphField.MASS: self.simulated_body.mass,
            SceneGraphField.FRICTION: asdict(friction) if friction else None,
        }


@dataclass(frozen=True)
class SceneGraph:
    """
    Every body of a controlled simulation, as the viewer lists them.
    """

    simulation: ControlledSimulation
    """
    The simulation whose bodies are listed.
    """

    bodies: List[SceneGraphBody]
    """
    The rows, in the order the simulation lists its bodies.
    """

    @classmethod
    def compose(
        cls,
        simulation: ControlledSimulation,
        transforms: TransformSnapshot,
        published_poses: Dict[str, List[float]],
    ) -> SceneGraph:
        """
        Combine snapshots taken on the simulation thread; reads no world state itself.

        :param simulation: The controlled simulation.
        :param transforms: The bridge's latest connection graph.
        :param published_poses: The poses the bridge published, by body name.
        """
        connections = {activity.child: activity for activity in transforms.activities}
        rows = []
        for simulated_body in simulation.simulated_bodies():
            name = str(simulated_body.body.name)
            rows.append(
                SceneGraphBody(
                    simulated_body=simulated_body,
                    connection=connections.get(name),
                    pose=cls._pose_of(simulated_body, published_poses.get(name)),
                )
            )
        return cls(simulation=simulation, bodies=rows)

    @staticmethod
    def _pose_of(
        simulated_body: SimulatedBody, published_pose: Optional[List[float]]
    ) -> Optional[List[float]]:
        """
        :return: Where a fixed body stands as the simulation recorded it, else the pose
            the bridge published for the body.
        """
        if simulated_body.pose is None:
            return published_pose
        return NumericPose.of_matrix(simulated_body.pose.to_np()).rounded(
            POSE_PRECISION
        )

    def to_payload(self) -> Dict[str, Any]:
        """
        The scene graph in the JSON shape the viewer reads.
        """
        return {
            SceneGraphField.AVAILABLE: True,
            SceneGraphField.STATE: self.simulation.state,
            SceneGraphField.BODIES: [row.to_payload() for row in self.bodies],
        }

    @staticmethod
    def unavailable_payload() -> Dict[str, Any]:
        """
        What the viewer is told when no simulation is attached.
        """
        return {SceneGraphField.AVAILABLE: False}


# %% edit requests


@dataclass(frozen=True)
class SimulationEditRequest:
    """
    A validated request to change one body of a controlled simulation.
    """

    body_name: str
    """
    The prefixed name of the body to change.
    """

    change: SimulationChange
    """
    What is changed about the body.
    """

    value: Any
    """
    The new value, shaped as :attr:`change` requires; checked by :meth:`edit_for`.
    """

    @classmethod
    def from_json(cls, payload: object) -> SimulationEditRequest:
        """
        :param payload: The decoded JSON request body.
        :raises InvalidSimulationEditRequest: If the body or a field is malformed.
        """
        if not isinstance(payload, dict):
            raise InvalidSimulationEditRequest("body must be a JSON object")
        body_name = payload.get(SimulationEditField.BODY)
        if not isinstance(body_name, str):
            raise InvalidSimulationEditRequest("body must name a body")
        change = payload.get(SimulationEditField.CHANGE)
        if change not in tuple(SimulationChange):
            raise InvalidSimulationEditRequest(
                "change must be one of %s" % ", ".join(SimulationChange)
            )
        return cls(
            body_name=body_name,
            change=SimulationChange(change),
            value=payload.get(SimulationEditField.VALUE),
        )

    def edit_for(self, simulation: ControlledSimulation) -> SimulationEdit:
        """
        :param simulation: The simulation the requested body belongs to.
        :return: The change to submit to ``simulation``.
        :raises UnknownSimulatedBody: If ``simulation`` simulates no such body.
        :raises InvalidSimulationEditRequest: If the value is malformed.
        :raises SimulationEditError: If the value is not one the body can take.
        """
        simulated_body = self._simulated_body(simulation)
        body = simulated_body.body
        if self.change is SimulationChange.MASS:
            return BodyMassChange(body, self._number(self.value, self.change))
        if self.change is SimulationChange.FRICTION:
            return BodyFrictionChange(body, self._friction())
        return BodyPoseChange(body, self._pose(simulation))

    def _simulated_body(self, simulation: ControlledSimulation) -> SimulatedBody:
        """
        :raises UnknownSimulatedBody: If ``simulation`` simulates no such body.
        """
        for simulated_body in simulation.simulated_bodies():
            if str(simulated_body.body.name) == self.body_name:
                return simulated_body
        raise UnknownSimulatedBody(self.body_name)

    def _friction(self) -> ContactFriction:
        """
        :raises InvalidSimulationEditRequest: If the value is not a set of coefficients.
        """
        names = [coefficient.name for coefficient in fields(ContactFriction)]
        coefficients = self._numbers_by_name(names)
        return ContactFriction(**coefficients)

    def _pose(
        self, simulation: ControlledSimulation
    ) -> HomogeneousTransformationMatrix:
        """
        :raises InvalidSimulationEditRequest: If the value is not a complete pose.
        """
        coordinates = self._numbers_by_name(list(PoseField))
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            **coordinates, reference_frame=simulation.simulation.world.root
        )

    def _numbers_by_name(self, names: List[str]) -> Dict[str, float]:
        """
        :param names: The keys the value has to hold, and no others.
        :return: The value's numbers by key.
        :raises InvalidSimulationEditRequest: If the value holds other keys or a value
            is not a number.
        """
        if not isinstance(self.value, dict) or set(self.value) != set(names):
            raise InvalidSimulationEditRequest(
                "%s must hold exactly %s" % (self.change, ", ".join(names))
            )
        return {name: self._number(self.value[name], name) for name in names}

    @staticmethod
    def _number(value: object, name: str) -> float:
        """
        :raises InvalidSimulationEditRequest: If ``value`` is not a number.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidSimulationEditRequest("%s must be a number" % name)
        return float(value)
