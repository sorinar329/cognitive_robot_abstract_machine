from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type, Self

from typing_extensions import Dict, List, TYPE_CHECKING

from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    from_json,
    SubclassJSONSerializer,
    to_json,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.definitions import JointStateType
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

if TYPE_CHECKING:
    from semantic_digital_twin.robots.abstract_robot import AbstractRobot
    from semantic_digital_twin.world import World


@dataclass
class JointState(SubclassJSONSerializer):
    mapping: Dict[ActiveConnection1DOF, float]
    """
    Mapping of connection to the connection position
    """

    state_type: JointStateType = field(default=None)
    """
    A type to better describe this state
    """

    name: PrefixedName = field(default=PrefixedName("JointState"))
    """
    A Name for this JointState
    """

    _connections: List[ActiveConnection1DOF] = field(init=False, default_factory=list)
    """
    All connections in this state
    """

    _target_values: List[float] = field(init=False, default_factory=list)
    """
    All target values in this state, order has to correspond to the order of connections
    """

    _robot: AbstractRobot = field(init=False, default=None)

    def __post_init__(self):
        for connection, target in self.mapping.items():
            self._connections.append(connection)
            self._target_values.append(target)

    def __len__(self):
        return len(self._connections)

    def __hash__(self):
        """
        Returns the hash of the joint state, which is based on the joint state name.
        This allows for proper comparison and storage in sets or dictionaries.
        """
        return hash((self._connections, self._target_values))

    def assign_to_robot(self, robot: AbstractRobot):
        """
        Assigns the joint state to the given robot. This method ensures that the joint state is only assigned
        to one robot at a time, and raises an error if it is already assigned to another robot.
        """
        if self._robot is not None and self._robot != robot:
            raise ValueError(
                f"Joint State {self.name} is already assigned to another robot: {self._robot.name}."
            )
        if self._robot is not None:
            return
        self._robot = robot

    def items(self):
        return zip(self._connections, self._target_values)

    @classmethod
    def from_str_dict(cls, mapping: Dict[str, float], world: World):
        mapping = {
            world.get_connection_by_name(connection_name): target
            for connection_name, target in mapping.items()
        }
        return cls(mapping=mapping)

    @classmethod
    def from_lists(cls, connections: List[ActiveConnection1DOF], targets: List[float]):
        return cls(mapping=dict(zip(connections, targets)))

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "_connections": [
                to_json(connection.name) for connection in self._connections
            ],
            "_target_values": self._target_values,
            "joint_state_type": to_json(self.state_type),
            "name": to_json(self.name),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        world = tracker._world
        if world:
            connections = [
                world.get_connection_by_name(from_json(name, **kwargs))
                for name in data["_connections"]
            ]
        else:
            connections = from_json(data["_connections"])
        target_values = from_json(data["_target_values"])
        state_type = from_json(data["joint_state_type"])
        name = from_json(data["name"])
        return cls(
            dict(zip(connections, target_values)), state_type=state_type, name=name
        )


GripperState = JointState
ArmState = JointState
TorsoState = JointState
