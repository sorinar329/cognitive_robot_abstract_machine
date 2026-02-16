import dataclasses

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation, Body


@dataclasses.dataclass(unsafe_hash=True)
class Cup(SemanticAnnotation):
    body: Body


@dataclasses.dataclass(unsafe_hash=True)
class Milk(SemanticAnnotation):
    body: Body


@dataclasses.dataclass(unsafe_hash=True)
class Bueno(SemanticAnnotation):
    body: Body


@dataclasses.dataclass(unsafe_hash=True)
class Salt(SemanticAnnotation):
    body: Body


@dataclasses.dataclass(unsafe_hash=True)
class Cornflakes(SemanticAnnotation):
    body: Body
