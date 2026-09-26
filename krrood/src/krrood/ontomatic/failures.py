from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from krrood.utils import module_and_class_name


@dataclass
class UnMonitoredContainerTypeForDescriptor(Exception):
    """
    Raised when a descriptor is used on a field with a container type that is not
    monitored (i.e., is not a subclass of MonitoredContainer).

    This happens when your type hint of the field is using a container type that is not
    supported.
    """

    clazz: type
    field_name: str
    container_type: type

    def __post_init__(self):
        super().__init__(
            f"Unmonitored container type '{self.container_type.__name__}' used for field '{self.field_name}' "
            f"in class '{self.clazz.__name__}'."
        )


@dataclass
class DuplicateOWLClassName(DataclassException):
    """
    Raised when two classes converted into one OWL ontology share a name, so both would
    be given the same OWL class IRI.
    """

    first_class: type
    """
    The class that claimed the name first.
    """

    second_class: type
    """
    The other class with the same name.
    """

    def error_message(self) -> str:
        return (
            f"Classes {module_and_class_name(self.first_class)} and "
            f"{module_and_class_name(self.second_class)} share the name "
            f"'{self.first_class.__name__}' and would become the same OWL class."
        )

    def suggest_correction(self) -> str:
        return "Rename one of the classes or convert them into separate ontologies."
