from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from krrood.entity_query_language.predicate import Symbol
    from krrood.ontomatic.property_descriptor.property_descriptor import (
        PropertyDescriptor,
    )


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class UnionOfAnonymousClassesAsFiller(DataclassException):
    """
    Raised when a restriction restricts the values of a property to a union that holds
    an anonymous class expression, which cannot be checked as a tuple of classes.
    """

    union_name: str
    """
    The name of the union.
    """

    def error_message(self) -> str:
        return (
            f"The union {self.union_name} restricts the values of a property but holds an "
            f"anonymous class expression."
        )

    def suggest_correction(self) -> str:
        return "Name the anonymous class expression in the ontology."


@dataclass(kw_only=True)
class UntranslatedComplement(DataclassException):
    """
    Raised when an intersection holds the complement of something other than a named
    class or a restriction.
    """

    complemented_name: str
    """
    The name of the complemented class expression.
    """

    def error_message(self) -> str:
        return f"The complement of {self.complemented_name} cannot be translated."

    def suggest_correction(self) -> str:
        return "Complement a named class or a restriction."


@dataclass(kw_only=True)
class DescriptorNotDeclaredForDomain(DataclassException):
    """
    Raised when neither a class nor any class it inherits fields from declares a field
    managed by a descriptor type.
    """

    descriptor_type: type
    """
    The descriptor type.
    """

    domain_type: type
    """
    The class without such a field.
    """

    def error_message(self) -> str:
        return (
            f"No field of {self.domain_type.__name__} is managed by "
            f"{self.descriptor_type.__name__}."
        )

    def suggest_correction(self) -> str:
        return (
            f"Declare the field on {self.domain_type.__name__} or on a class it inherits from, "
            f"and assign a {self.descriptor_type.__name__} to it."
        )


@dataclass(kw_only=True)
class NoObjectOfIndividualDeclaresProperty(DataclassException):
    """
    Raised when a fact is stored for an individual none of whose objects, its root role
    taker and its roles, has the field of the property.
    """

    individual_object: Symbol
    """
    An object of the individual.
    """

    property_descriptor_class: type[PropertyDescriptor]
    """
    The property of the fact.
    """

    def error_message(self) -> str:
        return (
            f"No object of the individual of {self.individual_object!r} declares the "
            f"property {self.property_descriptor_class.__name__}."
        )

    def suggest_correction(self) -> str:
        return (
            f"Give the individual a role, or a class, that declares a field managed by "
            f"{self.property_descriptor_class.__name__}."
        )
