from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UnMonitoredContainerTypeForDescriptor(Exception):
    """
    Raised when a descriptor is used on a field with a container type that is not monitored (i.e., is not a subclass of
    MonitoredContainer). This happens when your type hint of the field is using a container type that is not supported.
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
class NoObjectOfIndividualDeclaresProperty(Exception):
    """
    Raised when an inferred fact is stored for an individual none of whose objects (its root role taker and roles) is
    an instance of the domain of the property descriptor that manages the fact.
    """

    individual_object: object
    """
    An object of the individual.
    """
    property_descriptor_class: type
    """
    The property of the fact.
    """

    def __post_init__(self):
        super().__init__(
            f"No object of the individual of {self.individual_object!r} declares the property "
            f"'{self.property_descriptor_class.__name__}'."
        )


@dataclass
class UnionOfAnonymousClassesAsFiller(Exception):
    """
    Raised when a restriction restricts the values of a property to a union that holds an anonymous class
    expression, which cannot be checked against the classes of a value.
    """

    union_name: str
    """
    The name ontomatic gives the union.
    """

    def __post_init__(self):
        super().__init__(
            f"The union '{self.union_name}' restricts the values of a property but holds an anonymous class "
            f"expression; only unions of named classes can be translated there."
        )


@dataclass
class UntranslatedComplement(Exception):
    """
    Raised when a class expression is the complement of something other than a named class or a restriction.
    """

    complemented_name: str
    """
    The name ontomatic gives the complemented class expression.
    """

    def __post_init__(self):
        super().__init__(
            f"The complement of '{self.complemented_name}' cannot be translated; only complements of named classes "
            f"and of restrictions are."
        )
