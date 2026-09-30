from __future__ import annotations

from dataclasses import dataclass

from rdflib import URIRef

from krrood.exceptions import DataclassException
from krrood.utils import module_and_class_name


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


@dataclass
class PropertyIsBothObjectAndDatatype(DataclassException):
    """
    Raised when fields named alike refer to instances of classes in one converted class
    and to builtin values in another, so one OWL property would have to be both an
    ``owl:ObjectProperty`` and an ``owl:DatatypeProperty``, which OWL 2 DL forbids.
    """

    property_iri: URIRef
    """
    The IRI both kinds of property would be given.
    """

    class_with_object_field: type
    """
    A class whose field of that name refers to instances of another class.
    """

    class_with_datatype_field: type
    """
    A class whose field of that name holds builtin values.
    """

    def error_message(self) -> str:
        return (
            f"The property {self.property_iri} relates instances of "
            f"{module_and_class_name(self.class_with_object_field)} to other instances, but holds "
            f"builtin values on {module_and_class_name(self.class_with_datatype_field)}."
        )

    def suggest_correction(self) -> str:
        return (
            "Rename one of the fields or convert the classes into separate ontologies."
        )
