from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional, Type

# %% classes outside the converted hierarchy


@dataclass
class ReferencedOnlyClass:
    """
    A class outside the converted hierarchy that converted classes refer to.
    """

    name: str
    """
    A builtin-typed field, which is not a relation to another class.
    """


@dataclass
class SubclassOfReferencedOnlyClass(ReferencedOnlyClass):
    """
    A subclass of a class that is only referenced, and therefore not converted.
    """


@dataclass
class UnrelatedClass:
    """
    A class neither in the converted hierarchy nor referenced by it.
    """

    root: ConvertedRoot
    """
    A relation into the converted hierarchy, which does not make this class part of it.
    """


# %% converted hierarchy


@dataclass
class ConvertedRoot:
    """
    The root class handed to the converter.
    """


@dataclass
class Handle(ConvertedRoot):
    """
    A class with a single-valued relation to a class outside the hierarchy.
    """

    body: ReferencedOnlyClass
    """
    A relation to a class outside the converted hierarchy.
    """


@dataclass
class SpecializedHandle(Handle):
    """
    A subclass that inherits its relations unchanged.
    """


@dataclass
class Drawer(ConvertedRoot):
    """
    A class with single-valued relations inside the hierarchy.
    """

    handle: Handle
    """
    A required single-valued relation.
    """

    front: Optional[Handle] = None
    """
    An optional single-valued relation.
    """


@dataclass
class DrawerWithSpecializedHandle(Drawer):
    """
    A subclass that narrows the type of an inherited relation.
    """

    handle: SpecializedHandle
    """
    The inherited relation, narrowed to a subclass of its original type.
    """


@dataclass
class Cabinet(ConvertedRoot):
    """
    A class with a many-valued relation and a relation to a class object.
    """

    drawers: List[Drawer] = field(default_factory=list)
    """
    A many-valued relation.
    """

    handle_type: Type[Handle] = Handle
    """
    A reference to a class rather than to an instance of it.
    """
