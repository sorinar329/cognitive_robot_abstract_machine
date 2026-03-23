from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from krrood.ormatic.dao import AlternativeMapping


@dataclass(eq=False)
class Dependency:
    """
    A domain class that uses an AlternativeMapping.
    Its to_domain_object just returns itself but we will check if its 'value' is set.
    """

    name: str
    value: int = 0


@dataclass(eq=False)
class Main:
    """
    A domain class that uses an AlternativeMapping.
    It depends on 'Dependency' and accesses its 'value' in to_domain_object.
    """

    name: str
    dependency: Optional[Dependency] = None


@dataclass(eq=False)
class DependencyMapping(AlternativeMapping[Dependency]):
    name: str
    value: int

    @classmethod
    def from_domain_object(cls, obj: Dependency):
        return cls(name=obj.name, value=obj.value)

    def to_domain_object(self) -> Dependency:
        # This mapping just returns a Dependency instance.
        return Dependency(name=self.name, value=self.value)


@dataclass(eq=False)
class MainMapping(AlternativeMapping[Main]):
    name: str
    dependency: Optional[Dependency]

    @classmethod
    def from_domain_object(cls, obj: Main):
        return cls(name=obj.name, dependency=obj.dependency)

    def to_domain_object(self) -> Main:
        # The bug: if 'dependency' is still a DependencyMapping instance,
        # it might not have its 'value' attribute populated correctly
        # (if it was a scalar that ORMatic fills later).
        # Actually, in the current ORMatic, scalars are filled in Pass 2.1.
        # But if 'dependency' is an AlternativeMapping, it is NOT resolved
        # until its own turn in Pass 2.2.

        # If we access something that only exists on the DOMAIN object 'Dependency'
        # but not on the 'DependencyMapping' (or vice versa), we'd get an error.

        if self.dependency is not None:
            # Check if it is already the domain object
            if not isinstance(self.dependency, Dependency):
                raise TypeError(
                    f"dependency is {type(self.dependency)}, expected Dependency"
                )

            # Access an attribute to ensure it is fully initialized
            if self.dependency.value != 42:
                raise ValueError(
                    f"dependency.value is {self.dependency.value}, expected 42"
                )

        return Main(name=self.name, dependency=self.dependency)


@dataclass
class CyclicDependentAlternativeMappingContainer:
    """
    A container that ensures Main is discovered AFTER Dependency.
    """

    dependency: Dependency
    main: Main
