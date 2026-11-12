"""
Individuals whose classes are still being inferred, and counting their values of a
class.
"""

from __future__ import annotations

import uuid

from krrood.ontomatic.anonymous_class import AnonymousClass, NumberOfValuesOfClass

# %% the classes the counted values belong to


class Topping:
    """
    A class values can belong to.
    """


class CheeseTopping(Topping):
    """
    A subclass values can belong to.
    """


class Base:
    """
    A class no topping belongs to.
    """


def value_of(*types: type) -> AnonymousClass:
    """
    :return: An individual of the classes.
    """
    value = AnonymousClass(uri=f"http://example.org/pizza#{uuid.uuid4().hex}")
    for value_type in types:
        value.add_type(cls=value_type)
    return value


# %% types


def test_an_individual_records_the_classes_added_to_it():
    """
    The classes added to an individual are its types.
    """
    assert value_of(Topping, CheeseTopping).types == {Topping, CheeseTopping}


# %% counting values of a class


def test_the_values_of_a_class_and_its_subclasses_are_counted():
    """
    A value counts when it is of the class or of a subclass of it.
    """
    values = [value_of(CheeseTopping), value_of(Topping), value_of(Base)]
    assert NumberOfValuesOfClass.count(values=values, value_class=Topping) == 2
