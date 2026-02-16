"""CAS condition checking utilities.

This module provides annotators for checking conditions in the Common Analysis System (CAS).
The annotators follow these principles:

* Return SUCCESS if condition is true
* Return FAILURE if condition is false
* Never return RUNNING status

.. note::
   For RUNNING status checks, use CASCondition from cas_condition.py instead.
"""
import functools
from typing import Callable

import py_trees

import robokudo.annotators.core
import robokudo.cas
import robokudo.types


class CASCheckFunc(robokudo.annotators.core.BaseAnnotator):
    """Function-based CAS condition checker.
    
    Evaluates a given function that checks conditions in the CAS and returns:
    
    * SUCCESS if function returns True
    * FAILURE if function returns False
    
    .. warning::
       Will raise an exception if initialized without a function.
    """

    def __init__(self, name="CASCheckFunc", func: Callable[[robokudo.cas.CAS], bool] = None, raise_with_str: str = ""):
        """Initialize the CAS condition checker.

        :param name: Name of this node in the behavior tree, defaults to "CASCheckFunc"
        :type name: str, optional
        :param func: Function that evaluates CAS conditions, must return bool, defaults to None
        :type func: Callable[[robokudo.cas.CAS], bool], optional
        :param raise_with_str: Error message to raise on failure, empty string disables raising, defaults to ""
        :type raise_with_str: str, optional
        :raises Exception: If func is None
        """
        super(CASCheckFunc, self).__init__(name=name)
        if func is None:
            raise Exception("CASCheckFunc needs a function to work properly. Please pass 'func'.")
        self.func = func
        self.raise_with_str = raise_with_str

    def update(self):
        """Check the CAS condition.

        :return: SUCCESS if condition is True, FAILURE if False
        :rtype: py_trees.Status
        :raises Exception: If raise_with_str is set and condition is False
        """
        cas = self.get_cas()
        if self.func(cas):
            return py_trees.common.Status.SUCCESS
        else:
            if self.raise_with_str != "":
                raise Exception(self.raise_with_str)
            else:
                return py_trees.common.Status.FAILURE


def any_of_type_present(annotation_type, cas: robokudo.cas.CAS):
    """Check if any annotations of a specific type exist in the CAS.

    :param annotation_type: Type of annotation to check for
    :type annotation_type: type
    :param cas: Common Analysis System instance
    :type cas: robokudo.cas.CAS
    :return: True if at least one annotation exists, False otherwise
    :rtype: bool
    """
    annotations = cas.filter_annotations_by_type(annotation_type)
    return len(annotations) > 0


class CASCheckAnnotationTypeExists(CASCheckFunc):
    """Check for existence of specific annotation types.
    
    Specialized CASCheckFunc that:
    
    * Checks for presence of specific annotation types
    * Returns SUCCESS if at least one annotation exists
    * Returns FAILURE if no annotations exist
    """

    def __init__(self, name="CASCheckAnnotationTypeExists", annotation_type=None, raise_with_str: str = ""):
        """Initialize annotation type checker.

        :param name: Name of this node in the behavior tree, defaults to "CASCheckAnnotationTypeExists"
        :type name: str, optional
        :param annotation_type: Type of annotation to check for, defaults to None
        :type annotation_type: type, optional
        :param raise_with_str: Error message to raise on failure, empty string disables raising, defaults to ""
        :type raise_with_str: str, optional
        """
        func = functools.partial(any_of_type_present, annotation_type)
        super(CASCheckAnnotationTypeExists, self).__init__(name=name, func=func, raise_with_str=raise_with_str)


class CASCheckOHExists(CASCheckAnnotationTypeExists):
    """Check for existence of ObjectHypothesis annotations.
    
    Specialized CASCheckAnnotationTypeExists that:
    
    * Specifically checks for ObjectHypothesis annotations
    * Returns SUCCESS if any ObjectHypothesis exists
    * Returns FAILURE if no ObjectHypothesis exists
    """

    def __init__(self, name="CASCheckOHExists", raise_with_str: str = ""):
        """Initialize ObjectHypothesis checker.

        :param name: Name of this node in the behavior tree, defaults to "CASCheckOHExists"
        :type name: str, optional
        :param raise_with_str: Error message to raise on failure, empty string disables raising, defaults to ""
        :type raise_with_str: str, optional
        """
        super(CASCheckOHExists, self).__init__(name=name,
                                               annotation_type=robokudo.types.scene.ObjectHypothesis,
                                               raise_with_str=raise_with_str)
