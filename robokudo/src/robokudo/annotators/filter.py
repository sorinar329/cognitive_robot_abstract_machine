"""
Filter annotator for RoboKudo.

This module provides an annotator for filtering annotations based on custom conditions.
It supports:

* Dynamic filtering through callable functions
* Custom filter arguments
* In-place annotation list modification
* Flexible condition evaluation

The module is used for:

* Annotation filtering
* Data preprocessing
* Result refinement
* Conditional processing
"""
import py_trees

import robokudo


class FilterAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Annotator for applying filter conditions to annotations.

    This annotator applies a provided filter function to the current set of
    annotations, modifying the annotation list in-place based on the filter results.

    :ivar descriptor: Configuration descriptor containing filter parameters
    :type descriptor: FilterAnnotator.Descriptor
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """
        Configuration descriptor for filter annotator.

        :ivar parameters: Filter parameters
        :type parameters: FilterAnnotator.Descriptor.Parameters
        """

        class Parameters:
            """
            Parameter container for filter configuration.

            :ivar func: Filter function to apply
            :type func: callable
            :ivar func_args: Positional arguments for filter function
            :type func_args: tuple
            :ivar func_kwargs: Keyword arguments for filter function
            :type func_kwargs: dict
            """

            def __init__(self):
                self.func = None
                self.func_args = None
                self.func_kwargs = None

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="FilterAnnotator", descriptor=Descriptor()):
        """Initialize the filter annotator.

        :param name: Annotator name, defaults to "FilterAnnotator"
        :type name: str, optional
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: FilterAnnotator.Descriptor, optional
        """
        super().__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self):
        """
        Apply the filter function to current annotations.

        The filter function is applied to each annotation with the configured
        arguments. Only annotations that pass the filter are kept.

        :return: SUCCESS status
        :rtype: py_trees.Status
        """
        func = self.descriptor.parameters.func

        if func:
            func_args = self.descriptor.parameters.func_args or []
            func_kwargs = self.descriptor.parameters.func_kwargs or {}

            annotations = self.get_cas().annotations
            annotations = [annotation for annotation in annotations
                           if func(annotation, *func_args, **func_kwargs)]
            self.get_cas().annotations = annotations

        return py_trees.Status.SUCCESS
