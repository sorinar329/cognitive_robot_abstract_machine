"""
Lambda function annotator for RoboKudo.

This module provides an annotator for executing arbitrary functions.
It supports:

* Dynamic function execution
* Custom function arguments
* Flexible parameter passing
* Generic function handling

The module is used for:

* Custom processing
* Dynamic behavior
* Testing and debugging
* Quick prototyping
"""
import py_trees

import robokudo


class LambdaFunctionAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Annotator for executing arbitrary functions.

    This annotator executes a provided function with configurable arguments,
    allowing for dynamic behavior definition without creating new annotator classes.

    :ivar descriptor: Configuration descriptor containing function parameters
    :type descriptor: LambdaFunctionAnnotator.Descriptor
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """
        Configuration descriptor for lambda function annotator.

        :ivar parameters: Function parameters
        :type parameters: LambdaFunctionAnnotator.Descriptor.Parameters
        """

        class Parameters:
            """
            Parameter container for function configuration.

            :ivar func: Function to execute
            :type func: callable
            :ivar func_args: Positional arguments for function
            :type func_args: tuple
            :ivar func_kwargs: Keyword arguments for function
            :type func_kwargs: dict
            """
            def __init__(self):
                self.func = None
                self.func_args = None
                self.func_kwargs = None

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="LambdaFunctionAnnotator", descriptor=Descriptor()):
        """
        Initialize the lambda function annotator. Minimal one-time init!

        :param name: Annotator name, defaults to "LambdaFunctionAnnotator"
        :type name: str, optional
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: LambdaFunctionAnnotator.Descriptor, optional
        """
        super().__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self):
        """
        Execute the configured function.

        The function is called with the annotator instance as first argument,
        followed by any configured positional and keyword arguments.

        :return: SUCCESS status
        :rtype: py_trees.Status
        """
        func = self.descriptor.parameters.func

        if func:
            func_args = self.descriptor.parameters.func_args or []
            func_kwargs = self.descriptor.parameters.func_kwargs or {}

            func(self, *func_args, **func_kwargs)

        return py_trees.Status.SUCCESS
