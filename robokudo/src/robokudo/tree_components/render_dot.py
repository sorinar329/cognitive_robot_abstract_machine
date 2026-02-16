"""Behavior tree visualization using DOT format.

This module provides functionality to render behavior trees in DOT format for visualization.
It supports both one-time rendering and decorator-based automatic rendering on status changes.

The module provides:

* Directory management for output
* Threaded rendering for performance
* Customizable rendering triggers
* Tree traversal utilities
"""

import os
from timeit import default_timer
from concurrent.futures import ThreadPoolExecutor

import py_trees

import robokudo.display
import robokudo.utils.tree


def create_dir_if_not_exists(path):
    """Create directory if it doesn't exist.

    :param path: Directory path to create
    :type path: str
    """
    if not os.path.exists(path):
        os.makedirs(path)


def render_now(behaviour: py_trees.Behaviour):
    """Generate behavior tree snapshot and save to disk.

    This method:
    * Creates output directory if needed
    * Finds root of tree
    * Renders tree to DOT format
    * Updates rendering statistics

    :param behaviour: Behavior node requesting the render
    :type behaviour: py_trees.Behaviour
    """
    start_timer = default_timer()

    if behaviour.create_dir_for_path:
        create_dir_if_not_exists(behaviour.path)

    # Go up until we find the root
    root = robokudo.utils.tree.find_root(behaviour)

    robokudo.display.render_dot_tree(root, name=f'RKTree{behaviour.suffix}-{behaviour.counter}',
                                     threadpool_executor=behaviour.executor, path_prefix=behaviour.path)

    behaviour.counter += 1
    end_timer = default_timer()
    behaviour.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'


class RenderTreeToDot(py_trees.Behaviour):
    """Behavior that renders tree to DOT format when ticked.

    This behavior renders the entire tree to DOT format each time it is ticked,
    saving the output to the specified directory.

    :ivar path: Output directory path
    :type path: str
    :ivar counter: Number of renders performed
    :type counter: int
    :ivar create_dir_for_path: Whether to create output directory
    :type create_dir_for_path: bool
    :ivar executor: Thread pool for rendering
    :type executor: ThreadPoolExecutor
    :ivar suffix: Suffix to append to output filenames
    :type suffix: str
    """

    def __init__(self, path=None, suffix=""):
        """Initialize render behavior.

        :param path: Output directory path
        :type path: str
        :param suffix: Suffix for output filenames
        :type suffix: str
        """
        super().__init__(name="RenderToDot")
        self.path = path  # This should be a directory.
        self.counter = 0
        self.create_dir_for_path = True
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.suffix = suffix

    def update(self):
        """Render tree on each tick.

        :return: Always returns SUCCESS
        :rtype: :class:`py_trees.common.Status`
        """
        render_now(self)
        return py_trees.Status.SUCCESS


class RenderTreeToDotDecorator(py_trees.decorators.Decorator):
    """Decorator that renders tree when child returns specific status.

    This decorator monitors its child's status and triggers a tree render
    when the status matches configured triggers.

    :ivar path: Output directory path
    :type path: str
    :ivar counter: Number of renders performed
    :type counter: int
    :ivar create_dir_for_path: Whether to create output directory
    :type create_dir_for_path: bool
    :ivar executor: Thread pool for rendering
    :type executor: ThreadPoolExecutor
    :ivar suffix: Suffix to append to output filenames
    :type suffix: str
    :ivar trigger_when_status_is: List of status values that trigger rendering
    :type trigger_when_status_is: list
    """

    def __init__(self, child=None, path=None, suffix="", trigger_when_status_is=None):
        """Initialize render decorator.

        :param child: Child behavior to monitor
        :type child: py_trees.Behaviour
        :param path: Output directory path
        :type path: str
        :param suffix: Suffix for output filenames
        :type suffix: str
        :param trigger_when_status_is: Status values that trigger rendering
        :type trigger_when_status_is: list
        """
        super().__init__(name="RenderToDotDecorator", child=child)

        if trigger_when_status_is is None:
            trigger_when_status_is = [py_trees.Status.SUCCESS, py_trees.Status.FAILURE]
        self.trigger_when_status_is = trigger_when_status_is

        self.path = path  # This should be a directory.
        self.counter = 0
        self.create_dir_for_path = True
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.suffix = suffix

    def update(self):
        """Check child status and render if triggered.

        :return: Status of decorated child
        :rtype: :class:`py_trees.common.Status`
        """
        if self.decorated.status in self.trigger_when_status_is:
            render_now(self)

        return self.decorated.status
