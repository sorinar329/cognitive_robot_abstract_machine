"""Enhanced parallel behavior tree components.

This module provides enhanced parallel behavior tree components that improve upon
the standard py_trees parallel implementation. It supports:

* Configurable success policies
* Synchronization options
* Child status management
* Improved interrupt handling

The module is primarily used for:

* Complex parallel task execution
* Synchronized behavior coordination
* Robust failure handling
"""

# thanks to simon
# https://github.com/SemRoCo/giskardpy/blob/devel/src/giskardpy/tree/composites/better_parallel.py

import py_trees.composites
from py_trees.common import Status


class ParallelPolicy(object):
    """Configurable policies for parallel behavior execution.

    This class provides policy configurations that determine how parallel
    behaviors complete based on their children's status.

    :ivar synchronise: Whether to synchronize child execution
    :type synchronise: bool
    """

    class Base(object):
        """Base class for parallel policies.

        .. warning::
           Should never be used directly. Use derived policy classes instead.

        :ivar synchronise: Whether to stop ticking successful children
        :type synchronise: bool
        """
        def __init__(self, synchronise=False):
            """Initialize base policy.

            :param synchronise: Stop ticking successful children until policy met
            :type synchronise: bool
            """
            self.synchronise = synchronise

    class SuccessOnAll(Base):
        """Policy requiring all children to succeed.

        Returns SUCCESS only when each child returns SUCCESS. With synchronization,
        successful children are skipped until all succeed or one fails.

        :ivar synchronise: Whether to stop ticking successful children
        :type synchronise: bool
        """
        def __init__(self, synchronise=True):
            """Initialize SuccessOnAll policy.

            :param synchronise: Stop ticking successful children until all succeed
            :type synchronise: bool
            """
            super().__init__(synchronise=synchronise)

    class SuccessOnOne(Base):
        """Policy requiring only one child to succeed.

        Returns SUCCESS when at least one child succeeds and others are RUNNING.

        :ivar synchronise: Always False for this policy
        :type synchronise: bool
        """
        def __init__(self):
            """Initialize SuccessOnOne policy.

            No configuration needed as synchronization is always disabled.
            """
            super().__init__(synchronise=False)

    class SuccessOnSelected(Base):
        """Policy requiring specific children to succeed.

        Returns SUCCESS when all specified children succeed. With synchronization,
        successful children are skipped until all specified succeed or one fails.

        :ivar synchronise: Whether to stop ticking successful children
        :type synchronise: bool
        :ivar children: List of children that must succeed
        :type children: list
        """
        def __init__(self, children, synchronise=True):
            """Initialize SuccessOnSelected policy.

            :param children: List of children that must succeed
            :type children: list
            :param synchronise: Stop ticking successful children until specified succeed
            :type synchronise: bool
            """
            super().__init__(synchronise=synchronise)
            self.children = children


class Parallel(py_trees.composites.Parallel):
    """Enhanced parallel behavior tree node.

    This class extends py_trees.composites.Parallel with:
    
    * Improved policy handling
    * Better child status management
    * Proper interrupt propagation
    
    .. note::
       Children are ticked in sequence but may run concurrently.
    """

    def tick(self):
        """Tick all children according to policy.

        This method:
        * Initializes if not running
        * Ticks each child according to policy
        * Updates status based on children and policy
        * Handles interrupts for running children

        :yield: Reference to self or children during traversal
        :rtype: :class:`~py_trees.behaviour.Behaviour`
        """
        if self.status != Status.RUNNING:
            # subclass (user) handling
            self.initialise()
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        # process them all first
        for child in self.children:
            if self.policy.synchronise and child.status == Status.SUCCESS:
                continue
            for node in child.tick():
                yield node
        # new_status = Status.SUCCESS if self.policy == common.ParallelPolicy.SUCCESS_ON_ALL else Status.RUNNING
        new_status = Status.RUNNING
        if any([c.status == Status.FAILURE for c in self.children]):
            new_status = Status.FAILURE
        else:
            if isinstance(self.policy, ParallelPolicy.SuccessOnAll):
                if all([c.status == Status.SUCCESS for c in self.children]):
                    new_status = Status.SUCCESS
            elif isinstance(self.policy, ParallelPolicy.SuccessOnOne):
                if any([c.status == Status.SUCCESS for c in self.children]):
                    new_status = Status.SUCCESS
        # special case composite - this parallel may have children that are still running
        # so if the parallel itself has reached a final status, then these running children
        # need to be made aware of it too
        if new_status != Status.RUNNING:
            for child in self.children:
                if child.status == Status.RUNNING:
                    # interrupt it (exactly as if it was interrupted by a higher priority)
                    child.stop(Status.INVALID)
            self.stop(new_status)
        self.status = new_status
        yield self