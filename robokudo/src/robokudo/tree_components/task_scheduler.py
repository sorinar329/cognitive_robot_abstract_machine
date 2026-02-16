"""Task scheduling components for behavior trees.

This module provides base classes for implementing task schedulers in behavior trees.
Task schedulers are responsible for dynamically arranging and managing behavior tree
nodes during execution.

The module supports:

* Dynamic behavior arrangement
* Task scheduling policies
* Job sequence management
* Tree structure validation
"""

import logging
from typing import Optional

import py_trees
from py_trees.composites import Sequence

import robokudo.utils.tree
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.tree import add_child_to_parent, fix_parent_relationship_of_childs


class TaskSchedulerBase(py_trees.behaviour.Behaviour):
    """Base class for task scheduling behaviors.

    This Behaviour enables a dynamic arrangement of known Behaviours.
    It assumes that it is placed in a certain configuration in a behaviour tree:
    .. code-block:: text
                  JOB_SCHEDULING [SEQUENCE]
                 /             |
          JOB_SCHEDULER      JOB [SEQUENCE]
                               |

    During startup this class will save the Job Sequence which contains as a (direct) children all
    the Annotators that might need to get scheduled.

    ... note::
    In order to use this class, please use one of the deriving classes.

    :ivar fix_parent_relationships_after_plan: Whether to fix parent relationships after planning
    :type fix_parent_relationships_after_plan: bool
    """

    def __init__(self, name: str="TaskSchedulerBase"):
        """Initialize the task scheduler.

        :param name: Name of the scheduler node
        :type name: str
        """
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)
        self.fix_parent_relationships_after_plan: bool = True

    def initialise(self) -> None:
        """Initialize and validate tree structure.

        Performs sanity checks to ensure the scheduler is in a correctly
        configured environment.
        """
        self.logger.debug("%s.initialise()" % self.__class__.__name__)

        # Make sanity checks that we are in a correctly configured environment
        parent = self.parent
        assert (isinstance(parent, Sequence))

        assert (parent.children[0] == self)

    def plan_new_job(self) -> Optional[Sequence]:
        """Get the new job that should be applied by the JobScheduler.

        It is the responsibility of your method to return a valid py_trees.Sequence.
        This means especially that you have to make sure that your parent and children relations
        should be intact. This is important if you have to keep Instances of your Behaviours/Annotators
        which might get changed when being put into different py_trees.Behaviours.

        :return: py_trees.Sequence if it can be computed or None if no plan could be found.
        :rtype: Optional[py_trees.Sequence]
        """
        return None

    @catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:
        """Update the scheduler state.

        Called every time the behavior is ticked.

        This will happen only once for the job scheduling.

        :return: SUCCESS if job planned and added, FAILURE otherwise
        :rtype: :class:`py_trees.common.Status`
        """

        parent = self.parent
        assert (isinstance(parent, Sequence))

        new_job = self.plan_new_job()

        if new_job is None:
            self.logger.debug("Couldn't find solution for Job Scheduling. Aborting...")
            self.feedback_message = "Couldn't find solution for Job Scheduling. Aborting..."
            raise Exception(self.feedback_message)

        if len(parent.children) > 1:
            parent.remove_child(parent.children[1])  # remove the old job, if existing

        if self.fix_parent_relationships_after_plan:
            robokudo.utils.tree.fix_parent_relationship_of_childs(new_job)
        parent.add_child(new_job)  # add new job with instances from the original one

        return py_trees.common.Status.SUCCESS


class IterativeTaskScheduler(TaskSchedulerBase):
    """Task scheduler that cycles through a list of subtrees.

    A Task Scheduler that cycles iteratively through a list of given subtrees.
    Repeats from the beginning after the end of the list is reached.

    :ivar tree_list: List of subtrees to cycle through
    :type tree_list: list
    :ivar idx: Current index in tree_list
    :type idx: int
    """

    def __init__(self, name: str ="IterativeTaskScheduler", tree_list: list = []):
        """Initialize the iterative scheduler.

        :param name: Name of the scheduler node
        :type name: str
        :param tree_list: List of subtrees to cycle through
        :type tree_list: list
        """
        super().__init__(name)
        self.tree_list: list = tree_list

        # Save a reference to the initial job with all the possible sub-behaviours
        self.idx: int = 0

    def setup(self, timeout: float) -> bool:
        """Set up all trees in the list.

        TODO Since we might have the same node in multiple trees, we might call setup multiple times
            => Find a way to get around this

        .. note::
           Since nodes may appear in multiple trees, setup may be called
           multiple times on the same node.

        :param timeout: Maximum time allowed for setup
        :type timeout: float
        :return: True if setup successful
        :rtype: bool
        """
        for tree in self.tree_list:
            robokudo.utils.tree.setup_with_descendants_on_behavior(tree)
        return True

    def plan_new_job(self) -> Optional[Sequence]:
        """Plan the next job by selecting the next tree in sequence.

        :return: New job sequence with next tree, or None if tree_list empty
        :rtype: Optional[py_trees.Sequence]
        """
        parent = self.parent
        assert (isinstance(parent, Sequence))

        assert (len(self.tree_list) > 0)

        new_job = Sequence(name="Task", memory=True)
        new_subtree = self.tree_list[self.idx]
        add_child_to_parent(new_job, new_subtree)
        fix_parent_relationship_of_childs(new_job)

        if self.idx < len(self.tree_list) - 1:
            self.idx += 1
        else:
            self.idx = 0

        return new_job
