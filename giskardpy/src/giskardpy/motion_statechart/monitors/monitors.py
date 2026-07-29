from __future__ import annotations

import abc
from abc import ABC
from dataclasses import field
from typing_extensions import List, Optional

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.utils.decorators import dataclass
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom


def local_minimum_expression(
    dofs: List[DegreeOfFreedom],
    context: MotionStatechartContext,
    joint_convergence_threshold: float = 0.01,
    minimum_threshold: float = 0.01,
    maximum_threshold: float = 0.06,
    min_time: float = 1.0,
    reference_cycle_variable: Optional[FloatVariable] = None,
) -> sm.Scalar:
    """
    Build the "has settled" observation expression shared by
    :class:`LocalMinimumReached` and any task that wants to tolerate stalling (e.g. a
    gripper closing on a grasped object): true once at least.

    ``min_time`` seconds have elapsed and every DOF in ``dofs`` has a
    velocity below its own ``max_velocity * joint_convergence_threshold``
    (clamped to ``[minimum_threshold, maximum_threshold]``).

    :param dofs: The degrees of freedom to check.
    :param context: The motion statechart context, for the control timestep
        and elapsed-cycle-count symbol.
    :param joint_convergence_threshold: See :attr:`LocalMinimumReached.joint_convergence_threshold`.
    :param minimum_threshold: See :attr:`LocalMinimumReached.minimum_threshold`.
    :param maximum_threshold: See :attr:`LocalMinimumReached.maximum_threshold`.
    :param min_time: Minimum elapsed control time before the observation can become true.
    :param reference_cycle_variable: Cycle count elapsed time is measured from,
        instead of the start of the whole motion chart. Pass a variable a
        caller updates in its own ``on_start`` (e.g. a per-task "I started
        running at cycle N" marker) so ``min_time`` gates on how long *that
        caller* has been active, not on how many cycles the entire chart --
        including earlier, unrelated tasks -- has already ticked through.
        ``None`` keeps the previous chart-wide behaviour.
    :return: The observation expression.
    """
    ref = []
    symbols = []
    for dof in dofs:
        velocity_limit = dof.limits.upper.velocity
        velocity_limit *= joint_convergence_threshold
        velocity_limit = min(max(minimum_threshold, velocity_limit), maximum_threshold)
        ref.append(velocity_limit)
        symbols.append(dof.variables.velocity)
    ref = sm.Vector(ref)
    vel_symbols = sm.Vector(symbols)

    dt = (
        context.qp_controller_config.control_dt
        or context.qp_controller_config.model_predictive_control_time_step
    )
    elapsed_cycles = context.control_cycle_variable
    if reference_cycle_variable is not None:
        elapsed_cycles = elapsed_cycles - reference_cycle_variable
    traj_longer_than_min_time = elapsed_cycles * dt > min_time
    return sm.trinary_logic_and(
        traj_longer_than_min_time, sm.logic_all(sm.abs(vel_symbols) < ref)
    )


@dataclass
class ThreadedPayloadMonitor(MotionStatechartNode, ABC):
    """
    A monitor which executes its __call__ function when start_condition becomes True.

    Subclass this and implement __init__.py and __call__. The __call__ method should
    change self.state to True when it's done. Calls __call__ in a separate thread. Use
    for expensive operations
    """

    state: ObservationStateValues = field(
        init=False, default=ObservationStateValues.UNKNOWN
    )

    @abc.abstractmethod
    def __call__(self):
        pass


@dataclass
class LocalMinimumReached(MotionStatechartNode):
    """
    Checks if the robot has reached a local minimum in the trajectory, by checking if
    all velocities are below a degree of freedoms' max velocity
    *`joint_convergence_threshold`.
    """

    joint_convergence_threshold: float = 0.01
    """
    If a degree of freedom velocity is below its maximum velocity * this value, it is
    considered as not moving.
    """

    minimum_threshold: float = 0.01
    """
    Minimum value for degree of freedom velocity * joint_convergence_threshold.
    """

    maximum_threshold: float = 0.06
    """
    Maximum value for degree of freedom velocity * joint_convergence_threshold.
    """

    windows_size: int = 1
    """
    Windows size for joint convergence check.
    """

    dofs: Optional[List[DegreeOfFreedom]] = None
    """
    Degrees of freedom to check for convergence.

    Defaults to ``context.world.active_degrees_of_freedom`` (every active DOF) if left
    ``None``.
    """

    min_time: float = 1.0
    """
    Minimum elapsed control time before the observation can become true.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        dofs = (
            self.dofs if self.dofs is not None else context.world.active_degrees_of_freedom
        )
        artifacts.observation = local_minimum_expression(
            dofs=dofs,
            context=context,
            joint_convergence_threshold=self.joint_convergence_threshold,
            minimum_threshold=self.minimum_threshold,
            maximum_threshold=self.maximum_threshold,
            min_time=self.min_time,
        )
        return artifacts
