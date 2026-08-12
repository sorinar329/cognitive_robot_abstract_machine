"""
Gating of segmind's support detection on the arm having reached its park position.

A finished plan means the parking motion was commanded and the kinematic belief
considers it converged, which the simulated arm still trails. Sampling support relations
at that point reads a scene that is still moving, so ``inference.py`` waits here first
and only then asks the detector what it sees.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from coraplex.datastructures.enums import Arms
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom


class ArmParkDeviations(ABC):
    """
    Reports how far an arm's joints currently sit from their park targets.
    """

    @abstractmethod
    def deviations_from_park_targets(self) -> list[float]:
        """
        :return: Each arm joint's absolute distance from its park target, in radians.
        """


@dataclass
class RobotArmParkDeviations(ArmParkDeviations):
    """
    Measures the deviations against a robot's live joint positions.
    """

    world: World
    """
    The world whose state carries the simulated joint positions.
    """

    robot: AbstractRobot
    """
    The robot whose arm joints are compared against their park targets.
    """

    @property
    def park_targets(self) -> dict[DegreeOfFreedom, float]:
        """
        Each arm joint's park position, keyed by the degree of freedom carrying it.

        Read from the same joint states
        :class:`~coraplex.robot_plans.actions.core.robot_body.ParkArmsAction` commands,
        so the target waited for and the target driven to stay in step.
        """
        targets = {}
        for arm_view in ViewManager.get_all_arm_views(Arms.BOTH, self.robot):
            joint_state = arm_view.get_joint_state_by_type(StaticJointState.PARK)
            for connection, target in zip(
                joint_state.connections, joint_state.target_values
            ):
                targets[connection.raw_dof] = target
        return targets

    def deviations_from_park_targets(self) -> list[float]:
        return [
            abs(self.world.state[degree_of_freedom.id].position - target)
            for degree_of_freedom, target in self.park_targets.items()
        ]


@dataclass
class ParkedArmDetectionGate:
    """
    Holds support detection back until the arm has parked and the scene has settled.
    """

    arm: ArmParkDeviations
    """
    Reports the arm's distance from its park targets.
    """

    position_tolerance: float = 0.05
    """
    How far (in radians) a joint may sit from its park target and still count as
    arrived.
    """

    arrival_timeout: float = 5.0
    """
    Seconds spent waiting for the arm to arrive before giving up on it.
    """

    settle_time: float = 0.5
    """
    Seconds to keep waiting once the wait for the arm is over, so the cubes are read
    after they have stopped moving too.
    """

    poll_interval: float = 0.05
    """
    Seconds between two checks of the arm's joint positions while waiting for arrival.
    """

    def arm_has_arrived(self) -> bool:
        """
        Whether every arm joint currently sits within :attr:`position_tolerance` of its
        park target.
        """
        return all(
            deviation <= self.position_tolerance
            for deviation in self.arm.deviations_from_park_targets()
        )

    def wait_for_parked_arm(self) -> bool:
        """
        Block until the arm has parked and the scene has settled.

        The settling time is spent either way, so a caller that proceeds despite a miss
        still reads a scene that had a moment to come to rest.

        :return: Whether the arm arrived within :attr:`arrival_timeout`.
        """
        arrived = self._wait_for_arrival()
        time.sleep(self.settle_time)
        return arrived

    def _wait_for_arrival(self) -> bool:
        """
        Poll the arm's joint positions until they reach their park targets.

        :return: Whether the arm arrived before :attr:`arrival_timeout` ran out.
        """
        deadline = time.time() + self.arrival_timeout
        while not self.arm_has_arrived():
            if time.time() >= deadline:
                return False
            time.sleep(self.poll_interval)
        return True
