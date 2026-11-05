"""
When the world SegMind ticks was observed, so that what happens is read over time rather
than over ticks.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ObservationClock(ABC):
    """
    Tells when the state of the world about to be ticked was observed.
    """

    @abstractmethod
    def observe(self) -> float:
        """
        Observe the world for the tick about to happen.

        :return: When the state about to be ticked was observed, in seconds on a clock
            that never goes backwards.
        """


@dataclass
class TimeOfTick(ObservationClock):
    """
    Takes the world to be observed the moment it is ticked, as when it is watched while
    it changes.
    """

    def observe(self) -> float:
        return time.monotonic()


@dataclass
class ClockAdvancingEachTick(ObservationClock):
    """
    Takes every tick to observe the world a fixed time after the one before, as when a
    world is stepped tick by tick rather than watched while it changes.
    """

    seconds_per_tick: float = 0.5
    """
    The time between the observations of two ticks in a row.
    """

    ticks: int = field(default=0, init=False)
    """
    How many ticks have observed the world so far.
    """

    def observe(self) -> float:
        self.ticks += 1
        return self.ticks * self.seconds_per_tick
