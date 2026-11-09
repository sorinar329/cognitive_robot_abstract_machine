#!/usr/bin/env python
"""The IJCAI expo demo with failures as they come: the Panda stacks four cubes
into a tower in MuJoCo, and only some cubes are slippery.

Each cube's first attempt samples its friction: most cubes grip well, a share of them
are slippery. A slippery cube slides out of the fingers when it is lifted or drops out of them
right after; a cube that grips well is stacked whatever the speeds. A failed
attempt is diagnosed and retried as in :mod:`inference3b`.

Run it from this folder::

    python varied_failures.py --iterations 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from inference3b import (
    DiagnosedStackingDemonstration,
    ParameterPrior,
    StackingParameters,
)

# %% the frictions


@dataclass
class CubeFriction:
    """How slippery the cubes of a tower are: most grip well, a share of them
    are slippery."""

    slippery_share: float = 0.3
    """The share of cubes that are slippery."""

    gripping: ParameterPrior = field(
        default_factory=lambda: ParameterPrior(1.0, 0.3, 0.5, 1.5)
    )
    """The friction of a cube that grips well."""

    slippery: ParameterPrior = field(
        default_factory=lambda: ParameterPrior(0.065, 0.01, 0.05, 0.075)
    )
    """The friction of a slippery cube, low enough that the cube is lost before
    it is carried away.

    .. note:: From about 0.075 on, a slippery cube may twist in the fingers and land
       tilted, or be dropped somewhere out of reach, which this demonstration leaves out.
    """


# %% the demonstration


@dataclass
class VariedFailuresDemonstration(DiagnosedStackingDemonstration):
    """Stacks the cubes with a realistic friction for each, so that only some
    attempts fail and they fail in different ways."""

    friction: CubeFriction = field(default_factory=CubeFriction)
    """How slippery the cubes are."""

    report: Path = Path(__file__).parent / "varied_failures_report.md"

    parks_while_carrying: bool = False
    """The cube is carried straight from the pick-up to the tower."""

    def sampled_parameters(self) -> StackingParameters:
        """Extends :meth:`DiagnosedStackingDemonstration.sampled_parameters` by
        giving the cube a gripping or a slippery friction instead of an always
        slippery one."""
        parameters = super().sampled_parameters()
        prior = (
            self.friction.slippery
            if self.generator.random() < self.friction.slippery_share
            else self.friction.gripping
        )
        parameters.pick_up["object_friction"] = prior.sample(self.generator)
        return parameters


def main() -> None:
    """Run the demonstration from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--live-port", type=int, default=0)
    arguments = parser.parse_args()
    VariedFailuresDemonstration(
        iterations=arguments.iterations,
        seed=arguments.seed,
        headless=arguments.headless,
        live_port=arguments.live_port,
    ).run()


if __name__ == "__main__":
    main()
