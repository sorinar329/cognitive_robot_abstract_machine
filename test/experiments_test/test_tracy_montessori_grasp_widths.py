"""
Tests for :mod:`experiments.tracy_experiments.montessori.grasp_widths`: the close
setpoint picked for a shape is its override when it has one, and the shared default
otherwise, and a table for smaller pieces closes the fingers proportionally further.
"""

from __future__ import annotations

import pytest

from experiments.montessori.semantics import MontessoriShapeCategory
from experiments.tracy_experiments.montessori.grasp_widths import (
    DEFAULT_CLOSE_SETPOINT,
    RECTANGULAR_PRISM_CLOSE_SETPOINT,
    GraspCloseTable,
)
from experiments.tracy_experiments.montessori.gripper_feedback import (
    FULLY_CLOSED_KNUCKLE_POSITION,
)
from experiments.tracy_experiments.robotiq_gripper import FingerSetpoint

PIECE_SCALE = 0.8
"""
An arbitrary factor below one to scale the pieces by.
"""


def test_default_close_setpoint_is_the_controllers_own_closed_setpoint():
    assert DEFAULT_CLOSE_SETPOINT == float(FingerSetpoint.CLOSED)


def test_a_shape_without_an_override_gets_the_default_setpoint():
    table = GraspCloseTable()

    assert table.setpoint_for(MontessoriShapeCategory.CUBE) == DEFAULT_CLOSE_SETPOINT
    assert (
        table.setpoint_for(MontessoriShapeCategory.CYLINDER) == DEFAULT_CLOSE_SETPOINT
    )
    assert (
        table.setpoint_for(MontessoriShapeCategory.TRIANGULAR_PRISM)
        == DEFAULT_CLOSE_SETPOINT
    )


def test_the_rectangular_prism_is_closed_further_than_the_default():
    table = GraspCloseTable()

    setpoint = table.setpoint_for(MontessoriShapeCategory.RECTANGULAR_PRISM)

    assert setpoint == RECTANGULAR_PRISM_CLOSE_SETPOINT
    assert setpoint > DEFAULT_CLOSE_SETPOINT


def test_overrides_can_be_supplied_explicitly():
    table = GraspCloseTable(
        default_setpoint=0.4, overrides={MontessoriShapeCategory.SPHERE: 0.55}
    )

    assert table.setpoint_for(MontessoriShapeCategory.SPHERE) == 0.55
    assert table.setpoint_for(MontessoriShapeCategory.CUBE) == 0.4


# %% smaller pieces


def test_a_table_for_smaller_pieces_leaves_a_proportionally_smaller_opening():
    table = GraspCloseTable()

    scaled = table.for_pieces_scaled_by(PIECE_SCALE)

    for category in (
        MontessoriShapeCategory.CUBE,
        MontessoriShapeCategory.RECTANGULAR_PRISM,
    ):
        assert FULLY_CLOSED_KNUCKLE_POSITION - scaled.setpoint_for(
            category
        ) == pytest.approx(
            PIECE_SCALE * (FULLY_CLOSED_KNUCKLE_POSITION - table.setpoint_for(category))
        )
