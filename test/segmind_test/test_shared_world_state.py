"""
Tests for the state of a world kept in memory that another process reads while it is
written.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from segmind.shared_world_state import SharedWorldState

LARGE_STATE_SHAPE = (4, 250_000)
"""
A state big enough that copying it takes long enough for a reader to come in between.
"""

WRITES = 200
"""
How many states a test writes while another thread reads.
"""


@pytest.fixture
def shared_state():
    """
    A state of the large shape in newly created memory, freed afterwards.
    """
    state = SharedWorldState.create(LARGE_STATE_SHAPE)
    yield state
    state.close()
    state.unlink()


# %% reading what was written


def test_a_state_is_read_back_as_it_was_written(shared_state):
    """
    A state read is the state written last.
    """
    written = np.arange(np.prod(LARGE_STATE_SHAPE), dtype=float).reshape(
        LARGE_STATE_SHAPE
    )

    shared_state.write(written)

    assert np.array_equal(shared_state.read(), written)


def test_a_state_read_while_it_is_written_is_never_half_written(shared_state):
    """
    Every state written is the same number throughout, so a read that mixes two writes
    holds two numbers.
    """
    writing = threading.Thread(
        target=lambda: [
            shared_state.write(np.full(LARGE_STATE_SHAPE, float(number)))
            for number in range(1, WRITES + 1)
        ]
    )

    writing.start()
    mixed_reads = 0
    while writing.is_alive():
        read = shared_state.read()
        mixed_reads += int(read.min() != read.max())
    writing.join()

    assert mixed_reads == 0


def test_a_state_read_from_another_handle_on_the_same_memory_is_the_one_written(
    shared_state,
):
    """
    A reader attached to the memory by its name reads what the writer wrote.
    """
    written = np.full(LARGE_STATE_SHAPE, 3.0)
    shared_state.write(written)
    other_handle = SharedWorldState.attach(shared_state.name, LARGE_STATE_SHAPE)

    read = other_handle.read()
    other_handle.close()

    assert np.array_equal(read, written)


# %% replacing the memory


def test_a_state_is_not_replaced_until_it_is_marked_so(shared_state):
    """
    New memory holds the current state until it is marked replaced.
    """
    assert not shared_state.is_replaced


def test_a_state_marked_replaced_is_seen_replaced_from_another_handle(shared_state):
    """
    A reader learns from the memory itself that the state moved elsewhere.
    """
    other_handle = SharedWorldState.attach(shared_state.name, LARGE_STATE_SHAPE)

    shared_state.mark_replaced()
    replaced = other_handle.is_replaced
    other_handle.close()

    assert replaced
