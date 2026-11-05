"""
The state of a world kept in memory that another process reads while it is written.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np
from typing_extensions import Self


class SharedWorldStateHeader(enum.IntEnum):
    """
    The numbers kept ahead of the state, by their position.
    """

    WRITES = 0
    """
    How many writes have begun and how many have ended, together: odd while one is
    under way.
    """

    REPLACED = 1
    """
    Not zero once the state is kept in other memory, because the world changed shape.
    """


@dataclass(eq=False)
class SharedWorldState:
    """
    The state of a world in memory shared between processes, which one process writes
    and others read at the same time without ever reading half of a write.

    .. note:: One writer at a time: whoever writes from several threads holds a lock.
    """

    memory: shared_memory.SharedMemory
    """
    The memory the header and the state are kept in.
    """

    shape: tuple[int, int]
    """
    The shape of the state: derivatives by degrees of freedom.
    """

    @classmethod
    def create(cls, shape: tuple[int, int]) -> Self:
        """
        :param shape: The shape of the state to be kept.
        :return: The state in newly created memory, all zero.
        """
        numbers = len(SharedWorldStateHeader) + shape[0] * shape[1]
        memory = shared_memory.SharedMemory(
            create=True, size=numbers * np.dtype(float).itemsize
        )
        return cls(memory=memory, shape=tuple(shape))

    @classmethod
    def attach(cls, name: str, shape: tuple[int, int]) -> Self:
        """
        :param name: The name of memory another process created.
        :param shape: The shape of the state kept in it.
        :return: The state kept in that memory.
        """
        return cls(memory=shared_memory.SharedMemory(name=name), shape=tuple(shape))

    @property
    def name(self) -> str:
        """
        The name another process attaches to the memory by.
        """
        return self.memory.name

    @property
    def _numbers(self) -> np.ndarray:
        """
        The header followed by the state, as the numbers kept in the memory.
        """
        return np.ndarray(
            (len(SharedWorldStateHeader) + self.shape[0] * self.shape[1],),
            dtype=float,
            buffer=self.memory.buf,
        )

    def write(self, data: np.ndarray) -> None:
        """
        Keep ``data`` as the state.
        """
        numbers = self._numbers
        numbers[SharedWorldStateHeader.WRITES] += 1
        numbers[len(SharedWorldStateHeader) :] = data.ravel()
        numbers[SharedWorldStateHeader.WRITES] += 1

    def read(self) -> np.ndarray:
        """
        :return: A copy of the state written last, whole.
        """
        numbers = self._numbers
        while True:
            writes_before = numbers[SharedWorldStateHeader.WRITES]
            if writes_before % 2 == 1:
                continue
            data = numbers[len(SharedWorldStateHeader) :].reshape(self.shape).copy()
            if numbers[SharedWorldStateHeader.WRITES] == writes_before:
                return data

    @property
    def is_replaced(self) -> bool:
        """
        Whether the state is kept in other memory now.
        """
        return bool(self._numbers[SharedWorldStateHeader.REPLACED])

    def mark_replaced(self) -> None:
        """
        Tell every reader that the state is kept in other memory from now on.
        """
        self._numbers[SharedWorldStateHeader.REPLACED] = 1

    def close(self) -> None:
        """
        Let go of the memory in this process.
        """
        self.memory.close()

    def unlink(self) -> None:
        """
        Free the memory once every process has let go of it.
        """
        self.memory.unlink()
