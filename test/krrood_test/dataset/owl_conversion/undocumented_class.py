from __future__ import annotations

from dataclasses import dataclass


# The class below deliberately has no docstring: it stands for a class whose only
# documentation is the signature ``dataclass`` writes into ``__doc__``.
@dataclass
class UndocumentedRoot:
    size: int = 0
