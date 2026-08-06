"""
The colour cycle loose scene objects are drawn in.

Both the onboarder (which bakes colours into a scene bundle) and the live bridge (which
assigns them on the fly) use this one cycle, so the same object keeps its colour whether
the viewer renders the recording or the running world.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List

#: distinct, muted colours that read well against the viewer's dark stage
OBJECT_COLORS = (
    "#f3f0ea",
    "#cf5b3a",
    "#b8bcc4",
    "#e7c26a",
    "#7fb069",
    "#5b8cff",
    "#c98bdb",
    "#ff9d6b",
    "#6bd0c0",
    "#d0c86b",
)


def css_color(red: float, green: float, blue: float) -> str:
    """
    :param red: Red channel in ``[0, 1]``.
    :param green: Green channel in ``[0, 1]``.
    :param blue: Blue channel in ``[0, 1]``.
    :return: The color as a css hex string, e.g. ``#cc6633``.
    """
    return "#%02x%02x%02x" % tuple(
        min(255, max(0, round(channel * 255))) for channel in (red, green, blue)
    )


@dataclass(frozen=True)
class ObjectPalette:
    """
    Assigns object colours by position, wrapping around when it runs out.
    """

    colors: List[str] = field(default_factory=lambda: list(OBJECT_COLORS))
    """
    The colour cycle, in assignment order.
    """

    def color_for(self, index: int) -> str:
        """
        The colour of the object at the given position in the scene.
        """
        return self.colors[index % len(self.colors)]
