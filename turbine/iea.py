"""IEA-3.4-130-RWT reference data (references/iea34/geometry.json), plain JSON so
Blender's Python can read it too."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
with open(os.path.join(ROOT, "references", "iea34", "geometry.json")) as _f:
    DATA = json.load(_f)

BLADE = DATA["blade"]
AIRFOILS = DATA["airfoils"]
TOWER = DATA["tower"]
DRIVETRAIN = DATA["drivetrain"]
HUB_DIAMETER = DATA["hub_diameter"]
CONE_ANGLE_DEG = DATA["cone_angle_deg"]
HUB_HEIGHT = DATA["hub_height"]


def interpolate(grid, values, t):
    """Piecewise-linear value at normalized position t on grid."""
    if t <= grid[0]:
        return values[0]
    for i in range(1, len(grid)):
        if t <= grid[i]:
            w = (t - grid[i - 1]) / (grid[i] - grid[i - 1])
            return values[i - 1] * (1 - w) + values[i] * w
    return values[-1]
