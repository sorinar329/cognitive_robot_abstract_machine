"""Hub, spinner and blades (IEA-3.4-130-RWT). Frame ``hub``: on the rotor axis at
the hub centre, X along the (tilted) axis pointing upwind."""
import math

from turbine import iea
from turbine.dims import site

HUB_CENTER_ABOVE_TOWER_TOP = iea.DRIVETRAIN["distance_tt_hub"]          # 2.0 m
OVERHANG = iea.DRIVETRAIN["overhang"]                                     # 5.0 m
TILT = math.radians(iea.DRIVETRAIN["uptilt"])                             # 5 deg
CONE = math.radians(iea.CONE_ANGLE_DEG)                                   # 3 deg
HUB_R = iea.HUB_DIAMETER / 2                                              # 2.0 m: blade root plane
HUB_IN_NACELLE = (OVERHANG, 0.0, HUB_CENTER_ABOVE_TOWER_TOP - site.YAW_BEARING_H)

# spinner (estimate): surface of revolution around the hub X axis
SPINNER_X = (-2.1, 2.9)
SPINNER_R = 2.25
BLADE_OPENING_R = 1.45
PITCH_BEARING_R = (1.12, 1.42)                                           # ring in the root plane
CASTING_R = 1.7

BLADES = 3
BLADE_LENGTH = iea.BLADE["ref_axis_z"][-1]                                # 63 m
PITCH_LIMITS = (math.radians(-5), math.radians(95))
