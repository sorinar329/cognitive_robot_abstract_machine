"""Ground, foundation, tower, entrance and yaw bearing (world frame: origin at
ground level on the tower axis, Z up). Tower geometry from IEA-3.4-130-RWT."""
import math

from turbine import iea

# ------------------------------------------------------------------ ground
GROUND_RADIUS = 70.0
CRANE_PAD_X, CRANE_PAD_Y = (-12.0, 14.0), (6.0, 26.0)     # gravel crane pad beside the tower (estimate)
ROAD_WIDTH = 4.5

# ------------------------------------------------------------------ foundation (estimate)
PLINTH_R = 3.9                   # pedestal above the buried spread footing
PLINTH_Z = (-0.8, 0.24)          # top 0.24 m above ground
GROUT_Z = (0.24, 0.30)
GROUT_R = (2.72, 3.22)

# ------------------------------------------------------------------ tower (IEA)
TOWER_BASE_Z = GROUT_Z[1]
TOWER_Z = iea.TOWER["z"]                                   # 0 .. 108 m above the tower base
TOWER_D = iea.TOWER["outer_diameter"]
TOWER_T = iea.TOWER["wall_thickness"]
TOWER_HEIGHT = TOWER_Z[-1]
TOWER_TOP_Z = TOWER_BASE_Z + TOWER_HEIGHT
TOWER_FLANGES = (TOWER_Z[2], TOWER_Z[5], TOWER_Z[8])     # 4 sections: 21.6 / 32.4 / 32.4 / 21.6 m
TOWER_SEGMENTS = 96


def tower_radius(z):
    """Outer radius at height z above the tower base."""
    for i in range(1, len(TOWER_Z)):
        if z <= TOWER_Z[i]:
            w = (z - TOWER_Z[i - 1]) / (TOWER_Z[i] - TOWER_Z[i - 1])
            return (TOWER_D[i - 1] * (1 - w) + TOWER_D[i] * w) / 2
    return TOWER_D[-1] / 2


def tower_wall(z):
    for i in range(1, len(TOWER_Z)):
        if z <= TOWER_Z[i]:
            w = (z - TOWER_Z[i - 1]) / (TOWER_Z[i] - TOWER_Z[i - 1])
            return TOWER_T[i - 1] * (1 - w) + TOWER_T[i] * w
    return TOWER_T[-1]


# base flange (T-flange) with the anchor bolt nuts outside the tower wall
BASE_FLANGE_R_OUT, BASE_FLANGE_H = 3.32, 0.12
ANCHOR_NUT_R, ANCHOR_NUTS = 3.18, 96
FLANGE_WIDTH, FLANGE_H = 0.22, 0.14                         # internal L-flanges at section joints

# entrance: door facing -Y (azimuth -90 deg), outside stairs with a landing
DOOR_AZIMUTH = -math.pi / 2
DOOR_WIDTH, DOOR_HEIGHT = 0.9, 2.1
DOOR_SILL_Z = 0.96                                           # above the tower base
STEP_RISE, STEP_RUN = 0.18, 0.28
LANDING_DEPTH, STAIR_WIDTH = 1.4, 1.2

# earthing strap from the base flange to a terminal on the plinth (azimuth -120 deg)
EARTHING_AZIMUTH = -2 * math.pi / 3

# ------------------------------------------------------------------ yaw bearing
YAW_BEARING_H = 0.35
YAW_BEARING_R = (1.25, 1.62)
NACELLE_ORIGIN_Z = TOWER_TOP_Z + YAW_BEARING_H               # nacelle floor on the yaw axis
