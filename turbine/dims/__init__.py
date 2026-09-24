"""Single source of truth for all dimensions (metres, radians).

Imported by the Blender part scripts (geometry) and by the URDF generator
(joint origins, collision primitives), so both always agree.

This module holds the nacelle interior (frame ``nacelle``: origin on the yaw
axis at floor level, X upwind, Z up). ``site`` covers ground, foundation, tower
and yaw bearing, ``rotor`` the hub and blades. Values from the IEA-3.4-130-RWT
reference are marked IEA; everything else is an estimate for a 3-3.5 MW turbine.
"""

# ------------------------------------------------------------------ nacelle
NACELLE_X = (-8.8, 2.8)       # estimate: ~11.6 m long; front ends just behind the spinner
NACELLE_Y = (-2.1, 2.1)        # estimate: 4.2 m wide
NACELLE_Z = (0.0, 4.2)         # estimate: 4.2 m high
WALL = 0.06
FLOOR = 0.08
INNER_X = (NACELLE_X[0] + WALL, NACELLE_X[1] - WALL)
INNER_Y = (NACELLE_Y[0] + WALL, NACELLE_Y[1] - WALL)

SHAFT_Z = 1.5                 # main shaft / gearbox input axis height (level; the IEA 5 deg tilt comes with the nacelle rework)
SHAFT_HOLE_R = 0.75
ROOF_HATCH_X, ROOF_HATCH_Y = (-5.6, -4.6), (-0.5, 0.5)
FLOOR_HATCH_X, FLOOR_HATCH_Y = (-1.2, -0.5), (-1.97, -1.33)   # beside the bedplate girders, over the tower ladder
HATCH_THICKNESS = 0.03

# ------------------------------------------------------------------ bedplate
# Frame ``bedplate`` is the nacelle frame shifted by DRIVETRAIN_OFFSET_X: the
# drivetrain below (sized for a V90-like nacelle) sits 0.9 m further back so it
# stays behind the IEA hub. Replaced by the IEA drivetrain in the nacelle step.
DRIVETRAIN_OFFSET_X = -0.9
# Two side girders carry the main
# bearing (front block) and the torque arm brackets; between them a low deck
# with the cable opening on the yaw axis. The gearbox hangs down between them.
BP_GIRDER_X, BP_GIRDER_Y = (-0.3, 3.0), (0.9, 1.25)    # Y mirrored on both sides
BP_TOP_Z = 0.7
BP_FRONT_X = (2.2, 3.0)
BP_DECK_X, BP_DECK_Z = (-0.3, 2.2), 0.25
BP_CABLE_HOLE_R = 0.25
BP_BRACKET_XS = ((-0.34, -0.24), (0.84, 0.94))          # pin plates, gap in front of the bushings for inspection
BP_BRACKET_HALF_Y = 0.12

# ------------------------------------------------------------------ main bearing, main shaft, rotor lock
# Frame ``main_bearing``: bearing centre on the shaft axis, X upwind.
# ``main_shaft`` (continuous about X) and ``rotor_lock`` share this origin.
MB_XYZ = (2.65, 0.0, SHAFT_Z)
MB_R, MB_BORE_R, MB_HALF_W = 0.62, 0.37, 0.25
MB_FOOT_Y, MB_FOOT_Z = 0.85, (BP_TOP_Z - SHAFT_Z, BP_TOP_Z - SHAFT_Z + 0.35)
MB_SEAL_R = 0.46
MB_COLLECTOR_XY, MB_COLLECTOR_R, MB_COLLECTOR_H = (0.12, -0.7), 0.06, 0.18   # stands on the foot
MB_GREASE_OK, MB_GREASE_FULL = 0.25, 0.95                                    # fill fraction

MS_FLANGE_X, MS_FLANGE_R = (0.97, 1.09), 0.66          # hub flange, just outside the cover
MS_SEGMENTS = [                                          # (radius, x range) stepped shaft
    (0.40, (0.80, 0.97)), (0.36, (0.26, 0.80)), (0.36, (-0.26, 0.26)),
    (0.33, (-1.45, -0.26)), (0.26, (-1.65, -1.45)),
]
MS_LOCK_DISC_X, MS_LOCK_DISC_R = (0.6, 0.7), 0.62
MS_LOCK_HOLE_R, MS_LOCK_HOLE_PCD, MS_LOCK_HOLES = 0.05, 0.5, 12
MS_SHRINK_X, MS_SHRINK_R = (-1.62, -1.44), 0.45          # on the gearbox input stub
MS_SHRINK_BOLT_PCD, MS_SHRINK_BOLTS = 0.37, 16
MS_LOOSE_BOLT = 4                                          # index of the bolt that backs out

# rotor lock: housing on a bracket on the bearing foot, pin moves +X into the disc
# holes; at shaft angle 0 one hole sits exactly at (y=RL_Y, z=0)
# the 0.1 m gap between housing and disc shows whether the pin is engaged
RL_HOUSING_X, RL_Y, RL_HALF = (0.28, 0.5), -0.5, 0.12
RL_PIN_R, RL_PIN_LEN, RL_STROKE = 0.045, 0.2, 0.18

# ------------------------------------------------------------------ gearbox
# Frame ``gearbox``: on the input axis at the front face of the input stub,
# X upwind. Mounted at GEARBOX_XYZ in the nacelle frame.
GEARBOX_XYZ = (1.0, 0.0, SHAFT_Z)          # in the bedplate frame
GB_INPUT_R, GB_INPUT_X = 0.28, (0.0, 0.2)          # hollow input shaft stub
GB_FRONT_COVER_R, GB_FRONT_COVER_X = 0.8, (-0.15, 0.0)
GB_PLANET_R, GB_PLANET_X = 0.95, (-1.2, -0.15)       # planetary stage housing
GB_HELICAL_X, GB_HELICAL_Y, GB_HELICAL_Z = (-2.3, -1.15), (-0.7, 0.7), (-0.8, 0.8)
GB_OUTPUT_Z = 0.3                                   # high-speed output axis offset
GB_OUTPUT_R, GB_OUTPUT_X = 0.1, (-2.5, -2.3)
# torque arms (both sides) and their elastomer bushings (axis along X)
GB_ARM_X, GB_ARM_Z = (-0.95, -0.45), (-0.3, 0.2)
GB_ARM_Y_OUT = 1.32        # leaves ~0.37 m of walkway beside the arms
GB_ARM_Y_IN = 0.7
GB_BUSHING_R, GB_BUSHING_Y, GB_BUSHING_Z = 0.14, 1.12, -0.05
GB_PIN_R, GB_PIN_EXT = 0.05, 0.3                  # pin sticks out into the bedplate brackets
GB_RIB_X = (-0.3, -1.05)                           # clear of the torque arms
GB_BUSHING_X = (-1.0, -0.4)
# oil cooler on top of the helical stage
GB_COOLER_X, GB_COOLER_Y, GB_COOLER_Z = (-2.25, -1.8), (-0.5, 0.5), (0.8, 1.25)
# bolted inspection cover on top of the helical stage
GB_COVER_X, GB_COVER_Y, GB_COVER_T = (-1.75, -1.25), (-0.35, 0.35), 0.03
GB_COVER_BOLT_R = 0.018
# oil sight glass on the -Y face of the helical stage
GB_GLASS_XZ, GB_GLASS_R = (-1.7, -0.45), 0.065
GB_OIL_LEVEL_OK, GB_OIL_LEVEL_LOW = 0.6, 0.12       # fraction of the glass
# offline oil filter on the -Y side, clogging indicator on top
GB_FILTER_XY, GB_FILTER_R, GB_FILTER_Z = (-2.05, -0.86), 0.11, (-0.35, 0.3)

# ------------------------------------------------------------------ roof: weather mast (estimate)
MAST_X = NACELLE_X[0] + 0.9
MAST_HEIGHT = 1.6
MAST_ARM = 0.55                 # crossbar half length (anemometer at +Y, wind vane at -Y)
COVER_CRACK_XZ = (-5.8, 2.6)    # on the -Y side wall
