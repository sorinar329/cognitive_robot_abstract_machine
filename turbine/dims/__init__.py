"""Single source of truth for all dimensions (metres, radians).

Imported by the Blender part scripts (geometry) and by the URDF generator
(joint origins, collision primitives), so both always agree.

This module holds the nacelle interior (frame ``nacelle``: origin on the yaw
axis at floor level, X upwind, Z up). ``site`` covers ground, foundation, tower
and yaw bearing, ``rotor`` the hub and blades. Values from the IEA-3.4-130-RWT
reference are marked IEA; everything else is an estimate for a 3-3.5 MW turbine.
"""

import math

from turbine.dims import rotor as _rotor

# ------------------------------------------------------------------ nacelle cover (estimate)
NACELLE_X = (-9.8, 2.8)        # 12.6 m: drivetrain, generator, transformer and cabinets in a row
NACELLE_Y = (-2.1, 2.1)        # 4.2 m wide
NACELLE_Z = (0.0, 4.2)         # 4.2 m high
WALL = 0.06
FLOOR = 0.08
INNER_X = (NACELLE_X[0] + WALL, NACELLE_X[1] - WALL)
INNER_Y = (NACELLE_Y[0] + WALL, NACELLE_Y[1] - WALL)
WALKWAY_Y = 1.35               # walkways from +-1.35 m to the wall: ~0.69 m, enough for the G1

SHAFT_HOLE_R = 1.0             # front wall opening around the main shaft
ROOF_HATCH_X, ROOF_HATCH_Y = (-6.2, -5.2), (-0.5, 0.5)
FLOOR_HATCH_X, FLOOR_HATCH_Y = (-0.95, -0.25), (-2.0, -1.36)   # tower access, in the -Y walkway
HATCH_THICKNESS = 0.03

# ------------------------------------------------------------------ drivetrain frame
# Frame ``drivetrain``: origin at the hub centre, X along the main shaft axis
# (tilted up by the IEA 5 deg), pointing upwind. Main shaft, bearings, gearbox,
# fast shaft, brake, coupling and generator are placed along -X in this frame.
TILT = _rotor.TILT
HUB_IN_NACELLE = _rotor.HUB_IN_NACELLE


def drivetrain_to_nacelle(x, z=0.0, y=0.0):
    """Nacelle-frame point of a drivetrain-frame point."""
    hx, hy, hz = HUB_IN_NACELLE
    c, s = math.cos(TILT), math.sin(TILT)
    return (hx + x * c - z * s, hy + y, hz + x * s + z * c)


def nacelle_height_in_drivetrain(x, z_nacelle):
    """Drivetrain-frame z of the level plane at nacelle height z_nacelle, at drivetrain x."""
    return (z_nacelle - HUB_IN_NACELLE[2] - x * math.sin(TILT)) / math.cos(TILT)


SHAFT_HOLE_Z = drivetrain_to_nacelle((NACELLE_X[1] - HUB_IN_NACELLE[0]) / math.cos(TILT))[2]

# ------------------------------------------------------------------ bedplate (nacelle frame, level)
BP_FRONT_X = (-3.4, 2.7)                 # cast front bedplate: main bearings and torque arms
BP_GIRDER_Y = (0.9, 1.3)                 # side girders, mirrored in Y
BP_TOP_Z = 0.45
BP_CABLE_HOLE_R = 0.3                    # cable loop opening on the yaw axis
BP_REAR_X = (-9.7, -3.4)                 # welded rear frame: generator, transformer
BP_REAR_Y = (0.5, 0.8)
BP_REAR_TOP_Z = 0.3

# ------------------------------------------------------------------ main shaft, bearings, rotor lock (drivetrain frame)
MS_FLANGE_X, MS_FLANGE_R = (-2.0, -1.8), 0.9           # bolted to the hub flange
MS_SEGMENTS = [                                         # (radius, x range); IEA LSS OD 0.577 m
    (0.42, (-2.25, -2.0)), (0.33, (-3.3, -2.25)), (0.3, (-4.5, -3.3)),
    (0.32, (-5.1, -4.5)), (0.26, (-6.0, -5.1)),
]
MS_LOCK_DISC_X, MS_LOCK_DISC_R = (-2.62, -2.52), 0.9
MS_LOCK_HOLE_R, MS_LOCK_HOLE_PCD, MS_LOCK_HOLES = 0.05, 0.76, 12   # holes outside the bearing housing
MS_SHRINK_X, MS_SHRINK_R = (-5.7, -5.5), 0.5           # on the gearbox input stub
MS_SHRINK_BOLT_PCD, MS_SHRINK_BOLTS = 0.42, 20
MS_LOOSE_BOLT = 6

MB_XS = (-2.95, -4.75)                                   # IEA: CARB (front) and SRB (rear)
MB_R, MB_BORE_R, MB_HALF_W = 0.62, 0.34, 0.26
MB_FOOT_Y = 1.15
MB_SEAL_R = 0.45
MB_COLLECTOR_R, MB_COLLECTOR_H = 0.06, 0.18
MB_COLLECTOR_Y = -0.95                                    # stands on the front bearing's foot, -Y side
MB_GREASE_OK, MB_GREASE_FULL = 0.25, 0.95

RL_HOUSING_X, RL_Y, RL_HALF = (-3.1, -2.72), -MS_LOCK_HOLE_PCD, 0.1     # clamped to the front bearing housing
RL_PIN_R, RL_PIN_LEN, RL_STROKE = 0.045, 0.3, 0.18

# ------------------------------------------------------------------ gearbox (frame ``gearbox``: front face on the shaft axis)
GEARBOX_X = -5.75                        # gearbox frame origin, drivetrain x
GB_INPUT_R, GB_INPUT_X = 0.3, (0.0, 0.4)
GB_FRONT_COVER_R, GB_FRONT_COVER_X = 0.9, (-0.15, 0.0)
GB_PLANET_R, GB_PLANET_X = 1.0, (-1.05, -0.15)          # stage 1 planetary
GB_PLANET2_R, GB_PLANET2_X = 0.85, (-1.7, -1.05)        # stage 2 planetary
GB_HELICAL_X, GB_HELICAL_Y, GB_HELICAL_Z = (-2.5, -1.7), (-0.75, 0.75), (-0.75, 1.0)   # parallel stage
GB_RIB_X = (-0.35, -0.9)
GB_OUTPUT_Z = 0.55                                        # fast shaft axis above the main shaft axis
GB_OUTPUT_R, GB_OUTPUT_X = 0.144, (-2.7, -2.5)
GB_ARM_X, GB_ARM_Z = (-0.85, -0.35), (-0.3, 0.2)
GB_ARM_Y_IN, GB_ARM_Y_OUT = 0.9, 1.3
GB_BUSHING_R, GB_BUSHING_Y, GB_BUSHING_Z = 0.13, 1.1, -0.05
GB_BUSHING_X = (-0.9, -0.3)
GB_PIN_R, GB_PIN_EXT = 0.05, 0.25
GB_COOLER_X, GB_COOLER_Y, GB_COOLER_Z = (-2.45, -2.05), (-0.5, 0.5), (1.0, 1.45)
GB_COVER_X, GB_COVER_Z, GB_COVER_T = (-2.35, -1.85), (0.05, 0.55), 0.03   # on the +Y side face, towards the walkway
GB_COVER_BOLT_R = 0.018
GB_GLASS_XZ, GB_GLASS_R = (-2.35, -0.3), 0.065            # -Y face of the parallel stage, clear of the filter
GB_OIL_LEVEL_OK, GB_OIL_LEVEL_LOW = 0.6, 0.12
GB_FILTER_XY, GB_FILTER_R, GB_FILTER_Z = (-1.95, -0.92), 0.11, (-0.2, 0.45)

# ------------------------------------------------------------------ fast shaft, brake, coupling (drivetrain frame, axis at GB_OUTPUT_Z)
HSS_X, HSS_R = (-9.75, GEARBOX_X + GB_OUTPUT_X[0]), 0.144     # IEA: HSS diameter 0.288 m
BRAKE_DISC_X, BRAKE_DISC_R = (-8.8, -8.74), 0.45
CALIPER_Y_HALF, CALIPER_Z = 0.14, (0.28, 0.52)                 # radial position above the axis
BRAKE_PAD_OK, BRAKE_PAD_WORN = 0.025, 0.004                    # pad thickness
COUPLING_X, COUPLING_R = (-9.6, -9.05), 0.3

# ------------------------------------------------------------------ generator, DFIG (drivetrain frame, axis at GB_OUTPUT_Z)
GEN_X, GEN_HALF = (-12.2, -9.75), 0.78                          # square housing with cooling fins
SLIP_X, SLIP_HALF = (-12.75, -12.2), 0.45                       # slip-ring housing with brush access cover
GEN_TERMINAL_X = (-11.4, -10.6)

# ------------------------------------------------------------------ electrical, systems (nacelle frame)
TRAFO_X, TRAFO_Y, TRAFO_Z = (-9.65, -8.0), (-0.9, 0.9), (0.0, 2.3)
CABINET_X, CABINET_Y, CABINET_Z = (-9.7, -9.1), (1.1, 1.95), (0.0, 2.0)   # mirrored: converter +Y, controller -Y
HYDRAULIC_X, HYDRAULIC_Y, HYDRAULIC_Z = (-5.4, -4.5), (0.85, 1.3), (0.3, 1.05)
COOLING_X, COOLING_Y, COOLING_Z = (-6.3, -5.4), (-1.3, -0.85), (0.3, 1.25)
CRANE_X, CRANE_Z, CRANE_TRAVEL = (-9.5, 1.2), 3.9, (-9.0, 0.6)
EXTINGUISHER_XY = (-1.6, -2.0)
ESTOP_XZ = (-0.1, 1.25)                                  # on the -Y wall next to the tower hatch

# ------------------------------------------------------------------ roof: weather mast (estimate)
MAST_X = NACELLE_X[0] + 0.9
MAST_HEIGHT = 1.6
MAST_ARM = 0.55                 # crossbar half length (anemometer at +Y, wind vane at -Y)
COVER_CRACK_XZ = (-5.8, 2.6)    # on the -Y side wall

# ------------------------------------------------------------------ task T3: bring the tool (nacelle frame)
TOOL_RACK_X, TOOL_RACK_Y, TOOL_RACK_TOP = (-8.95, -8.35), (INNER_Y[0], -1.72), 0.85   # rear, -Y wall
TOOL_TRAY_X, TOOL_TRAY_Y, TOOL_TRAY_TOP = (1.0, 1.4), (-1.25, -0.98), 0.85           # on the -Y girder
TOOL_CASE = (0.09, 0.09, 0.2)             # torque tool case (socket set), small enough for the G1 hand
TECHNICIAN_XY = (2.0, -1.7)               # on the -Y walkway by the front main bearing, facing the machine
