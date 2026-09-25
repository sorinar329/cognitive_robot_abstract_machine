"""Start cameras for CRAMERA scenes and videos: view name -> (camera, show the nacelle cover)."""
from turbine.dims import site

_N = site.NACELLE_ORIGIN_Z
VIEWS = {
    # view: (camera, show the nacelle cover)
    "turbine": ({"position": [150.0, -210.0, 90.0], "target": [0.0, 0.0, 85.0]}, True),
    "tower_base": ({"position": [11.0, -17.0, 5.3], "target": [0.0, -2.0, 1.5]}, True),
    "nacelle": ({"position": [-1.5, -14.0, _N + 8.5], "target": [-3.5, 0.0, _N + 0.8]}, False),
    # robot recordings: the G1's start and its round around the tower base / along the nacelle walkway
    "g1_ground": ({"position": [19.0, -25.0, 9.0], "target": [2.0, -7.0, 1.0]}, True),
    "g1_nacelle": ({"position": [1.5, -9.5, _N + 5.5], "target": [-1.5, -1.2, _N + 0.9]}, False),
    "g1_nacelle_video": ({"position": [-3.2, -8.6, _N + 4.4], "target": [-3.4, -1.3, _N + 0.9]}, False),
    "g1_nacelle_wide": ({"position": [-2.5, -14.5, _N + 8.0], "target": [-3.3, -1.0, _N + 0.8]}, False),
    # the climbs: CRAMERA starts at the ground; videos follow the robot ("follow": camera offset
    # from the robot base, "look": target offset) and cut the tower wall facing the camera ("cut_tower")
    "g1_hoist": ({"position": [4.0, -19.0, 6.0], "target": [-8.0, -2.0, 1.5],
                  "follow": [5.0, -7.5, 6.5], "look": [0.0, 0.0, 0.0]}, False),
    "g1_close": ({"position": [2.5, -8.0, 1.2], "target": [0.0, -5.5, 0.6],
                  "follow": [2.0, -2.4, 0.5], "look": [0.0, 0.0, -0.25]}, True),
    "g1_tower": ({"position": [1.4, -1.9, 2.4], "target": [-0.7, 0.2, 0.9],
                  "follow": [3.2, -4.2, 1.6], "look": [0.0, 0.0, -0.1], "cut_tower": [0.6, -0.8]}, True),
}
MAX_CAMERA_DISTANCE = 320.0   # read by CRAMERA (rendering.maxCameraDistance) so the whole turbine fits
