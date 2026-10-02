"""
Show an offshore wind farm of three NREL 5 MW turbines on OC3 monopiles.

The turbines stand in a row across the unyawed downwind direction, with the substation
behind the middle one. Their blades are pitched in and their rotors turn at rated speed
until the script is interrupted.

Run with ``cramera-live`` to see it in the browser viewer, or directly for RViz (fixed
frame ``map``, ``MarkerArray`` display on ``/semworld/viz_marker``).
"""

from __future__ import annotations

import time

import numpy as np

from coraplex.testing import start_visualization
from semantic_digital_twin.semantic_annotations.wind_farm import (
    NREL_5MW_OC3_MONOPILE,
    OC3_MONOPILE,
    OffshoreWindFarmLayout,
    WindFarm,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

TURBINE_SPACING_IN_ROTOR_DIAMETERS = 7.0
"""
Distance between neighboring turbines in rotor diameters, as at Horns Rev 1.
"""

SUBSTATION_DISTANCE_IN_ROTOR_DIAMETERS = 3.0
"""
How far downwind of the turbine row the substation stands, in rotor diameters.
"""

SUBSTATION_TOPSIDE = Scale(30.0, 20.0, 12.0)
"""
Drawn size of the substation topside in meters; not taken from a reference design.
"""

RATED_ROTOR_SPEED = 12.1 * 2 * np.pi / 60
"""
Rotor speed of the NREL 5 MW turbine at rated wind speed in radians per second (12.1
rpm).
"""

UPDATE_RATE = 20.0
"""
How often per second the rotors are advanced.
"""


def demo_layout() -> OffshoreWindFarmLayout:
    """
    :return: Three turbines in a row along the world y axis, and the substation
        downwind of the middle one.
    """
    rotor_diameter = 2 * NREL_5MW_OC3_MONOPILE.geometry.tip_radius
    spacing = TURBINE_SPACING_IN_ROTOR_DIAMETERS * rotor_diameter
    return OffshoreWindFarmLayout(
        turbine_model=NREL_5MW_OC3_MONOPILE,
        foundation=OC3_MONOPILE,
        turbine_poses=[
            HomogeneousTransformationMatrix.from_xyz_rpy(y=row_index * spacing)
            for row_index in (-1, 0, 1)
        ],
        substation_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=SUBSTATION_DISTANCE_IN_ROTOR_DIAMETERS * rotor_diameter
        ),
        substation_topside=SUBSTATION_TOPSIDE,
    )


def spin(wind_farm: WindFarm) -> None:
    """
    Pitch every blade in and turn every rotor at rated speed until interrupted.
    """
    for turbine in wind_farm.turbines:
        for pitch_connection in turbine.pitch_connections:
            pitch_connection.position = 0.0
    time_step = 1 / UPDATE_RATE
    while True:
        for turbine in wind_farm.turbines:
            rotor = turbine.rotor_connection
            rotor.position = (rotor.position + RATED_ROTOR_SPEED * time_step) % (
                2 * np.pi
            )
        time.sleep(time_step)


def main() -> None:
    world = World.create_with_root_body()
    wind_farm = WindFarm.create_offshore_with_new_bodies_in_world(
        "offshore_wind_farm", world, demo_layout()
    )
    visualization = start_visualization(world)
    try:
        spin(wind_farm)
    except KeyboardInterrupt:
        pass
    finally:
        visualization.stop()


if __name__ == "__main__":
    main()
