"""Nacelle: cover, floor, tower access hatch, and on the roof the weather mast
(anemometer, wind vane, aviation light). Turns on the yaw bearing (continuous Z)."""
from turbine import dims as d
from turbine.dims import site
from turbine.urdf import fault, floor_boxes, inspection_point, link, span_box

F, W = d.FLOOR, d.WALL
NX, NY, NZ, IX, IY = d.NACELLE_X, d.NACELLE_Y, d.NACELLE_Z, d.INNER_X, d.INNER_Y
HX, HY = d.FLOOR_HATCH_X, d.FLOOR_HATCH_Y
KX, KY = d.CRANE_HATCH_X, d.CRANE_HATCH_Y
SZ, SH = d.SHAFT_HOLE_Z, d.SHAFT_HOLE_R

COLLISIONS = [
    # floor, open at the tower access hatch and the crane hatch
    *floor_boxes((-F, 0), IX, IY, holes=[(HX, HY), (KX, KY)]),
    # side walls, rear wall, roof (the roof hatch is not cut out yet)
    span_box(NX, (NY[0], IY[0]), NZ),
    span_box(NX, (IY[1], NY[1]), NZ),
    span_box((NX[0], IX[0]), NY, NZ),
    span_box(NX, NY, (NZ[1] - W, NZ[1])),
    # front wall, split around the main shaft opening
    span_box((IX[1], NX[1]), NY, (NZ[0], SZ - SH)),
    span_box((IX[1], NX[1]), NY, (SZ + SH, NZ[1])),
    span_box((IX[1], NX[1]), (NY[0], -SH), (SZ - SH, SZ + SH)),
    span_box((IX[1], NX[1]), (SH, NY[1]), (SZ - SH, SZ + SH)),
]

LINKS = [
    link("nacelle", "yaw_bearing", mesh=["nacelle/floor.obj", "nacelle/cover.obj"], xyz=(0, 0, site.YAW_BEARING_H),
         joint="continuous", axis=(0, 0, 1), collisions=COLLISIONS),
    link("nacelle_floor_hatch", "nacelle", mesh="nacelle/floor_hatch.obj",
         xyz=((HX[0] + HX[1]) / 2, HY[1], 0.0), joint="revolute", axis=(-1, 0, 0), limits=(0.0, 1.9),
         collisions=[span_box(((HX[0] - HX[1]) / 2, (HX[1] - HX[0]) / 2), (HY[0] - HY[1], 0.0), (0.0, d.HATCH_THICKNESS))]),
    # rear hatch under the crane: opened (pi/2, standing at its +Y edge) to hoist the lifting platform
    link("crane_hatch", "nacelle", mesh="nacelle/crane_hatch.obj",
         xyz=((KX[0] + KX[1]) / 2, KY[1], 0.0), joint="revolute", axis=(-1, 0, 0), limits=(0.0, 1.9),
         collisions=[span_box(((KX[0] - KX[1]) / 2, (KX[1] - KX[0]) / 2), (KY[0] - KY[1], 0.0), (0.0, d.HATCH_THICKNESS))]),
]
MAST_TOP = NZ[1] + d.MAST_HEIGHT
LINKS += [
    link("weather_mast", "nacelle", mesh="nacelle/weather_mast.obj",
         collisions=[span_box((d.MAST_X - 0.05, d.MAST_X + 0.05), (-d.MAST_ARM, d.MAST_ARM), (NZ[1], MAST_TOP + 0.2))]),
    link("anemometer", "weather_mast",
         variants={"ok": "nacelle/anemometer_ok.obj", "damaged": "nacelle/anemometer_damaged.obj"}),
    link("aviation_light", "weather_mast",
         variants={"ok": "nacelle/aviation_light_ok.obj", "broken": "nacelle/aviation_light_broken.obj"}),
]

INSPECTION_POINTS = [
    inspection_point("nacelle_inspect_weather_mast", "nacelle", (d.MAST_X, 0.2, MAST_TOP + 0.12), (-0.6, -1, 0.35),
                     "anemometer, wind vane and aviation light on the roof", distance=1.6, outside=True),
    inspection_point("nacelle_inspect_cover", "nacelle", (d.COVER_CRACK_XZ[0] + 0.5, NY[0], d.COVER_CRACK_XZ[1]), (0, -1, 0.1),
                     "nacelle cover, -Y side: cracks in the GRP shell", distance=4.0, outside=True),
]

FAULTS = {
    "nacelle.anemometer_damaged": fault(
        "nacelle", "anemometer", "Cup anemometer damaged: one cup torn off, one arm bent",
        "nacelle_inspect_weather_mast", ["drone_rgb"], severity="medium",
        set_variants={"anemometer": "damaged"}, signals={"anemometer_plausible": False}),
    "nacelle.aviation_light_failed": fault(
        "nacelle", "aviation obstruction light", "Aviation light dark with a broken dome",
        "nacelle_inspect_weather_mast", ["drone_rgb"], severity="high",
        set_variants={"aviation_light": "broken"}, signals={"aviation_light_on": False}),
    "nacelle.cover_crack": fault(
        "nacelle", "nacelle cover", "Crack in the GRP cover on the -Y side, rain can get in",
        "nacelle_inspect_cover", ["drone_rgb"], severity="low",
        overlays=[link("nacelle_fault_cover_crack", "nacelle", mesh="nacelle/fault_cover_crack.obj")]),
}

SIGNALS = {"anemometer_plausible": True, "aviation_light_on": True}
