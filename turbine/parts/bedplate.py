"""Part: bedplate (main frame), same frame as the nacelle."""
from turbine import dims as d
from turbine.urdf import link, span_box

gy = d.BP_GIRDER_Y
pin_z = d.SHAFT_Z + d.GB_BUSHING_Z
COLLISIONS = [span_box(d.BP_FRONT_X, (-gy[1], gy[1]), (0, d.BP_TOP_Z)),
              span_box(d.BP_DECK_X, (-gy[0], gy[0]), (0, d.BP_DECK_Z))]
for side in (-1, 1):
    COLLISIONS.append(span_box(d.BP_GIRDER_X, sorted((side * gy[0], side * gy[1])), (0, d.BP_TOP_Z)))
    yc = side * d.GB_BUSHING_Y
    for bx in d.BP_BRACKET_XS:
        COLLISIONS.append(span_box(bx, (yc - d.BP_BRACKET_HALF_Y, yc + d.BP_BRACKET_HALF_Y), (d.BP_TOP_Z, pin_z + 0.14)))

LINKS = [link("bedplate", "nacelle", mesh="bedplate/bedplate.obj", xyz=(d.DRIVETRAIN_OFFSET_X, 0, 0),
              collisions=COLLISIONS)]
INSPECTION_POINTS = []
FAULTS = {}
