"""Part: bedplate (nacelle frame): cast front bedplate, rear frame, torque arm brackets."""
from turbine import dims as d
from turbine.urdf import link, span_box

gy = d.BP_GIRDER_Y
COLLISIONS = [span_box((-0.5, d.BP_FRONT_X[1]), (-gy[1], gy[1]), (0, d.BP_TOP_Z))]
for side in (-1, 1):
    COLLISIONS.append(span_box(d.BP_FRONT_X, sorted((side * gy[0], side * gy[1])), (0, d.BP_TOP_Z)))
    COLLISIONS.append(span_box(d.BP_REAR_X, sorted((side * d.BP_REAR_Y[0], side * d.BP_REAR_Y[1])), (0, d.BP_REAR_TOP_Z)))
    for lx in (d.GB_BUSHING_X[0] - d.GB_PIN_EXT + 0.05, d.GB_BUSHING_X[1] + d.GB_PIN_EXT - 0.05):
        x, _, z = d.drivetrain_to_nacelle(d.GEARBOX_X + lx, d.GB_BUSHING_Z)
        yc = side * d.GB_BUSHING_Y
        COLLISIONS.append(span_box((x - 0.05, x + 0.05), (yc - 0.12, yc + 0.12), (d.BP_TOP_Z, z + 0.14)))

LINKS = [link("bedplate", "nacelle", mesh="bedplate/bedplate.obj", collisions=COLLISIONS)]
INSPECTION_POINTS = []
FAULTS = {}
