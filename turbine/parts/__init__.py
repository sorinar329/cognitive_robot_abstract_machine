"""Parts in assembly order; each module provides LINKS, INSPECTION_POINTS, FAULTS
(and optionally SIGNALS, the healthy baseline of its non-visual sensor values)."""
from turbine.parts import bedplate, gearbox, main_shaft, nacelle, rotor, site

ALL = [site, nacelle, rotor, bedplate, main_shaft, gearbox]
