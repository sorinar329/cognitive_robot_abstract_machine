"""Parts in assembly order; each module provides LINKS, INSPECTION_POINTS, FAULTS
(and optionally SIGNALS, the healthy baseline of its non-visual sensor values)."""
from turbine.parts import bedplate, gearbox, generator, main_shaft, nacelle, rotor, site, systems

ALL = [site, nacelle, bedplate, main_shaft, rotor, gearbox, generator, systems]
