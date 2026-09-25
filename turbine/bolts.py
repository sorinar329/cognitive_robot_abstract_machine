"""Tower flange bolts: layout, preload estimated from torque markings, and what to do.

Torque marking: after tightening, a paint line is drawn across the nut, washer and
bolt end. If the nut turns back, the line on the nut no longer lines up; the offset
angle tells how far it turned. Turning a nut back by the angle it was turned from
snug to full preload releases all preload, so the remaining preload is estimated as

    preload ratio = 1 - offset / turn_to_preload

with turn_to_preload = 360 deg * F_p (1/k_bolt + 1/k_joint) / pitch. This only sees a
nut that turned. Preload lost by settling or relaxation leaves the marking aligned,
which is why the rules below still ask for a tension check on a sample.

Assessment follows ACP RP 401 (anchor bolts; used here by analogy, the turbine
manufacturer's manual sets the real values): a bolt below 85 % of the specified
preload, or a flange average below 90 %, means re-tensioning the whole flange.
"""
import math

from turbine.dims import site as s

FUB = 1000e6                  # property class 10.9: ultimate tensile strength (Pa)
E_STEEL = 210e9
JOINT_TO_BOLT_STIFFNESS = 3.0  # clamped flanges are stiffer than the bolt (estimate)
K_FACTOR = 0.13               # torque coefficient, lubricated HV set (estimate)
BOLT_MIN, FLANGE_MEAN_MIN, RETIGHTEN_BELOW, LOST_BELOW = 0.85, 0.90, 0.90, 0.10
SAMPLE_FRACTION = 0.10        # tension check on 10 % of the bolts (ACP RP 401)
ZS = [0.0] + list(s.TOWER_FLANGES) + [s.TOWER_HEIGHT]


def spec(size):
    b = dict(s.BOLT_SPECS[size], size=size)
    b["preload"] = 0.7 * FUB * b["stress_area"]                          # F_p,C (EN 1993-1-8)
    grip = 2 * s.FLANGE_H + 2 * b["washer_h"]
    k_bolt = E_STEEL * b["stress_area"] / grip
    stretch = b["preload"] * (1 / k_bolt + 1 / (JOINT_TO_BOLT_STIFFNESS * k_bolt))
    b["turn_to_preload_deg"] = 360.0 * stretch / b["pitch"]
    b["torque_nm"] = K_FACTOR * b["preload"] * b["d"]
    return b


def flange(k):
    """Flange k (1 = lowest): height, bolt circle, bolts [(index, azimuth, x, y)] in the tower frame."""
    z = s.TOWER_FLANGES[k - 1]
    ri = s.tower_radius(z) - s.tower_wall(z)
    r = ri - s.FLANGE_WIDTH / 2
    size, n = s.FLANGE_BOLTS[k - 1]
    bolts = [(i, 2 * math.pi * i / n, r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    return dict(k=k, z=z, bolt_radius=r, inner_radius=ri, spec=spec(size), bolts=bolts,
                section=k, local_z=z - ZS[k - 1])


def nut_height(k):
    """Centre of the nut's side face below the flange joint, tower frame (z above the tower base)."""
    b = spec(s.FLANGE_BOLTS[k - 1][0])
    return s.TOWER_FLANGES[k - 1] - s.FLANGE_H - b["washer_h"] - b["nut_h"] / 2


def preload_ratio(offset_deg, bolt_spec):
    return max(0.0, min(1.0, 1.0 - abs(offset_deg) / bolt_spec["turn_to_preload_deg"]))


def ranges(indices):
    """[3, 4, 5, 9] -> '3-5, 9'"""
    out, run = [], []
    for i in sorted(indices):
        if run and i != run[-1] + 1:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = []
        run.append(i)
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ", ".join(out)


def bolt_status(ratio):
    if ratio < LOST_BELOW:
        return "lost"
    if ratio < BOLT_MIN:
        return "loose"
    if ratio < RETIGHTEN_BELOW:
        return "retighten"
    return "ok"


def assess(k, offsets_seen, total):
    """offsets_seen: {bolt index: marking offset (deg)} of the bolts the robot could read."""
    b = spec(s.FLANGE_BOLTS[k - 1][0])
    ratios = {i: preload_ratio(o, b) for i, o in offsets_seen.items()}
    seen = len(ratios)
    mean = sum(ratios.values()) / seen if seen else None
    worst = min(ratios.values()) if seen else None
    by = {st: sorted(i for i, r in ratios.items() if bolt_status(r) == st) for st in ("lost", "loose", "retighten")}
    unseen = sorted(set(range(total)) - set(ratios))
    whole = seen and (worst < BOLT_MIN or mean < FLANGE_MEAN_MIN)
    actions = []
    if whole:
        actions.append(f"Re-tension all {total} bolts of flange {k}: {b['size']} 10.9, preload {b['preload'] / 1e3:.0f} kN "
                       f"(about {b['torque_nm']:.0f} Nm with a hydraulic torque wrench). Work in a cross pattern, "
                       f"in two passes (50 %, then 100 %).")
    if by["lost"]:
        actions.append(f"Replace bolt, nut and washers at position{'s' if len(by['lost']) > 1 else ''} "
                       f"{', '.join(map(str, by['lost']))}: a bolt that ran loose under the tower's load cycles may be "
                       "fatigue-damaged. Check the two neighbours on each side and the flange gap before re-tensioning.")
    if not whole and (by["loose"] or by["retighten"]):
        few = by["loose"] + by["retighten"]
        actions.append(f"Re-tighten bolt{'s' if len(few) > 1 else ''} {', '.join(map(str, few))} to "
                       f"{b['torque_nm']:.0f} Nm; the flange as a whole is within limits.")
    if unseen:
        actions.append(f"{len(unseen)} bolts were not readable for the robot (behind the lift or at a flat angle): "
                       f"a technician checks the markings of bolts {ranges(unseen)} from the platform.")
    if seen and not whole and not by["lost"]:
        n = max(1, math.ceil(SAMPLE_FRACTION * total))
        actions.append(f"Tension check on a random {int(SAMPLE_FRACTION * 100)} % sample ({n} bolts) at the next "
                       "service, since preload lost without turning leaves the markings aligned.")
    if by["lost"] or by["loose"] or by["retighten"]:
        actions.append("Afterwards: draw new torque markings and record the values in the turbine log.")
    verdict = ("re-tension the flange" if whole else "re-tighten single bolts" if by["retighten"] or by["loose"]
               else "no action" if seen else "not inspected")
    return dict(flange=k, size=b["size"], bolts=total, seen=seen, tightness_ratio=mean, worst_ratio=worst,
                lost=by["lost"], loose=by["loose"], retighten=by["retighten"], unseen=len(unseen), unseen_bolts=unseen,
                verdict=verdict, actions=actions, turn_to_preload_deg=b["turn_to_preload_deg"],
                ratios={i: round(r, 3) for i, r in ratios.items() if r < 1.0})
