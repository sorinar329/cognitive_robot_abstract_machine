"""Blade surface lofted from the IEA-3.4-130-RWT planform and airfoils.

Frame ``blade``: origin at the root centre in the pitch bearing plane, Z along
the pitch axis towards the tip, X upwind (pressure side), leading edge towards
-Y (the direction the blade moves for a positive rotor rotation about hub X).

Pure Python so both Blender (mesh building) and the URDF generator
(collision boxes, inspection points) can use it.
"""
import math

from turbine import iea

POINTS_PER_SIDE = 48          # samples per airfoil surface, cosine-spaced (dense at LE and TE)


# %% airfoils
def _cosine_stations(n):
    return [0.5 * (1 - math.cos(math.pi * i / (n - 1))) for i in range(n)]


def _surface_at(xs, ys, x):
    """Linear interpolation of y(x) on one airfoil surface (xs ascending)."""
    for i in range(1, len(xs)):
        if x <= xs[i]:
            span = xs[i] - xs[i - 1]
            w = 0.0 if span == 0 else (x - xs[i - 1]) / span
            return ys[i - 1] * (1 - w) + ys[i] * w
    return ys[-1]


def _resample(airfoil, stations):
    """Upper and lower surface y at the given chord stations (0 = LE, 1 = TE)."""
    xs, ys = airfoil["x"], airfoil["y"]
    le = min(range(len(xs)), key=lambda i: xs[i])
    upper = sorted(zip(xs[: le + 1], ys[: le + 1]))           # TE -> LE in the file
    lower = sorted(zip(xs[le:], ys[le:]))                      # LE -> TE
    if sum(y for _, y in upper) < sum(y for _, y in lower):   # make "upper" the suction (+y) side
        upper, lower = lower, upper
    ux, uy = zip(*upper)
    lx, ly = zip(*lower)
    return [_surface_at(ux, uy, s) for s in stations], [_surface_at(lx, ly, s) for s in stations]


STATIONS = _cosine_stations(POINTS_PER_SIDE)
_AIRFOILS = sorted(
    ((a["rthick"], _resample(a, STATIONS)) for a in iea.AIRFOILS), key=lambda item: item[0]
)


def airfoil_for_thickness(rthick):
    """Upper/lower y at STATIONS for a relative thickness, blending the two
    neighbouring reference airfoils."""
    if rthick <= _AIRFOILS[0][0]:
        return _AIRFOILS[0][1]
    for (t0, (u0, l0)), (t1, (u1, l1)) in zip(_AIRFOILS, _AIRFOILS[1:]):
        if rthick <= t1:
            w = (rthick - t0) / (t1 - t0)
            return ([a * (1 - w) + b * w for a, b in zip(u0, u1)],
                    [a * (1 - w) + b * w for a, b in zip(l0, l1)])
    return _AIRFOILS[-1][1]


# %% sections
def section_parameters(t):
    """Planform at normalized span t (0 root, 1 tip)."""
    b = iea.BLADE
    return dict(
        z=iea.interpolate(b["ref_axis_grid"], b["ref_axis_z"], t),
        prebend=-iea.interpolate(b["ref_axis_grid"], b["ref_axis_x"], t),   # windIO -x is upwind
        chord=iea.interpolate(b["grid"], b["chord"], t),
        twist=iea.interpolate(b["grid"], b["twist_rad"], t),
        pitch_axis=iea.interpolate(b["grid"], b["pitch_axis_from_le"], t),
        rthick=iea.interpolate(b["grid"], b["rthick"], t),
    )


def chord_point(p, s, y_airfoil):
    """Blade-frame point for chord station s (0 LE .. 1 TE) and airfoil y (+ suction)."""
    along = s * p["chord"] - p["pitch_axis"]      # TE towards +Y
    bx, by = -y_airfoil * p["chord"], along       # suction side faces downwind (-X)
    c, sn = math.cos(p["twist"]), math.sin(p["twist"])
    return (bx * c - by * sn + p["prebend"], bx * sn + by * c, p["z"])


def section_ring(t):
    """Closed loop of blade-frame points: suction side TE->LE, then pressure side LE->TE.
    Ring index i < POINTS_PER_SIDE is the suction side at STATIONS[-1 - i]."""
    p = section_parameters(t)
    upper, lower = airfoil_for_thickness(p["rthick"])
    n = POINTS_PER_SIDE
    ring = [chord_point(p, STATIONS[n - 1 - i], upper[n - 1 - i]) for i in range(n)]
    ring += [chord_point(p, STATIONS[i], lower[i]) for i in range(1, n - 1)]
    return ring


def ring_index(side, s):
    """Ring index nearest to chord station s on side 'suction' or 'pressure'."""
    n = POINTS_PER_SIDE
    j = min(range(n), key=lambda i: abs(STATIONS[i] - s))
    if side == "suction":
        return n - 1 - j
    return n - 1 + j if 0 < j < n - 1 else (n - 1 if j == 0 else 0)


def ring_position(side, s):
    """Fractional ring index at chord station s (0 LE .. 1 TE) on a side."""
    n = POINTS_PER_SIDE
    s = min(max(s, 0.0), 1.0)
    for j in range(1, n):
        if s <= STATIONS[j]:
            f = (j - 1) + (s - STATIONS[j - 1]) / (STATIONS[j] - STATIONS[j - 1])
            break
    else:
        f = n - 1
    return (n - 1) - f if side == "suction" else (n - 1) + f


def span_stations():
    """Normalized span positions used for the loft: the IEA grid plus extra near the tip."""
    grid = list(iea.BLADE["grid"])
    extra = [0.955, 0.965, 0.975, 0.985, 0.992, 0.997]
    return sorted(set(grid + extra))


# %% meshes
def surface_mesh():
    """Vertices and quad/triangle faces of the closed blade surface."""
    spans = span_stations()
    rings = [section_ring(t) for t in spans]
    m = len(rings[0])
    verts = [v for ring in rings for v in ring]
    faces = []
    for k in range(len(rings) - 1):
        a, b = k * m, (k + 1) * m
        for i in range(m):
            j = (i + 1) % m
            faces.append((a + i, a + j, b + j, b + i))
    # root and tip caps (fan around the ring centroid)
    for k, flip in ((0, True), (len(rings) - 1, False)):
        ring = rings[k]
        c = tuple(sum(p[i] for p in ring) / m for i in range(3))
        verts.append(c)
        ci = len(verts) - 1
        for i in range(m):
            j = (i + 1) % m
            tri = (k * m + i, k * m + j, ci)
            faces.append(tri[::-1] if flip else tri)
    return verts, faces


def _normal(rings, k, i):
    """Outward surface normal at ring k, index i (finite differences)."""
    m = len(rings[k])
    p_prev, p_next = rings[k][(i - 1) % m], rings[k][(i + 1) % m]
    q_prev = rings[max(k - 1, 0)][i]
    q_next = rings[min(k + 1, len(rings) - 1)][i]
    t = [p_next[a] - p_prev[a] for a in range(3)]
    s = [q_next[a] - q_prev[a] for a in range(3)]
    n = [t[1] * s[2] - t[2] * s[1], t[2] * s[0] - t[0] * s[2], t[0] * s[1] - t[1] * s[0]]
    c = [sum(p[a] for p in rings[k]) / m for a in range(3)]
    p = rings[k][i]
    if sum(n[a] * (p[a] - c[a]) for a in range(3)) < 0:
        n = [-x for x in n]
    ln = math.sqrt(sum(x * x for x in n)) or 1.0
    return [x / ln for x in n]


def surface_patch(span_range, index_fn, offset=0.004, samples=24):
    """A thin patch lying on the blade surface, for fault overlays.

    ``index_fn(t, u)`` returns the (fractional) ring index at span t for u in
    [0, 1] across the patch. Returns (verts, faces).
    """
    t0, t1 = span_range
    spans = [t0 + (t1 - t0) * k / (samples - 1) for k in range(samples)]
    rings = [section_ring(t) for t in spans]
    m = len(rings[0])
    width = 9
    verts, faces = [], []
    for k, t in enumerate(spans):
        for w in range(width):
            f = index_fn(t, w / (width - 1))
            i0 = int(math.floor(f)) % m
            i1 = (i0 + 1) % m
            a = f - math.floor(f)
            p = [rings[k][i0][c] * (1 - a) + rings[k][i1][c] * a for c in range(3)]
            n = _normal(rings, k, i0)
            verts.append(tuple(p[c] + n[c] * offset for c in range(3)))
    for k in range(samples - 1):
        for w in range(width - 1):
            a, b = k * width + w, (k + 1) * width + w
            faces.append((a, a + 1, b + 1, b))
    return verts, faces


def surface_point(t, side, s, offset=0.0):
    """Blade-frame point on the surface at span t, side, chord station s."""
    rings = [section_ring(max(t - 0.005, 0.0)), section_ring(t), section_ring(min(t + 0.005, 1.0))]
    i = ring_index(side, s)
    n = _normal(rings, 1, i)
    return tuple(rings[1][i][c] + n[c] * offset for c in range(3))


def collision_boxes(n=6):
    """Coarse boxes along the span: (x range, y range, z range) in the blade frame."""
    spans = span_stations()
    edges = [k / n for k in range(n + 1)]
    boxes = []
    for t0, t1 in zip(edges, edges[1:]):
        pts = [p for t in spans if t0 <= t <= t1 for p in section_ring(t)]
        pts += section_ring(t0) + section_ring(t1)
        xs, ys, zs = zip(*pts)
        boxes.append(((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))))
    return boxes
