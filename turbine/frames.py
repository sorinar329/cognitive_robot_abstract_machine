"""Small rotation helpers shared by Blender scripts and the URDF generator."""
import math

from turbine.dims import rotor as r


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def apply(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def to_rpy(m):
    """URDF roll-pitch-yaw of a rotation matrix (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = math.asin(max(-1.0, min(1.0, -m[2][0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(m[2][1], m[2][2])
        yaw = math.atan2(m[1][0], m[0][0])
    else:
        roll, yaw = math.atan2(-m[1][2], m[1][1]), 0.0
    return (roll, pitch, yaw)


def homogeneous(m, t=(0, 0, 0)):
    return [list(m[0]) + [t[0]], list(m[1]) + [t[1]], list(m[2]) + [t[2]], [0, 0, 0, 1]]


def blade_azimuth(i):
    return 2 * math.pi * i / r.BLADES


def blade_rotation(i):
    """Hub frame -> blade i frame: azimuth about hub X, then the cone towards upwind."""
    return matmul(rot_x(blade_azimuth(i)), rot_y(r.CONE))


def blade_origin(i):
    return apply(blade_rotation(i), (0.0, 0.0, r.HUB_R))
