# ABOUTME: The LIBERO-Plus fork's five-step agentview transform (V, H, S, R, E, rounded to 4 decimals after
# ABOUTME: each step) for real-valued poses; integer poses give the fork's exact floats (test_cameras.py runs its code).
"""Fork pin 4976dc30, `libero/libero/envs/problems/*.py`: `rotate_around_y`, `rotate_around_z`,
`scale_distance_from_pivot` and each problem class's `_setup_camera`. The fork parses the pose with int(); here the
conditions compare the real value with 0 (or S with 100), so a non-integer pose is applied as given.

Runs in the render image's Python 3.8 as well as locally: numpy and scipy only."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from scipy.spatial.transform import Rotation

PIVOT = np.array([0, 0, 0.8])
Y_AXIS = np.array([0, 1, 0])
DECIMALS = 4
NOMINAL = (0.0, 0.0, 100.0, 0.0, 0.0)
PERCENT = 100

_TABLETOP = (
    (0.6586131746834771, 0.0, 1.6103500240372423),
    (0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349),
)
_LOW_QUAT = (0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979)
BASE_CAMERAS = {
    "libero_tabletop_manipulation": _TABLETOP,
    "libero_kitchen_tabletop_manipulation": _TABLETOP,
    "libero_study_tabletop_manipulation": ((0.4586131746834771, 0.0, 1.6103500240372423), _TABLETOP[1]),
    "libero_living_room_tabletop_manipulation": ((0.6065773716836134, 0.0, 0.96), _LOW_QUAT),
    "libero_floor_manipulation": ((0.8965773716836134, 5.216182733499864e-07, 0.65), _LOW_QUAT),
}
CAMERA_CLASS = {
    "libero_tabletop_manipulation": "tabletop",
    "libero_kitchen_tabletop_manipulation": "tabletop",
    "libero_study_tabletop_manipulation": "study",
    "libero_living_room_tabletop_manipulation": "living_room",
    "libero_floor_manipulation": "floor",
}


def _rounded(values) -> list[float]:
    return [round(float(x), DECIMALS) for x in values]


def _compose(rotation: Rotation, quat) -> list[float]:
    original = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    q = cast(Rotation, rotation * original).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _about_y(degrees: float) -> Rotation:
    return Rotation.from_rotvec(np.radians(-degrees) * Y_AXIS)


def _about_z(degrees: float) -> Rotation:
    return Rotation.from_euler("z", degrees, degrees=True)


def camera(problem: str, pose) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(pos, quat wxyz) the fork sets on "agentview" for `pose` = (H, V, S, R, E) in the scene class `problem`."""
    h, v, s, r, e = (float(x) for x in pose)
    pos, quat = BASE_CAMERAS[problem]
    if v != 0:
        up = _about_y(v)
        pos = _rounded(up.apply(np.array(pos) - PIVOT) + PIVOT)
        quat = _rounded(_compose(up, quat))
    around = _about_z(h)
    pos, quat = _rounded(around.apply(pos)), _rounded(_compose(around, quat))
    if s / PERCENT != 1.0:
        pos = _rounded(PIVOT + (np.array(pos) - PIVOT) * (s / PERCENT))
    if r != 0:
        quat = _rounded(_compose(_about_z(r), quat))
    if e != 0:
        quat = _rounded(_compose(_about_y(e), quat))
    return tuple(pos), tuple(quat)


def camera_key(pos, quat) -> tuple[float, ...]:
    """The comparison key of a set camera: q and -q are one rotation (350 and -10 degrees), so the quaternion's sign is
    fixed by its first non-zero component, w first. 0.0 and -0.0 compare and hash equal."""
    sign = next((math.copysign(1.0, c) for c in quat if c != 0), 1.0)
    return tuple(pos) + tuple(sign * c for c in quat)
