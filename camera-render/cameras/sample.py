# ABOUTME: Training camera draws (continuous ranges up to 1.3x, one family per view, seeded by episode and frame)
# ABOUTME: and the 24-view val/test ladder; each view is checked against its scene's 451 test cameras, redrawn on a hit.
"""The training views and the validation/test ladder. A pose is (H, V, S, R, E) with signed angles. The check
compares the camera actually set (transform.camera: rounded position and quaternion, sign-normalised) with the test
cameras computed the same way from the case names. Runs in the render image's Python 3.8 as well as locally."""

from __future__ import annotations

import hashlib

import numpy as np
from cameras.spec import FAMILY_CASES as TEST_CASES
from cameras.spec import load_testset, signed
from cameras.transform import camera, camera_key

FAMILY_CASES = {
    "orbit": TEST_CASES["orbit_level"] + TEST_CASES["orbit_raised"],
    "distance": TEST_CASES["distance"],
    "turn_tilt": TEST_CASES["turn_tilt"],
}
LEVEL_SHARE = TEST_CASES["orbit_level"] / FAMILY_CASES["orbit"]
FAIR = 0.5
TEST_MAX = {"H": 75.0, "V": 15.0, "S": 200.0, "R": 10.0, "E": 10.0}
TEST_MIN = {"H": 1.0, "S": 115.0, "R": 2.0, "E": 2.0}  # the test's smallest non-zero |H|, |R|, |E| and smallest S
LOW_FRAC, HIGH_FRAC = 0.7, 1.3
# The ranges (set 2026-09-29): H and R/E from the test's smallest value up to 1.3x its largest (H: +-1 to +-97.5,
# R and E: +-2 to +-13); V keeps 0.7x-1.3x of 15 for the raised mode; S from 1.15x to 2.30x
ORBIT_ABS_H = (TEST_MIN["H"], round(HIGH_FRAC * TEST_MAX["H"], 6))
RAISED_V = (round(LOW_FRAC * TEST_MAX["V"], 6), round(HIGH_FRAC * TEST_MAX["V"], 6))
DISTANCE_S = (115.0, 230.0)
TURN_ABS = (TEST_MIN["R"], round(HIGH_FRAC * TEST_MAX["R"], 6))
TRAIN_SEED, LADDER_SEED = 20260929, 20260930
SPLIT_CODES = {"val": 1, "test": 2}
MAX_REDRAWS = 100

ORBIT_RUNGS = ((7.5, 0.1), (22.5, 0.3), (37.5, 0.5), (56.25, 0.75), (74.5, 1.0), (97.5, 1.3), (112.5, 1.5))
LADDER_RAISED_V = 14.5
DISTANCE_RUNGS = ((110.5, 0.1), (150.5, 0.5), (199.5, 1.0), (230.0, 1.3), (250.0, 1.5))
TURN_RUNGS = ((1.0, 0.1), (5.0, 0.5), (9.5, 1.0), (13.0, 1.3), (15.0, 1.5))
TRAINING_CAP_FRAC, BEYOND_TRAINING_FRAC = 1.3, 1.5
BELOW_TEST = {("distance", 110.5), ("turn_tilt", 1.0)}


def episode_seed(episode: str) -> int:
    return int(hashlib.sha256(episode.encode()).hexdigest()[:16], 16)


def beyond_test(pose) -> bool:
    """At least one field beyond the test's largest value for it (V > 15 counts)."""
    h, v, s, r, e = pose
    return abs(h) > TEST_MAX["H"] or v > TEST_MAX["V"] or s > TEST_MAX["S"] or max(abs(r), abs(e)) > TEST_MAX["R"]


_TEST_KEYS: dict[str, frozenset] = {}


def test_keys(problem: str) -> frozenset:
    """The fork sets each test camera from the case name's mod-360 angles; the signed-angle camera is added too, since
    a rounding boundary can put the two 1e-4 apart (test_cameras.py measures where)."""
    if problem in _TEST_KEYS:
        return _TEST_KEYS[problem]
    keys = set()
    for h, v, s, r, e in load_testset()["poses"]:
        for pose in ((h, v, s, r, e), (signed(h), signed(v), s, signed(r), signed(e))):
            keys.add(camera_key(*camera(problem, pose)))
    return _TEST_KEYS.setdefault(problem, frozenset(keys))


def hits_test(problem: str, pose) -> bool:
    return camera_key(*camera(problem, pose)) in test_keys(problem)


def pick_family(rng: np.random.Generator) -> str:
    u, total = rng.random(), sum(FAMILY_CASES.values())
    edge = 0.0
    for name, n in FAMILY_CASES.items():
        edge += n / total
        if u < edge:
            return name
    return name


def magnitudes(rng: np.random.Generator, family: str) -> tuple[float, ...]:
    def sign() -> float:
        return 1.0 if rng.random() < FAIR else -1.0

    if family == "orbit":
        h = sign() * rng.uniform(*ORBIT_ABS_H)
        v = 0.0 if rng.random() < LEVEL_SHARE else rng.uniform(*RAISED_V)
        return (h, v, 100.0, 0.0, 0.0)
    if family == "distance":
        return (0.0, 0.0, rng.uniform(*DISTANCE_S), 0.0, 0.0)
    r, e = sign() * rng.uniform(*TURN_ABS), sign() * rng.uniform(*TURN_ABS)
    return (0.0, 0.0, 100.0, r, e)


def train_view(episode: str, frame: int, problem: str) -> dict:
    """The training view of kept frame `frame` of `episode`: seeded by both, redrawn inside its family on a test hit."""
    rng = np.random.default_rng([TRAIN_SEED, episode_seed(episode), frame])
    family = pick_family(rng)
    for redraws in range(MAX_REDRAWS):
        pose = magnitudes(rng, family)
        if not hits_test(problem, pose):
            return {
                "pose": [float(x) for x in pose],
                "family": family,
                "redraws": redraws,
                "bin": "training_range_beyond_test" if beyond_test(pose) else "in_test_range",
            }
    raise RuntimeError(f"{episode} frame {frame}: {MAX_REDRAWS} draws all hit a test camera")


def ladder_bin(family: str, magnitude: float, frac: float) -> str:
    if frac >= BEYOND_TRAINING_FRAC:
        return "beyond_training"
    if frac >= TRAINING_CAP_FRAC:
        return "training_range_beyond_test"
    return "below_test_range" if (family, magnitude) in BELOW_TEST else "in_test_range"


def ladder(split: str, episode: str, moment: int, problem: str) -> list[dict]:
    """The 24 fixed-magnitude views of one state; signs by a fixed per-state rule (seeded by split, episode, moment).
    Refuses if any rung's camera equals a test camera."""
    rng = np.random.default_rng([LADDER_SEED, SPLIT_CODES[split], episode_seed(episode), moment])
    views = []
    for v, family in ((0.0, "orbit_level"), (LADDER_RAISED_V, "orbit_raised")):
        views += [((float(rng.choice([-1, 1])) * h, v, 100.0, 0.0, 0.0), family, h, f) for h, f in ORBIT_RUNGS]
    views += [((0.0, 0.0, s, 0.0, 0.0), "distance", s, f) for s, f in DISTANCE_RUNGS]
    for m, f in TURN_RUNGS:
        r, e = (float(x) * m for x in rng.choice([-1, 1], size=2))
        views.append(((0.0, 0.0, 100.0, r, e), "turn_tilt", m, f))
    out = []
    for pose, family, magnitude, frac in views:
        if hits_test(problem, pose):
            raise RuntimeError(f"ladder pose {pose} sets a test camera in {problem}")
        base = family.split("_")[0] if family.startswith("orbit") else family
        out.append({"pose": list(pose), "family": family, "frac": frac, "bin": ladder_bin(base, magnitude, frac)})
    return out
