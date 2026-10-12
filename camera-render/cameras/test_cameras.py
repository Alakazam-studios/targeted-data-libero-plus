# ABOUTME: C1 (math) and C3 checks: transform.camera against the fork's own _setup_camera code run on the case-name
# ABOUTME: values, spec re-derivation, and the sampler's shares, uniformity, seeding and test-camera redraws.
from __future__ import annotations

import ast
import json
import types
from functools import cache

import numpy as np
import pytest
from cameras import sample, spec
from cameras.transform import BASE_CAMERAS, NOMINAL, camera, camera_key
from scipy import stats
from scipy.spatial.transform import Rotation

PROBLEMS = sorted(BASE_CAMERAS)
HELPERS = {"scale_distance_from_pivot", "rotate_around_y", "rotate_around_z"}
N_DRAWS = 100_000
SHARE_TOL = 0.02
KS_N, KS_P = 10_000, 0.01
BEYOND = {"orbit": 0.445, "distance": 0.261, "turn_tilt": 0.471}  # measured under the ranges of 2026-09-29 (sample.py)
BEYOND_ALL = 0.414
MAX_BOUNDARY_POSES = 3
# test-camera redraws per attempt, measured on these 100k draws (2026-09-29, H 1-97.5, R/E 2-13): distance 0.80%,
# orbit 0.17%, turn/tilt 0 of 18,569; the bounds hold a factor of about 2 around each
REDRAW_RATE = {"distance": (0.004, 0.012), "orbit": (0.0008, 0.0035), "turn_tilt": (0.0, 0.0005)}
LADDER_VIEWS = 24
MOMENTS = 5


@cache
def fork_setup_camera(problem: str):
    """The fork's own helper functions and `problem`'s _setup_camera, compiled from the pinned checkout's source."""
    path = spec.FORK / "libero/libero/envs/problems" / f"{problem}.py"
    tree = ast.parse(path.read_text())
    cls = "_".join(w.capitalize() for w in problem.split("_"))
    nodes: list[ast.stmt] = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    nodes += [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "_setup_camera"]
    ns = {"np": np, "Rotation": Rotation}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns["_setup_camera"]


def fork_camera(problem: str, name_pose) -> tuple:
    """What the fork sets for a case name's fields: env_wrapper.py:216-218 passes H, V, R, E as the name's strings and
    S as float(int(S) / 100)."""
    h, v, s, r, e = (str(x) for x in name_pose)
    this = types.SimpleNamespace(
        horizon_view=h, vertical_view=v, scale_factor=float(int(s) / 100), end_point_rot=r, end_point_vertical=e
    )
    got = {}
    arena = types.SimpleNamespace(set_camera=lambda camera_name, pos, quat: got.setdefault(camera_name, (pos, quat)))
    fork_setup_camera(problem)(this, arena)
    return tuple(got["agentview"][0]), tuple(got["agentview"][1])


def signed(pose):
    h, v, s, r, e = pose
    return [spec.signed(h), spec.signed(v), s, spec.signed(r), spec.signed(e)]


@pytest.fixture(scope="module")
def testset():
    return json.loads(spec.TESTSET.read_text())


@pytest.mark.parametrize("problem", PROBLEMS)
def test_transform_equals_fork_at_every_test_pose(problem, testset):
    for pose in testset["poses"]:
        assert camera(problem, pose) == fork_camera(problem, pose), pose


def test_fork_base_cameras_are_ours():
    for problem in PROBLEMS:
        path = spec.FORK / "libero/libero/envs/problems" / f"{problem}.py"
        text = path.read_text()
        fn = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "_setup_camera")
        assigns = {
            t.id: ast.literal_eval(a.value)
            for a in fn.body
            if isinstance(a, ast.Assign)
            for t in a.targets
            if isinstance(t, ast.Name) and t.id in ("pos_av", "quat_av")
        }
        assert (tuple(assigns["pos_av"]), tuple(assigns["quat_av"])) == BASE_CAMERAS[problem]


def test_signed_angles_set_the_same_camera(testset):
    """C1: 350 and -10 are one rotation with opposite quaternion signs, so keys are sign-normalised. One pose,
    (330, 15), rounds its position 1e-4 apart between 330 and -30 in three scene classes (a rounding boundary);
    the test-camera set holds both cameras."""
    sign_only, boundary = 0, []
    for problem in PROBLEMS:
        for pose in testset["poses"]:
            a, b = camera(problem, pose), camera(problem, signed(pose))
            sign_only += a != b and camera_key(*a) == camera_key(*b)
            if camera_key(*a) != camera_key(*b):
                boundary.append((problem, tuple(pose)))
            assert {camera_key(*a), camera_key(*b)} <= sample.test_keys(problem)
    assert sign_only > 0
    assert len(boundary) <= MAX_BOUNDARY_POSES, boundary
    r350, r_10 = camera(PROBLEMS[0], (0, 0, 100, 350, 0)), camera(PROBLEMS[0], (0, 0, 100, -10, 0))
    assert r350 != r_10 and camera_key(*r350) == camera_key(*r_10)


def test_spec_rederives_the_committed_testset():
    assert spec.main([]) == 0


def test_spec_refuses_a_changed_table():
    raw = json.loads(spec.load_cases(None))
    raw["libero_spatial"] = [
        r for r in raw["libero_spatial"] if not r["name"].endswith("_view_0_0_115_0_0_initstate_0")
    ]
    tables, _ = spec.derive(raw, spec.FORK)
    with pytest.raises(SystemExit, match="disagrees"):
        spec.check(tables)


@pytest.fixture(scope="module")
def draws():
    return [
        sample.train_view(f"libero_spatial/task:demo_{i % 50}", i, PROBLEMS[i % len(PROBLEMS)]) for i in range(N_DRAWS)
    ]


def test_family_and_beyond_shares(draws):
    total = sum(sample.FAMILY_CASES.values())
    for family, n in sample.FAMILY_CASES.items():
        mine = [d for d in draws if d["family"] == family]
        assert abs(len(mine) / N_DRAWS - n / total) < SHARE_TOL
        share = sum(d["bin"] == "training_range_beyond_test" for d in mine) / len(mine)
        assert abs(share - BEYOND[family]) < SHARE_TOL, family
    assert abs(sum(d["bin"] == "training_range_beyond_test" for d in draws) / N_DRAWS - BEYOND_ALL) < SHARE_TOL


def test_magnitudes_are_uniform(draws):
    def ks(values, bounds):
        lo, hi = bounds
        return stats.kstest(values[:KS_N], stats.uniform(loc=lo, scale=hi - lo).cdf).pvalue

    orbit = [d["pose"] for d in draws if d["family"] == "orbit"]
    turn = [d["pose"] for d in draws if d["family"] == "turn_tilt"]
    level = sum(p[1] == 0 for p in orbit) / len(orbit)
    assert abs(level - sample.LEVEL_SHARE) < SHARE_TOL
    assert ks([abs(p[0]) for p in orbit], sample.ORBIT_ABS_H) > KS_P
    assert ks([p[1] for p in orbit if p[1]], sample.RAISED_V) > KS_P
    assert ks([d["pose"][2] for d in draws if d["family"] == "distance"], sample.DISTANCE_S) > KS_P
    assert ks([abs(p[3]) for p in turn], sample.TURN_ABS) > KS_P
    assert ks([abs(p[4]) for p in turn], sample.TURN_ABS) > KS_P
    assert abs(np.mean([p[0] > 0 for p in orbit]) - sample.FAIR) < SHARE_TOL


def test_no_draw_sets_a_test_camera_and_redraws_happen(draws):
    by_problem = {p: [d for i, d in enumerate(draws) if PROBLEMS[i % len(PROBLEMS)] == p] for p in PROBLEMS}
    for problem, ds in by_problem.items():
        assert not any(sample.hits_test(problem, d["pose"]) for d in ds)
    for family, (low, high) in REDRAW_RATE.items():
        redraws = [d["redraws"] for d in draws if d["family"] == family]
        assert low <= sum(redraws) / (len(redraws) + sum(redraws)) < high, family


def test_turn_and_tilt_signs_are_independent(draws):
    """A shared sign would make them agree on every draw; independent signs agree half the time."""
    turn = [d["pose"] for d in draws if d["family"] == "turn_tilt"]
    agree = np.mean([(p[3] > 0) == (p[4] > 0) for p in turn])
    assert abs(agree - sample.FAIR) < SHARE_TOL


def test_draws_are_seeded_by_episode_and_frame():
    a = sample.train_view("libero_goal/x:demo_3", 17, PROBLEMS[0])
    assert a == sample.train_view("libero_goal/x:demo_3", 17, PROBLEMS[0])
    assert a != sample.train_view("libero_goal/x:demo_3", 18, PROBLEMS[0])
    assert a != sample.train_view("libero_goal/x:demo_4", 17, PROBLEMS[0])


def test_the_check_catches_a_test_camera():
    """The guard must fire on a real test pose, including one written with a signed angle."""
    assert sample.hits_test(PROBLEMS[0], (0, 0, 150, 0, 0))
    assert sample.hits_test(PROBLEMS[0], (0, 0, 100, -10, -2))
    assert not sample.hits_test(PROBLEMS[0], (0, 0, 150.5, 0, 0))


def test_ladder_has_24_views_off_the_test_cameras_and_off_the_draws(draws):
    drawn = {tuple(d["pose"]) for d in draws}
    for problem in PROBLEMS:
        for split in sample.SPLIT_CODES:
            for moment in range(MOMENTS):
                views = sample.ladder(split, "libero_10/t:demo_1", moment, problem)
                assert len(views) == LADDER_VIEWS
                assert not drawn & {tuple(v["pose"]) for v in views}
    bins = {v["bin"] for v in sample.ladder("val", "e", 0, PROBLEMS[0])}
    assert bins == {"below_test_range", "in_test_range", "training_range_beyond_test", "beyond_training"}


def test_ladder_signs_follow_the_state():
    a = sample.ladder("val", "libero_10/t:demo_1", 0, PROBLEMS[0])
    assert a == sample.ladder("val", "libero_10/t:demo_1", 0, PROBLEMS[0])
    assert [abs(x) for v in a for x in v["pose"]] == [
        abs(x) for v in sample.ladder("test", "libero_10/t:demo_9", 3, PROBLEMS[0]) for x in v["pose"]
    ]


def test_nominal_is_the_fork_default():
    for problem in PROBLEMS:
        assert camera(problem, NOMINAL) == fork_camera(problem, [0, 0, 100, 0, 0])
