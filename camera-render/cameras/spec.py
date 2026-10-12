# ABOUTME: Re-derives the LIBERO-Plus test-camera tables from the case list and the fork, refusing on any
# ABOUTME: mismatch, and writes testset.json (the 451 test poses as the case names write them, each task's scene class).
"""uv run python -m cameras.spec [--cases <task_classification.json>] [--fork <LIBERO-plus checkout>] [--write]

Defaults: --fork $LIBERO_PLUS_FORK (else ./LIBERO-plus); --cases the fork's own
libero/libero/benchmark/task_classification.json. Without --write it only checks that the committed testset.json
equals the derivation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

from cameras.transform import BASE_CAMERAS, NOMINAL, PIVOT

ROOT = Path(__file__).resolve().parents[1]
TESTSET = ROOT / "cameras" / "testset.json"
CASES_IN_FORK = Path("libero/libero/benchmark/task_classification.json")
FORK_PIN = "4976dc30028e805ff8094b55501d532c48fec182"
FORK = Path(os.environ.get("LIBERO_PLUS_FORK", "LIBERO-plus"))
CATEGORY = "Camera Viewpoints"
NAME = re.compile(r"^(.+)_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_initstate_0$")
FULL_TURN, HALF_TURN = 360, 180
RAISED_V = 15
BIN_WIDTH = 10
TURN_VALUES = {2, 4, 6, 8, 10}

FAMILY_CASES: dict[str, int] = {"orbit_level": 443, "orbit_raised": 549, "distance": 313, "turn_tilt": 294}
EXPECTED = {
    "cases": 1599,
    "distinct_poses": 450,
    "family_cases": FAMILY_CASES,
    "orbit_distinct_h": {"orbit_level": 128, "orbit_raised": 149},
    "orbit_h_range": [-75, 75],
    "orbit_abs_h_bins": {
        "orbit_level": [8, 29, 53, 73, 77, 80, 76, 47],
        "orbit_raised": [53, 68, 72, 80, 78, 77, 73, 48],
    },
    "distance_range": [115, 200],
    "distance_distinct": 83,
    "turn_tilt_values": sorted(TURN_VALUES | {-t for t in TURN_VALUES}),
    "tasks_per_problem": {
        "libero_tabletop_manipulation": 20,
        "libero_kitchen_tabletop_manipulation": 4,
        "libero_study_tabletop_manipulation": 1,
        "libero_living_room_tabletop_manipulation": 5,
        "libero_floor_manipulation": 10,
    },
}
DISTANCE_BINS = {"110s": 7, "120s": 31, "200": 4}
DISTANCE_MIDDLE_BINS = (36, 40)
BASE_GEOMETRY = {"tabletop": (1.04, 51), "study": (0.93, 60), "living_room": (0.63, 15), "floor": (0.91, -9.5)}
GEOMETRY_TOL_M, GEOMETRY_TOL_DEG = 0.01, 0.5
ORBIT_LAST_BIN = 7


def signed(angle: float) -> float:
    return angle - FULL_TURN if angle > HALF_TURN else angle


def family(pose) -> str:
    h, v, s, r, e = signed(pose[0]), signed(pose[1]), pose[2], signed(pose[3]), signed(pose[4])
    moved = {"orbit": bool(h or v), "distance": s != NOMINAL[2], "turn_tilt": bool(r or e)}
    if sum(moved.values()) != 1:
        raise SystemExit(f"REFUSED: pose {pose} moves {sum(moved.values())} families, not one")
    name = next(k for k, m in moved.items() if m)
    if name == "orbit":
        return "orbit_raised" if v == RAISED_V else "orbit_level"
    return name


def derive(cases_json: dict, fork: Path) -> tuple[dict, dict]:
    """(tables, testset): the checked tables and the testset.json payload."""
    rows = [(suite, r["name"]) for suite, rs in cases_json.items() for r in rs if r["category"] == CATEGORY]
    parsed = []
    for suite, name in rows:
        m = NAME.match(name)
        if not m:
            raise SystemExit(f"REFUSED: camera case {name} does not parse")
        parsed.append((f"{suite}/{m.group(1)}", tuple(int(x) for x in m.groups()[1:])))
    fams = {pose: family(pose) for _, pose in parsed}
    per_case = Counter(fams[pose] for _, pose in parsed)
    orbit_h = {f: sorted({signed(p[0]) for _, p in parsed if fams[p] == f}) for f in ("orbit_level", "orbit_raised")}
    bins = {
        f: [
            sum(1 for _, p in parsed if fams[p] == f and min(abs(signed(p[0])) // BIN_WIDTH, ORBIT_LAST_BIN) == b)
            for b in range(ORBIT_LAST_BIN + 1)
        ]
        for f in orbit_h
    }
    s_all = [p[2] for _, p in parsed if fams[p] == "distance"]
    turn = {signed(p[i]) for _, p in parsed if fams[p] == "turn_tilt" for i in (3, 4)}
    if any(signed(p[3]) == 0 or signed(p[4]) == 0 for _, p in parsed if fams[p] == "turn_tilt"):
        raise SystemExit("REFUSED: a turn/tilt case leaves R or E at 0")
    problems = {}
    for task in sorted({t for t, _ in parsed}):
        text = (fork / "libero/libero/bddl_files" / f"{task}.bddl").read_text()
        found = re.search(r"\(problem\s+(\S+)\)", text)
        if not found:
            raise SystemExit(f"REFUSED: {task}.bddl names no problem class")
        problems[task] = found.group(1).lower()
    tables = {
        "cases": len(parsed),
        "distinct_poses": len(fams),
        "family_cases": dict(per_case),
        "orbit_distinct_h": {f: len(h) for f, h in orbit_h.items()},
        "orbit_h_range": [min(min(h) for h in orbit_h.values()), max(max(h) for h in orbit_h.values())],
        "orbit_abs_h_bins": bins,
        "distance_range": [min(s_all), max(s_all)],
        "distance_distinct": len(set(s_all)),
        "turn_tilt_values": sorted(turn),
        "tasks_per_problem": dict(Counter(problems.values())),
    }
    s_bins = Counter(f"{s // BIN_WIDTH * BIN_WIDTH}s" if s < max(s_all) else str(s) for s in s_all)
    middle = [n for k, n in s_bins.items() if k not in DISTANCE_BINS]
    tables["distance_bins_ok"] = all(s_bins[k] == n for k, n in DISTANCE_BINS.items()) and all(
        DISTANCE_MIDDLE_BINS[0] <= n <= DISTANCE_MIDDLE_BINS[1] for n in middle
    )
    poses = sorted(fams) + [tuple(int(x) for x in NOMINAL)]
    testset = {"fork_pin": FORK_PIN, "poses": [list(p) for p in poses], "problems": problems}
    return tables, testset


def check_geometry() -> None:
    from cameras.transform import CAMERA_CLASS

    for problem, (pos, _) in BASE_CAMERAS.items():
        dx, dz = math.hypot(pos[0] - PIVOT[0], pos[1] - PIVOT[1]), pos[2] - PIVOT[2]
        dist, elev = BASE_GEOMETRY[CAMERA_CLASS[problem]]
        if (
            abs(math.hypot(dx, dz) - dist) > GEOMETRY_TOL_M
            or abs(math.degrees(math.atan2(dz, dx)) - elev) > GEOMETRY_TOL_DEG
        ):
            raise SystemExit(f"REFUSED: base camera of {problem} is not {dist} m / {elev} deg from the pivot")


def check(tables: dict) -> None:
    bad = {k: (tables[k], v) for k, v in EXPECTED.items() if tables[k] != v}
    if bad or not tables["distance_bins_ok"]:
        raise SystemExit(f"REFUSED: the case list disagrees with the expected tables: {bad or 'distance bins'}")
    check_geometry()


def load_testset() -> dict:
    return json.loads(TESTSET.read_text())


def load_cases(path: str | None, fork: Path = FORK) -> bytes:
    if path:
        return Path(path).read_bytes()
    return (fork / CASES_IN_FORK).read_bytes()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases")
    ap.add_argument("--fork", type=Path, default=FORK)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    raw = load_cases(a.cases, a.fork)
    tables, testset = derive(json.loads(raw), a.fork)
    check(tables)
    testset["cases_sha256"] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(testset, indent=1, sort_keys=True) + "\n"
    if a.write:
        TESTSET.write_text(text)
    elif TESTSET.read_text() != text:
        raise SystemExit(f"REFUSED: {TESTSET} differs from the derivation; rerun with --write")
    print(json.dumps({"result": "PASS", **{k: v for k, v in tables.items() if k != "orbit_abs_h_bins"}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
