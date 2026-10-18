# ABOUTME: Recomputes the paper's scores from the per-case files in this folder: each model's 7 axis rates and 7-axis
# ABOUTME: mean, and with --pair A B the paired difference A - B per axis with its paired SE and a z = 1.96 verdict.
"""python3 results/recompute.py                       every file in MANIFEST.json: axis rates and 7-axis mean (%)
python3 results/recompute.py --pair main-model pi05-libero    paired difference, mxri-lg-va-1 minus base pi0.5

Standard library only. Units: pt = percentage points of the axis's cases that succeeded. The 7-axis mean is the
unweighted mean of the 7 axis rates. A pair (A, B) on one axis of n cases: a_only = cases A succeeded and B failed,
b_only the reverse; d = 100 (a_only - b_only) / n; paired SE = 100 sqrt(a_only + b_only - (a_only - b_only)^2 / n) / n.
7-axis: d = mean of the 7 axis d, SE = sqrt(sum of the 7 axis SE^2) / 7 (the axes are disjoint case sets). The
difference is called beyond noise when |d| > 1.96 SE (two-sided 95 %)."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
AXES = ("Camera Viewpoints", "Robot Initial States", "Language Instructions", "Light Conditions",
        "Background Textures", "Sensor Noise", "Objects Layout")
SHORT = ("Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout")
Z = 1.96
CASES = 10030


def load(stem: str, entry: dict) -> dict[str, dict]:
    body = gzip.decompress((HERE / entry["file"]).read_bytes())
    if hashlib.sha256(body).hexdigest() != entry["rows_sha256_uncompressed"]:
        raise SystemExit(f"{stem}: rows differ from MANIFEST.json")
    rows = {r["case_id"]: r for r in map(json.loads, body.decode().splitlines())}
    if len(rows) != CASES:
        raise SystemExit(f"{stem}: {len(rows)} cases, expected {CASES}")
    return rows


def rates(rows: dict[str, dict]) -> dict[str, float]:
    out = {}
    for axis in AXES:
        hits = [r["success"] for r in rows.values() if r["axis"] == axis]
        out[axis] = 100 * sum(hits) / len(hits)
    return out


def pair(a: dict[str, dict], b: dict[str, dict]) -> list[tuple[str, float, float]]:
    if a.keys() != b.keys():
        raise SystemExit("the two files do not hold the same cases")
    readings = []
    for axis in AXES:
        ids = [k for k, r in a.items() if r["axis"] == axis]
        n = len(ids)
        a_only = sum(a[k]["success"] and not b[k]["success"] for k in ids)
        b_only = sum(b[k]["success"] and not a[k]["success"] for k in ids)
        d = 100 * (a_only - b_only) / n
        se = 100 * math.sqrt(a_only + b_only - (a_only - b_only) ** 2 / n) / n
        readings.append((axis, d, se))
    d7 = sum(d for _, d, _ in readings) / len(AXES)
    se7 = math.sqrt(sum(se**2 for _, _, se in readings)) / len(AXES)
    return readings + [("7-axis mean", d7, se7)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), help="file stems, e.g. main-model pi05-libero")
    args = ap.parse_args()
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    entries = {e["file"].removesuffix(".jsonl.gz"): e for e in manifest["files"]}
    if args.pair:
        a, b = (load(s, entries[s]) for s in args.pair)
        print(f"{args.pair[0]} minus {args.pair[1]}, paired by case (pt = percentage points)")
        for name, d, se in pair(a, b):
            verdict = "beyond noise" if abs(d) > Z * se else "within noise"
            print(f"  {name:22s} d = {d:+7.2f} pt   SE = {se:5.2f} pt   95% CI [{d - Z * se:+.2f}, {d + Z * se:+.2f}]"
                  f"   {verdict}")
        return
    print(f"{'file':24s} {'7-axis %':>8s} " + " ".join(f"{s:>10s}" for s in SHORT) + "   model")
    for stem, entry in entries.items():
        r = rates(load(stem, entry))
        mean = sum(r.values()) / len(AXES)
        print(f"{stem:24s} {mean:8.2f} " + " ".join(f"{r[a]:10.2f}" for a in AXES) + f"   {entry['model']}")


if __name__ == "__main__":
    main()
