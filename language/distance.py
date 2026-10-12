# ABOUTME: Distance from each LIBERO-Plus language test instruction to its nearest training instruction pair, and the
# ABOUTME: paper's distance bands. Usage: python distance.py --cases CASES.jsonl --pairs PAIRS.jsonl --out OUT.jsonl
"""The distance is the normalised edit distance of the paper (Section VIII): both strings are lowercased, every
character other than a letter, a digit or a space becomes a space, runs of spaces collapse to one; then the
Levenshtein distance (single-character insertions, deletions and substitutions) is divided by the length of the longer
string. Bands include their upper bound: d <= 0.10, 0.10 < d <= 0.30, d > 0.30. The bounds are compared in exact
integer arithmetic (10 * edits <= k * length) wherever a float distance is at a bound, so no rounding moves a case
across one.

CASES.jsonl: one object per language test case, {"case_id": "<suite>/<name>", "instruction": "<the prompt the benchmark
serves>"}; you build it from your own copy of LIBERO-Plus (case_ids as in ../results). PAIRS.jsonl: the training
instruction pairs, {"instruction": ..., "source": ...}; rows with source "identity" are skipped. OUT.jsonl gets
{"case_id", "distance", "band"} per case, in the order of CASES.jsonl. Prints the case count per band and, per
source, the share of pairs within 0.30 of a test instruction. Needs rapidfuzz and numpy.
"""

import argparse
import json
import re
from collections import Counter

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

TENTHS = 10  # bounds are k tenths: 1 -> 0.10, 3 -> 0.30
NEAR, MIDDLE = 1, 3
BANDS = ("d <= 0.10", "0.10 < d <= 0.30", "d > 0.30")
DECIMALS = 4
TIE = 1e-9  # float distances this close to a bound are re-decided in integers


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())


def distances(tests: list[str], pairs: list[str]) -> np.ndarray:
    return cdist(tests, pairs, scorer=Levenshtein.normalized_distance, workers=-1, dtype=np.float64)


def within(d: np.ndarray, tests: list[str], pairs: list[str], k: int) -> np.ndarray:
    """True where the distance is at most k tenths. Entries within TIE of the bound are decided in integers."""
    bound = k / TENTHS
    out = d <= bound
    for i, j in zip(*np.nonzero(np.abs(d - bound) < TIE), strict=True):
        out[i, j] = TENTHS * Levenshtein.distance(tests[i], pairs[j]) <= k * max(len(tests[i]), len(pairs[j]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = [json.loads(line) for line in open(args.cases) if line.strip()]
    rows = [json.loads(line) for line in open(args.pairs) if line.strip()]
    rows = [r for r in rows if r["source"] != "identity"]
    tests = sorted({norm(c["instruction"]) for c in cases})
    pairs = [norm(r["instruction"]) for r in rows]
    d = distances(tests, pairs)
    near, middle = within(d, tests, pairs, NEAR), within(d, tests, pairs, MIDDLE)
    nearest = d.min(axis=1)
    index = {t: i for i, t in enumerate(tests)}
    counts = Counter()
    with open(args.out, "w") as out:
        for c in cases:
            i = index[norm(c["instruction"])]
            band = BANDS[0] if near[i].any() else BANDS[1] if middle[i].any() else BANDS[2]
            counts[band] += 1
            out.write(json.dumps({"case_id": c["case_id"], "distance": round(float(nearest[i]), DECIMALS), "band": band}))
            out.write("\n")
    print("cases per band:", {b: counts[b] for b in BANDS})
    close = middle.any(axis=0)
    sources = [r["source"].split(":")[0] for r in rows]
    for s in sorted(set(sources)):
        sel = [c for c, src in zip(close, sources, strict=True) if src == s]
        print(f"pairs of source {s} within 0.30 of a test instruction: {sum(sel)} of {len(sel)} ({100 * sum(sel) / len(sel):.2f} %)")


if __name__ == "__main__":
    main()
