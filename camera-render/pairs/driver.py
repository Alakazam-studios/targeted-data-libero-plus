# ABOUTME: Renders each job's demo states at the nominal camera and its moved views by the direct pose path (the fork's
# ABOUTME: transform set on "agentview"), asserts wrist and state bit-identical, writes one npz + receipt per job.
"""python -m pairs.driver --jobs jobs.json --kept kept.json --hdf5-dir /h5 --splits splits.json --out /out
[--workers N]

Runs inside a render image (Python 3.8, OSMesa, the pinned fork) with PYTHONPATH holding this folder and a `render`
module with its clip-generation dependency (not part of this release). The scene env is built once per job at the fork's
default camera and fixtures are placed as the demo recorded them (render.align_fixtures); each view then sets the
camera's compiled pos/quat directly. The quaternion is normalised by MuJoCo's own XML compiler, exactly as the fork's
name-encoded path (robosuite set_camera -> XML) is. Kind "c1" also builds the fork's env per pose and compares."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, cast

import numpy as np
from cameras import sample
from cameras.spec import signed
from cameras.transform import NOMINAL, camera, camera_key

MOMENTS = (0.1, 0.3, 0.5, 0.7, 0.9)
# the base recipe's training horizon (its trainer's ACTION_HORIZON = 10, openpi pi05_libero the same)
ACTION_HORIZON = 10
TRACE_CHARS = 2000
DATA_KINDS = ("train", "ladder")  # views that become training or evaluation data must never set a test camera
CAMERA_XML = '<mujoco><worldbody><camera name="c" pos="{}" quat="{}"/></worldbody></mujoco>'


class Refused(RuntimeError):
    pass


def as_text(values) -> str:
    """robosuite's array_to_string, which the fork's set_camera writes into the scene XML."""
    return " ".join(f"{float(x)}" for x in values)


def compiled(pos, quat) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    m = mujoco.MjModel.from_xml_string(CAMERA_XML.format(as_text(pos), as_text(quat)))
    return np.array(m.cam_pos[0]), np.array(m.cam_quat[0])


def set_agentview(env, pose, problem: str) -> list[float]:
    cid = env.sim.model.camera_name2id("agentview")
    p, q = compiled(*camera(problem, pose))
    env.sim.model.cam_pos[cid] = p
    env.sim.model.cam_quat[cid] = q
    return [float(x) for x in np.concatenate([p, q])]


def signed_pose(pose) -> list[float]:
    h, v, s, r, e = pose
    return [signed(h), signed(v), s, signed(r), signed(e)]


def frames_of(spec: str, n: int) -> list[int]:
    named = {"all": list(range(n)), "first": [0], "middle": [n // 2]}
    return named[spec] if spec in named else [min(n - 1, int(f * n)) for f in MOMENTS]


def views_of(job: dict, t: int, moment: int) -> list[dict]:
    if job["views"] == "draw":
        return [sample.train_view(job["episode"], t, job["problem"])]
    if job["views"] == "ladder":
        return sample.ladder(job["split"], job["episode"], moment, job["problem"])
    return [dict(v) for v in job["views"]]


def chunk_rows(kept: list[int], t: int, horizon: int = ACTION_HORIZON) -> list[int]:
    """HDF5 rows of the action chunk at kept frame t, as the base recipe's training loader builds it: the
    next `horizon` kept frames from t itself, the episode's last kept frame repeated past its end."""
    return [kept[min(t + j, len(kept) - 1)] for j in range(horizon)]


def identical(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))


def max_abs_diff(a, b) -> int | None:
    """Largest per-pixel difference, or None when the shapes differ."""
    return int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()) if a.shape == b.shape else None


def c1_row(v: dict, fork_camera: list, same_key: bool, a_fork, a_direct, a_signed, a_rerender) -> dict:
    """The C1 comparisons of one pose. a_direct and a_signed were rendered on the job's env before any fork env was
    built, so the signed claim cannot pass on a renderer that changed afterwards; a_rerender (the unsigned pose again,
    after the fork env) is a recorded diagnostic of that change (a smoke run, 2026-09-29), never gated."""
    return {
        "pose": v["pose"],
        "model_camera_equal": fork_camera == v["camera"],
        "agentview_identical": identical(a_fork, a_direct),
        "signed_camera_key_equal": same_key,
        "signed_agentview_identical": identical(a_signed, a_direct),
        "rerender_identical": identical(a_rerender, a_direct),
        "fork_vs_direct_max": max_abs_diff(a_fork, a_direct),
        "signed_vs_direct_max": max_abs_diff(a_signed, a_direct),
        "rerender_vs_direct_max": max_abs_diff(a_rerender, a_direct),
    }


def render_views(cv, env, job: dict, vs: list[dict], state, problem: str, wrist, robot) -> tuple[list, list, int]:
    """Each view of one state on the job's env: (images, signed-pose images for a c1 job, identity failures). The
    signed pose is rendered right after its direct render, before fork_compare builds any other env."""
    imgs, signed, failures = [], [], 0
    for v in vs:
        if job["kind"] in DATA_KINDS and sample.hits_test(problem, v["pose"]):
            raise Refused(f"view {v['pose']} sets a test camera")
        v["camera"] = set_agentview(env, v["pose"], problem)
        a, w, x = cv.read_frame(cv.restore(env, state))
        failures += not (identical(w, wrist) and identical(x, robot))
        imgs.append(a)
        if job["kind"] == "c1":
            set_agentview(env, signed_pose(v["pose"]), problem)
            signed.append(cv.read_frame(cv.restore(env, state))[0])
    return imgs, signed, failures


def fork_compare(cv, bddl: str, xml: str, state, env, views: list[dict], direct: list, signed: list, problem: str):
    """C1: each pose through the fork's own name-encoded env vs the direct path; the signed pose (350 -> -10) was
    rendered before (render_views); the unsigned pose is re-rendered after the fork env as a diagnostic."""
    out = []
    for i, v in enumerate(views):
        fenv = cv.make_env(bddl, v["pose"], cv.RES)
        cv.align_fixtures(fenv, xml)
        fork_camera = cv.agentview_pose(fenv)
        a_fork = cv.read_frame(cv.restore(fenv, state))[0]
        fenv.close()
        set_agentview(env, v["pose"], problem)
        a_rerender = cv.read_frame(cv.restore(env, state))[0]
        sp = signed_pose(v["pose"])
        same_key = camera_key(*camera(problem, sp)) == camera_key(*camera(problem, v["pose"]))
        out.append(c1_row(v, fork_camera, same_key, a_fork, direct[i], signed[i], a_rerender))
    return out


def c1_pass(rows: list[dict]) -> bool:
    return bool(rows) and all(
        r["model_camera_equal"]
        and r["agentview_identical"]
        and (r["signed_agentview_identical"] or not r["signed_camera_key_equal"])
        for r in rows
    )


def view_arrays(labels: list[list[dict]]) -> dict:
    """Per [frame, view]: pose, the camera set, and the family and range bin as fixed-width unicode (numpy loads them
    without allow_pickle), so a consumer of the npz alone needs no receipt."""
    return {
        "pose": np.array([[v["pose"] for v in vs] for vs in labels]),
        "camera": np.array([[v["camera"] for v in vs] for vs in labels]),
        "family": np.array([[v["family"] for v in vs] for vs in labels], dtype=str),
        "bin": np.array([[v["bin"] for v in vs] for vs in labels], dtype=str),
    }


def render_job(payload) -> dict:
    job, info, hdf5_dir, rel, out_dir = payload
    out = Path(out_dir)
    receipt_path = out / "receipts" / (job["name"] + ".json")
    t0 = time.time()
    try:
        if receipt_path.is_file() and (out / "pairs" / (job["name"] + ".npz")).is_file():
            try:
                return {**json.loads(receipt_path.read_text()), "resumed": True}
            except ValueError:  # a receipt copied back truncated: render the job again
                pass
        return run_job(job, info, hdf5_dir, rel, out, receipt_path, t0)
    except Exception as e:  # one job's failure is recorded, never fatal to the others
        return {
            "name": job["name"],
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-TRACE_CHARS:],
        }


def write_atomic(path: Path, text: str) -> None:
    """Never a half-written file under its final name (a copier of these dirs can skip .partial- files)."""
    tmp = path.with_name(".partial-" + path.name)
    tmp.write_text(text)
    os.replace(str(tmp), str(path))


def run_all(fn, items: list, workers: int) -> list[dict]:
    """fn(payload) for each (name, payload), at most `workers` at once, each in its own fresh spawned process (the
    renderer leaks memory per process). A worker that dies (OOM kill) fails its job as BrokenProcessPool
    instead of hanging the run, as mp.Pool would."""
    spawn = mp.get_context("spawn")
    queue, running, results = list(items), {}, []
    while queue or running:
        while queue and len(running) < workers:
            name, payload = queue.pop(0)
            ex = ProcessPoolExecutor(max_workers=1, mp_context=spawn)
            running[ex.submit(fn, payload)] = (ex, name)
        done, _ = wait(list(running), return_when=FIRST_COMPLETED)
        for f in done:
            ex, name = running.pop(f)
            try:
                results.append(f.result())
            except BrokenProcessPool as e:
                results.append({"name": name, "ok": False, "error": f"worker process died: {e}"})
            except Exception as e:  # e.g. a result that cannot be pickled back: fail the job, keep every other result
                results.append({"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"})
            ex.shutdown(wait=True)
    return results


def run_job(job: dict, info: dict, hdf5_dir: str, rel: str, out: Path, receipt_path: Path, t0: float) -> dict:
    import h5py
    import render as cv

    with h5py.File(str(Path(hdf5_dir) / rel), "r") as f:
        g = f["data"][job["demo"]]
        states, actions, xml = g["states"][()], g["actions"][()], g.attrs["model_file"]
    kept, problem = info["kept"], job["problem"]
    env = cv.make_env(info["bddl"], None, cv.RES)
    if type(env.env).__name__.lower() != problem:
        raise Refused(f"env class {type(env.env).__name__} is not {problem}")
    fixtures = cv.align_fixtures(env, xml)
    if cv.agentview_pose(env) != set_agentview(env, NOMINAL, problem):
        raise Refused("the fork's default camera differs from transform.camera(NOMINAL)")
    ts = frames_of(job["frames"], len(kept))
    nominal, views, wrist, state, labels, c1 = [], [], [], [], [], []
    identity_failures = 0
    for moment, t in enumerate(ts):
        s_k = states[kept[t]]
        set_agentview(env, NOMINAL, problem)
        a0, w0, x0 = cv.read_frame(cv.restore(env, s_k))
        vs = views_of(job, t, moment)
        imgs, signed, failures = render_views(cv, env, job, vs, s_k, problem, w0, x0)
        identity_failures += failures
        if job["kind"] == "c1":
            c1 += fork_compare(cv, info["bddl"], xml, s_k, env, vs, imgs, signed, problem)
        nominal.append(a0)
        views.append(np.stack(imgs))
        wrist.append(w0)
        state.append(x0)
        labels.append(vs)
    env.close()
    arrays = {
        "nominal": np.stack(nominal),
        "views": np.stack(views),
        "wrist": np.stack(wrist),
        "state": np.stack(state).astype(np.float32),
        "frame": np.array(ts),
        "action": np.ascontiguousarray(actions[[kept[t] for t in ts]], dtype=np.float32),
        "chunk": np.ascontiguousarray(np.stack([actions[chunk_rows(kept, t)] for t in ts]), dtype=np.float32),
        **view_arrays(labels),
    }
    npz = out / "pairs" / (job["name"] + ".npz")
    tmp = npz.with_name(".partial-" + npz.name)
    with open(str(tmp), "wb") as fh:
        np.savez_compressed(fh, **cast("dict[str, Any]", arrays))
    os.replace(str(tmp), str(npz))
    receipt = {
        "name": job["name"],
        "ok": identity_failures == 0 and (job["kind"] != "c1" or c1_pass(c1)),
        "job": {k: v for k, v in job.items() if k != "views"},
        "instruction": info["instruction"],
        "frames": ts,
        "renders": len(ts) * (1 + len(labels[0])),
        "identity_failures": identity_failures,
        "redraws": sum(v.get("redraws", 0) for vs in labels for v in vs),
        "fixtures_moved_mm": fixtures["max_mm"],
        "views": labels,
        "c1": c1,
        "seconds": round(time.time() - t0, 1),
    }
    write_atomic(receipt_path, json.dumps(receipt) + "\n")
    return receipt


def manifest(out: Path) -> dict:
    """sha256 of the bank's content: every npz, every job receipt, the kept frames and the run summary."""
    files = [*sorted((out / "pairs").glob("[!.]*.npz")), *sorted((out / "receipts").glob("[!.]*.json"))]
    files += [p for p in (out / "kept.json", out / "pairs.json") if p.is_file()]
    return {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


# A smoke is the real run shrunk to 3 units. A unit is one
# problem class's slice: its C1 check, one corner job, one train episode, one ladder episode. These three classes have
# the most different base cameras (the common tabletop one, the living room's closest one, the floor's below-pivot one);
# the study and kitchen classes are covered by the full run's own C1 check.
SMOKE_PROBLEMS = (
    "libero_tabletop_manipulation",
    "libero_living_room_tabletop_manipulation",
    "libero_floor_manipulation",
)


def smoke_jobs(jobs: list[dict]) -> list[dict]:
    """The first job of each (kind, problem class) for the SMOKE_PROBLEMS, in the stage's order, for every kind the
    job set holds (the training set: c1, corner, train, ladder; the val set: ladder)."""
    seen: dict = {}
    for j in jobs:
        if j["problem"] in SMOKE_PROBLEMS:
            seen.setdefault((j["kind"], j["problem"]), j)
    kinds = sorted({j["kind"] for j in jobs})
    missing = [(k, p) for p in SMOKE_PROBLEMS for k in kinds if (k, p) not in seen]
    if missing:
        raise SystemExit(f"REFUSED: the smoke has no job for {missing}")
    return list(seen.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--jobs", "--kept", "--hdf5-dir", "--out", "--splits"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="render one job per (kind, problem class) for 3 problem classes (the run, shrunk)",
    )
    a = ap.parse_args(argv)
    jobs, kept = json.loads(Path(a.jobs).read_text()), json.loads(Path(a.kept).read_text())
    if a.smoke:
        jobs = smoke_jobs(jobs)
    tasks = json.loads(Path(a.splits).read_text())["tasks"]
    out = Path(a.out)
    for d in ("pairs", "receipts"):
        (out / d).mkdir(parents=True, exist_ok=True)
    payloads = [(j, kept[j["episode"]], a.hdf5_dir, tasks[j["task"]]["hdf5"], str(out)) for j in jobs]
    t0 = time.time()
    results = run_all(render_job, [(p[0]["name"], p) for p in payloads], a.workers)
    failed = sorted(r["name"] for r in results if not r["ok"])
    c1 = [row for r in results for row in r.get("c1", [])]
    summary = {
        "mode": "smoke" if a.smoke else "pairs",
        "result": "PASS" if not failed and len(results) == len(jobs) else "FAIL",
        "jobs": len(jobs),
        "ok": len(jobs) - len(failed),
        "failed": failed[:20],
        "errors": {r["name"]: r["error"] for r in results if "error" in r},
        "renders": sum(r.get("renders", 0) for r in results),
        "identity_failures": sum(r.get("identity_failures", 0) for r in results),
        "redraws": sum(r.get("redraws", 0) for r in results),
        "c1_poses": len(c1),
        "c1_pass": c1_pass(c1),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "pairs.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest(out), indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}))
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
