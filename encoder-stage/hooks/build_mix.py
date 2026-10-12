# ABOUTME: The encoder stage's data hook: the base mix plus the moved-camera training bank, and the exclusion
# ABOUTME: input's bank frames listed in meta/exclude_frames.i64 for the trainer port to weight 0.
"""Run by the trainer port as `python build_mix.py <base mix, read-only> <out dir>` with P4C_INPUTS set (/inputs).
Two inputs: `exclusion` (the exclusion step's output: exclusion.npy, receipt.json, READY.json) and the bank, the
other input under $P4C_INPUTS, or the one named by ENCFT_BANK_INPUT. The bank is the camera re-render's LeRobot v2.1
train dataset: the base's features plus observation.images.front_nominal, and meta/frame_tags.npy (one row per
bank frame index).

The one change from the earlier data hook (exclusion rule amendment 2): the exclusion list's bank-local
frame indices are mapped to the built mix's `index` space and written as meta/exclude_frames.i64 (raw little-endian
int64, strictly increasing), declared on the bank source in mix_receipt.json (`exclude_frames`: path, sha256, count,
rule). The port's split.py sets those frames' training weight to 0, beside the validation hold-out; validation is
unchanged. Nothing else is dropped or re-rendered: every bank episode and frame is still appended.

`--weights-only --split <validation split json>` prints the spec's sampling weights on the EFFECTIVE training frames:
each source less its validation hold-out (Plus: the split's held-out demos' episodes; clips: the held-out clip sources
times the clip plan's frames and copies; bank: the episodes plus_links ties to a held-out Plus demo), and the bank also
less the excluded frames. It reads meta files only (the base's and the bank's), no parquet or video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

FPS = 20
CODEBASE = "v2.1"
FRONT = "observation.images.front"
WRIST = "observation.images.wrist"
NOMINAL = "observation.images.front_nominal"
STATE, ACTION = "observation.state", "action"
GRIPPER_VALUES = (-1.0, 1.0)
BANK_SHARE = 0.33
SHARE_ENV = "ENCFT_BANK_SHARE"
PLUS_SOURCE, CLIPS_SOURCE, BANK_SOURCE = "plus", "clips", "bank"
BASE_TAG = "p4c"
RFT_RECEIPT = "rft_receipt.json"
# The base recipe's clip plan: 648,000 clip frames of 2,886,036, 212,464 of them steer-back. Decision of 2026-10-01:
# every arm keeps the clips at their absolute drawn share, so steer-back is drawn at the base recipe's own share; the
# floor is that exact share, never the rounded 7.4 %, which the base recipe itself fails.
P4C_FRAMES, P4C_CLIP_FRAMES, P4C_STEER_FRAMES = 2_886_036, 648_000, 212_464
STEER_FLOOR = P4C_STEER_FRAMES / P4C_FRAMES
STEER_PER_CLIP_FRAME = P4C_STEER_FRAMES / P4C_CLIP_FRAMES
FLOAT32_REL_TOL = 1e-6
CLIPS_MIN_DRAWN_SHARE = STEER_FLOOR / STEER_PER_CLIP_FRAME * (1.0 - FLOAT32_REL_TOL)
# The trainer port's every-source hold-out: the bank re-renders Plus demos, so it declares
# `plus_demo` and writes meta/plus_links.jsonl, an index of the Plus episodes each bank episode re-renders, built from
# the two datasets' episodes_stats (task, length, action min / max and mean), independent of the port's action-sha
# matcher, which reads the parquet bytes. MEASURED on both datasets: the two stats writers' action means
# differ by at most 1.6e-4 relative; min and max agree in float32.
HOLDOUT, PLUS_DEMO_RULE, PLUS_LINKS = "holdout", "plus_demo", "plus_links.jsonl"
LINK_MEAN_RTOL = 1e-3
BANK_REASON = "same-MuJoCo-state re-renders of LIBERO human demos from a moved scene camera; human actions"
TRAIN_SPLIT = "train"
TAGS = "frame_tags.npy"
INSTRUCTION = "instruction"
INPUTS_ENV, BANK_ENV = "P4C_INPUTS", "ENCFT_BANK_INPUT"
EXCLUSION_INPUT = "exclusion"
EXCLUSION_FILES = ("exclusion.npy", "receipt.json", "READY.json")
EXCLUDE_FILE = "exclude_frames.i64"
EXCLUDE_DTYPE = np.dtype("<i8")
EXCLUSION_RULE = "pre-registered exclusion rule, amendment 2"
INDEX = "index"
POSE_FIELDS, CAMERA_FIELDS = 5, 7
FAMILY_WIDTH, BIN_WIDTH, SPLIT_WIDTH = 16, 32, 8
TAG_STRINGS = {"family": FAMILY_WIDTH, "bin": BIN_WIDTH, "split": SPLIT_WIDTH}
TAG_VECTORS = {"pose": POSE_FIELDS, "camera": CAMERA_FIELDS}
# The camera re-render's training sampler (camera-render/cameras/sample.py): FAMILY_CASES keys and train_view's
# bins.
TRAIN_FAMILIES = ("orbit", "distance", "turn_tilt")
TRAIN_BINS = ("training_range_beyond_test", "in_test_range")
PARITY_RECEIPT = "parity_receipt.json"
TAGS_SCHEMA = "frame_tags.json"
RECORDED_DIFFERENCES = ("state_max_abs", "front_mae_vs_plus", "wrist_mae_vs_plus")
MIN_MATCHED_EPISODES = 3
MIN_PARITY_FRAMES = 15
MOVED_MIN_MSE = 1.0
CODEC_NOISE_MSE = 25.0
MOVED_VIEW_EPISODES = 8
UNICODE_BYTES = 4
UNMOVED_POSE = (0.0, 0.0, 100.0, 0.0, 0.0)
BASE_FAMILY, BASE_BIN = "none", "original"
MIX_TAG_DTYPE = np.dtype(
    [("source", "U8"), ("family", f"U{FAMILY_WIDTH}"), ("bin", f"U{BIN_WIDTH}"), ("pose", "f4", (POSE_FIELDS,))]
)
ORIENTATION_TASKS = 8
ORIENTATION_BASE_EPISODES = 4
ORIENTATION_MARGIN = 0.8
PROBLEMS_SHOWN = 10


class BuildRefusal(RuntimeError):
    pass


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


def refuse_if(problems: list[str], what: str) -> None:
    if problems:
        shown = "\n  ".join(problems[:PROBLEMS_SHOWN])
        raise BuildRefusal(f"{what}: {len(problems)} problem(s):\n  {shown}")


class Dataset:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        meta = self.root / "meta"
        self.info = json.loads((meta / "info.json").read_text())
        self.tasks = {int(t["task_index"]): t["task"] for t in jsonl(meta / "tasks.jsonl")}
        self.episodes = jsonl(meta / "episodes.jsonl")
        self.stats = {int(s["episode_index"]): s["stats"] for s in jsonl(meta / "episodes_stats.jsonl")}
        self.chunk_size = int(self.info["chunks_size"])

    def data_path(self, episode: int) -> Path:
        chunk = episode // self.chunk_size
        return self.root / self.info["data_path"].format(episode_chunk=chunk, episode_index=episode)

    def video_path(self, episode: int, key: str) -> Path:
        chunk = episode // self.chunk_size
        return self.root / self.info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=episode)

    def video_keys(self) -> list[str]:
        return [key for key, feature in self.info["features"].items() if feature["dtype"] == "video"]

    def frames(self) -> int:
        return sum(int(e["length"]) for e in self.episodes)


def bank_root(inputs: str | None) -> Path:
    if not inputs:
        raise BuildRefusal(f"{INPUTS_ENV} is not set: the port stages the bank there")
    named = os.environ.get(BANK_ENV)
    if named:
        return Path(inputs) / named
    entries = sorted(p for p in Path(inputs).iterdir() if p.is_dir() and p.name != EXCLUSION_INPUT)
    if len(entries) != 1:
        raise BuildRefusal(f"{inputs} holds {len(entries)} inputs; name the bank with {BANK_ENV}")
    return entries[0]


def first_frame(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            return frame.to_ndarray(format="rgb24")
    raise BuildRefusal(f"{path} has no video frame")


ORIENTATIONS = {
    "as stored": lambda x: x,
    "rotated 180": lambda x: x[::-1, ::-1],
    "flipped up-down": lambda x: x[::-1],
    "flipped left-right": lambda x: x[:, ::-1],
}


def orientation_problems(base: Dataset, bank: Dataset, task_map: dict[str, int]) -> tuple[list[str], dict]:
    """Frame 0 of a bank episode's NOMINAL render against frame 0 of base episodes of the same task, under each
    orientation. Proves the bank is not rotated or flipped relative to the base where scenes of one task look alike;
    it cannot prove a pixel match with the same demo (the base carries no demo id), nor anything about later frames."""
    by_task: dict[int, list[int]] = {}
    base_index = {norm(text): index for index, text in base.tasks.items()}
    strays = sorted({e["tasks"][0] for e in base.episodes if norm(e["tasks"][0]) not in base_index})
    refuse_if([f"base episode task not in tasks.jsonl: {t!r}" for t in strays], "base episodes.jsonl")
    for e in base.episodes:
        by_task.setdefault(base_index[norm(e["tasks"][0])], []).append(int(e["episode_index"]))
    checked, problems, unmatched = {}, [], []
    seen: set[int] = set()
    for e in bank.episodes:
        task = task_map[norm(e["tasks"][0])]
        if task in seen or len(seen) >= ORIENTATION_TASKS:
            continue
        seen.add(task)
        candidates = by_task.get(task, [])[:ORIENTATION_BASE_EPISODES]
        if not candidates:
            unmatched.append(base.tasks[task])
            continue
        nominal = first_frame(bank.video_path(int(e["episode_index"]), NOMINAL)).astype(np.float64)
        refs = [first_frame(base.video_path(c, FRONT)).astype(np.float64) for c in candidates]
        if any(ref.shape != nominal.shape for ref in refs):
            problems.append(f"task {task}: nominal frame {nominal.shape} != base frame {refs[0].shape}")
            continue
        errors = {
            name: float(min(np.mean((turn(nominal) - ref) ** 2) for ref in refs)) for name, turn in ORIENTATIONS.items()
        }
        best_other = min(v for k, v in errors.items() if k != "as stored")
        checked[base.tasks[task]] = {"min_mse": errors, "as_stored_over_best_other": errors["as stored"] / best_other}
        if not errors["as stored"] < ORIENTATION_MARGIN * best_other:
            best = min(errors, key=errors.__getitem__)
            problems.append(f"task {task}: the bank's nominal view matches the base best {best}, not as stored")
    if not checked:
        problems.append(f"no bank task has a base episode to compare orientation with (unmatched: {unmatched})")
    return problems, {"checked": checked, "unmatched_tasks": unmatched}


def read_table(path: Path):
    import pyarrow.parquet as pq

    return pq.read_table(path)


def parity_problems(base: Dataset, bank: Dataset, tags: np.ndarray, task_map: dict[str, int]) -> tuple[list, dict]:
    problems: list[str] = []
    for name, data in (("base", base), ("bank", bank)):
        if data.info.get("codebase_version") != CODEBASE or data.info.get("fps") != FPS:
            problems.append(f"{name}: codebase {data.info.get('codebase_version')}, fps {data.info.get('fps')}")
    features = {k: v for k, v in bank.info["features"].items() if k != NOMINAL}
    if NOMINAL not in bank.info["features"]:
        problems.append(f"bank has no {NOMINAL}")
    if features != base.info["features"]:
        differ = sorted(
            k for k in features.keys() | base.info["features"].keys() if features.get(k) != base.info["features"].get(k)
        )
        problems.append(f"bank features differ from the base's (without {NOMINAL}): {differ}")
    base_schema = read_table(base.data_path(int(base.episodes[0]["episode_index"]))).schema
    stat_keys = set(base.stats[int(base.episodes[0]["episode_index"])])
    seen_index: list[np.ndarray] = []
    for e in bank.episodes:
        episode = int(e["episode_index"])
        table = read_table(bank.data_path(episode))
        if not table.schema.equals(base_schema, check_metadata=False):
            problems.append(f"bank episode {episode}: parquet schema differs from the base's")
            continue
        if table.num_rows == 0:
            problems.append(f"bank episode {episode}: empty parquet")
            continue
        if set(bank.stats.get(episode, {})) - {NOMINAL} != stat_keys:
            problems.append(f"bank episode {episode}: episodes_stats keys differ from the base's")
        gripper = np.asarray(table.column(ACTION).to_pylist(), dtype=np.float32)[:, -1]
        if not np.isin(gripper, GRIPPER_VALUES).all():
            problems.append(f"bank episode {episode}: gripper actions are not exactly {GRIPPER_VALUES}")
        index = np.asarray(table.column(INDEX).to_pylist(), dtype=np.int64)
        seen_index.append(index)
        if not np.array_equal(index, np.arange(index[0], index[0] + len(index))):
            problems.append(f"bank episode {episode}: parquet index is not contiguous")
        elif index.min() < 0 or index.max() >= len(tags):
            problems.append(f"bank episode {episode}: frame index outside frame_tags")
        elif (tags["split"][index] != TRAIN_SPLIT).any():
            problems.append(f"bank episode {episode}: frames whose split is not {TRAIN_SPLIT}")
        elif INSTRUCTION in tags.dtype.names and {norm(t) for t in tags[INSTRUCTION][index]} != {norm(e["tasks"][0])}:
            problems.append(f"bank episode {episode}: frame_tags instructions differ from the episode's task")
    every = np.sort(np.concatenate(seen_index)) if seen_index else np.zeros(0, np.int64)
    if not np.array_equal(every, np.arange(len(tags))):
        problems.append(f"bank parquet index values are not unique and 0..{len(tags) - 1}, one per frame_tags row")
    for key in base.video_keys():
        shape = tuple(base.info["features"][key]["shape"])
        frame = first_frame(bank.video_path(int(bank.episodes[0]["episode_index"]), key))
        if frame.shape != shape:
            problems.append(f"bank {key}: decoded {frame.shape} != feature shape {shape}")
    orientation, evidence = orientation_problems(base, bank, task_map)
    moved, moved_evidence = moved_view_problems(bank, tags)
    return problems + orientation + moved, evidence | {"moved_view": moved_evidence}


def moved_view_problems(bank: Dataset, tags: np.ndarray) -> tuple[list[str], dict]:
    """Frame 0 of the bank's front (moved) view against its front_nominal: different where the tag's pose is moved,
    equal within codec noise where it is the unmoved pose. Catches front and front_nominal being one render; it cannot
    tell whether the camera moved to the pose the tag states."""
    problems, measured = [], {}
    for e in bank.episodes[:MOVED_VIEW_EPISODES]:
        episode = int(e["episode_index"])
        index = read_table(bank.data_path(episode)).column(INDEX)
        if len(index) == 0:
            continue
        first = int(index[0].as_py())
        unmoved = bool(np.allclose(tags["pose"][first], UNMOVED_POSE))
        front = first_frame(bank.video_path(episode, FRONT)).astype(np.float64)
        nominal = first_frame(bank.video_path(episode, NOMINAL)).astype(np.float64)
        mse = float(np.mean((front - nominal) ** 2))
        measured[episode] = {"unmoved_pose": unmoved, "front_vs_nominal_mse": mse}
        if unmoved and mse > CODEC_NOISE_MSE:
            problems.append(f"bank episode {episode}: unmoved pose but front differs from nominal (MSE {mse:.3g})")
        if not unmoved and mse <= MOVED_MIN_MSE:
            problems.append(f"bank episode {episode}: moved pose but front equals nominal (MSE {mse:.3g})")
    return problems, measured


def task_mapping(base: Dataset, bank: Dataset) -> dict[str, int]:
    """Bank instruction (normalised) -> the base's task_index; every bank episode's instruction must be a base task."""
    base_index = {norm(text): index for index, text in base.tasks.items()}
    texts = {norm(e["tasks"][0]) for e in bank.episodes} | {norm(text) for text in bank.tasks.values()}
    unknown = sorted(text for text in texts if text not in base_index)
    refuse_if([f"unknown instruction: {text!r}" for text in unknown], "bank tasks not in the base's tasks.jsonl")
    return {text: base_index[text] for text in texts}


def load_tags(bank: Dataset) -> np.ndarray:
    """The bank's frame tags, checked field by field and returned in frame-index order (row i = bank index i)."""
    tags = np.load(bank.root / "meta" / TAGS, allow_pickle=False)
    names = tags.dtype.names or ()
    missing = [f for f in (INDEX, *TAG_STRINGS, *TAG_VECTORS) if f not in names]
    refuse_if([f"missing field {f}" for f in missing], f"bank {TAGS}")
    problems = []
    if len(tags) != bank.frames():
        problems.append(f"{len(tags)} rows for {bank.frames()} frames")
    if not np.issubdtype(tags.dtype[INDEX], np.integer):
        problems.append(f"{INDEX} is {tags.dtype[INDEX]}, not an integer")
    elif not np.array_equal(np.sort(tags[INDEX]), np.arange(len(tags))):
        problems.append(f"{INDEX} is not a permutation of 0..{len(tags) - 1} (duplicated or missing values)")
    for name, width in TAG_VECTORS.items():
        if tags[name].shape != (len(tags), width) or not np.issubdtype(tags[name].dtype, np.floating):
            problems.append(f"{name} is {tags[name].dtype} {tags[name].shape[1:]}, not float ({width},)")
    for name, width in TAG_STRINGS.items():
        if tags.dtype[name].kind != "U":
            problems.append(f"{name} is {tags.dtype[name]}, not a string")
        elif len(tags) and int(np.char.str_len(tags[name]).max()) > width:
            problems.append(f"{name} has values longer than {width} characters")
    problems += schema_problems(bank, tags)
    for name, allowed in (("family", TRAIN_FAMILIES), ("bin", TRAIN_BINS)):
        if tags.dtype[name].kind == "U" and (stray := sorted(set(tags[name].tolist()) - set(allowed))):
            problems.append(f"{name} values {stray} are not crossview's training {name}s {list(allowed)}")
    refuse_if(problems, f"bank {TAGS}")
    return tags[np.argsort(tags[INDEX], kind="stable")]


def schema_problems(bank: Dataset, tags: np.ndarray) -> list[str]:
    """crossview's meta/frame_tags.json declares each string field's width; it must agree with the npy dtype."""
    path = bank.root / "meta" / TAGS_SCHEMA
    if not path.is_file():
        return [f"the bank has no meta/{TAGS_SCHEMA} (the sidecar's declared string widths)"]
    declared = json.loads(path.read_text()).get("string_widths", {})
    problems = []
    for name in tags.dtype.names or ():
        if tags.dtype[name].kind == "U":
            have = tags.dtype[name].itemsize // UNICODE_BYTES
            if declared.get(name) != have:
                problems.append(f"{name} is U{have} in {TAGS}, {TAGS_SCHEMA} declares {declared.get(name)}")
    return problems


def is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def crossview_parity(bank: Dataset) -> dict:
    """The camera re-render's C4 demo-level receipt, gated field by field (2026-09-29: its `verdict` is ignored,
    because the bank renders from the original HDF5 states while Plus frames are openvla re-simulations).
    Gated: orientation confirmed, actions bit-identical, and enough matched episodes and frames. The state and image
    differences are RECORDED, not bounded (decision of 2026-09-30): state identity is dropped (unreachable by
    design), and the measured 5-7 pixel-error and 1-4 px offsets are recorded as a known difference; a numeric bound
    set then would have been set after seeing the data, so none is set. The file's sha256 is
    recorded too."""
    path = bank.root / "meta" / PARITY_RECEIPT
    if not path.is_file():
        raise BuildRefusal(f"the bank has no meta/{PARITY_RECEIPT} (crossview's C4 parity check)")
    record = json.loads(path.read_text())
    fields = {
        "orientation_confirmed": ("orientation_confirmed",),
        "action_bit_identical": ("action_bit_identical",),
        "state_max_abs": ("state_max_abs",),
        "front_mae_vs_plus": ("front", "mae_vs_plus"),
        "wrist_mae_vs_plus": ("wrist", "mae_vs_plus"),
        "matched_episodes": ("counts", "matched"),
        "tried_episodes": ("counts", "tried"),
        "frames": ("frames",),
    }
    measured, problems = {}, []
    for name, path_in_record in fields.items():
        value, dotted = record, ".".join(path_in_record)
        for key in path_in_record:
            if not isinstance(value, dict) or key not in value:
                problems.append(f"lacks {dotted}")
                value = None
                break
            value = value[key]
        else:
            if value is None:
                problems.append(f"{dotted} is null")
        measured[name] = value
    refuse_if(problems, f"crossview's meta/{PARITY_RECEIPT}")
    flags = ("orientation_confirmed", "action_bit_identical")
    problems = [
        f"{name} is {measured[name]!r}, not a finite number"
        for name in fields
        if name not in flags and not is_finite_number(measured[name])
    ]
    refuse_if(problems, f"crossview's meta/{PARITY_RECEIPT}")
    counted = ("matched_episodes", "tried_episodes", "frames")
    problems = [
        f"{name} is {measured[name]!r}, not an integer" for name in counted if not isinstance(measured[name], int)
    ]
    refuse_if(problems, f"crossview's meta/{PARITY_RECEIPT}")
    if measured["tried_episodes"] <= 0 or measured["matched_episodes"] > measured["tried_episodes"]:
        refuse_if(
            [f"counts matched {measured['matched_episodes']} / tried {measured['tried_episodes']}"],
            f"crossview's meta/{PARITY_RECEIPT}",
        )
    problems = [f"{name} is not true" for name in flags if measured[name] is not True]
    lower = {"matched_episodes": MIN_MATCHED_EPISODES, "frames": MIN_PARITY_FRAMES}
    problems += [
        f"{name} {measured[name]} < minimum {least}" for name, least in lower.items() if measured[name] < least
    ]
    refuse_if(problems, f"crossview's meta/{PARITY_RECEIPT}")
    recorded = {
        "match_ratio": measured["matched_episodes"] / measured["tried_episodes"],
        "stopped": record.get("stopped"),
        "known_difference": {name: measured[name] for name in RECORDED_DIFFERENCES},
        "known_difference_ruling": "decision of 2026-09-30: recorded, no bound set",
    }
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "measured": measured,
        "bounds": {"at_least": lower},
        "recorded_not_gated": recorded,
    }


def check_inputs(base: Dataset, bank: Dataset) -> tuple[dict[str, int], np.ndarray, list[str], dict]:
    task_map = task_mapping(base, bank)
    tags = load_tags(bank)
    crossview = crossview_parity(bank)
    problems, evidence = parity_problems(base, bank, tags, task_map)
    return task_map, tags, problems, {"crossview_c4": crossview, "orientation": evidence}


def place_base(base: Dataset, out: Path) -> dict[str, int]:
    """Base data and video files: an absolute link (a Plus file) is re-made with the same target; the port's pack
    refuses any link outside the Plus mirror. Everything else is copied as a real file."""
    counts = {"links": 0, "copies": 0}
    for top in ("data", "videos"):
        for src in sorted((base.root / top).rglob("*")):
            if src.is_dir() and not src.is_symlink():
                continue
            dst = out / src.relative_to(base.root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            target = os.readlink(src) if src.is_symlink() else None
            if target is not None and os.path.isabs(target):
                os.symlink(target, dst)
                counts["links"] += 1
            else:
                shutil.copyfile(src, dst)
                counts["copies"] += 1
    return counts


def vec_stats(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return {
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "count": [len(x)],
    }


def append_bank(base: Dataset, bank: Dataset, out: Dataset, task_map: dict[str, int]) -> tuple[list, list, list]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = read_table(base.data_path(int(base.episodes[0]["episode_index"]))).schema
    episode, frame = len(base.episodes), base.frames()
    episode_lines, stats_lines, bank_rows = [], [], []
    for e in bank.episodes:
        source = int(e["episode_index"])
        table = read_table(bank.data_path(source))
        length = table.num_rows
        task = task_map[norm(e["tasks"][0])]
        bank_rows.append(np.asarray(table.column("index").to_pylist(), dtype=np.int64))
        new = {
            "episode_index": np.full(length, episode, np.int64),
            "index": np.arange(frame, frame + length, dtype=np.int64),
            "task_index": np.full(length, task, np.int64),
        }
        for name, values in new.items():
            table = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
        target = out.data_path(episode)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.cast(schema), target)
        for key in base.video_keys():
            video = out.video_path(episode, key)
            video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bank.video_path(source, key), video)
        stats = {k: v for k, v in bank.stats[source].items() if k != NOMINAL}
        stats |= {name: vec_stats(values) for name, values in new.items()}
        episode_lines.append({"episode_index": episode, "tasks": [base.tasks[task]], "length": length})
        stats_lines.append({"episode_index": episode, "stats": stats})
        episode, frame = episode + 1, frame + length
    return episode_lines, stats_lines, bank_rows


def mix_tags(base_frames: int, bank_tags: np.ndarray, bank_rows: list[np.ndarray]) -> np.ndarray:
    rows = np.concatenate(bank_rows) if bank_rows else np.zeros(0, np.int64)
    tags = np.zeros(base_frames + len(rows), dtype=MIX_TAG_DTYPE)
    tags["source"][:base_frames] = BASE_TAG
    tags["family"][:base_frames] = BASE_FAMILY
    tags["bin"][:base_frames] = BASE_BIN
    tags["pose"][:base_frames] = UNMOVED_POSE
    tags["source"][base_frames:] = BANK_SOURCE
    tags["family"][base_frames:] = bank_tags["family"][rows]
    tags["bin"][base_frames:] = bank_tags["bin"][rows]
    tags["pose"][base_frames:] = bank_tags["pose"][rows]
    return tags


def base_split(base: Dataset) -> tuple[int, dict[str, int]]:
    """(first clip episode, frames per base source) from the base's rft_receipt.json: episodes [0, orig_episodes) are
    the Plus demos, the rest the base's steer-back clips. Refused when absent or inconsistent with episodes.jsonl."""
    path = base.root / "meta" / RFT_RECEIPT
    if not path.is_file():
        raise BuildRefusal(f"the base has no meta/{RFT_RECEIPT}: the Plus / clips boundary is unknown")
    receipt = json.loads(path.read_text())
    n0, total = receipt.get("orig_episodes"), receipt.get("total_episodes")
    lengths = [int(e["length"]) for e in base.episodes]
    problems = []
    if not isinstance(n0, int) or not isinstance(total, int):
        refuse_if([f"orig_episodes {n0!r} / total_episodes {total!r} are not integers"], f"base {RFT_RECEIPT}")
    if total != len(lengths):
        problems.append(f"total_episodes {total} != {len(lengths)} episodes in episodes.jsonl")
    if not 0 < n0 < len(lengths):
        problems.append(f"orig_episodes {n0} leaves no Plus or no clip episode of {len(lengths)}")
    refuse_if(problems, f"base {RFT_RECEIPT}")
    frames = {PLUS_SOURCE: sum(lengths[:n0]), CLIPS_SOURCE: sum(lengths[n0:])}
    for name, key in ((PLUS_SOURCE, "orig_frames"), (CLIPS_SOURCE, "appended_frames")):
        if key in receipt and receipt[key] != frames[name]:
            problems.append(f"{key} {receipt[key]} != {frames[name]} {name} frames in episodes.jsonl")
    refuse_if(problems, f"base {RFT_RECEIPT}")
    return n0, frames


def share_from_env() -> float:
    raw = os.environ.get(SHARE_ENV, str(BANK_SHARE))
    try:
        return float(raw)
    except ValueError:
        raise BuildRefusal(f"{SHARE_ENV}={raw!r} is not a number") from None


def sampling_weights(frames: dict[str, int], share: float = BANK_SHARE) -> dict[str, float]:
    """Per-frame multipliers (mix.py:148-158 expands w[source] x total / sum(w x len)) for three sources: the clips
    drawn at their own share of the base (c = clips / (plus + clips)), the bank at `share`, Plus the rest
    (1 - c - share).
    w_source / w_plus = (target / frames) / ((1 - c - share) / plus frames)."""
    clips = frames[CLIPS_SOURCE] / (frames[PLUS_SOURCE] + frames[CLIPS_SOURCE])
    plus = 1.0 - clips - share
    if not (math.isfinite(share) and share > 0 and plus > 0):
        raise BuildRefusal(f"bank share {share} is outside (0, {1.0 - clips:.6f}): the clips hold {clips:.6f}")
    per_plus = plus / frames[PLUS_SOURCE]
    return {
        PLUS_SOURCE: 1.0,
        CLIPS_SOURCE: (clips / frames[CLIPS_SOURCE]) / per_plus,
        BANK_SOURCE: (share / frames[BANK_SOURCE]) / per_plus,
    }


def drawn_shares(frames: dict[str, int], weights: dict[str, float]) -> dict[str, float]:
    """Each source's share of drawn frames under the float32 per-frame weights mix.py writes (mean 1)."""
    total = sum(frames.values())
    scale = total / sum(weights[name] * n for name, n in frames.items())
    mass = {name: float(np.float32(weights[name] * scale)) * n for name, n in frames.items()}
    return {name: m / sum(mass.values()) for name, m in mass.items()}


def steer_back(drawn: dict[str, float]) -> dict:
    """Steer-back's drawn share (the clips' share times the base recipe's steer-back fraction of clip frames),
    refused below the base recipe's."""
    share = drawn[CLIPS_SOURCE] * STEER_PER_CLIP_FRAME
    if share < STEER_FLOOR * (1.0 - FLOAT32_REL_TOL):
        raise BuildRefusal(
            f"steer-back drawn share {share:.6f} < the base's {STEER_FLOOR:.6f} (clips {drawn[CLIPS_SOURCE]:.6f})"
        )
    return {"drawn_share": share, "floor": STEER_FLOOR}


def spec_sampling(frames: dict[str, int], share: float) -> dict:
    """What a spec's data.sampling must say for `share`, with the target and drawn shares and the steer-back check."""
    weights = sampling_weights(frames, share)
    drawn = drawn_shares(frames, weights)
    clips = frames[CLIPS_SOURCE] / (frames[PLUS_SOURCE] + frames[CLIPS_SOURCE])
    target = {PLUS_SOURCE: 1.0 - clips - share, CLIPS_SOURCE: clips, BANK_SOURCE: share}
    return {"sampling": weights, "target_share": target, "drawn_share": drawn, "steer_back": steer_back(drawn)}


def action_key(episode: dict, stats: dict) -> tuple:
    action = stats["action"]
    exact = tuple(tuple(np.float32(action[k]).tolist()) for k in ("min", "max"))
    return norm(episode["tasks"][0]), int(episode["length"]), *exact


def plus_links(base: Dataset, bank: Dataset, first_clip: int, n0: int) -> list[dict]:
    """One {"episode_index" (in the mix), "plus_episodes"} per bank episode whose task, length and action min / max
    equal those of Plus episodes (the base's [0, first_clip)), with action means within LINK_MEAN_RTOL. A bank episode
    whose candidates disagree on the mean (two source demos) is refused; one with no candidate is left out. An index
    that links nothing is refused (the port refuses it too)."""
    index: dict[tuple, list[int]] = {}
    for e in base.episodes[:first_clip]:
        episode = int(e["episode_index"])
        index.setdefault(action_key(e, base.stats[episode]), []).append(episode)
    lines, problems = [], []
    for offset, e in enumerate(bank.episodes):
        stats = bank.stats[int(e["episode_index"])]
        hits = index.get(action_key(e, stats), [])
        mean = np.asarray(stats["action"]["mean"], dtype=np.float64)
        close = [h for h in hits if np.allclose(base.stats[h]["action"]["mean"], mean, rtol=LINK_MEAN_RTOL, atol=0)]
        if len(close) != len(hits):
            problems.append(f"bank episode {e['episode_index']}: Plus candidates {hits} disagree on the action mean")
        elif hits:
            lines.append({"episode_index": n0 + offset, "plus_episodes": sorted(hits)})
    refuse_if(problems, f"meta/{PLUS_LINKS}")
    if not lines:
        raise BuildRefusal(f"meta/{PLUS_LINKS} links no bank episode to a Plus episode")
    return lines


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_exclusion(root: Path, bank: Dataset) -> tuple[np.ndarray, dict]:
    """The exclusion input's bank-local frame indices, refused unless: its three files exist, exclusion.npy's sha256
    is the one READY.json and receipt.json both name, receipt.json's is READY's, it was computed on THIS bank's frame
    tags (sha256) and frame count, and it is a strictly increasing int64 list inside [0, bank frames)."""
    missing = [name for name in EXCLUSION_FILES if not (root / name).is_file()]
    if missing:
        raise BuildRefusal(f"no exclusion input at {root} (missing {missing}): this data hook needs it")
    ready = json.loads((root / "READY.json").read_text())
    receipt = json.loads((root / "receipt.json").read_text())
    have = {"exclusion.npy": file_sha256(root / "exclusion.npy"), "receipt.json": file_sha256(root / "receipt.json")}
    problems = []
    if have["exclusion.npy"] != ready.get("exclusion_npy_sha256"):
        problems.append(f"exclusion.npy sha256 {have['exclusion.npy']} != READY's {ready.get('exclusion_npy_sha256')}")
    if have["exclusion.npy"] != receipt.get("files", {}).get("exclusion.npy", {}).get("sha256"):
        problems.append("exclusion.npy sha256 differs from the one receipt.json records")
    if have["receipt.json"] != ready.get("receipt_sha256"):
        problems.append(f"receipt.json sha256 {have['receipt.json']} != READY's {ready.get('receipt_sha256')}")
    tags_sha = file_sha256(bank.root / "meta" / TAGS)
    if receipt.get("inputs", {}).get("frame_tags", {}).get("sha256") != tags_sha:
        problems.append(f"computed on other frame tags than this bank's ({tags_sha})")
    if receipt.get("bank", {}).get("frames") != bank.frames():
        problems.append(
            f"computed on {receipt.get('bank', {}).get('frames')} bank frames, this bank has {bank.frames()}"
        )
    refuse_if(problems, "exclusion input")
    local = np.load(root / "exclusion.npy", allow_pickle=False)
    if local.dtype != np.int64 or local.ndim != 1:
        refuse_if([f"exclusion.npy is {local.dtype} {local.shape}, not 1-D int64"], "exclusion input")
    if len(local) != ready.get("count") or not len(local):
        refuse_if([f"{len(local)} indices, READY says {ready.get('count')}"], "exclusion input")
    if not (np.diff(local) > 0).all():
        refuse_if(["indices are not strictly increasing"], "exclusion input")
    if local[0] < 0 or local[-1] >= bank.frames():
        refuse_if([f"index out of range [0, {bank.frames()}): {int(local[0])}..{int(local[-1])}"], "exclusion input")
    record = {
        "exclusion_npy_sha256": have["exclusion.npy"],
        "receipt_sha256": have["receipt.json"],
        "ready_sha256": file_sha256(root / "READY.json"),
        "count": len(local),
        "bank_frames": bank.frames(),
        "rule": EXCLUSION_RULE,
    }
    return local, record


def global_exclusion(local: np.ndarray, bank_rows: list[np.ndarray], base_frames: int) -> np.ndarray:
    """Bank-local indices -> the built mix's `index` (base frames, then the bank rows in append order), sorted."""
    rows = np.concatenate(bank_rows) if bank_rows else np.zeros(0, np.int64)
    position = np.full(int(rows.max()) + 1 if rows.size else 0, -1, np.int64)
    position[rows] = np.arange(len(rows))
    if local.size and (local[-1] >= len(position) or (position[local] < 0).any()):
        raise BuildRefusal("an excluded bank frame was not appended to the mix")
    mapped = np.sort(base_frames + position[local])
    if not (np.diff(mapped) > 0).all():
        raise BuildRefusal("the mapped exclusion indices are not unique")
    return mapped.astype(EXCLUDE_DTYPE)


def write_exclusion(meta: Path, mapped: np.ndarray) -> dict:
    path = meta / EXCLUDE_FILE
    path.write_bytes(mapped.astype(EXCLUDE_DTYPE).tobytes())
    return {"path": f"meta/{EXCLUDE_FILE}", "sha256": file_sha256(path), "count": len(mapped), "rule": EXCLUSION_RULE}


def held_plus_episodes(split: dict, first_clip: int) -> set[int]:
    held = {int(e) for g in split["plus"] for e in g["episodes"]}
    if any(e >= first_clip for e in held):
        raise BuildRefusal(f"the split holds out Plus episodes outside the base's Plus block [0, {first_clip})")
    return held


def held_clip_frames(base: Dataset, split: dict) -> int:
    """The held-out clip sources' frames in the base: sources x frames per clip x copies, from the clip plan the base's
    rft_receipt.json records ([[dir, frames, copies]]), checked against the split's own frames-per-copy count."""
    plans = json.loads((base.root / "meta" / RFT_RECEIPT).read_text()).get("clips", [])
    if len(plans) != 1:
        raise BuildRefusal(f"the base's {RFT_RECEIPT} records {len(plans)} clip plans, not 1")
    frames, copies = int(plans[0][1]), int(plans[0][2])
    held = split["held_out"]
    if held["clip_sources"] * frames != held["clip_frames_per_copy"]:
        raise BuildRefusal(
            f"split: {held['clip_sources']} clip sources x {frames} frames != {held['clip_frames_per_copy']}"
        )
    return held["clip_frames_per_copy"] * copies


def effective_frames(base: Dataset, bank: Dataset, local: np.ndarray, split: dict) -> tuple[dict, dict]:
    """(training frames per source after the validation hold-out and the exclusion, how each was counted)."""
    first_clip, frames = base_split(base)
    lengths = {int(e["episode_index"]): int(e["length"]) for e in base.episodes}
    held_plus = held_plus_episodes(split, first_clip)
    plus_held = sum(lengths[e] for e in held_plus)
    if plus_held != split["held_out"]["plus_frames"]:
        raise BuildRefusal(f"held-out Plus frames {plus_held} != the split's {split['held_out']['plus_frames']}")
    clips_held = held_clip_frames(base, split)
    n0 = len(base.episodes)
    links = plus_links(base, bank, first_clip, n0)
    held_bank = sorted(
        int(bank.episodes[r["episode_index"] - n0]["episode_index"])
        for r in links
        if held_plus & set(r["plus_episodes"])
    )
    tags = load_tags(bank)
    counts = np.bincount(tags["episode_index"], minlength=max(int(e["episode_index"]) for e in bank.episodes) + 1)
    wrong = [int(e["episode_index"]) for e in bank.episodes if counts[int(e["episode_index"])] != int(e["length"])]
    refuse_if([f"bank episode {e}: frame_tags rows != its length" for e in wrong], f"bank {TAGS}")
    excluded = np.zeros(len(tags), bool)
    excluded[local] = True
    in_held = np.isin(tags["episode_index"], held_bank)
    bank_kept = int((~excluded).sum())
    bank_effective = int((~excluded & ~in_held).sum())
    effective = {
        PLUS_SOURCE: frames[PLUS_SOURCE] - plus_held,
        CLIPS_SOURCE: frames[CLIPS_SOURCE] - clips_held,
        BANK_SOURCE: bank_effective,
    }
    basis = {
        "built": frames | {BANK_SOURCE: bank.frames()},
        "held_out": {
            PLUS_SOURCE: plus_held,
            CLIPS_SOURCE: clips_held,
            BANK_SOURCE: int(in_held.sum()),
            "bank_episodes": len(held_bank),
        },
        "bank_excluded": len(local),
        "bank_kept": bank_kept,
        "bank_kept_and_held_out": bank_kept - bank_effective,
    }
    return effective, basis


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def build(base_root: Path, out_root: Path, bank_path: Path, exclusion_root: Path) -> dict:
    base, bank = Dataset(base_root), Dataset(bank_path)
    if out_root.exists() and any(out_root.iterdir()):
        raise BuildRefusal(f"{out_root} exists and is not empty")
    local, exclusion = load_exclusion(exclusion_root, bank)
    first_clip, base_parts = base_split(base)
    task_map, tags, problems, evidence = check_inputs(base, bank)
    refuse_if(problems, "observation parity")
    links = plus_links(base, bank, first_clip, len(base.episodes))
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    placed = place_base(base, out_root)
    out = Dataset.__new__(Dataset)
    out.root, out.info, out.chunk_size = out_root, base.info, base.chunk_size
    episode_lines, stats_lines, bank_rows = append_bank(base, bank, out, task_map)
    n0, total = len(base.episodes), len(base.episodes) + len(episode_lines)
    base_frames, bank_frames = base.frames(), sum(e["length"] for e in episode_lines)
    frames = base_parts | {BANK_SOURCE: bank_frames}
    info = dict(base.info)
    info |= {
        "total_episodes": total,
        "total_frames": base_frames + bank_frames,
        "total_videos": int(base.info["total_videos"]) + len(base.video_keys()) * len(episode_lines),
        "total_chunks": math.ceil(total / base.chunk_size),
        "splits": {TRAIN_SPLIT: f"0:{total}"},
    }
    meta = out_root / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    shutil.copyfile(base.root / "meta" / "tasks.jsonl", meta / "tasks.jsonl")
    write_jsonl(meta / "episodes.jsonl", base.episodes + episode_lines)
    base_stats = [
        {"episode_index": int(e["episode_index"]), "stats": base.stats[int(e["episode_index"])]} for e in base.episodes
    ]
    write_jsonl(meta / "episodes_stats.jsonl", base_stats + stats_lines)
    np.save(meta / TAGS, mix_tags(base_frames, tags, bank_rows), allow_pickle=False)
    write_jsonl(meta / PLUS_LINKS, links)
    excluded = write_exclusion(meta, global_exclusion(local, bank_rows, base_frames))
    receipt = {
        "sources": [
            {"name": PLUS_SOURCE, "episodes": [[0, first_clip]], "p4c_mix": True, "human_derived": False},
            {
                "name": CLIPS_SOURCE,
                "episodes": [[first_clip, n0]],
                "p4c_mix": True,
                "human_derived": False,
                "min_drawn_share": CLIPS_MIN_DRAWN_SHARE,
            },
            {
                "name": BANK_SOURCE,
                "episodes": [[n0, total]],
                "human_derived": True,
                "reason": BANK_REASON,
                HOLDOUT: PLUS_DEMO_RULE,
                "exclude_frames": excluded,
                # the train role's drawn_share check then refuses a bank drawn under its share, as the weights were
                # computed on --weights-only's effective counts (a hold-out counted differently would shift it)
                "min_drawn_share": share_from_env() * (1.0 - FLOAT32_REL_TOL),
            },
        ],
        "exclusion": exclusion,
        "frames": frames,
        "steer_back_floor": STEER_FLOOR,
        "plus_links": {"linked": len(links), "unlinked": len(episode_lines) - len(links), "mean_rtol": LINK_MEAN_RTOL},
        "parity": evidence,
        "base_files": placed,
    }
    (meta / "mix_receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encoder-stage data hook: the base mix plus the moved-camera bank.")
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument(
        "--weights-only", action="store_true", help=f"print the spec's data.sampling for the share in {SHARE_ENV}"
    )
    parser.add_argument("--split", type=Path, help="with --weights-only: the validation split json")
    parser.add_argument("--parity-only", action="store_true", help="run the checks, print the margins, write nothing")
    args = parser.parse_args(argv)
    inputs = os.environ.get(INPUTS_ENV)
    bank_path = bank_root(inputs)
    exclusion_root = Path(str(inputs)) / EXCLUSION_INPUT
    if args.weights_only:
        if args.split is None:
            parser.error("--weights-only needs --split: the weights are computed on the effective training frames")
        base, bank = Dataset(args.base), Dataset(bank_path)
        local, exclusion = load_exclusion(exclusion_root, bank)
        split = json.loads(args.split.read_text())
        frames, basis = effective_frames(base, bank, local, split)
        record = {"frames": frames, "frames_basis": basis, "exclusion": exclusion}
        record["split_sha256"] = file_sha256(args.split)
        print(json.dumps(spec_sampling(frames, share_from_env()) | record, sort_keys=True))
        return 0
    if args.parity_only:
        *_, problems, evidence = check_inputs(Dataset(args.base), Dataset(bank_path))
        print(json.dumps({"problems": problems, "parity": evidence}, indent=1, sort_keys=True))
        return 1 if problems else 0
    if args.out is None:
        parser.error("the output directory is required unless --weights-only")
    receipt = build(args.base, args.out, bank_path, exclusion_root)
    print(json.dumps({"frames": receipt["frames"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
