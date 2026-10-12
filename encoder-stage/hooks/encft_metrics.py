# ABOUTME: Encoder-stage metrics patch for RLinf d5ec669 (both arms): per-sample flow-matching loss split by the frame's
# ABOUTME: source, camera family and bin from the mix's meta/frame_tags.npy, plus drawn shares. As a script it installs.
"""One file, like encoder_lora.py: `python encft_metrics.py <rlinf-root>` (the port's apply-script form) copies itself
to rlinf/data/datasets/openpi_rlinf/encft_metrics.py and adds three anchor-applied call sites:
  - the SFT train loader (official_sft_data_loader.py, after the loader patches): the loader becomes a subclass of
    openpi's DataLoaderImpl (same constructor and attributes, so the port's validation eval can reuse it) whose
    training instance carries each batch's frame `index` past openpi's transforms (which drop every key the
    Observation does not know) and hands it to the model step through this module;
  - sft_forward (openpi_action_model.py, after `vla_loss = loss.mean()`): the per-element loss is reduced
    per sample and accumulated by source, family and bin;
  - run_training (fsdp_sft_worker.py): once per optimizer step, the sums are all-reduced over ranks and returned as
    metrics (RLinf prefixes them with `train/`).
The frame tags are the builder's meta/frame_tags.npy, named by ENCFT_FRAME_TAGS (the hook's env)."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

TAGS_ENV = "ENCFT_FRAME_TAGS"
INDEX_KEY = "encft_index"
SOURCE_INDEX = "index"
BANK, REPLAY = "bank", "p4c"
IN_RANGE_BIN, BEYOND_BIN = "in_test_range", "training_range_beyond_test"
TAG_FIELDS = ("source", "family", "bin")
PREFIX_SHARE = "data/share"


class MetricsMisaligned(RuntimeError):
    pass


class NonFiniteLoss(RuntimeError):
    pass


@dataclass
class Tracker:
    tags: np.ndarray
    families: tuple[str, ...]
    pending: tuple[np.ndarray, np.ndarray] | None = None
    sums: dict[str, float] | None = None
    counts: dict[str, int] | None = None

    def keys(self) -> list[str]:
        loss = ["loss_shifted", "loss_replay", "loss_shifted/in_range", "loss_shifted/beyond"]
        loss += [f"loss_shifted/{family}" for family in self.families]
        counted = ["n/all", "n/bank", "n/beyond"] + [f"n/{family}" for family in self.families]
        return loss + counted

    def reset(self) -> None:
        self.sums = dict.fromkeys(self.keys(), 0.0)
        self.counts = dict.fromkeys(self.keys(), 0)


_TRACKER: Tracker | None = None


class KeepIndex:
    """Wraps an openpi TransformedDataset's composed transform: the raw item's LeRobot `index` is carried past the
    transforms under INDEX_KEY. A module-level class, so spawn workers can unpickle it."""

    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, item: Mapping) -> dict:
        index = int(np.asarray(item[SOURCE_INDEX]).reshape(-1)[0])
        out = dict(self.transform(item))
        out[INDEX_KEY] = np.int64(index)
        return out


def fingerprint(actions) -> np.ndarray:
    """The first action step of every sample, on the host: enough to tell one batch from another."""
    import torch

    values = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)
    return values.detach().float().reshape(values.shape[0], -1)[:, : values.shape[-1]].cpu().numpy()


class IndexedIteration:
    """Mixed in before openpi's DataLoaderImpl by indexed_class: the class keeps DataLoaderImpl's constructor
    (data_config, data_loader), data_config() and _data_loader, because the port's validation eval reuses the training
    loader object and rebuilds a batch with type(loader)(loader.data_config(), [raw]). Only the instance wrap_loader
    marks (its observation_from_dict set) records each batch's indices and an actions fingerprint for record_loss; any
    other instance of the class is plain openpi and never touches the tracker. An unmarked instance's batches still
    carry INDEX_KEY: openpi's Observation.from_dict reads only its named keys (rlinf-openpi 0.1.1,
    openpi/models/model.py:110-129), so the extra key is ignored there."""

    _encft_observation_from_dict: Callable | None = None
    _data_loader: Iterable[dict]

    def __iter__(self):
        observation_from_dict = self._encft_observation_from_dict
        if observation_from_dict is None:
            yield from super().__iter__()  # ty: ignore[unresolved-attribute]
            return
        tracker = _tracker()
        for batch in self._data_loader:
            index = np.asarray(batch.pop(INDEX_KEY)).astype(np.int64).reshape(-1)
            actions = batch["actions"]
            tracker.pending = (index, fingerprint(actions))
            yield observation_from_dict(batch), actions


_INDEXED_CLASSES: dict[type, type] = {}


def indexed_class(base: type) -> type:
    """The subclass of `base` (openpi's DataLoaderImpl, or the tests' stand-in) with IndexedIteration first."""
    if base not in _INDEXED_CLASSES:
        _INDEXED_CLASSES[base] = type(f"Indexed{base.__name__}", (IndexedIteration, base), {"__module__": __name__})
    return _INDEXED_CLASSES[base]


def load_tags(path: str | None, frames: int) -> np.ndarray:
    if not path:
        raise MetricsMisaligned(f"{TAGS_ENV} must name the mix's meta/frame_tags.npy")
    tags = np.load(path, allow_pickle=False)
    missing = [f for f in TAG_FIELDS if f not in (tags.dtype.names or ())]
    if missing or len(tags) != frames:
        raise MetricsMisaligned(f"{path}: {len(tags)} rows for {frames} frames; missing fields {missing}")
    unknown = set(np.unique(tags["source"]).tolist()) - {BANK, REPLAY}
    if unknown:
        raise MetricsMisaligned(f"{path}: unknown sources {sorted(unknown)}")
    return tags


def install_tracker(tags: np.ndarray) -> Tracker:
    global _TRACKER
    families = tuple(sorted(set(tags["family"][tags["source"] == BANK].tolist())))
    _TRACKER = Tracker(tags=tags, families=families)
    _TRACKER.reset()
    return _TRACKER


def _tracker() -> Tracker:
    if _TRACKER is None:
        raise MetricsMisaligned("encft_metrics.wrap_loader() never ran in this process")
    return _TRACKER


def _openpi_observation_from_dict() -> Callable:
    from openpi.models import model as openpi_model  # ty: ignore[unresolved-import]

    return openpi_model.Observation.from_dict


def wrap_loader(impl, tags_path: str | None = None, observation_from_dict: Callable | None = None):
    """The training loader: an instance of indexed_class(type(impl)) built by impl's own constructor from impl's
    data_config() and _data_loader, marked to record. The dataset object stays the same, indexable by frame index."""
    dataset = impl._data_loader._data_loader.dataset
    if not hasattr(dataset, "_transform"):
        raise MetricsMisaligned(f"the train dataset is a {type(dataset).__name__}, not openpi's TransformedDataset")
    tags = load_tags(tags_path if tags_path is not None else os.environ.get(TAGS_ENV), len(dataset))
    install_tracker(tags)
    dataset._transform = KeepIndex(dataset._transform)
    loader = indexed_class(type(impl))(impl.data_config(), impl._data_loader)
    loader._encft_observation_from_dict = observation_from_dict or _openpi_observation_from_dict()
    return loader


def take_pending(tracker: Tracker, loss, data) -> np.ndarray:
    """The pending batch's frame indices, consumed; refuses unless `data` is the batch they were recorded for."""
    actions = data[1] if isinstance(data, tuple) else data["actions"]
    if tracker.pending is None:
        raise MetricsMisaligned("a model step without a freshly loaded batch (its frame indices are unknown)")
    index, expected = tracker.pending
    tracker.pending = None
    if len(index) != loss.shape[0] or not np.array_equal(fingerprint(actions), expected):
        raise MetricsMisaligned("the loss batch is not the batch whose frame indices were recorded")
    return index


def record_loss(loss, data, training: bool) -> None:
    """Called in sft_forward on the per-element loss (batch first), with the model's mode (sft_forward's
    self.training); the batch must be the loader's latest one. Only a forward with grad on records. With grad off:
      - eval mode (the port's validation eval runs model.eval() then torch.no_grad) records nothing, and refuses if a
        batch's indices are pending: the worker draws a training batch right before its training forward, so a
        pending batch at an eval-mode no-grad forward means a validation batch reached the tracker;
      - train mode (the port's CPU rehearsal: model.train(), then a no-grad forward of a loaded training batch)
        consumes that batch's pending indices and records nothing."""
    import torch

    tracker = _tracker()
    if not torch.is_grad_enabled():
        if tracker.pending is None:
            return
        if not training:
            raise MetricsMisaligned("a no-grad (validation) forward while a batch's frame indices are pending")
        take_pending(tracker, loss, data)
        return
    index = take_pending(tracker, loss, data)
    per_sample = loss.detach().float().reshape(loss.shape[0], -1).mean(dim=1).cpu().numpy().astype(np.float64)
    rows = tracker.tags[index]
    bank = rows["source"] == BANK
    groups = {
        "loss_shifted": bank,
        "loss_replay": rows["source"] == REPLAY,
        "loss_shifted/in_range": bank & (rows["bin"] == IN_RANGE_BIN),
        "loss_shifted/beyond": bank & (rows["bin"] == BEYOND_BIN),
        **{f"loss_shifted/{f}": bank & (rows["family"] == f) for f in tracker.families},
    }
    if tracker.sums is None or tracker.counts is None:
        raise MetricsMisaligned("the tracker was never reset")
    for key, mask in groups.items():
        tracker.sums[key] += float(per_sample[mask].sum())
        tracker.counts[key] += int(mask.sum())
    counted = {
        "n/all": np.ones(len(index), bool),
        "n/bank": bank,
        "n/beyond": bank & (rows["bin"] == BEYOND_BIN),
        **{f"n/{f}": bank & (rows["family"] == f) for f in tracker.families},
    }
    for key, mask in counted.items():
        tracker.counts[key] += int(mask.sum())


def _collective_device(dist) -> str | torch.device:
    """The rank's GPU whenever CUDA is available, else the host (the gloo CPU tests). RLinf creates its default group
    with a bare init_process_group(), so the backend's name varies with the torch build and cannot pick the device
    (the first GPU code smoke passed a CPU tensor to NCCL that way)."""
    import torch

    del dist
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return "cpu"


def _all_reduce(values: np.ndarray) -> np.ndarray:
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return values
    device = _collective_device(dist)
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


def step_metrics() -> dict[str, float]:
    """Once per optimizer step on every rank: global means and shares over all ranks and micro-batches, then reset.
    A key appears only when its group was drawn this step (the same on every rank, after the all-reduce)."""
    tracker = _tracker()
    if tracker.sums is None or tracker.counts is None:
        raise MetricsMisaligned("the tracker was never reset")
    keys = tracker.keys()
    totals = _all_reduce(np.array([tracker.sums[k] for k in keys] + [tracker.counts[k] for k in keys], np.float64))
    sums, counts = dict(zip(keys, totals[: len(keys)], strict=True)), dict(zip(keys, totals[len(keys) :], strict=True))
    tracker.reset()
    out = {k: float(sums[k] / counts[k]) for k in keys if not k.startswith("n/") and counts[k] > 0}
    if counts["n/all"] > 0:
        out[f"{PREFIX_SHARE}_shifted"] = float(counts["n/bank"] / counts["n/all"])
    if counts["n/bank"] > 0:
        out[f"{PREFIX_SHARE}_beyond"] = float(counts["n/beyond"] / counts["n/bank"])
        out |= {f"{PREFIX_SHARE}_{f}": float(counts[f"n/{f}"] / counts["n/bank"]) for f in tracker.families}
    if not all(math.isfinite(v) for v in out.values()):
        raise NonFiniteLoss(f"non-finite training loss in {sorted(k for k, v in out.items() if not math.isfinite(v))}")
    return out


MODULE_PATH = "rlinf/data/datasets/openpi_rlinf/encft_metrics.py"
IMPORT = "from rlinf.data.datasets.openpi_rlinf import encft_metrics\n\n"


@dataclass(frozen=True)
class Anchor:
    path: str
    text: str
    insert: str
    before: bool = False


ANCHORS = (
    Anchor(
        "rlinf/data/datasets/openpi_rlinf/official_sft_data_loader.py",
        "    return data_loader, data_loader.data_config()\n",
        f"    if not eval_dataset:\n        {IMPORT}        data_loader = encft_metrics.wrap_loader(data_loader)\n",
        before=True,
    ),
    Anchor(
        "rlinf/models/embodiment/openpi/openpi_action_model.py",
        "        vla_loss = loss.mean()\n",
        f"        {IMPORT}        encft_metrics.record_loss(loss, data, self.training)\n",
    ),
    Anchor(
        "rlinf/workers/sft/fsdp_sft_worker.py",
        '                    "grad_norm": grad_norm_value,\n                },\n            )\n',
        f"            {IMPORT}            append_to_dict(metrics, encft_metrics.step_metrics())\n",
    ),
)


class ApplyRefused(RuntimeError):
    pass


def apply(root: Path) -> list[str]:
    module_path = root / MODULE_PATH
    source = Path(__file__).read_bytes()
    texts = {a.path: (root / a.path).read_text() for a in ANCHORS}
    patched = [a for a in ANCHORS if (a.insert + a.text if a.before else a.text + a.insert) in texts[a.path]]
    if module_path.exists():
        if module_path.read_bytes() != source:
            raise ApplyRefused(f"{module_path} exists and differs from this hook")
        if len(patched) != len(ANCHORS):
            raise ApplyRefused(f"{module_path} exists but not every call site is in place")
        return []
    if patched:
        raise ApplyRefused("call sites exist without the module: a partial or foreign application")
    for anchor in ANCHORS:
        count = texts[anchor.path].count(anchor.text)
        if count != 1:
            raise ApplyRefused(f"anchor found {count} times (need 1) in {anchor.path}: {anchor.text!r}")
        new = anchor.insert + anchor.text if anchor.before else anchor.text + anchor.insert
        texts[anchor.path] = texts[anchor.path].replace(anchor.text, new)
    for path, text in texts.items():
        (root / path).write_text(text)
    module_path.write_bytes(source)
    return sorted({*texts, MODULE_PATH})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the encoder-stage metrics patch (both arms) to an RLinf tree.")
    parser.add_argument("rlinf_root", nargs="?", default=".", type=Path)
    args = parser.parse_args(argv)
    changed = apply(args.rlinf_root)
    for path in changed:
        print(f"[encft-metrics] patched {path}")
    if not changed:
        print("[encft-metrics] already applied with this exact hook; nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
