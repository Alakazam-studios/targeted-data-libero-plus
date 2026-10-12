# ABOUTME: Encoder-stage RLinf d5ec669 hook, every arm: a freeze set, optional LoRA r16 on the SigLIP MLPs (one FSDP1
# ABOUTME: unit per adapter matrix, or the frozen SigLIP tower as one unit), frozen-weight checks, and a checked save.
"""One file on purpose: the trainer port applies it as an `apply-script` hook (`python encoder_lora.py --arm
fla|control|frozen-encoder <rlinf-root>`), which copies this same file into RLinf as
`rlinf/hybrid_engines/fsdp/encoder_lora.py` and adds six anchor-applied call sites naming the arm.
The copy is byte-identical, so the hook's sha256 covers the runtime module."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
ADAPTER_DTYPE = torch.float32
SO400M_TRAINABLE = 4_713_984
METRICS_INTERVAL_STEPS = 100
TARGET_PATTERN = r"(?:.*\.)?vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc[12]"
TARGET_LEAVES = ("fc1", "fc2")
LM_PATTERN = r"(?:.*\.)?paligemma\.(?:model\.language_model|lm_head)\..+"
LM_EMBEDDING_PATTERN = r"(?:.*\.)?paligemma\.model\.language_model\.embed_tokens\.weight"
LM_HEAD_PATTERN = r"(?:.*\.)?paligemma\.lm_head\.weight"
TOWER_PATTERN = r"(?:.*\.)?vision_tower\.vision_model\..+"
TOWER_CLASS = "SiglipVisionTransformer"
EVERYTHING_PATTERN = r".+"
PEFT_ADAPTER = "default"
LEAF_MARK = "_encft_lora_leaf"
TOWER_UNIT_MARK = "_encft_tower_unit"
TARGET_MARK = "_encft_target"
PROBLEMS_SHOWN = 10
ARM_ENV = "ENCFT_ARM"
LAYOUT_ENV = "ENCFT_LAYOUT"
SAVE_RECORD = "encft_save.json"


class HookMisconfigured(RuntimeError):
    pass


class FrozenChanged(RuntimeError):
    pass


class SaveRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class Arm:
    name: str
    freeze: tuple[str, ...]
    lora: bool
    lm_trains: bool = False
    freezes_exactly_the_tower: bool = False
    # False keeps the base's FSDP1 setting and the model's tie; the frozen tower is then its own unit, since one flat
    # parameter cannot mix requires_grad under it. True needs the untie (untie below).
    use_orig_params: bool = True


ARMS = {
    "fla": Arm("fla", freeze=(EVERYTHING_PATTERN,), lora=True),
    "control": Arm("control", freeze=(LM_PATTERN,), lora=False),
    # The fine-tune-first arm's stage 2: the base's full fine-tune (use_orig_params=False, tie intact), SigLIP frozen
    "frozen-encoder": Arm(
        "frozen-encoder",
        freeze=(TOWER_PATTERN,),
        lora=False,
        lm_trains=True,
        freezes_exactly_the_tower=True,
        use_orig_params=False,
    ),
}


@dataclass(frozen=True)
class TensorRecord:
    dtype: torch.dtype
    shape: tuple[int, ...]
    digest: str


@dataclass
class HookState:
    arm: Arm
    reference: dict[str, TensorRecord]
    targets: dict[str, int]
    base_norms: dict[str, float]
    scale: float = LORA_ALPHA / LORA_RANK
    frozen_shards: dict[str, str] | None = None
    wrap_policy_applied: bool = False
    checked_once: bool = False
    pending_save_metrics: dict[str, float] = field(default_factory=dict)
    untied: list[list[str]] = field(default_factory=list)
    frozen_names: frozenset[str] = frozenset()
    tower: nn.Module | None = None

    def reference_sha(self) -> str:
        lines = "".join(f"{key}:{rec.digest}\n" for key, rec in sorted(self.reference.items()))
        return hashlib.sha256(lines.encode()).hexdigest()

    def is_frozen(self, name: str) -> bool:
        return any(re.fullmatch(pattern, name) for pattern in self.arm.freeze)


_ACTIVE: HookState | None = None


def is_lm(name: str) -> bool:
    return re.fullmatch(LM_PATTERN, name) is not None


def tensor_digest(tensor: torch.Tensor) -> str:
    flat = tensor.detach().reshape(-1).contiguous().cpu()
    return hashlib.sha256(flat.view(torch.uint8).numpy()).hexdigest()


def full_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    state_dict = get_model_state_dict(model)
    tensors = {key: value for key, value in state_dict.items() if isinstance(value, torch.Tensor)}
    if tensors.keys() != state_dict.keys():
        raise HookMisconfigured(f"non-tensor state dict entries: {sorted(state_dict.keys() - tensors.keys())}")
    return tensors


def find_targets(model: nn.Module) -> dict[str, int]:
    targets: dict[str, int] = {}
    for name, module in model.named_modules():
        match = re.fullmatch(TARGET_PATTERN, name)
        if match is None:
            continue
        if not isinstance(module, nn.Linear):
            raise HookMisconfigured(f"{name} is a {type(module).__name__}, not an nn.Linear")
        targets[name] = int(match.group(1))
    layers = sorted(set(targets.values()))
    if not layers or layers != list(range(len(layers))) or len(targets) != len(TARGET_LEAVES) * len(layers):
        raise HookMisconfigured(f"SigLIP MLP targets are not fc1+fc2 of layers 0..n-1: {sorted(targets)}")
    return targets


def inject_lora(model: nn.Module, targets: Mapping[str, int], expected_trainable: int | None = None) -> int:
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora import LoraLayer

    config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=TARGET_PATTERN)
    inject_adapter_in_model(config, model)
    modules = dict(model.named_modules())
    lora_layers = {name for name, module in modules.items() if isinstance(module, LoraLayer)}
    if lora_layers != set(targets):
        raise HookMisconfigured(f"peft wrapped {sorted(lora_layers ^ set(targets))} beyond or short of the targets")
    expected = 0
    for name in targets:
        layer = modules[name]
        if not math.isclose(layer.scaling[PEFT_ADAPTER], LORA_ALPHA / LORA_RANK):
            raise HookMisconfigured(f"{name}: peft scaling {layer.scaling[PEFT_ADAPTER]} != alpha/r")
        setattr(layer, TARGET_MARK, name)
        for leaf in (layer.lora_A[PEFT_ADAPTER], layer.lora_B[PEFT_ADAPTER]):
            leaf.to(ADAPTER_DTYPE)
            leaf.weight.requires_grad_(True)
            setattr(leaf, LEAF_MARK, True)
        base = layer.base_layer
        expected += LORA_RANK * (base.in_features + base.out_features)
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    count = sum(param.numel() for param in trainable.values())
    adapter_infixes = (f".lora_A.{PEFT_ADAPTER}.", f".lora_B.{PEFT_ADAPTER}.")
    stray = [name for name in trainable if not any(infix in name for infix in adapter_infixes)]
    if stray or count != expected or (expected_trainable is not None and count != expected_trainable):
        raise HookMisconfigured(
            f"trainable {count} (expected {expected}, required {expected_trainable}); stray {stray}"
        )
    return count


def layout_problems(reference: Mapping[str, TensorRecord], layout: Mapping[str, Mapping]) -> list[str]:
    missing, extra = sorted(layout.keys() - reference.keys()), sorted(reference.keys() - layout.keys())
    problems = [f"keys missing from the model: {missing}"] if missing else []
    problems += [f"keys not in the layout: {extra}"] if extra else []
    for key in sorted(layout.keys() & reference.keys()):
        have = {"shape": list(reference[key].shape), "dtype": str(reference[key].dtype)}
        want = {"shape": list(layout[key]["shape"]), "dtype": layout[key]["dtype"]}
        if have != want:
            problems.append(f"{key}: loaded {have} != layout {want}")
    return problems


def load_layout(path: str | None) -> dict[str, Mapping]:
    if not path:
        raise HookMisconfigured(f"{LAYOUT_ENV} must name the base checkpoint's layout manifest")
    return json.loads(Path(path).read_text())["tensors"]


def shared_groups(model: nn.Module) -> list[list[str]]:
    """State-dict keys that share storage, found over state_dict(keep_vars=True) by data_ptr (named_parameters() lists
    a tied parameter once, under its first name, and hides the alias)."""
    by_storage: dict[int, list[str]] = {}
    for key, value in model.state_dict(keep_vars=True).items():
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            by_storage.setdefault(value.data_ptr(), []).append(key)
    return [keys for keys in by_storage.values() if len(keys) > 1]


def tower_name(model: nn.Module) -> str:
    """The SigLIP tower's module path, found by module class, not by name pattern, so a freeze pattern is checked
    against something it did not produce."""
    towers = [name for name, module in model.named_modules() if type(module).__name__ == TOWER_CLASS]
    if len(towers) != 1:
        raise HookMisconfigured(f"need exactly one {TOWER_CLASS} in the model, found {towers}")
    return towers[0]


def tower_parameters(model: nn.Module) -> set[str]:
    tower = tower_name(model)
    return {f"{tower}.{name}" for name, _ in model.get_submodule(tower).named_parameters()}


def tower_as_own_unit(model: nn.Module) -> nn.Module:
    """Marks the tower for its own FSDP1 unit (unit_wrap_policy). Its flat parameter then holds only the tower, all
    frozen, provided no tower tensor shares storage with one outside it (a tie would put it in two units)."""
    tower = tower_name(model)
    crossing = [keys for keys in shared_groups(model) if len({key.startswith(f"{tower}.") for key in keys}) > 1]
    if crossing:
        raise HookMisconfigured(f"tensors shared across the SigLIP tower's boundary: {crossing}")
    module = model.get_submodule(tower)
    setattr(module, TOWER_UNIT_MARK, True)
    return module


def refuse_unless_tower_frozen(model: nn.Module) -> None:
    tower = tower_parameters(model)
    frozen = {name for name, param in model.named_parameters() if not param.requires_grad}
    if frozen != tower:
        extra, short = sorted(frozen - tower), sorted(tower - frozen)
        raise HookMisconfigured(
            f"the frozen set is not exactly the SigLIP tower: {len(extra)} frozen outside it {extra[:PROBLEMS_SHOWN]}, "
            f"{len(short)} of its parameters trainable {short[:PROBLEMS_SHOWN]}"
        )


def untie(model: nn.Module) -> list[list[str]]:
    """Every alias after the first gets its own Parameter: a clone with the same dtype and requires_grad. FSDP1 with
    use_orig_params=True restores only one alias of a shared parameter to its unsharded view, so the other module
    (the language model's embedding, tied to lm_head in rlinf-openpi) would read a flat shard (a code smoke,
    2026-09-30: "'weight' must be 2-D"). Values are unchanged, and the saved layout keeps both keys."""
    groups = shared_groups(model)
    for keys in groups:
        for key in keys[1:]:
            owner, _, leaf = key.rpartition(".")
            module = model.get_submodule(owner)
            shared = getattr(module, leaf)
            if not isinstance(shared, nn.Parameter):
                raise HookMisconfigured(f"{key} shares storage with {keys[0]} but is not a parameter")
            setattr(module, leaf, nn.Parameter(shared.detach().clone(), requires_grad=shared.requires_grad))
    refuse_shared(model)
    return groups


def refuse_shared(model: nn.Module) -> None:
    left = shared_groups(model)
    if left:
        raise HookMisconfigured(f"shared parameter left tied: {left}")


def install(
    model: nn.Module, arm: str, expected_trainable: int | None = None, layout: Mapping[str, Mapping] | None = None
) -> HookState:
    from peft.tuners.lora import LoraLayer

    global _ACTIVE
    if arm not in ARMS:
        raise HookMisconfigured(f"unknown arm {arm!r}; one of {sorted(ARMS)}")
    if any(isinstance(module, LoraLayer) for module in model.modules()):
        raise HookMisconfigured("the model already carries LoRA layers (is actor.model.is_lora on?)")
    names = [name for name, _ in model.named_parameters()]
    if not any(re.fullmatch(LM_EMBEDDING_PATTERN, name) for name in names):
        raise HookMisconfigured(f"no parameter matches the language model's embedding {LM_EMBEDDING_PATTERN}")
    untied = untie(model) if ARMS[arm].use_orig_params else []
    reference = {
        key: TensorRecord(value.dtype, tuple(value.shape), tensor_digest(value))
        for key, value in full_state_dict(model).items()
    }
    if layout is not None and (problems := layout_problems(reference, layout)):
        shown = "\n  ".join(problems[:PROBLEMS_SHOWN])
        raise HookMisconfigured(f"the loaded model does not match the layout manifest, {len(problems)}:\n  {shown}")
    targets = find_targets(model) if ARMS[arm].lora else {}
    modules = dict(model.named_modules())
    base_norms = {name: float(modules[name].weight.detach().float().norm()) for name in targets}
    state = HookState(arm=ARMS[arm], reference=reference, targets=targets, base_norms=base_norms, untied=untied)
    for name, param in model.named_parameters():
        param.requires_grad_(not state.is_frozen(name))
    if state.arm.freezes_exactly_the_tower:
        refuse_unless_tower_frozen(model)
    if not state.arm.use_orig_params:
        state.tower = tower_as_own_unit(model)
    if state.arm.lora:
        inject_lora(model, targets, expected_trainable)
    state.frozen_names = frozenset(name for name, param in model.named_parameters() if not param.requires_grad)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable == 0:
        raise HookMisconfigured(f"arm {arm} leaves nothing to train")
    _ACTIVE = state
    print(
        f"[encft] arm {arm}: {trainable} trainable parameters, {len(targets)} SigLIP MLP linears with LoRA; "
        f"untied {untied}; reference sha {state.reference_sha()}",
        flush=True,
    )
    return state


def is_lora_leaf(module: nn.Module) -> bool:
    return bool(getattr(module, LEAF_MARK, False))


def is_own_unit(module: nn.Module) -> bool:
    return is_lora_leaf(module) or bool(getattr(module, TOWER_UNIT_MARK, False))


def unit_wrap_policy(base_policy: Callable | None) -> Callable:
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

    leaf_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=is_own_unit)
    if base_policy is None:
        return leaf_policy
    return functools.partial(_or_policy, policies=[base_policy, leaf_policy])


def flat_units(model: nn.Module) -> list[tuple[list[str], torch.Tensor]]:
    """Each FSDP1 unit's flat parameter with the full names of the tensors flattened into it, in its order."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp._common_utils import clean_tensor_name

    units = []
    for name, module in model.named_modules():
        handle = module._handle if isinstance(module, FSDP) else None
        if handle is not None:
            flat = handle.flat_param
            units.append(([clean_tensor_name(f"{name}.{fqn}" if name else fqn) for fqn in flat._fqns], flat))
    return units


def mixed_units(model: nn.Module, frozen_names: frozenset[str]) -> list[str]:
    """Flat parameters that mix frozen and trainable tensors, or whose requires_grad disagrees with their tensors',
    and frozen tensors in no flat parameter. Frozen means frozen at install, by name."""
    problems, covered = [], set()
    for names, flat in flat_units(model):
        frozen = [name for name in names if name in frozen_names]
        covered.update(frozen)
        if frozen and len(frozen) != len(names):
            trainable = len(names) - len(frozen)
            problems.append(f"{len(frozen)} frozen and {trainable} trainable in one: {names[:PROBLEMS_SHOWN]}")
        elif bool(frozen) == flat.requires_grad:
            problems.append(f"requires_grad {flat.requires_grad} on a flat parameter of {names[:PROBLEMS_SHOWN]}")
    if missing := sorted(frozen_names - covered):
        problems.append(f"{len(missing)} frozen tensors in no flat parameter: {missing[:PROBLEMS_SHOWN]}")
    return problems


def frozen_digests(model: nn.Module) -> dict[str, str]:
    return {name: tensor_digest(param) for name, param in model.named_parameters() if not param.requires_grad}


def record_frozen(model: nn.Module, state: HookState) -> None:
    state.frozen_shards = frozen_digests(model)
    if not state.frozen_shards:
        raise HookMisconfigured("no parameter is frozen after wrapping")
    if not state.arm.lm_trains and not any(is_lm(name) for name in state.frozen_shards):
        raise HookMisconfigured("no frozen parameter is in the language model after wrapping")


def frozen_status(model: nn.Module, state: HookState) -> tuple[bool, bool]:
    if state.frozen_shards is None:
        raise HookMisconfigured("frozen parameters were never recorded after wrapping")
    now = frozen_digests(model)
    if now.keys() != state.frozen_shards.keys():
        return False, False
    changed = {name for name, digest in now.items() if digest != state.frozen_shards[name]}
    return not changed, not any(is_lm(name) for name in changed)


def _collective_device(dist) -> str | torch.device:
    """The rank's GPU whenever CUDA is available, else the host (the gloo CPU tests). RLinf creates its default group
    with a bare init_process_group(), so the backend's name varies with the torch build and cannot pick the device
    (the first GPU code smoke passed a CPU tensor to NCCL that way)."""
    del dist
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return "cpu"


def named_gradients(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Each parameter's gradient by name. A flat parameter that is itself a parameter (FSDP1 use_orig_params=False)
    is split into its tensors' slices of this rank's shard."""
    flats = {id(flat): names for names, flat in flat_units(model)}
    out = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if id(param) not in flats:
            out.append((name, param.grad))
            continue
        for full, info in zip(flats[id(param)], param._shard_param_infos, strict=True):
            if info.in_shard:
                out.append((full, param.grad[info.offset_in_shard : info.offset_in_shard + info.numel_in_shard]))
    return out


def lm_grad_norm(model: nn.Module) -> float:
    import torch.distributed as dist

    squares = [grad.detach().float().square().sum() for name, grad in named_gradients(model) if is_lm(name)]
    total = torch.stack(squares).sum().cpu() if squares else torch.zeros(())
    if dist.is_available() and dist.is_initialized():
        device = _collective_device(dist)
        total = total.to(device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return math.sqrt(float(total))


def _full_weight(leaf: nn.Module) -> torch.Tensor:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(leaf, FSDP):
        with FSDP.summon_full_params(leaf, recurse=False, writeback=False):
            return leaf.module.get_parameter("weight").detach().float().clone()
    return leaf.get_parameter("weight").detach().float().clone()


def lora_metrics(model: nn.Module, state: HookState) -> dict[str, float]:
    delta_sq: dict[int, float] = {}
    base_sq: dict[int, float] = {}
    b_sq = 0.0
    for module in model.modules():
        target = getattr(module, TARGET_MARK, None)
        if target is None:
            continue
        a = _full_weight(module.get_submodule(f"lora_A.{PEFT_ADAPTER}"))
        b = _full_weight(module.get_submodule(f"lora_B.{PEFT_ADAPTER}"))
        block = state.targets[target]
        delta_sq[block] = delta_sq.get(block, 0.0) + float((state.scale * (b @ a)).square().sum())
        base_sq[block] = base_sq.get(block, 0.0) + state.base_norms[target] ** 2
        b_sq += float(b.square().sum())
    metrics = {f"lora/delta_rel_norm/{block:02d}": math.sqrt(delta_sq[block] / base_sq[block]) for block in delta_sq}
    metrics["lora/B_norm"] = math.sqrt(b_sq)
    return metrics


def _all_ranks_true(flags: Sequence[bool]) -> list[bool]:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return list(flags)
    device = _collective_device(dist)
    values = torch.tensor([1 if flag else 0 for flag in flags], dtype=torch.int32, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MIN)
    return [bool(value) for value in values.tolist()]


def step_metrics(
    model: nn.Module, state: HookState, step: int, lm_grad: float, losses: Sequence[float] = ()
) -> dict[str, float]:
    (finite,) = _all_ranks_true([all(math.isfinite(loss) for loss in losses)])
    if not finite:
        raise FrozenChanged(f"non-finite loss at step {step} on some rank (here: {list(losses)})")
    if lm_grad != 0.0 and not state.arm.lm_trains:
        raise FrozenChanged(f"grad/norm_lm is {lm_grad} at step {step}: the language model is receiving gradients")
    metrics = {"grad/norm_lm": lm_grad, **state.pending_save_metrics}
    state.pending_save_metrics = {}
    if state.checked_once and step % METRICS_INTERVAL_STEPS:
        return metrics
    state.checked_once = True
    all_ok, lm_ok = _all_ranks_true(frozen_status(model, state))
    if not (all_ok and lm_ok):
        raise FrozenChanged(f"step {step}: frozen/checksum_ok {all_ok}, frozen/lm_hash_ok {lm_ok}")
    metrics |= (
        {"frozen/checksum_ok": 1.0} if state.arm.lm_trains else {"frozen/checksum_ok": 1.0, "frozen/lm_hash_ok": 1.0}
    )
    if state.arm.lora:
        metrics |= lora_metrics(model, state)
    return metrics


def _adapter_key(target: str, matrix: str) -> str:
    return f"{target}.lora_{matrix}.{PEFT_ADAPTER}.weight"


def _base_key(target: str, leaf: str) -> str:
    return f"{target}.base_layer.{leaf}"


def merge_weight(
    base: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float, dtype: torch.dtype
) -> torch.Tensor:
    return (base.float() + scale * (b.float() @ a.float())).to(dtype)


def merged_state_dict(gathered: Mapping[str, torch.Tensor], state: HookState) -> dict[str, torch.Tensor]:
    renames = {_base_key(t, leaf): f"{t}.{leaf}" for t in state.targets for leaf in ("weight", "bias")}
    adapters = {_adapter_key(t, matrix) for t in state.targets for matrix in ("A", "B")}
    out = {renames.get(key, key): value for key, value in gathered.items() if key not in adapters}
    for target in state.targets:
        key = f"{target}.weight"
        if key in out and _adapter_key(target, "A") in gathered and _adapter_key(target, "B") in gathered:
            a, b = gathered[_adapter_key(target, "A")], gathered[_adapter_key(target, "B")]
            out[key] = merge_weight(out[key], a, b, state.scale, out[key].dtype)
    return out


def _check_merged(
    key: str,
    tensor: torch.Tensor,
    record: TensorRecord,
    target: str,
    gathered: Mapping[str, torch.Tensor],
    scale: float,
) -> list[str]:
    base = gathered.get(_base_key(target, "weight"))
    a, b = gathered.get(_adapter_key(target, "A")), gathered.get(_adapter_key(target, "B"))
    if base is None or a is None or b is None:
        return [f"{key}: base or adapter missing from the gathered state dict"]
    problems = []
    if tensor_digest(base) != record.digest:
        problems.append(f"{key}: the live base weight differs from the start (merged or trained in place?)")
    if not torch.equal(tensor, merge_weight(base, a, b, scale, record.dtype)):
        problems.append(f"{key}: not start + scale*B@A in the saved dtype {record.dtype}")
    if torch.equal(tensor, base):
        problems.append(f"{key}: equals the start weight, so the adapter is not in it")
    return problems


def check_save(
    to_save: Mapping[str, torch.Tensor], gathered: Mapping[str, torch.Tensor], state: HookState
) -> list[str]:
    problems: list[str] = []
    missing, extra = state.reference.keys() - to_save.keys(), to_save.keys() - state.reference.keys()
    if missing or extra:
        problems.append(f"key layout differs from the start: missing {sorted(missing)}, extra {sorted(extra)}")
    weights = {f"{target}.weight": target for target in state.targets}

    def guarded(key: str) -> bool:
        return state.is_frozen(key) or (is_lm(key) and not state.arm.lm_trains)

    for key, record in state.reference.items():
        tensor = to_save.get(key)
        if tensor is None:
            continue
        if tensor.dtype != record.dtype or tuple(tensor.shape) != record.shape:
            problems.append(f"{key}: {tensor.dtype} {tuple(tensor.shape)} != start {record.dtype} {record.shape}")
        elif key in weights:
            problems += _check_merged(key, tensor, record, weights[key], gathered, state.scale)
        elif guarded(key) and tensor_digest(tensor) != record.digest:
            problems.append(f"{key}: a frozen tensor differs from the start")
    return problems


def delta_kept(
    to_save: Mapping[str, torch.Tensor], gathered: Mapping[str, torch.Tensor], state: HookState
) -> dict[str, float]:
    """<saved - base, delta> / |delta|^2 per SigLIP layer and pooled: the share of the adapter's change that survives
    rounding into the saved dtype (the base is already in that dtype, so saved - base is the rounded change)."""
    dots: dict[int, float] = {}
    norms: dict[int, float] = {}
    for target, block in state.targets.items():
        base = gathered[_base_key(target, "weight")].float()
        a, b = gathered[_adapter_key(target, "A")].float(), gathered[_adapter_key(target, "B")].float()
        delta = state.scale * (b @ a)
        dots[block] = dots.get(block, 0.0) + float(((to_save[f"{target}.weight"].float() - base) * delta).sum())
        norms[block] = norms.get(block, 0.0) + float(delta.square().sum())
    pooled = sum(norms.values())
    kept = {f"lora/delta_kept/{block:02d}": dots[block] / norms[block] if norms[block] else 0.0 for block in dots}
    kept["lora/delta_kept"] = sum(dots.values()) / pooled if pooled else 0.0
    return kept


def head_equals_embed(saved: Mapping[str, torch.Tensor]) -> bool | None:
    """Whether the saved lm_head equals the saved embedding (the start dir sets lm_head := embed); None if either
    key is absent."""
    head = [k for k in saved if re.fullmatch(LM_HEAD_PATTERN, k)]
    embed = [k for k in saved if re.fullmatch(LM_EMBEDDING_PATTERN, k)]
    if len(head) != 1 or len(embed) != 1:
        return None
    return bool(torch.equal(saved[head[0]], saved[embed[0]]))


def _active() -> HookState:
    if _ACTIVE is None:
        raise HookMisconfigured("encoder_lora.install() never ran in this process")
    return _ACTIVE


def _rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _from_rank0(problems: list[str], kept: dict[str, float]) -> tuple[list[str], dict[str, float]]:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return problems, kept
    box: list[object] = [(problems, kept)]
    dist.broadcast_object_list(box, src=0)
    received = box[0]
    if not isinstance(received, tuple):
        raise HookMisconfigured(f"rank 0 sent {type(received).__name__}, not (problems, kept)")
    return [str(p) for p in received[0]], {str(k): float(v) for k, v in received[1].items()}


def save_state_dict(
    state_dict: dict[str, torch.Tensor], state: HookState, record_dir: str | None = None
) -> dict[str, torch.Tensor]:
    out, problems, kept = state_dict, [], {}
    if _rank() == 0:
        out = merged_state_dict(state_dict, state)
        problems = check_save(out, state_dict, state)
        if state.targets and not problems:
            kept = delta_kept(out, state_dict, state)
        record: dict[str, object] = {"arm": state.arm.name, "reference_sha": state.reference_sha()}
        record |= {"delta_kept": kept, "refused": problems, "untied": state.untied}
        record |= {"lm_head_equals_embed": head_equals_embed(out)}
        print(f"[encft] save {json.dumps(record, sort_keys=True)}", flush=True)
        if record_dir is not None:
            Path(record_dir).mkdir(parents=True, exist_ok=True)
            (Path(record_dir) / SAVE_RECORD).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    problems, kept = _from_rank0(problems, kept)
    state.pending_save_metrics = kept
    if problems:
        shown = "\n  ".join(problems[:PROBLEMS_SHOWN])
        raise SaveRefused(f"full_weights.pt refused, {len(problems)} problem(s):\n  {shown}")
    return out


def rlinf_install(module: nn.Module, arm: str) -> None:
    layout = load_layout(os.environ.get(LAYOUT_ENV))
    install(module, arm, expected_trainable=SO400M_TRAINABLE if ARMS[arm].lora else None, layout=layout)


def rlinf_wrap_policy(base_policy: Callable | None, use_orig_params: bool) -> Callable:
    state = _active()
    if use_orig_params != state.arm.use_orig_params:
        raise HookMisconfigured(f"the encoder hook needs actor.fsdp_config.use_orig_params={state.arm.use_orig_params}")
    policy = unit_wrap_policy(base_policy)
    if state.tower is not None and not policy(module=state.tower, recurse=False, nonwrapped_numel=0):
        raise HookMisconfigured("the wrap policy does not make the frozen SigLIP tower its own FSDP unit")
    state.wrap_policy_applied = True
    return policy


def rlinf_record_frozen(model: nn.Module) -> None:
    state = _active()
    if not state.wrap_policy_applied:
        raise HookMisconfigured("the FSDP1 wrap policy hook never ran (actor.fsdp_config.strategy is not fsdp?)")
    if not state.arm.use_orig_params and (problems := mixed_units(model, state.frozen_names)):
        shown = "\n  ".join(problems[:PROBLEMS_SHOWN])
        raise HookMisconfigured(f"FSDP flat parameters mix frozen and trainable tensors, {len(problems)}:\n  {shown}")
    record_frozen(model, state)


def rlinf_lm_grad_norm(model: nn.Module) -> float:
    return lm_grad_norm(model)


def rlinf_step_metrics(model: nn.Module, step: int, lm_grad: float, metrics: Mapping[str, list]) -> dict[str, float]:
    return step_metrics(model, _active(), step, lm_grad, [float(loss) for loss in metrics.get("loss", [])])


def rlinf_save_state_dict(state_dict: dict[str, torch.Tensor], record_dir: str) -> dict[str, torch.Tensor]:
    return save_state_dict(state_dict, _active(), record_dir)


MODULE_PATH = "rlinf/hybrid_engines/fsdp/encoder_lora.py"
IMPORT = "from rlinf.hybrid_engines.fsdp import encoder_lora\n\n"


@dataclass(frozen=True)
class Anchor:
    path: str
    after: str
    insert: str


def anchors(arm: str) -> tuple[Anchor, ...]:
    manager, worker = "rlinf/hybrid_engines/fsdp/fsdp_model_manager.py", "rlinf/workers/sft/fsdp_sft_worker.py"
    return (
        Anchor(
            manager,
            "        module = self.model_provider_func()\n",
            f"        {IMPORT}        encoder_lora.rlinf_install(module, arm={arm!r})\n",
        ),
        Anchor(
            manager,
            "        self.model = self._strategy.wrap_model(\n            model=module, device_mesh=self._device_mesh\n"
            "        )\n",
            "        encoder_lora.rlinf_record_frozen(self.model)\n",
        ),
        Anchor(
            "rlinf/hybrid_engines/fsdp/strategy/fsdp.py",
            "            is_lora=self.cfg.model.is_lora,\n"
            "            model_type=self.cfg.model.model_type,\n        )\n",
            f"        {IMPORT}        auto_wrap_policy = encoder_lora.rlinf_wrap_policy(\n"
            "            auto_wrap_policy, use_orig_params=self.cfg.fsdp_config.use_orig_params\n        )\n",
        ),
        Anchor(
            "rlinf/hybrid_engines/fsdp/strategy/base.py",
            "            model_state_dict = get_model_state_dict(model=model, options=opts)\n",
            f"            {IMPORT}"
            "            model_state_dict = encoder_lora.rlinf_save_state_dict(model_state_dict, sd_save_path)\n",
        ),
        Anchor(
            worker,
            "            # in one step do the optimizer step\n",
            f"            {IMPORT}            encft_lm_grad = encoder_lora.rlinf_lm_grad_norm(self.model)\n",
        ),
        Anchor(
            worker,
            '                    "grad_norm": grad_norm_value,\n                },\n            )\n',
            "            append_to_dict(\n                metrics,\n"
            "                encoder_lora.rlinf_step_metrics(self.model, self.global_step, encft_lm_grad, metrics),\n"
            "            )\n",
        ),
    )


def apply(root: Path, arm: str) -> list[str]:
    if arm not in ARMS:
        raise HookMisconfigured(f"unknown arm {arm!r}; one of {sorted(ARMS)}")
    module_path = root / MODULE_PATH
    source = Path(__file__).read_bytes()
    edits = anchors(arm)
    texts = {anchor.path: (root / anchor.path).read_text() for anchor in edits}
    if module_path.exists():
        if module_path.read_bytes() != source:
            raise HookMisconfigured(f"{module_path} exists and differs from this hook")
        if any(texts[a.path].count(a.after + a.insert) != 1 for a in edits):
            raise HookMisconfigured(f"{module_path} exists but the call sites for arm {arm} are not all in place")
        return []
    if any("encoder_lora" in text for text in texts.values()):
        raise HookMisconfigured("call sites exist without the module: a partial or foreign application")
    for anchor in edits:
        count = texts[anchor.path].count(anchor.after)
        if count != 1:
            raise HookMisconfigured(f"anchor found {count} times (need 1) in {anchor.path}: {anchor.after!r}")
        texts[anchor.path] = texts[anchor.path].replace(anchor.after, anchor.after + anchor.insert)
    for path, text in texts.items():
        (root / path).write_text(text)
    module_path.write_bytes(source)
    if module_path.read_bytes() != source:
        raise HookMisconfigured(f"{module_path} is not a byte copy of the hook")
    return sorted({*texts, MODULE_PATH})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the encoder hook (freeze set + optional SigLIP LoRA) to RLinf.")
    parser.add_argument("rlinf_root", nargs="?", default=".", type=Path)
    parser.add_argument("--arm", choices=sorted(ARMS), default=os.environ.get(ARM_ENV))
    args = parser.parse_args(argv)
    if args.arm is None:
        parser.error(f"name the arm with --arm or {ARM_ENV}")
    changed = apply(args.rlinf_root, args.arm)
    for path in changed:
        print(f"[encft] arm {args.arm}: patched {path}")
    if not changed:
        print(f"[encft] arm {args.arm}: already applied with this exact hook; nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
