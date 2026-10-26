"""Stage S8 Phase 2 three-branch continuation runner.

Loads the full-state dump emitted by Stage S8 Phase 1 (base retrain) for a
single ``(dataset, seed, anchor_class)`` task, reconstructs the main
optimizer state (including private m/v/step -- the key S8 upgrade over
S7-B's fresh-Adam approximation), then runs H_max continuation main
batches through three branches -- F, T, DH -- simultaneously in one
process.  Every step writes ``continuation_steps.csv`` /
``continuation_validation.csv`` / ``continuation_geometry.csv``; the per-
branch best state (by macro AUC, strict >) is recorded in
``continuation_best_{F,T,DH}.pth`` and ``continuation_best.csv``.

Per Algorithm Specification ``stage_S8_design.md`` §4.3.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M3OE_ROOT = os.path.join(ROOT, "m3oe")
EXAMPLES_ROOT = os.path.join(M3OE_ROOT, "examples")
for _path in (HERE, M3OE_ROOT, EXAMPLES_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from MDMTRec.utils.data import TorchDataset  # noqa: E402

from stage_E1_instrumentation import (  # noqa: E402
    batch_fingerprint,
    cell_index,
    masked_bce_terms,
    preserve_rng_state,
    rng_state_digest,
    safe_auc,
    safe_logloss,
    write_json,
    write_rows_csv,
)
from stage_S1_data import load_stage_s1_bundle  # noqa: E402
from stage_S1_metrics import LOGLOSS_EPS  # noqa: E402
from stage_S1_models import build_model, initial_state_hash  # noqa: E402
from stage_S1_run import setup_seed  # noqa: E402
from stage_S3_gradient_access import (  # noqa: E402
    GROUP_ARCH,
    GROUP_PRIVATE,
    GROUP_SHARED,
    assert_expected_parameter_counts,
    loss_components,
    partition_parameters,
)
from stage_S3_protocol import (  # noqa: E402
    MODEL_SOURCE_KEY,
    MODEL_SOURCE_SHA256,
    target_for,
)
from stage_S4_gradient_controls import (  # noqa: E402
    adam_deltas,
    assert_vector_match,
    cosine,
    gradients_one_forward,
    vector_norm,
)
from stage_S4_protocol import SEEDS as S4_SEEDS  # noqa: E402
from stage_S7_probe import (  # noqa: E402
    CELL_NUM,
    DOMAIN_NUM,
    MODULES,
    TASK_NUM,
    module_slices,
    shared_module_of,
    validation_cell_gradients,
)

import stage_S8_protocol as proto  # noqa: E402


# Step-row schema (per design §6.2).  Validation / geometry columns only
# populated at the val-gradient sample steps; ``NaN`` otherwise (collector
# filters them).
STEP_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "seed", "anchor_class", "anchor_epoch", "branch",
    "epoch", "step", "batch_fingerprint", "domain_counts",
    "L_full", "L_target", "L_non",
    "g_full_norm", "g_target_norm", "cosine_gf_gt",
    "m_norm_persistent", "v_norm_persistent",
    "delta_full_norm", "delta_target_norm", "delta_applied_norm", "delta_cosine",
    "private_delta_norm", "applied_formula_residual", "private_full_residual",
    "theta_distance_to_F_shared",
    "eff_align_full", "eff_align_target", "eff_dot_full", "eff_dot_target",
    "rng_digest", "validity_L_full_diff", "validity_delta_norm_rel_diff",
)

GEOMETRY_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "seed", "anchor_class", "anchor_epoch", "branch", "step",
    "source_domain", "source_task", "target_domain", "target_task",
    "pair_class", "shared_module",
    "gradient_norm", "pairwise_cosine", "sample_alignment",
    "validation_dot_product", "weighted_source_target_contribution",
    "validation_gradient_norm", "source_weight",
)

VALIDATION_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "seed", "anchor_class", "anchor_epoch", "branch", "step",
    "kind", "scope", "domain", "task", "cell",
    "val_bce_raw", "auc", "logloss",
)

BEST_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "seed", "anchor_class", "anchor_epoch", "branch",
    "horizon", "horizon_steps", "best_step",
    "best_macro_auc", "best_target_auc", "best_macro_logloss", "best_worst_cell_auc",
    "best_incl_anchor_macro_auc",
    "anchor_macro_auc", "final_macro_auc",
    "f_original_best_macro_auc", "f_original_selected_epoch",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_counts(values: Sequence[int]) -> str:
    return json.dumps([int(v) for v in values], separators=(",", ":"))


def _to_fp64_flat(tensors: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    pieces: List[torch.Tensor] = []
    for value in tensors:
        if value is None:
            continue
        pieces.append(value.detach().reshape(-1).to(dtype=torch.float64, device="cpu"))
    if not pieces:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(pieces)


def _norm_fp64(a: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(a).detach().cpu())


def _cosine_fp64(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    na, nb = _norm_fp64(a), _norm_fp64(b)
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return max(-1.0, min(1.0, float(torch.dot(a, b).detach().cpu()) / (na * nb)))


def _dot_fp64(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float(torch.dot(a, b).detach().cpu())


def _set_grad(parameters: Mapping[str, torch.nn.Parameter],
              names: Sequence[str], values: Sequence[Optional[torch.Tensor]]) -> None:
    if len(names) != len(values):
        raise AssertionError("set_grad length mismatch")
    for name, value in zip(names, values):
        parameters[name].grad = None if value is None else value.detach().clone()


def _name_of(parameters: Mapping[str, torch.nn.Parameter], target: torch.nn.Parameter) -> str:
    for name, param in parameters.items():
        if param is target:
            return name
    raise KeyError("parameter not found in dict")


def _target_local_domain(dataset: str) -> int:
    return int(target_for(str(dataset).lower(), proto.target_setting(dataset))["local_domain"])


def _branch_state_bit_equal(model_a: torch.nn.Module, opt_a: torch.optim.Adam,
                             arch_opt_a: Optional[torch.optim.Adam],
                             model_b: torch.nn.Module, opt_b: torch.optim.Adam,
                             arch_opt_b: Optional[torch.optim.Adam],
                             shared_names: Sequence[str],
                             private_names: Sequence[str]) -> Tuple[bool, List[Dict[str, object]]]:
    """Bit-equal pre-step check across two (model, main_opt [, arch_opt])
    tuples -- returns ``(all_equal, per_param_diff_log)``.

    Because ``branch_state_full`` creates fresh Parameter objects (the
    source ``opt_a.state`` is keyed by old params; the cloned ``opt_b``
    is keyed by new params), we cannot compare ``state`` keys by
    identity.  Instead we walk ``opt_a.param_groups`` and ``opt_b.param_groups``
    in lock-step and compare the corresponding ``state`` tensor data.
    """
    rows: List[Dict[str, object]] = []
    pa = dict(model_a.named_parameters())
    pb = dict(model_b.named_parameters())
    if set(pa.keys()) != set(pb.keys()):
        rows.append({"scope": "model_param_names", "equal": False})
        return False, rows
    rows.append({"scope": "model_param_names", "equal": True,
                  "param_count": len(pa)})
    all_equal = True
    for name in pa:
        if not torch.equal(pa[name].detach().cpu(), pb[name].detach().cpu()):
            diff = float(torch.max(torch.abs(pa[name].detach().cpu() - pb[name].detach().cpu())))
            rows.append({"scope": "model_param", "name": name, "equal": False,
                          "max_abs_diff": diff})
            all_equal = False
    # Main optimizer state -- compare group-by-group in lock-step.
    if len(opt_a.param_groups) != len(opt_b.param_groups):
        rows.append({"scope": "main_opt_group_count", "a": len(opt_a.param_groups),
                      "b": len(opt_b.param_groups), "equal": False})
        return False, rows
    total_a_state = 0
    total_b_state = 0
    for ga, gb in zip(opt_a.param_groups, opt_b.param_groups):
        if len(ga["params"]) != len(gb["params"]):
            rows.append({"scope": "main_opt_group_params_count_mismatch",
                          "a": len(ga["params"]), "b": len(gb["params"]),
                          "equal": False})
            all_equal = False
            continue
        for pa_, pb_ in zip(ga["params"], gb["params"]):
            a_state = opt_a.state.get(pa_, {})
            b_state = opt_b.state.get(pb_, {})
            total_a_state += 1 if a_state else 0
            total_b_state += 1 if b_state else 0
            if set(a_state.keys()) != set(b_state.keys()):
                rows.append({"scope": "main_opt_state_keys",
                              "param_name_a": _name_of(pa, pa_),
                              "param_name_b": _name_of(pb, pb_),
                              "equal": False})
                all_equal = False
                continue
            for subkey, av in a_state.items():
                bv = b_state.get(subkey)
                if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor):
                    eq = bool(torch.equal(av.detach().cpu(), bv.detach().cpu()))
                    if not eq:
                        diff = float(torch.max(torch.abs(av.detach().cpu() - bv.detach().cpu())))
                        rows.append({"scope": "main_opt_state_tensor",
                                      "param_name": _name_of(pa, pa_),
                                      "key": subkey, "equal": False,
                                      "max_abs_diff": diff})
                        all_equal = False
                else:
                    if av != bv:
                        rows.append({"scope": "main_opt_state_scalar",
                                      "param_name": _name_of(pa, pa_),
                                      "key": subkey, "equal": False})
                        all_equal = False
    rows.append({"scope": "main_opt_state_param_count",
                  "a": total_a_state, "b": total_b_state,
                  "equal": total_a_state == total_b_state})
    # Arch optimizer (optional) -- same lock-step comparison.
    if arch_opt_a is not None and arch_opt_b is not None:
        if len(arch_opt_a.param_groups) != len(arch_opt_b.param_groups):
            rows.append({"scope": "arch_opt_group_count",
                          "a": len(arch_opt_a.param_groups),
                          "b": len(arch_opt_b.param_groups), "equal": False})
            return False, rows
        for ga, gb in zip(arch_opt_a.param_groups, arch_opt_b.param_groups):
            if len(ga["params"]) != len(gb["params"]):
                rows.append({"scope": "arch_opt_group_params_count_mismatch",
                              "a": len(ga["params"]), "b": len(gb["params"]),
                              "equal": False})
                all_equal = False
                continue
            for pa_, pb_ in zip(ga["params"], gb["params"]):
                a_state = arch_opt_a.state.get(pa_, {})
                b_state = arch_opt_b.state.get(pb_, {})
                if set(a_state.keys()) != set(b_state.keys()):
                    rows.append({"scope": "arch_opt_state_keys",
                                  "param_name_a": _name_of(pa, pa_),
                                  "param_name_b": _name_of(pb, pb_),
                                  "equal": False})
                    all_equal = False
                    continue
                for subkey, av in a_state.items():
                    bv = b_state.get(subkey)
                    if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor):
                        eq = bool(torch.equal(av.detach().cpu(), bv.detach().cpu()))
                        if not eq:
                            diff = float(torch.max(torch.abs(av.detach().cpu() - bv.detach().cpu())))
                            rows.append({"scope": "arch_opt_state_tensor",
                                          "param_name": _name_of(pa, pa_),
                                          "key": subkey, "equal": False,
                                          "max_abs_diff": diff})
                            all_equal = False
                    else:
                        if av != bv:
                            rows.append({"scope": "arch_opt_state_scalar",
                                          "param_name": _name_of(pa, pa_),
                                          "key": subkey, "equal": False})
                            all_equal = False
    elif arch_opt_a is not None or arch_opt_b is not None:
        rows.append({"scope": "arch_opt_state_presence", "equal": False})
        all_equal = False
    rows.append({"scope": "all_equal", "equal": bool(all_equal)})
    return all_equal, rows


# ---------------------------------------------------------------------------
# Source-run reading
# ---------------------------------------------------------------------------

def _read_s8_base_manifest(base_run_dir: str) -> Dict[str, object]:
    with open(os.path.join(base_run_dir, "run_manifest.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _read_s8_base_batch_trace(base_run_dir: str) -> List[Dict[str, str]]:
    """Return all batch-trace rows from the S8 base run (kind in {train, architecture})."""
    path = os.path.join(base_run_dir, "batch_trace.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_s8_base_metrics_epoch(base_run_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(base_run_dir, "metrics_epoch.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_s8_base_gradient_step(base_run_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(base_run_dir, "gradient_step.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("kind") == "main"]


def _epochs_metrics_lookup(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    """Map epoch -> one representative metrics_epoch.csv row.

    The S1/S6 ``metrics_epoch.csv`` schema is PER-CELL: every row carries
    ``domain``/``task`` and repeats the epoch-level aggregates
    (``validation_macro_auc`` etc.) on each of the 6 cell rows.  We group by
    epoch and verify the macro AUC is consistent across the cell rows.
    """
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        try:
            epoch = int(row.get("epoch", "-1"))
        except (TypeError, ValueError):
            continue
        if "validation_macro_auc" not in row:
            continue
        if epoch in out:
            prev = _safe_float(out[epoch].get("validation_macro_auc", ""))
            cur = _safe_float(row.get("validation_macro_auc", ""))
            if prev is not None and cur is not None and abs(prev - cur) > 0.0:
                raise ValueError(
                    "metrics_epoch.csv macro AUC inconsistent within epoch {}".format(epoch))
            continue
        out[epoch] = row
    return out


def _main_rows_for(base_rows: List[Dict[str, str]], start_epoch: int, h: int) -> List[Dict[str, str]]:
    """Filter S8 base main rows to ``epoch >= start_epoch`` and the first ``h`` rows
    ordered by (epoch, step).  Returns the slice, raising if fewer than ``h`` rows exist."""
    candidates: List[Tuple[int, int, Dict[str, str]]] = []
    for row in base_rows:
        try:
            epoch = int(row.get("epoch", "-1"))
            step = int(row.get("step", "-1"))
        except (TypeError, ValueError):
            continue
        if epoch < start_epoch:
            continue
        candidates.append((epoch, step, row))
    candidates.sort(key=lambda t: (t[0], t[1]))
    selected = [t[2] for t in candidates[:int(h)]]
    if len(selected) < int(h):
        raise ValueError(
            "S8 base main rows insufficient: epoch>={} wanted {} got {}".format(
                start_epoch, h, len(selected)))
    return selected


def _arch_rows_for(base_rows: List[Dict[str, str]], epoch: int) -> List[Dict[str, str]]:
    candidates: List[Tuple[int, Dict[str, str]]] = []
    for row in base_rows:
        if str(row.get("kind", "")) != "architecture":
            continue
        try:
            ep = int(row.get("epoch", "-1"))
        except (TypeError, ValueError):
            continue
        if ep != int(epoch):
            continue
        candidates.append((ep, row))
    return [t[1] for t in candidates]


# ---------------------------------------------------------------------------
# Optimizer state reconstruction from full-state dump
# ---------------------------------------------------------------------------

def _reconstruct_main_adam(parameters: Mapping[str, torch.nn.Parameter],
                            shared_names: Sequence[str], private_names: Sequence[str],
                            main_dump_state: Mapping[str, object],
                            main_lr: float, main_betas: Tuple[float, float],
                            main_wd: float) -> torch.optim.Adam:
    """Build a fresh main Adam (main params in the same order as the
    source ``S1Runner`` optimizer: ``model.named_parameters()`` excluding
    architecture params) and ``load_state_dict`` from the full-state
    dump.

    The dump's optimizer state was saved with the source's parameter
    ordering.  Re-ordering would misalign ``state[i]`` with
    ``param_groups[0]['params'][i]`` and break the Adam step.  We use
    ``shared_names`` and ``private_names`` as the authoritative name
    sets but preserve the source's ``model.named_parameters()`` order
    for the actual param list.
    """
    allowed_names = set(shared_names) | set(private_names)
    main_params: List[torch.nn.Parameter] = []
    for name, p in parameters.items():
        if name in allowed_names:
            main_params.append(p)
    optimizer = torch.optim.Adam(main_params, lr=float(main_lr),
                                  betas=tuple(main_betas),
                                  weight_decay=float(main_wd))
    optimizer.load_state_dict(main_dump_state)
    return optimizer


def _reconstruct_arch_adam(parameters: Mapping[str, torch.nn.Parameter],
                            arch_names: Sequence[str],
                            arch_dump_state: Mapping[str, object],
                            arch_lr: float, arch_betas: Tuple[float, float],
                            arch_wd: float) -> torch.optim.Adam:
    arch_params = [parameters[name] for name in arch_names]
    optimizer = torch.optim.Adam(arch_params, lr=float(arch_lr),
                                  betas=tuple(arch_betas),
                                  weight_decay=float(arch_wd))
    optimizer.load_state_dict(arch_dump_state)
    return optimizer


def reconstruct_anchor_full(model: torch.nn.Module, base_run_dir: str,
                            anchor_epoch: int, device: torch.device,
                            shared_names: Sequence[str],
                            private_names: Sequence[str],
                            arch_names: Sequence[str]
                            ) -> Dict[str, object]:
    """Load the full-state dump at ``anchor_epoch``, attach the model
    parameters, and rebuild the main and arch optimizers with their saved
    state.  Returns a dict with the optimizers and audit info."""
    dump_path = os.path.join(base_run_dir, proto.full_state_filename(int(anchor_epoch)))
    if not os.path.isfile(dump_path):
        raise FileNotFoundError("S8 full-state dump missing: " + dump_path)
    payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    model_state = payload["model"]
    if not isinstance(model_state, dict):
        raise ValueError("S8 dump model state must be a dict")
    model.load_state_dict(model_state, strict=True)
    main_state = payload["main_optimizer"]
    arch_state = payload["arch_optimizer"]
    if not isinstance(main_state, dict):
        raise ValueError("S8 dump main_optimizer must be a dict")
    if not isinstance(arch_state, dict):
        raise ValueError("S8 dump arch_optimizer must be a dict")
    parameters = dict(model.named_parameters())
    main_opt = _reconstruct_main_adam(parameters, shared_names, private_names,
                                       main_state, 0.01, (0.9, 0.999), 1e-5)
    arch_opt = _reconstruct_arch_adam(parameters, arch_names, arch_state,
                                       0.001, (0.9, 0.999), 1e-5)
    # Verify shared Adam step == expected (bpe * (anchor + 1)).
    shared_steps: Set[int] = set()
    for name in shared_names:
        state = main_opt.state.get(parameters[name], {})
        if "step" in state:
            value = state["step"]
            shared_steps.add(int(value.detach().cpu().item()) if isinstance(value, torch.Tensor)
                              else int(value))
    if not shared_steps:
        raise AssertionError("S8 anchor reconstruction: no shared Adam step loaded")
    if len(shared_steps) != 1:
        raise AssertionError("S8 anchor reconstruction: shared Adam step non-uniform: {}"
                              .format(sorted(shared_steps)))
    shared_step = int(next(iter(shared_steps)))
    # Private state coverage.
    private_state_count = int(sum(1 for name in private_names
                                    if main_opt.state.get(parameters[name])))
    # Arch step.
    arch_steps: Set[int] = set()
    for name in arch_names:
        state = arch_opt.state.get(parameters[name], {})
        if "step" in state:
            value = state["step"]
            arch_steps.add(int(value.detach().cpu().item()) if isinstance(value, torch.Tensor)
                            else int(value))
    if len(arch_steps) != 1:
        raise AssertionError("S8 anchor reconstruction: arch Adam step non-uniform: {}"
                              .format(sorted(arch_steps)))
    arch_step = int(next(iter(arch_steps)))
    audit = {
        "anchor_epoch": int(anchor_epoch),
        "full_state_dump_path": os.path.abspath(dump_path),
        "full_state_dump_sha256": _hash_file(dump_path),
        "shared_step_observed": int(shared_step),
        # Populated by run_continuation (needs the dataset); placeholder 0.
        "shared_step_expected": 0,
        "private_state_count": int(private_state_count),
        "private_state_coverage": (
            int(private_state_count) == len(list(private_names))),
        "arch_step_observed": int(arch_step),
        "optimizer_steps_in_payload": int(payload.get("optimizer_steps", -1)),
        "arch_optimizer_steps_in_payload": int(payload.get("architecture_optimizer_steps", -1)),
        "rng_digest_in_payload": str(payload.get("rng_digest", "")),
        "model_param_count": int(len(parameters)),
        "shared_param_count": int(len(list(shared_names))),
        "private_param_count": int(len(list(private_names))),
        "arch_param_count": int(len(list(arch_names))),
    }
    # ``shared_step_expected`` stays a placeholder here because
    # ``proto.expected_shared_step`` needs the dataset; run_continuation
    # overwrites it before the V5 comparison.
    return {"model": model, "optimizer": main_opt, "arch_optimizer": arch_opt,
            "audit": audit, "payload": payload}


def branch_state_full(model: torch.nn.Module, optimizer: torch.optim.Adam,
                       arch_optimizer: Optional[torch.optim.Adam],
                       shared_names: Sequence[str],
                       private_names: Sequence[str]) -> Tuple[torch.nn.Module,
                                                                torch.optim.Adam,
                                                                Optional[torch.optim.Adam]]:
    """Deep-copy a (model, main_opt, arch_opt) tuple while preserving
    the new optimizer's ``param_groups`` and ``state`` keyed by the new
    model's parameters.

    The naive ``copy.deepcopy(optimizer)`` creates new Parameter objects
    (because torch.optim.Optimizer has no ``__deepcopy__`` hook in this
    PyTorch version) and leaves ``optimizer.state`` keyed by the OLD
    parameters, which is broken.  We work around this by deep-copying
    only the model (which produces new parameters), then rebuilding the
    two optimizers in the SAME param order as the source (per
    ``_reconstruct_main_adam`` ordering rule: ``model.named_parameters()``
    filtering by the allowed name sets), and re-keying ``state`` to the
    new params.

    Hard parameter equality with the source is enforced (V3).
    """
    parameters = dict(model.named_parameters())
    new_model = copy.deepcopy(model)
    new_params = dict(new_model.named_parameters())
    allowed_names = set(shared_names) | set(private_names)
    # Build a new main optimizer in the SOURCE ordering (model.named_parameters()
    # filtering by the allowed name sets) so state keys map correctly.
    src_main_param_order = [name for name in parameters.keys() if name in allowed_names]
    new_main_params: List[torch.nn.Parameter] = []
    for name in src_main_param_order:
        new_main_params.append(new_params[name])
    new_opt = torch.optim.Adam(new_main_params, lr=float(optimizer.defaults.get("lr", 0.01)),
                                betas=tuple(optimizer.defaults.get("betas", (0.9, 0.999))),
                                weight_decay=float(optimizer.defaults.get("weight_decay", 1e-5)))
    # Re-key optimizer.state from old params to new params (matching by name
    # via the source ordering).
    new_opt.state.clear()
    for name in src_main_param_order:
        old_param = parameters[name]
        new_param = new_params[name]
        if old_param in optimizer.state:
            old_state = optimizer.state[old_param]
            new_opt.state[new_param] = {
                key: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
                for key, value in old_state.items()}
    new_arch_opt = None
    if arch_optimizer is not None:
        src_arch_param_order = [name for name in parameters.keys() if name not in allowed_names]
        new_arch_params: List[torch.nn.Parameter] = []
        for name in src_arch_param_order:
            new_arch_params.append(new_params[name])
        new_arch_opt = torch.optim.Adam(new_arch_params,
                                          lr=float(arch_optimizer.defaults.get("lr", 0.001)),
                                          betas=tuple(arch_optimizer.defaults.get("betas", (0.9, 0.999))),
                                          weight_decay=float(arch_optimizer.defaults.get("weight_decay", 1e-5)))
        new_arch_opt.state.clear()
        for name in src_arch_param_order:
            old_param = parameters[name]
            new_param = new_params[name]
            if old_param in arch_optimizer.state:
                old_state = arch_optimizer.state[old_param]
                new_arch_opt.state[new_param] = {
                    key: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
                    for key, value in old_state.items()}
    # Hard parameter equality with the source (V3).
    for name, old_param in parameters.items():
        if name not in new_params:
            raise AssertionError("deepcopy lost parameter: " + name)
        if not torch.equal(old_param.detach().cpu(), new_params[name].detach().cpu()):
            raise AssertionError("parameter inequality after deepcopy: " + name)
    return new_model, new_opt, new_arch_opt


# ---------------------------------------------------------------------------
# Validation pass (reused across branches at each step + final)
# ---------------------------------------------------------------------------

def _full_validation_metrics(model: torch.nn.Module, bundle, device: torch.device,
                              chunk: int = 40960
                              ) -> Tuple[Dict[int, Dict[str, float]],
                                          Dict[int, float], Dict[int, int]]:
    domain_num, task_num = bundle.domain_num, bundle.task_num
    val_x = np.ascontiguousarray(bundle.val_x.astype(np.float32))
    val_y = np.ascontiguousarray(bundle.val_y.astype(np.float32))
    raw_bce: Dict[int, float] = {t * domain_num + d: 0.0
                                  for t in range(task_num) for d in range(domain_num)}
    sample_count: Dict[int, int] = {t * domain_num + d: 0
                                     for t in range(task_num) for d in range(domain_num)}
    truth: Dict[int, List[float]] = {t * domain_num + d: []
                                       for t in range(task_num) for d in range(domain_num)}
    preds: Dict[int, List[float]] = {key: [] for key in truth}
    model.eval()
    with torch.no_grad():
        for left in range(0, len(val_x), chunk):
            right = min(len(val_x), left + chunk)
            x_chunk = torch.from_numpy(val_x[left:right]).to(device=device, dtype=torch.float32)
            y_chunk = torch.from_numpy(val_y[left:right]).to(device=device, dtype=torch.float32)
            pred = model(x_chunk.transpose(0, 1))
            if not bool(torch.isfinite(pred).all().detach().cpu()):
                raise FloatingPointError("non-finite validation prediction")
            domain_ids = x_chunk[:, -1].to(device=device, dtype=torch.long)
            for task in range(task_num):
                for domain in range(domain_num):
                    key = task * domain_num + domain
                    mask = domain_ids == domain
                    if not bool(mask.any()):
                        continue
                    p_cell = pred[mask, task].detach().cpu().numpy().astype(np.float64)
                    y_cell = y_chunk[mask, task].detach().cpu().numpy().astype(np.float64)
                    p_clip = np.clip(p_cell, LOGLOSS_EPS, 1.0 - LOGLOSS_EPS)
                    bce = -(y_cell * np.log(p_clip) + (1.0 - y_cell) * np.log(1.0 - p_clip))
                    raw_bce[key] += float(bce.sum())
                    sample_count[key] += int(mask.sum().item())
                    truth[key].extend(y_cell.tolist())
                    preds[key].extend(p_cell.tolist())
    records: Dict[int, Dict[str, float]] = {}
    for key in truth:
        labels = truth[key]
        scores = preds[key]
        if len(labels) == 0 or len(set(float(v) for v in labels)) < 2:
            raise ValueError("validation cell {} empty or missing class".format(key))
        records[key] = {"sample_count": int(len(labels)),
                          "auc": float(safe_auc(labels, scores)),
                          "logloss": float(safe_logloss(labels, scores))}
    raw_mean = {key: float(raw_bce[key] / max(1, sample_count[key])) for key in raw_bce}
    return records, raw_mean, sample_count


def _safe_validation_cell_gradients(model: torch.nn.Module, bundle, shared_params,
                                       device: torch.device, chunk: int = 40960
                                       ) -> Dict[int, torch.Tensor]:
    val_x = np.ascontiguousarray(bundle.val_x.astype(np.float32))
    val_y = np.ascontiguousarray(bundle.val_y.astype(np.float32))
    total = len(val_x)
    total_dim = sum(int(p.numel()) for p in shared_params)
    accumulators: Dict[int, torch.Tensor] = {
        idx: torch.zeros(total_dim, dtype=torch.float32, device=device)
        for idx in range(CELL_NUM)
    }
    counts_by_domain: Dict[int, int] = {d: 0 for d in range(DOMAIN_NUM)}
    model.train()
    for left in range(0, total, chunk):
        right = min(total, left + chunk)
        if left >= right:
            continue
        x_chunk = torch.from_numpy(val_x[left:right]).to(device=device, dtype=torch.float32)
        y_chunk = torch.from_numpy(val_y[left:right]).to(device=device, dtype=torch.float32)
        pred = model(x_chunk.transpose(0, 1))
        if not bool(torch.isfinite(pred).all().detach().cpu()):
            raise FloatingPointError("non-finite validation prediction")
        domain_ids = x_chunk[:, -1].to(device=device, dtype=torch.long)
        terms, counts = masked_bce_terms(pred, y_chunk, domain_ids, DOMAIN_NUM, TASK_NUM)
        for d in range(DOMAIN_NUM):
            counts_by_domain[d] += int(counts[d])
        for idx, term in enumerate(terms):
            d = idx % DOMAIN_NUM
            count = int(counts[d])
            if count == 0:
                continue
            grads = torch.autograd.grad(term, list(shared_params), retain_graph=True,
                                         allow_unused=True)
            pieces: List[torch.Tensor] = []
            for grad, param in zip(grads, shared_params):
                if grad is None:
                    pieces.append(torch.zeros_like(param.detach()).reshape(-1).to(
                        dtype=torch.float32, device=param.device))
                else:
                    pieces.append(grad.detach().reshape(-1).to(dtype=torch.float32,
                                                                  device=param.device))
            flat = torch.cat(pieces)
            accumulators[idx] = accumulators[idx] + float(count) * flat
    grads: Dict[int, torch.Tensor] = {}
    for idx in range(CELL_NUM):
        d = idx % DOMAIN_NUM
        n_d = counts_by_domain[d]
        if n_d == 0:
            raise ValueError(
                "Stage S8 validation split is missing domain {}".format(d))
        grads[idx] = (accumulators[idx] / float(n_d)).detach()
    return grads


def _aggregate_validation(records: Mapping[int, Mapping[str, float]],
                           raw_bce_sum: Mapping[int, float],
                           raw_bce_count: Mapping[int, int],
                           target_local_domain: int,
                           domain_shares: Mapping[int, float]) -> Dict[str, float]:
    domain_num, task_num = 3, 2
    shares = {d: float(domain_shares[d]) for d in range(domain_num)}
    macro_auc = float(np.mean([records[key]["auc"] for key in records]))
    macro_ll = float(np.mean([records[key]["logloss"] for key in records]))
    target_keys = [key for key in records if key % domain_num == target_local_domain]
    target_auc = float(np.mean([records[key]["auc"] for key in target_keys]))
    target_ll = float(np.mean([records[key]["logloss"] for key in target_keys]))
    non_target_weights = {d: (shares[d] / max(1e-12, (1.0 - shares[target_local_domain])))
                          for d in range(domain_num) if d != target_local_domain}
    weight_sum = sum(non_target_weights.values())
    if weight_sum > 0:
        non_target_weights = {d: v / weight_sum for d, v in non_target_weights.items()}
    non_target_keys = [key for key in records if key % domain_num != target_local_domain]
    weighted_auc = 0.0
    weighted_ll = 0.0
    weight_denom = 0.0
    for key in non_target_keys:
        d = key % domain_num
        weighted_auc += non_target_weights[d] * records[key]["auc"]
        weighted_ll += non_target_weights[d] * records[key]["logloss"]
        weight_denom += non_target_weights[d]
    if weight_denom > 0:
        weighted_auc /= weight_denom
        weighted_ll /= weight_denom
    target_loss = 0.5 * sum(raw_bce_sum[task * domain_num + target_local_domain]
                              for task in range(task_num))
    non_target_loss = 0.0
    for d in range(domain_num):
        if d == target_local_domain:
            continue
        non_target_loss += non_target_weights.get(d, 0.0) * 0.5 * sum(
            raw_bce_sum[task * domain_num + d] for task in range(task_num))
    return {"macro_auc": macro_auc, "macro_logloss": macro_ll,
            "target_domain_auc": target_auc, "target_domain_logloss": target_ll,
            "non_target_weighted_auc": float(weighted_auc),
            "non_target_weighted_logloss": float(weighted_ll),
            "target_loss": float(target_loss),
            "non_target_loss": float(non_target_loss)}


# ---------------------------------------------------------------------------
# Diagnostics + step (per-branch)
# ---------------------------------------------------------------------------

def _flatten_shared(loss: torch.Tensor, shared_params: Sequence[torch.nn.Parameter],
                     retain_graph: bool = True) -> torch.Tensor:
    grads = torch.autograd.grad(loss, list(shared_params), retain_graph=retain_graph,
                                 allow_unused=True)
    pieces: List[torch.Tensor] = []
    for grad, param in zip(grads, shared_params):
        if grad is None:
            pieces.append(torch.zeros_like(param.detach()).reshape(-1).to(
                dtype=torch.float32, device=param.device))
        else:
            pieces.append(grad.detach().reshape(-1).to(dtype=torch.float32, device=param.device))
    return torch.cat(pieces)


def _safe_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def diagnostics_at_theta(model: torch.nn.Module, branch_tag: str, step: int,
                          batch: Mapping[str, object], bundle,
                          shared_names: Sequence[str], private_names: Sequence[str],
                          slices: Mapping[str, Tuple[int, int]],
                          device: torch.device, val_chunk: int = 40960,
                          sample_val_grad: bool = False) -> Dict[str, object]:
    """Compute one branch's PRE-STEP diagnostics on the current main batch:
    cell losses + grad norms, shared/private grad norms, g_S = sum w_i g_i
    reconstruction, and -- at val-gradient sample steps only -- the
    validation cell gradients plus the validation FULL / TARGET objective
    gradients (sample semantics).

    Validation METRICS are deliberately NOT computed here: they must be
    evaluated on the POST-step parameter state by the caller (the step row
    ``k`` records the state after step ``k``).
    """
    model.train()
    branch_parameters = dict(model.named_parameters())
    shared_params = [branch_parameters[name] for name in shared_names]
    private_params = [branch_parameters[name] for name in private_names]
    x_cpu = batch["x"]
    y_cpu = batch["y"]
    x_device = x_cpu.to(device=device, dtype=torch.float32)
    y_device = y_cpu.to(device=device, dtype=torch.float32)
    domain_ids = x_cpu[:, -1].to(device=device, dtype=torch.long)
    target_local = _target_local_domain(str(bundle.dataset))
    val_grads: Optional[Dict[int, torch.Tensor]] = None
    if sample_val_grad:
        val_grads = _safe_validation_cell_gradients(model, bundle, shared_params, device,
                                                     chunk=val_chunk)
        val_vecs_fp64 = {idx: _to_fp64_flat([val_grads[idx]]) for idx in range(CELL_NUM)}
        g_val_full, g_val_target = _validation_full_target_grads(
            model, bundle, shared_params, device, target_local, chunk=val_chunk)
        g_val_full_fp64 = _to_fp64_flat([g_val_full])
        g_val_target_fp64 = _to_fp64_flat([g_val_target])
    else:
        val_vecs_fp64 = None
        g_val_full_fp64 = None
        g_val_target_fp64 = None
    model.train()
    pred = model(x_device.transpose(0, 1))
    if not bool(torch.isfinite(pred).all().detach().cpu()):
        raise FloatingPointError("non-finite train prediction")
    terms, _ = masked_bce_terms(pred, y_device, domain_ids, DOMAIN_NUM, TASK_NUM)
    cell_grads: Dict[int, torch.Tensor] = {}
    cell_losses: Dict[int, float] = {}
    for idx, term in enumerate(terms):
        cell_grads[idx] = _flatten_shared(term, shared_params, retain_graph=True)
        cell_losses[idx] = float(term.detach().cpu())
    losses = loss_components(pred, y_device, domain_ids, target_local,
                               "sample", {})
    full_loss, target_loss = losses["full"], losses["target"]
    non_target_loss = losses["non_target"]
    g_full_vec = _flatten_shared(full_loss, shared_params, retain_graph=True)
    g_target_vec = _flatten_shared(target_loss, shared_params, retain_graph=True)
    g_private_tup = torch.autograd.grad(full_loss, list(private_params),
                                          retain_graph=False, allow_unused=True)
    g_private_norm = float(vector_norm(list(g_private_tup)))
    weights = batch["w"]
    weighted_sum = torch.zeros_like(g_full_vec, dtype=torch.float32,
                                     device=g_full_vec.device)
    for idx in range(CELL_NUM):
        weighted_sum = weighted_sum + float(weights[idx]) * cell_grads[idx]
    recon_residual_gS = float(torch.linalg.vector_norm(
        g_full_vec.double() - weighted_sum.double()).detach().cpu())
    g_full_norm = float(torch.linalg.vector_norm(g_full_vec).detach().cpu())
    g_target_norm = float(torch.linalg.vector_norm(g_target_vec).detach().cpu())
    cos_gf_gt = _cosine_fp64(_to_fp64_flat([g_full_vec]),
                               _to_fp64_flat([g_target_vec]))
    if not math.isfinite(cos_gf_gt):
        cos_gf_gt = 0.0
    out: Dict[str, object] = {
        "branch": branch_tag, "step": int(step),
        "cell_grads": cell_grads, "cell_losses": cell_losses,
        "g_full_vec": g_full_vec, "g_target_vec": g_target_vec,
        "g_private_norm": g_private_norm,
        "weights": weights,
        "g_full_norm": g_full_norm, "g_target_norm": g_target_norm,
        "cos_gf_gt": float(cos_gf_gt),
        "recon_residual_gS": float(recon_residual_gS),
        "domain_counts": list(batch["domain_counts"]),
        "fingerprint": str(batch["fingerprint"]),
        "L_full": float(full_loss.detach().cpu()),
        "L_target": float(target_loss.detach().cpu()),
        "L_non": float(non_target_loss.detach().cpu()),
        "target_local": int(target_local),
        "slices": {module: (int(slices[module][0]), int(slices[module][1]))
                    for module in slices},
    }
    if val_grads is not None and val_vecs_fp64 is not None:
        out["val_grads"] = val_grads
        out["val_vecs_fp64"] = val_vecs_fp64
        out["g_val_full_fp64"] = g_val_full_fp64
        out["g_val_target_fp64"] = g_val_target_fp64
    return out


def controlled_continuation_step(model: torch.nn.Module, optimizer: torch.optim.Adam,
                                  branch: str, batch: Mapping[str, object],
                                  shared_names: Sequence[str], private_names: Sequence[str],
                                  device: torch.device, target_local: int,
                                  theta_before_F: Optional[Dict[str, torch.Tensor]] = None
                                  ) -> Dict[str, object]:
    """Run one main batch through ``branch`` semantics.

    F: install g_F, step, no overwrite.  applied = Delta_F.
    T: install g_T, step, no overwrite.  applied = Delta_T.
    DH: install g_F, step (committing g_F to moments), overwrite shared
        with theta_before + Delta_T.  applied = Delta_T.
    private parameters always Adam(g_P).  V1 hard gate inside.
    """
    if branch not in proto.BRANCHES:
        raise ValueError("branch must be one of {}".format(proto.BRANCHES))
    parameters = dict(model.named_parameters())
    shared_params = [parameters[name] for name in shared_names]
    private_params = [parameters[name] for name in private_names]
    x_cpu = batch["x"]
    y_cpu = batch["y"]
    x_device = x_cpu.to(device=device, dtype=torch.float32)
    y_device = y_cpu.to(device=device, dtype=torch.float32)
    domain_ids = x_cpu[:, -1].to(device=device, dtype=torch.long)
    pred = model(x_device.transpose(0, 1))
    if not bool(torch.isfinite(pred).all().detach().cpu()):
        raise FloatingPointError("non-finite S8 prediction")
    losses = loss_components(pred, y_device, domain_ids, target_local, "sample", {})
    full_loss, target_loss = losses["full"], losses["target"]
    gf, gt, gp = gradients_one_forward(full_loss, target_loss, parameters,
                                        shared_names, private_names)
    if any(g is None for g in gf):
        raise ValueError("S8 shared FULL gradient has None entries")
    if any(g is None for g in gt):
        raise ValueError("S8 shared TARGET gradient has None entries")
    df, _ = adam_deltas(parameters, shared_names, gf, optimizer)
    dt, _ = adam_deltas(parameters, shared_names, gt, optimizer)
    dp, _ = adam_deltas(parameters, private_names, gp, optimizer)
    if any(d is None for d in df):
        raise ValueError("S8 shared FULL Adam delta has None entries")
    if any(d is None for d in dt):
        raise ValueError("S8 shared TARGET Adam delta has None entries")
    if any(d is None for d in dp):
        raise ValueError("S8 private Adam delta has None entries")
    # m/v norms (pre-step) -- only meaningful when state exists; the
    # continuation's full-state reconstruction has installed state.
    m_norms = []
    v_norms = []
    for name in shared_names:
        state = optimizer.state.get(parameters[name], {})
        if state:
            m_norms.append(vector_norm([state["exp_avg"]]))
            v_norms.append(vector_norm([state["exp_avg_sq"]]))
        else:
            m_norms.append(0.0)
            v_norms.append(0.0)
    m_norm_persistent = float(sum(m_norms))
    v_norm_persistent = float(sum(v_norms))
    before = [parameters[name].detach().clone() for name in shared_names]
    private_before = [parameters[name].detach().clone() for name in private_names]
    history_source = proto.BRANCH_HISTORY[branch]
    current_source = proto.BRANCH_CURRENT[branch]
    g_h = gf if history_source == "FULL" else gt
    g_d = gf if current_source == "FULL" else dt
    installed = [g.detach().clone() for g in g_h]
    installed_private = [None if g is None else g.detach().clone() for g in gp]
    _set_grad(parameters, shared_names, installed)
    _set_grad(parameters, private_names, installed_private)
    rng_before = rng_state_digest()
    optimizer.step()
    rng_after = rng_state_digest()
    if rng_before != rng_after:
        raise AssertionError(
            "S8 {} main step consumed RNG (digest {} -> {})".format(
                branch, rng_before[:16], rng_after[:16]))
    # Branch-specific overwrite (DH only).
    if branch == "DH":
        with torch.no_grad():
            for name, old, target_delta in zip(shared_names, before, dt):
                parameters[name].copy_(old + target_delta)
    observed = [parameters[name].detach() - old for name, old in zip(shared_names, before)]
    expected_applied = df if current_source == "FULL" else dt
    applied_residual = assert_vector_match(observed, expected_applied,
                                            "S8 {} observed == Delta".format(branch))
    private_observed = [parameters[name].detach() - old
                         for name, old in zip(private_names, private_before)]
    private_residual = assert_vector_match(private_observed, dp,
                                            "S8 private observed == Delta_P")
    delta_cos = cosine(df, dt)
    if delta_cos is None or not math.isfinite(delta_cos):
        delta_cos = 0.0
    # theta distance to F (T/DH only).
    theta_dist_F = float("nan")
    if theta_before_F is not None and branch in ("T", "DH"):
        total = 0.0
        for name in shared_names:
            diff = parameters[name].detach().to(dtype=torch.float64, device="cpu") \
                 - theta_before_F[name].detach().to(dtype=torch.float64, device="cpu")
            total += float(torch.sum(diff * diff).item())
        theta_dist_F = math.sqrt(total)
    elif theta_before_F is not None and branch == "F":
        theta_dist_F = 0.0
    return {
        "delta_full_norm": float(vector_norm(df)),
        "delta_target_norm": float(vector_norm(dt)),
        "delta_applied_norm": float(vector_norm(expected_applied)),
        "delta_cosine": float(delta_cos),
        "private_delta_norm": float(vector_norm(dp)),
        "applied_formula_residual": float(applied_residual),
        "private_full_residual": float(private_residual),
        "theta_distance_to_F_shared": float(theta_dist_F),
        "m_norm_persistent": float(m_norm_persistent),
        "v_norm_persistent": float(v_norm_persistent),
        "_delta_full_vec_fp64": _to_fp64_flat(df),
        "_delta_target_vec_fp64": _to_fp64_flat(dt),
        "_delta_applied_vec_fp64": _to_fp64_flat(expected_applied),
    }


def _validation_full_target_grads(model: torch.nn.Module, bundle, shared_params,
                                   device: torch.device, target_local: int,
                                   chunk: int = 40960
                                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validation-set FULL and TARGET objective gradients wrt the shared
    parameters, under the exact ``loss_components(..., "sample", {})``
    semantics:

    - full   = (1/N) * sum over ALL N validation examples of the
      per-example mean-BCE (sample weights all 1);
    - target = (1/N) * sum over the target-domain examples of the
      per-example mean-BCE (denominator is the FULL N, not n_target).

    Gradients of chunked sums are accumulated and divided by N once
    (linearity of the gradient).  Returns two flat fp32 vectors in
    ``shared_params`` order.
    """
    val_x = np.ascontiguousarray(bundle.val_x.astype(np.float32))
    val_y = np.ascontiguousarray(bundle.val_y.astype(np.float32))
    total = len(val_x)
    if total == 0:
        raise ValueError("Stage S8 validation split is empty")
    total_dim = sum(int(p.numel()) for p in shared_params)
    acc_full = torch.zeros(total_dim, dtype=torch.float32, device=device)
    acc_target = torch.zeros(total_dim, dtype=torch.float32, device=device)
    model.train()
    for left in range(0, total, chunk):
        right = min(total, left + chunk)
        x_chunk = torch.from_numpy(val_x[left:right]).to(device=device, dtype=torch.float32)
        y_chunk = torch.from_numpy(val_y[left:right]).to(device=device, dtype=torch.float32)
        pred = model(x_chunk.transpose(0, 1))
        if not bool(torch.isfinite(pred).all().detach().cpu()):
            raise FloatingPointError("non-finite validation prediction")
        per_example = F.binary_cross_entropy(pred, y_chunk, reduction="none").mean(dim=1)
        domain_ids = x_chunk[:, -1].to(device=device, dtype=torch.long)
        target_mask = (domain_ids == int(target_local)).to(dtype=per_example.dtype)
        full_sum = per_example.sum()
        target_sum = (per_example * target_mask).sum()
        grads_full = torch.autograd.grad(full_sum, list(shared_params), retain_graph=True,
                                          allow_unused=True)
        grads_target = torch.autograd.grad(target_sum, list(shared_params), retain_graph=False,
                                            allow_unused=True)
        for grads, acc in ((grads_full, acc_full), (grads_target, acc_target)):
            pieces: List[torch.Tensor] = []
            for grad, param in zip(grads, shared_params):
                if grad is None:
                    pieces.append(torch.zeros_like(param.detach()).reshape(-1).to(
                        dtype=torch.float32, device=param.device))
                else:
                    pieces.append(grad.detach().reshape(-1).to(dtype=torch.float32,
                                                                  device=param.device))
            acc.add_(torch.cat(pieces))
    return (acc_full / float(total)).detach(), (acc_target / float(total)).detach()


def _eff_alignment(applied_vec_fp64: torch.Tensor,
                    g_val_full_fp64: torch.Tensor,
                    g_val_target_fp64: torch.Tensor) -> Dict[str, float]:
    """``eff_align_*`` / ``eff_dot_*`` between the applied effective Adam
    delta (fp64 flat, shared params) and the validation FULL / TARGET
    objective gradients (same layout)."""
    cos_full = _cosine_fp64(applied_vec_fp64, g_val_full_fp64)
    cos_target = _cosine_fp64(applied_vec_fp64, g_val_target_fp64)
    return {
        "eff_align_full": float(cos_full),
        "eff_align_target": float(cos_target),
        "eff_dot_full": float(_dot_fp64(applied_vec_fp64, g_val_full_fp64)),
        "eff_dot_target": float(_dot_fp64(applied_vec_fp64, g_val_target_fp64)),
    }


# ---------------------------------------------------------------------------
# Replay helper for smoke-replay
# ---------------------------------------------------------------------------

def replay_main_batches(bundle, seed: int, start_epoch: int, h: int,
                         shared_names: Sequence[str]) -> Tuple[List[Dict[str, object]],
                                                                  Dict[str, object]]:
    """Replay (and stream) the first ``h`` main batches of epoch
    ``start_epoch`` after consuming ``3 * start_epoch`` iterator draws
    (matches S6 / S7-B RNG bookkeeping)."""
    setup_seed(int(seed))
    build_model(bundle.features, DOMAIN_NUM, TASK_NUM, "m3oe",
                expert_num=4, fcn_dims=(512, 256, 256, 64), seed=int(seed))
    train_loader = DataLoader(
        TorchDataset(bundle.train_x, bundle.train_y),
        batch_size=40960, shuffle=True, num_workers=0, drop_last=False)
    n_iters = 3 * int(start_epoch)
    for _ in range(n_iters):
        _ = iter(train_loader)
    train_iterator = iter(train_loader)
    batches: List[Dict[str, object]] = []
    for step in range(int(h)):
        x, y = next(train_iterator)
        fp = batch_fingerprint(x, y)
        domain_ids = x[:, -1].detach().cpu().numpy().astype(np.int64)
        domain_counts = [int((domain_ids == d).sum()) for d in range(DOMAIN_NUM)]
        batch_rows_actual = int(x.shape[0])
        weights = {cell_index(task, d, DOMAIN_NUM): float(domain_counts[d]) / float(2 * batch_rows_actual)
                   for task in range(TASK_NUM) for d in range(DOMAIN_NUM)}
        batches.append({
            "x": x.detach().cpu().contiguous(),
            "y": y.detach().cpu().contiguous(),
            "fingerprint": fp,
            "domain_counts": domain_counts,
            "batch_size": batch_rows_actual,
            "w": weights,
            "step": int(step),
        })
    audit = {"start_epoch": int(start_epoch), "h_requested": int(h),
              "iter_consumed": int(n_iters),
              "materialized_fingerprints": [b["fingerprint"] for b in batches]}
    return batches, audit


def arch_batch_for_epoch(bundle, dataset: str, seed: int, epoch: int
                          ) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Rebuild the architecture batch the source run consumed at the end of
    ``epoch`` (design §4.3.2).

    The source arch iterator is created lazily inside
    ``preserve_rng_state`` and reused across ``bpe`` consecutive epochs
    (one generation); a new generation starts when ``next`` exhausts it.
    Generation ``g = epoch // bpe`` is created at the arch-update point of
    epoch ``g * bpe``, i.e. from the global RNG state after
    ``3 * (g * bpe + 1)`` draws (main-iter creation + first next +
    train-records-iter creation per epoch).  Within the generation, the
    batch for ``epoch`` is at offset ``epoch % bpe`` (0-based).
    """
    bpe = int(proto.BATCHES_PER_EPOCH[str(dataset).lower()])
    epoch_start = (int(epoch) // bpe) * bpe
    offset = int(epoch) - epoch_start
    with preserve_rng_state():
        setup_seed(int(seed))
        _ = build_model(bundle.features, DOMAIN_NUM, TASK_NUM, "m3oe",
                         expert_num=4, fcn_dims=(512, 256, 256, 64),
                         seed=int(seed))
        train_loader = DataLoader(
            TorchDataset(bundle.train_x, bundle.train_y),
            batch_size=40960, shuffle=True, num_workers=0, drop_last=False)
        for _ in range(3 * (int(epoch_start) + 1)):
            _ = iter(train_loader)
        arch_iterator = iter(train_loader)
        x_arch, y_arch = next(arch_iterator)
        for _ in range(offset):
            x_arch, y_arch = next(arch_iterator)
    return x_arch, y_arch, batch_fingerprint(x_arch, y_arch)


def _arch_controlled_step(model: torch.nn.Module, arch_optimizer: torch.optim.Adam,
                           branch: str, x_cpu: torch.Tensor, y_cpu: torch.Tensor,
                           arch_names: Sequence[str], target_local: int,
                           device: torch.device) -> Dict[str, object]:
    """One architecture update under ``branch`` semantics (mirrors S6
    ``_controlled_step(kind="architecture")``):

    F:  install g_F(arch), step, applied = Delta_F.
    T:  install g_T(arch), step, applied = Delta_T.
    DH: install g_F(arch), step (committing g_F to the arch moments),
        overwrite arch params with theta_before + Delta_T.

    Hard gates: None-gradient fail-closed, non-empty Adam state for every
    arch param, RNG digest invariant across the step, and
    ``assert_vector_match`` on the applied delta.
    """
    if branch not in proto.BRANCHES:
        raise ValueError("branch must be one of {}".format(proto.BRANCHES))
    parameters = dict(model.named_parameters())
    arch_params = [parameters[name] for name in arch_names]
    model.train()
    x_device = x_cpu.to(device=device, dtype=torch.float32)
    y_device = y_cpu.to(device=device, dtype=torch.float32)
    domain_ids = x_cpu[:, -1].to(device=device, dtype=torch.long)
    pred = model(x_device.transpose(0, 1))
    if not bool(torch.isfinite(pred).all().detach().cpu()):
        raise FloatingPointError("non-finite S8 arch prediction")
    losses = loss_components(pred, y_device, domain_ids, int(target_local),
                               "sample", {})
    gf, gt, _ = gradients_one_forward(losses["full"], losses["target"],
                                       parameters, arch_names, [])
    if any(g is None for g in gf):
        raise ValueError("S8 arch FULL gradient has None entries")
    if any(g is None for g in gt):
        raise ValueError("S8 arch TARGET gradient has None entries")
    df, _ = adam_deltas(parameters, arch_names, gf, arch_optimizer)
    dt, _ = adam_deltas(parameters, arch_names, gt, arch_optimizer)
    if any(d is None for d in df):
        raise ValueError("S8 arch FULL Adam delta has None entries")
    if any(d is None for d in dt):
        raise ValueError("S8 arch TARGET Adam delta has None entries")
    for name in arch_names:
        if not arch_optimizer.state.get(parameters[name]):
            raise AssertionError("S8 arch Adam state empty for " + name)
    before = [parameters[name].detach().clone() for name in arch_names]
    history_source = proto.BRANCH_HISTORY[branch]
    current_source = proto.BRANCH_CURRENT[branch]
    g_h = gf if history_source == "FULL" else gt
    _set_grad(parameters, arch_names, [g.detach().clone() for g in g_h])
    rng_before = rng_state_digest()
    arch_optimizer.step()
    rng_after = rng_state_digest()
    if rng_before != rng_after:
        raise AssertionError("S8 {} arch step consumed RNG".format(branch))
    if branch == "DH":
        with torch.no_grad():
            for name, old, target_delta in zip(arch_names, before, dt):
                parameters[name].copy_(old + target_delta)
    observed = [parameters[name].detach() - old for name, old in zip(arch_names, before)]
    expected_applied = df if current_source == "FULL" else dt
    applied_residual = assert_vector_match(observed, expected_applied,
                                            "S8 arch {} applied == Delta".format(branch))
    return {
        "arch_applied_formula_residual": float(applied_residual),
        "arch_delta_full_norm": float(vector_norm(df)),
        "arch_delta_target_norm": float(vector_norm(dt)),
        "arch_delta_applied_norm": float(vector_norm(expected_applied)),
    }


# ---------------------------------------------------------------------------
# Main continuation runner
# ---------------------------------------------------------------------------

def run_continuation(dataset: str, seed: int, anchor_class: str,
                      out_dir: str, h_max_epochs: int = proto.H_MAX_EPOCHS_CORE,
                      dataset_path: str = "",
                      smoke_restore: bool = False,
                      smoke_replay: bool = False,
                      smoke_onestep: bool = False) -> Dict[str, object]:
    dataset = str(dataset).lower()
    if dataset not in proto.DATASETS:
        raise ValueError("dataset must be kuairand or movielens")
    if int(seed) not in proto.SEEDS:
        raise ValueError("seed outside matrix")
    if anchor_class not in proto.ANCHOR_CLASSES:
        raise ValueError("anchor_class outside matrix")
    if sum([smoke_restore, smoke_replay, smoke_onestep]) > 1:
        raise ValueError("only one smoke mode at a time")
    started = time.time()
    anchor_epoch = proto.anchor_epoch_for(dataset, int(seed), anchor_class)
    base_run_dir = proto.base_run_dir(dataset, int(seed))
    if not os.path.isdir(base_run_dir):
        raise FileNotFoundError("S8 base run missing: " + base_run_dir)
    base_manifest = _read_s8_base_manifest(base_run_dir)
    base_batch_trace = _read_s8_base_batch_trace(base_run_dir)
    base_main_rows = [row for row in base_batch_trace if str(row.get("kind", "")) == "train"]
    base_arch_rows = [row for row in base_batch_trace if str(row.get("kind", "")) == "architecture"]
    base_metrics_epoch = _epochs_metrics_lookup(_read_s8_base_metrics_epoch(base_run_dir))
    initial_hash_expected = str(base_manifest.get("initial_parameter_state_hash", ""))
    setting = proto.target_setting(dataset)
    bundle = load_stage_s1_bundle(dataset, dataset_path or None, setting, int(seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(int(seed))
    model = build_model(bundle.features, DOMAIN_NUM, TASK_NUM, "m3oe",
                        expert_num=4, fcn_dims=(512, 256, 256, 64), seed=int(seed))
    model = model.to(device)
    initial_hash_observed = initial_state_hash(model)
    if initial_hash_expected and str(initial_hash_observed) != initial_hash_expected:
        raise AssertionError(
            "Stage S8 initial hash mismatch: observed {} != expected {}".format(
                initial_hash_observed, initial_hash_expected))
    partition = partition_parameters(model)
    assert_expected_parameter_counts(partition, dataset)
    parameters = dict(model.named_parameters())
    shared_names = list(partition["groups"][GROUP_SHARED])
    private_names = list(partition["groups"][GROUP_PRIVATE])
    arch_names = list(partition["groups"][GROUP_ARCH])
    slices = module_slices(shared_names, parameters)
    target_local = _target_local_domain(dataset)
    bpe = proto.BATCHES_PER_EPOCH[dataset]
    horizons = proto.horizon_steps(dataset, int(h_max_epochs))
    h_max_steps = max(horizons.values())
    valgrad_steps = set(proto.valgrad_sample_steps(h_max_steps, bpe))
    # ------------------------------------------------------------------
    # Anchor reconstruction (full state).
    # ------------------------------------------------------------------
    reconstruction = reconstruct_anchor_full(model, base_run_dir, anchor_epoch,
                                              device, shared_names, private_names,
                                              arch_names)
    anchor_audit = reconstruction["audit"]
    expected_shared_step = proto.expected_shared_step(dataset, int(anchor_epoch))
    anchor_audit["shared_step_expected"] = int(expected_shared_step)
    if int(anchor_audit["shared_step_observed"]) != int(expected_shared_step):
        raise AssertionError(
            "S8 anchor shared Adam step {} != expected {}".format(
                anchor_audit["shared_step_observed"], expected_shared_step))
    expected_arch_step = proto.expected_arch_step(int(anchor_epoch))
    if int(anchor_audit["arch_step_observed"]) != int(expected_arch_step):
        raise AssertionError(
            "S8 anchor arch Adam step {} != expected {}".format(
                anchor_audit["arch_step_observed"], expected_arch_step))
    if not bool(anchor_audit["private_state_coverage"]):
        raise AssertionError(
            "S8 anchor private state coverage failed: {} of {}".format(
                anchor_audit["private_state_count"],
                anchor_audit["private_param_count"]))
    model = reconstruction["model"]
    optimizer = reconstruction["optimizer"]
    arch_optimizer = reconstruction["arch_optimizer"]
    # ------------------------------------------------------------------
    # smoke-restore: only the reconstruction + a single sanity bit-equal
    # clone of the model + optimizer state (V3 equivalent).
    # ------------------------------------------------------------------
    if smoke_restore:
        new_model, new_opt, new_arch = branch_state_full(model, optimizer, arch_optimizer,
                                                           shared_names, private_names)
        same, rows = _branch_state_bit_equal(model, optimizer, arch_optimizer,
                                              new_model, new_opt, new_arch,
                                              shared_names, private_names)
        report = {
            "smoke_restore_ok": bool(same),
            "dataset": dataset, "seed": int(seed),
            "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
            "shared_step_observed": int(anchor_audit["shared_step_observed"]),
            "shared_step_expected": int(expected_shared_step),
            "arch_step_observed": int(anchor_audit["arch_step_observed"]),
            "arch_step_expected": int(expected_arch_step),
            "private_state_count": int(anchor_audit["private_state_count"]),
            "private_state_coverage": bool(anchor_audit["private_state_coverage"]),
            "v3_state_bit_equality": bool(same),
            "full_state_dump_sha256": str(anchor_audit["full_state_dump_sha256"]),
            "base_run_dir": base_run_dir,
        }
        if not same:
            report["v3_per_param"] = rows
            report["smoke_restore_ok"] = False
        return report
    # ------------------------------------------------------------------
    # Branch construction (deepcopy, bit-equal at step 0).
    # ------------------------------------------------------------------
    f_model, f_opt, f_arch = branch_state_full(model, optimizer, arch_optimizer,
                                                 shared_names, private_names)
    t_model, t_opt, t_arch = branch_state_full(model, optimizer, arch_optimizer,
                                                 shared_names, private_names)
    dh_model, dh_opt, dh_arch = branch_state_full(model, optimizer, arch_optimizer,
                                                    shared_names, private_names)
    # V3 cross-branch pre-step bit-equality.
    v3_branch_pairs: List[Tuple[str, str, torch.nn.Module, torch.optim.Adam,
                                  Optional[torch.optim.Adam], torch.nn.Module,
                                  torch.optim.Adam, Optional[torch.optim.Adam]]] = [
        ("F", "T", f_model, f_opt, f_arch, t_model, t_opt, t_arch),
        ("F", "DH", f_model, f_opt, f_arch, dh_model, dh_opt, dh_arch),
        ("T", "DH", t_model, t_opt, t_arch, dh_model, dh_opt, dh_arch),
    ]
    v3_per_pair: Dict[str, Dict[str, object]] = {}
    for tag_a, tag_b, ma, oa, aa, mb, ob, ab in v3_branch_pairs:
        ok, rows = _branch_state_bit_equal(ma, oa, aa, mb, ob, ab,
                                            shared_names, private_names)
        v3_per_pair["{}_{}".format(tag_a, tag_b)] = {"all_equal": bool(ok), "rows": rows}
        if not ok:
            raise AssertionError("S8 V3 branch bit-equality failed: {}/{}".format(tag_a, tag_b))
    # Anchor-time parameter snapshot of the F branch (used by
    # smoke-onestep as the theta_before_F placeholder).
    f_params_initial = {name: param.detach().clone() for name, param in f_model.named_parameters()}
    # ------------------------------------------------------------------
    # Step 0: validation metrics for each branch (anchor point).
    # ------------------------------------------------------------------
    step0_val_rows: List[Dict[str, object]] = []
    step0_aggregates: Dict[str, Dict[str, float]] = {}
    for branch_name, branch_model in (("F", f_model), ("T", t_model), ("DH", dh_model)):
        records, raw_mean, raw_count = _full_validation_metrics(branch_model, bundle, device)
        aggregates = _aggregate_validation(records, raw_mean, raw_count,
                                            target_local, bundle.domain_shares)
        step0_aggregates[branch_name] = aggregates
        for cell_key in sorted(records):
            d = cell_key % DOMAIN_NUM
            t = cell_key // DOMAIN_NUM
            step0_val_rows.append({
                "stage": "S8", "dataset": dataset, "seed": int(seed),
                "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
                "branch": branch_name, "step": 0, "kind": "anchor",
                "scope": "cell", "domain": d, "task": t, "cell": cell_key,
                "val_bce_raw": float(raw_mean[cell_key]),
                "auc": float(records[cell_key]["auc"]),
                "logloss": float(records[cell_key]["logloss"]),
            })
        for scope_name in ("macro", "target_domain", "non_target_weighted"):
            if scope_name == "macro":
                auc_v, ll_v = aggregates["macro_auc"], aggregates["macro_logloss"]
                loss_v = float("nan")
            elif scope_name == "target_domain":
                auc_v, ll_v = (aggregates["target_domain_auc"],
                                aggregates["target_domain_logloss"])
                loss_v = float(aggregates["target_loss"])
            else:
                auc_v, ll_v = (aggregates["non_target_weighted_auc"],
                                aggregates["non_target_weighted_logloss"])
                loss_v = float(aggregates["non_target_loss"])
            step0_val_rows.append({
                "stage": "S8", "dataset": dataset, "seed": int(seed),
                "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
                "branch": branch_name, "step": 0, "kind": "anchor",
                "scope": scope_name, "domain": -1, "task": -1, "cell": -1,
                "val_bce_raw": float(loss_v),
                "auc": float(auc_v),
                "logloss": float(ll_v),
            })
    # Compare step-0 F macro AUC vs base metrics_epoch anchor row.
    base_anchor_row = base_metrics_epoch.get(int(anchor_epoch), {})
    base_anchor_macro_auc_str = str(base_anchor_row.get("validation_macro_auc", ""))
    base_anchor_macro_auc = _safe_float(base_anchor_macro_auc_str)
    f_step0_macro_auc = float(step0_aggregates["F"]["macro_auc"])
    if base_anchor_macro_auc is not None:
        step0_v0_fail = bool(abs(f_step0_macro_auc - base_anchor_macro_auc)
                              > proto.V0_VALIDATION_MACRO_AUC_ABS)
    else:
        step0_v0_fail = False
    # ------------------------------------------------------------------
    # smoke-onestep: only step 0 / step 1 of the F/T branches.
    # ------------------------------------------------------------------
    if smoke_onestep:
        # Replay just one main batch.
        batches, replay_audit = replay_main_batches(bundle, int(seed),
                                                     int(anchor_epoch) + 1, 1,
                                                     shared_names)
        # Source main rows for step 0 of epoch anchor+1.
        source_main = _main_rows_for(base_main_rows, int(anchor_epoch) + 1, 1)
        f_diag = diagnostics_at_theta(f_model, "F", 0, batches[0], bundle,
                                        shared_names, private_names, slices, device)
        f_step = controlled_continuation_step(f_model, f_opt, "F", batches[0],
                                                shared_names, private_names, device,
                                                target_local,
                                                theta_before_F=f_params_initial)
        t_diag = diagnostics_at_theta(t_model, "T", 0, batches[0], bundle,
                                        shared_names, private_names, slices, device)
        t_step = controlled_continuation_step(t_model, t_opt, "T", batches[0],
                                                shared_names, private_names, device,
                                                target_local,
                                                theta_before_F=f_params_initial)
        return {
            "smoke_onestep_ok": bool(
                str(batches[0]["fingerprint"])
                == str(source_main[0].get("batch_fingerprint", ""))
                and not bool(step0_v0_fail)),
            "dataset": dataset, "seed": int(seed),
            "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
            "F_applied_formula_residual": float(f_step["applied_formula_residual"]),
            "T_applied_formula_residual": float(t_step["applied_formula_residual"]),
            "F_private_full_residual": float(f_step["private_full_residual"]),
            "T_private_full_residual": float(t_step["private_full_residual"]),
            "F_L_full": float(f_diag["L_full"]),
            "T_L_full": float(t_diag["L_full"]),
            "F_delta_cosine": float(f_step["delta_cosine"]),
            "T_delta_cosine": float(t_step["delta_cosine"]),
            "F_step0_fingerprint": str(batches[0]["fingerprint"]),
            "source_F_step0_fingerprint": str(source_main[0].get("batch_fingerprint", "")),
            "fingerprint_match": (str(batches[0]["fingerprint"])
                                   == str(source_main[0].get("batch_fingerprint", ""))),
            "F_step0_macro_auc": f_step0_macro_auc,
            "base_anchor_macro_auc": base_anchor_macro_auc,
            "v0_step0_fail": bool(step0_v0_fail),
        }
    # ------------------------------------------------------------------
    # smoke-replay: replay the first 5 main batches + the first arch
    # batch of epoch anchor+1; verify their fingerprints vs the S8
    # base batch_trace.csv.
    # ------------------------------------------------------------------
    if smoke_replay:
        replay_h = min(5, len(base_main_rows))
        batches, replay_audit = replay_main_batches(bundle, int(seed),
                                                     int(anchor_epoch) + 1, replay_h,
                                                     shared_names)
        source_fps: List[str] = []
        for row in base_main_rows:
            try:
                ep = int(row.get("epoch", "-1"))
            except (TypeError, ValueError):
                continue
            if ep != int(anchor_epoch) + 1:
                continue
            source_fps.append(str(row.get("batch_fingerprint", "")))
        source_fps = source_fps[:replay_h]
        match_rows: List[Dict[str, object]] = []
        for idx, fp in enumerate(replay_audit["materialized_fingerprints"]):
            match_rows.append({"index": idx, "match": fp == source_fps[idx]
                                if idx < len(source_fps) else False,
                                "observed": fp,
                                "expected": source_fps[idx] if idx < len(source_fps) else ""})
        # Arch batch fingerprint for epoch anchor+1.
        arch_epoch = int(anchor_epoch) + 1
        arch_rows = _arch_rows_for(base_batch_trace, arch_epoch)
        arch_source_fp = str(arch_rows[0].get("batch_fingerprint", "")) if arch_rows else ""
        # Reconstruct arch batch via the same RNG replay as the source.
        # The source ``_architecture_update`` is invoked at the END of
        # each epoch.  The arch iterator is built inside
        # ``preserve_rng_state`` and reused across epochs (``bpe`` arch
        # updates per iter).  The iter is rebuilt when ``next`` raises
        # StopIteration.  To get the arch batch at epoch ``arch_epoch``:
        #   1. Consume global draws up to the start of the iter cycle
        #      containing ``arch_epoch``'s batch (i.e., 3*(epoch_start+1)
        #      global draws, where epoch_start = (arch_epoch // bpe) * bpe).
        #   2. Inside preserve_rng_state, build the iter and take
        #      ``arch_epoch - epoch_start`` next() calls to advance to
        #      the right batch.
        # For ML bpe=5, arch_epoch=8: epoch_start = 5.  Consume 18
        # global draws, build iter, take 3 next() calls, the 4th is
        # the arch batch.
        bpe_value = int(proto.BATCHES_PER_EPOCH[dataset])
        epoch_start = (int(arch_epoch) // bpe_value) * bpe_value
        n_iters = 3 * (int(epoch_start) + 1)
        offset = int(arch_epoch) - epoch_start
        with preserve_rng_state():
            setup_seed(int(seed))
            _ = build_model(bundle.features, DOMAIN_NUM, TASK_NUM, "m3oe",
                             expert_num=4, fcn_dims=(512, 256, 256, 64),
                             seed=int(seed))
            train_loader = DataLoader(
                TorchDataset(bundle.train_x, bundle.train_y),
                batch_size=40960, shuffle=True, num_workers=0, drop_last=False)
            for _ in range(n_iters):
                _ = iter(train_loader)
            arch_iterator = iter(train_loader)
            x_arch, y_arch = next(arch_iterator)
            for _ in range(offset):
                x_arch, y_arch = next(arch_iterator)
        arch_fp_observed = batch_fingerprint(x_arch, y_arch)
        return {
            "smoke_replay_ok": bool(all(r.get("match", False) for r in match_rows)
                                     and (not arch_source_fp or arch_fp_observed == arch_source_fp)),
            "dataset": dataset, "seed": int(seed),
            "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
            "epoch_target": int(anchor_epoch) + 1,
            "materialized_fingerprints": replay_audit["materialized_fingerprints"],
            "source_fingerprints": source_fps,
            "fingerprint_rows": match_rows,
            "arch_source_fingerprint": arch_source_fp,
            "arch_observed_fingerprint": arch_fp_observed,
            "arch_fingerprint_match": arch_fp_observed == arch_source_fp,
        }
    # ------------------------------------------------------------------
    # Formal run: out_dir non-empty refusal.
    # ------------------------------------------------------------------
    out_dir_abs = os.path.abspath(out_dir)
    if os.path.isdir(out_dir_abs):
        existing = [name for name in os.listdir(out_dir_abs)
                    if name not in ("failed_attempts", "stdout.log", "process_interval.json")]
        if existing:
            raise FileExistsError(
                "Stage S8 refusing to overwrite non-empty run directory: " + out_dir_abs)
    os.makedirs(out_dir_abs, exist_ok=True)
    # The continuation must stay inside the 50-epoch trained trajectory
    # (batch fingerprints / arch fingerprints / gradient_step rows are read
    # from the S8-BASE run, which ends at epoch 49).
    if int(anchor_epoch) + int(h_max_epochs) > 49:
        raise ValueError(
            "S8 continuation horizon exceeds the trained trajectory: "
            "anchor {} + h_max {} > 49".format(anchor_epoch, h_max_epochs))
    # ------------------------------------------------------------------
    # f_original references: the ORIGINAL S6 canonical manifest (design
    # §9.11), hard-crosschecked against the S8-BASE retrain manifest
    # (selected_epoch equality + selected AUC |diff| <= 1e-6) and, for
    # anchor_class == "selected", against the frozen anchor pin.
    # ------------------------------------------------------------------
    canonical_dir = proto.source_run_dir(dataset, int(seed))
    canonical_manifest_path = os.path.join(canonical_dir, "run_manifest.json")
    if not os.path.isfile(canonical_manifest_path):
        raise FileNotFoundError("S6 canonical manifest missing: " + canonical_manifest_path)
    with open(canonical_manifest_path, encoding="utf-8") as handle:
        canonical_manifest = json.load(handle)
    f_original_best_macro_auc = _safe_float(
        canonical_manifest.get("selected_validation_macro_auc", ""))
    try:
        f_original_selected_epoch = int(canonical_manifest.get("selected_epoch", -1))
    except (TypeError, ValueError):
        f_original_selected_epoch = -1
    if f_original_best_macro_auc is None or f_original_selected_epoch < 0:
        raise ValueError(
            "S6 canonical manifest lacks selected_epoch/selected_validation_macro_auc: "
            + canonical_manifest_path)
    base_selected_auc = _safe_float(base_manifest.get("selected_validation_macro_auc", ""))
    try:
        base_selected_epoch = int(base_manifest.get("selected_epoch", -2))
    except (TypeError, ValueError):
        base_selected_epoch = -2
    if base_selected_epoch != f_original_selected_epoch:
        raise AssertionError(
            "S8-BASE selected_epoch {} != canonical {}".format(
                base_selected_epoch, f_original_selected_epoch))
    if base_selected_auc is None or \
            abs(base_selected_auc - f_original_best_macro_auc) > 1e-6:
        raise AssertionError(
            "S8-BASE selected AUC {} vs canonical {} diff above 1e-6".format(
                base_selected_auc, f_original_best_macro_auc))
    if anchor_class == "selected" and int(anchor_epoch) != f_original_selected_epoch:
        raise AssertionError(
            "S8 anchor pin {} != canonical selected_epoch {} (dataset={} seed={})".format(
                anchor_epoch, f_original_selected_epoch, dataset, seed))
    f_original_crosscheck = {
        "f_original_source": "s6_canonical_manifest",
        "canonical_run_dir": canonical_dir,
        "canonical_selected_epoch": int(f_original_selected_epoch),
        "canonical_selected_validation_macro_auc": float(f_original_best_macro_auc),
        "base_selected_epoch": int(base_selected_epoch),
        "base_selected_validation_macro_auc": (float(base_selected_auc)
                                                 if base_selected_auc is not None else None),
        "base_vs_canonical_auc_abs_diff": (abs(base_selected_auc - f_original_best_macro_auc)
                                             if base_selected_auc is not None else None),
        "anchor_pin_matches_canonical_selected": bool(
            int(anchor_epoch) == int(f_original_selected_epoch)),
    }
    # ------------------------------------------------------------------
    # Source-trajectory lookups (S8-BASE == F-arm retrain of the source).
    # ------------------------------------------------------------------
    source_step_rows: Dict[Tuple[int, int], Dict[str, str]] = {}
    for row in _read_s8_base_gradient_step(base_run_dir):
        try:
            key = (int(row.get("epoch", "-1")), int(row.get("step", "-1")))
        except (TypeError, ValueError):
            continue
        source_step_rows[key] = row
    source_main_fp: Dict[Tuple[int, int], str] = {}
    for row in base_main_rows:
        try:
            key = (int(row.get("epoch", "-1")), int(row.get("step", "-1")))
        except (TypeError, ValueError):
            continue
        source_main_fp[key] = str(row.get("batch_fingerprint", ""))
    source_arch_fp: Dict[int, str] = {}
    for row in base_arch_rows:
        try:
            ep = int(row.get("epoch", "-1"))
        except (TypeError, ValueError):
            continue
        source_arch_fp[ep] = str(row.get("batch_fingerprint", ""))
    for e in range(int(anchor_epoch) + 1, int(anchor_epoch) + 1 + int(h_max_epochs)):
        for s in range(int(bpe)):
            if (e, s) not in source_main_fp:
                raise ValueError("S8 base batch_trace missing train row epoch={} step={}".format(e, s))
            if (e, s) not in source_step_rows:
                raise ValueError("S8 base gradient_step missing main row epoch={} step={}".format(e, s))
        if e not in source_arch_fp:
            raise ValueError("S8 base batch_trace missing architecture row epoch={}".format(e))
    # ------------------------------------------------------------------
    # V0 anchor-point hard gate: step-0 (anchor state) F macro AUC must
    # equal the S8-BASE metrics_epoch row at the anchor epoch (same state,
    # same eval) within V0_VALIDATION_MACRO_AUC_ABS.
    # ------------------------------------------------------------------
    v0_step0_macro_auc = {
        "observed": float(f_step0_macro_auc),
        "expected": (float(base_anchor_macro_auc)
                      if base_anchor_macro_auc is not None else None),
        "abs_diff": (float(abs(f_step0_macro_auc - base_anchor_macro_auc))
                      if base_anchor_macro_auc is not None else None),
        "fail": bool(step0_v0_fail),
    }
    # ------------------------------------------------------------------
    # Run state containers (streaming -- no per-step tensor history is
    # retained; only scalar metrics and one best state_dict per branch).
    # ------------------------------------------------------------------
    branches: Dict[str, Dict[str, object]] = {
        "F": {"model": f_model, "opt": f_opt, "arch_opt": f_arch},
        "T": {"model": t_model, "opt": t_opt, "arch_opt": t_arch},
        "DH": {"model": dh_model, "opt": dh_opt, "arch_opt": dh_arch},
    }
    shared_set = set(shared_names)
    step_rows: List[Dict[str, object]] = []
    geometry_rows: List[Dict[str, object]] = []
    val_rows: List[Dict[str, object]] = list(step0_val_rows)
    metrics_by_step: Dict[str, Dict[int, Dict[str, float]]] = {
        b: {} for b in proto.BRANCHES}
    best_score: Dict[str, float] = {b: float("-inf") for b in proto.BRANCHES}
    best_step_of: Dict[str, Optional[int]] = {b: None for b in proto.BRANCHES}
    best_states: Dict[str, Optional[Dict[str, torch.Tensor]]] = {b: None for b in proto.BRANCHES}
    rng_digest_per_step: List[str] = []
    consumed_fingerprints: List[str] = []
    arch_fingerprint_rows: List[Dict[str, object]] = []
    epoch_end_f_macro_auc_vs_source: List[Dict[str, object]] = []
    v0_violations: List[Dict[str, object]] = []
    v2_violations: List[Dict[str, object]] = []
    review_flagged = False
    aborted = bool(step0_v0_fail)
    abort_reason = "v0_anchor_macro_auc_above_threshold" if step0_v0_fail else ""
    rng_digest_continuation_before = ""
    rng_digest_continuation_after: Optional[str] = None
    rng_within_epoch_invariant = True
    global_step = 0
    if not aborted:
        # --------------------------------------------------------------
        # Streaming RNG replay (design §4.3.1/§4.3.6): reproduce the
        # source's global-RNG draw chain exactly.  Per epoch:
        #   iter(train_loader) [1 draw] -> first next [1 draw] ->
        #   remaining nexts [0 draws] -> iter(train_records_loader) [1 draw].
        # Pre-consume 3*(anchor+1) draws to land at the start of epoch
        # anchor+1.
        # --------------------------------------------------------------
        setup_seed(int(seed))
        _ = build_model(bundle.features, DOMAIN_NUM, TASK_NUM, "m3oe",
                         expert_num=4, fcn_dims=(512, 256, 256, 64), seed=int(seed))
        train_loader = DataLoader(
            TorchDataset(bundle.train_x, bundle.train_y),
            batch_size=40960, shuffle=True, num_workers=0, drop_last=False)
        train_records_loader = DataLoader(
            TorchDataset(bundle.train_x, bundle.train_y),
            batch_size=40960, shuffle=False, num_workers=0, drop_last=False)
        for _ in range(3 * (int(anchor_epoch) + 1)):
            _ = iter(train_loader)
        rng_digest_continuation_before = rng_state_digest()
        for e_offset in range(int(h_max_epochs)):
            if aborted:
                break
            e = int(anchor_epoch) + 1 + int(e_offset)
            train_iterator = iter(train_loader)  # 1 draw (epoch main iter)
            for s in range(int(bpe)):
                x_batch, y_batch = next(train_iterator)  # first next: 1 draw
                global_step += 1
                fp = batch_fingerprint(x_batch, y_batch)
                consumed_fingerprints.append(fp)
                expected_fp = source_main_fp.get((e, s), "")
                if fp != expected_fp:
                    aborted = True
                    abort_reason = "v4_main_batch_fingerprint_mismatch_epoch_{}_step_{}".format(e, s)
                    global_step -= 1
                    consumed_fingerprints.pop()
                    rng_digest_per_step = rng_digest_per_step[:global_step]
                    break
                step_digest = rng_state_digest()
                rng_digest_per_step.append(step_digest)
                domain_ids_np = x_batch[:, -1].detach().cpu().numpy().astype(np.int64)
                domain_counts = [int((domain_ids_np == d).sum()) for d in range(DOMAIN_NUM)]
                batch_rows_actual = int(x_batch.shape[0])
                weights = {cell_index(task, d, DOMAIN_NUM):
                             float(domain_counts[d]) / float(2 * batch_rows_actual)
                           for task in range(TASK_NUM) for d in range(DOMAIN_NUM)}
                batch = {"x": x_batch, "y": y_batch, "fingerprint": fp,
                          "domain_counts": domain_counts, "batch_size": batch_rows_actual,
                          "w": weights, "step": int(s)}
                sample_val_grad = int(global_step) in valgrad_steps
                f_post_shared: Optional[Dict[str, torch.Tensor]] = None
                for branch_name in proto.BRANCHES:
                    b_model = branches[branch_name]["model"]
                    b_opt = branches[branch_name]["opt"]
                    diag = diagnostics_at_theta(b_model, branch_name, int(global_step),
                                                  batch, bundle, shared_names, private_names,
                                                  slices, device,
                                                  sample_val_grad=sample_val_grad)
                    step_met = controlled_continuation_step(
                        b_model, b_opt, branch_name, batch, shared_names, private_names,
                        device, target_local,
                        theta_before_F=(f_post_shared if branch_name in ("T", "DH") else None))
                    if branch_name == "F":
                        f_post_shared = {
                            name: p.detach().to(device="cpu", dtype=torch.float64).clone()
                            for name, p in b_model.named_parameters() if name in shared_set}
                    # V0 (global step 1, hard) / V2 (later steps, review
                    # tripwire): F branch vs the S8-BASE source trajectory.
                    validity_l_full_diff = float("nan")
                    validity_delta_rel = float("nan")
                    if branch_name == "F":
                        source_row = source_step_rows[(e, s)]
                        source_l_full = _safe_float(source_row.get("L_full", ""))
                        if source_l_full is not None:
                            validity_l_full_diff = abs(float(diag["L_full"]) - source_l_full)
                        source_delta = _safe_float(source_row.get("delta_observed_step_norm", ""))
                        f_delta = float(step_met["delta_full_norm"])
                        if source_delta is not None and source_delta > 0:
                            validity_delta_rel = abs(f_delta - source_delta) / source_delta
                        if global_step == 1:
                            v0_l_full_fail = bool(
                                (not math.isfinite(validity_l_full_diff))
                                or validity_l_full_diff > proto.V0_L_FULL_DIFF_ABS)
                            v0_delta_rel_fail = bool(
                                (not math.isfinite(validity_delta_rel))
                                or validity_delta_rel > proto.V0_DELTA_NORM_REL_DIFF)
                            v0_violations.append({
                                "step": int(global_step),
                                "L_full_abs_diff": float(validity_l_full_diff),
                                "delta_norm_rel_diff": (float(validity_delta_rel)
                                                          if math.isfinite(validity_delta_rel) else None),
                                "v0_l_full_fail": bool(v0_l_full_fail),
                                "v0_delta_rel_fail": bool(v0_delta_rel_fail),
                            })
                            if v0_l_full_fail or v0_delta_rel_fail:
                                aborted = True
                                abort_reason = "v0_l_full_or_delta_rel_above_threshold_step_1"
                                break
                        else:
                            v2_review = bool(
                                math.isfinite(validity_l_full_diff)
                                and validity_l_full_diff > proto.V2_REVIEW_TRIPWIRE_L_FULL_DIFF)
                            if v2_review:
                                review_flagged = True
                            v2_violations.append({
                                "step": int(global_step),
                                "L_full_abs_diff": float(validity_l_full_diff),
                                "delta_norm_rel_diff": (float(validity_delta_rel)
                                                          if math.isfinite(validity_delta_rel) else None),
                                "review_flagged": bool(v2_review),
                            })
                    # Post-step validation (the step-k validation row
                    # records the state AFTER step k).
                    records, raw_mean, raw_count = _full_validation_metrics(
                        b_model, bundle, device)
                    aggregates = _aggregate_validation(records, raw_mean, raw_count,
                                                        target_local, bundle.domain_shares)
                    metrics_by_step[branch_name][int(global_step)] = {
                        "macro_auc": float(aggregates["macro_auc"]),
                        "target_domain_auc": float(aggregates["target_domain_auc"]),
                        "macro_logloss": float(aggregates["macro_logloss"]),
                        "worst_cell_auc": float(min(r["auc"] for r in records.values())),
                    }
                    score = float(aggregates["macro_auc"])
                    if score > best_score[branch_name]:
                        best_score[branch_name] = score
                        best_step_of[branch_name] = int(global_step)
                        best_states[branch_name] = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in b_model.state_dict().items()}
                    _emit_validation_rows(val_rows, dataset, int(seed), anchor_class,
                                           int(anchor_epoch), branch_name, int(global_step),
                                           "post_step", records, raw_mean, aggregates)
                    # eff-alignment + geometry rows at sample steps.
                    eff_values = {"eff_align_full": float("nan"),
                                    "eff_align_target": float("nan"),
                                    "eff_dot_full": float("nan"),
                                    "eff_dot_target": float("nan")}
                    if sample_val_grad and "val_vecs_fp64" in diag:
                        eff_values = _eff_alignment(step_met["_delta_applied_vec_fp64"],
                                                      diag["g_val_full_fp64"],
                                                      diag["g_val_target_fp64"])
                        geometry_rows.extend(_geometry_rows_for_step(
                            dataset, int(seed), anchor_class, int(anchor_epoch),
                            branch_name, int(global_step), diag))
                    step_rows.append({
                        "stage": "S8", "dataset": dataset, "seed": int(seed),
                        "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
                        "branch": branch_name,
                        "epoch": int(e),
                        "step": int(global_step),
                        "batch_fingerprint": str(fp),
                        "domain_counts": _json_counts(diag["domain_counts"]),
                        "L_full": float(diag["L_full"]),
                        "L_target": float(diag["L_target"]),
                        "L_non": float(diag["L_non"]),
                        "g_full_norm": float(diag["g_full_norm"]),
                        "g_target_norm": float(diag["g_target_norm"]),
                        "cosine_gf_gt": float(diag["cos_gf_gt"]),
                        "m_norm_persistent": float(step_met["m_norm_persistent"]),
                        "v_norm_persistent": float(step_met["v_norm_persistent"]),
                        "delta_full_norm": float(step_met["delta_full_norm"]),
                        "delta_target_norm": float(step_met["delta_target_norm"]),
                        "delta_applied_norm": float(step_met["delta_applied_norm"]),
                        "delta_cosine": float(step_met["delta_cosine"]),
                        "private_delta_norm": float(step_met["private_delta_norm"]),
                        "applied_formula_residual": float(step_met["applied_formula_residual"]),
                        "private_full_residual": float(step_met["private_full_residual"]),
                        "theta_distance_to_F_shared": float(step_met["theta_distance_to_F_shared"]),
                        "eff_align_full": float(eff_values["eff_align_full"]),
                        "eff_align_target": float(eff_values["eff_align_target"]),
                        "eff_dot_full": float(eff_values["eff_dot_full"]),
                        "eff_dot_target": float(eff_values["eff_dot_target"]),
                        "rng_digest": str(step_digest),
                        "validity_L_full_diff": float(validity_l_full_diff),
                        "validity_delta_norm_rel_diff": float(validity_delta_rel),
                    })
                if aborted:
                    break
                if rng_state_digest() != step_digest:
                    rng_within_epoch_invariant = False
                    raise AssertionError(
                        "S8 RNG consumed within continuation step {}".format(global_step))
            if aborted:
                break
            # Epoch end: mirror the source's train-records iterator draw
            # (1 draw), then the arch batch + per-branch arch update (all
            # inside preserve_rng_state / RNG-invariant), then per-branch
            # post-arch validation.
            _ = iter(train_records_loader)
            x_arch, y_arch, arch_fp = arch_batch_for_epoch(bundle, dataset, int(seed), e)
            expected_arch_fp = source_arch_fp.get(e, "")
            arch_fingerprint_rows.append({
                "epoch": int(e), "observed": str(arch_fp),
                "expected": str(expected_arch_fp),
                "match": bool(arch_fp == expected_arch_fp)})
            if arch_fp != expected_arch_fp:
                aborted = True
                abort_reason = "v4_arch_batch_fingerprint_mismatch_epoch_{}".format(e)
                break
            for branch_name in proto.BRANCHES:
                _arch_controlled_step(branches[branch_name]["model"],
                                       branches[branch_name]["arch_opt"],
                                       branch_name, x_arch, y_arch,
                                       arch_names, target_local, device)
            for branch_name in proto.BRANCHES:
                b_model = branches[branch_name]["model"]
                records, raw_mean, raw_count = _full_validation_metrics(b_model, bundle, device)
                aggregates = _aggregate_validation(records, raw_mean, raw_count,
                                                    target_local, bundle.domain_shares)
                _emit_validation_rows(val_rows, dataset, int(seed), anchor_class,
                                       int(anchor_epoch), branch_name, int(global_step),
                                       "epoch_end_post_arch", records, raw_mean, aggregates)
                if branch_name == "F":
                    base_row = base_metrics_epoch.get(int(e), {})
                    base_macro_auc = _safe_float(base_row.get("validation_macro_auc", ""))
                    if base_macro_auc is not None:
                        diff = abs(float(aggregates["macro_auc"]) - base_macro_auc)
                        flagged = bool(diff > proto.V2_REVIEW_TRIPWIRE_MACRO_AUC)
                        if flagged:
                            review_flagged = True
                        epoch_end_f_macro_auc_vs_source.append({
                            "epoch": int(e),
                            "f_post_arch_macro_auc": float(aggregates["macro_auc"]),
                            "base_macro_auc": float(base_macro_auc),
                            "abs_diff": float(diff),
                            "review_flagged": bool(flagged),
                        })
                    else:
                        epoch_end_f_macro_auc_vs_source.append({
                            "epoch": int(e),
                            "f_post_arch_macro_auc": float(aggregates["macro_auc"]),
                            "base_macro_auc": None,
                            "abs_diff": None,
                            "review_flagged": True,
                        })
                        review_flagged = True
        rng_digest_continuation_after = rng_state_digest()
    replay_audit = {
        "mode": "streaming",
        "pre_consumed_iters": int(3 * (int(anchor_epoch) + 1)),
        "materialized_fingerprints": list(consumed_fingerprints),
        "consumed_fingerprints": list(consumed_fingerprints),
    }
    payload = _write_outputs(
        out_dir_abs, dataset, int(seed), anchor_class, int(anchor_epoch), base_run_dir,
        anchor_audit, replay_audit,
        step_rows, geometry_rows, val_rows,
        metrics_by_step, step0_aggregates,
        best_step_of, best_states,
        int(bpe), int(h_max_steps), int(h_max_epochs), horizons, valgrad_steps, started,
        aborted=aborted, abort_reason=abort_reason,
        v0_violations=v0_violations, v2_violations=v2_violations,
        review_flagged=review_flagged,
        v3_per_pair=v3_per_pair,
        rng_digest_continuation_before=rng_digest_continuation_before,
        rng_digest_per_step=rng_digest_per_step,
        rng_digest_continuation_after=rng_digest_continuation_after,
        rng_within_epoch_invariant=bool(rng_within_epoch_invariant),
        arch_fingerprint_rows=arch_fingerprint_rows,
        v0_step0_macro_auc=v0_step0_macro_auc,
        f_original_best_macro_auc=f_original_best_macro_auc,
        f_original_selected_epoch=f_original_selected_epoch,
        f_original_crosscheck=f_original_crosscheck,
        epoch_end_f_macro_auc_vs_source=epoch_end_f_macro_auc_vs_source)
    return payload


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def _emit_validation_rows(val_rows: List[Dict[str, object]], dataset: str, seed: int,
                           anchor_class: str, anchor_epoch: int, branch_name: str,
                           step: int, kind: str,
                           records: Mapping[int, Mapping[str, float]],
                           raw_mean: Mapping[int, float],
                           aggregates: Mapping[str, float]) -> None:
    """Append the 9 validation rows (6 cells + 3 scopes) for one
    (branch, step, kind) point."""
    for cell_key in sorted(records):
        d = cell_key % DOMAIN_NUM
        t = cell_key // DOMAIN_NUM
        val_rows.append({
            "stage": "S8", "dataset": dataset, "seed": int(seed),
            "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
            "branch": branch_name, "step": int(step), "kind": str(kind),
            "scope": "cell", "domain": d, "task": t, "cell": cell_key,
            "val_bce_raw": float(raw_mean[cell_key]),
            "auc": float(records[cell_key]["auc"]),
            "logloss": float(records[cell_key]["logloss"]),
        })
    for scope_name in ("macro", "target_domain", "non_target_weighted"):
        if scope_name == "macro":
            auc_v, ll_v = aggregates["macro_auc"], aggregates["macro_logloss"]
            loss_v = float("nan")
        elif scope_name == "target_domain":
            auc_v, ll_v = (aggregates["target_domain_auc"],
                            aggregates["target_domain_logloss"])
            loss_v = float(aggregates["target_loss"])
        else:
            auc_v, ll_v = (aggregates["non_target_weighted_auc"],
                            aggregates["non_target_weighted_logloss"])
            loss_v = float(aggregates["non_target_loss"])
        val_rows.append({
            "stage": "S8", "dataset": dataset, "seed": int(seed),
            "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
            "branch": branch_name, "step": int(step), "kind": str(kind),
            "scope": scope_name, "domain": -1, "task": -1, "cell": -1,
            "val_bce_raw": float(loss_v),
            "auc": float(auc_v),
            "logloss": float(ll_v),
        })


def _geometry_rows_for_step(dataset: str, seed: int, anchor_class: str, anchor_epoch: int,
                             branch_name: str, global_step: int,
                             diag: Mapping[str, object]) -> List[Dict[str, object]]:
    """The 36 cell-pair x 8 module geometry rows for one (branch, sample
    step).  Requires ``diag`` to carry ``val_vecs_fp64``."""
    rows: List[Dict[str, object]] = []
    full_vec_fp64 = _to_fp64_flat([diag["g_full_vec"]])
    for s in range(CELL_NUM):
        s_d, s_t = s % DOMAIN_NUM, s // DOMAIN_NUM
        s_vec_full = _to_fp64_flat([diag["cell_grads"][s]])
        for t_cell in range(CELL_NUM):
            t_d, t_t = t_cell % DOMAIN_NUM, t_cell // DOMAIN_NUM
            t_cell_vec_full = _to_fp64_flat([diag["cell_grads"][t_cell]])
            val_vec_full = diag["val_vecs_fp64"][t_cell]
            if s_d == t_d and s_t == t_t:
                pc = "self"
            elif s_d == t_d:
                pc = "same_domain_cross_task"
            elif s_t == t_t:
                pc = "cross_domain_same_task"
            else:
                pc = "cross_domain_cross_task"
            for module in MODULES:
                if module == "ALL":
                    s_slice = s_vec_full
                    t_slice = t_cell_vec_full
                    g_full_slice = full_vec_fp64
                    val_slice = val_vec_full
                else:
                    start, end = diag["slices"][module]
                    s_slice = s_vec_full[start:end].clone()
                    t_slice = t_cell_vec_full[start:end].clone()
                    g_full_slice = full_vec_fp64[start:end].clone()
                    val_slice = val_vec_full[start:end].clone()
                norm_s = _norm_fp64(s_slice)
                cs_st = _cosine_fp64(s_slice, t_slice)
                if not math.isfinite(cs_st):
                    cs_st = 0.0
                cs_st_clamped = max(-1.0, min(1.0, cs_st))
                align_s = _cosine_fp64(s_slice, g_full_slice)
                if not math.isfinite(align_s):
                    align_s = 0.0
                u_t = _dot_fp64(g_full_slice, val_slice)
                h_st = float(diag["weights"][s]) * _dot_fp64(s_slice, val_slice)
                rows.append({
                    "stage": "S8", "dataset": dataset, "seed": int(seed),
                    "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
                    "branch": branch_name, "step": int(global_step),
                    "source_domain": s_d, "source_task": s_t,
                    "target_domain": t_d, "target_task": t_t,
                    "pair_class": pc, "shared_module": module,
                    "gradient_norm": float(norm_s),
                    "pairwise_cosine": float(cs_st_clamped),
                    "sample_alignment": float(align_s),
                    "validation_dot_product": float(u_t),
                    "weighted_source_target_contribution": float(h_st),
                    "validation_gradient_norm": float(_norm_fp64(val_slice)),
                    "source_weight": float(diag["weights"][s]),
                })
    return rows


def _write_outputs(out_dir_abs: str, dataset: str, seed: int, anchor_class: str,
                    anchor_epoch: int, base_run_dir: str,
                    anchor_audit: Dict[str, object], replay_audit: Dict[str, object],
                    step_rows: List[Dict[str, object]],
                    geometry_rows: List[Dict[str, object]],
                    val_rows: List[Dict[str, object]],
                    metrics_by_step: Dict[str, Dict[int, Dict[str, float]]],
                    step0_aggregates: Dict[str, Dict[str, float]],
                    best_step_of: Dict[str, Optional[int]],
                    best_states: Dict[str, Optional[Dict[str, torch.Tensor]]],
                    bpe: int, h_max_steps: int, h_max_epochs: int,
                    horizons: Dict[str, int],
                    valgrad_steps: Set[int], started: float,
                    aborted: bool, abort_reason: str,
                    v0_violations: List[Dict[str, object]],
                    v2_violations: List[Dict[str, object]],
                    review_flagged: bool,
                    v3_per_pair: Dict[str, Dict[str, object]],
                    rng_digest_continuation_before: str,
                    rng_digest_per_step: List[str],
                    rng_digest_continuation_after: Optional[str],
                    rng_within_epoch_invariant: bool,
                    arch_fingerprint_rows: List[Dict[str, object]],
                    v0_step0_macro_auc: Dict[str, object],
                    f_original_best_macro_auc: Optional[float],
                    f_original_selected_epoch: Optional[int],
                    f_original_crosscheck: Dict[str, object],
                    epoch_end_f_macro_auc_vs_source: List[Dict[str, object]]) -> Dict[str, object]:
    # Per-branch best table rows (nested-horizon prefixes over the post-step
    # macro AUC trajectory; anchor excluded from ``best``; strict > with
    # earliest tie).
    best_rows: List[Dict[str, object]] = []
    for branch_name in proto.BRANCHES:
        branch_metrics = metrics_by_step.get(branch_name, {})
        present_steps = sorted(branch_metrics.keys())
        anchor_macro_auc = float(step0_aggregates[branch_name]["macro_auc"])
        for horizon_name, horizon_step_count in horizons.items():
            limit = min(int(horizon_step_count), present_steps[-1] if present_steps else 0)
            best_step: Optional[int] = None
            best_score_h = float("-inf")
            for s in present_steps:
                if s > limit:
                    break
                score = float(branch_metrics[s]["macro_auc"])
                if score > best_score_h:
                    best_score_h = score
                    best_step = int(s)
            if best_step is not None:
                best_metrics = branch_metrics[best_step]
                best_macro = float(best_metrics["macro_auc"])
                best_target = float(best_metrics["target_domain_auc"])
                best_logloss = float(best_metrics["macro_logloss"])
                best_worst_cell = float(best_metrics["worst_cell_auc"])
            else:
                best_macro = float(anchor_macro_auc)
                best_target = float("nan")
                best_logloss = float("nan")
                best_worst_cell = float("nan")
            if limit >= 1:
                final_macro = float(branch_metrics[limit]["macro_auc"])
            else:
                final_macro = float(anchor_macro_auc)
            best_incl_anchor = max(float(anchor_macro_auc), float(best_macro))
            best_rows.append({
                "stage": "S8", "dataset": dataset, "seed": int(seed),
                "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
                "branch": branch_name,
                "horizon": str(horizon_name),
                "horizon_steps": int(horizon_step_count),
                "best_step": (int(best_step) if best_step is not None else 0),
                "best_macro_auc": float(best_macro),
                "best_target_auc": float(best_target),
                "best_macro_logloss": float(best_logloss),
                "best_worst_cell_auc": float(best_worst_cell),
                "best_incl_anchor_macro_auc": float(best_incl_anchor),
                "anchor_macro_auc": float(anchor_macro_auc),
                "final_macro_auc": float(final_macro),
                "f_original_best_macro_auc": (float(f_original_best_macro_auc)
                                                if f_original_best_macro_auc is not None
                                                else float("nan")),
                "f_original_selected_epoch": (int(f_original_selected_epoch)
                                                if f_original_selected_epoch is not None else -1),
            })
        best_state = best_states.get(branch_name)
        if best_state is not None:
            torch.save(best_state,
                        os.path.join(out_dir_abs,
                                      "{}{}.pth".format(proto.BEST_PTH_PREFIX, branch_name)))
    write_rows_csv(os.path.join(out_dir_abs, proto.STEPS_CSV_NAME), step_rows, STEP_COLUMNS)
    write_rows_csv(os.path.join(out_dir_abs, proto.GEOMETRY_CSV_NAME), geometry_rows, GEOMETRY_COLUMNS)
    write_rows_csv(os.path.join(out_dir_abs, proto.VALIDATION_CSV_NAME), val_rows, VALIDATION_COLUMNS)
    write_rows_csv(os.path.join(out_dir_abs, proto.BEST_CSV_NAME), best_rows, BEST_COLUMNS)
    payload = {
        "stage": "S8",
        "status": "ABORTED" if aborted else "OK",
        "abort_reason": abort_reason,
        "dataset": dataset, "seed": int(seed), "anchor_class": anchor_class,
        "anchor_epoch": int(anchor_epoch),
        "key": proto.key_cont(dataset, int(seed), anchor_class),
        "base_run_dir": base_run_dir,
        "h_max_epochs": int(h_max_epochs),
        "h_max_steps": int(h_max_steps),
        "horizons": dict(horizons),
        "batches_per_epoch": int(bpe),
        "valgrad_sample_steps": sorted(int(s) for s in valgrad_steps),
        "anchor_reconstruction": anchor_audit,
        "replay_audit": replay_audit,
        "v3_per_pair": {k: {"all_equal": v.get("all_equal", False),
                              "rows": v.get("rows", [])}
                          for k, v in v3_per_pair.items()},
        "v0_step0_macro_auc": dict(v0_step0_macro_auc),
        "v0_violations": v0_violations,
        "v2_violations": v2_violations,
        "v2_review_flagged": bool(review_flagged),
        "arch_fingerprint_rows": list(arch_fingerprint_rows),
        "epoch_end_f_macro_auc_vs_source": list(epoch_end_f_macro_auc_vs_source),
        "f_original_best_macro_auc": f_original_best_macro_auc,
        "f_original_selected_epoch": f_original_selected_epoch,
        "f_original_crosscheck": dict(f_original_crosscheck),
        "rng_digest_before": str(rng_digest_continuation_before),
        "rng_digest_after": str(rng_digest_continuation_after or ""),
        "rng_digest_per_step": list(rng_digest_per_step),
        "rng_within_epoch_invariant": bool(rng_within_epoch_invariant),
        "rng_state_unchanged": bool(str(rng_digest_continuation_before)
                                      == str(rng_digest_continuation_after or "")),
        "wall_seconds": float(time.time() - started),
        "model_source_key": MODEL_SOURCE_KEY,
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "base_full_state_dump_sha256": str(anchor_audit.get("full_state_dump_sha256", "")),
    }
    write_json(os.path.join(out_dir_abs, proto.AUDIT_JSON_NAME), payload)
    write_json(os.path.join(out_dir_abs, proto.PROCESS_INTERVAL_NAME), {
        "started": float(started), "finished": float(time.time()),
        "wall_seconds": float(time.time() - started),
        "dataset": dataset, "seed": int(seed), "anchor_class": anchor_class,
        "anchor_epoch": int(anchor_epoch), "status": payload["status"],
        "h_max_epochs": int(h_max_epochs),
    })
    return payload
# ---------------------------------------------------------------------------
# Argparser + main
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S8 three-branch continuation")
    parser.add_argument("--dataset", choices=("kuairand", "movielens"), required=True)
    parser.add_argument("--seed", type=int, choices=tuple(int(s) for s in S4_SEEDS), required=True)
    parser.add_argument("--anchor-class", choices=tuple(proto.ANCHOR_CLASSES), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--h-max-epochs", type=int,
                        choices=(proto.H_MAX_EPOCHS_CORE, proto.H_MAX_EPOCHS_EXTENDED),
                        default=proto.H_MAX_EPOCHS_CORE)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--smoke-restore", action="store_true",
                        help="Smoke: anchor reconstruction + V3 bit-equality clone only.")
    parser.add_argument("--smoke-replay", action="store_true",
                        help="Smoke: replay 5 main + 1 arch batch; verify fingerprints.")
    parser.add_argument("--smoke-onestep", action="store_true",
                        help="Smoke: 1 batch through F/T branches; report residuals.")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    try:
        result = run_continuation(
            dataset=args.dataset, seed=int(args.seed), anchor_class=args.anchor_class,
            out_dir=args.out_dir, h_max_epochs=int(args.h_max_epochs),
            dataset_path=args.dataset_path or "",
            smoke_restore=bool(args.smoke_restore),
            smoke_replay=bool(args.smoke_replay),
            smoke_onestep=bool(args.smoke_onestep),
        )
        if isinstance(result, dict):
            if "smoke_restore_ok" in result:
                print("[stage_S8] smoke-restore dataset={} seed={} anchor={}/{} ok={} shared_step={}/{} private_coverage={} v3_equal={}".format(
                    args.dataset, int(args.seed), args.anchor_class,
                    int(result.get("anchor_epoch", -1)),
                    result.get("smoke_restore_ok"),
                    result.get("shared_step_observed"),
                    result.get("shared_step_expected"),
                    result.get("private_state_coverage"),
                    result.get("v3_state_bit_equality"),
                ), flush=True)
                return 0 if result.get("smoke_restore_ok") else 1
            if "smoke_replay_ok" in result:
                print("[stage_S8] smoke-replay dataset={} seed={} anchor={}/{} ok={} arch_match={}".format(
                    args.dataset, int(args.seed), args.anchor_class,
                    int(result.get("anchor_epoch", -1)),
                    result.get("smoke_replay_ok"),
                    result.get("arch_fingerprint_match"),
                ), flush=True)
                return 0 if result.get("smoke_replay_ok") else 1
            if "smoke_onestep_ok" in result:
                print("[stage_S8] smoke-onestep dataset={} seed={} anchor={}/{} F_resid={:.3e} T_resid={:.3e} fingerprint_match={} v0_fail={}".format(
                    args.dataset, int(args.seed), args.anchor_class,
                    int(result.get("anchor_epoch", -1)),
                    float(result.get("F_applied_formula_residual", 0.0)),
                    float(result.get("T_applied_formula_residual", 0.0)),
                    bool(result.get("fingerprint_match")),
                    bool(result.get("v0_step0_fail")),
                ), flush=True)
                return 0 if bool(result.get("fingerprint_match")) \
                           and not bool(result.get("v0_step0_fail")) else 1
        print("[stage_S8] OK dataset={} seed={} anchor={}/{} key={} wall={:.2f}s".format(
            args.dataset, int(args.seed), args.anchor_class,
            int(result.get("anchor_epoch", -1)),
            result.get("key", ""), float(result.get("wall_seconds", 0.0))), flush=True)
        return 0
    except Exception as exc:
        import traceback
        sys.stderr.write("[stage_S8] FAILED dataset={} seed={} anchor={} exception={}\n".format(
            args.dataset, int(args.seed), args.anchor_class, exc))
        sys.stderr.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
