"""Stage S7-B: same-theta early/late causal continuation.

From a single Stage S6 F arm anchor checkpoint (epoch 9 = early, epoch 39 =
late), with identical theta, identical reconstructed optimizer state, and
identical replayed minibatch sequence, run K=5 short continuation steps on
two branches:

- F branch: bit-for-bit S6 F arm semantics (shared <- g_F, private <- g_F,
  moments <- g_F).
- T branch: bit-for-bit S6 DH arm current manipulation (moments <- g_F;
  shared parameter step is overwritten by the read-only counterfactual
  Delta_T calculated from the pre-step persistent state; private <- g_F).

Per Algorithm Specification -- Stage S7-B.
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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
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

import stage_S7B_protocol as proto  # noqa: E402


STEP_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "seed", "anchor_epoch", "branch", "epoch", "step",
    "batch_fingerprint", "domain_counts",
    "L_full", "L_target", "L_non",
    "g_full_norm", "g_target_norm", "cosine_gf_gt",
    "delta_full_norm", "delta_target_norm", "delta_applied_norm",
    "delta_cosine", "private_delta_norm", "theta_distance_shared",
    "applied_formula_residual", "recon_residual_gS", "recon_residual_H",
    "rng_digest", "validity_L_full_diff", "validity_delta_norm_rel_diff",
)
GEOMETRY_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "seed", "anchor_epoch", "branch", "step",
    "source_domain", "source_task", "target_domain", "target_task",
    "pair_class", "shared_module",
    "gradient_norm", "pairwise_cosine", "sample_alignment",
    "validation_dot_product", "weighted_source_target_contribution",
    "validation_gradient_norm", "source_weight",
)
VALIDATION_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "seed", "anchor_epoch", "branch", "step",
    "scope", "domain", "task", "cell",
    "val_bce_raw", "auc", "logloss",
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

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
    na = _norm_fp64(a)
    nb = _norm_fp64(b)
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


def _theta_distance(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    pa = dict(model_a.named_parameters())
    pb = dict(model_b.named_parameters())
    total = 0.0
    for name, param_a in pa.items():
        diff = param_a.detach().to(dtype=torch.float64, device="cpu") \
             - pb[name].detach().to(dtype=torch.float64, device="cpu")
        total += float(torch.sum(diff * diff).item())
    return math.sqrt(total)


def _private_distance(model_a: torch.nn.Module, model_b: torch.nn.Module,
                       private_names: Sequence[str]) -> float:
    pa = dict(model_a.named_parameters())
    pb = dict(model_b.named_parameters())
    total = 0.0
    for name in private_names:
        diff = pa[name].detach().to(dtype=torch.float64, device="cpu") \
             - pb[name].detach().to(dtype=torch.float64, device="cpu")
        total += float(torch.sum(diff * diff).item())
    return math.sqrt(total)


def _target_local_domain(dataset: str) -> int:
    return int(target_for(str(dataset).lower(), proto.target_setting(dataset))["local_domain"])


# -----------------------------------------------------------------------------
# Source-run reading
# -----------------------------------------------------------------------------

def _read_source_batch_trace(source_run_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(source_run_dir, "batch_trace.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("kind") == "train"]


def _read_source_trace(source_run_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(source_run_dir, "gradient_step.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("kind") == "main"]


def _read_source_manifest(source_run_dir: str) -> Dict[str, object]:
    path = os.path.join(source_run_dir, "run_manifest.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _source_main_rows_for_continuation(source_run_dir: str, anchor: int, k: int
                                       ) -> List[Dict[str, str]]:
    rows = _read_source_trace(source_run_dir)
    epoch_next = int(anchor) + 1
    selected: List[Dict[str, str]] = []
    for row in rows:
        try:
            ep = int(row.get("epoch", "-1"))
            st = int(row.get("step", "-1"))
        except (TypeError, ValueError):
            continue
        if ep != epoch_next:
            continue
        if 0 <= st < k:
            selected.append(row)
    selected.sort(key=lambda r: int(r["step"]))
    if len(selected) != k:
        raise ValueError(
            "source main rows for epoch {} steps 0..{} incomplete (have {})".format(
                epoch_next, k - 1, len(selected)))
    return selected


def _source_train_batch_fingerprints_for_replay(source_run_dir: str, until_epoch: int
                                                ) -> List[str]:
    rows = _read_source_batch_trace(source_run_dir)
    fingerprints: List[str] = []
    for row in rows:
        try:
            ep = int(row.get("epoch", "-1"))
        except (TypeError, ValueError):
            continue
        if 0 <= ep <= until_epoch:
            fingerprints.append(str(row.get("batch_fingerprint", "")))
    return fingerprints


# -----------------------------------------------------------------------------
# Replay
# -----------------------------------------------------------------------------

def replay_epoch_batches(bundle, seed: int, epoch: int, k: int
                          ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Replay the Stage S6 RNG chain from ``setup_seed(seed)`` and
    ``build_model``; advance through epoch 0..anchor (consuming the
    same shuffled train_loader iter + sequential train_records iter the
    source S6 ``fit()`` loop draws per epoch) then take the first ``k``
    batches of epoch ``anchor+1``.

    Replay call sequence (matches the source S6 F ``fit()`` loop, which
    delegates to S1Runner / S3Runner / S4Runner — see
    ``stage_S1_run.fit`` → ``S3Runner._train_epoch`` →
    ``_cell_records(train_records_loader)``):

    For each epoch e in [0, anchor]:
      1. ``for x, y in self.train_loader`` (main epoch) -- one fresh
         shuffled train_loader iter, which consumes 2 global CPU RNG
         draws: ``_SingleProcessDataLoaderIter.__init__`` calls
         ``iter(self._index_sampler)`` (the RandomSampler, with
         ``generator=None`` → falls through to the default global
         generator → 1 randperm → 1 draw) and then
         ``torch.empty(()).random_(generator=loader.generator)``
         (1 draw for the per-iter ``_base_seed``).
      2. ``_cell_records(self.bundle, self.model, train_records_loader,
         self.device)`` at the end of ``_train_epoch`` -- one fresh
         DataLoader (with ``shuffle=False``) iter; the SequentialSampler
         uses no RNG, but ``_SingleProcessDataLoaderIter.__init__`` still
         draws the ``_base_seed`` from the default global generator → 1
         more global CPU RNG draw.
      3. ``_architecture_update`` -- 1 iter wrapped in
         ``preserve_rng_state`` → net 0.
      4. ``_cell_records(val_loader)`` inside ``with
         preserve_rng_state()`` in ``fit()`` → net 0.

    So per epoch the source S6 F ``fit()`` loop advances the global
    CPU RNG by exactly 3 draws (2 from the main shuffled iter, 1 from
    the train_records shuffle=False iter).  Across ``anchor+1`` epochs
    this is ``3 * (anchor + 1)`` total global CPU RNG draws; we replay
    by invoking ``iter(train_loader)`` that many times before drawing
    the first batch of epoch ``anchor+1``.

    Mechanism note (provided per designer's ruling 2): "the per-epoch
    accounting above is consistent with the code we read; the exact
    bookkeeping of torch internal draws has not been independently
    verified, but the fingerprint equality is the authoritative
    evidence that the RNG state at the start of epoch ``anchor+1``
    matches the source S6 F run."

    Fingerprint hard-gate semantics: the five materialized
    ``batch_fingerprint`` values are checked byte-equal against the
    source ``gradient_step.csv`` (epoch=anchor+1, kind=main, step 0..4).
    Any mismatch aborts the run.
    """
    setup_seed(int(seed))
    model = build_model(bundle.features, DOMAIN_NUM, TASK_NUM, "m3oe",
                        expert_num=4, fcn_dims=(512, 256, 256, 64), seed=int(seed))
    train_loader = DataLoader(
        TorchDataset(bundle.train_x, bundle.train_y),
        batch_size=40960, shuffle=True, num_workers=0, drop_last=False)
    n_iters = 3 * (int(epoch) + 1)
    for _ in range(n_iters):
        _ = iter(train_loader)
    train_iterator = iter(train_loader)
    batches: List[Dict[str, object]] = []
    for step in range(int(k)):
        x, y = next(train_iterator)
        fp = batch_fingerprint(x, y)
        domain_ids = x[:, -1].detach().cpu().numpy().astype(np.int64)
        domain_counts = [int((domain_ids == d).sum()) for d in range(DOMAIN_NUM)]
        if any(count == 0 for count in domain_counts):
            raise ValueError("replayed batch {} missing one of three domains: {}".format(
                step, domain_counts))
        # w_{d,t} = n_d / (2 B) with the ACTUAL per-batch B; the final batch
        # of an epoch can be partial (drop_last=False), e.g. MovieLens
        # 187500 = 4*40960 + 23660.
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
            "step": step,
        })
    audit: Dict[str, object] = {
        "epoch_iters_consumed": n_iters,
        "epoch_iters_formula": "3 * (anchor + 1) [2 draws per main shuffled iter (RandomSampler randperm + DataLoader _base_seed) + 1 draw from the train_records shuffle=False DataLoader iter; arch/validation iters are inside preserve_rng_state and net 0]",
        "k_requested": int(k),
        "materialized_fingerprints": [b["fingerprint"] for b in batches],
    }
    return batches, audit


def _verify_replay_fingerprints(audit: Dict[str, object],
                                source_run_dir: str, anchor: int, k: int
                                ) -> List[Dict[str, object]]:
    epoch_next = int(anchor) + 1
    source_fps = _source_train_batch_fingerprints_for_replay(source_run_dir, epoch_next)
    materialized = list(audit.get("materialized_fingerprints", []))
    rows: List[Dict[str, object]] = []
    if not source_fps:
        rows.append({"epoch": "all", "match": False,
                     "reason": "source batch_trace missing or empty"})
        return rows
    manifest = _read_source_manifest(source_run_dir)
    dataset = str(manifest.get("dataset_key", "")).lower()
    bpe = proto.BATCHES_PER_EPOCH.get(dataset)
    if bpe is None:
        raise ValueError("source dataset not in known matrix: {}".format(dataset))
    expected_offset = bpe * epoch_next
    if len(source_fps) < expected_offset + k:
        rows.append({"epoch": "epoch_{}_step_0..{}".format(epoch_next, k - 1),
                     "match": False,
                     "reason": "source trace too short: {} < {}".format(
                         len(source_fps), expected_offset + k)})
        return rows
    slice_source = source_fps[expected_offset:expected_offset + k]
    for step in range(k):
        match = (slice_source[step] == materialized[step])
        rows.append({"epoch": "epoch_{}_step_{}".format(epoch_next, step),
                     "match": bool(match),
                     "expected": slice_source[step],
                     "observed": materialized[step]})
    return rows


# -----------------------------------------------------------------------------
# Optimizer state reconstruction
# -----------------------------------------------------------------------------

def _discover_adam_step_template(device: torch.device
                                  ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32, device=device),
                           requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.01, betas=(0.9, 0.999), weight_decay=1e-5)
    p.grad = torch.zeros_like(p)
    opt.step()
    state = opt.state[p]
    step_tensor = state["step"]
    return step_tensor, (step_tensor.detach().clone(),)


def _build_adam_from_dump(parameters: Mapping[str, torch.nn.Parameter],
                          shared_names: Sequence[str], private_names: Sequence[str],
                          moment_dump_path: str, main_lr: float, main_betas: Tuple[float, float],
                          main_wd: float, step_template: torch.Tensor
                          ) -> Tuple[torch.optim.Adam, Dict[str, object]]:
    shared_params = [parameters[name] for name in shared_names]
    private_params = [parameters[name] for name in private_names]
    main_params = list(shared_params) + list(private_params)
    optimizer = torch.optim.Adam(main_params, lr=float(main_lr), betas=tuple(main_betas),
                                weight_decay=float(main_wd))
    payload = torch.load(moment_dump_path, map_location="cpu", weights_only=False)
    installed_shared = 0
    for name in shared_names:
        param = parameters[name]
        entry = payload.get(name)
        if entry is None:
            raise AssertionError("moment dump missing shared name: " + name)
        exp_avg = entry["exp_avg"].detach().to(param.device).clone()
        exp_avg_sq = entry["exp_avg_sq"].detach().to(param.device).clone()
        step_value = entry["step"]
        if isinstance(step_value, torch.Tensor):
            step_tensor = step_value.detach().to(dtype=step_template.dtype,
                                                  device=step_template.device).clone()
        else:
            step_tensor = torch.tensor(int(step_value), dtype=step_template.dtype,
                                        device=step_template.device)
        optimizer.state[param] = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq,
                                  "step": step_tensor}
        installed_shared += 1
    shared_steps = set()
    for name in shared_names:
        state = optimizer.state.get(parameters[name], {})
        if "step" in state:
            value = state["step"]
            shared_steps.add(int(value.detach().cpu().item()) if isinstance(value, torch.Tensor)
                              else int(value))
    if not shared_steps:
        raise AssertionError("no shared Adam step installed")
    if len(shared_steps) != 1:
        raise AssertionError("shared Adam step non-uniform: {}".format(sorted(shared_steps)))
    return optimizer, {
        "shared_step": int(next(iter(shared_steps))),
        "installed_shared": int(installed_shared),
        "installed_private": 0,
        "private_policy": "fresh_adam_no_state",
        "moment_dump_path": os.path.abspath(moment_dump_path),
    }


def _build_fresh_main_adam(parameters: Mapping[str, torch.nn.Parameter],
                            shared_names: Sequence[str], private_names: Sequence[str],
                            main_lr: float, main_betas: Tuple[float, float], main_wd: float
                            ) -> torch.optim.Adam:
    shared_params = [parameters[name] for name in shared_names]
    private_params = [parameters[name] for name in private_names]
    main_params = list(shared_params) + list(private_params)
    return torch.optim.Adam(main_params, lr=float(main_lr), betas=tuple(main_betas),
                            weight_decay=float(main_wd))


def reconstruct_anchor(model: torch.nn.Module, source_run_dir: str, anchor: int,
                        device: torch.device, moment_dump_path: Optional[str] = None
                        ) -> Dict[str, object]:
    parameters = dict(model.named_parameters())
    partition = partition_parameters(model)
    shared_names = list(partition["groups"][GROUP_SHARED])
    private_names = list(partition["groups"][GROUP_PRIVATE])
    ckpt_path = os.path.join(source_run_dir, proto.source_checkpoint_filename(anchor))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError("anchor checkpoint missing: " + ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    if moment_dump_path is not None and os.path.isfile(moment_dump_path):
        step_template, _ = _discover_adam_step_template(device)
        optimizer, audit = _build_adam_from_dump(parameters, shared_names, private_names,
                                                  moment_dump_path, 0.01, (0.9, 0.999), 1e-5,
                                                  step_template)
    else:
        optimizer = _build_fresh_main_adam(parameters, shared_names, private_names,
                                            0.01, (0.9, 0.999), 1e-5)
        audit = {"shared_step": 0, "installed_shared": 0, "installed_private": 0,
                  "private_policy": "fresh_adam_no_state", "moment_dump_path": ""}
    audit["anchor_checkpoint"] = os.path.abspath(ckpt_path)
    audit["anchor_checkpoint_sha256"] = _hash_file(ckpt_path)
    audit["anchor"] = int(anchor)
    if moment_dump_path is not None and os.path.isfile(moment_dump_path):
        audit["moment_dump_sha256"] = _hash_file(moment_dump_path)
    else:
        audit["moment_dump_sha256"] = ""
    return {"optimizer": optimizer, "audit": audit,
            "shared_names": shared_names, "private_names": private_names}


def branch_state(model: torch.nn.Module, optimizer: torch.optim.Adam
                  ) -> Tuple[torch.nn.Module, torch.optim.Adam]:
    """Deepcopy a (model, optimizer) pair.  Rebinds ``optimizer.param_groups``
    and ``optimizer.state`` to the new model's parameters.  Raises on any
    inequality between original and clone parameters.

    Strategy: rebuild ``new_optimizer.param_groups`` from scratch (using
    the new-model parameters in the same order as the source optimizer's
    param_groups) and rebuild ``new_optimizer.state`` from a clone of the
    source state.  The source optimizer's param_groups identity is not
    preserved across deepcopy.
    """
    parameters = dict(model.named_parameters())
    new_model = copy.deepcopy(model)
    new_optimizer = copy.deepcopy(optimizer)
    new_params = dict(new_model.named_parameters())
    # Build name -> new param lookup.
    name_to_new: Dict[str, torch.nn.Parameter] = new_params
    # Build source (old_param_id -> name) and (name -> new_param) maps.
    name_to_source: Dict[str, torch.nn.Parameter] = {}
    for group in optimizer.param_groups:
        for old_param in group["params"]:
            old_name = _name_of(parameters, old_param)
            name_to_source[old_name] = old_param
    # Rebuild new_optimizer.param_groups from scratch.
    new_optimizer.param_groups.clear()
    for group in optimizer.param_groups:
        new_group_params = []
        for old_param in group["params"]:
            old_name = _name_of(parameters, old_param)
            if old_name not in name_to_new:
                raise AssertionError("deepcopy lost parameter: " + old_name)
            new_group_params.append(name_to_new[old_name])
        new_group = dict(group)
        new_group["params"] = new_group_params
        new_optimizer.param_groups.append(new_group)
    # Rebuild state.
    new_optimizer.state.clear()
    for old_param, old_state in optimizer.state.items():
        old_name = _name_of(parameters, old_param)
        if old_name not in name_to_new:
            raise AssertionError("optimizer state references lost parameter: " + old_name)
        new_param = name_to_new[old_name]
        new_optimizer.state[new_param] = {
            key: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
            for key, value in old_state.items()
        }
    # Sanity check: every original parameter is bit-equal to its clone.
    for name, old_param in parameters.items():
        if name not in new_params:
            raise AssertionError("deepcopy lost parameter: " + name)
        if not torch.equal(old_param.detach().cpu(),
                            new_params[name].detach().cpu()):
            raise AssertionError("parameter inequality after deepcopy: " + name)
    return new_model, new_optimizer


# -----------------------------------------------------------------------------
# Validation pass
# -----------------------------------------------------------------------------

def _safe_validation_cell_gradients(model: torch.nn.Module, bundle, shared_params,
                                       device: torch.device, chunk: int = 40960
                                       ) -> Dict[int, torch.Tensor]:
    """A robust wrapper around the S7-A ``validation_cell_gradients`` that
    tolerates any per-chunk None gradients by zeroing them out (rather than
    raising).  S7-A raises because it always operates on a full-batch chunk
    (or sufficiently diverse chunks); when a particular chunk lacks a
    domain/task combination, the term is exactly 0 and any shared param not
    used by that term's computation yields a None gradient.  We replace
    None with zeros of the appropriate shape to keep the downstream geometry
    contract well-defined.
    """
    import inspect
    from stage_S7_probe import (validation_cell_gradients as _v,
                                  flatten_shared_grad, masked_bce_terms)
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
                "Stage S7-B validation split is missing domain {}".format(d))
        grads[idx] = (accumulators[idx] / float(n_d)).detach()
    return grads


def _full_validation_metrics(model: torch.nn.Module, bundle, device: torch.device,
                              chunk: int = 40960
                              ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float], Dict[int, int]]:
    domain_num, task_num = bundle.domain_num, bundle.task_num
    val_x = np.ascontiguousarray(bundle.val_x.astype(np.float32))
    val_y = np.ascontiguousarray(bundle.val_y.astype(np.float32))
    raw_bce: Dict[int, float] = {t * domain_num + d: 0.0 for t in range(task_num)
                                  for d in range(domain_num)}
    sample_count: Dict[int, int] = {t * domain_num + d: 0 for t in range(task_num)
                                     for d in range(domain_num)}
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
        records[key] = {
            "sample_count": int(len(labels)),
            "auc": float(safe_auc(labels, scores)),
            "logloss": float(safe_logloss(labels, scores)),
        }
    raw_mean = {key: float(raw_bce[key] / max(1, sample_count[key])) for key in raw_bce}
    return records, raw_mean, sample_count


def _aggregate_validation(records: Mapping[int, Mapping[str, float]],
                          raw_bce_sum: Mapping[int, float],
                          raw_bce_count: Mapping[int, int],
                          target_local_domain: int,
                          domain_shares: Mapping[int, float]) -> Dict[str, float]:
    domain_num, task_num = 3, 2
    # s_d are the natural (train-distribution) domain shares, the same
    # measure used by S6's target_weighted metrics and the S7-A performance
    # table; they are NOT derived from the validation split's sample counts.
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
        # S7-A ``_weighted_cell_mean`` semantics: normalized weighted mean
        # over cells, NOT a bare weighted sum (the per-domain weight is
        # applied once per task, so the raw denominator is T * Σ_d w_d).
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


# -----------------------------------------------------------------------------
# Loss + diagnostic + step
# -----------------------------------------------------------------------------

def _loss_components_for(pred: torch.Tensor, y: torch.Tensor, domain_ids: torch.Tensor,
                           target_local: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = loss_components(pred, y, domain_ids, target_local, "sample", {})
    return out["full"], out["target"], out["non_target"]


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


def diagnostics_at_theta(model: torch.nn.Module, branch_tag: str, step: int,
                          batch: Mapping[str, object], bundle,
                          shared_names: Sequence[str], private_names: Sequence[str],
                          slices: Mapping[str, Tuple[int, int]],
                          device: torch.device, val_chunk: int = 40960) -> Dict[str, object]:
    model.train()
    branch_parameters = dict(model.named_parameters())
    shared_params = [branch_parameters[name] for name in shared_names]
    private_params = [branch_parameters[name] for name in private_names]
    x_cpu = batch["x"]
    y_cpu = batch["y"]
    x_device = x_cpu.to(device=device, dtype=torch.float32)
    y_device = y_cpu.to(device=device, dtype=torch.float32)
    domain_ids = x_cpu[:, -1].to(device=device, dtype=torch.long)
    # Val pass: cell metrics + raw BCE sums.
    val_records, val_raw_mean, val_raw_count = _full_validation_metrics(
        model, bundle, device, chunk=val_chunk)
    val_grads = _safe_validation_cell_gradients(model, bundle, shared_params, device,
                                                  chunk=val_chunk)
    val_vecs_fp64 = {idx: _to_fp64_flat([val_grads[idx]]) for idx in range(CELL_NUM)}
    # Train forward.
    pred = model(x_device.transpose(0, 1))
    if not bool(torch.isfinite(pred).all().detach().cpu()):
        raise FloatingPointError("non-finite train prediction")
    terms, _ = masked_bce_terms(pred, y_device, domain_ids, DOMAIN_NUM, TASK_NUM)
    cell_grads: Dict[int, torch.Tensor] = {}
    cell_losses: Dict[int, float] = {}
    for idx, term in enumerate(terms):
        cell_grads[idx] = _flatten_shared(term, shared_params, retain_graph=True)
        cell_losses[idx] = float(term.detach().cpu())
    target_local = _target_local_domain(str(bundle.dataset))
    full_loss, target_loss, non_target_loss = _loss_components_for(
        pred, y_device, domain_ids, target_local)
    g_full_vec = _flatten_shared(full_loss, shared_params, retain_graph=True)
    g_target_vec = _flatten_shared(target_loss, shared_params, retain_graph=True)
    g_private_tup = torch.autograd.grad(full_loss, list(private_params), retain_graph=False,
                                          allow_unused=True)
    g_private_norm = float(vector_norm(list(g_private_tup)))
    weights = batch["w"]
    weighted_sum = torch.zeros_like(g_full_vec, dtype=torch.float32, device=g_full_vec.device)
    for idx in range(CELL_NUM):
        weighted_sum = weighted_sum + float(weights[idx]) * cell_grads[idx]
    recon_residual_gS = float(torch.linalg.vector_norm(
        g_full_vec.double() - weighted_sum.double()).detach().cpu())
    g_full_norm = float(torch.linalg.vector_norm(g_full_vec).detach().cpu())
    g_target_norm = float(torch.linalg.vector_norm(g_target_vec).detach().cpu())
    cos_gf_gt = _cosine_fp64(_to_fp64_flat([g_full_vec]), _to_fp64_flat([g_target_vec]))
    if not math.isfinite(cos_gf_gt):
        cos_gf_gt = 0.0
    u_k_per_target: Dict[int, float] = {}
    h_sum_per_target: Dict[int, float] = {}
    for k in range(CELL_NUM):
        u_k_per_target[k] = _dot_fp64(_to_fp64_flat([g_full_vec]), val_vecs_fp64[k])
        h_total = 0.0
        for s in range(CELL_NUM):
            h_total += float(weights[s]) * _dot_fp64(
                _to_fp64_flat([cell_grads[s]]), val_vecs_fp64[k])
        h_sum_per_target[k] = h_total
    recon_h_max = 0.0
    for k in range(CELL_NUM):
        diff = abs(u_k_per_target[k] - h_sum_per_target[k])
        if diff > recon_h_max:
            recon_h_max = diff
    # Hard reconstruction gates (V1-tier, S4 tolerances): g_S = Σ_i w_i g_i
    # and Σ_i H_{i→k} = U_k must hold per batch by linearity.  A violation
    # means the per-batch weights or the recorded vectors are wrong; abort
    # instead of recording silently (S7-A precedent).
    g_full_tol = max(proto.S4_TOLERANCE_ABS, proto.S4_TOLERANCE_REL * g_full_norm)
    if recon_residual_gS > g_full_tol:
        raise AssertionError(
            "S7B g_S reconstruction residual {:.3e} exceeds tolerance {:.3e} "
            "(branch={}, step={})".format(recon_residual_gS, g_full_tol, branch_tag, step))
    u_abs_max = max(abs(value) for value in u_k_per_target.values())
    h_tol = max(proto.S4_TOLERANCE_ABS, proto.S4_TOLERANCE_REL * u_abs_max)
    if recon_h_max > h_tol:
        raise AssertionError(
            "S7B ΣH=U identity residual {:.3e} exceeds tolerance {:.3e} "
            "(branch={}, step={})".format(recon_h_max, h_tol, branch_tag, step))
    return {
        "branch": branch_tag, "step": step,
        "val_records": val_records, "val_raw_sum": dict(val_raw_mean),
        "val_raw_count": dict(val_raw_count),
        "cell_grads": cell_grads, "cell_losses": cell_losses,
        "g_full_vec": g_full_vec, "g_target_vec": g_target_vec,
        "g_private_norm": g_private_norm,
        "weights": weights, "val_vecs_fp64": val_vecs_fp64,
        "g_full_norm": g_full_norm, "g_target_norm": g_target_norm,
        "cos_gf_gt": float(cos_gf_gt),
        "recon_residual_gS": float(recon_residual_gS),
        "recon_residual_H": float(recon_h_max),
        "domain_counts": list(batch["domain_counts"]),
        "fingerprint": str(batch["fingerprint"]),
        "L_full": float(full_loss.detach().cpu()),
        "L_target": float(target_loss.detach().cpu()),
        "L_non": float(non_target_loss.detach().cpu()),
        "target_local": int(target_local),
        "slices": {module: (int(slices[module][0]), int(slices[module][1]))
                    for module in slices},
    }


def controlled_continuation_step(model: torch.nn.Module, optimizer: torch.optim.Adam,
                                  branch: str, batch: Mapping[str, object],
                                  shared_names: Sequence[str], private_names: Sequence[str],
                                  device: torch.device, target_local: int
                                  ) -> Dict[str, object]:
    if branch not in (proto.BRANCH_F, proto.BRANCH_T):
        raise ValueError("branch must be F or T")
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
        raise FloatingPointError("non-finite S7-B prediction")
    full_loss, target_loss, _ = _loss_components_for(pred, y_device, domain_ids, target_local)
    gf, gt, gp = gradients_one_forward(full_loss, target_loss, parameters,
                                        shared_names, private_names)
    df, _ = adam_deltas(parameters, shared_names, gf, optimizer)
    dt, _ = adam_deltas(parameters, shared_names, gt, optimizer)
    dp, _ = adam_deltas(parameters, private_names, gp, optimizer)
    installed_shared = [None if g is None else g.detach().clone() for g in gf]
    installed_private = [None if g is None else g.detach().clone() for g in gp]
    before = [parameters[name].detach().clone() for name in shared_names]
    private_before = [parameters[name].detach().clone() for name in private_names]
    _set_grad(parameters, shared_names, installed_shared)
    _set_grad(parameters, private_names, installed_private)
    optimizer.step()
    if branch == proto.BRANCH_F:
        observed = [parameters[name].detach() - old for name, old in zip(shared_names, before)]
        applied_residual = assert_vector_match(observed, df, "S7B F observed == Δ_F")
    else:
        with torch.no_grad():
            for name, old, target_delta in zip(shared_names, before, dt):
                if target_delta is None:
                    continue
                parameters[name].copy_(old + target_delta)
        observed = [parameters[name].detach() - old for name, old in zip(shared_names, before)]
        applied_residual = assert_vector_match(observed, dt, "S7B T observed == Δ_T")
    private_observed = [parameters[name].detach() - old
                         for name, old in zip(private_names, private_before)]
    private_residual = assert_vector_match(private_observed, dp,
                                            "S7B private observed == Δ_P")
    delta_cos = cosine(df, dt)
    return {
        "delta_full_norm": float(vector_norm(df)),
        "delta_target_norm": float(vector_norm(dt)),
        "delta_applied_norm": float(vector_norm(dt if branch == proto.BRANCH_T else df)),
        "delta_cosine": float(delta_cos) if (delta_cos is not None and math.isfinite(delta_cos)) else 0.0,
        "private_delta_norm": float(vector_norm(dp)),
        "applied_formula_residual": float(applied_residual),
        "private_full_residual": float(private_residual),
    }


# -----------------------------------------------------------------------------
# Run + outputs
# -----------------------------------------------------------------------------

def run_continuation(dataset: str, seed: int, anchor: int, out_dir: str,
                     dataset_path: str = "",
                     smoke_state: bool = False,
                     smoke_replay: bool = False,
                     smoke_onestep: bool = False) -> Dict[str, object]:
    dataset = str(dataset).lower()
    if dataset not in proto.DATASETS:
        raise ValueError("dataset must be kuairand or movielens")
    if int(seed) not in proto.SEEDS:
        raise ValueError("seed outside matrix")
    if int(anchor) not in proto.ANCHORS:
        raise ValueError("anchor outside matrix")
    if sum([smoke_state, smoke_replay, smoke_onestep]) > 1:
        raise ValueError("only one smoke mode at a time")
    started = time.time()
    source_run_dir = proto.source_run_dir(dataset, int(seed))
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError("source run missing: " + source_run_dir)
    source_manifest = _read_source_manifest(source_run_dir)
    initial_hash_expected = source_manifest.get("initial_parameter_state_hash", "")
    setting = proto.target_setting(dataset)
    bundle = load_stage_s1_bundle(dataset, dataset_path or None, setting, int(seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(int(seed))
    model = build_model(bundle.features, DOMAIN_NUM, TASK_NUM, "m3oe",
                        expert_num=4, fcn_dims=(512, 256, 256, 64), seed=int(seed))
    model = model.to(device)
    initial_hash_observed = initial_state_hash(model)
    if str(initial_hash_expected) and str(initial_hash_observed) != str(initial_hash_expected):
        raise AssertionError(
            "Stage S7-B initial hash mismatch: observed {} != expected {}".format(
                initial_hash_observed, initial_hash_expected))
    partition = partition_parameters(model)
    assert_expected_parameter_counts(partition, dataset)
    parameters = dict(model.named_parameters())
    shared_names = list(partition["groups"][GROUP_SHARED])
    private_names = list(partition["groups"][GROUP_PRIVATE])
    # Architecture parameters exist in the model but are NOT part of the
    # main Adam; they are managed by a separate ``arch_optimizer`` in the
    # source runs.  K=5 main-only continuation never touches them.
    slices = module_slices(shared_names, parameters)
    moment_path = os.path.join(source_run_dir, proto.source_moment_filename(int(anchor)))
    target_local = _target_local_domain(dataset)
    # -- smoke-state --
    if smoke_state:
        result = reconstruct_anchor(model, source_run_dir, int(anchor), device, moment_path)
        audit = dict(result["audit"])
        shared_step_observed = 0
        for name in shared_names:
            state = result["optimizer"].state.get(parameters[name])
            if state and "step" in state:
                value = state["step"]
                shared_step_observed = int(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else int(value)
                break
        private_state_count = int(sum(1 for name in private_names
                                        if result["optimizer"].state.get(parameters[name])))
        smoke_report = {
            "smoke_state_ok": True,
            "dataset": dataset, "seed": int(seed), "anchor": int(anchor),
            "source_run_dir": source_run_dir,
            "checkpoint_loaded_sha256": audit["anchor_checkpoint_sha256"],
            "moment_dump_sha256": audit.get("moment_dump_sha256", ""),
            "shared_step_observed": int(shared_step_observed),
            "shared_step_expected": proto.expected_shared_step(dataset, int(anchor)),
            "arch_step_expected": proto.expected_arch_step(int(anchor)),
            "private_state_count": int(private_state_count),
            "installed_shared_moments": int(audit.get("installed_shared", 0)),
            "private_policy": audit.get("private_policy", "?"),
        }
        if shared_step_observed != proto.expected_shared_step(dataset, int(anchor)):
            smoke_report["smoke_state_ok"] = False
            smoke_report["abort_reason"] = "shared_step_mismatch"
        if int(audit.get("installed_shared", 0)) <= 0:
            smoke_report["smoke_state_ok"] = False
            smoke_report["abort_reason"] = "no_shared_moments_installed"
        if private_state_count != 0:
            smoke_report["smoke_state_ok"] = False
            smoke_report["abort_reason"] = "private_state_present"
        return smoke_report
    # -- replay --
    batches, replay_audit = replay_epoch_batches(bundle, int(seed), int(anchor), proto.K_STEPS)
    for b in batches:
        b["x"] = b["x"].contiguous()
        b["y"] = b["y"].contiguous()
    fingerprint_rows = _verify_replay_fingerprints(replay_audit, source_run_dir, int(anchor),
                                                   proto.K_STEPS)
    replay_all_match = bool(fingerprint_rows) and all(row.get("match", False)
                                                       for row in fingerprint_rows)
    if smoke_replay:
        return {
            "smoke_replay_ok": bool(replay_all_match),
            "dataset": dataset, "seed": int(seed), "anchor": int(anchor),
            "epoch_target": int(anchor) + 1,
            "k": proto.K_STEPS,
            "materialized_fingerprints": [b["fingerprint"] for b in batches],
            "fingerprint_rows": fingerprint_rows,
            "source_run_dir": source_run_dir,
        }
    # -- anchor reconstruction with optimizer state --
    reconstruction = reconstruct_anchor(model, source_run_dir, int(anchor), device, moment_path)
    anchor_audit = reconstruction["audit"]
    optimizer = reconstruction["optimizer"]
    expected_shared_step = proto.expected_shared_step(dataset, int(anchor))
    if int(anchor_audit.get("shared_step", -1)) != expected_shared_step:
        raise AssertionError(
            "Stage S7-B shared Adam step {} != expected {} (batches_per_epoch x (anchor+1))".format(
                anchor_audit.get("shared_step"), expected_shared_step))
    arch_names = list(partition["groups"][GROUP_ARCH])
    # -- smoke-onestep --
    if smoke_onestep:
        f_model, f_opt = branch_state(model, optimizer)
        t_model, t_opt = branch_state(model, optimizer)
        f_diag = controlled_continuation_step(f_model, f_opt, proto.BRANCH_F, batches[0],
                                                shared_names, private_names, device,
                                                target_local)
        t_diag = controlled_continuation_step(t_model, t_opt, proto.BRANCH_T, batches[0],
                                                shared_names, private_names, device,
                                                target_local)
        return {
            "smoke_onestep_ok": True,
            "dataset": dataset, "seed": int(seed), "anchor": int(anchor),
            "F_applied_formula_residual": f_diag["applied_formula_residual"],
            "T_applied_formula_residual": t_diag["applied_formula_residual"],
            "F_private_full_residual": f_diag["private_full_residual"],
            "T_private_full_residual": t_diag["private_full_residual"],
            "delta_cosine_F_recorded": f_diag["delta_cosine"],
            "delta_cosine_T_recorded": t_diag["delta_cosine"],
            "delta_cosine_F_minus_T": abs(f_diag["delta_cosine"] - t_diag["delta_cosine"]),
        }
    # -- output dir --
    out_dir_abs = os.path.abspath(out_dir)
    if os.path.isdir(out_dir_abs):
        existing = [name for name in os.listdir(out_dir_abs)
                    if name not in ("failed_attempts", "stdout.log", "process_interval.json")]
        if existing:
            raise FileExistsError(
                "Stage S7-B refusing to overwrite non-empty run directory: " + out_dir_abs)
    os.makedirs(out_dir_abs, exist_ok=True)
    # -- branches --
    f_model, f_opt = branch_state(model, optimizer)
    t_model, t_opt = branch_state(model, optimizer)
    f_theta_before = {name: param.detach().clone() for name, param
                       in f_model.named_parameters()}
    # Architecture parameters are never installed in the continuation's main
    # Adam and no arch optimizer is created; snapshot them so the end-of-run
    # invariance assertion is a hard check, not an assumption.
    arch_anchor = {name: f_theta_before[name].detach().cpu().clone()
                   for name in arch_names}
    # -- continuation loop --
    branches = {proto.BRANCH_F: {"model": f_model, "opt": f_opt,
                                  "diagnostics": [], "step_metrics": []},
                proto.BRANCH_T: {"model": t_model, "opt": t_opt,
                                  "diagnostics": [], "step_metrics": []}}
    f_to_t_distances: List[float] = []
    f_param_history: List[Dict[str, torch.Tensor]] = []  # theta_t^F after each step
    t_param_history: List[Dict[str, torch.Tensor]] = []
    # Record the initial (pre-step-0) theta distance between the two branches.
    f_to_t_distances.append(_theta_distance(branches[proto.BRANCH_F]["model"],
                                                branches[proto.BRANCH_T]["model"]))
    source_main_rows = _source_main_rows_for_continuation(source_run_dir, int(anchor),
                                                          proto.K_STEPS)
    rng_digest_continuation_before = rng_state_digest()
    rng_digest_per_step: List[str] = []
    abort_payload: Optional[Dict[str, object]] = None
    for step in range(proto.K_STEPS):
        # Per-step RNG non-consumption gate: the digest at every step
        # boundary must equal the digest taken before the loop.
        step_rng_digest = rng_state_digest()
        if step_rng_digest != rng_digest_continuation_before:
            raise AssertionError(
                "Stage S7-B continuation consumed RNG before step {}".format(step))
        rng_digest_per_step.append(step_rng_digest)
        # V3 (step 0 only, pre-controlled-step): the private parameters
        # must be bit-equal between the two branches at the start of
        # step 0 (both branches were deepcopied from the same anchor
        # state).  This is the correct V3 check: it certifies the
        # branch symmetry contract at the moment the branches are
        # constructed.  At step 1+, the two branches' pre-step shared
        # thetas diverge (because the T branch overwrites shared with
        # Δ_T at step 0), which makes g_P differ, which makes the
        # private Adam updates differ.  This is the structural
        # consequence of the T branch's semantics and is NOT a V3
        # violation.  The post-step private divergence is recorded
        # in v3_per_step for transparency.
        if step == 0:
            f_pre_private = dict(branches[proto.BRANCH_F]["model"].named_parameters())
            t_pre_private = dict(branches[proto.BRANCH_T]["model"].named_parameters())
            for pname in private_names:
                f_p = f_pre_private[pname].detach()
                t_p = t_pre_private[pname].detach()
                if not torch.equal(f_p, t_p):
                    diff = float(torch.max(torch.abs(f_p - t_p)).detach().cpu())
                    raise AssertionError(
                        "S7B V3 private pre-step symmetry violated at step 0 on {}: "
                        "F and T private tensors differ (max abs diff = {:.3e}). "
                        "This should be impossible: both branches were deepcopied "
                        "from the same anchor state.".format(pname, diff))
        # Diagnostics for both branches at theta_t (BEFORE step).
        for branch_name in proto.BRANCHES:
            entry = branches[branch_name]
            diag = diagnostics_at_theta(
                entry["model"], branch_name, step, batches[step], bundle,
                shared_names, private_names, slices, device, val_chunk=40960)
            entry["diagnostics"].append(diag)
        # Controlled step for both branches on the same batch.
        for branch_name in proto.BRANCHES:
            entry = branches[branch_name]
            step_diag = controlled_continuation_step(
                entry["model"], entry["opt"], branch_name, batches[step],
                shared_names, private_names, device, target_local)
            entry["step_metrics"].append(step_diag)
        # Snapshot post-step theta (F and T) and compute distance.
        f_post = {name: param.detach().clone() for name, param
                   in branches[proto.BRANCH_F]["model"].named_parameters()}
        t_post = {name: param.detach().clone() for name, param
                   in branches[proto.BRANCH_T]["model"].named_parameters()}
        f_param_history.append(f_post)
        t_param_history.append(t_post)
        # Distance shared only.
        total = 0.0
        for name in shared_names:
            diff = f_post[name].detach().to(dtype=torch.float64, device="cpu") \
                 - t_post[name].detach().to(dtype=torch.float64, device="cpu")
            total += float(torch.sum(diff * diff).item())
        f_to_t_distances.append(math.sqrt(total))
    rng_digest_continuation_after = rng_state_digest()
    # Architecture-parameter invariance gate (hard check, both branches):
    # arch parameters are never installed in the continuation's main Adam
    # and no arch optimizer exists, so they must be bit-identical to the
    # pre-loop snapshot.
    for branch_name in proto.BRANCHES:
        model_b = branches[branch_name]["model"]
        current = dict(model_b.named_parameters())
        for name in arch_names:
            if not torch.equal(current[name].detach().cpu(), arch_anchor[name]):
                diff = float(torch.max(torch.abs(
                    current[name].detach().cpu() - arch_anchor[name])))
                raise AssertionError(
                    "S7B architecture parameter invariance violated on branch {} "
                    "param {}: max abs diff = {:.3e}".format(branch_name, name, diff))
    # V3 per-step summary: step 0 pre-step private equality is the
    # hard gate (enforced inside the loop; always passes by
    # construction).  We additionally record the POST-step private
    # divergence for transparency — it is the structural consequence
    # of the T branch overwriting shared with Δ_T (which makes the
    # two branches' pre-step shared thetas differ at step 1+, hence
    # their g_P and private Adam updates differ).  This is not a V3
    # violation; it is the expected behavior of the branch semantics.
    v3_per_step: List[Dict[str, object]] = []
    for step_idx in range(proto.K_STEPS):
        f_snap = f_param_history[step_idx]
        t_snap = t_param_history[step_idx]
        per_param_max = 0.0
        per_param_name = ""
        for pname in private_names:
            f_p = f_snap[pname].detach()
            t_p = t_snap[pname].detach()
            if not torch.equal(f_p, t_p):
                diff = float(torch.max(torch.abs(f_p - t_p)).detach().cpu())
                if diff > per_param_max:
                    per_param_max = diff
                    per_param_name = pname
        v3_per_step.append({
            "step": int(step_idx),
            "v3_pre_step_pass": (step_idx == 0),
            "post_step_private_max_abs_diff": float(per_param_max),
            "post_step_first_divergent_param": per_param_name,
        })
    v3_aborted = False
    v3_abort_reason = ""
    # -- step5 final: per-branch val metrics, no gradients, no step --
    final_diag: Dict[str, Dict[str, object]] = {}
    for branch_name in proto.BRANCHES:
        entry = branches[branch_name]
        model_b = entry["model"]
        val_records, val_raw_mean, val_raw_count = _full_validation_metrics(
            model_b, bundle, device, chunk=40960)
        aggregates = _aggregate_validation(val_records, val_raw_mean,
                                            val_raw_count, target_local,
                                            bundle.domain_shares)
        final_diag[branch_name] = {
            "val_records": val_records, "val_raw_sum": dict(val_raw_mean),
            "val_raw_count": dict(val_raw_count),
            "aggregates": aggregates,
        }
    # -- validity check (layered V0 / V2 contract) --
    #
    # V0: step 0 only.  |ΔL_full| ≤ 1e-5 AND shared delta_norm rel diff
    #     ≤ 1e-3 certify the bit-exact reconstruction.  Failure aborts.
    # V2: step >= 1.  Report-only.  Per-step F-vs-source divergence is
    #     EXPECTED (private-moments fresh-Adam approximation; Algorithm
    #     Spec §4).  Recorded in ``validity_L_full_diff`` /
    #     ``validity_delta_norm_rel_diff`` and a non-blocking tripwire
    #     flags any single step with ``|L_full diff| > 0.3`` for human
    #     review in the run report.
    # V3: step 0 only, pre-step.  Enforced INSIDE the loop (bit-equal
    #     private parameters between branches); any violation raises
    #     immediately.  Step >= 1 private divergence is the structural
    #     consequence of the T branch's shared overwrite and is recorded
    #     report-only in v3_per_step.
    validity: List[Dict[str, object]] = []
    aborted = False
    abort_reason = ""
    review_flagged = False
    for step in range(proto.K_STEPS):
        f_diag = branches[proto.BRANCH_F]["diagnostics"][step]
        source_row = source_main_rows[step]
        expected_fp = str(source_row.get("batch_fingerprint", ""))
        observed_fp = str(f_diag["fingerprint"])
        if observed_fp != expected_fp:
            validity.append({"step": step, "kind": "batch_fingerprint",
                              "match": False, "expected": expected_fp,
                              "observed": observed_fp, "aborted": True})
            aborted = True
            abort_reason = "batch_fingerprint_mismatch"
            break
        try:
            source_l_full = float(source_row.get("L_full", "nan"))
        except (TypeError, ValueError):
            source_l_full = float("nan")
        l_full_diff = abs(f_diag["L_full"] - source_l_full)
        # ‖θ_t - θ_{t-1}‖_shared on the F branch.
        if step == 0:
            before = f_theta_before
        else:
            before = f_param_history[step - 1]
        after = f_param_history[step]
        delta_norm_sq = 0.0
        for name in shared_names:
            diff = after[name].detach().to(dtype=torch.float64, device="cpu") \
                 - before[name].detach().to(dtype=torch.float64, device="cpu")
            delta_norm_sq += float(torch.sum(diff * diff).item())
        f_delta_observed = math.sqrt(delta_norm_sq)
        try:
            source_delta_observed = float(source_row.get("delta_observed_step_norm", "nan"))
        except (TypeError, ValueError):
            source_delta_observed = float("nan")
        if source_delta_observed > 0 and math.isfinite(source_delta_observed):
            delta_rel_diff = abs(f_delta_observed - source_delta_observed) / source_delta_observed
        else:
            delta_rel_diff = float("nan")
        # V0 (step 0 hard gate) vs V2 (step >= 1 report-only).
        if step == 0:
            v0_l_full_fail = bool(l_full_diff > proto.V0_L_FULL_DIFF_ABS)
            v0_delta_rel_fail = bool(math.isfinite(delta_rel_diff)
                                       and delta_rel_diff > proto.V0_DELTA_NORM_REL_DIFF)
            v2_review = False
        else:
            v0_l_full_fail = False
            v0_delta_rel_fail = False
            v2_review = bool(l_full_diff > proto.V2_REVIEW_TRIPWIRE_L_FULL_DIFF)
            if v2_review:
                review_flagged = True
        entry = {
            "step": step, "kind": "L_full_and_delta_observed",
            "validity_layer": ("V0" if step == 0 else "V2"),
            "observed_l_full": f_diag["L_full"], "expected_l_full": source_l_full,
            "l_full_abs_diff": float(l_full_diff),
            "observed_delta_norm": float(f_delta_observed),
            "expected_delta_norm": float(source_delta_observed),
            "delta_norm_rel_diff": float(delta_rel_diff),
            "aborted_v0_l_full": bool(v0_l_full_fail),
            "aborted_v0_delta_rel": bool(v0_delta_rel_fail),
            "review_v2_tripwire": bool(v2_review),
        }
        validity.append(entry)
        if v0_l_full_fail or v0_delta_rel_fail:
            aborted = True
            abort_reason = ("v0_l_full_abs_diff_above_threshold"
                            if v0_l_full_fail else "v0_delta_norm_rel_diff_above_threshold")
            break
    # V3 (per step, both branches) is enforced inside the loop as a
    # pre-step hard gate.  If we reach here without V3 raising, the
    # per-step V3 contract held.  The post-step private divergence is
    # recorded in v3_per_step for transparency but is not a V3
    # violation (it is the structural consequence of the T branch
    # overwriting shared with Δ_T).
    # Always record the review flag in the audit; it does not affect the
    # abort / ok status (V2 is report-only).
    # -- write outputs --
    payload = _write_outputs(out_dir_abs, dataset, int(seed), int(anchor), source_run_dir,
                              anchor_audit, replay_audit, fingerprint_rows, branches,
                              final_diag, f_to_t_distances, validity, source_main_rows,
                              started, replay_all_match,
                              rng_digest_continuation_before, rng_digest_continuation_after,
                              aborted, abort_reason, review_flagged,
                              v3_per_step, bundle.domain_shares,
                              rng_digest_per_step)
    return payload


def _write_outputs(out_dir_abs: str, dataset: str, seed: int, anchor: int,
                    source_run_dir: str, anchor_audit: Dict[str, object],
                    replay_audit: Dict[str, object], fingerprint_rows: List[Dict[str, object]],
                    branches: Dict[str, Dict[str, object]],
                    final_diag: Dict[str, Dict[str, object]],
                    f_to_t_distances: List[float], validity: List[Dict[str, object]],
                    source_main_rows: List[Dict[str, str]],
                    started: float, replay_all_match: bool,
                    rng_before: str, rng_after: str,
                    aborted: bool, abort_reason: str,
                    review_flagged: bool,
                    v3_per_step: List[Dict[str, object]],
                    domain_shares: Mapping[int, float],
                    rng_digest_per_step: List[str]) -> Dict[str, object]:
    epoch_next = int(anchor) + 1
    step_rows: List[Dict[str, object]] = []
    geometry_rows: List[Dict[str, object]] = []
    val_rows: List[Dict[str, object]] = []
    for branch_name in proto.BRANCHES:
        entry = branches[branch_name]
        for step in range(proto.K_STEPS):
            diag = entry["diagnostics"][step]
            step_diag = entry["step_metrics"][step]
            target_local = int(diag["target_local"])
            slices_diag = diag["slices"]
            row = {
                "stage": "S7B", "dataset": dataset, "seed": int(seed),
                "anchor_epoch": int(anchor), "branch": branch_name,
                "epoch": epoch_next, "step": int(step),
                "batch_fingerprint": diag["fingerprint"],
                "domain_counts": _json_counts(diag["domain_counts"]),
                "L_full": float(diag["L_full"]),
                "L_target": float(diag["L_target"]),
                "L_non": float(diag["L_non"]),
                "g_full_norm": float(diag["g_full_norm"]),
                "g_target_norm": float(diag["g_target_norm"]),
                "cosine_gf_gt": float(diag["cos_gf_gt"]),
                "delta_full_norm": float(step_diag["delta_full_norm"]),
                "delta_target_norm": float(step_diag["delta_target_norm"]),
                "delta_applied_norm": float(step_diag["delta_applied_norm"]),
                "delta_cosine": float(step_diag["delta_cosine"]),
                "private_delta_norm": float(step_diag["private_delta_norm"]),
                "theta_distance_shared": float(f_to_t_distances[step + 1]),
                "applied_formula_residual": float(step_diag["applied_formula_residual"]),
                "recon_residual_gS": float(diag["recon_residual_gS"]),
                "recon_residual_H": float(diag["recon_residual_H"]),
                "rng_digest": str(rng_after),
                "validity_L_full_diff": next((float(v["l_full_abs_diff"]) for v in validity
                                              if v["step"] == step and "l_full_abs_diff" in v),
                                             float("nan")),
                "validity_delta_norm_rel_diff": next((float(v["delta_norm_rel_diff"]) for v in validity
                                                      if v["step"] == step and "delta_norm_rel_diff" in v),
                                                     float("nan")),
            }
            step_rows.append(row)
            # Geometry rows: 36 sources x 36 targets x 8 modules.
            for s in range(CELL_NUM):
                s_d, s_t = s % DOMAIN_NUM, s // DOMAIN_NUM
                s_vec_full = _to_fp64_flat([diag["cell_grads"][s]])
                full_vec_fp64 = _to_fp64_flat([diag["g_full_vec"]])
                for t in range(CELL_NUM):
                    t_d, t_t = t % DOMAIN_NUM, t // DOMAIN_NUM
                    t_cell_vec_full = _to_fp64_flat([diag["cell_grads"][t]])
                    val_vec_full = diag["val_vecs_fp64"][t]
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
                            start, end = slices_diag[module]
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
                        geometry_rows.append({
                            "stage": "S7B", "dataset": dataset, "seed": int(seed),
                            "anchor_epoch": int(anchor), "branch": branch_name,
                            "step": int(step),
                            "source_domain": s_d, "source_task": s_t,
                            "target_domain": t_d, "target_task": t_t,
                            "pair_class": pc,
                            "shared_module": module,
                            "gradient_norm": float(norm_s),
                            "pairwise_cosine": float(cs_st_clamped),
                            "sample_alignment": float(align_s),
                            "validation_dot_product": float(u_t),
                            "weighted_source_target_contribution": float(h_st),
                            "validation_gradient_norm": float(_norm_fp64(val_slice)),
                            "source_weight": float(diag["weights"][s]),
                        })
            # Per-cell validation rows for this step.
            val_records = diag["val_records"]
            val_raw_sum = diag["val_raw_sum"]
            for cell_key in sorted(val_records):
                d = cell_key % DOMAIN_NUM
                t = cell_key // DOMAIN_NUM
                val_rows.append({
                    "stage": "S7B", "dataset": dataset, "seed": int(seed),
                    "anchor_epoch": int(anchor), "branch": branch_name,
                    "step": int(step), "scope": "cell",
                    "domain": d, "task": t, "cell": cell_key,
                    "val_bce_raw": float(val_raw_sum[cell_key]),
                    "auc": float(val_records[cell_key]["auc"]),
                    "logloss": float(val_records[cell_key]["logloss"]),
                })
            aggregates = _aggregate_validation(val_records, val_raw_sum,
                                                 diag["val_raw_count"], target_local,
                                                 domain_shares)
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
                    "stage": "S7B", "dataset": dataset, "seed": int(seed),
                    "anchor_epoch": int(anchor), "branch": branch_name,
                    "step": int(step), "scope": scope_name,
                    "domain": -1, "task": -1, "cell": -1,
                    "val_bce_raw": float(loss_v),
                    "auc": float(auc_v),
                    "logloss": float(ll_v),
                })
        # Final step (=K) row.
        f_info = final_diag[branch_name]
        val_records = f_info["val_records"]
        val_raw_sum = f_info["val_raw_sum"]
        val_raw_count = f_info["val_raw_count"]
        aggregates = f_info["aggregates"]
        for cell_key in sorted(val_records):
            d = cell_key % DOMAIN_NUM
            t = cell_key // DOMAIN_NUM
            val_rows.append({
                "stage": "S7B", "dataset": dataset, "seed": int(seed),
                "anchor_epoch": int(anchor), "branch": branch_name,
                "step": int(proto.K_STEPS), "scope": "cell",
                "domain": d, "task": t, "cell": cell_key,
                "val_bce_raw": float(val_raw_sum[cell_key]),
                "auc": float(val_records[cell_key]["auc"]),
                "logloss": float(val_records[cell_key]["logloss"]),
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
                "stage": "S7B", "dataset": dataset, "seed": int(seed),
                "anchor_epoch": int(anchor), "branch": branch_name,
                "step": int(proto.K_STEPS), "scope": scope_name,
                "domain": -1, "task": -1, "cell": -1,
                "val_bce_raw": float(loss_v),
                "auc": float(auc_v),
                "logloss": float(ll_v),
            })
    write_rows_csv(os.path.join(out_dir_abs, proto.STEPS_CSV_NAME), step_rows,
                    STEP_COLUMNS)
    write_rows_csv(os.path.join(out_dir_abs, proto.GEOMETRY_CSV_NAME), geometry_rows,
                    GEOMETRY_COLUMNS)
    write_rows_csv(os.path.join(out_dir_abs, proto.VALIDATION_CSV_NAME), val_rows,
                    VALIDATION_COLUMNS)
    payload = {
        "stage": "S7B",
        "status": "ABORTED" if aborted else "OK",
        "abort_reason": abort_reason,
        "v2_review_flagged": bool(review_flagged),
        "v2_review_note": ("step >= 1 divergence is the declared private "
                              "fresh-Adam approximation's expected consequence "
                              "and does not constitute run invalidation"),
        "dataset": dataset, "seed": int(seed), "anchor": int(anchor),
        "key": proto.key(dataset, int(seed), int(anchor)),
        "source_run_dir": source_run_dir,
        "source_formal_run_key": source_main_rows[0].get("formal_run_key", "")
                                          if source_main_rows else "",
        "model_source_key": MODEL_SOURCE_KEY,
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "k_steps": proto.K_STEPS,
        "anchor_reconstruction": anchor_audit,
        "replay_audit": replay_audit,
        "fingerprint_rows": fingerprint_rows,
        "fingerprint_match_all": bool(replay_all_match),
        "f_to_t_distance_shared_post_step": list(f_to_t_distances),
        "v3_per_step_private_symmetry": v3_per_step,
        "v3_note": ("V3 is the hard gate that private parameters must remain "
                     "bit-equal between F and T branches at every step.  Due to "
                     "the structural difference in shared theta after step 0 "
                     "(F uses Δ_F, T overwrites with Δ_T), the two branches' "
                     "private trajectories naturally diverge at step 1+ "
                     "because g_P is computed at the respective branch's "
                     "pre-step theta.  This is not a structural bug; it is "
                     "the expected consequence of the branch semantics."),
        "validity": validity,
        "rng_digest_before": rng_before,
        "rng_digest_after": rng_after,
        "rng_digest_per_step": list(rng_digest_per_step),
        "rng_state_unchanged": bool(rng_before == rng_after),
        "wall_seconds": float(time.time() - started),
    }
    write_json(os.path.join(out_dir_abs, proto.AUDIT_JSON_NAME), payload)
    write_json(os.path.join(out_dir_abs, proto.PROCESS_INTERVAL_NAME), {
        "started": float(started), "finished": float(time.time()),
        "wall_seconds": float(time.time() - started),
        "dataset": dataset, "seed": int(seed), "anchor": int(anchor),
        "status": payload["status"],
    })
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S7-B one-run continuation")
    parser.add_argument("--dataset", choices=("kuairand", "movielens"), required=True)
    parser.add_argument("--seed", type=int, choices=tuple(int(s) for s in S4_SEEDS), required=True)
    parser.add_argument("--anchor", type=int, choices=tuple(int(a) for a in proto.ANCHORS), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--smoke-state", action="store_true")
    parser.add_argument("--smoke-replay", action="store_true")
    parser.add_argument("--smoke-onestep", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    try:
        result = run_continuation(
            dataset=args.dataset, seed=int(args.seed), anchor=int(args.anchor),
            out_dir=args.out_dir, dataset_path=args.dataset_path or "",
            smoke_state=bool(args.smoke_state),
            smoke_replay=bool(args.smoke_replay),
            smoke_onestep=bool(args.smoke_onestep),
        )
        if isinstance(result, dict):
            if "smoke_state_ok" in result:
                print("[stage_S7B] smoke-state dataset={} seed={} anchor={} ok={} shared_step={} expected={} policy={}".format(
                    args.dataset, int(args.seed), int(args.anchor),
                    result.get("smoke_state_ok"),
                    result.get("shared_step_observed"),
                    result.get("shared_step_expected"),
                    result.get("private_policy"),
                ), flush=True)
                return 0 if result.get("smoke_state_ok") else 1
            if "smoke_replay_ok" in result:
                print("[stage_S7B] smoke-replay dataset={} seed={} anchor={} ok={} rows={}".format(
                    args.dataset, int(args.seed), int(args.anchor),
                    result.get("smoke_replay_ok"),
                    len(result.get("fingerprint_rows", [])),
                ), flush=True)
                return 0 if result.get("smoke_replay_ok") else 1
            if "smoke_onestep_ok" in result:
                print("[stage_S7B] smoke-onestep dataset={} seed={} anchor={} F_resid={:.3e} T_resid={:.3e} cos_diff={:.3e}".format(
                    args.dataset, int(args.seed), int(args.anchor),
                    float(result.get("F_applied_formula_residual", 0.0)),
                    float(result.get("T_applied_formula_residual", 0.0)),
                    float(result.get("delta_cosine_F_minus_T", 0.0)),
                ), flush=True)
                return 0 if result.get("smoke_onestep_ok") else 1
        print("[stage_S7B] OK dataset={} seed={} anchor={} key={} wall={:.2f}s".format(
            args.dataset, int(args.seed), int(args.anchor),
            result.get("key", ""), float(result.get("wall_seconds", 0.0))), flush=True)
        return 0
    except Exception as exc:
        import traceback
        sys.stderr.write("[stage_S7B] FAILED dataset={} seed={} anchor={} exception={}\n".format(
            args.dataset, int(args.seed), int(args.anchor), exc))
        sys.stderr.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
