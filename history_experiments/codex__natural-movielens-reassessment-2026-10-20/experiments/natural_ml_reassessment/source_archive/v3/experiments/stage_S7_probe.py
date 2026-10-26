"""Stage S7-A: sample-objective MDMT Domain x Task gradient-dynamics probe.

Replays deterministic probe batches on a frozen Sample FULL checkpoint (from
the Stage S6 F arm) to compute per (source, target, shared_module) gradient
geometry.  Reuses S1 / S3 / S4 / E1 helpers; never mutates training state.

Per Algorithm Specification -- Stage S7-A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M3OE_ROOT = os.path.join(ROOT, "m3oe")
EXAMPLES_ROOT = os.path.join(M3OE_ROOT, "examples")
for _path in (HERE, M3OE_ROOT, EXAMPLES_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stage_E1_instrumentation import (  # noqa: E402
    batch_fingerprint,
    cell_index,
    masked_bce_terms,
    rng_state_digest,
    write_json,
    write_rows_csv,
)
from stage_S1_data import load_stage_s1_bundle  # noqa: E402
from stage_S1_models import build_model, initial_state_hash  # noqa: E402
from stage_S1_run import direct_per_example_bce, setup_seed  # noqa: E402
from stage_S3_gradient_access import (  # noqa: E402
    GROUP_SHARED,
    assert_expected_parameter_counts,
    partition_parameters,
)
from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256  # noqa: E402
from stage_S4_protocol import SEEDS as S4_SEEDS  # noqa: E402

# Frozen Stage S7 constants (Algorithm Specification section 4).
T_PROBE = 4
BATCH_SIZE = 40960
COMMON_EPOCHS: Tuple[int, ...] = (9, 19, 29, 39, 49)
CHECKPOINT_LABEL_ORDER: Tuple[str, ...] = (
    "epoch_9", "epoch_19", "epoch_29", "epoch_39", "epoch_49", "selected",
)
CHECKPOINT_FILENAMES: Mapping[str, str] = {
    "epoch_9": "common_epoch_9.pth",
    "epoch_19": "common_epoch_19.pth",
    "epoch_29": "common_epoch_29.pth",
    "epoch_39": "common_epoch_39.pth",
    "epoch_49": "common_epoch_49.pth",
    "selected": "selected_checkpoint.pth",
}
SHARED_MODULE_OF: Tuple[Tuple[str, Optional[Tuple[str, ...]]], ...] = (
    ("ALL", None),
    ("shared_scalars", ("shared_weight", "shared_bias")),
    ("embedding", ("embedding.",)),
    ("skip_conn", ("skip_conn.",)),
    ("star_mlp", ("star_mlp.",)),
    ("expert_shared", ("expert.",)),
    ("domain_expert", ("domain_expert.",)),
    ("task_expert", ("task_expert.",)),
)
MODULES: Tuple[str, ...] = tuple(entry[0] for entry in SHARED_MODULE_OF)

CHECKPOINT_LEVEL_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "setting", "seed", "checkpoint", "stage_epoch",
    "selected_coincide", "source_domain", "source_task", "source_cell",
    "target_domain", "target_task", "target_cell", "pair_class",
    "shared_module",
    "gradient_norm_mean", "gradient_norm_sd",
    "pairwise_cosine_mean", "pairwise_cosine_sd",
    "sample_alignment_mean", "sample_alignment_sd",
    "validation_dot_product_mean",
    "weighted_source_target_contribution_mean",
    "weighted_source_target_contribution_sd",
    "conflict_contribution_mean",
    "source_weight_mean",
    "validation_gradient_norm",
    "reconstruction_residual_max",
    "batch_count",
)
BATCH_LEVEL_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "setting", "seed", "checkpoint", "stage_epoch",
    "selected_coincide", "source_domain", "source_task", "source_cell",
    "target_domain", "target_task", "target_cell", "pair_class",
    "shared_module",
    "batch_index", "batch_fingerprint", "domain_counts",
    "gradient_norm_mean",
    "pairwise_cosine_mean",
    "sample_alignment_mean",
    "validation_dot_product_mean",
    "weighted_source_target_contribution_mean",
    "conflict_contribution_mean",
    "source_weight_mean",
    "validation_gradient_norm",
    "reconstruction_residual_max",
)
TASK_NUM = 2
DOMAIN_NUM = 3
CELL_NUM = TASK_NUM * DOMAIN_NUM
S4_TOLERANCE_ABS = 1e-6
S4_TOLERANCE_REL = 1e-4


def pair_class(source_cell: int, target_cell: int) -> str:
    if source_cell == target_cell:
        return "self"
    s_d, s_t = source_cell % DOMAIN_NUM, source_cell // DOMAIN_NUM
    t_d, t_t = target_cell % DOMAIN_NUM, target_cell // DOMAIN_NUM
    if s_d == t_d:
        return "same_domain_cross_task"
    if s_t == t_t:
        return "cross_domain_same_task"
    return "cross_domain_cross_task"


def shared_module_of(name: str) -> str:
    name = str(name)
    for module, prefixes in SHARED_MODULE_OF:
        if module == "ALL":
            continue
        if prefixes is None:
            continue
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix):
                return module
    raise ValueError("Stage S7 cannot classify shared parameter: {!r}".format(name))


def module_slices(shared_names: Sequence[str],
                  parameters: Mapping[str, torch.nn.Parameter]) -> Dict[str, Tuple[int, int]]:
    slices: Dict[str, Tuple[int, int]] = {module: (0, 0) for module in MODULES}
    running = 0
    for name in shared_names:
        size = int(parameters[name].numel())
        module = shared_module_of(name)
        prev_start, prev_end = slices[module]
        if prev_end == 0 and prev_start == 0:
            slices[module] = (running, running + size)
        else:
            slices[module] = (prev_start, prev_end + size)
        all_start, all_end = slices["ALL"]
        slices["ALL"] = (all_start, all_end + size)
        running += size
    total = running
    expected_total = sum(int(parameters[name].numel()) for name in shared_names)
    if total != expected_total:
        raise AssertionError("module_slices total {} != expected {}".format(total, expected_total))
    for module in MODULES:
        s, e = slices[module]
        if e < s:
            raise AssertionError("empty/inverted slice for module {}: ({},{})".format(module, s, e))
    all_start, all_end = slices["ALL"]
    if all_start != 0 or all_end != total:
        raise AssertionError("ALL slice ({},{}) must span (0,{})".format(
            all_start, all_end, total))
    for module in MODULES:
        if module == "ALL":
            continue
        s, e = slices[module]
        if s >= total or e > total:
            raise AssertionError("module {} slice ({},{}) out of bounds".format(module, s, e))
    for a, b in [(x, y) for x in MODULES for y in MODULES if x != y and x != "ALL" and y != "ALL"]:
        sa, ea = slices[a]
        sb, eb = slices[b]
        if max(sa, sb) < min(ea, eb):
            raise AssertionError("module slices {} and {} overlap".format(a, b))
    return slices


def derive_probe_batches(bundle, dataset: str, seed: int, t: int = T_PROBE,
                         batch_size: int = BATCH_SIZE) -> List[Dict[str, object]]:
    payload = "s7probe|{}|{}".format(dataset, seed).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    rng_seed = int.from_bytes(digest[:8], "big") % (2 ** 31 - 1)
    rng = np.random.RandomState(rng_seed)
    train_x = np.ascontiguousarray(bundle.train_x.astype(np.float32))
    train_y = np.ascontiguousarray(bundle.train_y.astype(np.float32))
    train_domains = train_x[:, -1].astype(np.int64)
    perm = rng.permutation(len(train_x))
    total_needed = t * batch_size
    if total_needed > len(train_x):
        raise ValueError("probe batch total {} exceeds train size {}".format(total_needed, len(train_x)))
    selected_perm = perm[:total_needed]
    batches: List[Dict[str, object]] = []
    two_b = float(2 * batch_size)
    for batch_index in range(t):
        left = batch_index * batch_size
        right = left + batch_size
        rows = selected_perm[left:right]
        x = torch.from_numpy(train_x[rows])
        y = torch.from_numpy(train_y[rows])
        domain_ids = train_domains[rows]
        domain_counts = [int((domain_ids == d).sum()) for d in range(DOMAIN_NUM)]
        if any(count == 0 for count in domain_counts):
            raise ValueError("probe batch {} missing one of three domains: {}".format(
                batch_index, domain_counts))
        weights = {cell_index(task, d, DOMAIN_NUM): float(domain_counts[d]) / two_b
                   for task in range(TASK_NUM) for d in range(DOMAIN_NUM)}
        fingerprint = batch_fingerprint(x, y)
        batches.append({
            "x": x,
            "y": y,
            "fingerprint": fingerprint,
            "domain_counts": domain_counts,
            "w": weights,
        })
    return batches


def flatten_shared_grad(loss: torch.Tensor, shared_params: Sequence[torch.nn.Parameter],
                        retain_graph: bool = True) -> torch.Tensor:
    grads = torch.autograd.grad(loss, list(shared_params), retain_graph=retain_graph,
                                allow_unused=True)
    pieces: List[torch.Tensor] = []
    for grad, param in zip(grads, shared_params):
        if grad is None:
            raise FloatingPointError(
                "Stage S7 shared gradient returned None for a tensor that participates "
                "in the forward graph; missing gradient is not allowed here.")
        pieces.append(grad.detach().reshape(-1).to(dtype=torch.float32, device=param.device))
    if not pieces:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(pieces)


def train_cell_gradients(model: torch.nn.Module, batch: Mapping[str, object],
                         shared_params: Sequence[torch.nn.Parameter],
                         device: torch.device) -> Dict[str, object]:
    shared_params = list(shared_params)
    x_cpu = batch["x"]
    y_cpu = batch["y"]
    x_device = x_cpu.to(device=device, dtype=torch.float32)
    y_device = y_cpu.to(device=device, dtype=torch.float32)
    domain_ids = x_cpu[:, -1].to(device=device, dtype=torch.long)
    pred = model(x_device.transpose(0, 1))
    if not bool(torch.isfinite(pred).all().detach().cpu()):
        raise FloatingPointError("Stage S7 non-finite probe prediction")
    terms, counts = masked_bce_terms(pred, y_device, domain_ids, DOMAIN_NUM, TASK_NUM)
    if any(c == 0 for c in counts):
        raise ValueError("probe batch missing one of three domains")
    g_cell: Dict[int, torch.Tensor] = {}
    cell_losses: Dict[int, float] = {}
    for idx, term in enumerate(terms):
        g_cell[idx] = flatten_shared_grad(term, shared_params, retain_graph=True)
        cell_losses[idx] = float(term.detach().cpu())
    full_loss = direct_per_example_bce(pred, y_device)
    g_full = flatten_shared_grad(full_loss, shared_params, retain_graph=False)
    weights = batch["w"]
    combined = torch.zeros_like(g_full)
    for idx in range(CELL_NUM):
        combined = combined + float(weights[idx]) * g_cell[idx]
    diff = g_full - combined
    g_full_norm = float(torch.linalg.vector_norm(g_full).detach().cpu())
    absolute_residual = float(torch.linalg.vector_norm(diff).detach().cpu())
    if absolute_residual > max(S4_TOLERANCE_ABS, S4_TOLERANCE_REL * g_full_norm):
        raise AssertionError(
            "Stage S7 reconstruction residual {} exceeds S4 tolerance (abs={}, rel={}*{:.3e}) "
            "for batch {}".format(absolute_residual, S4_TOLERANCE_ABS, S4_TOLERANCE_REL,
                                  g_full_norm, batch.get("fingerprint", "?")))
    return {"g_cell": g_cell, "g_full": g_full, "losses": cell_losses,
            "recon_residual": absolute_residual}


def validation_cell_gradients(model: torch.nn.Module, val_x, val_y,
                              shared_params: Sequence[torch.nn.Parameter],
                              device: torch.device,
                              chunk: int = BATCH_SIZE) -> Dict[int, torch.Tensor]:
    shared_params = list(shared_params)
    model.train()
    val_x_cpu = np.ascontiguousarray(np.asarray(val_x).astype(np.float32))
    val_y_cpu = np.ascontiguousarray(np.asarray(val_y).astype(np.float32))
    total = len(val_x_cpu)
    if total == 0:
        raise ValueError("Stage S7 requires a non-empty validation split")
    chunk = int(chunk)
    if chunk <= 0:
        raise ValueError("Stage S7 validation chunk size must be positive")
    total_dim = sum(int(p.numel()) for p in shared_params)
    if total_dim == 0:
        raise ValueError("Stage S7 shared parameter vector is empty")
    accumulators: Dict[int, torch.Tensor] = {
        idx: torch.zeros(total_dim, dtype=torch.float32, device=device)
        for idx in range(CELL_NUM)
    }
    counts_by_domain: Dict[int, int] = {d: 0 for d in range(DOMAIN_NUM)}
    starts = list(range(0, total, chunk))
    if starts[-1] != total:
        starts.append(total)
    for left, right in zip(starts[:-1], starts[1:]):
        if left >= right:
            continue
        rows = slice(int(left), int(right))
        x_chunk = torch.from_numpy(val_x_cpu[rows]).to(device=device, dtype=torch.float32)
        y_chunk = torch.from_numpy(val_y_cpu[rows]).to(device=device, dtype=torch.float32)
        domain_ids = x_chunk[:, -1].to(device=device, dtype=torch.long)
        pred = model(x_chunk.transpose(0, 1))
        if not bool(torch.isfinite(pred).all().detach().cpu()):
            raise FloatingPointError("Stage S7 non-finite validation prediction")
        terms, counts = masked_bce_terms(pred, y_chunk, domain_ids, DOMAIN_NUM, TASK_NUM)
        for d in range(DOMAIN_NUM):
            counts_by_domain[d] += int(counts[d])
        for idx, term in enumerate(terms):
            d = idx % DOMAIN_NUM
            count = int(counts[d])
            if count == 0:
                continue
            grad = flatten_shared_grad(term, shared_params, retain_graph=True).to(device=device)
            accumulators[idx] = accumulators[idx] + float(count) * grad
    grads: Dict[int, torch.Tensor] = {}
    for idx in range(CELL_NUM):
        d = idx % DOMAIN_NUM
        n_d = counts_by_domain[d]
        if n_d == 0:
            raise ValueError(
                "Stage S7 validation split is missing domain {}; aborting instead of "
                "emitting a zero gradient".format(d))
        grads[idx] = (accumulators[idx] / float(n_d)).detach()
    return grads


def _to_fp64_cpu(flat: torch.Tensor) -> torch.Tensor:
    return flat.detach().to(dtype=torch.float64).cpu().reshape(-1).contiguous()


def _norm_fp64(vec: torch.Tensor) -> float:
    if vec.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(vec).detach().cpu())


def _dot_fp64(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float(torch.dot(a, b).detach().cpu())


def _cosine_fp64(a: torch.Tensor, b: torch.Tensor) -> float:
    na = _norm_fp64(a)
    nb = _norm_fp64(b)
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return max(-1.0, min(1.0, _dot_fp64(a, b) / (na * nb)))


def _slice_module(vec_fp64: torch.Tensor, slices: Mapping[str, Tuple[int, int]],
                  module: str) -> torch.Tensor:
    start, end = slices[module]
    if end <= start:
        return torch.zeros(0, dtype=torch.float64)
    return vec_fp64[start:end].clone()


def probe_checkpoint(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor],
                     batches: Sequence[Mapping[str, object]],
                     val_grads: Mapping[int, torch.Tensor],
                     slices: Mapping[str, Tuple[int, int]],
                     device: torch.device,
                     shared_names: Optional[Sequence[str]] = None,
                     shared_params: Optional[Sequence[torch.nn.Parameter]] = None
                     ) -> Tuple[List[Dict[str, object]],
                                List[Dict[str, object]], float]:
    model.load_state_dict(state_dict, strict=True)
    model.train()
    if shared_params is None:
        if shared_names is None:
            raise ValueError("probe_checkpoint requires shared_names or shared_params")
        named_params = dict(model.named_parameters())
        shared_params = [named_params[name] for name in shared_names]
    val_vec_fp64 = {idx: _to_fp64_cpu(val_grads[idx].detach().to(device=device))
                    for idx in range(CELL_NUM)}
    val_norm_fp64: Dict[Tuple[int, str], float] = {}
    for idx in range(CELL_NUM):
        for module in MODULES:
            val_norm_fp64[(idx, module)] = _norm_fp64(_slice_module(val_vec_fp64[idx], slices, module))
    batch_rows: List[Dict[str, object]] = []
    per_pair: Dict[Tuple[int, int, str], List[Dict[str, float]]] = {}
    fingerprint_per_batch = [b["fingerprint"] for b in batches]
    domain_counts_per_batch = [b["domain_counts"] for b in batches]
    weights_per_batch = [b["w"] for b in batches]
    info = [train_cell_gradients(model, batch, shared_params, device) for batch in batches]
    cell_vecs_fp64 = [[_to_fp64_cpu(info[b]["g_cell"][idx].to(device=device))
                       for idx in range(CELL_NUM)] for b in range(len(batches))]
    full_vecs_fp64 = [_to_fp64_cpu(info[b]["g_full"].to(device=device)) for b in range(len(batches))]
    recon_per_batch = [float(info[b]["recon_residual"]) for b in range(len(batches))]
    sample_grad_norm_per_batch = [float(_norm_fp64(full_vecs_fp64[b])) for b in range(len(batches))]
    sample_grad_norm_max = max(sample_grad_norm_per_batch) if sample_grad_norm_per_batch else 0.0
    for batch_index in range(len(batches)):
        domain_counts = domain_counts_per_batch[batch_index]
        for s in range(CELL_NUM):
            for t in range(CELL_NUM):
                pc = pair_class(s, t)
                cs_full = _cosine_fp64(cell_vecs_fp64[batch_index][s],
                                       cell_vecs_fp64[batch_index][t])
                if s == t:
                    cc_full = 0.0
                else:
                    cs_safe = cs_full if math.isfinite(cs_full) else 0.0
                    cc_full = max(0.0, -cs_safe) * float(weights_per_batch[batch_index][s]) \
                        * float(weights_per_batch[batch_index][t])
                for module in MODULES:
                    s_slice = _slice_module(cell_vecs_fp64[batch_index][s], slices, module)
                    t_slice = _slice_module(cell_vecs_fp64[batch_index][t], slices, module)
                    full_slice = full_vecs_fp64[batch_index] if module == "ALL" \
                        else _slice_module(full_vecs_fp64[batch_index], slices, module)
                    val_slice = val_vec_fp64[t] if module == "ALL" \
                        else _slice_module(val_vec_fp64[t], slices, module)
                    norm_s = _norm_fp64(s_slice)
                    norm_t = _norm_fp64(t_slice)
                    cos_st = _cosine_fp64(s_slice, t_slice)
                    if not math.isfinite(cos_st):
                        cos_st = 0.0
                    align_s = _cosine_fp64(s_slice, full_slice)
                    if not math.isfinite(align_s):
                        align_s = 0.0
                    u_t = _dot_fp64(full_slice, val_slice)
                    h_st = float(weights_per_batch[batch_index][s]) * _dot_fp64(s_slice, val_slice)
                    cs_for_module = cos_st
                    if s != t:
                        cs_for_module = max(-1.0, min(1.0, cos_st))
                        cc_module = max(0.0, -cs_for_module) * float(weights_per_batch[batch_index][s]) \
                            * float(weights_per_batch[batch_index][t])
                    else:
                        cc_module = 0.0
                    row = {
                        "batch_index": batch_index,
                        "batch_fingerprint": fingerprint_per_batch[batch_index],
                        "domain_counts": json.dumps([int(c) for c in domain_counts]),
                        "gradient_norm_mean": float(norm_s),
                        "pairwise_cosine_mean": float(cs_for_module),
                        "sample_alignment_mean": float(align_s),
                        "validation_dot_product_mean": float(u_t),
                        "weighted_source_target_contribution_mean": float(h_st),
                        "conflict_contribution_mean": float(cc_module),
                        "source_weight_mean": float(weights_per_batch[batch_index][s]),
                        "validation_gradient_norm": float(val_norm_fp64[(t, module)]),
                        "reconstruction_residual_max": float(recon_per_batch[batch_index]),
                    }
                    per_pair.setdefault((s, t, module), []).append({**row, "pair_class": pc})
                    batch_rows.append({
                        **row,
                        "stage": "S7",
                        "dataset": "", "setting": "", "seed": 0,
                        "checkpoint": "", "stage_epoch": 0,
                        "selected_coincide": 0,
                        "source_domain": s % DOMAIN_NUM, "source_task": s // DOMAIN_NUM,
                        "source_cell": s,
                        "target_domain": t % DOMAIN_NUM, "target_task": t // DOMAIN_NUM,
                        "target_cell": t,
                        "pair_class": pc,
                        "shared_module": module,
                    })
    checkpoint_rows: List[Dict[str, object]] = []
    for (s, t, module), rows in per_pair.items():
        def stats(field: str) -> Tuple[float, float]:
            arr = np.asarray([r[field] for r in rows], dtype=np.float64)
            if not np.isfinite(arr).all():
                arr = np.nan_to_num(arr, nan=0.0)
            mean = float(arr.mean())
            sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            return mean, sd
        norm_m, sd_m = stats("gradient_norm_mean")
        cos_m, cos_sd = stats("pairwise_cosine_mean")
        align_m, align_sd = stats("sample_alignment_mean")
        h_m, h_sd = stats("weighted_source_target_contribution_mean")
        u_m, _ = stats("validation_dot_product_mean")
        cc_m, _ = stats("conflict_contribution_mean")
        w_m, _ = stats("source_weight_mean")
        val_norm = float(rows[0]["validation_gradient_norm"])
        recon = float(max(r["reconstruction_residual_max"] for r in rows))
        pc = rows[0]["pair_class"]
        checkpoint_rows.append({
            "stage": "S7",
            "dataset": "", "setting": "", "seed": 0,
            "checkpoint": "", "stage_epoch": 0,
            "selected_coincide": 0,
            "source_domain": s % DOMAIN_NUM, "source_task": s // DOMAIN_NUM, "source_cell": s,
            "target_domain": t % DOMAIN_NUM, "target_task": t // DOMAIN_NUM, "target_cell": t,
            "pair_class": pc, "shared_module": module,
            "gradient_norm_mean": norm_m, "gradient_norm_sd": sd_m,
            "pairwise_cosine_mean": cos_m, "pairwise_cosine_sd": cos_sd,
            "sample_alignment_mean": align_m, "sample_alignment_sd": align_sd,
            "validation_dot_product_mean": u_m,
            "weighted_source_target_contribution_mean": h_m,
            "weighted_source_target_contribution_sd": h_sd,
            "conflict_contribution_mean": cc_m,
            "source_weight_mean": w_m,
            "validation_gradient_norm": val_norm,
            "reconstruction_residual_max": recon,
            "batch_count": len(rows),
        })
    return checkpoint_rows, batch_rows, float(sample_grad_norm_max)


def _hash_file(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_run_manifest(source_run_dir: str) -> Dict[str, object]:
    path = os.path.join(source_run_dir, "run_manifest.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_probe(dataset: str, seed: int, source_run_dir: str, out_dir: str,
              dataset_path: Optional[str] = None,
              smoke_reconstruction: bool = False,
              smoke_checkpoint: bool = False,
              probe_batch_count: Optional[int] = None,
              val_chunk_count: Optional[int] = None) -> Dict[str, object]:
    started = time.time()
    setup_seed(seed)
    if seed not in S4_SEEDS:
        raise ValueError("seed {} outside S4 matrix {}".format(seed, S4_SEEDS))
    source_manifest = _read_run_manifest(source_run_dir)
    initial_hash_expected = source_manifest.get("initial_parameter_state_hash", "")
    if not initial_hash_expected:
        raise ValueError("source run_manifest lacks initial_parameter_state_hash")
    if str(source_manifest.get("dataset_key", "")) != dataset:
        raise ValueError("source dataset_key {} != requested {}".format(
            source_manifest.get("dataset_key"), dataset))
    if int(source_manifest.get("seed", -1)) != int(seed):
        raise ValueError("source seed {} != requested {}".format(
            source_manifest.get("seed"), seed))
    setting = source_manifest.get("setting")
    if not setting:
        raise ValueError("source run_manifest lacks setting")
    bundle = load_stage_s1_bundle(dataset, dataset_path, str(setting), int(seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(bundle.features, DOMAIN_NUM, TASK_NUM, "m3oe",
                        expert_num=4, fcn_dims=(512, 256, 256, 64), seed=seed)
    model = model.to(device)
    initial_hash_observed = initial_state_hash(model)
    if initial_hash_observed != initial_hash_expected:
        raise AssertionError(
            "Stage S7 initial hash mismatch: observed {} != expected {}".format(
                initial_hash_observed, initial_hash_expected))
    partition = partition_parameters(model)
    assert_expected_parameter_counts(partition, dataset)
    shared_names = list(partition["groups"][GROUP_SHARED])
    parameters = dict(model.named_parameters())
    for name in shared_names:
        if name not in parameters:
            raise AssertionError("shared name {} not in model".format(name))
    shared_params = [parameters[name] for name in shared_names]
    slices = module_slices(shared_names, parameters)
    t_use = max(1, min(T_PROBE, int(probe_batch_count) if probe_batch_count else T_PROBE))
    batches = derive_probe_batches(bundle, dataset, seed, t=t_use, batch_size=BATCH_SIZE)
    out_dir_abs = os.path.abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    for name in os.listdir(out_dir_abs):
        if name in ("failed_attempts", "stdout.log", "process_interval.json"):
            continue
        if name.startswith("gradient_probe") or name == "probe_manifest.json":
            raise FileExistsError(
                "Stage S7 refusing to overwrite non-empty run directory: {}".format(out_dir_abs))
    batch_fingerprints = [str(b["fingerprint"]) for b in batches]
    fingerprint_digest = hashlib.sha256(
        json.dumps(batch_fingerprints).encode("utf-8")).hexdigest()
    selected_epoch = int(source_manifest.get("selected_epoch", -1))
    if selected_epoch < 0:
        raise ValueError("source manifest has no selected_epoch")
    selected_overlap = selected_epoch in COMMON_EPOCHS
    rng_digests: Dict[str, Dict[str, str]] = {}
    checkpoint_summaries: List[Dict[str, object]] = []
    checkpoint_batch_rows: List[Dict[str, object]] = []
    checkpoint_meta: List[Dict[str, object]] = []
    for label in CHECKPOINT_LABEL_ORDER:
        filename = CHECKPOINT_FILENAMES[label]
        ckpt_path = os.path.join(source_run_dir, filename)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError("Stage S7 missing checkpoint: {}".format(ckpt_path))
        ckpt_hash = _hash_file(ckpt_path)
        is_selected_label = (label == "selected")
        if is_selected_label:
            epoch_to_load = int(selected_epoch)
        else:
            epoch_to_load = int(label.split("_")[1])
        if selected_overlap and is_selected_label:
            common_label = "epoch_{}".format(selected_epoch)
            common_path = os.path.join(source_run_dir, CHECKPOINT_FILENAMES[common_label])
            common_state = torch.load(common_path, map_location="cpu")
            selected_state = torch.load(ckpt_path, map_location="cpu")
            if set(common_state.keys()) != set(selected_state.keys()):
                raise AssertionError("selected/common key set mismatch")
            for key in common_state:
                if not torch.equal(common_state[key], selected_state[key]):
                    raise AssertionError(
                        "selected != common checkpoint at tensor {}".format(key))
            state_dict = common_state
            coincide = 1
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            coincide = 0
        rng_before = rng_state_digest()
        model.load_state_dict(state_dict, strict=True)
        chunk = int(val_chunk_count) if val_chunk_count else BATCH_SIZE
        val_grads = validation_cell_gradients(model, bundle.val_x, bundle.val_y,
                                              shared_params, device, chunk=chunk)
        cp_rows, batch_rows, sample_grad_norm_max = probe_checkpoint(
            model, state_dict, batches, val_grads, slices, device,
            shared_names=shared_names, shared_params=shared_params)
        rng_after = rng_state_digest()
        rng_digests[label] = {"before": rng_before, "after": rng_after,
                              "equal": rng_before == rng_after}
        if rng_before != rng_after:
            raise AssertionError("Stage S7 probe changed RNG state at label {}".format(label))
        for row in cp_rows:
            row["dataset"] = dataset
            row["setting"] = str(setting)
            row["seed"] = int(seed)
            row["checkpoint"] = label
            row["stage_epoch"] = int(epoch_to_load)
            row["selected_coincide"] = int(coincide)
        for row in batch_rows:
            row["dataset"] = dataset
            row["setting"] = str(setting)
            row["seed"] = int(seed)
            row["checkpoint"] = label
            row["stage_epoch"] = int(epoch_to_load)
            row["selected_coincide"] = int(coincide)
        checkpoint_summaries.extend(cp_rows)
        checkpoint_batch_rows.extend(batch_rows)
        checkpoint_meta.append({
            "label": label,
            "filename": filename,
            "sha256": ckpt_hash,
            "stage_epoch": int(epoch_to_load),
            "selected_coincide": int(coincide),
            "reconstruction_residual_max":
                float(max(r["reconstruction_residual_max"] for r in cp_rows)),
            "sample_gradient_norm_max": float(sample_grad_norm_max),
            "rng_digest_before": rng_before,
            "rng_digest_after": rng_after,
            "rng_state_unchanged": rng_before == rng_after,
        })
    write_rows_csv(os.path.join(out_dir_abs, "gradient_probe.csv"),
                   checkpoint_summaries, CHECKPOINT_LEVEL_COLUMNS)
    write_rows_csv(os.path.join(out_dir_abs, "gradient_probe_batches.csv"),
                   checkpoint_batch_rows, BATCH_LEVEL_COLUMNS)
    manifest = {
        "stage": "S7",
        "status": "OK",
        "dataset": dataset,
        "dataset_key": dataset,
        "setting": str(setting),
        "seed": int(seed),
        "source_run_dir": os.path.abspath(source_run_dir),
        "source_formal_run_key": source_manifest.get("formal_run_key", ""),
        "model_source_key": MODEL_SOURCE_KEY,
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "initial_parameter_state_hash_observed": initial_hash_observed,
        "initial_parameter_state_hash_expected": initial_hash_expected,
        "initial_parameter_state_hash_match": initial_hash_observed == initial_hash_expected,
        "shared_tensor_count": int(partition["counts"][GROUP_SHARED]["tensor_count"]),
        "shared_parameter_count": int(partition["counts"][GROUP_SHARED]["parameter_count"]),
        "partition_hash": partition["partition_hash"],
        "modules": list(MODULES),
        "module_slices": {module: list(slices[module]) for module in MODULES},
        "probe_seed_payload": "s7probe|{}|{}".format(dataset, seed),
        "batch_size": int(BATCH_SIZE),
        "probe_batch_count": int(t_use),
        "batch_fingerprints": batch_fingerprints,
        "batch_fingerprint_digest": fingerprint_digest,
        "rng_state_digests": rng_digests,
        "checkpoints": checkpoint_meta,
        "selected_epoch": int(selected_epoch),
        "selected_overlap_with_common": bool(selected_overlap),
        "wall_seconds": float(time.time() - started),
        "smoke_reconstruction": bool(smoke_reconstruction),
        "smoke_checkpoint": bool(smoke_checkpoint),
    }
    write_json(os.path.join(out_dir_abs, "probe_manifest.json"), manifest)
    write_json(os.path.join(out_dir_abs, "process_interval.json"), {
        "started": float(started),
        "finished": float(time.time()),
        "wall_seconds": float(time.time() - started),
        "dataset": dataset,
        "seed": int(seed),
    })
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S7 single-seed probe")
    parser.add_argument("--dataset", choices=("kuairand", "movielens"), required=True)
    parser.add_argument("--seed", type=int, choices=S4_SEEDS, required=True)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--smoke-reconstruction", action="store_true")
    parser.add_argument("--smoke-checkpoint", action="store_true")
    parser.add_argument("--probe-batch-count", type=int, default=None)
    parser.add_argument("--val-chunk-count", type=int, default=None)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    try:
        manifest = run_probe(
            dataset=args.dataset, seed=int(args.seed),
            source_run_dir=args.source_run_dir, out_dir=args.out_dir,
            dataset_path=args.dataset_path or None,
            smoke_reconstruction=bool(args.smoke_reconstruction),
            smoke_checkpoint=bool(args.smoke_checkpoint),
            probe_batch_count=args.probe_batch_count,
            val_chunk_count=args.val_chunk_count,
        )
        print("[stage_S7] OK dataset={} seed={} checkpoints={} wall={:.2f}s".format(
            args.dataset, int(args.seed), len(manifest["checkpoints"]),
            float(manifest.get("wall_seconds", 0.0))), flush=True)
        return 0
    except Exception as exc:
        import traceback
        sys.stderr.write("[stage_S7] FAILED dataset={} seed={} exception={}\n".format(
            args.dataset, int(args.seed), exc))
        sys.stderr.write(traceback.format_exc())
        return 1


def run_s7_reconstruction_smoke() -> Dict[str, object]:
    """Tiny in-memory smoke for the gradient decomposition contract."""
    from stage_S3_run import _smoke_feature_schema

    setup_seed(2022)
    schema = _smoke_feature_schema("kuairand")
    model = build_model(schema, DOMAIN_NUM, TASK_NUM, "m3oe",
                        expert_num=4, fcn_dims=(512, 256, 256, 64), seed=2022)
    part = partition_parameters(model)
    shared_names = list(part["groups"][GROUP_SHARED])
    parameters = dict(model.named_parameters())
    slices = module_slices(shared_names, parameters)
    shared_params = [parameters[name] for name in shared_names]

    expected = set(MODULES) - {"ALL"}
    seen = set()
    for name in shared_names:
        seen.add(shared_module_of(name))
    if seen != expected:
        raise AssertionError("shared_module_of missing modules: {} vs {}".format(seen, expected))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    B = 8
    x = torch.zeros(B, len(schema.columns), dtype=torch.float32)
    for i in range(B):
        x[i, 0] = float(i) / B
        x[i, 1] = float(i % 3)
        x[i, -1] = float(i % 3)
    y = torch.zeros(B, TASK_NUM, dtype=torch.float32)
    y[:, 0] = torch.tensor([(i % 2) for i in range(B)], dtype=torch.float32)
    y[:, 1] = torch.tensor([((i + 1) % 2) for i in range(B)], dtype=torch.float32)
    domain_counts = [int((x[:, -1] == d).sum()) for d in range(DOMAIN_NUM)]
    weights = {cell_index(t, d, DOMAIN_NUM): float(domain_counts[d]) / (2 * B)
               for t in range(TASK_NUM) for d in range(DOMAIN_NUM)}
    batch = {"x": x, "y": y, "fingerprint": "smoke_recon",
             "domain_counts": domain_counts, "w": weights}

    info = train_cell_gradients(model, batch, shared_params, device)
    g_full = info["g_full"].detach().to(dtype=torch.float64).cpu()
    g_cell = {idx: info["g_cell"][idx].detach().to(dtype=torch.float64).cpu()
              for idx in range(CELL_NUM)}

    combined = torch.zeros_like(g_full)
    for idx in range(CELL_NUM):
        combined = combined + float(weights[idx]) * g_cell[idx]
    diff_all = float(torch.linalg.vector_norm(g_full - combined).detach().cpu())
    base = float(torch.linalg.vector_norm(g_full).detach().cpu())
    if diff_all > max(S4_TOLERANCE_ABS, S4_TOLERANCE_REL * base):
        raise AssertionError("smoke reconstruction residual (ALL) {} exceeds tolerance".format(diff_all))
    for module in MODULES:
        s, e = slices[module]
        if e <= s:
            continue
        sub_diff = float(torch.linalg.vector_norm(
            g_full[s:e] - combined[s:e]).detach().cpu())
        sub_base = float(torch.linalg.vector_norm(g_full[s:e]).detach().cpu())
        if sub_diff > max(S4_TOLERANCE_ABS, S4_TOLERANCE_REL * sub_base):
            raise AssertionError(
                "smoke reconstruction residual ({}) {} exceeds tolerance".format(module, sub_diff))

    for i in range(CELL_NUM):
        vi = g_cell[i]
        for j in range(CELL_NUM):
            vj = g_cell[j]
            na = float(torch.linalg.vector_norm(vi))
            nb = float(torch.linalg.vector_norm(vj))
            if na > 0 and nb > 0:
                c_ij = float(torch.dot(vi, vj) / (na * nb))
                c_ji = float(torch.dot(vj, vi) / (nb * na))
                if abs(c_ij - c_ji) > 1e-7:
                    raise AssertionError("cosine asymmetry ({},{})".format(i, j))
                if i == j and abs(c_ij - 1.0) > 1e-6:
                    raise AssertionError("cosine diagonal not 1: ({},{})={}".format(i, j, c_ij))

    val_x = torch.cat([x.clone() for _ in range(4)], dim=0)
    val_y = torch.cat([y.clone() for _ in range(4)], dim=0)
    val_grads = validation_cell_gradients(model, val_x, val_y, shared_params, device,
                                          chunk=8)
    for k in range(CELL_NUM):
        u_k = torch.zeros_like(g_full)
        u_k = u_k  # noqa
        val_grad = val_grads[k].detach().to(dtype=torch.float64).cpu()
        h_sum = torch.zeros_like(g_full)
        for i in range(CELL_NUM):
            h_sum = h_sum + float(weights[i]) * g_cell[i]
        u_total = float(torch.dot(g_full, val_grad).detach().cpu())
        h_total = float(torch.dot(h_sum, val_grad).detach().cpu())
        if abs(u_total - h_total) > max(S4_TOLERANCE_ABS, S4_TOLERANCE_REL * abs(u_total)):
            raise AssertionError(
                "H sum identity violated for k={}: U={} H_sum={}".format(k, u_total, h_total))
    return {"smoke_recon_residual_all": diff_all,
            "smoke_recon_residual_base": base,
            "modules_covered": sorted(seen)}


def run_s7_checkpoint_smoke(dataset: str, seed: int) -> Dict[str, object]:
    """Real-checkpoint probe (epoch_9 only, T=1 truncated batch, val 2 chunks)."""
    dataset_root = {
        "kuairand": ("stage_S6_raw_results", "F"),
        "movielens": ("stage_S6ML_raw_results", "F"),
    }
    if dataset not in dataset_root:
        raise ValueError("dataset must be kuairand or movielens")
    raw_root, arm = dataset_root[dataset]
    source_run_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        raw_root, arm, "seed{}".format(int(seed)), "joint")
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError("Stage S7 source run missing: {}".format(source_run_dir))
    setup_seed(seed)
    source_manifest = _read_run_manifest(source_run_dir)
    initial_hash_expected = source_manifest.get("initial_parameter_state_hash", "")
    bundle = load_stage_s1_bundle(dataset, None, source_manifest["setting"], int(seed))
    model = build_model(bundle.features, DOMAIN_NUM, TASK_NUM, "m3oe",
                        expert_num=4, fcn_dims=(512, 256, 256, 64), seed=seed)
    initial_hash_observed = initial_state_hash(model)
    if initial_hash_observed != initial_hash_expected:
        raise AssertionError("smoke initial hash mismatch")
    partition = partition_parameters(model)
    assert_expected_parameter_counts(partition, dataset)
    shared_names = list(partition["groups"][GROUP_SHARED])
    parameters = dict(model.named_parameters())
    slices = module_slices(shared_names, parameters)
    shared_params = [parameters[name] for name in shared_names]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ckpt_path = os.path.join(source_run_dir, CHECKPOINT_FILENAMES["epoch_9"])
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    batches = derive_probe_batches(bundle, dataset, seed, t=1, batch_size=4096)
    val_grads = validation_cell_gradients(model, bundle.val_x, bundle.val_y,
                                          shared_params, device, chunk=40960)
    cp_rows, batch_rows, sample_grad_norm_max = probe_checkpoint(
            model, state_dict, batches, val_grads, slices, device,
            shared_names=shared_names, shared_params=shared_params)
    if not cp_rows:
        raise AssertionError("smoke produced no checkpoint rows")
    for row in cp_rows:
        for field in ("gradient_norm_mean", "pairwise_cosine_mean",
                      "sample_alignment_mean", "validation_dot_product_mean",
                      "weighted_source_target_contribution_mean",
                      "conflict_contribution_mean", "source_weight_mean",
                      "validation_gradient_norm", "reconstruction_residual_max"):
            v = row.get(field, None)
            if v is None or not math.isfinite(float(v)):
                raise AssertionError(
                    "smoke row field {} non-finite: {}".format(field, v))
    expected_rows = 1 * CELL_NUM * CELL_NUM * len(MODULES)
    if len(cp_rows) != expected_rows:
        raise AssertionError("smoke row count {} != {}".format(len(cp_rows), expected_rows))
    return {"smoke_dataset": dataset, "smoke_seed": int(seed),
            "smoke_rows": len(cp_rows), "smoke_initial_hash_ok": True}


if __name__ == "__main__":
    raise SystemExit(main())