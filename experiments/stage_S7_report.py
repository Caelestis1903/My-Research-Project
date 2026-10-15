"""Stage S7 validator, aggregator, and three output artefacts.

- ``validate_probe_run(run_dir)`` -> (ok, reasons).
- Aggregate 8 per-seed ``gradient_probe.csv`` into
  ``stage_S7_outputs/stage_S7_gradient_raw.csv``.
- Collect Stage S6 F arm performance CSV to
  ``stage_S7_outputs/stage_S7_performance_raw.csv``.
- Emit ``stage_S7_outputs/stage_S7_run_report.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "history_exp", "first"))

from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256  # noqa: E402
from stage_S4_protocol import SEEDS as S4_SEEDS  # noqa: E402
from stage_S7_probe import (  # noqa: E402
    CELL_NUM, DOMAIN_NUM, MODULES, TASK_NUM, T_PROBE, CHECKPOINT_LABEL_ORDER,
    S4_TOLERANCE_ABS, S4_TOLERANCE_REL,
)

OUTPUT_DIR = os.path.join(HERE, "stage_S7_outputs")
KUAIRAND_SOURCE_ROOT = "experiments/stage_S6_raw_results"
MOVIELENS_SOURCE_ROOT = "experiments/stage_S6ML_raw_results"
KUAIRAND_SETTINGS = ("natural",)
MOVIELENS_SETTINGS = ("d2_large",)
TARGET_LOCAL_DOMAIN = {"kuairand": 1, "movielens": 2}
IDENTITY_TOLERANCE_ABS = 1e-12

S4_TOLERANCE_ABS_LOCAL = S4_TOLERANCE_ABS
S4_TOLERANCE_REL_LOCAL = S4_TOLERANCE_REL

PERFORMANCE_COLUMNS: Tuple[str, ...] = (
    "dataset", "setting", "seed", "checkpoint", "stage_epoch", "domain", "task", "cell",
    "test_auc", "test_logloss", "validation_auc", "validation_logloss",
    "train_cell_logloss", "train_objective",
    "macro_auc", "macro_logloss",
    "target_weighted_auc", "target_weighted_logloss",
    "non_target_weighted_auc", "non_target_weighted_logloss",
    "target_domain_auc", "target_domain_logloss",
    "validation_macro_auc", "validation_macro_logloss",
    "validation_target_weighted_auc", "validation_target_weighted_logloss",
    "validation_non_target_weighted_auc", "validation_non_target_weighted_logloss",
    "validation_target_domain_auc", "validation_target_domain_logloss",
)


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str) -> Dict[str, object]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _finite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _safe_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_probe_run(run_dir: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not os.path.isdir(run_dir):
        return False, ["run_dir_missing: {}".format(run_dir)]
    expected_files = (
        "gradient_probe.csv", "gradient_probe_batches.csv",
        "probe_manifest.json", "process_interval.json",
    )
    for filename in expected_files:
        if not os.path.isfile(os.path.join(run_dir, filename)):
            reasons.append("missing_artifact: {}".format(filename))
    if reasons:
        return False, reasons
    rows = _read_csv(os.path.join(run_dir, "gradient_probe.csv"))
    expected_rows = len(CHECKPOINT_LABEL_ORDER) * CELL_NUM * CELL_NUM * len(MODULES)
    if len(rows) != expected_rows:
        reasons.append("checkpoint_row_count_mismatch: {} != {}".format(
            len(rows), expected_rows))
    batch_rows = _read_csv(os.path.join(run_dir, "gradient_probe_batches.csv"))
    expected_batches = len(CHECKPOINT_LABEL_ORDER) * T_PROBE * CELL_NUM * CELL_NUM * len(MODULES)
    if len(batch_rows) != expected_batches:
        reasons.append("batch_row_count_mismatch: {} != {}".format(
            len(batch_rows), expected_batches))
    manifest = _read_json(os.path.join(run_dir, "probe_manifest.json"))
    if not manifest:
        reasons.append("manifest_missing_or_empty")
        return False, reasons
    if str(manifest.get("model_source_sha256", "")) != MODEL_SOURCE_SHA256:
        reasons.append("model_source_hash_mismatch")
    if not bool(manifest.get("initial_parameter_state_hash_match", False)):
        reasons.append("initial_hash_mismatch")
    checkpoints = manifest.get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        reasons.append("checkpoint_metadata_missing")
    else:
        for entry in checkpoints:
            label = entry.get("label", "")
            if label not in CHECKPOINT_LABEL_ORDER:
                reasons.append("unknown_checkpoint_label: {}".format(label))
            sha = entry.get("sha256", "")
            if not sha:
                reasons.append("checkpoint_sha256_missing: {}".format(label))
            if not entry.get("rng_state_unchanged", False):
                reasons.append("rng_state_changed: {}".format(label))
    if rows:
        by_check_batch: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
        for row in rows:
            key = (str(row.get("checkpoint", "")), int(float(row.get("batch_count", 0))))
            by_check_batch.setdefault(key, []).append(row)
        for (cp_label, batch_count), group in by_check_batch.items():
            if batch_count != T_PROBE:
                reasons.append("batch_count_wrong: {}={}".format(cp_label, batch_count))
            if len(group) != CELL_NUM * CELL_NUM * len(MODULES):
                reasons.append("pair_count_wrong: {}={}".format(cp_label, len(group)))
    numeric_fields = ("gradient_norm_mean", "pairwise_cosine_mean", "sample_alignment_mean",
                       "validation_dot_product_mean",
                       "weighted_source_target_contribution_mean",
                       "conflict_contribution_mean", "source_weight_mean",
                       "validation_gradient_norm", "reconstruction_residual_max")
    for row in rows:
        for field in numeric_fields:
            value = row.get(field, "")
            if not _finite(value):
                reasons.append("nonfinite: {} {}={}".format(
                    row.get("checkpoint", ""), field, value))
                continue
            number = float(value)
            if field == "pairwise_cosine_mean" and abs(number) > 1.0 + 1e-9:
                reasons.append("cosine_out_of_range: {}={}".format(field, number))
            if field == "sample_alignment_mean" and abs(number) > 1.0 + 1e-9:
                reasons.append("alignment_out_of_range: {}={}".format(field, number))
    per_check_recon: Dict[str, float] = {}
    per_check_sample_norm: Dict[str, float] = {}
    for row in rows:
        cp = str(row.get("checkpoint", ""))
        recon = float(row.get("reconstruction_residual_max", "0") or 0.0)
        per_check_recon[cp] = max(per_check_recon.get(cp, 0.0), recon)
    for entry in (checkpoints or []):
        if not isinstance(entry, dict):
            continue
        label = entry.get("label", "")
        if not label:
            continue
        per_check_sample_norm[label] = float(entry.get("sample_gradient_norm_max", 0.0) or 0.0)
    for cp, recon in per_check_recon.items():
        sample_norm = per_check_sample_norm.get(cp, 0.0)
        threshold = max(S4_TOLERANCE_ABS_LOCAL, S4_TOLERANCE_REL_LOCAL * max(1e-6, sample_norm))
        if recon > threshold:
            reasons.append("reconstruction_residual_exceeded: {}={} > {}".format(
                cp, recon, threshold))
    h_check: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    for row in rows:
        cp = str(row.get("checkpoint", ""))
        tgt = int(row.get("target_cell", -1))
        module = str(row.get("shared_module", ""))
        h = float(row.get("weighted_source_target_contribution_mean", "0") or 0.0)
        u = float(row.get("validation_dot_product_mean", "0") or 0.0)
        entry = h_check.setdefault((cp, tgt, module), {"u": u, "h_sum": 0.0})
        entry["h_sum"] += h
        entry["u"] = u
    for (cp, tgt, module), entry in h_check.items():
        u = entry.get("u")
        h_sum = entry.get("h_sum", 0.0)
        if u is None:
            reasons.append("U_missing: {} tgt={} module={}".format(cp, tgt, module))
            continue
        threshold = max(S4_TOLERANCE_ABS_LOCAL, S4_TOLERANCE_REL_LOCAL * abs(float(u)))
        if abs(h_sum - float(u)) > threshold:
            reasons.append("H_sum_identity_violated: {} tgt={} module={} H={} U={}".format(
                cp, tgt, module, h_sum, u))
    manifest_fingerprints = manifest.get("batch_fingerprints", [])
    if isinstance(manifest_fingerprints, list) and len(manifest_fingerprints) != T_PROBE:
        reasons.append("batch_fingerprint_count_wrong: {}".format(len(manifest_fingerprints)))
    fp_by_checkpoint: Dict[str, Dict[int, str]] = {}
    fp_conflict = False
    for row in batch_rows:
        cp = str(row.get("checkpoint", ""))
        try:
            bidx = int(float(row.get("batch_index", -1)))
        except (TypeError, ValueError):
            reasons.append("batch_index_unparseable")
            continue
        fp = str(row.get("batch_fingerprint", ""))
        table = fp_by_checkpoint.setdefault(cp, {})
        if bidx in table and table[bidx] != fp:
            fp_conflict = True
        table[bidx] = fp
    if fp_conflict:
        reasons.append("batch_fingerprint_conflict_within_checkpoint")
    sequences: Dict[str, List[str]] = {}
    for cp, table in fp_by_checkpoint.items():
        if sorted(table) != list(range(T_PROBE)):
            reasons.append("batch_index_incomplete: {}".format(cp))
            continue
        sequences[cp] = [table[i] for i in range(T_PROBE)]
    if sequences:
        first_seq = next(iter(sequences.values()))
        for cp, seq in sequences.items():
            if seq != first_seq:
                reasons.append("batch_fingerprints_diverge_across_checkpoints: {}".format(cp))
                break
        if isinstance(manifest_fingerprints, list) and manifest_fingerprints \
                and [str(f) for f in manifest_fingerprints] != first_seq:
            reasons.append("batch_fingerprints_differ_from_manifest")
    return (not reasons), reasons


def _weighted_cell_mean(values: Mapping[int, Optional[float]],
                        weight_of_domain) -> object:
    """Share-style weighted mean over the 6 cells.

    ``weight_of_domain`` maps domain -> non-negative weight; the result is the
    normalized weighted mean ``(Σ s_d X_{d,t}) / (Σ s_d)`` which equals the
    S1 ``_target_weighted`` definition ``(1/T) Σ_t Σ_d s_d X_{d,t}`` when the
    per-domain weights sum to one over the D domains.
    """
    numerator = 0.0
    denominator = 0.0
    for cell in range(CELL_NUM):
        value = values.get(cell)
        if value is None:
            return ""
        weight = float(weight_of_domain(cell % DOMAIN_NUM))
        numerator += weight * value
        denominator += weight
    if denominator <= 0:
        return ""
    return numerator / denominator


def _target_domain_mean(values: Mapping[int, Optional[float]],
                        target_domain: int) -> object:
    """Mean over the 2 cells belonging to ``target_domain`` (task-major cells)."""
    target_cells = [target_domain + task * DOMAIN_NUM for task in range(TASK_NUM)]
    if TASK_NUM == 2:
        a = values.get(target_cells[0])
        b = values.get(target_cells[1])
    else:
        items = [values.get(cell) for cell in target_cells]
        if any(value is None for value in items):
            return ""
        return sum(items) / float(len(items))
    if a is None or b is None:
        return ""
    return (a + b) / float(TASK_NUM)


def _macro_cell_mean(values: Mapping[int, Optional[float]]) -> object:
    items = [values.get(cell) for cell in range(CELL_NUM)]
    if any(value is None for value in items):
        return ""
    return sum(items) / float(len(items))


def _load_performance(dataset_root: str, dataset: str, seed: int) -> List[Dict[str, object]]:
    run_dir = os.path.abspath(os.path.join(ROOT, dataset_root, "F",
                                            "seed{}".format(int(seed)), "joint"))
    test_rows = _read_csv(os.path.join(run_dir, "metrics_test.csv"))
    epoch_rows = _read_csv(os.path.join(run_dir, "metrics_epoch.csv"))
    source_manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    selected_epoch = int(source_manifest.get("selected_epoch", -1))

    # metrics_epoch.csv carries one row per (epoch, cell); epoch-level scalars
    # (train_objective, validation_macro_*, target_weighted_validation_*) are
    # repeated on every row of the epoch.
    epoch_cell: Dict[Tuple[int, int], Dict[str, str]] = {}
    for row in epoch_rows:
        try:
            epoch_cell[(int(row["epoch"]), int(row["cell"]))] = row
        except (KeyError, ValueError):
            continue

    # Natural domain shares are recorded on every metrics_test row.
    shares: Dict[int, float] = {}
    for row in test_rows:
        try:
            domain = int(row["domain"])
        except (KeyError, ValueError):
            continue
        share = _safe_float(row.get("domain_share"))
        if share is not None:
            shares[domain] = share
    target_domain_index = TARGET_LOCAL_DOMAIN.get(dataset)
    if target_domain_index is None:
        raise ValueError("Stage S7 cannot infer target_local_domain for dataset {}".format(dataset))
    share_target = float(shares.get(target_domain_index, 0.0))
    one_minus_share_target = max(0.0, 1.0 - share_target)

    def target_weight(d: int) -> float:
        return float(shares.get(d, 0.0))

    def non_target_weight(d: int) -> float:
        if d == target_domain_index:
            return 0.0
        return float(shares.get(d, 0.0)) / one_minus_share_target

    # Test rows indexed by (checkpoint tag, cell).  The epoch_49 tag only takes
    # common_epoch_49 rows; the duplicate "final" rows are never used.
    test_by_tag: Dict[str, Dict[int, Dict[str, str]]] = {}
    for row in test_rows:
        ck = str(row.get("checkpoint", ""))
        try:
            stage_epoch = int(row.get("stage_epoch", -1))
            cell = int(row["cell"])
        except (KeyError, ValueError):
            continue
        if ck.startswith("common_epoch_"):
            tag = "epoch_{}".format(stage_epoch)
        elif ck == "selected":
            tag = "selected"
        else:
            continue
        test_by_tag.setdefault(tag, {})[cell] = row

    rows: List[Dict[str, object]] = []
    for ck_tag in CHECKPOINT_LABEL_ORDER:
        if ck_tag == "selected":
            epoch_value = selected_epoch
        else:
            epoch_value = int(ck_tag.split("_")[1])
        test_cells = test_by_tag.get(ck_tag, {})
        if ck_tag == "selected" and test_cells:
            stage_epoch_seen = {int(r.get("stage_epoch", epoch_value))
                                for r in test_cells.values()}
            if stage_epoch_seen and stage_epoch_seen != {epoch_value}:
                epoch_value = sorted(stage_epoch_seen)[0]
        epoch_cell_rows = {cell: epoch_cell.get((epoch_value, cell), {})
                           for cell in range(CELL_NUM)}
        test_auc = {cell: _safe_float(test_cells.get(cell, {}).get("auc"))
                    for cell in range(CELL_NUM)}
        test_ll = {cell: _safe_float(test_cells.get(cell, {}).get("logloss"))
                   for cell in range(CELL_NUM)}
        val_auc = {cell: _safe_float(epoch_cell_rows[cell].get("validation_auc"))
                   for cell in range(CELL_NUM)}
        val_ll = {cell: _safe_float(epoch_cell_rows[cell].get("validation_logloss"))
                  for cell in range(CELL_NUM)}
        setting = str(source_manifest.get("setting", "")
                      or next(iter(test_cells.values()), {}).get("setting", ""))
        for cell in range(CELL_NUM):
            task = cell // DOMAIN_NUM
            domain = cell % DOMAIN_NUM
            test_row = test_cells.get(cell, {})
            epoch_row = epoch_cell_rows[cell]
            rows.append({
                "dataset": dataset,
                "setting": setting,
                "seed": int(seed),
                "checkpoint": ck_tag,
                "stage_epoch": int(epoch_value),
                "domain": int(domain),
                "task": int(task),
                "cell": int(cell),
                "share_target": share_target,
                "one_minus_share_target": one_minus_share_target,
                "test_auc": test_row.get("auc", ""),
                "test_logloss": test_row.get("logloss", ""),
                "validation_auc": epoch_row.get("validation_auc", ""),
                "validation_logloss": epoch_row.get("validation_logloss", ""),
                "train_cell_logloss": epoch_row.get("train_cell_logloss", ""),
                "train_objective": epoch_row.get("train_objective", ""),
                "macro_auc": _macro_cell_mean(test_auc),
                "macro_logloss": _macro_cell_mean(test_ll),
                "target_weighted_auc": _weighted_cell_mean(test_auc, target_weight),
                "target_weighted_logloss": _weighted_cell_mean(test_ll, target_weight),
                "non_target_weighted_auc": _weighted_cell_mean(test_auc, non_target_weight),
                "non_target_weighted_logloss": _weighted_cell_mean(test_ll, non_target_weight),
                "target_domain_auc": _target_domain_mean(test_auc, target_domain_index),
                "target_domain_logloss": _target_domain_mean(test_ll, target_domain_index),
                "validation_macro_auc": epoch_row.get("validation_macro_auc", ""),
                "validation_macro_logloss": epoch_row.get("validation_macro_logloss", ""),
                "validation_target_weighted_auc": epoch_row.get("target_weighted_validation_auc", ""),
                "validation_target_weighted_logloss": epoch_row.get("target_weighted_validation_logloss", ""),
                "validation_non_target_weighted_auc": _weighted_cell_mean(val_auc, non_target_weight),
                "validation_non_target_weighted_logloss": _weighted_cell_mean(val_ll, non_target_weight),
                "validation_target_domain_auc": _target_domain_mean(val_auc, target_domain_index),
                "validation_target_domain_logloss": _target_domain_mean(val_ll, target_domain_index),
            })
    return rows


def _verify_performance_identity(perf_rows: Sequence[Mapping[str, object]]) -> List[str]:
    """Confirm `target_weighted_X == s_{d*} * target_domain_X + (1 - s_{d*}) * non_target_weighted_X`.

    Verified per row across the four (scope, metric) combinations:
    test × {auc, logloss} and validation × {auc, logloss}.
    """
    if not perf_rows:
        return []
    identity_pairs = (
        ("test", "auc",
         "target_weighted_auc", "target_domain_auc", "non_target_weighted_auc"),
        ("test", "logloss",
         "target_weighted_logloss", "target_domain_logloss", "non_target_weighted_logloss"),
        ("validation", "auc",
         "validation_target_weighted_auc", "validation_target_domain_auc",
         "validation_non_target_weighted_auc"),
        ("validation", "logloss",
         "validation_target_weighted_logloss", "validation_target_domain_logloss",
         "validation_non_target_weighted_logloss"),
    )
    per_dataset_meta: Dict[str, Tuple[int, float, dict]] = {}
    reasons: List[str] = []
    for row_index, row in enumerate(perf_rows):
        dataset = str(row.get("dataset", ""))
        target_domain_index = TARGET_LOCAL_DOMAIN.get(dataset)
        if target_domain_index is None:
            reasons.append("unknown_target_domain: {}".format(dataset))
            continue
        if dataset not in per_dataset_meta:
            share_target = float(row.get("share_target", 0.0) or 0.0)
            one_minus_share_target = float(row.get("one_minus_share_target", 0.0) or 0.0)
            per_dataset_meta[dataset] = (target_domain_index, share_target, {
                "share_target": share_target, "one_minus_share_target": one_minus_share_target,
            })
        share_target = per_dataset_meta[dataset][2]["share_target"]
        one_minus_share_target = per_dataset_meta[dataset][2]["one_minus_share_target"]
        if one_minus_share_target == 0:
            reasons.append("degenerate_non_target_mass: dataset={}".format(dataset))
            continue
        for scope, _metric, tw_field, td_field, nt_field in identity_pairs:
            tw = _safe_float(row.get(tw_field))
            td = _safe_float(row.get(td_field))
            nt = _safe_float(row.get(nt_field))
            if tw is None or td is None or nt is None:
                reasons.append("missing_identity_field: dataset={} ck={} cell={} scope={} metric={}".format(
                    dataset, row.get("checkpoint", ""), row.get("cell", ""), scope, _metric))
                continue
            reconstructed = share_target * td + one_minus_share_target * nt
            diff = tw - reconstructed
            if abs(diff) > IDENTITY_TOLERANCE_ABS:
                reasons.append(
                    "identity_violation: dataset={} ck={} cell={} scope={} metric={} "
                    "tw={:.6e} td={:.6e} nt={:.6e} "
                    "s_d*={:.6e} (1-s_d*)={:.6e} reconstructed={:.6e} diff={:.3e}".format(
                        dataset, row.get("checkpoint", ""), row.get("cell", ""), scope, _metric,
                        tw, td, nt, share_target, one_minus_share_target, reconstructed, diff))
    return reasons


def aggregate() -> Dict[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    aggregate_rows: List[Dict[str, object]] = []
    for dataset, settings in (
        ("kuairand", KUAIRAND_SETTINGS),
        ("movielens", MOVIELENS_SETTINGS),
    ):
        for seed in S4_SEEDS:
            run_dir = os.path.join(ROOT, "experiments/stage_S7_raw_results",
                                    dataset, "seed{}".format(int(seed)))
            csv_path = os.path.join(run_dir, "gradient_probe.csv")
            if not os.path.isfile(csv_path):
                continue
            with open(csv_path, encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    row["dataset"] = dataset
                    row["seed"] = int(seed)
                    aggregate_rows.append(row)
    aggregate_path = os.path.join(OUTPUT_DIR, "stage_S7_gradient_raw.csv")
    if aggregate_rows:
        fields = sorted({key for row in aggregate_rows for key in row.keys()})
        with open(aggregate_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in aggregate_rows:
                writer.writerow(row)
    perf_rows: List[Dict[str, object]] = []
    for dataset, root, settings in (
        ("kuairand", KUAIRAND_SOURCE_ROOT, KUAIRAND_SETTINGS),
        ("movielens", MOVIELENS_SOURCE_ROOT, MOVIELENS_SETTINGS),
    ):
        for seed in S4_SEEDS:
            run_dir = os.path.join(ROOT, root, "F", "seed{}".format(int(seed)), "joint")
            if not os.path.isdir(run_dir):
                continue
            for setting in settings:
                rows = _load_performance(root, dataset, int(seed))
                perf_rows.extend(rows)
    perf_path = os.path.join(OUTPUT_DIR, "stage_S7_performance_raw.csv")
    identity_reasons: List[str] = []
    if perf_rows:
        identity_reasons = _verify_performance_identity(perf_rows)
        if identity_reasons:
            raise ValueError(
                "Stage S7 performance identity check failed ({} issues; first 5 shown): {}".format(
                    len(identity_reasons), identity_reasons[:5]))
        with open(perf_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PERFORMANCE_COLUMNS,
                                    extrasaction="ignore")
            writer.writeheader()
            for row in perf_rows:
                writer.writerow({key: row.get(key, "") for key in PERFORMANCE_COLUMNS})
    report_lines: List[str] = []
    report_lines.append("# Stage S7 run report")
    report_lines.append("")
    report_lines.append("Stage S7 sample-objective MDMT Domain x Task gradient-dynamics probe.")
    report_lines.append("")
    report_lines.append("- **Naming note (designer-fixed)**: `target_weighted_*` is the **full-population share-weighted** mean following the S1 naming lineage (`(1/T) Σ_t Σ_d s_d X_{d,t}`); `target_domain_*` is the **target-domain two-cell simple mean** (`(1/T) Σ_t X_{d*,t}`).")
    report_lines.append("- Identity checked on every row: `target_weighted_X = s_{d*} * target_domain_X + (1 - s_{d*}) * non_target_weighted_X` (abs tol 1e-12).")
    report_lines.append("")
    for dataset in ("kuairand", "movielens"):
        for seed in S4_SEEDS:
            run_dir = os.path.join(ROOT, "experiments/stage_S7_raw_results",
                                    dataset, "seed{}".format(int(seed)))
            manifest = _read_json(os.path.join(run_dir, "probe_manifest.json"))
            if not manifest:
                continue
            report_lines.append("## {dataset} seed {seed}".format(dataset=dataset, seed=int(seed)))
            report_lines.append("")
            report_lines.append("- source_run_dir: `{}`".format(manifest.get("source_run_dir", "")))
            report_lines.append("- source_formal_run_key: `{}`".format(
                manifest.get("source_formal_run_key", "")))
            report_lines.append("- initial hash match: {}".format(
                manifest.get("initial_parameter_state_hash_match", False)))
            report_lines.append("- selected_epoch: {} (overlap with common checkpoints: {}; {})".format(
                manifest.get("selected_epoch", -1),
                bool(manifest.get("selected_overlap_with_common", False)),
                "selected row reuses the asserted-equal common checkpoint probe"
                if bool(manifest.get("selected_overlap_with_common", False))
                else "selected checkpoint probed separately (reuse path not triggered)"))
            report_lines.append("- model source hash: `{}`".format(manifest.get("model_source_sha256", "")))
            report_lines.append("- batch_fingerprint_digest: `{}`".format(
                manifest.get("batch_fingerprint_digest", "")))
            for ck_meta in manifest.get("checkpoints", []):
                report_lines.append("  * {} (stage_epoch={}, sha256={}, recon={:.2e}, rng_unchanged={})".format(
                    ck_meta.get("label", ""), ck_meta.get("stage_epoch", -1),
                    ck_meta.get("sha256", "")[:12],
                    float(ck_meta.get("reconstruction_residual_max", 0.0)),
                    ck_meta.get("rng_state_unchanged", False)))
            report_lines.append("- wall_seconds: {:.2f}".format(
                float(manifest.get("wall_seconds", 0.0))))
            report_lines.append("")
    report_path = os.path.join(OUTPUT_DIR, "stage_S7_run_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")
    return {"gradient_csv": aggregate_path,
            "performance_csv": perf_path,
            "report_md": report_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage S7 report")
    parser.add_argument("--validate", help="path to a single probe run_dir")
    parser.add_argument("--aggregate", action="store_true",
                        help="aggregate all per-seed CSVs and emit three outputs")
    args = parser.parse_args()
    if args.validate:
        ok, reasons = validate_probe_run(args.validate)
        print(json.dumps({"run_dir": os.path.abspath(args.validate),
                          "ok": ok, "reasons": reasons}, indent=2))
        return 0 if ok else 1
    if args.aggregate:
        result = aggregate()
        print(json.dumps(result, indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())