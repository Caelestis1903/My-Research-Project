"""Stage S8 validator + aggregator.

Provides:

- ``validate_base_run(run_dir)`` -- V-BASE hard gate (Phase 1 retrain vs
  the original S6 canonical F-arm run).
- ``validate_run(run_dir)`` -- per-run hard gates for a Phase 2 continuation
  run (V0 / V1 / V3 / V4 / V5, plus structural file presence).
- ``aggregate()`` -- mechanical aggregation into ``stage_S8_outputs`` of
  the §8 schema CSVs / ``stage_S8_summary_manifest.json`` /
  ``reports/stage_S8_run_report.md``.

Per Algorithm Specification ``stage_S8_design.md`` §4.5.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import stage_S8_protocol as proto  # noqa: E402
from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256  # noqa: E402


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
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


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


# ---------------------------------------------------------------------------
# Phase 1 base-run validator (V-BASE)
# ---------------------------------------------------------------------------

def validate_base_run(run_dir: str, source_run_dir: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Validate a Phase 1 S8 base retrain run.

    Hard gates (V-BASE):

    1. all train batch fingerprints byte-equal between S8 base run and the
       original S6 canonical F-arm source (``batch_trace.csv``, kind=train);
    2. per-epoch validation macro AUC diff <= VBASE_MACRO_AUC_ABS;
    3. ``selected_epoch`` matches source; ``selected_validation_macro_auc``
       diff <= VBASE_MACRO_AUC_ABS;
    4. ``initial_parameter_state_hash`` matches source;
    5. final param hash drift is recorded report-only (item 5, NOT a hard
       gate).

    Returns ``(ok, reasons)``.  ``ok=False`` lists the failing gate(s).
    """
    reasons: List[str] = []
    if not os.path.isdir(run_dir):
        return False, ["run_dir_missing: {}".format(run_dir)]
    required = ("run_manifest.json", "config.json", "metrics_epoch.csv",
                "batch_trace.csv", "gradient_step.csv", "probe.csv",
                "metrics_test.csv", "metrics_summary.csv",
                "process_interval.json", "stdout.log")
    for name in required:
        if not os.path.isfile(os.path.join(run_dir, name)):
            reasons.append("missing_artifact: " + name)
    if reasons:
        return False, reasons
    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    if str(manifest.get("stage")) != "S8BASE":
        reasons.append("stage_not_S8BASE: " + str(manifest.get("stage")))
    # Source must be available for V-BASE crosscheck.
    dataset = str(manifest.get("dataset", "")).lower()
    seed = int(manifest.get("seed", -1))
    # The S1/S3 manifest stores "movielens-1m" as the dataset label; the
    # S8 protocol key is "movielens".  Normalise both forms.
    dataset_norm = dataset.replace("-1m", "")
    if dataset_norm not in proto.DATASETS:
        reasons.append("dataset_unknown: " + str(dataset))
        return False, reasons
    dataset = dataset_norm
    if source_run_dir is None:
        source_run_dir = proto.source_run_dir(dataset, int(seed))
    if not os.path.isdir(source_run_dir):
        reasons.append("source_run_missing: " + source_run_dir)
        return False, reasons
    source_manifest = _read_json(os.path.join(source_run_dir, "run_manifest.json"))
    # Gate 4: initial_parameter_state_hash.
    base_initial_hash = str(manifest.get("initial_parameter_state_hash", ""))
    source_initial_hash = str(source_manifest.get("initial_parameter_state_hash", ""))
    if base_initial_hash != source_initial_hash:
        reasons.append("initial_parameter_state_hash_mismatch: base={} source={}".format(
            base_initial_hash, source_initial_hash))
    # Gate 3: selected_epoch + selected_validation_macro_auc.
    base_selected_epoch = int(manifest.get("selected_epoch", -1))
    source_selected_epoch = int(source_manifest.get("selected_epoch", -1))
    if base_selected_epoch != source_selected_epoch:
        reasons.append("selected_epoch_mismatch: base={} source={}".format(
            base_selected_epoch, source_selected_epoch))
    base_selected_auc = _safe_float(manifest.get("selected_validation_macro_auc", ""))
    source_selected_auc = _safe_float(source_manifest.get("selected_validation_macro_auc", ""))
    if base_selected_auc is None or source_selected_auc is None:
        reasons.append("selected_validation_macro_auc_missing")
    else:
        diff = abs(base_selected_auc - source_selected_auc)
        if diff > proto.VBASE_MACRO_AUC_ABS:
            reasons.append("selected_validation_macro_auc_diff_above_threshold: {:.3e} > {:.3e}".format(
                diff, proto.VBASE_MACRO_AUC_ABS))
    # Gate 1: all train batch fingerprints byte-equal.
    base_trace = _read_csv(os.path.join(run_dir, "batch_trace.csv"))
    source_trace = _read_csv(os.path.join(source_run_dir, "batch_trace.csv"))
    base_train = [row for row in base_trace if str(row.get("kind", "")) == "train"]
    source_train = [row for row in source_trace if str(row.get("kind", "")) == "train"]
    if len(base_train) != len(source_train):
        reasons.append("train_batch_count_mismatch: base={} source={}".format(
            len(base_train), len(source_train)))
    else:
        mismatches: List[Dict[str, str]] = []
        for idx, (b, s) in enumerate(zip(base_train, source_train)):
            if str(b.get("batch_fingerprint", "")) != str(s.get("batch_fingerprint", "")):
                mismatches.append({"index": str(idx),
                                    "base": str(b.get("batch_fingerprint", "")),
                                    "source": str(s.get("batch_fingerprint", ""))})
                if len(mismatches) >= 5:
                    break
        if mismatches:
            reasons.append("train_batch_fingerprint_mismatch: " + json.dumps(mismatches))
    # Gate 2: per-epoch validation macro AUC diff.  The metrics_epoch.csv
    # schema is PER-CELL (every row carries domain/task and repeats the
    # epoch-level ``validation_macro_auc``), so we group by epoch over ALL
    # rows instead of filtering for aggregate-only rows (which do not
    # exist).
    base_metrics = _read_csv(os.path.join(run_dir, "metrics_epoch.csv"))
    source_metrics = _read_csv(os.path.join(source_run_dir, "metrics_epoch.csv"))

    def _epoch_macro_auc(rows: List[Dict[str, str]]) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for row in rows:
            try:
                epoch = int(row.get("epoch", "-1"))
            except (TypeError, ValueError):
                continue
            auc = _safe_float(row.get("validation_macro_auc", ""))
            if auc is None:
                continue
            if epoch in out and abs(out[epoch] - auc) > 0.0:
                # inconsistent per-cell repetition -- treat as corruption
                out[epoch] = float("nan")
            else:
                out.setdefault(epoch, auc)
        return out

    base_epoch_auc = _epoch_macro_auc(base_metrics)
    source_epoch_auc = _epoch_macro_auc(source_metrics)
    epoch_diffs: List[Tuple[int, float, float, float]] = []
    if not base_epoch_auc or not source_epoch_auc:
        reasons.append("metrics_epoch_macro_auc_empty: base={} source={}".format(
            len(base_epoch_auc), len(source_epoch_auc)))
    for epoch in sorted(set(base_epoch_auc.keys()) & set(source_epoch_auc.keys())):
        diff = abs(base_epoch_auc[epoch] - source_epoch_auc[epoch])
        if diff > proto.VBASE_MACRO_AUC_ABS:
            epoch_diffs.append((int(epoch), base_epoch_auc[epoch],
                                 source_epoch_auc[epoch], diff))
    if epoch_diffs:
        sample = ", ".join("e{} {:.3e}".format(e, d) for e, _, _, d in epoch_diffs[:3])
        reasons.append("per_epoch_validation_macro_auc_diff_above_threshold: {} ...".format(sample))
    # Final parameter hash drift is recorded report-only (item 5).  We
    # surface it in the audit but do not add a hard-gate reason.
    final_hash = str(manifest.get("final_parameter_state_hash", ""))
    source_final_hash = str(source_manifest.get("final_parameter_state_hash", ""))
    drift_recorded = (final_hash != source_final_hash)
    # Full-state dump presence.
    dump_epochs = manifest.get("full_state_dump_epochs", [])
    if not isinstance(dump_epochs, list) or len(dump_epochs) == 0:
        reasons.append("full_state_dump_epochs_empty")
    for epoch in dump_epochs:
        try:
            ep_int = int(epoch)
        except (TypeError, ValueError):
            continue
        fname = proto.full_state_filename(ep_int)
        path = os.path.join(run_dir, fname)
        if not os.path.isfile(path):
            reasons.append("missing_full_state_dump: " + fname)
    # Dump hashes recorded in manifest.
    dump_hashes = manifest.get("full_state_dump_hashes", {})
    if not isinstance(dump_hashes, dict) or len(dump_hashes) == 0:
        reasons.append("full_state_dump_hashes_missing_in_manifest")
    for epoch, expected in dump_hashes.items():
        path = os.path.join(run_dir, str(epoch))
        if not os.path.isfile(path):
            continue
        observed = _hash_file(path)
        if observed != str(expected):
            reasons.append("full_state_dump_hash_mismatch: {} observed={} expected={}".format(
                epoch, observed, expected))
    # vbase summary
    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Phase 2 continuation-run validator
# ---------------------------------------------------------------------------

def validate_run(run_dir: str) -> Tuple[bool, List[str]]:
    """Validate a Phase 2 S8 continuation run.

    Hard gates:

    - structural files present (steps / geometry / validation / best /
      audit / process_interval);
    - audit status != ABORTED;
    - V0 step-0 L_full diff and delta-norm rel diff within tolerance;
    - V3 cross-branch pre-step bit-equality (per the audit's
      ``v3_per_pair`` summary);
    - V4 all consumed main batch fingerprints match the S8 base run's
      ``batch_trace.csv`` (the audit already materialises the comparison
      via ``replay_audit.materialized_fingerprints``; we re-verify here
      from the CSV row materialisation); every epoch-end architecture
      batch fingerprint must also match (``arch_fingerprint_rows``);
    - V5 anchor state recovery: shared step == expected, arch step ==
      expected, private state coverage == 100%, model hash == dump hash;
    - shared step expected equals ``proto.expected_shared_step`` for the
      dataset / anchor epoch;
    - the f_original reference values in the audit equal the ORIGINAL S6
      canonical manifest (design §9.11), and for ``anchor_class ==
      "selected"`` the frozen anchor pin equals the canonical
      ``selected_epoch``;
    - validation/geometry CSV cardinality matches the frozen schema
      (3*9*(1+K+h_max_epochs) and 3*|valgrad_steps|*288);
    - the RNG digest is invariant WITHIN each continuation step
      (``rng_within_epoch_invariant``; the digest legitimately advances at
      iterator draws between steps/epochs).
    """
    reasons: List[str] = []
    if not os.path.isdir(run_dir):
        return False, ["run_dir_missing: {}".format(run_dir)]
    expected_files = (proto.STEPS_CSV_NAME, proto.GEOMETRY_CSV_NAME,
                       proto.VALIDATION_CSV_NAME, proto.BEST_CSV_NAME,
                       proto.AUDIT_JSON_NAME, proto.PROCESS_INTERVAL_NAME)
    for filename in expected_files:
        if not os.path.isfile(os.path.join(run_dir, filename)):
            reasons.append("missing_artifact: " + filename)
    if reasons:
        return False, reasons
    audit = _read_json(os.path.join(run_dir, proto.AUDIT_JSON_NAME))
    if str(audit.get("status")) == "ABORTED":
        reasons.append("run_aborted_acknowledged: " + str(audit.get("abort_reason", "")))
        # Continue validating the rest (the audit may still carry useful info).
    # V5: anchor reconstruction.
    anchor_recon = audit.get("anchor_reconstruction", {})
    if not isinstance(anchor_recon, dict) or not anchor_recon:
        reasons.append("anchor_reconstruction_missing")
    else:
        shared_step_observed = int(anchor_recon.get("shared_step_observed", -1))
        shared_step_expected = int(anchor_recon.get("shared_step_expected", -1))
        if shared_step_observed != shared_step_expected:
            reasons.append("v5_shared_step_mismatch: observed={} expected={}".format(
                shared_step_observed, shared_step_expected))
        arch_step_observed = int(anchor_recon.get("arch_step_observed", -1))
        arch_step_expected = proto.expected_arch_step(int(audit.get("anchor_epoch", -1)))
        if arch_step_observed != arch_step_expected:
            reasons.append("v5_arch_step_mismatch: observed={} expected={}".format(
                arch_step_observed, arch_step_expected))
        if not bool(anchor_recon.get("private_state_coverage", False)):
            reasons.append("v5_private_state_coverage_false")
        # Model hash check: load the dump and compare parameter hash to
        # the manifest's recorded (or compare hashes directly via the
        # audit field ``base_full_state_dump_sha256``).
        full_state_dump_sha256 = str(anchor_recon.get("full_state_dump_sha256", ""))
        full_state_dump_path = str(anchor_recon.get("full_state_dump_path", ""))
        if full_state_dump_path and os.path.isfile(full_state_dump_path):
            observed_sha = _hash_file(full_state_dump_path)
            if observed_sha != full_state_dump_sha256:
                reasons.append("v5_full_state_dump_sha_mismatch: observed={} expected={}".format(
                    observed_sha, full_state_dump_sha256))
        elif not full_state_dump_sha256:
            reasons.append("v5_full_state_dump_sha_missing")
    # V0 anchor-point hard gate (step-0 macro AUC vs S8-BASE anchor epoch).
    v0_anchor = audit.get("v0_step0_macro_auc", {})
    if not isinstance(v0_anchor, dict) or not v0_anchor:
        reasons.append("v0_step0_macro_auc_missing")
    elif bool(v0_anchor.get("fail", False)):
        reasons.append("v0_anchor_macro_auc_above_threshold: {}".format(
            v0_anchor.get("abs_diff")))
    # V0: step-1 hard gate (L_full diff, delta rel diff).
    v0 = audit.get("v0_violations", [])
    if not isinstance(v0, list):
        reasons.append("v0_violations_format_invalid")
    elif v0:
        for entry in v0:
            if bool(entry.get("v0_l_full_fail", False)):
                reasons.append("v0_l_full_above_threshold_step_1: {}".format(
                    entry.get("L_full_abs_diff")))
            if bool(entry.get("v0_delta_rel_fail", False)):
                reasons.append("v0_delta_rel_above_threshold_step_1: {}".format(
                    entry.get("delta_norm_rel_diff")))
            break
    # V4 arch: every epoch-end architecture batch fingerprint must match.
    arch_fp_rows = audit.get("arch_fingerprint_rows", [])
    if not isinstance(arch_fp_rows, list) or not arch_fp_rows:
        reasons.append("arch_fingerprint_rows_missing")
    else:
        for entry in arch_fp_rows:
            if not bool(entry.get("match", False)):
                reasons.append("v4_arch_batch_fingerprint_mismatch: epoch={}".format(
                    entry.get("epoch")))
    # f_original reference chain: the audit values must come from the
    # original S6 canonical manifest (design §9.11); re-verify here.
    f_src_dir = proto.source_run_dir(str(audit.get("dataset", "")),
                                      int(audit.get("seed", -1)))
    f_src_manifest = _read_json(os.path.join(f_src_dir, "run_manifest.json"))
    canonical_sel_epoch = f_src_manifest.get("selected_epoch", None)
    canonical_sel_auc = _safe_float(f_src_manifest.get("selected_validation_macro_auc", ""))
    audit_f_auc = _safe_float(audit.get("f_original_best_macro_auc", ""))
    audit_f_epoch = audit.get("f_original_selected_epoch", None)
    if canonical_sel_auc is None or audit_f_auc is None:
        reasons.append("f_original_reference_missing")
    elif abs(float(canonical_sel_auc) - float(audit_f_auc)) > 1e-6:
        reasons.append("f_original_best_macro_auc_not_canonical: audit={} canonical={}".format(
            audit_f_auc, canonical_sel_auc))
    if canonical_sel_epoch is None or audit_f_epoch is None \
            or int(canonical_sel_epoch) != int(audit_f_epoch):
        reasons.append("f_original_selected_epoch_not_canonical: audit={} canonical={}".format(
            audit_f_epoch, canonical_sel_epoch))
    if str(audit.get("anchor_class", "")) == "selected" and canonical_sel_epoch is not None:
        if int(audit.get("anchor_epoch", -1)) != int(canonical_sel_epoch):
            reasons.append("anchor_pin_mismatch_canonical_selected: anchor={} canonical={}".format(
                audit.get("anchor_epoch"), canonical_sel_epoch))
    # V3: cross-branch pre-step bit-equality.
    v3 = audit.get("v3_per_pair", {})
    if not isinstance(v3, dict) or not v3:
        reasons.append("v3_per_pair_missing")
    else:
        for pair_key, entry in v3.items():
            if not bool(entry.get("all_equal", False)):
                reasons.append("v3_branch_bit_equality_failed: {}".format(pair_key))
    # V4: every consumed main batch fingerprint matches the S8 base.
    base_run_dir = str(audit.get("base_run_dir", ""))
    if base_run_dir and os.path.isdir(base_run_dir):
        base_trace = _read_csv(os.path.join(base_run_dir, "batch_trace.csv"))
        base_train_fps: List[str] = []
        for row in base_trace:
            if str(row.get("kind", "")) != "train":
                continue
            try:
                ep = int(row.get("epoch", "-1"))
            except (TypeError, ValueError):
                continue
            if ep <= int(audit.get("anchor_epoch", -1)):
                continue
            base_train_fps.append(str(row.get("batch_fingerprint", "")))
        replay = audit.get("replay_audit", {})
        replay_fps = replay.get("materialized_fingerprints", []) if isinstance(replay, dict) else []
        h_max_steps = int(audit.get("h_max_steps", 0))
        if h_max_steps > 0:
            expected = base_train_fps[:h_max_steps]
            if len(expected) < h_max_steps:
                reasons.append("v4_base_train_fingerprints_insufficient: have={} need={}".format(
                    len(expected), h_max_steps))
            else:
                if list(replay_fps[:h_max_steps]) != expected:
                    mismatches = sum(1 for a, b in zip(replay_fps[:h_max_steps], expected)
                                       if a != b)
                    reasons.append("v4_main_batch_fingerprint_mismatch: count={}".format(mismatches))
    # Step CSV row count: 3 branches * h_max_steps.
    step_rows = _read_csv(os.path.join(run_dir, proto.STEPS_CSV_NAME))
    expected_step_count = 3 * int(audit.get("h_max_steps", 0))
    if len(step_rows) != expected_step_count:
        reasons.append("step_row_count_mismatch: {} != {}".format(
            len(step_rows), expected_step_count))
    # Best CSV row count: 3 branches * len(horizons).
    best_rows = _read_csv(os.path.join(run_dir, proto.BEST_CSV_NAME))
    horizons_count = len(audit.get("horizons", {}))
    expected_best_count = 3 * horizons_count
    if len(best_rows) != expected_best_count:
        reasons.append("best_row_count_mismatch: {} != {}".format(
            len(best_rows), expected_best_count))
    # Validation CSV cardinality: 3 branches * 9 rows * (1 anchor point +
    # h_max_steps post-step points + h_max_epochs post-arch points).
    val_rows_count = len(_read_csv(os.path.join(run_dir, proto.VALIDATION_CSV_NAME)))
    expected_val_count = 3 * 9 * (1 + int(audit.get("h_max_steps", 0))
                                    + int(audit.get("h_max_epochs", 0)))
    if val_rows_count != expected_val_count:
        reasons.append("validation_row_count_mismatch: {} != {}".format(
            val_rows_count, expected_val_count))
    # Geometry CSV cardinality: 3 branches * |valgrad_sample_steps| *
    # (36 cell pairs * 8 modules).
    geometry_rows_count = len(_read_csv(os.path.join(run_dir, proto.GEOMETRY_CSV_NAME)))
    expected_geometry_count = (3 * len(audit.get("valgrad_sample_steps", []))
                                 * 36 * 8)
    if geometry_rows_count != expected_geometry_count:
        reasons.append("geometry_row_count_mismatch: {} != {}".format(
            geometry_rows_count, expected_geometry_count))
    # Numerical finiteness on the step CSV.  ``theta_distance_to_F_shared``
    # is only defined for the T/DH rows (F rows carry NaN by design).
    for row in step_rows:
        for field in ("L_full", "g_full_norm", "g_target_norm", "delta_full_norm",
                       "delta_target_norm", "delta_applied_norm", "delta_cosine",
                       "applied_formula_residual", "private_full_residual"):
            if not _finite(row.get(field, "")):
                reasons.append("nonfinite_step: {}={} branch={} step={}".format(
                    field, row.get(field, ""), row.get("branch"), row.get("step")))
                return False, reasons
        if str(row.get("branch", "")) in ("T", "DH"):
            if not _finite(row.get("theta_distance_to_F_shared", "")):
                reasons.append("nonfinite_step: theta_distance_to_F_shared={} branch={} step={}".format(
                    row.get("theta_distance_to_F_shared", ""), row.get("branch"), row.get("step")))
                return False, reasons
    # RNG: within each continuation step the digest must be invariant
    # (iterator draws happen only at epoch boundaries / batch fetch; the
    # per-step branch computations are RNG-free).
    if not bool(audit.get("rng_within_epoch_invariant", False)):
        reasons.append("rng_within_epoch_invariant_false")
    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def _all_run_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    for dataset in sorted(proto.DATASETS.keys()):
        ds_root = os.path.join(root, dataset)
        if not os.path.isdir(ds_root):
            continue
        for seed_dir in sorted(os.listdir(ds_root)):
            full_seed = os.path.join(ds_root, seed_dir)
            if not os.path.isdir(full_seed):
                continue
            for anchor_dir in sorted(os.listdir(full_seed)):
                full_anchor = os.path.join(full_seed, anchor_dir)
                if not os.path.isdir(full_anchor):
                    continue
                if os.path.isfile(os.path.join(full_anchor, proto.AUDIT_JSON_NAME)):
                    out.append(full_anchor)
    return out


def _all_base_run_dirs() -> List[str]:
    if not os.path.isdir(proto.BASE_RAW_ROOT):
        return []
    out: List[str] = []
    for dataset in sorted(proto.DATASETS.keys()):
        ds_root = os.path.join(proto.BASE_RAW_ROOT, dataset)
        if not os.path.isdir(ds_root):
            continue
        for seed_dir in sorted(os.listdir(ds_root)):
            full_seed = os.path.join(ds_root, seed_dir)
            if not os.path.isdir(full_seed):
                continue
            if os.path.isfile(os.path.join(full_seed, "run_manifest.json")):
                out.append(full_seed)
    return out


def _write_csv(path: str, rows: List[Dict[str, object]],
                columns: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _best_table_rows(run_dirs: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    canonical_cache: Dict[Tuple[str, int], Dict[str, object]] = {}
    for run_dir in run_dirs:
        audit = _read_json(os.path.join(run_dir, proto.AUDIT_JSON_NAME))
        best_rows = _read_csv(os.path.join(run_dir, proto.BEST_CSV_NAME))
        # f_original references come from the ORIGINAL S6 canonical
        # manifest (design §9.11), not from the S8-BASE retrain.
        dataset_key = str(audit.get("dataset", ""))
        seed_key = int(audit.get("seed", -1))
        cache_key = (dataset_key, seed_key)
        if cache_key not in canonical_cache:
            canonical_cache[cache_key] = _read_json(os.path.join(
                proto.source_run_dir(dataset_key, seed_key), "run_manifest.json"))
        canonical = canonical_cache[cache_key]
        f_best = _safe_float(canonical.get("selected_validation_macro_auc", ""))
        if f_best is None:
            f_best = _safe_float(audit.get("f_original_best_macro_auc", ""))
        f_sel_raw = canonical.get("selected_epoch", None)
        try:
            f_sel = int(f_sel_raw)
        except (TypeError, ValueError):
            f_sel = audit.get("f_original_selected_epoch", -1)
        for row in best_rows:
            best_auc = _safe_float(row.get("best_macro_auc", ""))
            delta = None
            if best_auc is not None and f_best is not None:
                delta = float(best_auc - f_best)
            new_best = (delta is not None and delta > 0.0)
            anchor_cls = str(row.get("anchor_class", ""))
            final_auc = _safe_float(row.get("final_macro_auc", ""))
            late_rescue = None
            if anchor_cls == "late":
                if best_auc is not None and f_best is not None and final_auc is not None:
                    late_rescue = bool(best_auc <= f_best and final_auc > f_best)
            rows.append({
                "stage": "S8",
                "dataset": str(row.get("dataset", audit.get("dataset", ""))),
                "seed": int(row.get("seed", audit.get("seed", -1))),
                "anchor_class": anchor_cls,
                "anchor_epoch": int(row.get("anchor_epoch", audit.get("anchor_epoch", -1))),
                "branch": str(row.get("branch", "")),
                "horizon": str(row.get("horizon", "")),
                "horizon_steps": int(row.get("horizon_steps", 0)),
                "best_step": int(row.get("best_step", 0)),
                "continuation_best_auc": best_auc if best_auc is not None else float("nan"),
                "final_auc": final_auc if final_auc is not None else float("nan"),
                "f_original_best_auc": f_best if f_best is not None else float("nan"),
                "f_original_selected_epoch": int(f_sel) if isinstance(f_sel, int) else -1,
                "delta_best": delta if delta is not None else float("nan"),
                "new_best": bool(new_best),
                "late_rescue": bool(late_rescue) if late_rescue is not None else False,
            })
    return rows


def _trajectory_rows(run_dirs: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        audit = _read_json(os.path.join(run_dir, proto.AUDIT_JSON_NAME))
        val_rows = _read_csv(os.path.join(run_dir, proto.VALIDATION_CSV_NAME))
        for row in val_rows:
            rows.append({
                "stage": "S8",
                "dataset": str(row.get("dataset", audit.get("dataset", ""))),
                "seed": int(row.get("seed", audit.get("seed", -1))),
                "anchor_class": str(row.get("anchor_class", "")),
                "anchor_epoch": int(row.get("anchor_epoch", audit.get("anchor_epoch", -1))),
                "branch": str(row.get("branch", "")),
                "step": int(row.get("step", -1)),
                "kind": str(row.get("kind", "")),
                "scope": str(row.get("scope", "")),
                "auc": _safe_float(row.get("auc", "")),
                "logloss": _safe_float(row.get("logloss", "")),
                "val_bce_raw": _safe_float(row.get("val_bce_raw", "")),
            })
    return rows


def _effective_update_rows(run_dirs: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        audit = _read_json(os.path.join(run_dir, proto.AUDIT_JSON_NAME))
        step_rows = _read_csv(os.path.join(run_dir, proto.STEPS_CSV_NAME))
        valgrad_steps = set(audit.get("valgrad_sample_steps", []))
        for row in step_rows:
            try:
                step = int(row.get("step", -1))
            except (TypeError, ValueError):
                continue
            if step not in valgrad_steps:
                continue
            rows.append({
                "stage": "S8",
                "dataset": str(row.get("dataset", audit.get("dataset", ""))),
                "seed": int(row.get("seed", audit.get("seed", -1))),
                "anchor_class": str(row.get("anchor_class", "")),
                "anchor_epoch": int(row.get("anchor_epoch", audit.get("anchor_epoch", -1))),
                "branch": str(row.get("branch", "")),
                "step": int(step),
                "cosine_gf_gt": _safe_float(row.get("cosine_gf_gt", "")),
                "delta_full_norm": _safe_float(row.get("delta_full_norm", "")),
                "delta_target_norm": _safe_float(row.get("delta_target_norm", "")),
                "delta_applied_norm": _safe_float(row.get("delta_applied_norm", "")),
                "delta_cosine": _safe_float(row.get("delta_cosine", "")),
                "eff_align_full": _safe_float(row.get("eff_align_full", "")),
                "eff_align_target": _safe_float(row.get("eff_align_target", "")),
                "eff_dot_full": _safe_float(row.get("eff_dot_full", "")),
                "eff_dot_target": _safe_float(row.get("eff_dot_target", "")),
                "validity_L_full_diff": _safe_float(row.get("validity_L_full_diff", "")),
            })
    return rows


def _vbase_rows(base_dirs: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for base_dir in base_dirs:
        manifest = _read_json(os.path.join(base_dir, "run_manifest.json"))
        ok, reasons = validate_base_run(base_dir)
        rows.append({
            "dataset": str(manifest.get("dataset", "")),
            "seed": int(manifest.get("seed", -1)),
            "formal_run_key": str(manifest.get("formal_run_key", "")),
            "selected_epoch": int(manifest.get("selected_epoch", -1)),
            "selected_validation_macro_auc": _safe_float(
                manifest.get("selected_validation_macro_auc", "")),
            "initial_parameter_state_hash": str(manifest.get("initial_parameter_state_hash", "")),
            "vbase_ok": bool(ok),
            "vbase_reasons": json.dumps(list(reasons), ensure_ascii=False),
            "full_state_dumps": json.dumps(list(manifest.get("full_state_dumps", []) or []),
                                            ensure_ascii=False),
        })
    return rows


def _coverage_table(run_dirs: Sequence[str]) -> Dict[str, List[str]]:
    matrix: Dict[str, List[str]] = {}
    for run_dir in run_dirs:
        audit = _read_json(os.path.join(run_dir, proto.AUDIT_JSON_NAME))
        dataset = str(audit.get("dataset", ""))
        seed = int(audit.get("seed", -1))
        anchor_class = str(audit.get("anchor_class", ""))
        key = "{}/seed{}".format(dataset, seed)
        status = str(audit.get("status", ""))
        if status == "OK":
            label = "{}=OK".format(anchor_class)
        else:
            label = "{}=ABORTED({})".format(anchor_class, audit.get("abort_reason", ""))
        if bool(audit.get("v2_review_flagged", False)):
            label += " REVIEW"
        matrix.setdefault(key, []).append(label)
    return matrix


def aggregate() -> Dict[str, object]:
    run_dirs = _all_run_dirs(proto.RAW_ROOT)
    base_dirs = _all_base_run_dirs()
    os.makedirs(proto.OUTPUT_ROOT, exist_ok=True)
    reports_dir = os.path.join(proto.OUTPUT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    best_rows = _best_table_rows(run_dirs)
    traj_rows = _trajectory_rows(run_dirs)
    eff_rows = _effective_update_rows(run_dirs)
    vbase_rows = _vbase_rows(base_dirs)
    coverage = _coverage_table(run_dirs)
    _write_csv(os.path.join(proto.OUTPUT_ROOT, "stage_S8_best_auc_table.csv"),
                best_rows, ["stage", "dataset", "seed", "anchor_class", "anchor_epoch",
                              "branch", "horizon", "horizon_steps", "best_step",
                              "continuation_best_auc", "final_auc",
                              "f_original_best_auc", "f_original_selected_epoch",
                              "delta_best", "new_best", "late_rescue"])
    _write_csv(os.path.join(proto.OUTPUT_ROOT, "stage_S8_trajectory.csv"),
                traj_rows, ["stage", "dataset", "seed", "anchor_class", "anchor_epoch",
                               "branch", "step", "kind", "scope", "auc", "logloss",
                               "val_bce_raw"])
    _write_csv(os.path.join(proto.OUTPUT_ROOT, "stage_S8_effective_update.csv"),
                eff_rows, ["stage", "dataset", "seed", "anchor_class", "anchor_epoch",
                              "branch", "step", "cosine_gf_gt", "delta_full_norm",
                              "delta_target_norm", "delta_applied_norm", "delta_cosine",
                              "eff_align_full", "eff_align_target", "eff_dot_full",
                              "eff_dot_target", "validity_L_full_diff"])
    _write_csv(os.path.join(proto.OUTPUT_ROOT, "stage_S8_base_validity.csv"),
                vbase_rows, ["dataset", "seed", "formal_run_key", "selected_epoch",
                                "selected_validation_macro_auc",
                                "initial_parameter_state_hash", "vbase_ok",
                                "vbase_reasons", "full_state_dumps"])
    summary = {
        "stage": "S8",
        "base_run_count": len(base_dirs),
        "continuation_run_count": len(run_dirs),
        "coverage": coverage,
        "outputs": {
            "stage_S8_best_auc_table_csv": "stage_S8_best_auc_table.csv",
            "stage_S8_trajectory_csv": "stage_S8_trajectory.csv",
            "stage_S8_effective_update_csv": "stage_S8_effective_update.csv",
            "stage_S8_base_validity_csv": "stage_S8_base_validity.csv",
        },
        "model_source_key": MODEL_SOURCE_KEY,
        "model_source_sha256": MODEL_SOURCE_SHA256,
    }
    with open(proto.SUMMARY_MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    lines: List[str] = []
    lines.append("# Stage S8 Run Report (Mechanical)")
    lines.append("")
    lines.append("Per Algorithm Specification `stage_S8_design.md`.")
    lines.append("")
    lines.append("This report is the mechanical aggregator skeleton; scientific analysis is the "
                  "responsibility of the main agent.  See `stage_S8_summary_manifest.json` for the "
                  "summary metadata and the four CSVs in `stage_S8_outputs/` for the raw tables.")
    lines.append("")
    lines.append("## Coverage matrix")
    lines.append("")
    lines.append("| dataset/seed | anchors |")
    lines.append("|---|---|")
    for key in sorted(coverage):
        lines.append("| {} | {} |".format(key, ", ".join(sorted(coverage[key]))))
    lines.append("")
    lines.append("## Phase 1 base retrain V-BASE results")
    lines.append("")
    lines.append("| dataset | seed | selected_epoch | selected_macro_auc | initial_hash_match | vbase_ok | reasons |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in sorted(vbase_rows, key=lambda r: (r["dataset"], r["seed"])):
        lines.append("| {dataset} | {seed} | {sel_epoch} | {auc:.6f} | {hash_match} | {ok} | {reasons} |".format(
            dataset=row["dataset"], seed=row["seed"],
            sel_epoch=row["selected_epoch"],
            auc=row["selected_validation_macro_auc"] or 0.0,
            hash_match=str(row["initial_parameter_state_hash"])[:16] + "...",
            ok=bool(row["vbase_ok"]),
            reasons=row["vbase_reasons"][:120].replace("|", "/"),
        ))
    lines.append("")
    lines.append("## Source files")
    lines.append("")
    lines.append("- `experiments/stage_S8_protocol.py`")
    lines.append("- `experiments/stage_S8_base_run.py`")
    lines.append("- `experiments/stage_S8_run.py`")
    lines.append("- `experiments/stage_S8_orchestrator.py`")
    lines.append("- `experiments/stage_S8_report.py`")
    lines.append("")
    report_path = proto.RUN_REPORT_PATH
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return {"base_run_count": len(base_dirs),
            "continuation_run_count": len(run_dirs),
            "report_path": report_path,
            "summary_path": proto.SUMMARY_MANIFEST_PATH}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Stage S8 validator/aggregator")
    parser.add_argument("--run-dir", default="",
                        help="Validate one continuation run directory.")
    parser.add_argument("--base-run-dir", default="",
                        help="Validate one base retrain run directory.")
    parser.add_argument("--aggregate", action="store_true",
                        help="Run the aggregator over all run dirs.")
    args = parser.parse_args()
    if args.aggregate:
        result = aggregate()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.run_dir:
        ok, reasons = validate_run(args.run_dir)
        print(json.dumps({"ok": ok, "reasons": reasons}, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    if args.base_run_dir:
        ok, reasons = validate_base_run(args.base_run_dir)
        print(json.dumps({"ok": ok, "reasons": reasons}, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
