"""Stage S7-B validator, aggregator, run report.

- ``validate_run(run_dir)`` -> (ok, reasons).
- Aggregate 16 per-seed ``*.csv`` into three raw CSVs and a run report.
- Emit ``stage_S7B_outputs/stage_S7B_run_report.md``.

Per Algorithm Specification -- Stage S7-B.
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

import stage_S7B_protocol as proto  # noqa: E402
from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256  # noqa: E402
from stage_S4_protocol import SEEDS as S4_SEEDS  # noqa: E402
from stage_S7_probe import CELL_NUM, DOMAIN_NUM, MODULES  # noqa: E402


S4_TOLERANCE_ABS = proto.S4_TOLERANCE_ABS
S4_TOLERANCE_REL = proto.S4_TOLERANCE_REL


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


def validate_run(run_dir: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not os.path.isdir(run_dir):
        return False, ["run_dir_missing: {}".format(run_dir)]
    expected_files = (proto.STEPS_CSV_NAME, proto.GEOMETRY_CSV_NAME,
                       proto.VALIDATION_CSV_NAME, proto.AUDIT_JSON_NAME,
                       proto.PROCESS_INTERVAL_NAME)
    for filename in expected_files:
        if not os.path.isfile(os.path.join(run_dir, filename)):
            reasons.append("missing_artifact: {}".format(filename))
    if reasons:
        return False, reasons
    audit = _read_json(os.path.join(run_dir, proto.AUDIT_JSON_NAME))
    # An ABORTED run is NOT a passing deliverable: appending the abort
    # reason below makes validate_run return ok=False.  The per-step data
    # is still written and the abort reason is recorded in the audit, so
    # the run report can surface what happened.
    if str(audit.get("status", "")) == "ABORTED":
        reasons.append("run_aborted_acknowledged: {}".format(audit.get("abort_reason", "")))
    # Required fingerprint match.
    fp_rows = audit.get("fingerprint_rows", [])
    if not isinstance(fp_rows, list) or not fp_rows:
        reasons.append("fingerprint_rows_empty")
    else:
        for row in fp_rows:
            if not row.get("match", False):
                reasons.append("fingerprint_mismatch: " + json.dumps(row))
                return False, reasons
    # Step CSV row count: 2 branches * 5 steps = 10 rows.
    step_rows = _read_csv(os.path.join(run_dir, proto.STEPS_CSV_NAME))
    if len(step_rows) != 2 * proto.K_STEPS:
        reasons.append("step_row_count_mismatch: {} != {}".format(
            len(step_rows), 2 * proto.K_STEPS))
    # Geometry CSV row count: 2 branches * 5 steps * 36 sources * 36 targets * 8 modules.
    geom_rows = _read_csv(os.path.join(run_dir, proto.GEOMETRY_CSV_NAME))
    expected_geom = 2 * proto.K_STEPS * CELL_NUM * CELL_NUM * len(MODULES)
    if len(geom_rows) != expected_geom:
        reasons.append("geometry_row_count_mismatch: {} != {}".format(
            len(geom_rows), expected_geom))
    # Validation CSV row count: 2 branches * 6 steps (0..5) * 6 cells + 2 branches * 6 steps * 3 aggregates.
    val_rows = _read_csv(os.path.join(run_dir, proto.VALIDATION_CSV_NAME))
    expected_val = 2 * (proto.K_STEPS + 1) * CELL_NUM + 2 * (proto.K_STEPS + 1) * 3
    if len(val_rows) != expected_val:
        reasons.append("validation_row_count_mismatch: {} != {}".format(
            len(val_rows), expected_val))
    # Numerical finiteness.
    for row in step_rows:
        for field in ("L_full", "g_full_norm", "g_target_norm", "cosine_gf_gt",
                       "delta_full_norm", "delta_target_norm", "delta_applied_norm",
                       "delta_cosine", "theta_distance_shared",
                       "applied_formula_residual", "recon_residual_gS", "recon_residual_H"):
            if not _finite(row.get(field, "")):
                reasons.append("nonfinite_step: {}={}".format(field, row.get(field, "")))
                return False, reasons
    # Reconstructed step count + private policy.
    anchor_audit = audit.get("anchor_reconstruction", {})
    if not anchor_audit.get("moment_dump_path"):
        reasons.append("moment_dump_path_missing")
    if str(anchor_audit.get("private_policy", "")) != "fresh_adam_no_state":
        reasons.append("private_policy_unexpected: " + str(anchor_audit.get("private_policy", "")))
    # Reconstructed shared step = expected.
    shared_step = int(anchor_audit.get("shared_step", -1))
    dataset = str(audit.get("dataset", ""))
    anchor = int(audit.get("anchor", -1))
    if shared_step != proto.expected_shared_step(dataset, anchor):
        reasons.append("shared_step_mismatch: {} != {}".format(
            shared_step, proto.expected_shared_step(dataset, anchor)))
    # Cross-branch step0 cos(ΔF,ΔT) consistency within 1e-6.
    by_branch_step: Dict[Tuple[str, int], Dict[str, str]] = {}
    for row in step_rows:
        key = (str(row.get("branch", "")), int(row.get("step", -1)))
        by_branch_step[key] = row
    f0 = by_branch_step.get((proto.BRANCH_F, 0))
    t0 = by_branch_step.get((proto.BRANCH_T, 0))
    if f0 is not None and t0 is not None:
        f_cos = _safe_float(f0.get("delta_cosine"))
        t_cos = _safe_float(t0.get("delta_cosine"))
        if f_cos is not None and t_cos is not None:
            if abs(f_cos - t_cos) > proto.COS_DF_DT_CROSS_BRANCH_TOL:
                reasons.append("cos_df_dt_cross_branch_mismatch: F={} T={} diff={}".format(
                    f_cos, t_cos, abs(f_cos - t_cos)))
        # theta_distance_shared is recorded POST-update; spec's "step0 == 0"
        # check is enforced by the deepcopy parameter-equality assertion
        # inside ``branch_state`` (any pre-step divergence would have raised
        # before any step).  No additional CSV check needed here.
    # V3 step-0 pre-step private equality is enforced inside the run (it
    # raises there); reaching this point means the gate held.  The audit
    # additionally records the per-step post-step private divergence in
    # ``v3_per_step`` for transparency (structural at step >= 1).
    validity = audit.get("validity", [])
    if not isinstance(validity, list) or not validity:
        reasons.append("validity_missing")
    else:
        # The run completed without V3 raising; that is the gate.  We
        # additionally surface the per-step V0/V2 markers via the
        # report (see _validity_summary) but they do not affect the
        # validate_run ok flag when the run is non-aborted.
        pass
    # Replay audit fingerprint match all true.
    if not audit.get("fingerprint_match_all", False):
        reasons.append("fingerprint_match_all_false")
    # RNG state unchanged across continuation.
    if not audit.get("rng_state_unchanged", False):
        reasons.append("rng_state_changed")
    # Deep reconstruction checks (mirror the in-run hard gates, so a CSV
    # can be verified independently of the run):
    #   1. recon_residual_gS <= max(1e-6, 1e-4 * g_full_norm) per step row.
    #   2. recon_residual_H <= max(1e-6, 1e-4 * max|validation_dot_product|)
    #      over the geometry rows with shared_module == "ALL" of the same
    #      (branch, step).
    #   3. geometry source_weight == n_d / (2 B), where the counts come
    #      from the steps row's domain_counts (B = sum of counts).
    for row in step_rows:
        recon_gs = _safe_float(row.get("recon_residual_gS"))
        g_full = _safe_float(row.get("g_full_norm"))
        if recon_gs is not None and g_full is not None:
            tol = max(proto.S4_TOLERANCE_ABS, proto.S4_TOLERANCE_REL * g_full)
            if recon_gs > tol:
                reasons.append("recon_gS_above_tolerance: branch={} step={} {:.3e} > {:.3e}".format(
                    row.get("branch", ""), row.get("step", ""), recon_gs, tol))
    max_u_by_key: Dict[Tuple[str, int], float] = {}
    for row in geom_rows:
        if str(row.get("shared_module", "")) != "ALL":
            continue
        key = (str(row.get("branch", "")), int(row.get("step", -1)))
        u = _safe_float(row.get("validation_dot_product"))
        if u is None:
            continue
        if key not in max_u_by_key or abs(u) > max_u_by_key[key]:
            max_u_by_key[key] = abs(u)
    for row in step_rows:
        recon_h = _safe_float(row.get("recon_residual_H"))
        if recon_h is None:
            continue
        key = (str(row.get("branch", "")), int(row.get("step", -1)))
        u_max = max_u_by_key.get(key)
        if u_max is None:
            reasons.append("recon_H_reference_missing: branch={} step={}".format(*key))
            continue
        tol = max(proto.S4_TOLERANCE_ABS, proto.S4_TOLERANCE_REL * u_max)
        if recon_h > tol:
            reasons.append("recon_H_above_tolerance: branch={} step={} {:.3e} > {:.3e}".format(
                key[0], key[1], recon_h, tol))
    # source_weight identity: w_{d,t} = n_d / (2 B).
    counts_by_key: Dict[Tuple[str, int], List[int]] = {}
    for row in step_rows:
        key = (str(row.get("branch", "")), int(row.get("step", -1)))
        try:
            counts = [int(v) for v in json.loads(str(row.get("domain_counts", "[]")))]
        except (TypeError, ValueError):
            reasons.append("domain_counts_unparseable: branch={} step={}".format(*key))
            continue
        if len(counts) != DOMAIN_NUM:
            reasons.append("domain_counts_length_mismatch: branch={} step={}".format(*key))
            continue
        counts_by_key[key] = counts
    for row in geom_rows:
        key = (str(row.get("branch", "")), int(row.get("step", -1)))
        counts = counts_by_key.get(key)
        if counts is None:
            continue
        w_obs = _safe_float(row.get("source_weight"))
        if w_obs is None:
            continue
        s_d = int(row.get("source_domain", -1))
        batch_total = float(sum(counts))
        w_expected = float(counts[s_d]) / float(2.0 * batch_total)
        if abs(w_obs - w_expected) > 1e-9 * max(1.0, abs(w_expected)):
            reasons.append(
                "source_weight_identity_violation: branch={} step={} source_domain={} "
                "observed={:.12g} expected={:.12g}".format(
                    key[0], key[1], s_d, w_obs, w_expected))
            return False, reasons
    return (len(reasons) == 0, reasons)


def _all_run_dirs() -> List[str]:
    if not os.path.isdir(proto.RAW_ROOT):
        return []
    out: List[str] = []
    for dataset in sorted(proto.DATASETS.keys()):
        ds_root = os.path.join(proto.RAW_ROOT, dataset)
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


def _aggregate_csvs(run_dirs: Sequence[str]) -> Tuple[List[Dict[str, str]],
                                                       List[Dict[str, str]],
                                                       List[Dict[str, str]]]:
    step_rows: List[Dict[str, str]] = []
    geom_rows: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []
    for run_dir in run_dirs:
        step_rows.extend(_read_csv(os.path.join(run_dir, proto.STEPS_CSV_NAME)))
        geom_rows.extend(_read_csv(os.path.join(run_dir, proto.GEOMETRY_CSV_NAME)))
        val_rows.extend(_read_csv(os.path.join(run_dir, proto.VALIDATION_CSV_NAME)))
    return step_rows, geom_rows, val_rows


STEPS_AGGREGATED_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "seed", "anchor_epoch", "branch", "epoch", "step",
    "batch_fingerprint", "domain_counts",
    "L_full", "L_target", "L_non",
    "g_full_norm", "g_target_norm", "cosine_gf_gt",
    "delta_full_norm", "delta_target_norm", "delta_applied_norm",
    "delta_cosine", "private_delta_norm", "theta_distance_shared",
    "applied_formula_residual", "recon_residual_gS", "recon_residual_H",
    "rng_digest", "validity_L_full_diff", "validity_delta_norm_rel_diff",
)
GEOMETRY_AGGREGATED_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "seed", "anchor_epoch", "branch", "step",
    "source_domain", "source_task", "target_domain", "target_task",
    "pair_class", "shared_module",
    "gradient_norm", "pairwise_cosine", "sample_alignment",
    "validation_dot_product", "weighted_source_target_contribution",
    "validation_gradient_norm", "source_weight",
)
VALIDATION_AGGREGATED_COLUMNS: Tuple[str, ...] = (
    "stage", "dataset", "seed", "anchor_epoch", "branch", "step",
    "scope", "domain", "task", "cell",
    "val_bce_raw", "auc", "logloss",
)


def _write_csv(path: str, rows: List[Dict[str, object]],
                columns: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _coverage_matrix(run_dirs: Sequence[str]) -> Dict[str, List[str]]:
    matrix: Dict[str, List[str]] = {}
    for run_dir in run_dirs:
        audit = _read_json(os.path.join(run_dir, proto.AUDIT_JSON_NAME))
        dataset = str(audit.get("dataset", ""))
        seed = int(audit.get("seed", -1))
        anchor = int(audit.get("anchor", -1))
        key = "{}/seed{}".format(dataset, seed)
        status = str(audit.get("status", ""))
        if status == "OK":
            label = "anchor{}=OK".format(anchor)
        else:
            label = "anchor{}=ABORTED({})".format(anchor, audit.get("abort_reason", ""))
        if bool(audit.get("v2_review_flagged", False)):
            label += " REVIEW"
        matrix.setdefault(key, []).append(label)
    return matrix


def _validity_summary(run_dirs: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        audit = _read_json(os.path.join(run_dir, proto.AUDIT_JSON_NAME))
        dataset = str(audit.get("dataset", ""))
        seed = int(audit.get("seed", -1))
        anchor = int(audit.get("anchor", -1))
        for entry in audit.get("validity", []):
            if entry.get("kind") == "L_full_and_delta_observed":
                rows.append({
                    "dataset": dataset, "seed": seed, "anchor": anchor, "step": entry.get("step"),
                    "validity_layer": entry.get("validity_layer", ""),
                    "l_full_abs_diff": entry.get("l_full_abs_diff"),
                    "delta_norm_rel_diff": entry.get("delta_norm_rel_diff"),
                    "aborted_v0_l_full": entry.get("aborted_v0_l_full"),
                    "aborted_v0_delta_rel": entry.get("aborted_v0_delta_rel"),
                    "review_v2_tripwire": entry.get("review_v2_tripwire"),
                })
    return rows


def _fingerprint_summary(run_dirs: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        audit = _read_json(os.path.join(run_dir, proto.AUDIT_JSON_NAME))
        dataset = str(audit.get("dataset", ""))
        seed = int(audit.get("seed", -1))
        anchor = int(audit.get("anchor", -1))
        fp_rows = audit.get("fingerprint_rows", [])
        all_match = bool(audit.get("fingerprint_match_all", False))
        rows.append({"dataset": dataset, "seed": seed, "anchor": anchor,
                      "all_match": all_match, "row_count": len(fp_rows)})
    return rows


def aggregate() -> Dict[str, object]:
    run_dirs = _all_run_dirs()
    os.makedirs(proto.OUTPUT_ROOT, exist_ok=True)
    step_rows, geom_rows, val_rows = _aggregate_csvs(run_dirs)
    _write_csv(os.path.join(proto.OUTPUT_ROOT, "stage_S7B_steps_raw.csv"),
                step_rows, STEPS_AGGREGATED_COLUMNS)
    _write_csv(os.path.join(proto.OUTPUT_ROOT, "stage_S7B_geometry_raw.csv"),
                geom_rows, GEOMETRY_AGGREGATED_COLUMNS)
    _write_csv(os.path.join(proto.OUTPUT_ROOT, "stage_S7B_validation_raw.csv"),
                val_rows, VALIDATION_AGGREGATED_COLUMNS)
    coverage = _coverage_matrix(run_dirs)
    validity = _validity_summary(run_dirs)
    fingerprints = _fingerprint_summary(run_dirs)
    # Build report markdown.
    lines: List[str] = []
    lines.append("# Stage S7-B: Same-θ Early/Late Causal Continuation Run Report")
    lines.append("")
    lines.append("Per Algorithm Specification -- Stage S7-B.  T-branch semantics = S6 DH-arm current "
                  "manipulation (moments<-g_F, shared step overwritten by pre-step Δ_T).  F-branch "
                  "semantics = S6 F arm (bit-for-bit on step 0).")
    lines.append("")
    lines.append("**Geometry口径差异声明**：S7-B geometry 是**单 actual-batch**（每个 step = 一次 train "
                  "batch 的真实更新），与 S7-A 的 4-batch 平均**不可直接跨阶段比较绝对值**。本报告所有 "
                  "U_k / H 读出、cell 几何仅在 S7-B 内部比较。")
    lines.append("")
    lines.append("**Validity contract (V0 / V1 / V2 / V3)**：")
    lines.append("- **V0 (硬门, step 0 only)**: fingerprint 逐位相等（已）+ |ΔL_full| ≤ 1e-5 + "
                  "shared `delta_observed_step_norm` 相对差 ≤ 1e-3；任一失败即 abort。step 0 认证重建正确性。")
    lines.append("- **V1 (硬门, 每步, 双 branch)**: `controlled_continuation_step` 内既有断言 "
                  "（shared `assert_vector_match(observed, Δ)`、installed cosine、private 残差）— "
                  "V1 失败时 step 不会产生 CSV 行。")
    lines.append("- **V2 (仅上报, step ≥ 1)**: F-branch 对 S6 原轨迹的 L_full diff 与 shared "
                  "delta 相对差逐步记录（沿用 ``validity_L_full_diff`` / "
                  "``validity_delta_norm_rel_diff`` 列）。**不设 abort 阈值**。任一步 "
                  "|L_full diff| > 0.3 在该 run 的 report 行打 `REVIEW` 旗标。")
    lines.append("- **V3 (硬门, 仅 step 0, pre-step)**: 双 branch 间 private 参数逐 tensor "
                  "`torch.equal`；失败意味着 branch 对称性被破坏 → run abort。step ≥ 1 的 "
                  "private 分歧是 T-branch 在 step 0 覆写 shared（Δ_T）后双 branch 的 pre-step "
                  "shared θ 分歧、进而 g_P 分歧的结构后果，**非 V3 违反**，仅以 "
                  "`v3_per_step` 记录。step-0 跨 branch cos(Δ_F, Δ_T) 一致性断言保留。")
    lines.append("")
    lines.append("**Known approximation**：private moments not in any S6/S6ML product; per Algorithm "
                  "Spec §4, private parameters are installed with no state (fresh Adam, step=0).  At "
                  "step 0 the two branches' private parameters are bit-identical (same anchor "
                  "deepcopy); at step ≥ 1 the private trajectories diverge together with the shared "
                  "θ divergence — a structural consequence of the T-branch semantics, so the F vs T "
                  "causal contrast on shared parameters is unaffected.  Per V2 above, step ≥ 1 的 "
                  "F-branch 对 S6 原轨迹分歧是已声明的 private fresh-Adam 近似的预期后果，"
                  "**不构成 run 无效**。")
    lines.append("")
    lines.append("## Run matrix coverage")
    lines.append("")
    lines.append("| dataset/seed | anchors |")
    lines.append("|---|---|")
    for key in sorted(coverage):
        lines.append("| {} | {} |".format(key, ", ".join(sorted(coverage[key]))))
    lines.append("")
    lines.append("## Fingerprint replay audit (epoch {} step 0..{})".format(
        "anchor+1", proto.K_STEPS - 1))
    lines.append("")
    lines.append("| dataset | seed | anchor | all_match | row_count |")
    lines.append("|---|---|---|---|---|")
    for row in sorted(fingerprints, key=lambda r: (r["dataset"], r["seed"], r["anchor"])):
        lines.append("| {dataset} | {seed} | {anchor} | {all_match} | {row_count} |".format(**row))
    lines.append("")
    lines.append("## Validity (F branch vs S6 F source) -- V0 step 0 / V2 step >= 1")
    lines.append("")
    lines.append("| dataset | seed | anchor | step | layer | l_full_abs_diff | delta_norm_rel_diff | v0_abort | v2_review |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in sorted(validity, key=lambda r: (r["dataset"], r["seed"], r["anchor"], r["step"])):
        v0_abort = bool(row.get("aborted_v0_l_full") or row.get("aborted_v0_delta_rel"))
        v2_review = bool(row.get("review_v2_tripwire"))
        lines.append("| {dataset} | {seed} | {anchor} | {step} | {layer} | {l_full_abs_diff:.3e} | {delta_norm_rel_diff:.3e} | {v0_abort} | {v2_review} |".format(
            dataset=row["dataset"], seed=row["seed"], anchor=row["anchor"], step=row["step"],
            layer=row.get("validity_layer", ""),
            l_full_abs_diff=row.get("l_full_abs_diff") or 0.0,
            delta_norm_rel_diff=row.get("delta_norm_rel_diff") or 0.0,
            v0_abort=v0_abort, v2_review=v2_review))
    lines.append("")
    lines.append("## Failure log")
    lines.append("")
    has_failure = False
    for run_dir in run_dirs:
        audit = _read_json(os.path.join(run_dir, proto.AUDIT_JSON_NAME))
        if str(audit.get("status")) != "OK":
            has_failure = True
            lines.append("- `{}`: status={} abort_reason={}".format(
                run_dir, audit.get("status"), audit.get("abort_reason")))
    if not has_failure:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Source files")
    lines.append("")
    lines.append("- `experiments/stage_S7B_protocol.py`")
    lines.append("- `experiments/stage_S7B_run.py`")
    lines.append("- `experiments/stage_S7B_orchestrator.py`")
    lines.append("- `experiments/stage_S7B_report.py`")
    lines.append("")
    report_path = proto.RUN_REPORT_PATH
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return {"run_count": len(run_dirs), "step_rows": len(step_rows),
            "geometry_rows": len(geom_rows), "validation_rows": len(val_rows),
            "report_path": report_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage S7-B validator/aggregator")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        result = aggregate()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.run_dir:
        ok, reasons = validate_run(args.run_dir)
        print(json.dumps({"ok": ok, "reasons": reasons}, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
