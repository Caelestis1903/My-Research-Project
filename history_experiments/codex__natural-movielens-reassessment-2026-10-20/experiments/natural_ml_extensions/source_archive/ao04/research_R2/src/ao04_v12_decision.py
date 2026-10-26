"""Stage R2 decision module — validation-only gate.

Reads the validation-side artifacts from completed R2-A runs, computes
the D_s / T_s statistics and guards, and produces an immutable
``r2a_decision_<dataset>.json`` manifest.

No test array is ever read or scored here.  The module-level
``TEST_PREDICTION_CALLS`` counter must equal zero before any verdict is
computed; that assertion is enforced before invoking the gate.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

R2_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(R2_ROOT))
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")
M3OE_ROOT = os.path.join(PROJECT_ROOT, "m3oe")
for _path in (PROJECT_ROOT, EXPERIMENTS_ROOT, M3OE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stage_E1_instrumentation import write_json  # noqa: E402

from ao04_v12_protocol import (  # noqa: E402
    BUDGET_PER_DATASET,
    DATASETS,
    DATASET_SETTINGS,
    DOMAIN_NUM,
    EPOCHS,
    EXPLORATORY_D_MEAN,
    GUARD_G_A,
    GUARD_G_B,
    GUARD_G_C,
    GUARD_G_D,
    GUARD_G_E,
    GUARD_G_F,
    PRACTICAL_D_MEAN,
    R2A_ARMS,
    SEEDS,
    TASK_NUM,
    TASK_GAIN_MEAN,
)
import ao04_v12_runner as _runner_mod  # noqa: E402
from ao04_v12_runner import select_checkpoint_epoch  # noqa: E402


# ---------------------------------------------------------------------------
# CSV / manifest loaders.
# ---------------------------------------------------------------------------

def _read_metrics_epoch(metrics_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(metrics_path)
    with open(metrics_path, encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def _read_selection(selection_path: str) -> Dict[str, Any]:
    if not os.path.exists(selection_path):
        raise FileNotFoundError(selection_path)
    with open(selection_path, encoding="utf-8") as h:
        return json.load(h)


def _read_run_manifest(manifest_path: str) -> Dict[str, Any]:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(manifest_path)
    with open(manifest_path, encoding="utf-8") as h:
        return json.load(h)


def _read_finite_float(row: Mapping[str, Any], field: str) -> float:
    raw = row.get(field, "")
    value = float(raw)
    if not math.isfinite(value):
        raise FloatingPointError(
            "field {!r} not finite: {!r}".format(field, raw))
    return value


# ---------------------------------------------------------------------------
# Per-run metrics collector.
# ---------------------------------------------------------------------------

@dataclass
class PerRunMetrics:
    run_dir: str
    arm: str
    seed: int
    selected_epoch: int
    selected_macro_auc: float
    per_cell_auc: Dict[Tuple[int, int], float]  # (task, domain) -> AUC
    per_cell_logloss: Dict[Tuple[int, int], float]
    per_task_macro_auc: Dict[int, float]
    per_task_macro_logloss: Dict[int, float]
    train_domain_shares: Dict[int, float]
    target_weighted_auc: float
    target_weighted_logloss: float
    macro_logloss: float
    formal_run_key: str
    run_manifest_sha256: str
    initial_parameter_state_hash: str
    final_parameter_state_hash: str
    batch_order_hash: str
    architecture_batch_order_hash: str
    status: str
    epochs_in_metrics: int
    macro_auc_by_epoch: Dict[int, float]
    paired_init_manifest_sha256: str
    epochs_completed: int
    resumed_from: Optional[str]


def collect_per_run_metrics(run_dir: str, arm: str, seed: int,
                             *, domain_num: int = DOMAIN_NUM,
                             task_num: int = TASK_NUM) -> PerRunMetrics:
    """Load one run's validation metrics from CSV."""

    metrics_path = os.path.join(run_dir, "metrics_epoch.csv")
    selection_path = os.path.join(run_dir, "selection.json")
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    rows = _read_metrics_epoch(metrics_path)
    selection = _read_selection(selection_path)
    manifest = _read_run_manifest(manifest_path)
    selected_epoch = int(manifest["selected_epoch"])
    # Filter rows at the selected epoch only.
    selected_rows = [r for r in rows if int(r["epoch"]) == selected_epoch]
    if not selected_rows:
        raise RuntimeError(
            "no metrics rows for selected epoch {} in {}".format(
                selected_epoch, metrics_path))

    per_cell_auc: Dict[Tuple[int, int], float] = {}
    per_cell_logloss: Dict[Tuple[int, int], float] = {}
    per_task_auc_acc: Dict[int, List[float]] = {t: [] for t in range(task_num)}
    per_task_ll_acc: Dict[int, List[float]] = {t: [] for t in range(task_num)}
    for row in selected_rows:
        t = int(row["task"]); d = int(row["domain"])
        per_cell_auc[(t, d)] = _read_finite_float(row, "validation_auc")
        per_cell_logloss[(t, d)] = _read_finite_float(row, "validation_logloss")
        per_task_auc_acc[t].append(per_cell_auc[(t, d)])
        per_task_ll_acc[t].append(per_cell_logloss[(t, d)])
    per_task_macro_auc = {t: float(sum(v) / len(v))
                          for t, v in per_task_auc_acc.items() if v}
    per_task_macro_logloss = {t: float(sum(v) / len(v))
                              for t, v in per_task_ll_acc.items() if v}

    # Target-weighted AUC requires domain shares; pull them from the
    # bound bundle manifest via the row 'run_manifest.json.dataset..'.
    # In our schema these are already echoed in metrics_epoch row via
    # 'dataset' and 'cell'.  We compute using the manifest's
    # 'dataset_domain_shares' field if present.
    domain_shares_field = manifest.get("dataset_domain_shares", {})
    shares: Dict[int, float] = {}
    if domain_shares_field:
        shares = {int(k): float(v) for k, v in domain_shares_field.items()}
    else:
        # Fall back to the expected train counts in the protocol.
        from ao04_v12_protocol import EXPECTED_TRAIN_SHARES
        dataset_key = manifest.get("dataset", "")
        if dataset_key in EXPECTED_TRAIN_SHARES:
            shares = EXPECTED_TRAIN_SHARES[dataset_key]
    if not shares:
        raise RuntimeError("domain shares unavailable for {}".format(run_dir))
    target_weighted_auc = sum(
        shares[d] * per_cell_auc[(t, d)] for t in range(task_num) for d in range(domain_num)
    ) / float(task_num)
    target_weighted_logloss = sum(
        shares[d] * per_cell_logloss[(t, d)] for t in range(task_num) for d in range(domain_num)
    ) / float(task_num)
    macro_logloss = float(np_mean([per_cell_logloss[(t, d)]
                                    for t in range(task_num)
                                    for d in range(domain_num)]))
    macro_auc = float(manifest["selected_validation_macro_auc"])
    selected_macro_auc = macro_auc
    macro_auc_by_epoch = {int(k): float(v) for k, v in
                          selection.get("macro_auc_by_epoch", {}).items()}
    return PerRunMetrics(
        run_dir=run_dir,
        arm=arm,
        seed=seed,
        selected_epoch=selected_epoch,
        selected_macro_auc=selected_macro_auc,
        per_cell_auc=per_cell_auc,
        per_cell_logloss=per_cell_logloss,
        per_task_macro_auc=per_task_macro_auc,
        per_task_macro_logloss=per_task_macro_logloss,
        train_domain_shares=shares,
        target_weighted_auc=target_weighted_auc,
        target_weighted_logloss=target_weighted_logloss,
        macro_logloss=macro_logloss,
        formal_run_key=("joint|" + str(manifest["dataset"]) + "|" +
                         str(manifest["setting"]) + "|m3oe|" + str(seed)),
        run_manifest_sha256=_file_sha256(manifest_path),
        initial_parameter_state_hash=str(manifest.get("initial_parameter_state_hash", "")),
        final_parameter_state_hash=str(manifest.get("final_parameter_state_hash", "")),
        batch_order_hash=str(manifest.get("batch_order_hash", "")),
        architecture_batch_order_hash=str(manifest.get("architecture_batch_order_hash", "")),
        status=str(manifest.get("status", "")),
        epochs_in_metrics=len({int(r["epoch"]) for r in rows}),
        macro_auc_by_epoch=macro_auc_by_epoch,
        paired_init_manifest_sha256=str(
            manifest.get("paired_init_manifest_sha256", "")),
        epochs_completed=int(manifest.get("epochs_completed", 0)),
        resumed_from=manifest.get("resumed_from"),
    )


def np_mean(values: Iterable[float]) -> float:
    arr = [float(v) for v in values]
    if not arr:
        raise ValueError("np_mean called on empty iterable")
    return sum(arr) / len(arr)


def _file_sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Guard evaluators (B1 − B0 per seed).
# ---------------------------------------------------------------------------

@dataclass
class GuardsResult:
    g_a_per_task_mean: Dict[int, float]   # by task
    g_a_pass: bool
    g_b_W_mean: float
    g_b_pass: bool
    g_c_per_cell_mean: Dict[Tuple[int, int], float]
    g_c_pass: bool
    g_d_per_cell_single_seed: Dict[Tuple[int, int, int], float]  # (task, domain, seed)
    g_d_pass: bool
    g_e_per_cell_mean: Dict[Tuple[int, int], float]
    g_e_pass: bool
    g_f_macro_logloss_mean: float
    g_f_pass: bool

    def all_pass(self) -> bool:
        return all([self.g_a_pass, self.g_b_pass, self.g_c_pass,
                    self.g_d_pass, self.g_e_pass, self.g_f_pass])


def evaluate_guards(b0_by_seed: Dict[int, PerRunMetrics],
                     b1_by_seed: Dict[int, PerRunMetrics]) -> GuardsResult:
    """Compute G-a..G-f means and pass flags."""

    seeds = sorted(set(b0_by_seed.keys()) & set(b1_by_seed.keys()))
    if not seeds:
        raise ValueError("no overlapping seeds for guards evaluation")
    # G-a: per task macro AUC difference mean across seeds >= GUARD_G_A.
    g_a_per_task_mean: Dict[int, float] = {}
    g_a_pass_all: List[bool] = []
    for t in range(TASK_NUM):
        diffs = [b1_by_seed[s].per_task_macro_auc[t] - b0_by_seed[s].per_task_macro_auc[t]
                 for s in seeds]
        mean = float(sum(diffs) / len(diffs))
        g_a_per_task_mean[t] = mean
        g_a_pass_all.append(mean >= GUARD_G_A)
    # G-b: W difference mean >= GUARD_G_B.
    diffs_w = [b1_by_seed[s].target_weighted_auc - b0_by_seed[s].target_weighted_auc
               for s in seeds]
    g_b_mean = float(sum(diffs_w) / len(diffs_w))
    g_b_pass = g_b_mean >= GUARD_G_B
    # G-c: per-cell AUC difference mean >= GUARD_G_C.
    g_c_per_cell_mean: Dict[Tuple[int, int], float] = {}
    g_c_pass_all: List[bool] = []
    for t in range(TASK_NUM):
        for d in range(DOMAIN_NUM):
            diffs = [b1_by_seed[s].per_cell_auc[(t, d)] - b0_by_seed[s].per_cell_auc[(t, d)]
                     for s in seeds]
            mean = float(sum(diffs) / len(diffs))
            g_c_per_cell_mean[(t, d)] = mean
            g_c_pass_all.append(mean >= GUARD_G_C)
    # G-d: per-cell, per-seed single minimum >= GUARD_G_D.
    g_d_per_cell: Dict[Tuple[int, int, int], float] = {}
    g_d_pass_all: List[bool] = []
    for t in range(TASK_NUM):
        for d in range(DOMAIN_NUM):
            for s in seeds:
                diff = b1_by_seed[s].per_cell_auc[(t, d)] - b0_by_seed[s].per_cell_auc[(t, d)]
                key = (t, d, s)
                g_d_per_cell[key] = diff
                g_d_pass_all.append(diff >= GUARD_G_D)
    # G-e: per-cell LogLoss difference mean <= GUARD_G_E.
    g_e_per_cell: Dict[Tuple[int, int], float] = {}
    g_e_pass_all: List[bool] = []
    for t in range(TASK_NUM):
        for d in range(DOMAIN_NUM):
            diffs = [b1_by_seed[s].per_cell_logloss[(t, d)] - b0_by_seed[s].per_cell_logloss[(t, d)]
                     for s in seeds]
            mean = float(sum(diffs) / len(diffs))
            g_e_per_cell[(t, d)] = mean
            g_e_pass_all.append(mean <= GUARD_G_E)
    # G-f: macro LogLoss diff mean <= GUARD_G_F.
    diffs_ll = [b1_by_seed[s].macro_logloss - b0_by_seed[s].macro_logloss
                for s in seeds]
    g_f_mean = float(sum(diffs_ll) / len(diffs_ll))
    g_f_pass = g_f_mean <= GUARD_G_F

    return GuardsResult(
        g_a_per_task_mean=g_a_per_task_mean,
        g_a_pass=all(g_a_pass_all),
        g_b_W_mean=g_b_mean,
        g_b_pass=g_b_pass,
        g_c_per_cell_mean=g_c_per_cell_mean,
        g_c_pass=all(g_c_pass_all),
        g_d_per_cell_single_seed=g_d_per_cell,
        g_d_pass=all(g_d_pass_all),
        g_e_per_cell_mean=g_e_per_cell,
        g_e_pass=all(g_e_pass_all),
        g_f_macro_logloss_mean=g_f_mean,
        g_f_pass=g_f_pass,
    )


# ---------------------------------------------------------------------------
# Decision gate.
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    dataset: str
    decision: str  # "YES" or "NO"
    signal_class: str  # "practical", "exploratory", "none"
    D_s_by_seed: Dict[int, float]
    T_s_by_seed: Dict[int, float]
    mean_D: float
    mean_T: float
    guards: GuardsResult
    reasons: List[str]
    valid: bool
    validity_note: str
    decision_manifest_path: Optional[str]
    test_prediction_calls: int


def compute_r2a_decision(dataset: str,
                          run_dirs: Mapping[Tuple[str, int], str],
                          *,
                          write_to: Optional[str] = None,
                          code_hashes: Optional[Mapping[str, str]] = None,
                          ) -> DecisionResult:
    """Compute and (optionally) write the R2-A decision manifest.

    Parameters
    ----------
    dataset : str
        One of ``DATASETS``.

    run_dirs : Mapping
        Maps ``(arm, seed)`` pairs to completed run directories.
        Must contain B0 (2022, 2023), B1 (2022, 2023), B2 (2022,
        2023).  Missing entries are treated as invalid.

    write_to : str, optional
        Path to write the immutable decision manifest.  If ``None``
        the manifest is returned without writing.
    """

    if dataset not in DATASETS:
        raise ValueError("unsupported dataset: {}".format(dataset))
    if _runner_mod.TEST_PREDICTION_CALLS != 0:
        raise RuntimeError(
            "TEST_PREDICTION_CALLS={} (must be 0 before decision)".format(
                _runner_mod.TEST_PREDICTION_CALLS))

    validity_reasons: List[str] = []
    b0: Dict[int, PerRunMetrics] = {}
    b1: Dict[int, PerRunMetrics] = {}
    b2: Dict[int, PerRunMetrics] = {}
    for seed in SEEDS:
        for arm, store in (("B0", b0), ("B1", b1), ("B2", b2)):
            if (arm, seed) not in run_dirs:
                validity_reasons.append(
                    "missing {} run for seed {}".format(arm, seed))
                continue
            try:
                store[seed] = collect_per_run_metrics(
                    run_dirs[(arm, seed)], arm, seed)
            except Exception as exc:
                validity_reasons.append(
                    "run {} seed {} unreadable: {}: {}".format(
                        arm, seed, type(exc).__name__, exc))

    # Per-run validity: state-machine status, full epoch coverage, and an
    # independent recomputation of checkpoint selection from selection.json.
    collected: List[PerRunMetrics] = ([r for r in b0.values()]
                                      + [r for r in b1.values()]
                                      + [r for r in b2.values()])
    for r in collected:
        tag = "{} seed {}".format(r.arm, r.seed)
        if r.status != "COMPLETE":
            validity_reasons.append(
                "{} run manifest status {!r} != COMPLETE".format(tag, r.status))
        if r.epochs_in_metrics != EPOCHS:
            validity_reasons.append(
                "{} metrics_epoch.csv holds {} distinct epochs != {}".format(
                    tag, r.epochs_in_metrics, EPOCHS))
        if r.macro_auc_by_epoch:
            recomputed = select_checkpoint_epoch(r.macro_auc_by_epoch)
            if recomputed != r.selected_epoch:
                validity_reasons.append(
                    "{} selection recompute {} != manifest selected_epoch {}".format(
                        tag, recomputed, r.selected_epoch))
        else:
            validity_reasons.append(
                "{} selection.json lacks macro_auc_by_epoch".format(tag))

    # Exposure pairing: for each seed the three arms must share identical
    # train / architecture batch order hashes and the same paired-init
    # manifest (Specification §8.4).
    for seed in SEEDS:
        group = [store.get(seed) for store in (b0, b1, b2)]
        if any(r is None for r in group):
            continue
        for field_name in ("batch_order_hash", "architecture_batch_order_hash",
                           "paired_init_manifest_sha256"):
            values = {getattr(r, field_name) for r in group}
            if len(values) != 1:
                validity_reasons.append(
                    "seed {} exposure pairing broken on {}: {}".format(
                        seed, field_name, sorted(values)))

    valid = (len(b0) == len(SEEDS) and len(b1) == len(SEEDS)
             and len(b2) == len(SEEDS) and not validity_reasons)
    validity_note = ""
    if not valid:
        validity_note = "missing or invalid runs: " + "; ".join(validity_reasons)

    D_by_seed: Dict[int, float] = {}
    T_by_seed: Dict[int, float] = {}
    if valid:
        for s in SEEDS:
            D_by_seed[s] = float(b1[s].selected_macro_auc - b0[s].selected_macro_auc)
            T_by_seed[s] = float(b1[s].selected_macro_auc - b2[s].selected_macro_auc)

    guards: Optional[GuardsResult] = None
    signal_class = "none"
    decision = "NO"
    reasons: List[str] = []
    if valid:
        guards = evaluate_guards(b0, b1)
        mean_D = sum(D_by_seed.values()) / len(D_by_seed)
        mean_T = sum(T_by_seed.values()) / len(T_by_seed)
        if (D_by_seed[SEEDS[0]] > 0 and D_by_seed[SEEDS[1]] > 0
                and mean_D >= PRACTICAL_D_MEAN and guards.all_pass()):
            signal_class = "practical"
        elif (D_by_seed[SEEDS[0]] > 0 and D_by_seed[SEEDS[1]] > 0
              and EXPLORATORY_D_MEAN <= mean_D < PRACTICAL_D_MEAN
              and guards.all_pass()):
            signal_class = "exploratory"
        else:
            signal_class = "none"
            reasons.append("D_s thresholds not satisfied or guards failed")
        practical_signal = signal_class in ("practical", "exploratory")
        task_gain = (T_by_seed[SEEDS[0]] > 0 and T_by_seed[SEEDS[1]] > 0
                      and mean_T >= TASK_GAIN_MEAN)
        if practical_signal and task_gain:
            decision = "YES"
        else:
            if not task_gain:
                reasons.append("task_gain not satisfied (T_s <=0 or mean(T)<{})".format(TASK_GAIN_MEAN))
        if guards is not None and not guards.all_pass():
            reasons.append("at least one guard failed")
    else:
        reasons.append("validity check failed")
        mean_D = float("nan")
        mean_T = float("nan")

    manifest: Dict[str, Any] = {
        "dataset": dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "signal_class": signal_class,
        "D_s_by_seed": {str(s): float(v) for s, v in D_by_seed.items()},
        "T_s_by_seed": {str(s): float(v) for s, v in T_by_seed.items()},
        "mean_D": float(mean_D) if valid else float("nan"),
        "mean_T": float(mean_T) if valid else float("nan"),
        "valid": bool(valid),
        "validity_note": validity_note,
        "reasons": reasons,
        "guard_thresholds": {
            "G_a": GUARD_G_A, "G_b": GUARD_G_B,
            "G_c": GUARD_G_C, "G_d": GUARD_G_D,
            "G_e": GUARD_G_E, "G_f": GUARD_G_F,
        },
        "decision_thresholds": {
            "PRACTICAL_D_MEAN": PRACTICAL_D_MEAN,
            "EXPLORATORY_D_MEAN": EXPLORATORY_D_MEAN,
            "TASK_GAIN_MEAN": TASK_GAIN_MEAN,
        },
        "test_prediction_calls_at_decision": int(_runner_mod.TEST_PREDICTION_CALLS),
        "code_hashes": dict(code_hashes or {}),
        "decision_code_sha256": _file_sha256(
            os.path.join(R2_ROOT, "ao04_v12_decision.py")),
        "runs": [
            {
                "arm": r.arm,
                "seed": r.seed,
                "formal_run_key": r.formal_run_key,
                "run_manifest_sha256": r.run_manifest_sha256,
                "selected_epoch": int(r.selected_epoch),
                "selected_validation_macro_auc": float(r.selected_macro_auc),
                "epochs_completed": int(r.epochs_completed),
                "resumed_from": r.resumed_from,
                "status": r.status,
            }
            for r in sorted(collected, key=lambda x: (x.arm, x.seed))
        ],
        "budget": {
            "r2a_attempts_allowed": int(BUDGET_PER_DATASET["r2a_attempts"]),
            "attempts_used": int(len(collected)),
            "epochs_per_attempt": int(BUDGET_PER_DATASET["epochs_per_attempt"]),
            "epochs_used": int(sum(r.epochs_completed for r in collected)),
            "resume_per_run": int(BUDGET_PER_DATASET["resume_per_run"]),
            "resume_per_dataset_queue": int(
                BUDGET_PER_DATASET["resume_per_dataset_queue"]),
            "resumes_used": int(sum(1 for r in collected if r.resumed_from)),
        },
    }
    if guards is not None:
        manifest["guards"] = {
            "g_a_per_task_mean": {str(t): v for t, v in guards.g_a_per_task_mean.items()},
            "g_a_pass": bool(guards.g_a_pass),
            "g_b_W_mean": float(guards.g_b_W_mean),
            "g_b_pass": bool(guards.g_b_pass),
            "g_c_per_cell_mean": {str(k): v for k, v in guards.g_c_per_cell_mean.items()},
            "g_c_pass": bool(guards.g_c_pass),
            "g_d_per_cell_single_seed": {str(k): v for k, v in guards.g_d_per_cell_single_seed.items()},
            "g_d_pass": bool(guards.g_d_pass),
            "g_e_per_cell_mean": {str(k): v for k, v in guards.g_e_per_cell_mean.items()},
            "g_e_pass": bool(guards.g_e_pass),
            "g_f_macro_logloss_mean": float(guards.g_f_macro_logloss_mean),
            "g_f_pass": bool(guards.g_f_pass),
            "all_pass": bool(guards.all_pass()),
        }
    if write_to is not None:
        os.makedirs(os.path.dirname(os.path.abspath(write_to)), exist_ok=True)
        write_json(write_to, manifest)
        manifest["decision_manifest_path"] = os.path.abspath(write_to)
    return DecisionResult(
        dataset=dataset,
        decision=decision,
        signal_class=signal_class,
        D_s_by_seed=D_by_seed,
        T_s_by_seed=T_by_seed,
        mean_D=manifest["mean_D"],
        mean_T=manifest["mean_T"],
        guards=guards or GuardsResult({}, False, float("nan"), False, {},
                                       False, {}, False, {}, False, float("nan"), False),
        reasons=list(reasons),
        valid=bool(valid),
        validity_note=validity_note,
        decision_manifest_path=manifest.get("decision_manifest_path"),
        test_prediction_calls=int(_runner_mod.TEST_PREDICTION_CALLS),
    )


__all__ = [
    "PerRunMetrics",
    "GuardsResult",
    "DecisionResult",
    "collect_per_run_metrics",
    "evaluate_guards",
    "compute_r2a_decision",
]
