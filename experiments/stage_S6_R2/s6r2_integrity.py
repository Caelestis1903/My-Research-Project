"""s6r2_integrity.py

Cross-cuts integrity assertions for the S6-R2 retrospective analysis.
Writes/merges the entries listed in design §7 into
`experiments/stage_S6_R2_raw_results/s6r2_integrity_report.json`.

Categories collected:
    - macro/WD/WC per-epoch current computation vs precomputed column (TOL_EXACT)
    - npz recomputation vs metrics_test.csv (TOL_EXACT)
    - R0 reset vs selected_checkpoint flag epoch equality (per run)
    - validation_sample_count cross-epoch constancy (per run)
    - final vs common_epoch_49 per-cell AUC equality (TOL_EXACT via csv)
    - Part D: partition completeness + state_dict key/shape parity + anchor
      reproduction residuals (written by s6r2_partD_swap.main)

Run via:
    python -m experiments.stage_S6_R2.s6r2_integrity [--dataset kuairand|movielens]

The full data set {KR, ML} x {F, DH} x {4 seeds} is iterated once at runtime.
No training happens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from stage_S6_R2 import s6r2_common as c  # noqa: E402


def _check_final_vs_ce49(spec: c.DatasetSpec, arm: str, seed: int) -> List[Dict[str, float]]:
    df = pd.read_csv(
        c.run_joint_dir(spec, arm, seed) / "metrics_test.csv", encoding="utf-8-sig",
    )
    finals = df[df["checkpoint"] == "final"].sort_values(["domain", "task"])
    ce49 = df[df["checkpoint"] == "common_epoch_49"].sort_values(["domain", "task"])
    if len(finals) != 6 or len(ce49) != 6:
        raise AssertionError(
            f"{spec.name}/{arm}/seed{seed}: final/ce49 row counts {len(finals)}/{len(ce49)}"
        )
    out: List[Dict[str, float]] = []
    for r_f, r_49 in zip(finals.itertuples(index=False), ce49.itertuples(index=False)):
        if int(r_f.domain) != int(r_49.domain) or int(r_f.task) != int(r_49.task):
            raise AssertionError("final/ce49 row alignment differs")
        d = float(r_f.auc) - float(r_49.auc)
        if abs(d) >= 1e-6:
            raise AssertionError(
                f"{spec.name}/{arm}/seed{seed}/d{int(r_f.domain)}t{int(r_f.task)}: "
                f"final vs common_epoch_49 |ΔAUC|={abs(d):.2e} >=1e-6"
            )
        out.append({
            "domain": int(r_f.domain), "task": int(r_f.task),
            "final_auc": float(r_f.auc), "ce49_auc": float(r_49.auc),
            "diff": d,
        })
    return out


def run_for_dataset(spec: c.DatasetSpec) -> Dict[str, object]:
    print(f"[Integrity] dataset={spec.name}")
    out: Dict[str, object] = {}
    for arm in c.ARMS:
        for seed in c.SEEDS:
            tag = f"{arm}_seed{int(seed)}"
            sub: Dict[str, object] = {}
            # 1. selected_checkpoint flag equality with R0 selection.
            c.assert_epoch_aggregations_match_precomputed(spec, arm, seed)
            c.assert_val_counts_constant(spec, arm, seed)
            e_canon = c.canonical_selected_epoch(spec, arm, seed)
            sub["canonical_selected_epoch"] = int(e_canon)
            # 2. R0 selection matches selected_checkpoint flag.
            e_R0 = c.select_rule_R0(spec, arm, seed)[0]
            sub["R0_recomputed_epoch"] = int(e_R0)
            assert e_R0 == e_canon, (
                f"{spec.name}/{arm}/seed{seed}: R0 {e_R0} != flag {e_canon}"
            )
            # 3. test npz vs metrics_test.csv for the 7 checkpoints.
            npz_residuals: Dict[str, Dict[str, float]] = {}
            for ck in c.SAVED_TEST_NAMES:
                metrics = c.load_test_checkpoint(spec, arm, seed, ck)
                df = pd.read_csv(
                    c.run_joint_dir(spec, arm, seed) / "metrics_test.csv",
                    encoding="utf-8-sig")
                ref = df[df["checkpoint"] == ck]
                res: Dict[str, float] = {}
                for _, row in ref.iterrows():
                    label = c.cell_label(int(row["domain"]), int(row["task"]))
                    diff = float(metrics[label]["auc"]) - float(row["auc"])
                    if abs(diff) > c.TOL_EXACT:
                        raise AssertionError(
                            f"{spec.name}/{arm}/seed{seed}/{ck}/{label}: "
                            f"npz vs csv diff {diff:+.2e} > {c.TOL_EXACT}"
                        )
                    res[label] = diff
                npz_residuals[ck] = res
            sub["test_npz_vs_csv_residuals_max"] = {
                k: max(map(abs, v.values())) if v else 0.0
                for k, v in npz_residuals.items()
            }
            # 4. final == common_epoch_49 (within TOL_EXACT, assertion).
            sub["final_vs_ce49"] = _check_final_vs_ce49(spec, arm, seed)
            out[tag] = sub
    return out


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(c.SPEC.keys()))
    args = parser.parse_args(argv)

    if args.dataset is None:
        to_run = list(c.SPEC.keys())
    else:
        to_run = [args.dataset]

    int_path = c.OUT_ROOT / "s6r2_integrity_report.json"
    if int_path.exists():
        existing = json.loads(int_path.read_text(encoding="utf-8"))
    else:
        existing = {}
    existing.setdefault("partA_partB_partC", {})

    for ds in to_run:
        spec = c.SPEC[ds]
        c.ensure_out_dir(spec)
        existing["partA_partB_partC"][ds] = run_for_dataset(spec)
        int_path.parent.mkdir(parents=True, exist_ok=True)
        int_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"[Integrity] wrote {int_path}")


if __name__ == "__main__":
    main()
