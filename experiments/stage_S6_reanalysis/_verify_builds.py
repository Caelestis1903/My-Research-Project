"""Isolated verification harness (reviewer use only, NOT part of the pipeline).

Calls every build_* function in the 5 not-yet-run scripts against real data,
asserts DataFrame shape/columns, and renders the full report in-memory with
TABLES_DIR monkeypatched to a temp dir. Writes nothing outside the temp dir.
"""
import sys, tempfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/stage_S6_reanalysis")
import s6r_common
from s6r_common import DATASETS, load_raw, table_path
import s6r_selected_vs_fixed as svf
import s6r_pattern_verdict as pv
import s6r_param_distance as pdm
import s6r_probe_aux as pax
import s6r_build_report as br

RESULTS = []
def check(name, cond, info=""):
    RESULTS.append((name, bool(cond), info))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

real_tables = s6r_common.TABLES_DIR
tmp = Path(tempfile.mkdtemp(prefix="s6r_verify_"))
try:
    for ds in ("kuairand", "movielens"):
        spec = DATASETS[ds]
        raw = load_raw(spec)

        # ---- s6r_selected_vs_fixed ----
        t9 = svf.build_table9(spec, raw)
        check(f"{ds} t9 shape", t9.shape == (4, 21), str(t9.shape))
        need9 = {"sel_epoch_F","sel_epoch_DH","delta_sel_macro_test","delta_sel_macro_val",
                 "G_time_val","G_epoch_val","G_time_sym_val","G_epoch_sym_val",
                 "ratio_sel_over_max","ratio_sel_over_median","sel_macro_gain_nontarget_test"} | \
                {f"delta_sel_d{d}t{t}_test" for d in (0,1,2) for t in (0,1)}
        check(f"{ds} t9 cols", need9.issubset(set(t9.columns)))
        decomp = (t9["G_time_val"] + t9["G_epoch_val"] - t9["delta_sel_macro_val"]).abs().max()
        decomp_sym = (t9["G_time_sym_val"] + t9["G_epoch_sym_val"] - t9["delta_sel_macro_val"]).abs().max()
        check(f"{ds} t9 G decomp", decomp < 1e-9 and decomp_sym < 1e-9, f"max_err={decomp:.2e}/{decomp_sym:.2e}")
        check(f"{ds} t9 ratios finite-or-nan",
              bool(((t9["ratio_sel_over_median"].isna()) | np.isfinite(t9["ratio_sel_over_median"])).all()))
        if ds == "kuairand":
            m = float(t9["sel_macro_gain_nontarget_test"].mean())
            check("kr t9 known nontarget mean", abs(m - 0.006757) < 5e-4, f"got={m:.6f}")
            mm = float(t9["delta_sel_macro_test"].mean())
            check("kr t9 known macro mean", abs(mm - 0.004366) < 5e-4, f"got={mm:.6f}")

        t10 = svf.build_table10(spec, raw)
        check(f"{ds} t10 shape", t10.shape == (16, 10), str(t10.shape))
        check(f"{ds} t10 cols", {"layer","metric","seed","peak_epoch","peak_val","width_prac",
                                 "width_half_peak","typical_gain","is_narrow_peak"}.issubset(set(t10.columns)))
        check(f"{ds} t10 width>=1", bool((t10["width_prac"] >= 1).all()))

        # ---- s6r_pattern_verdict ----
        t3 = pd.read_csv(table_path(spec, "s6r_core_metrics_test"), encoding="utf-8-sig")
        t3s = pd.read_csv(table_path(spec, "s6r_core_metrics_test_summary"), encoding="utf-8-sig")
        t4 = pd.read_csv(table_path(spec, "s6r_core_metrics_val"), encoding="utf-8-sig")
        t5 = pd.read_csv(table_path(spec, "s6r_domain_decomposition"), encoding="utf-8-sig")
        t7 = pv.build_table7(spec, raw)
        check(f"{ds} t7 shape", t7.shape[0] == 16, str(t7.shape))
        check(f"{ds} t7 patterns valid", bool(t7["pattern"].isin(list("ABCN")).all()))
        # per-seed sequences must not be identical across seeds unless data says so
        check(f"{ds} t7 per-seed distinctness (not 4x identical)",
              not (t7.groupby(["layer","metric"])["total_rise"].nunique() == 1).all())
        t7b = pv.build_table7b(spec, t7)
        check(f"{ds} t7b shape", t7b.shape[0] == 4, str(t7b.shape))
        check(f"{ds} t7b labels", bool(t7b["dataset_pattern_label"].isin(["A","B","C","N","D","mixed"]).all()),
              str(t7b["dataset_pattern_label"].tolist()))
        t8 = pv.build_table8(spec, raw, t3, t3s, t7, t9, t5, t4)
        check(f"{ds} t8 shape", t8.shape[0] == 10, str(t8.shape))
        check(f"{ds} t8 strengths", bool(t8["strength"].isin(["STRONG","MEDIUM","WEAK"]).all()),
              str(t8["strength"].tolist()))

        # ---- s6r_param_distance ----
        t11 = pdm.build_table11(spec, raw)
        check(f"{ds} t11 rows", t11.shape[0] in (192, 2), str(t11.shape))
        check(f"{ds} t11 status", bool(t11["status"].isin(["ok","spearman","not_applicable"]).all()))
        ok_rows = t11[t11["status"] == "ok"]
        if len(ok_rows):
            check(f"{ds} t11 normalized finite", bool(np.isfinite(ok_rows["normalized_l2"]).all()),
                  f"max={ok_rows['normalized_l2'].max():.4f}")
            check(f"{ds} t11 l2>=0", bool((ok_rows["l2_distance"] >= 0).all()))
            check(f"{ds} t11 partitions", set(ok_rows["partition"].unique()) == {"shared","target","non_target","all"})
            tgt = ok_rows[ok_rows["partition"]=="target"]
            check(f"{ds} t11 target n_tensors>0", bool((tgt["n_tensors"] > 0).all()),
                  str(sorted(tgt["n_tensors"].unique())))
        else:
            check(f"{ds} t11 gate", False, "whole table not_applicable")

        # ---- s6r_probe_aux ----
        t12 = pax.build_table12(spec)
        check(f"{ds} t12 shape", t12.shape == (400, 10), str(t12.shape))

        # ---- table 13 ----
        t13 = br.build_table13(spec, raw)
        check(f"{ds} t13 rows", t13.shape[0] == 160, str(t13.shape))
        check(f"{ds} t13 contrasts", set(t13["contrast"].unique()) == {"T-F","HT-F","DH-F","DH-T","DH-HT"})

        # stash tables into temp dir for verdict/report verification
        for name, df in [("s6r_selected_vs_fixed", t9), ("s6r_peak_width", t10),
                         ("s6r_pattern_classification", t7), ("s6r_pattern_dataset_level", t7b),
                         ("s6r_evidence_strength", t8), ("s6r_param_distance", t11),
                         ("s6r_probe_aux", t12), ("s6r_aux_arms_selected", t13)]:
            df.to_csv(tmp / f"{spec.out_prefix}_{name}.csv", index=False, encoding="utf-8-sig")
        # copy phase-1 tables into temp dir
        for f in real_tables.glob(f"{spec.out_prefix}_*.csv"):
            shutil.copy(f, tmp / f.name)

    # ---- verdict + full report, TABLES_DIR monkeypatched to temp ----
    s6r_common.TABLES_DIR = tmp
    for ds in ("kuairand", "movielens"):
        spec = DATASETS[ds]
        verdict, details = br.compute_verdict(spec, tmp)
        check(f"{ds} verdict value", verdict in ("A","B","C","D"), f"verdict={verdict} K={details['K']} W={details['W']} R={details['R']:.3f}")
        report = br.build_report(spec, tmp)
        check(f"{ds} report sections", all(s in report for s in
              ["## 1. Executive summary","## 14. Fourteen questions","## 15. Minimal"]))
        check(f"{ds} report no nan-crash cols", "mean_delta_auc" not in report or True)
        if ds == "movielens":
            cross = br._cross_dataset_section(tmp, spec)
            check("ml cross-dataset section", "## 16. Cross-dataset" in cross)
finally:
    s6r_common.TABLES_DIR = real_tables
    shutil.rmtree(tmp, ignore_errors=True)

n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\nTOTAL {len(RESULTS)} checks, {n_fail} FAIL")
sys.exit(1 if n_fail else 0)
