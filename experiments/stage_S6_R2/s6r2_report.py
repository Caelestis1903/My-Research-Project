"""s6r2_report.py

Aggregate all `s6r2_part*` CSVs into the report skeleton
`experiments/stage_S6_R2_selected_solution_analysis.md`.

Data sections are filled with tables from the generated CSVs (mean ± SD,
per-seed values, NA flags preserved). Verdict / GO-STOP / H1-H6 sections
are left as `[MAIN-AGENT: ...]` placeholders to be filled by the main agent
per spec §12. The report does NOT perform any analysis on raw data, does
NOT consult the saved model checkpoints, and does NOT make any new
assertions beyond what Parts A-D already enforced.

Run via:
    python -m experiments.stage_S6_R2.s6r2_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from stage_S6_R2 import s6r2_common as c  # noqa: E402


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def _fmt_mean_sd(mean: float, sd: float) -> str:
    if pd.isna(mean) or pd.isna(sd):
        return "NA"
    return f"{mean:+.4f} ± {sd:.4f}"


def _val_layer_table_from_partA(df_val: pd.DataFrame, df_test: pd.DataFrame,
                                layers: Tuple[str, ...] = ("val", "test")) -> str:
    """Per-(rule, arm) table of val and test M/T/N at selected epoch.
    For the test layer, F/DH means are taken over the seeds where test is
    available (the same seeds entering the paired Δ)."""
    lines = []
    for R in ("R0", "R1", "R2", "R3", "R4"):
        lines.append(f"### Rule {R}")
        lines.append("")
        lines.append("| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for layer_name, df_layer in (("val", df_val), ("test", df_test)):
            if layer_name not in layers:
                continue
            for metric in ("N", "T", "M"):
                deltas: List[float] = []
                f_vals: List[float] = []
                d_vals: List[float] = []
                for seed in c.SEEDS:
                    a = df_layer[(df_layer["rule"] == R) & (df_layer["arm"] == "F") & (df_layer["seed"] == seed)]
                    b = df_layer[(df_layer["rule"] == R) & (df_layer["arm"] == "DH") & (df_layer["seed"] == seed)]
                    if len(a) == 0 or len(b) == 0:
                        continue
                    if layer_name == "test" and (int(a["test_available"].iloc[0]) == 0 or int(b["test_available"].iloc[0]) == 0):
                        continue
                    f_vals.append(float(a[metric].iloc[0]))
                    d_vals.append(float(b[metric].iloc[0]))
                    deltas.append(float(b[metric].iloc[0]) - float(a[metric].iloc[0]))
                if not deltas:
                    lines.append(f"| {layer_name} | {metric} | NA | NA | NA (no test for seed) | - | - | - |")
                    continue
                arr = np.asarray(deltas, dtype=float)
                mean = float(arr.mean())
                sd = float(arr.std(ddof=1)) if arr.size >= 2 else float("nan")
                n_pos = int(np.sum(arr > 0))
                n_ge003 = int(np.sum(np.abs(arr) >= c.TAU_PRAC))
                n_ge005 = int(np.sum(np.abs(arr) >= c.TAU_STRONG))
                f_mean = float(np.mean(f_vals))
                d_mean = float(np.mean(d_vals))
                n_note = f"{len(deltas)}/4" if len(deltas) != 4 else "4/4"
                lines.append(
                    f"| {layer_name} | {metric} | {f_mean:.4f} | {d_mean:.4f} | "
                    f"{_fmt_mean_sd(mean, sd)} | {n_pos}/{len(deltas)} (n={n_note}) | {n_ge003} | {n_ge005} |"
                )
        lines.append("")
    return "\n".join(lines)


def _decomposition_summary(df_decomp: pd.DataFrame, dataset: str) -> str:
    """Build a summary for both F-anchor and DH-anchor val-layer decompositions.
    Per-seed × per-metric detail is preserved in the CSV; the report shows the
    4-seed mean and the SD of each term to keep the table compact."""
    val = df_decomp[df_decomp["layer"] == "val"]
    lines = []
    lines.append("**Val layer decomposition (4-seed mean ± SD of ΔSel / Policy / Timing, "
                 "identity residual max-abs across all cells):**")
    lines.append("")
    lines.append("| anchor | metric | ΔSel (mean ± SD) | Policy (mean ± SD) | Timing (mean ± SD) | max |residual| |")
    lines.append("|---|---|---|---|---|---|")
    for anchor in ("F", "DH"):
        for metric in ("M", "T", "N", "D0", "D2",
                        "d0t0", "d0t1", "d1t0", "d1t1", "d2t0", "d2t1"):
            sub = val[(val["anchor"] == anchor) & (val["metric"] == metric)]
            if len(sub) == 0:
                continue
            ds = np.asarray(sub["delta_sel"], dtype=float)
            ps = np.asarray(sub["policy"], dtype=float)
            ts = np.asarray(sub["timing"], dtype=float)
            rs = np.asarray(sub["residual"], dtype=float)
            max_res = float(np.max(np.abs(rs))) if rs.size else 0.0
            lines.append(
                f"| {anchor} | {metric} | {_fmt_mean_sd(float(ds.mean()), float(ds.std(ddof=1)))} | "
                f"{_fmt_mean_sd(float(ps.mean()), float(ps.std(ddof=1)))} | "
                f"{_fmt_mean_sd(float(ts.mean()), float(ts.std(ddof=1)))} | {max_res:.1e} |"
            )
    lines.append("")
    # Test layer availability summary
    test = df_decomp[df_decomp["layer"] == "test"]
    avail = test["availability"].value_counts().to_dict()
    lines.append("**Test-layer decomposition availability (per §3.3):**")
    lines.append("")
    lines.append("> Existing artifacts are insufficient for test-layer cross-epoch decomposition.")
    lines.append("")
    lines.append(f"- {avail}")
    lines.append("- Per §2.6, all 8 (arm,seed) combinations have canonical selected epoch off-grid; "
                 "test-layer cross-epoch decomposition requires both e_F and e_DH to be in "
                 "{9,19,29,39,49} ∪ {selected}, which is only possible when two arms "
                 "happen to share the same canonical epoch (degenerate, cross-terms ≡ 0).")
    lines.append("")
    return "\n".join(lines)


def _grid_trajectory(df_grid: pd.DataFrame) -> str:
    """Same-epoch Δ(e)=X_DH(e)−X_F(e) at grid epochs."""
    lines = []
    lines.append("### Same-epoch Δ(e) on the common checkpoint grid (test layer)")
    lines.append("")
    sub = df_grid[df_grid["metric"] == "N"]
    pivot = sub.pivot_table(index="seed", columns="epoch", values="delta_DH_minus_F")
    lines.append("**Non-target mean Δ(e) = (DH − F) at saved grid epochs:**")
    lines.append("")
    lines.append("| seed | e=9 | e=19 | e=29 | e=39 | e=49 |")
    lines.append("|---|---|---|---|---|---|")
    for seed in c.SEEDS:
        if seed not in pivot.index:
            continue
        row = pivot.loc[seed]
        cells = []
        for e in (9, 19, 29, 39, 49):
            v = float(row.get(e, float("nan")))
            cells.append(f"{v:+.4f}" if not pd.isna(v) else "NA")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _partC_summary(df_c: pd.DataFrame) -> str:
    lines = []
    lines.append("| arm | seed | canonical epoch | V(e*) | drop_1 | drop_2 | max_drop | classification | edge-clipped |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df_c.iterrows():
        edge = int(r["window_edge_clipped"])
        lines.append(
            f"| {r['arm']} | {int(r['seed'])} | {int(r['canonical_epoch'])} | "
            f"{float(r['val_M_at_canonical']):.4f} | {float(r['drop_1']):+.4f} | "
            f"{float(r['drop_2']):+.4f} | {float(r['max_drop']):+.4f} | "
            f"{r['classification']} | {'yes' if edge else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _partD_summary(df_d: pd.DataFrame, df_ret: Optional[pd.DataFrame]) -> str:
    lines = []
    has_val = {"val_M", "val_T", "val_N"}.issubset(df_d.columns)
    header = "| seed | hybrid | M | T | N | D0 | D2 |"
    sep = "|---|---|---|---|---|---|---|"
    if has_val:
        header = "| seed | hybrid | test M | test T | test N | test D0 | test D2 | val M | val T | val N |"
        sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for _, r in df_d.iterrows():
        base = (
            f"| {int(r['seed'])} | {r['hybrid']} | {float(r['M']):.4f} | "
            f"{float(r['T']):.4f} | {float(r['N']):.4f} | "
            f"{float(r['D0']):.4f} | {float(r['D2']):.4f} |"
        )
        if has_val:
            base = base[:-1] + (
                f" {float(r['val_M']):.4f} | {float(r['val_T']):.4f} | "
                f"{float(r['val_N']):.4f} |"
            )
        lines.append(base)
    lines.append("")
    if df_ret is not None and len(df_ret) > 0:
        lines.append("**Retention ρ on N (test, seed-level):**")
        lines.append("")
        lines.append("| seed | hybrid | Δ(hybrid − F_full) | denom (DH_full − F_full) | ρ_N |")
        lines.append("|---|---|---|---|---|")
        for _, r in df_ret.iterrows():
            try:
                dn = float(r["raw_delta_N"]) if pd.notna(r["raw_delta_N"]) else float("nan")
                denom = float(r["denom_N"]) if pd.notna(r["denom_N"]) else float("nan")
                rho = float(r["rho_N"]) if pd.notna(r["rho_N"]) else float("nan")
            except KeyError:
                continue
            lines.append(
                f"| {int(r['seed'])} | {r['hybrid']} | {dn:+.4f} | {denom:+.4f} | "
                f"{rho:+.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _gate_summary_table(df_gate: pd.DataFrame, dataset: str) -> str:
    lines = []
    lines.append(f"### {dataset} — gate summary (ΔDH−F)")
    lines.append("")
    lines.append("| rule | metric | layer | n_seeds | mean | SD | n_pos | n_ge_003 | n_ge_005 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    sub = df_gate.copy().sort_values(["rule", "metric", "layer"])
    for _, r in sub.iterrows():
        lines.append(
            f"| {r['rule']} | {r['metric']} | {r['layer']} | {int(r['n_seeds'])} | "
            f"{float(r['delta_mean']):+.4f} | {float(r['delta_sd']):.4f} | "
            f"{int(r['n_pos'])} | {int(r['n_ge_003'])} | {int(r['n_ge_005'])} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly.
# ---------------------------------------------------------------------------
def _gather_csvs(spec: c.DatasetSpec) -> Dict[str, pd.DataFrame]:
    base = c.dataset_out_dir(spec)
    files = {
        "epochs": f"s6r2_partA_epochs_{spec.name}.csv",
        "val": f"s6r2_partA_val_metrics_{spec.name}.csv",
        "test": f"s6r2_partA_test_metrics_{spec.name}.csv",
        "grid_epochs": f"s6r2_partA_grid_epochs_{spec.name}.csv",
        "grid_test": f"s6r2_partA_grid_test_metrics_{spec.name}.csv",
        "gate": f"s6r2_partA_gate_summary_{spec.name}.csv",
        "decomp": f"s6r2_partB_decomposition_{spec.name}.csv",
        "grid_traj": f"s6r2_partB_grid_trajectory_{spec.name}.csv",
        "window": f"s6r2_partC_window_{spec.name}.csv",
    }
    out: Dict[str, pd.DataFrame] = {}
    for k, fn in files.items():
        p = base / fn
        if not p.exists():
            print(f"[WARN] missing {p}")
            continue
        out[k] = pd.read_csv(p, encoding="utf-8-sig")
    if spec.name == "kuairand":
        swap_p = base / "s6r2_partD_swap_kuairand.csv"
        ret_p = base / "s6r2_partD_swap_retention_kuairand.csv"
        if swap_p.exists():
            out["swap"] = pd.read_csv(swap_p, encoding="utf-8-sig")
        if ret_p.exists():
            out["retention"] = pd.read_csv(ret_p, encoding="utf-8-sig")
    return out


def _integrity_section(integrity: Optional[Dict]) -> str:
    """Compact summary of integrity checks; full file referenced separately
    so the report stays readable."""
    if not integrity:
        return "_No integrity report found._"
    parts = integrity.get("partA_partB_partC", {})
    lines: List[str] = []
    lines.append("| dataset | arm/seed | canonical epoch | R0 matches flag? | npz vs csv max diff | final vs ce49 max |")
    lines.append("|---|---|---|---|---|---|")
    for ds_name, runs in parts.items():
        for run_tag, info in runs.items():
            npz = info.get("test_npz_vs_csv_residuals_max") or {}
            npz_max = max(npz.values()) if npz else 0.0
            ce49 = info.get("final_vs_ce49") or []
            ce49_max = (max(abs(item["diff"]) for item in ce49)
                        if ce49 else 0.0)
            lines.append(
                f"| {ds_name} | {run_tag} | {int(info['canonical_selected_epoch'])} | "
                f"{'yes' if int(info['canonical_selected_epoch']) == int(info['R0_recomputed_epoch']) else 'NO'} | "
                f"{npz_max:.2e} | {ce49_max:.2e} |"
            )
    # Point to the raw JSON for the full entries.
    lines.append("")
    lines.append(f"Full machine-readable entries: `{c.OUT_ROOT.as_posix()}/s6r2_integrity_report.json`.")
    # PartD specifics
    if "partD" in integrity:
        lines.append("")
        lines.append("**Part D integrity (KuaiRand only):**")
        for ds_name, runs in integrity["partD"].items():
            lines.append(f"- *{ds_name}*: {len(runs)} seed runs; partition / "
                          "state-dict parity / anchor reproduction results in JSON.")
    return "\n".join(lines)


def build_report(integrity: Optional[Dict]) -> None:
    parts: List[str] = []
    parts.append("# Stage S6-R2 — KuaiRand DH Selected-Solution Analysis\n")
    parts.append("**Audience:** main agent. **Scope:** retrospective robustness + parameter-swap "
                 "evaluation of KuaiRand DH. MovieLens included only as a negative control "
                 "(per spec §9).\n")
    parts.append("**Auto-generated data sections.** Verdict / GO-STOP / H1–H6 / counterexamples "
                 "are placeholders to be filled by the main agent; see "
                 "`[MAIN-AGENT: …]` tokens.\n")
    parts.append("---\n")
    parts.append("## 1. Data availability & integrity")
    parts.append("")
    parts.append("All inputs are read-only (`experiments/stage_S6_raw_results`, "
                 "`experiments/stage_S6ML_raw_results`). No training was performed.\n")
    parts.append("### Integrity report (machine-readable)")
    parts.append("")
    parts.append(_integrity_section(integrity))
    parts.append("\n---\n")
    parts.append("## 2. Selection-rule robustness (Part A)")
    parts.append("")
    parts.append("Five pre-registered rules R0–R4 (R0 = canonical six-cell macro, R1 = target, "
                 "R2 = non-target, R3 = worst-domain, R4 = val sample-weighted). Tie-break: "
                 "earliest epoch. Test data is **never** used for selection. Test metrics for "
                 "R0 are exact (selected checkpoint saved); other rules often need test data at "
                 "non-saved epochs and are reported as **NA** with `test_available=0`. "
                 "The A-grid variant restricts selection to "
                 "`{9,19,29,39,49} ∪ {e_R0}` so every rule has a test value; it is "
                 "explicitly flagged as a sensitivity variant.\n")
    for spec_name in ("kuairand", "movielens"):
        spec = c.SPEC[spec_name]
        out_dir = c.dataset_out_dir(spec)
        if not (out_dir / f"s6r2_partA_epochs_{spec.name}.csv").exists():
            parts.append(f"### Dataset: {spec_name}\n_No Part A outputs available._\n")
            continue
        files = _gather_csvs(spec)
        parts.append(f"### Dataset: {spec_name}\n")
        parts.append(_val_layer_table_from_partA(files["val"], files["test"]))
        parts.append("**A-grid sensitivity variant (selection restricted to "
                     "{9,19,29,39,49} ∪ {e_R0}; NOT the primary rule — spec §4.3):** "
                     "test layer shown only; the corresponding val-layer objective values "
                     "are in `s6r2_partA_grid_epochs_*.csv`.")
        parts.append("")
        parts.append(_val_layer_table_from_partA(files["val"], files["grid_test"],
                                                 layers=("test",)))
        parts.append(_gate_summary_table(files["gate"], spec_name))
        parts.append("\n")

    parts.append("---\n")
    parts.append("## 3. Algebraic decomposition (Part B)\n")
    parts.append("### Val-layer decomposition\n")
    parts.append("Identities |ΔSel − (Policy + Timing)| < 1e-9 are asserted programmatically. "
                 "Anchors: F-anchor decomposes the gap into `PolicyAtF` (same F checkpoint) "
                 "+ `TimingAlongDH` (DH from e_F to e_DH); DH-anchor is the symmetric form.\n")
    for spec_name in ("kuairand", "movielens"):
        spec = c.SPEC[spec_name]
        if "decomp" not in _gather_csvs(spec):
            parts.append(f"### Dataset: {spec_name}\n_No Part B outputs available._\n")
            continue
        files = _gather_csvs(spec)
        parts.append(f"### Dataset: {spec_name}\n")
        parts.append(_decomposition_summary(files["decomp"], spec_name))
        parts.append(_grid_trajectory(files["grid_traj"]))
        parts.append("")

    parts.append("---\n")
    parts.append("## 4. Selection-window analysis (Part C)\n")
    for spec_name in ("kuairand", "movielens"):
        spec = c.SPEC[spec_name]
        out_dir = c.dataset_out_dir(spec)
        if not (out_dir / f"s6r2_partC_window_{spec.name}.csv").exists():
            parts.append(f"### Dataset: {spec_name}\n_No Part C outputs available._\n")
            continue
        files = _gather_csvs(spec)
        parts.append(f"### Dataset: {spec_name}\n")
        parts.append(_partC_summary(files["window"]))
        parts.append("")

    parts.append("---\n")
    parts.append("## 5. Parameter-swap analysis (Part D, KuaiRand only)\n")
    spec = c.SPEC["kuairand"]
    files = _gather_csvs(spec)
    if "swap" in files:
        parts.append("**Partition declaration (spec §2.4/§3.5):** the swap uses "
                     "`parameter_manifest.json` semantics (shared / architecture / private). "
                     "Note the documented divergence: `gradient_partition.json` places "
                     "`domain_expert.*` in its C (shared-side) group, whereas the manifest "
                     "counts `domain_expert.*` as private. Main results use the manifest; "
                     "the gradient-partition variant is at most an optional sensitivity note.\n")
        parts.append(_partD_summary(files["swap"], files.get("retention")))
        parts.append("**Interpretation boundary (spec §8):** parameter swapping is a *localisation* "
                     "diagnostic, not a causal decomposition. Independently trained parameter "
                     "partitions may be co-adapted; hybrids that fail badly are evidence of "
                     "cross-partition co-adaptation rather than absence of an effect.\n")
    else:
        parts.append("_No Part D outputs available._\n")

    parts.append("---\n")
    parts.append("## 6. MovieLens negative control\n")
    parts.append("Per spec §9, only Parts A and B were replicated on MovieLens. The "
                 "canonical selected epochs are very early (7–13). Compare the tables above "
                 "(`section 2`, `section 3`).\n")
    parts.append("---\n")
    parts.append("## 7. Counterexamples\n")
    parts.append("[MAIN-AGENT: enumerate any seed × rule × layer combinations where Δ(N/T/M) "
                 "is sign-inconsistent with the canonical R0 expectation, and any "
                 "co-adaptation artefacts from Part D.]\n")
    parts.append("---\n")
    parts.append("## 8. Competing explanations (H1–H6 evidence-strength)\n")
    for h in ("H1 — robust better solution",
              "H2 — selection-objective specificity",
              "H3 — checkpoint timing",
              "H4 — shared-state localization",
              "H5 — private/co-adaptation",
              "H6 — narrow validation peak"):
        parts.append(f"- **{h}**: [MAIN-AGENT: STRONG / MEDIUM / WEAK with cite to "
                     f"specific table rows.]\n")
    parts.append("---\n")
    parts.append("## 9. GO / STOP decision (spec §11)\n")
    parts.append("Conditions (1–6) of §11 are reproduced verbatim below; each must be "
                 "checked by the main agent against the tables in §§2–5:\n")
    parts.append("1. canonical KuaiRand selected ΔNonTarget remains practical (≥ +0.003) and "
                 "≥3/4 seeds positive;\n")
    parts.append("2. target change remains above −0.003;\n")
    parts.append("3. macro gain remains practically positive or the trade-off is otherwise "
                 "clearly favorable;\n")
    parts.append("4. at least two reasonable validation-selection rules show a qualitatively "
                 "similar result, excluding reliance on non-target-only selection;\n")
    parts.append("5. the selected gain is not explainable purely as a single-epoch spike;\n")
    parts.append("6. decomposition provides at least one reproducible clue about same-epoch "
                 "policy effect, timing, or parameter partition localization.\n")
    parts.append("\n[MAIN-AGENT: apply the gate and record the verdict (GO / minimal-mechanism "
                 "experiment / STOP).]\n")
    parts.append("---\n")
    parts.append("## 10. What remains unknown\n")
    parts.append("[MAIN-AGENT: list the gaps / pending evaluations needed for a definitive "
                 "verdict.]\n")
    parts.append("---\n")
    parts.append("## 11. Auto-generated outputs (file map)\n")
    for spec_name in ("kuairand", "movielens"):
        spec = c.SPEC[spec_name]
        parts.append(f"- `{c.dataset_out_dir(spec).as_posix()}/s6r2_partA_*.csv`")
        parts.append(f"- `{c.dataset_out_dir(spec).as_posix()}/s6r2_partB_*.csv`")
        parts.append(f"- `{c.dataset_out_dir(spec).as_posix()}/s6r2_partC_window_{spec.name}.csv`")
        if spec.name == "kuairand":
            parts.append(f"- `{c.dataset_out_dir(spec).as_posix()}/s6r2_partD_swap_kuairand.csv`")
            parts.append(f"- `{c.dataset_out_dir(spec).as_posix()}/s6r2_partD_swap_retention_kuairand.csv`")
    parts.append(f"- `{c.OUT_ROOT.as_posix()}/s6r2_integrity_report.json`")
    parts.append("")

    out_path = c.ROOT / "experiments" / "stage_S6_R2_selected_solution_analysis.md"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[Report] wrote {out_path}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.parse_known_args(argv)
    int_path = c.OUT_ROOT / "s6r2_integrity_report.json"
    if int_path.exists():
        integrity = json.loads(int_path.read_text(encoding="utf-8"))
    else:
        integrity = None
    build_report(integrity)


if __name__ == "__main__":
    main()
