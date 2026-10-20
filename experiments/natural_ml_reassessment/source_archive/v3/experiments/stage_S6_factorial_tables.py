"""Render descriptive S6 analysis tables/plots from main-agent CSV outputs."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stage_S6_analysis_tables"
REPORT = ROOT / "reports" / "stage_S6_complete_tables.md"
ARMS = ("F", "T", "HT", "DH")
CPS = ("selected", "common9", "common19", "common29", "common39", "common49", "final")
CELLS = ("d0t0", "d0t1", "d1t0", "d1t1", "d2t0", "d2t1", "macro", "non_target_mean", "worst_domain", "worst_cell")
NAMES = ("T-F", "HT-F", "DH-F", "HT-T", "DH-T", "history_main", "current_main", "interaction")


def read(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def table(lines, heads, rows):
    lines += ["| " + " | ".join(heads) + " |", "|" + "|".join(["---"] * len(heads)) + "|"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    lines.append("")


def main():
    abs_rows, con_rows, seed_rows = [read(n) for n in
        ("absolute_summary.csv", "factorial_summary.csv", "factorial_seed_contrasts.csv")]
    a = {(r["dataset"], r["objective"], r["checkpoint"], r["cell"], r["metric"], r["arm"]): r for r in abs_rows}
    c = {(r["dataset"], r["objective"], r["checkpoint"], r["cell"], r["metric"], r["contrast"]): r for r in con_rows}
    s = {(r["dataset"], r["objective"], r["checkpoint"], r["cell"], r["metric"], r["contrast"], int(r["seed"])): r for r in seed_rows}
    lines = ["# S6 complete descriptive tables — 2026-10-08", "",
        "数据：四臂 × 四 seeds × 两数据集。以下为 Sample，n=4，SD 为 seed 间样本标准差。",
        "AUC 差为正表示指标更高，LogLoss 差为负表示指标更低。均值后括号为相应方向 seed 数/4；交互项符号没有直接的优劣含义。",
        "selected 为各臂独立 validation-selected epoch；common 是固定 epoch；final 与 common49 同一结果。",
        "worst-cell 和 worst-domain 先在每个 seed/arm 内取最差再跨 seed 平均；最差身份可能改变。",
        "完整逐 seed 六个 cell 的原始绝对指标、聚合指标、所有条件效应和交互项见 `stage_S6_analysis_tables` CSV。", ""]
    for dataset, target in (("kuairand", 1), ("movielens", 2)):
        lines += [f"## {dataset}: absolute metrics", ""]
        for cp in CPS:
            for metric in ("auc", "logloss"):
                lines += [f"### {cp} / {metric}: mean ± SD", ""]
                table(lines, ("cell", *ARMS), [[cell, *[
                    f"{float(a[dataset,'sample',cp,cell,metric,arm]['mean']):.6f} ± {float(a[dataset,'sample',cp,cell,metric,arm]['sd']):.6f}"
                    for arm in ARMS]] for cell in CELLS])
        lines += [f"## {dataset}: all cell / global contrasts", ""]
        for cp in CPS:
            for metric in ("auc", "logloss"):
                lines += [f"### {cp} / Δ{metric}: mean (positive-AUC / negative-LogLoss seeds/4)", ""]
                table(lines, ("cell", *NAMES), [[cell, *[
                    f"{float(c[dataset,'sample',cp,cell,metric,name]['mean']):+.6f} ({c[dataset,'sample',cp,cell,metric,name]['positive' if metric == 'auc' else 'negative']}/4)"
                    for name in NAMES]] for cell in CELLS])
        lines += [f"## {dataset}: per-seed target contrasts", ""]
        for cp in ("selected", "final"):
            for metric in ("auc", "logloss"):
                for cell in (f"d{target}t0", f"d{target}t1"):
                    lines += [f"### {cp} / {cell} / Δ{metric}", ""]
                    table(lines, ("seed", *NAMES), [[seed, *[
                        f"{float(s[dataset,'sample',cp,cell,metric,name,seed]['value']):+.6f}"
                        for name in NAMES]] for seed in (2022, 2023, 2024, 2025)])
    lines += ["## Equal → Sample anchor comparison", "", "同 seed 的 contrast 差仅作描述性 objective 对照；未报告 p 值。", ""]
    o = read("objective_comparison_summary.csv")
    for dataset, target in (("kuairand", 1), ("movielens", 2)):
        for metric in ("auc", "logloss"):
            lines += [f"### {dataset} / {metric}", ""]
            rows = [r for r in o if r["dataset"] == dataset and r["metric"] == metric
                    and r["cell"] in (f"d{target}t0", f"d{target}t1")
                    and r["contrast"] in ("T-F", "HT-F", "DH-F", "interaction")]
            table(lines, ("checkpoint", "cell", "contrast", "Equal mean", "Sample mean", "Sample−Equal", "SD"),
                  [[r["checkpoint"], r["cell"], r["contrast"], f"{float(r['equal_mean']):+.6f}",
                    f"{float(r['sample_mean']):+.6f}", f"{float(r['objective_interaction_mean']):+.6f}",
                    f"{float(r['objective_interaction_sd']):.6f}"] for r in rows])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    colors = {"F": "#333333", "T": "#cc6677", "HT": "#4477aa", "DH": "#228833"}
    epochs = (9, 19, 29, 39, 49)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for row, (dataset, target) in enumerate((("kuairand", 1), ("movielens", 2))):
        for task in (0, 1):
            ax = axes[row, task]
            for arm in ARMS:
                vals = [float(a[dataset, "sample", f"common{e}", f"d{target}t{task}", "auc", arm]["mean"]) for e in epochs]
                ax.plot(epochs, vals, "o-", label=arm, color=colors[arm], lw=1.6, markersize=4)
            ax.set_title(f"{dataset} target d{target}t{task}")
            ax.set_ylabel("Test AUC (4-seed mean)")
            ax.grid(alpha=.2)
            if row == 1:
                ax.set_xlabel("Common epoch (zero-based)")
    axes[0, 0].legend(ncol=4, fontsize=9)
    fig.suptitle("S6 Sample: target trajectories under four optimizer policies")
    fig.tight_layout()
    fig.savefig(OUT / "target_auc_trajectories.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    contrast_colors = {"T-F": "#cc6677", "HT-F": "#4477aa", "DH-F": "#228833"}
    for row, (dataset, target) in enumerate((("kuairand", 1), ("movielens", 2))):
        for task in (0, 1):
            ax = axes[row, task]
            for name, color in contrast_colors.items():
                vals = [float(c[dataset, "sample", f"common{e}", f"d{target}t{task}", "auc", name]["mean"]) for e in epochs]
                sd = [float(c[dataset, "sample", f"common{e}", f"d{target}t{task}", "auc", name]["sd"]) for e in epochs]
                ax.plot(epochs, vals, "o-", color=color, label=name, lw=1.6, markersize=4)
                ax.fill_between(epochs, [x-y for x,y in zip(vals,sd)], [x+y for x,y in zip(vals,sd)], color=color, alpha=.10)
            ax.axhline(0, color="black", lw=.8)
            ax.axhline(.003, color="grey", ls="--", lw=.8)
            ax.axhline(-.003, color="grey", ls="--", lw=.8)
            ax.set_title(f"{dataset} target d{target}t{task}")
            ax.set_ylabel("Paired ΔAUC mean ± seed SD")
            ax.grid(alpha=.2)
            if row == 1:
                ax.set_xlabel("Common epoch (zero-based)")
    axes[0, 0].legend(ncol=3, fontsize=9)
    fig.suptitle("S6 Sample: paired target contrasts (shading is SD, not CI)")
    fig.tight_layout()
    fig.savefig(OUT / "target_auc_contrasts.png", dpi=180)
    plt.close(fig)
    print(REPORT)
    print(OUT / "target_auc_trajectories.png")
    print(OUT / "target_auc_contrasts.png")


if __name__ == "__main__":
    main()
