#!/usr/bin/env python
"""Render the completed descriptive analysis without selecting runs or models."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = ["O", "B", "D", "G", "Gtau0", "Dconst01"]


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"] +
        ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"


def main():
    out = HERE / "analysis"
    r = pd.read_csv(out / "all_seed_results.csv")
    a = pd.read_csv(out / "aggregate_test_metrics.csv")
    p = pd.read_csv(out / "paired_test_deltas.csv")
    assert len(r) == 48 and not r.duplicated(["dataset", "arm", "seed"]).any()
    def value(ds, arm, metric):
        x = a[(a.dataset == ds) & (a.arm == arm) & (a.metric == metric)].iloc[0]
        return f"{x['mean']:.6f} ± {x.sd:.6f}"
    text = ["# 完整统计表\n\n单位为 seed，n=4，± 为样本 SD；CI 为配对差的描述性 t 95% 区间。AUC 越高、LogLoss 越低越好。所有配置以 validation 选择 checkpoint。\n"]
    rows = []
    for ds in ["movielens", "kuairand"]:
        for arm in ARMS:
            q = r[(r.dataset == ds) & (r.arm == arm)].sort_values("seed")
            rows.append([ds, arm] + [value(ds, arm, m) for m in ["macro_auc", "macro_logloss", "pooled_task_mean_auc", "worst_cell_auc"]] +
                ["/".join(str(int(v)) for v in q.selected_epoch)])
    main_table = table(["数据集", "臂", "macro AUC", "macro LogLoss", "pooled AUC", "worst-cell AUC", "best epoch (2022–25)"], rows)
    (out / "main_table.md").write_text(main_table)
    text += ["## 整体终点\n\n", main_table]
    rows = []
    for ds in ["movielens", "kuairand"]:
        for contrast in p.contrast.drop_duplicates():
            q = p[(p.dataset == ds) & (p.contrast == contrast)].set_index("metric")
            auc, loss = q.loc["macro_auc"], q.loc["macro_logloss"]
            pos = sum(auc[f"seed{s}"] >= .001 for s in [2022,2023,2024,2025])
            neg = sum(auc[f"seed{s}"] <= -.001 for s in [2022,2023,2024,2025])
            rows.append([ds, contrast, f"{auc['mean']:+.6f} [{auc.ci95_low:+.6f}, {auc.ci95_high:+.6f}]",
                f"{int(auc.positive)}/{int(auc.negative)}", f"{pos}/{neg}",
                f"{loss['mean']:+.6f} [{loss.ci95_low:+.6f}, {loss.ci95_high:+.6f}]"])
    text += ["\n## 配对比较\n\n", table(["数据集", "差值", "Δ macro AUC [CI]", "AUC 正/负种子数", "AUC ≥.001/≤−.001", "Δ macro LogLoss [CI]"], rows)]
    for ds in ["movielens", "kuairand"]:
        for level, count in [("cell",6), ("domain",3), ("task",2)]:
            rows=[]
            for k in range(count):
                for metric in ["auc", "logloss"]:
                    name = f"d{k//2}t{k%2}_{metric}" if level == "cell" else f"{level}{k}_macro_{metric}"
                    rows.append([name] + [value(ds, arm, name) for arm in ARMS])
            text += [f"\n## {ds}：{level} 终点\n\n", table(["指标"]+ARMS,rows)]
    rows=[]
    for _, q in r.sort_values(["dataset","arm","seed"]).iterrows():
        rows.append([q.dataset,q.arm,int(q.seed),f"{q.macro_auc:.9f}",f"{q.macro_logloss:.9f}",
            f"{q.pooled_task_mean_auc:.9f}",f"{q.worst_cell_auc:.9f}",int(q.selected_epoch),int(q.stopped_epoch)])
    text += ["\n## 全部逐 seed 结果\n\n", table(["数据集","臂","seed","macro AUC","macro LogLoss","pooled AUC","worst AUC","best epoch","stop epoch"],rows)]
    (out / "RESULT_TABLES.md").write_text("".join(text))
    print("Rendered all 48 seeds and all cell/domain/task endpoints.")


if __name__ == "__main__":
    main()
