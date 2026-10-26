"""Read-only derivation/plots from completed C2.1 numeric oracle outputs."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from c2_numeric_oracle_20261016 import OUT, sha
from b1_reconstruction_20261016 import save_json


def main():
    source_sha = sha(__file__)
    receipt = json.loads((OUT / "run_receipt.json").read_text())
    assert receipt["status"] == "PASS_NUMERIC_ORACLE_ONLY" and receipt["cases_completed"] == 8
    assert receipt["training_runs"] == 0 and not receipt["c2_training_go"]
    for filename, digest in receipt["artifact_hashes"].items():
        assert sha(OUT / filename) == digest
    d = pd.read_csv(OUT / "estimator_moments.csv")
    g = pd.read_csv(OUT / "partition_moments.csv")
    targets = pd.read_csv(OUT / "target_checks.csv")
    assert len(d) == 96 and len(g) == 288 and len(targets) == 16
    analysis = OUT / "analysis"
    analysis.mkdir(exist_ok=True)
    covariance_ratios = g.pivot(index=["dataset", "seed", "objective", "batch_size", "partition"],
                               columns="sampler", values="full_gradient_covariance_trace").reset_index()
    covariance_ratios["Balanced_over_Natural"] = covariance_ratios.Balanced / covariance_ratios.Natural
    covariance_ratios["Stress_over_Natural"] = covariance_ratios.Stress / covariance_ratios.Natural
    summary = covariance_ratios.groupby(["dataset", "objective", "batch_size", "partition"])[
        ["Balanced_over_Natural", "Stress_over_Natural"]].agg(["mean", "min", "max"]).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    covariance_ratios.to_csv(analysis / "covariance_ratios_per_seed.csv", index=False)
    summary.to_csv(analysis / "covariance_ratio_seed_summary.csv", index=False)
    variance_scaling = []
    for key, group in d.groupby(["dataset", "seed", "objective", "sampler"]):
        v16 = group[group.batch_size == 16].exact_loss_variance.iloc[0]
        v128 = group[group.batch_size == 128].exact_loss_variance.iloc[0]
        residual = abs(v16 / v128 - 8)
        assert residual < 1e-10
        variance_scaling.append(dict(zip(("dataset", "seed", "objective", "sampler"), key),
                                     B16_over_B128=float(v16 / v128), residual=residual))
    pd.DataFrame(variance_scaling).to_csv(analysis / "iid_loss_variance_scaling.csv", index=False)
    # This is an upper bound on the unconditional batch marginal for ordinary
    # shuffled without-replacement Natural sampling, not an empirical result.
    bounds = []
    for dataset in ("kuairand", "movielens"):
        manifest = json.loads((OUT / dataset / "seed2022/manifest.json").read_text())
        p = np.asarray(manifest["full_train_domain_probabilities"])
        train_n = sum(manifest["data_manifest"]["train_domain_counts"].values())
        last_batch = train_n - ((train_n + 40959) // 40960 - 1) * 40960
        for stage, batch_size in (("full_main_batch", 40960), ("last_main_batch", last_batch)):
            for domain in range(3):
                log10_bound = batch_size * np.log1p(-p[domain]) / np.log(10)
                bounds.append(dict(dataset=dataset, domain=domain, stage=stage, batch_size=batch_size,
                    natural_p=float(p[domain]), log10_iid_absence_upper_bound=float(log10_bound),
                    scope="mathematical upper bound for uniformly shuffled batch marginal; no training"))
    bounds = pd.DataFrame(bounds)
    bounds.to_csv(analysis / "formal_natural_batch_absence_bounds.csv", index=False)
    adam_max, model_batch_max = [], []
    for dataset in ("kuairand", "movielens"):
        for seed in (2022, 2023, 2024, 2025):
            m = json.loads((OUT / dataset / f"seed{seed}/manifest.json").read_text())
            assert m["model_theta_unchanged"]
            adam_max.append(m["adam_max_absolute_residual"])
            model_batch_max.append(m["batch_prediction_max_residual"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dataset in zip(axes, ("kuairand", "movielens")):
        for offset, objective, color in ((-.12, "Equal", "#2878b5"), (.12, "Sample", "#d55e00")):
            sub = covariance_ratios[(covariance_ratios.dataset == dataset)
                & (covariance_ratios.objective == objective) & (covariance_ratios.batch_size == 16)]
            for j, partition in enumerate(("C", "P", "A_arch")):
                values = sub[sub.partition == partition].Balanced_over_Natural.to_numpy()
                ax.scatter(np.repeat(j + offset, len(values)), values, color=color, alpha=.65,
                           label=objective if j == 0 else None)
                ax.plot([j + offset - .06, j + offset + .06], [values.mean()] * 2, color=color)
        ax.axhline(1, color="black", linestyle="--", linewidth=.8)
        ax.set_yscale("log")
        ax.set_xticks(range(3), ("C", "P", "A_arch"))
        ax.set_title(dataset)
        ax.set_ylabel("Balanced / Natural gradient covariance trace")
        ax.legend()
    fig.suptitle("Fixed theta, finite training pools; four initializations; no training")
    fig.tight_layout()
    fig.savefig(analysis / "fixed_theta_covariance_ratios.png", dpi=180)
    fig.savefig(analysis / "fixed_theta_covariance_ratios.pdf")
    plt.close(fig)
    assert sha(__file__) == source_sha
    save_json(analysis / "analysis_receipt.json", {
        "status": "PASS_DESCRIPTIVE_NUMERICAL_ANALYSIS", "training_runs": 0,
        "cases": 8, "exact_gradient_max_relative_residual": float(d.exact_full_gradient_relative_residual.max()),
        "aggregate_autograd_max_relative_residual": float(targets.full_coordinate_autograd_relative_residual.max()),
        "mc_loss_max_standardized_residual": float(d.mc_loss_standardized_residual.max()),
        "mc_gradient_max_standardized_residual": float(d.mc_gradient_max_standardized_residual.max()),
        "adam_max_absolute_residual": max(adam_max), "batch_prediction_max_residual": max(model_batch_max),
        "c2_training_go": False, "source_sha256": source_sha, "run_receipt_sha256": sha(OUT / "run_receipt.json"),
        "verified_raw_artifact_count": len(receipt["artifact_hashes"]), "timestamp": time.time(),
        "artifact_hashes": {p.name: sha(p) for p in analysis.iterdir() if p.is_file() and p.name != "analysis_receipt.json"}})
    print(summary.to_string(index=False), flush=True)
    print(bounds.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
