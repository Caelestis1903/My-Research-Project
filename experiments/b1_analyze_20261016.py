"""Prediction-level numerical analysis; scientific verdict is written by main agent.

Default input requires all eight actual complete runs. Explicit engineering
mode has a different output directory and never increments formal run counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

from b1_reconstruction_20261016 import OUT, ROOT, STUDY, save_json, sha

EPS = 1e-12
SEEDS = (2022, 2023, 2024, 2025)
REPLICATES = 1000


def bce(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def predictions(path, raw_labels, raw_domains):
    with np.load(path, allow_pickle=False) as z:
        ids = z["row_ids"].astype(np.int64)
        domains, labels, scores = z["local_domains"], z["labels"], z["predictions"]
    assert len(np.unique(ids)) == len(ids)
    assert (np.diff(ids) > 0).all(), "row-order window analysis requires original sorted row IDs"
    assert np.array_equal(labels, raw_labels[ids])
    assert np.array_equal(domains, raw_domains[ids])
    assert labels.shape == scores.shape == (len(ids), 2)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    return ids, domains, labels.astype(np.float64), scores.astype(np.float64)


def aggregates(records):
    result = {}
    for metric in ("auc", "logloss"):
        a = np.asarray([[records[d, t][metric] for t in range(2)] for d in range(3)])
        counts = np.asarray([[records[d, t]["n"] for t in range(2)] for d in range(3)])
        q = {"macro": a.mean(), "target": a[1].mean(), "non_target": a[[0, 2]].mean(),
             "eval_row_weighted": np.sum(a * counts) / counts.sum(),
             "worst_cell": a.min() if metric == "auc" else a.max(),
             "worst_domain": a.mean(1).min() if metric == "auc" else a.mean(1).max()}
        q.update({f"domain{d}": a[d].mean() for d in range(3)})
        q.update({f"task{t}": a[:, t].mean() for t in range(2)})
        q.update({f"d{d}t{t}": a[d, t] for d in range(3) for t in range(2)})
        result[metric] = q
    return result


def audit_state(payload, partition, epoch, batches):
    assert payload["epoch"] == epoch and payload["study_id"] == STUDY
    assert payload["optimizer_steps"] == batches * (epoch + 1)
    assert payload["architecture_optimizer_steps"] == epoch + 1
    assert {"torch_cpu", "torch_cuda", "python", "numpy"}.issubset(payload["rng"])
    assert payload["arch_iterator_creation_rng"] is not None
    assert 1 <= payload["arch_iterator_batches_consumed"] <= batches
    state_count = {}
    for key, names_key, groups in (
        ("optimizer", "optimizer_parameter_names", ("C", "P")),
        ("arch_optimizer", "arch_optimizer_parameter_names", ("A_arch",))):
        state = payload[key]
        parameter_names = payload[names_key]
        assert len(state["param_groups"]) == len(parameter_names)
        names = [n for group in parameter_names for n in group]
        assert len(names) == len(set(names))
        assert set(names) == {n for group in groups for n in partition["groups"][group]}
        mapping = {pid: name for group, group_names in zip(state["param_groups"], parameter_names)
                   for pid, name in zip(group["params"], group_names)}
        assert len(mapping) == len(names)
        assert set(state["state"]).issubset(mapping)
        for pid, s in state["state"].items():
            assert {"step", "exp_avg", "exp_avg_sq"}.issubset(s)
            assert s["exp_avg"].shape == s["exp_avg_sq"].shape == payload["model"][mapping[pid]].shape
            assert torch.isfinite(s["exp_avg"]).all() and torch.isfinite(s["exp_avg_sq"]).all()
        state_count[key] = len(state["state"])
    return state_count


def weighted_auc_setup(y, p):
    order = np.argsort(p, kind="stable")
    starts = np.r_[0, np.flatnonzero(p[order][1:] != p[order][:-1]) + 1]
    return order, starts, y[order]


def weighted_auc(weights, setup, npos, nneg):
    order, starts, y = setup
    w = weights[order]
    pos = np.add.reduceat(w * y, starts)
    neg = np.add.reduceat(w * (1 - y), starts)
    return np.sum(pos * (np.cumsum(neg) - .5 * neg)) / (npos * nneg)


def bootstrap(selected, seeds):
    rng = np.random.default_rng(20261016)
    result, draws = [], {}
    for split in ("validation", "test"):
        first = selected[seeds[0], "F", split]
        ids, domains, labels, _ = first
        for seed in seeds:
            for arm in ("F", "DH"):
                current = selected[seed, arm, split]
                assert all(np.array_equal(a, b) for a, b in zip(first[:3], current[:3]))
        for d in range(3):
            for t in range(2):
                mask = domains == d
                y = labels[mask, t]
                ip, ine = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
                npos, nneg = len(ip), len(ine)
                setups, ll_diffs = [], []
                point = []
                for seed in seeds:
                    f = selected[seed, "F", split][3][mask, t]
                    h = selected[seed, "DH", split][3][mask, t]
                    setups.append((weighted_auc_setup(y, f), weighted_auc_setup(y, h)))
                    ll_diffs.append(bce(y, h) - bce(y, f))
                    point.append((roc_auc_score(y, h) - roc_auc_score(y, f), np.mean(ll_diffs[-1])))
                auc_draws = np.empty((REPLICATES, len(seeds)))
                ll_draws = np.empty_like(auc_draws)
                for repeat in range(REPLICATES):
                    rows = np.r_[ip[rng.integers(npos, size=npos)], ine[rng.integers(nneg, size=nneg)]]
                    weights = np.bincount(rows, minlength=len(y))
                    for j, ((f_setup, h_setup), ll_diff) in enumerate(zip(setups, ll_diffs)):
                        auc_draws[repeat, j] = weighted_auc(weights, h_setup, npos, nneg) - weighted_auc(weights, f_setup, npos, nneg)
                        ll_draws[repeat, j] = np.dot(weights, ll_diff) / len(y)
                for metric, samples, points in (("auc", auc_draws, np.asarray(point)[:, 0]),
                                                ("logloss", ll_draws, np.asarray(point)[:, 1])):
                    draws[f"{split}_d{d}t{t}_{metric}"] = samples
                    for j, seed in enumerate(list(seeds) + ["fixed_model_mean"]):
                        values = samples[:, j] if j < len(seeds) else samples.mean(1)
                        estimate = points[j] if j < len(seeds) else points.mean()
                        lo, hi = np.quantile(values, [.025, .975])
                        result.append(dict(split=split, domain=d, task=t, metric=metric,
                            seed=seed, model_count=1 if j < len(seeds) else len(seeds), n=len(y), n_pos=npos,
                            delta=estimate, ci95_low=lo, ci95_high=hi, replicates=REPLICATES,
                            uncertainty="row sampling conditional on fixed fitted models; marginal unadjusted"))
                print(f"bootstrap {split} d{d}t{t}", flush=True)
    return pd.DataFrame(result), draws


def main():
    checked_source_sha = sha(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineering-fixture", type=Path)
    a = parser.parse_args()
    engineering = a.engineering_fixture is not None
    runs = a.engineering_fixture.resolve() if engineering else OUT / "runs"
    seeds = (2022,) if engineering else SEEDS
    output = OUT / "engineering_analysis" / runs.name if engineering else OUT / "analysis"
    paths = {(seed, arm): runs / arm if engineering else runs / f"seed{seed}" / arm
             for seed in seeds for arm in ("F", "DH")}
    manifests = {}
    for key, path in paths.items():
        manifest = json.loads((path / "run_manifest.json").read_text())
        progress = json.loads((path / "progress.json").read_text())
        assert manifest["status"] == "OK" and progress["phase"] == "COMPLETE"
        assert manifest["seed"] == key[0] and manifest["arm"] == key[1]
        assert bool(manifest.get("engineering_fixture", False)) == engineering
        assert bool(manifest["formal_valid"]) != engineering
        assert manifest["study_id"] == ("B1_ENGINEERING_INTEGRATION" if engineering else STUDY)
        assert manifest["dense_state_count"] == 50
        manifests[key] = manifest
    # No partial input can reach creation of formal analysis products.
    output.mkdir(parents=True, exist_ok=True)
    raw_path = Path("/home/xzf/data/kuairand/processed/kuai_rand.csv")
    dataset_sha = sha(raw_path)
    assert dataset_sha == "a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f"
    raw = pd.read_csv(raw_path, usecols=["tab", "is_click", "long_view"], dtype="int8")
    raw_labels = raw[["is_click", "long_view"]].to_numpy()
    raw_domains = raw.tab.map({0: 0, 1: 1, 4: 2}).to_numpy()
    cells, quantities, state_checks, selected, selections, windows, calibration, train_rows = [], [], [], {}, {}, [], [], []
    for (seed, arm), path in paths.items():
        manifest = manifests[seed, arm]
        if not engineering:
            train_n = int(len(raw) * .8)
            assert manifest["examples_processed"] == train_n * 50
            assert manifest["batches_per_epoch"] == (train_n + 40959) // 40960
        config_path = path / "config.json"
        assert sha(config_path) == manifest["artifact_hashes"]["config.json"]
        selection = json.loads((path / "selection_frozen.json").read_text())["selection"]
        selections[seed, arm] = selection
        index = json.loads((path / "dense/index.json").read_text())
        assert [v["epoch"] for v in index] == list(range(50))
        partition = json.loads((path / "gradient_partition.json").read_text())
        epoch_csv = pd.read_csv(path / "metrics_epoch.csv")
        dense_csv = pd.read_csv(path / "dense/metrics_test.csv")
        assert len(epoch_csv) == len(dense_csv) == 300
        for epoch in range(50):
            entry = index[epoch]
            state_path = path / "dense" / entry["path"]
            digest = sha(state_path)
            assert digest == entry["sha256"] == manifest["dense_artifact_hashes"][f"dense/{entry['path']}"]
            payload = torch.load(state_path, map_location="cpu", weights_only=False)
            counts = audit_state(payload, partition, epoch, int(manifest["batches_per_epoch"]))
            assert payload["initial_parameter_state_hash"] == manifest["initial_parameter_state_hash"]
            state_checks.append(dict(seed=seed, arm=arm, epoch=epoch, sha256=digest, **counts))
            train_info = epoch_csv[epoch_csv.epoch == epoch]
            assert len(train_info) == 6 and train_info.train_objective.nunique() == 1
            train_rows.append(dict(seed=seed, arm=arm, epoch=epoch, train_loss=train_info.train_objective.iloc[0],
                                   selected=epoch == selection["R0"], common=epoch in (9, 19, 29, 39, 49)))
            for split in ("validation", "test"):
                prediction_path = path / "dense" / f"{split}_epoch{epoch:02d}.npz"
                assert sha(prediction_path) == manifest["dense_artifact_hashes"][str(prediction_path.relative_to(path))]
                data = predictions(prediction_path, raw_labels, raw_domains)
                ids, domains, labels, scores = data
                assert (ids >= int(len(raw) * .8)).all() and (ids < len(raw)).all()
                if split == "validation":
                    assert (ids < int(len(raw) * .9)).all()
                else:
                    assert (ids >= int(len(raw) * .9)).all()
                if not engineering:
                    begin = int(len(raw) * (.8 if split == "validation" else .9))
                    end = int(len(raw) * .9) if split == "validation" else len(raw)
                    assert np.array_equal(ids, np.arange(begin, end, dtype=np.int64)), "incomplete evaluation split"
                if epoch == selection["R0"]:
                    selected[seed, arm, split] = data
                rec = {}
                for d in range(3):
                    for t in range(2):
                        mask = domains == d
                        y, p = labels[mask, t], scores[mask, t]
                        auc, ll = roc_auc_score(y, p), bce(y, p).mean()
                        original = (train_info if split == "validation" else dense_csv[dense_csv.epoch == epoch])
                        original = original[(original.domain == d) & (original.task == t)].iloc[0]
                        assert abs(auc - original["validation_auc" if split == "validation" else "auc"]) < 1e-12
                        assert abs(ll - original["validation_logloss" if split == "validation" else "logloss"]) < 1e-12
                        row = dict(seed=seed, arm=arm, split=split, epoch=epoch, domain=d, task=t,
                                   n=len(y), n_pos=int(y.sum()), auc=auc, logloss=ll)
                        rec[d, t] = row
                        cells.append(row)
                        if epoch == selection["R0"]:
                            for window, positions in enumerate(np.array_split(np.arange(len(ids)), 4)):
                                wm = np.intersect1d(np.flatnonzero(mask), positions)
                                yy, pp = labels[wm, t], scores[wm, t]
                                windows.append(dict(seed=seed, arm=arm, split=split, domain=d, task=t,
                                    window=window, kind="global row-order quartile proxy", n=len(yy), n_pos=int(yy.sum()),
                                    auc=roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan,
                                    logloss=bce(yy, pp).mean() if len(yy) else np.nan))
                            bins = np.minimum((p * 10).astype(int), 9)
                            for bin_id in range(10):
                                bm = bins == bin_id
                                calibration.append(dict(seed=seed, arm=arm, split=split, domain=d, task=t,
                                    bin=bin_id, n=int(bm.sum()), positive_count=int(y[bm].sum()),
                                    mean_prediction=p[bm].mean() if bm.any() else np.nan,
                                    positive_rate=y[bm].mean() if bm.any() else np.nan,
                                    brier=(p - y).dot(p - y) / len(y)))
                for metric, values in aggregates(rec).items():
                    quantities.extend(dict(seed=seed, arm=arm, split=split, epoch=epoch, metric=metric,
                                           quantity=name, value=float(value)) for name, value in values.items())
            del payload
        print(f"audited {seed}/{arm}", flush=True)
    for seed in seeds:
        f, h = paths[seed, "F"], paths[seed, "DH"]
        fi, hi = (json.loads((p / "reconstruction_identity.json").read_text()) for p in (f, h))
        assert fi["actual_initial_parameter_hash"] == hi["actual_initial_parameter_hash"]
        assert fi["dataset"] == hi["dataset"]
        findex, hindex = (json.loads((p / "dense/index.json").read_text()) for p in (f, h))
        for left, right in zip(findex, hindex):
            for field in ("initial_parameter_hash", "main_batch_hash", "arch_batch_hash"):
                assert left[field] == right[field]
    q = pd.DataFrame(quantities)
    lookup = q.set_index(["seed", "arm", "split", "epoch", "metric", "quantity"]).value
    for seed in seeds:
        for arm in ("F", "DH"):
            for rule, quantity in zip(("R0", "R1", "R2", "R3", "R4"),
                                      ("macro", "target", "non_target", "worst_domain", "eval_row_weighted")):
                expected = max(range(50), key=lambda e: (lookup.loc[seed, arm, "validation", e, "auc", quantity], -e))
                assert selections[seed, arm][rule] == expected
    same_epoch = q.pivot(index=["seed", "split", "epoch", "metric", "quantity"], columns="arm", values="value").reset_index()
    same_epoch["delta"] = same_epoch.DH - same_epoch.F
    comparisons = []
    for seed in seeds:
        for rule in ("R0", "R1", "R2", "R3", "R4"):
            ef, eh = selections[seed, "F"][rule], selections[seed, "DH"][rule]
            for split in ("validation", "test"):
                for metric in ("auc", "logloss"):
                    for quantity in q.quantity.unique():
                        f0, f1 = (lookup.loc[seed, "F", split, e, metric, quantity] for e in (ef, eh))
                        h0, h1 = (lookup.loc[seed, "DH", split, e, metric, quantity] for e in (ef, eh))
                        delta, policy_f, timing_h, policy_h, timing_f = h1 - f0, h0 - f0, h1 - h0, h1 - f1, f1 - f0
                        assert abs(delta - policy_f - timing_h) < 1e-12
                        assert abs(delta - policy_h - timing_f) < 1e-12
                        comparisons.append(dict(seed=seed, split=split, rule=rule, metric=metric, quantity=quantity,
                            F_epoch=ef, DH_epoch=eh, F=f0, DH=h1, delta=delta,
                            policy_at_F=policy_f, timing_on_DH=timing_h, policy_at_DH=policy_h, timing_on_F=timing_f))
    paired = pd.DataFrame(comparisons)
    summary = paired.groupby(["split", "rule", "metric", "quantity"]).delta.agg(["count", "mean", "std", "min", "max"]).reset_index()
    for idx, row in summary.iterrows():
        subset = paired[(paired.split == row.split) & (paired.rule == row.rule)
                        & (paired.metric == row.metric) & (paired.quantity == row.quantity)]
        beneficial = subset.delta if row.metric == "auc" else -subset.delta
        summary.loc[idx, "beneficial_seed_count"] = int((beneficial > 0).sum())
        if row.metric == "auc":
            summary.loc[idx, "delta_ge_003_count"] = int((subset.delta >= .003).sum())
            summary.loc[idx, "delta_ge_005_count"] = int((subset.delta >= .005).sum())
            summary.loc[idx, "delta_le_minus003_count"] = int((subset.delta <= -.003).sum())
    boot, draws = bootstrap(selected, seeds)
    calibration_frame = pd.DataFrame(calibration)
    calibration_summary = []
    for key, group in calibration_frame.groupby(["seed", "arm", "split", "domain", "task"]):
        finite = group[group.n > 0]
        ece = np.sum(finite.n * np.abs(finite.mean_prediction - finite.positive_rate)) / finite.n.sum()
        calibration_summary.append(dict(zip(("seed", "arm", "split", "domain", "task"), key),
                                        ece_10bins=ece, brier=group.brier.iloc[0]))
    for name, table in (("cell_metrics", pd.DataFrame(cells)), ("aggregate_metrics", q),
                        ("state_checks", pd.DataFrame(state_checks)), ("training_observations", pd.DataFrame(train_rows)),
                        ("same_epoch_differences", same_epoch), ("selected_timing_decomposition", paired),
                        ("seed_summary", summary), ("selected_paired_bootstrap", boot),
                        ("selected_row_order_windows", pd.DataFrame(windows)), ("selected_calibration", calibration_frame),
                        ("selected_calibration_summary", pd.DataFrame(calibration_summary))):
        table.to_csv(output / f"{name}.csv", index=False)
    np.savez_compressed(output / "bootstrap_draws.npz", **draws)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for i, split in enumerate(("validation", "test")):
        for j, quantity in enumerate(("macro", "non_target", "target")):
            ax = axes[i, j]
            for arm, color in (("F", "#2878b5"), ("DH", "#d55e00")):
                sub = q[(q.split == split) & (q.metric == "auc") & (q.quantity == quantity) & (q.arm == arm)]
                for seed in seeds:
                    run = sub[sub.seed == seed].sort_values("epoch")
                    ax.plot(run.epoch, run.value, color=color, alpha=.3, linewidth=.8)
                    selected_epoch = selections[seed, arm]["R0"]
                    ax.scatter(selected_epoch, run[run.epoch == selected_epoch].value.iloc[0],
                               color=color, marker="x", s=16)
                mean = sub.groupby("epoch").value.mean()
                ax.plot(mean.index, mean.values, color=color, linewidth=1.8, label=arm)
            ax.set_title(f"{split}: {quantity}")
            ax.set_xlabel("Epoch (0-based)")
            ax.set_ylabel("AUC")
            ax.legend()
    fig.suptitle("ENGINEERING FIXTURE — NO SCIENTIFIC INFERENCE" if engineering else "B1 reconstruction: trajectories; x = validation R0 checkpoint")
    fig.tight_layout()
    fig.savefig(output / "auc_trajectories.png", dpi=180)
    fig.savefig(output / "auc_trajectories.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ("validation", "test")):
        sub = boot[(boot.seed == "fixed_model_mean") & (boot.split == split) & (boot.metric == "auc")].sort_values(["domain", "task"])
        ax.hlines(np.arange(len(sub)), sub.ci95_low, sub.ci95_high, color="#2878b5")
        ax.scatter(sub.delta, np.arange(len(sub)), color="#2878b5", s=20)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_yticks(np.arange(len(sub)), [f"d{d}t{t}" for d, t in zip(sub.domain, sub.task)])
        ax.set_xlabel("Selected DH - F AUC; row bootstrap marginal 95% CI")
        ax.set_title(split)
    fig.suptitle("ENGINEERING FIXTURE ONLY" if engineering else "Conditional on fixed fitted models; not a seed-population CI")
    fig.tight_layout()
    fig.savefig(output / "selected_cell_intervals.png", dpi=180)
    fig.savefig(output / "selected_cell_intervals.pdf")
    plt.close(fig)
    assert sha(raw_path) == dataset_sha
    assert sha(__file__) == checked_source_sha, "analysis source changed while executing"
    save_json(output / "analysis_receipt.json", {"status": "PASS_NUMERICAL_AUDIT_ONLY",
        "scope": "ENGINEERING_ONLY" if engineering else "B1_RECONSTRUCTION_FORENSIC",
        "formal_runs_completed": 0 if engineering else len(paths), "run_count_audited": len(paths),
        "state_count_audited": len(state_checks), "prediction_files_audited": len(paths) * 100,
        "dataset_sha256": dataset_sha, "analysis_source_sha256": checked_source_sha, "timestamp": time.time(),
        "scientific_verdict": "PENDING_MAIN_AGENT_ANALYSIS", "independent_review": False,
        "sources": {str(path.relative_to(ROOT)): sha(path / "run_manifest.json") for path in paths.values()},
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != "analysis_receipt.json"}})
    print(json.dumps({"output": str(output), "formal_runs_completed": 0 if engineering else len(paths)}), flush=True)


if __name__ == "__main__":
    main()
