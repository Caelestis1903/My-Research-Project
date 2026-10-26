"""C2.1 finite-population, fixed-theta CPU numerical oracle; zero training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from b1_reconstruction_20261016 import ROOT, save_json, sha
from stage_S1_data import load_stage_s1_bundle
from stage_S1_models import build_model, initial_state_hash
from stage_S3_gradient_access import partition_parameters, assert_expected_parameter_counts
from stage_S3_protocol import source_inventory

OUT = ROOT / "research_plans/2026-10-15/execution_20261015/C2_numeric_oracle_20261016"
GROUPS = ("C", "P", "A_arch")
SAMPLES_PER_DOMAIN = 16
MC_BATCHES = 10000
CONFIGS = {
    "kuairand": ("natural", "/home/xzf/data/kuairand/processed", "kuai_rand.csv",
                  "a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f"),
    "movielens": ("d2_large", "/home/xzf/data/ml-1m", "ml-1m.csv",
                  "f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4"),
}


def flatten_gradient(loss, parameters, retain=False):
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain)
    return np.concatenate([(np.zeros(p.numel(), dtype=np.float32) if g is None
                           else g.detach().cpu().reshape(-1).numpy()) for p, g in zip(parameters, gradients)])


def adam_formula(gradients, theta, lr):
    effective = gradients + 1e-5 * theta
    # Valid at fresh Adam step0 after bias correction, betas=(.9,.999).
    return -lr * effective / (np.abs(effective) + 1e-8)


def adam_torch_check(gradient, theta, selected_groups):
    parameters, groups, ranges = [], [], []
    for name in GROUPS:
        ids = np.flatnonzero(selected_groups == name)
        parameter = torch.nn.Parameter(torch.tensor(theta[ids], dtype=torch.float64))
        parameters.append(parameter)
        ranges.append(ids)
        groups.append({"params": [parameter], "lr": .001 if name == "A_arch" else .01})
    optimizer = torch.optim.Adam(groups, betas=(.9, .999), eps=1e-8,
                                 weight_decay=1e-5, foreach=False)
    before = [p.detach().clone() for p in parameters]
    for p, ids in zip(parameters, ranges):
        p.grad = torch.tensor(gradient[ids], dtype=torch.float64)
    optimizer.step()
    actual = np.empty_like(theta)
    for p, old, ids in zip(parameters, before, ranges):
        actual[ids] = (p.detach() - old).numpy()
    lr = np.where(selected_groups == "A_arch", .001, .01)
    residual = float(np.max(np.abs(actual - adam_formula(gradient, theta, lr))))
    assert residual < 1e-12
    return residual


def numeric_case(dataset, seed):
    setting, directory, filename, expected_hash = CONFIGS[dataset]
    case_dir = OUT / dataset / f"seed{seed}"
    case_dir.mkdir(parents=True, exist_ok=False)
    assert sha(Path(directory) / filename) == expected_hash
    frozen_sources = source_inventory(str(ROOT))
    frozen_sources[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    bundle = load_stage_s1_bundle(dataset, directory, setting, seed)
    model = build_model(bundle.features, 3, 2, "m3oe", expert_num=4,
                        fcn_dims=(512, 256, 256, 64), seed=seed).cpu().eval()
    initial_hash = initial_state_hash(model)
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    assert not any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules())
    weighted_modules = [m for m in model.modules() if hasattr(m, "softmax_type")]
    assert weighted_modules and all(m.softmax_type == 3 for m in weighted_modules)
    attrs = [(m, m.tau) for m in weighted_modules]

    def forward(x):
        value = model(x)
        # tau is unused in sigmoid softmax_type=3; keep even Python state fixed.
        for m, tau in attrs:
            m.tau = tau
        return value

    partition = partition_parameters(model)
    assert_expected_parameter_counts(partition, dataset)
    named = list(model.named_parameters())
    parameters = [p for _, p in named]
    theta = np.concatenate([p.detach().numpy().reshape(-1) for p in parameters]).astype(np.float64)
    offsets, offset, group_indices, coordinate_names = {}, 0, {g: [] for g in GROUPS}, []
    name_groups = {n: group for group in GROUPS for n in partition["groups"][group]}
    for name, parameter in named:
        offsets[name] = (offset, offset + parameter.numel())
        group_indices[name_groups[name]].extend(range(offset, offset + parameter.numel()))
        offset += parameter.numel()
    group_indices = {k: np.asarray(v, dtype=np.int64) for k, v in group_indices.items()}
    rng = np.random.default_rng(20261016)
    row_positions = np.concatenate([rng.choice(np.flatnonzero(bundle.train_x[:, -1] == d),
                                   size=SAMPLES_PER_DOMAIN, replace=False) for d in range(3)])
    x = torch.tensor(bundle.train_x[row_positions], dtype=torch.float32)
    y = torch.tensor(bundle.train_y[row_positions], dtype=torch.float32)
    domains = np.repeat(np.arange(3), SAMPLES_PER_DOMAIN)
    pool_ids = bundle.train_row_ids[row_positions]
    n = len(row_positions)
    gradient_matrix = np.empty((n, len(theta)), dtype=np.float32)
    losses = np.empty((n, 2), dtype=np.float64)
    individual_predictions, cell_rows = [], []
    for i in range(n):
        prediction = forward(x[i:i + 1].T)
        individual_predictions.append(prediction.detach().numpy()[0])
        terms = F.binary_cross_entropy(prediction, y[i:i + 1], reduction="none")[0]
        losses[i] = terms.detach().numpy()
        gg = [flatten_gradient(terms[t], parameters, retain=t == 0) for t in range(2)]
        gradient_matrix[i] = (gg[0] + gg[1]) * .5
        for t in range(2):
            for group in GROUPS:
                grad = gg[t][group_indices[group]].astype(np.float64)
                cell_rows.append(dict(dataset=dataset, seed=seed, domain=int(domains[i]), task=t,
                    row_id=int(pool_ids[i]), partition=group, loss=losses[i, t], gradient_norm=float(np.linalg.norm(grad))))
    pred_all = forward(x.T)
    prediction_residual = float(np.max(np.abs(pred_all.detach().numpy() - np.asarray(individual_predictions))))
    assert prediction_residual < 1e-6
    p_domain = np.asarray([bundle.train_counts[d] for d in range(3)], dtype=np.float64)
    p_domain /= p_domain.sum()
    loss_per_row = losses.mean(1)
    exact_gradient = {}
    target_checks = []
    for objective, alpha in (("Equal", np.ones(3) / 3), ("Sample", p_domain)):
        coefficients = alpha[domains] / SAMPLES_PER_DOMAIN
        expected = coefficients @ gradient_matrix
        exact_gradient[objective] = expected
        prediction = forward(x.T)
        terms = F.binary_cross_entropy(prediction, y, reduction="none").mean(1)
        target_loss = (terms * torch.tensor(coefficients, dtype=terms.dtype)).sum()
        actual = flatten_gradient(target_loss, parameters).astype(np.float64)
        residual = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12))
        assert residual < 1e-4
        target_checks.append(dict(dataset=dataset, seed=seed, objective=objective,
            full_coordinate_autograd_relative_residual=residual, partition_size=len(theta),
            exact_target_loss=float(coefficients @ loss_per_row)))
    selected, selected_groups = [], []
    for group in GROUPS:
        indices = group_indices[group]
        support = indices[np.any(gradient_matrix[:, indices] != 0, axis=0)]
        if group == "A_arch":
            chosen = indices
        else:
            assert len(support)
            chosen = support[np.linspace(0, len(support) - 1, min(32, len(support))).astype(int)]
        selected.extend(chosen)
        selected_groups.extend([group] * len(chosen))
    selected = np.asarray(selected, dtype=np.int64)
    selected_groups = np.asarray(selected_groups)
    for index, group in zip(selected, selected_groups):
        name = next(name for name, (left, right) in offsets.items() if left <= index < right)
        coordinate_names.append(dict(index=int(index), partition=str(group), parameter=name,
                                     within_parameter_index=int(index - offsets[name][0])))
    projected = gradient_matrix[:, selected].astype(np.float64)
    row_norms = {group: np.asarray([np.dot(row[ids].astype(np.float64), row[ids].astype(np.float64))
                 for row in gradient_matrix]) for group, ids in group_indices.items()}
    rows, partition_rows, covariance = [], [], {}
    adam_max_residual = 0.
    lr = np.where(selected_groups == "A_arch", .001, .01)
    for objective, alpha in (("Equal", np.ones(3) / 3), ("Sample", p_domain)):
        target = exact_gradient[objective]
        target_loss = float(np.dot(alpha, losses.mean(1).reshape(3, SAMPLES_PER_DOMAIN).mean(1)))
        target_projected = target[selected]
        for sampler, q in (("Natural", p_domain), ("Balanced", np.ones(3) / 3),
                           ("Stress", np.asarray([.0001, .9998, .0001]))):
            prob = q[domains] / SAMPLES_PER_DOMAIN
            importance = alpha[domains] / q[domains]
            exact = (prob * importance) @ gradient_matrix
            exact_residual = float(np.linalg.norm(exact - target) / max(np.linalg.norm(target), 1e-12))
            assert exact_residual <= 1e-10
            assert abs(np.dot(prob * importance, loss_per_row) - target_loss) <= 1e-10
            for batch_size in (16, 128):
                mc_rng = np.random.default_rng(20261016 + seed + (1 if dataset == "movielens" else 0))
                counts = mc_rng.multinomial(batch_size, prob, size=MC_BATCHES)
                weighted = counts * importance / batch_size
                estimates = weighted @ projected
                loss_estimates = weighted @ loss_per_row
                cov_exact = ((prob * importance**2)[:, None] * projected).T @ projected
                cov_exact = (cov_exact - np.outer(target_projected, target_projected)) / batch_size
                cov_observed = np.cov(estimates, rowvar=False)
                variance_loss = float((np.dot(prob * importance**2, loss_per_row**2) - target_loss**2) / batch_size)
                loss_z = float(abs(loss_estimates.mean() - target_loss) / (np.sqrt(max(variance_loss, 0) / MC_BATCHES) + 1e-9))
                grad_z = float(np.max(np.abs(estimates.mean(0) - target_projected)
                                     / (np.sqrt(np.maximum(np.diag(cov_exact), 0) / MC_BATCHES) + 1e-8)))
                assert loss_z <= 8 and grad_z <= 8
                delta = adam_formula(estimates, theta[selected], lr)
                delta_mean_gradient = adam_formula(target_projected, theta[selected], lr)
                for g in (estimates[0], estimates[MC_BATCHES // 2], estimates[-1], target_projected):
                    adam_max_residual = max(adam_max_residual, adam_torch_check(g, theta[selected], selected_groups))
                domain_counts = counts.reshape(MC_BATCHES, 3, SAMPLES_PER_DOMAIN).sum(2)
                domain_loss_sum = (counts * loss_per_row).reshape(MC_BATCHES, 3, SAMPLES_PER_DOMAIN).sum(2)
                cell_normalized = np.divide(domain_loss_sum, domain_counts, out=np.zeros_like(domain_loss_sum), where=domain_counts > 0).mean(1)
                population_equal = float(loss_per_row.reshape(3, SAMPLES_PER_DOMAIN).mean(1).mean())
                expected_cell = float(np.mean((1 - (1 - q)**batch_size) * loss_per_row.reshape(3, SAMPLES_PER_DOMAIN).mean(1)))
                key = f"{objective}_{sampler}_B{batch_size}"
                covariance[key + "_exact"] = cov_exact
                covariance[key + "_mc"] = cov_observed
                rows.append(dict(dataset=dataset, seed=seed, objective=objective, sampler=sampler,
                    batch_size=batch_size, mc_batches=MC_BATCHES, target_loss=target_loss,
                    mc_loss_mean=float(loss_estimates.mean()), mc_loss_variance=float(loss_estimates.var(ddof=1)),
                    exact_loss_variance=variance_loss, exact_full_gradient_relative_residual=exact_residual,
                    mc_loss_standardized_residual=loss_z, mc_gradient_max_standardized_residual=grad_z,
                    projected_covariance_relative_error=float(np.linalg.norm(cov_exact - cov_observed) / max(np.linalg.norm(cov_exact), 1e-12)),
                    mean_unique_rows=float((counts > 0).sum(1).mean()),
                    exact_mean_unique_rows=float(np.sum(1 - (1 - prob)**batch_size)),
                    reuse_fraction=float(1 - (counts > 0).sum(1).mean() / batch_size),
                    batch_cell_equal_expected=expected_cell, batch_cell_equal_mc=float(cell_normalized.mean()),
                    batch_cell_equal_bias_vs_population_equal=expected_cell - population_equal,
                    **{f"absent_d{d}_exact": float((1 - q[d])**batch_size) for d in range(3)},
                    **{f"absent_d{d}_mc": float((domain_counts[:, d] == 0).mean()) for d in range(3)}))
                for group in GROUPS:
                    ids, mask = group_indices[group], selected_groups == group
                    exact_trace = float((np.dot(prob * importance**2, row_norms[group]) - np.dot(target[ids], target[ids])) / batch_size)
                    partition_rows.append(dict(dataset=dataset, seed=seed, objective=objective, sampler=sampler,
                        batch_size=batch_size, partition=group, full_coordinate_count=len(ids),
                        full_gradient_covariance_trace=exact_trace, expected_gradient_norm=float(np.linalg.norm(target[ids])),
                        projected_coordinate_count=int(mask.sum()),
                        projected_mean_adam_delta_norm=float(np.linalg.norm(delta.mean(0)[mask])),
                        projected_delta_of_mean_gradient_norm=float(np.linalg.norm(delta_mean_gradient[mask])),
                        projected_adam_jensen_difference_norm=float(np.linalg.norm((delta.mean(0) - delta_mean_gradient)[mask]))))
    assert initial_state_hash(model) == initial_hash
    assert all(parameter.grad is None for parameter in parameters)
    assert all(m.tau == tau for m, tau in attrs)
    assert sha(Path(directory) / filename) == expected_hash
    end_sources = source_inventory(str(ROOT))
    end_sources[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    assert end_sources == frozen_sources
    save_json(case_dir / "manifest.json", {"scope": "FINITE_POPULATION_FIXED_THETA_ORACLE", "training_runs": 0,
        "dataset": dataset, "setting": setting, "seed": seed, "dataset_sha256": expected_hash,
        "actual_initial_parameter_hash": initial_hash, "pool_row_ids": pool_ids.tolist(),
        "pool_domain_labels": domains.tolist(), "full_train_domain_probabilities": p_domain.tolist(),
        "pool_samples_per_domain": SAMPLES_PER_DOMAIN, "partition": partition,
        "selected_coordinates": coordinate_names, "adam_initial_state": "fresh step0; detached coordinate copies only",
        "adam_max_absolute_residual": adam_max_residual, "batch_prediction_max_residual": prediction_residual,
        "data_manifest": bundle.manifest_fields(), "target_checks": target_checks,
        "source_sha256": sha(__file__), "source_hashes": frozen_sources, "model_theta_unchanged": True,
        "limitations": ["finite small training pool, not full population", "selected-coordinate covariance matrix plus full-coordinate covariance trace",
                        "no selected AUC or training evidence", "no C2 training GO", "self-review only"]})
    pd.DataFrame(rows).to_csv(case_dir / "estimator_moments.csv", index=False)
    pd.DataFrame(partition_rows).to_csv(case_dir / "partition_moments.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(case_dir / "cell_gradient_observations.csv", index=False)
    np.savez_compressed(case_dir / "projected_covariance.npz", **covariance)
    return rows, partition_rows, target_checks


def main():
    assert not torch.cuda.is_available(), "CPU oracle requires CUDA_VISIBLE_DEVICES=''"
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_source = sha(__file__)
    started = time.time()
    all_rows, all_partitions, target_checks = [], [], []
    for dataset in CONFIGS:
        for seed in (2022, 2023, 2024, 2025):
            case_path = OUT / dataset / f"seed{seed}"
            if case_path.exists():
                raise FileExistsError(f"Inspect existing case before rerunning: {case_path}")
            rows, parts, checks = numeric_case(dataset, seed)
            all_rows.extend(rows)
            all_partitions.extend(parts)
            target_checks.extend(checks)
            save_json(OUT / "progress.json", {"phase": "RUN", "last_completed_dataset": dataset,
                "last_completed_seed": seed, "completed_cases": len(target_checks) // 2,
                "training_runs": 0, "source_sha256": frozen_source})
            print(json.dumps({"dataset": dataset, "seed": seed, "status": "ORACLE_CASE_PASS"}), flush=True)
    assert sha(__file__) == frozen_source
    pd.DataFrame(all_rows).to_csv(OUT / "estimator_moments.csv", index=False)
    pd.DataFrame(all_partitions).to_csv(OUT / "partition_moments.csv", index=False)
    pd.DataFrame(target_checks).to_csv(OUT / "target_checks.csv", index=False)
    save_json(OUT / "run_receipt.json", {"status": "PASS_NUMERIC_ORACLE_ONLY", "scope": "FINITE_POOL_FIXED_THETA",
        "cases_completed": 8, "training_runs": 0, "c2_training_go": False,
        "source_sha256": frozen_source, "elapsed_seconds": time.time() - started,
        "independent_review": False, "scientific_analysis": "PENDING_MAIN_AGENT",
        "artifact_hashes": {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob("*") if p.is_file() and p.name != "run_receipt.json"}})


if __name__ == "__main__":
    main()
