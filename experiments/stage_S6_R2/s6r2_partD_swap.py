"""s6r2_partD_swap.py

Part D: parameter-swap hybrid inference (KuaiRand only per spec §3.5, §5).

For each (seed), we:
    1. Load the chosen F and DH state_dicts (canonical selected).
    2. Verify keys/shapes match, partition completeness, partition
       disjointness, partition union = state_dict keys (§3.5 / §2.4).
    3. Rebuild the M3OE model from the joint DataBundle (same seed).
    4. Evaluate F_full and DH_full on test → six-cell AUCs + T + N + M.
       These act as **anchor reproduction**: must match `metrics_test.csv`
       per-cell AUC for the `selected` row of each arm at |Δ| < 1e-4.
       On failure the script exits non-zero.
    5. Construct hybrid checkpoints:
         C  = DH(S∪A) + F(P)
         D  = F(S∪A) + DH(P)
         (optional) E = DH(S) + F(A) + F(P)
         (optional) F = F(S) + DH(A) + F(P)
    6. Evaluate each hybrid on test; report six-cell AUC + T + N + M.
    7. Compute retention ρ_X = (X(hybrid) - X(F_full)) / (X(DH_full) - X(F_full))
       on the test layer for X ∈ {N, T, M, six cells} and on val for {N, T, M}.
    8. Emit the integrity report contribution: anchor reproduction residuals,
       partition completeness, key/shape equality, F/DH state_dict equality.

Outputs (single file, KuaiRand only):
    s6r2_partD_swap_kuairand.csv

Run via:
    python -m experiments.stage_S6_R2.s6r2_partD_swap [--device cpu|cuda:0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from stage_S6_R2 import s6r2_common as c  # noqa: E402

import stage_S1_data as s1data  # noqa: E402
import stage_S1_models as s1models  # noqa: E402
import stage_S1_metrics as s1metrics  # noqa: E402

from m3oe.MDMTRec.utils.data import TorchDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


DEFAULT_BATCH_SIZE = 40960


# ---------------------------------------------------------------------------
# Integrity helpers.
# ---------------------------------------------------------------------------
def _verify_partitions(
    spec: c.DatasetSpec,
    manifest_f: Dict[str, object],
    manifest_dh: Dict[str, object],
) -> Dict[str, object]:
    report: Dict[str, object] = {}
    for arm, m in (("F", manifest_f), ("DH", manifest_dh)):
        names = set(m["names"])
        S = set(m["shared_names"])
        A = set(m["architecture_names"])
        P = set(m["private_names"])
        report[f"{arm}_partition_count"] = {
            "shared": len(S), "architecture": len(A),
            "private": len(P), "union_names": len(S | A | P),
        }
        assert (S | A | P) == names, (
            f"{arm}: union of S/A/P must equal names set"
        )
        assert S.isdisjoint(A) and S.isdisjoint(P) and A.isdisjoint(P), (
            f"{arm}: partition groups must be mutually exclusive"
        )
        assert ("domain_expert" not in " ".join(S) + " ".join(P)
                or "domain_expert" in " ".join(P)), (
            f"{arm}: domain_expert ownership disagrees between manifest and "
            "gradient_partition (documented in design §2.4 / §3.5)"
        )
    # Manifest parity.
    assert (manifest_f["shared_names"] == manifest_dh["shared_names"]
            and manifest_f["architecture_names"] == manifest_dh["architecture_names"]
            and manifest_f["private_names"] == manifest_dh["private_names"]), (
        "shared/architecture/private name sets must be identical across F/DH"
    )
    return report


def _verify_state_dicts(
    sd_f: Dict[str, torch.Tensor],
    sd_dh: Dict[str, torch.Tensor],
    manifest: Dict[str, object],
) -> Dict[str, object]:
    kf = set(sd_f.keys())
    kd = set(sd_dh.keys())
    mn = set(manifest["names"])
    assert kf == kd == mn, (
        f"F/DH/manifest key sets differ: |F|={len(kf)} |DH|={len(kd)} |M|={len(mn)}"
    )
    # Same shape per key.
    diff_shapes: List[str] = []
    for k in sorted(mn):
        if sd_f[k].shape != sd_dh[k].shape:
            diff_shapes.append(k)
    if diff_shapes:
        raise AssertionError(f"shape mismatch on {len(diff_shapes)} keys: {diff_shapes[:5]}")
    # Across the 16 F/DH runs we expect no key/shape mismatch (we only check the
    # pair of this seed; per-seed parity is reported but not asserted beyond pairs).
    return {
        "key_count": len(mn),
        "shared_keys": len(set(manifest["shared_names"]) & kf),
        "architecture_keys": len(set(manifest["architecture_names"]) & kf),
        "private_keys": len(set(manifest["private_names"]) & kf),
        "shape_mismatches": 0,
    }


# ---------------------------------------------------------------------------
# Inference.
# ---------------------------------------------------------------------------
def _make_data_loaders(
    spec: c.DatasetSpec, arm: str, seed: int, batch_size: int,
) -> Tuple[DataLoader, DataLoader, object]:
    cfg = c.load_config(spec, arm, seed)
    setting = cfg.get("setting", spec.setting)
    bundle = s1data.load_stage_s1_bundle(spec.name, None, str(setting), seed)
    val_loader = DataLoader(
        TorchDataset(bundle.val_x, bundle.val_y),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        TorchDataset(bundle.test_x, bundle.test_y),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return val_loader, test_loader, bundle


def _build_model(bundle, expert_num: int, fcn_dims: list) -> torch.nn.Module:
    return s1models.build_model(
        bundle.features, domain_num=3, task_num=2,
        backbone="m3oe", expert_num=int(expert_num),
        fcn_dims=list(fcn_dims),
    )


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    domain_num: int, task_num: int,
    return_predictions: bool = False,
) -> Dict[str, object]:
    """Return per-cell AUC (computed via safe_auc) + T/N/M/D0/D2 + (optional)
    raw prediction arrays."""
    truth: Dict[int, List[float]] = {t * domain_num + d: []
                                     for t in range(task_num) for d in range(domain_num)}
    predictions: Dict[int, List[float]] = {k: [] for k in truth}
    pred_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    domain_parts: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            x_device = xb.transpose(0, 1).to(device=device, dtype=torch.float32)
            y_device = yb.to(device=device, dtype=torch.float32)
            pred = model(x_device)
            if not torch.isfinite(pred).all():
                raise FloatingPointError("non-finite evaluation prediction")
            domain_ids = x_device[-1, :].long()
            for t in range(task_num):
                for d in range(domain_num):
                    key = t * domain_num + d
                    mask = domain_ids == d
                    if mask.any():
                        truth[key].extend(y_device[mask, t].cpu().tolist())
                        predictions[key].extend(pred[mask, t].cpu().tolist())
            if return_predictions:
                pred_parts.append(pred.cpu().numpy().astype(np.float32))
                label_parts.append(yb.cpu().numpy().astype(np.float32))
                domain_parts.append(xb[:, -1].cpu().numpy().astype(np.int64))
    result: Dict[str, Dict[str, float]] = {}
    for key in truth:
        labels = np.asarray(truth[key], dtype=np.float64)
        scores = np.asarray(predictions[key], dtype=np.float64)
        if labels.size == 0:
            result[key] = {"sample_count": 0, "auc": float("nan"), "logloss": float("nan")}
            continue
        if not np.isfinite(labels).all() or not np.isfinite(scores).all():
            raise FloatingPointError("non-finite inputs to safe_auc")
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            result[key] = {"sample_count": int(labels.size), "auc": float("nan"),
                           "logloss": float("nan")}
        else:
            auc = s1metrics.safe_auc(labels, scores)
            result[key] = {
                "sample_count": int(labels.size),
                "auc": float(auc),
                "logloss": float(s1metrics.safe_logloss(labels, scores)),
            }
    if return_predictions:
        result["__raw__"] = {
            "predictions": np.concatenate(pred_parts, axis=0) if pred_parts else np.zeros((0, task_num), np.float32),
            "labels": np.concatenate(label_parts, axis=0) if label_parts else np.zeros((0, task_num), np.float32),
            "domains": np.concatenate(domain_parts, axis=0) if domain_parts else np.zeros((0,), np.int64),
        }
    return result


def _agg_from_per_cell(
    spec: c.DatasetSpec,
    per_cell: Dict[int, Dict[str, float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    cells_list = [per_cell[t * 3 + d]["auc"] for t in range(2) for d in range(3)]
    target_d = spec.target_domain
    target_vals = [per_cell[0 * 3 + target_d]["auc"], per_cell[1 * 3 + target_d]["auc"]]
    non_target_vals = [
        per_cell[t * 3 + d]["auc"]
        for t in range(2) for d in range(3) if d != target_d
    ]
    out["M"] = float(np.mean(cells_list))
    out["T"] = float(np.mean(target_vals))
    out["N"] = float(np.mean(non_target_vals))
    out["D0"] = float(np.mean([per_cell[t * 3 + 0]["auc"] for t in range(2)]))
    out["D2"] = float(np.mean([per_cell[t * 3 + 2]["auc"] for t in range(2)]))
    for d in range(3):
        for t in range(2):
            out[c.cell_label(d, t)] = per_cell[t * 3 + d]["auc"]
    return out


def _evaluate_one(
    spec: c.DatasetSpec,
    seed: int,
    state_dict: Dict[str, torch.Tensor],
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    expert_num: int,
    fcn_dims: list,
    bundle_obj: object,
) -> Dict[str, Dict[str, float]]:
    """Build the model, load the state_dict, and return aggregated metrics on
    val and test layers (per the 10-metric dict). The data bundle is supplied
    by the caller (loaded once per seed) so the dataset is not re-read for
    every hybrid."""
    model = _build_model(bundle_obj, expert_num, fcn_dims).to(device)
    res = model.load_state_dict(state_dict, strict=True)
    if res.missing_keys or res.unexpected_keys:
        raise AssertionError(
            f"load_state_dict mismatch: missing={res.missing_keys} "
            f"unexpected={res.unexpected_keys}"
        )
    per_cell_val = _evaluate_loader(model, val_loader, device, 3, 2)
    per_cell_test = _evaluate_loader(model, test_loader, device, 3, 2)
    return {
        "val": _agg_from_per_cell(spec, per_cell_val),
        "test": _agg_from_per_cell(spec, per_cell_test),
    }


# ---------------------------------------------------------------------------
# Hybrid construction.
# ---------------------------------------------------------------------------
def _make_hybrid(
    sd_f: Dict[str, torch.Tensor],
    sd_dh: Dict[str, torch.Tensor],
    S: set, A: set, P: set,
    kind: str,
) -> Dict[str, torch.Tensor]:
    """kind ∈ {"C","D","E","F"} per spec §3.5."""
    out: Dict[str, torch.Tensor] = {}
    if kind == "C":
        # DH(S∪A) + F(P)
        return {k: (sd_dh[k] if k in (S | A) else sd_f[k]) for k in sd_f}
    if kind == "D":
        # F(S∪A) + DH(P)
        return {k: (sd_f[k] if k in (S | A) else sd_dh[k]) for k in sd_f}
    if kind == "E":
        # DH(S) + F(A∪P)
        return {k: (sd_dh[k] if k in S else sd_f[k]) for k in sd_f}
    if kind == "F":
        # F(S) + DH(A) + F(P)
        return {k: (sd_f[k] if k in (S | P) else sd_dh[k]) for k in sd_f}
    raise ValueError(f"unknown hybrid kind {kind}")


# ---------------------------------------------------------------------------
# Anchor reproduction: compare to metrics_test.csv selected row.
# ---------------------------------------------------------------------------
def _test_anchor_for_arm_seed(spec: c.DatasetSpec, arm: str, seed: int) -> Dict[str, float]:
    """Load saved `selected` row of metrics_test.csv → 10-metric dict."""
    csv_p = c.run_joint_dir(spec, arm, seed) / "metrics_test.csv"
    df = pd.read_csv(csv_p, encoding="utf-8-sig")
    sub = df[df["checkpoint"] == "selected"]
    if len(sub) != 6:
        raise AssertionError(
            f"{spec.name}/{arm}/seed{seed}: expected 6 'selected' rows, got {len(sub)}"
        )
    per_cell_auc: Dict[Tuple[int, int], float] = {}
    for _, row in sub.iterrows():
        per_cell_auc[(int(row["task"]), int(row["domain"]))] = float(row["auc"])
    # Compose 10-metric dict in same shape as `_agg_from_per_cell`.
    cells_list = [per_cell_auc[(t, d)] for t in range(2) for d in range(3)]
    target_d = spec.target_domain
    target_vals = [per_cell_auc[(0, target_d)], per_cell_auc[(1, target_d)]]
    nt_vals = [per_cell_auc[(t, d)]
               for t in range(2) for d in range(3) if d != target_d]
    out = {
        "M": float(np.mean(cells_list)),
        "T": float(np.mean(target_vals)),
        "N": float(np.mean(nt_vals)),
        "D0": float(np.mean([per_cell_auc[(t, 0)] for t in range(2)])),
        "D2": float(np.mean([per_cell_auc[(t, 2)] for t in range(2)])),
    }
    for d in range(3):
        for t in range(2):
            out[c.cell_label(d, t)] = per_cell_auc[(t, d)]
    return out


def _safe_load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    sd = torch.load(path, map_location="cpu")
    if not isinstance(sd, dict):
        raise TypeError(f"{path}: expected state_dict dict")
    return {str(k): v for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def _run_for_seed(
    spec: c.DatasetSpec,
    seed: int,
    device: torch.device,
    batch_size: int,
    integrity: Dict[str, object],
) -> List[Dict[str, object]]:
    cfg_f = c.load_config(spec, "F", seed)
    cfg_dh = c.load_config(spec, "DH", seed)
    # F and DH of the same seed must share architecture/setting (controlled
    # variables; spec §4).
    for key in ("setting", "expert_num", "fcn_dims", "backbone"):
        assert cfg_f.get(key) == cfg_dh.get(key), (
            f"{spec.name}/seed{seed}: F/DH config mismatch on {key}: "
            f"{cfg_f.get(key)} vs {cfg_dh.get(key)}"
        )
    expert_num = int(cfg_f.get("expert_num", 4))
    fcn_dims = list(cfg_f.get("fcn_dims", [512, 256, 256, 64]))
    setting_f = str(cfg_f.get("setting", spec.setting))

    print(f"  [PartD] seed={seed} setting={setting_f} expert_num={expert_num} "
          f"fcn_dims={fcn_dims} device={device}")

    # 1. Manifests + state_dicts.
    manifest_f = c.read_parameter_manifest(spec, "F", seed)
    manifest_dh = c.read_parameter_manifest(spec, "DH", seed)
    integrity[f"seed{seed}_partition"] = _verify_partitions(spec, manifest_f, manifest_dh)
    sd_f = _safe_load_state_dict(
        c.run_joint_dir(spec, "F", seed) / "selected_checkpoint.pth", device)
    sd_dh = _safe_load_state_dict(
        c.run_joint_dir(spec, "DH", seed) / "selected_checkpoint.pth", device)
    integrity[f"seed{seed}_state_dict_parity"] = _verify_state_dicts(
        sd_f, sd_dh, manifest_f)

    # 2. Data loaders (bundle loaded once per seed and reused for every eval).
    val_loader, test_loader, bundle = _make_data_loaders(
        spec, "F", seed, batch_size)

    S = set(manifest_f["shared_names"])
    A = set(manifest_f["architecture_names"])
    P = set(manifest_f["private_names"])

    rows: List[Dict[str, object]] = []

    def _row_with_val(hybrid: str, m: Dict[str, Dict[str, float]]) -> Dict[str, object]:
        # Test layer is primary; val-layer M/T/N recorded for reference
        # (spec §3.5 / §8).
        return {
            "seed": int(seed), "hybrid": hybrid,
            **m["test"],
            "val_M": m["val"]["M"], "val_T": m["val"]["T"],
            "val_N": m["val"]["N"],
        }

    def _check_anchor(arm: str, metrics: Dict[str, float]) -> Dict[str, float]:
        anchor = _test_anchor_for_arm_seed(spec, arm, seed)
        residuals: Dict[str, float] = {}
        for k in ("M", "T", "N", "D0", "D2",
                  "d0t0", "d0t1", "d1t0", "d1t1", "d2t0", "d2t1"):
            res = float(metrics[k]) - float(anchor[k])
            residuals[k] = res
            if abs(res) >= c.TOL_ANCHOR:
                raise AssertionError(
                    f"{spec.name}/{arm}/seed{seed} anchor reproduction "
                    f"mismatch on {k}: pipe={metrics[k]:.9f} csv={anchor[k]:.9f} "
                    f"diff={res:+.2e} (tol={c.TOL_ANCHOR})"
                )
        return residuals

    # 3. F_full anchor.
    t0 = time.time()
    f_metrics = _evaluate_one(
        spec, seed, sd_f, val_loader, test_loader, device,
        expert_num, fcn_dims, bundle,
    )
    integrity[f"seed{seed}_F_anchor_residuals"] = _check_anchor("F", f_metrics["test"])
    rows.append(_row_with_val("F_full", f_metrics))
    print(f"    F_full M={f_metrics['test']['M']:.6f} N={f_metrics['test']['N']:.6f} "
          f"({time.time()-t0:.1f}s)")

    # 4. DH_full anchor.
    t0 = time.time()
    d_metrics = _evaluate_one(
        spec, seed, sd_dh, val_loader, test_loader, device,
        expert_num, fcn_dims, bundle,
    )
    integrity[f"seed{seed}_DH_anchor_residuals"] = _check_anchor("DH", d_metrics["test"])
    rows.append(_row_with_val("DH_full", d_metrics))
    print(f"    DH_full M={d_metrics['test']['M']:.6f} N={d_metrics['test']['N']:.6f} "
          f"({time.time()-t0:.1f}s)")

    # 5. Hybrids.
    for kind in ("C", "D"):
        t0 = time.time()
        sd = _make_hybrid(sd_f, sd_dh, S, A, P, kind)
        m = _evaluate_one(
            spec, seed, sd, val_loader, test_loader, device,
            expert_num, fcn_dims, bundle,
        )
        rows.append(_row_with_val(f"{kind}_hybrid", m))
        print(f"    {kind}_hybrid M={m['test']['M']:.6f} N={m['test']['N']:.6f} "
              f"({time.time()-t0:.1f}s)")

    # Optional E and F.
    for kind in ("E", "F"):
        t0 = time.time()
        sd = _make_hybrid(sd_f, sd_dh, S, A, P, kind)
        m = _evaluate_one(
            spec, seed, sd, val_loader, test_loader, device,
            expert_num, fcn_dims, bundle,
        )
        rows.append(_row_with_val(f"{kind}_hybrid", m))
        print(f"    {kind}_hybrid M={m['test']['M']:.6f} N={m['test']['N']:.6f} "
              f"({time.time()-t0:.1f}s)")

    return rows


def _retention_rows(test_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """For each (seed, hybrid) compute ρ_X = (X(hybrid) - X(F_full)) / (X(DH_full) - X(F_full))
    on X ∈ {M, T, N, six cells}. Test layer only. `NA` if denominator near 0."""
    df = pd.DataFrame(test_rows)
    metrics = ("M", "T", "N", "D0", "D2",
               "d0t0", "d0t1", "d1t0", "d1t1", "d2t0", "d2t1")
    out: List[Dict[str, object]] = []
    for seed in sorted(set(int(s) for s in df["seed"].unique())):
        per_seed = df[df["seed"] == seed]
        f_full = per_seed[per_seed["hybrid"] == "F_full"].iloc[0]
        dh_full = per_seed[per_seed["hybrid"] == "DH_full"].iloc[0]
        for _, row in per_seed.iterrows():
            H = row["hybrid"]
            if H in ("F_full", "DH_full"):
                continue
            rec: Dict[str, object] = {"seed": int(seed), "hybrid": H}
            for m in metrics:
                denom = float(dh_full[m]) - float(f_full[m])
                num = float(row[m]) - float(f_full[m])
                if abs(denom) < 1e-9:
                    rec[f"rho_{m}"] = float("nan")
                    rec[f"raw_delta_{m}"] = num
                    rec[f"denom_{m}"] = denom
                else:
                    rec[f"rho_{m}"] = num / denom
                    rec[f"raw_delta_{m}"] = num
                    rec[f"denom_{m}"] = denom
            out.append(rec)
    return out


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(c.SPEC.keys()), required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)
    spec = c.SPEC[args.dataset]
    if spec.name != "kuairand":
        raise SystemExit("Part D runs only on kuairand per spec §3.5.")
    out_dir = c.ensure_out_dir(spec)

    device = torch.device(args.device)
    print(f"[PartD] dataset={spec.name} device={device}")

    integrity: Dict[str, object] = {}
    rows: List[Dict[str, object]] = []
    for seed in c.SEEDS:
        rows.extend(_run_for_seed(spec, seed, device, args.batch_size, integrity))
    df = pd.DataFrame(rows).sort_values(["seed", "hybrid"]).reset_index(drop=True)
    c.write_csv(df, out_dir / "s6r2_partD_swap_kuairand.csv")
    print(f"[PartD] wrote {len(df)} hybrid eval rows.")

    retention = _retention_rows(rows)
    df_ret = pd.DataFrame(retention).sort_values(["seed", "hybrid"]).reset_index(drop=True)
    c.write_csv(df_ret, out_dir / "s6r2_partD_swap_retention_kuairand.csv")
    print(f"[PartD] wrote {len(df_ret)} retention rows.")

    # Append integrity report to the global integrity JSON if present.
    int_path = c.OUT_ROOT / "s6r2_integrity_report.json"
    if int_path.exists():
        with open(int_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}
    existing.setdefault("partD", {})[spec.name] = integrity
    int_path.parent.mkdir(parents=True, exist_ok=True)
    with open(int_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"[PartD] integrity report merged into {int_path}")


if __name__ == "__main__":
    main()
