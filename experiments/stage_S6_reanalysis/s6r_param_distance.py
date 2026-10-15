"""s6r_param_distance.py

Generate table 11 — DH vs F parameter & optimizer-moment L2 distance, grouped by
shared / target / non_target / all partitions at E_common, plus Spearman with
non-target val gain (5 points only, weak evidence ceiling).

Partition assignment follows the actual state_dict keys (gradient_partition.json
in the run directory is the authoritative key universe) and the MDMTRec model
semantics (m3oe/MDMTRec/models/multi_domain/model.py):

  - ``slot_weight.{d}`` / ``slot_bias.{d}`` / ``domain_expert.{d}.*`` are per-domain
  - ``tower.{i}.*`` / ``gate.{i}.*`` are per-cell with i = task*domain_num + domain
    (model.py: ``output[:, t*self.domain_num+d]``), so domain = i % 3
  - ``*.domain_specific.{0,1}`` inside expert./task_expert./skip_conn/star_mlp is a
    *layer* index (Linear/LayerNorm), NOT a domain index -> shared
  - everything else (embedding, shared_*, expert.*, task_expert.*, _weight_*) -> shared

Pre-flight gate: for each state_type, F vs DH key-sets and per-tensor shapes must
match for EVERY (seed, E_common) pair, and (param only) the key universe must match
gradient_partition.json. Any failure marks the whole state_type not_applicable —
partial computation is forbidden by spec.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from s6r_common import (
    DatasetSpec,
    E_COMMON,
    SEEDS,
    delta_pair,
    load_raw,
    load_val_epoch,
    non_target_cells,
    parse_dataset,
    table_path,
    write_csv,
)

DOMAIN_NUM = 3  # domains 0,1,2; cell index i = task*DOMAIN_NUM + domain
PARTITIONS = ("shared", "target", "non_target", "all")


def _checkpoint_path(spec: DatasetSpec, arm: str, seed: int, epoch: int, state_type: str) -> Path:
    if state_type == "param":
        return Path(spec.run_root) / arm / f"seed{int(seed)}" / "joint" / f"common_epoch_{int(epoch)}.pth"
    # moment
    return Path(spec.run_root) / arm / f"seed{int(seed)}" / "joint" / f"moment_state_epoch{int(epoch)}.pth"


def _partition_json_path(spec: DatasetSpec) -> Path:
    return Path(spec.run_root) / "F" / f"seed{SEEDS[0]}" / "joint" / "gradient_partition.json"


def _key_domain(key: str) -> Optional[int]:
    """Domain index attributed to a state_dict key, or None if shared."""
    m = re.match(r"^(slot_weight|slot_bias)\.(\d+)$", key)
    if m:
        return int(m.group(2))
    m = re.match(r"^domain_expert\.(\d+)\.", key)
    if m:
        return int(m.group(1))
    m = re.match(r"^(tower|gate)\.(\d+)\.", key)
    if m:
        return int(m.group(2)) % DOMAIN_NUM
    return None


def _partition_of(key: str, target_d: int) -> str:
    d = _key_domain(key)
    if d is None:
        return "shared"
    return "target" if d == int(target_d) else "non_target"


def _flat_tensors(sd: Dict) -> Dict[str, List[torch.Tensor]]:
    """Normalize a loaded state dict to {key: [tensors]}.

    param checkpoints map key -> tensor; moment checkpoints map
    key -> {'exp_avg': t, 'exp_avg_sq': t, 'step': ...}. Non-tensor
    scalars (e.g. step counters) are skipped.
    """
    out: Dict[str, List[torch.Tensor]] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = [v]
        elif isinstance(v, dict):
            out[k] = [vv for _, vv in sorted(v.items()) if torch.is_tensor(vv)]
        else:
            out[k] = []
    return out


def _gate_pair(flat_a: Dict[str, List[torch.Tensor]], flat_b: Dict[str, List[torch.Tensor]]) -> Tuple[bool, Optional[str]]:
    if set(flat_a.keys()) != set(flat_b.keys()):
        return False, "key-set mismatch"
    for k in flat_a:
        ta, tb = flat_a[k], flat_b[k]
        if len(ta) != len(tb) or any(x.shape != y.shape for x, y in zip(ta, tb)):
            return False, f"shape mismatch at {k}"
    return True, None


def _sq_dist(flat_a: Dict[str, List[torch.Tensor]], flat_b: Dict[str, List[torch.Tensor]],
             partition: str, target_d: int) -> float:
    total = 0.0
    for k in flat_a:
        if partition != "all" and _partition_of(k, target_d) != partition:
            continue
        for x, y in zip(flat_a[k], flat_b[k]):
            diff = x.detach().to(torch.float64).flatten() - y.detach().to(torch.float64).flatten()
            total += float(torch.sum(diff * diff).item())
    return total


def _sq_norm(flat: Dict[str, List[torch.Tensor]]) -> float:
    total = 0.0
    for k in flat:
        for x in flat[k]:
            v = x.detach().to(torch.float64).flatten()
            total += float(torch.sum(v * v).item())
    return total


def _partition_counts(flat: Dict[str, List[torch.Tensor]], target_d: int) -> Dict[str, int]:
    counts = {"shared": 0, "target": 0, "non_target": 0}
    for k in flat:
        counts[_partition_of(k, target_d)] += 1
    return counts


def _json_key_universe(spec: DatasetSpec) -> Optional[set]:
    p = _partition_json_path(spec)
    if not p.exists():
        return None
    groups = json.loads(p.read_text(encoding="utf-8"))["groups"]
    keys: set = set()
    for g in groups.values():
        keys.update(g)
    return keys


def build_table11(spec: DatasetSpec, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_d = spec.target_domain
    # val non-target gain per seed/epoch (for the descriptive Spearman)
    val_rows = []
    for arm in ("F", "DH"):
        for seed in SEEDS:
            v = load_val_epoch(spec, arm, int(seed))
            val_rows.append(v)
    val_df = pd.concat(val_rows, ignore_index=True)
    d_val = delta_pair(val_df, "DH", "F", "validation_auc")
    nt = non_target_cells(spec)
    mask = d_val["domain"].astype(int).isin([t[0] for t in nt])
    nt_val_per_seed_epoch = (
        d_val[mask].groupby(["seed", "epoch"])["delta"].mean().reset_index()
    )

    json_keys = _json_key_universe(spec)

    for state_type in ("param", "moment"):
        # ---- pre-flight gate over EVERY (seed, epoch) pair before any math ----
        flats: Dict[Tuple[int, int, str], Dict[str, List[torch.Tensor]]] = {}
        gate_fail: Optional[str] = None
        for seed in SEEDS:
            for epoch in E_COMMON:
                sd_f = torch.load(_checkpoint_path(spec, "F", seed, epoch, state_type), map_location="cpu")
                sd_dh = torch.load(_checkpoint_path(spec, "DH", seed, epoch, state_type), map_location="cpu")
                flat_f, flat_dh = _flat_tensors(sd_f), _flat_tensors(sd_dh)
                ok, why = _gate_pair(flat_f, flat_dh)
                if not ok:
                    gate_fail = f"{state_type} F-vs-DH gate failed at seed{seed}/epoch{epoch}: {why}"
                    break
                if state_type == "param" and json_keys is not None:
                    if set(flat_f.keys()) != json_keys:
                        gate_fail = (
                            f"{state_type} key universe != gradient_partition.json at "
                            f"seed{seed}/epoch{epoch}"
                        )
                        break
                flats[(int(seed), int(epoch), "F")] = flat_f
                flats[(int(seed), int(epoch), "DH")] = flat_dh
            if gate_fail:
                break
        if gate_fail:
            rows.append({
                "dataset": spec.name, "seed": -1, "epoch": -1,
                "state_type": state_type, "partition": "all",
                "l2_distance": float("nan"), "normalized_l2": float("nan"),
                "spearman_with_nontarget_gain_val": float("nan"),
                "n_tensors": 0,
                "status": "not_applicable",
                "note": gate_fail,
            })
            continue

        counts = _partition_counts(flats[(int(SEEDS[0]), int(E_COMMON[0]), "F")], target_d)
        coverage_note = (
            "tensors per partition: "
            + ", ".join(f"{p}={counts[p]}" for p in ("shared", "target", "non_target"))
        )

        for seed in SEEDS:
            seed_rows = []
            for epoch in E_COMMON:
                flat_f = flats[(int(seed), int(epoch), "F")]
                flat_dh = flats[(int(seed), int(epoch), "DH")]
                # normalization denominator: ||theta^F||_2 over the full F state
                norm_F = float(np.sqrt(_sq_norm(flat_f)))
                for partition in PARTITIONS:
                    l2 = float(np.sqrt(_sq_dist(flat_dh, flat_f, partition, target_d)))
                    n2 = (l2 / norm_F) if norm_F > 0 else float("nan")
                    seed_rows.append({
                        "dataset": spec.name, "seed": int(seed), "epoch": int(epoch),
                        "state_type": state_type, "partition": partition,
                        "l2_distance": l2, "normalized_l2": n2,
                        "n_tensors": int(sum(counts.values())) if partition == "all" else int(counts.get(partition, 0)),
                        "status": "ok",
                        "note": coverage_note if epoch == E_COMMON[0] else "",
                    })
            seed_df = pd.DataFrame(seed_rows)
            for partition in PARTITIONS:
                # Spearman vs val non-target gain at the same 5 epochs (descriptive only)
                dist_series = seed_df[seed_df["partition"] == partition].set_index("epoch")["l2_distance"].reindex(list(E_COMMON))
                nt_series = (
                    nt_val_per_seed_epoch[nt_val_per_seed_epoch["seed"] == int(seed)]
                    .set_index("epoch")["delta"].reindex(list(E_COMMON))
                )
                rho, _ = spearmanr(dist_series.values, nt_series.values)
                rho = float(rho) if not np.isnan(rho) else float("nan")
                seed_rows.append({
                    "dataset": spec.name, "seed": int(seed), "epoch": -1,
                    "state_type": state_type, "partition": partition,
                    "l2_distance": float("nan"), "normalized_l2": float("nan"),
                    "spearman_with_nontarget_gain_val": rho,
                    "n_tensors": 0,
                    "status": "spearman",
                    "note": "descriptive only; never upgrades evidence strength",
                })
            rows.extend(seed_rows)
    return pd.DataFrame(rows).sort_values(["state_type", "partition", "seed", "epoch"]).reset_index(drop=True)


def main() -> None:
    spec = parse_dataset()
    raw = load_raw(spec)
    t11 = build_table11(spec, raw)
    write_csv(t11, table_path(spec, "s6r_param_distance"))
    print(f"{spec.name}: table 11 written ({len(t11)})")


if __name__ == "__main__":
    main()
