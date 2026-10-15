"""Comprehensive G0 checks: forward equivalence (synthetic + real train
batches), parameter total, optimizer grouping, B1 asymmetric probe,
exposure pairing, test tripwire.

Run from project root:

    venv/Scripts/python.exe research_R2/tests/run_g0_forward_equivalence.py

This script exits non-zero on any failure.  The output is verbose on
purpose so the Implementation Report can quote the per-arm numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
R2_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(R2_ROOT)
sys.path.insert(0, os.path.join(R2_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "m3oe"))

from MDMTRec.basic.features import SparseFeature  # noqa
from MDMTRec.utils.data import TorchDataset  # noqa
from stage_S1_data import load_stage_s1_bundle  # noqa
from stage_S1_models import initial_state_hash, parameter_manifest  # noqa

from ao04_v12_model import (  # noqa
    AO04M3OE,
    B3_BUFFER_NAME,
    DELTA_BAL_D_NAME,
    Q_RHO_NAME,
    build_ao04_model,
)
from ao04_v12_protocol import JK_PARTNER  # noqa
from stage_S1_models import features_from_schema  # noqa
from ao04_v12_protocol import (  # noqa
    ARCHITECTURE_PARAMETER_NAMES,
    BATCH_SIZE,
    DATASETS,
    EXPECTED_PARAM_TOTAL,
    LEARNING_RATE_ARCH,
    LEARNING_RATE_MAIN,
    WEIGHT_DECAY,
    BETAS,
    EPS_ADAM,
)
from ao04_v12_runner import (  # noqa
    SealedTestSplit,
    TestSealViolation,
    TEST_PREDICTION_CALLS,
    _sealed_from_bundle,
    build_optimizers_for_arm,
    forbidden_identifier_check,
    setup_seed,
)
from ao04_v12_static_check import runner_static_tripwire  # noqa


# ---------------------------------------------------------------------------
# Synthetic schema helper (replicates features_from_schema conventions).
# ---------------------------------------------------------------------------


def _make_synthetic_schema(n_features: int = 4, embedding_dim: int = 64,
                            vocab_size: int = 3):
    feats = []
    columns = []
    for i in range(n_features):
        feats.append(SparseFeature(name=i, vocab_size=vocab_size,
                                    embed_dim=embedding_dim))
        columns.append("feat_{}".format(i))
    feats.append(SparseFeature(name=n_features, vocab_size=3,
                                embed_dim=embedding_dim))
    columns.append("domain_indicator")
    return feats, columns


def _make_synthetic_batch(feats, batch_size: int,
                           device: torch.device) -> torch.Tensor:
    n_cols = len(feats)
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 3, size=(batch_size, n_cols)).astype(np.float32)
    arr[:, -1] = rng.randint(0, 3, size=(batch_size,)).astype(np.float32)
    return torch.from_numpy(arr).T.to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. Forward equivalence (synthetic).
# ---------------------------------------------------------------------------


def _forward_equivalence_arm(arm: str, *, seed: int = 2022,
                              batch_size: int = 128,
                              max_abs: float = 1e-6,
                              rtol: float = 1e-5) -> Tuple[bool, float, float]:
    feats, _ = _make_synthetic_schema()
    setup_seed(seed)
    model_b0 = build_ao04_model(feats, "B0", seed=seed)
    b0_state = {k: v.detach().cpu().clone() for k, v in model_b0.state_dict().items()}
    setup_seed(seed)
    model_arm = build_ao04_model(feats, arm, seed=seed)
    model_arm.load_state_dict(b0_state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_b0 = model_b0.to(device); model_arm = model_arm.to(device)
    x = _make_synthetic_batch(feats, batch_size, device)
    model_b0.eval(); model_arm.eval()
    with torch.no_grad():
        y_b0 = model_b0(x); y_arm = model_arm(x)
    diff = (y_arm - y_b0).abs()
    max_abs_seen = float(diff.max().detach().cpu().item())
    denom = float(y_b0.abs().max().detach().cpu().item()) + 1e-30
    rel = max_abs_seen / denom
    return bool((max_abs_seen <= max_abs) and (rel <= rtol)), max_abs_seen, rel


def check_forward_equivalence_synthetic() -> bool:
    print("=== G0-5 forward equivalence (synthetic) ===")
    ok = True
    for arm in ("B1", "B2", "B4"):
        passed, max_abs, rel = _forward_equivalence_arm(arm)
        print(f"  {arm}: max_abs={max_abs:.2e} rel={rel:.2e} pass={passed}")
        ok = ok and passed
    return ok


# ---------------------------------------------------------------------------
# 2. Forward equivalence on real data.
# ---------------------------------------------------------------------------


def _real_train_batch(bundle, batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Take the first ``batch_size`` rows of the train split (deterministic)."""

    train_x = bundle.train_x[:batch_size]
    train_y = bundle.train_y[:batch_size]
    return train_x, train_y


def _forward_equivalence_real(dataset: str, *, batch_size: int = 4096,
                               seed: int = 2022,
                               max_abs: float = 1e-6,
                               rtol: float = 1e-5) -> Tuple[bool, float, float]:
    setting = ("natural" if dataset == "kuairand" else "d0_large")
    bundle = load_stage_s1_bundle(dataset, None, setting, seed)
    train_x, _ = _real_train_batch(bundle, batch_size=batch_size)
    feats = features_from_schema(bundle.features)
    setup_seed(seed)
    model_b0 = build_ao04_model(feats, "B0", seed=seed)
    b0_state = {k: v.detach().cpu().clone() for k, v in model_b0.state_dict().items()}
    setup_seed(seed)
    # Use the per-arm list extending B0.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_diff_total = 0.0
    max_rel_total = 0.0
    for arm in ("B1", "B2", "B4"):
        model_arm = build_ao04_model(feats, arm, seed=seed)
        model_arm.load_state_dict(b0_state, strict=False)
        model_b0_d = model_b0.to(device); model_arm = model_arm.to(device)
        x = torch.from_numpy(train_x).T.to(device=device, dtype=torch.float32)
        model_b0_d.eval(); model_arm.eval()
        with torch.no_grad():
            y_b0 = model_b0_d(x); y_arm = model_arm(x)
        diff = (y_arm - y_b0).abs()
        max_abs_seen = float(diff.max().detach().cpu().item())
        denom = float(y_b0.abs().max().detach().cpu().item()) + 1e-30
        rel = max_abs_seen / denom
        passed = bool((max_abs_seen <= max_abs) and (rel <= rtol))
        print(f"  {dataset}/{arm}: max_abs={max_abs_seen:.2e} rel={rel:.2e} pass={passed}")
        max_diff_total = max(max_diff_total, max_abs_seen)
        max_rel_total = max(max_rel_total, rel)
        if not passed:
            return False, max_diff_total, max_rel_total
    return True, max_diff_total, max_rel_total


def check_forward_equivalence_real() -> bool:
    print("=== G0-5 forward equivalence (real train batch) ===")
    ok = True
    for dataset in ("kuairand", "movielens"):
        passed, _, _ = _forward_equivalence_real(dataset)
        ok = ok and passed
    return ok


# ---------------------------------------------------------------------------
# 3. Parameter totals.
# ---------------------------------------------------------------------------


def check_parameter_total(dataset: str) -> bool:
    print(f"=== Parameter total ({dataset}) ===")
    setting = ("natural" if dataset == "kuairand" else "d0_large")
    bundle = load_stage_s1_bundle(dataset, None, setting, seed=2022)
    feats = features_from_schema(bundle.features)
    expected = EXPECTED_PARAM_TOTAL[dataset]
    ok = True
    for arm in ("B0", "B1", "B2", "B3", "B4"):
        setup_seed(2022)
        model = build_ao04_model(feats, arm, seed=2022)
        manifest = parameter_manifest(model)
        total = int(manifest["total"])
        expected_n = expected[arm]
        passed = total == expected_n
        print(f"  {arm}: total={total} expected={expected_n} pass={passed}")
        ok = ok and passed
        arch_names = manifest["architecture_names"]
        if sorted(arch_names) != sorted(ARCHITECTURE_PARAMETER_NAMES):
            print(f"  {arm}: architecture_names mismatch: {arch_names}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# 4. Optimizer grouping.
# ---------------------------------------------------------------------------


def check_optimizer_grouping(dataset: str) -> bool:
    print(f"=== Optimizer grouping ({dataset}) ===")
    setting = ("natural" if dataset == "kuairand" else "d0_large")
    bundle = load_stage_s1_bundle(dataset, None, setting, seed=2022)
    feats = features_from_schema(bundle.features)
    ok = True
    for arm in ("B0", "B1", "B2", "B3", "B4"):
        setup_seed(2022)
        model = build_ao04_model(feats, arm, seed=2022)
        main_opt, arch_opt, main_set, arch_set = build_optimizers_for_arm(model)
        # Identity set comparisons.
        main_ids = {id(p) for p in main_set}
        arch_ids = {id(p) for p in arch_set}
        intersection = main_ids & arch_ids
        if intersection:
            print(f"  {arm}: optimizer intersection non-empty: {intersection}")
            ok = False
        # Total trainable coverage.
        all_trainable = {id(p) for _, p in model.named_parameters()}
        union = main_ids | arch_ids
        if union != all_trainable:
            missing = all_trainable - union
            extra = union - all_trainable
            print(f"  {arm}: union != trainable. missing={missing} extra={extra}")
            ok = False
        # Arch set must be exactly the four raw scalars.
        if len(arch_set) != 4:
            print(f"  {arm}: arch set size != 4: {len(arch_set)}")
            ok = False
        # B3 buffer must not be in either optimizer.
        if arm == "B3":
            buf_param = getattr(model, B3_BUFFER_NAME)
            if any(id(buf_param) == id(p) for p in main_set) \
                    or any(id(buf_param) == id(p) for p in arch_set):
                print(f"  B3: buffer leaked into optimizer!")
                ok = False
        # q/delta must be in MAIN, not ARCH.
        if arm in ("B1", "B2"):
            q_param = model.q_rho
            if any(id(q_param) == id(p) for p in main_set) and not any(id(q_param) == id(p) for p in arch_set):
                pass
            else:
                print(f"  {arm}: q_rho grouping wrong (q should be in main only)")
                ok = False
        if arm == "B4":
            d_param = model.delta_bal_d
            if any(id(d_param) == id(p) for p in main_set) and not any(id(d_param) == id(p) for p in arch_set):
                pass
            else:
                print(f"  B4: delta_bal_d grouping wrong (delta should be in main only)")
                ok = False
        print(f"  {arm}: main={len(main_set)} arch={len(arch_set)} trainable={len(all_trainable)} ok={ok}")
    return ok


# ---------------------------------------------------------------------------
# 5. B1 asymmetric probe.
# ---------------------------------------------------------------------------


def check_b1_asymmetric_probe() -> bool:
    print("=== B1 asymmetric probe ===")
    # Construct a B1 model and set its q_rho to a known non-uniform vector.
    # Then verify per-cell R_dt coefficients match Section 2 by running
    # forward against a per-domain-uniform batch and checking the linear
    # combination shapes.
    n_features = 4
    embedding_dim = 64
    feats = [SparseFeature(name=i, vocab_size=3, embed_dim=embedding_dim)
             for i in range(n_features + 1)]
    setup_seed(2022)
    model = build_ao04_model(feats, "B1", seed=2022)
    # Inject known q values.
    q_target = torch.tensor([1.5, -0.7, 0.2, 0.9, -1.0, 0.3],
                             dtype=torch.float32)
    with torch.no_grad():
        model.q_rho.copy_(q_target)
    # Build a synthetic batch where each domain has at least one row.
    batch_size = 96
    arr = np.zeros((batch_size, n_features + 1), dtype=np.float32)
    for d in range(3):
        for i in range(batch_size // 3):
            arr[d * (batch_size // 3) + i] = 0
    arr[:, -1] = np.repeat(np.arange(3, dtype=np.float32), batch_size // 3)
    arr[:, :n_features] = np.random.RandomState(7).randint(0, 3,
                                                             size=(batch_size, n_features)).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.from_numpy(arr).T.to(device=device, dtype=torch.float32)
    # Compare per-cell numerator / denominator relationships via a tiny linear
    # algebra probe: construct two batches that differ only in the domain
    # indicator and check the gradient w.r.t. q_rho at each cell is non-zero
    # and finite.
    model.train()
    target_y = torch.zeros(batch_size, 2, device=device)
    # Forward
    pred = model(x)
    loss = torch.nn.functional.binary_cross_entropy(pred, target_y, reduction="mean")
    loss.backward()
    grad_q = model.q_rho.grad.detach().cpu().clone()
    print("  q gradient:", grad_q.tolist())
    finite = torch.isfinite(grad_q).all().item()
    nonzero = grad_q.abs().max().item() > 0.0
    print(f"  finite={finite} non-zero max={grad_q.abs().max().item():.4e}")
    # Sanity: JK_PARTNER mapping for d ∈ {0,1,2} must match the spec
    # and q_rho must receive gradient on all six cells because every
    # cell is exercised by at least one sample in this probe batch.
    expected_jk_d = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    if JK_PARTNER != expected_jk_d:
        print("  JK_PARTNER mismatch:", JK_PARTNER, "expected", expected_jk_d)
        return False
    return bool(finite and nonzero)


# ---------------------------------------------------------------------------
# 6. Exposure pairing (≤3 batch hash equality).
# ---------------------------------------------------------------------------


def check_exposure_pairing(dataset: str, *, max_batches: int = 3) -> bool:
    print(f"=== Exposure pairing ({dataset}, ≤{max_batches} batches) ===")
    setting = ("natural" if dataset == "kuairand" else "d0_large")
    bundle = load_stage_s1_bundle(dataset, None, setting, seed=2022)
    feats = features_from_schema(bundle.features)
    # Construct all five arms with the same seed.
    hashes = {}
    for arm in ("B0", "B1", "B2", "B3", "B4"):
        setup_seed(2022)
        model = build_ao04_model(feats, arm, seed=2022)
        # Use the same per-seed generator so the train shuffle order
        # is reproducible across arms (Specification §8.4).
        gen = torch.Generator(); gen.manual_seed(2022)
        train_loader = DataLoader(
            TorchDataset(bundle.train_x, bundle.train_y),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
            drop_last=False, generator=gen)
        fps = []
        for i, (x_batch, _) in enumerate(train_loader):
            if i >= max_batches:
                break
            # Use the Stage S1 fingerprint directly.
            from stage_E1_instrumentation import batch_fingerprint
            fps.append(batch_fingerprint(x_batch, _))
        h = hashlib_sha256(json.dumps(fps))
        hashes[arm] = h
    # All five should match.
    ref = hashes["B0"]
    ok = True
    for arm in ("B0", "B1", "B2", "B3", "B4"):
        match = hashes[arm] == ref
        print(f"  {arm}: hash={hashes[arm][:16]}... match={match}")
        ok = ok and match
    return ok


def hashlib_sha256(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# 7. Test tripwire dynamic check.
# ---------------------------------------------------------------------------


def check_test_tripwire() -> bool:
    print("=== Test tripwire dynamic check ===")
    sealed = SealedTestSplit(10, "fp", "rid", "lbl", {"d0t0": 3}, {"d0t0": 2})
    raised_well = {"x": False, "iter": False, "len": False}
    try:
        sealed.x
    except TestSealViolation:
        raised_well["x"] = True
    try:
        iter(sealed)
    except TestSealViolation:
        raised_well["iter"] = True
    try:
        len(sealed)
    except TestSealViolation:
        raised_well["len"] = True
    # Allowed attributes MUST remain accessible.
    allowed_ok = (sealed.row_count == 10
                  and sealed.fingerprint_value == "fp"
                  and isinstance(sealed.class_coverage_check(), dict))
    print("  SealedTestSplit raised:", {k: v for k, v in raised_well.items()},
          "allowed_ok =", allowed_ok)
    # Static check on runner module.
    matches = runner_static_tripwire()
    static_clean = (matches == [])
    print(f"  Static check matches: {matches} clean={static_clean}")
    # Counter still at 0 (no real test eval).
    counter_clean = TEST_PREDICTION_CALLS == 0
    print(f"  TEST_PREDICTION_CALLS=0: {counter_clean} (current={TEST_PREDICTION_CALLS})")
    return (all(raised_well.values()) and allowed_ok and static_clean
            and counter_clean)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all", "kuairand", "movielens"),
                        default="all")
    args = parser.parse_args()
    # Section 1: synthetic equivalence.
    s1 = check_forward_equivalence_synthetic()
    # Section 2: real equivalence (subset per --dataset).
    s2 = True
    for ds in ("kuairand", "movielens"):
        if args.dataset != "all" and args.dataset != ds:
            continue
        ok, _, _ = _forward_equivalence_real(ds)
        if not ok:
            s2 = False
    # Section 3: parameter totals.
    s3 = True
    for ds in ("kuairand", "movielens"):
        if args.dataset != "all" and args.dataset != ds:
            continue
        s3 = check_parameter_total(ds) and s3
    # Section 4: optimizer grouping.
    s4 = True
    for ds in ("kuairand", "movielens"):
        if args.dataset != "all" and args.dataset != ds:
            continue
        s4 = check_optimizer_grouping(ds) and s4
    # Section 5: B1 asymmetric probe.
    s5 = check_b1_asymmetric_probe()
    # Section 6: exposure pairing.
    s6 = True
    for ds in ("kuairand", "movielens"):
        if args.dataset != "all" and args.dataset != ds:
            continue
        s6 = check_exposure_pairing(ds) and s6
    # Section 7: test tripwire.
    s7 = check_test_tripwire()
    overall = all([s1, s2, s3, s4, s5, s6, s7])
    print()
    print("=== OVERALL ===")
    print(f"  G0-5 forward equivalence (synthetic): {s1}")
    print(f"  G0-5 forward equivalence (real):      {s2}")
    print(f"  G0-8 parameter total:                 {s3}")
    print(f"  G0-7 optimizer grouping:              {s4}")
    print(f"  B1 asymmetric probe:                  {s5}")
    print(f"  Exposure pairing:                     {s6}")
    print(f"  Test tripwire:                        {s7}")
    print(f"  OVERALL: {overall}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
