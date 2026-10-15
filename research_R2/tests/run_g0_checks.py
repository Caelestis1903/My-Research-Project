"""Helper script that performs G0 checks on a small synthetic batch.

Usage (from the project root):

    venv/Scripts/python.exe research_R2/tests/run_g0_checks.py

This script is intentionally self-contained so it can be re-run by a
reviewer without depending on the full Stage R2 runner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import types
from typing import Dict, List, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R2_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(R2_ROOT)
sys.path.insert(0, os.path.join(R2_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "m3oe"))

from MDMTRec.basic.features import SparseFeature  # noqa

from stage_S1_data import FeatureSchema, load_stage_s1_bundle  # noqa
from stage_S1_models import initial_state_hash  # noqa

from ao04_v12_model import build_ao04_model, AO04M3OE  # noqa
from ao04_v12_protocol import EXPECTED_PARAM_TOTAL  # noqa
from ao04_v12_runner import build_optimizers_for_arm, setup_seed  # noqa


def _make_synthetic_schema(n_sparse: int = 4, vocab_size: int = 8,
                            embed_dim: int = 4) -> FeatureSchema:
    """A tiny synthetic schema that mirrors the M3OE column conventions."""

    columns = ["feat_{}".format(i) for i in range(n_sparse)] + ["domain_indicator"]
    vocab_sizes = {"feat_{}".format(i): vocab_size for i in range(n_sparse)}
    vocab_sizes["domain_indicator"] = 3
    return FeatureSchema(
        dataset="synthetic",
        columns=tuple(columns),
        dense_columns=(),
        sparse_columns=tuple(columns),
        vocab_sizes=vocab_sizes,
        embedding_dim=embed_dim,
    )


def _make_features(schema: FeatureSchema):
    """Real SparseFeature descriptors (no DenseFeatures in synthetic).

    Per ``stage_S1_models.features_from_schema``, the column NAMES are
    the integer column indices — i.e. ``SparseFeature`` is keyed by
    column index so that the forward can call ``x[k]`` against the
    transposed tensor.
    """

    feats = []
    for index, column in enumerate(schema.columns):
        feats.append(SparseFeature(
            name=index, vocab_size=int(schema.vocab_sizes[column]),
            embed_dim=int(schema.embedding_dim)))
    return feats


def _make_synthetic_batch(schema: FeatureSchema, batch_size: int,
                            device: torch.device) -> torch.Tensor:
    """Generate one synthetic domain-routed batch with deterministic contents."""

    n_cols = len(schema.columns)
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 3, size=(batch_size, n_cols)).astype(np.float32)
    arr[:, -1] = rng.randint(0, 3, size=(batch_size,))
    arr[:, -1] = arr[:, -1].astype(np.float32)
    return torch.from_numpy(arr).T.to(device=device, dtype=torch.float32)


def forward_equivalence_check(arm: str, *, seed: int = 2022,
                               batch_size: int = 128,
                               max_abs: float = 1e-6,
                               rtol: float = 1e-5) -> Tuple[bool, float, float]:
    """Compare B<arm> zero-init eval-mode output to B0 (same seed)."""

    schema = _make_synthetic_schema()
    feats = _make_features(schema)
    setup_seed(seed)
    model_b0 = build_ao04_model(feats, "B0", seed=seed)
    b0_state = {k: v.detach().cpu().clone() for k, v in model_b0.state_dict().items()}

    setup_seed(seed)
    model_arm = build_ao04_model(feats, arm, seed=seed)
    result = model_arm.load_state_dict(b0_state, strict=False)
    print(f"  {arm}: load_state_dict missing={result.missing_keys} "
          f"unexpected={result.unexpected_keys}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_b0 = model_b0.to(device)
    model_arm = model_arm.to(device)
    x = _make_synthetic_batch(schema, batch_size, device)
    model_b0.eval(); model_arm.eval()
    with torch.no_grad():
        y_b0 = model_b0(x)
        y_arm = model_arm(x)
    diff = (y_arm - y_b0).abs()
    max_abs_seen = float(diff.max().detach().cpu().item())
    denom = float(y_b0.abs().max().detach().cpu().item()) + 1e-30
    rel = max_abs_seen / denom
    passed = bool((max_abs_seen <= max_abs) and (rel <= rtol))
    print(f"  {arm} max_abs={max_abs_seen:.2e} rel={rel:.2e} pass={passed}")
    return passed, max_abs_seen, rel


def main() -> int:
    print("=== G0-5 forward equivalence check ===")
    for arm in ("B1", "B2", "B4"):
        ok, max_abs, rel = forward_equivalence_check(arm)
        if not ok:
            print("  FAIL for arm", arm, "max_abs=", max_abs, "rel=", rel)
            return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
