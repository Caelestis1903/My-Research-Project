import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_E4_orchestrator import build_oracle_tasks, build_tasks  # noqa: E402
from stage_E4_weighting import (  # noqa: E402
    DatasetBundle,
    EXPECTED_SEEDS,
    TASK_NUM,
    WEIGHTING_KEYS,
    filter_split_first_domain,
    split_row_fingerprint,
)


def _tiny_bundle():
    # Deliberately non-random ordered rows make a pre-filter split easy to
    # distinguish from an invalid filter-then-split implementation.
    x = np.asarray([
        [10, 0], [11, 1], [12, 2], [13, 1], [14, 0], [15, 2],
        [16, 1], [17, 0], [18, 2], [19, 1],
    ], dtype=np.int64)
    y = np.asarray([[i % 2, (i + 1) % 2] for i in range(len(x))], dtype=np.float32)
    train_end, val_end = 8, 9
    splits = {
        "train": (x[:train_end], y[:train_end]),
        "validation": (x[train_end:val_end], y[train_end:val_end]),
        "test": (x[val_end:], y[val_end:]),
    }
    metadata = {
        "split_row_fingerprints": {k: split_row_fingerprint(*v) for k, v in splits.items()},
        "split_counts": {k: {d: int((v[0][:, -1] == d).sum()) for d in range(3)} for k, v in splits.items()},
    }
    return DatasetBundle([], [], x[:train_end], y[:train_end], x[train_end:val_end], y[train_end:val_end],
                         x[val_end:], y[val_end:], len(x), "tiny.csv", "tiny.csv", {}, metadata)


def test_oracle_filters_after_global_split_and_remaps_local_domain():
    bundle = _tiny_bundle()
    filtered, info = filter_split_first_domain(bundle, 1)
    # Global split was made at rows [0:8], [8:9], [9:].  Domain-1 rows are
    # therefore train rows 1,3,6 and test row 9; validation has none.
    assert filtered["train"][0][:, 0].tolist() == [11, 13, 16]
    assert filtered["validation"][0].shape[0] == 0
    assert filtered["test"][0][:, 0].tolist() == [19]
    assert np.all(filtered["train"][0][:, -1] == 0)
    assert info["filter_after_global_split"] is True
    assert info["local_domain_mapping"] == {"1": 0}
    assert info["split_row_fingerprints"]["train"] == split_row_fingerprint(*filtered["train"])


def test_formal_queue_has_exact_12_joint_and_12_oracle_paths(tmp_path):
    joint = build_tasks(str(tmp_path / "raw"))
    oracle = build_oracle_tasks(str(tmp_path / "raw" / "oracle_reference"))
    assert len(joint) == len(WEIGHTING_KEYS) * len(EXPECTED_SEEDS) == 12
    assert len(oracle) == 3 * len(EXPECTED_SEEDS) == 12
    assert {(row["weighting"], row["seed"]) for row in joint} == {
        (weighting, seed) for weighting in WEIGHTING_KEYS for seed in EXPECTED_SEEDS
    }
    assert {(row["original_physical_domain"], row["seed"]) for row in oracle} == {
        (domain, seed) for domain in range(3) for seed in EXPECTED_SEEDS
    }
