import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_F0_data import load_movielens_natural, make_sdmt_view


def _write_movie_csv(path, rows=120):
    ages = (18, 25, 35)
    frame = pd.DataFrame({
        "user_id": np.arange(rows) + 1,
        "movie_id": np.arange(rows) % 17 + 1,
        "gender": np.where(np.arange(rows) % 2, "M", "F"),
        "age": [ages[i % 3] for i in range(rows)],
        "occupation": np.arange(rows) % 4,
        "zip": [str(10000 + i % 11) for i in range(rows)],
        "genres": ["Drama|Comedy" if i % 2 else "Action|Drama" for i in range(rows)],
        "rating": np.arange(rows) % 5 + 1,
    })
    frame.to_csv(path, index=False)


def test_natural_global_pool_and_sdmt_preserve_global_schema(tmp_path):
    path = tmp_path / "ml-1m.csv"
    _write_movie_csv(path)
    joint = load_movielens_natural(str(path))
    assert len(joint.train_x) == 96
    assert len(joint.val_x) == 12
    assert len(joint.test_x) == 12
    assert joint.preprocessing_metadata["global_base_train_size"] == 96
    assert joint.features.vocab_sizes["domain_indicator"] == 3
    np.testing.assert_array_equal(joint.train_row_ids, np.arange(96))
    np.testing.assert_array_equal(joint.train_y[:, 0], (np.arange(96) % 5 + 1 > 3).astype(np.float32))
    np.testing.assert_array_equal(joint.train_y[:, 1], (np.arange(96) % 5 + 1 > 4).astype(np.float32))

    view = make_sdmt_view(joint, 1)
    assert view.domain_num == 1
    assert view.local_to_physical == (1,)
    assert view.features.vocab_sizes == joint.features.vocab_sizes
    assert set(np.unique(view.train_x[:, -1])) == {0.0}
    assert set(view.train_row_ids).issubset(set(joint.train_row_ids))
    assert view.preprocessing_metadata["global_embedding_schema_preserved"] is True
