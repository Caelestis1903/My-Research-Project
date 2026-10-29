import copy
import random

import numpy as np
import pytest
import torch

from experiments.mdmt_sota_baselines_2526.config import load_config
from experiments.mdmt_sota_baselines_2526.data import Data, tensor
from experiments.mdmt_sota_baselines_2526.models import build_model


torch.set_num_threads(1)


def _prepared_fixture(tmp_path, dataset="movielens"):
    root = tmp_path / dataset
    root.mkdir()
    n = 64
    x = np.column_stack((np.arange(n) % 8 + 1, np.arange(n) % 9 + 1,
                         np.arange(n) % 3)).astype(np.float32)
    y = np.column_stack((np.arange(n) % 2, (np.arange(n) // 2) % 2)).astype(np.float32)
    np.save(root / "x.npy", x, allow_pickle=False)
    np.save(root / "y.npy", y, allow_pickle=False)
    components = ("fit", "P_label", "P_cal", "P_update", "Q_explore", "validation", "test", "Q_confirm")
    for name, ids in zip(components, np.array_split(np.arange(n, dtype=np.int64), len(components))):
        np.save(root / f"rows_{name}.npy", ids, allow_pickle=False)
    (root / "DATA_MANIFEST.json").write_text(
        '{"status":"READY_DIRECT_VALUE_CHECKS","dataset":"movielens",'
        '"protocol":"MDMT-SOTA-2526-v1","features":{"columns":["user_id",'
        '"movie_id","domain_indicator"],"dense_columns":[],"sparse_columns":['
        '"user_id","movie_id","domain_indicator"],"vocab_sizes":{"user_id":9,'
        '"movie_id":10,"domain_indicator":3},"embedding_dim":16}}', encoding="utf-8")
    return root, x, y


def _numpy_state_equal(left, right):
    return (left[0] == right[0] and np.array_equal(left[1], right[1])
            and left[2:] == right[2:])


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_tree_equal(first, second)
    else:
        assert left == right


def test_data_constructor_uses_readonly_ram_x_and_keeps_y_lazy_and_role_locked(tmp_path, monkeypatch):
    root, source_x, _ = _prepared_fixture(tmp_path)
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = torch.get_rng_state().clone()
    data = Data("movielens", tmp_path)
    assert type(data.x) is np.ndarray and not isinstance(data.x, np.memmap)
    assert data.x.base is None and not data.x.flags.writeable
    assert data.x.dtype == np.dtype("float32") and data.x.shape == source_x.shape
    assert np.array_equal(data.x, source_x)
    assert isinstance(data.y, np.memmap)
    assert data.y.flags.writeable is False
    assert random.getstate() == python_state
    assert _numpy_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    with pytest.raises(PermissionError):
        data.labels("E", "final")
    with pytest.raises(PermissionError):
        data.labels("S", "transaction")
    with pytest.raises(PermissionError):
        data.labels("legacy", "development")

    class LazyY:
        shape = (len(source_x), 2)
        indexed = 0

        def __len__(self):
            return len(source_x)

        def __getitem__(self, item):
            self.indexed += 1
            raise AssertionError("Data constructor must not index labels")

    original_load = np.load
    lazy_y = LazyY()

    def load_spy(path, *args, **kwargs):
        if str(path).endswith("y.npy"):
            return lazy_y
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", load_spy)
    Data("movielens", tmp_path)
    assert lazy_y.indexed == 0
    with pytest.raises(PermissionError):
        Data("movielens", tmp_path).labels("E", "final")
    assert lazy_y.indexed == 0


def test_m3oe_logits_loss_adam_update_gradients_and_rng_match_memmap_input(tmp_path):
    root, _, source_y = _prepared_fixture(tmp_path)
    data = Data("movielens", tmp_path)
    ids = data.train_ids[:8]
    mapped_x = np.load(root / "x.npy", mmap_mode="r", allow_pickle=False)
    r1_batch = np.asarray(mapped_x[ids])
    r2_batch, r2_labels, r2_mask = data.training_batch(ids)
    labels = np.asarray(data.y[ids])
    mask = np.ones_like(labels, dtype=bool)
    assert np.array_equal(r1_batch, r2_batch)
    assert np.array_equal(labels, r2_labels) and np.array_equal(labels, source_y[ids])
    assert np.array_equal(mask, r2_mask)

    schema = {
        "dataset": "movielens", "columns": ["user_id", "movie_id", "domain_indicator"],
        "dense_columns": [], "sparse_columns": ["user_id", "movie_id", "domain_indicator"],
        "vocab_sizes": {"user_id": 9, "movie_id": 10, "domain_indicator": 3}, "embedding_dim": 16,
    }
    cfg = load_config()
    model_config = {**cfg["models"]["M3OE_FIXED1"], "dataset": "movielens", "seed": 2022}
    torch.manual_seed(24)
    r1_model = build_model("M3OE_FIXED1", schema, model_config, cfg["source_roots"])
    r2_model = copy.deepcopy(r1_model)
    optimizer_config = cfg["models"]["M3OE_FIXED1"]
    make_optimizer = lambda model: torch.optim.Adam(
        model.parameters(), lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"], betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["eps"])
    r1_optimizer, r2_optimizer = make_optimizer(r1_model), make_optimizer(r2_model)

    python_rng = random.getstate()
    numpy_rng = copy.deepcopy(np.random.get_state())
    torch_rng = torch.get_rng_state().clone()

    def step(model, optimizer, batch):
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        x = tensor(batch, "cpu")
        y = tensor(labels, "cpu")
        valid = torch.as_tensor(mask, dtype=torch.bool)
        logits = model(x, domain=x[:, -1].long())
        objective = model.objective(x, y, valid)
        objective["loss"].backward()
        gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                     for name, parameter in model.named_parameters()}
        optimizer.step()
        return {
            "logits": logits.detach().clone(), "loss": objective["loss"].detach().clone(),
            "gradients": gradients,
            "parameters": {name: value.detach().clone() for name, value in model.state_dict().items()},
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "python_rng": random.getstate(), "numpy_rng": copy.deepcopy(np.random.get_state()),
            "torch_rng": torch.get_rng_state().clone(),
        }

    r1 = step(r1_model, r1_optimizer, r1_batch)
    r2 = step(r2_model, r2_optimizer, r2_batch)
    assert torch.equal(r1["logits"], r2["logits"])
    assert torch.equal(r1["loss"], r2["loss"])
    _assert_tree_equal(r1["gradients"], r2["gradients"])
    _assert_tree_equal(r1["parameters"], r2["parameters"])
    _assert_tree_equal(r1["optimizer"], r2["optimizer"])
    assert r1["python_rng"] == r2["python_rng"]
    assert _numpy_state_equal(r1["numpy_rng"], r2["numpy_rng"])
    assert torch.equal(r1["torch_rng"], r2["torch_rng"])
