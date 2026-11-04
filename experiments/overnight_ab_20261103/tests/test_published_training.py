"""Small real-adapter fixtures for RNG, optimizer and AutoIFS stage recovery."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.overnight_ab_20261103 import published_training as training
from experiments.overnight_ab_20261103.published_models import build_published
from experiments.overnight_ab_20261103.state import state_digest


torch.set_num_threads(1)


@pytest.fixture
def bundle():
    schema = dict(dataset="movielens", columns=["user_id", "age", "movie_id", "domain_indicator"],
                  dense_columns=["age"], sparse_columns=["user_id", "movie_id", "domain_indicator"],
                  vocab_sizes={"user_id": 7, "movie_id": 9, "domain_indicator": 3})
    x = np.array([[1 + i % 6, i / 20, 1 + i % 8, i // 4] for i in range(12)], dtype=np.float32)
    y = np.array([[i % 2, (i // 2) % 2] for i in range(12)], dtype=np.float32)
    return SimpleNamespace(features=SimpleNamespace(as_dict=lambda: schema),
                           train_x=x, train_y=y, train_row_ids=np.arange(12),
                           val_x=x.copy(), val_y=y.copy(), val_row_ids=np.arange(12))


def job(method):
    root = Path(__file__).resolve().parents[3]
    protocol = json.loads((root / "research_plans/26-11-03_overnight_AB/protocol.json").read_text())
    cfg = copy.deepcopy(protocol[method.lower()])
    if method == "AUTOIFS":
        cfg["rank"] = protocol["datasets"]["movielens"]["autoifs_rank"]
    return dict(job_id=f"movielens.{method}.s2022", trajectory_id="fixture",
                config_hash=method, source_hash="fixture", data_hash="fixture", method=method,
                arm=method, dataset="movielens", seed=2022, config=cfg, protocol=protocol)


def scientific(snapshot):
    return {key: snapshot[key] for key in ("model_state", "optimizer_states", "model_meta", "rng",
                                          "stage", "next_stage_epoch", "completed_search", "completed_final",
                                          "commit_index", "global_step", "selection", "snapshots")}


@pytest.mark.parametrize("method", ["HYPERGATE", "AUTOIFS"])
def test_real_adapter_cpu_objective_and_two_epoch_ram_continuation(bundle, method):
    specification = job(method)
    uninterrupted = training.create_session(specification, bundle, "cpu")
    first = training.advance_epoch(uninterrupted)
    frozen = training.export_session(uninterrupted)
    second = training.advance_epoch(uninterrupted)
    expected = training.export_session(uninterrupted)
    resumed = training.restore_session(specification, bundle, "cpu", frozen)
    resumed_second = training.advance_epoch(resumed)
    actual = training.export_session(resumed)
    assert state_digest(scientific(actual)) == state_digest(scientific(expected))
    assert resumed_second["data_order_hash"] == second["data_order_hash"]
    assert first["cost"]["sample_exposures"] == len(bundle.train_x)
    assert first["cost"]["label_exposures"] == 2 * len(bundle.train_x)
    expected_calls = 2 if method == "AUTOIFS" else 1
    assert first["cost"]["optimizer_calls"] == expected_calls
    assert all(np.isfinite(value) for value in first["train_objective"].values())
    assert not resumed["complete"]


def test_autoifs_search9_transition_and_reuse1_recover_exactly(bundle):
    specification = job("AUTOIFS")
    session = training.create_session(specification, bundle, "cpu")
    for _ in range(9):
        training.advance_epoch(session)
    entering_final_search = training.export_session(session)
    expected_rewind = state_digest(session["model"].state_dict())
    last_search = training.advance_epoch(session)
    reuse_ready = training.export_session(session)
    assert last_search["stage"] == "search" and last_search["epoch"] == 9
    assert reuse_ready["completed_search"] == 10 and reuse_ready["commit_index"] == 10
    assert reuse_ready["stage"] == "reuse" and reuse_ready["next_stage_epoch"] == 1
    assert state_digest(reuse_ready["rewind_state"]) == expected_rewind
    assert reuse_ready["selection"]["selected"] is None
    assert reuse_ready["model_meta"] == dict(ticket=True, temp=100., gate_frozen=True)
    assert not reuse_ready["optimizer_states"]["base"]["state"]
    assert training._selector_matches(session["model"], reuse_ready["selector_state"])

    restored_search9 = training.restore_session(specification, bundle, "cpu", entering_final_search)
    training.advance_epoch(restored_search9)
    assert state_digest(scientific(training.export_session(restored_search9))) == state_digest(scientific(reuse_ready))

    first_reuse = training.advance_epoch(session)
    expected = training.export_session(session)
    restored_reuse = training.restore_session(specification, bundle, "cpu", reuse_ready)
    actual_record = training.advance_epoch(restored_reuse)
    assert state_digest(scientific(training.export_session(restored_reuse))) == state_digest(scientific(expected))
    assert first_reuse["stage"] == "reuse" and first_reuse["epoch"] == 1
    assert actual_record["cost"]["optimizer_calls"] == 1
    assert restored_reuse["selection"]["selected"]["stage"] == "reuse"
    assert all(not p.requires_grad for name, p in restored_reuse["model"].network.named_parameters()
               if "gate" in name)


def test_hypergate_maps_domain_once_and_preserves_source_objective(bundle):
    cfg = job("HYPERGATE")["config"]
    torch.manual_seed(2022)
    model = build_published("HYPERGATE", bundle, 2022, cfg).eval()
    x, y = torch.from_numpy(bundle.train_x), torch.from_numpy(bundle.train_y)
    batch, domains = model._batch(x, labels=y)
    assert torch.equal(batch["domain_indicator"], domains + 1)
    reference = model.network.score(batch)
    names = model.network.frating
    logits = torch.stack([reference[name]["score"] for name in names], dim=1)
    torch.testing.assert_close(model(x), logits, rtol=1e-6, atol=1e-6)
    predictive = sum(torch.nn.functional.binary_cross_entropy_with_logits(logits[:, i], y[:, i])
                     for i in range(2))
    torch.testing.assert_close(model.objective(x, y)["loss"], predictive + reference["cl_loss"],
                               rtol=1e-6, atol=1e-6)

