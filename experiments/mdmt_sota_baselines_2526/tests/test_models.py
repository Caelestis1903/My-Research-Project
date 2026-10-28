from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from experiments.mdmt_sota_baselines_2526.config import load_config
from experiments.mdmt_sota_baselines_2526.hypergate import TASK_FIELDS, _FeatureData
from experiments.mdmt_sota_baselines_2526.models import build_model
from experiments.mdmt_sota_baselines_2526.sources import (
    load_hypergate_class, load_hypergate_initializer,
)


torch.set_num_threads(1)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def schema():
    return {
        "dataset": "movielens",
        "columns": ["user_id", "age", "movie_id", "domain_indicator"],
        "dense_columns": ["age"],
        "sparse_columns": ["user_id", "movie_id", "domain_indicator"],
        "vocab_sizes": {"user_id": 7, "movie_id": 9, "domain_indicator": 3},
        "embedding_dim": 16,
        "domain_column": "domain_indicator",
        "oov_index": 0,
    }


def _sources(config):
    return config["source_roots"]


def _x():
    return torch.tensor([
        [1.0, 0.20, 4.0, 0.0],
        [2.0, 0.55, 3.0, 2.0],
        [4.0, 0.80, 8.0, 2.0],
    ])


def _y():
    return torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def _all_grads(module):
    grads = [p.grad for p in module.parameters() if p.requires_grad]
    assert grads and any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_hypergate_upstream_logits_aux_and_gradient_parity(config, schema):
    model = build_model("HYPERGATE", schema, config["models"]["HYPERGATE"], _sources(config))
    model.eval()
    x, y = _x(), _y()

    logits = model(x)
    batch, _ = model._batch(x, labels=y)
    assert torch.equal(batch["domain_indicator"], torch.tensor([1, 3, 3]))
    reference = model.network.score(batch)
    reference_logits = torch.stack([reference[name]["score"] for name in TASK_FIELDS], dim=1)
    torch.testing.assert_close(logits, reference_logits, rtol=1e-6, atol=1e-6)

    objective = model.objective(x, y)
    predictive = sum(F.binary_cross_entropy_with_logits(reference_logits[:, i], y[:, i])
                     for i in range(2))
    reference_loss = predictive + reference["cl_loss"]
    torch.testing.assert_close(objective["predictive"], predictive, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(objective["domain_aux"] + objective["task_aux"],
                               reference["cl_loss"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(objective["loss"], reference_loss, rtol=1e-6, atol=1e-6)

    objective["loss"].backward()
    core = model.network
    for module in (core.domain_aug_module, core.domain_anchor_embedding,
                   core.task_aug_module, core.task_anchor_embedding, core.domain_gate,
                   core.domain_embedding,
                   core.biasing_gate, core.fusion_gates, core.experts, core.towers):
        _all_grads(module)
    selected = next(p for p in core.fusion_gates[TASK_FIELDS[0]].parameters() if p.grad is not None)
    adapter_grad = selected.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    reference = core.score(batch)
    reference_logits = torch.stack([reference[name]["score"] for name in TASK_FIELDS], dim=1)
    ref_loss = sum(F.binary_cross_entropy_with_logits(reference_logits[:, i], y[:, i])
                   for i in range(2)) + reference["cl_loss"]
    ref_loss.backward()
    torch.testing.assert_close(adapter_grad, selected.grad, rtol=1e-6, atol=1e-6)

    assert core.enable_domain_gate and core.enable_biasing_gate and core.enable_fusion_gate
    for domain_ids in ([1], [0, 2]):
        rows = torch.tensor(domain_ids)
        x_missing = _x()[rows]
        result = model(x_missing, domain=x_missing[:, -1].long())
        assert result.shape == (len(domain_ids), 2) and torch.isfinite(result).all()
    empty = torch.empty((0, len(schema["columns"])))
    assert model(empty).shape == (0, 2)

    masked_y = y.clone()
    masked_y[0, 0] = float("nan")
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, 0] = False
    masked = model.objective(x, masked_y, mask)
    assert all(torch.isfinite(value) for value in masked.values())


def test_hypergate_official_initializer_and_field_order(config, schema):
    source_root = _sources(config)["recstudio"]
    torch.manual_seed(2022)
    model = build_model("HYPERGATE", schema, config["models"]["HYPERGATE"], _sources(config))

    torch.manual_seed(2022)
    official_type = load_hypergate_class(source_root)
    reference = official_type(config=copy.deepcopy(model.network.config))
    reference._init_model(_FeatureData(schema), drop_unused_field=False)
    initializer = load_hypergate_initializer(source_root)
    for _, module in reference.named_children():
        module.apply(initializer)

    assert model.network.state_dict().keys() == reference.state_dict().keys()
    for name, value in model.network.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name]), name
    for module in model.network.modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, torch.nn.Embedding) and module.padding_idx is not None:
            assert torch.equal(module.weight[module.padding_idx],
                               torch.zeros_like(module.weight[module.padding_idx]))

    model.eval()
    reference.eval()
    x, y = _x(), _y()
    batch, _ = model._batch(x, labels=y)
    reference_scores = reference.score(batch)
    reference_logits = torch.stack([reference_scores[name]["score"] for name in TASK_FIELDS], dim=1)
    torch.testing.assert_close(model(x), reference_logits, rtol=1e-6, atol=1e-6)


def test_hypergate_same_torch_seed_across_python_hash_seeds(config):
    project_root = Path(__file__).resolve().parents[3]
    script = """
import json
import torch
from experiments.mdmt_sota_baselines_2526.config import load_config
from experiments.mdmt_sota_baselines_2526.models import build_model
torch.set_num_threads(1)
cfg = load_config()
torch.manual_seed(2022)
schema = {
    "dataset": "movielens",
    "columns": ["user_id", "age", "movie_id", "domain_indicator"],
    "dense_columns": ["age"],
    "sparse_columns": ["user_id", "movie_id", "domain_indicator"],
    "vocab_sizes": {"user_id": 7, "movie_id": 9, "domain_indicator": 3},
    "embedding_dim": 16,
}
model = build_model("HYPERGATE", schema, cfg["models"]["HYPERGATE"],
                    cfg["source_roots"]).eval()
x = torch.tensor([[1., .20, 4., 0.], [2., .55, 3., 2.], [4., .80, 8., 2.]])
with torch.inference_mode():
    print(json.dumps(model(x).tolist(), separators=(",", ":")))
"""
    outputs = []
    for hash_seed in ("1", "2"):
        env = os.environ.copy()
        env.update(PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-c", script], cwd=project_root,
                                env=env, text=True, capture_output=True, check=True)
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1]


def test_autoifs_forward_biases_search_groups_and_ticket_transition(config, schema):
    configured = copy.deepcopy(config["models"]["AUTOIFS"])
    configured.update(dataset="movielens", seed=2022)
    configured["lambda"] = [0.15, 0.2]
    model = build_model("AUTOIFS", schema, configured, _sources(config))
    model.eval()
    x, y = _x(), _y()
    core = model.network

    calls = []
    hook = core.register_forward_hook(lambda *_: calls.append(True))
    objective = model.objective(x, y)
    hook.remove()
    assert len(calls) == 1
    assert torch.isfinite(objective["loss"])
    torch.testing.assert_close(objective["loss"],
                               objective["predictive"] + objective["gate_penalty"])

    direct = core(x, x[:, model.domain_index].long().reshape(-1, 1))
    torch.testing.assert_close(model(x), torch.cat((direct[0], direct[1]), dim=1))
    assert model(torch.tensor([[1.0, 0.2, 4.0, 1.0]])).shape == (1, 2)
    empty = torch.empty((0, len(schema["columns"])))
    assert model(empty).shape == (0, 2)

    assert all(bias.shape == (kernel.shape[1],)
               for kernel, bias in zip(core.D_kernels, core.D_bias))
    assert all(bias.shape == (kernel.shape[1],)
               for kernel, bias in zip(core.T_kernels, core.T_bias))
    assert all(bias.shape == (core.domain_num, kernel.shape[1])
               for kernel, bias in zip(core.D_kernels, core.D_lora_bias))
    assert all(bias.shape == (core.task_num, kernel.shape[1])
               for kernel, bias in zip(core.T_kernels, core.T_lora_bias))
    assert core.D_kernels[0].shape[0] != core.D_bias[0].numel()
    assert core.T_kernels[0].shape[0] != core.T_bias[0].numel()

    groups = model.optimizer_groups()
    group_ids = {key: {id(p) for p in values} for key, values in groups.items()}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert group_ids["base"] and group_ids["gate"]
    assert not (group_ids["base"] & group_ids["gate"])
    assert group_ids["base"] | group_ids["gate"] == trainable_ids

    objective["loss"].backward()
    for module in (core.share_mlp, core.gate_hypernet, core.gate_head0, core.gate_head1):
        _all_grads(module)
    _all_grads(core.D_B_kernel)
    _all_grads(core.T_B_kernel)

    with torch.no_grad():
        core.D_B_kernel[0].fill_(0.01)
        core.T_B_kernel[0].fill_(0.01)
    model.zero_grad(set_to_none=True)
    model.objective(x, y)["loss"].backward()
    assert core.D_A_kernel[0].grad.abs().sum() > 0
    assert core.T_A_kernel[0].grad.abs().sum() > 0
    masked_y = y.clone()
    masked_y[0, 0] = float("nan")
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, 0] = False
    assert torch.isfinite(model.objective(x, masked_y, mask)["loss"])

    rewind = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        core.embedding.embedding.add_(0.125)
        core.embedding.dense_vector.add_(0.25)
        core.gate_head0.weight.zero_()
        core.gate_head0.bias.copy_(torch.tensor([1.0, -1.0, 1.0, -1.0]))
        core.gate_head1.weight.zero_()
        core.gate_head1.bias.copy_(torch.tensor([-1.0, 1.0, -1.0, 1.0]))
    selector = model.export_search_state()

    reuse = build_model("AUTOIFS", schema, configured, _sources(config))
    reuse.load_state_dict(rewind)
    reuse.import_selector_state(selector)
    reuse.network.ticket = True
    reuse.network.temp = 100.0
    reuse.freeze_gate()
    assert reuse.network.ticket
    torch.testing.assert_close(reuse.network.gate_embedding.embedding, core.embedding.embedding)
    torch.testing.assert_close(reuse.network.gate_embedding.dense_vector, core.embedding.dense_vector)
    reuse_groups = reuse.optimizer_groups()
    assert reuse_groups["base"] and reuse_groups["gate"] == []
    assert all(not p.requires_grad for name, p in reuse.network.named_parameters() if "gate" in name)

    reuse.train()
    before = reuse.network.D_kernels[0].detach().clone()
    logits = reuse(x)
    assert logits.shape == (3, 2) and torch.isfinite(logits).all()
    assert torch.all((reuse.network.weights[0] > 0) ==
                     torch.tensor([[True, False, True, False]]).expand(3, -1))
    loss = reuse.objective(x, y)["loss"]
    loss.backward()
    optimizer = torch.optim.Adam(reuse_groups["base"], lr=1e-3)
    optimizer.step()
    assert not torch.equal(before, reuse.network.D_kernels[0])
    assert all(p.grad is None for name, p in reuse.network.named_parameters() if "gate" in name)


def test_m3oe_fixed1_singleton_and_masked_objective(config, schema):
    configured = copy.deepcopy(config["models"]["M3OE_FIXED1"])
    configured.update(dataset="movielens", seed=2022)
    model = build_model("M3OE_FIXED1", schema, configured, _sources(config)).eval()
    assert model.initialization["learned_fusion"] is False
    x = torch.tensor([[2.0, 0.35, 5.0, 1.0]])
    logits = model(x, domain=torch.tensor([1]))
    assert logits.shape == (1, 2) and torch.isfinite(logits).all()
    labels = torch.tensor([[float("nan"), 1.0]])
    mask = torch.tensor([[False, True]])
    objective = model.objective(x, labels, mask)
    assert torch.isfinite(objective["loss"])
    empty = torch.empty((0, len(schema["columns"])))
    assert model(empty).shape == (0, 2)
    with pytest.raises(ValueError, match="domain IDs"):
        model(x, domain=torch.tensor([1.5]))

    x_full, y_full = _x(), _y()
    objective = model.objective(x_full, y_full)
    native_loss = F.binary_cross_entropy_with_logits(model(x_full), y_full)
    torch.testing.assert_close(objective["loss"], native_loss, rtol=1e-6, atol=1e-6)
    objective["loss"].backward()
    adapter_grads = {name: (None if parameter.grad is None else parameter.grad.detach().clone())
                     for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(model(x_full), y_full).backward()
    for name, parameter in model.named_parameters():
        actual = adapter_grads[name]
        if actual is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(actual, parameter.grad, rtol=1e-6, atol=1e-6)

    uneven_labels = y_full.clone()
    uneven_labels[1, 1] = float("nan")
    uneven_mask = torch.tensor([[True, True], [True, False], [False, True]])
    uneven_logits = model(x_full)
    per_label = F.binary_cross_entropy_with_logits(
        uneven_logits, torch.nan_to_num(uneven_labels), reduction="none")
    expected = (per_label * uneven_mask).sum() / uneven_mask.sum()
    uneven = model.objective(x_full, uneven_labels, uneven_mask)
    torch.testing.assert_close(uneven["loss"], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("method", ["HYPERGATE", "AUTOIFS", "M3OE_FIXED1"])
def test_sparse_range_and_domain_validation(config, schema, method):
    model_config = copy.deepcopy(config["models"][method])
    model_config.update(dataset="movielens", seed=2022)
    model = build_model(method, schema, model_config, _sources(config))
    max_valid = _x()[:1].clone()
    max_valid[0, 0] = schema["vocab_sizes"]["user_id"] - 1
    max_valid[0, 2] = schema["vocab_sizes"]["movie_id"] - 1
    assert model(max_valid).shape == (1, 2)
    oov = max_valid.clone()
    oov[0, 0] = 0
    assert model(oov).shape == (1, 2)
    bad_x = _x()[:1].clone()
    bad_x[0, 0] = schema["vocab_sizes"]["user_id"]
    with pytest.raises(ValueError, match="outside its integer vocabulary"):
        model(bad_x)
    with pytest.raises(ValueError, match="Explicit domain IDs"):
        model(_x()[:1], domain=torch.tensor([0.5]))
