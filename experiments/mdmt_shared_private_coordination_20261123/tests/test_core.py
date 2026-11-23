"""Block-update invariants and source fixed1 fixture identity."""
import copy
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.mdmt_shared_private_coordination_20261123 import blocks, source

CFG = source.config("movielens")


@pytest.fixture
def native():
    source.setup(2022, "cpu")
    features = json.loads((source.RECEIPTS / "movielens_data_identity.json").read_bytes())["features"]
    bundle = source.TrainValidation(source.raw.FeatureSchema(**features), *([None] * 6))
    model = source.make_model(bundle, "movielens")
    return model, features


def _batch(features, size, seed):
    rng = torch.Generator().manual_seed(seed)
    x = torch.stack([
        torch.rand(size, generator=rng) if name in features["dense_columns"] else
        torch.randint(features["vocab_sizes"][name], (size,), generator=rng).float()
        for name in features["columns"]
    ])
    y = torch.randint(2, (size, 2), generator=rng).float()
    return x, y


def _state(model, opt, ema):
    return source.state_digest((model.state_dict(), opt.state_dict(), ema.state_dict()))


def _donor_substep(model, opt, ema, x, y, names):
    opt.zero_grad(set_to_none=True)
    F.binary_cross_entropy(model(x), y, reduction="none").mean(1).mean().backward()
    for name, param in model.named_parameters():
        if name not in names:
            param.grad = None
    opt.step()
    with torch.no_grad():
        for target, current in zip(ema.parameters(), model.parameters()):
            target.mul_(CFG["rho"]).add_(current, alpha=1 - CFG["rho"])
        for target, current in zip(ema.buffers(), model.buffers()):
            target.copy_(current)


def test_joint_fixture_exact_and_groups(native):
    model, features = native
    groups = blocks.groups(model)
    a, b, c = (groups[group] for group in blocks.CORE_GROUPS)
    assert not (a & b or a & c or b & c)
    assert a | b | c == {name for name, p in model.named_parameters() if p.requires_grad}
    assert groups["P_FULL"] == b | c
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = source.optimizer(model, CFG)
    reference, reference_opt, reference_ema = copy.deepcopy((model, opt, ema))
    batch = _batch(features, 17, 12)
    with torch.no_grad():
        probability, logits = source.forward_logits(model, batch[0])
        torch.testing.assert_close(probability, logits.sigmoid(), atol=1e-7, rtol=1e-6)
        expected = F.binary_cross_entropy(reference(batch[0]), batch[1], reduction="none").mean(1).mean()
    result = blocks.step(model, opt, ema, (batch,), CFG, detailed=False)
    _donor_substep(reference, reference_opt, reference_ema, *batch, a | b | c)
    assert result["loss"] == float(expected)
    assert result["rows"] == 17
    assert result["optimizer_steps"] == result["ema_updates"] == 1
    assert result["groups"] == {}
    assert _state(model, opt, ema) == _state(reference, reference_opt, reference_ema)


@pytest.mark.parametrize("second_size", [17, 29])
def test_accumulation_gradients_and_ema(native, second_size):
    model, features = native
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = source.optimizer(model, CFG)
    blocks.step(model, opt, ema, (_batch(features, 21, 13),), CFG)
    reference = copy.deepcopy(model)
    expected_ema = copy.deepcopy(ema)
    source.update_ema(expected_ema, model, CFG["rho"])
    a, b = _batch(features, 17, 12), _batch(features, second_size, 33)
    x, y = torch.cat((a[0], b[0]), dim=1), torch.cat((a[1], b[1]))
    F.binary_cross_entropy(reference(x), y, reduction="none").mean(1).mean().backward()
    result = blocks.step(model, opt, ema, (a, b), CFG)
    source.update_ema(expected_ema, model, CFG["rho"])
    assert result["ema_updates"] == result["training_forwards"] == result["parameter_backward_calls"] == 2
    assert result["optimizer_steps"] == 1 and result["rows"] == 17 + second_size
    for p, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, expected.grad, rtol=2e-5, atol=2e-7)
    assert source.state_digest(ema.state_dict()) == source.state_digest(expected_ema.state_dict())
    for stats in result["groups"].values():
        assert stats["step_delta_range"] == [1, 1]


@pytest.mark.parametrize("sequence", [("U", "P_FULL"), ("P_FULL", "U")])
def test_sequential_poststates_and_block_clocks(native, sequence):
    model, features = native
    opt = source.optimizer(model, CFG)
    ema = copy.deepcopy(model).requires_grad_(False)
    blocks.step(model, opt, ema, (_batch(features, 21, 13),), CFG)
    reference, reference_opt, reference_ema = copy.deepcopy((model, opt, ema))
    stale = copy.deepcopy(model)
    groups = blocks.groups(model)
    batches = (_batch(features, 17, 12), _batch(features, 23, 33))
    old_reference_grads = torch.autograd.grad(
        blocks.natural_loss(stale(batches[1][0]), batches[1][1]), tuple(stale.parameters())
    )
    for group, batch in zip(sequence, batches):
        result = blocks.step(model, opt, ema, (batch,), CFG, names=groups[group])
        _donor_substep(reference, reference_opt, reference_ema, *batch, groups[group])
        assert _state(model, opt, ema) == _state(reference, reference_opt, reference_ema)
        assert result["ema_updates"] == 1
        for core in blocks.CORE_GROUPS:
            expected_delta = int(groups[core] <= groups[group])
            assert result["groups"][core]["step_delta_range"] == [expected_delta, expected_delta]
    assert any(
        not torch.equal(before, p.grad)
        for (name, p), before in zip(model.named_parameters(), old_reference_grads, strict=True)
        if name in groups[sequence[1]]
    )


class ExplicitZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_weight = nn.Parameter(torch.tensor([.28, -.11]))
        self.gate = nn.Linear(2, 2)
        self.factor = 1

    def forward(self, x):
        return (x.T * self.shared_weight + self.factor * self.gate(x.T)).sigmoid()


def test_none_and_explicit_zero_preserve_native_adam_semantics():
    source.setup(2022, "cpu")
    model = ExplicitZero()
    opt = source.optimizer(model, CFG)
    ema = copy.deepcopy(model).requires_grad_(False)
    groups = blocks.groups(model)
    batch = (torch.tensor([[1., 2., 3.], [-1., 1., 2.]]), torch.tensor([[0., 0.], [1., 0.], [1., 1.]]))
    blocks.step(model, opt, ema, (batch,), CFG, names=groups["U"])
    private = list(model.gate.parameters())
    assert all(p.grad is None and p not in opt.state for p in private)
    blocks.step(model, opt, ema, (batch,), CFG)
    assert all(opt.state[p]["exp_avg"].count_nonzero() and opt.state[p]["exp_avg_sq"].count_nonzero()
               for p in private)
    before = source.state_digest([(p, opt.state[p]) for p in private])
    model.factor = 0
    skipped = blocks.step(model, opt, ema, (batch,), CFG, names=groups["U"])
    assert all(p.grad is None for p in private)
    assert source.state_digest([(p, opt.state[p]) for p in private]) == before
    assert skipped["groups"]["P_CELL"]["step_delta_range"] == [0, 0]
    reference, reference_opt, reference_ema = copy.deepcopy((model, opt, ema))
    updated = blocks.step(model, opt, ema, (batch,), CFG, names=groups["P_CELL"])
    _donor_substep(reference, reference_opt, reference_ema, *batch, groups["P_CELL"])
    assert _state(model, opt, ema) == _state(reference, reference_opt, reference_ema)
    assert updated["groups"]["P_CELL"]["explicit_zero_gradients"] == 2
    assert updated["groups"]["P_CELL"]["coupled_decay_norm"] > 0
    assert updated["groups"]["P_CELL"]["displacement_norm"] > 0
    assert updated["groups"]["P_CELL"]["step_delta_range"] == [1, 1]
    assert source.state_digest([(p, opt.state[p]) for p in private]) != before
