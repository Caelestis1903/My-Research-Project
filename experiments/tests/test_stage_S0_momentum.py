import os
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S0_run import DomainEMA, GAMMA_EQ, GAMMA_P, gamma_vector, memory_statistics


def test_domain_gamma_formula_and_uniform_controls():
    shares = np.asarray([150000, 34500, 3000], dtype=np.float64) / 187500.0
    domain = gamma_vector(shares, "comp1_domain")
    assert domain == pytest.approx([0.9, 0.977, 0.99])
    assert gamma_vector(shares, "uniform_p_tau") == pytest.approx([GAMMA_P] * 3)
    assert gamma_vector(shares, "uniform_eq_tau") == pytest.approx([GAMMA_EQ] * 3)


def test_memory_statistics_definitions():
    item = memory_statistics([0.9])[0]
    assert item["tau"] == pytest.approx(10.0)
    assert item["effective_n"] == pytest.approx(19.0)
    assert item["half_life"] == pytest.approx(np.log(0.5) / np.log(0.9))


def test_ema_updates_each_domain_once_and_aggregates():
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    ema = DomainEMA([parameter], [0.5, 0.5, 0.5])
    ema.update([[torch.tensor([2.0, 0.0])], [torch.tensor([0.0, 4.0])],
                [torch.tensor([2.0, 2.0])]])
    ema.aggregate_into_grads()
    assert ema.update_counts == [1, 1, 1]
    assert parameter.grad == pytest.approx(torch.tensor([2.0, 3.0]))


def test_ema_fails_fast_on_disconnected_or_nonfinite_shared_gradient():
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    ema = DomainEMA([parameter], [0.9])
    with pytest.raises(RuntimeError, match="disconnected"):
        ema.update([[None]])
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        ema.update([[torch.tensor([float("nan")])]])


def test_split_adam_default_beta_is_update_equivalent():
    left_a = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    left_b = torch.nn.Parameter(torch.tensor([3.0]))
    right_a = torch.nn.Parameter(left_a.detach().clone())
    right_b = torch.nn.Parameter(left_b.detach().clone())
    single = torch.optim.Adam([left_a, left_b], lr=0.01, weight_decay=1e-5,
                              betas=(0.9, 0.999))
    split = torch.optim.Adam([{"params": [right_a], "betas": (0.9, 0.999)},
                              {"params": [right_b], "betas": (0.9, 0.999)}],
                             lr=0.01, weight_decay=1e-5)
    for _ in range(3):
        for param, grad in ((left_a, [0.2, -0.1]), (right_a, [0.2, -0.1]),
                            (left_b, [0.3]), (right_b, [0.3])):
            param.grad = torch.tensor(grad)
        single.step()
        split.step()
    assert torch.equal(left_a, right_a)
    assert torch.equal(left_b, right_b)


def test_read_only_observer_gradients_do_not_change_update():
    base = torch.nn.Linear(2, 1, bias=False)
    observed = torch.nn.Linear(2, 1, bias=False)
    observed.load_state_dict(base.state_dict())
    x = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
    y = torch.tensor([[1.0], [0.0]])
    opt_a = torch.optim.Adam(base.parameters(), lr=0.01)
    opt_b = torch.optim.Adam(observed.parameters(), lr=0.01)
    loss_a = torch.nn.functional.binary_cross_entropy_with_logits(base(x), y)
    opt_a.zero_grad(); loss_a.backward(); opt_a.step()
    loss_b = torch.nn.functional.binary_cross_entropy_with_logits(observed(x), y)
    torch.autograd.grad(loss_b, tuple(observed.parameters()), retain_graph=True)
    opt_b.zero_grad(); loss_b.backward(); opt_b.step()
    assert torch.equal(base.weight, observed.weight)
