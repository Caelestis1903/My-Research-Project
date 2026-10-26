"""Bounded numerical/gradient checks required by RCF-v1; no research runs."""
from __future__ import annotations

import copy
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import expit

from .demand import profiled_moments, symmetric_metric, exhaustive_assignment
from .formation import (DeltaMLP, RowCycle, bernoulli_kl, constraint_values,
                        feasible_scale, initialize_delta)


def run_required(config, *, real_bundle=None):
    torch.set_num_threads(1)
    rng = np.random.default_rng(90626)
    phi, b = rng.normal(size=(37, 32)), rng.normal(size=37)
    y = rng.integers(2, size=37).astype(float)
    moments = profiled_moments(phi, b, y, config)
    p, n = expit(b), np.column_stack((np.ones(37), b))
    vv = p*(1-p)
    hp = (phi.T*vv) @ n/37
    hn = (n.T*vv) @ n/37
    explicit = (phi.T @ (y-p))/37-hp @ np.linalg.solve(
        hn+moments['nuisance_ridge']*np.eye(2), n.T @ (y-p)/37)
    profile_error = float(np.max(np.abs(moments['score']-explicit)))
    assert profile_error < 1e-12
    metric, invroot, details = symmetric_metric(
        [moments['H_phi_phi']]*2, np.array([.3, .7]), phi, config)
    identity_error = float(np.max(np.abs(invroot @ metric @ invroot-np.eye(32))))
    assert identity_error < 1e-9
    assignment, objective = exhaustive_assignment(np.array([[1., 0], [1.1,0], [-1,0],[-1.1,0]]),
        np.array([.1,.2,.3,.4]), [(0,0),(0,1),(1,0),(1,1)])
    assert assignment[0,0] == 1 and np.array_equal(assignment.argmax(1), [0,0,1,1])

    # Unequal domain/group sizes: omega reproduces pi once, including a short batch.
    masses, sizes = np.array([.2,.3,.5]), np.array([2,3,5])
    x = torch.tensor(np.arange(1,11), dtype=torch.float64)
    parameter = torch.tensor(.17, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0,1,0,1,1,0,1,0,1,1], dtype=torch.float64)
    pi_row = np.repeat(masses/sizes, sizes)
    omega = torch.tensor(10*pi_row, dtype=torch.float64)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(parameter*x, labels, reduction='none')
    weighted = (omega*loss).mean()
    explicit_loss = sum(float(masses[g])*loss[sum(sizes[:g]):sum(sizes[:g+1])].mean()
                        for g in range(3))
    grad_weighted = torch.autograd.grad(weighted, parameter, retain_graph=True)[0]
    grad_explicit = torch.autograd.grad(explicit_loss, parameter, retain_graph=True)[0]
    # Accumulating real short batches uses their true sample count for this identity.
    short_recombined = sum((omega[i:i+4]*loss[i:i+4]).mean()*len(loss[i:i+4])/10
                           for i in range(0,10,4))
    assert torch.allclose(weighted, explicit_loss, atol=1e-14, rtol=0)
    assert torch.allclose(grad_weighted, grad_explicit, atol=1e-14, rtol=0)
    assert torch.allclose(weighted, short_recombined, atol=1e-14, rtol=0)
    assert abs(float(omega.mean())-1) < 1e-14

    # Exact zero and correct first/second-step learning, with frozen teacher inputs.
    torch.manual_seed(1026)
    weight, bias = torch.randn(32,256)*.1, torch.ones(32)*.01
    model = DeltaMLP(weight, bias)
    if real_bundle is None:
        h = torch.randn(31,256)
        target = torch.randint(0,2,(31,)).float()
        base = torch.randn(31)
        real_status = 'NOT_RUN_WITH_REAL_BUNDLE'
    else:
        view = real_bundle.views['P_form']
        h = torch.from_numpy(np.array(view.h[:31]))
        target = torch.from_numpy(np.array(view.y[:31,0]))
        base = torch.from_numpy(np.array(real_bundle.base_oof[:31,0]))
        real_status = getattr(real_bundle, 'verification_source', 'REAL_FROZEN_PREFIX_MINIBATCH')
        if hasattr(real_bundle, 'pilots'):
            model = initialize_delta(real_bundle,0,0,0,config)
    assert torch.equal(model(h), torch.zeros(len(h)))
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    loss0 = torch.nn.functional.binary_cross_entropy_with_logits(base+model(h), target)
    loss0.backward()
    out_gradient = float(model.output.weight.grad.norm())
    hidden_gradient0 = float(model.hidden.weight.grad.norm())
    assert out_gradient > 0 and hidden_gradient0 == 0
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.nn.functional.binary_cross_entropy_with_logits(base+model(h),target).backward()
    hidden_gradient1 = float(model.hidden.weight.grad.norm())
    assert hidden_gradient1 > 0
    dormant = torch.nn.Linear(256,1)
    dormant_before = copy.deepcopy(dormant.state_dict())
    other = torch.optim.Adam(list(model.parameters())+list(dormant.parameters()), lr=.001,weight_decay=.01)
    other.step()
    assert all(p.grad is None and p not in other.state for p in dormant.parameters())
    assert all(torch.equal(v,dormant.state_dict()[k]) for k,v in dormant_before.items())

    a = torch.tensor([-40.,-3.,0.,3.,40.], dtype=torch.float64)
    delta = torch.tensor([.2,-.3,.5,.6,-.7], dtype=torch.float64)
    kl = bernoulli_kl(a,delta)
    explicit_kl = np.logaddexp(0,a.numpy()+delta.numpy())-np.logaddexp(0,a.numpy())-expit(a.numpy())*delta.numpy()
    assert np.max(np.abs(kl.numpy()-explicit_kl)) < 1e-13
    assert float(kl.min()) >= -1e-14

    base_np, target_np = np.zeros(40), np.tile([0.,1.],20)
    ds = np.full(40, 3.)
    c_s = feasible_scale(base_np,target_np,ds,[np.zeros(20),np.zeros(10)],config)
    root_s = brentq(lambda c:np.logaddexp(0,3*c)-np.log(2)-1.5*c-
                   config['formation']['source_ll_budget'],0,1)
    assert 0 <= root_s-c_s < 2e-11
    c_r = feasible_scale(base_np,target_np,np.zeros(40),[np.full(20,2.),np.full(10,2.)],config)
    assert abs(c_r-.25) < 2e-11
    c_both = feasible_scale(base_np,target_np,ds,[np.full(20,20.),np.full(10,20.)],config)
    assert abs(c_both-min(root_s,.025)) < 2e-11
    c_local = feasible_scale(base_np,target_np,np.full(40,100),[np.full(20,2.)],config,
                             source_constraint=False,local=True)
    assert abs(c_local-.25) < 2e-11
    assert constraint_values(base_np,target_np,ds,[np.ones(10),np.ones(10)],scale=0)[0] == 0

    # In-memory full next-step replay with Adam and independent aux cursor/RNG.
    torch.manual_seed(809)
    net = DeltaMLP(weight,bias)
    optimizer = torch.optim.Adam(net.parameters(),lr=.001)
    cycle = RowCycle(np.arange(7),np.random.default_rng(19))
    def step():
        idx = cycle.take(11)%len(h)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.binary_cross_entropy_with_logits(base[idx]+net(h[idx]),target[idx]).backward()
        optimizer.step()
    step()
    snapshot = dict(model=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
                    cycle=copy.deepcopy(cycle),rng=torch.get_rng_state().clone(),nu=1.3,xi=.7)
    step()
    continuous = copy.deepcopy(net.state_dict())
    continuous_opt = copy.deepcopy(optimizer.state_dict())
    net.load_state_dict(snapshot['model'])
    optimizer.load_state_dict(snapshot['optimizer'])
    cycle = copy.deepcopy(snapshot['cycle'])
    torch.set_rng_state(snapshot['rng'])
    step()
    replay_error = max(float((continuous[k]-net.state_dict()[k]).abs().max()) for k in continuous)
    assert replay_error == 0
    for key, state in continuous_opt['state'].items():
        for name, value in state.items():
            other_value = optimizer.state_dict()['state'][key][name]
            assert torch.equal(value,other_value) if torch.is_tensor(value) else value == other_value
    return dict(status='PASS', profile_max_abs=profile_error, metric_identity_max_abs=identity_error,
        assignment_objective=objective, omega_mean=float(omega.mean()),
        omega_gradient_error=float(abs(grad_weighted-grad_explicit)),
        delta_zero_exact=True, output_first_gradient_norm=out_gradient,
        hidden_first_gradient_norm=hidden_gradient0, hidden_second_gradient_norm=hidden_gradient1,
        inactive_none_preserved=True, projection_c_source=c_s, projection_c_rms=c_r,
        projection_c_both=c_both, local_projection_ignores_source=True,
        memory_resume_max_abs=replay_error, real_minibatch=real_status,
        model_files_written=False, research_training=False)
