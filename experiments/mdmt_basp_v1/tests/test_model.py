"""Numerical and semantic checks directly required by the BASP contract."""
import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from experiments.mdmt_basp_v1.config import load_config, arms
from experiments.mdmt_basp_v1.model import BASPModel, TargetModel, make_model
from experiments.mdmt_basp_v1.benefit import exact_benefit, measure_benefits, refresh_teacher, align_cache
from experiments.mdmt_basp_v1.relation import effective_sample_size, GainTable, reading_policy, weighted_low_rank
from experiments.mdmt_basp_v1.losses import training_losses, project_sources
from experiments.mdmt_basp_v1.permissions import make_optimizer, apply_gradients, add_new_parameters, ALLOWED
from experiments.mdmt_basp_v1.conflict import conditional_partition, source_gradient_sketch
from experiments.mdmt_basp_v1.transaction import decide, run_transaction


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(2022)


def packet(n=48, dtype=torch.float32):
    p = dict(h=torch.randn(n, 256, dtype=dtype), z=torch.randn(n, 2, dtype=dtype),
             phi=torch.randn(n, 3, 64, dtype=dtype))
    return p, torch.arange(n) % 3, torch.randint(0, 2, (n, 2)).to(dtype)


def model(cfg=None, dtype=torch.float32):
    cfg = cfg or load_config()
    return BASPModel(cfg, teacher_w=torch.randn(3, 2, 64)*.1).to(dtype=dtype)


def nontrivial(m):
    with torch.no_grad():
        for w in m.readers.values():
            w.normal_(0, .1)
        for head in m.gain_heads:
            head.base.weight.normal_(0, .1)
            head.base.bias.normal_(0, .1)
        for tree in m.needs:
            tree.initialized = True
            tree.centers.base.normal_(0, .2)
            for local in tree.locals.values():
                local.weight.normal_(0, .1)
        for tree in m.sources:
            for block in tree.blocks.values():
                block.U.normal_(0, .1)
        for table in m.tables:
            table.policy_gain.uniform_(-.01, .02)
        m.benefit_scale.fill_(.01)
    m.set_epoch(3)


def test_parameter_counts_and_label_free_initialization():
    cfg = load_config()
    m = model(cfg)
    t = TargetModel(cfg)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 63680
    assert sum(p.numel() for p in t.parameters() if p.requires_grad) == 63966
    p, d, y = packet()
    assert torch.equal(m(p, d).logits, p['z'])
    assert torch.equal(t(p, d).logits, p['z'])
    before = m(p, d).logits
    y = y.flip(0)
    assert torch.equal(before, m(p, d).logits)
    with pytest.raises(TypeError):
        m(p, d, y=y)
    with pytest.raises(TypeError):
        m(dict(p, gain=y), d)
    empty, ed, _ = packet(0)
    assert m(empty, ed).logits.shape == (0, 2)


def test_exact_bce_signs_off_and_local_expansion():
    y = torch.tensor([0., 0.], dtype=torch.float64)
    ref = torch.zeros_like(y)
    correction = torch.tensor([1., -1.], dtype=torch.float64)
    gain = exact_benefit(y, ref, correction)
    assert gain[0] < 0 < gain[1]
    assert torch.equal(exact_benefit(y, ref, torch.zeros_like(y)), torch.zeros_like(y))
    assert torch.allclose(gain, np.log(2)-F.softplus(correction), atol=1e-15, rtol=0)
    small = correction*1e-4
    first = (y-ref.sigmoid())*small
    second = first-.5*ref.sigmoid()*(1-ref.sigmoid())*small.square()
    exact = exact_benefit(y, ref, small)
    assert (exact-second).abs().max() < (exact-first).abs().max()


def test_gain_support_policy_and_unique_count():
    mass = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    assert torch.equal(effective_sample_size(mass), torch.tensor([2., 1.]))
    gain = torch.tensor([[[1., -2.]], [[3., -4.]], [[9., 8.]]])
    table = GainTable(1, 2, 2)
    table.measure(gain, mass, [0, 1, 2], minimum_ess=2, version='v1')
    assert torch.equal(table.observed_gain[0, 0], torch.tensor([2., -3.]))
    assert not table.observed[0, 1].any()
    assert table.unique_count[0, 0, 0] == 2
    before = table.policy_gain.clone()
    table.split_source(0)
    assert torch.equal(table.policy_gain[..., 0], table.policy_gain[..., -1])
    assert not table.observed[..., [0, -1]].any()
    assert not table.unique_count[..., [0, -1]].any()
    alpha, null = reading_policy(torch.tensor([[[-1., -2.]]]), torch.tensor([[.4, .6]]), .1)
    assert not alpha.any() and torch.equal(null, torch.ones_like(null))
    with pytest.raises(ValueError):
        GainTable(1, 2, 2).measure(gain, mass, [0, 0, 2])


def test_teacher_measurement_is_detached_and_pure():
    m = model()
    nontrivial(m)
    p, d, y = packet()
    before = [t.policy_gain.clone() for t in m.tables]
    measurement = measure_benefits(m, p, d, y)
    assert not measurement['benefits'].requires_grad
    assert all(torch.equal(a, t.policy_gain) for a, t in zip(before, m.tables))
    cache = refresh_teacher(m, p, d, y, np.arange(len(y)), freeze_scale=True, version='epoch1')
    assert cache.version == 'epoch1'
    assert not any(p.requires_grad for p in cache.teacher.parameters())
    first = next(cache.teacher.parameters()).clone()
    with torch.no_grad():
        next(m.parameters()).add_(1)
    assert torch.equal(first, next(cache.teacher.parameters()))
    cache.to('cpu')
    assert cache.subset(np.arange(5))['target'].shape == (5, 2, 4)


def test_actual_loss_graph_permissions_and_foreign_detach():
    m = model()
    m.flags['SOURCE_WRITE'] = False
    nontrivial(m)
    p, d, y = packet()
    for value in p.values():
        value.requires_grad_(True)
    cache = refresh_teacher(m, p, d, y, np.arange(len(y)), freeze_scale=True)
    owner = dict(packet=p, domain=d, y=y, weights=torch.tensor([.2, .3, .5]))
    losses = training_losses(m, p, d, y, probe=cache.subset(np.arange(len(y))), owner=owner, epoch=5)
    named = list(m.named_parameters())
    categories = m.parameter_categories()
    for name, loss in losses.items():
        if not loss.requires_grad:
            continue
        grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True, retain_graph=True)
        touched = {categories[n] for (n, _), g in zip(named, grads) if g is not None and bool(g.abs().any())}
        assert touched <= ALLOWED[name], (name, touched)
        if name == 'rec':
            assert 'source' not in touched and 'owner' not in touched
        if name == 'owner':
            assert 'source' in touched
        if name in ('gain', 'cluster', 'contrast', 'shadow', 'owner', 'anchor'):
            external = torch.autograd.grad(loss, list(p.values()), allow_unused=True, retain_graph=True)
            assert all(value is None or not value.abs().any() for value in external)


def test_cold_start_owner_and_shadow_in_all_null_policy():
    m = model()
    p, d, y = packet()
    m.set_epoch(3)
    with torch.no_grad():
        for table in m.tables:
            table.policy_gain.fill_(-1)
    before = m(p, d)
    assert torch.equal(before.logits, before.reference_logits)
    losses = training_losses(m, p, d, y, owner=dict(packet=p, domain=d, y=y), epoch=3)
    u = m.sources[0].blocks['s0'].U
    v = m.sources[0].blocks['s0'].V
    ug, vg = torch.autograd.grad(losses['owner'], [u, v], retain_graph=True)
    assert ug.abs().sum() > 0 and v.abs().sum() > 0
    assert not vg.abs().any()
    reader = m.readers['d0s1'].clone()
    apply_gradients(losses, m, make_optimizer(m), 3)
    assert not torch.equal(reader, m.readers['d0s1'])


@pytest.mark.parametrize('dtype,atol', [(torch.float32, 1e-6), (torch.float64, 1e-12)])
@pytest.mark.parametrize('kind', ['NEED_SPLIT', 'SOURCE_SPLIT'])
def test_split_conservation_nonuniform_routing_and_null(dtype, atol, kind):
    m = model(dtype=dtype)
    nontrivial(m)
    p, d, y = packet(dtype=dtype)
    before = m(p, d)
    weight = torch.randn(258, dtype=dtype)*.01
    proposal = dict(kind=kind, domain=0, parent=0, weight=weight, bias=torch.tensor(.9, dtype=dtype))
    if kind == 'NEED_SPLIT':
        proposal['centers'] = torch.randn(2, 2, 4, dtype=dtype)
    rng_before = torch.get_rng_state().clone()
    m.add_structure('SPLIT', proposal)
    assert torch.equal(rng_before, torch.get_rng_state())
    after = m(p, d)
    assert (before.logits-after.logits).abs().max() <= atol
    assert torch.allclose(after.need_mass.sum(-1), torch.ones(len(y), dtype=dtype), atol=atol, rtol=0)
    for dd in range(3):
        rows = d == dd
        kn = len(m.needs[dd].leaves)
        total = after.read_mass[rows, :, :kn].sum((-1, -2))+after.null_mass[rows, :, :kn]
        assert torch.allclose(total, torch.ones_like(total), atol=atol, rtol=0)
        assert not after.read_mass[rows, :, :, dd].any()
    if kind == 'SOURCE_SPLIT':
        assert torch.allclose(m.action_weight(1).sum(), torch.tensor(1., dtype=dtype))
        assert not m.tables[1].unique_count[..., [0, -1]].any()


def test_new_parameter_state_zero_old_state_unchanged_and_cache_alignment():
    m = model()
    nontrivial(m)
    p, d, y = packet()
    cache = refresh_teacher(m, p, d, y, np.arange(len(y)), freeze_scale=True)
    opt = make_optimizer(m)
    apply_gradients(training_losses(m, p, d, y, epoch=3), m, opt, 3)
    old = {name: (param, copy.deepcopy(opt.state.get(param))) for name, param in m.named_parameters()}
    m.add_structure('SPLIT', dict(kind='SOURCE_SPLIT', domain=0, parent=0))
    added = add_new_parameters(m, opt)
    assert added
    for name, param in m.named_parameters():
        if name in old:
            previous, state = old[name]
            assert param is previous
            if state is not None:
                assert all(torch.equal(state[k], opt.state[param][k]) for k in state)
        else:
            assert param not in opt.state
    aligned = align_cache(cache, m)
    assert aligned.benefits.shape[-1] == 5
    for domain in (1, 2):
        rows = d == domain
        assert torch.equal(aligned.benefits[rows, :, 0], aligned.benefits[rows, :, -1])


def test_auxiliary_norm_cap_and_warmup_no_source_state():
    m = model()
    opt = make_optimizer(m)
    rec = m.trunk.weight.sum()*.001
    gain = m.trunk.weight.sum()
    report = apply_gradients(dict(rec=rec, gain=gain), m, opt, 3)
    assert report['aux_gamma'] < 1
    assert report['aux_gamma']*report['trunk_aux_norm'] <= .25*max(report['trunk_rec_norm'], .001)+1e-9
    p, d, y = packet()
    apply_gradients(training_losses(m, p, d, y, epoch=1), m, opt, 1)
    assert all(param not in opt.state for tree in m.sources for param in tree.parameters())


def test_full_support_projection_and_gamma_zero_reference():
    m = model()
    p, d, y = packet(96)
    with torch.no_grad():
        for tree in m.sources:
            for block in tree.blocks.values():
                block.V.normal_(0, 1)
                block.U.normal_(0, 8)
    own = lambda: iter([dict(packet=p, domain=d, y=y)])
    foreign = lambda: iter([dict(packet=p, domain=d)])
    records = project_sources(m, own, foreign)
    assert len(records) == 3
    assert any(record['gamma'] < 1 for record in records)
    for record in records:
        assert record['maximum_residual_rms_after'] <= .5+1e-8
        assert np.max(np.array(record['projected_logloss'])-record['baseline_logloss']) <= .00100002
    with torch.no_grad():
        for source, tree in enumerate(m.sources):
            for block in tree.blocks.values():
                block.U.zero_()
            expected = p['z']+F.linear(p['phi'][:, source], m.teacher_w[source], m.teacher_b[source])
            assert torch.allclose(m.owner_logits(p, source), expected, atol=1e-6, rtol=0)


def test_switches_source_mask_and_no_foreign_write():
    cfg = load_config()
    p, d, y = packet()
    for switch in ('GAIN_CLUSTER', 'GAIN_RELATION', 'DYNAMIC_SPLIT', 'CONTRASTIVE', 'SOURCE_WRITE', 'LOW_RANK'):
        arm = copy.deepcopy(arms()['FULL_DENSE'])
        arm['switches'][switch] = False
        if switch == 'GAIN_CLUSTER':
            arm['aux_matched'] = True
        m = make_model(cfg, arm, seed=2022)
        out = m(p, d)
        assert torch.isfinite(out.logits).all()
        if switch == 'GAIN_RELATION':
            assert torch.equal(out.logits, out.reference_logits)
    arm = copy.deepcopy(arms()['STATIC_LOCAL'])
    arm['switches']['CONTRASTIVE'] = True
    with pytest.raises(ValueError):
        make_model(cfg, arm)
    m = make_model(cfg, arms()['CLUSTER_ONLY'])
    with pytest.raises(ValueError):
        m.add_structure('SPLIT', dict(kind='SOURCE_SPLIT', domain=0, parent=0))


def test_same_x_opposite_labels_do_not_make_deployable_partition():
    n = 256
    features = np.ones((n, 4))
    responses = np.tile(np.array([[-1.], [1.]]), (n//2, 1))
    folds = np.tile([0, 0, 1, 1], n//4)
    partition, diagnostic = conditional_partition(features, responses, np.ones(n), folds, 2022, load_config()['split'])
    assert partition is None and diagnostic['oof_skill'] <= 0
    g1, g2 = np.array([1., 2.]), np.array([1., -2.])
    average = (g1+g2)/2
    assert g1@g2 < 0 < min(g1@average, g2@average)


def test_source_gradient_sketch_matches_autograd():
    m = model()
    nontrivial(m)
    m.flags['SOURCE_WRITE'] = True
    p, d, y = packet(6)
    out = m(p, d)
    sketch, mass = source_gradient_sketch(m, out, p, d, y, 0, 0, 2022)
    generator = torch.Generator().manual_seed(2022)
    j = (torch.randint(0, 2, (16, 256), generator=generator)*2-1).float()/4
    for i in (1, 2, 4, 5):
        for task in range(2):
            gradient = torch.autograd.grad(F.binary_cross_entropy_with_logits(out.logits[i, task], y[i, task]),
                                           m.sources[0].blocks['s0'].U, retain_graph=True)[0]
            assert np.allclose(sketch[i, task*16:(task+1)*16], (j@gradient.flatten()).detach().numpy(), atol=1e-7, rtol=1e-5)


def test_transaction_wide_win_has_no_split_claim_and_one_panel_read():
    def metric(ll, added, matched=True):
        return dict(macro_logloss=ll, local_logloss=ll, local_mass=.5,
                    cells=[dict(logloss=ll)]*6, added_parameters=added, capacity_match=matched)
    metrics = dict(KEEP=metric(.5, 0), READER=metric(.498, 10), WIDE=metric(.497, 148), SPLIT=metric(.497, 148))
    assert decide(metrics, load_config())['winner'] == 'WIDE'
    calls = []
    callbacks = dict(fork=lambda snapshot, variant, proposal: dict(variant=variant, step=snapshot['step']),
                     train=lambda branch, n: branch.update(step=branch['step']+n), project=lambda branch: None,
                     evaluate=lambda branches: calls.append('S') or metrics,
                     capture=lambda branch: copy.deepcopy(branch))
    original = dict(step=7)
    result = run_transaction(original, {}, callbacks, load_config())
    assert result['winner'] == 'WIDE' and not result['decision']['split_accepted']
    assert result['snapshot']['step'] == 15 and original['step'] == 7
    assert calls == ['S']


def test_low_rank_mask_does_not_manufacture_observations():
    values = torch.tensor([[1., 2., 0.], [2., 4., 0.], [3., 6., 0.]])
    weight = torch.tensor([[1., 1., 0.]]).expand_as(values)
    reconstructed = weighted_low_rank(values, weight)
    assert torch.isfinite(reconstructed).all()
    assert (reconstructed[:, :2]-values[:, :2]).abs().max() < .01
    table = GainTable(1, 2, 2)
    gain = torch.ones(10, 1, 2)
    table.measure(gain, torch.ones(10, 2)/2, np.arange(10), minimum_ess=64, low_rank=True)
    assert not table.observed.any() and not table.policy_gain.any()


def test_asymmetric_source_split_refresh_and_true_capacity_cost():
    m = model()
    nontrivial(m)
    p, d, y = packet(96)
    proposal = dict(kind='SOURCE_SPLIT', domain=0, parent=0)
    wide = copy.deepcopy(m)
    wide_cost = wide.add_structure('WIDE', proposal)
    split_cost = m.add_structure('SPLIT', proposal)
    assert split_cost['added_macs'] == 1490
    assert wide_cost['capacity_match']
    cache = refresh_teacher(m, p, d, y, np.arange(len(y)), panels=np.arange(len(y)) % 2, freeze_scale=True)
    assert cache.benefits.shape[-1] == 5
    assert len(m.action_maps[0]) == 4 and len(m.action_maps[1]) == 5
    local = make_model(load_config(), arms()['STATIC_LOCAL'])
    calls = []
    handles = [tree.blocks['s0'].register_forward_hook(lambda *args: calls.append(True)) for tree in local.sources]
    local(p, d)
    assert not calls
    measure_benefits(local, p, d, y)
    assert calls
    for handle in handles:
        handle.remove()
