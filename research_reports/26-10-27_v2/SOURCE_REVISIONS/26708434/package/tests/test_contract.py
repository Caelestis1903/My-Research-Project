import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.mdmt_relation_basis_v2.config import DEFAULT, METHODS, load_config
from experiments.mdmt_relation_basis_v2.losses import masked_bce
from experiments.mdmt_relation_basis_v2.metrics import evaluate
from experiments.mdmt_relation_basis_v2.model import Model
from experiments.mdmt_relation_basis_v2.relation import Relation
from experiments.mdmt_relation_basis_v2.state import capture, restore, state_hash, configure
from experiments.mdmt_relation_basis_v2.data import Data
from experiments.mdmt_relation_basis_v2.source_functions import SourceFunctions
from experiments.mdmt_relation_basis_v2.vendor_snapshot.mdmt_dfd_20261026.functions import PilotBank
from experiments.mdmt_relation_basis_v2.vendor_snapshot.mdmt_dfd_20261026.native_adapter import make_native


@pytest.fixture(autouse=True)
def cpu():
    configure()
    torch.manual_seed(2022)


def batch(n=6):
    return (torch.randn(n, 256), torch.randn(n, 2), torch.arange(n) % 3,
            torch.randn(n, 3, 64), (torch.arange(n*2).reshape(n, 2) % 2).float())


def test_config(tmp_path):
    cfg = load_config()
    assert cfg['seeds']['development'] == [2022, 2023, 2024, 2025]
    for text in [DEFAULT.read_text()+'\nmethods: [DIRECT]\n',
                 DEFAULT.read_text().replace('h_dim: 256', 'h_dim: 255'),
                 DEFAULT.read_text().replace('  - DIRECT', '  - UNKNOWN'),
                 DEFAULT.read_text().replace('  phi_dim: 64\n', ''),
                 DEFAULT.read_text()+'\nunknown: 1\n']:
        path = tmp_path/'bad.yaml'
        path.write_text(text)
        with pytest.raises(ValueError):
            load_config(path)


def test_parameters_initialization_and_routes():
    expected = [49926, 49926, 50718, 62406, 62166, 62846]
    h, z, d, phi, y = batch()
    for method, number in zip(METHODS, expected):
        model = Model(method)
        assert model.parameter_counts()['adaptation'] == number
        assert torch.equal(model(h, z, d, phi), z)
        if model.reads_source:
            alpha, null = model.relation(torch.cat((h, z), -1), d)
            assert alpha.shape == (6, 2, 2, 2)
            torch.testing.assert_close(alpha.sum((-1, -2))+null, torch.ones(6, 2), atol=1e-6, rtol=0)
            for target in range(3):
                assert target not in model.relation.foreign[target].tolist()
            with torch.no_grad():
                model.w.normal_()
            torch.testing.assert_close(model(h, z, d, phi, force_null=True), z+model.target_residual(torch.cat((h, z), -1), d))
        if model.forms_basis:
            for source in range(3):
                torch.testing.assert_close(model.function(phi[:, source], source), phi[:, source, None, :].expand(-1, 2, -1), atol=1e-7, rtol=0)
            assert not torch.equal(model.V[:, 0], model.V[:, 1])
    assert Model('LOWRANK_RELATION').parameter_counts()['relation'] == 112
    assert Model('FULL_RELATION').parameter_counts()['relation'] == 792
    assert Model('LOWRANK_RELATION', basis_count=1)(h, z, d, phi).shape == (6, 2)


def test_gradients_warmup_projection_and_small_batches():
    h, z, d, phi, y = batch()
    model = Model('LOWRANK_RELATION')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model.warmup(True)
    masked_bce(model(h, z, d, phi), y).backward()
    assert model.U.grad is None and model.relation.bias.grad is None
    optimizer.step()
    model.warmup(False)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        masked_bce(model(h, z, d, phi), y).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
        optimizer.step()
    assert model.U.grad.abs().sum() > 0
    assert model.V.grad.abs().sum() > 0
    assert model.relation.target_task.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.w.fill_(10.)
    model.project()
    assert float(model.w.detach().norm(dim=-1).max()) <= 2.000001
    one = model(h[:1], z[:1], d[:1], phi[:1])
    masked_bce(one, y[:1]).backward()
    assert one.shape == (1, 2)


def test_mask_and_logit_metrics():
    z = torch.tensor([[1000., -1000.], [0., 3.]], requires_grad=True)
    y = torch.tensor([[1., 0.], [float('nan'), 1.]])
    mask = torch.tensor([[True, True], [False, True]])
    loss = masked_bce(z, y, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(z[mask], y[mask])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert z.grad[1, 0] == 0
    assert masked_bce(z, y, torch.zeros_like(mask)) is None
    labels = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], float)
    logits = np.array([[-1000, -1], [1000, 2], [-2, 1], [3, -2]], float)
    metrics = evaluate(labels, logits, np.array([0, 0, 1, 1]))
    assert metrics['macro_auc'] == 1.
    assert metrics['cells'][2]['auc'] is None
    expected = (np.logaddexp(0., logits)-labels*logits).mean()
    assert abs(metrics['natural_logloss']-expected) < 1e-15
    wrong = evaluate(np.array([[0., 1.]]), np.array([[1000., -1000.]]), np.array([0]))
    assert wrong['natural_logloss'] == 1000.


def test_ram_restore_next_update_and_identity():
    model = Model('LOWRANK_RELATION')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    h, z, d, phi, y = batch()
    identity = dict(config='a', data='b', row_plan='c', source='d', native='e', preprocessor='f')
    masked_bce(model(h, z, d, phi), y).backward()
    optimizer.step()
    snapshot = capture(model, optimizer, identity, epoch=1, global_step=1, row_cursor=0)
    before = model(h, z, d, phi).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    masked_bce(model(h, z, d, phi), y).backward()
    optimizer.step()
    expected = state_hash(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
    restored = restore(snapshot, model, optimizer, identity)
    assert restored['global_step'] == 1
    torch.testing.assert_close(model(h, z, d, phi), before, atol=1e-6, rtol=0)
    optimizer.zero_grad(set_to_none=True)
    masked_bce(model(h, z, d, phi), y).backward()
    optimizer.step()
    assert state_hash(dict(model=model.state_dict(), optimizer=optimizer.state_dict())) == expected
    with pytest.raises(ValueError, match='identity'):
        restore(snapshot, model, optimizer, dict(identity, config='wrong'))


def test_native_and_source_teacher():
    schema = dict(columns=['feature', 'domain_indicator'], dense_columns=[],
                  vocab_sizes=dict(feature=4, domain_indicator=3), embedding_dim=16)
    native, receipt = make_native(schema, 2022, 'movielens', 'cpu')
    native.eval()
    x = torch.tensor([[1., 0.], [2., 1.], [3., 2.]])
    packet = native.encode(x)
    assert packet.h.shape == (3, 256) and packet.logits.shape == (3, 2)
    probability = native.native(x.T)
    torch.testing.assert_close(packet.logits.sigmoid(), probability, atol=1e-6, rtol=0)
    bank = PilotBank(lambda s, q, k: 2022+s*10+q)
    with torch.no_grad():
        for head in bank.heads.values():
            head.output.weight.normal_(std=.02)
            head.output.bias.fill_(.1)
    h = torch.randn(24, 256)
    raw = torch.stack([torch.cat([torch.relu(bank.head(s, q).hidden(h)) for q in range(2)], -1) for s in range(3)], 1)
    means, stds = raw.mean(0).detach(), raw.std(0, correction=0).clamp_min(.001).detach()
    source = SourceFunctions(bank, means, stds)
    phi = source(h)
    z = torch.randn(24, 2)
    for s in range(3):
        direct = z+torch.stack([bank.head(s, t)(h) for t in range(2)], -1)
        torch.testing.assert_close(source.teacher(phi[:, s], z, s), direct, atol=1e-6, rtol=0)


def test_label_permissions():
    data = Data.__new__(Data)
    data.roles = dict(T0=np.array([0, 1]), F=np.array([2, 3]), V=np.array([4]), E=np.array([5]))
    data.y = np.zeros((6, 2), np.float32)
    data.outer_frozen = None
    data.access_log = None
    for role in ('V', 'E'):
        with pytest.raises(PermissionError):
            data.labels(role, 'adapt')
    with pytest.raises(PermissionError, match='locked'):
        data.labels('E', 'final')
    with pytest.raises(PermissionError):
        data.labels('F', 'adapt', [4])
    with pytest.raises(PermissionError):
        data.authorize_final(dict(status='FROZEN', workers={}), dict(seed=2022))


def test_audit_plan_do_not_train_or_create_files(tmp_path, monkeypatch, capsys):
    from experiments.mdmt_relation_basis_v2.cli import main
    import experiments.mdmt_relation_basis_v2.data as data_module
    monkeypatch.setattr(data_module, 'prepare', lambda *a, **k: pytest.fail('audit/plan started preparation'))
    monkeypatch.chdir(tmp_path)
    main(['--artifact-root', str(tmp_path/'absent'), 'plan'])
    assert not list(tmp_path.iterdir())
    main(['--artifact-root', str(tmp_path/'absent'), 'audit'])
    assert not list(tmp_path.iterdir())


def test_adapter_cpu_pipeline_retains_states_and_defers_E(tmp_path):
    from experiments.mdmt_relation_basis_v2.trainer import train_method
    from experiments.mdmt_relation_basis_v2.diagnostics import diagnose
    data = Data.__new__(Data)
    data.dataset = 'movielens'
    data.roles = {r:np.arange(i*12, (i+1)*12) for i, r in enumerate(('F', 'V', 'Q', 'S', 'E'))}
    data.y = np.tile(np.array([[0., 1.], [1., 0.]], np.float32), (30, 1))
    data.domain = np.tile(np.repeat(np.arange(3), 4), 5)
    data.outer_frozen, data.access_log = None, None
    data.manifest = dict(unit_fixture=True)
    source = SourceFunctions(PilotBank(), np.zeros((3, 64), np.float32), np.ones((3, 64), np.float32))
    views = {}
    for role in ('F', 'V', 'Q', 'S'):
        h, z, _, _, _ = batch(12)
        views[role] = dict(ids=data.roles[role], domain=data.domain[data.roles[role]],
                           h=h.numpy(), z=z.numpy(), phi=source(h).detach().numpy())
    result = train_method('LOWRANK_RELATION', data, views, 2022, load_config(), tmp_path, 'cpu', dict(unit_fixture=True))
    assert result['receipt']['global_steps'] == 20
    assert 5 <= result['receipt']['selected_epoch'] <= 20
    assert result['final']['position']['epoch'] == 20
    assert all(v.device.type == 'cpu' for v in result['selected']['model'].values())
    assert not list(tmp_path.rglob('*.pth'))
    assert not list(tmp_path.rglob('E_*'))
    metrics = diagnose(result['model'], data, views, source, 'cpu')
    assert metrics['status'] == 'COMPLETE'
    assert len(metrics['source_retention']) == 6
    assert len(metrics['basis_output_rms_distance']) == 3
    assert np.isfinite(metrics['all_root_minus_full']['macro_logloss'])

