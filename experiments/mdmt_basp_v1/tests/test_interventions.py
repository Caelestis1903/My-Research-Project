"""The finite reading diagnostic's action, selection and isolation acceptance."""
import copy
import json
import types
import numpy as np
import torch
from experiments.mdmt_basp_v1.interventions import (
    fixed_logits, reader_logits, select_readapt_path, run_read_interventions, ram_equal, _score_rows)
from experiments.mdmt_basp_v1.metrics import evaluate
from experiments.mdmt_basp_v1.config import load_config, arms
from experiments.mdmt_basp_v1.roles import Data
from experiments.mdmt_basp_v1.trainer import Trainer


def test_off_mass_moves_to_null_and_root_stays_phi_during_updates():
    torch.set_num_threads(1)
    path = (0, 1, 's0a')
    reader = torch.nn.Module()
    reader.readers = torch.nn.ParameterDict({'d0s1': torch.nn.Parameter(torch.tensor([[1., 0.]]))})
    item = dict(rows=torch.tensor([0]), coefficient=torch.tensor([[.25]]),
                fixed_coefficient=torch.tensor([[.2]]), correction=torch.tensor([[4.]]),
                root_correction=torch.tensor([[2.]]), values=torch.tensor([[4., 0.]]),
                phi=torch.tensor([[2., 0.]]))
    cache = dict(reference=torch.tensor([[1.], [3.]]), keep=torch.tensor([[2.], [3.]]),
                 paths={path: item})
    assert torch.equal(fixed_logits(cache, 'OFF_PATH', path), torch.tensor([[1.], [3.]]))
    assert torch.equal(fixed_logits(cache, 'ROOT_PATH', path), torch.tensor([[1.5], [3.]]))
    assert torch.allclose(fixed_logits(cache, 'FIXED_RELATION_FIXED'), torch.tensor([[1.8], [3.]]))
    assert torch.equal(fixed_logits(cache, 'ALL_NULL_FIXED'), cache['reference'])
    optimizer = torch.optim.Adam(reader.parameters(), lr=.001)
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        off = reader_logits(reader, cache, 'OFF_PATH_READAPT', path)
        root = reader_logits(reader, cache, 'ROOT_PATH_READAPT', path)
        assert not off.requires_grad
        assert torch.allclose(root[0], 1+.25*2*reader.readers['d0s1'][0, 0])
        root.sum().backward()
        optimizer.step()


def test_f_selection_uses_natural_domain_mass_and_stable_ties():
    first, second = (0, 1, 's0'), (1, 0, 's0')
    cache = dict(reference=torch.zeros(2, 2), keep=torch.ones(2, 2), domain=torch.tensor([0, 1]),
        paths={first: dict(rows=torch.tensor([0]), coefficient=torch.ones(1, 2), correction=torch.ones(1, 2)),
               second: dict(rows=torch.tensor([1]), coefficient=torch.ones(1, 2), correction=torch.ones(1, 2)*.5)})
    y, mask = np.ones((2, 2)), np.ones((2, 2), bool)
    chosen, scores = select_readapt_path(cache, y, mask, [1, 9])
    assert chosen == second
    assert scores[0]['score']/scores[0]['natural_domain_mass'] > scores[1]['score']/scores[1]['natural_domain_mass']
    before = cache['keep'].numpy()
    prediction = before.copy()
    prediction[0] = 0
    baseline = evaluate(y, before, cache['domain'].numpy(), mask, domains=2)
    baseline['_loss'] = np.logaddexp(0., before)-y*before
    stub = types.SimpleNamespace(D=2, T=2, needs=[])
    global_row = _score_rows({}, 'F_probe', 'OFF_PATH', first, y, mask, prediction,
        cache['domain'].numpy(), np.ones((2, 1)), stub, baseline, natural_domain_counts=[1, 9])[0]
    ll = np.logaddexp(0., prediction.astype(np.float64))-y*prediction
    expected = .1*ll[0].mean()+.9*ll[1].mean()
    assert np.isclose(global_row['natural_logloss'], expected)
    assert not np.isclose(global_row['natural_logloss'], ll.mean())
    assert np.isclose(global_row['delta_natural_logloss'], .1*(ll[0]-baseline['_loss'][0]).mean())
    qrow = _score_rows({}, 'Q', 'OFF_PATH', first, y, mask, prediction,
        cache['domain'].numpy(), np.ones((2, 1)), stub, baseline)[0]
    assert np.isclose(qrow['natural_logloss'], ll.mean())
    cache['paths'][second]['correction'].fill_(1)
    assert select_readapt_path(cache, y, mask, [1, 1])[0] == first
    cache['paths'][first]['correction'].zero_()
    cache['paths'][second]['correction'].zero_()
    assert select_readapt_path(cache, y, mask, [1, 1])[0] is None


def test_three_original_adam_branches_lock_before_q_and_leave_mother_unchanged(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(2022)
    data = Data.__new__(Data)
    data.dataset = 'movielens'
    data.roles = {role: np.arange(i*18, (i+1)*18) for i, role in enumerate(('T0', 'F', 'S', 'Q', 'V', 'E'))}
    total = 108
    data.domain = np.arange(total) % 3
    data.x = np.zeros((total, 8), np.float32)
    data._y = np.ones((total, 2), np.float32)
    data.probe = data.roles['F'].copy()
    data.folds = np.arange(18) % 2
    data.owner_weights = np.ones(3)/3
    data.access_log = None
    data.used_s, data.read_s = set(), set()
    data.final_authorization = None
    data.s_windows = [v.copy() for v in np.array_split(data.roles['S'], 3)]
    views = {role: dict(ids=ids, domain=data.domain[ids], h=np.zeros((18, 256), np.float32),
                       z=np.zeros((18, 2), np.float32), phi=np.ones((18, 3, 64), np.float32))
             for role, ids in data.roles.items()}
    source = torch.nn.Module()
    source.register_buffer('teacher_w', torch.zeros(3, 2, 64))
    source.register_buffer('teacher_b', torch.zeros(3, 2))
    trainer = Trainer(arms()['FIXED_READ'], data, views, dict(source=source, native=torch.nn.Identity()),
                      2022, load_config(), tmp_path/'trainer', 'cpu')
    trainer.state['epoch'] = 3
    trainer.model.set_epoch(3)
    with torch.no_grad():
        trainer.model.readers['d0s1'][:, 0] = .8
    trainer.refresh(2, freeze_scale=True)
    trainer.step()
    trainer.selected = trainer.snapshot()
    trainer.final = copy.deepcopy(trainer.selected)
    before = trainer.snapshot()
    original = data.labels
    qreads = []
    def labels(self, role, purpose, ids=None, **context):
        if role == 'Q':
            plan = json.loads((trainer.root/'interventions'/'selected'/'INTERVENTION_PLAN.json').read_text())
            assert plan['status'] == 'BRANCHES_LOCKED_Q_PENDING'
            assert plan['selected_path'] is not None
            assert [branch['branch'] for branch in plan['branches']] == ['KEEP_READAPT', 'OFF_PATH_READAPT', 'ROOT_PATH_READAPT']
            assert all(branch['attempted_steps'] == 8 and branch['frozen_after_eight_steps'] for branch in plan['branches'])
            assert all(branch['same_original_adam_state'] for branch in plan['branches'])
            qreads.append(role)
        assert role not in ('S', 'E', 'V')
        return original(role, purpose, ids, **context)
    data.labels = types.MethodType(labels, data)
    result = run_read_interventions(trainer, 'selected')
    assert result['status'] == 'COMPLETE'
    assert result['mother_state_parity'] == 'PASS'
    assert result['fixed_actions_per_role'] == 27
    assert result['cost']['attempted_reader_steps'] == 24
    assert qreads == ['Q']
    assert ram_equal(before, trainer.snapshot())
    expected_plan = result['shared_next_F_row_plan']
    assert len(expected_plan) == 8
    initial_steps = [branch['original_reader_adam_steps'] for branch in result['branches']]
    assert initial_steps[0] == initial_steps[1] == initial_steps[2]
    for branch in result['branches']:
        assert all(name.startswith('readers.') for step in branch['updated_parameters'] for name in step)
    reused = run_read_interventions(trainer, 'final')
    assert reused['reused_from'] == 'selected'
    assert reused['cost']['additional_reader_steps'] == 0
    assert qreads == ['Q']
    assert ram_equal(before, trainer.snapshot())
    assert not any(path.suffix in ('.pt', '.pth', '.pkl') for path in tmp_path.rglob('*'))
