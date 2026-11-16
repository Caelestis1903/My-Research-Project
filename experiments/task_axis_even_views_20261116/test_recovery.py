"""CPU recovery and RESOURCE-1 admission/provenance fixtures; mocked IPC only."""
import copy
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import pytest
from . import protocol as p, runner as r


def test_acknowledged_continuation():
    with TemporaryDirectory() as directory, ExitStack() as stack:
        attempt = Path(directory).resolve()
        stack.enter_context(patch.object(p, 'ATTEMPT', attempt))
        manifest = dict(source_hash='f'*64, design_bundle_sha256=p.BUNDLE, cache_dir='unused')
        p.write_json(attempt/'run_manifest.json', manifest)
        j = p.job('movielens', 'AXIS-EVEN', 2022, manifest['source_hash']); key = j['key']
        p.write_json(attempt/'jobs'/f'{key}.json', j)
        template = torch.nn.Linear(2, 1)
        initial_hash = p.state_digest(template.state_dict())
        model = copy.deepcopy(template); opt = p.optimizer(model, j['config'])
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        opt.step()
        loader = torch.Generator().manual_seed(13022)
        owner = dict(owner_tool_session='fixture-parent')
        status = dict(jobs={}, retained_bytes=0, retained_states=1)
        stack.enter_context(patch.object(r, 'validate_attempt', return_value=manifest))
        stack.enter_context(patch.object(p, 'effective_run_manifest', return_value=manifest))
        stack.enter_context(patch.object(r, 'require_storage', return_value=(status, owner)))
        stack.enter_context(patch.object(r, 'storage_status', return_value=(status, {})))
        stack.enter_context(patch.object(r, 'live_worker_pids', return_value=[]))
        stack.enter_context(patch.object(r, 'stop_requested', return_value=False))
        ipc = stack.enter_context(patch.object(r, 'request'))
        unlink = stack.enter_context(patch.object(Path, 'unlink'))
        receipt_write = stack.enter_context(patch.object(r, 'immutable'))
        output_write = stack.enter_context(patch.object(r, 'write_json'))
        stack.enter_context(patch.object(r, 'assert_old_unchanged', return_value={}))

        def fixture(epoch):
            step = epoch*j['config']['steps_per_epoch']
            optimizer = p.cpu_copy(opt.state_dict())
            for value in optimizer['state'].values():
                value['step'].fill_(step)
            values = dict(online=dict(macro_auc=.6), fixed_ema=dict(macro_auc=.61))
            state = p.cpu_copy(dict(job=j, model=model.state_dict(), ema=model.state_dict(),
                optimizer=optimizer, rng=p.rng_state('cpu'), loader=loader.get_state(),
                epoch=epoch, global_step=step, selected_models={k:model.state_dict() for k in values},
                selected_epochs={k:epoch for k in values}, selected_validation=values,
                common_validation={str(e):values for e in (10, 30, 50) if e <= epoch},
                best={k:v['macro_auc'] for k,v in values.items()},
                counts=dict(optimizer_updates=step, ema_updates=step), initial_hash=initial_hash,
                peak_allocated_bytes=1024, peak_reserved_bytes=2048, physical_worker_seconds=12.5))
            record = {name:state[name] for name in ('epoch', 'global_step', 'counts',
                'selected_epochs', 'peak_allocated_bytes', 'peak_reserved_bytes')}
            record.update({f'{name}_hash':p.state_digest(state[name])
                           for name in ('model', 'ema', 'optimizer', 'rng', 'loader')})
            record.update(selected_hashes={k:p.state_digest(v) for k,v in state['selected_models'].items()},
                source_hash=manifest['source_hash'], design_bundle_sha256=p.BUNDLE,
                continuation_hash=p.state_digest(state))
            path = attempt/'runs'/key/'attempt1'/f'epoch_{epoch:02d}.json'
            p.write_json(path, record)
            ack = dict(key=key, epoch=epoch, attempt=1, epoch_file=str(path), status='RAM_COMMITTED')
            (attempt/'epoch_commit_ledger.jsonl').write_text(json.dumps(ack)+'\n')
            status['jobs'][key] = dict(job=j, epoch=epoch, counts=state['counts'].copy(), status='PAUSED')
            # A later, unacknowledged record must never replace the ledger's receipt.
            p.write_json(attempt/'runs'/key/'attempt99'/path.name, dict(epoch=epoch, continuation_hash='bad'))
            return state, record, path, ack

        def recover(state, job_status='PAUSED', invalid=False):
            status['jobs'][key]['status'] = job_status
            ipc.reset_mock(); unlink.reset_mock()
            ipc.side_effect = lambda _, command, **kw: (dict(state=state) if command == 'get'
                else dict(status='RECOVERY_READY', epoch=status['jobs'][key]['epoch']))
            if invalid:
                try:
                    r.recover(attempt, key)
                except (AssertionError, KeyError, FileNotFoundError):
                    pass
                else:
                    raise AssertionError('Invalid continuation accepted')
                assert [c.args[1] for c in ipc.call_args_list] == ['get']
                unlink.assert_not_called()
            else:
                assert r.recover(attempt, key)['status'] == 'RECOVERY_READY'
                assert [c.args[1] for c in ipc.call_args_list] == (['get'] if job_status == 'PAUSED' else ['get', 'recover'])
                assert unlink.call_count == (job_status == 'PAUSED')

        status['jobs'][key] = dict(job=j, epoch=0, status='PAUSED')
        recover(None)
        recover(dict(job=j, epoch=0), invalid=True)
        state, record, path, ack = fixture(1)
        p.validate_epoch_state(state, record)
        mutations = [
            ('model', 'weight'), ('ema', 'weight'), ('optimizer', 'state', 0, 'exp_avg'),
            ('optimizer', 'state', 0, 'exp_avg_sq'), ('optimizer', 'state', 0, 'step'),
            ('optimizer', 'param_groups', 0, 'lr'), ('rng', 'cpu'), ('rng', 'numpy', 1),
            ('rng', 'python'), ('loader',), ('selected_models', 'online', 'weight'),
            ('selected_models', 'fixed_ema', 'weight'), ('epoch',), ('global_step',),
            ('counts', 'optimizer_updates'), ('counts', 'ema_updates'), ('selected_epochs', 'online'),
            ('selected_validation', 'online', 'macro_auc'), ('best', 'online'),
            ('common_validation',), ('initial_hash',), ('physical_worker_seconds',),
            ('peak_reserved_bytes',), ('job', 'source_hash')]
        for job_status in ('PAUSED', 'FAILED', 'RUNNING'):
            recover(state, job_status)
            for keys in mutations:
                changed = p.cpu_copy(state); target = changed
                for name in keys[:-1]:
                    target = target[name]
                value = target[keys[-1]]
                if torch.is_tensor(value) or hasattr(value, 'flat'):
                    value.reshape(-1)[0] += 1
                elif isinstance(value, tuple):
                    target[keys[-1]] = (value[0]+1, *value[1:])
                else:
                    target[keys[-1]] = 'changed' if isinstance(value, (dict, str, type(None))) else value+1
                recover(changed, job_status, invalid=True)
            recover(dict(job=j, epoch=1), job_status, invalid=True)
            recover(None, job_status, invalid=True)
        for field in ('continuation_hash', 'source_hash', 'design_bundle_sha256', 'model_hash',
                      'selected_hashes', 'selected_epochs', 'counts', 'global_step', 'epoch'):
            changed = dict(record); changed.pop(field)
            p.write_json(path, changed); recover(state, invalid=True)
        p.write_json(path, record)
        for changed_ack in (dict(ack, epoch=2), dict(ack, epoch_file=str(path.parent/'missing.json'))):
            (attempt/'epoch_commit_ledger.jsonl').write_text(json.dumps(changed_ack)+'\n')
            recover(state, invalid=True)
        (attempt/'epoch_commit_ledger.jsonl').write_text('')
        recover(state, invalid=True)

        state, record, path, ack = fixture(50)
        recover(state)
        p.validate_continuation(attempt, key, state, status['jobs'][key])
        receipt_write.reset_mock(); output_write.reset_mock()
        r.attach(attempt, 'fixture-parent'); r.handoff(attempt, 'fixture-parent', accept=True)
        assert receipt_write.call_count == output_write.call_count == 1
        bad = p.cpu_copy(state); bad['model']['weight'].add_(1)
        ipc.side_effect = lambda *_a, **_kw: dict(state=bad)
        for boundary in (lambda:r.attach(attempt, 'fixture-parent'),
                         lambda:r.handoff(attempt, 'fixture-parent', accept=True)):
            receipt_write.reset_mock(); output_write.reset_mock()
            try:
                boundary()
            except AssertionError:
                pass
            else:
                raise AssertionError('Invalid reattachment accepted')
            receipt_write.assert_not_called(); output_write.assert_not_called()

        p.write_json(attempt/'reused_reference_index.json', dict(initial_hashes={'movielens/2022':initial_hash}))
        stack.enter_context(patch.object(r, 'parent_alive', return_value=True))
        stack.enter_context(patch.object(r, 'pipe_lost', return_value=False))
        stack.enter_context(patch.object(r, 'install_signals'))
        stack.enter_context(patch.object(Path, 'touch'))
        data = stack.enter_context(patch.object(r, 'dataset_cache', return_value=(None, {})))
        stack.enter_context(patch.object(r, 'setup'))
        make_model = stack.enter_context(patch.object(r, 'make_model', side_effect=lambda *_:copy.deepcopy(template)))
        restore = stack.enter_context(patch.object(r, 'restore_rng'))
        stack.enter_context(patch.object(torch.cuda, 'reset_peak_memory_stats'))
        train = stack.enter_context(patch.object(r, 'train_step'))
        for invalid in (bad, dict(job=j, epoch=50)):
            ipc.reset_mock(); data.reset_mock(); make_model.reset_mock(); restore.reset_mock()
            ipc.side_effect = lambda _, command, **kw: dict(attempt=2, state=invalid) if command == 'register' else {}
            try:
                r.worker(attempt, key, 'cpu', 123)
            except (AssertionError, KeyError):
                pass
            else:
                raise AssertionError('Invalid worker load accepted')
            data.assert_not_called(); make_model.assert_not_called(); restore.assert_not_called()
            assert [c.args[1] for c in ipc.call_args_list] == ['register', 'failed']
        ipc.reset_mock()
        ipc.side_effect = lambda _, command, **kw: dict(attempt=2, state=state) if command == 'register' else dict(status='COMPLETE')
        r.worker(attempt, key, 'cpu', 123)
        assert [c.args[1] for c in ipc.call_args_list] == ['register', 'complete']
        train.assert_not_called(); restore.assert_called_once()
        result = ipc.call_args_list[-1].kwargs['result']
        assert result['global_step'] == 1000 and result['counts'] == state['counts']
        assert result['final_hashes']['online'] == record['model_hash']


def test_conservative_power_admission(monkeypatch):
    smoke = json.loads((p.ATTEMPT/'SMOKE_REPORT.json').read_text())
    before = copy.deepcopy(smoke)
    budgets = p.conservative_board_power_budgets(smoke)
    assert budgets == {'movielens': 195.61, 'kuairand': 206.31} and smoke == before
    monkeypatch.setattr(r, 'host_admissible', lambda _: True)
    loaded = dict(index=0, memory=100, total=24000, power=283, limit=350)
    idle = dict(loaded, index=1, power=20)
    assert r.choose_device([0], 1024, budgets['kuairand'], rows=[loaded]) is None
    assert r.choose_device([1], 1024, budgets['kuairand'], rows=[idle]) == 1
    assert r.choose_device([0, 1], 1024, budgets['kuairand'], rows=[loaded, idle]) == 1
    assert r.choose_device([1], 1024, budgets['kuairand'], pending={1}, rows=[idle]) is None
    assert r.choose_device([1], 0, budgets['kuairand'], rows=[dict(idle, memory=20400)]) is None
    assert r.choose_device([1], 0, 277.5, rows=[idle]) is None
    monkeypatch.setattr(r, 'host_admissible', lambda _: False)
    assert r.choose_device([1], 0, budgets['kuairand'], rows=[idle]) is None


@pytest.fixture
def provenance(tmp_path, monkeypatch):
    design = p.design(); cycle = tmp_path/'cycle_01'; attempt = cycle/'attempt1'
    here = tmp_path/'experiments/task_axis_even_views_20261116'
    here.mkdir(parents=True); attempt.mkdir(parents=True)
    prefix = str(here.relative_to(tmp_path))+'/'
    old_files = []
    for name in ('runner.py', 'protocol.py', 'test_recovery.py'):
        original = p.CYCLE/'BEFORE_SOURCE_ARCHIVE'/prefix/name
        before = cycle/'BEFORE_SOURCE_ARCHIVE'/prefix/name
        before.parent.mkdir(parents=True, exist_ok=True); before.write_bytes(original.read_bytes())
        old_files.append(dict(path=prefix+name, sha256=p.sha_file(before)))
        (here/name).write_bytes((p.HERE/name).read_bytes())
    (here/'scientific.py').write_text('SCIENTIFIC_DEPENDENCY = 1\n')
    old_files.append(dict(path=prefix+'scientific.py', sha256=p.sha_file(here/'scientific.py')))
    old_files.sort(key=lambda row: row['path']); bootstrap = p.digest(old_files)

    def source_manifest():
        files = [dict(path=str(path.relative_to(tmp_path)), sha256=p.sha_file(path))
                 for path in sorted(here.glob('*.py'))]
        return dict(files=files, sha256=p.digest(files), design_bundle_sha256=p.BUNDLE,
                    selection_hash=p.digest(p.SELECTION))

    original_review = cycle/'ORIGINAL_REVIEW.json'
    p.write_json(original_review, dict(verdict='PASS', source_hash=bootstrap))
    index = dict(entries=[]); environment = dict(fixture='CPU_ONLY')
    p.write_json(attempt/'reused_reference_index.json', index)
    p.write_json(attempt/'source_receipt.json', dict(files=old_files, sha256=bootstrap,
        design_bundle_sha256=p.BUNDLE, selection_hash=p.digest(p.SELECTION)))
    p.write_json(attempt/'run_manifest.json', dict(source_hash=bootstrap, design_bundle_sha256=p.BUNDLE,
        review_path=str(original_review), review_sha256=p.sha_file(original_review),
        controls_index_hash=p.digest(index), environment=environment, test_accessed=False))
    smoke = json.loads((p.ATTEMPT/'SMOKE_REPORT.json').read_text()); smoke['source_hash'] = bootstrap
    p.write_json(attempt/'SMOKE_REPORT.json', smoke)
    endpoint = dict(root=str(attempt), pid=12345, socket='/tmp/resource-1-fixture.sock')
    endpoint['endpoint_hash'] = p.digest([endpoint['root'], endpoint['pid'], endpoint['socket']])
    owner = dict(endpoint, owner_tool_session='fixture-parent', source_hash=bootstrap,
        design_bundle_sha256=p.BUNDLE, ownership='FOREGROUND_PARENT_CURRENT_CONVERSATION', test_accessed=False)
    p.write_json(attempt/'STORAGE_OWNER.json', owner)
    p.write_json(attempt/'guardian.json', dict(pid=endpoint['pid'], address=endpoint['socket']))
    p.write_json(attempt/'STORAGE_HANDOFF.json', dict(owner, jobs=[]))
    p.write_json(attempt/'PARENT_STORAGE_ACCEPTANCE.json', dict(owner, jobs=[], parent_accepted=True))
    pins = {name: p.sha_file(attempt/name) for name in p.RESOURCE_1_ORIGINALS}
    for name, value in dict(ROOT=tmp_path, HERE=here, CYCLE=cycle, ATTEMPT=attempt,
                            RESOURCE_1_SOURCE=bootstrap, RESOURCE_1_ORIGINALS=pins).items():
        monkeypatch.setattr(p, name, value)
    monkeypatch.setattr(p, 'source_manifest', source_manifest)
    monkeypatch.setattr(p, 'design', lambda: design)
    monkeypatch.setattr(p, 'environment_identity', lambda: environment)
    monkeypatch.setattr(p, 'assert_old_unchanged', lambda _: {})
    monkeypatch.setattr(r, 'environment_identity', lambda: environment)
    monkeypatch.setattr(r, 'assert_old_unchanged', lambda _: {})
    monkeypatch.setattr(r, 'stop_requested', lambda: False)
    status = dict(jobs={}, retained_states=0, retained_bytes=0)
    monkeypatch.setattr(r, 'storage_status', lambda _: (status, endpoint))
    monkeypatch.setattr(r, 'live_worker_pids', lambda *_: [])
    review = cycle/'FRESH_REVIEW.json'
    p.write_json(review, dict(verdict='PASS', blocking_issues=[], source_hash=source_manifest()['sha256'],
        design_bundle_sha256=p.BUNDLE, cpu_checks_passed=True,
        resource_1_evidence_reuse=p.resource_1_evidence(attempt)))
    return dict(attempt=attempt, here=here, review=review, owner=owner, status=status, pins=pins)


def test_reviewed_supersession_and_original_receipts(provenance):
    f = provenance; attempt = f['attempt']
    with pytest.raises(AssertionError):
        p.effective_run_manifest(attempt)
    adopted = r.adopt_resource_1(attempt, f['review'], 'fixture-parent')
    manifest = p.validate_attempt(attempt, require_smoke=True)
    assert adopted['effective_source_hash'] == manifest['source_hash'] == p.source_manifest()['sha256']
    assert manifest['storage_bootstrap_source_hash'] == manifest['smoke_source_hash'] == p.RESOURCE_1_SOURCE
    assert manifest['power_budgets_watts'] == {'movielens': 195.61, 'kuairand': 206.31}
    j = p.job('movielens', 'AXIS-EVEN', 2022, manifest['source_hash'])
    record = dict(job=j, epoch=0)
    p.validate_continuation(attempt, j['key'], None, record)
    f['status']['jobs'][j['key']] = record
    assert r.require_storage(attempt)[1] == f['owner']
    f['status']['jobs'].clear()
    for accept in (False, True):
        handoff = r.handoff(attempt, 'fixture-parent', accept=accept)
        assert handoff['source_hash'] == p.RESOURCE_1_SOURCE
        assert handoff['source_hash_verified'] == manifest['source_hash']
        assert handoff['supersession_receipt_sha256'] == p.sha_file(attempt/p.RESOURCE_1_RECEIPT)
    assert (attempt/'RESOURCE_1_STORAGE_HANDOFF.json').exists()
    assert (attempt/'RESOURCE_1_PARENT_STORAGE_ACCEPTANCE.json').exists()
    assert {name: p.sha_file(attempt/name) for name in f['pins']} == f['pins']
    with pytest.raises(AssertionError, match='exactly one'):
        r.adopt_resource_1(attempt, f['review'], 'fixture-parent')


@pytest.mark.parametrize('mutation', ('smoke', 'scientific', 'worker', 'endpoint', 'reuse', 'review_source'))
def test_evidence_changes_refused(provenance, mutation):
    f = provenance; attempt = f['attempt']
    if mutation in ('smoke', 'endpoint'):
        path = attempt/('SMOKE_REPORT.json' if mutation == 'smoke' else 'STORAGE_OWNER.json')
        data = json.loads(path.read_text()); data['tampered'] = True; p.write_json(path, data)
    elif mutation == 'scientific':
        (f['here']/'scientific.py').write_text('SCIENTIFIC_DEPENDENCY = 2\n')
    elif mutation == 'worker':
        path = f['here']/'runner.py'; path.write_text(path.read_text().replace('def worker(', 'def altered_worker(', 1))
    else:
        review = json.loads(f['review'].read_text())
        if mutation == 'reuse':
            review.pop('resource_1_evidence_reuse')
        else:
            review['source_hash'] = p.RESOURCE_1_SOURCE
        p.write_json(f['review'], review)
    with pytest.raises(AssertionError):
        r.adopt_resource_1(attempt, f['review'], 'fixture-parent')
    assert not (attempt/p.RESOURCE_1_RECEIPT).exists()


@pytest.mark.parametrize('activity', ('dispatch.json', 'progress.json', 'launch_ledger.jsonl',
    'epoch_commit_ledger.jsonl', 'jobs/key.json', 'runs/key', 'registered_key', 'live_worker'))
def test_pre_formal_only(provenance, monkeypatch, activity):
    f = provenance; attempt = f['attempt']
    if activity == 'registered_key':
        f['status']['jobs']['formal-key'] = {}
    elif activity == 'live_worker':
        monkeypatch.setattr(r, 'live_worker_pids', lambda *_: [98765])
    else:
        path = attempt/activity; path.parent.mkdir(parents=True, exist_ok=True); path.touch()
    with pytest.raises(AssertionError):
        r.adopt_resource_1(attempt, f['review'], 'fixture-parent')
    assert not (attempt/p.RESOURCE_1_RECEIPT).exists()


@pytest.mark.parametrize('mutation', ('budget', 'review', 'removed_receipt'))
def test_effective_manifest_cannot_downgrade(provenance, mutation):
    f = provenance; attempt = f['attempt']
    r.adopt_resource_1(attempt, f['review'], 'fixture-parent')
    path = attempt/p.RESOURCE_1_RECEIPT
    if mutation == 'budget':
        receipt = json.loads(path.read_text()); receipt['run_manifest']['power_budgets_watts']['kuairand'] = 0
        p.write_json(path, receipt)
    elif mutation == 'review':
        review = json.loads(f['review'].read_text()); review['cpu_checks_passed'] = False
        p.write_json(f['review'], review)
    else:
        path.unlink()
    with pytest.raises(AssertionError):
        p.effective_run_manifest(attempt)
        assert result['optimizer_hash'] == record['optimizer_hash']


if __name__ == '__main__':
    test_acknowledged_continuation()
    print('PASS: acknowledged continuation, mutation refusals, attachment/load gates, epoch50 without updates')
