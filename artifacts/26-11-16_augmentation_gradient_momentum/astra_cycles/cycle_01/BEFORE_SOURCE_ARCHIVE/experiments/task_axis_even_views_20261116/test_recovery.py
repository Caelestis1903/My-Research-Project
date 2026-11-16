"""One CPU recovery fixture; temporary numerical receipts and mocked IPC only."""
import copy
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
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
        assert result['optimizer_hash'] == record['optimizer_hash']


if __name__ == '__main__':
    test_acknowledged_continuation()
    print('PASS: acknowledged continuation, mutation refusals, attachment/load gates, epoch50 without updates')
