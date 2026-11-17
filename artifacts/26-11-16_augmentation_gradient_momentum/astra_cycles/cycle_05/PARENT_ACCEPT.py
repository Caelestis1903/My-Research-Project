"""Direct foreground-parent acceptance; no training or research implementation."""
import argparse
from datetime import datetime, timezone
import json

from experiments.cell_mass_bce_20261117 import protocol as p, runner as r


def engineering():
    a = p.ATTEMPT
    manifest = p.validate_attempt(a)
    status = p.request(a, 'status')
    rows, acks = [], []
    for number, (device, dataset, arm) in enumerate(r.CASES, 1):
        case_id = f'case{number}_gpu{device}_{dataset}'
        paths = list((a / 'smoke' / case_id).glob('attempt*/case.json'))
        assert len(paths) == 1
        case = p.read_json(paths[0])
        key, job = case['key'], case['job']
        record = status['jobs'][key]
        assert (device, dataset, arm) == (case['device'], case['dataset'], case['arm'])
        assert case['source_hash'] == manifest['source_hash']
        assert case['verified_plan_counts'] == p.ENGINEERING_PLAN
        assert record['status'] == 'PAUSED' and record['epoch'] == 2
        assert not r._alive(record.get('worker_pid'))
        got = p.request(a, 'get', key=key)
        state = got['state']
        r._verify_saved_ack(a, job, state)
        proof = r._engineering_proof(a, job, state)
        assert state['counts'] == dict(optimizer_updates=4, ema_updates=4)
        assert state['scientific_epochs_completed'] == 0
        assert state['engineering_state_step'] == 4 and state['engineering_commit_ordinal'] == 2
        assert p.scientific_identity(state) == case['suffix_scientific_identity_hash']
        case_acks = [r._receipt(a / 'acks' / key / f'epoch_{epoch:02d}.json') for epoch in (1, 2)]
        for receipt in case_acks:
            p.receipt_json(receipt)
        acks.extend(case_acks)
        rows.append(dict(case_id=case_id, key=key, case_receipt=r._receipt(paths[0]),
                         proof=case['proof'], profile=case['profile'], acks=case_acks,
                         continuation_hash=p.state_digest(state),
                         scientific_identity_hash=p.scientific_identity(state),
                         counts=state['counts'], scientific_epochs_completed=0,
                         engineering_state_step=4, engineering_commit_ordinal=2,
                         proof_verified=bool(proof)))
        del got, state
    assert status['retained_states'] == 4 and len(acks) == 8
    receipt = dict(status='ACCEPTED', parent_accepted=True, source_hash=manifest['source_hash'],
                   design_bundle_sha256=p.DESIGN_HASH, research_bundle_sha256=p.RESEARCH_HASH,
                   owner=p.owner_binding(a), cases=rows, acks=acks,
                   retained_engineering_keys=[row['key'] for row in rows],
                   retained_states=status['retained_states'], retained_tensor_bytes=status['retained_bytes'],
                   engineering_keys_released=False, resource_review_pending=True,
                   recorded_at_utc=datetime.now(timezone.utc).isoformat(), test_accessed=False)
    p.write_json(a / 'PARENT_ENGINEERING_SUPERVISION_ACCEPTANCE.json', receipt)
    return {key: receipt[key] for key in ('status', 'retained_states', 'retained_tensor_bytes', 'source_hash')}


def formal():
    a = p.ATTEMPT
    manifest = p.validate_attempt(a, True)
    status = p.request(a, 'status')
    rows, acks, total_bytes, updates = [], [], 0, 0
    for dataset in p.DATASETS:
        for arm in p.ARMS:
            for seed in p.SEEDS:
                job = p.job(dataset, arm, seed, manifest['source_hash'])
                key = job['key']
                record = status['jobs'][key]
                assert record['status'] == 'COMPLETE' and record['epoch'] == 50
                assert not r._alive(record.get('worker_pid'))
                got = p.request(a, 'get', key=key)
                state, result = got['state'], got['result']
                assert set(state) == {'selected_models', 'selected_epochs', 'selected_validation'}
                assert result == p.read_json(a / 'runs' / key / 'result.json') and result['job'] == job
                hashes = {kind: p.state_digest(value) for kind, value in state['selected_models'].items()}
                assert set(hashes) == {'online', 'fixed_ema'} and hashes == result['selected_hashes']
                assert state['selected_epochs'] == result['selected_epochs']
                assert state['selected_validation'] == result['selected_validation']
                epochs = p.epoch_files(a, result)
                assert set(epochs) == set(range(1, 51))
                p.verify_selection(result, epochs)
                assert result['counts'] == dict(optimizer_updates=50 * p.STEPS[dataset], ema_updates=50 * p.STEPS[dataset])
                size = sum(t.numel() * t.element_size() for model in state['selected_models'].values() for t in model.values())
                total_bytes += size
                updates += result['global_step']
                acks.extend(r._receipt(a / 'acks' / key / f'epoch_{epoch:02d}.json') for epoch in range(1, 51))
                rows.append(dict(key=key, dataset=dataset, arm=arm, seed=seed, selected_hashes=hashes,
                                 selected_epochs=result['selected_epochs'], selected_validation=result['selected_validation'],
                                 tensor_bytes=size, verified_epoch_ACKs=50,
                                 result_receipt=r._receipt(a / 'runs' / key / 'result.json')))
                del got, state, result, epochs
    assert len(rows) == 10 and len(acks) == 500 and updates == 17250
    assert status['retained_states'] == 10 and status['retained_bytes'] == total_bytes
    handoff = p.read_json(a / 'RUNNER_HANDOFF.json')
    assert handoff['source_hash'] == manifest['source_hash'] and handoff['test_accessed'] is False
    assert handoff['retained_tensor_bytes'] == total_bytes and handoff['committed_epochs'] == 500
    receipt = dict(status='ACCEPTED', parent_accepted=True, source_hash=manifest['source_hash'],
                   design_bundle_sha256=p.DESIGN_HASH, research_bundle_sha256=p.RESEARCH_HASH,
                   owner=p.owner_binding(a), rows=rows, acks=acks, verified_selected_models=20,
                   verified_epoch_ACKs=500, useful_adam_steps=updates, useful_ema_steps=updates,
                   retained_states=10, retained_bytes=total_bytes, handoff_sha256=p.sha_file(a / 'RUNNER_HANDOFF.json'),
                   all_formal_workers_exited=True, recorded_at_utc=datetime.now(timezone.utc).isoformat(), test_accessed=False)
    p.write_json(a / 'PARENT_STORAGE_ACCEPTANCE.json', receipt)
    return {key: receipt[key] for key in ('status', 'verified_selected_models', 'verified_epoch_ACKs', 'useful_adam_steps', 'retained_bytes')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('engineering', 'formal'))
    args = parser.parse_args()
    print(json.dumps(engineering() if args.stage == 'engineering' else formal()))
