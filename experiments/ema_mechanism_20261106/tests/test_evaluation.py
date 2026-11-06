"""Test gates reject incomplete budgets and changed endpoint identities."""
from types import SimpleNamespace

import pytest

from experiments.ema_mechanism_20261106 import analysis as a, evaluation as e
from experiments.overnight_ab_20261103.state import read_json, sha_file, write_json


def _study():
    jobs = [dict(job_id=f'{dataset}_{objective}_{seed}', dataset=dataset, objective=objective,
                 seed=seed, steps_per_epoch=20 if dataset == 'movielens' else 49)
            for dataset in ('movielens', 'kuairand') for objective in a.CONDITIONS for seed in a.SEEDS]
    return SimpleNamespace(jobs=jobs, manifest_hash='manifest', identity={'source': 'frozen'},
                           manifest={'datasets': {'movielens': {'steps_per_epoch': 20}, 'kuairand': {'steps_per_epoch': 49}}})


def test_no_endpoint_freeze_without_twenty_complete_fifty_epoch_jobs(tmp_path):
    study = _study()
    with pytest.raises(a.Incomplete, match='epochs'):
        e._complete_summaries(study, tmp_path)


def test_changed_selection_bytes_rejected_before_test_arrays(tmp_path):
    write_json(tmp_path/'SELECTION_FROZEN.json', dict(endpoints=[], input_hashes={}, seed_ids=list(a.SEEDS)))
    write_json(tmp_path/'FREEZE_RECEIPT.json', dict(selection_hash=sha_file(tmp_path/'SELECTION_FROZEN.json')))
    value = read_json(tmp_path/'SELECTION_FROZEN.json')
    value['endpoints'] = [{'state_hash': 'tampered'}]
    write_json(tmp_path/'SELECTION_FROZEN.json', value)
    with pytest.raises(ValueError, match='modified'):
        e.validate_freeze(_study(), tmp_path)


def test_test_evaluation_is_a_separate_process(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'validate_freeze', lambda *_: ({}, {}))
    write_json(tmp_path/'PREPARED.json', {'paths': {'manifest': '/fixed/manifest.json'}})
    write_json(tmp_path/'TEST_EVALUATION_RECEIPT.json', {'status': 'COMPLETED'})
    calls = []
    monkeypatch.setattr(e.subprocess, 'run', lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0))
    assert e.evaluate_frozen(_study(), tmp_path)['status'] == 'COMPLETED'
    assert calls[0][1:3] == ['-m', 'experiments.ema_mechanism_20261106.evaluation']
    assert '--internal-evaluate' in calls[0]
