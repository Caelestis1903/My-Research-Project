"""Freeze completed development decisions before admitting test readout jobs."""
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from n19_data import OUT, ROOT, dump


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    freeze = OUT/'TEST_FREEZE.json'
    if freeze.exists():
        raise RuntimeError('Test freeze already exists; preserve the original decision')
    decision = OUT/'DEVELOPMENT_DECISION_RECORD.md'
    if not decision.exists():
        raise RuntimeError('Primary scientific development disposition is missing')
    manifest_path = OUT/'queue_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    tasks = [t for t in manifest['tasks'] if 'experiments/n19_train.py' in t['command'] and '--engineering' not in t['command']]
    groups = defaultdict(list)
    records = {}
    for task in tasks:
        run = OUT/'runs'/task['name']
        config = json.loads((run/'config.json').read_text())
        receipt = json.loads((run/'receipt.json').read_text())
        selection = json.loads((run/'selection_frozen.json').read_text())
        budget = 50 if config['dataset'] == 'kuairand' else 100
        assert not config['engineering'] and config['from_scratch'] and not config['backbone_frozen']
        assert receipt['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
        assert config['epochs'] == receipt['epochs'] == selection['budget'] == budget
        assert receipt['selected_epoch'] == selection['selected_epoch']
        assert not receipt['test_read'] and not selection['test_read']
        assert not (run/'test_receipt.json').exists()
        assert not list(run.glob('test_*.npz'))
        if config['method'] in ['rank_uniform', 'rank_protected']:
            assert receipt['selected_epoch'] >= 10
        for filename in ['selected.pt', 'last.pt']:
            assert (Path(config['state_directory'])/filename).exists()
        groups[re.sub(r'__seed\d+', '', run.name)].append(config['seed'])
        records[run.name] = dict(selected_epoch=receipt['selected_epoch'],epochs=budget,
            config_sha256=sha(run/'config.json'),training_receipt_sha256=sha(run/'receipt.json'),
            selection_sha256=sha(run/'selection_frozen.json'))
    assert len(records) == len(tasks) and records
    for name, seeds in groups.items():
        assert sorted(seeds) == [2022,2023,2024,2025], (name,seeds)
    sources = sorted((ROOT/'experiments').glob('n19_*.py'))
    dirty = subprocess.check_output(['git','diff','HEAD','--name-only','--','experiments'],cwd=ROOT,text=True)
    assert not dirty, 'Commit research source before freezing test decisions'
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dump(freeze,dict(status='DEVELOPMENT_DECISIONS_FROZEN',created_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=source_commit,review='Primary self-review under revised start.md; not independent review',
        decision_record=str(decision.relative_to(ROOT)),decision_sha256=sha(decision),
        training_manifest_sha256=sha(manifest_path),formal_networks=len(records),groups=len(groups),
        seeds=[2022,2023,2024,2025],runs=sorted(records),training_records=records,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        test_scope='Previously used historical test population; frozen readout, not fresh confirmatory evidence',
        no_post_test_method_changes=True))
    for name in sorted(records):
        manifest['tasks'].append(dict(name='evaluate__'+name,
            command=['/home/xzf/miniconda3/envs/phase0-3-env/bin/python','-u','experiments/n19_evaluate.py',
                '--run',str(OUT/'runs'/name),'--device','{device}'],
            log=str(OUT/'logs'/('evaluate__'+name+'.log'))))
    manifest['sealed'] = True
    dump(manifest_path,manifest)
    print(f'Frozen {len(records)} networks in {len(groups)} four-seed groups; {len(manifest["tasks"])} total queue tasks')


if __name__ == '__main__':
    main()
