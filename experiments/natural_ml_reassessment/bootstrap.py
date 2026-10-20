"""Import only this study's archived producers, never mutable sibling trees."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'source_archive'
ROOT = Path('/home_new/xzf_natural_movielens_reassessment')
PYTHON = '/home/xzf/miniconda3/envs/phase0-3-env/bin/python'

def activate(label='core'):
    root = ARCHIVE / label
    for path in (root/'m3oe/examples', root/'m3oe', root/'experiments'):
        sys.path.insert(0, str(path))
    return root

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def source_hashes(extra=()):
    inventory = json.loads((ARCHIVE/'inventory.json').read_text())
    result = {}
    for name, record in inventory.items():
        actual = sha(ARCHIVE/name)
        if actual != record['sha256']:
            raise RuntimeError('Frozen producer changed: '+name)
        result['source_archive/'+name] = actual
    stable=('__init__.py','bootstrap.py','data.py','manifest.py','telemetry.py',
            'stage2d.py','mechanism.py','run.py','queue.py','basiccheck.py')
    for name in sorted(set(stable+tuple(extra))):
        path=HERE/name
        if not path.is_file():continue
        result[path.name] = sha(path)
    return result

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)
