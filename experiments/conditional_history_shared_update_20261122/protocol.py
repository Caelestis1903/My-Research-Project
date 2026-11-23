"""Finite core queue, immutable source identities and scalar filesystem records."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from experiments.conditional_information_momentum_20261102_v3 import protocol as v3

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
PLAN = ROOT / 'research_plans/26-11-22_conditional_history_shared_update'
REPORT = ROOT / 'research_reports/conditional_history_shared_update_20261122'
ART = ROOT / 'artifacts/conditional_history_shared_update_20261122'
DATASETS = ('movielens', 'kuairand')
SEEDS = tuple(range(2022, 2027))
MISSING = (2023, 2025, 2026)
ORIGIN = ROOT.parent / 'phase0-3_git'
M = '875215c1e509ab55c1b6632d4f5d13e8b94782da'
P = 'a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288'
PLAN_COMMIT = '4a3e4f1b21f1aebb6daa2df6635c3d7ca547a983'
V3_MANIFEST = ROOT / 'research_reports/2026-11-02_mdmt_information_momentum_v3/IMPLEMENTATION_SOURCE_MANIFEST.json'
A_FILES = ('__init__.py', 'protocol.py', 'state.py', 'reuse.py', 'checks.py', 'runner.py', 'scheduler.py', 'cli.py')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')
    os.replace(out.name, path)


def append(path, value):
    with Path(path).open('a') as out:
        out.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ORIGIN), *args])


def pinned(path, commit):
    rel = Path(path).relative_to(ROOT)
    actual = sha(path)
    blob = git('rev-parse', f'{commit}:{rel}').decode().strip()
    if hashlib.sha256(git('cat-file', 'blob', blob)).hexdigest() != actual:
        raise ValueError(f'Pinned source differs: {rel}')
    return dict(path=str(rel), commit=commit, blob=blob, sha256=actual)


def source(stage):
    names = list(A_FILES)
    if stage == 'B_CORE':
        names += sorted(p.name for p in PACKAGE.iterdir() if (p.name.startswith('b_') and p.suffix in ('.py','.cpp')) or
                        p.stem in ('groups','probes','utility','virtual_adam','rollout'))
        if 'b_observer.py' not in names:
            raise ValueError('B observer implementation pending')
    elif stage != 'A_CORE':
        raise ValueError(stage)
    files = {name:sha(PACKAGE/name) for name in names}
    old = read(V3_MANIFEST)['files']
    if sha(V3_MANIFEST) != '7034a3bb59e6d08e3296183258e0e7b16ab74d0248ee0a3b75c526a680462cc0':
        raise ValueError('v3 source manifest changed')
    for row in old:
        if sha(v3.HERE / row['path']) != row['sha256']:
            raise ValueError(f"Source mismatch {row['path']}")
    authority = {'plan':sha(PLAN/'experiment_manifest.json'), 'design':sha(REPORT/'DESIGN_SPECIFICATION.md')}
    return dict(files=files, v3_sha256=sha(V3_MANIFEST), **authority,
                source_hash=digest([files, sha(V3_MANIFEST), authority]))


def jobs(stage, source_hash):
    if stage == 'A_CORE':
        arms, seeds = ('A10', 'A30'), MISSING
    elif stage == 'B_CORE':
        arms, seeds = ('A00',), SEEDS
    else:
        raise ValueError(stage)
    names = {'A00':'A0', 'A10':'P00', 'A30':'P10'}
    result = []
    for seed in seeds:
        for arm in arms:
            for dataset in DATASETS:
                config = v3.config(names[arm], dataset, 0 if arm == 'A00' else .05)
                job = dict(dataset=dataset, seed=seed, arm=arm, stage=stage, split_id='original',
                           source_hash=source_hash, config=config)
                job['key'] = digest(job)
                result.append(job)
    return result
