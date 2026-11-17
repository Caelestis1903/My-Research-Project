"""Offline, NOT_FINAL C05 inventory and full immediate-directory Git tree plan.

Copies the real index; never changes HEAD/the real index or calls a remote API.
Does not pack raw records, inspect owner credentials, or declare evidence final.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SERIES = Path('artifacts/26-11-16_augmentation_gradient_momentum')
A = ROOT / SERIES / 'astra_cycles/cycle_05'
N = A / 'attempt1'
P = ROOT / 'research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05'
E = ROOT / 'experiments/cell_mass_bce_20261117'
R = Path('/tmp/phase0-3-c05-01a1e9a9/attempt1')
EXPECTED_HEAD = 'cbccf5ae234e46bc3f976467de2b3327139f021b'
BASE_TREE = 'e874ff09381557117d9e357048e3b59a475bd366'
SOURCE_HASH = 'f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d'
C04_RECEIPT = ROOT / SERIES / 'astra_cycles/cycle_04/PUBLICATION_RECEIPT.json'
C04_RECEIPT_SHA = '319b921da73f3b5a157f45922bfdafb475673f77f987b4fc409525a958cc12cd'
SOURCE_NAMES = ('__init__.py', 'core.py', 'checks.py', 'protocol.py', 'runner.py', 'tables.py', 'README.md')
CONTEXT_NAMES = (
    'Experimental_Contradiction_Register.md', 'Experimental_Dead_Ends.md',
    'Experimental_Evidence_Ledger.md', 'Experimental_Experiment_Index.md',
    'Experimental_Global_Summary.md', 'Experimental_Hypothesis_History.md',
    'Experimental_Open_Questions.md', 'Experimental_Stable_Findings.md',
)
PRIVATE = {'guardian.json', 'STORAGE_OWNER.json'}
PUBLIC_SUFFIXES = {'.py', '.md', '.json', '.csv', '.png'}
GIT_ENV = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_TERMINAL_PROMPT='0')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def object_sha(kind, data):
    return hashlib.sha1(kind.encode() + b' ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def git(*args, data=None, env=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, input=data, env=env or GIT_ENV)


def text_git(*args, **kwargs):
    return git(*args, **kwargs).decode().strip()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def indexed(env):
    entries = {}
    for record in git('ls-files', '--stage', '-z', env=env).split(b'\0'):
        if not record:
            continue
        meta, name = record.split(b'\t', 1)
        mode, oid, stage = meta.decode().split()
        require(stage == '0', 'Unmerged index entry: ' + name.decode())
        entries[name.decode()] = (mode, oid)
    return entries


def verify_source():
    receipt = read(R / 'source_receipt.json')
    files = receipt['files']
    bundle = json.dumps(files, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    require(len(files) == 55 and hashlib.sha256(bundle).hexdigest() == receipt['sha256'] == SOURCE_HASH,
            'Frozen source55 receipt identity differs')
    for row in files:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid source receipt path')
        path = ROOT / relative
        require(path.resolve() == path and path.is_file() and sha(path) == row['sha256'],
                'Frozen source changed: ' + row['path'])
    require({row['path'] for row in files if row['path'].startswith(str(E.relative_to(ROOT)) + '/')}
            == {str((E / name).relative_to(ROOT)) for name in SOURCE_NAMES}, 'C05 source must be exactly seven files')
    return files


def inventory():
    paths = {ROOT / name for name in ('AGENTS.md', 'buchong.md', 'EXP_AI_CONTEXT.md', 'MDMT_ALGORITHM_METRICS_LEDGER.md')}
    paths.add(ROOT / 'research_plans/26-11-16_augmentation_gradient_momentum/00_EXECUTION_RUNBOOK.md')
    paths.update(ROOT / 'research_context/experiments' / name for name in CONTEXT_NAMES)
    paths.update(E / name for name in SOURCE_NAMES)
    paths.add(C04_RECEIPT)
    for directory in (OUT, A, P):
        paths.update(p for p in directory.iterdir()
                     if p.is_file() and p.suffix in PUBLIC_SUFFIXES and p.name not in PRIVATE)
    paths.discard(A / 'PUBLICATION_RECEIPT.json')  # Own receipt follows remote verification.
    paths.update(path for path in N.glob('*.json') if path.is_file() and path.name not in PRIVATE)
    package_names = ('RAW_RUNS_MANIFEST.json', 'RAW_RUNS_PART_INDEX.json')
    missing = [str((N / name).relative_to(ROOT)) for name in package_names if not (N / name).exists()]
    for name in package_names:
        if (N / name).is_file():
            paths.add(N / name)
    manifest_path = N / 'RAW_RUNS_MANIFEST.json'
    if manifest_path.exists():
        archive = read(manifest_path)
        require(archive['source_hash'] == SOURCE_HASH, 'Archive source identity differs')
        require(bool(archive['parts']), 'Empty archive part list')
        combined = hashlib.sha256()
        total = 0
        seen = set()
        for i, part in enumerate(archive['parts']):
            name = part['path']
            require(Path(name).name == name and name.startswith('RAW_RUNS.tar.gz.part-') and name not in seen,
                    'Unsafe/duplicate archive part path')
            seen.add(name)
            path = N / name
            data = path.read_bytes()
            require(len(data) == part['bytes'] and 0 < len(data) <= 8 * 1024 * 1024
                    and (i == len(archive['parts']) - 1 or len(data) == 8 * 1024 * 1024)
                    and hashlib.sha256(data).hexdigest() == part['sha256'], 'Archive part differs: ' + name)
            combined.update(data)
            total += len(data)
            paths.add(path)
        require(total == archive['bytes'] and combined.hexdigest() == archive['sha256'], 'Combined archive differs')
    return sorted(paths), missing


def full_tree_plan(tree, changed_paths):
    parents = {''}
    for name in changed_paths:
        parents.update('' if parent == Path('.') else str(parent) for parent in Path(name).parents)
    plan = []
    for parent in sorted(parents, key=lambda p: (-len(Path(p).parts), p)):
        spec = tree + ':' + parent if parent else tree
        expected = text_git('rev-parse', spec)
        entries = []
        serialized = b''
        for record in git('ls-tree', '-z', expected).split(b'\0'):
            if not record:
                continue
            meta, name = record.split(b'\t', 1)
            mode, kind, oid = meta.decode().split()
            require(b'/' not in name, 'Tree entry must be an immediate basename')
            entries.append(dict(path=name.decode(), mode=mode, type=kind, sha=oid))
            serialized += mode.lstrip('0').encode() + b' ' + name + b'\0' + bytes.fromhex(oid)
        require(object_sha('tree', serialized) == expected, 'Full tree serialization differs: ' + parent)
        plan.append(dict(path=parent, sha=expected, entries=entries))
    require(plan[-1]['path'] == '' and plan[-1]['sha'] == tree, 'Root must be last')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--base-tree', required=True)
    parser.add_argument('--output-prefix', default='/tmp/phase0-3-cycle05')
    args = parser.parse_args()
    require(args.expected_head == EXPECTED_HEAD and args.base_tree == BASE_TREE, 'C04 base differs; obtain a fresh parent plan')
    require(text_git('rev-parse', 'HEAD') == EXPECTED_HEAD, 'HEAD changed')
    require(text_git('rev-parse', 'HEAD^{tree}') == BASE_TREE, 'HEAD tree changed')
    original = Path(text_git('rev-parse', '--git-path', 'index')).resolve()
    head_file = Path(text_git('rev-parse', '--git-path', 'HEAD')).resolve()
    index_sha, head_sha = sha(original), sha(head_file)
    source = verify_source()
    require(sha(C04_RECEIPT) == C04_RECEIPT_SHA, 'C04 publication receipt differs')
    require(read(C04_RECEIPT)['current_selected_ram_available'] is False, 'C04 RAM disclosure differs')
    paths, missing_package = inventory()
    staging = Path(tempfile.mkdtemp(prefix='phase0-3-cycle05-publication-'))
    backup, prepared = staging / 'original.index', staging / 'prepared.index'
    shutil.copyfile(original, backup)
    shutil.copyfile(original, prepared)
    env = dict(GIT_ENV, GIT_INDEX_FILE=str(prepared))
    require(sha(backup) == sha(prepared) == index_sha, 'Index copy differs')
    before = indexed(env)
    require(text_git('write-tree', '--missing-ok', env=env) == BASE_TREE, 'Existing index has unrelated staged changes')
    rows = []
    base_blobs = {entry[1] for entry in before.values()}
    for path in paths:
        require(path.is_file() and path.resolve() == path and path.is_relative_to(ROOT), 'Missing/linked public file: ' + str(path))
        require(path.name not in PRIVATE and path.suffix.lower() not in ('.pt', '.pth', '.ckpt', '.pkl', '.safetensors', '.npy', '.npz'),
                'Private/tensor file in inventory')
        name = str(path.relative_to(ROOT))
        data = path.read_bytes()
        require(len(data) < 100 * 1024 * 1024, 'File exceeds GitHub blob limit: ' + name)
        oid = object_sha('blob', data)
        require(text_git('hash-object', '-w', '--stdin', data=data) == oid, 'Git blob SHA differs: ' + name)
        mode = before.get(name, ('100644', None))[0]
        rows.append(dict(path=name, absolute_path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                         git_sha1=oid, mode=mode, upload_required=oid not in base_blobs))
    updates = b''.join(f"{r['mode']} {r['git_sha1']}\t{r['path']}".encode() + b'\0' for r in rows)
    git('update-index', '-z', '--add', '--index-info', data=updates, env=env)
    after = indexed(env)
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    require(set(changed) <= {r['path'] for r in rows}, 'Unallowlisted index change')
    tree = text_git('write-tree', '--missing-ok', env=env)
    plan = full_tree_plan(tree, changed)
    for row in rows:
        require(sha(Path(row['absolute_path'])) == row['sha256'], 'Public file changed during preparation: ' + row['path'])
    require(verify_source() == source, 'Source changed during preparation')
    require(sha(original) == index_sha and sha(head_file) == head_sha and text_git('rev-parse', 'HEAD') == EXPECTED_HEAD,
            'Real HEAD/index changed during preparation')
    prefix = Path(args.output_prefix)
    manifest_path = Path(str(prefix) + '-publication-manifest.json')
    tree_path = Path(str(prefix) + '-tree-plan.json')
    manifest = dict(phase='NOT_FINAL_PREPARATION_ONLY', publication_authorized=False,
                    repository='Caelestis1903/My-Research-Project', branch='main', expected_head=EXPECTED_HEAD,
                    base_tree_sha=BASE_TREE, expected_tree=tree, files=rows, changed_paths=changed,
                    prepared_index=str(prepared), original_index=str(original), original_index_backup=str(backup),
                    original_index_sha256=index_sha, original_head_file_sha256=head_sha,
                    source_hash=SOURCE_HASH, frozen_source_files_verified=len(source), c05_source_files=7,
                    source_receipt_sha256=sha(R / 'source_receipt.json'), c04_receipt_sha256=C04_RECEIPT_SHA,
                    excluded_private=sorted(PRIVATE), raw_policy='R5 archive only; no individual raw files or C04 parts',
                    missing_package_paths=missing_package, tree_plan=str(tree_path),
                    pending=['Training, parent reports/context, final evidence review, final archive/index and parent publication authorization remain external gates.'])
    write(manifest_path, manifest)
    write(tree_path, plan)
    print(json.dumps(dict(status=manifest['phase'], manifest=str(manifest_path), tree_plan=str(tree_path),
                          files=len(rows), changed_paths=len(changed), upload_blobs=len({r['git_sha1'] for r in rows if r['upload_required']}),
                          bytes=sum(r['bytes'] for r in rows), expected_tree=tree, tree_calls=len(plan),
                          source55_verified=True, actual_head_index_unchanged=True, missing_package_paths=missing_package)))


if __name__ == '__main__':
    main()
