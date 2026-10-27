"""Freeze this experiment's two reviewed FP32 diagnostic revisions."""
from __future__ import annotations
from pathlib import Path
import argparse
import ast
import difflib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.mdmt_relation_basis_v2.config import METHODS, DATASETS, SEEDS, load_config
from experiments.mdmt_relation_basis_v2.state import digest, sha_file, write_json

OLD = '72f3e825dac422784fd575bd24450e7858e99ce138f6c8261b3ae7636b8b13fc'
NEW = '26708434c81106dd424d52b4de258bee8aad7b35640b1d50b8699290694a2a3c'
SNAPSHOTS = Path(__file__).resolve().parent/'SOURCE_REVISIONS'


def read(path):
    return json.loads(Path(path).read_text())


def source_equivalence():
    versions = {}
    for expected in (OLD, NEW):
        directory = SNAPSHOTS/expected[:8]
        identity = read(directory/'runtime_source_identity.json')
        actual = {str(p.relative_to(directory/'package')):sha_file(p)
                  for p in sorted((directory/'package').rglob('*.py'))}
        if identity['source_hash'] != expected or identity['files'] != actual or digest(actual) != expected:
            raise ValueError('Source snapshot identity mismatch: '+expected)
        versions[expected] = identity
    old, new = versions[OLD]['files'], versions[NEW]['files']
    if set(old) != set(new):
        raise ValueError('Different source file sets')
    changed = [p for p in old if old[p] != new[p]]
    if set(changed) != {'diagnostics.py', 'tests/test_runtime_numerics.py'}:
        raise ValueError('Changes exceed reviewed diagnostics/tests: '+str(changed))
    training = {p:h for p,h in old.items() if p != 'diagnostics.py' and not p.startswith('tests/')}
    other_training = {p:h for p,h in new.items() if p != 'diagnostics.py' and not p.startswith('tests/')}
    if training != other_training:
        raise ValueError('Scientific training files differ')
    texts = [(SNAPSHOTS/h[:8]/'package'/'diagnostics.py').read_text() for h in (OLD, NEW)]
    trees = [ast.parse(text) for text in texts]
    for tree in trees:
        tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name == 'compare_raw_logits')]
    if ast.dump(trees[0], include_attributes=False) != ast.dump(trees[1], include_attributes=False):
        raise ValueError('Non-comparison diagnostics behavior differs')
    return dict(old_full_source_hash=OLD, new_full_source_hash=NEW,
                training_source_hash=digest(training), training_source_files=training,
                excluded_only=['diagnostics.py', 'tests/'], changed_files=changed,
                diagnostic_scientific_functions_identical=True,
                diagnostics_diff=''.join(difflib.unified_diff(texts[0].splitlines(True), texts[1].splitlines(True), fromfile=OLD, tofile=NEW))), versions


def build_views(root, cfg, decision, equivalence, versions, review_record):
    """Read eight complete resident workers; keep each actual identity intact."""
    root = Path(root)
    if decision.get('status') != 'FINAL_METHODS_FROZEN' or set(decision.get('stage2', {})) != set(DATASETS):
        raise ValueError('Methods and both dataset stage2 decisions must be closed')
    if any(v not in ('NOT_TRIGGERED', 'COMPLETED') for v in decision['stage2'].values()):
        raise ValueError('Unfinished stage2 decision')
    workers, worker_view = {}, {}
    role_identities = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            directory = root/'workers'/dataset/f'seed{seed}'
            ready = read(directory/'READY.json')
            identity = ready['identity']
            full_hash = identity['source_hash']
            if ready['status'] != 'READY_FOR_GLOBAL_FREEZE' or full_hash not in (OLD, NEW):
                raise ValueError('Worker missing readiness or unreviewed revision')
            if (identity['dataset'], identity['seed'], identity['config_hash']) != (dataset, seed, cfg['config_hash']):
                raise ValueError('Dataset/seed/config identity mismatch')
            if set(identity['selected']) != set(METHODS) or set(identity['final']) != set(METHODS):
                raise ValueError('Worker methods incomplete')
            recorded = read(directory/'source_identity.json')
            if recorded != versions[full_hash]:
                raise ValueError('Worker source manifest differs from exact reviewed snapshot')
            effective = read(directory/'effective_config.json')
            if effective['config'] != cfg:
                raise ValueError('Effective study configurations differ')
            data = read(directory/'data_manifest.json')
            role_pair = (data['data_role_hash'], data['preprocessor_hash'])
            if role_pair != (identity['data_role_hash'], identity['preprocessor_hash']):
                raise ValueError('Worker data/preprocessor identity mismatch')
            if dataset in role_identities and role_identities[dataset] != role_pair:
                raise ValueError('Dataset roles differ across seeds')
            role_identities[dataset] = role_pair
            for method in METHODS:
                receipt = read(root/'runs'/dataset/method/f'seed{seed}'/'run_receipt.json')
                if receipt['identity'] != dict({k:v for k,v in identity.items() if k not in ('selected','final')}, method=method):
                    raise ValueError('Run and worker identities differ')
                if (receipt['selected_model_hash'], receipt['final_model_hash']) != (identity['selected'][method], identity['final'][method]):
                    raise ValueError('Run state identity mismatch')
            key = f'{dataset}/seed{seed}'
            workers[key] = identity
            worker_view[key] = 'GLOBAL_FREEZE.json' if full_hash == OLD else 'GLOBAL_FREEZE_FP32_STANDARD.json'
    common = dict(status='FROZEN', protocol=cfg['protocol'], config_hash=cfg['config_hash'],
                  methods=list(METHODS), workers=workers, decision=decision,
                  thresholds=cfg['comparison'], historical_test=True,
                  training_source_hash=equivalence['training_source_hash'],
                  source_equivalence_hash=digest(equivalence),
                  diagnostic_revisions={OLD:dict(atol=1e-6, rtol=1.3e-6), NEW:dict(atol=1e-5, rtol=1.3e-6)},
                  review_record=review_record)
    canonical_identity = digest(common)
    views = {filename:dict(common, source_hash=full_hash, canonical_identity=canonical_identity)
             for full_hash,filename in [(OLD,'GLOBAL_FREEZE.json'),(NEW,'GLOBAL_FREEZE_FP32_STANDARD.json')]}
    return common, canonical_identity, views, worker_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--review-record', type=Path, required=True)
    args = parser.parse_args()
    equivalence, versions = source_equivalence()
    review = dict(path=str(args.review_record.resolve()), sha256=sha_file(args.review_record))
    common, canonical_identity, views, worker_view = build_views(args.artifact_root, load_config(), read(args.decision), equivalence, versions, review)
    names = list(views)+['GLOBAL_FREEZE_CANONICAL.json', 'SOURCE_EQUIVALENCE.json']
    if any((args.artifact_root/name).exists() for name in names):
        raise FileExistsError('Freeze/equivalence output already exists')
    write_json(args.artifact_root/'SOURCE_EQUIVALENCE.json', equivalence, exclusive=True)
    outputs = {}
    for name, value in views.items():
        write_json(args.artifact_root/name, value, exclusive=True)
        outputs[name] = dict(path=str((args.artifact_root/name).resolve()), sha256=sha_file(args.artifact_root/name),
                             view_identity=digest(value))
    canonical = dict(status='CANONICAL_FROZEN', canonical_identity=canonical_identity,
                     common=common, views=outputs, worker_view=worker_view)
    write_json(args.artifact_root/'GLOBAL_FREEZE_CANONICAL.json', canonical, exclusive=True)
    print(json.dumps(dict(canonical_identity=canonical_identity, views=outputs,
                          training_source_hash=equivalence['training_source_hash']), indent=2))


if __name__ == '__main__':
    main()
