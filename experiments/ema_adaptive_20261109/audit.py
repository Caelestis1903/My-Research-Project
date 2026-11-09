"""Read-only source, archived scalar, and observation panel admission."""
import csv
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'research_reports/2026-11-09_ema_adaptive'
SEEDS = (2022, 2023, 2024, 2025, 2026)
IDENTITY_KEYS = ('schema_version', 'job_id', 'dataset', 'objective', 'seed', 'epoch',
                 'global_step', 'commit_index', 'init_hash', 'order_hash', 'layout_hash', 'group_hash', 'identity_id')


class Incomplete(RuntimeError):
    exit_code = 5


class Conflict(Incomplete):
    exit_code = 2


class ResourceBlock(RuntimeError):
    exit_code = 4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def blob(path):
    b = Path(path).read_bytes()
    return hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def code_identity():
    files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}
    return {'files': files, 'sha256': digest(files)}


def resolve(manifest):
    m = read_json(manifest)
    if m['study_id'] != 'MDMT_ADAPTIVE_EMA_20261109' or m['seeds']['historical_development'] != list(SEEDS):
        raise Conflict('Study or historical seed matrix changed')
    remote = read_json(REPORT / 'REMOTE_SOURCE_IDENTITY.json')
    return {'manifest': m, 'manifest_sha256': sha(manifest), 'source_head': remote['source_head'],
            'main_head': remote['main_head'], 'remote': remote,
            'design_sha256': sha(REPORT / 'DESIGN_SPECIFICATION.md'), 'implementation': code_identity(),
            'permitted_seeds': list(SEEDS), 'future_seed_override': 'start.md ONLY 2022-2026; independent A1 needs a reviewed amendment',
            'source': str(ROOT / m['stage_A0']['source']), 'new_training_runs': 0, 'new_test_accesses': 0}


def check_input(path, ref, inventory, purpose, expected_blob=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise Incomplete('Missing or empty source: ' + str(path))
    rel = str(path.relative_to(ROOT))
    expected = expected_blob or git('rev-parse', f'{ref}:{rel}')
    actual = blob(path)
    inventory.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha(path),
                      'git_blob': actual, 'expected_git_blob': expected, 'source_commit': ref,
                      'purpose': purpose, 'verified': actual == expected})
    if actual != expected:
        raise Conflict('Source blob mismatch: ' + rel)


def _normalized(value):
    if value is None or value == '':
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if value in ('True', 'False'):
        return value == 'True'
    try:
        return float(value)
    except ValueError:
        return value


def archive_check(source, out, ref):
    """Compare each archived pulse scalar against its committed CSV row, without extracting files."""
    tables = {}
    keys = {'pulse_readouts': ['job_id', 'epoch', 'panel_id', 'physical_readout_id', 'domain', 'task'],
            'pulse_metrics': ['job_id', 'epoch', 'domain', 'task', 'group_id', 'panel_id']}
    names = {'pulse_readouts': 'PULSE_READOUTS.csv', 'pulse_metrics': 'PULSE_METRICS.csv'}
    recovered = {}
    for kind, name in names.items():
        path = source / name
        if not path.is_file():
            path = out / name
            try:
                content = subprocess.check_output(['git', 'show', f'{ref}:{(source / name).relative_to(ROOT)}'], cwd=ROOT)
            except subprocess.CalledProcessError as error:
                raise Incomplete('Committed source CSV cannot be restored: ' + name) from error
            path.write_bytes(content)
            recovered[name] = str(path)
        if path.is_file():
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            table = {}
            for row in rows:
                key = tuple(_normalized(row[k]) for k in keys[kind])
                val = {k: _normalized(v) for k, v in row.items()}
                if key in table and table[key] != val:
                    raise Incomplete('Conflicting archived CSV identity: ' + name)
                table[key] = val
            tables[kind] = table
    counts = dict.fromkeys(keys, 0)
    with tarfile.open(source / 'raw_epoch_records.tar.gz', 'r|gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.json'):
                continue
            with archive.extractfile(member) as stream:
                record = json.load(stream)
            for kind in keys:
                for observation in record.get(kind, []):
                    row = {**{k: record.get(k) for k in IDENTITY_KEYS}, **observation}
                    key = tuple(_normalized(row[k]) for k in keys[kind])
                    if kind in tables:
                        expected = tables[kind].pop(key, None)
                        if expected is None or any(_normalized(row.get(k)) != v for k, v in expected.items()):
                            raise Incomplete(f'Archive/CSV scalar conflict: {kind} {key}')
                    counts[kind] += 1
    if any(table for table in tables.values()):
        raise Incomplete('CSV has pulse rows absent from archived records')
    return {'verified_rows': counts, 'recovered_tables': recovered}


def panel_check(source, inventory, ref):
    from experiments.ema_mechanism_20261106.panels import load_panels
    panels = read_json(source / 'PROBE_MANIFEST.json')['panels']
    checked = []
    for dataset, seeds in panels.items():
        if set(map(int, seeds)) != set(SEEDS):
            raise Incomplete('Panel five-seed coverage differs')
        for seed, entries in seeds.items():
            for receipt in entries.values():
                path = source / receipt['path']
                check_input(path, ref, inventory, 'frozen panel identity')
                if sha(path) != receipt['sha256']:
                    raise Incomplete('Panel hash differs')
            values = load_panels(source, dataset, int(seed))
            for name, value in values.items():
                key = 'rank_pairs' if name in ('P_A', 'P_B') else 'pairs'
                if entries[name]['pair_hash'] != entries[name]['identity']['arrays'][key]:
                    raise Incomplete('Pair receipt hash differs')
            a, b, v = values['P_A'], values['P_B'], values['V_mech']
            rows_a = np.concatenate([a[k].ravel() for k in ('rank_row_ids', 'natural_row_ids')])
            rows_b = np.concatenate([b[k].ravel() for k in ('rank_row_ids', 'natural_row_ids')])
            if np.intersect1d(rows_a, rows_b).size:
                raise Incomplete('P_A/P_B overlap')
            mask, sides = v['indices'] >= 0, v['validation_row_side']
            if not np.isin(sides[mask], [0, 1]).all() or not (sides[~mask] == -1).all():
                raise Incomplete('Invalid Validation mask/row side')
            if np.intersect1d(v['row_ids'][mask & (sides == 0)], v['row_ids'][mask & (sides == 1)]).size:
                raise Incomplete('V_A/V_B overlap')
            if np.intersect1d(np.concatenate([rows_a, rows_b]), v['row_ids'][mask]).size:
                raise Incomplete('Training/Validation panel overlap')
            checked.append({'dataset': dataset, 'seed': int(seed), 'V_counts': mask.sum(axis=1).tolist()})
    return checked


def audit(study, out):
    source = Path(study['source'])
    inventory = []
    required = [ROOT / p for p in study['manifest']['stage_A0']['required_source_artifacts']]
    extra = ['PREDICTIVE_ROWS.csv', 'PREDICTIVE_FOLDS.json', 'RESOLVED_PROTOCOL.json', 'GROUP_MAP.json',
             'PROBE_MANIFEST.json', 'SOURCE_RECEIPT.json', 'ANALYSIS_REVISION.json', 'raw_epoch_records.tar.gz']
    recovery = archive_check(source, out, study['source_head'])
    for path in required + [source / name for name in extra]:
        if path.name in recovery['recovered_tables']:
            restored = Path(recovery['recovered_tables'][path.name])
            expected = git('rev-parse', f'{study["source_head"]}:{path.relative_to(ROOT)}')
            if blob(restored) != expected:
                raise Incomplete('Recovered table differs from committed CSV bytes')
            inventory.append({'path': str(restored.relative_to(ROOT)), 'bytes': restored.stat().st_size,
                              'sha256': sha(restored), 'git_blob': blob(restored), 'expected_git_blob': expected,
                              'source_commit': study['source_head'], 'purpose': 'archive restoration', 'verified': True})
        else:
            check_input(path, study['source_head'], inventory, 'A0 source scalar/log identity')
    receipt = read_json(source / 'SOURCE_RECEIPT.json')
    revision = read_json(source / 'ANALYSIS_REVISION.json')
    if revision['collection_implementation_hash'] != receipt['implementation_hash']:
        raise Incomplete('Collection/revision identity chain differs')
    for rel, expected in receipt['source']['files'].items():
        path = ROOT / rel
        check_input(path, study['source_head'], inventory, 'frozen model/loss/data dependency')
        if sha(path) != expected:
            raise Incomplete('Collection source dependency changed: ' + rel)
    for rel, expected in receipt['implementation']['files'].items():
        path = ROOT / rel
        check_input(path, study['source_head'], inventory, 'historical collection/analysis implementation')
        change = revision['changed_files'].get(rel)
        if change and change['collection_sha256'] != expected:
            raise Incomplete('Analysis revision has wrong collection parent: ' + rel)
        if sha(path) != (change['current_sha256'] if change else expected):
            raise Incomplete('Historical implementation revision differs: ' + rel)
    for row in study['remote']['plan_files']:
        check_input(ROOT / row['path'], study['main_head'], inventory, 'frozen plan', row['blob_sha'])
    for row in study['remote']['main_context_files']:
        check_input(REPORT / 'source_main' / Path(row['path']).name, study['main_head'], inventory,
                    'main context snapshot', row['blob_sha'])
    panels = panel_check(source, inventory, study['source_head'])
    group_map = read_json(source / 'GROUP_MAP.json')
    resources = {}
    metrics = pd.read_csv(recovery['recovered_tables'].get('PULSE_METRICS.csv', source / 'PULSE_METRICS.csv'), float_precision='round_trip')
    for dataset, layout in group_map.items():
        count = sum(r['numel'] for r in layout['records'] if r['group'] == 'S')
        if count != layout['counts']['S'] or count != study['manifest']['resource_limits']['shared_parameter_S_numel'][dataset]:
            raise Incomplete('Shared S coordinate count differs')
        observed = metrics[(metrics.dataset == dataset) & (metrics.group_id == 'S')].actual_count.unique()
        if list(observed) != [count]:
            raise Incomplete('Metric S parameter count differs')
        resources[dataset] = {'S_coordinates': count, 'S_fp32_payload_bytes': count * 4, 'allocated_parameter_bytes': 0}
    main, source_ref = study['main_head'], study['source_head']
    ahead = git('rev-list', '--left-right', '--count', f'{source_ref}...{main}').split()
    graph = {'source_head': source_ref, 'main_head': main, 'merge_base': git('merge-base', source_ref, main),
             'source_ahead_main': int(ahead[0]), 'main_ahead_source': int(ahead[1]),
             'main_mechanism_code': git('ls-tree', main, 'experiments/ema_mechanism_20261106'),
             'main_mechanism_report': git('ls-tree', main, 'research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md')}
    record = {'status': 'VERIFIED', 'inventory': inventory, 'graph': graph, 'archive': recovery, 'panels': panels}
    write_json(out / 'AUDIT_INVENTORY.json', record)
    write_json(out / 'RESOLVED_PROTOCOL.json', study)
    write_json(out / 'RESOURCE_ESTIMATE.json', {'stage': 'A0', 'parameter_layouts': resources,
               'source_bytes': sum(r['bytes'] for r in inventory), 'new_GPU_seconds': 0, 'new_Adam_steps': 0,
               'prediction_copy_bytes': 0, 'output_cap_bytes': 2 ** 31})
    pd.DataFrame(inventory).to_csv(out / 'EVIDENCE_REUSE.csv', index=False)
    (out / 'SOURCE_AUDIT.md').write_text('# Source audit\n\n' + json.dumps(graph, indent=2) +
        '\n\nAll listed source blobs, scalar archive rows and observation panel identities verified. '
        'The mechanism report is present on current main; its implementation remains on the source branch.\n'
        'A0 reads metrics/logs and panel identities; new training, parameter states and Test access: 0.\n')
    return record
