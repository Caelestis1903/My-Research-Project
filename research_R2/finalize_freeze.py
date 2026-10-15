"""Main-controller bookkeeping: freeze valid completed runs before final test.

Contains no training, model construction, prediction, or test metrics.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'formal_runs_20261012'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text())

def write_once(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def main():
    fixes = ROOT / 'R2_Test_Path_Independent_Review.md'
    assert fixes.exists(), 'test-path independent review missing'
    review = fixes.read_text()
    assert 'PASS' in review and 'FAIL' not in review, 'test-path review not passed'
    term = ROOT / 'formal_decisions_20261012/kuairand_r2a_termination_manifest.json'
    decision = ROOT / 'formal_decisions_20261012/r2a_decision_movielens.json'
    ml = read(decision)
    assert read(term)['status'] == 'INCOMPLETE'
    assert ml['valid'] and ml['decision'] == 'NO'
    assert ml['test_prediction_calls_at_decision'] == 0
    mapping = read(ROOT / 'preflight/formal_execution_mapping_20261012.json')
    code = {}
    for rel, old_hash in mapping['code_hashes'].items():
        path = ROOT.parent / rel
        observed = digest(path)
        if rel != 'research_R2/src/ao04_v12_test_evaluator.py':
            assert observed == old_hash, f'code drift: {rel}'
        else:
            assert observed == 'bb49baacb2d005ff93b1d510a1d085b1f2267692b04e80d34eeff6a98dbf8a06', 'unreviewed evaluator drift'
        code[rel] = observed
    frozen = []
    for path in sorted(RUNS.rglob('run_manifest.json')):
        m = read(path)
        if m['status'] != 'COMPLETE':
            continue
        assert m['epochs_completed'] == 50 and not m['smoke_completed']
        assert not m['test_evaluated_during_training'] and not m['test_predictions_written']
        run = path.parent
        assert not (run / 'metrics_test.csv').exists(), 'test already performed'
        with (run / 'metrics_epoch.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 300
        history = {}
        for epoch in range(50):
            group = [r for r in rows if int(r['epoch']) == epoch]
            assert {(int(r['task']), int(r['domain'])) for r in group} == {(t,d) for t in range(2) for d in range(3)}
            values = [float(r['validation_auc']) for r in group]
            assert all(math.isfinite(v) for v in values)
            history[epoch] = sum(values) / 6
        selected = max(history, key=history.get)
        assert selected == m['selected_epoch'], 'selection mismatch'
        config = read(run / 'config.json')
        runtime_config = read(ROOT / f"configs/runtime_20261012/{m['dataset']}_{m['setting']}_m3oe.json")
        for field, value in runtime_config.items():
            if field in config:
                assert config[field] == value, f'config drift: {field}'
        assert digest(Path(runtime_config['dataset_path'])) == runtime_config['dataset_sha256']
        init_path = Path(m['paired_init_manifest'])
        assert digest(init_path) == m['paired_init_manifest_sha256']
        required = ['run_manifest.json', 'config.json', 'selection.json', 'metrics_epoch.csv',
                    'fusion_log.csv', 'parameter_manifest.json', 'batch_trace.csv',
                    'validation_predictions_selected.npz', 'selected_checkpoint.pth',
                    'final_checkpoint.pth', 'optimizer_state.pth', 'architecture_optimizer_state.pth',
                    'rng_state.pth', 'last_resume.pt', 'stdout.log', 'stderr.log']
        required += [f'common_epoch_{e}.pth' for e in (9,19,29,39,49)]
        files = {}
        for name in required:
            artifact = run / name
            assert artifact.is_file(), f'missing {artifact}'
            if name.endswith(('.pth', '.pt', '.npz')):
                assert artifact.stat().st_size > 0
            files[name] = {'sha256': digest(artifact), 'bytes': artifact.stat().st_size}
        frozen.append({'dataset': m['dataset'], 'setting': m['setting'], 'arm': m['arm'],
                       'seed': m['seed'], 'backbone': 'm3oe', 'run_dir': str(run),
                       'status': 'COMPLETE', 'epochs_completed': 50, 'selected_epoch': selected,
                       'batch_order_hash': m['batch_order_hash'],
                       'architecture_batch_order_hash': m['architecture_batch_order_hash'],
                       'paired_init_manifest_sha256': m['paired_init_manifest_sha256'], 'files': files})
    assert len(frozen) == 9, 'unexpected evaluation scope'
    assert {(r['arm'], r['seed']) for r in frozen if r['dataset']=='movielens'} == {(a,s) for a in ('B0','B1','B2') for s in (2022,2023)}
    assert {(r['arm'], r['seed']) for r in frozen if r['dataset']=='kuairand'} == {('B0',2022),('B0',2023),('B1',2022)}
    for dataset, seeds, arms in [('movielens',(2022,2023),('B0','B1','B2')), ('kuairand',(2022,),('B0','B1'))]:
        for seed in seeds:
            group = [r for r in frozen if r['dataset']==dataset and r['seed']==seed and r['arm'] in arms]
            for key in ('batch_order_hash','architecture_batch_order_hash','paired_init_manifest_sha256'):
                assert len({r[key] for r in group}) == 1, f'pairing mismatch {dataset} {seed}'
    freeze = {'created_at': datetime.now(timezone.utc).isoformat(), 'signed_by':'Main Controller',
              'stage':'R2', 'global_status':'INCOMPLETE_RESTRICTED_FINALIZATION',
              'hpost_enabled':False, 'training_closed':True, 'runs':frozen,
              'code_hashes':code, 'test_path_review_sha256':digest(fixes),
              'dataset_outcomes':{'kuairand':{'status':'INCOMPLETE','termination_manifest':str(term),'sha256':digest(term)},
                                  'movielens':{'status':'VALID_R2A_NO','decision_manifest':str(decision),'sha256':digest(decision)}},
              'excluded_runs':['kuairand/B1/2023: INCOMPLETE; not eligible for test',
                               'kuairand/B2/2022,2023: NOT_RUN_AFTER_TERMINATION',
                               'all B3/B4: NOT_SCHEDULED'],
              'test_scope':'selected/final/common 9,19,29,39,49 for only the nine COMPLETE runs',
              'test_prediction_calls_before_unlock':0,
              'prohibitions':['no new training','no reselect by test','no retry failed run','no R3/R4']}
    target = ROOT / 'locks/global_selection_freeze.json'
    write_once(target, freeze)
    write_once(ROOT / 'locks/global_test_unlock.json',
               {'global_test_unlock':True, 'created_at':freeze['created_at'], 'signed_by':'Main Controller',
                'freeze_manifest':str(target), 'freeze_manifest_sha256':digest(target),
                'valid_complete_runs':frozen, 'hpost_enabled':False,
                'scope':'restricted final report after frozen native-failure termination; no complete dual-dataset claim',
                'dataset_outcomes':freeze['dataset_outcomes'], 'test_prediction_calls_before_unlock':0})
    print(json.dumps({'freeze':str(target),'unlock':str(ROOT/'locks/global_test_unlock.json'),'eligible_runs':len(frozen)}))

if __name__ == '__main__':
    main()
