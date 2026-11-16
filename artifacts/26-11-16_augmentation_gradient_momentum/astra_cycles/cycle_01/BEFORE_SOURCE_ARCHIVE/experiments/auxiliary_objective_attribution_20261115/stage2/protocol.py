"""Independent Stage 2 identities and the reviewed Stage 1 CLEAN evidence index."""
import json
import subprocess
from pathlib import Path
from ..protocol import (ROOT,REPORT,SEEDS,DATASETS,SELECTION,digest,immutable,sha_file,
    load_protocol,dataset_cache,optimizer,make_model,setup,cpu_copy,state_digest,rng_state,
    restore_rng,evaluate,write_json,request,ensure_guardian,alive)
from ..protocol import source_manifest as stage1_source,results as stage1_results,epoch_files as stage1_epochs

HERE = Path(__file__).resolve().parent
STAGE1 = ROOT/'artifacts/auxiliary_objective_attribution_20261115/attempt1'
EXPECTED_STAGE1_HASH = 'a78bce2f713cf62f9af32548b50bc360b134420ff9ed6ac6990dca22d76a2ad0'
PROVENANCE = [REPORT/'STAGE2_DESIGN_SPECIFICATION.md',REPORT/'decision.json',REPORT/'STAGE1_FINAL_REVIEW.json',
    STAGE1/'source_receipt.json',STAGE1/'tables/SUPPORT_SUMMARY.csv',STAGE1/'tables/WEIGHTS.csv']


def source_manifest():
    original = stage1_source(); assert original['sha256'] == EXPECTED_STAGE1_HASH
    files = original['files']+[dict(path=str(p.relative_to(ROOT)),sha256=sha_file(p)) for p in
        [HERE/n for n in ('__init__.py','core.py','protocol.py','runner.py')]+PROVENANCE]
    return dict(files=files,sha256=digest(files),stage1_source_hash=EXPECTED_STAGE1_HASH)


def config(dataset):
    assert dataset in DATASETS
    p = load_protocol()['datasets'][dataset]
    return dict(dataset=dataset,arm='NDP-A',alpha=.03,mixture_lambda=.25,epochs=50,batch_size=40960,
        precision='float32',lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=p['weight_decay'],rho=p['rho'],
        objective='natural_row_task_convex_view_mixture_v1',support='all_2B_entries_including_zero_directions',
        gate_training_detached=True,probe_gate_full_derivative=True,
        data_hash=digest(p['source']),selection_hash=digest(SELECTION))


def job(dataset,seed):
    assert seed in SEEDS
    c = config(dataset)
    value = dict(dataset=dataset,seed=seed,config=c,config_hash=digest(c),source_hash=source_manifest()['sha256'])
    value['key'] = digest(value)
    return value


def check_review(path):
    review = json.loads(Path(path).read_text())
    assert review['verdict'] in ('PASS','PASS WITH FIXES') and not review.get('blocking_issues')
    assert review['source_hash'] == source_manifest()['sha256']
    return review


def prepare(attempt,cache_dir):
    attempt = Path(attempt); attempt.mkdir(parents=True,exist_ok=True)
    decision = json.loads((REPORT/'decision.json').read_text())
    review = json.loads((REPORT/'STAGE1_FINAL_REVIEW.json').read_text())
    assert decision['status'] == 'STAGE1_COMPLETED_ADVANCE_STAGE2' and decision['stage_advancement']['stage2']
    assert decision['independent_review']['verdict'] == review['verdict'] == 'PASS' and not review['blocking_issues']
    assert review['decision_hash'] == sha_file(REPORT/'decision.json') and not decision['test_accessed']
    source = source_manifest(); immutable(attempt/'source_receipt.json',source)
    clean = [r for r in stage1_results(STAGE1) if r['job']['config']['arm'] == 'CLEAN']
    assert len(clean) == 10 and {(r['job']['dataset'],r['job']['seed']) for r in clean} == {(d,s) for d in DATASETS for s in SEEDS}
    entries = []
    for result in clean:
        assert result['epochs'] == 50 and not result['test_accessed']
        path = STAGE1/'reused/artifacts/heterogeneity_aware_20261115/attempt1/runs'/result['job']['key']/'result.json'
        entries.append(dict(dataset=result['job']['dataset'],seed=result['job']['seed'],key=result['job']['key'],
            result_path=str(path.relative_to(ROOT)),result_sha256=sha_file(path)))
    immutable(attempt/'reused_clean_index.json',dict(trajectories=10,
        stage1_reuse_receipt_sha256=sha_file(STAGE1/'reuse_receipt.json'),entries=entries,
        zero_counterpart='Requires final same-source CPU and full-batch zero-equivalence smoke PASS'))
    immutable(attempt/'run_manifest.json',dict(stage='STAGE2',source_hash=source['sha256'],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seeds=list(SEEDS),datasets=list(DATASETS),arms=['NDP-A'],new_trajectories=10,reused_clean_trajectories=10,
        epochs=50,batch_size=40960,optimizer_updates=17250,ema_updates=17250,
        configs={d:config(d) for d in DATASETS},cache_dir=str(Path(cache_dir).resolve()) if cache_dir else None,
        selection=SELECTION,selection_hash=digest(SELECTION),model_storage='RAM_ONLY',test_accessed=False,
        timing_scope='CUDA synchronized wall time for probe, delta construction, and forwards/ledger/backward/Adam/EMA; validation separately',
        provenance=[dict(path=str(p.relative_to(ROOT)),sha256=sha_file(p)) for p in PROVENANCE],
        stage1_final_decision_sha256=sha_file(REPORT/'decision.json'),stage1_final_review_sha256=sha_file(REPORT/'STAGE1_FINAL_REVIEW.json')))
    for dataset in DATASETS:
        _,manifest = dataset_cache(dataset,cache_dir)
        immutable(attempt/f'{dataset}_data_manifest.json',manifest)
    return dict(status='PREPARED',source_hash=source['sha256'],new_trajectories=10,reused_clean=10)


def results(attempt):
    attempt = Path(attempt)
    paths = sorted((attempt/'runs').glob('*/result.json'))
    index = json.loads((attempt/'reused_clean_index.json').read_text())
    for row in index['entries']:
        path = ROOT/row['result_path']; assert sha_file(path) == row['result_sha256']; paths.append(path)
    return [json.loads(p.read_text()) for p in paths]


def epoch_files(attempt,result):
    if result['job']['config']['arm'] == 'CLEAN':
        return stage1_epochs(STAGE1,result)
    by_epoch = {}
    for path in sorted((Path(attempt)/'runs'/result['job']['key']).glob('attempt*/epoch_*.json')):
        value = json.loads(path.read_text()); by_epoch[value['epoch']] = (path,value)
    return by_epoch
