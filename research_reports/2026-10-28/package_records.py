"""Package scalar/formation records and retain an explicit index of local row data."""
from pathlib import Path
import csv
import json
import shutil
import tarfile

REPORT = Path(__file__).resolve().parent
REPO = REPORT.parents[1]
ROOT = REPO/'artifacts/mdmt_basp_v1'
CANONICAL = ROOT/'attempt2'


def package():
    frozen = json.loads((CANONICAL/'GLOBAL_FREEZE.json').read_text())
    assert frozen['status']=='FROZEN' and len(frozen['workers'])==4
    identities, costs = [], []
    for key, worker in frozen['workers'].items():
        directory = CANONICAL/'workers'/key
        assert json.loads((directory/'STATUS.json').read_text())['status']=='COMPLETE'
        receipt = json.loads((directory/'HISTORICAL_E_RECEIPT.json').read_text())
        assert receipt['identity']==worker and receipt['frozen_version']==frozen['version']
        assert (directory/'EVALUATE.json').stat().st_mtime >= (CANONICAL/'GLOBAL_FREEZE.json').stat().st_mtime
        for name, method in worker['methods'].items():
            for state in ('selected','final'):
                identities.append(dict(dataset=worker['dataset'],seed=worker['seed'],method=name,state=state,
                    epoch=method[state]['epoch'],main_step=method[state]['main_step'],storage='RAM_ONLY_RELEASED_AFTER_E',
                    identity=f'{key}/{name}/{state}/epoch{method[state]["epoch"]}/step{method[state]["main_step"]}'))
    training = [json.loads(p.read_text()) for p in CANONICAL.glob('runs/*/*/seed*/TRAINING_RECEIPT.json')]
    assert len(training)==24 and all(x['main_updates']==x['expected_main_updates'] for x in training)
    assert len(list(CANONICAL.glob('runs/*/*/seed*/E_*_metrics.json')))==48
    assert len(list(CANONICAL.glob('workers/*/seed*/E_NATIVE_RAW_metrics.json')))==4
    for p in CANONICAL.glob('runs/*/*/seed*/EPOCH_METRICS.jsonl'):
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        assert [r['epoch'] for r in rows]==list(range(1,21))
    plans=[json.loads(p.read_text()) for p in CANONICAL.glob('runs/*/*/seed*/interventions/*/INTERVENTION_PLAN.json')]
    assert len(plans)==24 and all(p['mother_state_parity']=='PASS' for p in plans)
    with (REPORT/'CHECKPOINT_IDENTITIES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,identities[0]);writer.writeheader();writer.writerows(identities)
    shutil.copyfile(CANONICAL/'GLOBAL_FREEZE.json',REPORT/'GLOBAL_FREEZE.json')
    shutil.copyfile(CANONICAL/'workers/movielens/seed2022/EXECUTION_CONFIG.json',REPORT/'EXECUTION_CONFIG.json')
    archive=REPORT/'RAW_RECORDS.tar.gz'
    included,local=[],[]
    forbidden={'.pt','.pth','.ckpt','.pkl','.pickle','.safetensors'}
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file():continue
            assert p.suffix not in forbidden, f'Model-state file is forbidden: {p}'
            keep=p.suffix in {'.json','.csv','.log'} or (p.suffix=='.jsonl' and p.name!='LABEL_ACCESS.jsonl')
            keep |= p.suffix=='.npz' and (p.name.startswith('F_') or '_F_FOLD_GAIN_PREDICTIONS' in p.name or p.name.endswith('_MEASURED_RELATION.npz'))
            if 'row_data' in p.parts:keep=False
            record=dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size)
            if keep:
                tar.add(p,arcname=record['path'],recursive=False);included.append(record)
            else:
                record['category']='row_cache' if 'row_data' in p.parts else 'full_label_access' if p.name=='LABEL_ACCESS.jsonl' else 'row_predictions_or_gain_observations'
                local.append(record)
    index=dict(archive=archive.name,archive_bytes=archive.stat().st_size,included_files=len(included),
        included_uncompressed_bytes=sum(x['bytes'] for x in included),
        included_categories='All scalar JSON/CSV, epoch and execution logs, F gain/fold arrays, and measured relation arrays, including failed/reallocated attempts.',
        local_large_data='Full row predictions, Q/E per-row gains, full label-access logs and cached prefix outputs remain at the indexed workspace paths.',
        local_files=local,model_state_files=0,canonical_root=str(CANONICAL))
    (REPORT/'PUBLISH_DATA_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(status='COMPLETE_UNDER_PREREGISTERED_STOP',canonical_root=str(CANONICAL),
        datasets=['movielens','kuairand'],development_seeds=[2022,2023],locked_seeds=[2024,2025,2026],locked_status='NOT_TRIGGERED',
        canonical_prefixes=4,native_epochs=96,pilot_passes=48,adaptation_trajectories=24,adaptation_epochs=480,
        adaptation_main_updates=sum(x['main_updates'] for x in training),historical_E_adaptation_evaluations=48,native_E_scores=4,
        scalar_metric_records=156,cell_metric_records=936,intervention_state_diagnostics=24,
        additional_reader_updates=sum(p['cost']['additional_reader_steps'] for p in plans),
        static_contract_checks=21,core_tests=29,intervention_tests=3,
        stage_decision='STAGE_DECISION.json',frozen_selection='GLOBAL_FREEZE.json',model_storage='RAM_ONLY_RELEASED_AFTER_E',
        failures='EXPERIMENT_EXECUTION_REPORT_ATTEMPT1.md',runtime_fix='RUNTIME_FIX1_REVIEW.md',
        resource_reallocation='artifacts/mdmt_basp_v1/attempt2/resource_reschedule_attempt1/RESOURCE_RESCHEDULE.json',
        publication=dict(repository='Caelestis1903/My-Research-Project',requested_branch='26-10-28'),all_worker_statuses='COMPLETE')
    (REPORT/'RUN_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    context=json.loads((REPORT/'EXECUTION_CONTEXT.json').read_text())
    context.update(status='EXPERIMENT_AND_ANALYSIS_COMPLETE',canonical_artifact_root=str(CANONICAL))
    context['stages'].update(IMPLEMENT='COMPLETE',REVIEW='PASS_WITH_FIXES_ZERO_OPEN_BLOCKING',RUN='COMPLETE',ANALYZE='COMPLETE')
    (REPORT/'EXECUTION_CONTEXT.json').write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n')
    return dict(included_files=len(included),archive_bytes=archive.stat().st_size,local_files=len(local),manifest=manifest)


if __name__=='__main__':
    result=package();print(json.dumps({k:v for k,v in result.items() if k!='manifest'}))
