"""Run the registered A40/A41 matrix after its independent review."""
import argparse
import os
from pathlib import Path

from . import history_stage
from .dose_runner import dose_config
from .protocol import ART, DATASETS, PACKAGE, REPORT, SEEDS, digest, read, sha, write
from .results import DOSE, stage_arms
from .state import generation

STAGE = 'A_DOSE'


def stopping(art, stage=STAGE):
    return history_stage.stopping(art, stage)


def gate(art):
    directory = art/'tables/A_HISTORY'
    previous = read(directory/'decisions.json')
    stage_arms(STAGE, previous)
    target = previous['target_dataset']
    assert (target, previous['history_form'], previous['scope']) == ('movielens','A21','S_DENSE')
    assert previous['datasets'][target]['eligible_history'] and previous['dose_motivation']['current_cell_tradeoff']
    assert previous['seeds'] == list(SEEDS) and not previous['missing_primary']
    assert (previous['split'], previous['readout'], previous['checkpoint_rule']) == ('validation','fixed_ema','selected')
    assert previous['previous_decision_sha256'] == digest(history_stage.gate(art))
    assert previous['statistics_script_sha256'] == sha(PACKAGE/'results.py')
    assert all(sha(directory/f) == h for f,h in previous['evidence'].items())
    return previous


def dose_source():
    prior = history_stage.history_source()
    files = dict(prior['files'])
    for name in ('dose_stage.py', 'dose_runner.py', 'dose_scheduler.py', 'dose_checks.py'):
        files[name] = sha(PACKAGE/name)
    current = dict(prior, files=files)
    del current['source_hash']
    return dict(current, source_hash=digest(current))


def jobs(stage, source_hash):
    if stage != STAGE:
        raise ValueError(stage)
    result = []
    for seed in SEEDS:
        for arm in DOSE:
            for dataset in DATASETS:
                job = dict(dataset=dataset, seed=seed, arm=arm, stage=stage, split_id='original',
                           source_hash=source_hash, config=dose_config(arm,dataset))
                result.append(dict(job, key=digest(job)))
    return result


def audit(art):
    previous = gate(art)
    prior = history_stage.reviewed(art)
    assert read(art/'A_HISTORY_RUN.json')['status'] == 'COMPLETE'
    current = dose_source()
    frozen = jobs(STAGE,current['source_hash'])
    result = dict(**current, stage=STAGE, status='SOURCE_READY', datasets=prior['datasets'],
                  a_core_source=prior['a_core_source'], a_history_source=prior['source_hash'],
                  gate_sha256=digest(previous), reuse_sha256=prior['reuse_sha256'],
                  owner_generation=generation(art/'attempt1'), job_count=len(frozen),
                  cuda_usable_total_bytes=prior['cuda_usable_total_bytes'])
    write(art/f'{STAGE}_SOURCE_MANIFEST.json',dict(**current,jobs=frozen))
    write(art/f'{STAGE}_AUDIT.json',result)
    return result


def reviewed(art):
    current = dose_source()
    audit_record = read(art/f'{STAGE}_AUDIT.json')
    check = read(art/f'{STAGE}_CHECK.json')
    review = read(REPORT/f'{STAGE}_REVIEW_RECEIPT.json')
    assert review['stage'] == STAGE and review['verdict'] in ('PASS','PASS WITH FIXES') and not review['blocking_issues']
    assert review['design_sha256'] == current['design'] and review['check_sha256'] == sha(art/f'{STAGE}_CHECK.json')
    assert current['source_hash'] == audit_record['source_hash'] == check['source_hash'] == review['source_hash']
    assert read(art/f'{STAGE}_SOURCE_MANIFEST.json') == dict(**current,jobs=jobs(STAGE,current['source_hash']))
    assert check['status'] == 'PASS' and audit_record['status'] == 'SOURCE_READY'
    assert check['owner_generation'] == audit_record['owner_generation'] == generation(art/'attempt1')
    assert audit_record['reuse_sha256'] == sha(art/'REUSE_MANIFEST.json')
    assert audit_record['gate_sha256'] == digest(gate(art))
    return audit_record


def analyze(art, localization=False):
    from .protocol import jobs as core_jobs
    from .results import audited_reuse, completed_rows, write_stage
    authority = reviewed(art)
    assert read(art/f'{STAGE}_RUN.json')['status'] == 'COMPLETE'
    previous = gate(art)
    frozen = core_jobs('A_CORE',authority['a_core_source'])
    frozen += history_stage.jobs('A_HISTORY',authority['a_history_source'])
    frozen += jobs(STAGE,authority['source_hash'])
    include = set(stage_arms(STAGE,previous))
    rows = [r for r in audited_reuse(read(art/'REUSE_MANIFEST.json')) if r['arm'] in include]
    for job in frozen:
        if job['arm'] not in include:
            continue
        path = art/'attempt1/runs'/job['key']/'result.json'
        result = read(path)
        assert result['job'] == job and result['data_hash'] == authority['datasets'][job['dataset']]
        assert result['global_step'] == (1000 if job['dataset']=='movielens' else 2450)
        assert result['test_evaluations'] == 0 and len(result['rows']) == 6
        rows.extend(completed_rows(path,result))
    output = art/'tables'/('A_DOSE' if localization else 'A_DOSE_PRELIM')
    records = None
    if localization:
        receipt = read(art/'A_TARGET_RUN.json')
        path = art/'tables/A_TARGET/localization.json'
        assert receipt['status'] == 'COMPLETE' and receipt['localization_sha256'] == sha(path)
        records = read(path)
    return write_stage(STAGE,rows,output,previous,records)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=('audit','check','run','_worker','analyze'))
    parser.add_argument('--artifacts',type=Path,default=ART)
    parser.add_argument('--stage',choices=(STAGE,),default=STAGE)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--key')
    parser.add_argument('--parent',type=int)
    parser.add_argument('--localization',action='store_true')
    args = parser.parse_args()
    art = args.artifacts.resolve()
    if args.action != 'analyze' and stopping(art):
        raise RuntimeError('Stop or pause control is active')
    if args.action == 'audit':
        result = audit(art)
    elif args.action == 'check':
        from .dose_checks import check
        result = check(art,args.gpu)
    elif args.action == 'run':
        from .dose_scheduler import supervise
        result = supervise(art,STAGE,reviewed(art))
    elif args.action == '_worker':
        from .dose_runner import worker
        if not args.parent or os.getppid() != args.parent:
            raise PermissionError('Mechanical worker requires foreground supervisor')
        authority = reviewed(art)
        job = next(j for j in jobs(STAGE,authority['source_hash']) if j['key']==args.key)
        result = worker(art/'attempt1',job,f'cuda:{args.gpu}',authority,
                        lambda:os.getppid()!=args.parent or stopping(art))
    else:
        result = analyze(art,args.localization)
    if result is not None:
        print(result['status'])


if __name__=='__main__':
    main()
