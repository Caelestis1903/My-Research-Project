"""Stage-scoped CLI for the registered A_HISTORY controls."""
import argparse
import os
from pathlib import Path

from .cli import reviewed as reviewed_core
from .history_controls import history_config
from .protocol import ART, DATASETS, PACKAGE, REPORT, ROOT, SEEDS, digest, read, sha, source, write
from .state import generation

STAGE = 'A_HISTORY'


def gate(art):
    from .results import stage_arms
    directory = art/'tables/A_CORE'
    previous = read(directory/'decisions.json')
    stage_arms(STAGE, previous)
    assert previous['seeds'] == list(SEEDS) and not previous['missing_primary']
    assert previous['split']=='validation' and previous['readout']=='fixed_ema' and previous['checkpoint_rule']=='selected'
    assert any(previous['datasets'][d]['history_vs_adam']['delta_auc']>0 for d in DATASETS)
    assert previous['statistics_script_sha256']==sha(PACKAGE/'results.py')
    assert all(sha(directory/f)==h for f,h in previous['evidence'].items())
    return previous


def history_source():
    current = source('A_CORE')
    files = dict(current['files'])
    for name in ('history_controls.py','history_runner.py','history_stage.py','history_scheduler.py',
                 'history_checks.py','results.py'):
        files[name] = sha(PACKAGE/name)
    value = dict(current,files=files)
    del value['source_hash']
    return dict(value,source_hash=digest(value))


def jobs(stage, source_hash):
    assert stage==STAGE
    result = []
    for seed in SEEDS:
        for arm in ('A21','A22'):
            for dataset in DATASETS:
                job = dict(dataset=dataset,seed=seed,arm=arm,stage=stage,split_id='original',
                           source_hash=source_hash,config=history_config(arm,dataset))
                result.append(dict(job,key=digest(job)))
    return result


def stopping(art,stage=STAGE):
    from .scheduler import stopping as stage_stop
    if stage_stop(art,stage):
        return True
    for workspace in (ROOT,ROOT.parent/'phase0-3_tvg_rm_20261116'):
        folder = workspace/'artifacts/26-11-16_augmentation_gradient_momentum'
        if (folder/'STOP_AUTONOMOUS').exists():
            return True
        # The retired Nov 16 driver's final USER_STOPPED receipt is historical.
        if stage_stop(workspace/'artifacts'/ART.name,stage):
            return True
    return False


def audit(art):
    base = reviewed_core(art,'A_CORE')
    previous = gate(art)
    current = history_source()
    assert read(art/'A_CORE_RUN.json')['status']=='COMPLETE'
    table_review = read(REPORT/'TABLES_REVIEW_RECEIPT.json')
    assert table_review['verdict'] in ('PASS','PASS WITH FIXES') and table_review['blocking_issues']==[]
    assert table_review['files']['results.py']==current['files']['results.py']
    frozen = jobs(STAGE,current['source_hash'])
    result = dict(**current,stage=STAGE,status='SOURCE_READY',datasets=base['datasets'],
                  a_core_gate_sha256=sha(art/'tables/A_CORE/decisions.json'),a_core_source=base['source_hash'],
                  gate_sha256=digest(previous),reuse_sha256=sha(art/'REUSE_MANIFEST.json'),
                  owner_generation=generation(art/'attempt1'),job_count=len(frozen),
                  cuda_usable_total_bytes=base['cuda_usable_total_bytes'])
    write(art/f'{STAGE}_SOURCE_MANIFEST.json',dict(**current,jobs=frozen))
    write(art/f'{STAGE}_AUDIT.json',result)
    return result


def reviewed(art):
    authority = read(art/f'{STAGE}_AUDIT.json')
    check = read(art/f'{STAGE}_CHECK.json')
    review = read(REPORT/f'{STAGE}_REVIEW_RECEIPT.json')
    current = history_source()
    assert review['verdict'] in ('PASS','PASS WITH FIXES') and not review['blocking_issues']
    assert review['stage']==STAGE and review['design_sha256']==current['design']
    assert review['source_hash']==current['source_hash']==authority['source_hash']==check['source_hash']
    assert review['check_sha256']==sha(art/f'{STAGE}_CHECK.json')
    assert authority['status']=='SOURCE_READY' and check['status']=='PASS'
    assert authority['gate_sha256']==digest(gate(art))
    assert authority['reuse_sha256']==sha(art/'REUSE_MANIFEST.json')
    assert authority['owner_generation']==check['owner_generation']==generation(art/'attempt1')
    return authority


def analyze(art):
    from .protocol import jobs as core_jobs
    from .results import audited_reuse, completed_rows, write_stage
    authority = reviewed(art)
    assert read(art/f'{STAGE}_RUN.json')['status']=='COMPLETE'
    rows = list(audited_reuse(read(art/'REUSE_MANIFEST.json')))
    frozen = core_jobs('A_CORE',authority['a_core_source']) + jobs(STAGE,authority['source_hash'])
    for job in frozen:
        path = art/'attempt1/runs'/job['key']/'result.json'
        result = read(path)
        assert result['job']==job and result['data_hash']==authority['datasets'][job['dataset']]
        rows.extend(completed_rows(path,result))
    previous = gate(art)
    return write_stage(STAGE,rows,art/'tables'/STAGE,previous)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=('audit','check','run','_worker','analyze'))
    parser.add_argument('--artifacts',type=Path,default=ART)
    parser.add_argument('--stage',choices=(STAGE,),default=STAGE)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--key')
    parser.add_argument('--parent',type=int)
    args = parser.parse_args()
    art = args.artifacts.resolve()
    if args.action in ('audit','check','run','_worker') and stopping(art):
        raise RuntimeError('Stop or pause control is active')
    if args.action=='audit':
        result = audit(art)
    elif args.action=='check':
        from .history_checks import check
        result = check(art,args.gpu)
    elif args.action=='run':
        from .history_scheduler import supervise
        result = supervise(art,STAGE,reviewed(art))
    elif args.action=='_worker':
        from .history_runner import worker
        if not args.parent or os.getppid()!=args.parent:
            raise PermissionError('Mechanical worker requires foreground supervisor')
        authority = reviewed(art)
        job = next(j for j in jobs(STAGE,authority['source_hash']) if j['key']==args.key)
        result = worker(art/'attempt1',job,f'cuda:{args.gpu}',authority,
                        lambda:os.getppid()!=args.parent or stopping(art))
    else:
        result = analyze(art)
    if result is not None:
        print(result['status'])


if __name__=='__main__':
    main()
