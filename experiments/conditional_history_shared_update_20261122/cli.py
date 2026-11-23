"""Runbook CLI for staged, supervised execution."""
import argparse
import json
import os
from pathlib import Path

from .protocol import ART, PLAN, REPORT, digest, jobs, read, sha, source, write
from .state import generation


def reviewed(art,stage):
    current = source(stage)
    audit = read(art/f'{stage}_AUDIT.json')
    check = read(art/f'{stage}_CHECK.json')
    review = read(REPORT/f'{stage}_REVIEW_RECEIPT.json')
    assert review['verdict'] in ('PASS','PASS WITH FIXES') and review['blocking_issues']==[]
    assert review['stage']==stage and review['source_hash']==current['source_hash']
    assert review['design_sha256']==current['design']
    assert audit['source_hash']==check['source_hash']==current['source_hash']
    assert audit['status']=='SOURCE_READY' and check['status']=='PASS'
    assert review['check_sha256']==sha(art/f'{stage}_CHECK.json')
    assert audit['reuse_sha256']==sha(art/'REUSE_MANIFEST.json')
    assert generation(art/'attempt1')==check['owner_generation']
    return audit


def analyze(art,stage):
    if stage == 'B_CORE':
        from .b_stage import analyze_core
        return analyze_core(art,reviewed(art,stage))
    from .results import write_stage, audited_reuse, completed_rows
    authority = reviewed(art,stage)
    assert stage=='A_CORE'
    rows = list(audited_reuse(read(art/'REUSE_MANIFEST.json')))
    for job in jobs(stage,authority['source_hash']):
        out = art/'attempt1/runs'/job['key']
        result = read(out/'result.json')
        assert result['job']==job and result['epochs']==50
        assert digest(result)==read(out/'COMPLETE_VERIFIED.json')['result_hash']
        rows.extend(completed_rows(out/'result.json',result))
    return write_stage(stage,rows,art/'tables'/stage)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=('audit','check','run','_worker','analyze','advance','freeze','evaluate','report'))
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--artifacts',type=Path,default=ART)
    parser.add_argument('--stage',choices=('A_CORE','B_CORE'),default='A_CORE')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--key')
    parser.add_argument('--parent',type=int)
    args = parser.parse_args()
    art,stage = args.artifacts.resolve(),args.stage
    if args.plan.resolve()!=PLAN:
        parser.error('Expected registered 2026-11-22 plan path')
    if args.action=='audit':
        from .checks import audit
        result = audit(art,stage)
    elif args.action=='check':
        if stage=='B_CORE':
            from .b_checks import check
        else:
            from .checks import check
        result = check(art,stage,args.gpu)
    elif args.action=='run':
        from .scheduler import supervise,stopping
        if stopping(art,stage):
            raise RuntimeError('Stage is paused. Resolve the logged cause before resuming.')
        if stage=='B_CORE' and not (art/'tables/A_CORE/decisions.json').exists():
            raise RuntimeError('S2 A decision pending before S3 replay')
        result = supervise(art,stage,reviewed(art,stage))
    elif args.action=='_worker':
        from .runner import worker
        from .scheduler import stopping
        if not args.parent or os.getppid()!=args.parent:
            raise PermissionError('Mechanical worker requires its foreground supervisor')
        authority = reviewed(art,stage)
        job = next(j for j in jobs(stage,authority['source_hash']) if j['key']==args.key)
        result = worker(art/'attempt1',job,f'cuda:{args.gpu}',authority,
                        lambda:os.getppid()!=args.parent or stopping(art,stage))
    elif args.action=='analyze':
        result = analyze(art,stage)
    elif args.action=='advance':
        if not (art/'A_CORE_RUN.json').exists() or read(art/'A_CORE_RUN.json')['status']!='COMPLETE':
            result = dict(next='A_CORE',protocol='S1')
        elif not (art/'tables/A_CORE/decisions.json').exists():
            result = dict(next='analyze A_CORE',protocol='S2')
        elif not (art/'B_CORE_RUN.json').exists() or read(art/'B_CORE_RUN.json')['status']!='COMPLETE':
            result = dict(next='B_CORE',protocol='S3')
        else:
            from .b_stage import advance
            result = advance(art)
    else:
        raise RuntimeError(f'{args.action} requires the reviewed S5 implementation and completed prescribed stages')
    if result is not None:
        print(json.dumps({k:v for k,v in result.items() if k in ('status','source_hash','next','protocol')},ensure_ascii=False))


if __name__=='__main__':
    main()
