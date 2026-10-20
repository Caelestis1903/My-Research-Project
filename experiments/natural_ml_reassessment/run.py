"""One independently reviewed formal configuration; no automatic shortcuts."""
import argparse
import json
from pathlib import Path
from .bootstrap import ROOT, source_hashes
from .manifest import configurations

def check_review(path,cfg):
    gate=json.loads(Path(path).read_text())
    if gate.get('verdict') not in ('PASS','PASS WITH FIXES') or gate.get('blocking_issues'):
        raise RuntimeError('independent review is missing a passing resolved verdict')
    if cfg['stage'] not in gate.get('stages',[]):raise RuntimeError('review does not cover requested stage')
    extra=('controlled.py',) if cfg['stage'] in ('AC-S5','AC-S6') else ('v9.py',) if cfg['stage']=='AC-V9' else ()
    current=source_hashes(extra)
    reviewed=gate.get('source_hashes',{})
    if any(reviewed.get(k)!=v for k,v in current.items()):
        raise RuntimeError('reviewed source identity differs from current source')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--review-report',default=str(ROOT/'review_gate.json'));args=parser.parse_args()
    cfg=next(x for x in configurations() if x['run_id']==args.run_id)
    check_review(args.review_report,cfg)
    if cfg['stage']=='AC-2D':
        from .stage2d import Runner
    elif cfg['stage']=='AC-V9':
        from .v9 import Runner
    else:
        from .controlled import Runner
    runner=Runner(cfg,args.device);runner.fit()

if __name__=='__main__':main()
