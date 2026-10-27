"""Read-only planning plus explicit data preparation, resident runs and release."""
from pathlib import Path
import argparse
import json
import sys
from .config import ROOT,DATASETS,DEV_SEEDS,LOCKED_SEEDS,load_config,plan
from .identity import plain,write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--artifact-root',type=Path,default=ROOT/'artifacts/mdmt_basp_v1')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('plan');sub.add_parser('audit')
    prepare=sub.add_parser('prepare');prepare.add_argument('--dataset',choices=DATASETS,required=True)
    train=sub.add_parser('worker');train.add_argument('--dataset',choices=DATASETS,required=True)
    train.add_argument('--seed',type=int,choices=DEV_SEEDS+LOCKED_SEEDS,required=True)
    train.add_argument('--device',default='cuda:0');train.add_argument('--locked-plan',type=Path)
    frozen=sub.add_parser('freeze');frozen.add_argument('--decision',type=Path,required=True)
    frozen.add_argument('--workers',nargs='+',required=True)
    evaluate=sub.add_parser('evaluate');evaluate.add_argument('--dataset',choices=DATASETS,required=True)
    evaluate.add_argument('--seed',type=int,choices=DEV_SEEDS+LOCKED_SEEDS,required=True)
    report=sub.add_parser('report');report.add_argument('--output',type=Path,default=ROOT/'research_reports/2026-10-28')
    args=parser.parse_args(argv);cfg=load_config(args.config)
    if args.command=='plan':
        result=plan(cfg)
    elif args.command=='audit':
        import subprocess
        import torch
        from .roles import RAW,PREPARED
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,power.draw,power.limit','--format=csv'],capture_output=True,text=True)
        result=dict(status='READ_ONLY',python=sys.version,torch=torch.__version__,
                    cuda_available=torch.cuda.is_available(),gpu=gpu.stdout,
                    data={d:dict(raw=str(RAW[d]),raw_exists=RAW[d].is_file(),prepared=str(PREPARED[d]),
                                 prepared_exists=(PREPARED[d]/'DATA_MANIFEST.json').is_file()) for d in DATASETS})
    elif args.command=='prepare':
        from .roles import verify_prepared
        result=verify_prepared(args.dataset,args.artifact_root/'data'/args.dataset/'DIRECT_VALIDATION.json')
    elif args.command=='worker':
        from .campaign import worker
        result=worker(cfg,args.dataset,args.seed,args.artifact_root,args.device,args.locked_plan)
    elif args.command=='freeze':
        from .campaign import freeze_campaign
        result=freeze_campaign(args.artifact_root,args.decision,args.workers)
    elif args.command=='evaluate':
        from .campaign import release_evaluation
        result=release_evaluation(args.artifact_root,args.dataset,args.seed)
    elif args.command=='report':
        from .report import collect
        result=collect(args.artifact_root,args.output)
    print(json.dumps(plain(result),ensure_ascii=False,indent=2,allow_nan=False),flush=True)
    return result


if __name__=='__main__':
    main()
