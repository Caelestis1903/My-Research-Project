"""Read-only audit/plan and explicit preparation, resident training and export."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import torch
from .config import DEFAULT, ROOT, DATASETS, SEEDS, load_config, plan
from .data import RAW, EXPECTED, prepare_data
from .state import plain, sha_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--artifact-root', type=Path, default=ROOT/'artifacts/mdmt_relation_basis_v2')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('audit')
    sub.add_parser('plan')
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--dataset', choices=DATASETS, required=True)
    for name in ('train', 'worker'):
        train_parser = sub.add_parser(name)
        train_parser.add_argument('--dataset', choices=DATASETS, required=True)
        train_parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
        train_parser.add_argument('--device', default='cuda:0')
        train_parser.add_argument('--freeze-file', type=Path)
    report_parser = sub.add_parser('report')
    report_parser.add_argument('--report-root', type=Path, default=ROOT/'research_reports/26-10-27_v2')
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--decision', type=Path, required=True)
    evaluate_parser = sub.add_parser('evaluate')
    evaluate_parser.add_argument('--dataset', choices=DATASETS, required=True)
    evaluate_parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == 'plan':
        result = plan(cfg)
    elif args.command == 'audit':
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,power.draw,power.limit', '--format=csv'], capture_output=True, text=True, check=False).stdout
        except FileNotFoundError:
            gpu = 'nvidia-smi unavailable'
        result = dict(mode='READ_ONLY', python=sys.version, torch=torch.__version__,
                      cuda_available=torch.cuda.is_available(), gpu=gpu,
                      data={d:dict(raw_path=p, exists=Path(p).exists(),
                                   raw_sha256=sha_file(p) if Path(p).is_file() else None,
                                   expected_sha256=EXPECTED[d],
                                   prepared=(args.artifact_root/'data'/d/'VALIDATED.json').exists())
                            for d, p in RAW.items()}, plan=plan(cfg))
    elif args.command == 'prepare':
        result = prepare_data(args.dataset, args.artifact_root)
    elif args.command in ('train', 'worker'):
        from .campaign import worker
        result = worker(cfg, args.dataset, args.seed, args.artifact_root, args.device, args.freeze_file)
    elif args.command == 'report':
        from .report import write_report
        report = write_report(args.artifact_root, args.report_root)
        result = {k:report[k] for k in ('status', 'completed_adaptation_runs', 'expected_adaptation_runs', 'development_gates')}
    elif args.command == 'freeze':
        from .report import freeze_manifest
        result = freeze_manifest(args.artifact_root, cfg, args.decision)
    elif args.command == 'evaluate':
        from .report import release_evaluation
        result = release_evaluation(args.artifact_root, args.dataset, args.seed)
    print(json.dumps(plain(result), ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()

