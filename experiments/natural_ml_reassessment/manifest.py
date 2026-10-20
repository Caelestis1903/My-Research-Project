"""120 mandatory core runs; every budget and seed is explicit before training."""
import argparse
from .bootstrap import ROOT, PYTHON, dump, source_hashes
from .data import POPULATIONS

SEEDS = (2022, 2023, 2024, 2025)

def configurations():
    result = []
    for population, budget in [(p, 'full50') for p in POPULATIONS]+[(POPULATIONS[0], 'matched250')]:
        for seed in SEEDS:
            for arm in ('SN', 'SB', 'EN', 'EB'):
                result.append(dict(stage='AC-2D', population=population, budget=budget, seed=seed, arm=arm))
    for stage, arms in [('AC-V9', ('SN','SB')), ('AC-S5', ('F','T','HT','DH')), ('AC-S6', ('F','DH'))]:
        for budget in (('full50',) if stage=='AC-V9' else ('full50','matched250')):
            for seed in SEEDS:
                for arm in arms:
                    result.append(dict(stage=stage, population=POPULATIONS[0], budget=budget, seed=seed, arm=arm))
    for cfg in result:
        cfg['run_id'] = '__'.join(str(cfg[k]) for k in ('stage','population','budget','arm','seed'))
        cfg.update(epochs=50, batch_size=40960, architecture_steps=0 if cfg['stage']=='AC-V9' else 50,
                   main_steps=250 if cfg['budget']=='matched250' or cfg['population']!=POPULATIONS[0] else 1000,
                   expected_main_examples_per_epoch=187500 if cfg['budget']=='matched250' or cfg['population']!=POPULATIONS[0] else 800167)
        cfg['run_dir'] = str(ROOT/'runs'/cfg['run_id'])
        cfg['command'] = [PYTHON, '-m', 'experiments.natural_ml_reassessment.run', '--run-id', cfg['run_id'], '--device', 'cuda:0']
        cfg['status'] = 'PENDING_INDEPENDENT_REVIEW'
    assert len(result)==120
    assert len({r['run_id'] for r in result})==120
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default=str(ROOT/'core_manifest.json'));args=p.parse_args()
    from .mechanism import configurations as diagnostics
    runs=configurations()
    dump(args.output,dict(study='natural_movielens_reassessment',review_gate='independent PASS required before training',
                          sources=source_hashes(),formal_training_runs=120,offline_diagnostic_runs_separate=True,
                          runs=runs,offline_diagnostics=diagnostics(runs)))

if __name__=='__main__': main()
