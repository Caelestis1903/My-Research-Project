"""Check archived endpoint identities and expose only Validation to the A gate."""
import csv
import json
from pathlib import Path
from .protocol import DATASETS, MISSING, SEEDS, M, P, ROOT, pinned, read, v3

SUP = ROOT/'research_reports/2026-11-02_mdmt_information_momentum_v3/supplement'
FIVE = ROOT/'research_reports/2026-11-03_mdmt_five_seed_completion/endpoints.csv'


def inventory():
    sources = [pinned(FIVE,M), pinned(SUP/'endpoints.csv',P), pinned(SUP/'selection_frozen.json',P)]
    old = ROOT/'research_reports/2026-11-02_mdmt_information_momentum_v3/SELECTION_FROZEN.json'
    sources.append(pinned(old,P))
    frozen = {(r['key'],r['checkpoint'],r['model_kind']):r for r in read(old)['endpoints'] + read(SUP/'selection_frozen.json')['endpoints']}
    rows = []
    for file in (FIVE, SUP/'endpoints.csv'):
        with file.open(newline='') as stream:
            for row in csv.DictReader(stream):
                five = file == FIVE
                if row['split' if five else 'role'] != 'validation':
                    continue
                if row['arm'] not in (('A0','P01') if five else ('P00','P10')):
                    continue
                dataset, seed = row['dataset'], int(row['seed'])
                kind = row['model_kind' if five else 'estimator']
                arm = {'P00':'A10','P01':'A20','P10':'A30','A0':'A00'}[row['arm']]
                checkpoint = row['checkpoint']
                model_hash = row['model_hash' if five else 'state_hash']
                if five:
                    if row['config_json']:
                        config = json.loads(row['config_json'])
                    else:
                        assert arm == 'A00'
                        archived = ROOT / Path(row['source_result']).parent
                        run = 'S5' if seed in MISSING else 'S1'
                        trial = 'A0' if seed in MISSING else 'C'
                        cfg = archived/f'runs/{run}/{dataset}/{trial}/seed{seed}/attempt1/resolved_config.json'
                        identity = read(cfg)
                        assert (identity['run_hash'],identity['config_hash']) == (row['key'],row['configuration'])
                        raw = identity['config']
                        rho = raw['shadow_rates']['H2' if dataset == 'movielens' else 'ema995']
                        assert not any(raw[n] for n in ('use_aig_m','use_sad','use_augmentation'))
                        assert raw['betas']==[.9,.999] and raw['eps']==1e-8
                        config = dict(arm='A0',dataset=identity['dataset'],rho=rho,lr=raw['lr'],
                                      weight_decay=raw['weight_decay'],epochs=identity['epochs'],
                                      batch_size=identity['batch_size'],legacy_config_ref=str(cfg.relative_to(ROOT)))
                else:
                    item = frozen[(row['key'],checkpoint,kind)]
                    config = item['job']['config']
                    assert item['model_hash'] == model_hash and item['epoch'] == int(row['epoch'])
                expected = v3.config(row['arm'],dataset,0 if arm=='A00' else .05)
                # Five-seed baseline config may include further explicit Adam/source identity fields.
                for name,value in expected.items():
                    if name in config:
                        assert config[name] == value, (dataset,arm,seed,name)
                for name in ('arm','dataset','lr','weight_decay','rho','batch_size','epochs'):
                    assert name in config
                metrics = {k:float(v) for k,v in row.items() if v and k.endswith(('_auc','_logloss','_count','_positive','_negative'))}
                assert len(model_hash)==64 and all(f'd{d}t{t}_auc' in metrics and f'd{d}t{t}_logloss' in metrics for d in range(3) for t in range(2))
                rows.append(dict(dataset=dataset, arm=arm, seed=seed, checkpoint_rule=checkpoint,readout=kind,split='validation',reuse_status='REUSE_NUMERIC',
                                 epoch=int(row['epoch']), model_hash=model_hash, config=config, source_key=row['key'],
                                 source_ref=str(file.relative_to(ROOT)), trajectory_id=row['key'], **metrics))
    expected = {(dataset,arm,seed,checkpoint,kind) for dataset in DATASETS for arm in ('A00','A10','A20','A30')
                for seed in SEEDS if arm in ('A00','A20') or seed not in MISSING
                for checkpoint in ('selected','E10','E50') for kind in ('online','fixed_ema')}
    observed = {(r['dataset'],r['arm'],r['seed'],r['checkpoint_rule'],r['readout']) for r in rows}
    assert observed == expected and len(rows) == len(expected)
    return dict(sources=sources, numeric_trajectories=28, rows=rows, selected_RAM_claims=0,
                cost_class='REUSE_NUMERIC', historical_test='HISTORICAL_EXPOSED', new_test_evaluations=0)
