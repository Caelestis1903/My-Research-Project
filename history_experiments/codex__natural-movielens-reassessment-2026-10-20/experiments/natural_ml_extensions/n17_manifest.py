"""N17 72 prespecified configurations, symmetric pilot freeze and J-only H200 gate."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .n17_bootstrap import OUT,task_path,sources,sha,dump
SEEDS=(2022,2023,2024,2025)
LRS=(.0003,.001,.003,.01)

def configuration(group,arm,lr,seed,population='movielens_natural',epoch_examples=None,sampling='iid'):
    return dict(id=f'{group}_{arm}_lr{lr:g}_seed{seed}',group=group,arm=arm,lr=lr,seed=seed,
        population=population,epochs=100,batch_size=40960,epoch_examples=epoch_examples,
        sampling=sampling,run_dir=str(task_path(group,arm,lr,seed)),
        optimizer='adam',weight_decay=1e-5,betas=[.9,.999],eps=1e-8,
        exploratory_replication=True,test_deferred=True)

def base_configurations():
    configs=[configuration('pilot',arm,lr,seed) for lr in LRS for arm in ('SN','SB') for seed in (2030,2031)]
    for group,pop,n in [('formal','movielens_natural',None),
                        ('matched_budget','movielens_natural',187500),
                        ('matched_n','movielens_natural_matched_n187500',None)]:
        configs.extend(configuration(group,arm,lr,seed,pop,n) for lr in LRS[:2] for arm in ('SN','SB') for seed in SEEDS)
    # Quota is required; LR is unresolved until all sixteen full pilots complete.
    configs.extend(dict(id=f'quota_{arm}_seed{seed}',group='quota',arm=arm,seed=seed,
        lr='PILOT_COMMON_LR',epochs=100,population='movielens_natural',sampling='quota',
        blocked_by='winners_frozen.json') for arm in ('SN','SB') for seed in SEEDS)
    assert len(configs)==72
    return configs

def emit(configs,name):
    folder=OUT/'configs';folder.mkdir(parents=True,exist_ok=True)
    for config in configs:
        if isinstance(config['lr'],(int,float)):
            path=folder/(config['id']+'.json');dump(path,config)
            config['config_path']=str(path)
    manifest=dict(status='REVIEW_PASS',configurations=configs,
        sources=sources(),test_read=False,seeds=list(SEEDS),pilot_seeds=[2030,2031])
    dump(OUT/name,manifest);return manifest

def prepare():return emit(base_configurations(),'base_manifest.json')

def freeze_winners():
    rows=[]
    for config in base_configurations()[:16]:
        path=Path(config['run_dir']);receipt=json.loads((path/'receipt.json').read_text())
        assert receipt['epochs']==100 and receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
        table=pd.read_csv(path/'trajectory.csv');assert list(table.epoch)==list(range(1,101))
        assert np.isfinite(table.selection_macro_auc).all()
        rows.append(dict(seed=config['seed'],arm=config['arm'],lr=config['lr'],
            selected_score=float(table.selection_macro_auc.max()),receipt_sha256=sha(path/'receipt.json')))
    frame=pd.DataFrame(rows);OUT.mkdir(parents=True,exist_ok=True);frame.to_csv(OUT/'pilot_response.csv',index=False)
    def winner(table):
        means=table.groupby('lr').selected_score.mean()
        return float(sorted(means.index,key=lambda lr:(-means[lr],lr))[0])
    common=winner(frame);own={arm:winner(frame[frame.arm==arm]) for arm in ('SN','SB')}
    freeze=dict(status='FROZEN',common_lr=common,own_lr=own,tie_rule='smaller LR',
        response_sha256=sha(OUT/'pilot_response.csv'),test_read=False,sources=sources())
    dump(OUT/'winners_frozen.json',freeze)
    extra=[];uses=[]
    for arm in ('SN','SB'):
        for lr in sorted({common,own[arm]}):
            for seed in SEEDS:
                config=configuration('formal',arm,lr,seed)
                uses.append(dict(arm=arm,lr=lr,seed=seed,common=lr==common,own=lr==own[arm],run_dir=config['run_dir']))
                if lr not in LRS[:2]:extra.append(config)
    quota=[configuration('quota',arm,common,seed,sampling='quota') for arm in ('SN','SB') for seed in SEEDS]
    emit(extra+quota,'post_pilot_manifest.json')
    dump(OUT/'winner_run_usage.json',dict(uses=uses,required_base_four_grid_retained=True,
        additional_formal_runs=len(extra),quota_runs=len(quota),winners_sha256=sha(OUT/'winners_frozen.json')))
    return freeze

def common_parent_configurations():
    frozen=json.loads((OUT/'winners_frozen.json').read_text());assert frozen['status']=='FROZEN'
    return [configuration('formal',arm,frozen['common_lr'],seed) for arm in ('SN','SB') for seed in SEEDS]

def plateau_gate():
    parents=common_parent_configurations();records=[]
    for config in parents:
        path=Path(config['run_dir']);receipt=json.loads((path/'receipt.json').read_text())
        assert receipt['epochs']==100 and receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
        table=pd.read_csv(path/'trajectory.csv');assert list(table.epoch)==list(range(1,101))
        j=table.train_empirical_objective.to_numpy();assert np.isfinite(j).all()
        first=float(j[80:90].mean());last=float(j[90:100].mean());rho=(first-last)/max(abs(first),1e-12)
        records.append(dict(arm=config['arm'],seed=config['seed'],rho=rho,
            mean_J81_90=first,mean_J91_100=last,receipt_sha256=sha(path/'receipt.json')))
    groups={arm:[row['rho'] for row in records if row['arm']==arm] for arm in ('SN','SB')}
    trigger=any(np.mean(vals)>.005 and sum(v>0 for v in vals)>=3 for vals in groups.values())
    decision=dict(status='FROZEN',trigger_H200=bool(trigger),records=records,test_read=False,
        rule='all 8 finite complete H100; either q mean rho>.005 and >=3/4 positive; extend both q all4',
        historical_rule='original 8 seeds and 6 positive; explicitly migrated to 4 and 3',sources=sources())
    dump(OUT/'plateau_frozen.json',decision)
    if trigger:
        configs=[]
        for original in parents:
            config=configuration('H200',original['arm'],original['lr'],original['seed'])
            config.update(epochs=200,resume_from=original['run_dir'],selection_budget='1..200; H100 source remains frozen')
            configs.append(config)
        emit(configs,'H200_manifest.json')
    return decision

def stage_continuation(config):
    if not config.get('resume_from'):raise ValueError('not a registered H200 continuation')
    gate=json.loads((OUT/'plateau_frozen.json').read_text());assert gate['trigger_H200']
    source=Path(config['resume_from']);destination=Path(config['run_dir'])
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,destination)
    dump(destination/'H100_parent_frozen.json',dict(path=str(source),
        selection_sha256=sha(source/'selection_frozen.json'),receipt_sha256=sha(source/'receipt.json'),
        continuation_config=config,gate_sha256=sha(OUT/'plateau_frozen.json')))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze-winners','plateau']);a=p.parse_args()
    {'prepare':prepare,'freeze-winners':freeze_winners,'plateau':plateau_gate}[a.action]()
if __name__=='__main__':main()
