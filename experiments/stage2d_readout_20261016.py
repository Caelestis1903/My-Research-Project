"""Descriptive Stage2D readouts; no scientific gate or training policy changes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research_plans/2026-10-16/stage2d_execution_20261016'
A=OUT/'analysis'
SEEDS=[2022,2023,2024,2025]
CONTRASTS=[('SB','SN'),('EB','EN')]


def runpath(seed,arm):
    return OUT/'runs'/f'seed{seed}'/arm/'attempt1'


def vectors(state,names,field):
    if field=='theta':return torch.cat([state['model'][n].double().flatten() for n in names])
    mapping={}
    for optkey,namekey in [('optimizer','optimizer_parameter_names'),('arch_optimizer','arch_optimizer_parameter_names')]:
        opt=state[optkey]
        ids=[i for group in opt['param_groups'] for i in group['params']]
        assert len(ids)==len(state[namekey])
        mapping.update({name:opt['state'][i] for name,i in zip(state[namekey],ids)})
    key={'m':'exp_avg','v':'exp_avg_sq'}[field]
    return torch.cat([mapping[n][key].double().flatten() for n in names])


def main():
    assert json.loads((A/'receipt.json').read_text())['training_runs_complete']==16
    torch.set_num_threads(2)
    a=pd.read_csv(A/'aggregate_metrics.csv');selected=pd.read_csv(A/'selected_aggregate_metrics.csv')
    timing=[];common=[];ranking=[]
    metrics=['auc_macro','auc_target_d1','auc_non_target','logloss_macro']
    for seed in SEEDS:
        for candidate,baseline in CONTRASTS:
            v=a[(a.seed==seed)&(a.split=='validation')].set_index(['arm','epoch'])
            s=selected[(selected.seed==seed)&(selected.split=='validation')].set_index('arm')
            ec=int(s.loc[candidate,'selected_epoch']);eb=int(s.loc[baseline,'selected_epoch'])
            for metric in metrics:
                delta=v.loc[(candidate,ec),metric]-v.loc[(baseline,eb),metric]
                same=v.loc[(candidate,eb),metric]-v.loc[(baseline,eb),metric]
                shift=v.loc[(candidate,ec),metric]-v.loc[(candidate,eb),metric]
                assert abs(delta-same-shift)<1e-12
                timing.append(dict(seed=seed,contrast=candidate+'-'+baseline,metric=metric,
                    candidate_epoch=ec,baseline_epoch=eb,selected_delta=delta,
                    same_epoch_at_baseline_selection=same,candidate_timing_contribution=shift))
            for split in ['validation','test']:
                for e in [9,19,29,39,49]:
                    x=a[(a.seed==seed)&(a.split==split)&(a.epoch==e)]
                    if split=='test':x=x[x.checkpoint==f'common{e:02d}']
                    x=x.set_index('arm')
                    common.append(dict(seed=seed,contrast=candidate+'-'+baseline,split=split,epoch=e,
                        **{metric:x.loc[candidate,metric]-x.loc[baseline,metric] for metric in a.columns
                           if metric.startswith(('auc_','logloss_'))}))
        for arm in ['SN','SB','EN','EB']:
            val=a[(a.seed==seed)&(a.arm==arm)&(a.split=='validation')].set_index('epoch')
            test=a[(a.seed==seed)&(a.arm==arm)&(a.split=='test')&a.checkpoint.str.startswith('common')].set_index('epoch')
            for metric in [c for c in a.columns if c.startswith('auc_')]:
                ranking.append(dict(seed=seed,arm=arm,metric=metric,epochs=5,
                    spearman=float(spearmanr(val.loc[test.index,metric],test[metric]).statistic)))
    pd.DataFrame(timing).to_csv(A/'validation_timing_decomposition.csv',index=False)
    pd.DataFrame(common).to_csv(A/'common_epoch_paired_differences.csv',index=False)
    pd.DataFrame(ranking).to_csv(A/'common_epoch_val_test_ranking.csv',index=False)

    pairs=pd.read_csv(A/'selected_paired_differences.csv')
    interactions=[]
    for (seed,split),v in selected.groupby(['seed','split']):
        v=v.set_index('arm')
        for metric in [c for c in selected.columns if c.startswith(('auc_','logloss_'))]:
            natural=v.loc['EN',metric]-v.loc['SN',metric]
            balanced=v.loc['EB',metric]-v.loc['SB',metric]
            interactions.append(dict(seed=seed,split=split,metric=metric,
                equal_minus_sample_natural=natural,equal_minus_sample_balanced=balanced,
                difference_of_differences=balanced-natural,
                interpretation='SECONDARY_OBJECTIVE_BY_SAMPLING_INTERACTION_NOT_SAMPLING_ONLY'))
    pd.DataFrame(interactions).to_csv(A/'objective_sampling_interaction.csv',index=False)
    gates=[]
    auc_cells=[f'auc_d{d}t{t}' for d in range(3) for t in range(2)]
    ll_cells=[f'logloss_d{d}t{t}' for d in range(3) for t in range(2)]
    for contrast,v in pairs[pairs.split=='validation'].groupby('contrast'):
        assert len(v)==4
        mean=float(v.auc_macro.mean());positive=int((v.auc_macro>0).sum())
        guard_auc=bool((v[auc_cells].mean()>=-.003).all())
        guard_seed=bool((v[auc_cells]>=-.005).all().all())
        guard_ll=bool((v[ll_cells].mean()<=.010).all())
        gates.append(dict(contrast=contrast,validation_selected_mean_auc_delta=mean,
            positive_seeds=positive,practical_positive_seeds=int((v.auc_macro>=.003).sum()),
            strong_positive_seeds=int((v.auc_macro>=.005).sum()),
            practical_and_direction=bool(mean>=.003 and positive>=3),
            strong_and_direction=bool(mean>=.005 and positive>=3),
            cell_mean_auc_guard=guard_auc,individual_cell_auc_guard=guard_seed,
            cell_mean_logloss_guard=guard_ll,all_nonharm_guards=guard_auc and guard_seed and guard_ll,
            minimum_cell_mean_auc=float(v[auc_cells].mean().min()),
            minimum_individual_cell_auc=float(v[auc_cells].min().min()),
            maximum_cell_mean_ll=float(v[ll_cells].mean().max()),
            scientific_decision='PENDING_PRIMARY_AGENT'))
    (A/'frozen_threshold_readout.json').write_text(json.dumps(gates,indent=2)+'\n')

    exposure=pd.read_csv(A/'exposure_and_loss.csv');expected=[]
    counts=np.array([172782,1743898,89452]);n=counts.sum()
    for r in exposure.itertuples():
        q=counts/n if r.arm.endswith('N') else np.ones(3)/3
        for d in range(3):
            expected.append(dict(seed=r.seed,arm=r.arm,epoch=r.epoch,domain=d,
                observed_unique=getattr(r,f'unique_cumulative_d{d}'),
                expected_unique=float(counts[d]*(-np.expm1(n*(r.epoch+1)*np.log1p(-q[d]/counts[d])))),
                expected_main_exposures=float(n*(r.epoch+1)*q[d]),train_unique_available=int(counts[d])))
    pd.DataFrame(expected).to_csv(A/'unique_exposure_comparison.csv',index=False)

    differences=[]
    for seed in SEEDS:
        for candidate,baseline in CONTRASTS:
            groups=json.loads((runpath(seed,baseline)/'partition.json').read_text())['groups']
            for epoch in [9,19,29,39,49]:
                c=torch.load(runpath(seed,candidate)/f'epoch{epoch:02d}.pt',map_location='cpu',weights_only=False)
                b=torch.load(runpath(seed,baseline)/f'epoch{epoch:02d}.pt',map_location='cpu',weights_only=False)
                for partition,names in groups.items():
                    for field in ['theta','m','v']:
                        x=vectors(c,names,field);y=vectors(b,names,field)
                        nx=float(torch.linalg.vector_norm(x));ny=float(torch.linalg.vector_norm(y))
                        diff=float(torch.linalg.vector_norm(x-y))
                        differences.append(dict(seed=seed,contrast=candidate+'-'+baseline,epoch=epoch,
                            partition=partition,field=field,candidate_norm=nx,baseline_norm=ny,
                            difference_norm=diff,relative_to_baseline=diff/max(ny,1e-30),
                            cosine=float(torch.dot(x,y))/max(nx*ny,1e-30),
                            interpretation='DESCRIPTIVE_DIFFERENT_THETA_NOT_CAUSAL'))
                del c,b
    pd.DataFrame(differences).to_csv(A/'common_state_vector_differences.csv',index=False)
    (A/'readout_receipt.json').write_text(json.dumps(dict(status='DESCRIPTIVE_TABLES_COMPLETE',
        timing_rows=len(timing),common_rows=len(common),ranking_rows=len(ranking),state_rows=len(differences),
        state_interpretation='different theta; no m/v component causal identification',
        scientific_decisions='PRIMARY_AGENT_ONLY'),indent=2)+'\n')
    print('Descriptive readout tables complete')


if __name__=='__main__':main()
