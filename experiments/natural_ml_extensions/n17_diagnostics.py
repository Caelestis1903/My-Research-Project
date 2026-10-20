"""Mechanical full-coordinate N2 and paired N3 diagnostics; no scientific verdicts."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from .n17_bootstrap import OUT,prepare,sources,sha,dump
from .n17_manifest import SEEDS,common_parent_configurations

def configurations():
    configs=[]
    for parent in common_parent_configurations():
        for epoch in (10,40):
            anchor=f'{parent["arm"]}_seed{parent["seed"]}_E{epoch:03d}'
            checkpoint=str(Path(parent['run_dir'])/f'epoch{epoch:03d}.pt')
            base=dict(anchor_arm=parent['arm'],seed=parent['seed'],lr=parent['lr'],epoch=epoch,checkpoint=checkpoint,
                      population='movielens_natural',optimizer='adam',batch_size=40960)
            configs.append(dict(base,id='N2_'+anchor,kind='n2',k=512,out=str(OUT/'N2'/anchor)))
            for bank in (0,1):
                configs.append(dict(base,id=f'N3_{anchor}_bank{bank}',kind='n3',draws=32,start_draw=bank*32,bank=bank,
                    independent_bank_definition='disjoint consecutive IID draws from the same paired stream',
                    out=str(OUT/'N3'/anchor/f'bank{bank}')))
    assert sum(c['kind']=='n2' for c in configs)==16
    assert sum(c['kind']=='n3' for c in configs)==32
    return configs

def manifest():
    configs=configurations()
    for config in configs:dump(OUT/'configs'/(config['id']+'.json'),config)
    dump(OUT/'diagnostic_manifest.json',dict(configurations=configs,sources=sources(),test_read=False,
        n2_batches_per_estimator=512,n3_draws_per_bank=32,banks=2,anchors=16,
        adaptive_precision_status='FROZEN_PRIMARY_AMENDMENT',
        optional_64_draws='MCSE > 0.000025 or undefined; two disjoint extra32 chunks; maximum64 per bank'))

def raw_local_contrasts(folder,config):
    table=pd.read_csv(folder/'macro_readouts.csv');baseline=table[table.case=='unupdated_anchor'].set_index('split')
    rows=[]
    for (draw,split),group in table[table.draw>=0].groupby(['draw','split']):
        sn=group[group.case=='native_SN'].iloc[0];sb=group[group.case=='native_SB'].iloc[0]
        for _,row in group.iterrows():
            item=dict(draw=int(draw),bank=config['bank'],split=split,case=row['case'],block=row['block'],
                seed=config['seed'],anchor_arm=config['anchor_arm'],epoch=config['epoch'],
                direction_undefined=bool(row.direction_undefined))
            for metric in ('auc','logloss','train_J'):
                item[metric+'_minus_SB']=float(row[metric]-sb[metric])
                item[metric+'_minus_SN']=float(row[metric]-sn[metric])
                item[metric+'_minus_anchor']=float(row[metric]-baseline.loc[split,metric])
            gap=float(sn.auc-sb.auc)
            item['native_SN_minus_SB_auc']=gap
            item['attenuation_auxiliary']=(float(row.auc-sb.auc)/gap if abs(gap)>=1e-4 else None)
            rows.append(item)
    pd.DataFrame(rows).to_csv(folder/'raw_contrasts.csv',index=False)

def precision_gate():
    configs=configurations();decisions=[];extra=[]
    for base in [c for c in configs if c['kind']=='n3' and c['bank']==0]:
        banks=[c for c in configs if c['kind']=='n3' and c['checkpoint']==base['checkpoint']]
        records=[]
        for config in banks:
            folder=Path(config['out'])
            assert json.loads((folder/'extension_receipt.json').read_text())['status']=='COMPLETE'
            table=pd.read_csv(folder/'raw_contrasts.csv')
            for case in ('C_full','C_norm','C_direction'):
                subset=table[(table.split=='diagnostic')&(table.case==case)]
                values=subset.loc[~subset.direction_undefined.astype(bool),'auc_minus_SB'].to_numpy()
                finite=values[np.isfinite(values)]
                mcse=float(np.std(finite,ddof=1)/np.sqrt(len(finite))) if len(finite)>1 else None
                records.append(dict(bank=config['bank'],case=case,n=len(finite),mcse=mcse,
                    trigger=bool(len(finite)!=32 or mcse is None or mcse>0.000025)))
        trigger=any(r['trigger'] for r in records)
        decisions.append(dict(checkpoint=base['checkpoint'],extend_to64_per_bank=trigger,records=records))
        if trigger:
            for config in banks:
                config=dict(config,id=config['id']+'_extra32',start_draw=64+32*config['bank'],
                    out=config['out']+'_extra32',draws=32,combine_only_with_bank=config['bank'])
                dump(OUT/'configs'/(config['id']+'.json'),config);extra.append(config)
    dump(OUT/'precision_decision.json',dict(decisions=decisions,threshold=0.000025,
        metric='paired development macro-AUC MCSE',sign_or_hypothesis_gate=False,test_read=False,sources=sources()))
    dump(OUT/'extra_precision_manifest.json',dict(configurations=extra,sources=sources(),test_read=False))

def execute(config,device='cuda:0'):
    gate=json.loads((OUT/'review_gate.json').read_text())
    assert gate['verdict']=='PASS' and gate['source_hashes']==sources()
    b,sp,c,root=prepare(config['population'])
    checkpoint=Path(config['checkpoint']);parent=json.loads((checkpoint.parent/'config.json').read_text())
    assert parent['arm']==config['anchor_arm'] and parent['seed']==config['seed'] and parent['lr']==config['lr']
    assert parent['population']=='movielens_natural'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);assert state['epoch']==config['epoch']
    # Both native optimizer moments and step come from this same anchor checkpoint.
    module=__import__('n17_geometry' if config['kind']=='n2' else 'n17_local')
    argv=[module.__file__,'--checkpoint',str(checkpoint),'--seed',str(config['seed']),
        '--optimizer','adam','--lr',str(config['lr']),'--anchor-arm',config['anchor_arm'],
        '--device',device,'--out',config['out'],'--batch-size','40960']
    if config['kind']=='n2':argv+=['--k','512','--adam-map']
    else:argv+=['--draws',str(config['draws']),'--start-draw',str(config['start_draw'])]
    old_argv=sys.argv
    try:sys.argv=argv;module.main()
    finally:sys.argv=old_argv
    folder=Path(config['out'])
    if config['kind']=='n2':
        # SGD is affine on the same full gradient draws, so these are exact raw
        # distribution transforms, without rerunning or changing the estimator.
        rows=[]
        model=c.base.fixed_model(b,config['seed']);model.load_state_dict(state['model'])
        from n17_geometry import indices
        groups,_=indices(model,torch.device('cpu'))
        theta=torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).double().numpy()
        sgd_means={};lag_variances={}
        optstate=state['optimizer']['state'];ids=state['optimizer']['param_groups'][0]['params']
        steps=[float(optstate[key]['step']) for key in ids];assert len(set(steps))==1
        native_v=torch.cat([optstate[key]['exp_avg_sq'].flatten() for key in ids]).double().numpy()
        denominator=np.sqrt(native_v/(1-.999**max(steps[0],1)))+1e-8
        lag_coefficient=-config['lr']*.1/(1-.9**(steps[0]+1))/denominator
        for estimator_arm in ('SN','SB'):
            for variant in ('IID','quota'):
                with np.load(folder/f'{estimator_arm}_{variant}_moments.npz') as moments:
                    sgd_means[(estimator_arm,variant)]=-config['lr']*(moments['mean']+1e-5*theta)
                    lag_variances[(estimator_arm,variant)]=lag_coefficient**2*moments['m2']/(config['k']-1)
                np.savez_compressed(folder/f'{estimator_arm}_{variant}_same_state_sgd.npz',mean_update=sgd_means[(estimator_arm,variant)])
        for row in json.loads((folder/'geometry.json').read_text()):
            rows.append(dict(seed=config['seed'],anchor_arm=config['anchor_arm'],epoch=config['epoch'],
                arm=row['arm'],variant=row['variant'],block=row['block'],
                lagged_common_denominator_update_covariance_trace=float(lag_variances[(row['arm'],row['variant'])][groups[row['block']].numpy()].sum()),
                sgd_mean_update_norm=float(np.linalg.norm(sgd_means[(row['arm'],row['variant'])][groups[row['block']].numpy()])),
                sgd_update_covariance_trace=config['lr']**2*row['empirical_covariance_trace'],
                sgd_update_mc_rms_se=config['lr']*row['mc_rms_se'],momentum=0,weight_decay=1e-5,
                mapping='delta=-lr*(g+WD*theta), same theta, exact affine covariance transform'))
        dump(folder/'same_state_sgd_mapping.json',rows)
        dump(folder/'finite_pool_definition.json',dict(population_size=len(b.train_x),
            domain_counts=b.train_counts,domain_sampling='IID uniform WR within fixed finite empirical pool',
            covariance_sigma='empirical finite-pool per-example training gradient covariance; estimated using independent domain batch means around exact finite-pool domain means',
            all_coordinates=True,coordinate_subsampling=False,
            iid_formula='(sum_d p_d^2/q_d Sigma_d + sum_d p_d^2/q_d mu_d mu_d^T - mu mu^T)/B',
            quota_formula='sum_d p_d^2 Sigma_d/n_d conditional on systematic quotas',
            mcse='per-block raw energy MCSE and per-domain variance-estimate MCSE in geometry.json',
            derivative_semantics='original training graph, routing input detached; not a functional Jacobian'))
    else:raw_local_contrasts(folder,config)
    dump(folder/'extension_receipt.json',dict(status='COMPLETE',config=config,sources=sources(),
        native_receipt_sha256=sha(folder/'receipt.json'),test_read=False,scientific_analysis=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path);p.add_argument('--manifest',action='store_true');p.add_argument('--precision-gate',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.manifest:manifest()
    elif a.precision_gate:precision_gate()
    else:execute(json.loads(a.config.read_text()),a.device)
if __name__=='__main__':main()
