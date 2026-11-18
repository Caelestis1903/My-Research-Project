"""Recompute per-cell and paired seed tables after a complete stage."""
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from cm19_train import OUT,dump

FIELDS=['stage','dataset','model','variant','optimizer','decay_mode','fraction','schedule','max_steps']

def conditions(c):return {k:c.get(k,1. if k=='fraction' else 'normal_epochs' if k=='schedule' else 0 if k=='max_steps' else '') for k in FIELDS}

def group_id(c):return '|'.join(str(conditions(c)[k]) for k in FIELDS)

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','selection','test'],required=True);p.add_argument('--stage',choices=['G0'],help='Read a fully completed G0 development stage while later preregistered stages continue.');a=p.parse_args()
    if a.stage and a.split=='test':p.error('Stage-only readout is restricted to validation; test requires the complete frozen experiment.')
    manifest=json.loads((OUT/'run_manifest.json').read_text());runs=[c for c in manifest['runs'] if not a.stage or c['stage']==a.stage]
    out=OUT/('test_readout' if a.split=='test' else 'development_g0_readout' if a.stage else 'development_readout');out.mkdir(exist_ok=True)
    missing=[c['run_id'] for c in runs if not (OUT/'runs'/c['run_id']/'receipt.json').exists()]
    if missing:raise RuntimeError(f'Readout requires the whole requested scope: {len(missing)} receipts missing.')
    cells_all=[];trajectories=[];online_metrics=[];costs=[];paths=[];strengths=[];gradients=[];counterfactuals=[];configs={}
    for c in runs:
        rr=OUT/'runs'/c['run_id'];receipt=json.loads((rr/'receipt.json').read_text());cfg=json.loads((rr/'config.json').read_text());configs[c['run_id']]=cfg
        assert receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
        traj=pd.read_csv(rr/'trajectory.csv');traj['run_id']=c['run_id'];trajectories.append(traj.assign(**conditions(c),seed=c['seed']))
        vv=pd.read_csv(rr/'metrics_validation.csv')
        online_metrics.append(vv[vv.split=='training_online'].assign(**conditions(c),seed=c['seed'],run_id=c['run_id']))
        if a.split=='test':
            assert (rr/'test_receipt.json').exists();df=pd.read_csv(rr/'metrics_test.csv')
        else:
            vv=vv[vv.split==a.split];chunks=[]
            for label,epoch in [('selected',receipt['selected_epoch']),('final',receipt['epochs'])]:chunks.append(vv[vv.epoch==epoch].assign(checkpoint=label,intervention='none'))
            df=pd.concat(chunks)
        df=df.assign(**conditions(c),seed=c['seed'],run_id=c['run_id'],group_id=group_id(c));cells_all.append(df)
        costs.append(dict(run_id=c['run_id'],seed=c['seed'],**conditions(c),**{k:receipt.get(k) for k in ['wall_seconds','peak_reserved_mib','epochs','steps','examples','selected_epoch']},parameters=cfg['parameters'],trainable_parameters=cfg['trainable_parameters']))
        for file,frames in [('path_diagnostics.csv',paths),('path_strength.csv',strengths),('gradient_updates.csv',gradients),('local_counterfactual.csv',counterfactuals)]:
            f=rr/file
            if f.exists():frames.append(pd.read_csv(f).assign(**conditions(c),seed=c['seed'],run_id=c['run_id'],group_id=group_id(c)))
    cc=pd.concat(cells_all,ignore_index=True);cc.to_csv(out/'seed_cell_metrics.csv',index=False)
    pd.concat(trajectories,ignore_index=True).to_csv(out/'training_trajectories.csv',index=False);pd.DataFrame(costs).to_csv(out/'run_costs.csv',index=False)
    pd.concat(online_metrics,ignore_index=True).to_csv(out/'training_online_metrics.csv.gz',index=False,compression='gzip')
    for file,frames in [('path_diagnostics.csv',paths),('path_strength.csv',strengths),('gradient_updates.csv',gradients),('local_counterfactual.csv',counterfactuals)]:
        if frames:
            target=out/(file+'.gz' if file=='path_diagnostics.csv' else file)
            pd.concat(frames,ignore_index=True).to_csv(target,index=False,compression='gzip' if file=='path_diagnostics.csv' else None)
    macro=[];taskrows=[];domainrows=[]
    original_counts={}
    for c in runs:
        if c['stage']=='G0' and c['model']=='fixed':original_counts[c['dataset']]=configs[c['run_id']]['main_domain_counts']
    for (run,label),df in cc[cc.intervention=='none'].groupby(['run_id','checkpoint']):
        cfg=configs[run];weights=np.asarray(original_counts[cfg['dataset']],float);weights/=weights.sum();w=np.array([weights[int(d)]/len(cfg['task_ids']) for d in df.domain])
        macro.append(dict(run_id=run,checkpoint=label,seed=cfg['seed'],**conditions(cfg),group_id=group_id(cfg),macro_auc=df.auc.mean(),macro_logloss=df.logloss.mean(),weighted_cell_auc=float((df.auc*w).sum()),weighted_cell_logloss=float((df.logloss*w).sum()),worst_cell_auc=df.auc.min(),macro_skill_logloss=df.skill_logloss.mean()))
        for task,dd in df.groupby('task'):taskrows.append(dict(run_id=run,checkpoint=label,task=task,auc=dd.auc.mean(),logloss=dd.logloss.mean(),**conditions(cfg),seed=cfg['seed']))
        for domain,dd in df.groupby('domain'):domainrows.append(dict(run_id=run,checkpoint=label,domain=domain,auc=dd.auc.mean(),logloss=dd.logloss.mean(),**conditions(cfg),seed=cfg['seed']))
    mm=pd.DataFrame(macro);mm.to_csv(out/'seed_macro_metrics.csv',index=False);pd.DataFrame(taskrows).to_csv(out/'seed_task_metrics.csv',index=False);pd.DataFrame(domainrows).to_csv(out/'seed_domain_metrics.csv',index=False)
    metrics=['macro_auc','macro_logloss','weighted_cell_auc','weighted_cell_logloss','worst_cell_auc','macro_skill_logloss']
    summary=mm.groupby(FIELDS+['checkpoint'],dropna=False)[metrics].agg(['mean','std','count']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'condition_summary.csv',index=False)
    # Preregistered pairs: cross-optimizer G0, each intervention/native and normalized joint/private.
    pairs=[];seed_pairs=[];cell_pairs=[]
    def compare(cand,base,label):
        for checkpoint in ['selected','final']:
            aa=mm[(mm.group_id==cand)&(mm.checkpoint==checkpoint)];bb=mm[(mm.group_id==base)&(mm.checkpoint==checkpoint)]
            if not len(aa) or not len(bb):return
            jj=aa.merge(bb,on='seed',suffixes=('_candidate','_baseline'));assert len(jj)==4,(label,len(jj))
            for metric in metrics:
                delta=jj[metric+'_candidate']-jj[metric+'_baseline'];mu=delta.mean();sd=delta.std(ddof=1);half=student_t.ppf(.975,3)*sd/2
                pairs.append(dict(comparison=label,candidate=cand,baseline=base,checkpoint=checkpoint,metric=metric,mean_delta=mu,sd_delta=sd,ci95_low=mu-half,ci95_high=mu+half,positive_seeds=int((delta>0).sum()),negative_seeds=int((delta<0).sum()),zero_seeds=int((delta==0).sum()),n=4,interval_scope='optimization seeds on one fixed population; descriptive; no multiplicity correction'))
                for seed,value in zip(jj.seed,delta):seed_pairs.append(dict(comparison=label,checkpoint=checkpoint,metric=metric,seed=int(seed),delta=float(value)))
            ca=cc[(cc.group_id==cand)&(cc.checkpoint==checkpoint)&(cc.intervention=='none')];cb=cc[(cc.group_id==base)&(cc.checkpoint==checkpoint)&(cc.intervention=='none')]
            cj=ca.merge(cb,on=['seed','domain','task'],suffixes=('_candidate','_baseline'))
            for row in cj.itertuples():cell_pairs.append(dict(comparison=label,checkpoint=checkpoint,seed=row.seed,domain=row.domain,task=row.task,delta_auc=row.auc_candidate-row.auc_baseline,delta_logloss=row.logloss_candidate-row.logloss_baseline,n=row.n_candidate))
    unique={group_id(c):c for c in runs}
    for gid,c in unique.items():
        base=dict(c)
        if c['stage']=='G0' and c['optimizer']=='adamw':base['optimizer']='adam';label=f"G0_optimizer/{c['dataset']}/{c['model']}"
        elif c['stage'] in ['G1','G2','G3','G4','G4LONG']:
            base.update(stage='G0',variant='native',decay_mode='standard');base.pop('fraction',None);base.pop('schedule',None);base.pop('max_steps',None)
            label=f"{c['stage']}/{c['dataset']}/{c['model']}/{c['variant']}/{c['optimizer']}/{c['decay_mode']}/fraction{c.get('fraction',1)}/{c.get('schedule','normal_epochs')}"
        elif c['stage']=='G0M':
            base.update(stage='G0MC',model=c['model'].replace('_joint','_private_joint'));label=f"task_sharing/{c['dataset']}/{c['model']}/{c['optimizer']}"
        else:continue
        compare(gid,group_id(base),label)
    for gid,c in unique.items():
        if c['stage']=='G0' and c['model'] in ['original','original_sample','pepnet','mmoe','ple']:
            base={**c,'model':'fixed'};compare(gid,group_id(base),f"baseline_reference/{c['dataset']}/{c['model']}_vs_fixed/{c['optimizer']}")
            if c['model']=='original':compare(gid,group_id({**c,'model':'original_sample'}),f"protocol_bridge/{c['dataset']}/original_vs_original_sample/{c['optimizer']}")
        if c['stage']=='G0M':compare(gid,group_id({**c,'stage':'G0','model':'fixed'}),f"joint_baseline_reference/{c['dataset']}/{c['model']}_vs_fixed/{c['optimizer']}")
        if c['stage']=='G1' and c['decay_mode']!='wd0':
            compare(gid,group_id({**c,'decay_mode':'wd0'}),f"strong_WD0_reference/{c['dataset']}/{c['model']}/{c['decay_mode']}")
            compare(gid,group_id({**c,'stage':'G0','decay_mode':'standard','optimizer':'adamw'}),f"strong_AdamW_reference/{c['dataset']}/{c['model']}/{c['decay_mode']}")
        pd.DataFrame(pairs).to_csv(out/'paired_comparisons.csv',index=False);pd.DataFrame(seed_pairs).to_csv(out/'paired_seed_deltas.csv',index=False);pd.DataFrame(cell_pairs).to_csv(out/'paired_cell_deltas.csv',index=False)
    pd.DataFrame(manifest['aliases']).to_csv(out/'equivalent_configuration_aliases.csv',index=False)
    dump(out/'collection_receipt.json',dict(split=a.split,stage=a.stage or 'all',independent_runs=len(runs),cell_rows=len(cc),macro_rows=len(mm),paired_metric_rows=len(pairs),test_metrics_used=a.split=='test',all_seeds_present=True))
    print(json.dumps(dict(split=a.split,runs=len(runs),cell_rows=len(cc),comparisons=len(pairs))))
if __name__=='__main__':main()
