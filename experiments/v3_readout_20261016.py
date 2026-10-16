"""Numerical v3 readouts. Scientific decisions remain with the orchestrator."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import v3_audit_20261016 as audit
import v3_train_20261016 as train
base=audit.base;OUT=audit.OUT

def auc_order(labels,scores):
    order=np.argsort(scores,kind='stable');s=scores[order]
    return order,labels[order],np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]

def ordered_auc(prepared,weights):
    order,y,starts=prepared;w=weights[:,order].astype(np.float64)
    pos=np.add.reduceat(w*y[None,:],starts,axis=1)
    neg=np.add.reduceat(w*(1-y)[None,:],starts,axis=1)
    before=np.cumsum(neg,axis=1)-neg
    return np.sum(pos*(before+.5*neg),axis=1)/(pos.sum(1)*neg.sum(1))

def weighted_auc(labels,scores,weights):
    return ordered_auc(auc_order(labels,scores),weights)

def bootstrap_curves(labels,domains,predictions,subset,k,seed,progress=None):
    # Domain-stratified ordinary row bootstrap, same row multiplicities across
    # arms/epochs/tasks. Cached ranks change runtime only, not resampling law.
    arms,epochs,_,_=predictions.shape
    result=np.zeros((k,arms,epochs,2),dtype=float);rng=np.random.default_rng(seed)
    for d in range(3):
        rows=subset[domains[subset]==d];n=len(rows);y=labels[rows]
        prepared={};losses={}
        for arm in range(arms):
            for epoch in range(epochs):
                for task in range(2):
                    key=(arm,epoch,task);p=predictions[arm,epoch,rows,task];yy=y[:,task]
                    prepared[key]=auc_order(yy,p)
                    clipped=np.clip(p.astype(float),1e-12,1-1e-12)
                    losses[key]=-(yy*np.log(clipped)+(1-yy)*np.log(1-clipped))
        for offset in range(0,k,16):
            m=min(16,k-offset);weights=rng.multinomial(n,np.full(n,1/n),size=m)
            for (arm,epoch,task),ranks in prepared.items():
                result[offset:offset+m,arm,epoch,0]+=ordered_auc(ranks,weights)/6
                result[offset:offset+m,arm,epoch,1]+=(weights@losses[(arm,epoch,task)])/n/6
            if progress:base.dump(progress,dict(status='BOOTSTRAP',domain=d,replicates_completed=offset+m,k=k,time=time.time()))
    assert np.isfinite(result).all()
    return result

def selection_indices(curves):
    # curves [..., epoch, metric], AUC max and loss min, np.argmax earliest ties.
    shape=curves.shape[:-2]
    return dict(macro_auc=np.argmax(curves[...,0],axis=-1),macro_logloss=np.argmin(curves[...,1],axis=-1),
        early_auc_1_10=np.argmax(curves[...,:10,0],axis=-1),fixed_budget=np.full(shape,curves.shape[-2]-1,dtype=int))

def onset(point,boot,train_objective):
    e=len(point)
    for start in range(1,e-2):
        peak=int(np.argmax(point[:start,0]));window=np.arange(start,start+3)
        uncertainty=np.quantile(boot[:,window,0]-boot[:,peak,None,0],.975,axis=0)
        if np.all(point[window,0]<point[peak,0]) and np.all(uncertainty<0):
            ll_worse=bool(np.all(point[window,1]>point[peak,1]))
            loss_down=bool(train_objective[start+2]<train_objective[peak])
            return dict(auc_degradation_onset=start+1,peak_epoch=peak+1,confirmed_overfitting=ll_worse and loss_down,
                logloss_worse=ll_worse,train_objective_decreased=loss_down)
    return dict(auc_degradation_onset=None,peak_epoch=None,confirmed_overfitting=False,logloss_worse=False,train_objective_decreased=False)

def paired_summary(values,seed=20261016):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(seed);n=len(values)
    boot=np.mean(values[rng.integers(0,n,size=(20000,n))],axis=1)
    perms=np.array([np.mean(values*np.array(sign)) for sign in itertools.product([-1,1],repeat=n)])
    return dict(n=n,mean=float(values.mean()),sd=float(values.std(ddof=1)) if n>1 else None,median=float(np.median(values)),
        positive=int((values>0).sum()),negative=int((values<0).sum()),seed_values=values.tolist(),ci_low=float(np.quantile(boot,.025)),
        ci_high=float(np.quantile(boot,.975)),paired_sign_randomization_p=float(np.mean(np.abs(perms)>=abs(values.mean())-1e-15)))

def pair(args):
    freeze=json.loads((OUT/'analysis_rules_frozen.json').read_text());assert freeze['status']=='FROZEN' and freeze['test_used_to_choose_conditions'] is False
    out=args.out;out.mkdir(parents=True,exist_ok=False);configs=[json.loads((p/'config.json').read_text()) for p in (args.natural,args.balanced)]
    assert configs[0]['arm']=='SN' and configs[1]['arm']=='SB'
    for key in ('dataset','seed','optimizer','lr','batch_size','replacement','order','duplication','corruption','init_sha256','split_sha256','data'):
        assert configs[0][key]==configs[1][key],f'Pair mismatch {key}'
    dataset=configs[0]['dataset'];seed=configs[0]['seed'];b=audit.bundle(dataset,seed)
    frozen=np.load(OUT/'splits'/f'{dataset}_seed{seed}.npz');domains=b.val_x[:,-1].astype(int)
    directories=(args.natural,args.balanced);files=[sorted(p.glob('validation_epoch*.npz')) for p in directories]
    epochs=[int(p.stem.split('epoch')[1]) for p in files[0]]
    assert epochs==[int(p.stem.split('epoch')[1]) for p in files[1]]
    assert epochs==list(range(1,len(epochs)+1)) and len(epochs)>=50
    predictions=np.array([[np.load(p)['predictions'] for p in paths] for paths in files])
    rows=[];point=np.empty((2,2,len(epochs),2))
    for split_index,split in enumerate(('selection','diagnostic')):
        for arm,p in enumerate(directories):
            for i,epoch in enumerate(epochs):
                cells=train.metrics(b,predictions[arm,i],'val',frozen[split])
                point[split_index,arm,i]=[np.mean([c[m] for c in cells]) for m in ('auc','logloss')]
                for row in cells:row.update(epoch=epoch,split=split,arm=('SN','SB')[arm],seed=seed,run=str(p));rows.append(row)
    pd.DataFrame(rows).to_csv(out/'cell_trajectory.csv',index=False)
    # Diagnostic labels/predictions only opened after explicit LR/rule freeze receipt.
    selection=bootstrap_curves(b.val_y,domains,predictions,frozen['selection'],args.bootstraps,seed+20261016,out/'progress.json')
    diagnostic=bootstrap_curves(b.val_y,domains,predictions,frozen['diagnostic'],200,seed+20261017,out/'progress.json')
    np.savez_compressed(out/'bootstrap_curves.npz',selection=selection,diagnostic=diagnostic,point=point,epochs=epochs)
    indices=selection_indices(selection);chosen=[]
    for rule,ix in indices.items():
        for k in range(args.bootstraps):
            n,bidx=int(ix[k,0]),int(ix[k,1])
            chosen.append(dict(replicate=k,rule=rule,SN_epoch=epochs[n],SB_epoch=epochs[bidx],
                selection_auc_difference=float(selection[k,1,bidx,0]-selection[k,0,n,0]),
                selection_logloss_difference=float(selection[k,1,bidx,1]-selection[k,0,n,1]),
                diagnostic_auc_at_reselected_epochs=float(point[1,1,bidx,0]-point[1,0,n,0])))
    pd.DataFrame(chosen).to_csv(out/'selection_bootstrap.csv',index=False)
    training=[pd.read_csv(p/'trajectory.csv') for p in directories]
    onsets=[onset(point[1,i],diagnostic[:,i],training[i]['train_empirical_objective'].to_numpy()) for i in range(2)]
    selection_onsets=[onset(point[0,i],selection[:,i],training[i]['train_empirical_objective'].to_numpy()) for i in range(2)]
    budget=[]
    for i,arm in enumerate(('SN','SB')):
        losses=training[i]['train_empirical_objective'].to_numpy();prior=float(np.mean(losses[-10:-5]));recent=float(np.mean(losses[-5:]));gain=(prior-recent)/max(abs(prior),1e-12)
        budget.append(dict(arm=arm,prior5_objective=prior,last5_objective=recent,relative_improvement=gain,practical_plateau=bool(gain<=.005),selection_confirmed_overfitting=selection_onsets[i]['confirmed_overfitting'],extension_required=bool(gain>.005 and not selection_onsets[i]['confirmed_overfitting'])))
    stop=min([o['auc_degradation_onset']-1 for o in onsets if o['auc_degradation_onset'] is not None]+[len(epochs)])
    summaries=[]
    for split_index,split in enumerate(('selection','diagnostic')):
        for epoch in (5,10,15,20,30,40,50,100):
            if epoch in epochs:
                i=epochs.index(epoch)
                for j,metric in enumerate(('auc','logloss')):summaries.append(dict(split=split,readout=f'common_epoch{epoch}',metric=metric,SN=float(point[split_index,0,i,j]),SB=float(point[split_index,1,i,j]),difference=float(point[split_index,1,i,j]-point[split_index,0,i,j])))
        for horizon in (10,20,stop):
            if horizon<=0:continue
            h=min(horizon,len(epochs))
            for j,metric in enumerate(('auc','logloss')):
                # Mean over completed-epoch values: discretized AULC divided by epoch budget.
                summaries.append(dict(split=split,readout=f'AULC_1_{h}',metric=metric,SN=float(np.mean(point[split_index,0,:h,j])),SB=float(np.mean(point[split_index,1,:h,j])),difference=float(np.mean(point[split_index,1,:h,j]-point[split_index,0,:h,j]))))
        for rule,ix in selection_indices(point[0]).items():
            for j,metric in enumerate(('auc','logloss')):summaries.append(dict(split=split,readout=f'selected_{rule}',metric=metric,SN=float(point[split_index,0,ix[0],j]),SB=float(point[split_index,1,ix[1],j]),difference=float(point[split_index,1,ix[1],j]-point[split_index,0,ix[0],j])))
    pd.DataFrame(summaries).to_csv(out/'paired_readouts.csv',index=False)
    base.dump(out/'receipt.json',dict(status='COMPLETE',seed=seed,dataset=dataset,optimizer=configs[0]['optimizer'],
        source_runs=list(map(str,directories)),onsets=onsets,selection_onsets=selection_onsets,optimization_budget=budget,common_pre_degradation_epochs=stop,bootstrap_count=args.bootstraps,
        bootstrap_unit='domain-stratified validation rows, paired across arms epochs tasks',seed_uncertainty_separate=True,
        hashes={p.name:base.sha(p) for p in out.iterdir() if p.is_file()}))

def test(args):
    frozen=json.loads((OUT/'analysis_rules_frozen.json').read_text());assert frozen['status']=='FROZEN'
    directory=args.natural;config=json.loads((directory/'config.json').read_text());assert frozen.get('all_condition_definitions_frozen') is True;assert (directory/'receipt.json').exists()
    device=torch.device(args.device);torch.set_num_threads(2)
    if device.type=='cuda':torch.cuda.set_device(device)
    b=audit.bundle(config['dataset'],config['seed']);train.corruption(b,config['corruption'],int(np.argmin(config['p'])))
    model=base.fixed_model(b,config['seed']).to(device)
    rows=[];epochs=[]
    for path in sorted(directory.glob('epoch*_model.pt')):
        state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
        predictions=train.predict(model,b,'test',device);epoch=state['epoch'];epochs.append(epoch)
        np.savez_compressed(directory/f'test_epoch{epoch:03d}.npz',predictions=predictions)
        cells=train.metrics(b,predictions,'test')
        for row in cells:row.update(epoch=epoch,split='test',arm=config['arm'],seed=config['seed']);rows.append(row)
    pd.DataFrame(rows).to_csv(directory/'metrics_test.csv',index=False)
    base.dump(directory/'test_readout_receipt.json',dict(status='COMPLETE',epochs=epochs,freeze_sha=base.sha(OUT/'analysis_rules_frozen.json'),
        test_previously_exposed=True,hashes={p.name:base.sha(p) for p in directory.glob('test_epoch*.npz')}))

def check(args):
    from stage_S1_metrics import safe_auc
    y=np.array([0,1,0,1,1]);p=np.array([.1,.4,.4,.8,.8]);counts=np.array([[1,1,1,1,1],[2,1,2,3,1]])
    actual=weighted_auc(y,p,counts);expected=[safe_auc(np.repeat(y,c),np.repeat(p,c)) for c in counts]
    assert np.allclose(actual,expected,atol=1e-14)
    assert paired_summary([1,1,1,1])['paired_sign_randomization_p']==.125
    base.dump(args.out,dict(status='PASS',weighted_tie_auc_max_error=float(np.max(abs(actual-expected))),
        exact_sign_permutation_n4_min_two_sided_p=.125,source_hash=base.sha(__file__)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pair','test','check'],required=True)
    p.add_argument('--natural',type=Path);p.add_argument('--balanced',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--bootstraps',type=int,default=1000);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.mode=='pair':pair(args)
    elif args.mode=='test':test(args)
    else:check(args)

if __name__=='__main__':main()
