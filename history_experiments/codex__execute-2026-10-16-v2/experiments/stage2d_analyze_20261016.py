"""Recompute Stage2D metrics and paired tables. Scientific decisions belong to main agent."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research_plans/2026-10-16/stage2d_execution_20261016'
ARMS=['SN','SB','EN','EB'];SEEDS=[2022,2023,2024,2025]
CONTRASTS=[('SB','SN'),('EB','EN')]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def aggregates(cells,p):
    out={}
    for metric in ['auc','logloss']:
        values={(int(r.domain),int(r.task)):float(getattr(r,metric)) for r in cells.itertuples()}
        out[metric+'_macro']=np.mean(list(values.values()))
        for d in range(3):out[f'{metric}_d{d}']=np.mean([values[d,t] for t in range(2)])
        for t in range(2):out[f'{metric}_t{t}']=np.mean([values[d,t] for d in range(3)])
        out[metric+'_target_d1']=out[metric+'_d1']
        out[metric+'_non_target']=np.mean([values[d,t] for d in [0,2] for t in range(2)])
        out[metric+'_W_train']=sum(p[d]*values[d,t] for d in range(3) for t in range(2))/2
        counts=cells.groupby('domain').n.first();shares=counts/counts.sum()
        out[metric+'_W_split']=sum(shares[d]*values[d,t] for d in range(3) for t in range(2))/2
        if metric=='auc':
            out['auc_worst_cell']=min(values.values());out['auc_worst_domain']=min(out[f'auc_d{d}'] for d in range(3))
        for (d,t),value in values.items():out[f'{metric}_d{d}t{t}']=value
    return out


def raw_metrics(path,raw,split):
    with np.load(path,allow_pickle=False) as z:
        ids=z['row_ids'];domains=z['domains'];labels=z['labels'];pred=z['predictions'].astype(float)
    assert len(np.unique(ids))==len(ids)
    assert np.array_equal(raw.iloc[ids][['is_click','long_view']].to_numpy(),labels)
    assert np.array_equal(raw.iloc[ids].tab.map({0:0,1:1,4:2}).to_numpy(),domains)
    lo=int(.8*len(raw)) if split=='validation' else int(.9*len(raw))
    hi=int(.9*len(raw)) if split=='validation' else len(raw)
    assert ((ids>=lo)&(ids<hi)).all()
    rows=[]
    for d in range(3):
        mask=domains==d
        for t in range(2):
            y=labels[mask,t].astype(float);pr=pred[mask,t];pp=np.clip(pr,1e-12,1-1e-12)
            rows.append(dict(domain=d,task=t,n=len(y),n_pos=int(y.sum()),auc=float(roc_auc_score(y,pr)),
                logloss=float(-(y*np.log(pp)+(1-y)*np.log1p(-pp)).mean())))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    target=OUT/('live_audit' if args.preflight else 'analysis');target.mkdir(exist_ok=True)
    records=[];configs={};traces={};validation=[];test=[];selected=[];hash_checks=[];source_sets=[]
    raw=None if args.preflight else pd.read_csv('/home/xzf/data/kuairand/processed/kuai_rand.csv',usecols=['tab','is_click','long_view'])
    if not args.preflight:assert sha('/home/xzf/data/kuairand/processed/kuai_rand.csv')=='a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f'
    for seed in SEEDS:
        for arm in ARMS:
            run=OUT/'runs'/f'seed{seed}'/arm/'attempt1'
            configpath=run/'config.json'
            if not configpath.exists():
                assert args.preflight,(seed,arm,'not launched');continue
            config=json.loads(configpath.read_text());configs[seed,arm]=config
            source_sets.append(config['source_hashes'])
            for source,expected in config['source_hashes'].items():assert sha(ROOT/source)==expected
            path=run/'batch_trace.csv'
            if path.exists():traces[seed,arm]=pd.read_csv(path)
            metrics=run/'metrics_epoch.csv'
            if metrics.exists():
                v=pd.read_csv(metrics);assert not v.duplicated(['epoch','domain','task']).any()
                validation.append(v)
                for epoch,c in v.groupby('epoch'):
                    assert len(c)==6 and c.step_main.nunique()==1 and c.step_arch.nunique()==1
                    records.append(dict(seed=seed,arm=arm,split='validation',epoch=epoch,checkpoint='trajectory',
                        **aggregates(c,np.array([172782,1743898,89452])/2006132)))
            if args.preflight:continue
            manifest=json.loads((run/'run_manifest.json').read_text());assert manifest['status']=='COMPLETE'
            assert manifest['step_main']==2450 and manifest['step_arch']==50 and manifest['epochs_completed']==50
            assert len(v)==300 and set(v.epoch)==set(range(50))
            for name,h in manifest['artifact_hashes'].items():
                actual=sha(run/name);hash_checks.append(dict(seed=seed,arm=arm,path=name,match=actual==h));assert actual==h,(run,name)
            scores=v.groupby('epoch').auc.mean();epoch=int(scores.idxmax());assert epoch==manifest['selected_epoch']
            selection=json.loads((run/'selection_frozen.json').read_text());assert selection['selected_epoch']==epoch
            assert selection['test_access_started'] is False
            p=np.array([172782,1743898,89452],dtype=float)/2006132
            for split,predfile,ref in [('validation','validation_selected.npz',v[v.epoch==epoch]),
                ('test','test_selected.npz',pd.read_csv(run/'metrics_test.csv').query("checkpoint == 'selected'"))]:
                computed=raw_metrics(run/predfile,raw,split).sort_values(['domain','task'])
                reference=ref.sort_values(['domain','task'])
                for field in ['auc','logloss']:assert np.max(np.abs(computed[field].to_numpy()-reference[field].to_numpy()))<1e-10
                for row in computed.to_dict('records'):selected.append(dict(seed=seed,arm=arm,split=split,epoch=epoch,**row))
            t=pd.read_csv(run/'metrics_test.csv');test.append(t)
            for (checkpoint,ep),c in t.groupby(['checkpoint','epoch']):
                assert len(c)==6
                records.append(dict(seed=seed,arm=arm,split='test',epoch=ep,checkpoint=checkpoint,**aggregates(c,p)))
    for seed in SEEDS:
        conf=[configs[seed,a] for a in ARMS if (seed,a) in configs]
        assert len({c['init_sha'] for c in conf})<=1
        for left,right in [('SN','EN'),('SB','EB')]:
            if (seed,left) not in traces or (seed,right) not in traces:continue
            x=traces[seed,left];y=traces[seed,right]
            joined=x.merge(y,on=['epoch','batch','kind'],suffixes=('_a','_b'),validate='one_to_one')
            assert np.array_equal(joined.row_ids_sha256_a,joined.row_ids_sha256_b)
            if not args.preflight:assert len(joined)==2500
    pd.DataFrame(records).to_csv(target/'aggregate_metrics.csv',index=False)
    receipt=dict(status='LIVE_PAIRING_PASS' if args.preflight else 'PASS_NUMERICAL_AUDIT_ONLY',
        run_configs_checked=len(configs),training_runs_complete=0 if args.preflight else 16,
        same_q_pairing=True,init_pairing=True,source_sets_consistent=all(x==source_sets[0] for x in source_sets),
        source_sha256=sha(__file__),independent_review=False,scientific_analysis='PENDING_MAIN_AGENT')
    if args.preflight:
        (target/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    assert all(x==source_sets[0] for x in source_sets)
    a=pd.DataFrame(records);sel=pd.DataFrame(selected);sel.to_csv(target/'selected_cell_metrics.csv',index=False)
    pd.concat(validation).to_csv(target/'validation_cell_metrics.csv',index=False);pd.concat(test).to_csv(target/'test_cell_metrics.csv',index=False)
    pd.DataFrame(hash_checks).to_csv(target/'artifact_hash_checks.csv',index=False)
    own=[]
    for (seed,arm,split),v in sel.groupby(['seed','arm','split']):
        own.append(dict(seed=seed,arm=arm,split=split,selected_epoch=int(v.epoch.iloc[0]),**aggregates(v,np.array([172782,1743898,89452])/2006132)))
    own=pd.DataFrame(own);own.to_csv(target/'selected_aggregate_metrics.csv',index=False)
    pairs=[];same=[]
    for candidate,baseline in CONTRASTS:
        for (seed,split),v in own.groupby(['seed','split']):
            c=v[v.arm==candidate].iloc[0];b=v[v.arm==baseline].iloc[0]
            row=dict(seed=seed,split=split,contrast=candidate+'-'+baseline,candidate_epoch=int(c.selected_epoch),baseline_epoch=int(b.selected_epoch))
            for col in own.columns:
                if col.startswith(('auc_','logloss_')):row[col]=float(c[col]-b[col])
            pairs.append(row)
        v=a[(a.split=='validation')&(a.arm.isin([candidate,baseline]))]
        for (seed,epoch),g in v.groupby(['seed','epoch']):
            c=g[g.arm==candidate].iloc[0];b=g[g.arm==baseline].iloc[0]
            same.append(dict(seed=seed,epoch=epoch,contrast=candidate+'-'+baseline,
                **{col:float(c[col]-b[col]) for col in a.columns if col.startswith(('auc_','logloss_'))}))
    pairs=pd.DataFrame(pairs);pairs.to_csv(target/'selected_paired_differences.csv',index=False)
    pd.DataFrame(same).to_csv(target/'same_epoch_validation_differences.csv',index=False)
    long=pairs.melt(id_vars=['seed','split','contrast','candidate_epoch','baseline_epoch'],var_name='metric',value_name='delta')
    summary=long.groupby(['split','contrast','metric']).agg(n=('seed','size'),mean=('delta','mean'),sd=('delta','std'),
        minimum=('delta','min'),maximum=('delta','max'),positive_seeds=('delta',lambda x:int((x>0).sum())),
        practical_positive_seeds=('delta',lambda x:int((x>=.003).sum())))
    summary.to_csv(target/'seed_summary.csv')
    fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    for i,seed in enumerate(SEEDS):
        for arm in ARMS:
            v=a[(a.seed==seed)&(a.arm==arm)&(a.split=='validation')].sort_values('epoch')
            axes[0,i].plot(v.epoch,v.auc_macro,label=arm);axes[1,i].plot(v.epoch,v.logloss_macro,label=arm)
        axes[0,i].set_title(f'seed {seed}');axes[0,i].set_ylabel('validation macro AUC')
        axes[1,i].set_ylabel('validation macro LogLoss');axes[1,i].set_xlabel('epoch (0 based)')
    axes[0,0].legend();fig.savefig(target/'validation_trajectories.png',dpi=150);fig.savefig(target/'validation_trajectories.pdf');plt.close(fig)
    observations=[];exposure=[]
    for seed,arm in configs:
        run=OUT/'runs'/f'seed{seed}'/arm/'attempt1'
        o=pd.read_csv(run/'optimizer_observations.csv');o['seed']=seed;o['arm']=arm;observations.append(o)
        e=pd.read_csv(run/'epoch_observations.csv');e['seed']=seed;e['arm']=arm;exposure.append(e)
    pd.concat(observations).to_csv(target/'optimizer_observations.csv',index=False)
    pd.concat(exposure).to_csv(target/'exposure_and_loss.csv',index=False)
    receipt.update(artifact_hash_checks=len(hash_checks),selected_prediction_files_recomputed=32,
        output_hashes={p.name:sha(p) for p in target.iterdir() if p.is_file() and p.name!='receipt.json'})
    (target/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
