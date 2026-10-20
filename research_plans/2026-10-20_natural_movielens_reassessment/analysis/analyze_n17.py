"""Primary N17 phase analysis, only after the global training/selection test freeze."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score

ART=Path('/home_new/xzf_natural_movielens_reassessment')
ROOT=ART/'N17'
OUT=Path(__file__).resolve().parent/'n17'
SEEDS={2022,2023,2024,2025}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stats(values):
    v=np.asarray(values);n=len(v);mean=v.mean();sd=v.std(ddof=1);half=t.ppf(.975,n-1)*sd/np.sqrt(n)
    return dict(n=n,mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half,
        positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()))

def analyze():
    OUT.mkdir(exist_ok=True)
    freeze=json.loads((ROOT/'test_frozen.json').read_text());assert freeze['status']=='FROZEN'
    winners=json.loads((ROOT/'winners_frozen.json').read_text())
    records=[];source=[];audit=[];readouts=[];allconfigs={}
    for path,entry in freeze['runs'].items():
        config=entry['config'];run=Path(path);assert config['group']!='pilot'
        receipt=json.loads((run/'test_receipt.json').read_text());assert receipt['status']=='COMPLETE' and receipt['zero_updates']
        assert receipt['freeze_sha256']==sha(ROOT/'test_frozen.json')
        assert receipt['metrics_sha256']==sha(run/'metrics_test.csv')
        assert sha(run/'config.json')==entry['config_sha256']
        f=pd.read_csv(run/'metrics_test.csv');assert len(f)==6*config['epochs'] and set(f.epoch)==set(range(1,config['epochs']+1))
        assert not f.duplicated(['epoch','domain','task']).any()
        data=json.loads((ART/'data'/(config['population']+'.json')).read_text())['data']
        counts=data['train_domain_counts'];total=sum(counts.values())
        f['population']=config['population'];f['group']=config['group'];f['run_dir']=path
        f['lr']=config['lr'];f['arm']=config['arm'];f['seed']=config['seed']
        f['train_p_weight']=f.domain.map({int(d):n/total/2 for d,n in counts.items()})
        f['selected']=f.epoch==entry['selected_epoch'];f['final']=f.epoch==config['epochs']
        with np.load(ART/'data'/(config['population']+'.npz')) as z:
            y=z['test_y'];d=z['test_x'][:,-1].astype(int);test_ids=z['test_row_ids']
        prediction_index=json.loads((run/'test_prediction_index.json').read_text())
        assert prediction_index['row_ids_sha256']==hashlib.sha256(test_ids.astype(np.int64).tobytes()).hexdigest()
        saved_prediction_hashes={v['epoch']:v['sha256'] for v in prediction_index['predictions']}
        for epoch in sorted({entry['selected_epoch'],config['epochs']}):
            prediction=run/f'test_epoch{epoch:03d}.npz'
            assert sha(prediction)==saved_prediction_hashes[epoch]
            with np.load(prediction) as z:p=z['predictions'].astype(float)
            for row in f[f.epoch==epoch].itertuples():
                ix=d==row.domain;yy=y[ix,row.task].astype(float);pp=np.clip(p[ix,row.task],1e-12,1-1e-12)
                auc=roc_auc_score(yy,p[ix,row.task]);loss=np.mean(-yy*np.log(pp)-(1-yy)*np.log1p(-pp))
                assert abs(auc-row.auc)<1e-12 and abs(loss-row.logloss)<1e-12
                audit.append(dict(run_dir=path,epoch=epoch,domain=row.domain,task=row.task,
                    auc_residual=auc-row.auc,logloss_residual=loss-row.logloss,prediction_sha256=sha(prediction)))
        records.append(f);allconfigs[path]=config
        source.append(dict(run_dir=path,config_sha256=sha(run/'config.json'),metrics_sha256=sha(run/'metrics_test.csv'),
            test_receipt_sha256=sha(run/'test_receipt.json'),freeze_sha256=sha(ROOT/'test_frozen.json')))
        for label,subset in [('selected',f[f.selected]),('final',f[f.final]),
                *[(f'E{e}',f[f.epoch==e]) for e in (10,20,50,100,200) if e<=config['epochs']],
                *[(f'AULC1_{e}',f[f.epoch<=e]) for e in sorted({20,100,config['epochs']})]]:
            for task in (-1,0,1):
                g=subset if task==-1 else subset[subset.task==task]
                readouts.append(dict(run_dir=path,group=config['group'],population=config['population'],arm=config['arm'],
                    seed=config['seed'],lr=config['lr'],readout=label,task=task,
                    auc=g.auc.mean(),logloss=g.logloss.mean(),
                    train_p_cell_auc=np.average(g.auc,weights=g.train_p_weight),
                    eval_p_cell_auc=np.average(g.auc,weights=g.n),
                    train_p_cell_logloss=np.average(g.logloss,weights=g.train_p_weight),
                    eval_p_cell_logloss=np.average(g.logloss,weights=g.n)))
    pd.concat(records).to_csv(OUT/'all_test_cell_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'selected_final_prediction_recalculation.csv',index=False)
    pd.DataFrame(source).to_csv(OUT/'source_index.csv',index=False)
    read=pd.DataFrame(readouts);read.to_csv(OUT/'seed_readouts.csv',index=False)
    rows=[];summary=[]
    def pair(name,a,b,estimand):
        aa=a.set_index(['seed','readout','task']);bb=b.set_index(['seed','readout','task'])
        assert set(aa.index)==set(bb.index) and set(aa.index.get_level_values('seed'))==SEEDS
        for ix in aa.index:
            item=dict(comparison=name,estimand=estimand,seed=ix[0],readout=ix[1],task=ix[2],
                candidate_run_dir=aa.loc[ix,'run_dir'],reference_run_dir=bb.loc[ix,'run_dir'],
                candidate_lr=float(aa.loc[ix,'lr']),reference_lr=float(bb.loc[ix,'lr']),
                candidate_arm=aa.loc[ix,'arm'],reference_arm=bb.loc[ix,'arm'],
                **{m:float(aa.loc[ix,m]-bb.loc[ix,m]) for m in ('auc','logloss','train_p_cell_auc','eval_p_cell_auc','train_p_cell_logloss','eval_p_cell_logloss')})
            rows.append(item)
        for (label,task),gg in pd.DataFrame(rows)[lambda x:x.comparison==name].groupby(['readout','task']):
            for metric in ('auc','logloss','train_p_cell_auc','eval_p_cell_auc','train_p_cell_logloss','eval_p_cell_logloss'):
                summary.append(dict(comparison=name,estimand=estimand,readout=label,task=task,metric=metric,
                    reference_mean=float(b[(b.readout==label)&(b.task==task)][metric].mean()),
                    candidate_mean=float(a[(a.readout==label)&(a.task==task)][metric].mean()),**stats(gg[metric])))
    for group in ('formal','matched_budget','matched_n','quota','H200'):
        g=read[read.group==group]
        for lr in sorted(g.lr.unique()):
            a=g[(g.lr==lr)&(g.arm=='SB')];b=g[(g.lr==lr)&(g.arm=='SN')]
            if not a.empty and not b.empty:pair(f'{group}_sameLR{lr:g}_SB-SN',a,b,'same population, objective and LR; different estimator')
    g=read[read.group=='formal']
    pair('formal_commonLR_SB-SN',g[(g.arm=='SB')&(g.lr==winners['common_lr'])],
        g[(g.arm=='SN')&(g.lr==winners['common_lr'])],'symmetric pilot common-grid winner; fixed LR for both q')
    pair('formal_ownLR_SB-SN',g[(g.arm=='SB')&(g.lr==winners['own_lr']['SB'])],
        g[(g.arm=='SN')&(g.lr==winners['own_lr']['SN'])],'each q own pilot winner; includes LR differences when winners differ')
    for arm in ('SN','SB'):
        lr=winners['common_lr']
        pair('quota-minus-IID_'+arm,read[(read.group=='quota')&(read.arm==arm)&(read.lr==lr)],
            read[(read.group=='formal')&(read.arm==arm)&(read.lr==lr)],'systematic quota versus IID; same common LR and q')
        for lr in (.0003,.001):
            for candidate,baseline in [('matched_n','matched_budget'),('formal','matched_budget')]:
                pair(f'{candidate}-minus-{baseline}_{arm}_lr{lr:g}',
                    read[(read.group==candidate)&(read.arm==arm)&(read.lr==lr)],
                    read[(read.group==baseline)&(read.arm==arm)&(read.lr==lr)],
                    'support/exposure control; population and optimization effects stated separately')
        h200=read[(read.group=='H200')&(read.arm==arm)&(read.lr==winners['common_lr'])]
        if not h200.empty:
            h100=read[(read.group=='formal')&(read.arm==arm)&(read.lr==winners['common_lr'])]
            common=set(h200.readout)&set(h100.readout)
            pair('H200-minus-H100_'+arm,h200[h200.readout.isin(common)],h100[h100.readout.isin(common)],
                'same saved H100 prefix continued to H200; selected changes search horizon, final changes epoch; shared-prefix readouts are duplicate evidence')
    pairs=pd.DataFrame(rows);pairs.to_csv(OUT/'paired_seed_deltas.csv',index=False)
    byrun={str(f.run_dir.iloc[0]):f for f in records};cellpairs=[]
    def subset(f,label):
        if label=='selected':return f[f.selected]
        if label=='final':return f[f.final]
        if label.startswith('E'):return f[f.epoch==int(label[1:])]
        return f[f.epoch<=int(label.split('_')[-1])]
    for row in pairs[pairs.task==-1].itertuples():
        a=subset(byrun[row.candidate_run_dir],row.readout).groupby(['domain','task'])[['auc','logloss']].mean()
        b=subset(byrun[row.reference_run_dir],row.readout).groupby(['domain','task'])[['auc','logloss']].mean()
        for d,task in a.index:
            cellpairs.append(dict(comparison=row.comparison,seed=row.seed,readout=row.readout,domain=d,task=task,
                auc=a.loc[(d,task),'auc']-b.loc[(d,task),'auc'],
                logloss=a.loc[(d,task),'logloss']-b.loc[(d,task),'logloss'],baseline_auc=b.loc[(d,task),'auc']))
    pcs=pd.DataFrame(cellpairs);pcs.to_csv(OUT/'paired_cell_deltas.csv',index=False)
    pcs.groupby(['comparison','seed','readout','domain'])[['auc','logloss','baseline_auc']].mean().reset_index().to_csv(OUT/'paired_domain_deltas.csv',index=False)
    trades=[]
    for key,g in pcs.groupby(['comparison','seed','readout']):
        means=g.groupby('domain').auc.mean()
        trades.append(dict(comparison=key[0],seed=key[1],readout=key[2],
            domain_definition='natural smallest d0, largest d2; physical IDs preserved',
            small_gain_large_decline=bool(means[0]>0 and means[2]<0),
            both_smaller_gain_largest_decline=bool(means[0]>0 and means[1]>0 and means[2]<0)))
    pd.DataFrame(trades).to_csv(OUT/'domain_size_tradeoffs.csv',index=False)
    table=pd.DataFrame(summary);table.to_csv(OUT/'paired_summary.csv',index=False)
    interaction=[]
    for arm in ('SN','SB'):
        z=pairs[pairs.comparison=='quota-minus-IID_'+arm].set_index(['seed','readout','task'])
        if arm=='SN':sn=z
        else:
            for ix in z.index:
                interaction.append(dict(seed=ix[0],readout=ix[1],task=ix[2],
                    **{m:float(z.loc[ix,m]-sn.loc[ix,m]) for m in ('auc','logloss','train_p_cell_auc','eval_p_cell_auc','train_p_cell_logloss','eval_p_cell_logloss')}))
    pd.DataFrame(interaction).to_csv(OUT/'quota_by_q_interaction_seed.csv',index=False)
    old=Path('/home/xzf/Code/phase0-3-20261017-git/research_plans/2026-10-17/execution_20261017')
    historical_sources=[]
    for name in ('FINAL_STAGE_MACRO_AUC.csv','FINAL_SELECTED_CELL_METRICS.csv','N5_winners_frozen.json'):
        target=OUT/('historical_'+name);target.write_bytes((old/name).read_bytes())
        historical_sources.append(dict(path=str(old/name),sha256=sha(old/name),copy=str(target),
            evidence='historical recorded table; old large checkpoints not claimed available'))
    (OUT/'historical_source_index.json').write_text(json.dumps(historical_sources,indent=2)+'\n')
    (OUT/'analysis_receipt.json').write_text(json.dumps(dict(status='COMPLETE',formal_runs=len(allconfigs),
        test_freeze_sha256=sha(ROOT/'test_frozen.json'),new_formal_seeds=sorted(SEEDS),
        historical_N5_seeds=list(range(2032,2040)),historical_and_new_population_deltas_not_seed_paired=True,
        intervals='pointwise descriptive t95 over four optimization seeds; not independent population confirmation'),indent=2)+'\n')
    print(table[(table.readout.isin(['selected','final']))&(table.task==-1)&(table.metric=='auc')].to_string(index=False))

if __name__=='__main__':analyze()
