"""Primary full-seed core analysis with prediction-level metric reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score

ART=Path('/home_new/xzf_natural_movielens_reassessment')
OUT=Path(__file__).resolve().parent/'core'
SEEDS={2022,2023,2024,2025}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def interval(values):
    values=np.asarray(values,dtype=float);n=len(values);mean=values.mean();sd=values.std(ddof=1)
    half=t.ppf(.975,n-1)*sd/np.sqrt(n)
    return dict(n=n,mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half,
        positive=int((values>0).sum()),negative=int((values<0).sum()),zero=int((values==0).sum()))

def analyze(stage):
    out=OUT/stage;out.mkdir(parents=True,exist_ok=True)
    planned=[r for r in json.loads((ART/'core_manifest.json').read_text())['runs'] if r['stage']==stage]
    complete=[];missing=[]
    for cfg in planned:
        run=Path(cfg['run_dir']);manifest=run/'run_manifest.json'
        if not manifest.exists() or json.loads(manifest.read_text()).get('status')!='COMPLETE':
            missing.append(cfg['run_id']);continue
        complete.append(cfg)
    # Scientific comparisons only use a whole preregistered four-seed cohort.
    expected=pd.DataFrame(planned).groupby(['population','budget']).size().to_dict()
    available=pd.DataFrame(complete).groupby(['population','budget']).size().to_dict() if complete else {}
    admitted={key for key,n in expected.items() if available.get(key)==n}
    complete=[c for c in complete if (c['population'],c['budget']) in admitted]
    (out/'analysis_status.json').write_text(json.dumps(dict(stage=stage,planned=len(planned),
        missing_run_ids=missing,admitted_complete_cohorts=[list(k) for k in sorted(admitted)],
        full_stage_complete=not missing,partial_seed_means_used=False),indent=2)+'\n')
    if not complete:print(stage,'no complete cohort yet');return
    reference=np.load(ART/'data/movielens_natural.npz')
    test_ids=reference['test_row_ids'];test_y=reference['test_y'];test_d=reference['test_x'][:,-1].astype(int)
    data={pop:json.loads((ART/'data'/(pop+'.json')).read_text())['data'] for pop in {c['population'] for c in complete}}
    frames=[];provenance=[];audits=[];trajectories=[]
    for cfg in complete:
        run=Path(cfg['run_dir']);config=json.loads((run/'config.json').read_text())
        assert config['seed']==cfg['seed'] and config['arm']==cfg['arm']
        f=pd.read_csv(run/'metrics_test.csv');assert not f.duplicated(['checkpoint','domain','task']).any()
        assert f.groupby('checkpoint').size().eq(6).all()
        for ck,g in f.groupby('checkpoint'):
            path=run/('test_'+ck+'.npz')
            with np.load(path) as z:
                assert np.array_equal(z['row_ids'],test_ids) and np.array_equal(z['labels'],test_y)
                assert np.array_equal(z['domains'],test_d)
                predictions=z['predictions'].astype(float)
            for row in g.itertuples():
                mask=test_d==row.domain;y=test_y[mask,row.task].astype(float);p=predictions[mask,row.task]
                clipped=np.clip(p,1e-12,1-1e-12)
                auc=roc_auc_score(y,p);loss=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped)))
                da=auc-row.auc;dl=loss-row.logloss
                assert abs(da)<1e-12 and abs(dl)<1e-12
                audits.append(dict(run_id=cfg['run_id'],checkpoint=ck,domain=row.domain,task=row.task,
                    auc=auc,logloss=loss,auc_residual=da,logloss_residual=dl,prediction_sha256=sha(path)))
        f['run_id']=cfg['run_id'];f['population']=cfg['population'];f['budget']=cfg['budget'];f['stage']=stage
        f['completed_epoch']=f.epoch+1
        counts=data[cfg['population']]['train_domain_counts'];total=sum(counts.values())
        f['train_p_weight']=f.domain.map({int(d):n/total/2 for d,n in counts.items()})
        constants=[]
        for row in f.itertuples():
            rate=data[cfg['population']]['cell_stats']['train'][f'd{row.domain}t{row.task}']['positive_rate']
            test_rate=row.n_pos/row.n if hasattr(row,'n_pos') else float(test_y[test_d==row.domain,row.task].mean())
            constants.append(-test_rate*np.log(rate)-(1-test_rate)*np.log1p(-rate))
        f['train_constant_test_logloss']=constants;f['logloss_skill_vs_train_constant']=1-f.logloss/f.train_constant_test_logloss
        frames.append(f)
        v=pd.read_csv(run/'metrics_epoch.csv');v['run_id']=cfg['run_id'];v['population']=cfg['population'];v['budget']=cfg['budget'];v['stage']=stage
        v['completed_epoch']=v.epoch+1;trajectories.append(v)
        provenance.append(dict(run_id=cfg['run_id'],config_sha256=sha(run/'config.json'),metrics_sha256=sha(run/'metrics_test.csv'),
            manifest_sha256=sha(run/'run_manifest.json'),path=str(run),init_sha256=config.get('init_sha',config.get('init_sha256'))))
    cell=pd.concat(frames,ignore_index=True);cell.to_csv(out/'seed_cell_metrics.csv',index=False)
    pd.DataFrame(provenance).to_csv(out/'source_index.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'prediction_metric_recalculation.csv',index=False)
    pd.concat(trajectories,ignore_index=True).to_csv(out/'validation_cell_trajectories.csv',index=False)
    keys=['stage','population','budget','arm','seed','checkpoint']
    rows=[]
    for key,g in cell.groupby(keys):
        assert len(g)==6
        rows.append(dict(zip(keys,key),completed_epoch=int(g.completed_epoch.iloc[0]),
            main_steps=int(g.step_main.iloc[0]),arch_steps=int(g.step_arch.iloc[0]),
            macro_auc=g.auc.mean(),macro_logloss=g.logloss.mean(),
            train_p_cell_auc=(g.auc*g.train_p_weight).sum(),train_p_cell_logloss=(g.logloss*g.train_p_weight).sum(),
            eval_p_cell_auc=np.average(g.auc,weights=g.n),eval_p_cell_logloss=np.average(g.logloss,weights=g.n),
            macro_logloss_skill=g.logloss_skill_vs_train_constant.mean()))
    macro=pd.DataFrame(rows);macro.to_csv(out/'seed_macro_metrics.csv',index=False)
    arm_contrasts={'AC-2D':[('SB','SN'),('EB','EN'),('EN','SN'),('EB','SB')],
        'AC-S5':[('T','F'),('HT','F'),('DH','F'),('DH','HT')], 'AC-S6':[('DH','F')],'AC-V9':[('SB','SN')]}[stage]
    paired=[];summary=[];paircells=[]
    metrics=['macro_auc','macro_logloss','train_p_cell_auc','train_p_cell_logloss','eval_p_cell_auc','eval_p_cell_logloss','macro_logloss_skill']
    for (pop,budget,ck),g in macro.groupby(['population','budget','checkpoint']):
        for candidate,reference_arm in arm_contrasts:
            a=g[g.arm==candidate].set_index('seed');b=g[g.arm==reference_arm].set_index('seed')
            if a.empty or b.empty:continue
            assert set(a.index)==set(b.index)==SEEDS
            contrast=candidate+'-'+reference_arm
            for seed in sorted(SEEDS):
                paired.append(dict(stage=stage,population=pop,budget=budget,checkpoint=ck,contrast=contrast,seed=seed,
                    **{metric:float(a.loc[seed,metric]-b.loc[seed,metric]) for metric in metrics}))
            for metric in metrics:
                summary.append(dict(stage=stage,population=pop,budget=budget,checkpoint=ck,contrast=contrast,metric=metric,
                    reference_mean=float(b[metric].mean()),candidate_mean=float(a[metric].mean()),**interval(a[metric]-b[metric])))
            cc=cell[(cell.population==pop)&(cell.budget==budget)&(cell.checkpoint==ck)]
            aa=cc[cc.arm==candidate].set_index(['seed','domain','task']);bb=cc[cc.arm==reference_arm].set_index(['seed','domain','task'])
            for ix in aa.index:
                paircells.append(dict(stage=stage,population=pop,budget=budget,checkpoint=ck,contrast=contrast,
                    seed=ix[0],domain=ix[1],task=ix[2],auc=aa.loc[ix,'auc']-bb.loc[ix,'auc'],
                    logloss=aa.loc[ix,'logloss']-bb.loc[ix,'logloss'],baseline_auc=bb.loc[ix,'auc']))
    pd.DataFrame(paired).to_csv(out/'paired_seed_deltas.csv',index=False)
    sums=pd.DataFrame(summary);sums.to_csv(out/'paired_summary.csv',index=False)
    pcs=pd.DataFrame(paircells);pcs.to_csv(out/'paired_cell_deltas.csv',index=False)
    for level in ('domain','task'):
        pcs.groupby(['stage','population','budget','checkpoint','contrast','seed',level])[['auc','logloss','baseline_auc']].mean().reset_index().to_csv(out/('paired_'+level+'_deltas.csv'),index=False)
    trade=[]
    for key,g in pcs.groupby(['stage','population','budget','checkpoint','contrast','seed']):
        domains=g.groupby('domain').auc.mean();counts=data[key[1]]['train_domain_counts'];ordered=sorted((int(d) for d in counts),key=lambda d:counts[str(d)])
        trade.append(dict(zip(['stage','population','budget','checkpoint','contrast','seed'],key),
            smallest_domain=ordered[0],largest_domain=ordered[-1],
            small_gain_large_decline=bool(domains[ordered[0]]>0 and domains[ordered[-1]]<0),
            both_smaller_gain_largest_decline=bool(all(domains[d]>0 for d in ordered[:2]) and domains[ordered[-1]]<0)))
    pd.DataFrame(trade).to_csv(out/'domain_size_tradeoffs.csv',index=False)
    population_deltas=[];population_cells=[]
    cohorts=[('natural_full','movielens_natural','full50'),('artificial_full','movielens_artificial_d2_fixed2022','full50'),
        ('natural_equal_N','movielens_natural_matched_n187500','full50'),('natural_matched_exposure','movielens_natural','matched250')]
    lookup={name:macro[(macro.population==pop)&(macro.budget==budget)] for name,pop,budget in cohorts}
    for candidate,baseline in [('natural_full','artificial_full'),('natural_equal_N','artificial_full'),
        ('natural_matched_exposure','artificial_full'),('natural_full','natural_matched_exposure'),('natural_equal_N','natural_matched_exposure')]:
        ca=next((pop,budget) for name,pop,budget in cohorts if name==candidate)
        cb=next((pop,budget) for name,pop,budget in cohorts if name==baseline)
        aa=cell[(cell.population==ca[0])&(cell.budget==ca[1])].set_index(['arm','seed','checkpoint','domain','task'])
        bb=cell[(cell.population==cb[0])&(cell.budget==cb[1])].set_index(['arm','seed','checkpoint','domain','task'])
        for ix in aa.index.intersection(bb.index):
            population_cells.append(dict(contrast=candidate+'-'+baseline,arm=ix[0],seed=ix[1],checkpoint=ix[2],domain=ix[3],task=ix[4],
                auc=aa.loc[ix,'auc']-bb.loc[ix,'auc'],logloss=aa.loc[ix,'logloss']-bb.loc[ix,'logloss'],baseline_auc=bb.loc[ix,'auc']))
        a=lookup[candidate].set_index(['arm','seed','checkpoint']);b=lookup[baseline].set_index(['arm','seed','checkpoint'])
        for ix in a.index.intersection(b.index):
            population_deltas.append(dict(contrast=candidate+'-'+baseline,arm=ix[0],seed=ix[1],checkpoint=ix[2],
                **{m:float(a.loc[ix,m]-b.loc[ix,m]) for m in metrics}))
    pd.DataFrame(population_deltas).to_csv(out/'population_budget_seed_deltas.csv',index=False)
    pd.DataFrame(population_cells).to_csv(out/'population_budget_cell_deltas.csv',index=False)
    print(stage,'admitted runs',len(complete),'of',len(planned),'; prediction cells',len(audits))
    print(sums[(sums.metric=='macro_auc')&sums.checkpoint.isin(['selected','final'])].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',default='AC-2D',choices=['AC-2D','AC-S5','AC-S6','AC-V9'])
    analyze(p.parse_args().stage)
