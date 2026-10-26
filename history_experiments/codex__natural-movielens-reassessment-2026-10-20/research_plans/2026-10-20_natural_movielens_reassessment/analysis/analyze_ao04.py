"""Primary AO04 full-test performance and separate local fusion sensitivity."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from analyze_core import interval

ART=Path('/home_new/xzf_natural_movielens_reassessment')
ROOT=ART/'AO04'
OUT=Path(__file__).resolve().parent/'ao04'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def analyze():
    freeze=json.loads((ROOT/'TEST_FREEZE.json').read_text())
    receipt=json.loads((ROOT/'test_receipt.json').read_text())
    assert freeze['global_test_unlock'] and receipt['status']=='COMPLETE_FROZEN_TEST' and receipt['optimization_steps']==0
    assert receipt['freeze_sha256']==sha(ROOT/'TEST_FREEZE.json')
    runs=json.loads((ROOT/'manifest.json').read_text())['configurations'];assert len(runs)==12
    with np.load(ART/'data/movielens_natural.npz') as z:
        ids=z['test_row_ids'];y=z['test_y'];d=z['test_x'][:,-1].astype(int)
    data=json.loads((ART/'data/movielens_natural.json').read_text())['data']
    OUT.mkdir(exist_ok=True);frames=[];trajectories=[];audit=[];sources=[];sensitivity=[];risks=[]
    for cfg in runs:
        run=Path(cfg['run_dir']);train=json.loads((run/'receipt.json').read_text())
        assert train['epochs']==50 and train['steps']==1000 and train['arch_steps']==50
        frozen=freeze['runs'][cfg['id']]
        assert sha(run/'config.json')==frozen['config_sha256']
        assert sha(run/'selection.json')==frozen['selection_sha256']
        assert all(sha(run/name)==value for name,value in frozen['checkpoints'].items())
        f=pd.read_csv(run/'metrics_test.csv').rename(columns={'test_auc':'auc','test_logloss':'logloss',
            'test_sample_count':'n','test_positive_count':'n_pos'})
        assert not f.duplicated(['checkpoint','domain','task']).any() and f.groupby('checkpoint').size().eq(6).all()
        with np.load(run/'predictions_test.npz') as z:
            assert np.array_equal(z['row_ids'],ids) and np.array_equal(z['labels'],y)
            for row in f.itertuples():
                pred=z['pred_'+row.checkpoint].astype(float);mask=d==row.domain
                yy=y[mask,row.task];pp=pred[mask,row.task]
                auc=roc_auc_score(yy,pp);pp=np.clip(pp,1e-12,1-1e-12)
                ll=-np.mean(yy*np.log(pp)+(1-yy)*np.log1p(-pp))
                assert abs(auc-row.auc)<1e-12 and abs(ll-row.logloss)<1e-12
                audit.append(dict(run_id=cfg['id'],checkpoint=row.checkpoint,domain=row.domain,task=row.task,
                    auc_residual=auc-row.auc,logloss_residual=ll-row.logloss))
        f['run_id']=cfg['id'];f['arm']=cfg['arm'];f['seed']=cfg['seed'];frames.append(f)
        v=pd.read_csv(run/'metrics_epoch.csv');v['run_id']=cfg['id'];trajectories.append(v)
        local=run/'local_fusion_diagnostics';rr=json.loads((local/'receipt.json').read_text())
        assert rr['status']=='COMPLETE_RAW_DIAGNOSTIC' and rr['training_updates']==0 and not rr['test_read']
        for checkpoint in rr['checkpoints']:
            assert sha(checkpoint['source'])==checkpoint['sha256']
            label=checkpoint['checkpoint']
            with np.load(local/(label+'_sensitivity.npz')) as z:
                domains=z['domains'];q=z['logit_jacobian_wrt_q'];rho=z['logit_jacobian_wrt_rho']
                diff=z['other_expert_differences']
                for domain in range(3):
                    mask=domains==domain;actualdiff=diff[mask,domain]
                    for task in range(2):
                        sensitivity.append(dict(run_id=cfg['id'],arm=cfg['arm'],seed=cfg['seed'],checkpoint=label,
                            domain=domain,task=task,n=int(mask.sum()),
                            other_expert_difference_RMS=float(np.sqrt(np.mean(np.square(actualdiff)))),
                            raw_logit_q_Jacobian_RMS=float(np.sqrt(np.mean(np.square(q[mask,task])))),
                            raw_logit_rho_Jacobian_RMS=float(np.sqrt(np.mean(np.square(rho[mask,task])))),
                            a_D_times_one_minus_b_D=checkpoint['a_D_times_one_minus_b_D'],
                            scope=checkpoint['B0_probe'] or 'trained arm, fixed theta, first8 validation rows/domain'))
        risk=pd.read_csv(local/'probe_cell_risks.csv');risk['run_id']=cfg['id'];risk['arm']=cfg['arm'];risk['seed']=cfg['seed']
        baseline=risk[risk.case=='baseline'].set_index(['checkpoint','domain','task'])
        risk['delta_logloss']=[row.logloss-baseline.loc[(row.checkpoint,row.domain,row.task),'logloss'] for row in risk.itertuples()]
        risk['delta_auc']=[row.auc-baseline.loc[(row.checkpoint,row.domain,row.task),'auc'] if row.auc_defined else np.nan for row in risk.itertuples()]
        risks.append(risk)
        sources.append(dict(run_id=cfg['id'],run_dir=str(run),config_sha256=sha(run/'config.json'),
            metrics_sha256=sha(run/'metrics_test.csv'),predictions_sha256=sha(run/'predictions_test.npz'),
            local_receipt_sha256=sha(local/'receipt.json'),paired_init_hash=train['paired_initial_state_sha256']))
    cell=pd.concat(frames,ignore_index=True);cell.to_csv(OUT/'seed_cell_metrics.csv',index=False)
    pd.concat(trajectories,ignore_index=True).to_csv(OUT/'validation_train_cell_trajectories.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'prediction_recalculation.csv',index=False)
    pd.DataFrame(sources).to_csv(OUT/'source_index.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'local_sensitivity.csv',index=False)
    pd.concat(risks,ignore_index=True).to_csv(OUT/'local_probe_perturbation_risks.csv',index=False)
    macro=[]
    for key,g in cell.groupby(['arm','seed','checkpoint']):
        p=g.domain.map({int(k):v for k,v in data['domain_shares'].items()})
        for task in (-1,0,1):
            gg=g if task==-1 else g[g.task==task];weights=p.loc[gg.index]
            macro.append(dict(zip(['arm','seed','checkpoint'],key),task=task,macro_auc=gg.auc.mean(),macro_logloss=gg.logloss.mean(),
                train_p_cell_auc=np.average(gg.auc,weights=weights),eval_p_cell_auc=np.average(gg.auc,weights=gg.n)))
    macro=pd.DataFrame(macro);macro.to_csv(OUT/'seed_macro_task_metrics.csv',index=False)
    paired=[];summary=[];paircells=[]
    for (checkpoint,task),g in macro.groupby(['checkpoint','task']):
        for ca,ref in [('B1','B0'),('B2','B0'),('B1','B2')]:
            a=g[g.arm==ca].set_index('seed');b=g[g.arm==ref].set_index('seed')
            assert set(a.index)==set(b.index)=={2022,2023,2024,2025}
            for metric in ('macro_auc','macro_logloss','train_p_cell_auc','eval_p_cell_auc'):
                values=a[metric]-b[metric]
                summary.append(dict(checkpoint=checkpoint,task=task,contrast=ca+'-'+ref,metric=metric,
                    reference_mean=b[metric].mean(),candidate_mean=a[metric].mean(),**interval(values)))
                for seed in values.index:paired.append(dict(checkpoint=checkpoint,task=task,contrast=ca+'-'+ref,
                    metric=metric,seed=seed,delta=float(values.loc[seed])))
    for ck,g in cell.groupby('checkpoint'):
        for ca,ref in [('B1','B0'),('B2','B0'),('B1','B2')]:
            a=g[g.arm==ca].set_index(['seed','domain','task']);b=g[g.arm==ref].set_index(['seed','domain','task'])
            for ix in a.index:
                paircells.append(dict(checkpoint=ck,contrast=ca+'-'+ref,seed=ix[0],domain=ix[1],task=ix[2],
                    auc=a.loc[ix,'auc']-b.loc[ix,'auc'],logloss=a.loc[ix,'logloss']-b.loc[ix,'logloss'],baseline_auc=b.loc[ix,'auc']))
    pd.DataFrame(paired).to_csv(OUT/'paired_seed_deltas.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'paired_summary.csv',index=False)
    pd.DataFrame(paircells).to_csv(OUT/'paired_cell_deltas.csv',index=False)
    (OUT/'analysis_status.json').write_text(json.dumps(dict(complete=True,runs=12,formal_seeds=4,
        local_probe_rows_per_domain=8,local_sensitivity_is_not_test_performance=True,
        full_prediction_cells_recomputed=len(audit)),indent=2)+'\n')
    print('AO04 complete; cell checks',len(audit))

if __name__=='__main__':analyze()
