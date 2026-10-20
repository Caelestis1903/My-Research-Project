"""History/current controls: actual formulas, batch identity and state telemetry."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ART=Path('/home_new/xzf_natural_movielens_reassessment')
OUT=Path(__file__).resolve().parent/'controlled_updates'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def analyze():
    planned=[r for r in json.loads((ART/'core_manifest.json').read_text())['runs'] if r['stage'] in ('AC-S5','AC-S6')]
    configs=pd.DataFrame(planned);admitted=[];missing=[]
    for key,g in configs.groupby(['stage','population','budget']):
        complete=all((Path(r.run_dir)/'run_manifest.json').exists() and
            json.loads((Path(r.run_dir)/'run_manifest.json').read_text()).get('status')=='COMPLETE' for r in g.itertuples())
        if complete:admitted+=g.to_dict('records')
        else:missing.append(list(key))
    OUT.mkdir(exist_ok=True)
    (OUT/'analysis_status.json').write_text(json.dumps(dict(planned_runs=48,admitted_runs=len(admitted),
        pending_complete_four_seed_cohorts=missing,partial_seed_means_used=False),indent=2)+'\n')
    if not admitted:print('No whole controlled-update cohort ready');return
    windows=[];residuals=[];probes=[];indices=[];batchpairs=[];traces={};budgets=[];exposures=[]
    for cfg in admitted:
        run=Path(cfg['run_dir']);native=json.loads((run/'config.json').read_text())
        base={key:cfg[key] for key in ('run_id','stage','population','budget','seed','arm')}
        g=pd.read_csv(run/'gradient_step.csv');g['completed_epoch']=g.epoch+1
        g['window']=pd.cut(g.completed_epoch,bins=[0,10,40,50],labels=['E1-10','E11-40','E41-50'])
        for (kind,window),gg in g.groupby(['kind','window'],observed=True):
            fields=['g_full_norm','g_target_norm','g_non_norm','gradient_cosine','m_norm_persistent','v_norm_persistent',
                'delta_full_norm','delta_target_norm','delta_history_norm','delta_applied_norm','private_delta_norm',
                'applied_cosine_full','applied_cosine_target','installed_cosine_history']
            for metric in fields:
                windows.append(dict(base,record='original_control_gradient',kind=kind,partition='original shared control block',
                    window=str(window),metric=metric,steps=len(gg),mean=float(gg[metric].mean()),
                    minimum=float(gg[metric].min()),maximum=float(gg[metric].max())))
        for kind,gg in g.groupby('kind'):
            residuals.append(dict(base,kind=kind,history_source=native['history_source'],current_source=native['current_source'],
                observed_steps=len(gg),decomposition_maxabs=float(gg.decomposition_residual.abs().max()),
                history_formula_maxabs=float(gg.history_formula_residual.abs().max()),
                applied_formula_maxabs=float(gg.applied_formula_residual.abs().max()),
                private_full_maxabs=float(gg.private_full_residual.abs().max()),
                zero_target_gradient_steps=int(gg.zero_target_gradient_step.astype(bool).sum()),
                persistent_m_and_v='both true native historical states; this intervention does not isolate m necessity'))
        o=pd.read_csv(run/'optimizer_full_observations.csv');o['completed_epoch']=o.epoch+1
        o['window']=pd.cut(o.completed_epoch,bins=[0,10,40,50],labels=['E1-10','E11-40','E41-50'])
        o['decay_over_data_gradient_norm']=np.divide(o.weight_decay_gradient_norm,o.raw_gradient_norm,
            out=np.full(len(o),np.nan),where=o.raw_gradient_norm.to_numpy()!=0)
        for (kind,partition,window),gg in o.groupby(['kind','partition','window'],observed=True):
            for metric in ['raw_gradient_norm','weight_decay_gradient_norm','effective_gradient_norm','persistent_m_norm',
                    'persistent_v_norm','applied_delta_norm','decay_over_data_gradient_norm']:
                windows.append(dict(base,record='new_full_coordinate_optimizer_observation',kind=kind,partition=partition,
                    window=str(window),metric=metric,steps=len(gg),mean=float(gg[metric].mean()),
                    minimum=float(gg[metric].min()),maximum=float(gg[metric].max())))
        p=pd.read_csv(run/'probe.csv')
        for key,value in base.items():p[key]=value
        probes.append(p)
        traces[cfg['run_id']]=pd.read_csv(run/'batch_trace.csv')
        epochs=pd.read_csv(run/'epoch_observations.csv');assert len(epochs)==50 and set(epochs.epoch)==set(range(50))
        for kind,gg in traces[cfg['run_id']].groupby('kind'):
            budgets.append(dict(base,kind=kind,steps=len(gg),actual_examples=int(gg.n.sum()),
                **{'domain'+str(d)+'_examples':int(gg['d'+str(d)].sum()) for d in range(3)}))
        for d in range(3):
            last=epochs.iloc[-1];available=native['data']['train_domain_counts'][str(d)]
            exposures.append(dict(base,domain=d,available_train_rows=available,
                main_examples=int(epochs['exposure_d'+str(d)].sum()),
                final_cumulative_unique=int(last['unique_cumulative_d'+str(d)]),
                rows_never_seen_in_main=available-int(last['unique_cumulative_d'+str(d)]),
                within_epoch_repeats=int(epochs['repeat_epoch_d'+str(d)].sum()),
                meaning='unique support and exposure of main shuffle only; architecture exposures counted separately'))

        for filename in ['config.json','gradient_step.csv','probe.csv','optimizer_full_observations.csv','batch_trace.csv','epoch_observations.csv']:
            indices.append(dict(run_id=cfg['run_id'],path=str(run/filename),sha256=sha(run/filename)))
    # Controls must receive the same actual batches within each seed/budget.
    frame=pd.DataFrame(admitted)
    for key,g in frame.groupby(['stage','population','budget','seed']):
        baseline=g[g.arm=='F'].iloc[0];b=traces[baseline.run_id]
        columns=['epoch','batch','kind','n','d0','d1','d2','row_ids_sha256']
        for row in g.itertuples():
            a=traces[row.run_id]
            equal=a[columns].reset_index(drop=True).equals(b[columns].reset_index(drop=True))
            assert equal,'different training batches in paired control '+row.run_id
            batchpairs.append(dict(stage=key[0],population=key[1],budget=key[2],seed=key[3],arm=row.arm,
                reference_arm='F',all_main_and_architecture_batches_identical=equal,observed_batches=len(a)))
    pd.DataFrame(windows).to_csv(OUT/'seed_window_gradient_state_summary.csv',index=False)
    window_table=pd.DataFrame(windows)
    positive=window_table[window_table.record=='new_full_coordinate_optimizer_observation']
    contrasts=[]
    for key,g in positive.groupby(['stage','population','budget','seed','kind','partition','window','metric']):
        g=g.set_index('arm');reference=g.loc['F','mean']
        for arm in sorted(set(g.index)-{'F'}):
            candidate=g.loc[arm,'mean']
            contrasts.append(dict(zip(['stage','population','budget','seed','kind','partition','window','metric'],key),
                contrast=arm+'-F',reference_mean=reference,candidate_mean=candidate,
                difference=candidate-reference,ratio=candidate/reference if reference else np.nan,
                interpretation='same-seed trajectory comparison; actual states differ; norm ratios do not identify a unique mediator'))
    pd.DataFrame(contrasts).to_csv(OUT/'paired_seed_window_state_norm_context.csv',index=False)

    pd.DataFrame(residuals).to_csv(OUT/'control_formula_and_none_context.csv',index=False)
    pd.concat(probes,ignore_index=True).to_csv(OUT/'original_sparse_probes.csv',index=False)
    pd.DataFrame(batchpairs).to_csv(OUT/'paired_batch_identity.csv',index=False)
    pd.DataFrame(budgets).to_csv(OUT/'realized_main_arch_budgets.csv',index=False)
    pd.DataFrame(exposures).to_csv(OUT/'realized_main_support_by_domain.csv',index=False)
    pd.DataFrame(indices).to_csv(OUT/'source_index.csv',index=False)
    print('Controlled-update analysis admitted',len(admitted),'complete-cohort runs')

if __name__=='__main__':analyze()
