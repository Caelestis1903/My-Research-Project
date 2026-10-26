"""Fixed-probe path activity, actual clocks and nominal float32 decay scope."""
from pathlib import Path
import hashlib
import gzip
import json
import numpy as np
import pandas as pd

ART=Path('/home_new/xzf_natural_movielens_reassessment')
OLD=Path('/home/xzf/Code/phase0-3-20261019-git/research_plans/2026-10-19/execution/runs')
OUT=Path(__file__).resolve().parent/'cm19_paths'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def analyze():
    OUT.mkdir(exist_ok=True);planned=[]
    for run in sorted(OLD.iterdir()):
        path=run/'config.json'
        if not path.exists():continue
        cfg=json.loads(path.read_text())
        if cfg.get('smoke'):continue
        pop='kuairand_natural_historical' if cfg['dataset']=='kuairand' else 'movielens_artificial_historical'
        planned.append((run,cfg,pop,'historical',False))
    manifest=json.loads((ART/'transfer_manifest.json').read_text())
    complete_scopes=set()
    for scope in {r['scope'] for r in manifest['runs'] if r['study']=='CM19'}:
        configs=[r for r in manifest['runs'] if r['study']=='CM19' and r['scope']==scope]
        if all((Path(r['run_dir'])/'reassessment_receipt.json').exists() and
               json.loads((Path(r['run_dir'])/'reassessment_receipt.json').read_text())['status']=='COMPLETE_TRAINING_TEST_DEFERRED' for r in configs):
            complete_scopes.add(scope)
            for config in configs:
                run=Path(config['run_dir']);planned.append((run,json.loads((run/'config.json').read_text()),config['population'],scope,True))
    probes=[];endpoints=[];strengths=[];gradients=[];local=[];moments=[];costs=[];runs=[];source=[]
    for run,cfg,pop,scope,new in planned:
        rid='CM19__'+scope+'__'+run.name
        receipt=json.loads((run/'receipt.json').read_text())
        base=dict(global_run_id=rid,historical_run_id=run.name,population=pop,scope=scope,dataset=cfg['dataset'],
            seed=cfg['seed'],model=cfg['model'],optimizer=cfg['optimizer'],stage=cfg['stage'],variant=cfg['variant'],
            decay_mode=cfg['decay_mode'],schedule=cfg.get('schedule','normal_epochs'),fraction=cfg.get('fraction',1.),lr=cfg['lr'])
        runs.append(dict(base,final_steps=receipt['steps'],final_examples=receipt['examples'],epochs=receipt['epochs'],
            parameters=receipt['parameters'],final_state_sha256=receipt['final_state_sha256'],
            nominal_global_factor_float32=float(np.float32(1-cfg['lr']*1e-5)),path=str(run),
            evidence='new full-budget frozen-source run' if new else 'historical saved CSV/config/receipt; original heavy tensors not reloaded'))
        def read(name):
            path=run/name
            if not path.exists():return None
            source.append(dict(global_run_id=rid,path=str(path),sha256=sha(path)))
            return pd.read_csv(path)
        path=read('path_diagnostics.csv')
        if path is not None and len(path):
            for key,g in path.groupby(['label','probe','intervention']):
                assert g[['domain','task']].drop_duplicates().shape[0]==len(g)
                probes.append(dict(base,label=key[0],probe=key[1],intervention=key[2],epoch=int(g.epoch.iloc[0]),
                    step=int(g.step.iloc[0]),examples=int(g.examples.iloc[0]),cells=len(g),
                    measured_cell_rows=int(g.n.sum()),nonzero=int(g.nonzero.sum()),exactly_equal_on_this_probe=bool(g.nonzero.eq(0).all()),
                    max_delta_p=float(g.max_delta.max()),mean_cell_delta_p=g.delta_p.mean(),
                    row_task_weighted_delta_p=np.average(g.delta_p,weights=g.n),macro_delta_auc=g.delta_auc.mean(),
                    macro_delta_logloss=g.delta_logloss.mean(),
                    scope_warning='finite fixed meta/development probe and implemented neutral action; not global irreversibility or retrained model value'))
            e=path[path.label.isin(['selected','final'])].copy()
            for key,value in base.items():e[key]=value
            endpoints.append(e)
        for name,target in [('path_strength.csv',strengths),('gradient_updates.csv',gradients),
                ('offline_local_counterfactual.csv',local),('offline_moment_statistics.csv',moments)]:
            table=read(name)
            if table is None or not len(table):continue
            if name=='gradient_updates.csv' and 'optimizer' in table:
                table=table.rename(columns={'optimizer':'group_optimizer_class'})
            for key,value in base.items():table[key]=value
            target.append(table)
        cp=run/'cost_profile.json'
        if cp.exists():
            cost=json.loads(cp.read_text());costs.append(dict(base,**{k:v for k,v in cost.items() if k!='run_id'}))
            source.append(dict(global_run_id=rid,path=str(cp),sha256=sha(cp)))
    f=pd.DataFrame(probes);f.to_csv(OUT/'probe_checkpoint_summary.csv',index=False)
    for name,frames in [('probe_endpoint_cells',endpoints),('path_strengths',strengths),('actual_gradient_updates',gradients),
            ('same_state_local_counterfactuals',local),('same_state_moments',moments)]:
        table=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
        table.to_csv(OUT/(name+'.csv'),index=False)
        if name=='actual_gradient_updates' and len(table):
            first=table[(table.stage=='G0')&(table.optimizer=='adam')&table.model.isin(['fixed','star0','star1'])&
                (table.step==1)&table.group.isin(['shared','slot'])].copy()
            first['L2_vector_over_data_gradient_norm']=first.wd_vector_norm/first.data_grad_norm
            first['interpretation']='first observed step, parameter-space scale only; not a function learning rate or unique explanation'
            first.to_csv(OUT/'first_step_L2_gradient_context.csv',index=False)
    pd.DataFrame(costs).to_csv(OUT/'cost_profiles.csv',index=False)
    runindex=pd.DataFrame(runs);runindex.to_csv(OUT/'run_budget_and_state_index.csv',index=False)
    pd.DataFrame(source).to_csv(OUT/'source_index.csv',index=False)
    onset=[]
    for key,g in f.groupby(['global_run_id','probe','intervention']):
        clock=g.sort_values('step');zero=clock[clock.exactly_equal_on_this_probe]
        first=int(zero.step.min()) if len(zero) else None
        onset.append(dict(global_run_id=key[0],probe=key[1],intervention=key[2],first_observed_exact_zero_step=first,
            total_observations=len(g),observations_with_exact_zero=len(zero),
            observed_nonzero_after_first_zero=bool(first is not None and (~clock[clock.step>first].exactly_equal_on_this_probe).any()),
            boundary='sparse measured checkpoints only; selected is an earlier clock revisit, not new training'))
    pd.DataFrame(onset).to_csv(OUT/'observed_zero_boundaries.csv',index=False)
    identities=[]
    for key,g in runindex.groupby(['population','scope','model','seed']):
        a=g[(g.stage=='G0')&(g.variant=='native')&(g.optimizer=='adamw')&(g.decay_mode=='standard')]
        b=g[(g.stage=='G1')&(g.variant=='native')&(g.optimizer=='adam')&(g.decay_mode=='wd0')]
        if len(a)!=1 or len(b)!=1:continue
        a=a.iloc[0];b=b.iloc[0]
        assert a.final_steps==b.final_steps and a.final_examples==b.final_examples and a.lr==b.lr
        identities.append(dict(population=key[0],scope=key[1],model=key[2],seed=key[3],nominal_float32_factor=a.nominal_global_factor_float32,
            AdamW_run=a.global_run_id,Adam_WD0_run=b.global_run_id,final_state_sha256_equal=a.final_state_sha256==b.final_state_sha256,
            actual_steps=a.final_steps,examples=a.final_examples,
            evidence='canonical model-state hash in original training receipts, not optimizer checkpoint-file hash'))
    pd.DataFrame(identities).to_csv(OUT/'nominal_AdamW_vs_WD0_model_identities.csv',index=False)
    (OUT/'analysis_status.json').write_text(json.dumps(dict(historical_runs=sum(not r[-1] for r in planned),
        new_runs=sum(r[-1] for r in planned),admitted_new_complete_scopes=sorted(complete_scopes),
        all_new_complete=len(complete_scopes)==4,test_read_by_this_analysis=False,
        scope='path equality is checkpoint/probe/implemented-action conditional; activity is not competitive predictive benefit'),indent=2)+'\n')
    archives=[]
    for path in sorted(OUT.glob('*.csv')):
        if path.name=='large_table_archives.csv' or path.stat().st_size<=5_000_000:continue
        compressed=path.with_suffix(path.suffix+'.gz')
        compressed.write_bytes(gzip.compress(path.read_bytes(),mtime=0))
        archives.append(dict(raw_path=str(path),raw_sha256=sha(path),raw_bytes=path.stat().st_size,
            portable_gzip=compressed.name,gzip_sha256=sha(compressed),gzip_bytes=compressed.stat().st_size,
            recovery='gzip -dc <portable_gzip> recreates the exact raw CSV bytes'))
    pd.DataFrame(archives).to_csv(OUT/'large_table_archives.csv',index=False)
    print('Path records',len(f),'runs',len(runindex),'new scopes',sorted(complete_scopes))

if __name__=='__main__':analyze()
