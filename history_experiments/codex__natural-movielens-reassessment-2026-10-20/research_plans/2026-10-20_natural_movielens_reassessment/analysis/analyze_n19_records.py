"""Known-simulator audit support, rank budgets and realized per-visit masks."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ART=Path('/home_new/xzf_natural_movielens_reassessment')
OLD=Path('/home/xzf/Code/phase0-3-n19-git/research_plans/2026-10-18/N19/runs')
OUT=Path(__file__).resolve().parent/'n19_records'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def analyze():
    OUT.mkdir(exist_ok=True);runs=[];source=[]
    for run in sorted(OLD.iterdir()):
        path=run/'config.json'
        if not path.exists():continue
        cfg=json.loads(path.read_text())
        if cfg.get('engineering'):continue
        pop='kuairand_natural_historical' if cfg['dataset']=='kuairand' else 'movielens_artificial_historical'
        runs.append((run,cfg,pop,'historical'))
    plan=[r for r in json.loads((ART/'transfer_manifest.json').read_text())['runs'] if r['study']=='N19']
    admitted=[]
    for scope in sorted({r['scope'] for r in plan}):
        group=[r for r in plan if r['scope']==scope]
        if not all((Path(r['run_dir'])/'reassessment_receipt.json').exists() and
            json.loads((Path(r['run_dir'])/'reassessment_receipt.json').read_text())['status']=='COMPLETE_TRAINING_TEST_DEFERRED' for r in group):continue
        admitted.append(scope)
        for r in group:runs.append((Path(r['run_dir']),json.loads((Path(r['run_dir'])/'config.json').read_text()),r['population'],scope))
    counts=[];rates=[];rmse=[];fits=[];budgets=[];rank=[];copulas=[]
    for run,cfg,pop,scope in runs:
        base=dict(global_run_id='N19__'+scope+'__'+run.name,population=pop,scope=scope,dataset=cfg['dataset'],
            method=cfg['method'],suite=cfg['suite'],seed=cfg['seed'],optimizer=cfg['optimizer'],
            backbone=cfg.get('backbone') or 'm3oe',noise_interaction=cfg.get('noise_interaction',0.))
        def read(name):
            path=run/name
            if not path.exists():return None
            source.append(dict(global_run_id=base['global_run_id'],path=str(path),sha256=sha(path)))
            return json.loads(path.read_text())
        receipt=read('receipt.json');read('config.json')
        if cfg['suite']=='noise':
            calibration=read('audit_calibration.json');estimate=read('noise_estimates.json');truth=cfg['noise_rates']
            if calibration:
                support=sorted(calibration['counts'],key=lambda c:(c['domain'],c['task'],c['clean']))
                support_sha=hashlib.sha256(json.dumps(support,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                for row in support:counts.append(dict(base,**row,audit_count_and_flip_signature=support_sha,
                    replication_scope='optimization seeds share fixed audit draw; these are not independent audit replicates'))
                for fit in calibration.get('fits',[]):fits.append(dict(base,fit=json.dumps(fit,sort_keys=True)))
            for stage,record in [('audit_initial',calibration),('final',estimate)]:
                if not record:continue
                errors=[]
                for label in ('a','b'):
                    array=np.asarray(record[label]);true=np.asarray(truth[label]);assert array.shape==true.shape==(3,2)
                    for d in range(3):
                        for task in range(2):
                            error=float(array[d,task]-true[d,task]);errors.append(error)
                            rates.append(dict(base,stage=stage,domain=d,task=task,rate=label,
                                estimate=float(array[d,task]),simulator_truth=float(true[d,task]),error=error))
                rmse.append(dict(base,stage=stage,macro_rate_RMSE=float(np.sqrt(np.mean(np.square(errors)))),
                    scope_warning='known synthetic channel, fixed audit draw; not real corruption identification'))
        if cfg['suite']=='sampling':
            trajectory=pd.read_csv(run/'trajectory.csv');examples=int(trajectory.examples.iloc[-1])
            policy=receipt.get('policy_counts') or {};policy_rows=policy.get('rows',0)
            budgets.append(dict(base,epochs=receipt['epochs'],main_steps=receipt['steps'],main_row_visits=examples,
                labels_queried=receipt['labels_queried'],pairs_queried=receipt['pairs_queried'],
                labels_per_row_visit=receipt['labels_queried']/examples,
                pair_events_per_row_visit=receipt['pairs_queried']/examples,
                high_endpoint_rows=policy.get('high_rows'),policy_observed_rows=policy_rows,
                high_endpoint_fraction=policy.get('high_rows',0)/policy_rows if policy_rows else np.nan,
                lower_endpoint_fraction=1-policy.get('high_rows',0)/policy_rows if policy_rows else np.nan,
                cost_scope='per-visit simulated observation of static cacheable labels; dense model/EMA computation remains'))
            last=read('mechanism_last.json')
            for index,row in enumerate(last.get('batches',[]) if last else []):
                copulas.append(dict(base,epoch=last['epoch'],batch=index,raw_fields=json.dumps(row,sort_keys=True)))
        allocation=read('rank_allocation.json')
        if allocation:
            assert sum(allocation['active_ranks'])==48
            for d,n in enumerate(allocation['active_ranks']):
                rank.append(dict(base,domain=d,active_rank=n,total_active_rank=48,
                    initial_config_parameter_count=cfg['parameters'],declared_deployed_parameters=receipt['parameters'],
                    raw_allocation=json.dumps(allocation,sort_keys=True),
                    scope_warning='active factor rank/storage is not full shared STAR operator rank or predictive utility'))
    for name,rows in [('audit_class_support',counts),('audit_rate_estimates',rates),('audit_rate_RMSE',rmse),
            ('audit_fit_convergence_records',fits),('realized_mask_budgets',budgets),('rank_allocations',rank),
            ('final_mask_policy_raw_batches',copulas),('source_index',source)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    (OUT/'analysis_status.json').write_text(json.dumps(dict(historical_runs=sum(r[-1]=='historical' for r in runs),
        admitted_new_scopes=admitted,all_new_scopes_complete=len(admitted)==3,test_read=False,
        audit_optimization_seeds_are_not_audit_replicates=True,mask_counts_are_not_unique_annotation_cost=True),indent=2)+'\n')
    print('N19 records',len(runs),'runs; new scopes',admitted)

if __name__=='__main__':analyze()
