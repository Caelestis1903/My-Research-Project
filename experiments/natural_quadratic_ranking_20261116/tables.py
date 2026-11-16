"""Numeric Cycle02 export. Scientific decisions remain with the parent."""
import argparse
import csv
import gzip
import json
import math
import statistics
from contextlib import ExitStack
from pathlib import Path

from .protocol import (ATTEMPT, ARMS, KINDS, CELLS, SEEDS, DATASETS, cycle_path, local_path,
    validate_attempt, results, epoch_files, read_json, sha_file, write_json, immutable,
    assert_old_unchanged)


def write_csv(path, rows):
    assert rows
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=names); writer.writeheader(); writer.writerows(rows)


def summary(values):
    assert len(values) == 5 and all(math.isfinite(v) for v in values)
    mean = statistics.mean(values); sd = statistics.stdev(values)
    half = 2.7764451051977987*sd/math.sqrt(5)
    return dict(n=5,mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half,
                positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),zero=sum(v==0 for v in values))


def base(result):
    j = result['job']
    return dict(key=j['key'],dataset=j['dataset'],arm=j['config']['arm'],seed=j['seed'],
                source_hash=j['source_hash'],reused=j['config']['arm']=='CLEAN',test_accessed=False)


def state_hash(record, kind):
    return record['model_hash' if kind=='online' else 'ema_hash']


def paired_tables(endpoints):
    index = {(r['dataset'],r['arm'],r['seed'],r['endpoint'],r['estimator']):r for r in endpoints}
    contrasts = [(ARMS[0],'fixed_ema','CLEAN','fixed_ema'),(ARMS[0],'fixed_ema','CLEAN','online'),
                 (ARMS[0],'online','CLEAN','online'),(ARMS[0],'fixed_ema',ARMS[1],'fixed_ema'),
                 (ARMS[0],'fixed_ema',ARMS[2],'fixed_ema'),(ARMS[1],'fixed_ema','CLEAN','fixed_ema'),
                 (ARMS[2],'fixed_ema','CLEAN','fixed_ema')]
    common = [(a,k,'CLEAN',k) for a in ARMS for k in KINDS]
    common += [(ARMS[0],k,a,k) for a in ARMS[1:] for k in KINDS]
    paired=[]; grouped={}
    for dataset in DATASETS:
        for endpoint in ('selected','E10','E30','E50'):
            for left,lk,right,rk in contrasts if endpoint=='selected' else common:
                comparison=f'{left}/{lk} minus {right}/{rk}'
                for seed in SEEDS:
                    l=index[dataset,left,seed,endpoint,lk]; r=index[dataset,right,seed,endpoint,rk]
                    metrics=[name for name in l if name.endswith(('_auc','_logloss'))]
                    for name in metrics+['epoch']:
                        assert l[name] is not None and r[name] is not None
                        row=dict(dataset=dataset,comparison=comparison,endpoint=endpoint,seed=seed,metric=name,
                            left_key=l['key'],right_key=r['key'],left_epoch=l['epoch'],right_epoch=r['epoch'],
                            left_state_hash=l['state_hash'],right_state_hash=r['state_hash'],
                            left=l[name],right=r[name],delta=l[name]-r[name])
                        paired.append(row); grouped.setdefault((dataset,comparison,endpoint,name),[]).append(row['delta'])
    summaries=[dict(dataset=d,comparison=c,endpoint=e,metric=m,**summary(values)) for (d,c,e,m),values in grouped.items()]
    return paired,summaries


def batch_records(path):
    opener = gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as f:
        try:
            for line in f:
                if line.strip(): yield json.loads(line)
        except EOFError:
            # A killed physical attempt can end before the gzip footer; prior complete lines remain raw costs.
            return


def export(attempt=ATTEMPT):
    attempt=cycle_path(attempt); manifest=validate_attempt(attempt,require_smoke=True)
    records=results(attempt); target=local_path(attempt,attempt/'tables'); target.mkdir(parents=True,exist_ok=True)
    curves=[]; cells=[]; endpoints=[]; endpoint_cells=[]; costs=[]; attempts=[]; masses=[]; stats=[]; inputs=[]
    committed_logs=[]
    for result in records:
        b=base(result); epochs=epoch_files(attempt,result); assert len(epochs)==50
        for epoch,(path,record) in sorted(epochs.items()):
            inputs.append(dict(path=str(path),sha256=sha_file(path),purpose='committed_epoch',key=b['key']))
            for kind in KINDS:
                v=record['validation'][kind]
                row=dict(**b,epoch=epoch,estimator=kind,state_hash=state_hash(record,kind),
                    epoch_order_hash=record['epoch_order_hash'],**v)
                curves.append(row)
                for cell in CELLS:
                    cells.append(dict(**b,epoch=epoch,estimator=kind,cell=cell,state_hash=row['state_hash'],
                        **{m:v[cell+'_'+m] for m in ('auc','logloss','count','positive','negative')}))
                for endpoint in ('selected','E10','E30','E50'):
                    wanted=result['selected_epochs'][kind] if endpoint=='selected' else int(endpoint[1:])
                    if epoch!=wanted: continue
                    endpoints.append(dict(**row,endpoint=endpoint))
                    for cell in CELLS:
                        endpoint_cells.append(dict(**b,endpoint=endpoint,epoch=epoch,estimator=kind,cell=cell,
                            state_hash=row['state_hash'],**{m:v[cell+'_'+m] for m in ('auc','logloss','count','positive','negative')}))
            if b['reused']: continue
            log=local_path(attempt,record['batch_log']); assert sha_file(log)==record['batch_log_sha256']
            committed_logs.append((b,epoch,log))
            for name,r in record['objective_ledger']['masses'].items():
                masses.append(dict(**b,epoch=epoch,mass=name,batches=record['batch_count'],**r))
            for cell,r in record['objective_ledger']['cells'].items():
                for name,value in r['statistics'].items():
                    stats.append(dict(**b,epoch=epoch,cell=cell,statistic=name,n=r['n'],n0=r['n0'],n1=r['n1'],
                        denominator_units='class_rows' if name in ('mu0','mu1','var0','var1','bbar0','bbar1') else 'cell_rows',
                        **value))
        physical=[]
        if not b['reused']:
            for path in sorted((attempt/'runs'/b['key']).glob('attempt*/physical_attempt.json'),key=lambda p:int(p.parent.name[7:])):
                r=read_json(path); assert r['key']==b['key'] and r['status']!='RUNNING'
                physical.append(r)
                profile=r.get('profile') or {}
                numeric_costs={k:v for k,v in r['costs'].items() if isinstance(v,(int,float))}
                attempts.append(dict(**b,attempt=r['attempt'],status=r['status'],seconds=r['seconds'],
                    path=str(path),sha256=sha_file(path),failure=json.dumps(r.get('failure')),**numeric_costs,
                    full_process_peak_bytes=profile.get('full_process_peak_bytes'),
                    torch_reserved_peak_bytes=profile.get('torch_reserved_peak_bytes'),
                    peak_host_rss_bytes=profile.get('peak_host_rss_bytes')))
            assert physical
        numeric_names=set().union(*(r['costs'] for r in physical)) if physical else set(result.get('costs',{}))
        totals={name:sum(r['costs'].get(name,0) for r in physical) for name in numeric_names
                if physical and all(isinstance(r['costs'].get(name,0),(int,float)) for r in physical)}
        for name in ('training_seconds','evaluation_seconds','diagnostic_seconds','serialization_seconds','transfer_seconds',
                     'serialization_bytes','ordinary_forwards','scalar_backwards','evaluation_forwards','evaluation_rows'):
            totals.setdefault(name,None)
        if totals.get('unmeasured_transfer_transactions',0):
            for name in ('serialization_seconds','transfer_seconds','serialization_bytes'): totals[name]=None
        costs.append(dict(**b,physical_attempts=len(physical) if physical else None,
            physical_worker_seconds=sum(r['seconds'] for r in physical) if physical else result.get('worker_seconds'),
            death_observation_upper_bounds=sum(r.get('seconds_semantics')=='launch_to_observed_death_upper_bound' for r in physical) if physical else None,
            useful_optimizer_updates=result['counts']['optimizer_updates'],useful_ema_updates=result['counts']['ema_updates'],
            peak_allocated_bytes=result.get('peak_allocated_bytes'),peak_reserved_bytes=result.get('peak_reserved_bytes'),
            peak_full_process_bytes=max((r.get('profile',{}).get('full_process_peak_bytes',0) for r in physical if r.get('profile')),default=None),
            cost_provenance='all_physical_attempt_receipts' if physical else 'historical_result_missing_fields_null',**totals))
    assert len(endpoints)==320 and len(curves)==4000 and len(cells)==24000 and len(endpoint_cells)==1920
    assert sum(r['useful_optimizer_updates'] for r in costs if not r['reused'])==51750
    weight_fields=['key','dataset','arm','seed','purpose','epoch','batch','global_step','row_order_hash','branch',
        'loss','clean_loss','completed_loss','B','entries','mixture_lambda','total_mass','task0_mass','task1_mass',
        'bce_mass','opposite_mass','cell','n','n0','n1','pi0','pi1','weight','cell_bce_mass','cell_opposite_mass',
        'bce_coefficient0','bce_coefficient1','mu0','mu1','var0','var1','bbar0','bbar1','S','Q_R','Q_B','Q_M','risk']
    diag_fields=['key','dataset','arm','seed','purpose','epoch','batch','global_step','row_order_hash','parameter_group',
        'data_gradient_norm','parameter_update_norm','active_parameters','none_parameters','parameter_step_min','parameter_step_max',
        'coordinates','parameter_names']
    batch_count=0; weight_count=0; diagnostic_count=0
    with ExitStack() as stack:
        weights=csv.DictWriter(stack.enter_context((target/'WEIGHTS.csv').open('w',newline='')),fieldnames=weight_fields); weights.writeheader()
        diagnostics=csv.DictWriter(stack.enter_context((target/'DIAGNOSTICS.csv').open('w',newline='')),fieldnames=diag_fields); diagnostics.writeheader()
        for b,epoch,path in committed_logs:
            batch_inputs=0
            for record in batch_records(path):
                batch_count+=1; batch_inputs+=1; ledger=record['ledger']
                context={k:b[k] for k in ('key','dataset','arm','seed')}
                context.update(purpose='formal_committed',epoch=epoch,batch=record['batch'],global_step=record['global_step'],row_order_hash=record['row_order_hash'])
                shared=dict(context,branch=record['branch'],**{k:record[k] for k in ('loss','clean_loss','completed_loss')},
                    **{k:ledger[k] for k in ('B','entries','mixture_lambda','total_mass','bce_mass','opposite_mass')},
                    task0_mass=ledger['task_mass']['0'],task1_mass=ledger['task_mass']['1'])
                for cell,value in ledger['cells'].items():
                    value=dict(value); value['cell_bce_mass']=value.pop('bce_mass'); value['cell_opposite_mass']=value.pop('opposite_mass')
                    weights.writerow(dict(shared,cell=cell,**value)); weight_count+=1
                d=record['diagnostics']
                if d is not None:
                    common={k:v for k,v in d.items() if k!='parameter_groups'}
                    diagnostics.writerow(dict(context,parameter_group='ALL',**common)); diagnostic_count+=1
                    for prefix,value in d['parameter_groups'].items():
                        value=dict(value); value['parameter_names']=json.dumps(value['parameter_names'])
                        diagnostics.writerow(dict(context,parameter_group=prefix,**value)); diagnostic_count+=1
            assert batch_inputs==next(j['config']['steps_per_epoch'] for j in manifest['jobs'] if j['key']==b['key'])
            inputs.append(dict(path=str(path),sha256=sha_file(path),purpose='committed_batches',key=b['key']))
    assert batch_count==51750 and weight_count==310500
    # Keep every failed/replayed physical ledger separately from useful batches.
    physical_logs=[]; useful_paths={str(p) for _,_,p in committed_logs}
    for j in manifest['jobs']:
        for path in sorted((attempt/'runs'/j['key']).glob('attempt*/batches_epoch_*.jsonl.gz')):
            if str(path) in useful_paths: continue
            count=sum(1 for _ in batch_records(path))
            physical_logs.append(dict(key=j['key'],path=str(path),sha256=sha_file(path),returned_batches=count,purpose='uncommitted_physical_attempt'))
    paired,paired_summary=paired_tables(endpoints)
    groups={}
    for row in endpoints:
        for metric in (k for k in row if k.endswith(('_auc','_logloss'))):
            groups.setdefault((row['dataset'],row['arm'],row['estimator'],row['endpoint'],metric),[]).append(row[metric])
    descriptive=[dict(dataset=d,arm=a,estimator=k,endpoint=e,metric=m,**summary(v)) for (d,a,k,e,m),v in groups.items()]
    tables={'ENDPOINTS.csv':endpoints,'ENDPOINT_CELLS.csv':endpoint_cells,'CURVES.csv':curves,'CELL_CURVES.csv':cells,
        'COSTS.csv':costs,'PHYSICAL_ATTEMPTS.csv':attempts,'MASS_SUMMARIES.csv':masses,'STATISTICS.csv':stats,
        'PAIRED.csv':paired,'PAIRED_SUMMARY.csv':paired_summary,'DESCRIPTIVE.csv':descriptive}
    smoke = read_json(attempt/'SMOKE_REPORT.json')
    tables['SMOKE_COSTS.csv'] = [dict(key=r['key'],dataset=r['dataset'],arm=r['arm'],purpose='engineering_smoke',
        seconds=r['seconds'],full_process_peak_bytes=r['profile']['full_process_peak_bytes'],
        **{k:v for k,v in r['costs'].items() if isinstance(v,(int,float))}) for r in smoke['cases']]
    for name,rows in tables.items(): write_csv(target/name,rows)
    write_json(target/'UNCOMMITTED_PHYSICAL_LOGS.json',physical_logs)
    extra=[]
    for name in ('admission.jsonl','launch_ledger.jsonl','foreground_runs.jsonl','SMOKE_REPORT.json'):
        path=attempt/name
        if path.exists(): extra.append(dict(path=str(path),sha256=sha_file(path)))
    report=dict(source_hash=manifest['source_hash'],design_bundle_sha256=manifest['design_bundle_sha256'],
        trajectories=40,new_trajectories=30,reused_clean=10,endpoints=320,curves=4000,cell_curves=24000,
        useful_batches=batch_count,weights=weight_count,diagnostics=diagnostic_count,
        inputs=inputs,physical_logs=physical_logs,extra=extra,test_accessed=False,
        files=[dict(path=str(target/name),rows=len(rows),sha256=sha_file(target/name)) for name,rows in tables.items()])
    write_json(target/'EXPORT_RECEIPT.json',report)
    write_json(attempt/'OLD_INVENTORY_AFTER_EXPORT.json',assert_old_unchanged(attempt))
    return dict(paths=[r['path'] for r in report['files']],trajectories=40,endpoints=320,curves=4000,
                cell_curves=24000,useful_batches=batch_count,test_accessed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--attempt',default=str(ATTEMPT))
    print(json.dumps(export(parser.parse_args().attempt)))
