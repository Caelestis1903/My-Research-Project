"""Numeric Cycle03 export. Scientific decisions remain with the parent."""
import argparse
import csv
import gzip
import json
import math
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

from .protocol import (ATTEMPT, ARMS, KINDS, CELLS, SEEDS, DATASETS, cycle_path, local_path,
    validate_attempt, results, epoch_files, read_json, sha_file, write_json, immutable,
    assert_old_unchanged, PHASE_FIELDS)


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
                source_hash=j['source_hash'],reused=j['config']['arm'] not in ARMS,test_accessed=False)


def state_hash(record, kind):
    return record['model_hash' if kind=='online' else 'ema_hash']


def paired_tables(endpoints):
    index = {(r['dataset'],r['arm'],r['seed'],r['endpoint'],r['estimator']):r for r in endpoints}
    contrasts = [(ARMS[0],'CLEAN'),(ARMS[0],ARMS[1]),(ARMS[0],'NQR-025'),
                 (ARMS[1],'CLEAN'),(ARMS[1],'NQR-025')]
    paired=[]; grouped={}
    for dataset in DATASETS:
        for endpoint in ('selected','E10','E30','E40','E50'):
            for kind in KINDS:
                for left,right in contrasts:
                    comparison=f'{left} minus {right}'
                    for seed in SEEDS:
                        l=index[dataset,left,seed,endpoint,kind]; r=index[dataset,right,seed,endpoint,kind]
                        metrics=[name for name in l if name.endswith(('_auc','_logloss'))]
                        for name in metrics+['epoch']:
                            available=l.get(name) is not None and r.get(name) is not None
                            delta=l[name]-r[name] if available else None
                            favorable=(delta<0 if name.endswith('_logloss') else delta>0) if available and name!='epoch' else None
                            row=dict(dataset=dataset,comparison=comparison,estimator=kind,endpoint=endpoint,seed=seed,metric=name,
                                primary_metric=name=='macro_auc',left_key=l['key'],right_key=r['key'],left_epoch=l['epoch'],right_epoch=r['epoch'],
                                left_state_hash=l['state_hash'],right_state_hash=r['state_hash'],
                                left=l.get(name),right=r.get(name),delta=delta,available=available,
                                favorable_direction='negative' if name.endswith('_logloss') else 'positive' if name!='epoch' else None,
                                favorable=favorable)
                            paired.append(row); grouped.setdefault((dataset,comparison,kind,endpoint,name),[]).append(delta)
    summaries=[]
    for (d,c,k,e,m),values in grouped.items():
        available=all(v is not None for v in values)
        numeric=summary(values) if available else dict(n=5,mean=None,sd=None,ci95_low=None,ci95_high=None,positive=None,negative=None,zero=None)
        summaries.append(dict(dataset=d,comparison=c,estimator=k,endpoint=e,metric=m,available=available,
            favorable=numeric['negative'] if m.endswith('_logloss') else numeric['positive'] if m!='epoch' else None,
            unfavorable=numeric['positive'] if m.endswith('_logloss') else numeric['negative'] if m!='epoch' else None,
            **numeric))
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
    tick=time.perf_counter()
    attempt=cycle_path(attempt); manifest=validate_attempt(attempt,require_smoke=True)
    records=results(attempt); target=local_path(attempt,attempt/'tables'); target.mkdir(parents=True,exist_ok=True)
    curves=[]; cells=[]; endpoints=[]; endpoint_cells=[]; costs=[]; attempts=[]; masses=[]; stats=[]; inputs=[]
    committed_logs=[]; historical_diagnostics=[]; diagnostic_availability=[]; phase_rows=[]; phase_expected={}
    result_paths={j['key']:attempt/'runs'/j['key']/'result.json' for j in manifest['jobs']}
    result_paths.update({e['key']:Path(e['result_path']) for e in read_json(attempt/'reused_reference_index.json')['entries']})
    for result in records:
        b=base(result); epochs=epoch_files(attempt,result); assert len(epochs)==50
        result_path=result_paths[b['key']]
        inputs.append(dict(path=str(result_path),sha256=sha_file(result_path),purpose='trajectory_result',key=b['key']))
        for epoch,(path,record) in sorted(epochs.items()):
            inputs.append(dict(path=str(path),sha256=sha_file(path),purpose='committed_epoch',key=b['key']))
            for kind in KINDS:
                v=record['validation'][kind]
                row=dict(**b,epoch=epoch,estimator=kind,state_hash=state_hash(record,kind),
                    epoch_order_hash=record['epoch_order_hash'],schedule_available=not b['reused'],
                    schedule_phase=record.get('schedule',{}).get('schedule_phase'),
                    effective_lambda=record.get('schedule',{}).get('effective_lambda'),**v)
                curves.append(row)
                for cell in CELLS:
                    cells.append(dict(**b,epoch=epoch,estimator=kind,cell=cell,state_hash=row['state_hash'],
                        **{m:v[cell+'_'+m] for m in ('auc','logloss','count','positive','negative')}))
                for endpoint in ('selected','E10','E30','E40','E50'):
                    wanted=result['selected_epochs'][kind] if endpoint=='selected' else int(endpoint[1:])
                    if epoch!=wanted: continue
                    endpoints.append(dict(**row,endpoint=endpoint))
                    for cell in CELLS:
                        endpoint_cells.append(dict(**b,endpoint=endpoint,epoch=epoch,estimator=kind,cell=cell,
                            state_hash=row['state_hash'],**{m:v[cell+'_'+m] for m in ('auc','logloss','count','positive','negative')}))
            if epoch in (1,10,11,25,40,41,50) and (b['reused'] or epoch!=25):
                source_log=record.get('batch_log'); diagnostic=None
                if b['reused'] and source_log and Path(source_log).is_file():
                    assert sha_file(source_log)==record['batch_log_sha256']
                    first=next(batch_records(Path(source_log)),None)
                    diagnostic=first.get('diagnostics') if first else None
                    if diagnostic is not None:
                        historical_diagnostics.append((b,epoch,first,source_log))
                diagnostic_availability.append(dict(**b,epoch=epoch,batch=1,
                    registered_observation=epoch in (1,10,11,40,41,50),
                    available=bool(diagnostic) if b['reused'] else epoch in (1,10,11,40,41,50),
                    provenance=source_log if b['reused'] and diagnostic else 'historical_unavailable' if b['reused'] else record['batch_log']))
            if b['reused']: continue
            phase_rows.append(dict(**b,epoch=epoch,global_step=record['global_step'],
                **record['schedule'],**record['phase_counters'],counts=json.dumps(record['counts']),
                adam_step_counters=json.dumps(record['adam_step_counters']),
                last_parameter_activity=json.dumps(record['last_parameter_activity']),
                phase_losses=json.dumps(record['phase_losses']),prefix_check=json.dumps(record['prefix_check'])))
            log=local_path(attempt,record['batch_log']); assert sha_file(log)==record['batch_log_sha256']
            committed_logs.append((b,epoch,log))
            phase_expected[b['key'],epoch]=record['phase_counters']
            for name,r in record['objective_ledger']['masses'].items():
                masses.append(dict(**b,epoch=epoch,phase=record['schedule']['schedule_phase'],mass=name,batches=record['batch_count'],**r))
            for cell,r in record['objective_ledger']['cells'].items():
                for name,value in r['statistics'].items():
                    stats.append(dict(**b,epoch=epoch,phase=record['schedule']['schedule_phase'],cell=cell,statistic=name,n=r['n'],n0=r['n0'],n1=r['n1'],
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
        if b['reused']:
            totals={name:value for name,value in result.get('costs',{}).items() if isinstance(value,(int,float))}
        for name in ('training_seconds','evaluation_seconds','diagnostic_seconds','serialization_seconds','transfer_seconds',
                     'serialization_bytes','ordinary_forwards','scalar_backwards','probe_forwards','view_forwards',
                     'algorithm_rng_draws','evaluation_forwards','evaluation_rows'):
            totals.setdefault(name,None)
        if totals.get('unmeasured_transfer_transactions',0):
            for name in ('serialization_seconds','transfer_seconds','serialization_bytes'): totals[name]=None
        costs.append(dict(**b,physical_attempts=len(physical) if physical else result.get('physical_attempts'),
            physical_worker_seconds=sum(r['seconds'] for r in physical) if physical else result.get('worker_seconds'),
            death_observation_upper_bounds=sum(r.get('seconds_semantics')=='launch_to_observed_death_upper_bound' for r in physical) if physical else None,
            useful_optimizer_updates=result['counts']['optimizer_updates'],useful_ema_updates=result['counts']['ema_updates'],
            peak_allocated_bytes=result.get('peak_allocated_bytes'),peak_reserved_bytes=result.get('peak_reserved_bytes'),
            peak_full_process_bytes=max((r.get('profile',{}).get('full_process_peak_bytes',0) for r in physical if r.get('profile')),default=None),
            cost_provenance='all_physical_attempt_receipts' if physical else 'historical_result_available_fields_otherwise_null',**totals))
    assert len(endpoints)==400 and len(curves)==4000 and len(cells)==24000 and len(endpoint_cells)==2400
    assert len(phase_rows)==1000
    assert sum(r['useful_optimizer_updates'] for r in costs if not r['reused'])==34500
    weight_fields=['key','dataset','arm','seed','purpose','epoch','batch','global_step','row_order_hash','branch',
        'schedule_id','schedule_epoch','schedule_phase','scheduled_active','effective_lambda','external_arm','inner_arm','clock_kind',
        'loss','clean_loss','completed_loss','B','entries','mixture_lambda','total_mass','task0_mass','task1_mass',
        'bce_mass','opposite_mass','cell','n','n0','n1','pi0','pi1','weight','cell_bce_mass','cell_opposite_mass',
        'bce_coefficient0','bce_coefficient1','mu0','mu1','var0','var1','bbar0','bbar1','S','Q_R','Q_B','Q_M','risk']
    diag_fields=['key','dataset','arm','seed','purpose','epoch','batch','global_step','row_order_hash','parameter_group',
        'data_gradient_norm','parameter_update_norm','active_parameters','none_parameters','parameter_step_min','parameter_step_max',
        'coordinates','parameter_names','available','reused','source_path']
    batch_count=0; weight_count=0; diagnostic_count=0; schedule_count=0; logged_phase={}
    schedule_fields=['key','dataset','arm','seed','purpose','epoch','batch','global_step','row_order_hash',
        'schedule_id','schedule_epoch','schedule_phase','scheduled_active','effective_lambda','external_arm','inner_arm','clock_kind']
    schedule_fields += list(PHASE_FIELDS)+['phase_'+n for n in PHASE_FIELDS]+['optimizer_updates','ema_updates']
    with ExitStack() as stack:
        weights=csv.DictWriter(stack.enter_context((target/'WEIGHTS.csv').open('w',newline='')),fieldnames=weight_fields); weights.writeheader()
        diagnostics=csv.DictWriter(stack.enter_context((target/'DIAGNOSTICS.csv').open('w',newline='')),fieldnames=diag_fields); diagnostics.writeheader()
        schedules=csv.DictWriter(stack.enter_context((target/'SCHEDULES.csv').open('w',newline='')),fieldnames=schedule_fields); schedules.writeheader()
        for b,epoch,path in committed_logs:
            batch_inputs=0
            for record in batch_records(path):
                batch_count+=1; batch_inputs+=1; ledger=record['ledger']
                context={k:b[k] for k in ('key','dataset','arm','seed')}
                context.update(purpose='formal_committed',epoch=epoch,batch=record['batch'],global_step=record['global_step'],row_order_hash=record['row_order_hash'])
                cumulative=logged_phase.setdefault(b['key'],{n:0 for n in PHASE_FIELDS})
                assert record['phase_increments']['nonzero_opportunity_updates']==int(ledger['opposite_mass']>0)
                for name in PHASE_FIELDS: cumulative[name]+=record['phase_increments'][name]
                assert cumulative==record['phase_counters'], ('LOG_PHASE_MISMATCH',b['key'],epoch,record['batch'])
                shared=dict(context,**record['schedule'],branch=record['branch'],**{k:record[k] for k in ('loss','clean_loss','completed_loss')},
                    **{k:ledger[k] for k in ('B','entries','mixture_lambda','total_mass','bce_mass','opposite_mass')},
                    task0_mass=ledger['task_mass']['0'],task1_mass=ledger['task_mass']['1'])
                schedules.writerow(dict(context,**record['schedule'],**record['phase_increments'],
                    **{'phase_'+n:v for n,v in record['phase_counters'].items()},**record['counts'])); schedule_count+=1
                for cell,value in ledger['cells'].items():
                    value=dict(value); value['cell_bce_mass']=value.pop('bce_mass'); value['cell_opposite_mass']=value.pop('opposite_mass')
                    weights.writerow(dict(shared,cell=cell,**value)); weight_count+=1
                d=record['diagnostics']
                if d is not None:
                    common={k:v for k,v in d.items() if k!='parameter_groups'}
                    common.update(available=True,reused=False,source_path=str(path))
                    diagnostics.writerow(dict(context,parameter_group='ALL',**common)); diagnostic_count+=1
                    for prefix,value in d['parameter_groups'].items():
                        value=dict(value); value['parameter_names']=json.dumps(value['parameter_names'])
                        diagnostics.writerow(dict(context,parameter_group=prefix,available=True,reused=False,source_path=str(path),**value)); diagnostic_count+=1
            assert batch_inputs==next(j['config']['steps_per_epoch'] for j in manifest['jobs'] if j['key']==b['key'])
            assert logged_phase[b['key']]==phase_expected[b['key'],epoch]
            inputs.append(dict(path=str(path),sha256=sha_file(path),purpose='committed_batches',key=b['key']))
        for b,epoch,record,source_log in historical_diagnostics:
            d=record['diagnostics']; context={n:b[n] for n in ('key','dataset','arm','seed')}
            context.update(purpose='historical_numeric',epoch=epoch,batch=record['batch'],global_step=record['global_step'],
                row_order_hash=record['row_order_hash'],available=True,reused=True,source_path=source_log)
            diagnostics.writerow(dict(context,parameter_group='ALL',**{n:v for n,v in d.items() if n!='parameter_groups'})); diagnostic_count+=1
            for prefix,value in d['parameter_groups'].items():
                value=dict(value); value['parameter_names']=json.dumps(value['parameter_names'])
                diagnostics.writerow(dict(context,parameter_group=prefix,**value)); diagnostic_count+=1
            inputs.append(dict(path=source_log,sha256=sha_file(source_log),purpose='historical_diagnostics',key=b['key']))
        for row in diagnostic_availability:
            if row['available']: continue
            diagnostics.writerow(dict(**{n:row[n] for n in ('key','dataset','arm','seed','epoch','batch','reused')},
                purpose='historical_unavailable',parameter_group='ALL',available=False,source_path=row['provenance']))
            diagnostic_count+=1
    assert batch_count==34500 and weight_count==207000 and schedule_count==34500
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
        'PHASE_COUNTERS.csv':phase_rows,'DIAGNOSTIC_AVAILABILITY.csv':diagnostic_availability,
        'PAIRED.csv':paired,'PAIRED_SUMMARY.csv':paired_summary,'DESCRIPTIVE.csv':descriptive}
    smoke = read_json(attempt/'SMOKE_REPORT.json')
    tables['SMOKE_COSTS.csv'] = [dict(key=r['key'],dataset=r['dataset'],arm=r['arm'],purpose='engineering_smoke',
        seconds=r['seconds'],full_process_peak_bytes=r['profile']['full_process_peak_bytes'],
        **{k:v for k,v in r['costs'].items() if isinstance(v,(int,float))}) for r in smoke['cases']]
    for name,rows in tables.items(): write_csv(target/name,rows)
    write_json(target/'UNCOMMITTED_PHYSICAL_LOGS.json',physical_logs)
    extra=[]
    for path in sorted((attempt/'runs').glob('*/attempt*/resource_samples.jsonl')):
        extra.append(dict(path=str(local_path(attempt,path)),sha256=sha_file(path),purpose='measured_resource_samples'))
    for name in ('admission.jsonl','smoke_admission.jsonl','launch_ledger.jsonl','foreground_runs.jsonl','SMOKE_REPORT.json',
        'SMOKE_REVIEW.json','STORAGE_OWNER.json','STORAGE_HANDOFF.json','run_manifest.json','source_receipt.json',
        'reused_reference_index.json','OLD_INVENTORY_BEFORE.json','PREPARATION_COST.json'):
        path=attempt/name
        if path.exists(): extra.append(dict(path=str(path),sha256=sha_file(path)))
    report=dict(source_hash=manifest['source_hash'],design_bundle_sha256=manifest['design_bundle_sha256'],
        trajectories=40,new_trajectories=20,reused_clean=10,reused_constant_nqr=10,endpoints=400,curves=4000,cell_curves=24000,
        useful_batches=batch_count,weights=weight_count,diagnostics=diagnostic_count,
        inputs=inputs,physical_logs=physical_logs,extra=extra,export_seconds=time.perf_counter()-tick,test_accessed=False,
        files=[dict(path=str(target/name),rows=len(rows),sha256=sha_file(target/name)) for name,rows in tables.items()])
    report['files'] += [dict(path=str(target/name),rows=count,sha256=sha_file(target/name)) for name,count in
        (('WEIGHTS.csv',weight_count),('DIAGNOSTICS.csv',diagnostic_count),('SCHEDULES.csv',schedule_count))]
    write_json(target/'EXPORT_RECEIPT.json',report)
    write_json(attempt/'OLD_INVENTORY_AFTER_EXPORT.json',assert_old_unchanged(attempt))
    return dict(paths=[r['path'] for r in report['files']],trajectories=40,endpoints=400,curves=4000,
                cell_curves=24000,useful_batches=batch_count,test_accessed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--attempt',default=str(ATTEMPT))
    print(json.dumps(export(parser.parse_args().attempt)))
