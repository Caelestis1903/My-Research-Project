"""Receipt-bound numeric extraction. No efficacy decisions or winner selection."""
import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from . import protocol as p

ENDPOINTS = ('selected','E1','E10','E30','E50')
STATISTICS = ('mu0','mu1','var0','var1','bbar0','bbar1','H','Q_R')


def csv_write(path, rows, columns=()):
    path=p.new_path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys([*columns,*(k for r in rows for k in r)]))
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields); w.writeheader()
        for row in rows:
            p.json_safe(row)
            w.writerow({k:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False) if isinstance(v,(dict,list,tuple)) else v) for k,v in row.items()})
    return dict(path=str(path),sha256=p.sha_file(path),rows=len(rows))


def paired_summary(values, *, favorable_direction):
    valid=[x for x in values if x is not None]
    n=len(valid); mean=statistics.mean(valid) if n else None
    sd=statistics.stdev(valid) if n>1 else None
    half=2.7764451051977987*sd/math.sqrt(5) if n==5 else None
    sign=1 if favorable_direction=='positive' else -1
    return dict(available=n==5,n=n,mean=mean,sd=sd,ci95_low=None if half is None else mean-half,
                ci95_high=None if half is None else mean+half,positive=sum(x>0 for x in valid),
                negative=sum(x<0 for x in valid),zero=sum(x==0 for x in valid),
                favorable=sum(sign*x>0 for x in valid),unfavorable=sum(sign*x<0 for x in valid),
                null_reason=None if n==5 else 'incomplete_five_seed_coverage')


def comparisons():
    w,s=p.ARMS; c='CLEAN'; n='NQR-025'
    result=[(f'{left}-{right}/{kind}',left,kind,right,kind) for kind in p.ESTIMATORS
            for left,right in ((w,c),(s,c),(w,s),(n,w),(n,c))]
    result.append(('W_fixed_ema-C_online',w,'fixed_ema',c,'online'))
    result.extend((f'{arm}/fixed_ema-online',arm,'fixed_ema',arm,'online') for arm in (w,s,n,c))
    return result


def endpoint_rows(result, epochs, reused):
    j=result['job']; arm=j.get('arm',j['config']['arm']); rows=[]; cells=[]; curves=[]; cell_curves=[]
    base=dict(key=j['key'],dataset=j['dataset'],arm=arm,seed=j['seed'],source_hash=j['source_hash'],
              design_bundle_sha256=result.get('design_bundle_sha256',j.get('design_bundle_sha256')),
              reused=reused,test_accessed=False,available=True,null_reason=None)
    for kind in p.ESTIMATORS:
        for epoch in range(1,51):
            q=epochs[epoch][1] if isinstance(epochs[epoch],tuple) else epochs[epoch]
            v=q['validation'][kind]; h=q['model_hash' if kind=='online' else 'ema_hash']
            cr=dict(base,estimator=kind,epoch=epoch,endpoint='curve',selected_epoch=result['selected_epochs'][kind],
                    state_hash=h,epoch_order_hash=q['epoch_order_hash'],**v)
            curves.append(cr)
            for cell in p.CELLS:
                cell_curves.append(dict({k:v for k,v in cr.items() if k not in q['validation'][kind]},cell=cell,
                    **{name:v[cell+'_'+name] for name in ('auc','logloss','count','positive','negative')}))
        for endpoint in ENDPOINTS:
            epoch=result['selected_epochs'][kind] if endpoint=='selected' else int(endpoint[1:])
            q=epochs[epoch][1] if isinstance(epochs[epoch],tuple) else epochs[epoch]
            v=q['validation'][kind]
            row=dict(base,estimator=kind,endpoint=endpoint,epoch=epoch,selected_epoch=result['selected_epochs'][kind],
                     state_hash=q['model_hash' if kind=='online' else 'ema_hash'],epoch_order_hash=q['epoch_order_hash'],**v)
            rows.append(row)
            for cell in p.CELLS:
                cells.append(dict({k:v for k,v in row.items() if k not in q['validation'][kind]},cell=cell,
                    **{name:v[cell+'_'+name] for name in ('auc','logloss','count','positive','negative')}))
    return rows,cells,curves,cell_curves


def paired_tables(endpoints):
    lookup={(r['dataset'],r['arm'],r['seed'],r['estimator'],r['endpoint']):r for r in endpoints}
    rows=[]; groups=defaultdict(list)
    metrics=sorted({k for r in endpoints for k in r if k.endswith(('_auc','_logloss'))})
    for d in p.DATASETS:
        for label,left,lk,right,rk in comparisons():
            for endpoint in ENDPOINTS:
                for metric in metrics:
                    direction='negative' if metric.endswith('logloss') else 'positive'
                    primary=left==p.ARMS[0] and right=='CLEAN' and lk==rk=='fixed_ema' and endpoint=='selected' and metric=='macro_auc'
                    for seed in p.SEEDS:
                        a=lookup.get((d,left,seed,lk,endpoint)); b=lookup.get((d,right,seed,rk,endpoint))
                        av=None if a is None else a.get(metric); bv=None if b is None else b.get(metric)
                        available=av is not None and bv is not None; delta=av-bv if available else None
                        r=dict(dataset=d,comparison=label,left_arm=left,left_estimator=lk,right_arm=right,right_estimator=rk,
                            endpoint=endpoint,metric=metric,primary_metric=primary,seed=seed,left_key=None if a is None else a['key'],
                            right_key=None if b is None else b['key'],left_epoch=None if a is None else a['epoch'],right_epoch=None if b is None else b['epoch'],
                            left_state_hash=None if a is None else a['state_hash'],right_state_hash=None if b is None else b['state_hash'],
                            left=av,right=bv,delta=delta,available=available,null_reason=None if available else 'not_recorded_in_source',
                            favorable_direction=direction,favorable=None if delta is None else (delta<0 if direction=='negative' else delta>0))
                        rows.append(r); groups[(d,label,lk,rk,endpoint,metric,primary,direction)].append(delta)
    summaries=[]
    for key,values in groups.items():
        d,label,lk,rk,endpoint,metric,primary,direction=key
        summaries.append(dict(dataset=d,comparison=label,left_estimator=lk,right_estimator=rk,endpoint=endpoint,
            metric=metric,primary_metric=primary,favorable_direction=direction,**paired_summary(values,favorable_direction=direction)))
    return rows,summaries


def weights_rows(base,batch):
    ledger=batch['ledger']; context=batch['context']; result=[]
    provenance=dict(base,attempt=batch['attempt'],**context,row_order_hash=batch['row_order_hash'],B=ledger['B'],entries=ledger['entries'],
                    branch=ledger['branch'],mixture_lambda=ledger['mixture_lambda'],m=ledger['bce_mass_scale'],r=ledger['removed_opposite_mass'],
                    actual_total_mass=ledger['total_mass'],actual_bce_mass=ledger['bce_mass'],actual_opposite_mass=ledger['opposite_mass'],
                    actual_task_mass=ledger['task_mass'],completion_task_mass=ledger['completion_task_mass'],diagnose=ledger['diagnose'])
    for cell,v in ledger['cells'].items():
        r=dict(provenance,cell=cell,**v)
        for a in (0,1):
            coefficient=v['completion_coefficient'+str(a)]
            r['q'+str(a)]=None if coefficient is None else 2*ledger['B']*coefficient
        result.append(r)
    return result


def mass_add(acc,ledger,B):
    def add(scope,sid,kind,value):
        a=acc[(scope,sid,kind)]; a['update_sum']+=value; a['row_weighted_sum']+=B*value
        a['batches']+=1; a['rows']+=B
    for name in ('bce_mass_scale','removed_opposite_mass','total_mass','bce_mass','opposite_mass'):
        add('total','ALL',name,ledger[name])
    for name in ('task_mass','completion_task_mass'):
        for task,v in ledger[name].items(): add('task',task,name,v)
    for cell,v in ledger['cells'].items():
        for name in ('weight','opportunity','removed_opposite_mass','completion_bce_mass','scale_bce_mass','bce_mass','opposite_mass'):
            add('cell',cell,name,v[name])


def mass_rows(base,acc):
    return [dict(base,scope=scope,scope_id=sid,mass_kind=kind,**a,update_mean=a['update_sum']/a['batches'],
                 row_weighted_mean=a['row_weighted_sum']/a['rows'],denominator_units='updates; training_rows')
            for (scope,sid,kind),a in acc.items()]


def statistic_add(acc,ledger):
    for cell,c in ledger['cells'].items():
        for statistic in STATISTICS:
            a=acc[(cell,statistic)]; v=c[statistic]
            count=c['n0'] if statistic.endswith('0') else c['n1'] if statistic.endswith('1') else c['n']
            if v is None:
                a['null_batches']+=1
                reason=c['statistics_status'].get(statistic,'not_scheduled')
                a['null_reasons'][reason]+=1
            else:
                a['observed_batches']+=1; a['sum']+=v; a['count_weighted_sum']+=count*v; a['observed_count']+=count


def statistic_rows(base,acc):
    rows=[]
    for (cell,statistic),a in acc.items():
        a=dict(a); a['null_reasons']=dict(a['null_reasons']); n=a['observed_batches']; count=a['observed_count']
        rows.append(dict(base,cell=cell,statistic=statistic,**a,update_mean=None if n==0 else a['sum']/n,
            count_weighted_mean=None if count==0 else a['count_weighted_sum']/count,
            denominator_units='class0_rows' if statistic.endswith('0') else 'class1_rows' if statistic.endswith('1') else 'observed_cell_rows',
            null_reason=None if n else (';'.join(sorted(a['null_reasons'])) or 'not_scheduled')))
    return rows


def diagnostic_rows(base,batch):
    diag=batch['diagnostics']
    if diag is None: return []
    context=batch['context']; rows=[]
    common=dict(base,host_arm=diag['host_arm'],epoch=context['epoch'],batch=context['batch'],global_step=context['global_step'],
        row_order_hash=diag['row_order_hash'],**{n:diag[n] for n in ('pre_model_hash','pre_optimizer_hash','pre_rng_hash','pre_loader_hash')},
        extra_autograd_sweeps=diag['extra_autograd_sweeps'],virtual_adam_steps=diag['virtual_adam_steps'],
        counterfactual_at_host_state=diag['counterfactual_at_host_state'],actual_virtual_exact=diag['actual_virtual_exact'],
        actual_post=diag['actual_post'],virtual_post=diag['virtual_post'],objective_losses=diag['objective_losses'],
        m=diag['bce_mass_scale'],r=diag['removed_opposite_mass'])
    for group,values in diag['parameter_groups'].items():
        group_base=dict(common,parameter_group=group,coordinates=values['coordinates'],parameter_names=values['parameter_names'])
        for objective,metrics in values['objectives'].items():
            activity={k:v for k,v in metrics.items() if k.endswith(('_parameters','_coordinates'))}
            for name,value in metrics.items():
                if isinstance(value,dict):
                    for sub,x in value.items():
                        rows.append(dict(group_base,objective_or_pair=objective,metric=name+'.'+sub,value=x,
                            defined=x is not None,null_reason=None if x is not None else 'uninitialized',activity_counts=activity))
                else:
                    rows.append(dict(group_base,objective_or_pair=objective,metric=name,value=value,
                        defined=value is not None,null_reason=None if value is not None else 'zero_norm',activity_counts=activity))
        for pair,metrics in values['pairs'].items():
            for name,value in metrics.items():
                if isinstance(value,dict):
                    for sub,x in value.items():
                        if sub=='null_reason': continue
                        rows.append(dict(group_base,objective_or_pair=pair,metric=name+'.'+sub,value=x,
                            defined=x is not None,null_reason=value.get('null_reason') if x is None else None,activity_counts={}))
                else:
                    rows.append(dict(group_base,objective_or_pair=pair,metric=name,value=value,
                        defined=value is not None,null_reason=None,activity_counts={}))
    return rows


def seed_means(rows,group_fields,value_field='value'):
    groups=defaultdict(list)
    for r in rows:
        if r.get(value_field) is None or (isinstance(r.get(value_field),(int,float)) and not isinstance(r[value_field],bool)):
            groups[tuple(r[k] for k in group_fields)].append(r)
    output=[]
    for key,rs in groups.items():
        by_seed=defaultdict(list)
        for r in rs:
            if r[value_field] is not None: by_seed[r['seed']].append(r[value_field])
        values=[statistics.mean(v) for v in by_seed.values()]
        output.append(dict(zip(group_fields,key),seed_aggregation='mean_within_seed_then_across_seeds',valid_seed_count=len(values),
            mean=statistics.mean(values) if values else None,sd=statistics.stdev(values) if len(values)>1 else None,
            min=min(values) if values else None,max=max(values) if values else None,
            available=len(values)==5,null_reason=None if len(values)==5 else
                ('incomplete_five_seed_coverage' if values else ';'.join(sorted({r.get('null_reason') or 'undefined' for r in rs})))))
    return output


def _jsonl(path):
    if not path.exists(): return []
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def diagnostic_cost_rows(base,batches,source_path):
    rows=[]
    for batch in batches:
        value=batch.get('value',batch)
        if 'source_loss' in batch: continue
        context=batch['context']; seconds=value.get('diagnostic_seconds')
        rows.append(dict(base,key=batch.get('key',base.get('key')),attempt=batch['attempt'],cost_kind='diagnostic',
            context=context,call=batch.get('call'),seconds=seconds,seconds_semantics='returned_core_diagnostic_seconds',
            null_reason=None if seconds is not None else 'not_returned',source_path=str(source_path),
            extra_autograd_sweeps=value.get('extra_autograd_sweeps'),virtual_adam_steps=value.get('virtual_adam_steps'),
            source_hash=batch.get('source_hash',base.get('source_hash')),design_bundle_sha256=p.DESIGN_HASH,test_accessed=False))
    return rows


def physical_cost_rows(a):
    rows=_jsonl(a/'costs.jsonl')
    paths=[*sorted((a/'runs').glob('*/attempt*/batches_epoch_*.jsonl.gz')),
           *sorted((a/'smoke').glob('*/attempt*/batches.jsonl'))]
    for path in paths:
        batches=[]; partial=False; opener=gzip.open if path.suffix=='.gz' else open
        try:
            with opener(path,'rt') as stream:
                for line in stream:
                    try: batches.append(json.loads(line))
                    except json.JSONDecodeError: partial=True; break
        except (EOFError,OSError): partial=True
        purpose='formal' if path.is_relative_to(a/'runs') else 'engineering_smoke'
        rows+=diagnostic_cost_rows(dict(purpose=purpose),batches,path)
        if partial:
            rows.append(dict(cost_kind='diagnostic',purpose=purpose,source_path=str(path),seconds=None,
                null_reason='incomplete_physical_batch_log',test_accessed=False))
    for path in sorted((a/'transactions').glob('*/attempt*/epoch_*.json')):
        transaction=p.read_json(path)
        rows.append(dict(transaction,cost_kind='physical_snapshot_serialization',transaction_file=str(path.relative_to(a)),
            seconds=transaction['serialization']['seconds'],seconds_semantics='actual_BytesIO_preflight'))
    return rows


def export(attempt=p.ATTEMPT):
    a=p.new_root(attempt); manifest=p.validate_attempt(a,True); p.load_reference_index()
    entries=[]
    for entry in p.load_reference_index()['entries']:
        r=p.receipt_json(entry['result']); entries.append((r,p.epoch_files(a,r),True))
    for d in p.DATASETS:
        for seed in p.SEEDS:
            for arm in p.ARMS:
                j=p.job(d,arm,seed,manifest['source_hash']); r=p.read_json(a/'runs'/j['key']/'result.json')
                entries.append((r,p.epoch_files(a,r),False))
    endpoints=[]; endpoint_cells=[]; curves=[]; cell_curves=[]; selections=[]
    weights=[]; masses=[]; stats=[]; diagnostics=[]; availability=[]; counters=[]; costs=[]
    for result,epochs,reused in entries:
        p.verify_selection(result,epochs); j=result['job']; arm=j.get('arm',j['config']['arm'])
        base=dict(key=j['key'],dataset=j['dataset'],arm=arm,seed=j['seed'],source_hash=j['source_hash'],
                  design_bundle_sha256=result.get('design_bundle_sha256',j.get('design_bundle_sha256')),test_accessed=False)
        e,c,cv,cc=endpoint_rows(result,epochs,reused); endpoints+=e; endpoint_cells+=c; curves+=cv; cell_curves+=cc
        for kind in p.ESTIMATORS:
            v=result['selected_validation'][kind]
            selections.append(dict(base,estimator=kind,selected_epoch=result['selected_epochs'][kind],selected_hash=result['selected_hashes'][kind],
                selected_macro_auc=v['macro_auc'],selected_macro_logloss=v['macro_logloss'],selection_hash=p.digest(p.SELECTION),
                first_strict_max_verified=True,reused=reused))
        for epoch,(path,record) in epochs.items():
            observed=not reused and epoch in p.DIAGNOSTIC_EPOCHS
            availability.append(dict(base,epoch=epoch,metric_family='common_state_counterfactual_gradients_and_Adam',available=observed,
                null_reason=None if observed else ('not_recorded_in_source' if reused else 'not_scheduled'),
                source_path=str(path),source_sha256=p.sha_file(path)))
            for field in ('rng_hash','loader_hash','counts','continuation_hash','test_accessed'):
                availability.append(dict(base,epoch=epoch,metric_family='recorded_epoch_'+field,available=field in record,
                    null_reason=None if field in record else 'not_recorded_in_source',
                    source_path=str(path),source_sha256=p.sha_file(path)))
            if reused: continue
            ack=p.read_json(a/'acks'/j['key']/f'epoch_{epoch:02d}.json')
            costs.append(dict(base,epoch=epoch,cost_kind='snapshot_transfer',attempt=ack['attempt'],**{k:ack.get(k) for k in
                ('serialization','commit_status_get_seconds','events','continuation_hash')}))
            costs.append(dict(base,epoch=epoch,cost_kind='logical_epoch',seconds=record['seconds'],validation_cost=record['validation_cost'],snapshot_costs=record['costs']))
            counters.append(dict(base,clock_kind=record['clock_kind'],epoch=epoch,global_step=record['global_step'],counts=record['counts'],
                **record['counts'],**record['operations'],branch_counts=record['branch_counts'],private_steps=record['adam_step_counters'],activity=record['last_parameter_activity'],
                mixture_lambda=.25,next_epoch=record['next_epoch'],scientific_epochs_completed=record['scientific_epochs_completed']))
            p.require(p.sha_file(a/record['batch_log'])==record['batch_log_sha256'],'Committed batch bytes differ')
            batches=_jsonl(a/record['batch_log']); p.require(len(batches)==record['batch_count']==p.STEPS[j['dataset']],'Batch coverage differs')
            acc=defaultdict(lambda:dict(update_sum=0.,row_weighted_sum=0.,batches=0,rows=0))
            sa=defaultdict(lambda:dict(observed_batches=0,null_batches=0,sum=0.,count_weighted_sum=0.,observed_count=0,null_reasons=defaultdict(int)))
            for batch in batches:
                weights+=weights_rows(base,batch); mass_add(acc,batch['ledger'],batch['B']); statistic_add(sa,batch['ledger'])
                diagnostics+=diagnostic_rows(base,batch)
            masses+=mass_rows(dict(base,epoch=epoch),acc); stats+=statistic_rows(dict(base,epoch=epoch),sa)
    paired,summary=paired_tables(endpoints)
    means=[]
    for aggregation in ('update_mean','count_weighted_mean'):
        rows=seed_means(stats,('dataset','arm','epoch','cell','statistic'),aggregation)
        means.extend(dict(r,aggregation=aggregation,n_seed=5) for r in rows)
    diag_means=seed_means(diagnostics,('dataset','host_arm','epoch','parameter_group','objective_or_pair','metric'))
    attempts=_jsonl(a/'physical_attempts.jsonl'); physical_costs=physical_cost_rows(a)
    costs+=physical_costs
    smoke_costs=[r for r in physical_costs if r.get('purpose')=='engineering_smoke']
    signoff=p.read_json(a/'ENGINEERING_SIGNOFF.json')
    for er in signoff['cases']:
        case=p.receipt_json(er); smoke_costs.append(case)
    # Include every physical failed/replayed/uncommitted record, independent of logical ACK selection.
    for path in sorted((a/'smoke').glob('*/attempt*/case.json')):
        if not any(x.get('key')==p.read_json(path)['key'] and 'verified_plan_counts' in x for x in smoke_costs): smoke_costs.append(p.read_json(path))
    counts=dict(endpoints=len(endpoints),endpoint_cells=len(endpoint_cells),curves=len(curves),cell_curves=len(cell_curves),weights=len(weights),phase_counters=len(counters))
    p.require(counts==dict(endpoints=400,endpoint_cells=2400,curves=4000,cell_curves=24000,weights=207000,phase_counters=1000),'Formal extraction coverage differs')
    last=[r for r in counters if r['epoch']==50]
    logical={name:sum(r[name] for r in last) for name in p.OPERATIONS}
    expected=dict(training_rows=1403149500,task_entries=2806299000,ordinary_forwards=34500,scalar_backwards=34500,
        live_adam_steps=34500,live_ema_steps=34500,diagnostic_points=80,extra_autograd_sweeps=240,virtual_adam_steps=320,
        probe_forwards=0,view_forwards=0,algorithm_rng_draws=0)
    p.require(logical==expected,'Formal operation budget differs')
    audit=dict(source_hash=manifest['source_hash'],design_bundle_sha256=p.DESIGN_HASH,controls_index_hash=p.CONTROLS_HASH,
        counts=counts,logical_operations=logical,engineering_signoff=dict(path=str(a/'ENGINEERING_SIGNOFF.json'),sha256=p.sha_file(a/'ENGINEERING_SIGNOFF.json')),
        engineering_cases=signoff['cases'],measured_profiles=signoff['profiles'],
        admissions=_jsonl(a/'admission.jsonl'),allocation_acks=_jsonl(a/'allocation_acks.jsonl'),
        physical_attempts=attempts,committed_epochs=1000,verified_control_files=1020,owner=p.owner_binding(a),
        parent_storage_acceptance=None if not (a/'PARENT_STORAGE_ACCEPTANCE.json').exists() else
            dict(path=str(a/'PARENT_STORAGE_ACCEPTANCE.json'),sha256=p.sha_file(a/'PARENT_STORAGE_ACCEPTANCE.json')),
        unknown_partial_calls=sum(bool(r.get('unknown_partial_call')) for r in attempts),test_accessed=False)
    output={}
    families={'ENDPOINTS':endpoints,'ENDPOINT_CELLS':endpoint_cells,'CURVES':curves,'CELL_CURVES':cell_curves,
        'ALL_PAIRED_SEEDS':paired,'ALL_PAIRED_SUMMARY':summary,'WEIGHTS':weights,'MASS_SUMMARIES':masses,
        'STATISTICS':stats,'STATISTIC_MEANS':means,'COMMON_STATE_DIAGNOSTICS':diagnostics,'DIAGNOSTIC_MEANS':diag_means,
        'DIAGNOSTIC_AVAILABILITY':availability,'SELECTION_STRUCTURE':selections,'PHASE_COUNTERS':counters,
        'PHYSICAL_ATTEMPTS':attempts,'COSTS':costs,'SMOKE_COSTS':smoke_costs}
    for name,rows in families.items(): output[name]=csv_write(a/'tables'/f'{name}.csv',rows)
    p.write_json(a/'tables/RESOURCE_EXECUTION_AUDIT.json',audit)
    p.write_json(a/'tables/TABLE_RECEIPT.json',dict(source_hash=manifest['source_hash'],design_bundle_sha256=p.DESIGN_HASH,files=output,test_accessed=False))
    return dict(status='EXPORTED',files=output,test_accessed=False)


def finite_checks():
    assert len(comparisons())==15
    summary=paired_summary([1.,2.,3.,4.,5.],favorable_direction='positive')
    assert summary['n']==5 and summary['mean']==3. and summary['positive']==5
    assert paired_summary([None,1.],favorable_direction='negative')['n']==1
    assert paired_summary([-1.,-1.,-1.,-1.,-1.],favorable_direction='negative')['favorable']==5
    acc=defaultdict(lambda:dict(update_sum=0.,row_weighted_sum=0.,batches=0,rows=0))
    ledger=dict(bce_mass_scale=.9,removed_opposite_mass=.1,total_mass=.9,bce_mass=.9,opposite_mass=0.,
                task_mass={'0':.45,'1':.45},completion_task_mass={'0':.4,'1':.5},cells={})
    mass_add(acc,ledger,40960); mass_add(acc,{**ledger,'total_mass':1.},21927)
    rows=mass_rows({},acc); row=next(r for r in rows if r['mass_kind']=='total_mass')
    assert row['update_mean']==.95 and row['row_weighted_mean']==(.9*40960+21927)/(40960+21927)
    sa=defaultdict(lambda:dict(observed_batches=0,null_batches=0,sum=0.,count_weighted_sum=0.,observed_count=0,null_reasons=defaultdict(int)))
    cell=dict(n=10,n0=3,n1=7,**{s:None for s in STATISTICS},statistics_status={s:'not_scheduled' for s in STATISTICS})
    statistic_add(sa,dict(cells={'d0t0':cell})); statistic_add(sa,dict(cells={'d0t0':dict(cell,mu0=.2)}))
    sr=statistic_rows({},sa); mu=next(r for r in sr if r['statistic']=='mu0')
    assert mu['observed_count']==3 and mu['null_batches']==1 and mu['update_mean']==.2
    assert next(r for r in sr if r['statistic']=='var0')['count_weighted_mean'] is None
    dm=seed_means([dict(seed=2022,value=1.,group='a'),dict(seed=2022,value=3.,group='a'),dict(seed=2023,value=10.,group='a')],('group',))
    assert dm[0]['mean']==6. and dm[0]['valid_seed_count']==2
    missing=seed_means([dict(seed=2022,value=None,group='a',null_reason='zero_norm')],('group',))
    assert missing[0]['mean'] is None and missing[0]['valid_seed_count']==0 and missing[0]['null_reason']=='zero_norm'
    cost=diagnostic_cost_rows(dict(key='fixture',source_hash='fixture'),[
        dict(attempt=1,context=dict(epoch=1,batch=1),diagnostic_seconds=.25,extra_autograd_sweeps=3,virtual_adam_steps=4),
        dict(attempt=2,context=dict(epoch=1,batch=1),diagnostic_seconds=.5,extra_autograd_sweeps=3,virtual_adam_steps=4),
        dict(attempt=3,context=dict(epoch=1,batch=1),diagnostic_seconds=None)],'fixture.jsonl')
    assert [r['seconds'] for r in cost]==[.25,.5,None] and [r['attempt'] for r in cost]==[1,2,3]
    assert cost[-1]['null_reason']=='not_returned'
    return dict(status='PASS',groups=5,comparisons=15,physical_diagnostic_cost_fixture=True,test_accessed=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--attempt',default=str(p.ATTEMPT)); parser.add_argument('--root',default=str(p.ROOT))
    args=parser.parse_args(); p.require(Path(args.root)==p.ROOT,'ROOT differs')
    print(json.dumps(export(args.attempt),allow_nan=False))


if __name__=='__main__': main()
