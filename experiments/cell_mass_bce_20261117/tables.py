"""Receipt-bound numeric extraction. No efficacy decisions or winner selection."""
import argparse
import csv
import gzip
import json
import math
import statistics
import tarfile
import hashlib
import shutil
import tempfile
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
    sign=1 if favorable_direction=='positive' else -1 if favorable_direction=='negative' else None
    return dict(available=n==5,n=n,mean=mean,sd=sd,ci95_low=None if half is None else mean-half,
                ci95_high=None if half is None else mean+half,positive=sum(x>0 for x in valid),
                negative=sum(x<0 for x in valid),zero=sum(x==0 for x in valid),
                favorable=None if sign is None else sum(sign*x>0 for x in valid),unfavorable=None if sign is None else sum(sign*x<0 for x in valid),
                direction_null_reason='no_favorable_direction' if sign is None else None,
                null_reason=None if n==5 else 'incomplete_five_seed_coverage')


def comparisons():
    arms=dict(U=p.ARMS[0],C='CLEAN',W='COMPLETION-BCE-025',S='MASS-SCALE-025')
    return [(f"{r['left_arm']}_{r['left_estimator']}-{r['right_arm']}_{r['right_estimator']}",
        arms[r['left_arm']],r['left_estimator'],arms[r['right_arm']],r['right_estimator'])
        for r in p.research_manifest()['analysis']['contrasts']]


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
    metrics=p.research_manifest()['analysis']['metric_names']
    p.require(len(metrics)==34 and metrics[-1]=='epoch','Metric34 contract differs')
    for d in p.DATASETS:
        for label,left,lk,right,rk in comparisons():
            for endpoint in ENDPOINTS:
                for metric in metrics:
                    direction=None if metric=='epoch' else 'negative' if metric.endswith('logloss') else 'positive'
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
                            favorable_direction=direction,direction_null_reason='no_favorable_direction' if direction is None else None,
                            favorable=None if delta is None or direction is None else (delta<0 if direction=='negative' else delta>0))
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
        result.append(r)
    return result


def mass_add(acc,ledger,B):
    def add(scope,sid,kind,value):
        a=acc[(scope,sid,kind)]; a['update_sum']+=value; a['row_weighted_sum']+=B*value
        a['batches']+=1; a['rows']+=B
    for name in ('bce_mass_scale','removed_opposite_mass','total_mass','bce_mass','opposite_mass'):
        add('total','ALL',name,ledger[name])
    for name in ('task_mass','completion_task_mass','cell_mass_task_mass','scale_task_mass','natural_task_mass'):
        for task,v in ledger[name].items(): add('task',task,name,v)
    for cell,v in ledger['cells'].items():
        for name in ('weight','opportunity','removed_opposite_mass','completion_bce_mass','cell_mass_bce_mass','natural_bce_mass','scale_bce_mass','bce_mass','opposite_mass'):
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
    reference=batch.get('diagnostics')
    if reference is None: return []
    diag=p.read_json(p.ATTEMPT/'diagnostics'/f"{reference['point_id']}.json")
    context=batch['context']; rows=[]
    common=dict(base,point_id=diag['point_id'],parameter_dictionary_id=diag['parameter_dictionary_id'],host_arm=diag['host_arm'],
        epoch=context['epoch'],batch=context['batch'],global_step=context['global_step'])
    for group,values in diag['parameter_groups'].items():
        for category in ('objectives','pairs'):
            for objective,metrics in values[category].items():
                for name,value in metrics.items():
                    if isinstance(value,dict):
                        for sub,x in value.items():
                            if sub=='null_reason': continue
                            rows.append(dict(common,parameter_group=group,objective_or_pair=objective,metric=name+'.'+sub,
                                value=x,defined=x is not None,null_reason=value.get('null_reason','uninitialized') if x is None else None))
                    else:
                        rows.append(dict(common,parameter_group=group,objective_or_pair=objective,metric=name,value=value,
                            defined=value is not None,null_reason='zero_norm' if value is None else None))
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
        for batch in batches:
            value=batch.get('value',batch)
            rows.append(dict(cost_kind='returned_train_call',purpose=purpose,key=batch['key'],attempt=batch['attempt'],
                seconds=batch.get('seconds',value.get('seconds')),B=batch['B'],entries=2*batch['B'],context=batch['context'],
                diagnostic_points=int(value.get('extra_autograd_sweeps',0)==3),
                extra_autograd_sweeps=value.get('extra_autograd_sweeps',0),virtual_adam_steps=value.get('virtual_adam_steps',0),
                source_path=str(path),source_hash=batch['source_hash'],test_accessed=False))
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
    p.require(len(paired)==28900 and len(summary)==5780,'17family/34metric counts differ')
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
    p.require(counts==dict(endpoints=400,endpoint_cells=2400,curves=4000,cell_curves=24000,weights=103500,phase_counters=500),'Formal extraction coverage differs')
    last=[r for r in counters if r['epoch']==50]
    logical={name:sum(r[name] for r in last) for name in p.OPERATIONS}
    expected=dict(training_rows=701574750,task_entries=1403149500,ordinary_forwards=17250,scalar_backwards=17250,
        live_adam_steps=17250,live_ema_steps=17250,diagnostic_points=40,extra_autograd_sweeps=120,virtual_adam_steps=160,
        probe_forwards=0,view_forwards=0,algorithm_rng_draws=0)
    p.require(logical==expected,'Formal operation budget differs')
    audit=dict(source_hash=manifest['source_hash'],design_bundle_sha256=p.DESIGN_HASH,controls_index_hash=p.controls_hash(),
        counts=counts,logical_operations=logical,engineering_signoff=dict(path=str(a/'ENGINEERING_SIGNOFF.json'),sha256=p.sha_file(a/'ENGINEERING_SIGNOFF.json')),
        engineering_cases=signoff['cases'],measured_profiles=signoff['profiles'],
        raw_journals={n:dict(path=str(a/n),sha256=p.sha_file(a/n),records=len(_jsonl(a/n))) for n in
            ('admission.jsonl','allocation_acks.jsonl','physical_attempts.jsonl','profiles/witness_registry.jsonl')},committed_epochs=500,verified_control_files=1530,owner=p.owner_binding(a),
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
    p.write_json(a/'tables/schema.json',dict(keys=dict(trajectories='key',parameters='parameter_dictionary_id, ordered index',
        points='point_id',diagnostics='point_id,parameter_group,objective_or_pair,metric'),
        joins=dict(parameter_dictionary='dictionaries/parameters_<parameter_dictionary_id>.json',diagnostic_point='diagnostics/<point_id>.json',
            epochs='acks/<key>/epoch_<epoch>.json -> epoch_file -> batch_log'),
        units=dict(AUC='probability_rank',logloss='ordinary unscaled source probability BCE',paired_unit='seed',counts='integer operations'),
        formulas=dict(sd='sample ddof1',ci95='mean +/- 2.7764451051977987 * sd / sqrt(5)',mass='update and row weighted'),
        null_reasons=['empty_cell','absent_class','not_scheduled','not_computed_delegate','requires_both_classes','not_recorded_in_source',
            'zero_norm','uninitialized','no_favorable_direction','incomplete_five_seed_coverage'],
        logical_counts=counts,paired_seed_rows=28900,paired_summary_rows=5780,test_accessed=False))
    p.write_json(a/'tables/TABLE_RECEIPT.json',dict(source_hash=manifest['source_hash'],design_bundle_sha256=p.DESIGN_HASH,files=output,test_accessed=False))
    return dict(status='EXPORTED',files=output,test_accessed=False)



PRIVATE_NAMES = {'guardian.json','STORAGE_OWNER.json'}
OPAQUE_SUFFIXES = {'.pt','.pth','.pkl','.pickle','.ckpt','.npy','.npz','.bin','.safetensors'}
PRIVATE_KEYS = {'address','authkey','password','access_token','secret','credentials'}
STATE_KEYS = {'selected_models','model','ema','optimizer','rng','loader'}


def member_name(name):
    from pathlib import PurePosixPath
    path=PurePosixPath(name)
    p.require(bool(name) and not path.is_absolute() and '..' not in path.parts and str(path)==name,'Archive traversal/noncanonical member')
    p.require(path.name not in PRIVATE_NAMES and not any(s in OPAQUE_SUFFIXES for s in path.suffixes),'Private/state archive member')
    return str(path)


def numeric_document(value):
    p.json_safe(value)
    if isinstance(value,dict):
        p.require(not PRIVATE_KEYS.intersection(value),'Private credential field in public record')
        # State names also label descriptive/hash provenance; reject their payload containers.
        p.require(all(not isinstance(value[k],(dict,list,tuple)) for k in STATE_KEYS.intersection(value)),
                  'Opaque training state field in public record')
        for v in value.values(): numeric_document(v)
    elif isinstance(value,(tuple,list)):
        for v in value: numeric_document(v)


def checked_member(path,name,category,*,root,expected_sha=None):
    path=p.contained(path,root); name=member_name(name)
    p.require(path.is_file() and not path.is_symlink(),'Only regular mapped archive files')
    p.require(path.name not in PRIVATE_NAMES,'Private credentials cannot enter archive')
    suffixes=path.suffixes
    p.require(not any(s in OPAQUE_SUFFIXES for s in suffixes),'Opaque state cannot enter archive')
    source=category=='source'
    if not source:
        if path.suffix=='.json': numeric_document(p.read_json(path))
        elif path.suffix=='.jsonl' or path.name.endswith('.jsonl.gz'):
            for row in _jsonl(path): numeric_document(row)
        elif path.suffix=='.csv':
            with path.open(newline='') as stream:
                for row in csv.DictReader(stream): p.json_safe(row)
        else: p.require(path.suffix=='.md' and category in ('design','receipt'),'Unstructured/non-numeric raw log')
    actual=p.sha_file(path)
    p.require(expected_sha is None or actual==expected_sha,'Archive source byte hash differs')
    return dict(member=name,source_path=str(path),category=category,bytes=path.stat().st_size,sha256=actual)


def archive_members(a):
    source=p.read_json(a/'source_receipt.json'); design=p.design_bundle(); entries=[]
    # New physical tree only. Historical artifacts are individually enumerated below.
    for path in sorted(a.rglob('*')):
        p.require(not path.is_symlink(),'Symlink in physical evidence tree')
        if not path.is_file() or path.name in PRIVATE_NAMES or path.is_relative_to(a/'source_snapshot'): continue
        if path.name in ('RAW_RUNS_MANIFEST.json','RAW_RUNS_PART_INDEX.json','PACK_RECEIPT.json') or path.suffix in ('.log',): continue
        if path.name=='PAUSE_TRAINING': continue
        if path.suffix not in ('.json','.jsonl','.gz','.csv'): continue
        entries.append(checked_member(path,'attempt1/'+str(path.relative_to(a)),'raw',root=a))
    for f in source['files']:
        path=a/'source_snapshot'/f['path']
        entries.append(checked_member(path,'attempt1/source_snapshot/'+f['path'],'source',root=a,expected_sha=f['sha256']))
    for e in p.load_reference_index()['entries']:
        prefix=f"reused_controls/{e['symbol']}/{e['dataset']}/seed{e['seed']}/{e['key']}"
        for receipt,name in [(e['result'],'result.json')]+[(r,f"epoch_{r['epoch']:02d}.json") for r in e['epochs']]:
            path=Path(receipt['path'])
            # Every old file is explicitly accepted in the design control list.
            entries.append(checked_member(path,prefix+'/'+name,'control',root=path.parent,expected_sha=receipt['sha256']))
    for name,path in design['logical_name_to_absolute_path'].items():
        entries.append(checked_member(Path(path),'design/'+name,'design',root=Path(path).parent,expected_sha=design['files'][name]))
    manifest=p.research_manifest()
    c04=manifest['c04']['publication_receipt']
    old_receipts=[(Path(c04['absolute_path']),'C04_PUBLICATION_RECEIPT.json',c04['sha256']),
        (p.A4/'SELECTED_RAM_AVAILABILITY_UPDATE.json','C04_SELECTED_RAM_AVAILABILITY_UPDATE.json',None),
        (p.A4/'attempt2/RAW_RUNS_MANIFEST.json','C04_RAW_RUNS_MANIFEST.json',None)]
    for path,name,sha in old_receipts:
        entries.append(checked_member(path,'receipts/'+name,'receipt',root=path.parent,expected_sha=sha))
    # Parent supplies public stage/report receipts explicitly, with byte hashes.
    inputs=p.read_json(p.LOGICAL/'PACK_INPUTS.json')
    p.require(inputs['source_hash']==source['sha256'] and inputs['design_bundle_sha256']==p.DESIGN_HASH,'Pack inputs identity differs')
    allowed_roots=(p.A5,p.LOGICAL,a)
    for r in inputs['receipts']:
        path=Path(r['path']); root=next((root for root in allowed_roots if path.is_relative_to(root)),None)
        p.require(root is not None,'Receipt outside authorized C05 roots')
        entries.append(checked_member(path,'receipts/'+member_name(r['logical_name']),'receipt',root=root,expected_sha=r['sha256']))
    p.require(len({e['member'] for e in entries})==len(entries),'Duplicate archive member')
    p.require(len([e for e in entries if e['category']=='control'])==1530 and
              len([e for e in entries if e['category']=='source'])==55,'Archive control/source closure incomplete')
    return sorted(entries,key=lambda e:e['member'])


def write_archive(path,members):
    with tarfile.open(path,'w:gz',compresslevel=3) as archive:
        for row in members:
            src=Path(row['source_path']); p.require(p.sha_file(src)==row['sha256'],'Archive input changed during packing')
            archive.add(src,arcname=row['member'],recursive=False)
    with tarfile.open(path,'r:gz') as archive:
        actual=archive.getmembers(); p.require([r.name for r in actual]==[r['member'] for r in members],'Archive mapping differs')
        for header,row in zip(actual,members):
            p.require(header.isfile() and header.size==row['bytes'],'Archive member type/length differs')
            with archive.extractfile(header) as stream:
                digest=hashlib.sha256()
                for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
            p.require(digest.hexdigest()==row['sha256'],'Archive member content differs')


def pack(attempt=p.ATTEMPT):
    a=p.new_root(attempt); manifest=p.validate_attempt(a,True)
    acceptance=p.read_json(a/'PARENT_STORAGE_ACCEPTANCE.json')
    p.require(acceptance['parent_accepted'] is True and acceptance['source_hash']==manifest['source_hash'] and
              acceptance['design_bundle_sha256']==p.DESIGN_HASH and acceptance['test_accessed'] is False,'Parent storage acceptance required before pack')
    p.require(acceptance['owner']['endpoint_hash']==p.owner_binding(a)['endpoint_hash'],'Pack owner binding differs')
    p.require(p.read_json(a/'tables/TABLE_RECEIPT.json')['source_hash']==manifest['source_hash'],'Raw export required before pack')
    p.disk_observation(a,phase='before_pack'); members=archive_members(a)
    p.STAGING.mkdir(parents=True,exist_ok=True); archive=p.staging_path(p.STAGING/'RAW_RUNS.tar.gz')
    need=sum(r['bytes'] for r in members)
    p.require(shutil.disk_usage(p.STAGING).free>=2*need+256*1024**2,'Staging capacity below measured raw/archive work')
    write_archive(archive,members); size=archive.stat().st_size
    required=3*size+256*1024**2+1024**3
    p.require(shutil.disk_usage(p.ROOT).free>=required,'Workspace staging lacks measured 3A+256MiB+1GiB headroom')
    parts=[]
    with archive.open('rb') as stream:
        for i in range((size+8388607)//8388608):
            part=p.staging_path(p.STAGING/f'RAW_RUNS.tar.gz.part-{i:02d}'); part.write_bytes(stream.read(8388608))
            parts.append(dict(path=part.name,bytes=part.stat().st_size,sha256=p.sha_file(part),logical_path=str(p.LOGICAL/part.name)))
    receipt=dict(archive=archive.name,sha256=p.sha_file(archive),bytes=size,member_count=len(members),uncompressed_bytes=need,
        parts=parts,members=members,source_hash=manifest['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
        controls_index_hash=p.controls_hash(),parent_storage_acceptance=dict(path=str(a/'PARENT_STORAGE_ACCEPTANCE.json'),sha256=p.sha_file(a/'PARENT_STORAGE_ACCEPTANCE.json')),
        tensor_files=0,credential_files_excluded=sorted(PRIVATE_NAMES),test_accessed=False)
    index=dict(archive_sha256=receipt['sha256'],archive_bytes=size,parts=parts,test_accessed=False)
    for path,value in ((p.STAGING/'RAW_RUNS_MANIFEST.json',receipt),(p.STAGING/'RAW_RUNS_PART_INDEX.json',index)):
        p.staging_path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    p.metadata_json(p.LOGICAL/'RAW_RUNS_MANIFEST.json',receipt); p.metadata_json(p.LOGICAL/'RAW_RUNS_PART_INDEX.json',index)
    for r in parts:
        dst=p.logical_path(p.LOGICAL/r['path']); shutil.copyfile(p.STAGING/r['path'],dst)
        p.require(p.sha_file(dst)==r['sha256'],'Published part staging differs')
    return dict(status='PACKED',archive_sha256=receipt['sha256'],archive_bytes=size,parts=len(parts),test_accessed=False)


def finite_checks():
    assert len(comparisons())==17
    summary=paired_summary([1.,2.,3.,4.,5.],favorable_direction='positive')
    assert summary['n']==5 and summary['mean']==3. and summary['sd']==math.sqrt(2.5)
    assert math.isclose(summary['ci95_high']-3.,2.7764451051977987/math.sqrt(2))
    epoch=paired_summary([1.,-1.,0.,2.,-2.],favorable_direction=None)
    assert epoch['positive']==2 and epoch['negative']==2 and epoch['zero']==1 and epoch['favorable'] is None
    assert epoch['direction_null_reason']=='no_favorable_direction'
    assert paired_summary([None,1.],favorable_direction='negative')['available'] is False
    # Explicit complete synthetic endpoint matrix verifies 17*2*5*34*5 arithmetic.
    metrics=p.research_manifest()['analysis']['metric_names']; endpoints=[]
    for d in p.DATASETS:
        for arm in (p.ARMS[0],*p.CONTROL_ARMS):
            for seed in p.SEEDS:
                for kind in p.ESTIMATORS:
                    for endpoint in ENDPOINTS:
                        endpoints.append(dict({m:float(seed-2021) for m in metrics},dataset=d,arm=arm,seed=seed,estimator=kind,
                            endpoint=endpoint,key=p.digest([d,arm,seed]),state_hash='fixture',epoch=seed-2021))
    paired,means=paired_tables(endpoints)
    assert len(endpoints)==400 and len(paired)==28900 and len(means)==5780
    assert all(r['favorable_direction'] is None and r['favorable'] is None for r in paired if r['metric']=='epoch')
    acc=defaultdict(lambda:dict(update_sum=0.,row_weighted_sum=0.,batches=0,rows=0))
    ledger=dict(bce_mass_scale=.9,removed_opposite_mass=.1,total_mass=.9,bce_mass=.9,opposite_mass=0.,
        **{n:{'0':.45,'1':.45} for n in ('task_mass','completion_task_mass','cell_mass_task_mass','scale_task_mass','natural_task_mass')},cells={})
    mass_add(acc,ledger,40960); mass_add(acc,dict(ledger,total_mass=1.),21927)
    row=next(r for r in mass_rows({},acc) if r['mass_kind']=='total_mass')
    assert row['update_mean']==.95 and row['row_weighted_mean']==(.9*40960+21927)/(40960+21927)
    # C05-R1: real design/engineering provenance belongs in the numeric archive.
    from .runner import observation
    import io
    model=p.torch.nn.Linear(2,2); ema=p.torch.nn.Linear(2,2)
    opt=p.optimizer(model,dict(lr=.01,weight_decay=1e-4))
    loader=p.torch.Generator(device='cpu').manual_seed(2022)
    observed=observation(model,ema,opt,loader,'cpu')
    names=[n for n,_ in model.named_parameters()]
    observed['activity']=[observed['activity'][n] for n in names]
    observed['parameter_dictionary_id']=p.digest([dict(name=n,shape=list(x.shape),dtype=str(x.dtype),numel=x.numel())
        for n,x in model.named_parameters()])
    assert all(isinstance(observed[k],str) and len(observed[k])==64 for k in ('model','ema','optimizer','rng','loader'))
    manifest=checked_member(p.P5/'EXPERIMENT_MANIFEST.json','design/EXPERIMENT_MANIFEST.json','design',root=p.P5,
        expected_sha='42320f2842f04f0d2d49006108db52e0927c9fe392631df2c96f423440fcf27e')
    # Only numeric/hash records reach this isolated /tmp fixture; state examples stay in RAM.
    with tempfile.TemporaryDirectory(prefix='c05-lovelace-map-') as directory:
        root=Path(directory); path=root/'numeric.json'; path.write_text('{"count": 3, "test_accessed": false}\n')
        row=checked_member(path,'attempt1/numeric.json','raw',root=root)
        path=root/'engineering.jsonl'
        path.write_text(json.dumps(dict(observation=observed,test_accessed=False),allow_nan=False)+'\n')
        engineering=checked_member(path,'attempt1/engineering.jsonl','raw',root=root)
        archive=root/'fixture.tar.gz'; write_archive(archive,[row,manifest,engineering]); assert archive.is_file()
        for name in ('../escape.json','/absolute.json','attempt1/guardian.json','attempt1/state.pt'):
            try: member_name(name)
            except p.TechnicalPause: pass
            else: raise AssertionError('Invalid archive member accepted')
        rejected=({'authkey':'secret'},{'nested':{'password':'secret'}},{'model':{'weights':[1,2]}},
            {'model':model.state_dict()},{'ema':ema.state_dict()},{'optimizer':opt.state_dict()},
            {'rng':p.rng_state('cpu')},{'loader':loader.get_state()},{'selected_models':{'online':model.state_dict()}},
            {'array':p.np.zeros(1)},{'bytes':b'state'},{'bytearray':bytearray(b'state')},
            {'opaque':io.BytesIO()},{'model':[1,2]},{'model':(1,2)})
        for value in rejected:
            try: numeric_document(value)
            except p.TechnicalPause: pass
            else: raise AssertionError('Private/state record accepted')
    return dict(status='PASS',groups=4,comparisons=17,metrics=34,paired_seed_rows=28900,paired_summary_rows=5780,
        archive_provenance=dict(manifest_sha256=manifest['sha256'],engineering_hash_fields=5,archive_readback=True,
            rejected_state_or_credential_records=len(rejected),train_calls=0,forwards=0,backwards=0,adam_steps=0),test_accessed=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('export','pack'))
    parser.add_argument('--attempt',default=str(p.ATTEMPT)); parser.add_argument('--root',default=str(p.ROOT))
    args=parser.parse_args(); p.require(Path(args.root)==p.ROOT,'ROOT differs')
    print(json.dumps(export(args.attempt) if args.command=='export' else pack(args.attempt),allow_nan=False))


if __name__=='__main__': main()
