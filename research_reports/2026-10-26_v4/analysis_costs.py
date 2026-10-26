"""Summarize recorded disjoint cost leaves; no training or imputed free work."""
from pathlib import Path
from datetime import datetime
import collections,csv,json,numpy as np

ROOT=Path('/home/xzf/Code/phase0-3/artifacts/mdmt_dfd_v4/v4_0')
OUT=ROOT/'analysis'/'costs'

def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csvsave(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def duration_union(intervals):
    total=0.;end=None
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total

events=[]
for path in sorted(list((ROOT/'prefix').glob('*/*/costs.jsonl'))+list((ROOT/'runs').glob('*/*/*/costs.jsonl'))):
    for line in path.read_text().splitlines():
        item=json.loads(line);item['ledger']=str(path.relative_to(ROOT));events.append(item)
OUT.mkdir(parents=True,exist_ok=True)
components=collections.defaultdict(lambda:[0,0.]);pids=collections.defaultdict(list);intervals=[]
for e in events:
    components[e['component']][0]+=1;components[e['component']][1]+=e['seconds']
    a=datetime.fromisoformat(e['started_utc']).timestamp();b=a+e['seconds'];pids[e['pid']].append((a,b));intervals.append((a,b))
rows=[dict(component=k,events=v[0],recorded_seconds=v[1]) for k,v in components.items()]
csvsave(OUT/'training_components.csv',rows)
spans=[(min(x[0] for x in v),max(x[1] for x in v)) for v in pids.values()]
save(OUT/'research_actual_recorded.json',dict(known_training_and_development_leaf_seconds=sum(e['seconds'] for e in events),
     leaf_events=len(events),observed_processes=len(pids),leaf_active_interval_union_seconds=duration_union(intervals),
     process_activity_spans_sum_seconds=sum(b-a for a,b in spans),process_activity_spans_union_seconds=duration_union(spans),
     caveat='Recorded components/activity spans, not complete process startup/finalize wall, isolated deployment retraining, GPU hours or energy. Missing initial/setup/weight-cache/I/O/review/wrapper pieces are not assigned zero. Analysis and final-evaluation costs are separate. Books overlap and must not be added.'))

standalone=[]
for seed in [5101,5102]:
    prefix=[e for e in events if e['ledger']==f'prefix/movielens/seed{seed}/costs.jsonl']
    native=sum(e['seconds'] for e in prefix if e['component'] in ['native_epoch','native_validation'])
    cache=sum(e['seconds'] for e in prefix if e['component']=='feature_cache')
    pilot=sum(e['seconds'] for e in prefix if e['component']=='pilot_pass')
    eta_fold=sum(e['seconds'] for e in prefix if e['component']=='eta_pass' and e.get('kind')=='eta' and e.get('rng_component') in [0,1])
    eta_refit=sum(e['seconds'] for e in prefix if e['component']=='eta_pass' and e.get('kind')=='eta' and e.get('rng_component')==2)
    uniform_form=sum(e['seconds'] for e in events if e['ledger']==f'runs/movielens/UNIFORM/seed{seed}/costs.jsonl' and e['component']=='formation_pass')
    for method in ['NATIVE','LOCAL','H_DIRECT','ETA_DIRECT','UNIFORM','GEOM','SHUFFLE','CONST_UNIFORM','LOCAL_CAP','DFD']:
        own=[e for e in events if e['ledger']==f'runs/movielens/{method}/seed{seed}/costs.jsonl']
        selected=[e for e in own if e['component'] not in ['evaluation','endpoint_evaluation']]
        prefix_cost=native
        if method!='NATIVE':prefix_cost+=cache
        if method not in ['NATIVE','H_DIRECT']:prefix_cost+=pilot
        if method in ['DFD','SHUFFLE']:prefix_cost+=eta_fold+eta_refit
        if method=='ETA_DIRECT':prefix_cost+=eta_refit
        if method=='CONST_UNIFORM':prefix_cost+=uniform_form
        standalone.append(dict(dataset='movielens',seed=seed,method=method,necessary_prefix_recorded_seconds=prefix_cost,
                               own_training_selection_recorded_seconds=sum(e['seconds'] for e in selected),
                               reconstructed_known_seconds=prefix_cost+sum(e['seconds'] for e in selected),
                               exact_total='UNKNOWN: unjournaled setup/cache bookkeeping not free; component rates share campaign GPU load'))
csvsave(OUT/'standalone_known_components.csv',standalone)
save(OUT/'development_information.json',dict(large_native24_trajectories=4,P1_neural_fold_heads=144,P1_const_target_task_fold_fits=48,
    P2_logical_outputs=20,P3='NO_START',P4='NO_START',P_form_training_label_passes_per_target_task=dict(DFD=60,SHUFFLE=60,UNIFORM=20,GEOM=20,LOCAL_CAP=20,H_DIRECT=20,ETA_DIRECT=20),
    note='DFD60=fold training20 + full refit20 + reader20. Fold20 is per-row, not40. Every soft/read direct additionally uses P_read10. Constant uses unique P_form union P_read with actual convex solver evaluations, not a fabricated SGD-pass count. Native/LOCAL no added target-pool fitting. These supervision counts are not all physical memory reads or independent labels.'))

support=[]
for seed in [5101,5102]:
    base=ROOT/'runs/movielens/DFD'/f'seed{seed}'
    w=json.loads((base/'weight_statistics.json').read_text());s=json.loads((ROOT/'prefix/movielens'/f'seed{seed}'/'RESPONSIBILITY_SUPPORT.json').read_text())
    for owner,v in w['owners'].items():
        for k in range(2):
            support.append(dict(seed=seed,owner=owner,intended_slot=k,source_n=v['n'],weight_mean=v['mean'][k],weight_sd=v['std'][k],weight_min=v['min'][k],weight_max=v['max'][k],mass=v['mass'][k],ESS=v['ess'][k],ESS_over_source_n=v['ess'][k]/v['n'],row_sum_max_error=w['row_sum_max_error'],max_mean_source_clip_rate=max(x['mean'] for x in s[owner]['source_support_clips'].values())))
csvsave(OUT/'formation_support.csv',support)
print('Cost analysis complete',len(events),sum(e['seconds'] for e in events),flush=True)
