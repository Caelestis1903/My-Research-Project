"""Export raw endpoints, paired seed statistics and measured costs for analysis."""
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from scipy.stats import t
from .protocol import REPORT, digest, write_json
from .decisions import recipe, config, read_decision, STAGES


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def stats(values):
    n=len(values);m=mean(values);sd=stdev(values) if n>1 else None
    margin=float(t.ppf(.975,n-1))*sd/math.sqrt(n) if n>1 else None
    return dict(n=n,mean=m,sample_sd=sd,ci95_low=m-margin if margin is not None else None,ci95_high=m+margin if margin is not None else None,positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),zero=sum(v==0 for v in values))


def report(attempt):
    attempt=Path(attempt);evaluation=json.loads((attempt/'final_evaluation.json').read_text());assert evaluation['status']=='COMPLETE'
    results=[json.loads(f.read_text()) for f in sorted((attempt/'runs').glob('*/result.json'))]
    endpoints=[];costs=[];epochs=[];run_manifest=[]
    for r in results:
        j=r['job'];base=dict(key=j['key'],dataset=j['dataset'],seed=j['seed'],arm=j['config']['arm'],configuration=digest(j['config']),config_json=json.dumps(j['config'],sort_keys=True),historical_reference=bool(r.get('provenance')))
        run_manifest.append(dict(**base,job=j,provenance=r.get('provenance'),epochs=r['epochs'],selected_epochs=r['selected_epochs']))
        for kind in ('online','fixed_ema'):
            for cp in ('selected','E10','E50'):
                endpoints.append(dict(**base,split='validation',model_kind=kind,checkpoint=cp,epoch=r['selected_epochs'][kind] if cp=='selected' else int(cp[1:]),model_hash=r[cp+'_hashes'][kind],**r[cp+'_validation'][kind]))
        cost=dict(**base,worker_wall_seconds=r.get('attempt_worker_seconds',r['worker_seconds']),logical_committed_worker_seconds=r['worker_seconds'],new_training_worker_seconds=0. if r.get('provenance') else r.get('attempt_worker_seconds',r['worker_seconds']),peak_memory_bytes=r.get('peak_memory_bytes'),peak_reserved_bytes=r.get('peak_reserved_bytes'),**r.get('physical_counts',r.get('counts',{})))
        cost['cost_role']='historical_reference' if r.get('provenance') else 'new_training';costs.append(cost)
        path=Path(r['provenance']['epoch_directory']) if r.get('provenance') else attempt/'runs'/j['key']
        files=sorted(path.glob('epoch_*.json')) if r.get('provenance') else sorted(path.glob('attempt*/epoch_*.json'))
        by_epoch={}
        for f in files:
            e=json.loads(f.read_text());by_epoch[e['epoch']]=e
        assert set(by_epoch)==set(range(1,51)),('Incomplete epoch history',j['key'])
        for e in by_epoch.values():
            for kind,metrics in e['validation'].items():epochs.append(dict(**base,epoch=e['epoch'],model_kind=kind,training_loss=e['loss'],training_real_loss=e['real_loss'],**metrics))
    by_key={r['job']['key']:r for r in results}
    for r in evaluation['rows']:
        original=by_key[r['key']];j=r['job'];base=dict(key=r['key'],dataset=j['dataset'],seed=j['seed'],arm=j['config']['arm'],configuration=digest(j['config']),config_json=json.dumps(j['config'],sort_keys=True),historical_reference=bool(original.get('provenance')))
        endpoints.append(dict(**base,split='test',model_kind=r['model_kind'],checkpoint=r['checkpoint'],epoch=r['epoch'],model_hash=r['model_hash'],**r['metrics']))
        costs.append(dict(**base,cost_role='historical_test_reference' if original.get('provenance') else 'test',checkpoint=r['checkpoint'],model_kind=r['model_kind'],**r['cost']))
    for f in sorted((attempt/'runs').glob('*/attempt*/failure.json')):
        failure=json.loads(f.read_text());costs.append(dict(key=failure['key'],cost_role='failed_physical_attempt',attempt=failure['attempt'],worker_wall_seconds=failure['worker_seconds'],error=failure['error'],failed_step_counts_complete=failure.get('failed_step_counts_complete',False),**failure.get('completed_step_counts',{})))
    cells=[]
    for r in endpoints:
        identity={k:r[k] for k in ('key','dataset','seed','arm','configuration','split','model_kind','checkpoint','epoch','model_hash','historical_reference')}
        for d in range(3):
            for task in range(2):cells.append(dict(**identity,domain=d,task=task,auc=r[f'd{d}t{task}_auc'],logloss=r[f'd{d}t{task}_logloss'],count=r[f'd{d}t{task}_count'],positive=r[f'd{d}t{task}_positive']))
    comparison_pairs=set()
    for dataset in ('movielens','kuairand'):
        configs={r['configuration']:r for r in endpoints if r['dataset']==dataset}
        a0=digest(config('A0f',dataset));baselines={a0}
        for stage in ('D1','REPLICATE','D2','CONFIRM'):
            file=attempt/'decisions'/f'{stage}.json'
            if file.exists():
                v=json.loads(file.read_text())['datasets'][dataset]
                if v.get('winners'):
                    b=v['winners'][2];baselines.add(digest(config(b['arm'],dataset,**{k:x for k,x in b.items() if k!='arm'})))
        for a in configs:
            for b in baselines:
                if a!=b:comparison_pairs.add((dataset,a,b))
        # All measured members of a family are useful paired component contrasts.
        for a,ra in configs.items():
            for b,rb in configs.items():
                if a==b:continue
                if ra['arm'] in ('H_MIX','TM','M') and rb['arm'] in ('H_REAL','C_MIX','H_U','H_NORM','M','T0','T_U','T_NORM','U_MIX','SCALE'):
                    ca=json.loads(ra['config_json']);cb=json.loads(rb['config_json'])
                    if all(ca.get(k)==cb.get(k) for k in ('tau','kappa','lambda_mix') if k in ca and k in cb):comparison_pairs.add((dataset,a,b))
    lookup={(r['dataset'],r['configuration'],r['seed'],r['split'],r['model_kind'],r['checkpoint']):r for r in endpoints}
    metrics=[k for k in endpoints[0] if k.endswith(('_auc','_logloss'))]
    paired=[];summaries=[]
    for dataset,a,b in sorted(comparison_pairs):
        for split in ('validation','test'):
            for kind in ('online','fixed_ema'):
                for cp in ('selected','E10','E50'):
                    found={s:(lookup[(dataset,a,s,split,kind,cp)],lookup[(dataset,b,s,split,kind,cp)]) for s in range(2022,2027) if (dataset,a,s,split,kind,cp) in lookup and (dataset,b,s,split,kind,cp) in lookup}
                    for metric in metrics:
                        values={s:ra[metric]-rb[metric] for s,(ra,rb) in found.items() if ra.get(metric) is not None and rb.get(metric) is not None}
                        for s,delta in values.items():paired.append(dict(dataset=dataset,candidate=a,baseline=b,seed=s,split=split,model_kind=kind,checkpoint=cp,metric=metric,candidate_value=found[s][0][metric],baseline_value=found[s][1][metric],delta=delta))
                        for subset,seeds in [('development',(2022,2024)),('nondevelopment',(2023,2025,2026)),('all',tuple(range(2022,2027)))]:
                            vs=[values[s] for s in seeds if s in values]
                            if vs:summaries.append(dict(dataset=dataset,candidate=a,baseline=b,split=split,model_kind=kind,checkpoint=cp,metric=metric,subset=subset,**stats(vs)))
    interactions=[]
    for r in results:
        j=r['job'];c=j['config']
        if c['arm']!='TM':continue
        d=j['dataset'];s=j['seed'];recipes=[('TM',c),('T0',config('T0',d,kappa=c['kappa'])),('M',config('M',d,lambda_mix=c['lambda_mix'],tau=c['tau'])),('A0f',config('A0f',d))]
        for split in ('validation','test'):
            for kind in ('online','fixed_ema'):
                for cp in ('selected','E10','E50'):
                    keys=[(d,digest(cc),s,split,kind,cp) for _,cc in recipes]
                    if all(k in lookup for k in keys):
                        for metric in metrics:
                            vals=[lookup[k].get(metric) for k in keys]
                            if None not in vals:interactions.append(dict(dataset=d,seed=s,configuration=digest(c),split=split,model_kind=kind,checkpoint=cp,metric=metric,interaction=vals[0]-vals[1]-vals[2]+vals[3]))
    for name,rows in [('endpoints',endpoints),('six_cell',cells),('paired_seed',paired),('paired_summary',summaries),('costs',costs),('validation_epochs',epochs),('transport_interactions',interactions)]:write_csv(REPORT/f'{name}.csv',rows)
    write_json(REPORT/'RUN_MANIFEST.json',run_manifest)
    return dict(status='COMPLETE',runs=len(results),endpoints=len(endpoints),six_cells=len(cells),paired_rows=len(paired),report_directory=str(REPORT))
