"""Parent arithmetic for Cycle04; run after all twenty new runs and export finish."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
N=ROOT/'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2'
TABLES=N/'tables'
ARMS=('COMPLETION-BCE-025','MASS-SCALE-025','NQR-025','CLEAN')
DATASETS=('movielens','kuairand')
SEEDS=tuple(range(2022,2027))
KINDS=('online','fixed_ema')
ENDPOINTS=('selected','E1','E10','E30','E50')
CONTRASTS=((ARMS[0],'CLEAN'),(ARMS[1],'CLEAN'),(ARMS[0],ARMS[1]),
           (ARMS[2],ARMS[0]),(ARMS[2],'CLEAN'))
CELLS=tuple(f'd{d}t{t}' for d in range(3) for t in range(2))


def read_csv(name):
    with (TABLES/name).open(newline='') as f:
        return list(csv.DictReader(f))


def dump_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def physical_attempts(events):
    """One row per STARTED process; supervisor wall time overlaps worker time."""
    starts=[r for r in events if r.get('purpose')=='formal' and r['status']=='STARTED']
    result=[]
    for start in starts:
        same=[r for r in events if r.get('pid')==start['pid'] and r.get('key')==start['key']]
        worker=[r for r in same if r.get('purpose')=='formal' and r['status']!='STARTED'
                and r.get('attempt')==start['attempt']]
        observed=[r for r in same if r['status']=='SUPERVISOR_OBSERVED_EXIT']
        assert len(worker)<=1 and len(observed)<=1 and (worker or observed)
        terminal=worker[0] if worker else None
        supervisor=observed[0] if observed else None
        result.append(dict(key=start['key'],pid=start['pid'],attempt=start['attempt'],device=start['device'],
            status=terminal['status'] if terminal else 'FAILED_OBSERVED',
            seconds=terminal.get('seconds') if terminal else None,
            worker_terminal_missing=terminal is None,
            returned_train_calls=terminal.get('returned_train_calls') if terminal else None,
            unknown_partial_call=(terminal or supervisor).get('unknown_partial_call'),
            supervisor_process_wall_seconds=supervisor.get('elapsed_wall_seconds') if supervisor else None,
            supervisor_time_semantics='process_wall_including_load_and_exit; overlaps_worker_seconds'))
    assert len({(r['pid'],r['key']) for r in result})==len(starts)
    return result


def describe(values,metric):
    assert len(values)==5 and all(math.isfinite(v) for v in values)
    mean=statistics.mean(values);sd=statistics.stdev(values)
    half=2.7764451051977987*sd/math.sqrt(5)
    positive=sum(v>0 for v in values);negative=sum(v<0 for v in values)
    return dict(n=5,mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half,
        positive=positive,negative=negative,zero=sum(v==0 for v in values),
        favorable=negative if metric.endswith('_logloss') else positive if metric!='epoch' else None,
        unfavorable=positive if metric.endswith('_logloss') else negative if metric!='epoch' else None,
        favorable_direction='negative' if metric.endswith('_logloss') else 'positive' if metric!='epoch' else None)


def main():
    acceptance=json.loads((N/'PARENT_STORAGE_ACCEPTANCE.json').read_text())
    assert acceptance['status']=='ACCEPTED'
    receipt=json.loads((TABLES/'TABLE_RECEIPT.json').read_text())
    assert receipt['test_accessed'] is False
    audit=json.loads((TABLES/'RESOURCE_EXECUTION_AUDIT.json').read_text())
    assert audit['committed_epochs']==1000 and audit['logical_operations']['live_adam_steps']==34500
    assert audit['logical_operations']['live_ema_steps']==34500 and audit['logical_operations']['diagnostic_points']==80
    rows=read_csv('ENDPOINTS.csv');assert len(rows)==400
    metrics=[k for k in rows[0] if k.endswith(('_auc','_logloss'))]+['epoch']
    index={}
    for original in rows:
        row=dict(original)
        for metric in metrics:row[metric]=float(row[metric]) if metric!='epoch' else int(row[metric])
        row['seed']=int(row['seed'])
        key=row['dataset'],row['arm'],row['seed'],row['endpoint'],row['estimator']
        assert key not in index;index[key]=row
    means=[];pairs=[];summaries=[]
    for dataset in DATASETS:
        for endpoint in ENDPOINTS:
            for kind in KINDS:
                for arm in ARMS:
                    rs=[index[dataset,arm,seed,endpoint,kind] for seed in SEEDS]
                    for metric in metrics:
                        means.append(dict(dataset=dataset,arm=arm,estimator=kind,endpoint=endpoint,metric=metric,
                                          **describe([r[metric] for r in rs],metric)))
                comparisons=[(a,kind,b,kind) for a,b in CONTRASTS]
                if kind=='fixed_ema':
                    comparisons += [(ARMS[0],'fixed_ema','CLEAN','online')]
                    comparisons += [(arm,'fixed_ema',arm,'online') for arm in ARMS]
                for a,ak,b,bk in comparisons:
                    comparison=f'{a}/{ak} minus {b}/{bk}'
                    for metric in metrics:
                        values=[]
                        for seed in SEEDS:
                            l,r=index[dataset,a,seed,endpoint,ak],index[dataset,b,seed,endpoint,bk]
                            delta=l[metric]-r[metric];values.append(delta)
                            pairs.append(dict(dataset=dataset,comparison=comparison,endpoint=endpoint,metric=metric,
                                seed=seed,left=l[metric],right=r[metric],delta=delta,
                                favorable=(delta<0 if metric.endswith('_logloss') else delta>0) if metric!='epoch' else None,
                                left_epoch=l['epoch'],right_epoch=r['epoch'],left_state_hash=l['state_hash'],right_state_hash=r['state_hash']))
                        summaries.append(dict(dataset=dataset,comparison=comparison,endpoint=endpoint,metric=metric,
                                              **describe(values,metric)))
    lookup={(r['dataset'],r['comparison'],r['endpoint'],r['metric']):r for r in summaries}
    primary={};comparison=f'{ARMS[0]}/fixed_ema minus CLEAN/fixed_ema'
    for dataset in DATASETS:
        auc,ll=(lookup[dataset,comparison,'selected',m] for m in ('macro_auc','macro_logloss'))
        cells={c:lookup[dataset,comparison,'selected',c+'_auc'] for c in CELLS}
        minimum=min(v['mean'] for v in cells.values())
        primary[dataset]=dict(auc=auc,logloss=ll,cell_auc=cells,min_cell_mean_delta=minimum,
            macro_target_met=auc['mean']>=.003,cell_protection_met=minimum>=-.001,
            auc_target_and_cell_protection_met=auc['mean']>=.003 and minimum>=-.001,
            mean_selected_loss_decrease=ll['mean']<0)
    positive=sum(primary[d]['auc']['mean']>0 for d in DATASETS)
    outcome=('NO_CONTINUATION_SIGNAL' if positive==0 else 'MIXED' if positive==1 else
             'VALIDATION_AUC_TARGET_AND_CELL_PROTECTION_MET_BOTH_DATASETS'
             if all(primary[d]['auc_target_and_cell_protection_met'] for d in DATASETS) else
             'POSITIVE_SIGNAL_BELOW_QUALIFICATION')
    diagnostic_groups=defaultdict(lambda:defaultdict(list));diagnostic_nulls=defaultdict(list)
    annotations=[]
    annotation_values={'ema_relationship':'deltaEMA=(1-rho)*deltaTheta','ema_relationship_algebraic':'True'}
    for r in read_csv('COMMON_STATE_DIAGNOSTICS.csv'):
        if r['metric'] in annotation_values:
            assert r['defined']=='True' and r['value']==annotation_values[r['metric']] and not r['null_reason']
            annotations.append({k:r[k] for k in ('dataset','host_arm','seed','epoch','parameter_group','objective_or_pair','metric','value')})
            continue
        key=(r['dataset'],r['host_arm'],int(r['epoch']),r['parameter_group'],r['objective_or_pair'],r['metric'])
        if r['defined']=='True' and r['value'] not in ('',None):
            v=float(r['value']);assert math.isfinite(v)
            diagnostic_groups[key][int(r['seed'])].append(v)
        else:diagnostic_nulls[key].append(dict(seed=r['seed'],reason=r['null_reason']))
    diagnostic_means=[]
    for key in sorted(set(diagnostic_groups)|set(diagnostic_nulls)):
        seeds={seed:statistics.mean(values) for seed,values in diagnostic_groups[key].items()}
        assert set(seeds).issubset(SEEDS)
        valid=list(seeds.values())
        d,a,e,g,o,m=key
        diagnostic_means.append(dict(dataset=d,host_arm=a,epoch=e,parameter_group=g,objective_or_pair=o,metric=m,
            valid_seed_count=len(valid),mean=statistics.mean(valid) if valid else None,
            sd=statistics.stdev(valid) if len(valid)>1 else None,
            seed_values=json.dumps(seeds,sort_keys=True),unavailable=json.dumps(diagnostic_nulls[key],sort_keys=True),
            seed_aggregation='within_seed_mean_then_equal_seed_summary',counterfactual_at_host_state=True))
    assert len(annotations)==4160 and {r['metric'] for r in annotations}==set(annotation_values)
    dump_csv('DIAGNOSTIC_ANNOTATIONS.csv',annotations)
    costs=read_csv('COSTS.csv');physical=read_csv('PHYSICAL_ATTEMPTS.csv');cost_means=[]
    # Physical files contain STARTED plus terminal events. Long component costs
    # overlap logical-epoch/worker durations and must never be added to them.
    terminal=physical_attempts(physical)
    assert sum(r['status']=='COMPLETE' for r in terminal)==20
    dump_csv('PHYSICAL_ATTEMPT_SUMMARY.csv',terminal)
    jobs={r['key']:(r['dataset'],r['arm'],int(r['seed'])) for r in rows if r['arm'] in ARMS[:2]}
    seed_costs=defaultdict(list);unknown_costs=defaultdict(int)
    for r in terminal:
        dataset,arm,seed=jobs[r['key']]
        k=(dataset,arm,'physical_worker_seconds',seed)
        if r.get('seconds') not in ('',None):seed_costs[k].append(float(r['seconds']))
        else:unknown_costs[k]+=1
        observed=r['supervisor_process_wall_seconds']
        if observed not in ('',None):
            seed_costs[dataset,arm,'supervisor_process_wall_seconds_overlapping',seed].append(float(observed))
    for r in costs:
        key=r.get('key')
        if key not in jobs:
            path=Path(r.get('source_path') or '.')
            if path.is_relative_to(N/'runs'):
                key=path.relative_to(N/'runs').parts[0]
        if key not in jobs or r.get('purpose')=='engineering_smoke':continue
        dataset,arm,seed=jobs[key];kind=r['cost_kind']
        field='commit_status_get_seconds' if kind=='snapshot_transfer' else 'seconds'
        k=(dataset,arm,kind+'_'+field,seed)
        if r.get(field) not in ('',None):seed_costs[k].append(float(r[field]))
        else:unknown_costs[k]+=1
    component_names=sorted({k[2] for k in set(seed_costs)|set(unknown_costs)})
    for dataset in DATASETS:
        for arm in ARMS[:2]:
            for metric in component_names:
                values={seed:sum(seed_costs[dataset,arm,metric,seed]) for seed in SEEDS
                        if seed_costs[dataset,arm,metric,seed] and not unknown_costs[dataset,arm,metric,seed]}
                vs=list(values.values())
                cost_means.append(dict(dataset=dataset,arm=arm,metric=metric,measured_seeds=len(vs),
                    mean=statistics.mean(vs) if vs else None,sd=statistics.stdev(vs) if len(vs)>1 else None,
                    seed_values=json.dumps(values,sort_keys=True),
                    unknown_component_records=sum(unknown_costs[dataset,arm,metric,seed] for seed in SEEDS),
                    provenance='all_returned_formal_physical_attempts_or_named_components; overlapping_components_not_additive'))
    for name,values in (('ENDPOINT_MEANS.csv',means),('ALL_PAIRED_SEEDS.csv',pairs),
                        ('ALL_PAIRED_SUMMARY.csv',summaries),('DIAGNOSTIC_MEANS.csv',diagnostic_means),('COST_MEANS.csv',cost_means)):
        dump_csv(name,values)
    appendix=['# Cycle04 numerical appendix','','Natural validation; seeds2022–2026; earliest strict Macro AUC maximum independently per estimator. LL uses that same state.','']
    for dataset in DATASETS:
        for endpoint in ENDPOINTS:
            for kind in KINDS:
                appendix += [f'## {dataset} / {endpoint} / {kind}','','| Arm | Seed | Epoch | Macro AUC | Macro LL | State SHA256 |','|---|---:|---:|---:|---:|---|']
                for arm in ARMS:
                    for seed in SEEDS:
                        r=index[dataset,arm,seed,endpoint,kind]
                        appendix.append(f'| {arm} | {seed} | {r["epoch"]} | {r["macro_auc"]:.12f} | {r["macro_logloss"]:.12f} | {r["state_hash"]} |')
                appendix.append('')
        for arm in ARMS:
            for kind in KINDS:
                appendix += [f'## {dataset} / selected / {kind} / {arm} — cells','','| Seed | '+' | '.join(f'{c} {m}' for c in CELLS for m in ('AUC','LL'))+' |','|---:|'+'---:|'*12]
                for seed in SEEDS:
                    r=index[dataset,arm,seed,'selected',kind]
                    appendix.append('| '+str(seed)+' | '+' | '.join(f'{r[c+"_"+m]:.12f}' for c in CELLS for m in ('auc','logloss'))+' |')
                appendix.append('')
    (OUT/'NUMERICAL_APPENDIX.md').write_text('\n'.join(appendix)+'\n')
    facts=dict(candidate=ARMS[0],decision=outcome,primary=primary,new_trajectories=20,reused_trajectories=20,
        new_epochs=1000,new_useful_updates=34500,test_accessed=False,champion_selected=False,
        source_hash=receipt['source_hash'],design_bundle_sha256=receipt['design_bundle_sha256'],
        physical_attempts=len(terminal),physical_status_counts={s:sum(r['status']==s for r in terminal) for s in set(r['status'] for r in terminal)},
        measured_physical_worker_seconds=sum(float(r['seconds']) for r in terminal if r.get('seconds') not in ('',None)),
        unknown_terminal_durations=sum(r.get('seconds') in ('',None) for r in terminal),
        supervisor_process_wall_seconds=sum(float(r['supervisor_process_wall_seconds']) for r in terminal
                                            if r['supervisor_process_wall_seconds'] not in ('',None)),
        physical_time_semantics='worker duration excludes missing terminal; supervisor duration overlaps it and includes process startup/exit',
        retired_attempt_costs=json.loads((OUT/'RETIRED_ATTEMPT_COSTS.json').read_text()),
        export_receipt_sha256=sha(TABLES/'TABLE_RECEIPT.json'),
        inputs={name:sha(TABLES/name) for name in ('ENDPOINTS.csv','COMMON_STATE_DIAGNOSTICS.csv','COSTS.csv','PHYSICAL_ATTEMPTS.csv')})
    availability_path=N.parent/'SELECTED_RAM_AVAILABILITY_UPDATE.json'
    availability=json.loads(availability_path.read_text())
    facts.update(selected_ram_available_at_parent_acceptance=True,current_selected_ram_available=availability['selected_ram_available'],
        availability_update=dict(path=str(availability_path),sha256=sha(availability_path)),final_test_eligible=False)
    (OUT/'DECISION.json').write_text(json.dumps(facts,indent=2)+'\n')
    print(json.dumps(dict(decision=outcome,primary=primary,paired_seed_rows=len(pairs),paired_summary_rows=len(summaries),test_accessed=False)))


if __name__=='__main__':
    main()
