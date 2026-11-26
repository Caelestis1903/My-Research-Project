"""Scalar tables and seed-level paired summaries; no scheduling decisions."""
import csv
import io
import math
import statistics
from pathlib import Path
from .state import read_json,write_json,sha_file
from .inventory import read_csv,scalar_row
from .selection import CELLS

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
def stats(values):
    n=len(values); mean=statistics.fmean(values) if n else None
    sd=statistics.stdev(values) if n>1 else None
    margin=2.7764451051977987*sd/math.sqrt(5) if n==5 else None
    return dict(n=n,mean=mean,sd=sd,ci95_low=mean-margin if margin is not None else None,
                ci95_high=mean+margin if margin is not None else None)
def decision(rows):
    seeds=[int(r['seed']) for r in rows]
    if len(seeds)!=len(set(seeds)): raise ValueError('Duplicate paired seed')
    if set(seeds)!=set(range(2022,2027)): return dict(status='INCOMPLETE',n=len(rows))
    auc=statistics.fmean(r['delta_macro_auc'] for r in rows)
    ll=statistics.fmean(r['delta_macro_logloss'] for r in rows)
    cells={c:statistics.fmean(r['delta_'+c+'_auc'] for r in rows) for c in CELLS}
    macro=auc>=.003; guards=all(v>=-.001 for v in cells.values())
    return dict(status='AUC_PASS_WITH_CELL_PROTECTION' if macro and guards else
                'AUC_MACRO_PASS_CELL_PROTECTION_FAIL' if macro else 'AUC_MACRO_BELOW_THRESHOLD',
        n=5,macro_auc_pass=macro,cell_protection_pass=guards,logloss_improved=ll<0,
        macro_auc_delta=auc,macro_logloss_delta=ll,cell_auc_deltas=cells)
def endpoint_rows(attempt):
    attempt=Path(attempt); rows=read_json(attempt/'REUSED_ENDPOINTS.json')
    for path in sorted((attempt/'runs').glob('*/DONE.json')):
        done=read_json(path); parent=path.parent
        if any(sha_file(parent/name)!=expected for name,expected in done['files'].items()):
            raise ValueError(f'Completed scalar archive changed: {parent}')
        rows.extend(read_json(parent/'endpoints.json'))
    identities=[(r['dataset'],r['arm'],int(r['seed']),r['estimator'],r['checkpoint'],r['split']) for r in rows]
    if len(identities)!=len(set(identities)): raise ValueError('Duplicate endpoint identity')
    return rows
def report(attempt):
    attempt=Path(attempt); prepared=read_json(attempt/'PREPARED.json'); p=prepared['protocol']
    destination=Path(prepared['report_root']); destination.mkdir(parents=True,exist_ok=True)
    rows=endpoint_rows(attempt); write_csv(destination/'ENDPOINTS.csv',rows)
    metrics=sorted({k for r in rows for k in r if k.endswith(('_auc','_logloss'))})
    grouped={}
    for r in rows:
        key=tuple(r[k] for k in ('dataset','arm','method','role','estimator','checkpoint','split'))
        grouped.setdefault(key,[]).append(r)
    absolute=[]
    for key,values in sorted(grouped.items()):
        base=dict(zip(('dataset','arm','method','role','estimator','checkpoint','split'),key))
        for metric in metrics:
            data=[r[metric] for r in values if r.get(metric) is not None]
            absolute.append(dict(base,metric=metric,**stats(data)))
    write_csv(destination/'ABSOLUTE_SUMMARY.csv',absolute)
    selected=[r for r in rows if r['checkpoint']=='selected' and r['split']=='test']
    lookup={(r['dataset'],r['arm'],int(r['seed']),r['estimator']):r for r in selected}
    pairs=[]; comparisons={}
    candidates=(('C_MIX','fixed_ema'),('P01','fixed_ema'),('C_MIX_P01_SERIAL_V1','fixed_ema'),
                ('GAUSSIAN_A1','fixed_ema'),('C_MIX_GAUSSIAN_V1','fixed_ema'))
    for ds in p['datasets']:
        for arm,kind in candidates:
            if arm.startswith('GAUSSIAN') or arm=='C_MIX_GAUSSIAN_V1':
                if not prepared.get('fallback_enabled'): continue
            refs=[('A0','online','total'),('A0','fixed_ema','component'),
                  ('HYPERGATE','online','published'),('AUTOIFS','online','published')]
            if arm.startswith('C_MIX_P01'): refs.extend([('C_MIX','fixed_ema','combination'),('P01','fixed_ema','combination')])
            for ref,refkind,role in refs:
                comparison=f'{arm}.{kind}_vs_{ref}.{refkind}'
                current=[]
                for seed in p['seeds']:
                    c=lookup.get((ds,arm,seed,kind)); b=lookup.get((ds,ref,seed,refkind))
                    if c is None or b is None: continue
                    if (c['protocol_id'],c['data_protocol'])!=(b['protocol_id'],b['data_protocol']):
                        raise ValueError('Paired protocols differ')
                    out=dict(dataset=ds,comparison_id=comparison,comparison_role=role,seed=seed,
                        candidate_arm=arm,reference_arm=ref,candidate_hash=c['model_hash'],
                        reference_hash=b['model_hash'],candidate_epoch=c['epoch'],reference_epoch=b['epoch'])
                    for metric in metrics:
                        if c.get(metric) is not None and b.get(metric) is not None:
                            out.update({f'candidate_{metric}':c[metric],f'reference_{metric}':b[metric],
                                        f'delta_{metric}':c[metric]-b[metric]})
                    pairs.append(out); current.append(out)
                comparisons[(ds,comparison)]=current
    write_csv(destination/'PAIRED_SEED.csv',pairs)
    paired_summary=[]; decisions=[]
    for (ds,comparison),values in sorted(comparisons.items()):
        verdict=decision(values); decisions.append(dict(dataset=ds,comparison_id=comparison,**verdict))
        for metric in metrics:
            valid=[r['delta_'+metric] for r in values if 'delta_'+metric in r]
            paired_summary.append(dict(dataset=ds,comparison_id=comparison,metric=metric,
                status=verdict['status'],**stats(valid)))
    write_csv(destination/'PAIRED_SUMMARY.csv',paired_summary); write_json(destination/'DECISIONS.json',decisions)
    factorial=[]
    for name,cells in p['factorials'].items():
        if name=='CG' and not prepared.get('fallback_enabled'): continue
        for ds in p['datasets']:
            for seed in p['seeds']:
                four={cell:lookup.get((ds,arm,seed,'fixed_ema')) for cell,arm in cells.items()}
                if any(row is None for row in four.values()): continue
                out=dict(dataset=ds,factorial=name,seed=seed)
                for metric in metrics:
                    if any(r.get(metric) is None for r in four.values()): continue
                    out.update({cell+'_'+metric:r[metric] for cell,r in four.items()})
                    out['interaction_'+metric]=four['11'][metric]-four['10'][metric]-four['01'][metric]+four['00'][metric]
                factorial.append(out)
    write_csv(destination/'FACTORIAL_SEED.csv',factorial); factorial_summary=[]
    for name in p['factorials']:
        for ds in p['datasets']:
            current=[r for r in factorial if r['dataset']==ds and r['factorial']==name]
            for metric in metrics:
                values=[r['interaction_'+metric] for r in current if 'interaction_'+metric in r]
                factorial_summary.append(dict(dataset=ds,factorial=name,metric=metric,
                    status='COMPLETE' if len(values)==5 else 'INCOMPLETE',**stats(values)))
    write_csv(destination/'FACTORIAL_SUMMARY.csv',factorial_summary)
    curves=read_json(attempt/'REUSED_VALIDATION_CURVES.json')
    costs=[]
    historical=Path(prepared['history_root'])/'research_reports/2026-11-03_mdmt_five_seed_completion/costs.csv'
    if historical.exists(): costs.extend(dict(r,cost_origin='historical_reuse',new_training_seconds=0)
        for r in read_csv(historical) if r.get('arm') in ('A0','C_MIX','P01','Gaussian'))
    for job in prepared['jobs']:
        parent=attempt/'runs'/job['job_id']
        for path in sorted(parent.glob('launch_*/epoch_*.json')):
            r=read_json(path); validation=r['validation']
            if 'macro_auc' in validation: validation={'online':validation}
            for estimator,values in validation.items():
                curves.append(dict(dataset=job['dataset'],arm=job['arm'],seed=job['seed'],
                    estimator=estimator,stage=r['stage'],epoch=r['epoch'],**values))
        if (parent/'cost.json').exists():
            costs.append(dict(dataset=job['dataset'],arm=job['arm'],seed=job['seed'],
                trajectory_id=job['trajectory_id'],cost_origin='new_training',**read_json(parent/'cost.json')))
        for path in parent.glob('launch_*/failure.json'):
            costs.append(dict(dataset=job['dataset'],arm=job['arm'],seed=job['seed'],
                cost_origin='failed_physical_attempt',**read_json(path)))
        for path in parent.glob('launch_*/physical_cost.json'):
            costs.append(dict(dataset=job['dataset'],arm=job['arm'],seed=job['seed'],
                cost_origin='physical_launch_including_RAM_IPC',**read_json(path)))
    write_csv(destination/'VALIDATION_CURVES.csv',curves); write_csv(destination/'COSTS.csv',costs)
    for name in ('A_REUSE_AUDIT.csv','B_REUSE_AUDIT.csv'):
        (destination/name).write_bytes((attempt/name).read_bytes())
    queue=read_json(attempt/'queue_state.json')
    statuses={name:r['status'] for name,r in queue['jobs'].items()}
    counts={status:list(statuses.values()).count(status) for status in sorted(set(statuses.values()))}
    expected_A={(ds,arm,seed,'online') for ds in p['datasets'] for arm in ('A0','HYPERGATE','AUTOIFS') for seed in p['seeds']}
    expected_B={(ds,arm,seed,'fixed_ema') for ds in p['datasets'] for arm in p['factorials']['CP'].values() for seed in p['seeds']}
    missingA=sorted(expected_A-set(lookup)); missingB=sorted(expected_B-set(lookup))
    complete=not missingA and not missingB and all(s in ('DONE','REUSE_DONE','CONDITIONAL_NOT_ENABLED') for s in statuses.values())
    status='FINAL_COMPLETE' if complete else 'WAIT_EXTERNAL' if 'WAIT_EXTERNAL' in counts else 'PARTIAL_ENGINEERING' if 'BLOCKED_ENGINEERING' in counts else 'RUNNING'
    progress=dict(status=status,counts=counts,A_complete=30-len(missingA),A_required=30,
        B_complete=40-len(missingB),B_required=40,missing_A=missingA,missing_B=missingB)
    write_json(attempt/'progress.json',progress)
    text='# Overnight A/B 实验进度\n\n'+f'状态：{status}。A端点 {30-len(missingA)}/30；B端点 {40-len(missingB)}/40。\n\n'
    text+='状态计数：'+', '.join(f'{key}={value}' for key,value in counts.items())+'。\n\n'
    text+='缺失A：'+str(missingA)+'\n\n缺失B：'+str(missingB)+'\n'
    (destination/'PROGRESS.md').write_text(text)
    (destination/'FINAL_REPORT.md').write_text(text+'\n逐seed原始值见ENDPOINTS.csv，配对统计见PAIRED_SUMMARY.csv，组合交互见FACTORIAL_SUMMARY.csv。\n')
    (destination/'LEDGER_APPEND.md').write_text(f'\n### MDMT_OVERNIGHT_AB_20261103 / AUC_SELECTED_V1\n\n状态 {status}；A {30-len(missingA)}/30；B {40-len(missingB)}/40。数据 FULLTRAIN_NATURAL80_10_10。\n')
    return progress

