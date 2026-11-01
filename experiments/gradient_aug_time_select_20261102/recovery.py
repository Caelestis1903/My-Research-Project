"""Numerical history interfaces and identity/cost continuity after RAM loss."""
import hashlib
import json
from pathlib import Path
import time
import pandas as pd
from .protocol import DEV_SEEDS,config,digest,write_json

MISSING=object()
FIXED_SECTIONS=('training','model','optimizer','rng')
COST_KEYS=('training_seconds','original_exposures','extra_views','view_draws','training_forwards',
    'gradient_calls','optimizer_steps','shadow_updates','validation_forwards','validation_wall_seconds',
    'validation_gpu_event_ms','training_gpu_event_ms','selection_checks','training_passes')


def read_identity(value,path):
    current=value
    for key in path:
        if not isinstance(current,dict) or key not in current:return MISSING
        current=current[key]
    return current


def historical_endpoint_core(rows,p,out,filename='historical_baseline_verification.json'):
    """Both CPU preflight and actual RAM adapters use these exact comparisons."""
    history=Path(p['history_dir']);datasets=list(p['datasets']);reasons={d:[] for d in datasets}
    checks=[];identity_checks=[];files={};loaded={}
    required=('validation_metrics.csv','development_selected_checkpoints.json','formal_configuration_frozen.json',
        'data_manifest.json','resolved_config.json')
    for name in required:
        path=history/name
        try:
            content=path.read_bytes();files[name]=dict(path=str(path),sha256=hashlib.sha256(content).hexdigest())
            loaded[name]=pd.read_csv(path) if name.endswith('.csv') else json.loads(content)
        except (OSError,ValueError) as exc:
            for d in datasets:reasons[d].append(f'{name}: {type(exc).__name__}: {exc}')
    def compare_identity(dataset,label,old,new,old_path,current_path):
        present=old is not MISSING and new is not MISSING and old is not None and new is not None
        match=bool(present and old==new)
        identity_checks.append(dict(dataset=dataset,field=label,historical_field=old_path,current_field=current_path,
            present=present,matches=match))
        if not match:reasons[dataset].append(f'{label}: '+('missing required identity' if not present else 'identity mismatch'))
        return match
    if len(loaded)==len(required):
        original=loaded['validation_metrics.csv'];development=read_identity(loaded['development_selected_checkpoints.json'],('checkpoints',))
        frozen=loaded['formal_configuration_frozen.json'];data=loaded['data_manifest.json'];resolved=loaded['resolved_config.json']
        columns={'dataset','arm','seed','model_kind','checkpoint','model_hash','epoch'}
        if not columns.issubset(original.columns):
            for d in datasets:reasons[d].append('validation_metrics.csv: missing columns '+','.join(sorted(columns-set(original.columns))))
        if not isinstance(development,list):
            for d in datasets:reasons[d].append('development_selected_checkpoints.json: missing checkpoints list')
        for d in datasets:
            for section in FIXED_SECTIONS:
                compare_identity(d,section,read_identity(resolved,(section,)),read_identity(p,(section,)),
                    f'resolved_config.{section}',f'protocol.{section}')
            for key in ('raw_sha256','array_hashes','split_row_id_hashes','split_row_fingerprints'):
                compare_identity(d,key,read_identity(data,(d,key)),read_identity(p,('datasets',d,key)),
                    f'data_manifest.{d}.{key}',f'protocol.datasets.{d}.{key}')
            for key in ('encoder_hash','scaler_hash'):
                compare_identity(d,key,read_identity(data,(d,'features',key)),read_identity(p,('datasets',d,key)),
                    f'data_manifest.{d}.features.{key}',f'protocol.datasets.{d}.{key}')
            counts=read_identity(data,(d,'train_domain_counts'))
            if isinstance(counts,dict):
                expected=read_identity(p,('datasets',d,'train_domain_counts'))
                keys=[str(i) for i in range(len(expected))] if isinstance(expected,list) else []
                counts=[counts[k] for k in keys] if keys and all(k in counts for k in keys) else MISSING
            compare_identity(d,'train_domain_counts',counts,read_identity(p,('datasets',d,'train_domain_counts')),
                f'data_manifest.{d}.train_domain_counts ordered d0/d1/d2',f'protocol.datasets.{d}.train_domain_counts')
            current_rows=[r for r in rows if r.get('dataset')==d]
            expected_keys={(arm,seed,kind,checkpoint) for arm in ('A0','A1') for seed in DEV_SEEDS
                for kind in ('online','fixed_ema') for checkpoint in ('selected','E10','E50')}
            actual_keys=[(r.get('arm'),r.get('seed'),r.get('model_kind'),r.get('checkpoint')) for r in current_rows]
            if set(actual_keys)!=expected_keys or len(actual_keys)!=len(expected_keys):
                reasons[d].append('current endpoints: expected 24 unique A0/A1 online/fixed selected/E10/E50 development records')
            for arm in ('A0','A1'):
                now=config(arm,d);old=read_identity(frozen,('configurations',d,arm,'config'))
                for key in ('lr','weight_decay','betas','eps','lambda_aug'):
                    compare_identity(d,f'{arm}.{key}',read_identity(old,(key,)),now[key],
                        f'formal_configuration_frozen.configurations.{d}.{arm}.config.{key}',f'config({arm},{d}).{key}')
                if arm=='A1':compare_identity(d,'A1.noise_scale',read_identity(old,('noise_scale',)),now['noise_scale'],
                    f'formal_configuration_frozen.configurations.{d}.A1.config.noise_scale',f'config(A1,{d}).noise_scale')
                kind=read_identity(frozen,('configurations',d,arm,'kind'))
                old_rho=read_identity(old,('shadow_rates',kind)) if isinstance(kind,str) else MISSING
                compare_identity(d,f'{arm}.fixed_rho',old_rho,now['rho_fixed'],
                    f'formal_configuration_frozen.configurations.{d}.{arm}.config.shadow_rates[selected kind]',f'config({arm},{d}).rho_fixed')
                for row in [r for r in current_rows if r.get('arm')==arm]:
                    rowkind=row.get('model_kind');oldkind='online' if rowkind=='online' else kind
                    sourcearm=arm if rowkind=='online' else arm+'_EMA';found=[];source='validation_metrics.csv'
                    if columns.issubset(original.columns) and isinstance(oldkind,str):
                        found=original[(original.dataset==d)&(original.arm==sourcearm)&(original.seed==row.get('seed'))&
                            (original.model_kind==oldkind)&(original.checkpoint==row.get('checkpoint'))].to_dict('records')
                    if not found and isinstance(development,list) and isinstance(oldkind,str):
                        trial='C' if arm=='A0' else 'G1' if d=='movielens' else 'G2';source='development_selected_checkpoints.json'
                        found=[r for r in development if r.get('dataset')==d and r.get('seed')==row.get('seed') and r.get('trial')==trial and
                            r.get('model_kind')==oldkind and r.get('checkpoint')==row.get('checkpoint') and r.get('stage')==('S1' if arm=='A0' else 'S2')]
                    old_hash=found[0].get('model_hash') if len(found)==1 else None
                    epoch_match=len(found)==1 and found[0].get('epoch') is not None and row.get('epoch') is not None and found[0]['epoch']==row['epoch']
                    match=bool(len(found)==1 and isinstance(old_hash,str) and old_hash and
                        isinstance(row.get('model_hash'),str) and old_hash==row['model_hash'] and epoch_match)
                    checks.append(dict(dataset=d,arm=arm,kind=rowkind,seed=row.get('seed'),checkpoint=row.get('checkpoint'),
                        current_model_hash=row.get('model_hash'),historical_model_hash=old_hash,current_epoch=row.get('epoch'),
                        historical_epoch=found[0].get('epoch') if len(found)==1 else None,source_file=source,matches=match,matches_found=len(found)))
                    if not match:reasons[d].append(f"{arm}/{rowkind}/seed{row.get('seed')}/{row.get('checkpoint')}: endpoint identity mismatch or nonunique/missing historical record")
    verified={d:not reasons[d] for d in datasets}
    receipt=dict(status='PASS' if all(verified.values()) else 'REUSE_REJECTED',verified=verified,reasons=reasons,
        checks=checks,identity_checks=identity_checks,endpoint_count=len(rows),matched_endpoints=sum(r['matches'] for r in checks),
        history_dir=str(history),source_files=files,source_commit='6f5742c2085c1e49d1767b0a7079bb9f28160eda',
        historical_test_accessed_for_decision=False,weights_loaded=False)
    write_json(Path(out)/filename,receipt)
    return receipt


def completed_attempt_runs(attempt):
    attempt=Path(attempt);ledger=pd.read_csv(attempt/'trial_ledger.csv');rows=ledger[ledger.status=='COMPLETE'].to_dict('records')
    runs={}
    for row in rows:
        directory=Path(row['output_dir']);stats=json.loads((directory/'runtime_stats.json').read_text())
        key=f"{stats['dataset']}|{stats['trial']}|{stats['seed']}"
        if key in runs:raise ValueError(f'Duplicate completed scientific run in recovery source: {key}')
        if stats['epochs']!=50:raise ValueError(f'Incomplete recovery source: {key}')
        runs[key]=dict(stats=stats,directory=str(directory),state_summary=str(directory/'state_summary.json'))
    expected={f'{d}|{a}|{s}' for d in ('movielens','kuairand') for a in ('A0','A1','B1','B2','B3') for s in DEV_SEEDS}
    if set(runs)!=expected:raise ValueError('Recovery source must contain exactly the original 20 complete base trajectories')
    return runs


def attempt_endpoint_rows(attempt):
    rows=[]
    for run in completed_attempt_runs(attempt).values():
        rows.extend(r for r in json.loads(Path(run['state_summary']).read_text())['checkpoints']
            if r['arm'] in ('A0','A1') and r['model_kind'] in ('online','fixed_ema'))
    return rows


def task_clock(task_start,limits,recovery_start=None):
    now=time.time() if recovery_start is None else float(recovery_start)
    start=now if task_start is None else float(task_start)
    if not start>0 or start>now:raise ValueError('Task start must be a positive past UNIX timestamp')
    return dict(task_start_unix=start,recovery_start_unix=now,elapsed_before_recovery_seconds=now-start,
        stop_dispatch_deadline=start+limits['stop_hours']*3600,dispatch_deadline=start+limits['hard_hours']*3600)


def recovery_context(attempt,task_start,p):
    attempt=Path(attempt);runs=completed_attempt_runs(attempt)
    original=json.loads((attempt/'resolved_config.json').read_text())
    for section in FIXED_SECTIONS:
        if read_identity(original,(section,)) is MISSING or original[section]!=p[section]:
            raise ValueError(f'Recovery source fixed configuration mismatch: {section}')
    launches=pd.read_csv(attempt/'launch_ledger.csv');first=float(launches['time'].min())
    if abs(float(task_start)-first)>1e-6:raise ValueError('Recovery task start differs from original first launch')
    forecast=json.loads((attempt/'throughput_calibration.json').read_text())['projected_development_worker_seconds']
    if not forecast>0:raise ValueError('Original development forecast must be positive')
    costs={k:sum(run['stats'].get(k,0) for run in runs.values()) for k in COST_KEYS}
    return dict(prior_attempt=str(attempt.resolve()),runs=runs,prior_costs=costs,
        prior_unique_scientific_trajectories=len(runs),original_development_forecast_seconds=forecast,
        original_execution_commit=original.get('execution_commit'),original_source_hash=original.get('source_manifest_hash'),
        original_task_start_unix=first,original_test_evaluated=False)


def verify_recomputed_trajectory(result,context,out):
    key=f"{result['dataset']}|{result['trial']}|{result['seed']}";source=context['runs'][key]
    old=source['stats'];directory=Path(source['directory']);primary=directory/'primary' if (directory/'primary').exists() else directory
    trace=pd.read_csv(primary/'batch_trace.csv');checks={}
    for field in ('config','init_hash','epoch_order_hashes','final_optimizer_hash','selected_epochs','selected_validation'):
        checks[field]=result[field]==old[field]
    checks['data_identity']=result['run_identity']['data_hash']==old['run_identity']['data_hash']
    checks['online_step_hashes']=result['step_hashes']==trace['online_model_hash'].tolist()
    checks['view_hashes']=result['view_hashes']==trace['view_hash'].tolist()
    checks['current_replay_verified']=not result['config']['replay_enabled'] or result.get('replay_verified') is True
    from .protocol import checkpoint_rows
    current_rows=[r for kind in result['selected_epochs'] for r in checkpoint_rows(result,kind)]
    old_rows=json.loads(Path(source['state_summary']).read_text())['checkpoints']
    index={(r['model_kind'],r['checkpoint']):r for r in old_rows};endpoint_checks=[]
    for row in current_rows:
        prior=index.get((row['model_kind'],row['checkpoint']),{})
        matches=all(field in prior and row[field]==prior[field] for field in
            ('epoch','global_step','model_hash','optimizer_hash','validation'))
        endpoint_checks.append(dict(kind=row['model_kind'],checkpoint=row['checkpoint'],matches=matches,
            current_model_hash=row['model_hash'],prior_model_hash=prior.get('model_hash')))
    checks['all_endpoint_identities']=len(current_rows)==len(old_rows)==len(index) and all(r['matches'] for r in endpoint_checks)
    receipt=dict(key=key,matches=all(checks.values()),checks=checks,endpoint_checks=endpoint_checks,
        prior_run_hash=old['run_hash'],current_run_hash=result['run_hash'],prior_source_hash=old['run_identity']['source_hash'],
        current_source_hash=result['run_identity']['source_hash'],prior_execution_commit=old['run_identity']['execution_commit'],
        current_execution_commit=result['run_identity']['execution_commit'],same_scientific_trajectory=all(checks.values()),
        prior_directory=str(directory),current_directory=result['output_dir'])
    write_json(Path(out)/'cross_attempt_identity'/f"{result['dataset']}_{result['trial']}_{result['seed']}.json",receipt)
    return receipt


def verify_recomputed_results(results,p,out):
    context=p.get('recovery')
    if context is None:return
    receipts=[verify_recomputed_trajectory(r,context,out) for r in results
        if f"{r['dataset']}|{r['trial']}|{r['seed']}" in context['runs']]
    matched=all(r['matches'] for r in receipts)
    write_json(Path(out)/'cross_attempt_identity.json',dict(status='PASS' if matched else 'MISMATCH',
        trajectories=receipts,verified_recomputations=len(receipts),prior_scientific_trajectories=20,
        scientific_repetitions_added=0,promotion_blocked=not matched,test_accessed=False))
    if not matched:raise RuntimeError('Cross-attempt trajectory identity mismatch; promotion blocked; independent review required')


def execution_costs(results,p):
    current={key:sum(r.get(key,0) for r in results) for key in COST_KEYS}
    prior=p.get('recovery',{}).get('prior_costs',{key:0 for key in COST_KEYS})
    context=p.get('recovery');overlap=0
    if context:overlap=sum(f"{r['dataset']}|{r['trial']}|{r['seed']}" in context['runs'] for r in results)
    unique=len(results)+(0 if context is None else context['prior_unique_scientific_trajectories']-overlap)
    return dict(prior_attempt=prior,current_attempt=current,combined={key:prior[key]+current[key] for key in COST_KEYS},
        prior_unique_scientific_trajectories=0 if context is None else context['prior_unique_scientific_trajectories'],
        verified_recomputed_trajectories=overlap,unique_scientific_trajectories=unique,
        current_complete_trajectories=len(results),wall_clock_start=p['task_start_unix'])
