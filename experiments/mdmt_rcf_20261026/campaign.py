"""Resident RAM-only campaign workers and registered stage decisions.

Workers own trained objects. Pipes carry commands and small result summaries;
no model, optimizer, or RNG snapshot is persisted or sent to the controller.
"""
from __future__ import annotations
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from scipy.special import expit
from .artifacts import _json,_sha,_plain,_metrics,_write_csv,export_bundle,verify_exports,paired_bootstrap,OUTER

S1_METHODS=('NATIVE','LOCAL','B_REFIT','H_DIRECT','ETA_DIRECT_MATCHED','H_DIRECT_EFFORT','GROUP_CAL','PILOT_PROB','PILOT_LOGIT','FULL_SOURCE','FULL_SOURCE_ALL','FULL_TARGET','RANDOM_FEATURE')
S2_METHODS=('RCF','POOLED_TARGET','LOCAL_ACTIVE')
S3_NONLINEAR=('ASSIGN_SHUFFLE','K1_WIDE','ANCHOR_OFF','SOURCE_INIT_RANDOM','SOURCE_LABEL_SHUFFLE')
S3_LINEAR=('SPECTRAL_K2','RRR_K2','UNGROUPED_RRR')
MINIMUM_TRAINING_SEEDS=4
MANDATORY_S3={'kuairand':('K1_WIDE','SOURCE_INIT_RANDOM')}

LOCKED_METHODS=('H_DIRECT','ETA_DIRECT_MATCHED','H_DIRECT_EFFORT','GROUP_CAL','FULL_SOURCE_ALL','LOCAL_ACTIVE','POOLED_TARGET','K1_WIDE','RCF')

def source_identity(config):
    root=Path(config['paths']['code']);files={str(p):_sha(p) for p in sorted(root.rglob('*.py')) if '__pycache__' not in str(p)}
    for name in ('SOURCE_MANIFEST.json','SCALE_SOURCE_MANIFEST.json'):
        p=root/'vendor_snapshot'/name
        if p.exists():files[str(p)]=_sha(p)
    payload=json.dumps(files,sort_keys=True,separators=(',',':'))
    return {'files':files,'source_hash':hashlib.sha256(payload.encode()).hexdigest(),'config_hash':config['config_hash']}

def build_run_manifest(config,audit):
    rows=[]
    for dataset in config['datasets']:
        for seed in config['seeds']['development']:
            for stage,methods in [('s0',('PREFIX',)),('s1',S1_METHODS),('s2',S2_METHODS)]:
                for method in methods:rows.append(dict(dataset=dataset,seed=seed,stage=stage,method=method,trigger='ALWAYS',status='NOT_RUN'))
            for method in S3_NONLINEAR+S3_LINEAR:rows.append(dict(dataset=dataset,seed=seed,stage='s3',method=method,trigger='FOUR_SEED_EVIDENCE_COMPLETION' if method in MANDATORY_S3.get(dataset,()) else 'registered four-seed S2 subset',status='NOT_RUN' if method in MANDATORY_S3.get(dataset,()) else 'NO_START_PENDING_S2'))
        for seed in config['seeds']['locked']:
            for method in ('PREFIX',)+LOCKED_METHODS:rows.append(dict(dataset=dataset,seed=seed,stage='s4',method=method,trigger='S3 retained contribution + practical gate',status='NO_START_PENDING_S3'))
    return dict(spec='RCF-v1',schema_version=1,identity=source_identity(config),audit=audit,runs=rows,storage='RAM_ONLY_MODEL_STATES',exposure='EXPOSED_BENCHMARK_REASSESSMENT',status='REGISTERED_NOT_RUN')

def _compare(candidate,reference,seeds,config):
    if len(set(seeds))<MINIMUM_TRAINING_SEEDS:return dict(passed=False,status='INCOMPLETE',minimum_training_seeds=MINIMUM_TRAINING_SEEDS,values=[])
    values=[];missing=[]
    for seed in seeds:
        if seed not in candidate or seed not in reference:missing.append(seed);continue
        a,b=candidate[seed],reference[seed]
        if a.get('macro_logloss') is None or b.get('macro_logloss') is None:missing.append(seed);continue
        values.append(dict(seed=seed,delta_ll=a['macro_logloss']-b['macro_logloss'],delta_auc=a['macro_auc']-b['macro_auc'],cells=[dict(domain=x['domain'],task=x['task'],delta_ll=x['logloss']-y['logloss'],delta_auc=x['auc']-y['auc']) for x,y in zip(a['cells'],b['cells'])]))
    if missing:return dict(passed=False,status='INCOMPLETE',missing_seeds=missing,values=values)
    p=config['promotion'];dll=np.array([v['delta_ll'] for v in values]);dauc=np.array([v['delta_auc'] for v in values]);cell=np.array([[c['delta_ll'] for c in v['cells']] for v in values])
    tests=dict(all_seeds_negative=bool((dll<0).all()),mean_ll=bool(dll.mean()<=p['mean_dll_max']),auc_protection=bool(dauc.mean()>=p['mean_dauc_min']),cell_protection=bool(cell.mean(0).max()<=p['max_cell_mean_dll']))
    return dict(passed=all(tests.values()),status='COMPLETE',values=values,mean_delta_ll=float(dll.mean()),mean_delta_auc=float(dauc.mean()),max_cell_mean_delta_ll=float(cell.mean(0).max()),tests=tests)

def decide_stage(stage,predictions,config):
    """Deterministic registered comparisons, with explicit competing triggers.

Input: {dataset: {seed: {'metrics':{method:Q_metrics},
                       'diagnostics':..., 'formation_records':...}}}.
    """
    out={};seeds=config['seeds']['development']
    if len(set(seeds))<MINIMUM_TRAINING_SEEDS:
        return dict(stage=stage,complete=False,minimum_training_seeds=MINIMUM_TRAINING_SEEDS,datasets={d:dict(status='BLOCKED_INCOMPLETE_STAGE',reason='Four distinct development training seeds required',triggered_methods=[],s4=False) for d in config['datasets']})
    for dataset in config['datasets']:
        entries=predictions.get(dataset,{})
        entries={int(k):v for k,v in entries.items()}
        missing=[s for s in seeds if s not in entries]
        if missing:out[dataset]=dict(status='BLOCKED_INCOMPLETE_STAGE',missing_seeds=missing,triggered_methods=[],s4=False);continue
        methods=sorted(set().union(*(set(entries[s].get('metrics',{})) for s in seeds)))
        ms={method:{s:entries[s]['metrics'][method] for s in seeds if method in entries[s].get('metrics',{})} for method in methods}
        comparisons={}
        for a in methods:
            for b in methods:
                if a!=b:comparisons[a+'__minus__'+b]=_compare(ms[a],ms[b],seeds,config)
        def passed(a,b):return comparisons.get(a+'__minus__'+b,{}).get('passed',False)
        strong=('H_DIRECT','H_DIRECT_EFFORT','FULL_SOURCE_ALL','POOLED_TARGET','LOCAL_ACTIVE')
        required=S1_METHODS+S2_METHODS if stage=='s2' else S1_METHODS+S2_METHODS
        absent={s:sorted(set(required)-set(entries[s].get('metrics',{}))) for s in seeds}
        absent={s:v for s,v in absent.items() if v}
        if absent:out[dataset]=dict(status='BLOCKED_INCOMPLETE_STAGE',missing_methods=absent,comparisons=comparisons,triggered_methods=[],s4=False);continue
        full=all(passed('RCF',ref) for ref in strong)
        any_strong_increment=any(passed(a,b) for a in S2_METHODS for b in ('H_DIRECT','H_DIRECT_EFFORT','FULL_SOURCE_ALL'))
        # Functional crossover is supplied by actual P_form-aligned diagnostic data.
        crossover_by_seed={s:_registered_crossovers(entries[s]) for s in seeds}
        def cross_id(r):return str(r.get('profile_id',r.get('id','')))
        sets=[{cross_id(r) for r in crossover_by_seed[s] if r.get('supported_vs_controls',False) and r.get('sufficient_support',False)} for s in seeds]
        repeated_crossovers=sorted(set.intersection(*sets)) if sets else []
        collapse=[]
        for s in seeds:
            records=entries[s].get('formation_records',{}).get('RCF',[])
            terminal=[r for r in records if 'c' in r or 'scale' in r]
            collapse.append(dict(seed=s,count=len(terminal),zero_count=sum(float(r.get('c',r.get('scale',1.)))==0 for r in terminal)))
        candidates=[];triggered=[];s4=False
        if full:
            status='FULL_PRACTICAL_INCREMENT';triggered=list(S3_NONLINEAR+S3_LINEAR);candidates.append('FULL_ATTRIBUTION')
        else:
            source_fit=any(all(passed(a,b) for b in ('H_DIRECT','H_DIRECT_EFFORT')) for a in ('RCF','POOLED_TARGET'))
            local_fit=all(passed('LOCAL_ACTIVE',b) for b in ('H_DIRECT','H_DIRECT_EFFORT'))
            if repeated_crossovers:
                candidates.append('MECHANISM_ONLY');triggered.append('K1_WIDE')
            if source_fit and not passed('RCF','POOLED_TARGET'):
                candidates.append('ORDINARY_ANCHORED_FINETUNING');triggered.extend(('K1_WIDE','SOURCE_INIT_RANDOM'))
            if local_fit and not passed('RCF','LOCAL_ACTIVE'):
                candidates.append('TARGET_REFITTING_EXPLANATION');triggered.append('SOURCE_INIT_RANDOM')
            if any(passed('RCF',b) for b in ('NATIVE','LOCAL','GROUP_CAL')) and not passed('RCF','FULL_SOURCE_ALL'):
                candidates.append('SIMPLE_READER_SUFFICIENT')
            if not any_strong_increment and not repeated_crossovers:
                status='NO_INCREMENT_IN_TESTED_SPACE';triggered=[]
            elif repeated_crossovers:status='MECHANISM_ONLY'
            elif 'ORDINARY_ANCHORED_FINETUNING' in candidates:status='ORDINARY_ANCHORED_FINETUNING'
            elif 'TARGET_REFITTING_EXPLANATION' in candidates:status='TARGET_REFITTING_EXPLANATION'
            else:status='SIMPLE_READER_SUFFICIENT'
        triggered=list(dict.fromkeys(triggered+list(MANDATORY_S3.get(dataset,()))))
        contribution={}
        if stage=='s3':
            prior=entries[seeds[0]].get('s2_decision',{})
            prior_methods=list(dict.fromkeys(list(prior.get('triggered_methods',triggered))+triggered+list(MANDATORY_S3.get(dataset,()))))
            absent_controls={s:sorted(set(prior_methods)-set(entries[s].get('metrics',{}))) for s in seeds}
            absent_controls={s:v for s,v in absent_controls.items() if v}
            if absent_controls:status='BLOCKED_INCOMPLETE_S3'
            elif full:
                attribution={name:passed('RCF',name) for name in prior_methods if name in methods}
                # Retained claims, without treating all components as mandatory success.
                contribution=dict(source=attribution.get('SOURCE_INIT_RANDOM',False) or attribution.get('SOURCE_LABEL_SHUFFLE',False),organization=attribution.get('ASSIGN_SHUFFLE',False),multifunction=attribution.get('K1_WIDE',False),nonlinear=attribution.get('SPECTRAL_K2',False) and attribution.get('RRR_K2',False))
                s4=any(contribution.values())
                status='S4_ELIGIBLE_WITH_RETAINED_CLAIMS' if s4 else 'NO_DISTINCT_RCF_CONTRIBUTION'
            else:contribution={};s4=False
            triggered=[]
        else:contribution={}
        out[dataset]=dict(status=status,full_practical_increment=full,triggered_methods=triggered,s4=s4,comparisons=comparisons,candidate_branches=candidates,repeated_crossovers=repeated_crossovers,crossovers_by_seed=crossover_by_seed,collapse=collapse,retained_contribution=contribution,training_seeds=list(seeds),minimum_training_seeds=MINIMUM_TRAINING_SEEDS,mandatory_evidence_completion=list(MANDATORY_S3.get(dataset,())),missing_control_methods=absent_controls if stage=='s3' else {},selection_basis='Q_explore four-seed registered comparisons; no exposed test',no_start_reason=None if triggered else 'No additional registered experiment justified by supported claim')
    return dict(stage=stage,datasets=out,complete=all(not v['status'].startswith('BLOCKED') for v in out.values()))

def _simple_prediction(bundle,method):
    from .protocol import PredictionBundle
    from .baselines import validate_raw_predictor,predict_direct_models
    if method=='B_REFIT':logits=bundle.base_logits
    elif method=='NATIVE':logits={r:v.z for r,v in bundle.views.items()}
    elif method=='LOCAL':
        logits={}
        for r,v in bundle.views.items():
            a=v.z.copy()
            for d in range(3):
                mask=v.domain==d
                for q in range(2):a[mask,q]+=bundle.pilots.residual(r,d,q)[mask]
            logits[r]=a
    else:raise ValueError(method)
    p={r:expit(a) for r,a in logits.items()}
    terminal={r:expit(a) for r,a in bundle.diagnostics.get('native_terminal_logits',logits).items()} if method=='NATIVE' else p
    result=PredictionBundle(method,p,terminal,{'selection':'fixed prefix state'}, {'eta':method=='LOCAL'},dict(bundle.identity,method=method))
    def calculate(role,h,z,source):
        if method=='NATIVE':return expit(z)
        if method=='B_REFIT':return expit(predict_direct_models(bundle,bundle.base_models,role,raw_h=h,raw_z=z))
        domain=bundle.views[role].domain
        return expit(z+source[np.arange(len(h)),domain,:])
    result.diagnostics['_raw_x_validate']=lambda b:validate_raw_predictor(b,result,calculate)
    from .memory_data import map_prediction
    return map_prediction(bundle,result)

def _small_diagnostics(value):
    return {m:{k:v for k,v in d.items() if k not in ('single_function_predictions','requests')} for m,d in value.items()}

def _summary(bundle,predictions,formations,diagnostics,config=None):
    v=bundle.views['Q_explore'];bundle.ledger.access('stage_decision','Q_explore',rows=v.row_id,purpose='evaluate')
    delivery={}
    if config is not None:
        from .delivery import collect_delivery
        delivery=collect_delivery(bundle,predictions,formations,diagnostics,config,benchmark=False)
    return dict(delivery=delivery,dataset=bundle.dataset,seed=bundle.seed,metrics={m:_metrics(v.y,p.probabilities['Q_explore'],v.domain) for m,p in predictions.items()},group_metrics=_group_summaries(v,predictions),diagnostics=_plain(_small_diagnostics(diagnostics)),formation_records={m:_plain(f.records) for m,f in formations.items()},identity=bundle.identity)

def _group_summaries(view,predictions):
    out={}
    for method,pred in predictions.items():
        rows=[]
        for t in range(3):
            for q in range(2):
                for g in range(2):
                    m=(view.domain==t)&(view.group[:,q]==g);yy=view.y[m,q];pp=np.clip(pred.probabilities['Q_explore'][m,q].astype(float),1e-7,1-1e-7)
                    rows.append(dict(target=t,task=q,group=g,n=int(m.sum()),positive=int(yy.sum()),negative=int(len(yy)-yy.sum()),logloss=float(np.mean(-yy*np.log(pp)-(1-yy)*np.log1p(-pp))) if len(yy) else None))
        out[method]=rows
    return out

def _registered_crossovers(entry):
    diag=entry.get('diagnostics',{}).get('RCF',{});raw=diag.get('crossovers',[]);profiles=diag.get('group_profiles',[]);out=[]
    def key(r):return (r['source'],r['task'],r['function_a'],r['function_b'],r['target_a'],r['group_a'],r['target_b'],r['group_b'])
    pf={key(r):r for r in raw if r.get('role')=='P_form'}
    for r in raw:
        if r.get('role')!='Q_explore' or not r.get('cross_advantage') or not r.get('sufficient_support'):continue
        anchor=pf.get(key(r))
        if not anchor or anchor['delta_a']==0:continue
        sign=1 if anchor['delta_a']<0 else -1
        oriented=(sign*r['delta_a'],sign*r['delta_b'])
        record=dict(r,profile_id=str((r['source'],r['task'],r['target_a'],r['group_a'],r['target_b'],r['group_b'],oriented[0]<0,oriented[1]<0)),alignment='P_form delta_a orientation only',supported_vs_controls=True)
        for target,group in ((r['target_a'],r['group_a']),(r['target_b'],r['group_b'])):
            choices=[p for p in profiles if p.get('role')=='Q_explore' and p.get('task')==r['task'] and p.get('target')==target and p.get('group')==group and p.get('action')=='FIXED_PLUS' and p.get('function_id') in (r['function_a'],r['function_b'])]
            if len(choices)!=2:record['supported_vs_controls']=False;continue
            winner=min(choices,key=lambda p:p['logloss'])
            if winner.get('delta_ll_vs_group_cal',0)>=0:record['supported_vs_controls']=False
            for method in ('POOLED_TARGET','LOCAL_ACTIVE'):
                ref=next((p for p in entry.get('group_metrics',{}).get(method,[]) if p['target']==target and p['task']==r['task'] and p['group']==group),None)
                if not ref or ref['logloss'] is None or winner['logloss']>=ref['logloss']:record['supported_vs_controls']=False
        out.append(record)
    return out

def host_memory_snapshot(pid=None):
    values={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value,*_=line.split()
        if key in ('MemAvailable:','MemTotal:'):values[key[:-1].lower()+'_bytes']=int(value)*1024
    try:
        for line in Path(f'/proc/{pid or os.getpid()}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):values['rss_bytes']=int(line.split()[1])*1024
    except FileNotFoundError:values['rss_bytes']=0
    return values


def _verify_replayed_prefix(bundle,config,prior=None):
    checks=[]
    candidates=[]
    if prior is not None:candidates.append(('same_artifact_prior_completion',prior))
    old_root=config.get('recovery',{}).get('parent_artifact')
    if old_root and bundle.seed in (5201,5202):
        old=Path(old_root)/'prefix'/bundle.dataset/f'seed{bundle.seed}'/'PREFIX_COMPLETE.json'
        if not old.exists():raise FileNotFoundError('Registered replay evidence missing: '+str(old))
        candidates.append((str(old),json.loads(old.read_text())))
    for source,receipt in candidates:
        expected=receipt['identity']
        keys=('native_hash','native_epoch24_hash','pilot_hash','selected_epoch')
        mismatch={k:dict(expected=expected.get(k),actual=bundle.identity.get(k)) for k in keys if expected.get(k)!=bundle.identity.get(k)}
        if mismatch:raise RuntimeError('Prefix replay changed trained values: '+str(mismatch))
        checks.append(dict(source=source,matching_fields=list(keys)))
    result=dict(status='PASS' if checks else 'NEW_REGISTERED_SEED',checks=checks,statistical_seed=bundle.seed,physical_replay_is_additional_seed=False,identity=bundle.identity)
    _json(Path(bundle.output_root)/'REPLAY_IDENTITY_CHECK.json',result)
    return result


def _method_completed(bundle,predictions,method,stage,config):
    from .memory_data import map_prediction,release_data_pages
    map_prediction(bundle,predictions[method])
    _json(Path(config['paths']['artifact'])/'runs'/bundle.dataset/f'seed{bundle.seed}'/method/'TRAINING_COMPLETE.json',dict(status='COMPLETE',stage=stage,method=method,statistical_seed=bundle.seed,identity=predictions[method].identity,prefix_identity=bundle.identity,selected_policy=predictions[method].selection,model_storage='RAM_ONLY'))
    release_data_pages(bundle)


def _worker(conn,dataset,seed,config,root,device):
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    from .prefix import build_prefix
    bundle=None;predictions={};formations={};diagnostics={};demand=None;frozen=False;s2_decision={}
    while True:
        command=conn.recv();op=command['op'];started=time.monotonic()
        try:
            if op=='prefix':
                if bundle is None:
                    prior_path=Path(root)/'prefix'/dataset/f'seed{seed}'/'PREFIX_COMPLETE.json'
                    prior=json.loads(prior_path.read_text()) if prior_path.exists() else None
                    bundle=build_prefix(dataset,seed,config,root,device=device,resume=True)
                    replay=_verify_replayed_prefix(bundle,config,prior)
                else:replay={'status':'ALREADY_RESIDENT'}
                result={'status':'PREFIX_RESIDENT','identity':bundle.identity,'dataset':dataset,'seed':seed,'replay_check':replay}
            elif op=='capture-fixture':
                from .baselines import fit_base
                from .fixture_export import export_reader_fixture
                if bundle is None:raise RuntimeError('A resident prefix is required for fixture capture')
                fit_base(bundle,config)
                result=export_reader_fixture(bundle,config)
            elif op=='reload-readers':
                from . import readers
                import importlib,marshal,hashlib
                from .fixture_export import live_model_hashes
                token=command['authorization']
                from .state import sha_file
                expected=token.get('new_readers_sha256')
                if token.get('config_hash')!=config['config_hash'] or token.get('verdict') not in ('PASS','PASS WITH FIXES') or token.get('blocking_issues',[]) or expected!=sha_file(readers.__file__):raise PermissionError('Reader reload authorization differs')
                before=live_model_hashes(bundle);old_code=hashlib.sha256(marshal.dumps(readers.fit_group.__code__)).hexdigest();old_dict=readers.__dict__.copy()
                try:importlib.invalidate_caches();importlib.reload(readers)
                except BaseException:
                    readers.__dict__.clear();readers.__dict__.update(old_dict);raise
                after=live_model_hashes(bundle)
                if before!=after:raise RuntimeError('Reader reload modified a trained model')
                result=dict(status='READERS_RELOADED',dataset=dataset,seed=seed,completed_s1_methods=sorted(predictions),models_unchanged=True,old_fit_group_code_hash=old_code,new_fit_group_code_hash=hashlib.sha256(marshal.dumps(readers.fit_group.__code__)).hexdigest(),new_readers_sha256=expected,protocol_prefix_reloaded=False,model_files_written=0)
            elif op=='s1':
                from .baselines import fit_base,fit_direct_baselines,fixed_feature_bank,fit_probability_reader
                from .readers import fit_reader
                if bundle is None:raise RuntimeError('Prefix model not resident; replay explicitly required')
                fit_base(bundle,config)
                for m in ('NATIVE','LOCAL','B_REFIT'):
                    if m not in predictions:predictions[m]=_simple_prediction(bundle,m)
                    _method_completed(bundle,predictions,m,'s1',config)
                direct_names=('H_DIRECT','ETA_DIRECT_MATCHED','H_DIRECT_EFFORT')
                present=[m for m in direct_names if m in predictions]
                if present and len(present)!=3:raise RuntimeError('Partial completed direct family cannot be silently retrained')
                if not present:predictions.update(fit_direct_baselines(bundle,config))
                for m in direct_names:_method_completed(bundle,predictions,m,'s1',config)
                if 'PILOT_PROB' not in predictions:predictions['PILOT_PROB']=fit_probability_reader(bundle,config)
                _method_completed(bundle,predictions,'PILOT_PROB','s1',config)
                for m in ('GROUP_CAL','PILOT_LOGIT','FULL_SOURCE','FULL_SOURCE_ALL','FULL_TARGET','RANDOM_FEATURE'):
                    if m in predictions:continue
                    fb=fixed_feature_bank(bundle,m,config)
                    predictions[m]=fit_reader(bundle,fb,m,config,train_roles=('P_form','P_read') if m=='FULL_SOURCE_ALL' else ('P_read',))
                    _method_completed(bundle,predictions,m,'s1',config)
                    del fb;gc.collect()
                result=_summary(bundle,predictions,formations,diagnostics,config)
            elif op=='s2':
                from .demand import compute_demand
                from .formation import form
                from .readers import fit_reader
                from .mechanisms import diagnose_functions
                if demand is None:demand=compute_demand(bundle,config)
                for m in S2_METHODS:
                    if m not in formations:formations[m]=form(bundle,demand,m,config)
                    if m not in predictions:predictions[m]=fit_reader(bundle,formations[m].features,m,config)
                    if m not in diagnostics:diagnostics[m]=diagnose_functions(bundle,formations[m],predictions[m],config)
                    _method_completed(bundle,predictions,m,'s2',config)
                result=_summary(bundle,predictions,formations,diagnostics,config)
            elif op=='s3':
                from .formation import form
                from .readers import fit_reader
                from .mechanisms import diagnose_functions,spectral_features,fit_rrr
                s2_decision=command['decision']
                for m in command['methods']:
                    if m in S3_NONLINEAR:
                        if m not in formations:formations[m]=form(bundle,demand,m,config,paired=formations['RCF'])
                        if m not in predictions:predictions[m]=fit_reader(bundle,formations[m].features,m,config)
                        if m not in diagnostics:diagnostics[m]=diagnose_functions(bundle,formations[m],predictions[m],config)
                    elif m=='SPECTRAL_K2':
                        if m not in predictions:predictions[m]=fit_reader(bundle,spectral_features(bundle,demand,config),m,config)
                    elif m in ('RRR_K2','UNGROUPED_RRR'):
                        if m not in predictions:
                            features=fit_rrr(bundle,demand,config,ungrouped=m=='UNGROUPED_RRR')
                            predictions[m]=fit_reader(bundle,features,m,config)
                    else:raise ValueError('Unregistered S3 method '+m)
                    _method_completed(bundle,predictions,m,'s3',config)
                result=_summary(bundle,predictions,formations,diagnostics,config);result['s2_decision']=s2_decision
            elif op=='locked':
                from .baselines import fit_base,fit_direct_baselines,fixed_feature_bank
                from .readers import fit_reader
                from .demand import compute_demand
                from .formation import form
                from .mechanisms import diagnose_functions
                fit_base(bundle,config)
                for m in ('NATIVE','LOCAL','B_REFIT'):
                    if m not in predictions:predictions[m]=_simple_prediction(bundle,m)
                    _method_completed(bundle,predictions,m,'locked',config)
                direct_names=('H_DIRECT','ETA_DIRECT_MATCHED','H_DIRECT_EFFORT')
                present=[m for m in direct_names if m in predictions]
                if present and len(present)!=3:raise RuntimeError('Partial completed direct family cannot be silently retrained')
                if not present:predictions.update(fit_direct_baselines(bundle,config))
                for m in direct_names:_method_completed(bundle,predictions,m,'locked',config)
                for m in ('GROUP_CAL','FULL_SOURCE_ALL'):
                    if m not in predictions:predictions[m]=fit_reader(bundle,fixed_feature_bank(bundle,m,config),m,config,train_roles=('P_form','P_read') if m=='FULL_SOURCE_ALL' else ('P_read',))
                    _method_completed(bundle,predictions,m,'locked',config)
                if demand is None:demand=compute_demand(bundle,config)
                for m in tuple(dict.fromkeys(S2_METHODS+('K1_WIDE',command['source_intervention']))):
                    if m not in formations:formations[m]=form(bundle,demand,m,config,paired=formations.get('RCF') if m not in S2_METHODS else None)
                    if m not in predictions:predictions[m]=fit_reader(bundle,formations[m].features,m,config)
                    if m not in diagnostics:diagnostics[m]=diagnose_functions(bundle,formations[m],predictions[m],config)
                    _method_completed(bundle,predictions,m,'locked',config)
                result=_summary(bundle,predictions,formations,diagnostics,config)
            elif op=='freeze':
                manifest=command['manifest'];bundle.ledger.freeze_outer(manifest)
                frozen=True;result=dict(status='FROZEN',identity=bundle.identity,methods=sorted(predictions))
            elif op=='export':
                from .mechanisms import diagnose_functions
                if not frozen:raise PermissionError('Worker is not globally frozen')
                cfg=dict(config,_outer_authorized=True)
                from .memory_data import release_data_pages
                for m,f in formations.items():
                    diagnostics[m]=diagnose_functions(bundle,f,predictions[m],cfg)
                    release_data_pages(bundle)
                for m,d in _small_diagnostics(diagnostics).items():_json(Path(root)/'runs'/dataset/f'seed{seed}'/m/'mechanisms.json',d)
                from .delivery import collect_delivery
                collect_delivery(bundle,predictions,formations,diagnostics,config,benchmark=True)
                index=export_bundle(bundle,dict(predictions=predictions,formations=formations,demand=demand,diagnostics=diagnostics,outer_authorized=True),root,config)
                acceptance=verify_exports(index,config);_json(Path(index['root'])/'EXPORT_ACCEPTANCE.json',acceptance)
                if acceptance['status']!='PASS':raise RuntimeError('Export verification failed: '+str(acceptance))
                result=dict(index=index,acceptance=acceptance,summary=_summary(bundle,predictions,formations,diagnostics,config))
            elif op=='bootstrap':
                if not frozen:raise PermissionError('Need global freeze before bootstrapping outer roles')
                from .protocol import derive_rng
                records=[]
                for role in ('Q_explore',)+OUTER:
                    v=bundle.views[role];users=None if bundle.data.user_ids is None else bundle.data.user_ids[v.row_id]
                    for ref in command.get('references',('H_DIRECT','FULL_SOURCE_ALL','POOLED_TARGET','LOCAL_ACTIVE')):
                        if 'RCF' not in predictions or ref not in predictions:continue
                        rng=derive_rng(seed,dataset,'bootstrap',unit=list(S1_METHODS+S2_METHODS).index(ref))
                        stat=paired_bootstrap(v.y,predictions['RCF'].probabilities[role],predictions[ref].probabilities[role],v.domain,users,rng,1000)
                        reps=stat.pop('replicate_deltas');_npz_path=Path(root)/'runs'/dataset/f'seed{seed}'/'bootstrap'/f'{role}_RCF_minus_{ref}.npz'
                        from .artifacts import _npz
                        _npz(_npz_path,deltas=reps)
                        records.append(dict(role=role,candidate='RCF',reference=ref,**stat))
                _json(Path(root)/'runs'/dataset/f'seed{seed}'/'BOOTSTRAP.json',records);result={'status':'COMPLETE','comparisons':len(records)}
            elif op=='summary':result=_summary(bundle,predictions,formations,diagnostics,config)
            elif op=='release-data':result=dict(status='DATA_PAGES_RELEASED')
            elif op=='shutdown':
                if bundle is not None and not command.get('abort',False) and not command.get('export_verified',False):raise PermissionError('Cannot release model before export acceptance')
                conn.send({'ok':True,'result':{'status':'RELEASED_MEMORY','dataset':dataset,'seed':seed}});break
            else:raise ValueError(op)
            if bundle is not None:
                from .memory_data import release_data_pages,data_cache_receipt
                release_data_pages(bundle)
                _json(Path(bundle.output_root)/'ROW_DATA_CACHE.json',data_cache_receipt(bundle))
                import resource
                result['resource_snapshot']=dict(host_memory_snapshot(),peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
            conn.send(dict(ok=True,result=result,seconds=time.monotonic()-started))
        except BaseException as exc:
            conn.send(dict(ok=False,error=repr(exc),traceback=traceback.format_exc(),op=op,dataset=dataset,seed=seed))
    conn.close()

def gpu_snapshot():
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,power.draw,power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        out=[]
        for line in p.stdout.splitlines():
            i,m,mt,power,limit=[float(v.strip()) for v in line.split(',')]
            out.append(dict(index=int(i),memory_used=m,memory_total=mt,power=power,power_limit=limit,eligible=m/mt<.85 and power/limit<.85))
        return out
    except (OSError,subprocess.CalledProcessError,ValueError):return []

class Campaign:
    def __init__(self,config,*,devices=None):
        self.config=config;self.root=Path(config['paths']['artifact']).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.devices=devices or ['cuda:'+str(g['index']) for g in gpu_snapshot()]
        self.workers={};self.summaries={};self.decisions={};self.audit={};self.indexes=[];self.frozen=None;self.ctx=mp.get_context('spawn');self.prefix_peak_bytes={}
        self.identity=source_identity(config)
    def write_status(self,stage,status,**extra):
        _json(self.root/'CAMPAIGN_STATUS.json',dict(stage=stage,status=status,identity=self.identity,resident_prefixes=[list(k) for k in self.workers],updated_unix=time.time(),**extra))
    def require_review(self,scope='all'):
        token=self.root/('REVIEW_AUTHORIZATION_S0.json' if scope=='s0' else 'REVIEW_AUTHORIZATION.json')
        if not token.exists():raise PermissionError('Independent reviewer PASS authorization is required before '+scope+' training')
        r=json.loads(token.read_text())
        if r.get('verdict') not in ('PASS','PASS WITH FIXES') or r.get('config_hash')!=self.config['config_hash'] or r.get('blocking_issues',[]):raise PermissionError('Review authorization config/verdict/blocking issues invalid')
        if r.get('files'):
            for path,expected in r['files'].items():
                if not Path(path).exists() or _sha(path)!=expected:raise PermissionError('Reviewed source changed: '+path)
        elif r.get('source_hash')!=source_identity(self.config)['source_hash']:
            raise PermissionError('Review source hash mismatch')
        if scope=='all':
            actual=source_identity(self.config)
            if actual!=self.identity:
                _json(self.root/'loaded_identity_updates'/f'{time.time_ns()}.json',dict(previous=self.identity,current_disk_source=actual,note='Normal stage boundary review accepted new source; retained prefix/model identities are unchanged. Explicit reader reload records loaded bytecode separately.'))
                self.identity=actual
        if scope=='s0' and r.get('scope','s0') not in ('s0','all'):raise PermissionError('Review scope mismatch')
    def _spawn(self,dataset,seed,device):
        key=(dataset,int(seed));parent,child=self.ctx.Pipe()
        p=self.ctx.Process(target=_worker,args=(child,dataset,int(seed),self.config,str(self.root),device),daemon=False);p.start();child.close()
        self.workers[key]=dict(process=p,conn=parent,device=device,pending=None)
        return self.workers[key]
    def _send(self,key,op,**args):
        worker=self.workers[key]
        if worker['pending']:raise RuntimeError('Worker busy')
        worker['conn'].send(dict(op=op,**args));worker['pending']=op
    def _recv(self,key):
        worker=self.workers[key]
        while not worker['conn'].poll(10):
            if not worker['process'].is_alive():raise RuntimeError(f'Worker {key} lost; deterministic replay required, no checkpoint exists')
        try:r=worker['conn'].recv()
        except EOFError:
            worker['pending']=None
            loss=dict(dataset=key[0],seed=key[1],pid=worker['process'].pid,exitcode=worker['process'].exitcode,reason='Worker pipe EOF; RAM model state unavailable; explicit same-seed replay required',model_checkpoint_exists=False)
            _json(self.root/'failures'/f'{key[0]}_{key[1]}_WORKER_LOST.json',loss)
            raise RuntimeError(str(loss)) from None
        worker['pending']=None
        if not r['ok']:
            _json(self.root/'failures'/f'{key[0]}_{key[1]}_{int(time.time())}.json',r);raise RuntimeError(r['traceback'])
        worker['last_resources']=r['result'].get('resource_snapshot',{})
        return r['result']
    def _remember(self,key,result):
        old_root=self.config.get('recovery',{}).get('parent_artifact')
        if old_root and key[1] in (5201,5202):
            old=Path(old_root)/'summaries'/f'{key[0]}_{key[1]}.json'
            if old.exists():
                prior=json.loads(old.read_text());common=sorted(set(prior.get('metrics',{}))&set(result.get('metrics',{})))
                equal={m:prior['metrics'][m]==result['metrics'][m] for m in common}
                _json(self.root/'replay_comparisons'/f'{key[0]}_{key[1]}.json',dict(status='PASS' if all(equal.values()) else 'DIFFERENCE_REQUIRES_REVIEW',Q_metrics_exact_equal=equal,source=str(old),new_config_and_source_identity=True,outer_metrics_used=False))
                if not all(equal.values()):raise RuntimeError('Same-seed Q predictions/metrics differ after data-only recovery; inspect replay comparisons')
        self.summaries.setdefault(key[0],{})[key[1]]=result
        _json(self.root/'summaries'/f'{key[0]}_{key[1]}.json',result)
    def _memory_admission(self,dataset,active):
        memory=host_memory_snapshot();reserve=int(self.config.get('recovery',{}).get('host_reserve_gib',8)*2**30)
        peak=max(int(self.config.get('recovery',{}).get('prefix_working_set_gib',{}).get(dataset,6)*2**30),int(1.15*self.prefix_peak_bytes.get(dataset,0)))
        pending=0
        for key in active:
            worker=self.workers[key];rss=host_memory_snapshot(worker['process'].pid).get('rss_bytes',0)
            other_peak=max(int(self.config.get('recovery',{}).get('prefix_working_set_gib',{}).get(key[0],6)*2**30),int(1.15*self.prefix_peak_bytes.get(key[0],0)))
            pending+=max(0,other_peak-rss)
        return dict(eligible=memory['memavailable_bytes']-pending>=reserve+peak,available_bytes=memory['memavailable_bytes'],pending_reserved_bytes=pending,new_worker_working_set_bytes=peak,system_reserve_bytes=reserve)
    def prefix(self,datasets=None,seeds=None):
        self.require_review('s0');datasets=datasets or list(self.config['datasets']);seeds=seeds or self.config['seeds']['development']
        if any(d not in self.config['datasets'] for d in datasets) or any(s not in self.config['seeds']['development']+self.config['seeds']['locked'] for s in seeds):raise ValueError('Unregistered dataset/seed')
        if self.frozen is not None:raise PermissionError('Frozen campaign cannot add or replay a training prefix')
        for key in [(d,int(s)) for d in datasets for s in seeds]:
            if key in self.workers and not self.workers[key]['process'].is_alive():
                w=self.workers.pop(key);w['conn'].close();w['process'].join(timeout=0)
                self.decisions.clear();self.summaries.get(key[0],{}).pop(key[1],None)
        queue=[(d,int(s)) for s in seeds for d in datasets if (d,int(s)) not in self.workers];active=[];completed=[]
        while queue or active:
            snapshots={f"cuda:{g['index']}":g for g in gpu_snapshot()}
            for device in self.devices:
                if not queue:break
                if device!='cpu' and not snapshots.get(device,{}).get('eligible',False):continue
                # Observe the first actual prefix peak per dataset before admitting more of that dataset.
                if queue[0][0] not in self.prefix_peak_bytes and any(k[0]==queue[0][0] for k in active):break
                admission=self._memory_admission(queue[0][0],active)
                if not admission['eligible']:
                    self.write_status('s0','WAITING_HOST_MEMORY',admission=admission,queued=[list(q) for q in queue],active=[list(k) for k in active])
                    if not active:raise MemoryError('Insufficient host RAM with required 8GiB reserve; no worker launched. A registered batch replay plan is required: '+str(admission))
                    break
                key=queue.pop(0);self._spawn(*key,device);self._send(key,'prefix');active.append(key)
                self.write_status('s0','RUNNING',admission=admission,queued=[list(q) for q in queue],active=[list(k) for k in active])
            progressed=False
            for key in active[:]:
                w=self.workers[key]
                if w['conn'].poll():
                    item=self._recv(key);completed.append(item)
                    self.prefix_peak_bytes[key[0]]=max(self.prefix_peak_bytes.get(key[0],0),item.get('resource_snapshot',{}).get('peak_rss_bytes',0))
                    active.remove(key);progressed=True
                elif not w['process'].is_alive():raise RuntimeError(f'Prefix worker died: {key}; explicit same-seed replay required')
            if not progressed and (queue or active):time.sleep(5)
        self.write_status('s0','COMPLETE',prefixes=completed);return completed
    def capture_fixture(self,dataset='movielens',seed=5201):
        self.require_review('s0');self.prefix([dataset],[int(seed)])
        key=(dataset,int(seed));self._send(key,'capture-fixture');result=self._recv(key)
        self.write_status('reader-fixture','EXPORTED',fixture=result);return result
    def reload_readers(self):
        self.require_review()
        path=self.root/'READER_RELOAD_AUTHORIZATION.json'
        if not path.exists():raise PermissionError('Dedicated reader numerical reload authorization required')
        token=json.loads(path.read_text())
        from . import readers
        if token.get('config_hash')!=self.config['config_hash'] or token.get('verdict') not in ('PASS','PASS WITH FIXES') or token.get('blocking_issues',[]) or token.get('new_readers_sha256')!=_sha(readers.__file__):raise PermissionError('Reader reload authorization is invalid')
        results=[]
        for key in self.workers:
            self._send(key,'reload-readers',authorization=token);results.append(self._recv(key))
        _json(self.root/'READER_RELOAD_RECEIPT.json',dict(results=results,identity=self.identity));return results
    def phase(self,phase,*,methods_by_dataset=None):
        self.require_review();results=[]
        for key,w in self.workers.items():
            if w['process'].is_alive() and w['pending'] is None:
                self._send(key,'release-data');self._recv(key)
        keys=[k for k in self.workers if k[1] in self.config['seeds']['development']]
        for key in keys:
            methods=(methods_by_dataset or {}).get(key[0],[])
            if phase=='s3' and not methods:continue
            self._send(key,phase,**({'methods':methods,'decision':self.decisions['s2']['datasets'][key[0]]} if phase=='s3' else {}))
            result=self._recv(key);self._remember(key,result);results.append(result)
            self.write_status(phase,'RUNNING',completed=[r['identity'] for r in results])
        self.write_status(phase,'COMPLETE');return results
    def decide(self,stage):
        decision=decide_stage(stage,self.summaries,self.config);self.decisions[stage]=decision
        _json(self.root/f'{stage.upper()}_DECISION.json',decision);return decision
    def diagnose(self):
        if 's2' not in self.decisions:self.decide('s2')
        if not self.decisions['s2']['complete']:raise PermissionError('Four-seed S2 must complete before any S3 decision')
        methods={d:r['triggered_methods'] for d,r in self.decisions['s2']['datasets'].items()}
        return self.phase('s3',methods_by_dataset=methods)
    def four_seed_inventory(self):
        rows=[];failures=[]
        for dataset in self.config['datasets']:
            cohorts=[('development',list(self.config['seeds']['development']))]
            if self.decisions.get('s3',{}).get('datasets',{}).get(dataset,{}).get('s4'):
                cohorts.append(('locked',list(self.config['seeds']['locked'])))
            for cohort,seeds in cohorts:
                expected=set(S1_METHODS+S2_METHODS) if cohort=='development' else set(('NATIVE','LOCAL','B_REFIT')+LOCKED_METHODS)
                if cohort=='development':expected.update(MANDATORY_S3.get(dataset,()))
                for seed in seeds:expected.update(self.summaries.get(dataset,{}).get(seed,{}).get('metrics',{}))
                for method in sorted(expected):
                    complete=[];missing=[]
                    for seed in seeds:
                        summary=self.summaries.get(dataset,{}).get(seed,{})
                        receipt=self.root/'runs'/dataset/f'seed{seed}'/method/'TRAINING_COMPLETE.json'
                        prefix=self.root/'prefix'/dataset/f'seed{seed}'/'PREFIX_COMPLETE.json'
                        valid=method in summary.get('metrics',{}) and receipt.exists() and prefix.exists()
                        if valid:
                            rec=json.loads(receipt.read_text());pre=json.loads(prefix.read_text())
                            valid=(rec.get('status')=='COMPLETE' and rec.get('statistical_seed')==seed and rec.get('prefix_identity',{}).get('config_hash')==self.config['config_hash'] and pre.get('identity',{}).get('config_hash')==self.config['config_hash'] and pre.get('native_epochs')==24)
                        (complete if valid else missing).append(seed)
                    row=dict(dataset=dataset,cohort=cohort,method=method,training_seeds=complete,count=len(set(complete)),required_seeds=seeds,missing_seeds=missing)
                    rows.append(row)
                    if missing or len(set(complete))<MINIMUM_TRAINING_SEEDS:failures.append(row)
        scale_ref=self.config.get('recovery',{}).get('scale_reuse');scale=[]
        if scale_ref:
            frozen=json.loads(Path(scale_ref).read_text())
            for dataset in self.config['datasets']:
                records=[r for r in frozen['runs'] if r['dataset']==dataset]
                complete=[]
                for row in records:
                    receipt=Path(row['receipt'])
                    if receipt.exists() and _sha(receipt)==row['sha256']:complete.append(int(row['seed']))
                item=dict(dataset=dataset,method='Dconst01_NO_AUG',training_seeds=sorted(set(complete)),count=len(set(complete)),source_manifest=scale_ref,source_config_hash=frozen.get('config_hash'),reused_without_retraining=True)
                scale.append(item)
                if len(set(complete))<MINIMUM_TRAINING_SEEDS:failures.append(item)
        result=dict(status='PASS' if not failures else 'BLOCKED_INCOMPLETE',minimum_distinct_training_seeds=MINIMUM_TRAINING_SEEDS,conditions=rows,scale=scale,failures=failures,config_hash=self.config['config_hash'],analysis_authorized=not failures)
        _json(self.root/'FOUR_SEED_ACCEPTANCE.json',result)
        return result
    def freeze(self):
        if len(set(self.config['seeds']['development']))<MINIMUM_TRAINING_SEEDS:raise PermissionError('Four distinct training seeds required before final freeze')
        self.decide('s2');self.decide('s3')
        if 's2' not in self.decisions:raise PermissionError('S2 decision incomplete')
        if not self.decisions['s2']['complete'] or 's3' not in self.decisions or not self.decisions['s3']['complete']:raise PermissionError('Cannot freeze incomplete triggered stage')
        for d,decision in self.decisions['s3']['datasets'].items():
            if decision.get('s4') and any(s not in self.summaries.get(d,{}) for s in self.config['seeds']['locked']):raise PermissionError('Triggered locked seed inventory incomplete')
        seed_acceptance=self.four_seed_inventory()
        if seed_acceptance['status']!='PASS':raise PermissionError('At least one method lacks four actual training completion receipts: '+str(seed_acceptance['failures']))
        manifest=dict(identity=self.identity,four_seed_acceptance_sha256=_sha(self.root/'FOUR_SEED_ACCEPTANCE.json'),decisions=self.decisions,methods={f'{d}:{s}':sorted(v['metrics']) for d,ss in self.summaries.items() for s,v in ss.items()},outer_roles=list(OUTER),frozen_unix=time.time())
        # Existing freeze is immutable; a changed method inventory needs a new explicit version.
        p=self.root/'FROZEN_MANIFEST.json'
        if p.exists():
            old=json.loads(p.read_text())
            if old['identity']!=manifest['identity'] or old['methods']!=manifest['methods']:raise ValueError('Frozen method inventory changed')
            manifest=old
        else:_json(p,manifest)
        self.frozen=manifest
        for key in self.workers:self._send(key,'freeze',manifest=manifest);self._recv(key)
        self.write_status('freeze','COMPLETE');return manifest
    def run_locked(self):
        if len(set(self.config['seeds']['locked']))<MINIMUM_TRAINING_SEEDS:raise PermissionError('Locked experiments require at least four distinct training seeds')
        if 's3' not in self.decisions or not self.decisions['s3']['complete']:raise PermissionError('Need complete S3 decision before locked runs')
        datasets=[d for d,r in self.decisions['s3']['datasets'].items() if r.get('s4')]
        if not datasets:return dict(status='NO_START',reason='No dataset passed the registered retained-contribution decision')
        locked_manifest=dict(identity=source_identity(self.config),datasets=datasets,seeds=self.config['seeds']['locked'],methods=list(LOCKED_METHODS),source_intervention='SOURCE_INIT_RANDOM',selection='fixed from completed S3 contribution decision; no locked/test tuning')
        _json(self.root/'LOCKED_METHOD_MANIFEST.json',locked_manifest)
        self.prefix(datasets,self.config['seeds']['locked'])
        for key in list(self.workers):
            if key[0] not in datasets or key[1] not in self.config['seeds']['locked']:continue
            self._send(key,'locked',source_intervention=locked_manifest['source_intervention']);result=self._recv(key)
            self._remember(key,result)
        return dict(status='COMPLETE',datasets=datasets,seeds=self.config['seeds']['locked'])
    def evaluate(self):
        if self.frozen is None:raise PermissionError('Need global freeze before evaluation')
        self.indexes=[]
        for key in self.workers:
            self._send(key,'export');r=self._recv(key);self.indexes.append(r['index']);self._remember(key,r['summary'])
            self._send(key,'bootstrap');self._recv(key)
        _json(self.root/'ARTIFACT_INDEX.json',dict(prefixes=self.indexes,identity=self.identity));self.write_status('evaluate','COMPLETE');return self.indexes
    def finalize(self):
        if not self.indexes:raise PermissionError('No final exports; memory may not be released')
        accept=[verify_exports(i,self.config) for i in self.indexes]
        if any(r['status']!='PASS' for r in accept):raise RuntimeError('Export acceptance failed')
        forbidden=[str(p) for p in self.root.rglob('*') if p.is_file() and p.suffix in ('.pt','.pth','.ckpt','.safetensors')]
        if forbidden:raise RuntimeError('Forbidden training state files: '+repr(forbidden))
        # Enumerate only explicitly recorded, current-campaign temporary row data.
        cleanup=[]
        for dataset,seed in self.workers:
            prefix_root=(self.root/'prefix'/dataset/f'seed{seed}').resolve();cache_root=(prefix_root/'row_data_cache').resolve()
            manifest=prefix_root/'ROW_DATA_CACHE.json'
            if not manifest.exists():raise FileNotFoundError('Owned row-data manifest missing before cleanup: '+str(manifest))
            owned=json.loads(manifest.read_text())
            for row in owned.get('mappings',[])+owned.get('label_access_masks',[]):
                path=Path(row['path']).resolve()
                if not path.is_relative_to(cache_root) or not cache_root.is_relative_to(self.root/'prefix'):raise PermissionError('Temporary row-data cleanup path escaped current owned root: '+str(path))
                if path.exists():cleanup.append(dict(path=str(path),bytes=path.stat().st_size,owner_manifest=str(manifest)))
        _json(self.root/'TEMPORARY_DATA_CLEANUP_MANIFEST.json',dict(files=cleanup,export_verified=True,model_files_written=0))
        for key in self.workers:self._send(key,'shutdown',export_verified=True);self._recv(key)
        for w in self.workers.values():
            w['process'].join(timeout=30)
            if w['process'].is_alive():raise RuntimeError('Worker must release RAM before deleting its temporary row mappings')
        deleted=[]
        for row in cleanup:
            path=Path(row['path'])
            if path.exists():path.unlink();deleted.append(row)
        for parent in sorted({Path(row['path']).parent for row in cleanup},key=lambda x:len(x.parts),reverse=True):
            if parent.exists() and not any(parent.iterdir()):parent.rmdir()
        retained=[i['root'] for i in self.indexes]
        if any(not Path(p).exists() for p in retained):raise RuntimeError('A final export root disappeared during temporary data cleanup')
        receipt=dict(status='COMPLETE',model_files_written=0,deleted_files=deleted,deleted_bytes=sum(r['bytes'] for r in deleted),memory_released=[list(k) for k in self.workers],retained_artifacts=retained,export_acceptance=accept,cleanup_scope='Only manifest-owned current-v2 prefix row_data_cache; final paired/candidate/function NPZ retained')
        _json(self.root/'EXPORT_ACCEPTANCE.json',dict(status='PASS',prefixes=accept));_json(self.root/'CLEANUP_RECEIPT.json',receipt);self.workers.clear();self.write_status('finalize','COMPLETE');return receipt
    def run_all(self):
        self.prefix();self.phase('s1');self.phase('s2');self.decide('s2');self.diagnose();self.decide('s3');self.run_locked();self.freeze();self.evaluate();return self.finalize()
