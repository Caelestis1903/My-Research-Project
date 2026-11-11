"""Frozen study resolution, source/data identity, and finite job expansion."""
import copy
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from experiments.overnight_ab_20261103.state import read_json, write_json, digest, sha_file

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE='5ef45b18b973ed11be6fbb5f654562349998de60'
PLAN='01979f6588935f713029b2b5e04557de9b8c153f'
SEEDS=list(range(2022,2027))
CELLS=[f'd{d}t{t}' for d in range(3) for t in range(2)]
BASELINES=Path('/home/xzf/Code/phase0-3_gaussian_coverage/research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv')


def implementation_identity():
    files={str(p.relative_to(ROOT)):sha_file(p) for p in sorted(HERE.rglob('*.py'))}
    return dict(files=files,sha256=digest(files))


def readout_menu(manifest,ds):
    d=manifest['datasets'][ds]
    result=[dict(id='O',family='O',forwards=1),dict(id='F',family='F',forwards=1),
            dict(id='B_SHARED',family='B_SHARED_ONLY',forwards=1)]
    for h in d['history_windows']:
        for scope in ('S','all'):
            for mode in ('logit','probability'):
                result.append(dict(id=f'B_H{h}_{scope}_{mode}',family='B_SPLIT',H=h,scope=scope,mode=mode,forwards=2))
    for mode in ('PRED','PARAM'):
        result.append(dict(id=f'C_GLOBAL_{mode}',family=f'C_GLOBAL_{mode}',H=d['history_base'],scope='S',mode='logit',forwards=2 if mode=='PRED' else 1))
        for s in (.5,1.):
            result.append(dict(id=f'C_CELL_{mode}_s{s:g}',family=f'C_CELL_{mode}',H=d['history_base'],scope='S',mode='logit',shrinkage=s,forwards=2))
    return result


def resolve_plan(plan_dir):
    plan=Path(plan_dir).resolve();raw=read_json(plan/'experiment_manifest.json');m=copy.deepcopy(raw)
    if m['source_commit']!=SOURCE or m['seeds']!=SEEDS:raise ValueError('Frozen source/seeds changed')
    if m['budgets']['logical_training']['core_epochs']!=740 or m['budgets']['logical_training']['all_max_epochs']!=2490:raise ValueError('Frozen matrix budget changed')
    if m['route_A']['anchors']!=[20,35] or m['route_A']['direction_alpha']!=[.5,1.]:raise ValueError('Frozen A menu changed')
    if m['distillation']['student_epochs']!=5 or m['source_protocol']['train_batch_size']!=40960:raise ValueError('Frozen training budget changed')
    correction=dict(field='route_C.risk_tie_order',original=m['route_C']['risk_tie_order'],
        resolved=['closest to current old mass a','closest to 0.5','smaller alpha'],authority='14_ROUTE_C.md section 3.2')
    m['route_C']['risk_tie_order']=correction['resolved']
    configs=copy.deepcopy(read_json(ROOT/'research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json')['configs'])
    for ds,c in configs.items():
        d=m['datasets'][ds]
        for key in ('rho','weight_decay','steps_per_epoch','parameter_count'):
            if c[key]!=d[key]:raise ValueError('Source protocol mismatch '+ds+'/'+key)
        c.update(rho_by_physical_shadow={'F':c['rho']},aliases={},target_domain=0)
    hashes={p.name:sha_file(p) for p in sorted(plan.iterdir()) if p.is_file()}
    provenance=ROOT/'research_reports/2026-11-11_ema_next/PLAN_PROVENANCE.json'
    if provenance.exists() and read_json(provenance)['files']!=hashes:raise ValueError('Pinned raw planning definitions changed')
    identity=dict(study_id=m['study_id'],source_commit=SOURCE,plan_commit=PLAN,plan_hash=digest(hashes),
        plan_files=hashes,resolved_manifest_hash=digest(m),implementation_hash=implementation_identity()['sha256'])
    study=dict(manifest=m,configs=configs,identity=identity,plan_dir=str(plan),corrections=[correction],
               readouts={d:readout_menu(m,d) for d in configs})
    study['jobs']=expand_jobs(study)
    return study


def make_job(study,ds,seed,family,*,objective='clean',start_epoch=0,end_epoch=50,parent_id=None,**extra):
    config=copy.deepcopy(study['configs'][ds]);identifier=extra.pop('job_id',None)
    scientific=dict(dataset=ds,seed=seed,family=family,objective=objective,start_epoch=start_epoch,end_epoch=end_epoch,parent_id=parent_id,**extra)
    scientific['config']=config
    scientific['config_hash']=digest(scientific)
    scientific['job_id']=identifier or f'{ds}__{objective}__{family}__s{seed}__{scientific["config_hash"][:12]}'
    scientific['steps_per_epoch']=config['steps_per_epoch']
    return scientific


def parent_id(ds,seed,objective='clean'):
    return f'P.{"C" if objective=="clean" else "G"}.{ds}.{seed}'


def expand_jobs(study):
    jobs=[]
    for ds in study['manifest']['dataset_order']:
        for seed in SEEDS:
            pid=parent_id(ds,seed)
            jobs.append(make_job(study,ds,seed,'PARENT',job_id=pid,mandatory=True,requires='S0_PASS'))
            for anchor in (20,35):
                for alpha in (.5,1.):
                    jobs.append(make_job(study,ds,seed,'A_MATCH',parent_id=pid,start_epoch=anchor,end_epoch=anchor+6,direction_alpha=alpha,
                        mandatory=True,requires='FIVE_PARENTS_AND_B_FREEZE',job_id=f'A.M.{ds}.{seed}.e{anchor}.a{alpha:g}'))
    return jobs


def history_schedule(ds,manifest):
    d=manifest['datasets'][ds];steps=d['steps_per_epoch'];users={}
    for e in range(1,51):
        for h in d['history_windows']:users.setdefault(max(0,e*steps-h),set()).add(e*steps)
    active={};peak=0;events=[]
    for t in range(50*steps+1):
        if t in users:active[t]=max(users[t])
        peak=max(peak,len(active))
        events.append(dict(step=t,capture=t in users,live_before_consume=len(active)))
        active={k:v for k,v in active.items() if v>t}
    return dict(origins={str(k):sorted(v) for k,v in users.items()},peak=peak,events=events)


def resource_forecast(study):
    # Each entry is a distinct worst-case packet; sharing is credited only within its layout.
    result={};prior=0
    for ds in study['manifest']['dataset_order']:
        d=study['manifest']['datasets'][ds];P=d['parameter_count'];S=d['shared_parameter_count'];N=P-S
        schedule=history_schedule(ds,study['manifest']);H=schedule['peak']
        best={'O_F_SHARED':3*P,'B_S':6*(2*S+N+P),'B_all':6*3*P,
              'C_GLOBAL_PRED':2*S+N+P,'C_GLOBAL_PARAM':P,
              'C_CELL_PRED':2*(2*S+N+P),'C_CELL_PARAM':2*(2*S+N)}
        prefix={'O_F':2*P,'B_S':6*(2*S+N),'B_all':6*2*P}
        anchors={'E20_E35_theta_m_v_F':8*P,'E20_E35_sparse_F':2*H*P}
        completed=sum(best.values())+sum(prefix.values())+sum(anchors.values())
        live=(4+H)*P;metadata=(1<<20)+200*d.get('runtime_buffer_bytes',0);events=[]
        for done in range(5):
            for epoch in range(1,51):
                prefix_now=sum(prefix.values()) if epoch>=20 else 0
                anchors_now=(4+H)*P*(int(epoch>=20)+int(epoch>=35))
                resident=prior+4*(done*completed+sum(best.values())+prefix_now+anchors_now+live)+(done+1)*metadata
                incoming=4*(live+(12*P+3*S))+metadata
                worker=4*live+metadata
                guardian=resident+incoming+worker
                physical=guardian+4*(completed+live+5*P)+metadata
                events.append(dict(completed_seeds=done,epoch=epoch,resident_bytes=resident,
                    incoming_bytes=incoming,worker_live_bytes=worker,guardian_predicate=guardian,physical_tensor_bytes=physical))
        peak=max(events,key=lambda e:e['guardian_predicate'])
        singles=[]
        for window in d['history_windows']:
            one=copy.deepcopy(study['manifest']);one['datasets'][ds]['history_windows']=[window]
            singles.append(history_schedule(ds,one)['peak'])
        h1=max(singles)
        phase_elements=dict(parent=5*completed+live,
            MATCH_RAW_short=5*((26+2*h1)*P+5*S)+10*(13+h1)*P,
            KD=190*P+25*S,
            Gaussian_A_or_AB=5*((26*P+5*S)+(29+13+h1)*P),
            Gaussian_KD=310*P+30*S,
            Gaussian_C=270*P+45*S)
        if any(v>phase_elements['parent'] for v in phase_elements.values()):raise ValueError('A conditional RAM phase is not covered by parent reservation')
        result[ds]=dict(layout_elements=dict(best=best,prefix=prefix,anchors=anchors,active=live),history_peak=H,conditional_phase_elements=phase_elements,single_window_history_peak=h1,
            event_count=len(events),events=events,peak=peak,prior_dataset_endpoint_bytes=prior,
            status='PASS' if peak['guardian_predicate']<=4<<30 and max(e['physical_tensor_bytes'] for e in events)<=7<<30 else 'BLOCKED_RESOURCE')
        # Frozen O/F, B winner+scope control, C chosen modes, A/AB, and three KD controls.
        endpoint_terms=dict(O_F_SHARED=3*P,B_winner_opposite=3*P+S,C_GLOBAL_modes=2*P+2*S,C_CELL_same_s_modes=2*P+2*S,A_online_F=4*P,AB=4*P,KD_four_online_F=8*P)
        gaussian_terms=dict(KD=10*P,AB=8*P,C=6*P+4*S,B=6*P+S,A=4*P)
        retained_elements=sum(endpoint_terms.values())+max(gaussian_terms.values())
        result[ds]['final_endpoint_layout_elements']=dict(clean=endpoint_terms,gaussian_alternatives=gaussian_terms,total=retained_elements)
        prior+=5*retained_elements*4
    return dict(datasets=result,status='PASS' if all(x['status']=='PASS' for x in result.values()) else 'BLOCKED_RESOURCE',
        guardian_cap=4<<30,physical_cap=7<<30,persisted_cap=2<<30,
        scalar_curve_bound_bytes=450<<20,pair_prediction_bound_bytes=650<<20,soft_target_bound_bytes=300<<20,
        metadata_bound_bytes=300<<20)


def audit_source(study,out):
    files={}
    for folder in ('aug_history_20261105','ema_mechanism_20261106','ema_causal_20261110','overnight_ab_20261103','mdmt_history_information_20261103'):
        for p in sorted((ROOT/'experiments'/folder).rglob('*.py')):
            rel=str(p.relative_to(ROOT));expected=subprocess.check_output(['git','rev-parse',SOURCE+':'+rel],cwd=ROOT,text=True).strip()
            actual=subprocess.check_output(['git','hash-object',str(p)],cwd=ROOT,text=True).strip()
            if actual!=expected:raise ValueError('SOURCE_MISMATCH '+rel)
            files[rel]=dict(git_blob=actual,sha256=sha_file(p))
    environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,
        platform=platform.platform(),cuda=torch.cuda.is_available(),devices=torch.cuda.device_count())
    result=dict(identity=study['identity'],source_files=files,environment=environment)
    write_json(Path(out)/'SOURCE_RECEIPT.json',result)
    return result


def prepare(study,out,cache_root=None):
    from experiments.overnight_ab_20261103.data_bridge import prepare_cache,cached_bundle
    from experiments.ema_mechanism_20261106.protocol import verify_data_identity
    from experiments.ema_mechanism_20261106.panels import build_panels,save_panels
    from experiments.aug_history_20261105.model_adapter import build_model
    from experiments.aug_history_20261105.group_registry import build_path_groups
    out=Path(out);out.mkdir(parents=True,exist_ok=True);m=study['manifest']
    cache=Path(cache_root or m['environment']['source_cache']);old=read_json(ROOT/'research_plans/26-11-6/experiment_manifest.json')
    paths={d:v['source_csv'] for d,v in m['datasets'].items()}
    if not all((cache/d/'manifest.json').exists() for d in paths):prepare_cache(cache,dict(datasets={d:dict(csv=p) for d,p in paths.items()}))
    data=verify_data_identity(SimpleNamespace(manifest=old),cache,paths);panels={};groups={}
    for ds in paths:
        bundle=cached_bundle(cache,ds,include_test=False);model=build_model(bundle,study['configs'][ds]);groups[ds]=build_path_groups(model,bundle).as_dict();groups[ds]['runtime_buffer_bytes']=sum(v.numel()*v.element_size() for v in model.buffers());del model
        panels[ds]={str(s):save_panels(out,ds,s,build_panels(bundle,s,old['panels'])) for s in SEEDS}
    forecast_study=copy.deepcopy(study)
    for ds in groups:forecast_study['manifest']['datasets'][ds]['runtime_buffer_bytes']=groups[ds]['runtime_buffer_bytes']
    forecast=resource_forecast(forecast_study)
    result=dict(study,out=str(out.resolve()),cache_root=str(cache.resolve()),data_identity=data,panels=panels,groups=groups,resource_estimate=forecast,
        published_baseline_path=str(BASELINES),published_baseline_sha256=sha_file(BASELINES))
    for job in result['jobs']:
        job.update(data_hash=data[job['dataset']]['data_hash'],group_hash=groups[job['dataset']]['group_hash'],identity=study['identity'])
    if (out/'PREPARED.json').exists() and read_json(out/'PREPARED.json')!=result:raise ValueError('PREPARED identity conflict')
    for name,value in [('PREPARED',result),('RESOLVED_PROTOCOL',dict(identity=study['identity'],configs=study['configs'],manifest=m)),
        ('DATA_MANIFEST',data),('PROBE_MANIFEST',dict(panels=panels)),('PANEL_MANIFEST',panels),('GROUP_MANIFEST',groups),('EXPANDED_JOBS',dict(core=study['jobs'],conditional_templates=m['job_templates'],budgets=m['budgets']['logical_training'])),
        ('READOUT_SCHEDULE',{d:history_schedule(d,m) for d in paths}),('RESOURCE_ESTIMATE',forecast),('SPECIFICATION_CORRECTIONS',study['corrections'])]:write_json(out/(name+'.json'),value)
    if forecast['status']!='PASS':raise MemoryError('BLOCKED_RESOURCE: full five-seed object graph exceeds RAM budget; see RESOURCE_ESTIMATE')
    return result
