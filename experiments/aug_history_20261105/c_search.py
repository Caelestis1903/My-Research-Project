"""Immutable numeric C stages and five-seed validation choices over the existing runtime."""
import copy
import math
import statistics
from pathlib import Path
from experiments.overnight_ab_20261103.state import read_json,write_json,sha_file,digest
from experiments.overnight_ab_20261103.selection import CELLS,score
from .protocol import ROOT,SEEDS,SOURCE_REF
from . import guardian
from .selection import Incomplete,rank_family,require_frozen

SEARCH=('C1_NEG','C2_POS','C3_WEAK','C4_CADENCE')
CONTROLS=('C_MERGED','C_ROTATED')


def scope_check(prepared,datasets):
    if list(datasets)!=prepared.get('scope_datasets') or list(datasets)!=['movielens']:
        raise ValueError('C is authorized only for prepared scope [movielens]')


def immutable_json(path,value):
    path=Path(path)
    if path.exists() and read_json(path)!=value:
        raise ValueError('Frozen C resolution/selection cannot be overwritten: '+str(path))
    if not path.exists(): write_json(path,value)
    return value


def parent_reference(prepared):
    origin=ROOT/'artifacts/aug_history_20261105/attempt1'
    path=origin/'B/selection_frozen.json'; frozen=require_frozen(origin,path)
    parent=read_json(origin/'PREPARED.json')
    if frozen['manifest_hash']!=prepared['identity']['manifest_hash'] or parent['source_receipt']['source_ref']!=SOURCE_REF:
        raise ValueError('Gamma-zero parent protocol/source differs')
    if parent['identity']['datasets']['movielens']!=prepared['identity']['datasets']['movielens']:
        raise ValueError('Gamma-zero parent population/preprocessing differs')
    wanted=('source','batch_size','epochs','lr','betas','eps','lambda_aug','kappa','rho','weight_decay','target_domain')
    old_job=next(j for j in parent['jobs'] if j['dataset']=='movielens')
    current=prepared['resolved_configs']['movielens']
    if any(old_job['config'][key]!=current[key] for key in wanted):
        raise ValueError('Gamma-zero parent training settings differ')
    points=[p for p in frozen['datasets']['movielens']['endpoints'] if p['candidate_id'] in ('F','online')]
    if (len(points)!=30 or {(p['candidate_id'],p['seed'],p['checkpoint']) for p in points}!=
        {(cid,seed,endpoint) for cid in ('F','online') for seed in SEEDS for endpoint in ('selected','E10','E50')}):
        raise ValueError('Gamma-zero parent endpoint matrix is incomplete')
    endpoint_file=origin/'B/ENDPOINTS.csv'
    return dict(origin=str(origin),prepared_path=str(origin/'PREPARED.json'),prepared_sha256=sha_file(origin/'PREPARED.json'),
        source_hash=parent['identity']['source_hash'],manifest_hash=frozen['manifest_hash'],selection_hash=frozen['selection_hash'],
        selection_path=str(path),selection_sha256=sha_file(path),endpoint_path=str(endpoint_file),
        endpoint_sha256=sha_file(endpoint_file),points=points,new_training=0)


def make_job(prepared,candidate,seed,resolution_hash):
    config=dict(prepared['resolved_configs']['movielens'],gamma=float(candidate['gamma']),
        refresh_steps=int(candidate['refresh_steps']),control_mode=candidate['control_mode'])
    normalized=dict(candidate,family='C_SEARCH' if candidate['id'] in SEARCH else 'C_CONTROL')
    job=dict(dataset='movielens',seed=seed,route='C',config=config,config_hash=digest(config),
        manifest_hash=prepared['identity']['manifest_hash'],source_hash=prepared['identity']['source_hash'],
        data_hash=prepared['identity']['datasets']['movielens'],layout_hash=None,candidates=[normalized],
        resolution_hash=resolution_hash,scope_datasets=['movielens'])
    key=digest(job); job.update(job_id=key,trajectory_id=key)
    return job


def resolve_stage(output,prepared,stage,configs,dependencies):
    scope_check(prepared,prepared['scope_datasets'])
    if any(c['gamma'] is None or c['refresh_steps'] not in (20,40) or c['control_mode'] not in ('task','merged','rotated') for c in configs):
        raise ValueError('C job configuration must already be numeric and complete')
    dependency_name={'II':'W12','III':'W123','IV':'C_WINNER'}.get(stage)
    if dependency_name:
        path=Path(output)/f'C/movielens/{dependency_name}.json'
        if not path.exists(): raise Incomplete('C stage awaits its five-seed frozen choice: '+dependency_name)
        choice=read_json(path)
        if (digest({k:v for k,v in choice.items() if k!='choice_hash'})!=choice['choice_hash'] or
            choice['source_hash']!=prepared['identity']['source_hash'] or
            dependencies!={dependency_name:choice['choice_hash']} or
            any(row['seed_ids']!=SEEDS or not row['complete'] for row in choice['rows'])):
            raise ValueError('C stage dependency identity/completeness changed')
        previous=choice['winner_config']
        if stage=='II': expected=[dict(id='C3_WEAK',gamma=.5*(1 if previous['gamma']>0 else -1),refresh_steps=20,control_mode='task')]
        elif stage=='III': expected=[dict(id='C4_CADENCE',gamma=previous['gamma'],refresh_steps=40,control_mode='task')]
        else: expected=[dict(id=cid,gamma=previous['gamma'],refresh_steps=previous['refresh_steps'],
            control_mode='merged' if cid=='C_MERGED' else 'rotated') for cid in CONTROLS]
        if configs!=expected: raise ValueError('C numeric resolution differs from its frozen dependency')
    elif stage!='I' or configs!=[dict(id='C1_NEG',gamma=-2.,refresh_steps=20,control_mode='task'),dict(id='C2_POS',gamma=2.,refresh_steps=20,control_mode='task')]:
        raise ValueError('Unknown C stage or initial grid')
    payload=dict(stage=stage,source_hash=prepared['identity']['source_hash'],manifest_hash=prepared['identity']['manifest_hash'],
        data_hash=prepared['identity']['datasets']['movielens'],scope_datasets=['movielens'],seed_ids=SEEDS,
        configurations=configs,dependencies=dependencies)
    resolution_hash=digest(payload)
    record=dict(payload,resolution_hash=resolution_hash,
        jobs=[make_job(prepared,c,seed,resolution_hash) for c in configs for seed in SEEDS])
    return immutable_json(Path(output)/f'C/movielens/STAGE_{stage}_RESOLVED.json',record)


def initialize(output,prepared):
    reference=parent_reference(prepared)
    immutable_json(Path(output)/'PARENT_REFERENCE.json',reference)
    configs=[dict(id='C1_NEG',gamma=-2.,refresh_steps=20,control_mode='task'),
             dict(id='C2_POS',gamma=2.,refresh_steps=20,control_mode='task')]
    return resolve_stage(output,prepared,'I',configs,dict(engineering_source_hash=prepared['identity']['source_hash']))['jobs']


def resolved_jobs(output,prepared):
    jobs=[]
    for stage in ('I','II','III','IV'):
        path=Path(output)/f'C/movielens/STAGE_{stage}_RESOLVED.json'
        if not path.exists(): continue
        record=read_json(path)
        payload={k:v for k,v in record.items() if k not in ('resolution_hash','jobs')}
        if digest(payload)!=record['resolution_hash'] or record['source_hash']!=prepared['identity']['source_hash']:
            raise ValueError('C stage identity changed')
        expected=[make_job(prepared,c,seed,record['resolution_hash']) for c in record['configurations'] for seed in SEEDS]
        if record['jobs']!=expected:
            raise ValueError('Resolved C numeric jobs changed')
        jobs.extend(expected)
    if len({j['job_id'] for j in jobs})!=len(jobs):
        raise ValueError('Duplicate C trajectory identity')
    return jobs


def candidate_rows(output,prepared,candidate_ids,summaries=None):
    output=Path(output)
    summaries=summaries if summaries is not None else guardian.request(output,'summaries')['summaries']
    reference=read_json(output/'PARENT_REFERENCE.json')
    parent={p['seed']:p['validation'] for p in reference['points'] if p['candidate_id']=='F' and p['checkpoint']=='selected'}
    jobs=resolved_jobs(output,prepared); rows=[]; inputs={}
    for cid in candidate_ids:
        selected_jobs=sorted([j for j in jobs if j['candidates'][0]['id']==cid],key=lambda j:j['seed'])
        if [j['seed'] for j in selected_jobs]!=SEEDS:
            raise Incomplete('C candidate requires five resolved seeds: '+cid)
        if any(j['job_id'] not in summaries or not summaries[j['job_id']]['complete'] or summaries[j['job_id']]['epoch']!=50 for j in selected_jobs):
            raise Incomplete('C candidate requires five full-budget RAM states: '+cid)
        points=[]; costs=[]
        for job in selected_jobs:
            key=job['job_id']; summary=summaries.get(key)
            if summary is None or not summary['complete'] or summary['epoch']!=50:
                raise Incomplete('C candidate requires five full-budget RAM states: '+cid)
            if set(summary['snapshots'])!={'online',cid}:
                raise ValueError('C estimator identity changed')
            if any(set(v)!={'selected','E10','E50'} for v in summary['snapshots'].values()):
                raise Incomplete('C endpoint state is incomplete')
            epoch_cost=[]
            for epoch in range(1,51):
                path=guardian.epoch_path(output,key,epoch); record=read_json(path)
                if set(record['validation'])!={'online',cid} or record['epoch']!=epoch:
                    raise ValueError('C full validation curve is incomplete')
                inputs[str(path.relative_to(output))]=sha_file(path)
                epoch_cost.append(record['cost']['training_seconds']+record['cost']['probe_seconds']+record['cost']['validation_seconds'])
            points.append(summary['snapshots'][cid]['selected'])
            costs.append(math.fsum(epoch_cost))
        values=[p['validation'] for p in points]
        deltas={cell:math.fsum(v[cell+'_auc']-parent[seed][cell+'_auc'] for seed,v in zip(SEEDS,values))/5 for cell in CELLS}
        config=selected_jobs[0]['candidates'][0]
        row=dict(dataset='movielens',candidate_id=cid,family=config['family'],config=config,
            gamma=config['gamma'],refresh_steps=config['refresh_steps'],seed_ids=SEEDS,complete=True,
            score=math.fsum(score(v) for v in values)/5,score_sd=statistics.stdev(score(v) for v in values),
            deviation=abs(config['gamma']),measured_cost_seconds=math.fsum(costs),cell_protected=all(d>=-.001 for d in deltas.values()),
            selected_epochs=[p['epoch'] for p in points],selected_validation_scores=[score(v) for v in values],
            **{cell+'_delta_F':value for cell,value in deltas.items()})
        for metric in values[0]:
            if all(v[metric] is not None for v in values):
                row[metric+'_mean']=statistics.mean(v[metric] for v in values)
                row[metric+'_sd']=statistics.stdev(v[metric] for v in values)
        rows.append(row)
    return rows,inputs


def freeze_choice(output,prepared,name,ids):
    rows,inputs=candidate_rows(output,prepared,ids)
    winner=rank_family(rows)
    chosen=next(row for row in rows if row['candidate_id']==winner['candidate_id'])
    value=dict(name=name,route='C',scope_datasets=['movielens'],seed_ids=SEEDS,candidate_ids=list(ids),
        source_hash=prepared['identity']['source_hash'],manifest_hash=prepared['identity']['manifest_hash'],
        selection=winner,winner_config=chosen['config'],rows=rows,input_validation_hashes=inputs)
    value['choice_hash']=digest(value)
    return immutable_json(Path(output)/f'C/movielens/{name}.json',value)


def train_stages(output,prepared,datasets,seeds,run_stage):
    scope_check(prepared,datasets)
    if seeds!=SEEDS: raise ValueError('Every C candidate requires all five seeds')
    entry=read_json(Path(output)/'ENTRY_CONTINUATION.json')
    if entry.get('scope_datasets')!=['movielens'] or not any(a['route']=='C' and a['dataset']=='movielens' for a in entry['allowed_actions']):
        raise ValueError('C scope is not authorized by immutable A continuation')
    stage=read_json(Path(output)/'C/movielens/STAGE_I_RESOLVED.json')
    run_stage(stage['jobs'])
    w12=freeze_choice(output,prepared,'W12',SEARCH[:2])
    weak=.5*(1 if w12['winner_config']['gamma']>0 else -1)
    stage=resolve_stage(output,prepared,'II',[dict(id='C3_WEAK',gamma=weak,refresh_steps=20,control_mode='task')],
        dict(W12=w12['choice_hash']))
    run_stage(stage['jobs'])
    w123=freeze_choice(output,prepared,'W123',SEARCH[:3])
    stage=resolve_stage(output,prepared,'III',[dict(id='C4_CADENCE',gamma=w123['winner_config']['gamma'],refresh_steps=40,control_mode='task')],
        dict(W123=w123['choice_hash']))
    run_stage(stage['jobs'])
    winner=freeze_choice(output,prepared,'C_WINNER',SEARCH)
    selected=winner['winner_config']
    stage=resolve_stage(output,prepared,'IV',[dict(id=cid,gamma=selected['gamma'],refresh_steps=selected['refresh_steps'],
        control_mode='merged' if cid=='C_MERGED' else 'rotated') for cid in CONTROLS],dict(C_WINNER=winner['choice_hash']))
    run_stage(stage['jobs'])
    return dict(status='C_TRAIN_COMPLETE',scope_datasets=['movielens'],independent_trajectories=30,
        search_winner=selected['id'],new_parent_training=0)


def audit_chain(output,prepared):
    output=Path(output)
    if any(not (output/f'C/movielens/STAGE_{stage}_RESOLVED.json').exists() for stage in ('I','II','III','IV')):
        raise Incomplete('All four C stages must be resolved before final selection')
    initial=[dict(id='C1_NEG',gamma=-2.,refresh_steps=20,control_mode='task'),dict(id='C2_POS',gamma=2.,refresh_steps=20,control_mode='task')]
    resolve_stage(output,prepared,'I',initial,dict(engineering_source_hash=prepared['identity']['source_hash']))
    w12=freeze_choice(output,prepared,'W12',SEARCH[:2])
    resolve_stage(output,prepared,'II',[dict(id='C3_WEAK',gamma=.5*(1 if w12['winner_config']['gamma']>0 else -1),
        refresh_steps=20,control_mode='task')],dict(W12=w12['choice_hash']))
    w123=freeze_choice(output,prepared,'W123',SEARCH[:3])
    resolve_stage(output,prepared,'III',[dict(id='C4_CADENCE',gamma=w123['winner_config']['gamma'],refresh_steps=40,
        control_mode='task')],dict(W123=w123['choice_hash']))
    winner=freeze_choice(output,prepared,'C_WINNER',SEARCH)
    selected=winner['winner_config']
    resolve_stage(output,prepared,'IV',[dict(id=cid,gamma=selected['gamma'],refresh_steps=selected['refresh_steps'],
        control_mode='merged' if cid=='C_MERGED' else 'rotated') for cid in CONTROLS],dict(C_WINNER=winner['choice_hash']))
    return winner


def select_C(output):
    from .report import write_csv
    output=Path(output); prepared=read_json(output/'PREPARED.json'); scope_check(prepared,prepared['scope_datasets'])
    winner=audit_chain(output,prepared)
    summaries=guardian.request(output,'summaries')['summaries']
    hpo,inputs=candidate_rows(output,prepared,(*SEARCH,*CONTROLS),summaries)
    jobs=resolved_jobs(output,prepared)
    if len(jobs)!=30: raise Incomplete('C requires thirty independent ML trajectories')
    curves=[]; endpoints=[]; selected_ids=[winner['winner_config']['id'],*CONTROLS]
    costs={}
    for job in jobs:
        cid=job['candidates'][0]['id']; summary=summaries[job['job_id']]
        records=[]
        for epoch in range(1,51):
            record=read_json(guardian.epoch_path(output,job['job_id'],epoch)); records.append(record)
            for kind,values in record['validation'].items():
                curves.append(dict(dataset='movielens',seed=job['seed'],trajectory_id=job['trajectory_id'],
                    candidate_id=cid if kind==cid else cid+'_online',method_id=cid,
                    estimator='fixed_ema' if kind==cid else 'online',epoch=epoch,
                    model_hash=record['model_hash'][kind],training_loss=record['training_loss'],clean_loss=record['clean_loss'],**values))
        costs[job['job_id']]={key:max(r['cost'][key] for r in records) if key.startswith('peak_') else
            sum(r['cost'][key] for r in records) for key in records[0]['cost']}
        if cid in selected_ids:
            for kind in (cid,'online'):
                for endpoint,point in summary['snapshots'][kind].items():
                    endpoints.append(dict(job_id=job['job_id'],trajectory_id=job['trajectory_id'],seed=job['seed'],
                        candidate_id=cid if kind==cid else cid+'_online',source_candidate_id=cid,state_key=kind,
                        estimator='fixed_ema' if kind==cid else 'online',checkpoint=endpoint,**point))
    route=output/'C'; write_csv(route/'VALIDATION_CURVES.csv',curves); write_csv(route/'HPO_SUMMARY.csv',hpo)
    value=dict(schema_version=1,route='C',status='HPO_FROZEN',scope_datasets=['movielens'],seed_ids=SEEDS,
        source_hash=prepared['identity']['source_hash'],manifest_hash=prepared['identity']['manifest_hash'],
        input_hashes=dict(curves=sha_file(route/'VALIDATION_CURVES.csv'),hpo=sha_file(route/'HPO_SUMMARY.csv'),
            validation_files=digest(inputs)),parent_reference=read_json(output/'PARENT_REFERENCE.json'),
        datasets=dict(movielens=dict(winners={family:dict(candidate_id=cid) for family,cid in
            (('C',selected_ids[0]),('MERGED','C_MERGED'),('ROTATED','C_ROTATED'))},
            report_candidates=selected_ids,candidates=[j['candidates'][0] for j in jobs if j['seed']==2022],
            endpoints=endpoints,hpo_rows=hpo,costs=costs,search_choice_hash=winner['choice_hash'])))
    immutable_json(route/'HPO_SELECTION.json',value)
    value=dict(value,selection_hash=digest(value))
    return immutable_json(route/'selection_frozen.json',value)


def decide_C(output):
    output=Path(output); route=output/'C'; frozen=require_frozen(output,route/'selection_frozen.json')
    current=frozen['datasets']['movielens']; selected=current['winners']['C']['candidate_id']
    rows=[r for r in current['hpo_rows'] if r['candidate_id'] in (selected,*CONTROLS)]
    chosen=rank_family(rows)['candidate_id']; best=next(r for r in rows if r['candidate_id']==chosen)
    merged=next(r for r in rows if r['candidate_id']=='C_MERGED')
    if (best['score']-merged['score']<=.0002 and merged['measured_cost_seconds']<best['measured_cost_seconds'] and
        (merged['cell_protected'] or not best['cell_protected'])):
        chosen='C_MERGED'
    decision=dict(schema_version=1,study_id='AUG_HISTORY_20261105',route='C',dataset='movielens',scope_datasets=['movielens'],
        status='DECIDED',manifest_hash=frozen['manifest_hash'],selection_hash=frozen['selection_hash'],seed_ids=SEEDS,
        all_required_candidates_complete=True,search_winner=selected,selected_candidate=chosen,
        validation=dict(comparisons=rows,selection_rule='cell_protection_then_score_cost_deviation_id; cheaper_merged_within_.0002'),
        decision_basis=dict(split='validation',test_consulted=False,historical_test_exposure=True),
        next_action='STOP_AND_REPORT',next_actions=[],matched_rule='C_FULL_COMPLETE',
        deferred_routes=dict(kuairand_C='DEFERRED_NOT_RUN'),new_independent_trajectories=30,new_parent_training=0)
    immutable_json(route/'movielens/DECISION.json',decision)
    return immutable_json(route/'DECISION.json',dict(route='C',basis='validation',scope_datasets=['movielens'],
        selection_hash=frozen['selection_hash'],datasets=dict(movielens=decision),next_actions=[],next_action='STOP_AND_REPORT'))


def evaluate_C(output,frozen_path,device):
    import torch
    from .report import write_csv,endpoint_row
    from .protocol import published_reference
    from .model_adapter import build_model
    from experiments.overnight_ab_20261103.data_bridge import cached_bundle
    from experiments.overnight_ab_20261103.evaluation import evaluate_model
    from experiments.overnight_ab_20261103.inventory import read_csv,scalar_row
    from experiments.overnight_ab_20261103.state import state_digest
    output=Path(output); route=output/'C'; prepared=read_json(output/'PREPARED.json')
    frozen=require_frozen(output,frozen_path); scope_check(prepared,frozen['scope_datasets'])
    if not (route/'DECISION.json').exists(): raise Incomplete('Freeze C validation dispatch before test access')
    destination=route/'ENDPOINTS.csv'
    if (route/'EVALUATED.json').exists():
        previous=read_json(route/'EVALUATED.json')
        if previous['selection_hash']!=frozen['selection_hash'] or previous['endpoints_sha256']!=sha_file(destination):
            raise ValueError('C evaluated endpoint archive changed')
        return previous
    rows=[]; points=frozen['datasets']['movielens']['endpoints']
    for job in resolved_jobs(output,prepared):
        current=[point for point in points if point['job_id']==job['job_id']]
        if not current: continue
        snapshot=guardian.request(output,'get',key=job['job_id'])['state']
        if snapshot is None or not snapshot['complete']: raise Incomplete('C RAM endpoints lost; replay train --resume')
        bundle=cached_bundle(prepared['cache_root'],'movielens',include_test=True)
        model=build_model(bundle,job['config']).to(device)
        for frozen_point in current:
            kind=frozen_point['state_key']; checkpoint=frozen_point['checkpoint']; point=snapshot['snapshots'][kind][checkpoint]
            if (point['epoch'],point['model_hash'])!=(frozen_point['epoch'],frozen_point['model_hash']):
                raise ValueError('C RAM endpoint differs from frozen identity')
            model.load_state_dict(point['model_state'])
            if state_digest(model.state_dict())!=point['model_hash']: raise ValueError('C model content hash changed')
            metrics=evaluate_model(model,bundle,'test','G-D0',job['config']['eval_batch_size'])
            for split,values in (('validation',point['validation']),('test',metrics)):
                row=endpoint_row(job,job['candidates'][0]['id'],checkpoint,point,split,values,snapshot['layout_hash'])
                row.update(candidate_id=frozen_point['candidate_id'],estimator=frozen_point['estimator'],
                    method_id=job['candidates'][0]['id'])
                rows.append(row)
        del model,snapshot,bundle
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    parent=frozen['parent_reference']; path=Path(parent['endpoint_path'])
    if sha_file(path)!=parent['endpoint_sha256']: raise ValueError('Frozen parent endpoint source changed')
    for row in map(scalar_row,read_csv(path)):
        if row['dataset']=='movielens' and row['candidate_id'] in ('F','online'):
            point=next(p for p in parent['points'] if (p['candidate_id'],p['seed'],p['checkpoint'])==
                (row['candidate_id'],row['seed'],row['checkpoint']))
            if (row['epoch'],row['model_hash'])!=(point['epoch'],point['model_hash']):
                raise ValueError('Reused gamma-zero endpoint differs from parent freeze')
            row.update(candidate_id='F' if row['candidate_id']=='F' else 'PARENT_online',
                estimator='fixed_ema' if row['candidate_id']=='F' else 'online',role='REUSED_GAMMA_ZERO_PARENT',new_training=0,
                parent_source_hash=parent['source_hash'],parent_selection_hash=parent['selection_hash'])
            rows.append(row)
    published,reference=published_reference(); rows.extend(r for r in published if r['dataset']=='movielens')
    write_csv(destination,rows)
    marker=dict(status='EVALUATED',scope_datasets=['movielens'],selection_hash=frozen['selection_hash'],
        endpoints_sha256=sha_file(destination),published_reference=reference,new_independent_trajectories=30,new_parent_training=0)
    write_json(route/'EVALUATED.json',marker)
    return marker


def report_C(output):
    from .report import write_csv
    from experiments.overnight_ab_20261103.inventory import read_csv,scalar_row
    output=Path(output); route=output/'C'
    if not (route/'EVALUATED.json').exists(): raise Incomplete('Evaluate frozen C test endpoints before report')
    frozen=require_frozen(output,route/'selection_frozen.json'); prepared=read_json(output/'PREPARED.json')
    rows=list(map(scalar_row,read_csv(route/'ENDPOINTS.csv')))
    selected=frozen['datasets']['movielens']['winners']['C']['candidate_id']
    methods=[selected,*CONTROLS]
    comparisons=[(selected,ref) for ref in ('F',*CONTROLS)]
    comparisons += [(cid,ref) for cid in methods for ref in ('M3oE_fixed1','HyperGate','AutoIFS')]
    comparisons += [(cid,'F') for cid in CONTROLS]
    comparisons += [(cid+'_online','PARENT_online') for cid in methods]
    metrics=[key for key in rows[0] if key.endswith(('_auc','_logloss'))]
    pairs=[]; summaries=[]
    for method,reference in comparisons:
        for split in ('validation','test'):
            for endpoint in ('selected','E10','E50'):
                peers={cid:{r['seed']:r for r in rows if r['candidate_id']==cid and r['split']==split and r['checkpoint']==endpoint}
                    for cid in (method,reference)}
                if any(sorted(v)!=SEEDS for v in peers.values()): raise Incomplete('C comparison needs exactly five matching seeds')
                for metric in metrics:
                    delta=[peers[method][seed][metric]-peers[reference][seed][metric] for seed in SEEDS]
                    mean=statistics.mean(delta); sd=statistics.stdev(delta); margin=2.7764451051977987*sd/math.sqrt(5)
                    for seed,difference in zip(SEEDS,delta):
                        pairs.append(dict(dataset='movielens',method=method,reference=reference,split=split,checkpoint=endpoint,
                            metric=metric,seed=seed,method_value=peers[method][seed][metric],reference_value=peers[reference][seed][metric],delta=difference))
                    summaries.append(dict(dataset='movielens',method=method,reference=reference,split=split,checkpoint=endpoint,
                        metric=metric,n=5,mean_delta=mean,sd_delta=sd,ci95_low=mean-margin,ci95_high=mean+margin,
                        positive_seeds=sum(d>0 for d in delta),method_mean=statistics.mean(peers[method][s][metric] for s in SEEDS),
                        method_sd=statistics.stdev(peers[method][s][metric] for s in SEEDS),reference_mean=statistics.mean(peers[reference][s][metric] for s in SEEDS)))
    write_csv(route/'PAIRED_SEED.csv',pairs); write_csv(route/'PAIRED_SUMMARY.csv',summaries)
    costs=[dict(dataset='movielens',candidate_id=job['candidates'][0]['id'],seed=job['seed'],trajectory_id=job['trajectory_id'],
        **frozen['datasets']['movielens']['costs'][job['job_id']]) for job in resolved_jobs(output,prepared)]
    write_csv(route/'COSTS.csv',costs)
    write_json(output/'RUN_MANIFEST.json',dict(prepared,status='C_EVALUATED',scope_datasets=['movielens'],
        resolved_jobs=resolved_jobs(output,prepared),selection_hash=frozen['selection_hash'],costs=costs,
        raw_tables={p.name:sha_file(p) for p in route.glob('*.csv')}))
    (route/'REPORT.md').write_text('# MovieLens C_FULL 实验数据报告\n\n30条独立训练、1500epoch、30000次Adam、3000次模型验证；γ0父控制复用，新增0条。\n\n全部候选HPO与V曲线、冻结端点、逐seed/配对数表和成本已生成。KuaiRand C为DEFERRED_NOT_RUN。科学解释由主代理根据原数表撰写。\n')
    return dict(status='C_REPORTED',scope_datasets=['movielens'],independent_trajectories=30,
        endpoint_rows=len(rows),paired_rows=len(pairs),summary_rows=len(summaries),next_action='STOP_AND_REPORT')
