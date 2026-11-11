"""Frozen endpoint evaluation and reproducible numerical tables."""
import csv
import gzip
import itertools
import json
import time
from pathlib import Path
from .protocol import SEEDS,CELLS,implementation_identity
from .state import request,disk_admit
from . import decision as D
from experiments.overnight_ab_20261103.state import read_json,write_json,digest,sha_file


def table(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'wt',newline='') as stream:
        writer=csv.DictWriter(stream,columns);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def rows(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as stream:return list(csv.DictReader(stream))


def inherited_records(out,job):
    from .scheduler import load_records
    result=load_records(out,job['job_id'])
    if job['family'] in ('A_MATCH','A_RAW'):
        parent=read_json(Path(out)/'runs'/job['parent_id']/'JOB.json')
        result=[r for r in inherited_records(out,parent) if r['epoch']<=job['start_epoch']]+result
    return result


def freeze(out,prepared,summaries):
    out=Path(out);live=request(out,'summaries')['summaries'];endpoints=[]
    for summary in summaries:
        for condition,registry in [('clean',summary['registry']),('gaussian_d0',summary['gaussian']['registry'])]:
            for candidate,entry in registry.items():
                if set(map(int,entry['owners']))!=set(SEEDS):raise ValueError('Incomplete endpoint seeds')
                for seed in SEEDS:
                    owner=entry['owners'][str(seed)];job_id=owner['job_id'];rid=owner['readout_id'];row=live[job_id]
                    if row['status']!='COMPLETE':raise ValueError('Unfinished endpoint owner '+job_id)
                    point=row['selected'][rid];packet=request(out,'get_object',object_id=point['packet_id'])['object']
                    curve=inherited_records(out,row['job']);chosen=D.selected(curve,rid)
                    if chosen['epoch']!=point['epoch'] or D.metrics(chosen)!=point['validation'] or chosen['state_id']!=packet['state_id']:
                        raise ValueError('Selected RAM endpoint differs from complete committed curve')
                    endpoints.append(dict(dataset=summary['dataset'],condition=row['job']['objective'],candidate=candidate,seed=seed,
                        owner_job_id=job_id,readout_id=rid,packet_id=point['packet_id'],state_id=packet['state_id'],epoch=point['epoch'],
                        validation=point['validation'],spec=entry['spec'],objective=row['job']['objective'],family=entry['spec']['family'],
                        config_hash=row['job']['config_hash'],data_hash=prepared['data_identity'][summary['dataset']]['data_hash']))
    frozen=dict(status='FROZEN',identity=prepared['identity'],seeds=SEEDS,endpoints=endpoints,
        state_ids=sorted({e['state_id'] for e in endpoints}),decisions_sha256=sha_file(out/'DECISIONS.json'),
        primary={s['dataset']:s['primary'] for s in summaries},primary_by_condition=[dict(dataset=s['dataset'],condition='clean',candidate=s['primary']) for s in summaries]+[dict(dataset=s['dataset'],condition='gaussian_d0_parent__clean_student' if s['gaussian']['inherited_spec']['family'] in ('B_DISTILL','KD_LABEL','KD_FTEACHER') else 'gaussian_d0',candidate=s['gaussian']['primary_candidate']) for s in summaries if s['gaussian']['status']=='COMPLETE'],published_baselines=dict(path=prepared['published_baseline_path'],sha256=prepared['published_baseline_sha256'],spec=prepared['manifest']['evaluation']['published_baselines']))
    path=out/'GLOBAL_SELECTION_FREEZE.json'
    if path.exists() and read_json(path)!=frozen:raise ValueError('Global freeze changed')
    write_json(path,frozen);return frozen


def evaluate_test(out,prepared,device):
    from .evaluation import make_context,evaluate_full_population
    from experiments.overnight_ab_20261103.data_bridge import cached_bundle
    out=Path(out);freeze=read_json(out/'GLOBAL_SELECTION_FREEZE.json')
    if freeze['identity']!=prepared['identity'] or freeze['decisions_sha256']!=sha_file(out/'DECISIONS.json'):raise ValueError('Freeze binding changed')
    endpoints=[];cache={};costs=[]
    for ds in prepared['manifest']['dataset_order']:
        bundle=cached_bundle(prepared['cache_root'],ds,include_test=True);context=make_context(bundle,prepared['configs'][ds],device)
        for e in [x for x in freeze['endpoints'] if x['dataset']==ds]:
            cache_key=(ds,e['state_id']);start=time.perf_counter()
            if cache_key not in cache:
                fetched=request(out,'get_object',object_id=e['packet_id']);packet=fetched['object']
                if packet['state_id']!=e['state_id']:raise ValueError('Frozen state missing or changed')
                cache[cache_key]=evaluate_full_population(packet,bundle,split='test',eval_batch_size=prepared['configs'][ds]['eval_batch_size'],context=context,freeze=freeze,objects=fetched['objects'])
            for split,values in [('validation',e['validation']),('test',cache[cache_key])]:
                endpoints.append({k:v for k,v in e.items() if k not in ('validation','spec')}|dict(split=split,checkpoint='selected',role='INTERNAL_CONTROL' if e['candidate'] in ('O','F') else 'CANDIDATE',**values))
            costs.append(dict(phase='test',dataset=ds,candidate=e['candidate'],seed=e['seed'],seconds=time.perf_counter()-start,state_id=e['state_id']))
        del bundle,context
    table(out/'ENDPOINTS.csv.gz',endpoints);table(out/'TEST_COSTS.csv.gz',costs)
    write_json(out/'TEST_COMPLETE.json',dict(status='COMPLETE',freeze_sha256=sha_file(out/'GLOBAL_SELECTION_FREEZE.json'),endpoint_count=len(endpoints)//2,unique_forward_objects=len(cache),endpoints_sha256=sha_file(out/'ENDPOINTS.csv.gz')))
    return endpoints


def paired_tables(out,prepared,endpoints):
    test=[r for r in endpoints if r['split']=='test'];groups={}
    for r in test:groups.setdefault((r['dataset'],r['condition'],r['candidate']),{})[int(r['seed'])]=r
    published=[]
    if sha_file(prepared['published_baseline_path'])!=prepared['published_baseline_sha256']:raise ValueError('Published baseline changed')
    spec=prepared['manifest']['evaluation']['published_baselines'];filters=spec['filters']
    for r in rows(prepared['published_baseline_path']):
        if r['split']!=spec['test_split']:continue
        if not all(r.get(k) in v if isinstance(v,list) else r.get(k)==v for k,v in filters.items()):continue
        published.append(r)
    comparisons=[]
    for (ds,condition,candidate),data in groups.items():
        controls={}
        for control in ('O','F','B_SHARED','C_GLOBAL_PRED','C_GLOBAL_PARAM'):
            key=(ds,'gaussian_d0' if condition=='gaussian_d0_parent__clean_student' else condition,control)
            if key in groups and control!=candidate:controls[control]=groups[key]
        for other in groups:
            if other[:2]!=(ds,condition) or other[2]==candidate:continue
            name=other[2]
            if (candidate.startswith('A_FULL_') and name.startswith('A_FULL_')) or (candidate.startswith('C_CELL_') and name.startswith('C_CELL_')) or (candidate.startswith('KD_') and name.startswith('KD_')) or (candidate.startswith('AB_') and (name.startswith('A_FULL_') or name.startswith('B_H'))):controls[name]=groups[other]
        for method in filters['method']:controls['PUBLISHED:'+method]={int(r['seed']):r for r in published if r['dataset']==ds and r['method']==method}
        for name,control in controls.items():
            if set(data)!=set(SEEDS) or set(control)!=set(SEEDS):raise ValueError('Incomplete comparison seeds')
            for metric in ['macro_auc','macro_logloss','sample_weighted_cell_auc','sample_weighted_cell_logloss','pooled_task_mean_auc','pooled_logloss','worst_cell_auc',*[c+s for c in CELLS for s in ('_auc','_logloss')]]:
                if any(data[s].get(metric) in (None,'') or control[s].get(metric) in (None,'') for s in SEEDS):continue
                delta=[float(data[s][metric])-float(control[s][metric]) for s in SEEDS];stats=D.statistics(delta)
                comparisons.append(dict(dataset=ds,condition=condition,candidate=candidate,control=name,metric=metric,**stats,
                    comparison_class='published_model' if name.startswith('PUBLISHED:') else 'ordinary_online' if name=='O' else 'component_vs_F' if name=='F' else 'simpler_alternative',
                    candidate_values=[float(data[s][metric]) for s in SEEDS],control_values=[float(control[s][metric]) for s in SEEDS]))
    table(out/'PAIRED.csv.gz',comparisons);return comparisons


def ab_interactions(out,decisions,endpoints):
    from .scheduler import load_records
    result=[];metric_names=['macro_auc','macro_logloss',*[c+s for c in CELLS for s in ('_auc','_logloss')]]
    for summary in decisions['datasets']:
        ds=summary['dataset'];registry=summary['registry']
        for key,entry in registry.items():
            if not key.startswith('AB_'):continue
            label=key[3:];a_key='A_FULL_'+label+'_F';b_id=next(v['readout_id'] for v in entry['owners'].values())
            for metric in metric_names:
                for split in ('validation','test'):
                    values=[]
                    for seed in SEEDS:
                        lookup={r['candidate']:float(r[metric]) for r in endpoints if r['dataset']==ds and r['condition']=='clean' and r['split']==split and int(r['seed'])==seed}
                        values.append(lookup[key]-lookup[a_key]-lookup[b_id]+lookup['F'])
                    result.append(dict(dataset=ds,candidate=key,comparison='independently_selected_difference_in_differences',split=split,metric=metric,**D.statistics(values)))
                values=[]
                for seed in SEEDS:
                    owner=entry['owners'][str(seed)];a=next(r for r in load_records(out,owner['job_id']) if r['epoch']==50)
                    parent=next(r for r in load_records(out,'P.C.'+ds+'.'+str(seed)) if r['epoch']==50)
                    values.append(D.metrics(a['readouts'][b_id])[metric]-D.metrics(a['readouts']['F'])[metric]-D.metrics(parent['readouts'][b_id])[metric]+D.metrics(parent['readouts']['F'])[metric])
                result.append(dict(dataset=ds,candidate=key,comparison='same_time_E50_difference_in_differences',split='validation',metric=metric,**D.statistics(values)))
    table(Path(out)/'AB_INTERACTIONS.csv.gz',result)
    return result


def report(out,prepared):
    out=Path(out);endpoints=rows(out/'ENDPOINTS.csv.gz');comparisons=paired_tables(out,prepared,endpoints);costs=[];curves=[]
    for path in sorted((out/'runs').glob('*/epochs/E*.json')):
        record=read_json(path)
        if record['epoch'] and not (path.parent.parent/'commits'/path.name).exists():continue
        costs.append(dict(job_id=record['job_id'],dataset=record['dataset'],seed=record['seed'],epoch=record['epoch'],family=record['family'],objective=record['objective'],**record.get('cost',{})))
        for rid,point in record['readouts'].items():curves.append(dict(job_id=record['job_id'],dataset=record['dataset'],seed=record['seed'],epoch=record['epoch'],readout=rid,state_id=point['state_id'],**D.metrics(point)))
    table(out/'COSTS.csv.gz',costs);table(out/'CURVES.csv.gz',curves)
    freeze=read_json(out/'GLOBAL_SELECTION_FREEZE.json');decisions=read_json(out/'DECISIONS.json');formal=[]
    ab_interactions(out,decisions,endpoints)
    for primary_record in freeze['primary_by_condition']:
        ds=primary_record['dataset'];primary=primary_record['candidate'];condition=primary_record['condition']
        for method in prepared['manifest']['evaluation']['published_baselines']['filters']['method']:
            subset=[r for r in comparisons if r['dataset']==ds and r['condition']==condition and r['candidate']==primary and r['control']=='PUBLISHED:'+method]
            macro=next(r for r in subset if r['metric']=='macro_auc');ll=next(r for r in subset if r['metric']=='macro_logloss');cells=[r for r in subset if r['metric'] in [c+'_auc' for c in CELLS]]
            passed=macro['mean']>=.003 and all(r['mean']>=-.001 for r in cells)
            formal.append(dict(dataset=ds,condition=condition,candidate=primary,control=method,primary_baseline=method==prepared['manifest']['datasets'][ds]['primary_published_baseline'],
                auc_pass=passed,delta_auc=macro['mean'],delta_logloss=ll['mean'],cell_guard=all(r['mean']>=-.001 for r in cells),
                decision='AUC_PASS_LL_COST' if passed and ll['mean']>=0 else 'AUC_PASS' if passed else 'AUC_BELOW_FORMAL_GATE'))
    write_json(out/'FORMAL_DECISIONS.json',formal)
    lines=['# EMA 下一阶段实验：数值结果','',f'规划提交 `{prepared["identity"]["plan_commit"]}`；源码提交 `{prepared["identity"]["source_commit"]}`。统计单位为五个 seed（2022–2026）。','',
        '| 数据集/条件 | 主候选 | 发表模型对照 | ΔMacro AUC | ΔMacro LogLoss | AUC 判定 |','|---|---|---|---:|---:|---|']
    for r in formal:lines.append(f'| {r["dataset"]}/{r["condition"]} | {r["candidate"]} | {r["control"]} | {r["delta_auc"]:+.8f} | {r["delta_logloss"]:+.8f} | {r["decision"]} |')
    lines+=['','所有阶段门及条件分支见 `DECISIONS.json`；逐 seed、六格、Macro、Traffic 和 Pooled 指标见 `ENDPOINTS.csv.gz`；配对原值、差值、样本标准差与 t4 区间见 `PAIRED.csv.gz`。完整验证曲线见 `CURVES.csv.gz`，实际训练与读出成本见 `COSTS.csv.gz`。','']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    appendix=['# 定量附录','','每项依序列出 seed 2022、2023、2024、2025、2026；置信区间使用 t4=2.7764451052。','',
              '| 数据集/条件 | 候选 − 对照 | 指标 | 五 seed 配对差 | 均值 | SD | 95% CI |','|---|---|---|---|---:|---:|---|']
    for r in comparisons:appendix.append(f'| {r["dataset"]}/{r["condition"]} | {r["candidate"]} − {r["control"]} | {r["metric"]} | '+', '.join(f'{v:+.8f}' for v in r['values'])+f' | {r["mean"]:+.8f} | {r["sd"]:.8f} | [{r["ci95"][0]:+.8f}, {r["ci95"][1]:+.8f}] |')
    (out/'QUANTITATIVE_APPENDIX.md').write_text('\n'.join(appendix)+'\n')
    (out/'PROVENANCE.md').write_text('# Provenance\n\n'+json.dumps(dict(identity=prepared['identity'],data=prepared['data_identity'],baseline_sha256=prepared['published_baseline_sha256'],freeze_sha256=sha_file(out/'GLOBAL_SELECTION_FREEZE.json')),indent=2)+'\n')
    index={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha_file(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('guardian.json','scheduler.lock','ARTIFACT_INDEX.json') and not p.name.endswith('.log')}
    write_json(out/'ARTIFACT_INDEX.json',dict(status='COMPLETE',files=index,logical_epoch_records=sum(r['epoch']>0 for r in costs),identity=prepared['identity']))
    disk_admit(out);return dict(status='COMPLETE',endpoint_rows=len(endpoints),paired_rows=len(comparisons),logical_epoch_records=len(costs))
