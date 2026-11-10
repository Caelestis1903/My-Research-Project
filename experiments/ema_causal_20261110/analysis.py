"""Committed scalar aggregation, explicit scientific freeze and five-state Test."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from .protocol import SEEDS,CONDITIONS,read_json,write_json,write_csv,sha_file,digest
from .state import request,disk_admit


def _complete(out):
    records=[]
    prepared=read_json(out/'PREPARED.json')
    for job in prepared['jobs']:
        paths=sorted((out/'runs'/job['job_id']/'epochs').glob('E*.json'))
        rows=[read_json(p) for p in paths]
        if len(rows)!=50 or [r['epoch'] for r in rows]!=list(range(1,51)) or rows[-1]['global_step']!=50*job['steps_per_epoch']:raise RuntimeError('Incomplete trajectory: '+job['job_id'])
        records.extend(rows)
    if sum(r['cost']['real_adam_steps'] for r in records)!=34500:raise ValueError('Adam count differs')
    return prepared,records


def summarize(values):
    values=np.asarray(values,dtype=float)
    if len(values)!=5 or not np.isfinite(values).all():raise ValueError('Five finite seed values required')
    mean=float(values.mean());sd=float(values.std(ddof=1));width=2.7764451052*sd/np.sqrt(5)
    return dict(mean=mean,sd=sd,ci95_low=mean-width,ci95_high=mean+width,n=5,positive=int((values>0).sum()),negative=int((values<0).sum()),seed_values=values.tolist())


def analyze(study,out,stage):
    prepared,records=_complete(out);curves=[];pairs=[]
    for r in records:
        meta={k:r[k] for k in ('job_id','dataset','objective','seed','epoch','global_step')}
        for state,values in r['validation'].items():curves.append(dict(meta,readout=state,**values))
        pairs.append(dict(meta,delta_auc=r['validation']['F']['macro_auc']-r['validation']['O']['macro_auc'],delta_logloss=r['validation']['F']['macro_logloss']-r['validation']['O']['macro_logloss']))
    write_csv(out/'VALIDATION_CURVES.csv',curves);write_csv(out/'SAME_TIME_SEED_EFFECTS.csv',pairs)
    stages=[]
    for ds,obj in CONDITIONS:
        for name,epochs in (('early',[5,10]),('middle',[20]),('late',[35,50])):
            for metric in ('delta_auc','delta_logloss'):
                values=[float(np.mean([r[metric] for r in pairs if r['dataset']==ds and r['objective']==obj and r['seed']==seed and r['epoch'] in epochs])) for seed in SEEDS]
                stages.append(dict(dataset=ds,objective=obj,training_stage=name,metric=metric,**summarize(values)))
    write_csv(out/'STAGE_SEED_SUMMARY.csv',stages)
    blocks=[]
    for ds,obj in CONDITIONS:
        p=out/'blocks'/(ds+'__'+obj)/'BLOCK_COMPLETE.json'
        if not p.exists():raise RuntimeError('Block live-state analysis incomplete: '+str(p))
        blocks.append(dict(block=ds+'__'+obj,path=str(p.relative_to(out)),sha256=sha_file(p)))
    receipt=dict(stage=stage,status='COMPLETE',jobs=20,seeds=SEEDS,real_adam_steps=34500,blocks=blocks,scientific_unit='seed',inference_owner='orchestrator')
    write_json(out/f'STAGE{stage}_COMPLETE.json',receipt);return receipt


def freeze(study,out):
    prepared,records=_complete(out)
    for stage in range(1,5):read_json(out/f'STAGE{stage}_COMPLETE.json')
    findings=read_json(out/'MECHANISM_FINDINGS.json')
    if not isinstance(findings,dict) or findings.get('author_role')!='orchestrator' or not findings.get('main_direction') in ('A','B','C','D'):raise ValueError('Orchestrator scientific findings and main direction required')
    questions=findings.get('findings',[])
    if {r.get('question_id') for r in questions}!={'Q1','Q2','Q3','Q4','Q5'}:raise ValueError('Q1–Q5 scientific findings required')
    rows=[]
    for job in prepared['jobs']:
        row=request(out,'get',key=job['job_id'])['row']
        if row is None or row['status']!='FINALIZED':raise RuntimeError('Finalized live endpoints required')
        selected=row['state']['selected']
        for kind in ('O','F'):
            curve=[r for r in records if r['job_id']==job['job_id']]
            best=max(curve,key=lambda r:(r['validation'][kind]['macro_auc'],-r['epoch']))
            if selected[kind]['epoch']!=best['epoch']:raise ValueError('Selected epoch changed')
        for name in ('O','F','OatF','S_atF','N_atF'):
            point=selected[name if name in selected else 'F']
            rows.append(dict(job_id=job['job_id'],dataset=job['dataset'],objective=job['objective'],seed=job['seed'],readout=name,epoch=point['epoch'],parent_hash=point['state_hash']))
    binding={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*.json')) if 'data_cache' not in p.parts and p.name not in ('guardian.json','SELECTION_FROZEN.json','MECHANISM_FREEZE.json','RELEASE_RECEIPT.json') and 'SMOKE' not in p.name}
    result=dict(status='FROZEN',findings_hash=sha_file(out/'MECHANISM_FINDINGS.json'),bindings=binding,selection=rows,test_states=['O','F','OatF','S_atF','N_atF'],time=time.time())
    write_json(out/'SELECTION_FROZEN.json',rows);write_json(out/'MECHANISM_FREEZE.json',result);return dict(status='FROZEN',logical_endpoints=len(rows))


def evaluate(study,out,device):
    from experiments.overnight_ab_20261103.data_bridge import cached_bundle
    from experiments.aug_history_20261105.model_adapter import build_model
    from experiments.aug_history_20261105.registry import ParameterLayout
    from experiments.aug_history_20261105.group_registry import build_path_groups
    from .interventions import group_masks,vector_hash
    from .trajectory import full_prediction
    freeze=read_json(out/'MECHANISM_FREEZE.json')
    for path,expected in freeze['bindings'].items():
        if sha_file(out/path)!=expected:raise ValueError('Frozen scientific input changed: '+path)
    prepared=read_json(out/'PREPARED.json');rows=[];cache={}
    for job in prepared['jobs']:
        ds=job['dataset']
        if ds not in cache:cache[ds]=cached_bundle(prepared['cache_root'],ds,include_test=True)
        bundle=cache[ds];model=build_model(bundle,job['config']).to(device);layout=ParameterLayout.from_model(model);masks=group_masks(build_path_groups(model,bundle));selected=request(out,'get',key=job['job_id'])['row']['state']['selected']
        states={k:selected[k]['state'] for k in ('O','F','OatF')}
        for group in ('S','N'):
            value=states['OatF'].clone();value[masks[group]]=states['F'][masks[group]];states[group+'_atF']=value
        physical={}
        for name,value in states.items():
            h=vector_hash(value)
            if h not in physical:layout.copy_to(model,value);physical[h]=full_prediction(model,bundle,job['config']['eval_batch_size'],split='test')[0]
            point=selected[name if name in selected else 'F']
            rows.append(dict(job_id=job['job_id'],dataset=ds,objective=job['objective'],seed=job['seed'],readout=name,epoch=point['epoch'],state_hash=h,split='test',**physical[h]))
        del model;torch.cuda.empty_cache()
    write_csv(out/'ENDPOINTS.csv',rows);write_json(out/'TEST_RECEIPT.json',dict(status='COMPLETE',logical_states=len(rows),freeze_hash=sha_file(out/'MECHANISM_FREEZE.json'),time=time.time()));return dict(status='COMPLETE',logical_states=len(rows))


def report(study,out):
    import csv
    findings=read_json(out/'MECHANISM_FINDINGS.json');test=read_json(out/'TEST_RECEIPT.json')
    with (out/'ENDPOINTS.csv').open() as f:rows=list(csv.DictReader(f))
    summaries=[]
    for ds,obj in CONDITIONS:
        for metric in ('macro_auc','macro_logloss'):
            vals=[]
            for seed in SEEDS:
                select={r['readout']:float(r[metric]) for r in rows if r['dataset']==ds and r['objective']==obj and int(r['seed'])==seed};vals.append(select['F']-select['O'])
            summaries.append(dict(dataset=ds,objective=obj,metric=metric,**summarize(vals)))
    write_csv(out/'SELECTED_TEST_PAIRED.csv',summaries)
    text=['# 固定 EMA 因果与功能归因研究报告','',f"主方向：{findings['main_direction']}。",'']
    for finding in findings['findings']:
        text.extend([f"## {finding['question_id']}",'',finding.get('claim',finding.get('conclusion','')),'',f"证据强度：{finding.get('evidence_strength','WEAK')}。",''])
    text+=['## 五 seed 独立选点 Test','', '| 数据 | 目标 | 指标F−O | 均值 | SD | 95% CI |','|---|---|---|---:|---:|---|']
    for r in summaries:text.append(f"| {r['dataset']} | {r['objective']} | {r['metric']} | {r['mean']:.7f} | {r['sd']:.7f} | [{r['ci95_low']:.7f}, {r['ci95_high']:.7f}] |")
    text+=['','完整逐seed、六域任务指标及状态身份见ENDPOINTS.csv；同刻和独立选点分别保留。每条件n=5；各epoch/cell不扩充样本量。','', '本轮20条轨迹、1000epoch、34,500次真实Adam；源训练配置、自然人口和数据身份见RESOLVED_PROTOCOL.json与DATA_MANIFEST.json。参数历史仅保留于RAM。', '', '功能/参数/时间/pair/桥接原始表见blocks各条件ARTIFACT_INDEX.json；统计和来源身份见同目录收据。']
    (out/'FINAL_REPORT.md').write_text('\n'.join(text)+'\n');disk_admit(out);return dict(status='COMPLETE',report=str(out/'FINAL_REPORT.md'))
