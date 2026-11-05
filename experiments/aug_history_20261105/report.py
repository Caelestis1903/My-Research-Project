"""Scalar tables and frozen-rule dispatch; scientific interpretation is written by the orchestrator."""
import csv
import json
import math
import statistics
from pathlib import Path
from experiments.overnight_ab_20261103.inventory import read_csv,scalar_row
from experiments.overnight_ab_20261103.selection import CELLS,score
from experiments.overnight_ab_20261103.state import read_json,write_json,sha_file,digest
from .protocol import SEEDS, ROOT, published_reference
from .selection import require_frozen, Incomplete


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def archived_online_cost(dataset):
    path=ROOT/'research_reports/2026-11-04_gaussian_domain_coverage/ENDPOINTS.csv'
    if not path.exists():
        return 'unknown',[]
    if sha_file(path) != '5de7c064542d19b79bcae49325c49f7e40fb35f41b21e7e1b0a236cb0bf69875':
        raise ValueError('Archived online comparison source changed')
    rows=[scalar_row(r) for r in read_csv(path)]
    groups={arm:{r['seed']:r for r in rows if r['dataset']==dataset and r['arm']==arm and
        r['estimator']=='online' and r['checkpoint']=='E50' and r['split']=='validation'} for arm in ('G-D0','M3OE')}
    if any(sorted(v)!=SEEDS for v in groups.values()):
        return 'unknown',[]
    delta={cell:math.fsum(groups['G-D0'][seed][cell+'_auc']-groups['M3OE'][seed][cell+'_auc'] for seed in SEEDS)/5 for cell in CELLS}
    return any(value<-.001 for value in delta.values()),[dict(path=str(path),sha256=sha_file(path),
        dataset=dataset,checkpoint='E50',estimator='online',split='validation',
        comparison='G-D0 minus M3OE',seed_ids=SEEDS,six_cell_auc_deltas=delta)]


def decide(output):
    output=Path(output); route=output/'B'
    frozen=require_frozen(output,route/'selection_frozen.json')
    published,_=published_reference()
    decisions={}
    for dataset,current in frozen['datasets'].items():
        points={(r['candidate_id'],r['seed']):r for r in current['endpoints'] if r['checkpoint']=='selected'}
        winners={f:row['candidate_id'] for f,row in current['winners'].items()}
        def v(cid):
            return math.fsum(score(points[cid,seed]['validation']) for seed in SEEDS)/5
        def cells(a,b):
            return {cell:math.fsum(points[a,seed]['validation'][cell+'_auc']-points[b,seed]['validation'][cell+'_auc'] for seed in SEEDS)/5 for cell in CELLS}
        d=winners['D']; gains=cells(d,'F')
        direction=min(v(d)-v(winners['S']),v(d)-v(current['matched_at_D']['S']))
        multi=min(v(d)-v(winners['M']),v(d)-v(current['matched_at_D']['M']))
        simple_near=v(d)-max(v(winners[f]) for f in ('U','M','S'))<=.0002
        tradeoff=any(value<-.001 for value in gains.values())
        online_supported,sources=archived_online_cost(dataset)
        residual={cell:math.fsum(points[d,seed]['validation'][cell+'_auc']-next(
            r[cell+'_auc'] for r in published if r['dataset']==dataset and r['candidate_id']=='M3oE_fixed1' and
            r['seed']==seed and r['checkpoint']=='selected' and r['split']=='validation') for seed in SEEDS)/5 for cell in CELLS}
        if v(d)<=v('F'):
            rule,action='B_NO_INCREMENT','RUN_C'
        elif tradeoff:
            rule,action='B_AGGREGATION_TRADEOFF','RUN_A'
        elif online_supported is True and any(value<-.001 for value in residual.values()):
            rule,action='B_ONLINE_COST','RUN_C'
        elif simple_near or direction<=0 or multi<=0:
            rule,action='B_SIMPLE_SUFFICIENT','RUN_A'
        else:
            rule,action='B_RETAIN','RETAIN_B_AND_REPORT'
        next_actions=[] if action=='RETAIN_B_AND_REPORT' else [dict(route=action[-1],dataset=dataset,matrix=action[-1]+'_FULL',seeds=SEEDS)]
        decision=dict(schema_version=1,study_id='AUG_HISTORY_20261105',route='B',dataset=dataset,status='DECIDED',
            manifest_hash=frozen['manifest_hash'],selection_hash=frozen['selection_hash'],seed_ids=SEEDS,
            all_required_candidates_complete=True,winners=winners,
            validation=dict(macro_auc_by_winner={f:v(cid) for f,cid in winners.items()},F_macro_auc=v('F'),
                six_cell_auc_deltas=gains,direction_gain=direction,multi_response_gain=multi,
                simple_near=simple_near,aggregation_tradeoff=tradeoff,
                online_cost_supported=online_supported,online_cost_sources=sources,
                residual_cell_reference='M3oE_fixed1 / A0 online selected validation',residual_cell_auc_deltas=residual),
            test_report=dict(status='NOT_YET_EVALUATED',formal_gains={},component_gains={},six_cell_deltas={}),
            decision_basis=dict(split='validation',test_consulted=False,historical_test_exposure=True,
                source_hashes=[s['sha256'] for s in sources]),
            cost=dict(online_trajectories=5,estimator_seed_instances=75,**current['cost']),
            matched_rule=rule,next_action=action,next_actions=next_actions,
            deferred_routes=[r for r in ('A','C') if r not in [a['route'] for a in next_actions]],
            reason=dict(D_minus_F=v(d)-v('F'),direction_gain=direction,multi_response_gain=multi,simple_near=simple_near))
        path=route/dataset/'DECISION.json'
        if path.exists() and read_json(path)!=decision:
            raise ValueError('Frozen validation decision changed')
        write_json(path,decision); decisions[dataset]=decision
    result=dict(route='B',basis='validation',selection_hash=frozen['selection_hash'],datasets=decisions,
        next_actions=[a for d in decisions.values() for a in d['next_actions']])
    write_json(route/'DECISION.json',result)
    return result


def endpoint_row(job,candidate,endpoint,point,split,values,layout_hash):
    config=next((r for r in job['candidates'] if r['id']==candidate),dict(id='online',family='ONLINE'))
    return dict(dataset=job['dataset'],candidate_id=candidate,family=config['family'],seed=job['seed'],
        trajectory_id=job['trajectory_id'],estimator=candidate,checkpoint=endpoint,epoch=point['epoch'],
        model_hash=point['model_hash'],stage='train',split=split,
        config_json=dict(parent=job['config'],candidate=config),config_hash=digest(dict(parent=job['config'],candidate=config)),
        source_hash=job['source_hash'],manifest_hash=job['manifest_hash'],data_hash=job['data_hash'],
        split_hash=digest(job['config']['source']['split_row_id_hashes']),
        preprocessing_hash=digest({k:job['config']['source'][k] for k in ('encoder_hash','scaler_hash')}),
        layout_hash=layout_hash,**values)


def evaluate(output,frozen_path,device='cuda:0'):
    import torch
    from . import guardian
    from .model_adapter import build_model
    from experiments.overnight_ab_20261103.data_bridge import cached_bundle
    from experiments.overnight_ab_20261103.evaluation import evaluate_model
    from experiments.overnight_ab_20261103.state import state_digest
    output=Path(output); route=output/'B'; prepared=read_json(output/'PREPARED.json')
    frozen=require_frozen(output,frozen_path)
    if not (route/'DECISION.json').exists():
        raise Incomplete('Freeze validation dispatch before test evaluation')
    destination=route/'ENDPOINTS.csv'
    if (route/'EVALUATED.json').exists():
        marker=read_json(route/'EVALUATED.json')
        if marker['selection_hash']!=frozen['selection_hash'] or marker['endpoints_sha256']!=sha_file(destination):
            raise ValueError('Previously evaluated endpoint archive changed')
        return marker
    rows=[]
    for job in prepared['jobs']:
        snapshot=guardian.request(output,'get',key=job['job_id'])['state']
        if snapshot is None or not snapshot['complete']:
            raise Incomplete('RAM endpoints lost; replay train --resume before evaluate')
        bundle=cached_bundle(prepared['cache_root'],job['dataset'],include_test=True)
        model=build_model(bundle,job['config']).to(device)
        for frozen_point in frozen['datasets'][job['dataset']]['endpoints']:
            if frozen_point['seed']!=job['seed']:
                continue
            cid,endpoint=frozen_point['candidate_id'],frozen_point['checkpoint']
            point=snapshot['snapshots'][cid][endpoint]
            if point['model_hash']!=frozen_point['model_hash'] or point['epoch']!=frozen_point['epoch']:
                raise ValueError('RAM checkpoint differs from frozen endpoint')
            model.load_state_dict(point['model_state'])
            if state_digest(model.state_dict())!=point['model_hash']:
                raise ValueError('RAM model content checksum changed')
            result=evaluate_model(model,bundle,'test','G-D0',job['config']['eval_batch_size'])
            for split,values in (('validation',point['validation']),('test',result)):
                rows.append(endpoint_row(job,cid,endpoint,point,split,values,snapshot['layout_hash']))
        del model,snapshot,bundle
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    published,reference=published_reference()
    rows.extend(published)
    write_csv(destination,rows)
    marker=dict(status='EVALUATED',selection_hash=frozen['selection_hash'],endpoints_sha256=sha_file(destination),
        new_endpoint_rows=len(rows)-len(published),published_reference=reference)
    write_json(route/'EVALUATED.json',marker)
    return marker


def report(output):
    output=Path(output); route=output/'B'
    if not (route/'EVALUATED.json').exists():
        raise Incomplete('Complete frozen test endpoints before report')
    frozen=require_frozen(output,route/'selection_frozen.json')
    rows=[scalar_row(r) for r in read_csv(route/'ENDPOINTS.csv')]
    metric_names=[k for k in rows[0] if k.endswith(('_auc','_logloss'))]
    pairs,summaries=[],[]
    for dataset,current in frozen['datasets'].items():
        w={k:v['candidate_id'] for k,v in current['winners'].items()}
        comparisons=[(w['D'],ref) for ref in dict.fromkeys(['F',w['U'],w['M'],w['S'],*current['matched_at_D'].values()])]
        comparisons += [(cid,ref) for cid in dict.fromkeys(['F',*w.values()]) for ref in ('M3oE_fixed1','HyperGate','AutoIFS')]
        for method,reference in comparisons:
            for split in ('validation','test'):
                for endpoint in ('selected','E10','E50'):
                    peers={cid:{r['seed']:r for r in rows if r['dataset']==dataset and r['candidate_id']==cid and
                        r['split']==split and r['checkpoint']==endpoint} for cid in (method,reference)}
                    if any(sorted(v)!=SEEDS for v in peers.values()):
                        raise Incomplete('Pairing requires exactly the five matching seeds')
                    for metric in metric_names:
                        delta=[peers[method][s][metric]-peers[reference][s][metric] for s in SEEDS]
                        mean=statistics.mean(delta); sd=statistics.stdev(delta); margin=2.7764451051977987*sd/math.sqrt(5)
                        for seed,difference in zip(SEEDS,delta):
                            pairs.append(dict(dataset=dataset,method=method,reference=reference,split=split,
                                checkpoint=endpoint,metric=metric,seed=seed,method_value=peers[method][seed][metric],
                                reference_value=peers[reference][seed][metric],delta=difference))
                        summaries.append(dict(dataset=dataset,method=method,reference=reference,split=split,
                            checkpoint=endpoint,metric=metric,n=5,mean_delta=mean,sd_delta=sd,
                            ci95_low=mean-margin,ci95_high=mean+margin,positive_seeds=sum(d>0 for d in delta),
                            method_mean=statistics.mean(peers[method][s][metric] for s in SEEDS),
                            method_sd=statistics.stdev(peers[method][s][metric] for s in SEEDS),
                            reference_mean=statistics.mean(peers[reference][s][metric] for s in SEEDS)))
    write_csv(route/'PAIRED_SEED.csv',pairs)
    write_csv(route/'PAIRED_SUMMARY.csv',summaries)
    prepared=read_json(output/'PREPARED.json')
    costs=[]
    for job in prepared['jobs']:
        records=[read_json(guardian_path) for guardian_path in sorted((output/'runs'/job['job_id']/'epochs').glob('E*.json'))]
        totals={key:max(r['cost'][key] for r in records) if key.startswith('peak_') else sum(r['cost'][key] for r in records) for key in records[0]['cost']}
        costs.append(dict(dataset=job['dataset'],seed=job['seed'],trajectory_id=job['trajectory_id'],**totals))
    write_csv(route/'COSTS.csv',costs)
    write_json(output/'RUN_MANIFEST.json',dict(prepared,status='B_EVALUATED',selection_hash=frozen['selection_hash'],
        raw_tables={p.name:sha_file(p) for p in route.glob('*.csv')},costs=costs))
    lines=['# 11/05 增强响应与历史估计：实验数据报告','',
        'B：两数据集、2022–2026 五 seed、每条 50 epoch，15 个非反馈影子；共 10 条在线轨迹。',
        '各候选完整验证，超参数与端点按 validation 冻结。A/C 的状态见 DECISION.json 和后续执行记录。',
        '', '| 数据集 | 家族 | 验证赢家 |', '|---|---|---|']
    for dataset,current in frozen['datasets'].items():
        for family,value in current['winners'].items():
            lines.append(f'| {dataset} | {family} | {value["candidate_id"]} |')
    lines += ['', '逐 seed、六格、Macro/Traffic/Pooled、selected/E10/E50、配对均差与 95% 区间分别见 ENDPOINTS.csv、PAIRED_SEED.csv、PAIRED_SUMMARY.csv；成本见 COSTS.csv。',
        'HyperGate 沿用来源标记 UNVERIFIED_PROPOSAL。科学解释由主代理结合完整数表撰写。','']
    (route/'REPORT.md').write_text('\n'.join(lines))
    return dict(status='B_REPORTED',rows=len(rows),paired_rows=len(pairs),summary_rows=len(summaries))
