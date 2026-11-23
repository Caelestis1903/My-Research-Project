"""Frozen GA31-R3 gates and numerical records; source states stay in RAM."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .vendor_ga31.protocol import (setup, optimizer, cpu_copy, equal, rng_state, restore_rng,
    snapshot, restore, tensor_bytes, send_memory, receive_memory, write_json, append_csv)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEFAULT_PROTOCOL=ROOT/'research_plans/26-10-31_v2/phase1_protocol.json'


def protocol(path=DEFAULT_PROTOCOL):
    p=json.loads(Path(path).read_text())
    assert p['study_id']=='GA31-R3' and p['seeds']==list(range(2022,2027))
    assert p['training']['epochs']==50 and p['training']['batch_size']==40960
    assert p['budget']['required_formal_runs']==35 and p['budget']['maximum_formal_runs']==45
    assert p['stage_B']['mandatory_count']==20
    for name in p['unsupported']:
        if p.get(name) or p.get('flags',{}).get(name):raise ValueError(f'Unsupported: {name}')
    old=json.loads((HERE/'vendor_ga31/resolved_protocol.json').read_text())
    p['model']=old['model'];p['datasets']=old['datasets']
    p['source_commit']=p['source']['commit']
    p['training']['save_epochs']=p['state']['snapshot_epochs']
    return p


def derive_seed(key):
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8],'little') & ((1<<63)-1)


def view_key(dataset,seed,phase,step,family,view,batch=None):
    middle=f'{phase}|{step}' if batch is None else f'{phase}|{step}|{batch}'
    return f'GA31-R3|{dataset}|{seed}|{middle}|{family}|{view}'


def arm_config(p,arm,dataset,baselines=None):
    for a in p['stage_A']['runs']:
        if a['arm']==arm:
            assert a['dataset']==dataset
            return dict(a,view_family='none',num_views=0,pairing='iid',stage='A')
    base=baselines[dataset] if isinstance(baselines[dataset],str) else baselines[dataset]['arm']
    config=arm_config(p,base,dataset)
    method=next(a for a in p['stage_B']['mandatory']+p['stage_B']['conditional'] if a['arm']==arm)
    return dict(config,arm=arm,stage='B',view_family=method['family'],num_views=method['num_views'],pairing=method['pairing'])


def optimizer_for(model,p,config):
    q=dict(p);q['optimizer']=dict(p['optimizer'],weight_decay=config['weight_decay'],eps=config['eps'])
    return optimizer(model,q)


def stage_a_jobs(p):
    return [(a['dataset'],a['arm'],s) for s in p['seeds'] for a in p['stage_A']['runs']]


def stage_b_jobs(p,gates):
    return [(d,arm,s) for s in p['seeds'] for d in p['datasets']
            for arm in ['M1','G-IID2']+(['G-ANTI2'] if gates[d]['triggered'] else [])]


def guards(deltas,p,dataset):
    largest=int(np.argmax(p['datasets'][dataset]['train_domain_counts']))
    keys=[f'd{largest}t{t}_auc' for t in range(2)]+['pooled_task_mean_auc','worst_cell_auc']
    result=[]
    for metric in keys+['macro_logloss']:
        v=np.asarray(deltas[metric],dtype=float)
        bad=(v>0) if metric=='macro_logloss' else (v<0)
        mean=float(v.mean());threshold=.002 if metric=='macro_logloss' else -.002
        if (mean>=threshold if metric=='macro_logloss' else mean<=threshold) and int(bad.sum())>=4:
            result.append(dict(metric=metric,mean=mean,adverse_seeds=int(bad.sum())))
    return result


def choose_baselines(results,p):
    by={(r['dataset'],r['arm'],r['seed']):r['selected_validation'] for r in results}
    assert len(by)==15
    records=[];deltas={}
    for seed in p['seeds']:
        a=by['movielens','L2-ML',seed];b=by['movielens','B-ML',seed]
        records.append(dict(seed=seed,baseline_auc=b['macro_auc'],l2_auc=a['macro_auc'],delta=a['macro_auc']-b['macro_auc']))
        for k in a:deltas.setdefault(k,[]).append(a[k]-b[k])
    values=np.array(deltas['macro_auc']);bad=guards(deltas,p,'movielens')
    substitute=abs(float(values.mean()))<=.0005 and int((abs(values)<=.001).sum())>=4 and not bad
    mean_base=float(np.mean([r['baseline_auc'] for r in records]));mean_l2=float(np.mean([r['l2_auc'] for r in records]))
    chosen='L2-ML' if substitute or mean_l2>=mean_base-1e-6 else 'B-ML'
    return dict(study_id=p['study_id'],baselines={'movielens':chosen,'kuairand':'B-KR'},
                validation_only=True,ml_practical_substitute=substitute,ml_pairs=records,
                validation_guards=bad,mean_validation_auc={'B-ML':mean_base,'L2-ML':mean_l2},
                reason='practical L2 substitution' if substitute else 'validation mean with L2 tie preference')


def anti_gate(observations,p):
    decisions={}
    for dataset in p['datasets']:
        ratios=[];detail=[]
        for seed in p['seeds']:
            rows=[r for r in observations if r['dataset']==dataset and r['seed']==seed and r['gate_anchor'] and r['valid_variance']]
            # Anchor batches are already deduplicated before measurement.
            valid=bool(rows)
            ratio=float(np.mean([r['V_anti'] for r in rows])/np.mean([r['V_iid'] for r in rows])) if valid else None
            detail.append(dict(seed=seed,valid_observations=len(rows),ratio=ratio))
            if valid:ratios.append(ratio)
        triggered=len(ratios)==5 and float(np.mean(ratios))<=.8 and sum(r<1 for r in ratios)>=4
        decisions[dataset]=dict(triggered=triggered,valid_seeds=len(ratios),mean_ratio=float(np.mean(ratios)) if ratios else None,
                                below_one=sum(r<1 for r in ratios),per_seed=detail)
    return decisions
