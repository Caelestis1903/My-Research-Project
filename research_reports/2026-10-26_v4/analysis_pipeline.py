"""Analysis only: paired fixed predictions, seed statistics, shared cluster draws."""
from pathlib import Path
import argparse, csv, hashlib, json, sys, time
import numpy as np

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from experiments.mdmt_dfd_20261026.data import RoleBundle
from experiments.mdmt_dfd_20261026.metrics import evaluate_probabilities,losses,paired_summary
from experiments.mdmt_dfd_20261026.plans import rng

ROOT=Path('/home/xzf/Code/phase0-3/artifacts/mdmt_dfd_v4/v4_0')
METHODS=['NATIVE','LOCAL','H_DIRECT','ETA_DIRECT','UNIFORM','GEOM','SHUFFLE','CONST_UNIFORM','LOCAL_CAP','DFD']
ROLE_COMPONENT={'P_form_OOF':1,'Q_explore':2,'validation':3,'test_exposed':4,'Q_legacy_confirm_exposed':5}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_prediction(path):
    with np.load(path) as a:
        ids=a['row_id'] if 'row_id' in a else a['rows'];order=np.argsort(ids)
        return dict(row_id=ids[order],y=a['y'][order],domain=a['domain'][order],p=a['p'][order],
                    checkpoint_hash=str(a['checkpoint_hash']),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

class FixedRisk:
    """Weighted BCE and tie-aware Mann-Whitney AUC, fixed-score order reused."""
    def __init__(self,y,p,domain):
        self.cells=[];ll=losses(y,p)
        for q in range(2):
            for d in range(3):
                ids=np.flatnonzero(domain==d);ids=ids[np.argsort(p[ids,q],kind='stable')]
                scores=p[ids,q];starts=np.r_[0,np.flatnonzero(np.diff(scores)!=0)+1] if len(ids) else np.array([],dtype=int)
                self.cells.append((ids,y[ids,q].astype(np.float64),ll[ids,q],starts))
    def compute(self,weights):
        ll=[];aa=[]
        for ids,y,loss,starts in self.cells:
            w=weights[ids];den=w.sum()
            ll.append(float(np.dot(w,loss)/den) if den else np.nan)
            if not len(ids):aa.append(np.nan);continue
            pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(1-y),starts)
            product=pos.sum()*neg.sum()
            aa.append(float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/product) if product else np.nan)
        return np.asarray([np.mean(ll),np.mean(aa),*ll,*aa],dtype=np.float64)

def bootstrap_all(data,seed,role,rows,y,domain,predictions,out):
    references=[m for m in predictions if m!='DFD']
    ids=None if data.user_ids is None else data.user_ids[rows]
    clusters=np.arange(len(rows)) if ids is None else ids
    unique,inverse=np.unique(clusters,return_inverse=True)
    generator=rng(seed,data.dataset,70,component=ROLE_COMPONENT[role])
    estimators={m:FixedRisk(y,p,domain) for m,p in predictions.items()}
    draws=np.empty((1000,len(references),14));max_check=0.;started=time.perf_counter()
    for b in range(1000):
        counts=np.bincount(generator.integers(len(unique),size=len(unique)),minlength=len(unique));weights=counts[inverse]
        values={m:e.compute(weights) for m,e in estimators.items()}
        if b==0:
            for m,p in predictions.items():
                official=evaluate_probabilities(y,p,domain,weights)
                reference=np.asarray([official['macro_logloss'],official['macro_auc'],*[c['logloss'] for c in official['cells']],*[c['auc'] for c in official['cells']]],dtype=np.float64)
                max_check=max(max_check,float(np.nanmax(np.abs(reference-values[m]))))
            if max_check>1e-12:raise RuntimeError('Weighted metric analysis reference mismatch')
        for j,m in enumerate(references):draws[b,j]=values['DFD']-values[m]
    names=['macro_LL','macro_AUC']+[f'cell{c}_LL' for c in range(6)]+[f'cell{c}_AUC' for c in range(6)]
    result=dict(dataset=data.dataset,seed=seed,role=role,repetitions=1000,
                cluster='original_user' if ids is not None else 'row_fallback_independence_unverified',
                unique_clusters=len(unique),direction='DFD minus reference: LL negative favorable, AUC positive favorable',
                common_draws_for_all_methods_tasks_domains=True,rng_coordinates=[seed,data.dataset_id,70,0,0,ROLE_COMPONENT[role],0],
                first_draw_weighted_metric_reference_max_abs=max_check,analysis_wall_seconds=time.perf_counter()-started,
                comparisons={})
    for j,m in enumerate(references):
        result['comparisons'][m]=dict(percentile95=dict(zip(names,np.nanquantile(draws[:,j],[.025,.975],axis=0).T.tolist())),
                                     invalid_resamples=dict(zip(names,(~np.isfinite(draws[:,j])).sum(0).tolist())))
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'seed{seed}_bootstrap1000_draws.npz',deltas=draws,reference=np.asarray(references),metric=np.asarray(names))
    write(out/f'seed{seed}_bootstrap1000.json',result)
    return result

def analyze_files(dataset,role,state,files_by_seed,output,with_bootstrap=True):
    data=RoleBundle(dataset);all_metrics={};audits=[];cellrows=[];macrorows=[]
    for seed,files in files_by_seed.items():
        bundles={m:load_prediction(Path(p)) for m,p in files.items()}
        base=next(iter(bundles.values()));rows,y,domain=base['row_id'],base['y'],base['domain']
        for m,b in bundles.items():
            if not np.array_equal(rows,b['row_id']) or not np.array_equal(y,b['y']) or not np.array_equal(domain,b['domain']):
                raise RuntimeError('Methods have different row/label/domain identity')
        metrics={m:evaluate_probabilities(y,b['p'],domain) for m,b in bundles.items()};all_metrics[seed]=metrics
        for m,v in metrics.items():
            macrorows.append(dict(dataset=dataset,seed=seed,method=m,role=role,state=state,macro_LL=v['macro_logloss'],macro_AUC=v['macro_auc'],natural_LL=v['natural_logloss'],pooled_AUC0=v['pooled_auc'][0],pooled_AUC1=v['pooled_auc'][1],clip_rate=v['clip_rate']))
            for c in v['cells']:cellrows.append(dict(dataset=dataset,seed=seed,method=m,role=role,state=state,**c))
        audits.append(dict(seed=seed,files={m:{k:b[k] for k in ['path','sha256','checkpoint_hash']} for m,b in bundles.items()}))
        target=output/f'seed{seed}_bootstrap1000.json'
        if with_bootstrap and 'DFD' in bundles and not target.exists():
            result=bootstrap_all(data,seed,role,rows,y,domain,{m:b['p'] for m,b in bundles.items()},output)
            print(dataset,role,state,seed,'bootstrap',result['analysis_wall_seconds'],flush=True)
    table(output/'macro.csv',macrorows);table(output/'six_cells.csv',cellrows)
    references=[m for m in all_metrics[next(iter(all_metrics))] if m!='DFD']
    comparisons={}
    if 'DFD' in all_metrics[next(iter(all_metrics))]:
        for m in references:
            comparisons[m]={field:paired_summary([all_metrics[s]['DFD'][field]-all_metrics[s][m][field] for s in all_metrics]) for field in ['macro_logloss','macro_auc']}
            comparisons[m]['cell_mean_LL_delta']=[float(np.mean([all_metrics[s]['DFD']['cells'][c]['logloss']-all_metrics[s][m]['cells'][c]['logloss'] for s in all_metrics])) for c in range(6)]
            comparisons[m]['cell_mean_AUC_delta']=[float(np.mean([all_metrics[s]['DFD']['cells'][c]['auc']-all_metrics[s][m]['cells'][c]['auc'] for s in all_metrics])) for c in range(6)]
    write(output/'paired_seed_statistics.json',dict(dataset=dataset,role=role,state=state,comparisons=comparisons,audits=audits))
    print(dataset,role,state,'analysis complete',flush=True)
    return all_metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',default='Q_explore');p.add_argument('--state',default='selected');p.add_argument('--no-bootstrap',action='store_true');a=p.parse_args()
    files={s:{m:ROOT/'runs'/'movielens'/m/f'seed{s}'/f'{a.role}_{a.state}.npz' for m in METHODS} for s in [5101,5102]}
    analyze_files('movielens',a.role,a.state,files,ROOT/'analysis'/'p2'/'movielens'/a.role/a.state,not a.no_bootstrap)
