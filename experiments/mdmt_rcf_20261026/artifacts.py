"""Research data exports, identity receipts, and paired metric recomputation.

This module never serializes a model, optimizer, RNG, or training snapshot.
"""
from __future__ import annotations
import csv
import dataclasses
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import expit

OUTER = ('test_exposed', 'Q_legacy_confirm_exposed')

def _plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name:_plain(getattr(value,f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k):_plain(v) for k,v in value.items() if not str(k).startswith('_') and not callable(v)}
    if isinstance(value,(list,tuple)): return [_plain(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    if isinstance(value,float) and not np.isfinite(value): return None
    if value is None or isinstance(value,(str,int,float,bool)): return value
    # In-memory predictors and optimizer objects have no disk representation.
    return {'in_memory_type':type(value).__name__,'serialized':False}

def _json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(_plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def _sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def _npz(path,**arrays):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    clean={k:np.asarray(v) for k,v in arrays.items() if v is not None}
    if any(v.dtype.hasobject for v in clean.values()): raise ValueError('Object array is not research data')
    np.savez_compressed(p,**clean)
    return {'path':str(p.resolve()),'sha256':_sha(p),'bytes':p.stat().st_size,'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in clean.items()}}

def _metrics(y,p,domain):
    from .protocol import evaluate
    return evaluate(y,p,domain)

def _write_csv(path,rows):
    rows=list(rows);p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(_plain(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple,np.ndarray)) else v for k,v in row.items()})

def metrics_rows(dataset,seed,method,role,y,p,domain,stage='selected'):
    m=_metrics(y,p,domain)
    common=dict(dataset=dataset,seed=int(seed),method=method,role=role,training_stage=stage)
    rows=[dict(common,level='macro',auc=m['macro_auc'],logloss=m['macro_logloss'],natural_logloss=m['natural_logloss'])]
    rows.extend(dict(common,level='cell',**c) for c in m['cells'])
    rows.extend(dict(common,level='task',**r) for r in m.get('task',[]))
    rows.extend(dict(common,level='domain',**r) for r in m.get('domain',[]))
    return rows

def _weighted_auc_plan(y,p):
    order=np.argsort(p,kind='mergesort');sp=np.asarray(p)[order]
    group=np.r_[0,np.cumsum(sp[1:]!=sp[:-1])].astype(np.int64)
    return order,group,np.asarray(y)[order].astype(float),int(group[-1]+1) if len(group) else 0

def _weighted_auc(plan,weights):
    order,group,y,n=plan;w=weights[order]
    pos=np.bincount(group,weights=w*y,minlength=n);neg=np.bincount(group,weights=w*(1-y),minlength=n)
    den=pos.sum()*neg.sum()
    return np.nan if den==0 else float(np.sum(pos*(np.cumsum(neg)-.5*neg))/den)

def paired_bootstrap(y,candidate,reference,domain,user_ids,rng,repetitions=1000):
    """Weighted, tie-correct AUC bootstrap; sorting reused across replicates."""
    y=np.asarray(y);domain=np.asarray(domain);n=len(y)
    clusters=np.arange(n) if user_ids is None else np.asarray(user_ids)
    unique,inverse=np.unique(clusters,return_inverse=True)
    masks=[domain==d for q in range(2) for d in range(3)]
    ys=[y[m,q] for q in range(2) for d,m in enumerate(masks[q*3:q*3+3])]
    plans=[];diff=[]
    for q in range(2):
        for d in range(3):
            m=domain==d
            plans.append((_weighted_auc_plan(y[m,q],candidate[m,q]),_weighted_auc_plan(y[m,q],reference[m,q])))
            pc=np.clip(candidate[m,q].astype(float),1e-7,1-1e-7);pr=np.clip(reference[m,q].astype(float),1e-7,1-1e-7);yy=y[m,q]
            diff.append(-(yy*np.log(pc)+(1-yy)*np.log1p(-pc))+(yy*np.log(pr)+(1-yy)*np.log1p(-pr)))
    out=np.empty((repetitions,2),float)
    for b in range(repetitions):
        weights=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inverse]
        losses=[];aucs=[]
        for m,dd,(cp,rp) in zip(masks,diff,plans):
            w=weights[m];losses.append(np.average(dd,weights=w) if w.sum() else np.nan)
            aucs.append(_weighted_auc(cp,w)-_weighted_auc(rp,w))
        out[b]=np.nanmean(losses),np.nanmean(aucs)
    return dict(cluster='raw_user' if user_ids is not None else 'row_fallback_independence_unverified',repetitions=repetitions,
                delta_ll_percentile95=np.nanquantile(out[:,0],[.025,.975]),delta_auc_percentile95=np.nanquantile(out[:,1],[.025,.975]),
                replicate_deltas=out,invalid_replicates=int((~np.isfinite(out).all(1)).sum()))

def validate_raw_cache(bundle,chunk=4096):
    """Validate all cached H/Z against still-resident native raw-X prediction."""
    import torch
    model=bundle.native
    if isinstance(model,dict): model=model.get('model',model.get('selected'))
    if not hasattr(model,'encode'): raise RuntimeError('Resident native predictor is required for raw-X verification')
    device=next(model.parameters()).device;was=model.training;model.eval();mh=mz=0.;count=0
    with torch.no_grad():
        for role,view in bundle.views.items():
            for start in range(0,len(view.row_id),chunk):
                ids=view.row_id[start:start+chunk]
                packet=model.encode(torch.as_tensor(np.array(bundle.data.x[ids]),device=device))
                h=packet.h.detach().cpu().numpy();z=packet.logits.detach().cpu().numpy()
                mh=max(mh,float(np.max(np.abs(h-view.h[start:start+len(ids)]))))
                mz=max(mz,float(np.max(np.abs(z-view.z[start:start+len(ids)]))))
                count+=len(ids)
    model.train(was)
    if max(mh,mz)>1e-6:raise ValueError(f'raw-X cached native mismatch H={mh} Z={mz}')
    return dict(status='PASS',rows=count,max_abs_h=mh,max_abs_z=mz,tolerance=1e-6,scope='all cached native H/Z; full predictor callback separately recorded')

def _function_arrays(bundle,formed,role):
    import torch
    view=bundle.views[role];values={}
    for fid,obj in formed.functions.items():
        cached=getattr(obj,'outputs',{}).get(role)
        if cached is not None:
            values[fid]=np.asarray(cached);continue
        mod=getattr(obj,'module',obj.get('module') if isinstance(obj,dict) else None)
        scale=getattr(obj,'scale',obj.get('scale',1.) if isinstance(obj,dict) else 1.)
        if mod is None:continue
        device=next(mod.parameters()).device;out=[]
        with torch.no_grad():
            for i in range(0,len(view.h),4096):out.append((mod(torch.as_tensor(np.asarray(view.h[i:i+4096]),device=device))*float(scale)).detach().cpu().numpy().reshape(-1))
        values[fid]=np.concatenate(out) if out else np.empty(0)
    return values

def export_bundle(bundle,results,output_root,config):
    """Export complete data objects only; called after the global method freeze."""
    root=Path(output_root)/'runs'/bundle.dataset/f'seed{bundle.seed}';root.mkdir(parents=True,exist_ok=True)
    predictions=results.get('predictions',results);formed=results.get('formations',{});demand=results.get('demand')
    files=[];metric_rows_all=[];groups=[];feature_files=[]
    if not results.get('outer_authorized',False): raise PermissionError('Global freeze must authorize final export/evaluation')
    for role,view in bundle.views.items():
        y=np.asarray(view.y)
        bundle.ledger.access('export',role,rows=view.row_id,purpose='evaluate' if role in OUTER+('Q_explore','validation') else 'audit')
        arrays=dict(row_id=view.row_id,domain=view.domain,y=y,native_logits=view.z,group=view.group,base_refit=bundle.base_logits[role])
        if role=='P_form':arrays['base_oof']=bundle.base_oof
        if getattr(bundle,'data',None) is not None and bundle.data.user_ids is not None:arrays['raw_user_id']=bundle.data.user_ids[view.row_id]
        supports=bundle.data.supports(view.row_id)
        if supports is not None:arrays['support']=supports
        for s in range(3):
            for q in range(2):
                arrays[f'pilot_s{s}_q{q}']=bundle.pilots.residual(role,s,q)
                arrays[f'pilot_phi_s{s}_q{q}']=bundle.pilots.hidden(role,s,q,standardized=True)
        files.append(_npz(root/'prefix'/f'{role}.npz',**arrays))
        if demand is not None:
            from .demand import outer_requests
            request=outer_requests(bundle,demand,role)
            files.append(_npz(root/'prefix'/f'{role}_frozen_requests.npz',**request))
        for owner,diag in results.get('diagnostics',{}).items():
            for fid,single in diag.get('single_function_predictions',{}).items():
                if role in single.probabilities:
                    files.append(_npz(root/owner/'single_function_predictions'/f'{role}_{fid}.npz',row_id=view.row_id,p=single.probabilities[role],y=y,domain=view.domain))
        for method,pred in predictions.items():
            if not hasattr(pred,'probabilities') or role not in pred.probabilities: continue
            p=np.asarray(pred.probabilities[role]);terminal=np.asarray(pred.terminal_probabilities.get(role,p))
            if p.shape!=y.shape or not np.isfinite(p).all():raise ValueError(f'Invalid export {method} {role}')
            files.append(_npz(root/method/f'{role}.npz',row_id=view.row_id,domain=view.domain,y=y,p=p,p_terminal=terminal,group=view.group))
            metric_rows_all.extend(metrics_rows(bundle.dataset,bundle.seed,method,role,y,p,view.domain))
            metric_rows_all.extend(metrics_rows(bundle.dataset,bundle.seed,method,role,y,terminal,view.domain,'terminal'))
            for q in range(2):
                for d in range(3):
                    for g in range(2):
                        mask=(view.domain==d)&(view.group[:,q]==g)
                        from sklearn.metrics import roc_auc_score
                        pp=np.clip(p[mask,q].astype(float),1e-7,1-1e-7);yy=y[mask,q]
                        groups.append(dict(dataset=bundle.dataset,seed=bundle.seed,method=method,role=role,domain=d,task=q,group=g,n=int(mask.sum()),positive=int(yy.sum()),logloss=float(np.mean(-(yy*np.log(pp)+(1-yy)*np.log1p(-pp)))) if len(yy) else None,auc=float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None))
        for method,f in formed.items():
            from .demand import outer_requests
            request_bundle=getattr(f,'request_bundle',bundle)
            if hasattr(f.requests,'units'):
                request=outer_requests(request_bundle,f.requests,role)
                feature_files.append(_npz(root/method/f'{role}_actual_frozen_requests.npz',**request))
            anchors={}
            for function in f.functions.values():
                provider=getattr(function,'anchor_pilots',None)
                if provider is not None:
                    ss,qq=function.source,function.task
                    anchors[f'anchor_logits_s{ss}_q{qq}']=provider.logits(role,ss,qq)
                    anchors[f'anchor_residual_s{ss}_q{qq}']=provider.residual(role,ss,qq)
            if anchors:feature_files.append(_npz(root/method/f'{role}_actual_anchors.npz',row_id=view.row_id,**anchors))
            vals=_function_arrays(bundle,f,role)
            feature_files.append(_npz(root/method/f'{role}_functions.npz',row_id=view.row_id,**vals))
            for t in range(3):
                for q in range(2):
                    value=f.features.values.get((role,t,q))
                    if value is not None:feature_files.append(_npz(root/method/f'{role}_read_features_t{t}q{q}.npz',row_id=view.row_id[view.domain==t],value=value))
    _write_csv(root/'metrics.csv',metric_rows_all);_write_csv(root/'groups.csv',groups)
    for method,pred in predictions.items():
        if hasattr(pred,'selection'):_json(root/method/'selection.json',dict(selection=pred.selection,identity=pred.identity,diagnostics={k:v for k,v in pred.diagnostics.items() if k in ('selected_coefficients','policies','checks','elapsed_seconds','records','eta','candidate_history','selected_pass','cost','parameter_count')}))
    for method,f in formed.items():
        _json(root/method/'formation.json',dict(identity=f.identity,records=f.records,requests=f.requests,columns=f.features.columns,bounds=f.features.bounds))
    if demand is not None:_json(root/'demand.json',demand)
    _json(root/'normalizers_and_thresholds.json',bundle.normalizers)
    raw=validate_raw_cache(bundle);_json(root/'RAW_X_CACHE_ACCEPTANCE.json',raw)
    # Optional complete predictor callback supplied by owner A; never serialized.
    checks={}
    for method,pred in predictions.items():
        validator=pred.diagnostics.get('_raw_x_validate') if hasattr(pred,'diagnostics') else None
        if validator is not None:checks[method]=validator(bundle)
        else:checks[method]={'status':'NOT_RUN','reason':'complete predictor callback unavailable'}
    _json(root/'RAW_X_FULL_PREDICTOR_ACCEPTANCE.json',checks)
    if checks and all(c.get('status')=='PASS' for c in checks.values()):
        from .baselines import release_raw_validation_cache
        release_raw_validation_cache(bundle)
    _json(root/'identity.json',dict(identity=bundle.identity,storage='RAM_MODELS_ONLY',raw_cache_acceptance=raw))
    for p in root.rglob('*'):
        if p.is_file() and p.suffix not in ('.npz',):files.append(dict(path=str(p.resolve()),sha256=_sha(p),bytes=p.stat().st_size))
    index=dict(dataset=bundle.dataset,seed=bundle.seed,root=str(root.resolve()),files=files+feature_files,prediction_methods=sorted(predictions),formation_methods=sorted(formed),roles=list(bundle.views),raw_x=raw,full_predictors=checks,
               outer_demand={'used_eta_mu':'NOT_APPLICABLE','actual_demand':'demand.json','groups':'prefix role files','functional_profiles':{m:'mechanisms.json' for m in results.get('diagnostics',{})}},status='EXPORTED')
    _json(root/'ARTIFACT_INDEX.json',index)
    return index

def verify_exports(index,config):
    problems=[];checked=0
    for f in index.get('files',[]):
        p=Path(f['path'])
        if not p.exists() or _sha(p)!=f['sha256']:problems.append(str(p));continue
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as a:
                if 'p' in a and ('y' not in a or a['p'].shape!=a['y'].shape or not np.isfinite(a['p']).all()):problems.append(str(p)+' invalid predictions')
                if 'p' in a and 'domain' in a:
                    _metrics(a['y'],a['p'],a['domain']);checked+=1
    if index.get('raw_x',{}).get('status')!='PASS':problems.append('raw-X native acceptance')
    for method,check in index.get('full_predictors',{}).items():
        if check.get('status')!='PASS':problems.append('complete raw-X predictor acceptance missing: '+method)
    if index.get('formation_methods') and not (Path(index['root'])/'demand.json').exists():problems.append('actual demand export missing')
    return dict(status='PASS' if not problems else 'FAIL',prediction_files_recomputed=checked,problems=problems)
