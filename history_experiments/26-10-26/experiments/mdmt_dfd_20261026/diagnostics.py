"""Registered arithmetic and bounded diagnostic records; no adaptive search."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .state import atomic_json,sha_file,digest
from .plans import DATASETS,DEV_SEEDS
from .metrics import losses,evaluate_probabilities

def decide_p1(root):
    root=Path(root);result={}
    for d in DATASETS:
        values=[];missing=[];invalid=[]
        for s in DEV_SEEDS:
            p=root/'prefix'/d/f'seed{s}'/'P1_METRICS.json'
            complete=p.parent/'P1_OOF_COMPLETE.json';oof=p.parent/'p1_oof.npz'
            if not p.exists() or not complete.exists() or not oof.exists():missing.append(s);continue
            try:
                meta=json.loads(p.read_text());done=json.loads(complete.read_text())
                from .demand import source_identity
                valid=(done['npz_sha256']==sha_file(oof) and done['checkpoint_hash']==meta['checkpoint_hash'] and done['identity']['source_hash']==source_identity())
                with np.load(oof) as stored:valid=valid and str(stored['checkpoint_hash'])==meta['checkpoint_hash']
                if not valid:invalid.append(dict(seed=s,reason='OOF complete/NPZ/source/checkpoint identity mismatch'));continue
            except (OSError,ValueError,KeyError,json.JSONDecodeError) as e:
                invalid.append(dict(seed=s,reason='Incomplete or invalid OOF export: '+str(e)));continue
            m=meta['metrics'];c=m['CONST'];a=m['DEMAND'];l=m['LOGIT']
            cg=np.asarray([x['logloss'] for x in c['cells']])-np.asarray([x['logloss'] for x in a['cells']])
            values.append(dict(seed=s,gain_const=c['macro_logloss']-a['macro_logloss'],gain_logit=l['macro_logloss']-a['macro_logloss'],nonnegative_cell_count=int((cg>=0).sum()),cell_gains=cg.tolist(),macro_metrics={k:{'auc':v['macro_auc'],'logloss':v['macro_logloss']} for k,v in m.items()}))
        if missing or invalid:result[d]=dict(status='INCOMPLETE',missing_seeds=missing,invalid_exports=invalid,rows=values);continue
        checks=dict(both_const_gain_positive=all(x['gain_const']>0 for x in values),mean_const_gain_at_least_0001=np.mean([x['gain_const'] for x in values])>=.0001,both_at_least_four_cells=all(x['nonnegative_cell_count']>=4 for x in values),mean_not_worse_than_logit=np.mean([x['gain_logit'] for x in values])>=0,metrics_and_implementation_complete=True)
        passed=all(checks.values());result[d]=dict(status='PASS' if passed else 'FAIL',checks=checks,rows=values,mean_const_gain=float(np.mean([x['gain_const'] for x in values])),mean_logit_gain=float(np.mean([x['gain_logit'] for x in values])),next_stage='P2' if passed else 'DFD_P2_NO_START',p3='CONDITIONAL_P2_PASS' if passed else 'NO_START',p4='CONDITIONAL_P3_PASS' if passed else 'NO_START')
    report=dict(stage='p1',score_role='P_form_OOF',datasets=result);atomic_json(root/'DECISION_P1.json',report);return report


def audit_rhd(root):
    root=Path(root);base=Path('/home_new/xzf_mdmt_next_v3_2026-10-25_15-56-04');rows=[]
    # This version requires completed epochs12/20; stage-start diagnostic files cannot replace them.
    import torch
    for seed in (3101,3102):
        directory=base/f'P1_kuairand_R0_BASE_{seed}'
        for epoch in (12,20):
            candidates=[]
            if directory.exists():
                for p in directory.glob('*.pt'):
                    if p.name in ('epoch12.pt','epoch20.pt','ll_best.pt','auc_best.pt','primary_best.pt','model_best.pt'):
                        x=torch.load(p,map_location='cpu',weights_only=False)
                        if x.get('epoch')==epoch:candidates.append(str(p))
            rows.append(dict(dataset='kuairand',seed=seed,completed_epoch=epoch,status='MISSING_ARTIFACT' if not candidates else 'CANDIDATE_REQUIRES_COMPLETE_STATE_AUDIT',candidate_paths=candidates,reason='No verified complete model/Adam/RNG/original main row-plan at exact completed anchor; no substitute/retraining permitted',updates_run=0))
    result=dict(status='MISSING_ARTIFACT' if all(r['status']=='MISSING_ARTIFACT' for r in rows) else 'NEEDS_REVIEW',rows=rows,maximum_updates=960,actual_updates=0,old_root_exists=base.exists());atomic_json(root/'RHD_DIAGNOSTIC_AUDIT.json',result);return result


def oof_support_tables(data,run_dir):
    """Frozen OOF demand groups; full tables retained, no label-selected cuts."""
    run_dir=Path(run_dir);a=np.load(run_dir/'p1_oof.npz');rows=a['rows'];out=[]
    for t in range(3):
        pos=np.flatnonzero(a['domain']==t);ids=rows[pos];support=data.supports(ids)
        for q in range(2):
            own=a['candidates'][pos,q,0]
            for k,s in enumerate(s for s in range(3) if s!=t):
                mu=a['mu'][pos,q,k];cuts=np.quantile(mu,[.25,.5,.75]);group=np.searchsorted(cuts,mu,side='right')
                actual=losses(a['y'][pos,q],own)-losses(a['y'][pos,q],a['candidates'][pos,q,k+1])
                for g in range(4):
                    mask=group==g;yy=a['y'][pos,q][mask]
                    out.append(dict(target=t,source=s,task=q,group=g,cuts=cuts,n=int(mask.sum()),positive=int(yy.sum()),negative=int(len(yy)-yy.sum()),mean_mu=float(mu[mask].mean()) if mask.any() else None,mean_actual_risk_gain=float(actual[mask].mean()) if mask.any() else None,raw_users=None if data.user_ids is None else int(len(np.unique(data.user_ids[ids[mask]]))),support_counts=None if support is None else {f'{u}_{i}':int(((support[mask,0]==u)&(support[mask,1]==i)).sum()) for u in (False,True) for i in (False,True)}))
    atomic_json(run_dir/'P1_DEMAND_GROUPS.json',out);return out
