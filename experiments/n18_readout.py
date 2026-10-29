"""Primary-analysis arithmetic for frozen N18 measurements; no training."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from n18_common import OUT,dump

def cell_operators():
    d,t=np.indices((3,2));A=np.c_[np.ones(6),(d.ravel()==1),(d.ravel()==2),(t.ravel()==1)].astype(float)
    Q=np.linalg.qr(A)[0];P=Q@Q.T
    return P,np.eye(6)-P

def gradient_readout(gram):
    P,I=cell_operators();A=np.c_[np.eye(6),np.zeros((6,18))];B=np.c_[np.zeros((6,6)),np.eye(6),np.zeros((6,12))]
    C=np.c_[np.zeros((6,12)),np.eye(6),np.zeros((6,6))];F=(A+B)/2
    J=np.c_[np.zeros((6,18)),np.eye(6)]
    norm=lambda X:float(np.trace(X@gram@X.T))
    ip=lambda X,Y:float(np.trace(X@gram@Y.T))
    denom=np.sqrt(max(norm(I@A)*norm(I@B),1e-300))
    reference=J@gram@F.T;cn=np.sqrt(np.diag(J@gram@J.T));fn=np.sqrt(np.diag(F@gram@F.T))
    refcos=reference/np.maximum(cn[:,None]*fn[None,:],1e-30)
    out=[];prediction_cosines={};prediction_inner={}
    for name,H in [('full_6cell',F),('additive_marginals',P@F)]:
        estimates=H@gram@F.T;hn=np.sqrt(np.diag(H@gram@H.T))
        cos=estimates/np.maximum(hn[:,None]*fn[None,:],1e-30)
        prediction_cosines[name]=cos;prediction_inner[name]=estimates
        valid=abs(refcos)>=.01;chosen=valid&(abs(cos)>=.01)
        out.append(dict(method=name,relative_gradient_squared_error=norm(H-C)/max(norm(C),1e-30),
            relative_interaction_squared_error=norm(I@(H-C))/max(norm(I@C),1e-30),
            decision_count=int(chosen.sum()),reference_decisions=int(valid.sum()),
            sign_accuracy=float(np.mean(np.sign(estimates[chosen])==np.sign(reference[chosen]))) if chosen.any() else None,
            interaction_energy_share=norm(I@C)/max(norm(C),1e-30),
            interaction_A_B_cosine=ip(I@A,I@B)/denom,
            relative_stop_gradient_jacobian_difference=norm(J-C)/max(norm(J),1e-30),
            calls_for_training_observation=12,relative_cost_to_full=1.))
    common=(abs(refcos)>=.01)&(abs(prediction_cosines['full_6cell'])>=.01)&(abs(prediction_cosines['additive_marginals'])>=.01)
    for row in out:
        row['matched_decision_count']=int(common.sum())
        row['matched_sign_accuracy']=float(np.mean(np.sign(prediction_inner[row['method']][common])==np.sign(reference[common]))) if common.any() else None
    return out,dict(reference_cosines=refcos.tolist(),full_prediction_cosines=(F@gram@F.T/np.maximum(fn[:,None]*fn[None,:],1e-30)).tolist(),
                    marginal_prediction_inner_products=(P@F@gram@F.T).tolist(),source_gram=(F@gram@F.T).tolist(),target_source_gram=reference.tolist())

def proposal_readout(samples,moments):
    p=np.asarray(moments['p']);groups={}
    for block in ['design','heldout']:
        groups[block]=[]
        for d in range(3):
            rows=[r for r in samples[block] if r['domain']==d]
            groups[block].append(dict(norm2=np.mean([r['norm2'] for r in rows]),
                projections=np.array([r['projections'] for r in rows]),
                logit=np.mean([r['logit_norm2'] for r in rows]),tasks=np.mean([r['task_residual2'] for r in rows],axis=0)))
    def optimal(values):
        q=p*np.sqrt(np.maximum(values,1e-30));q/=q.sum();return .98*q+.02*p
    dd=groups['design'];qnorm=optimal([v['norm2'] for v in dd])
    qdirection=optimal([np.mean(v['projections']**2) for v in dd]);qlogit=optimal([v['logit'] for v in dd])
    proposals=np.stack([p,qnorm,optimal([v['tasks'][0] for v in dd]),optimal([v['tasks'][1] for v in dd])])
    values=np.array([v['norm2'] for v in dd]);base_objective=float(np.dot(p,values))
    result=minimize(lambda weights:float(np.sum(p*p*values/(weights@proposals))/base_objective),np.ones(4)/4,
                    method='SLSQP',bounds=[(0,1)]*4,constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],
                    options=dict(ftol=1e-12,maxiter=100))
    if not result.success:raise RuntimeError(result.message)
    qmix=result.x@proposals
    rows=[]
    for name,q in [('natural',p),('norm_IS',qnorm),('EMAIS_logit_proxy',qlogit),('direction_IS',qdirection),('MIS_mixture_trace',qmix)]:
        for block in ['design','heldout']:
            vals=groups[block];mean=moments['blocks'][block]
            trace=float(np.sum(p*p/q*np.array([v['norm2'] for v in vals]))-mean['population_mean_norm2'])
            projected=np.sum((p*p/q)[:,None]*np.array([np.mean(v['projections']**2,axis=0) for v in vals]),axis=0)-np.square(mean['population_mean_projections'])
            rows.append(dict(method=name,block=block,q0=float(q[0]),q1=float(q[1]),q2=float(q[2]),covariance_trace=trace,
                 mean_direction_variance=float(projected.mean()),**{f'cell{k}_direction_variance':float(v) for k,v in enumerate(projected)},
                 batch40960_trace=trace/40960,mean_preservation_error=float(abs(q*(p/q)-p).max())))
    return rows,dict(mixture_weights=result.x.tolist(),proposals=proposals.tolist(),fit_success=bool(result.success),
                    scope='domain proposal; finite heldout population; proxy is not full EMAIS; mixture is not full online OMIS')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT/'readouts');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows4=[];rows6=[];rows3=[]
    for seed in range(2022,2026):
        source=OUT/'measurements_functional'/f'seed{seed}';grams=np.load(source/'P4_gram.npz')
        for block in ['all','shared']:
            rows,details=gradient_readout(grams[f'{block}_gram'])
            cost=json.loads((source/'P4_direct_cost.json').read_text())
            for row in rows:
                reduced=row['method']=='additive_marginals'
                row.update(seed=seed,block=block,calls_for_training_observation=8 if reduced else 12,
                    relative_cost_to_full=cost['time_ratio'] if reduced else 1.,
                    measured_training_seconds=cost['marginal_seconds'] if reduced else cost['full_seconds'])
            rows4.extend(rows);dump(a.out/f'P4_seed{seed}_{block}_decisions.json',details)
        rows,details=proposal_readout(json.loads((source/'P6_samples.json').read_text()),json.loads((source/'P6_moments.json').read_text()))
        for row in rows:row.update(seed=seed)
        rows6.extend(rows);dump(a.out/f'P6_seed{seed}_proposals.json',details)
        metrics=pd.read_csv(OUT/'P3_training'/f'seed{seed}'/'metrics.csv')
        pdom=np.array(json.loads((source/'P6_moments.json').read_text())['p'])
        for (split,model),group in metrics.groupby(['split','model']):
            teacher=metrics[(metrics.split==split)&(metrics.model=='teacher')].sort_values(['domain','task'])
            g=group.sort_values(['domain','task']);da=g.auc.to_numpy()-teacher.auc.to_numpy();dl=g.logloss.to_numpy()-teacher.logloss.to_numpy()
            rows3.append(dict(seed=seed,split=split,model=model,macro_auc=float(g.auc.mean()),macro_logloss=float(g.logloss.mean()),
              train_p_weighted_cell_auc=float(np.dot(np.repeat(pdom/2,2),g.auc)),train_p_weighted_cell_logloss=float(np.dot(np.repeat(pdom/2,2),g.logloss)),
              macro_auc_delta=float(da.mean()),macro_logloss_delta=float(dl.mean()),worst_auc_delta=float(da.min()),worst_logloss_delta=float(dl.max()),
              all_cell_screen=bool(np.all(da>=-.003)&np.all(dl<=.003))))
    pd.DataFrame(rows4).to_csv(a.out/'P4_interaction.csv',index=False)
    pd.DataFrame(rows6).to_csv(a.out/'P6_variance.csv',index=False)
    pd.DataFrame(rows3).to_csv(a.out/'P3_aggregate.csv',index=False)
    dump(a.out/'receipt.json',dict(status='COMPLETE',seeds=list(range(2022,2026)),test_access=False))

if __name__=='__main__':main()
