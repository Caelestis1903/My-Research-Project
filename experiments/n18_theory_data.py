"""Raw finite-support oracles and controlled data checks (P1/P2/P3/P5)."""
import argparse
import csv
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit
from n18_common import OUT, bundle, dump

def feasibility(A, target):
    return linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=target,
                   bounds=(0,None), method='highs')

def moments():
    # z,s in {-1,+1}; exact diagonal support cannot balance E[zs]=0.
    full=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float)
    records=[]
    for domain, points in [('diagonal',full[[0,3]]),('full',full)]:
        z,s=points.T; A=np.array([np.ones(len(z)),z,s,z*s])
        fit=feasibility(A,[1,0,0,0])
        # Preserve E[z]=E[s]=0 and minimize |E[zs]| with an epigraph LP.
        eq=np.c_[A[:3],np.zeros(3)]
        ub=np.array([np.r_[z*s,-1],np.r_[-z*s,-1]])
        relaxed=linprog(np.r_[np.zeros(len(z)),1],A_ub=ub,b_ub=[0,0],
                        A_eq=eq,b_eq=[1,0,0],bounds=(0,None),method='highs')
        records.append(dict(domain=domain,points=points.tolist(),zero_balance_feasible=bool(fit.success),
                            min_absolute_moment=float(relaxed.fun),mass=relaxed.x[:-1].tolist()))
    # Pooled XOR cancellation: per-domain dependency remains perfect.
    pooled=np.vstack([full[[0,3]],full[[1,2]]])
    joint=feasibility(np.array([[1,1],[-1,1],[-1,1.]]),[1,-.5,.5])
    separate=[feasibility(np.array([[1,1],[-1,1.]]),[1,m]).success for m in [-.5,.5]]
    # Repeated same support point gives arbitrarily large ESS but zero label diversity.
    return dict(support_oracles=records,pooled_cross_moment=float(np.prod(pooled,axis=1).mean()),
                domain_cross_moments=[1,-1],joint_shared_weights_feasible=bool(joint.success),
                separate_task_weights_feasible=[bool(x) for x in separate],
                task_target_moments=[-.5,.5],ess_counterexample=dict(replicas=10000,ess=10000,label_classes=1))

def negative_cycle(W,delta):
    D,T=W.shape; edges=[]
    for d in range(D):
        for t in range(T):
            edges += [(D+t,d,float(W[d,t]+delta[d,t])),(d,D+t,float(-W[d,t]+delta[d,t]))]
    dist=np.zeros(D+T); predecessor=np.full(D+T,-1); last=-1
    for _ in range(D+T):
        last=-1
        for i,j,cost in edges:
            if dist[j] > dist[i]+cost+1e-12:
                dist[j]=dist[i]+cost;predecessor[j]=i;last=j
    if last<0:return dict(feasible=True,potentials=dist.tolist(),cycle=[])
    node=last
    for _ in range(D+T):node=int(predecessor[node])
    cycle=[node];other=int(predecessor[node])
    while other!=node:cycle.append(other);other=int(predecessor[other])
    cycle.append(node)
    # Following predecessor traverses the cycle backwards.
    directed=list(reversed(cycle))
    cost=sum(next(v for i,j,v in edges if i==a and j==b) for a,b in zip(directed,directed[1:]))
    return dict(feasible=False,cycle=directed,cycle_cost=float(cost))

def budget_certificates():
    examples=[]
    W=np.array([[1.,-1.],[-1.,1.]])
    for scale in [.1,1.,1.2]:
        s=np.array([[1.,4.],[.5,2.]])
        # Nonuniform tolerated coefficient radii, sum controls feasibility.
        delta=scale*np.array([[.5,1.],[1.5,1.]])
        D,T=W.shape; A=np.zeros((D*T,D+T))
        for d in range(D):
            for t in range(T):A[d*T+t,d]=1;A[d*T+t,D+t]=-1
        lp=linprog(np.zeros(D+T),A_ub=np.r_[A,-A],b_ub=np.r_[(W+delta).ravel(),(-W+delta).ravel()],bounds=[(None,None)]*(D+T),method='highs')
        examples.append(dict(W=W.tolist(),s=s.tolist(),epsilon=(s*delta**2).tolist(),delta=delta.tolist(),
                    contrast=float(abs(W[0,0]-W[0,1]-W[1,0]+W[1,1])),radius_sum=float(delta.sum()),
                    formula_feasible=bool(4<=delta.sum()+1e-12),lp_feasible=bool(lp.success),certificate=negative_cycle(W,delta)))
    W3=np.array([[1,-1],[0,0],[-1,1.]])
    examples.append(dict(W=W3.tolist(),delta=(np.ones_like(W3)*.2).tolist(),certificate=negative_cycle(W3,np.ones_like(W3)*.2)))
    # A+e_c and A+(I-P_A)e_c span exactly the same prediction class.
    d,t=np.indices((3,2)); A=np.c_[np.ones(6),(d.ravel()==1),(d.ravel()==2),(t.ravel()==1)].astype(float)
    Q=np.linalg.qr(A)[0]; P=Q@Q.T; span=[]
    for cell in range(6):
        e=np.eye(6)[:,cell];r=(np.eye(6)-P)@e
        q1=np.linalg.qr(np.c_[Q,e])[0];q2=np.linalg.qr(np.c_[Q,r])[0]
        span.append(dict(cell=cell,projector_max_error=float(abs(q1@q1.T-q2@q2.T).max())))
    return dict(interval_examples=examples,ordinary_head_vs_centered_repair=span,
                dimensions=dict(additive=4,additive_one_residual=5,flat=6,interaction=2))

def bce_risk(p,eta):return -eta*np.log(p)-(1-eta)*np.log1p(-p)

def identification(b):
    raw=[];audit=[]
    for seed in range(2022,2026):
        rng=np.random.default_rng(seed)
        for d in range(3):
            y=b.train_y[b.train_x[:,-1]==d];n=len(y);y0,y1=y.T.astype(int)
            assert np.all(y1<=y0)
            for mechanism in ['MAR','MNAR']:
                prob=np.full(n,.4) if mechanism=='MAR' else np.where(y1==1,.7,.2)
                obs=rng.random(n)<prob
                observed_pos=int(y1[obs].sum());missing=~obs
                lo=observed_pos/n;upper=(observed_pos+missing.sum())/n
                logic_upper=(observed_pos+np.sum(missing & (y0==1)))/n
                # Pair f=.7 and ref=.3; risk difference linear in unknown eta.
                vals=[float(bce_risk(.7,eta)-bce_risk(.3,eta)) for eta in (lo,upper,logic_upper)]
                lp=linprog([-1.],A_ub=[[1.]],b_ub=[logic_upper],bounds=[(lo,upper)],method='highs')
                row=dict(seed=seed,domain=d,mechanism=mechanism,n=n,observed=int(obs.sum()),
                     observed_positive_rate=float(y1[obs].mean()),oracle_positive_rate=float(y1.mean()),
                     naive_lower=lo,naive_upper=upper,logic_lower=lo,logic_upper=logic_upper,
                     ordinary_same_information_lp_upper=float(-lp.fun),inferred_zeros=int(np.sum(missing&(y0==0))),
                     risk_difference_lower=min(vals[:2]),risk_difference_upper=max(vals[:2]),
                     logic_risk_difference_lower=min(vals[0],vals[2]),logic_risk_difference_upper=max(vals[0],vals[2]),
                     oracle_risk_difference=float(bce_risk(.7,y1.mean())-bce_risk(.3,y1.mean())))
                raw.append(row)
                # Audit the missing population only; observed totals are already known.
                pool=np.flatnonzero(missing);k=min(128,len(pool));random_ix=rng.choice(pool,k,False)
                est=(observed_pos+len(pool)*y1[random_ix].mean())/n
                se=len(pool)/n*np.sqrt((1-k/len(pool))*y1[random_ix].var(ddof=1)/k)
                # zero stratum known by logic, buy k labels in unknown positive Y0 stratum.
                pospool=np.flatnonzero(missing&(y0==1));k2=min(k,len(pospool));target_ix=rng.choice(pospool,k2,False)
                est2=(observed_pos+len(pospool)*y1[target_ix].mean())/n
                se2=len(pospool)/n*np.sqrt((1-k2/len(pospool))*y1[target_ix].var(ddof=1)/k2)
                audit.append(dict(seed=seed,domain=d,mechanism=mechanism,label_budget=k,
                     random_estimate=float(est),random_se=float(se),logic_stratified_estimate=float(est2),
                     logic_stratified_se=float(se2),oracle=float(y1.mean()),actual_stratified_labels=k2))
    return raw,audit

def symmetry():
    rng=np.random.default_rng(2022);x=rng.normal(size=(128,4));specific=rng.normal(size=(4,3));shared=rng.normal(size=(4,3))
    gate=expit(x@rng.normal(size=4));bsp=rng.normal(size=3);bsh=rng.normal(size=3)
    def pred(g):
        W=(g[:,None,None]*specific)*((1-g)[:,None,None]*shared)
        bias=(g[:,None]*bsp)*((1-g)[:,None]*bsh)
        return expit(np.einsum('ni,nij->nj',x,W)+bias)
    a=pred(gate);b=pred(1-gate)
    labels=(rng.random(a.shape)<.5).astype(float)
    # Gradient w.r.t identical function logits, same action coordinates.
    wa=.1+.5*gate;wb=.1+.5*(1-gate)
    ga=wa[:,None]*(a-labels);gb=wb[:,None]*(b-labels)
    return dict(prediction_max_error=float(abs(a-b).max()),weight_mean_absolute_change=float(abs(wa-wb).mean()),
                function_logit_gradient_change_norm=float(np.linalg.norm(ga-gb)),
                kind='published D3 product-fusion numerical construction; not official trained D3',
                downstream_prediction_regret='NOT_MEASURED')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=str,default=str(OUT/'theory_data'));a=p.parse_args()
    from pathlib import Path
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    dump(out/'P1_support.json',moments());dump(out/'P3_certificates.json',budget_certificates());dump(out/'P5_symmetry.json',symmetry())
    schema=[]
    for dataset in ['kuairand','movielens']:
        b=bundle(dataset)
        for split in ['train','val','test']:
            x=getattr(b,split+'_x');y=getattr(b,split+'_y')
            # Test labels deliberately excluded from this study.
            if split=='test':continue
            for d in range(3):
                yy=y[x[:,-1]==d]
                schema.append(dict(dataset=dataset,split=split,domain=d,n=len(yy),task0_rate=float(yy[:,0].mean()),
                    task1_rate=float(yy[:,1].mean()),nested_violations=int(np.sum(yy[:,1]>yy[:,0]))))
        dump(out/f'{dataset}_schema.json',dict(features=list(b.features.columns),metadata=b.preprocessing_metadata,
                    has_missing_indicator=False,has_random_audit=False,has_exposure_time=False,
                    raw_rating_timestamp_present=dataset=='movielens',
                    upload_dt_interpretation='video upload, not exposure timestamp' if dataset=='kuairand' else None))
        if dataset=='movielens':
            raw_table=pd.read_csv(b.preprocessing_metadata['raw_file'],usecols=['rating','timestamp'])
            train_table=raw_table.iloc[b.train_row_ids].copy()
            assert np.array_equal((train_table.rating.to_numpy()>3),b.train_y[:,0])
            cuts=np.quantile(train_table.timestamp,[.25,.5,.75])
            environments=np.searchsorted(cuts,train_table.timestamp.to_numpy(),side='right')
            envrows=[]
            for d in range(3):
                for e in range(4):
                    yy=b.train_y[(b.train_x[:,-1]==d)&(environments==e)]
                    envrows.append(dict(domain=d,time_quartile=e,n=len(yy),task0_rate=float(yy[:,0].mean()),task1_rate=float(yy[:,1].mean())))
            pd.DataFrame(envrows).to_csv(out/'P1_ML_rating_time_environments.csv',index=False)
            dump(out/'P1_ML_time_definition.json',dict(cuts=cuts.tolist(),source='raw rating timestamp joined by preserved row IDs',
                nuisance_or_stable_mechanism='NOT_IDENTIFIED',exposure_policy='NOT_OBSERVED'))
            raw,audit=identification(b);pd.DataFrame(raw).to_csv(out/'P2_mask_bounds.csv',index=False)
            pd.DataFrame(audit).to_csv(out/'P2_equal_budget_audit.csv',index=False)
    pd.DataFrame(schema).to_csv(out/'data_cells.csv',index=False)
    dump(out/'P2_worlds.json',dict(observed_2x2=[[.5,.5],[.5,None]],missing_probabilities=[.1,.9],
        shared_probability=.8,reference_probability=.2,
        risk_differences=[float(bce_risk(.8,e)-bce_risk(.2,e)) for e in [.1,.9]],
        connected_observation_graph=True,point_identified=False,
        warning='worlds illustrate lack of structural constraints; separate nested-label bounds use ML'))
    dump(out/'receipt.json',dict(status='COMPLETE',wall_seconds=time.time()-start,seeds=list(range(2022,2026)),test_access=False))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
