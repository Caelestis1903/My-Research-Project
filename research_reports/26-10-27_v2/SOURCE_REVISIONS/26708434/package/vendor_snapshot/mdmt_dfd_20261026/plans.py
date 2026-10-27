"""Frozen namespace, row and full run matrix plans."""
from __future__ import annotations
import numpy as np
from .state import array_hash,digest,atomic_json
DATASETS=('movielens','kuairand')
DEV_SEEDS=(5101,5102);FORMAL_SEEDS=(6101,6102,6103,6104)
NAMESPACES=dict(native_init=10,main_plan=20,pilot_init=30,pilot_plan=31,eta_init=40,eta_plan=41,formation_perturb=50,formation_plan=51,shuffle_control=52,geometry_subset=53,geometry_fit=54,read_init=60,read_plan=61,bootstrap=70)
# component0 native/pilot; eta FULL fold0=0/fold1=1/refit=2, LOGIT fold0=10/11,
# DIRECT fold0=20/21; readers DFD/UNIFORM/GEOM/SHUFFLE/LOCAL_CAP paired=0,
# H_DIRECT=10, ETA_DIRECT=11, U_BUDGET uses UNIFORM20 prefix=0, POOL=20.
COMPONENTS=dict(eta_full_fold0=0,eta_full_fold1=1,eta_refit=2,eta_logit_fold0=10,eta_logit_fold1=11,p1_direct_fold0=20,p1_direct_fold1=21,ordinary_reader=0,h_direct=10,eta_direct=11,pooled=20)

def sequence(seed,dataset,namespace,source=0,task=0,component=0,pass_id=0):
    d=DATASETS.index(dataset) if isinstance(dataset,str) else int(dataset)
    n=NAMESPACES[namespace] if isinstance(namespace,str) else int(namespace)
    return np.random.SeedSequence([int(seed),d,n,int(source),int(task),int(component),int(pass_id)])
def rng(seed,dataset,namespace,source=0,task=0,component=0,pass_id=0):return np.random.Generator(np.random.PCG64(sequence(seed,dataset,namespace,source,task,component,pass_id)))
def torch_seed(*a,**kw):return int(sequence(*a,**kw).generate_state(1,dtype=np.uint32)[0])
def batches(ids,seed,dataset,namespace,pass_id,source=0,task=0,component=0,batch_size=40960):
    order=rng(seed,dataset,namespace,source,task,component,pass_id).permutation(np.asarray(ids,dtype=np.int64))
    return [order[i:i+batch_size] for i in range(0,len(order),batch_size)]
def folds(data):
    ids=data.ids('P_form');out=np.empty(len(ids),dtype=np.int8)
    for d in range(3):
        pos=np.flatnonzero(data.domain[ids]==d)
        perm=np.random.Generator(np.random.PCG64(20261026+70000+1009*d)).permutation(np.sort(ids[pos]))
        out[np.searchsorted(ids,perm)]=np.arange(len(perm))%2
    return out

def full_plan(data_hashes=None):
    p2=['NATIVE','LOCAL','H_DIRECT','ETA_DIRECT','UNIFORM','GEOM','SHUFFLE','CONST_UNIFORM','LOCAL_CAP','DFD'];p3=['U_BUDGET','POOL'];rows=[]
    for d in DATASETS:
        for s in DEV_SEEDS:
            rows.append(dict(run_id=f'v4_{d}_s{s}_PREFIX_p1',dataset=d,seed=s,arm='PREFIX',stage='p1',depends=[],trigger='always',native_epochs=24,pilot_passes=12,eta_fold_passes=20))
            for arm in p2:rows.append(dict(run_id=f'v4_{d}_s{s}_{arm}_p2',dataset=d,seed=s,arm=arm,stage='p2',depends=[f'v4_{d}_s{s}_PREFIX_p1'],trigger=f'P1_PASS({d})'))
            for arm in p3:rows.append(dict(run_id=f'v4_{d}_s{s}_{arm}_p3',dataset=d,seed=s,arm=arm,stage='p3',depends=[f'v4_{d}_s{s}_DFD_p2'],trigger=f'P2_PASS({d})'))
        for s in FORMAL_SEEDS:
            for arm in p2+p3+['STEM_DOMAIN_TASK','STEM_DOMAIN_TASK_INFO','AutoIFS_MDMT','AutoIFS_MDMT_INFO']:
                rows.append(dict(run_id=f'v4_{d}_s{s}_{arm}_p4',dataset=d,seed=s,arm=arm,stage='p4',depends=[],trigger=f'P3_PASS({d})',from_scratch=True))
    result=dict(spec='DFD-v4.0',data_hashes=data_hashes or {},namespaces=NAMESPACES,components=COMPONENTS,rng_coordinates=['seed','dataset','namespace','source','task','component','pass'],runs=rows,max_large_trajectories=28,max_formal_logical_predictors=128,native=dict(epochs=24,batch_size=40960,microbatch=40960,physical_batch=40960,lr=.001,weight_decay=1e-5,betas=[.9,.999],eps=1e-8,precision='FP32',eligible=list(range(12,25)),selection='validation_macro_logloss'),passes=dict(pilot=12,eta_fold=20,eta_refit=20,formation=12,reader_form=20,reader_read=10,u_budget_form=60),metrics=dict(ll_clip=1e-7,cell_order='task-major/domain-minor'),promotion=dict(p1_mean_gain=.0001,p2_mean_gain=.0005,p3_mean_gain=.0005,auc_harm=.0005,cell_ll_harm=.003),budget_reserve=1.35)
    result['plan_hash']=digest(result);return result
