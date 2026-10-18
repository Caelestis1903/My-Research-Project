"""N19 populations, controlled mechanisms and frozen evaluation split."""
from pathlib import Path
from dataclasses import replace
import json,os,pickle
from types import SimpleNamespace
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research_plans/2026-10-18/N19'
ART=Path(os.environ.get('N19_ARTIFACT_ROOT','/home_new/xzf_n19_artifacts'))
HISTORY=Path('/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016')
PYTHON='/home/xzf/miniconda3/envs/phase0-3-env/bin/python'

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)

def population(dataset,suite,noise_interaction=0.):
    with (HISTORY/'local_cache'/f'{dataset}_2022.pkl').open('rb') as f:b=pickle.load(f)
    data=SimpleNamespace(features=b.features,metadata=b.preprocessing_metadata,dataset=dataset,suite=suite)
    for split in ['train','val','test']:
        setattr(data,split+'_x',getattr(b,split+'_x'));setattr(data,split+'_y',getattr(b,split+'_y'))
        setattr(data,split+'_ids',getattr(b,split+'_row_ids'))
    rng=np.random.default_rng(191019);selection=[];development=[]
    for d in range(3):
        pool=rng.permutation(np.flatnonzero(data.val_x[:,-1]==d));h=len(pool)//2
        selection.extend(pool[:h]);development.extend(pool[h:])
    data.selection=np.array(selection);data.development=np.array(development)
    rng=np.random.default_rng(202219);main=[];meta=[]
    for d in range(3):
        pool=rng.permutation(np.flatnonzero(data.train_x[:,-1]==d));h=max(1,int(len(pool)*.05))
        meta.extend(pool[:h]);main.extend(pool[h:])
    data.main=np.array(main);data.meta=np.array(meta)
    counts=np.bincount(data.train_x[data.main,-1].astype(int),minlength=3)
    data.p=counts/counts.sum();data.noise_rates=None
    data.audit=np.ones(len(data.train_x),bool);data.observed=data.train_y.copy()
    data.environment=np.zeros(len(data.train_x),int)
    if suite=='stability':
        assert dataset=='movielens'
        column='controlled_Z';old=data.features
        data.features=replace(old,columns=old.columns[:-1]+(column,old.columns[-1]),
            sparse_columns=old.sparse_columns+(column,),vocab_sizes={**old.vocab_sizes,column:3})
        for si,split in enumerate(['train','val','test']):
            rng=np.random.default_rng(191901+si);x=getattr(data,split+'_x');y0=getattr(data,split+'_y')[:,0]
            env=rng.integers(0,2,len(x));rate=np.where(env==0,.95,.75) if split=='train' else np.full(len(x),.1 if split=='val' else .05)
            z=np.where(rng.random(len(x))<rate,y0,1-y0).astype(int)
            y1=(rng.random(len(x))<(.2+.6*z)).astype(np.float32)
            setattr(data,split+'_x',np.c_[x[:,:-1],z+1,x[:,-1]].astype(np.float32))
            setattr(data,split+'_y',np.c_[y0,y1].astype(np.float32))
            if split=='train':data.environment=env
        data.observed=data.train_y.copy()
    if suite=='noise':
        assert dataset=='movielens'
        cross=noise_interaction*np.array([-1.,0.,1.])[:,None]*np.array([-1.,1.])[None,:]
        aa=.35*expit(np.array([-1.,.2,.8])[:,None]+np.array([-.3,.5])[None,:]+cross)
        bb=.35*expit(np.array([.5,-.5,.2])[:,None]+np.array([.4,-.4])[None,:]-cross)
        rng=np.random.default_rng(191902);domains=data.train_x[:,-1].astype(int);y=data.train_y
        flips=rng.random(y.shape)<np.where(y==0,aa[domains],bb[domains])
        observed=np.where(flips,1-y,y).astype(np.float32);audit=rng.random(len(y))<.02
        data.observed=observed;data.audit=audit;data.noise_rates=dict(a=aa.tolist(),b=bb.tolist())
    return data
