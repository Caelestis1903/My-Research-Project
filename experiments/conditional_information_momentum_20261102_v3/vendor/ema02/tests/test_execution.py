import copy
import json
import time
import numpy as np
import torch
from ..data import window_indices
from ..evaluate import metrics
from ..gates import paired, protection
from ..run import choose_device, initial_dispatch, budget_admission
from ..guardian import ensure_guardian, request
from ..protocol import config, make_optimizer, cpu_copy, state_digest, rng_state, restore_rng
from ..objectives import train_step
from .test_core import fixture


def test_window_and_undefined_cells():
    a,b,c=window_indices(103)
    assert [len(a),len(b),len(c)]==[61,11,10]
    assert np.array_equal(np.concatenate((a,b,c)),np.arange(82))
    y=np.zeros((9,2));p=np.full((9,2),.3);domain=np.repeat(np.arange(3),3)
    m=metrics(y,p,domain)
    assert m['macro_auc'] is None and m['worst_cell_auc'] is None
    assert np.isclose(m['macro_logloss'],-np.log(.7)) and m['d2t1_negative']==3


def test_seed_pair_and_harm_predicates():
    ref={s:dict(macro_auc=.7,macro_logloss=.5,**{f'd{d}t{t}_auc':.7 for d in range(3) for t in range(2)})
         for s in (2022,2024)}
    candidate=copy.deepcopy(ref)
    candidate[2022]['macro_auc']+=.002;candidate[2024]['macro_auc']+=.004
    e=paired(candidate,ref)
    assert e['n']==2 and e['positive']==2 and np.isclose(e['mean'],.003) and e['ci95'][0]<0
    for v in candidate.values():v['d0t0_auc']-=.003
    assert protection(candidate,ref,2)[0]['metric']=='d0t0_auc'


def test_independent_device_gate_and_initial_matrix():
    status=[dict(index=0,memory=2,total=24,power=90,limit=100),
            dict(index=1,memory=10,total=24,power=40,limit=100)]
    assert choose_device(status,[0,1],5*1024**2)==1
    dispatch=initial_dispatch()
    assert len(dispatch['jobs'])==24 and len({j['key'] for j in dispatch['jobs']})==24
    assert {j['seed'] for j in dispatch['jobs']}=={2022,2024}


def test_frozen_budget_reserves_active_next_and_final_work():
    assert not budget_admission(1200,1199,[],1000)['admitted']
    assert not budget_admission(1200,900,[200],50,100)['admitted']
    assert budget_admission(1200,900,[100],50,100)['admitted']
    assert not budget_admission(1200,1201,[],0)['admitted']


def test_guardian_committed_state_restores_next_step(tmp_path):
    ensure_guardian(tmp_path)
    try:
        m,x,y,ids,ctx=fixture();ema=copy.deepcopy(m).requires_grad_(False);c=config('A0','movielens')
        opt=make_optimizer(m,c);loader=torch.Generator().manual_seed(2022+11000)
        train_step(m,opt,ema,x,y,ids,c,ctx)
        snapshot=cpu_copy(dict(model=m.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
            rng=rng_state(),loader=loader.get_state(),epoch=1,physical_worker_seconds=.01,counts={}))
        request(tmp_path,'register',key='test',job=dict(key='test'),pid=999999999)
        ack=request(tmp_path,'commit',key='test',state=snapshot,epoch_file='scalar-only-test.json')
        assert ack==dict(status='RAM_COMMITTED',epoch=1)
        restored=request(tmp_path,'get',key='test')['state']
        ctx['global_step']=2;train_step(m,opt,ema,x,y,ids,c,ctx)
        expected=state_digest(dict(model=m.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict()))
        m2=copy.deepcopy(m);e2=copy.deepcopy(ema);o2=make_optimizer(m2,c)
        m2.load_state_dict(restored['model']);e2.load_state_dict(restored['ema']);o2.load_state_dict(restored['optimizer'])
        restore_rng(restored['rng']);loader.set_state(restored['loader'])
        train_step(m2,o2,e2,x,y,ids,c,ctx)
        assert expected==state_digest(dict(model=m2.state_dict(),ema=e2.state_dict(),optimizer=o2.state_dict()))
    finally:
        request(tmp_path,'shutdown')
