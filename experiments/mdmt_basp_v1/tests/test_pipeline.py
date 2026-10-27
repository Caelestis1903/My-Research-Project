"""Small checks for role safety, natural clocks, numerical metrics and RAM replay."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.mdmt_basp_v1.config import load_config,validate_schema,validate_arm,arms
from experiments.mdmt_basp_v1.identity import capture,restore,directly_equal,plain
from experiments.mdmt_basp_v1.metrics import evaluate,weighted_auc
from experiments.mdmt_basp_v1.roles import Data


def test_schema_and_seed_override():
    cfg=load_config()
    assert cfg['seeds']==dict(development=[2022,2023],locked_reevaluation=[2024,2025,2026])
    assert cfg['budgets']['p5_max_adaptation_trajectories']==24
    changed=copy.deepcopy(cfg);changed['extra']=True
    with pytest.raises(ValueError):
        validate_schema(changed,cfg)
    for arm in arms().values():
        validate_arm(arm)
    invalid=copy.deepcopy(arms()['STATIC_LOCAL'])
    invalid['switches']['CONTRASTIVE']=True
    with pytest.raises(ValueError):
        validate_arm(invalid)


def test_fp64_logit_metrics_and_single_class():
    y=np.array([[0,1],[1,0],[1,1]],float)
    z=np.array([[-1000,1000],[1000,-1000],[1000,1000]],float)
    m=evaluate(y,z,np.array([0,0,1]))
    assert m['natural_logloss']==0
    assert m['cells'][0]['auc']==1
    assert m['cells'][1]['auc'] is None
    assert m['macro_auc_denominator']==2
    assert weighted_auc(np.array([0,1]),np.array([1.,0.]),np.array([0.,1.])) is None
    empty=evaluate(np.empty((0,2)),np.empty((0,2)),np.empty(0,int))
    assert empty['natural_logloss'] is None


def small_data():
    data=Data.__new__(Data)
    data.roles=dict(F=np.array([0,2]),V=np.array([1]),S=np.array([3,4]),E=np.array([5]))
    data._y=np.ones((6,2),np.float32);data.x=np.ones((6,8),np.float32)
    data.used_s=set();data.read_s=set();data.final_authorization=None;data.access_log=None
    data.s_windows=[np.array([3]),np.array([4]),np.empty(0,int)]
    return data


def test_role_purpose_and_one_s_read():
    data=small_data()
    with pytest.raises(PermissionError):
        data.labels('V','gain')
    with pytest.raises(PermissionError):
        data.labels('F','adapt',[1])
    with pytest.raises(PermissionError):
        data.labels('E','final')
    with pytest.raises(PermissionError):
        data.labels('S','transaction',[3],window=0)
    ids=data.consume_s(0)
    assert data.labels('S','transaction',ids,window=0)[0].shape==(1,2)
    with pytest.raises(PermissionError):
        data.consume_s(0)
    with pytest.raises(PermissionError):
        data.labels('S','transaction',ids,window=0)
    for method in ('C_SPLIT','CG_SPLIT'):
        ids=data.consume_s(0,method)
        data.labels('S','transaction',ids,window=0,method=method)
        assert data.s_state(method)==({0},{0})
    data.restore_s('C_SPLIT',[],[])
    assert data.s_state('CG_SPLIT')==({0},{0})
    assert data.s_state('C_SPLIT')==(set(),set())


def test_ram_replay_preserves_adam_and_rng():
    torch.manual_seed(2022)
    model=torch.nn.Linear(3,2)
    factory=lambda m:torch.optim.Adam(m.parameters(),lr=.001)
    optimizer=factory(model)
    x=torch.randn(4,3)
    loss=model(x).square().mean();loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    snapshot=capture(model,optimizer,dict(seed=2022),dict(cursor=4,epoch=1))
    expected=None
    for repeat in range(2):
        resident,opt,state=restore(snapshot,'cpu',factory)
        noise=torch.randn(4,3)
        value=resident(x+noise).square().mean();value.backward();opt.step()
        observed=dict(model=resident.state_dict(),optimizer=opt.state_dict(),loss=value.detach(),state=state)
        if expected is None:
            expected=copy.deepcopy(observed)
        else:
            assert directly_equal(expected,observed)
    with pytest.raises(TypeError):
        plain(snapshot)


def test_gain_calibration_and_fold_unit():
    from experiments.mdmt_basp_v1.export import calibration,fold_ridge_skill
    x=np.arange(20.)[:,None]
    y=(x/20).reshape(20,1,1)
    m,p=fold_ridge_skill(x,y,np.arange(20)%2)
    assert m['skill']>.99
    assert p.shape==y.shape
    assert calibration(y,y)['best_action_regret']==0


def test_continuous_rows_keep_tails_and_replay():
    from experiments.mdmt_basp_v1.trainer import RowStream
    stream=RowStream(np.arange(7),2022,'movielens',40,3)
    one=np.concatenate([stream.next(),stream.next(),stream.next()])
    assert np.array_equal(np.sort(one),np.arange(7))
    saved=stream.state();a=stream.next();stream.load(saved)
    assert np.array_equal(a,stream.next())


def test_structure_window_records_proposal_diagnostic_epoch(monkeypatch):
    from types import SimpleNamespace
    from experiments.mdmt_basp_v1.model import make_model
    from experiments.mdmt_basp_v1.trainer import Trainer
    from experiments.mdmt_basp_v1 import conflict
    cfg=load_config()
    trainer=Trainer.__new__(Trainer)
    trainer.model=make_model(cfg,arms()['STATIC_LOCAL'],seed=2022)
    trainer.cfg=cfg;trainer.cache=None
    trainer.data=SimpleNamespace(folds=np.array([0,1]))
    trainer.state=dict(cooldown={},previous_diagnostic=None)
    trainer.diagnostics=[];trainer.proposals=[]
    diagnostic=dict(status='NO_SPLIT_PROPOSAL',epoch=4,nodes=[])
    monkeypatch.setattr(conflict,'propose',lambda *args,**kwargs:(None,diagnostic))
    trainer.structure_window(4)
    assert trainer.diagnostics==[diagnostic]
    assert trainer.state['previous_diagnostic']==diagnostic
    assert trainer.proposals==[dict(epoch=4,proposal=None,status='NO_SPLIT_PROPOSAL')]


def test_split_snapshot_restores_added_adam_groups():
    from experiments.mdmt_basp_v1.model import make_model
    from experiments.mdmt_basp_v1.permissions import make_optimizer,add_new_parameters
    cfg=load_config();arm=arms()['CLUSTER_RELATION']
    model=make_model(cfg,arm,seed=2022)
    optimizer=make_optimizer(model,cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.grad=torch.ones_like(p)*.01
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    old_steps={n:copy.deepcopy(optimizer.state[p]['step']) for n,p in model.named_parameters() if p in optimizer.state}
    model.add_structure('SPLIT',dict(kind='NEED_SPLIT',domain=0,parent=0,centers=torch.randn(2,2,4)))
    added=add_new_parameters(model,optimizer,cfg)
    assert added
    for group in optimizer.param_groups:
        if group['kind']=='center':
            assert group['lr']==.0003
    snapshot=capture(model,optimizer,dict(seed=2022),dict(cursor=7,epoch=6))
    resident,opt,state=restore(snapshot,'cpu',lambda m:make_optimizer(m,cfg))
    assert len(opt.param_groups)==len(optimizer.param_groups)
    assert directly_equal(opt.state_dict(),snapshot['optimizer'])
    for name,p in resident.named_parameters():
        if name in old_steps:
            assert torch.equal(opt.state[p]['step'],old_steps[name])
        elif name in added:
            assert p not in opt.state


def test_registered_source_coverage_excludes_owner_domain():
    from experiments.mdmt_basp_v1.model import make_model
    from experiments.mdmt_basp_v1.trainer import coverage_weights
    cfg=load_config();model=make_model(cfg,arms()['FIXED_READ'],seed=2022)
    domain=torch.tensor([0,0,1,1,2,2])
    packet=dict(h=torch.randn(6,256),z=torch.randn(6,2),phi=torch.randn(6,3,64))
    proposal=dict(kind='SOURCE_SPLIT',domain=0,parent=0,
                  coverage=dict(target_domain=None,excluded_domain=0,weight=[0.]*258,bias=1.,group=0))
    weights,hard=coverage_weights(model,packet,domain,proposal)
    assert np.all(weights[:2]==0) and np.all(~hard[:2])
    assert np.all(weights[2:]>0) and np.all(hard[2:])
    clustered=make_model(cfg,arms()['CLUSTER_RELATION'],seed=2022)
    need=dict(kind='NEED_SPLIT',domain=0,parent=0,
              centers=torch.stack([torch.zeros(2,4),torch.ones(2,4)]),
              coverage=dict(target_domain=0,excluded_domain=None,weight=[0.]*258,bias=-10.,group=0))
    weight,hard=coverage_weights(clustered,packet,domain,need)
    assert np.all(weight[:2]>0) and np.all(hard[:2])
    assert np.all(weight[2:]==0)


def test_actual_trainer_keep_replays_all_clocks(tmp_path):
    from experiments.mdmt_basp_v1.trainer import Trainer
    from experiments.mdmt_basp_v1.identity import capture_rng
    gen=np.random.default_rng(2022)
    data=Data.__new__(Data);data.dataset='movielens'
    data.roles={role:np.arange(i*18,(i+1)*18) for i,role in enumerate(('T0','F','S','Q','V','E'))}
    total=108;data.domain=np.arange(total)%3
    data.x=np.zeros((total,8),np.float32);data._y=gen.integers(0,2,(total,2)).astype(np.float32)
    data.probe=data.roles['F'].copy();data.folds=np.arange(18)%2
    data.owner_weights=np.ones(3)/3;data.access_log=None;data.used_s=set();data.read_s=set()
    data.final_authorization=None;data.s_windows=[v.copy() for v in np.array_split(data.roles['S'],3)]
    h=gen.normal(size=(total,256)).astype(np.float32)
    z=gen.normal(size=(total,2)).astype(np.float32)
    phi=gen.normal(size=(total,3,64)).astype(np.float32)
    views={role:dict(ids=ids,domain=data.domain[ids],h=h[ids],z=z[ids],phi=phi[ids]) for role,ids in data.roles.items()}
    source=torch.nn.Module();source.register_buffer('teacher_w',torch.randn(3,2,64)*.01)
    source.register_buffer('teacher_b',torch.zeros(3,2))
    prefix=dict(source=source,native=torch.nn.Identity())
    trainer=Trainer(arms()['CLUSTER_RELATION'],data,views,prefix,2022,load_config(),tmp_path/'trainer','cpu')
    trainer.state['epoch']=3;trainer.model.set_epoch(3)
    trainer.refresh(2,freeze_scale=True);trainer.step()
    snapshot=trainer.snapshot()
    expected=trainer.step();parameters=copy.deepcopy(trainer.model.state_dict())
    optimizer=copy.deepcopy(trainer.optimizer.state_dict());rngstate=capture_rng()
    clocks=dict(main=trainer.main.state(),owners=[s.state() for s in trainer.owners],
                probe=trainer.probe_stream.state(),step=trainer.state['step'],shadow=trainer.state['shadow_cursor'])
    branch=trainer.fork(snapshot,'KEEP',None)
    observed=branch.step()
    assert directly_equal(expected,observed)
    assert directly_equal(parameters,branch.model.state_dict())
    assert directly_equal(optimizer,branch.optimizer.state_dict())
    assert directly_equal(rngstate,capture_rng())
    assert directly_equal(clocks,dict(main=branch.main.state(),owners=[s.state() for s in branch.owners],
                                     probe=branch.probe_stream.state(),step=branch.state['step'],shadow=branch.state['shadow_cursor']))
    branch.selected=snapshot
    branch.export_state('Q','selected')
    assert (branch.root/'F_selected_GROUP_REGISTRATION.json').is_file()
    assert (branch.root/'Q_selected_GAIN_TABLES.npz').is_file()
    from experiments.mdmt_basp_v1.report import collect
    result=collect(tmp_path,tmp_path/'summary')
    assert result['status']=='RAW_COLLECTION'
