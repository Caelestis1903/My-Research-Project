import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.aug_history_20261105.registry import ParameterLayout
from experiments.ema_causal_20261110.interventions import source_direction_states,group_masks
from experiments.ema_mechanism_20261106.worker import create_session,train_batch
from experiments.ema_mechanism_20261106.shadows import FixedShadowBank
from experiments.overnight_ab_20261103.state import state_digest,read_json,cpu_copy
from experiments.overnight_ab_20261103.data_bridge import cached_bundle
from experiments.mdmt_history_information_20261103 import protocol as source
from experiments.ema_next_20261111.optimizer_direction import capture_embed_inputs,apply_matched_direction,set_embed_beta1_zero
from experiments.ema_next_20261111.history_readout import LaggedFStore,split_history,shared_only
from experiments.ema_next_20261111.conditional_readout import build_unique_pairs,choose_risk,fit_from_PA,apply_lagged_policy
from experiments.ema_next_20261111.evaluation import update_earliest_best,evaluate_full_population,materialize_packet
from experiments.ema_next_20261111.distillation import student_loss
from experiments.ema_next_20261111 import trajectory


def test_matched_direction_native_noop_oracle_and_feedback():
    torch.manual_seed(2022)
    model=torch.nn.Linear(3,2);layout=ParameterLayout.from_model(model);mask=torch.ones(layout.size,dtype=torch.bool)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.001,foreach=False,fused=False)
    x=torch.randn(4,3);y=torch.randn(4,2)
    for _ in range(2):
        optimizer.zero_grad();(model(x)-y).square().mean().backward();optimizer.step()
    optimizer.zero_grad();(model(x)-y).square().mean().backward()
    before=layout.flatten(model.parameters()).clone();g=layout.flatten(p.grad for p in model.parameters())
    pre=capture_embed_inputs(model,optimizer,layout,mask);optimizer.step();native=layout.flatten(model.parameters()).clone()
    before_hash=state_digest((model.state_dict(),optimizer.state_dict()))
    apply_matched_direction(pre,optimizer,0.)
    assert before_hash==state_digest((model.state_dict(),optimizer.state_dict()))
    v=layout.flatten(optimizer.state[p]['exp_avg_sq']/(1-.999**int(optimizer.state[p]['step'])) for p in model.parameters())
    m=layout.flatten(optimizer.state[p]['exp_avg']/(1-.9**int(optimizer.state[p]['step'])) for p in model.parameters())
    expected=source_direction_states(native,before,g,m,v,mask,lr=.01,eps=1e-8,weight_decay=.001)['current_preconditioned'][0]
    diagnostics=apply_matched_direction(pre,optimizer,1.)
    assert torch.allclose(layout.flatten(model.parameters()),expected,atol=6e-8,rtol=2e-7)
    assert diagnostics['relative_error']<=1e-4
    changed=copy.deepcopy(model);layout.copy_to(model,native)
    model.zero_grad();changed.zero_grad();model(x).square().mean().backward();changed(x).square().mean().backward()
    assert not torch.equal(model.weight.grad,changed.weight.grad)


def test_sparse_f_clock_barycenter_and_scope():
    for steps,windows,peak in ((20,[20,40,80],5),(49,[69,138,277],11)):
        store=LaggedFStore([e*steps for e in range(1,51)],windows,.995)
        assert store.schedule()['peak_vectors']==peak
        objects={}
        def put(value):
            key=str(len(objects));objects[key]=value;return key
        for step in range(steps*50+1):
            store.capture(step,torch.full((3,),float(step)),put)
            if step and step%steps==0:
                values=store.consume(step,objects)
                for h,v in values.items():assert torch.all(v==max(0,step-h))
        assert not store.records
    f=torch.tensor([.1,.2,.3,.4]);old=torch.tensor([.4,.3,.2,.1]);mask=torch.tensor([True,False,True,False])
    r,o,audit=split_history(f,old,.995,277,mask)
    assert torch.equal(r[~mask],f[~mask]) and torch.equal(o[~mask],f[~mask])
    assert audit['fp64_barycenter_max_abs']<1e-15 and audit['fp32_barycenter_max_abs']<1e-7
    online=f+1;shared=shared_only(f,online,mask)
    assert torch.equal(shared[mask],f[mask]) and torch.equal(shared[~mask],online[~mask])


def _panel():
    return dict(panel_id='P_A',panel_hash='p',rank_row_ids=np.tile(np.arange(4),(6,1)),rank_y=np.tile([1,1,0,0],(6,1)))


def test_c_unique_pairs_lag_and_documented_tie():
    pairs=build_unique_pairs(_panel(),'movielens',2022)
    assert all(len(t)==4 and len(set(map(tuple,t)))==4 for t in pairs['tables'])
    assert pairs['pair_hash']==build_unique_pairs(_panel(),'movielens',2022)['pair_hash']
    assert choose_risk(np.array([0.,.5]),np.array([1.,1.]),.25)==.5
    z=np.tile([1.,2.,0.,-1.],(6,1));pending=fit_from_PA(z,z,pairs,.37,1)
    assert pending['global_alpha']==.37
    assert apply_lagged_policy(None,.2,1)['weights']==[.2]*6
    used=apply_lagged_policy(pending,.5,2,1.)
    assert used['weights']==[.37]*6 and used['a_apply']==.5
    with pytest.raises(ValueError):apply_lagged_policy(pending,.5,3)


def test_earliest_selection_includes_zero_and_test_freeze():
    selected={};update_earliest_best(selected,'F','p0',0,{'macro_auc':.8},{})
    assert not update_earliest_best(selected,'F','p1',1,{'macro_auc':.8},{})
    assert selected['F']['epoch']==0
    packet={'kind':'single','state_id':'p'}
    with pytest.raises(ValueError):evaluate_full_population(packet,None,split='test')


def test_kd_teacher_equals_labels_and_row_alignment():
    torch.manual_seed(2022);model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.Sigmoid())
    x=torch.randn(4,3);y=torch.tensor([[0.,1.],[1.,0.],[1.,1.],[0.,0.]])
    ids=np.array([2,0,3,1]);targets=np.empty((4,2),np.float32);targets[ids]=y.numpy()
    session=dict(model=model,job={'family':'B_DISTILL','distill_weight':.5},soft_targets=targets)
    loss,_=student_loss(session,x,y,ids)
    exact=torch.nn.functional.binary_cross_entropy(model(x),y,reduction='none').mean(1).mean()
    assert torch.equal(loss,exact)
    loss.backward();assert all(p.grad is not None for p in model.parameters())


def test_source_step_alpha0_observer_free_equivalence():
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    cfg=read_json(root/'research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json')['configs']['movielens']
    cfg.update(rho_by_physical_shadow={'F':cfg['rho']},aliases={})
    bundle=cached_bundle('/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache','movielens',include_test=False)
    job=dict(config=cfg,dataset='movielens',seed=2022,objective='clean',family='A_MATCH',direction_alpha=0.,start_epoch=0)
    first=create_session(job,bundle,'cpu',{});second=create_session(job,bundle,'cpu',{})
    second.update(masks=group_masks(second['groups']),history=LaggedFStore([],[],cfg['rho']),objects={},new_objects={},training_updates=0)
    x=torch.tensor(np.array(bundle.train_x[:32])).T;y=torch.tensor(np.array(bundle.train_y[:32]))
    for _ in range(2):
        train_batch(first,x,y,observe=False);trajectory.train_one_step(second,x,y)
    for key in ('model','optimizer'):assert state_digest(first[key].state_dict())==state_digest(second[key].state_dict())
    assert state_digest(first['bank'].state_dict())==state_digest(second['bank'].state_dict())
    assert torch.equal(first['loader'].get_state(),second['loader'].get_state())
    assert not bool(second['masks']['EMBED'][second['masks']['N']].any())
    before=state_digest([second['optimizer'].state[p] for p in second['model'].parameters()])
    set_embed_beta1_zero(second['optimizer'],second['layout'],second['masks']['EMBED'],second['model'])
    assert before==state_digest([second['optimizer'].state[p] for p in second['model'].parameters()])
    assert {g['betas'][0] for g in second['optimizer'].param_groups}=={0.,.9}


def test_c_observations_use_shared_scope_only(monkeypatch):
    model=torch.nn.Linear(2,2);layout=ParameterLayout.from_model(model);mask=torch.tensor([True,True,True,False,False,False])
    natural=np.zeros((3,2,3),np.int64);natural[:,:,2]=np.arange(3)[:,None]
    panels={}
    for name in ('P_A','P_B'):
        p=_panel();p.update(panel_id=name,natural_x=natural,natural_y=np.tile([0.,1.],(3,2,1)),rank_pairs=np.tile([[0,2],[1,3]],(6,1,1)))
        panels[name]=p
    bundle=SimpleNamespace(val_x=np.column_stack((np.zeros((6,2)),np.repeat(np.arange(3),2))),val_y=np.tile([0.,1.],(6,1)))
    def pred(packet,*args,**kwargs):
        result={'p':np.full((6,2),.5,np.float32),'z':np.zeros((6,2),np.float32)}
        return ({'macro_auc':.5},result)
    def panel_pred(packet,*args,**kwargs):
        return {name:dict(natural_p=np.full((3,2,2),.5,np.float32),natural_z=np.zeros((3,2,2),np.float32),rank_p=np.tile([.8,.7,.3,.2],(6,1)).astype(np.float32),rank_z=np.tile([1.,.5,-.5,-1.],(6,1)).astype(np.float32)) for name in panels}
    monkeypatch.setattr(trajectory,'evaluate_full_population',pred);monkeypatch.setattr(trajectory,'predict_panels',panel_pred)
    monkeypatch.setattr(source,'metrics',lambda *a,**k:{'macro_auc':.5})
    menu=[dict(id='O',family='O'),dict(id='F',family='F')]
    menu += [dict(id='B_'+scope,family='B_SPLIT',H=2,scope=scope,mode='logit') for scope in ('S','all')]
    for mode in ('PRED','PARAM'):
        menu.append(dict(id='C_GLOBAL_'+mode,family='C_GLOBAL_'+mode,H=2,scope='S'))
        menu.append(dict(id='C_CELL_'+mode+'_s1',family='C_CELL_'+mode,H=2,scope='S',shrinkage=1.))
    s=dict(model=model,layout=layout,bank=FixedShadowBank(layout.flatten(model.parameters()),{'F':.9},{}),
           job=dict(config={'rho':.9,'eval_batch_size':20},dataset='movielens',seed=2022,objective='clean',job_id='test'),
           masks={'S':mask},objects={},new_objects={},selected={},menu=menu,global_step=2,bundle=bundle,panels=panels,context={},pending_c=None,
           pairs={name:build_unique_pairs(p,'movielens',2022) for name,p in panels.items()},history=LaggedFStore([2],[2],.9))
    s['history'].capture(0,layout.flatten(model.parameters()),lambda value:trajectory.put_object(s,value))
    calls=[];original=trajectory.fit_from_PA
    def fit(*a,**k):calls.append(1);return original(*a,**k)
    monkeypatch.setattr(trajectory,'fit_from_PA',fit)
    records,_=trajectory.observe_epoch(s,1)
    assert len(calls)==1
    for key,row in records.items():
        if key.startswith('C_') and key!='C_GLOBAL_PARAM':
            packet=materialize_packet(s['objects'][row['packet_id']],s['objects']);assert packet['scope']=='S'


def test_restore_has_independent_mutable_state_and_exact_next_source_step(monkeypatch):
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    cfg=read_json(root/'research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json')['configs']['movielens']
    cfg.update(rho_by_physical_shadow={'F':cfg['rho']},aliases={})
    bundle=cached_bundle('/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache','movielens',include_test=False)
    job=dict(config=cfg,dataset='movielens',seed=2022,objective='clean',family='PARENT',end_epoch=50,job_id='test')
    def make(job,prepared,device):
        s=create_session(job,bundle,device,{})
        s.update(masks=group_masks(s['groups']),history=LaggedFStore([],[],cfg['rho']),objects={},new_objects={},training_updates=0,
                 selected={},prefix={},anchors={},pending_c=None,raw_active=False)
        return s
    first=make(job,{},'cpu');x=torch.tensor(np.array(bundle.train_x[:32])).T;y=torch.tensor(np.array(bundle.train_y[:32]))
    trajectory.train_one_step(first,x,y);first['rng']=source.rng_state('cpu')
    snapshot=trajectory.export_session(first);monkeypatch.setattr(trajectory,'create_parent',make)
    restored=trajectory.restore_session(job,{},'cpu',snapshot,{})
    assert next(first['model'].parameters()).data_ptr()!=next(restored['model'].parameters()).data_ptr()
    trajectory.train_one_step(first,x,y);trajectory.train_one_step(restored,x,y)
    for key in ('model','optimizer'):assert state_digest(first[key].state_dict())==state_digest(restored[key].state_dict())
    assert state_digest(first['bank'].state_dict())==state_digest(restored['bank'].state_dict())


def test_cell_parameter_uses_actual_task_columns(monkeypatch):
    from experiments.ema_next_20261111 import evaluation
    weights=[0.,.2,.4,.6,.8,1.]
    packet=dict(kind='split',scope='all',recent=torch.zeros(1),old=torch.ones(1),buffers={},mode='parameter',a=.5,weights=weights)
    calls=[]
    def forward(vector,buffers,x,context):
        alpha=float(vector[0]);calls.append((alpha,int(x[-1,0])))
        values=torch.tensor([alpha+10,alpha+20]).expand(x.shape[1],2).clone()
        return values,values
    monkeypatch.setattr(evaluation,'_vector_prediction',forward)
    x=torch.tensor([[1.,1.,1.,1.],[2.,0.,1.,2.]])
    p,_=evaluation.predict_batch(packet,x,{'masks':{'S':torch.ones(1,dtype=torch.bool)}})
    for row,domain in enumerate([2,0,1,2]):
        assert torch.allclose(p[row],torch.tensor([10+weights[domain*2],20+weights[domain*2+1]]))
    assert len(calls)==6


def test_split_packet_and_stream_mix_have_identical_fp32_scores(monkeypatch):
    from experiments.ema_next_20261111 import evaluation
    packet=dict(kind='split',scope='all',recent=torch.zeros(1),old=torch.ones(1),buffers={},mode='logit',a=.37,weights=[.1,.2,.3,.4,.5,.6])
    x=torch.tensor([[1.,1.,1.],[0.,1.,2.]])
    recent=torch.tensor([[.1,-3.],[-.2,.2],[7.,-7.]],dtype=torch.float32)
    old=torch.tensor([[.3,2.],[.6,-.1],[-7.,7.]],dtype=torch.float32)
    def forward(vector,buffers,x,context):
        z=recent if float(vector[0])==0 else old
        return torch.sigmoid(z),z
    monkeypatch.setattr(evaluation,'_vector_prediction',forward)
    for mode in ('logit','probability'):
        packet['mode']=mode
        p,z=evaluation.predict_batch(packet,x,{'masks':{'S':torch.ones(1,dtype=torch.bool)}})
        expected=evaluation.combine_predictions(dict(p=torch.sigmoid(recent).numpy(),z=recent.numpy()),dict(p=torch.sigmoid(old).numpy(),z=old.numpy()),packet['weights'],x[-1].numpy(),mode)
        assert np.array_equal(p.numpy(),expected['p']) and np.array_equal(z.numpy(),expected['z'])
