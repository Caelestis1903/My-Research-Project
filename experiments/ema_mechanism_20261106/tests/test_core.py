"""Necessary arithmetic, transaction, and observer-purity checks."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.ema_mechanism_20261106.shadows import FixedShadowBank,PulseWindow
from experiments.ema_mechanism_20261106.readouts import group_hybrid,read_predictions,cell_metrics
from experiments.ema_mechanism_20261106.state import PredictionWriter,commit_epoch
from experiments.ema_mechanism_20261106.protocol import resolve_study,ROOT,DEFAULT_CACHE
from experiments.ema_mechanism_20261106.worker import create_session,train_batch,export_session,restore_session
from experiments.overnight_ab_20261103.data_bridge import cached_bundle
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.objectives import losses
from experiments.overnight_ab_20261103.state import state_digest


def test_shadow_alias_and_pulse_coordinate_equivalence():
    theta=torch.arange(12,dtype=torch.float32);rho=.9;bank=FixedShadowBank(theta,{'H2':rho},{'F':'H2'})
    pulse=PulseWindow(rho);pulse.begin(bank.get('F'),5,0)
    per_group=theta.clone();mask=torch.arange(12)%2==0
    for step in range(1,5):
        theta=theta+.25
        per_group[mask]=per_group[mask]*rho**2+theta[mask]*(1-rho**2)
        per_group[~mask]=per_group[~mask]*rho+theta[~mask]*(1-rho)
        bank.after_adam(theta,step);pulse.after_adam(theta,step)
    ended=pulse.finish(4);hybrid=torch.empty_like(theta)
    group_hybrid(bank.get('F'),ended['fast'],(~mask).long(),0,hybrid)
    assert torch.equal(bank.get('F'),bank.get('H2'))
    assert torch.allclose(hybrid,per_group,atol=2e-6)
    before=ended['fast'].clone();pulse.after_adam(theta+1,5);assert torch.equal(before,pulse.fast)
    with pytest.raises(ValueError): bank.after_adam(theta,6)


def test_original_probability_metrics_ties_floor_and_missing():
    value=cell_metrics(np.array([0,1]),np.array([.5,.5],np.float32))
    assert value['auc']==.5
    saturated=cell_metrics(np.array([1,0]),np.array([0,1],np.float32))
    assert saturated['logloss']==100
    assert cell_metrics(np.ones(2),np.ones(2,np.float32))['auc'] is None


def test_prediction_whitelist_caps_and_idempotence(tmp_path):
    writer=PredictionWriter(tmp_path);arrays=dict(p=np.full((3,4096,2),.5,np.float32))
    prediction=dict(readout_id='F',physical_readout_id='H2',panel_id='V_mech',state_hash='s',arrays=arrays)
    meta=dict(panel_hash='panel',pair_hash='pair')
    first,new=writer.write('job',1,prediction,meta);second,new2=writer.write('job',1,prediction,meta)
    assert new is not None and new2 is None and first==second
    with pytest.raises(ValueError): writer.write('job',2,dict(prediction,arrays=dict(parameters=np.zeros(3))),meta)
    with pytest.raises(MemoryError): PredictionWriter(tmp_path,cap=1).write('job',2,prediction,meta)


def test_ram_commit_idempotence_and_conflict_rollback(tmp_path):
    snapshot=dict(job={'job_id':'j'},epoch=1,global_step=1,complete=False,init_hash='i',layout_hash='l',group_hash='g',cost={},commit_index=1,snapshots={},online=torch.ones(2))
    row=dict(job=snapshot['job'],commit_index=0)
    record=dict(job_id='j',epoch=1,commit_index=1,model_hash={'online':'same'},cost={},predictions=[])
    writer=PredictionWriter(tmp_path)
    assert commit_epoch(tmp_path,'j',row,snapshot,record,[],writer,{})['status']=='COMMITTED'
    assert record['cost']['peak_guardian_ram_bytes']>0
    assert row['summary']['cost']['peak_guardian_ram_bytes']>0
    assert commit_epoch(tmp_path,'j',row,snapshot,record,[],writer,{})['status']=='ALREADY_COMMITTED'
    previous=row.copy()
    with pytest.raises(ValueError): commit_epoch(tmp_path,'j',row,dict(snapshot,commit_index=2),dict(record,commit_index=2,epoch=1,model_hash={'online':'conflict'}),[],writer,{})
    assert row['record_hash']==previous['record_hash'] and row['commit_index']==1


@pytest.fixture(scope='module')
def real_source():
    study=resolve_study(ROOT/'research_plans/26-11-6/experiment_manifest.json')
    bundle=cached_bundle(DEFAULT_CACHE,'movielens',include_test=False)
    ids=np.concatenate([np.flatnonzero(bundle.train_x[:,-1]==d)[:4] for d in range(3)])
    return study,bundle,torch.as_tensor(np.array(bundle.train_x[ids])).T,torch.as_tensor(np.array(bundle.train_y[ids]))


@pytest.mark.parametrize('objective',['clean','gaussian_d0'])
def test_source_loss_gradient_adam_f_and_ram_restore(real_source,objective):
    study,bundle,x,y=real_source
    job=next(j for j in study.jobs if j['dataset']=='movielens' and j['objective']==objective and j['seed']==2022)
    session=create_session(job,bundle,'cpu',{});reference=copy.deepcopy(session['model'])
    optimizer=torch.optim.Adam(reference.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=1e-4,foreach=False,fused=False,amsgrad=False)
    initial=session['bank'].get('F').clone()
    loss,_=losses(reference,x,y,0,dict(num_views=int(objective=='gaussian_d0'),view_family='gaussian',pairing='iid'),'movielens',2022,1,mix_lambda=.5,kappa=.05)
    loss.backward();optimizer.step();value=train_batch(session,x,y)
    assert value['loss']==float(loss.detach())
    for a,b in zip(reference.parameters(),session['model'].parameters()):
        assert torch.equal(a,b) and torch.equal(a.grad,b.grad)
    expected=initial.mul(job['config']['rho']).add(session['layout'].flatten(reference.parameters()),alpha=1-job['config']['rho'])
    assert torch.equal(expected,session['bank'].get('F'))
    snapshot=export_session(session);restored=restore_session(job,bundle,'cpu',{},snapshot)
    assert state_digest(restored['model'].state_dict())==state_digest(session['model'].state_dict())
    assert state_digest(restored['optimizer'].state_dict())==state_digest(session['optimizer'].state_dict())
    train_batch(session,x,y);train_batch(restored,x,y)
    assert state_digest(restored['model'].state_dict())==state_digest(session['model'].state_dict())


def test_full_functional_derivative_and_hook_purity(real_source):
    study,bundle,x,y=real_source;session=create_session(study.jobs[0],bundle,'cpu',{})
    scratch=session['scratch'];layout=session['layout'];flat=layout.flatten(scratch.parameters()).requires_grad_(True)
    params={name:flat[left:right].view(shape) for name,shape,(left,right) in zip(layout.names,layout.shapes,layout.offsets)}
    full=read_predictions(scratch,x,params=params,full_functional_graph=True)
    proxy=read_predictions(scratch,x,params=params,full_functional_graph=False)
    assert torch.equal(full.p,proxy.p) and torch.equal(full.z,proxy.z)
    g=torch.autograd.grad(full.z[:,0].mean(),flat)[0]
    direction=torch.zeros_like(flat);left,right=layout.offsets[layout.names.index('shared_weight')];direction[left:right]=g[left:right]
    direction/=direction.norm();epsilon=.001
    def forward(vector):
        pp={name:vector[l:r].view(shape) for name,shape,(l,r) in zip(layout.names,layout.shapes,layout.offsets)}
        return read_predictions(scratch,x,params=pp,full_functional_graph=True).z[:,0].mean()
    numeric=(forward(flat.detach()+epsilon*direction)-forward(flat.detach()-epsilon*direction))/(2*epsilon)
    assert torch.allclose(numeric,g.dot(direction),atol=2e-3,rtol=.02)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in scratch.modules())
    assert all(p.grad is None for p in session['model'].parameters())


def test_complete_observer_schema_and_no_online_feedback(real_source):
    from experiments.ema_mechanism_20261106.worker import collect_observations
    study,bundle,_,_=real_source
    panels={}
    for name in ('P_A','P_B'):
        indices=[];pairs=[]
        for d in range(3):
            pool=np.flatnonzero(bundle.train_x[:,-1]==d)
            for t in range(2):
                ids=np.concatenate([pool[bundle.train_y[pool,t]==label][:2] for label in (1,0)])
                indices.append(ids);pairs.append([[0,2],[1,3]])
        indices=np.array(indices)
        natural=np.stack([np.flatnonzero(bundle.train_x[:,-1]==d)[:8] for d in range(3)])
        panels[name]=dict(panel_id=name,rank_x=np.asarray(bundle.train_x[indices]),rank_y=np.stack([bundle.train_y[indices[c],c%2] for c in range(6)]),rank_pairs=np.array(pairs),natural_x=np.asarray(bundle.train_x[natural]),natural_y=np.asarray(bundle.train_y[natural]))
    val_ids=np.concatenate([np.flatnonzero(bundle.val_x[:,-1]==d)[:100] for d in range(3)])
    small=copy.copy(bundle);small.val_x=np.asarray(bundle.val_x[val_ids]);small.val_y=np.asarray(bundle.val_y[val_ids])
    indices=np.arange(300).reshape(3,100)[:,:8]
    panels['V_mech']=dict(panel_id='V_mech',natural_indices=indices,natural_x=small.val_x[indices],natural_y=small.val_y[indices],validation_row_side=np.tile(np.arange(8)%2,(3,1)))
    session=create_session(study.jobs[0],small,'cpu',panels)
    initial=state_digest(session['model'].state_dict());session['pulse'].begin(session['bank'].get('F'),1,0)
    record,predictions,cost=collect_observations(session,1)
    assert initial==state_digest(session['model'].state_dict())
    assert len(record['pulse_metrics'])==72 and sum(row['functional_support'] for row in record['pulse_metrics'])==24
    assert len(record['pulse_readouts'])==27*6*5
    assert len(predictions)==4+4+26*3
    assert all(p.grad is None for p in session['model'].parameters())
    for block,fields in study.manifest['predictive']['feature_blocks'].items():
        if block!='design': assert set(fields)<=set(record['pulse_metrics'][0])


def test_group_reductions_float64_cancellation_and_exact_repeat():
    from experiments.ema_mechanism_20261106.observers import reduce_groups
    values=torch.tensor([1e10,1.,-1e10]*12,dtype=torch.float32)
    groups=torch.arange(12).repeat_interleave(3)
    expected=torch.ones(12,dtype=torch.float64)
    first=reduce_groups(values,groups)
    assert first.dtype==torch.float64 and torch.equal(first,expected)
    assert torch.equal(first,reduce_groups(values,groups))
