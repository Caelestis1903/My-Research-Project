"""Focused mathematical, source-equivalence and transaction acceptance checks."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.mdmt_history_information_20261103 import protocol as history
from experiments.overnight_ab_20261103.data_bridge import cached_bundle
from experiments.overnight_ab_20261103.state import state_digest,write_json,read_json,digest
from experiments.overnight_ab_20261103.selection import consider
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.views import make_model,apply_view
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.objectives import losses as source_losses
from .. import protocol,worker,selection,guardian
from ..adam_counterfactual import snapshot_optimizer,predict_displacement
from ..directed_ema import update
from ..model_adapter import build_model
from ..objectives import loss_terms
from ..registry import ParameterLayout
from ..responses import extract_aug_grads
from ..shadow_bank import ShadowBank
from ..subspace import orthonormal_basis,signed_basis
from ..run import parser
from .. import run

MANIFEST=protocol.ROOT/'research_plans/26-11-5/experiment_manifest.json'
torch.set_num_threads(1)


def job_for(dataset):
    p=protocol.resolve(MANIFEST,dataset)
    return dict(dataset=dataset,seed=2022,job_id='ENGINEERING_ONLY_'+dataset,
        trajectory_id='ENGINEERING_ONLY_'+dataset,config=p.config,candidates=p.candidates)


def small_batch(bundle):
    ids=np.concatenate([np.flatnonzero(bundle.train_x[:,-1]==domain)[:4] for domain in range(3)])
    return torch.as_tensor(np.array(bundle.train_x[ids])).T,torch.as_tensor(np.array(bundle.train_y[ids]))


@pytest.mark.parametrize('dataset',['movielens','kuairand'])
def test_source_online_hash_and_loss(dataset):
    bundle=cached_bundle(protocol.CACHE,dataset,include_test=False)
    job=job_for(dataset); session=worker.create_session(job,bundle,'cpu')
    reference=copy.deepcopy(session['model'])
    reference_ema=[p.detach().clone() for p in reference.parameters()]
    c=job['config']
    opt=torch.optim.Adam(reference.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,
        weight_decay=c['weight_decay'],foreach=False,fused=False,amsgrad=False)
    x,y=small_batch(bundle)
    for step in range(1,3):
        opt.zero_grad(set_to_none=True)
        loss,info=source_losses(reference,x,y,0,dict(num_views=1,view_family='gaussian',pairing='iid'),
            dataset,2022,step,mix_lambda=.5,kappa=c['kappa'])
        loss.backward(); expected_grad=[p.grad.clone() if p.grad is not None else None for p in reference.parameters()]
        opt.step()
        with torch.no_grad():
            for phi,p in zip(reference_ema,reference.parameters()):
                phi.mul_(c['rho']).add_(p,alpha=1-c['rho'])
        result=worker.train_batch(session,x,y)
        assert result['loss']==float(loss.detach())
        assert all(torch.equal(a,b) for a,b in zip(reference.parameters(),session['model'].parameters()))
        assert all(g is None and p.grad is None or g is not None and torch.equal(g,p.grad)
                   for g,p in zip(expected_grad,session['model'].parameters()))
        assert torch.equal(session['bank'].phi['F'],session['layout'].flatten(reference_ema))
    assert session['bank'].updates==2
    assert len(session['bank'].phi)==15
    layout=ParameterLayout.from_model(session['model'])
    assert layout.size==c['parameter_count']
    assert len(layout.names)==len(list(session['model'].parameters()))
    logits=session['model'].forward_logits(x)
    torch.testing.assert_close(logits.sigmoid(),session['model'](x),atol=1e-7,rtol=1e-5)
    terms=loss_terms(session['model'],x,y,dict(c,seed=2022,step=3))
    expected=.5*(terms.metadata['augmented'][0][:,0]-terms.metadata['real'][terms.metadata['mask'],0]).sum()/y.numel()
    torch.testing.assert_close(expected,terms.aug_by_cell[0,0],atol=0,rtol=0)
    torch.testing.assert_close(terms.total,terms.clean+sum(terms.aug_by_cell.values()),atol=1e-7,rtol=1e-5)
    no_target=x[:,x[-1]!=0]; labels=y[x[-1]!=0]
    empty=loss_terms(session['model'],no_target,labels,dict(c,seed=2022,step=3))
    assert not empty.metadata['augmented'] and all(float(v)==0 for v in empty.aug_by_cell.values())
    assert not session['model'].tower[0]._forward_hooks


@pytest.mark.parametrize('weight_decay',[1e-4,1e-5])
def test_counterfactual_exact_and_pure(weight_decay):
    model=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([.2,-.3])),
        torch.nn.Parameter(torch.ones(2,2)),torch.nn.Parameter(torch.zeros(3))])
    opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=weight_decay,foreach=False,fused=False)
    for step in range(7):
        for index,p in enumerate(model):
            p.grad=None if (index==2 or index==1 and step%2==0) else torch.full_like(p,.1*(step-3))
        opt.step()
    grads=[torch.tensor([0.,-.2]),torch.zeros(2,2),None]
    for p,g in zip(model,grads): p.grad=g
    before=state_digest(dict(model=model.state_dict(),optimizer=opt.state_dict()))
    rng=torch.get_rng_state().clone()
    snapshot=snapshot_optimizer(model,opt)
    active=[g is not None for g in grads]
    predicted=predict_displacement(snapshot['theta'],grads,snapshot,active)
    assert state_digest(dict(model=model.state_dict(),optimizer=opt.state_dict()))==before
    assert torch.equal(rng,torch.get_rng_state())
    old=[p.detach().clone() for p in model]; opt.step()
    actual=torch.cat([(p-v).detach().reshape(-1) for p,v in zip(model,old)])
    assert torch.equal(actual,predicted)
    assert torch.equal(model[2],old[2])
    assert not torch.equal(model[1],old[1])


def test_first_step_none_zero_and_gradient_algebra():
    model=torch.nn.Linear(3,2).double()
    x=torch.arange(15,dtype=torch.float64).reshape(5,3)/15
    y=torch.zeros(5,2,dtype=torch.float64)
    clean=torch.nn.functional.binary_cross_entropy(model(x).sigmoid(),y,reduction='none')
    aug=torch.nn.functional.binary_cross_entropy(model(x+.01).sigmoid(),y,reduction='none')
    terms=[.5*(aug[:,t]-clean[:,t]).sum()/10 for t in range(2)]
    params=tuple(model.parameters())
    g=torch.autograd.grad(clean.mean()+sum(terms),params,retain_graph=True)
    g0=torch.autograd.grad(clean.mean(),params,retain_graph=True)
    ga=[torch.autograd.grad(t,params,retain_graph=True) for t in terms]
    for i in range(len(params)):
        torch.testing.assert_close(g[i],g0[i]+ga[0][i]+ga[1][i],atol=1e-12,rtol=1e-12)
    opt=torch.optim.Adam(params,lr=.01,weight_decay=.0001,foreach=False,fused=False)
    snapshot=snapshot_optimizer(model,opt)
    grads=[torch.zeros_like(params[0]),None]
    predicted=predict_displacement(snapshot['theta'],grads,snapshot,[True,False])
    for p,grad in zip(params,grads): p.grad=grad
    old=[p.clone() for p in params]; opt.step()
    assert torch.equal(predicted,torch.cat([(p-a).detach().reshape(-1) for p,a in zip(params,old)]))


def test_subspace_signs_and_ema():
    columns=torch.tensor([[1.,0.],[0.,2.],[0.,0.],[0.,0.]],dtype=torch.float64)
    q,_=orthonormal_basis(columns)
    other,_=orthonormal_basis(columns.flip(1)*-1)
    torch.testing.assert_close(q@q.T,other@other.T,atol=1e-10,rtol=1e-10)
    for columns,rank in ((torch.zeros(4,2),0),(torch.ones(4,2),1),
                         (torch.stack((torch.ones(4),-torch.ones(4)),1),1)):
        basis,metadata=orthonormal_basis(columns)
        assert basis.shape[1]==rank==metadata['rank']
    with pytest.raises(ValueError): orthonormal_basis(torch.full((3,2),float('nan')))
    q=torch.tensor([[1.],[1.],[0.],[0.]])/2**.5
    signed=signed_basis(q,torch.tensor([1.,-1.,1.,-1.]))
    assert torch.equal(q.abs(),signed.abs()) and not torch.allclose(q@q.T,signed@signed.T)
    old=torch.zeros(4); theta=torch.ones(4); phi=old.clone()
    update(phi,theta,q,.9,.5)
    expected=.1*theta+(.9-.9**2)*(q@(q.T@(theta-old)))
    torch.testing.assert_close(phi,expected,atol=1e-7,rtol=1e-5)
    for h,basis in ((1,q),(.5,q[:,:0])):
        phi=old.clone(); ref=old.clone().mul_(.9).add_(theta,alpha=.1)
        update(phi,theta,basis,.9,h); assert torch.equal(phi,ref)
    phi=theta.clone(); update(phi,theta,q,.9,2.)
    torch.testing.assert_close(phi,theta,atol=1e-7,rtol=1e-5)


def test_bank_clock_rng_and_resume():
    p=protocol.resolve(MANIFEST,'movielens')
    before=torch.get_rng_state().clone()
    bank=ShadowBank(torch.zeros(4),p.candidates,.9,'movielens',2022)
    assert torch.equal(before,torch.get_rng_state())
    q=torch.tensor([[1.],[0.],[0.],[0.]])
    for step in range(1,32):
        response=dict(D=q,M=q) if bank.refresh_due(step) else None
        bank.after_step(dict(step=step,theta=torch.ones(4)*step,response=response))
        if step==12:
            clone=ShadowBank(torch.zeros(4),p.candidates,.9,'movielens',2022)
            clone.load_state_dict(history.cpu_copy(bank.state_dict()),'cpu')
        if step>12: clone.after_step(dict(step=step,theta=torch.ones(4)*step,response=response))
    assert bank.last_refresh=={10:31,20:21}
    assert history.equal(bank.state_dict(),clone.state_dict())
    bank.after_step(dict(step=32,theta=torch.ones(4),response=None))
    with pytest.raises(ValueError): bank.after_step(dict(step=32,theta=torch.ones(4),response=None))


def test_gaussian_keeps_routing_dense_labels_and_detaches_rms():
    embedding=torch.tensor([[2.,3.,4.,5.]],requires_grad=True)
    view=dict(family='gaussian',noise=torch.ones(1,1,2))
    result=apply_view(embedding,[(0,2)],view,.1)
    assert torch.equal(result[:,2:],embedding[:,2:])
    gradient=torch.autograd.grad(result.sum(),embedding)[0]
    assert torch.equal(gradient,torch.ones_like(gradient))


def test_selection_five_seed_ties_and_test_gate(tmp_path):
    values={cell+'_auc':.7 for cell in ('d0t0','d0t1','d1t0','d1t1','d2t0','d2t1')}
    tracker=dict(final_stage='train',best_score=-float('inf'),selected=None)
    assert consider(tracker,'train',2,values,{'weight':torch.ones(1)})
    assert not consider(tracker,'train',3,values,{'weight':torch.zeros(1)})
    assert tracker['selected']['epoch']==2
    row=dict(seed_ids=protocol.SEEDS,complete=True,score=.7,measured_cost_seconds=1.,
        half_life_ratio=1.,candidate_id='F',cell_protected=True)
    assert selection.rank_family([row])['candidate_id']=='F'
    with pytest.raises(selection.Incomplete): selection.rank_family([dict(row,seed_ids=[2022])])
    with pytest.raises(selection.Incomplete): selection.require_frozen(tmp_path,tmp_path/'selection_frozen.json')
    commands=['preflight','validate','train','select','decide','report']
    for command in commands:
        assert parser().parse_args([command,'--output',str(tmp_path)]).command==command


def test_ram_export_restore_and_idempotent_scalar_commit(tmp_path):
    bundle=cached_bundle(protocol.CACHE,'movielens',include_test=False)
    job=job_for('movielens'); session=worker.create_session(job,bundle,'cpu')
    x,y=small_batch(bundle)
    worker.train_batch(session,x,y)
    snap=worker.export_session(session)
    restored=worker.restore_session(job,bundle,'cpu',snap)
    worker.train_batch(session,x,y); worker.train_batch(restored,x,y)
    assert state_digest(session['model'].state_dict())==state_digest(restored['model'].state_dict())
    assert history.equal(session['bank'].state_dict(),restored['bank'].state_dict())
    assert history.equal(session['optimizer'].state_dict(),restored['optimizer'].state_dict())
    row=dict(job_id='toy',epoch=1,commit_index=1,global_step=1,order_hash='order',init_hash='init',
        layout_hash='layout',model_hash={'F':'model'},training_loss=.5,clean_loss=.6,cost={})
    assert guardian.persist_epoch(tmp_path,'toy',row)=='SCALAR_COMMITTED'
    assert guardian.persist_epoch(tmp_path,'toy',row)=='REPLAY_VERIFIED'
    with pytest.raises(ValueError): guardian.persist_epoch(tmp_path,'toy',dict(row,model_hash={'F':'conflict'}))
    assert all(path.suffix in ('.json','.jsonl') for path in tmp_path.rglob('*') if path.is_file())


def test_resume_ownership_and_rejected_ram_commit(tmp_path,monkeypatch):
    monkeypatch.setattr(guardian,'alive',lambda owner:bool(owner) and owner['pid']==101)
    monkeypatch.setattr(guardian,'process_identity',lambda pid:dict(pid=pid))
    saved=dict(commit_index=7,theta=torch.ones(2))
    rows=dict(live=dict(status='RUNNING',worker=dict(pid=101),commit_index=7,state=saved),
        failed=dict(status='ENGINEERING_BLOCKED',worker=None,commit_index=7,state=saved),
        dead=dict(status='RUNNING',worker=dict(pid=303),commit_index=7,state=saved))
    jobs={key:dict(job_id=key) for key in rows}
    calls=[]
    def request(output,command,**payload):
        calls.append(command)
        if command=='retry':
            return guardian.retry_job(output,payload['key'],rows[payload['key']])
        raise ValueError('Another worker owns this registration')
    monkeypatch.setattr(guardian,'request',request)
    assert run.ready_jobs(tmp_path,[jobs['live']],rows,{},True,set())==[]
    with pytest.raises(RuntimeError): run.ready_jobs(tmp_path,[jobs['failed']],rows,{},False,set())
    retried=set()
    assert run.ready_jobs(tmp_path,[jobs['failed'],jobs['dead']],rows,{},True,retried)==[jobs['failed'],jobs['dead']]
    assert retried=={'failed','dead'} and calls==['retry','retry']
    assert rows['failed']['state'] is saved and rows['dead']['state'] is saved
    rows['failed']['status']='ENGINEERING_BLOCKED'
    with pytest.raises(RuntimeError): run.ready_jobs(tmp_path,[jobs['failed']],rows,{},True,retried)
    with pytest.raises(ValueError): guardian.fail_job(rows['live'],202,'foreign failure')
    assert rows['live']['status']=='RUNNING' and rows['live']['worker']==dict(pid=101)
    monkeypatch.setattr(run,'review_gate',lambda output,prepared:None)
    with pytest.raises(ValueError):
        run.execute_job(tmp_path,dict(jobs=[jobs['live']]),'live','cpu')
    assert calls==['retry','retry','register']
    assert rows['live']['status']=='RUNNING'
    guardian.fail_job(rows['live'],101,'own failure')
    assert rows['live']['status']=='ENGINEERING_BLOCKED'
    record=dict(job_id='toy',epoch=1,commit_index=1,global_step=1,order_hash='order',init_hash='init',
        layout_hash='layout',model_hash={'F':'model'},training_loss=.5,clean_loss=.6,cost={})
    guardian.persist_epoch(tmp_path,'toy',record)
    previous=dict(theta=torch.zeros(2))
    row=dict(job=jobs['failed'],state=previous,commit_index=0,status='RUNNING')
    snapshot=dict(job=jobs['failed'],theta=torch.ones(2),commit_index=1)
    with pytest.raises(ValueError):
        guardian.commit_epoch(tmp_path,'toy',row,snapshot,dict(record,model_hash={'F':'conflict'}))
    assert row['state'] is previous and row['commit_index']==0 and 'record_hash' not in row
    assert guardian.commit_epoch(tmp_path,'toy',row,snapshot,record)['status']=='REPLAY_VERIFIED'
    assert row['state'] is snapshot and row['commit_index']==1
    assert guardian.commit_epoch(tmp_path,'toy',row,snapshot,record)['status']=='ALREADY_COMMITTED'
    assert parser().parse_args(['train','--output',str(tmp_path),'--resume']).resume is True
