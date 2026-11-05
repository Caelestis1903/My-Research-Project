"""C's input derivative, covariance/loss, teacher/RAM and sequential five-seed contracts."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.mdmt_history_information_20261103 import protocol as history
from experiments.overnight_ab_20261103.data_bridge import cached_bundle
from experiments.overnight_ab_20261103.state import state_digest,write_json,read_json,digest
from .. import protocol,worker,c_search,selection
from ..model_adapter import ModelAdapter,build_model
from ..drift_probe import input_drift,DriftProbeSampler
from ..covariance import statistics,covariance_root,fixed_rotations,controlled_statistics
from ..conditional_objective import conditional_loss_terms
from ..conditional_views import draw_conditional_views
from ..objectives import source_losses
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.views import draw_view
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.protocol import view_key

MANIFEST=protocol.ROOT/'research_plans/26-11-5/experiment_manifest.json'
torch.set_num_threads(1)


def prepared_C():
    p=protocol.resolve(MANIFEST,'movielens','C')
    return dict(route='C',scope_datasets=['movielens'],resolved_configs=dict(movielens=p.config),
        identity=dict(source_hash='ENGINEERING_SOURCE',manifest_hash=p.manifest_hash,datasets=dict(movielens='ENGINEERING_DATA')))


def job_C(gamma=-2.,K=20,mode='task'):
    cid='C_ROTATED' if mode=='rotated' else 'C1_NEG'
    return c_search.make_job(prepared_C(),dict(id=cid,gamma=gamma,refresh_steps=K,control_mode=mode),2022,'ENGINEERING_RESOLUTION')


class TinyEmbedding(nn.Module):
    def __init__(self):
        super().__init__(); self.table=nn.Embedding(4,16,dtype=torch.float64)
    def forward(self,x): return self.table(x[0].long())


class TinyAdapter(ModelAdapter):
    def __init__(self):
        nn.Module.__init__(self)
        self.perturb_slices=[(0,16)]; self.embedding=TinyEmbedding()
        self.skip_conn=nn.Linear(16,4,dtype=torch.float64); self.star_mlp=nn.Linear(16,4,dtype=torch.float64)
        self.gate=nn.ModuleList([nn.Sequential(nn.Linear(4,2,dtype=torch.float64),nn.Softmax(1)) for _ in range(6)])
        self.tower=nn.ModuleList([nn.Linear(4,1,dtype=torch.float64) for _ in range(6)])
    def forward(self,x):
        h=self.embedding(x); emb=self.star_mlp(h)+self.skip_conn(h); output=[]
        for index in range(6):
            g=self.gate[index](emb.detach())
            output.append(self.tower[index](g[:,:1]*emb+g[:,1:]*emb.square()).squeeze(1).sigmoid())
        return torch.stack([sum(torch.where(x[-1]==d,output[3*t+d],torch.zeros_like(output[0])) for d in range(3)) for t in range(2)],1)


def test_full_gate_delta_derivative_and_own_embedding_purity():
    torch.manual_seed(2022); online=TinyAdapter(); teacher=copy.deepcopy(online)
    with torch.no_grad(): teacher.embedding.table.weight.add_(.3)
    x=torch.tensor([[0,1,2],[0,1,2]],dtype=torch.float64)
    delta=torch.zeros(3,1,16,dtype=torch.float64,requires_grad=True)
    full=online.forward_logits(x,delta=delta,probe_input_full_graph=True)
    torch.testing.assert_close(full.sigmoid(),online(x),atol=1e-12,rtol=1e-12)
    detached=online.forward_logits(x,delta=delta,probe_input_full_graph=False)
    assert torch.equal(full,detached)
    gradient=torch.autograd.grad(full[:,0].sum(),delta)[0]
    detached_gradient=torch.autograd.grad(detached[:,0].sum(),delta)[0]
    plus=torch.zeros_like(delta); plus[0,0,0]=1e-6
    finite=(online.forward_logits(x,delta=plus,probe_input_full_graph=True)[:,0].sum()-
        online.forward_logits(x,delta=-plus,probe_input_full_graph=True)[:,0].sum())/(2e-6)
    torch.testing.assert_close(gradient[0,0,0],finite,atol=1e-8,rtol=1e-6)
    assert abs(float(gradient[0,0,0]-detached_gradient[0,0,0]))>1e-8
    for p in online.parameters(): p.grad=torch.full_like(p,.123)
    before=state_digest(dict(online=online.state_dict(),teacher=teacher.state_dict(),grads=[p.grad for p in online.parameters()]))
    rng=history.rng_state('cpu')
    drift=input_drift(online,teacher,x,1)
    forward_difference=lambda d:(online.forward_logits(x,delta=d,probe_input_full_graph=True)[:,0]-
        teacher.forward_logits(x,delta=d,probe_input_full_graph=True)[:,0]).sum()
    expected=(forward_difference(plus)-forward_difference(-plus))/(2e-6)
    torch.testing.assert_close(drift[0,0,0,0],expected,atol=1e-8,rtol=1e-6)
    assert history.equal(rng,history.rng_state('cpu'))
    assert before==state_digest(dict(online=online.state_dict(),teacher=teacher.state_dict(),grads=[p.grad for p in online.parameters()]))
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in online.modules())
    assert torch.equal(input_drift(online,copy.deepcopy(online),x,1),torch.zeros_like(drift))


def test_psd_trace_roots_rotation_and_expected_noise_energy():
    u=torch.zeros(2,2,1,16,dtype=torch.float64); u[0,:,:,0]=2
    matrix=statistics(u)
    torch.testing.assert_close(matrix.diagonal(dim1=-2,dim2=-1).sum(-1),torch.full((2,1),.5,dtype=torch.float64),atol=1e-10,rtol=0)
    roots={gamma:covariance_root(matrix,gamma) for gamma in (-2.,0.,2.)}
    for root in roots.values():
        covariance=root@root.transpose(-1,-2)
        assert bool((torch.linalg.eigvalsh(covariance)>0).all())
        torch.testing.assert_close(covariance.diagonal(dim1=-2,dim2=-1).sum(-1),torch.full((2,1),16.,dtype=torch.float64),atol=1e-10,rtol=0)
    assert roots[2.][0,0,0,0]>roots[2.][0,0,1,1] and roots[-2.][0,0,0,0]<roots[-2.][0,0,1,1]
    assert torch.equal(covariance_root(torch.zeros_like(matrix),2.),torch.eye(16,dtype=torch.float64).expand_as(matrix))
    rng=torch.get_rng_state().clone(); rotation=fixed_rotations('movielens',2022,1)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(rotation,fixed_rotations('movielens',2022,1))
    torch.testing.assert_close(rotation@rotation.transpose(-1,-2),torch.eye(16,dtype=torch.float64)[None],atol=1e-10,rtol=0)
    rotated=controlled_statistics(matrix,'rotated',rotation)
    torch.testing.assert_close(torch.linalg.eigvalsh(rotated),torch.linalg.eigvalsh(matrix),atol=1e-10,rtol=0)
    merged=covariance_root(controlled_statistics(matrix,'merged'),-2.)
    assert torch.equal(merged[0],merged[1])
    generator=torch.Generator().manual_seed(2022)
    noise=torch.randn(20000,1,16,generator=generator)
    model=type('Fields',(),dict(perturb_slices=[(0,16)]))()
    base=dict(family='gaussian',noise=noise,key='engineering',seed=2022,hash='base')
    before=noise.clone(); views=draw_conditional_views(model,torch.empty(1,20000),roots[2.],base)
    assert torch.equal(noise,before)
    energy=float((.05*2*views[0]['noise']).square().sum(-1).mean())
    assert abs(energy-(.05*2)**2*16)<.0032
    bad=matrix.clone(); bad[...,1,1]=-1e-3
    with pytest.raises(FloatingPointError): covariance_root(bad,2.)


def test_train_probe_and_gamma_zero_source_loss_update_ema():
    bundle=cached_bundle(protocol.CACHE,'movielens',include_test=False)
    ids=np.concatenate([np.flatnonzero(bundle.train_x[:,-1]==d)[:2] for d in range(3)])
    raw=np.asarray(bundle.train_row_ids[ids]); x=torch.tensor(np.array(bundle.train_x[ids])).T; y=torch.tensor(np.array(bundle.train_y[ids]))
    sampler=DriftProbeSampler(bundle); values,metadata=sampler.sample(raw,2022,21)
    assert len(values)==128 and (values[:,-1]==0).all()
    assert metadata==sampler.sample(raw,2022,21)[1]
    empty,_=sampler.sample(bundle.train_row_ids,2022,21); assert len(empty)==0
    job=job_C(0.); session=worker.create_session(job,bundle,'cpu'); reference=copy.deepcopy(session['model']); teacher=copy.deepcopy(reference)
    c=job['config']; optimizer=torch.optim.Adam(reference.parameters(),lr=.01,weight_decay=1e-4,foreach=False,fused=False)
    for step in (1,2):
        optimizer.zero_grad(set_to_none=True)
        loss,_=source_losses(reference,x,y,0,dict(num_views=1,view_family='gaussian',pairing='iid'),'movielens',2022,step,mix_lambda=.5,kappa=.05)
        loss.backward(); optimizer.step()
        with torch.no_grad():
            for phi,theta in zip(teacher.parameters(),reference.parameters()): phi.mul_(c['rho']).add_(theta,alpha=1-c['rho'])
        rng=history.rng_state('cpu'); result=worker.train_batch(session,x,y,row_ids=raw)
        assert history.equal(rng,history.rng_state('cpu')) and result['loss']==float(loss.detach())
        assert all(torch.equal(a,b) and torch.equal(a.grad,b.grad) for a,b in zip(reference.parameters(),session['model'].parameters()))
        assert all(torch.equal(a,b) for a,b in zip(teacher.parameters(),session['bank'].teacher.parameters()))


def test_task_views_actual_tail_denominator():
    bundle=cached_bundle(protocol.CACHE,'movielens',include_test=False); config=job_C()['config']
    model=build_model(bundle,config)
    ids=np.concatenate([np.flatnonzero(bundle.train_x[:,-1]==d)[:2] for d in range(3)])
    x=torch.tensor(np.array(bundle.train_x[ids])).T; y=torch.tensor(np.array(bundle.train_y[ids]))
    root=torch.eye(16,dtype=torch.float64).expand(2,6,16,16).clone()
    matrix=torch.zeros_like(root); matrix[1,:,:, :]=torch.diag(torch.tensor([.8,.2]+[0.]*14,dtype=torch.float64))
    root=covariance_root(matrix,-2.); cfg=dict(config,seed=2022,step=1)
    terms=conditional_loss_terms(model,x,y,root,cfg); mask=x[-1]==0
    base=draw_view(model,int(mask.sum()),'gaussian',view_key('movielens',2022,'train',1,'gaussian',1),'cpu')
    views=draw_conditional_views(model,x[:,mask],root,base)
    real=F.binary_cross_entropy(model(x),y,reduction='none')
    losses=[F.binary_cross_entropy(model(x[:,mask],view=v,kappa=.05),y[mask],reduction='none') for v in views]
    expected=(real.sum()+.5*sum(losses[t][:,t].sum()-real[mask,t].sum() for t in range(2)))/y.numel()
    torch.testing.assert_close(terms.total,expected,atol=1e-7,rtol=1e-5)
    assert terms.metadata['extra_views']==4
    no_target=x[:,x[-1]!=0]; labels=y[x[-1]!=0]
    empty=conditional_loss_terms(model,no_target,labels,root,cfg)
    assert not empty.metadata['views']


@pytest.mark.parametrize('K',[20,40])
def test_old_teacher_refresh_and_ram_restore(K):
    bundle=cached_bundle(protocol.CACHE,'movielens',include_test=False)
    job=job_C(-2.,K,'rotated' if K==20 else 'task'); session=worker.create_session(job,bundle,'cpu')
    ids=np.concatenate([np.flatnonzero(bundle.train_x[:,-1]==d)[:2] for d in range(3)])
    x=torch.tensor(np.array(bundle.train_x[ids])).T; y=torch.tensor(np.array(bundle.train_y[ids])); raw=np.asarray(bundle.train_row_ids[ids])
    for step in range(1,K+1):
        value=worker.train_batch(session,x,y,row_ids=raw)
        if step==1:
            assert not bool(session['bank'].A.ne(0).any())
    snapshot=worker.export_session(session); restored=worker.restore_session(job,bundle,'cpu',snapshot)
    old_hash=state_digest(session['bank'].teacher.state_dict())
    for step in (K+1,K+2):
        result=worker.train_batch(session,x,y,row_ids=raw); worker.train_batch(restored,x,y,row_ids=raw)
        assert state_digest(session['model'].state_dict())==state_digest(restored['model'].state_dict())
        for key in ('A','root'):
            assert torch.equal(getattr(session['bank'],key),getattr(restored['bank'],key))
        assert history.equal(session['optimizer'].state_dict(),restored['optimizer'].state_dict())
        assert state_digest(session['bank'].teacher.state_dict())==state_digest(restored['bank'].teacher.state_dict())
        if step==K+1:
            assert result['response']['teacher_preupdate_step']==K
            assert result['response']['probe']['teacher_hash']==old_hash
    assert session['bank'].last_refresh==K+1==session['bank'].last_valid_step
    assert bool(session['bank'].A.ne(0).any())
    assert history.equal(session['bank'].rotation,restored['bank'].rotation)


def test_five_seed_dependency_numeric_identity_freeze_and_scope(tmp_path,monkeypatch):
    p=prepared_C(); configs=[dict(id='C1_NEG',gamma=-2.,refresh_steps=20,control_mode='task'),dict(id='C2_POS',gamma=2.,refresh_steps=20,control_mode='task')]
    initial=c_search.resolve_stage(tmp_path,p,'I',configs,dict(engineering_source_hash=p['identity']['source_hash']))
    jobs=c_search.resolved_jobs(tmp_path,p); assert len(jobs)==10
    changed=c_search.make_job(p,dict(configs[0],gamma=2.),2022,initial['resolution_hash'])
    assert changed['job_id']!=jobs[0]['job_id'] and all(isinstance(j['config']['gamma'],float) for j in jobs)
    with pytest.raises(selection.Incomplete):
        c_search.resolve_stage(tmp_path,p,'II',[dict(id='C3_WEAK',gamma=-.5,refresh_steps=20,control_mode='task')],{})
    write_json(tmp_path/'PARENT_REFERENCE.json',dict(points=[]))
    summaries={j['job_id']:dict(complete=True,epoch=50) for j in jobs[:4]}
    with pytest.raises(selection.Incomplete): c_search.candidate_rows(tmp_path,p,['C1_NEG'],summaries)
    with pytest.raises(ValueError): c_search.scope_check(p,['movielens','kuairand'])
    with pytest.raises(ValueError): protocol.resolve(MANIFEST,'kuairand','C')
    c_search.immutable_json(tmp_path/'frozen.json',dict(gamma=-2.,K=20))
    with pytest.raises(ValueError): c_search.immutable_json(tmp_path/'frozen.json',dict(gamma=2.,K=20))
    with pytest.raises(selection.Incomplete): selection.require_frozen(tmp_path,tmp_path/'C/selection_frozen.json')
