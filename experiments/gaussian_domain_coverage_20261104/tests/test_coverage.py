"""Checks limited to the coverage intervention, sample budget and RAM protocol."""
import copy
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from experiments.gaussian_domain_coverage_20261104 import objectives, worker, config
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.views import apply_view
from experiments.overnight_ab_20261103 import selection, state


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 2)
        self.head = torch.nn.Linear(3, 2)
        self.perturb_slices = [(0, 2)]
        self.calls = []

    def forward(self, x, *, view=None, kappa=.1):
        clean = self.embedding(x[0].long())
        embedding = torch.cat((clean, x[-1,None].T), dim=1)
        self.calls.append(dict(domains=x[-1].detach().clone(), view=view))
        if view is not None:
            altered = apply_view(embedding,self.perturb_slices,view,kappa)
            assert torch.equal(altered[:,-1],embedding[:,-1])
            embedding = altered
        return self.head(embedding).sigmoid()


def fixture():
    torch.manual_seed(2022)
    model = TinyModel()
    x = torch.tensor([[0,1,2,3,4,5,6], [0,0,0,1,1,2,2]],dtype=torch.float32)
    y = torch.tensor([[0,1],[1,0],[1,1],[0,0],[1,1],[0,1],[1,0]],dtype=torch.float32)
    return model, x, y


@pytest.mark.parametrize('domain',[0,2])
def test_singleton_is_unchanged_source_loss_and_gradient(domain):
    a,x,y = fixture()
    b=copy.deepcopy(a)
    loss,info = objectives.losses(a,x,y,[domain],seed=2023,step=19)
    expected,source = objectives.source_losses(b,x,y,domain,
        dict(num_views=1,view_family='gaussian',pairing='iid'),'kuairand',2023,19,mix_lambda=.5,kappa=.1)
    assert torch.equal(loss,expected)
    assert info['views'][0]['key']==source['views'][0]['key']
    assert torch.equal(info['views'][0]['noise'],source['views'][0]['noise'])
    loss.backward()
    expected.backward()
    assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))


def test_dual_coverage_budget_both_tasks_and_noise_sharing():
    model,x,y = fixture()
    loss,info = objectives.losses(model,x,y,[0,2],seed=2024,step=7)
    assert [v['domain'] for v in info['domain_views']]==[0,2]
    assert [v['rows'] for v in info['domain_views']]==[3,2]
    assert len(model.calls)==3 and not any(bool((r['domains']==1).any()) for r in model.calls[1:])
    weighted=info['real'].clone()
    for domain,aug in zip((0,2),info['augmented']):
        mask=x[-1]==domain
        weighted[mask]=.5*info['real'][mask]+.5*aug
        singleton=TinyModel()
        singleton.load_state_dict(model.state_dict())
        _,single=objectives.losses(singleton,x,y,[domain],seed=2024,step=7)
        view=next(v for v in info['domain_views'] if v['domain']==domain)
        assert torch.equal(view['noise'],single['views'][0]['noise'])
    assert torch.allclose(loss,weighted.sum()/(2*len(y)),rtol=0,atol=1e-7)
    grad=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
    direct=torch.autograd.grad(weighted.mean(),tuple(model.parameters()))
    assert all(torch.allclose(a,b,rtol=1e-6,atol=1e-8) for a,b in zip(grad,direct))
    for task in (0,1):
        changed=y.clone()
        changed[x[-1]==0,task]=1-changed[x[-1]==0,task]
        value,_=objectives.losses(model,x,changed,[0,2],seed=2024,step=7)
        assert value!=loss
    # The two tasks share the original sample denominator even in this seven-row tail batch.
    assert info['real'].numel()==14 and info['extra_views']==5


@pytest.mark.parametrize('domains,weight,kappa,only_d1',[
    ([0,2],0,.1,False),([0,2],.5,0,False),([0,2],.5,.1,True),
    ([0],0,.1,False),([0],.5,0,False)])
def test_disabled_or_empty_coverage_is_original_loss(domains,weight,kappa,only_d1):
    model,x,y=fixture()
    if only_d1:
        x=x[:,3:5]
        y=y[3:5]
    before=torch.get_rng_state().clone()
    loss,info=objectives.losses(model,x,y,domains,seed=2022,step=1,mix_lambda=weight,kappa=kappa)
    assert info['forwards']==1 and info['extra_views']==0 and len(model.calls)==1
    assert torch.equal(torch.get_rng_state(),before)
    assert torch.equal(loss,info['real'].mean(1).mean())


def test_rms_scale_is_detached_and_clean_embedding_has_gradient():
    clean=torch.tensor([[3.,4.,9.]],requires_grad=True)
    noise=torch.tensor([[[.5,-.25]]])
    view=dict(family='gaussian',noise=noise)
    result=apply_view(clean,[(0,2)],view,.1)
    result.sum().backward()
    assert torch.equal(clean.grad,torch.ones_like(clean))
    assert result[0,2]==9 and not noise.requires_grad


def test_one_optimizer_backward_ema_update_and_state_restore(monkeypatch):
    model,x,y=fixture()
    ema=copy.deepcopy(model).requires_grad_(False)
    c=dict(target_domains=[0,2],lambda_aug=.5,kappa=.1,rho=.995)
    opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-5,foreach=False,fused=False)
    before=copy.deepcopy(ema.state_dict())
    updates=[]
    original=opt.step
    def counted(*a,**kw):
        updates.append(1)
        return original(*a,**kw)
    monkeypatch.setattr(opt,'step',counted)
    value=objectives.train_step(model,opt,ema,x,y,np.arange(7),c,2022,1)
    assert len(updates)==1 and value['optimizer_steps']==value['ema_updates']==value['backward_calls']==1
    assert all(int(s['step'])==1 for s in opt.state.values())
    for name,p in ema.state_dict().items():
        assert torch.equal(p,before[name]*.995+model.state_dict()[name]*.005)
    assert all(p.grad is None for p in ema.parameters())
    saved=dict(model=copy.deepcopy(model.state_dict()),ema=copy.deepcopy(ema.state_dict()),
        optimizer=copy.deepcopy(opt.state_dict()),rng=state.capture_rng('cpu'))
    objectives.train_step(model,opt,ema,x,y,np.arange(7),c,2022,2)
    expected=(copy.deepcopy(model.state_dict()),copy.deepcopy(ema.state_dict()),copy.deepcopy(opt.state_dict()))
    model.load_state_dict(saved['model'])
    ema.load_state_dict(saved['ema'])
    opt.load_state_dict(saved['optimizer'])
    state.restore_rng(saved['rng'],'cpu')
    objectives.train_step(model,opt,ema,x,y,np.arange(7),c,2022,2)
    assert state.state_digest(expected)==state.state_digest((model.state_dict(),ema.state_dict(),opt.state_dict()))


def test_worker_export_restore_and_loader_selection(monkeypatch):
    model,x,y=fixture()
    c=dict(target_domains=[0,2],lambda_aug=.5,kappa=.1,rho=.995,batch_size=7,eval_batch_size=7,epochs=50)
    job=dict(seed=2022,config=c,job_id='fixture',arm='G-D02')
    def create(job,bundle,device):
        model,_,_=fixture()
        return dict(job=copy.deepcopy(job),bundle=bundle,device=device,model=model,
            ema=copy.deepcopy(model).requires_grad_(False),
            optimizers={'main':torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-5,foreach=False,fused=False)},
            loader=torch.Generator().manual_seed(2022+11000),rng=state.capture_rng(device),
            stage='train',next_stage_epoch=1,completed_search=0,completed_final=0,commit_index=0,global_step=0,
            selection={k:dict(final_stage='train',best_score=-float('inf'),selected=None) for k in ('online','fixed_ema')},
            snapshots={k:{} for k in ('online','fixed_ema')},cost={},complete=False,
            init_hash=state.state_digest(model.state_dict()))
    bundle=SimpleNamespace(train_x=x.T.numpy(),train_y=y.numpy(),train_row_ids=np.arange(7),
                           val_x=x.T.numpy(),val_y=y.numpy())
    monkeypatch.setattr(worker,'create_session',create)
    def evaluate(model,bundle,split,method,batch_size):
        with torch.no_grad():
            p=model(torch.tensor(bundle.val_x).T)
        value=float(p.mean())
        return dict(macro_auc=value,macro_logloss=float(F.binary_cross_entropy(p,torch.tensor(bundle.val_y))),
                    **{cell+'_auc':value for cell in selection.CELLS})
    monkeypatch.setattr(worker.evaluation,'evaluate_model',evaluate)
    first=create(job,bundle,'cpu')
    worker.advance_epoch(first)
    saved=worker.export_session(first)
    worker.advance_epoch(first)
    expected=worker.export_session(first)
    resumed=worker.restore_session(job,bundle,'cpu',saved)
    worker.advance_epoch(resumed)
    actual=worker.export_session(resumed)
    expected.pop('cost')
    actual.pop('cost')
    assert state.state_digest(actual)==state.state_digest(expected)
    tracker=dict(final_stage='train',best_score=-float('inf'),selected=None)
    v={cell+'_auc':.7 for cell in selection.CELLS}
    assert selection.consider(tracker,'train',1,v,{'p':torch.ones(1)})
    assert not selection.consider(tracker,'train',2,v,{'p':torch.zeros(1)})
    assert tracker['selected']['epoch']==1
    assert not selection.consider(tracker,'test',3,{cell+'_auc':.99 for cell in selection.CELLS},{})


def test_matrix_and_seed_classification_contract():
    jobs=[config.make_job(a,s,'data','source') for s in config.SEEDS for a in config.ARMS]
    assert len(jobs)==10 and sum(j['config']['epochs'] for j in jobs)==500
    assert {j['seed'] for j in jobs}==set(range(2022,2027))
    assert all(j['config']['batch_size']==40960 for j in jobs)
    assert all(j['config']['rho']==.995 for j in jobs)


def test_guardian_loss_preserves_scalars_and_starts_a_fresh_commit_chain(tmp_path):
    from experiments.gaussian_domain_coverage_20261104 import run
    from experiments.gaussian_domain_coverage_20261104.reporting import committed_chain
    key='kuairand.G-D0.s2022'
    parent=tmp_path/'runs'/key
    ledger=tmp_path/'epoch_commit_ledger.jsonl'
    for epoch in (1,2):
        path=parent/'launch_1'/f'epoch_{epoch:03d}.json'
        state.write_json(path,dict(epoch=epoch,cost={'worker_seconds':epoch}))
        state.append_json(ledger,dict(key=key,commit_index=epoch,epoch_record_sha=state.sha_file(path)))
    state.write_json(parent/'launch_1'/'physical_cost.json',dict(physical_worker_seconds=3.))
    original_ledger=ledger.read_bytes()
    queue_state=dict(jobs={key:dict(status='RUNNING',commit_index=2,worker=None)})
    run.reconcile(tmp_path,queue_state,dict(jobs={}))
    assert queue_state['jobs'][key]['status']=='STATE_LOST_REPLAY'
    assert queue_state['jobs'][key]['commit_index']==0
    preserved=tmp_path/'replay_history'/key/'replay_1'
    assert state.read_json(preserved/'launch_1'/'epoch_002.json')['epoch']==2
    assert state.read_json(preserved/'launch_1'/'physical_cost.json')['physical_worker_seconds']==3.
    assert ledger.read_bytes()==original_ledger
    for epoch in range(1,51):
        path=parent/'launch_1'/f'epoch_{epoch:03d}.json'
        state.write_json(path,dict(epoch=epoch,cost={'worker_seconds':epoch+10}))
        state.append_json(ledger,dict(key=key,commit_index=epoch,epoch_record_sha=state.sha_file(path)))
    chain=committed_chain(tmp_path,key)
    assert [row['commit_index'] for row in chain]==list(range(1,51))
    hashes={state.sha_file(path) for path in parent.glob('launch_*/epoch_*.json')}
    assert all(row['epoch_record_sha'] in hashes for row in chain)
    assert len(ledger.read_text().splitlines())==52


def test_completed_scalar_archive_releases_retained_ram(tmp_path,monkeypatch):
    from experiments.gaussian_domain_coverage_20261104 import run
    key='done'
    parent=tmp_path/'runs'/key
    state.write_json(parent/'endpoints.json',{'complete':True})
    state.write_json(parent/'DONE.json',dict(job_id=key,files={'endpoints.json':state.sha_file(parent/'endpoints.json')}))
    released=[]
    monkeypatch.setattr(run.guardian,'request',lambda attempt,command,**kw:released.append((command,kw['key'])))
    queue_state=dict(jobs={key:dict(status='RUNNING',commit_index=50)})
    run.reconcile(tmp_path,queue_state,dict(jobs={key:dict(status='TRAIN_COMPLETE')}))
    assert queue_state['jobs'][key]['status']=='DONE' and released==[('release',key)]
