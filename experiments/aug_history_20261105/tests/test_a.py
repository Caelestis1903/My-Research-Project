"""A's graph paths, training-only controller, source purity and RAM clocks."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.overnight_ab_20261103.state import state_digest,write_json,read_json,digest,sha_file
from experiments.overnight_ab_20261103.data_bridge import cached_bundle
from experiments.mdmt_history_information_20261103 import protocol as history
from .. import protocol,worker,grouped_ema,run
from ..group_registry import GROUPS,build_path_groups
from ..grouped_ema import GroupedEMAState,grouped_update,virtual_parameters,controller_step
from ..ranking_probe import RankProbeSampler,ProbeBatch
from ..model_adapter import build_model
from ..objectives import source_losses

MANIFEST=protocol.ROOT/'research_plans/26-11-5/experiment_manifest.json'
torch.set_num_threads(1)


def a_job(dataset):
    p=protocol.resolve(MANIFEST,dataset,'A')
    return dict(dataset=dataset,seed=2022,job_id='ENGINEERING_A_'+dataset,
        trajectory_id='ENGINEERING_A_'+dataset,route='A',config=p.config,candidates=p.candidates)


def small_rows(bundle):
    return np.concatenate([np.flatnonzero(bundle.train_x[:,-1]==d)[:2] for d in range(3)])


@pytest.mark.parametrize('dataset',['movielens','kuairand'])
def test_group_partition_and_actual_lookup(dataset):
    bundle=cached_bundle(protocol.CACHE,dataset,include_test=False)
    model=build_model(bundle,a_job(dataset)['config']); groups=build_path_groups(model,bundle)
    assert sum(groups.counts.values())==sum(p.numel() for p in model.parameters())
    assert groups.counts['INACTIVE']==0 and set(groups.coordinates.tolist())==set(range(12))
    lookup=f'embedding.embed_dict.{bundle.features.columns.index("domain_indicator")}.weight'
    rows=[r for r in groups.records if r['parameter_name']==lookup]
    assert [(r['group'],r['row_indices'],r['numel']) for r in rows]==[(f'D{d}',[d],16) for d in range(3)]
    for index in range(6):
        for prefix in ('gate','tower'):
            assert {r['group'] for r in groups.records if r['parameter_name'].startswith(f'{prefix}.{index}.')}=={f'DT{index%3}{index//3}'}


def test_single_cell_gradient_paths():
    bundle=cached_bundle(protocol.CACHE,'movielens',include_test=False)
    model=build_model(bundle,a_job('movielens')['config']); groups=build_path_groups(model,bundle)
    rows=small_rows(bundle); x=torch.tensor(np.array(bundle.train_x[rows])).T
    logits=model.forward_logits(x); parameters=tuple(model.parameters()); responsive=set()
    for domain in range(3):
        for task in range(2):
            gradients=torch.autograd.grad(logits[x[-1]==domain,task].sum(),parameters,retain_graph=True,allow_unused=True)
            flat=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,gradients)])
            for record in groups.records:
                active=any(bool(flat[left:right].ne(0).any()) for left,right in record['flat_ranges'])
                if [domain,task] not in record['source_cells']:
                    assert not active,record['parameter_name']
                if active: responsive.add(record['group'])
    assert responsive==set(GROUPS)


def test_probe_train_exclusion_rng_and_empty_cells():
    domains=np.repeat(np.arange(3),80)
    labels=np.stack((np.arange(240)%2,(np.arange(240)//2)%2),1).astype('float32')
    bundle=SimpleNamespace(dataset='movielens',train_x=np.stack((np.arange(240),domains),1).astype('float32'),
        train_y=labels,train_row_ids=np.arange(240)+1000,
        val_x=np.full((1,2),-100),val_y=np.ones((1,2)))
    before=history.rng_state('cpu')
    sampler=RankProbeSampler(bundle); excluded=bundle.train_row_ids[:8]
    probe=sampler.sample(excluded,2022,20); again=sampler.sample(excluded,2022,20)
    assert history.equal(before,history.rng_state('cpu'))
    assert np.array_equal(probe.row_ids,again.row_ids) and probe.metadata==again.metadata
    assert len(probe.cells)==6 and len(probe.row_ids)==192
    assert not np.isin(probe.row_ids,excluded).any()
    for cell in probe.cells:
        for kind,label in (('positive',1),('negative',0)):
            raw=probe.row_ids[cell[kind]]; assert len(np.unique(raw))==len(raw)==16
            assert (bundle.train_y[raw-1000,cell['task']]==label).all()
    bundle.train_y=np.zeros_like(bundle.train_y)
    empty=RankProbeSampler(bundle).sample([],2022,20)
    assert not empty.cells and empty.metadata['reason']=='NO_VALID_TRAIN_CELL'


def test_fixed_profile_q0_and_controller_bounds(monkeypatch):
    layout=SimpleNamespace(names=('weights',),shapes=((12,),),offsets=((0,12),))
    registry=SimpleNamespace(layout=layout,rho0=.9,counts=dict(INACTIVE=0),indices=lambda device:torch.arange(12,device=device))
    phi=torch.arange(12,dtype=torch.float32)/12; theta=phi+1
    expected=phi.clone().mul_(.9).add_(theta,alpha=1-.9)
    actual=phi.clone(); grouped_update(actual,theta,registry,[.9]*12)
    assert torch.equal(actual,expected)
    rates=[.9**2,.9**.5]*6; actual=phi.clone(); grouped_update(actual,theta,registry,rates)
    expected=torch.stack([old*r+new*(1-r) for old,new,r in zip(phi,theta,rates)])
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-5)
    same=theta.clone(); grouped_update(same,theta,registry,rates)
    torch.testing.assert_close(same,theta,atol=1e-7,rtol=1e-5)
    q=torch.zeros(12,requires_grad=True)
    params=virtual_parameters(theta,phi,q,20,registry)
    gradient=torch.autograd.grad(params['weights'].sum(),q)[0]
    assert torch.count_nonzero(gradient)==12
    for count in (1,12):
        state=GroupedEMAState(phi.clone(),torch.zeros(count),.2,20,20)
        probe=ProbeBatch(np.empty((0,1)),np.empty(0),[{}],{'step':20})
        monkeypatch.setattr(grouped_ema,'virtual_parameters',lambda theta,phi,q,K,registry:dict(q=q))
        monkeypatch.setattr(grouped_ema,'ranking_objective',lambda params,probe,model:params['q'].sum())
        controller_step(state,theta,probe,registry,None)
        assert torch.equal(state.q,torch.full_like(state.q,-.2)) and state.q.numel()==count
        state.q.fill_(-.99); controller_step(state,theta,probe,registry,None)
        assert torch.equal(state.q,torch.full_like(state.q,-1.))
        before=state.q.clone(); probe.cells=[]
        controller_step(state,theta,probe,registry,None); assert torch.equal(state.q,before)
        probe.cells=[{}]
        monkeypatch.setattr(grouped_ema,'ranking_objective',lambda params,probe,model:params['q'].sum()*float('nan'))
        with pytest.raises(FloatingPointError): controller_step(state,theta,probe,registry,None)


@pytest.mark.parametrize('dataset',['movielens','kuairand'])
def test_a_source_purity_clocks_and_restore(dataset):
    bundle=cached_bundle(protocol.CACHE,dataset,include_test=False); job=a_job(dataset)
    session=worker.create_session(job,bundle,'cpu'); reference=copy.deepcopy(session['model']); c=job['config']
    opt=torch.optim.Adam(reference.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,
        weight_decay=c['weight_decay'],foreach=False,fused=False,amsgrad=False)
    rows=small_rows(bundle); x=torch.tensor(np.array(bundle.train_x[rows])).T; y=torch.tensor(np.array(bundle.train_y[rows]))
    raw=np.asarray(bundle.train_row_ids[rows]); restored=None
    assert len(session['bank'].states)==15
    assert all(torch.equal(state.phi,session['layout'].flatten(session['model'].parameters())) for state in session['bank'].states.values())
    for step in range(1,42):
        opt.zero_grad(set_to_none=True)
        loss,_=source_losses(reference,x,y,0,dict(num_views=1,view_family='gaussian',pairing='iid'),
            dataset,2022,step,mix_lambda=.5,kappa=c['kappa'])
        loss.backward(); opt.step()
        rng=history.rng_state('cpu')
        result=worker.train_batch(session,x,y,row_ids=raw)
        assert history.equal(rng,history.rng_state('cpu'))
        if step in (1,20,40,41):
            assert all(torch.equal(a,b) for a,b in zip(reference.parameters(),session['model'].parameters()))
            assert all(torch.equal(a.grad,b.grad) for a,b in zip(reference.parameters(),session['model'].parameters()))
        if step==19: assert not result['response']['controllers']
        if step==20: assert len(result['response']['controllers'])==4
        if step==21:
            restored=worker.restore_session(job,bundle,'cpu',worker.export_session(session))
        elif step>21:
            worker.train_batch(restored,x,y,row_ids=raw)
    for cid,state in session['bank'].states.items():
        other=restored['bank'].states[cid]
        assert torch.equal(state.phi,other.phi) and torch.equal(state.q,other.q)
        assert state.next_refresh==other.next_refresh and state.last_probe==other.last_probe
        if state.K:
            assert state.controller_updates==41//state.K
            assert state.last_refresh==40 and state.next_refresh==(60 if state.K==20 else 80)
            assert bool(state.q.ne(0).any())
        else:
            assert state.cost['controller_seconds']==0
    assert state_digest(session['model'].state_dict())==state_digest(restored['model'].state_dict())


def test_old_b_continuation_uses_its_own_identity(tmp_path):
    origin=tmp_path/'old_B'; target=tmp_path/'new_A'; target.mkdir(); (origin/'B').mkdir(parents=True)
    identity=dict(source_hash='old_source',manifest_hash='old_manifest')
    write_json(origin/'PREPARED.json',dict(identity=identity))
    selection=dict(route='B',seed_ids=protocol.SEEDS,source_hash='old_source',manifest_hash='old_manifest')
    selection['selection_hash']=digest(selection)
    write_json(origin/'B/selection_frozen.json',selection)
    write_json(origin/'B/DECISION.json',dict(route='B',selection_hash=selection['selection_hash'],
        next_actions=[dict(route='A',dataset='movielens',matrix='A_FULL',seeds=protocol.SEEDS)]))
    before={p.name:sha_file(p) for p in (origin/'B').iterdir()}
    receipt=run.continuation(target,origin/'B/DECISION.json',MANIFEST)
    assert receipt['allowed_actions'][0]['route']=='A'
    assert receipt['source_prepared']['identity']==identity
    assert before=={p.name:sha_file(p) for p in (origin/'B').iterdir()}
