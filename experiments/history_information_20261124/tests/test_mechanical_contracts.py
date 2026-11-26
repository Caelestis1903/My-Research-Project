"""Small contracts for the synchronous owner, capacity gate, and real-source anchor."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.conditional_history_shared_update_20261122.history_controls import history_config, make_history_optimizer
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.protocol import (
    rng_state, state_digest, update_ema,
)
from experiments.history_information_20261124 import diagnostics, objectives, scheduler
from experiments.history_information_20261124.observers import HistoryObserver
from experiments.history_information_20261124.protocol import RunSpec, STUDY, digest
from experiments.history_information_20261124.runner import Supervisor, _append_transaction, _meta
from experiments.history_information_20261124.state import ANCHOR, EPOCH, SELECTED, RamStore, capture, live_hashes, restore


def test_capacity_thresholds_and_scoped_stop(tmp_path, monkeypatch):
    left,right = (tmp_path/name for name in ('left','right'))
    monkeypatch.setattr(scheduler,'WORKSPACES',(left,right))
    scheduler.register_stop_controls(tmp_path)
    archived = right/'artifacts'/'26-11-16_augmentation_gradient_momentum'/'PAUSE_TRAINING'
    archived.parent.mkdir(parents=True)
    archived.touch()
    assert scheduler.stop_status(tmp_path) is None
    current = right/'artifacts'/STUDY/'USER_STOPPED'
    current.parent.mkdir()
    current.touch()
    assert scheduler.stop_status(tmp_path) == ('USER_STOPPED',str(current))
    gpu = dict(uuid='gpu',used=80,total=100,watts=20,limit=30)
    assert scheduler.eligible(gpu,.89)
    assert scheduler.eligible(gpu,.89,physical_fit=True,cuda_usable=99,full_process=19)
    assert not scheduler.eligible(gpu,.89,physical_fit=True,cuda_usable=99,full_process=20)
    assert not scheduler.eligible(gpu,.90)
    assert not scheduler.eligible(dict(gpu,used=85),.10)
    assert not scheduler.eligible(dict(gpu,watts=25.5),.10)
    idle = dict(gpu,used=1)
    assert scheduler.quiescent_capacity(idle,[],99)
    assert not scheduler.quiescent_capacity(idle,[{'uuid':'gpu'}],99)


def test_owner_selected_readback_and_rollback(tmp_path):
    supervisor = Supervisor(tmp_path,emit=lambda event: None)
    queue = dict(schema_version=1,study_id=STUDY,stage='S3a',jobs=[],prerequisite_receipts={})
    queue['queue_hash'] = digest(queue)
    receipt = supervisor.run(queue)
    assert receipt['status']=='NOT_TRIGGERED'
    assert supervisor.run(queue)==receipt
    spec = RunSpec('movielens',2022,'B0')
    job_path = tmp_path/spec.stage
    job_path.mkdir()
    state = {'weight':torch.tensor([1.,2.])}
    identity = dict(_meta(spec,1,'cpu','cpu',supervisor.store.generation),model_hash=state_digest(state),readout='online')
    supervisor.jobs[spec.job_key] = dict(spec=spec,status='COMPLETE',selected={'online':identity['model_hash']})
    packet = dict(schema=SELECTED,identity=identity,epoch=1,global_step=20,model_state=state)
    ack = supervisor.store.commit(f'{spec.job_key}:selected:online',packet,identity)
    value = supervisor.selected_get(spec.job_key,'online')
    value['model_state']['weight'].zero_()
    assert supervisor.selected_get(spec.job_key,'online')['ram_hash']==ack['get_hash']
    assert torch.equal(supervisor.selected_get(spec.job_key,'online')['model_state']['weight'],state['weight'])
    log = job_path/'steps.jsonl'
    log.write_text('first\n')
    with pytest.raises(RuntimeError), _append_transaction([log,job_path/'predictions.jsonl']):
        log.write_text('first\nsecond\n')
        (job_path/'predictions.jsonl').touch()
        raise RuntimeError('interrupted output')
    assert log.read_text()=='first\n' and not (job_path/'predictions.jsonl').exists()
    supervisor.release_selected([])
    with pytest.raises(KeyError):
        supervisor.selected_get(spec.job_key,'online')
    supervisor.close()


def test_epoch_restore_replays_loader_rng_and_private_steps():
    torch.manual_seed(17)
    model = torch.nn.Linear(2,1)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(),foreach=False,fused=False)
    loader = torch.Generator().manual_seed(44)
    order = torch.randperm(8,generator=loader).numpy()

    def one_step():
        opt.zero_grad()
        model(torch.rand(2,2)).sum().backward()
        opt.step()
        update_ema(ema,model,.98)
        return live_hashes(model,opt,ema), torch.randperm(8,generator=loader)

    one_step()
    store = RamStore()
    identity = dict(job_key='tiny',owner_generation=store.generation,data_hash='data',source_hash='source')
    packet = capture(EPOCH,identity,model,opt,ema,loader,None,'cpu',1,1,order,{},cursor=len(order))
    ack = store.commit('epoch',packet,identity)
    continuous_hashes, continuous_order = one_step()
    loaded, actual_hash = store.get('epoch',identity)
    assert actual_hash==ack['put_hash']
    restore(loaded,identity,model,opt,ema,loader,None,'cpu',lambda *_:None)
    replayed_hashes, replayed_order = one_step()
    assert replayed_hashes==continuous_hashes and torch.equal(replayed_order,continuous_order)
    with pytest.raises(ValueError,match='cursor'):
        store.commit('epoch',dict(packet,cursor=0),identity)
    with pytest.raises(ValueError,match='owner generation'):
        RamStore().commit('other',packet,identity)


class MiniModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_weight = torch.nn.Parameter(torch.tensor([.2]))
        self.expert = torch.nn.ModuleList([torch.nn.Linear(1,1,bias=False)])
        self.tower = torch.nn.ModuleList(torch.nn.Linear(1,1) for _ in range(6))

    def forward(self,x):
        base = self.expert[0](x[0,:,None]*self.shared_weight)
        logits = [tower(base).squeeze(1) for tower in self.tower]
        values = [sum(torch.where(x[-1]==d,logits[3*t+d],0) for d in range(3)) for t in range(2)]
        return torch.stack(values,dim=1).sigmoid()


def test_actual_h0_anchor_replay_and_fixed_panel_contract():
    torch.manual_seed(11)
    raw_x = np.asarray([[i/60.,i//20] for i in range(60)],dtype=np.float32)
    labels = np.asarray([[(i%4)//2,i%2] for i in range(60)],dtype=np.float32)
    rows = np.arange(60)
    order = rows.copy()
    bundle = SimpleNamespace(train_x=raw_x,train_y=labels,train_ids=rows,
                             val_x=raw_x,val_y=labels,val_ids=rows+60)
    spec = RunSpec('movielens',2022,'H0')
    model = MiniModel()
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = make_history_optimizer(model,history_config('A21','movielens'))
    observer = HistoryObserver(spec,optimizer)
    loader = torch.Generator().manual_seed(9)
    store = RamStore()
    identity = dict(job_key=spec.job_key,owner_generation=store.generation)
    context = dict(dataset=spec.dataset,seed=spec.seed,epoch=5,global_step=1)
    pre_rng = rng_state('cpu')
    packet = None

    def hook(_,pair,raw):
        nonlocal packet
        packet = capture(ANCHOR,identity,model,optimizer,ema,loader,observer,'cpu',5,1,order,{},cursor=0,
                         row_ids=rows,batch_indices=order,native_grad={n:p.grad.clone() for n,p in model.named_parameters()},
                         pair=pair,raw_pair=raw,pre_forward_rng=pre_rng,pre_forward_buffers={})
        observer.observe(pair,raw)

    objectives.step(model,optimizer,ema,torch.from_numpy(raw_x).T,torch.from_numpy(labels),rows,spec,context,hook)
    observer.finish()
    packet['actual_post'] = live_hashes(model,optimizer,ema)
    store.commit('anchor',packet,identity)
    restored, _ = store.get('anchor',identity)
    bank = diagnostics.panels(spec.dataset,spec.seed,bundle)
    diagnostics.attach_pairs(bank,spec.dataset,spec.seed,bundle)
    assert not set(bank['V_A']['row_ids']).intersection(bank['V_B']['row_ids'])
    before = live_hashes(model,optimizer,ema),state_digest(rng_state('cpu')),state_digest(loader.get_state())
    output,cost = diagnostics.anchor_diagnostics(model,optimizer,observer,restored,bank,'cpu',40960)
    after = live_hashes(model,optimizer,ema),state_digest(rng_state('cpu')),state_digest(loader.get_state())
    assert before == after
    assert output[0]['actual_post']==packet['actual_post']
    assert cost['diagnostic_reference_steps']==1 and cost['diagnostic_virtual_steps']==10
    assert cost['diagnostic_panel_evaluations']==66 and len({r['branch_id'] for r in output if r['diagnostic']=='D2_PANEL'})==11
    assert cost['diagnostic_probe_autograd_sweeps']==6
    cross_error = max(r['cross_reconstruction_max_abs'] for r in observer.last)
    assert cross_error<1e-6
    frozen = diagnostics.fixed_pairs(spec.dataset,spec.seed,'VALIDATION',labels,raw_x[:,-1],'D4',32768)
    scored = diagnostics.pair_scores(labels,'d0t0',frozen['d0t0'])
    assert diagnostics.pair_transition(scored,scored)['maintained_correct']==32768
    print(json.dumps(dict(anchor=state_digest(restored),post=packet['actual_post'],counts=cost,
                          cross_reconstruction_max_abs=cross_error),sort_keys=True))


def test_d3_adjusted_operator_identity():
    from experiments.history_information_20261124.optimizers import transition

    spec = RunSpec('movielens',2022,'C2_A100')
    state, arithmetic = {}, torch.zeros(2,dtype=torch.float64)
    h_values = [torch.tensor([.2,-.4],dtype=torch.float64),torch.tensor([.5,.8],dtype=torch.float64)]
    d_values = []
    for i, h in enumerate(h_values,1):
        mixed = torch.tensor([.3,.5],dtype=torch.float64)*i
        state,_ = transition(state,mixed,h,1.,spec)
        d = (state['v']/(1-.999**i)).sqrt().add_(1e-8).reciprocal_()
        if i==1:
            torch.testing.assert_close(state['u'],state['q']*d,rtol=1e-10,atol=1e-12)
        arithmetic = .99*arithmetic+.01*d*h
        d_values.append(d)
    torch.testing.assert_close(state['u'],arithmetic,rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(state['u']-d_values[1]*state['q'],.0099*(d_values[0]-d_values[1])*h_values[0],
                               rtol=1e-10,atol=1e-12)
