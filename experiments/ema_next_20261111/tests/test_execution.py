import os
from pathlib import Path
import pytest
import torch
from experiments.ema_next_20261111.protocol import ROOT,resolve_plan,resource_forecast,history_schedule,SEEDS,CELLS
from experiments.ema_next_20261111.decision import value_gate,choose,match_decision,primary_choice
from experiments.ema_next_20261111.state import Guardian,unique_bytes


def point(value):
    return dict(macro_auc=value,macro_logloss=.5,**{c+'_auc':value for c in CELLS})


def test_finite_matrix_and_history():
    study=resolve_plan(ROOT/'research_plans/26-11-11_ema_next_phase')
    jobs=study['jobs']
    assert len(jobs)==50
    assert sum(j['end_epoch']-j['start_epoch'] for j in jobs)==740
    assert sum((j['end_epoch']-j['start_epoch'])*j['config']['steps_per_epoch'] for j in jobs)==25530
    assert all(len(menu)==21 for menu in study['readouts'].values())
    assert history_schedule('movielens',study['manifest'])['peak']==5
    assert history_schedule('kuairand',study['manifest'])['peak']==11
    assert study['manifest']['route_C']['risk_tie_order'][1]=='closest to 0.5'
    resources=resource_forecast(study)
    assert resources['status']=='PASS'
    assert resources['datasets']['kuairand']['prior_dataset_endpoint_bytes']>0
    assert all(v['event_count']==250 for v in resources['datasets'].values())


def test_five_seed_gate_and_equality():
    control={s:point(0.) for s in SEEDS};candidate={s:point(.0002) for s in SEEDS}
    assert value_gate('x',candidate,control)['passed']
    assert value_gate('x',{s:p for s,p in candidate.items() if s!=2026},control)['status']=='WAITING_FOR_FIVE_SEEDS'
    candidate[2022]['d0t0_auc']=-.01
    assert not value_gate('x',candidate,control)['passed']
    assert match_decision({}, {})['status']=='WAITING_FOR_FIVE_SEEDS'


def test_hpo_tie_does_not_change_checkpoint_selection():
    from experiments.ema_next_20261111.decision import selected
    records=[dict(job_id='j',epoch=1,readouts={'F':dict(metrics=point(.6))}),dict(job_id='j',epoch=2,readouts={'F':dict(metrics=point(.60000001))})]
    assert selected(records,'F')['epoch']==2
    candidates={'simple':{s:point(.6) for s in SEEDS},'complex':{s:point(.60000001) for s in SEEDS}}
    assert choose(candidates,order=['simple','complex'])=='simple'


def test_unique_storage_and_epoch_recovery(tmp_path):
    g=Guardian(tmp_path);pid=os.getpid();job=dict(start_epoch=0,end_epoch=2,family='PARENT',config={})
    g.handle(dict(command='register',key='j',job=job,pid=pid))
    first=torch.arange(4.);assert unique_bytes([first,first[:2]])==16
    base=dict(kind='single',vector=first,refs=[]);descriptor=dict(kind='predictor',base_ref='b1',refs=['b1'])
    state=dict(object_refs=['d1'],selected={'F':dict(packet_id='d1')},online=first.clone())
    rec=dict(epoch=1,job_id='j',cost={})
    g.handle(dict(command='commit',key='j',pid=pid,state=state,new_objects={'b1':base,'d1':descriptor},record=rec))
    recovered=g.handle(dict(command='get',key='j'))
    assert torch.equal(recovered['row']['state']['online'],first)
    assert set(recovered['objects'])=={'b1','d1'}
    second=dict(object_refs=['d1','d2'],selected={'F':dict(packet_id='d2')},prefix={'F':dict(packet_id='d1')},online=first+1)
    g.handle(dict(command='commit',key='j',pid=pid,state=second,new_objects={'b2':dict(kind='single',vector=first+1,refs=[]),'d2':dict(kind='predictor',base_ref='b2',refs=['b2'])},record=dict(epoch=2,job_id='j',cost={})))
    assert 'b1' in g.objects
    g.handle(dict(command='complete',key='j',keep_live=False))
    g.handle(dict(command='retain',key='j',readouts=['F'],anchors=False))
    assert set(g.objects)=={'b2','d2'}


def test_missing_packet_does_not_commit(tmp_path):
    g=Guardian(tmp_path);pid=os.getpid();g.handle(dict(command='register',key='j',job=dict(start_epoch=0,end_epoch=1),pid=pid))
    with pytest.raises(ValueError,match='Missing immutable'):
        g.handle(dict(command='commit',key='j',pid=pid,state=dict(object_refs=['absent']),new_objects={},record=dict(epoch=1,cost={})))
    assert g.rows['j']['epoch']==0
    assert not (tmp_path/'runs/j/epochs/E001.json').exists()


def test_independent_gpu_admission(monkeypatch):
    from experiments.ema_next_20261111 import scheduler
    monkeypatch.setattr(scheduler,'gpu_state',lambda _: [dict(index=0,total=24<<30,used=23<<30,power=340,power_limit=350),dict(index=1,total=24<<30,used=1<<30,power=50,power_limit=350)])
    p=dict(resource_estimate=dict(datasets={'d':dict(peak=dict(guardian_predicate=1,physical_tensor_bytes=1),layout_elements=dict(best={},prefix={},anchors={},active=0))}))
    device,receipt=scheduler.admit(p,dict(payload_bytes=0),dict(dataset='d',job_id='j',config=dict(parameter_count=1)),[0,1],0)
    assert device==1 and receipt['reason']=='ADMITTED'
