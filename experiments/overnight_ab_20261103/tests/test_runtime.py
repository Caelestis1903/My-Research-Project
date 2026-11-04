"""Small checks for selection, paired statistics, queue fairness and RAM IPC."""
import math
import os
import subprocess
import sys
import time
from pathlib import Path
import pytest
import torch
from experiments.overnight_ab_20261103 import guardian,selection,queue
from experiments.overnight_ab_20261103.reporting import decision
from experiments.overnight_ab_20261103.state import write_json

def validation(value):
    return {c+'_auc':value for c in selection.CELLS}
def test_selection_strict_tie_and_stage():
    tracker=dict(final_stage='reuse',best_score=-math.inf,selected=None); state={'w':torch.ones(2)}
    assert not selection.consider(tracker,'search',1,validation(.99),state)
    assert not selection.consider(tracker,'reuse',0,validation(.99),state)
    assert selection.consider(tracker,'reuse',1,validation(.7),state)
    assert selection.consider(tracker,'reuse',2,validation(.8),state)
    assert not selection.consider(tracker,'reuse',3,validation(.8),state)
    assert tracker['selected']['epoch']==2
    with pytest.raises(ValueError):
        selection.consider(tracker,'reuse',4,{c+'_auc':None for c in selection.CELLS},state)
def test_seed_decisions_independent():
    rows=[dict(seed=s,delta_macro_auc=.003,delta_macro_logloss=.01,
        **{'delta_'+c+'_auc':-.001 if c=='d0t0' else .0038 for c in selection.CELLS}) for s in range(2022,2027)]
    assert decision(rows)['macro_auc_pass'] and decision(rows)['cell_protection_pass']
    assert not decision(rows)['logloss_improved']
    assert decision(rows[:-1])['status']=='INCOMPLETE'
    with pytest.raises(ValueError): decision(rows+[rows[0]])
    for row in rows: row['delta_d0t0_auc']=-.00101
    assert not decision(rows)['cell_protection_pass']
def test_epoch_fairness_and_independent_gpu_gate(monkeypatch):
    jobs=[dict(job_id='a',lane='A',seed=2022,dataset='kuairand',arm='HYPERGATE'),
          dict(job_id='b',lane='B',seed=2022,dataset='kuairand',arm='C_MIX_P01_SERIAL_V1')]
    states={j['job_id']:dict(status='READY',commit_index=0) for j in jobs}
    assert queue.choose_job(jobs,states,'A',set())['job_id']=='a'
    assert queue.choose_job(jobs,states,'B',set())['job_id']=='b'
    monkeypatch.setattr(queue,'available_ram',lambda:(32<<30,64<<30))
    base=dict(memory_used_mib=1000,memory_total_mib=24576,power_draw_w=200,power_limit_w=350)
    assert queue.resources_allow(base,1<<30)
    assert not queue.resources_allow(dict(base,power_draw_w=300),1<<30)

@pytest.mark.skipif(torch.cuda.device_count()<2,reason='Two CUDA devices required')
def test_cuda_rng_maps_active_stream_to_new_device():
    from experiments.overnight_ab_20261103.state import capture_rng,restore_rng
    torch.cuda.set_device(0); torch.cuda.manual_seed(2022)
    saved=capture_rng('cuda:0')
    expected=torch.rand(16,device='cuda:0').cpu()
    torch.cuda.set_device(1); restore_rng(saved,'cuda:1')
    actual=torch.rand(16,device='cuda:1').cpu()
    assert torch.equal(expected,actual)
def test_guardian_normal_quanta_not_retries(tmp_path):
    guardian.ensure_guardian(tmp_path); key='fixture'
    job=dict(job_id=key,trajectory_id='t',arm='HYPERGATE',config_hash='c',source_hash='s',data_hash='d')
    try:
        for commit in range(1,4):
            registered=guardian.request(tmp_path,'register',key=key,job=job,pid=os.getpid())
            assert registered['launch_index']==commit
            snapshot=dict(commit_index=commit,completed_search=0,completed_final=commit,weight=torch.tensor([commit]))
            guardian.request(tmp_path,'commit',key=key,state=snapshot,identity=job,epoch_record_sha=str(commit))
            guardian.request(tmp_path,'yield',key=key)
        row=guardian.request(tmp_path,'status')['jobs'][key]
        assert row['failure_count']==0 and row['launch_index']==3
        assert torch.equal(guardian.request(tmp_path,'get',key=key)['state']['weight'],torch.tensor([3]))
        assert not list(tmp_path.rglob('*.pt')) and not list(tmp_path.rglob('*.pkl'))
        write_json(tmp_path/'runs'/key/'DONE.json',{})
        guardian.request(tmp_path,'release',key=key)
    finally:
        guardian.request(tmp_path,'shutdown')

