import os
import multiprocessing as mp
import torch
from experiments.ema_causal_20261110.state import Guardian
from experiments.overnight_ab_20261103.guardian import process_identity


def _failed_worker(conn):
    # A committed epoch survives a worker exiting before the next transaction.
    conn.send(os.getpid());conn.recv();os._exit(37)


def test_atomic_epoch_and_worker_failure(tmp_path):
    parent,child=mp.get_context('spawn').Pipe();process=mp.get_context('spawn').Process(target=_failed_worker,args=(child,));process.start();pid=parent.recv()
    guard=Guardian(tmp_path);job={'job_id':'engineering'}
    guard.handle(dict(command='register',key='engineering',job=job,pid=pid))
    state={'online':torch.tensor([1.,2.]),'optimizer':{'exp_avg':torch.tensor([.1,.2])}}
    record={'epoch':1,'cost':{},'scalar':.5}
    guard.handle(dict(command='commit',key='engineering',pid=pid,state=state,packets={'E1':{'online_fp32':torch.tensor([1.,2.])}},record=record))
    parent.send('committed');process.join(10);assert process.exitcode==37
    restored=guard.handle(dict(command='register',key='engineering',job=job,pid=os.getpid()))['state']
    assert torch.equal(restored['online'],state['online'])
    assert torch.equal(restored['optimizer']['exp_avg'],state['optimizer']['exp_avg'])
    assert guard.rows['engineering']['epoch']==1
    assert not list(tmp_path.rglob('*.pt'))
