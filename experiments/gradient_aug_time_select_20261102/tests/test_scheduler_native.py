"""CPU event injection through the actual schedule; no processes or training."""
from pathlib import Path
import pandas as pd
import pytest
from experiments.gradient_aug_time_select_20261102 import run
from experiments.gradient_aug_time_select_20261102.protocol import protocol
from experiments.gradient_aug_time_select_20261102.recovery import scientific_job_key


class Channel:
    def __init__(self):self.queue=[];self.eof=False
    def poll(self):return bool(self.queue) or self.eof
    def receive(self):
        if self.queue:return self.queue.pop(0)
        raise EOFError
    def close(self):pass


class EventContext:
    def __init__(self,outcomes):self.outcomes=outcomes;self.tick=0;self.events=[];self.processes=[]
    def Pipe(self,duplex=False):
        channel=Channel();return channel,channel
    def Process(self,target,args):
        context=self
        class Process:
            exitcode=None
            def __init__(self):
                self.channel,self.bundles,self.p,self.job,self.device,self.out,self.attempt=args
                self.finished=False;self.joined=False;context.processes.append(self)
            def start(self):
                self.mode,self.delay,self.code=context.outcomes(self.job,self.attempt)
                self.end=context.tick+self.delay
                self.channel.queue.append(dict(kind='allocated',peak_memory_bytes=1024**3))
                context.events.append(dict(event='start',seed=self.job['seed'],attempt=self.attempt,tick=context.tick))
            def is_alive(self):return not self.finished
            def join(self):
                assert self.finished
                self.exitcode=self.code;self.joined=True
                context.events.append(dict(event='join',seed=self.job['seed'],attempt=self.attempt,tick=context.tick,exitcode=self.exitcode))
        return Process()
    def advance(self,seconds):
        self.tick+=1
        for process in self.processes:
            if process.finished or self.tick<process.end:continue
            process.finished=True;job=process.job
            self.events.append(dict(event='finish',seed=job['seed'],attempt=process.attempt,tick=self.tick,mode=process.mode))
            if process.mode=='success':
                process.channel.queue.append(dict(kind='complete',result=dict(dataset=job['dataset'],trial=job['trial'],
                    seed=job['seed'],epochs=50,global_step=1000,selected_epochs={},output_dir=str(process.out))))
            elif process.mode=='python_error':
                process.channel.queue.append(dict(kind='error',error='AssertionError: injected identity failure',oom=False,recoverable=False))
            elif process.mode=='eof':process.channel.eof=True


def exercise(monkeypatch,out,outcomes,seeds=(2022,),prior_failures=0,deadline=None):
    out=Path(out);out.mkdir(parents=True);p=protocol();p.update(source_manifest_hash='cpu-injection',execution_commit='cpu-injection')
    jobs=run.make_jobs('development','B2',['movielens'],seeds)
    if prior_failures:p['recovery']=dict(native_failure_history=dict(failures_by_job={scientific_job_key(jobs[0],p):prior_failures}))
    context=EventContext(outcomes);origin=1793549242.1426222
    if deadline is not None:p['dispatch_deadline']=origin+deadline
    with monkeypatch.context() as patch:
        patch.setattr(run.mp,'get_context',lambda method:context)
        patch.setattr(run,'receive_memory',lambda channel:channel.receive())
        patch.setattr(run,'available_ram',lambda:100*1024**3)
        patch.setattr(run,'gpu_status',lambda:[dict(index=0,memory=1,total=24576,power=1,limit=350)])
        patch.setattr(run.time,'sleep',context.advance);patch.setattr(run.time,'time',lambda:origin+context.tick)
        result=[];error=None
        try:result=run.schedule({},p,out,jobs,[0])
        except RuntimeError as exc:error=str(exc)
    return dict(events=context.events,result=result,error=error,protocol=p,out=out,
        failures=pd.read_csv(out/'failure_ledger.csv').to_dict('records') if (out/'failure_ledger.csv').exists() else [])


def test_native_retry_serial_exclusion_and_unchanged_identity(monkeypatch,tmp_path):
    def outcomes(job,attempt):
        if job['seed']==2022:return ('eof',2,-11) if attempt==1 else ('success',3,0)
        return ('success',6 if job['seed']==2023 else 1,0)
    r=exercise(monkeypatch,tmp_path/'serial',outcomes,seeds=(2022,2023,2024))
    assert r['error'] is None and len(r['result'])==3 and len(r['failures'])==1
    failure=r['failures'][0];assert failure['exitcode']==-11 and failure['recoverable'] and not failure['oom']
    starts=[e for e in r['events'] if e['event']=='start'];finishes=[e for e in r['events'] if e['event']=='finish']
    retry=next(e for e in starts if e['seed']==2022 and e['attempt']==2)
    other_finish=next(e for e in finishes if e['seed']==2023)
    retry_finish=next(e for e in finishes if e['seed']==2022 and e['attempt']==2)
    last=next(e for e in starts if e['seed']==2024)
    assert retry['tick']>=other_finish['tick'] and last['tick']>=retry_finish['tick']
    running=pd.read_csv(r['out']/'trial_ledger.csv');running=running[(running.status=='RUNNING')&(running.seed==2022)]
    assert running.attempt.tolist()==[1,2] and running.run_id.nunique()==running.config_hash.nunique()==1


@pytest.mark.parametrize('prior,expected',[(0,[1,2,3]),(1,[2,3])])
def test_native_retry_exhaustion_and_prior_allowance(monkeypatch,tmp_path,prior,expected):
    r=exercise(monkeypatch,tmp_path/f'exhaust{prior}',lambda job,attempt:('dead' if prior==0 else 'eof',2,-11),prior_failures=prior)
    assert r['error'] and [e['attempt'] for e in r['events'] if e['event']=='start']==expected
    assert len(r['failures'])==len(expected) and r['failures'][-1]['attempt']==3


def test_carried_native_failure_starts_exclusively_as_attempt_two(monkeypatch,tmp_path):
    r=exercise(monkeypatch,tmp_path/'carried',lambda job,attempt:('success',3,0),seeds=(2022,2023),prior_failures=1)
    starts=[e for e in r['events'] if e['event']=='start'];finished=next(e for e in r['events'] if e['event']=='finish' and e['seed']==2022)
    assert r['error'] is None and len(r['result'])==2 and not r['failures']
    assert starts[0]['seed']==2022 and starts[0]['attempt']==2 and starts[1]['tick']>=finished['tick'] and starts[1]['attempt']==1
    launch=pd.read_csv(r['out']/'launch_ledger.csv');assert launch.iloc[0]['serial_retry'] and launch.iloc[0]['prior_native_failures']==1


@pytest.mark.parametrize('outcome',[('eof',2,-15),('python_error',2,0)])
def test_other_native_and_assertion_do_not_retry(monkeypatch,tmp_path,outcome):
    r=exercise(monkeypatch,tmp_path/str(outcome[2]),lambda job,attempt:outcome)
    assert r['error'] and len(r['failures'])==1 and not r['failures'][0]['recoverable']
    assert len([e for e in r['events'] if e['event']=='start'])==1


def test_original_deadline_prevents_native_retry(monkeypatch,tmp_path):
    r=exercise(monkeypatch,tmp_path/'deadline',lambda job,attempt:('eof',2,-11),deadline=1)
    assert r['error'] is None and not r['result'] and r['protocol']['budget_stopped']
    assert len([e for e in r['events'] if e['event']=='start'])==1
    assert 'NOT_DISPATCHED_HARD_LIMIT' in set(pd.read_csv(r['out']/'trial_ledger.csv').status)
