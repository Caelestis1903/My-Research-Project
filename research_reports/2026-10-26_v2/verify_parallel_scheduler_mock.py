"""Operational mocks only. Does not connect to or modify the resident service."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from experiments.mdmt_rcf_20261026 import campaign as original
from experiments.mdmt_rcf_20261026.phase_scheduler import parallel_phase,_admission,_validate_authorization,install_on_reader_import,function_fingerprint,SERVICE_PID,CONFIG_HASH,WORKSPACE,ARTIFACT_ROOT,TARGET

KEYS=[('movielens',5201),('movielens',5202)]
class Process:
    def __init__(self,pid):self.pid=pid
    def is_alive(self):return True
class Connection:
    def __init__(self,owner,key):self.owner=owner;self.key=key
    def poll(self):
        if self.owner.workers[self.key]['pending'] in ('s1','s2','s3'):
            self.owner.polls[self.key]+=1
            return self.owner.polls[self.key]> (2 if self.key==KEYS[0] else 0)
        return True
class Fake:
    def __init__(self,keys=KEYS,fail=None):
        self.config={'seeds':{'development':[5201,5202,5203,5204]},'recovery':{'host_reserve_gib':8,'prefix_working_set_gib':{'movielens':4,'kuairand':6}}}
        self.prefix_peak_bytes={};self.events=[];self.results={};self.status=[];self.polls={k:0 for k in keys};self.fail=fail
        self.workers={k:{'process':Process(i+100),'device':'cuda:0','pending':None} for i,k in enumerate(keys)}
        for key in keys:self.workers[key]['conn']=Connection(self,key)
        self.decisions={'s2':{'datasets':{'movielens':{'triggered_methods':['K1_WIDE']},'kuairand':{'triggered_methods':[]}}}}
    def require_review(self):self.events.append(('review',))
    def _send(self,key,op,**kw):
        assert self.workers[key]['pending'] is None
        self.events.append(('send',key,op,kw));self.workers[key]['pending']=op
    def _recv(self,key):
        op=self.workers[key]['pending'];self.events.append(('recv',key,op));self.workers[key]['pending']=None
        if op=='release-data':return {'status':'RELEASED'}
        if key==self.fail:raise RuntimeError('synthetic worker failure')
        return {'identity':{'dataset':key[0],'seed':key[1]},'metrics':{'synthetic':True}}
    def _remember(self,key,result):self.results[key]=result
    def write_status(self,phase,status,**kw):self.status.append((phase,status,kw))

def memory(pid=None):return {'memavailable_bytes':64*2**30,'rss_bytes':2**30}
def gpu(**kw):return [{'index':0,'memory_used':10.,'memory_total':100.,'power':10.,'power_limit':100.,'eligible':True,**kw}]

def run():
    out={}
    with patch.object(original,'host_memory_snapshot',side_effect=memory),patch.object(original,'gpu_snapshot',side_effect=gpu):
        f=Fake();results=parallel_phase(f,'s1');sends=[i for i,e in enumerate(f.events) if e[0]=='send' and e[2]=='s1'];receives=[i for i,e in enumerate(f.events) if e[0]=='recv' and e[2]=='s1']
        assert len(sends)==2 and sends[1]<receives[0]
        assert [r['identity']['seed'] for r in results]==[5201,5202]
        actual_order=[e[1][1] for e in f.events if e[0]=='recv' and e[2]=='s1'];assert actual_order==[5202,5201]
        assert set(f.results)==set(KEYS) and f.status[-1][1]=='COMPLETE'
        out['same_card_concurrency_and_out_of_order_binding']=True
        for metric in ('memory_used','power'):
            with patch.object(original,'gpu_snapshot',return_value=gpu(**{metric:85.})):
                assert not _admission(Fake(),KEYS[0],set())['gpu_eligible']
        out['either_gpu_limit_85_blocks_dispatch']=True
        with patch.object(original,'host_memory_snapshot',side_effect=lambda pid=None:{'memavailable_bytes':10*2**30,'rss_bytes':2**30}):
            assert not _admission(Fake(),KEYS[0],set())['host_eligible']
        out['host_8gib_reserve_and_pending_reservation']=True
        admission=_admission(Fake(),KEYS[1],{KEYS[0]});assert admission['active_additional_reservation_bytes']==3*2**30
        assert admission['candidate_additional_reservation_bytes']==3*2**30
        f=Fake(keys=KEYS+[('kuairand',5201)]);parallel_phase(f,'s3',methods_by_dataset={'movielens':['K1_WIDE'],'kuairand':[]})
        stage=[e for e in f.events if e[0]=='send' and e[2]=='s3'];assert len(stage)==2
        assert all(e[3]=={'methods':['K1_WIDE'],'decision':f.decisions['s2']['datasets']['movielens']} for e in stage)
        out['s3_subset_and_exact_message_contract']=True
        f=Fake(fail=KEYS[1])
        try:parallel_phase(f,'s2')
        except RuntimeError:pass
        else:raise AssertionError('Failure was hidden')
        assert f.status[-1][1]=='FAILED_DRAINED' and KEYS[0] in f.results and all(w['pending'] is None for w in f.workers.values())
        assert all(status!='COMPLETE' for _,status,_ in f.status)
        out['failure_drains_others_and_never_marks_complete']=True
    token=dict(verdict='PASS',blocking_issues=[],service_pid=SERVICE_PID,config_hash=CONFIG_HASH,workspace=str(WORKSPACE),artifact_root=str(ARTIFACT_ROOT),target=TARGET,old_phase_code_hash='old',new_phase_code_hash='new',scheduler_sha256='scheduler',readers_sha256='readers')
    _validate_authorization(token,old_hash='old',new_hash='new',scheduler_sha='scheduler',readers_sha='readers')
    for field,value in [('verdict','FAIL'),('old_phase_code_hash','wrong'),('scheduler_sha256','wrong'),('config_hash','wrong')]:
        bad=dict(token,**{field:value})
        try:_validate_authorization(bad,old_hash='old',new_hash='new',scheduler_sha='scheduler',readers_sha='readers')
        except PermissionError:pass
        else:raise AssertionError('Invalid authorization accepted')
    out['authorization_verdict_identity_and_hash_checks']=True
    assert install_on_reader_import()['status']=='NO_OP'
    out['ordinary_process_install_is_noop']=True
    reader=Path('experiments/mdmt_rcf_20261026/readers.py');new=reader.read_text();old=new[:new.index('\n# Fixed, independently authorized operational hook.')]
    auth=json.loads(Path('artifacts/mdmt_rcf_20261026/v2_four_seed/REVIEW_AUTHORIZATION.json').read_text())
    expected=auth['files'][str(reader)]
    assert hashlib.sha256(old.encode()).hexdigest()==expected
    defs=lambda source:[ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    assert defs(old)==defs(new)
    out['all_preexisting_reader_bytes_and_science_asts_unchanged']=True
    out.update(status='PASS',formal_training=False,scientific_results_generated=False,resident_service_contacted=False,model_files_written=0)
    return out
if __name__=='__main__':
    result=run();Path('research_reports/2026-10-26_v2/PARALLEL_SCHEDULER_BASIC_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
