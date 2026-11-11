"""Incremental immutable objects and recoverable epoch commits, held only in RAM."""
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from multiprocessing.connection import Listener
import numpy as np
import torch
from experiments.mdmt_history_information_20261103.protocol import send_memory,receive_memory
from experiments.overnight_ab_20261103.guardian import process_identity,alive
from experiments.ema_mechanism_20261106.state import request,guardian_memory
from experiments.overnight_ab_20261103.state import read_json,write_json,append_json,digest,state_digest

RAM_CAP=4<<30
PHYSICAL_CAP=7<<30
DISK_CAP=2<<30


def unique_bytes(value,seen=None):
    seen=set() if seen is None else seen
    if torch.is_tensor(value):
        storage=value.untyped_storage();key=('torch',str(value.device),storage.data_ptr(),storage.nbytes())
        if key in seen:return 0
        seen.add(key);return storage.nbytes()
    if isinstance(value,np.ndarray):
        root=value
        while isinstance(root.base,np.ndarray):root=root.base
        key=('numpy',id(root))
        if key in seen:return 0
        seen.add(key);return root.nbytes
    if isinstance(value,dict):return sum(unique_bytes(v,seen) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(unique_bytes(v,seen) for v in value)
    return 0


def closure(refs,objects):
    keep={};pending=list(refs)
    while pending:
        key=pending.pop()
        if key in keep:continue
        if key not in objects:raise ValueError('Missing immutable RAM object '+key)
        keep[key]=objects[key];pending.extend(objects[key].get('refs',[]))
    return keep


def selected_refs(state):
    refs=[]
    for name in ('selected','prefix'):
        refs.extend(v['packet_id'] for v in state.get(name,{}).values())
    refs.extend(state.get('anchors',{}).values())
    return list(dict.fromkeys(refs))


def disk_admit(out,extra=0):
    used=sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file() and 'data_cache' not in p.parts)
    if used+extra>DISK_CAP:raise MemoryError('Observation output exceeds 2 GiB')
    return used


def scientific_record(record):return {k:v for k,v in record.items() if k!='cost'}


class Guardian:
    def __init__(self,out):
        self.out=Path(out);self.rows={};self.objects={};self.selections={};self.teachers={}

    def collect(self):
        refs=[r for row in self.rows.values() for r in row.get('state',{}).get('object_refs',[])]
        self.objects=closure(refs,self.objects)

    def payload(self):return unique_bytes((self.objects,self.teachers,[r.get('state') for r in self.rows.values()]))

    def handle(self,msg):
        cmd=msg['command'];key=msg.get('key');row=self.rows.get(key)
        if cmd=='status':
            return dict(pid=os.getpid(),**guardian_memory(),payload_bytes=self.payload(),object_count=len(self.objects),
                jobs={k:{n:r[n] for n in ('status','epoch','launches','worker','job') if n in r} for k,r in self.rows.items()})
        if cmd=='teacher_get':return dict(target=self.teachers.get(key))
        if cmd=='teacher_put':
            target=msg['target'];values=target['probabilities']
            if not isinstance(values,np.ndarray) or values.dtype!=np.float32 or values.ndim!=2 or values.shape[1]!=2:raise ValueError('Teacher target schema')
            if self.payload()+unique_bytes(target)>RAM_CAP:raise MemoryError('Teacher RAM budget')
            self.teachers.setdefault(key,target);return dict(status='CACHED')
        if cmd=='teacher_clear':
            self.teachers.clear();return dict(status='RELEASED')
        if cmd=='get':
            if row is None:return dict(row=None,objects={})
            return dict(row=row,objects=closure(row.get('state',{}).get('object_refs',[]),self.objects))
        if cmd=='get_object':return dict(object=self.objects[msg['object_id']],objects=closure([msg['object_id']],self.objects))
        if cmd=='summaries':return dict(summaries={k:dict(job=r['job'],status=r['status'],epoch=r['epoch'],selected=r.get('state',{}).get('selected',{})) for k,r in self.rows.items()})
        if cmd=='register':
            if row is None:
                old=list((self.out/'runs'/key/'epochs').glob('E*.json'))
                if old:
                    entries=[__import__('json').loads(v) for v in (self.out/'REPLAYS.jsonl').read_text().splitlines()] if (self.out/'REPLAYS.jsonl').exists() else []
                    if any(v.get('job_id')==key and v.get('status')=='RAM_LOSS_REPLAY' for v in entries):raise RuntimeError('Full replay budget exhausted '+key)
                    append_json(self.out/'REPLAYS.jsonl',dict(job_id=key,status='RAM_LOSS_REPLAY',time=time.time()))
                row=dict(job=msg['job'],status='PENDING',epoch=msg['job']['start_epoch'],launches=0,worker=None,state={});self.rows[key]=row
            if row['status']=='COMPLETE':return dict(complete=True)
            if alive(row.get('worker')):raise RuntimeError('Worker already active '+key)
            if row['launches']>=2:raise RuntimeError('Worker restart budget exhausted '+key)
            row.update(worker=process_identity(msg['pid']),launches=row['launches']+1,status='RUNNING')
            return dict(state=row['state'],objects=closure(row['state'].get('object_refs',[]),self.objects))
        if cmd=='commit':
            if row is None or row.get('worker')!=process_identity(msg['pid']):raise ValueError('Epoch owner mismatch')
            record=msg['record'];epoch=record['epoch'];state=msg['state'];incoming=msg.get('new_objects',{})
            if epoch!=row['epoch']+1:raise ValueError('Nonconsecutive epoch')
            objects=dict(self.objects)
            for ident,obj in incoming.items():
                if ident in objects and state_digest(objects[ident])!=state_digest(obj):raise ValueError('Immutable object collision')
                objects[ident]=obj
            closure(state.get('object_refs',[]),objects)
            live=[r.get('state') for r in self.rows.values()]
            # The old row, incoming live state and worker-live copies coexist at commit.
            total=unique_bytes((objects,self.teachers,live,state))
            worker=sum(unique_bytes(r.get('state')) for r in self.rows.values() if r.get('status')=='RUNNING')
            if total+worker>RAM_CAP:raise MemoryError('Guardian commit predicate exceeds 4 GiB')
            if total+worker+unique_bytes(state)>PHYSICAL_CAP:raise MemoryError('Physical tensor payload exceeds 7 GiB')
            path=self.out/'runs'/key/'epochs'/f'E{epoch:03d}.json';old=read_json(path) if path.exists() else None
            science=scientific_record(record)
            if old is not None and scientific_record(old)!=science:raise ValueError('Replay scientific record mismatch '+key)
            disk_admit(self.out,len(__import__('json').dumps(record).encode())*2)
            old_state=row['state'];old_objects=self.objects;old_epoch=row['epoch']
            try:
                self.objects=objects;row.update(state=state,epoch=epoch)
                if old is None:write_json(path,record)
                else:append_json(self.out/'REPLAYS.jsonl',dict(job_id=key,epoch=epoch,status='REPLAY_VERIFIED',cost=record.get('cost',{})))
                receipt=dict(job_id=key,epoch=epoch,record_hash=digest(science),status='COMMITTED',time=time.time())
                write_json(self.out/'runs'/key/'commits'/f'E{epoch:03d}.json',receipt)
                append_json(self.out/'epoch_commit_ledger.jsonl',receipt)
            except Exception:
                row.update(state=old_state,epoch=old_epoch);self.objects=old_objects
                if old is None:path.unlink(missing_ok=True)
                raise
            self.collect();return dict(status='COMMITTED',payload_bytes=self.payload(),object_count=len(self.objects))
        if cmd=='complete':
            if row['epoch']!=row['job']['end_epoch']:raise ValueError('Incomplete training budget')
            if not msg.get('keep_live',False):
                state=row['state'];keep=('selected','prefix','anchors','job','epoch','global_step','cost') if row['job']['family']=='PARENT' else ('selected','job','epoch','global_step','cost');row['state']={k:state[k] for k in keep if k in state}
                row['state']['object_refs']=selected_refs(row['state'])
            row.update(status='COMPLETE',worker=None);self.collect();return dict(status='COMPLETE')
        if cmd=='prune_anchors':
            from .trajectory import _roots,_trim_history
            frozen=msg.get('frozen_B');keep={'O','F'}|({frozen['id']} if frozen else set())
            state=row['state'];state['prefix']={k:v for k,v in state.get('prefix',{}).items() if k in keep}
            anchors={}
            for epoch,ident in state.get('anchors',{}).items():
                original=self.objects[ident];snap=dict(original['snapshot'])
                for name in ('selected','prefix'):snap[name]={k:v for k,v in snap.get(name,{}).items() if k in keep}
                snap['history']=_trim_history(snap['history'],[frozen['H']] if frozen else [],int(epoch),row['job']['config']['steps_per_epoch'],row['job']['config']['rho']).state_dict()
                snap['object_refs']=_roots(snap);obj=dict(kind='anchor',snapshot=snap,refs=snap['object_refs']);new=state_digest(obj)
                self.objects[new]=obj;anchors[epoch]=new
            state['anchors']=anchors;state['object_refs']=selected_refs(state);self.collect()
            return dict(status='PRUNED',payload_bytes=self.payload())
        if cmd=='retain':
            if row is None:raise ValueError('Unknown row')
            state=row['state'];keep=msg['readouts'];state['selected']={k:v for k,v in state.get('selected',{}).items() if k in keep}
            if not msg.get('anchors',False):state['anchors']={};state['prefix']={}
            state['object_refs']=selected_refs(state)
            row['state']=state;self.collect();return dict(status='RETAINED',payload_bytes=self.payload())
        if cmd=='strip_teacher_refs':
            for value in row['state'].get('selected',{}).values():
                packet=self.objects[value['packet_id']]
                if packet.get('parent_F_ref'):
                    packet={k:v for k,v in packet.items() if k!='parent_F_ref'}
                    packet['refs']=[packet['base_ref']];ident=state_digest(packet);self.objects[ident]=packet;value['packet_id']=ident
            row['state']['object_refs']=selected_refs(row['state']);self.collect()
            return dict(status='RELEASED_TEACHER_INITIALIZERS',payload_bytes=self.payload())
        if cmd=='drop':
            if row and alive(row.get('worker')):raise RuntimeError('Cannot drop live worker')
            if row:row['state']={};row['status']='RELEASED'
            self.collect();return dict(status='RELEASED')
        if cmd=='failed':
            row.update(status='FAILED',worker=None,error=msg['error']);return dict(status='FAILED')
        if cmd=='put_immutable':
            ident=msg['object_id'];self.objects[ident]=msg['object'];return dict(object_id=ident)
        if cmd=='release':
            needed=('GLOBAL_SELECTION_FREEZE.json','TEST_COMPLETE.json','FINAL_REPORT.md','ARTIFACT_INDEX.json')
            if not all((self.out/n).exists() for n in needed):raise RuntimeError('Frozen Test/report consumers incomplete')
            amount=self.payload();self.rows.clear();self.objects.clear();self.teachers.clear()
            write_json(self.out/'RELEASE_RECEIPT.json',dict(status='RELEASED',released_tensor_bytes=amount,time=time.time()))
            return dict(stop=True,status='RELEASED')
        raise ValueError('Unknown guardian command '+cmd)


def ensure_guardian(out):
    out=Path(out).resolve();info=out/'guardian.json'
    if info.exists() and alive(read_json(info).get('owner')):return request(out,'status')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'guardian.log').open('ab') as log:
        p=subprocess.Popen([sys.executable,'-m','experiments.ema_next_20261111.run','_guardian','--out',str(out)],
            cwd=Path(__file__).resolve().parents[2],stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
    for _ in range(600):
        if info.exists() and read_json(info).get('pid')==p.pid:return request(out,'status')
        if p.poll() is not None:break
        time.sleep(.1)
    raise RuntimeError('Guardian startup failed')


def serve(out):
    out=Path(out);guard=Guardian(out);auth=secrets.token_bytes(32);address=f'/tmp/ema-next-{os.getuid()}-{secrets.token_hex(6)}.sock'
    try:
        with Listener(address,family='AF_UNIX',authkey=auth) as listener:
            write_json(out/'guardian.json',dict(pid=os.getpid(),owner=process_identity(os.getpid()),address=address,authkey=auth.hex()));os.chmod(out/'guardian.json',0o600)
            stop=False
            while not stop:
                with listener.accept() as conn:
                    try:result=guard.handle(receive_memory(conn));stop=bool(result.pop('stop',False))
                    except Exception as e:result=dict(error=f'{type(e).__name__}: {e}')
                    send_memory(conn,result)
    finally:Path(address).unlink(missing_ok=True)


def put_immutable(out,object_id,obj):return request(out,'put_immutable',object_id=object_id,object=obj)
def commit_epoch(out,job_id,live_state,new_objects,object_refs,record):
    live_state['object_refs']=object_refs
    return request(out,'commit',key=job_id,pid=os.getpid(),state=live_state,new_objects=new_objects,record=record)
def fork_snapshot(out,job_id):return request(out,'get',key=job_id)
def recover(out,job_id):return fork_snapshot(out,job_id)
def release_consumed(out,job_id,readouts,anchors=False):return request(out,'retain',key=job_id,readouts=readouts,anchors=anchors)
