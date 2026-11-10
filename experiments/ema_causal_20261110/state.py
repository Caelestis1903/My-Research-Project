"""Incremental immutable anchor packets and epoch transactions in a RAM guardian."""
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from multiprocessing.connection import Listener
from experiments.mdmt_history_information_20261103.protocol import send_memory,receive_memory
from experiments.overnight_ab_20261103.guardian import process_identity,alive
from experiments.ema_mechanism_20261106.state import request,guardian_memory
from experiments.overnight_ab_20261103.state import read_json,write_json,append_json,tensor_bytes,digest
RAM_CAP=4<<30
DISK_CAP=2<<30

def disk_admit(out,extra=0):
    used=sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file() and 'data_cache' not in p.parts)
    if used+extra>DISK_CAP: raise MemoryError('Observation output exceeds 2 GiB')
    return used

class Guardian:
    def __init__(self,out): self.out=Path(out);self.rows={};self.blocks=set()
    def handle(self,msg):
        cmd=msg['command'];key=msg.get('key');row=self.rows.get(key)
        if cmd=='status': return dict(pid=os.getpid(),**guardian_memory(),payload_bytes=tensor_bytes(self.rows),blocks=sorted(self.blocks),jobs={k:{n:r[n] for n in ('status','epoch','launches','worker') if n in r} for k,r in self.rows.items()})
        if cmd=='get': return dict(row=row)
        if cmd=='register':
            if row is None:
                replay=self.out/'REPLAYS.jsonl';records=[__import__('json').loads(v) for v in replay.read_text().splitlines()] if replay.exists() else []
                if list((self.out/'runs'/key/'epochs').glob('E*.json')):
                    if any(r['job_id']==key and r['status']=='RAM_LOSS_REPLAY' for r in records): raise RuntimeError('Replay budget exhausted: '+key)
                    append_json(replay,dict(job_id=key,status='RAM_LOSS_REPLAY',time=time.time()))
                row=dict(job=msg['job'],status='READY',epoch=0,launches=0,worker=None,state=None,packets={});self.rows[key]=row
            if row['status'] in ('COMPLETE','FINALIZED'): return dict(complete=True)
            if alive(row['worker']): raise RuntimeError('Worker already active')
            if row['launches']>=2: raise RuntimeError('Worker restart budget exhausted')
            row.update(worker=process_identity(msg['pid']),launches=row['launches']+1,status='RUNNING')
            return dict(state=row['state'])
        if cmd=='commit':
            if row is None or row['worker']!=process_identity(msg['pid']): raise ValueError('Epoch owner mismatch')
            state=msg['state'];record=msg['record'];epoch=record['epoch']
            if epoch!=row['epoch']+1: raise ValueError('Nonconsecutive epoch')
            newpackets={**row['packets'],**msg['packets']}
            size=tensor_bytes(self.rows)-tensor_bytes(row['state'])-tensor_bytes(row['packets'])+tensor_bytes(state)+tensor_bytes(newpackets)
            # Account for the commit's old/new payload coexistence.
            worker_payload=sum(tensor_bytes(r.get('state')) for r in self.rows.values() if r.get('status')=='RUNNING')
            if size+tensor_bytes(row['state'])+worker_payload>RAM_CAP: raise MemoryError('RAM payload commit cap')
            path=self.out/'runs'/key/'epochs'/f'E{epoch:03d}.json'
            science={k:v for k,v in record.items() if k!='cost'}
            if path.exists():
                old=read_json(path)
                if {k:v for k,v in old.items() if k!='cost'}!=science: raise ValueError('Replay scientific record mismatch')
                append_json(self.out/'REPLAYS.jsonl',dict(job_id=key,epoch=epoch,status='REPLAY_VERIFIED',cost=record['cost']))
            else:
                disk_admit(self.out,len(__import__('json').dumps(record).encode())*2);write_json(path,record)
            row.update(epoch=epoch,state=state,packets=newpackets)
            append_json(self.out/'epoch_commit_ledger.jsonl',dict(job_id=key,epoch=epoch,record_hash=digest(science),time=time.time()))
            return dict(status='COMMITTED',payload_bytes=size)
        if cmd=='complete':
            if row['epoch']!=50: raise ValueError('Training incomplete')
            keep=('job','selected','selected_packet','source_packets','matched_done','cost','init_hash','global_step')
            row['state']={k:row['state'][k] for k in keep};row.update(status='COMPLETE',worker=None)
            return dict(status='COMPLETE')
        if cmd=='failed': row.update(status='FAILED',worker=None,error=msg['error']);return dict(status='FAILED')
        if cmd=='finalize':
            keys=msg['keys']
            if len(keys)!=5 or any(self.rows[k]['status']!='COMPLETE' for k in keys): raise ValueError('Five complete live seeds required')
            for k in keys:
                r=self.rows[k];r['state']={n:r['state'][n] for n in ('job','selected','cost','global_step')};r['packets']={};r['status']='FINALIZED'
            self.blocks.add(msg['block']);return dict(status='FINALIZED')
        if cmd=='release':
            if len(self.blocks)!=4 or not (self.out/'FINAL_REPORT.md').exists() or not (self.out/'ENDPOINTS.csv').exists(): raise RuntimeError('Report/Test incomplete')
            amount=tensor_bytes(self.rows);self.rows.clear();write_json(self.out/'RELEASE_RECEIPT.json',dict(status='RELEASED',released_bytes=amount,time=time.time()));return dict(stop=True,status='RELEASED')
        raise ValueError('Unknown guardian command '+cmd)

def ensure_guardian(out):
    out=Path(out).resolve();info=out/'guardian.json'
    if info.exists() and alive(read_json(info).get('owner')): return request(out,'status')
    with (out/'guardian.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-m','experiments.ema_causal_20261110.run','_guardian','--out',str(out)],cwd=Path(__file__).resolve().parents[2],stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
    for _ in range(600):
        if info.exists() and read_json(info).get('pid')==child.pid: return request(out,'status')
        if child.poll() is not None: break
        time.sleep(.1)
    raise RuntimeError('Guardian startup failed')

def serve(out):
    out=Path(out);guard=Guardian(out);auth=secrets.token_bytes(32);address=f'/tmp/ema-causal-{os.getuid()}-{secrets.token_hex(6)}.sock'
    try:
        with Listener(address,family='AF_UNIX',authkey=auth) as listener:
            write_json(out/'guardian.json',dict(pid=os.getpid(),owner=process_identity(os.getpid()),address=address,authkey=auth.hex()));os.chmod(out/'guardian.json',0o600)
            stop=False
            while not stop:
                with listener.accept() as conn:
                    try: result=guard.handle(receive_memory(conn));stop=bool(result.pop('stop',False))
                    except Exception as e: result={'error':f'{type(e).__name__}: {e}'}
                    send_memory(conn,result)
    finally: Path(address).unlink(missing_ok=True)
