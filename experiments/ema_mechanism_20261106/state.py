"""RAM guardian and atomic scalar/prediction transactions; no model files."""
import copy
import hashlib
import io
import os
import resource
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from multiprocessing.connection import Listener
import numpy as np
from experiments.mdmt_history_information_20261103.protocol import send_memory,receive_memory
from experiments.overnight_ab_20261103.guardian import request as source_request,process_identity,alive
from experiments.overnight_ab_20261103.state import read_json,write_json,append_json,digest,tensor_bytes,cpu_copy

def request(out,command,**payload):
    try: return source_request(out,command,**payload)
    except RuntimeError as exc:
        if 'MemoryError:' in str(exc): raise MemoryError(str(exc)) from exc
        if 'FloatingPointError:' in str(exc): raise FloatingPointError(str(exc)) from exc
        if 'ValueError:' in str(exc): raise ValueError(str(exc)) from exc
        raise


PREDICTION_CAP=2<<30
RAM_CAP=4<<30


def guardian_memory():
    rss=next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
    return dict(guardian_rss_bytes=int(rss)*1024,
        peak_guardian_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def array_digest(arrays):
    h=hashlib.sha256()
    for key,value in sorted(arrays.items()):
        value=np.ascontiguousarray(value);h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


class PredictionWriter:
    def __init__(self,out,cap=PREDICTION_CAP):
        self.out=Path(out);self.cap=cap
        self.bytes=sum(f.stat().st_size for f in self.out.rglob('*') if f.is_file())
    def check(self,size):
        if self.bytes+size>self.cap: raise MemoryError('RESOURCE_BLOCKED: observation disk cap')
    def write(self,job,epoch,prediction,panel):
        arrays=prediction['arrays'];is_v=prediction['panel_id']=='V_mech'
        expected={'p':(3,4096,2),'z':(3,4096,2)} if is_v else {'rank_p':(6,512),'rank_z':(6,512),'natural_p':(3,512,2),'natural_z':(3,512,2)}
        if not arrays or not set(arrays)<=set(expected): raise ValueError('Forbidden prediction field')
        for key,value in arrays.items():
            if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or value.shape!=expected[key]: raise ValueError('Forbidden prediction dtype/shape')
            if not np.isfinite(value).all(): raise FloatingPointError('Nonfinite prediction')
        observation_hash=array_digest(arrays)
        relative=Path('predictions')/job/f'E{epoch:03d}'/(prediction['readout_id']+'__'+prediction['panel_id']+'.npz')
        path=self.out/relative;created=False
        if path.exists():
            with np.load(path,allow_pickle=False) as old:
                if array_digest({k:old[k] for k in old.files})!=observation_hash: raise ValueError('Prediction replay conflict')
        else:
            stream=io.BytesIO();np.savez_compressed(stream,**arrays);payload=stream.getvalue();self.check(len(payload))
            path.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
                temporary=tmp.name;tmp.write(payload);tmp.flush();os.fsync(tmp.fileno())
            os.replace(temporary,path);self.bytes+=len(payload);created=True
        receipt={k:v for k,v in prediction.items() if k!='arrays'}
        from .protocol import sha_file
        receipt.update(job_id=job,epoch=epoch,path=str(relative),sha256=sha_file(path),observation_hash=observation_hash,
            panel_hash=panel['panel_hash'],pair_hash=panel['pair_hash'],arrays=list(arrays))
        return receipt,path if created else None


def epoch_path(out,key,epoch): return Path(out)/'runs'/key/'epochs'/f'E{epoch:03d}.json'


def summary(snapshot):
    keys=('job','epoch','global_step','complete','init_hash','layout_hash','group_hash','cost','commit_index')
    result={k:snapshot[k] for k in keys}
    result['snapshots']={kind:{mark:{k:v for k,v in value.items() if k!='model_state'} for mark,value in entries.items()} for kind,entries in snapshot['snapshots'].items()}
    return result


def scientific_record(record):
    return {k:v for k,v in record.items() if k not in ('cost','predictions')}


def commit_epoch(out,key,row,snapshot,record,predictions,writer,panels,all_rows=None):
    index=snapshot['commit_index']
    if snapshot['job']!=row['job'] or record['commit_index']!=index: raise ValueError('Epoch identity changed')
    record_hash=digest(scientific_record(record))
    if index==row['commit_index'] and row.get('record_hash')==record_hash: return dict(status='ALREADY_COMMITTED',commit_index=index)
    if index!=row['commit_index']+1: raise ValueError('Nonconsecutive epoch commit')
    total=sum(tensor_bytes(r.get('state')) for r in (all_rows or {}).values())-tensor_bytes(row.get('state'))+tensor_bytes(snapshot)
    if total>RAM_CAP: raise MemoryError('RESOURCE_BLOCKED: guardian model RAM cap')
    path=epoch_path(out,key,record['epoch']);old_record=read_json(path) if path.exists() else None
    if old_record is not None and scientific_record(old_record)!=scientific_record(record): raise ValueError('Deterministic replay epoch conflict')
    previous=row.copy();created=[];previous_bytes=writer.bytes
    peak=guardian_memory()['peak_guardian_ram_bytes']
    record['cost']['peak_guardian_ram_bytes']=peak
    snapshot['cost']['peak_guardian_ram_bytes']=max(snapshot['cost'].get('peak_guardian_ram_bytes',0),peak)
    row.update(state=snapshot,commit_index=index,record_hash=record_hash,summary=summary(snapshot))
    try:
        receipts=[]
        for prediction in predictions:
            receipt,new=writer.write(key,index,prediction,panels[prediction['panel_id']]);receipts.append(receipt)
            if new: created.append(new)
        record=dict(record,predictions=receipts);writer.check(len(__import__('json').dumps(record).encode())*2)
        if old_record is None:
            write_json(path,record);created.append(path);writer.bytes+=path.stat().st_size
            status='COMMITTED'
        else:
            append_json(Path(out)/'REPLAYS.jsonl',dict(job_id=key,epoch=index,status='REPLAY_VERIFIED',physical_cost=record['cost'],time=time.time()))
            status='REPLAY_VERIFIED'
        append_json(Path(out)/'epoch_commit_ledger.jsonl',dict(job_id=key,epoch=index,commit_index=index,record_hash=record_hash,status=status,time=time.time()))
    except Exception:
        row.clear();row.update(previous)
        for file in created: file.unlink(missing_ok=True)
        writer.bytes=previous_bytes
        raise
    return dict(status=status,commit_index=index,prediction_bytes=writer.bytes)


class MechanismGuardian:
    def __init__(self,out):
        self.out=Path(out);self.rows={};self.writer=PredictionWriter(out)
        self.prepared=read_json(self.out/'PREPARED.json')
    def handle(self,msg):
        command=msg['command'];key=msg.get('key');row=self.rows.get(key)
        if command=='status':
            return dict(pid=os.getpid(),**guardian_memory(),snapshot_bytes=sum(tensor_bytes(r.get('state')) for r in self.rows.values()),
                jobs={k:{n:v for n,v in r.items() if n not in ('state','summary')} for k,r in self.rows.items()})
        if command=='summaries': return dict(summaries={k:r['summary'] for k,r in self.rows.items() if r.get('summary')})
        if command=='get': return dict(state=None if row is None else row.get('state'))
        if command=='register':
            if row is None:
                existing=list((self.out/'runs'/key/'epochs').glob('E*.json'))
                if existing:
                    replay_path=self.out/'REPLAYS.jsonl'
                    logs=[__import__('json').loads(line) for line in replay_path.read_text().splitlines()] if replay_path.exists() else []
                    if any(r.get('job_id')==key and r.get('status')=='RAM_LOSS_REPLAY' for r in logs): raise RuntimeError('One replay budget exhausted')
                    append_json(replay_path,dict(job_id=key,status='RAM_LOSS_REPLAY',time=time.time()))
                row=dict(job=msg['job'],commit_index=0,status='READY',launches=0);self.rows[key]=row
            if row['job']!=msg['job'] or alive(row.get('worker')): raise ValueError('Job changed or worker active')
            if row['status']=='COMPLETE': return dict(state=row['state'],complete=True)
            if row['launches']>=2: raise RuntimeError('One worker restart budget exhausted')
            row.update(worker=process_identity(msg['pid']),status='RUNNING',launches=row['launches']+1)
            return dict(state=row.get('state'))
        if command=='commit':
            if row is None or process_identity(msg['pid'])!=row.get('worker'): raise ValueError('Only owning worker commits')
            meta=self.prepared['panels'][row['job']['dataset']][str(row['job']['seed'])]
            return commit_epoch(self.out,key,row,msg['state'],msg['record'],msg['predictions'],self.writer,meta,self.rows)
        if command=='complete':
            if row is None or not row.get('state',{}).get('complete'): raise ValueError('Training incomplete')
            row.update(status='COMPLETE',worker=None);return dict(status='COMPLETE')
        if command=='failed':
            if row is None or process_identity(msg['pid'])!=row.get('worker'): raise ValueError('Only owning worker reports failure')
            row.update(status='ENGINEERING_BLOCKED',worker=None,error=msg['error']);return dict(status=row['status'])
        if command=='release':
            if not (self.out/'FINAL_REPORT.md').exists() or not (self.out/'ENDPOINTS.csv').exists(): raise ValueError('Report/Test incomplete')
            released=sum(tensor_bytes(r.get('state')) for r in self.rows.values());self.rows.clear()
            write_json(self.out/'RELEASE_RECEIPT.json',dict(status='RELEASED',released_parameter_bytes=released,time=time.time()))
            return dict(status='RELEASED',stop=True)
        raise ValueError('Unknown guardian command')


def ensure_guardian(out):
    out=Path(out).resolve();info=out/'guardian.json'
    if info.exists() and alive(read_json(info).get('owner')): return request(out,'status')
    with (out/'guardian.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-m','experiments.ema_mechanism_20261106.run','_guardian','--out',str(out)],cwd=Path(__file__).resolve().parents[2],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    for _ in range(600):
        if info.exists() and read_json(info).get('pid')==child.pid: return request(out,'status')
        if child.poll() is not None: break
        time.sleep(.1)
    raise RuntimeError('Guardian startup failed; see guardian.log')


def serve(out):
    out=Path(out);guard=MechanismGuardian(out);auth=secrets.token_bytes(32)
    address=f'/tmp/ema-mechanism-{os.getuid()}-{secrets.token_hex(6)}.sock'
    try:
        with Listener(address,family='AF_UNIX',authkey=auth) as listener:
            write_json(out/'guardian.json',dict(pid=os.getpid(),owner=process_identity(os.getpid()),address=address,authkey=auth.hex()))
            os.chmod(out/'guardian.json',0o600)
            stop=False
            while not stop:
                with listener.accept() as conn:
                    try:
                        result=guard.handle(receive_memory(conn));stop=bool(result.pop('stop',False))
                    except Exception as exc: result=dict(error=f'{type(exc).__name__}: {exc}')
                    send_memory(conn,result)
    finally:
        Path(address).unlink(missing_ok=True)
