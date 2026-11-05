"""A separate RAM owner commits each epoch before its scalar record is published."""
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from multiprocessing.connection import Listener
from experiments.mdmt_history_information_20261103.protocol import send_memory, receive_memory
from experiments.overnight_ab_20261103.guardian import request, process_identity, alive
from experiments.overnight_ab_20261103.state import read_json, write_json, append_json, digest, tensor_bytes
from .protocol import ROOT


def epoch_path(output,key,epoch):
    return Path(output)/'runs'/key/'epochs'/f'E{epoch:03d}.json'


def persist_epoch(output,key,record):
    path = epoch_path(output,key,record['epoch'])
    if path.exists():
        old = read_json(path)
        for field in ('job_id','epoch','commit_index','global_step','order_hash','init_hash','layout_hash',
                      'model_hash','training_loss','clean_loss'):
            if old[field] != record[field]:
                raise ValueError('Replay conflicts with committed epoch: '+field)
        append_json(Path(output)/'REPLAY_COSTS.jsonl',dict(job_id=key,epoch=record['epoch'],cost=record['cost'],time=time.time()))
        return 'REPLAY_VERIFIED'
    write_json(path,record)
    return 'SCALAR_COMMITTED'


def commit_epoch(output,key,row,snapshot,record):
    index = snapshot['commit_index']
    if snapshot['job'] != row['job'] or record['commit_index'] != index:
        raise ValueError('Epoch identity changed')
    scientific = {f:record[f] for f in ('job_id','epoch','commit_index','model_hash','order_hash')}
    record_hash = digest(scientific)
    if index == row['commit_index'] and row.get('record_hash') == record_hash:
        return dict(status='ALREADY_COMMITTED',commit_index=index)
    if index != row['commit_index']+1:
        raise ValueError('Epoch commit is not consecutive')
    previous = row.copy()
    # RAM precedes publication; a rejected scalar/replay transaction restores the old RAM epoch.
    row.update(state=snapshot,record=record,commit_index=index,record_hash=record_hash)
    try:
        status = persist_epoch(output,key,record)
    except Exception:
        row.clear(); row.update(previous)
        raise
    append_json(Path(output)/'epoch_commit_ledger.jsonl',dict(job_id=key,commit_index=index,
        record_hash=record_hash,status=status,time=time.time()))
    return dict(status=status,commit_index=index)


def retry_job(output,key,row):
    if alive(row.get('worker')) or row['status'] not in ('ENGINEERING_BLOCKED','RUNNING'):
        raise ValueError('Only a failed or dead-worker job can be resumed')
    previous = row['status']
    row.update(status='PAUSED_RAM' if row.get('state') is not None else 'READY',
        worker=None,retry_count=row.get('retry_count',0)+1)
    attempt = dict(job_id=key,previous_status=previous,status=row['status'],
        retry_count=row['retry_count'],commit_index=row['commit_index'],time=time.time())
    append_json(Path(output)/'RETRIES.jsonl',attempt)
    return attempt


def fail_job(row,pid,error):
    owner = process_identity(pid)
    if owner is None or row.get('worker') != owner:
        raise ValueError('Only the registered owning worker can report its failure')
    row.update(status='ENGINEERING_BLOCKED',worker=None,error=error)
    return dict(status='ENGINEERING_BLOCKED')


def ensure_guardian(output):
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    info_path = output/'guardian.json'
    if info_path.exists() and alive(read_json(info_path).get('owner')):
        return request(output,'status')
    with (output/'guardian.log').open('ab') as log:
        child = subprocess.Popen([sys.executable,'-m','experiments.aug_history_20261105.run',
            '_guardian','--output',str(output)],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=log,start_new_session=True)
    for _ in range(600):
        if info_path.exists():
            info = read_json(info_path)
            if info.get('pid') == child.pid:
                return request(output,'status')
        if child.poll() is not None:
            break
        time.sleep(.1)
    raise RuntimeError('RAM guardian failed; inspect guardian.log')


def serve(output):
    output = Path(output); records = {}; auth = secrets.token_bytes(32)
    address = f'/tmp/ah1105-{os.getuid()}-{secrets.token_hex(6)}.sock'
    with Listener(address,family='AF_UNIX',authkey=auth) as listener:
        write_json(output/'guardian.json',dict(pid=os.getpid(),owner=process_identity(os.getpid()),
            address=address,authkey=auth.hex()))
        os.chmod(output/'guardian.json',0o600)
        stop = False
        while not stop:
            with listener.accept() as connection:
                try:
                    message = receive_memory(connection)
                    command, key = message['command'],message.get('key')
                    if command == 'status':
                        result = dict(pid=os.getpid(),snapshot_bytes=sum(tensor_bytes(r.get('state')) for r in records.values()),
                            jobs={k:{f:v for f,v in row.items() if f not in ('state','record')} for k,row in records.items()})
                    elif command == 'register':
                        row = records.setdefault(key,dict(job=message['job'],commit_index=0,status='READY'))
                        if row['job'] != message['job'] or alive(row.get('worker')):
                            raise ValueError('Job identity changed or another worker owns it')
                        row.update(worker=process_identity(message['pid']),status='RUNNING',device=message.get('device'))
                        result = dict(state=row.get('state'))
                    elif command == 'commit':
                        result = commit_epoch(output,key,records[key],message['state'],message['record'])
                    elif command == 'get':
                        result = dict(state=records.get(key,{}).get('state'))
                    elif command == 'summaries':
                        summaries = {}
                        for name,row in records.items():
                            snapshot = row.get('state')
                            if snapshot is None:
                                continue
                            summaries[name] = dict(job=snapshot['job'],complete=snapshot['complete'],
                                epoch=snapshot['epoch'],cost=snapshot['cost'],candidate_cost=snapshot['candidate_cost'],
                                layout_hash=snapshot['layout_hash'],snapshots={candidate:{endpoint:{
                                    k:v for k,v in point.items() if k!='model_state'}
                                    for endpoint,point in endpoints.items()} for candidate,endpoints in snapshot['snapshots'].items()})
                            if snapshot['route']=='A':
                                summaries[name]['estimator_costs']={cid:data['cost'].copy() for cid,data in snapshot['bank']['states'].items()}
                        result = dict(summaries=summaries)
                    elif command == 'complete':
                        row = records[key]
                        if not row['state']['complete'] or row['state']['epoch'] != 50:
                            raise ValueError('Full training budget is incomplete')
                        row.update(status='TRAIN_COMPLETE',worker=None)
                        result = dict(status='TRAIN_COMPLETE')
                    elif command == 'failed':
                        result = fail_job(records[key],message['pid'],message['error'])
                    elif command == 'retry':
                        result = retry_job(output,key,records[key])
                    elif command == 'release':
                        route=read_json(output/'PREPARED.json').get('route','B')
                        if not (output/route/'EVALUATED.json').exists():
                            raise ValueError('Unevaluated endpoint state cannot be released')
                        for row in records.values():
                            if alive(row.get('worker')):
                                raise ValueError('Training is still active')
                            row.pop('state',None); row.pop('record',None); row['status']='RELEASED'
                        result = dict(status='RELEASED')
                    elif command == 'shutdown':
                        if any(row.get('state') is not None or alive(row.get('worker')) for row in records.values()):
                            raise ValueError('RAM state or active workers remain')
                        stop=True; result=dict(status='SHUTDOWN')
                    else:
                        raise ValueError('Unknown guardian command')
                    send_memory(connection,result)
                except (EOFError,BrokenPipeError,ConnectionResetError):
                    continue
                except Exception as error:
                    try:
                        send_memory(connection,dict(error=f'{type(error).__name__}: {error}'))
                    except (EOFError,BrokenPipeError,ConnectionResetError):
                        pass
    Path(address).unlink(missing_ok=True)
