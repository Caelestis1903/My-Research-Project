"""Independent RAM owner for epoch transactions; no parameter files."""
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from multiprocessing.connection import Client,Listener
from experiments.mdmt_history_information_20261103.protocol import send_memory,receive_memory
from .state import read_json,write_json,append_json,tensor_bytes

def process_identity(pid):
    try:
        raw=Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()
        if raw[0]=='Z': return None
        return dict(pid=int(pid),start=raw[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError,ValueError,TypeError): return None
def alive(identity):
    return bool(identity) and process_identity(identity['pid'])==identity
def request(attempt,command,**payload):
    info=read_json(Path(attempt)/'guardian.json')
    with Client(info['address'],family='AF_UNIX',authkey=bytes.fromhex(info['authkey'])) as connection:
        send_memory(connection,dict(command=command,**payload)); answer=receive_memory(connection)
    if 'error' in answer: raise RuntimeError(answer['error'])
    return answer
def ensure_guardian(attempt):
    attempt=Path(attempt)
    if (attempt/'guardian.json').exists():
        info=read_json(attempt/'guardian.json')
        if alive(info.get('owner')): return request(attempt,'status')
    with (attempt/'guardian.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-m','experiments.overnight_ab_20261103.run','_guardian',
            '--attempt',str(attempt)],cwd=Path(__file__).resolve().parents[2],stdin=subprocess.DEVNULL,
            stdout=log,stderr=log,start_new_session=True)
    for _ in range(300):
        if (attempt/'guardian.json').exists():
            try:
                info=read_json(attempt/'guardian.json')
                if info.get('pid')==child.pid: return request(attempt,'status')
            except (OSError,ValueError,ConnectionError): pass
        if child.poll() is not None: break
        time.sleep(.1)
    raise RuntimeError('Guardian startup failed; see guardian.log')
def serve(attempt):
    attempt=Path(attempt); records={}; auth=secrets.token_bytes(32)
    address=f'/tmp/overnight-ab-{os.getuid()}-{secrets.token_hex(6)}.sock'
    with Listener(address,family='AF_UNIX',authkey=auth) as server:
        write_json(attempt/'guardian.json',dict(pid=os.getpid(),owner=process_identity(os.getpid()),
            address=address,authkey=auth.hex())); os.chmod(attempt/'guardian.json',0o600)
        stop=False
        while not stop:
            with server.accept() as connection:
                try:
                    msg=receive_memory(connection); cmd=msg['command']; key=msg.get('key')
                    if cmd=='status':
                        answer=dict(pid=os.getpid(),jobs={k:{n:v for n,v in row.items() if n!='state'}
                            for k,row in records.items()},snapshot_bytes=sum(tensor_bytes(r.get('state')) for r in records.values()))
                    elif cmd=='register':
                        row=records.setdefault(key,dict(job=msg['job'],commit_index=0,status='READY',
                            failure_count=0,launch_index=0))
                        assert row['job']==msg['job'],'Scientific identity changed'
                        assert not alive(row.get('worker')),'Worker already active'
                        assert row['status'] not in ('DONE','BLOCKED_ENGINEERING'),'Job not runnable'
                        row.update(worker=process_identity(msg['pid']),status='RUNNING',launch_index=row['launch_index']+1,
                            device=msg.get('device'),allocated=False)
                        answer=dict(state=row.get('state'),launch_index=row['launch_index'])
                    elif cmd=='allocated':
                        records[key].update(peak_memory_bytes=msg['peak_memory_bytes'],allocated=True); answer=dict(status='ALLOCATED')
                    elif cmd=='commit':
                        row=records[key]; state=msg['state']
                        assert state['commit_index']==row['commit_index']+1,'Nonconsecutive commit'
                        assert msg['identity']==row['job'],'Commit identity changed'
                        row.update(state=state,commit_index=state['commit_index'],last_commit=time.time(),
                            snapshot_bytes=tensor_bytes(state))
                        append_json(attempt/'epoch_commit_ledger.jsonl',dict(key=key,commit_index=row['commit_index'],
                            launch_index=row['launch_index'],epoch_record_sha=msg['epoch_record_sha'],time=time.time()))
                        answer=dict(status='RAM_COMMITTED',commit_index=row['commit_index'])
                    elif cmd=='yield':
                        row=records[key]; assert row.get('state') is not None
                        row.update(status='PAUSED_RAM',worker=None); answer=dict(status='PAUSED_RAM')
                    elif cmd=='get': answer=dict(state=records.get(key,{}).get('state'))
                    elif cmd=='train_complete':
                        row=records[key]; state=row['state']; expected=10 if row['job']['arm']=='AUTOIFS' else 0
                        assert state['completed_search']==expected and state['completed_final']==50
                        row.update(status='TRAIN_COMPLETE'); answer=dict(status='TRAIN_COMPLETE')
                    elif cmd=='failed':
                        row=records[key]; failure=msg['failure']; count=row['failure_count']+1
                        retry=failure.get('infrastructure',False) and count<=2
                        row.update(status='RETRY_PENDING' if retry else 'BLOCKED_ENGINEERING',failure_count=count,
                            failure=failure,worker=None); answer=dict(status=row['status'],failure_count=count)
                    elif cmd=='retry':
                        row=records[key]; assert not alive(row.get('worker'))
                        assert row['status'] in ('BLOCKED_ENGINEERING','RETRY_PENDING')
                        row.update(status='PAUSED_RAM' if row.get('state') else 'READY',worker=None)
                        answer=dict(status=row['status'])
                    elif cmd=='release':
                        assert (attempt/'runs'/key/'DONE.json').exists(),'Endpoint archive missing'
                        records[key].pop('state',None); records[key].update(status='DONE',worker=None)
                        answer=dict(status='RELEASED')
                    elif cmd=='shutdown':
                        assert not any(alive(r.get('worker')) for r in records.values())
                        assert not any(r.get('state') is not None for r in records.values()),'Unevaluated state remains'
                        stop=True; answer=dict(status='SHUTDOWN')
                    else: raise ValueError(cmd)
                    send_memory(connection,answer)
                except (EOFError,BrokenPipeError,ConnectionResetError): continue
                except Exception as error:
                    try: send_memory(connection,dict(error=f'{type(error).__name__}: {error}'))
                    except (EOFError,BrokenPipeError,ConnectionResetError): pass
    Path(address).unlink(missing_ok=True)

