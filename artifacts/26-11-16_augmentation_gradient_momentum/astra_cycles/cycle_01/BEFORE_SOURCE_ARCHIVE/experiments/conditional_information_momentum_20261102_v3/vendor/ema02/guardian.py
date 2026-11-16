"""Independent RAM owner: every epoch is committed only after a BytesIO IPC ACK."""
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from multiprocessing.connection import Client, Listener
from .protocol import send_memory, receive_memory, write_json, state_digest


def request(attempt, command, **payload):
    info = json.loads((Path(attempt)/'guardian.json').read_text())
    with Client(info['address'], family='AF_UNIX', authkey=bytes.fromhex(info['authkey'])) as conn:
        send_memory(conn, dict(command=command, **payload))
        answer = receive_memory(conn)
    if 'error' in answer: raise RuntimeError(answer['error'])
    return answer


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (ProcessLookupError, FileNotFoundError, ValueError, TypeError): return False


def ensure_guardian(attempt):
    attempt = Path(attempt).resolve(); attempt.mkdir(parents=True, exist_ok=True)
    if (attempt/'guardian.json').exists():
        return request(attempt, 'status')
    with (attempt/'guardian.log').open('ab') as log:
        process=subprocess.Popen([sys.executable, '-m', 'experiments.ema_view_design_20261102_v2.run',
            '_guardian', '--attempt', str(attempt)],cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    for _ in range(600):
        if (attempt/'guardian.json').exists(): return request(attempt, 'status')
        if process.poll() is not None:break
        time.sleep(.1)
    raise RuntimeError('RAM guardian startup failed; see guardian.log')


def serve(attempt):
    attempt = Path(attempt).resolve(); auth = secrets.token_bytes(32)
    address = f'/tmp/ema02v2-{os.getuid()}-{secrets.token_hex(8)}.sock'
    records = {}
    with Listener(address, family='AF_UNIX', authkey=auth) as server:
        write_json(attempt/'guardian.json', dict(address=address, authkey=auth.hex(), pid=os.getpid()))
        os.chmod(attempt/'guardian.json', 0o600)
        stop = False
        while not stop:
            with server.accept() as conn:
                try:
                    msg = receive_memory(conn); cmd = msg['command']; key = msg.get('key')
                    if cmd == 'status':
                        answer = dict(pid=os.getpid(), jobs={k: {n:v for n,v in r.items() if n != 'state'}
                            for k,r in records.items()})
                    elif cmd == 'roundtrip': answer = dict(value=msg['value'])
                    elif cmd == 'register':
                        r = records.setdefault(key, dict(job=msg['job'], epoch=0, status='REGISTERED', attempts=0))
                        assert r['job'] == msg['job'], 'Scientific job identity changed'
                        assert r['status'] != 'COMPLETE', 'Completed job must not rerun'
                        assert not alive(r.get('worker_pid')), 'Existing worker still alive'
                        assert r['attempts'] < 2, 'Retry budget exhausted'
                        r.update(worker_pid=msg['pid'], status='RUNNING', attempts=r['attempts']+1,
                                 started=time.time(), allocated=False,
                                 resume_worker_seconds=r.get('state',{}).get('physical_worker_seconds',0.))
                        answer = dict(attempt=r['attempts'], state=r.get('state'))
                    elif cmd == 'allocated':
                        records[key].update(allocated=True, peak_memory_bytes=msg['peak_memory_bytes'])
                        answer = dict(status='ALLOCATED')
                    elif cmd == 'commit':
                        r = records[key]; state = msg['state']
                        assert state['epoch'] == r['epoch'] + 1
                        r.update(state=state, epoch=state['epoch'], last_commit=time.time(),
                            worker_seconds=state['physical_worker_seconds'], counts=state['counts'])
                        with (attempt/'epoch_commit_ledger.jsonl').open('a') as f:
                            f.write(json.dumps(dict(key=key, epoch=r['epoch'], attempt=r['attempts'],
                                epoch_file=msg['epoch_file'], time=time.time(), status='RAM_COMMITTED'))+'\n')
                        answer = dict(status='RAM_COMMITTED', epoch=r['epoch'])
                    elif cmd == 'complete':
                        r = records[key]; assert r['epoch'] == 50
                        r.update(status='COMPLETE', result=msg['result'], worker_pid=None)
                        r['state'].pop('optimizer', None); r['state'].pop('rng', None); r['state'].pop('loader', None)
                        write_json(attempt/'runs'/key/'result.json', msg['result'])
                        answer = dict(status='COMPLETE')
                    elif cmd == 'failed':
                        r = records[key]; r.update(status='FAILED', failure=msg['failure'], worker_pid=None)
                        answer = dict(status='FAILED')
                    elif cmd == 'get': answer = dict(state=records[key].get('state'), result=records[key].get('result'))
                    elif cmd == 'release':
                        records[key].pop('state', None); records[key]['state_released'] = True
                        answer = dict(status='RELEASED')
                    elif cmd == 'shutdown':
                        assert all(r.get('state_released') or r['status'] != 'COMPLETE' for r in records.values())
                        assert not any(alive(r.get('worker_pid')) for r in records.values())
                        stop = True; answer = dict(status='SHUTDOWN')
                    else: raise ValueError(cmd)
                    send_memory(conn, answer)
                except (EOFError, BrokenPipeError, ConnectionResetError):
                    continue
                except Exception as error:
                    try: send_memory(conn, dict(error=f'{type(error).__name__}: {error}'))
                    except (EOFError, BrokenPipeError, ConnectionResetError): pass
    Path(address).unlink(missing_ok=True)
