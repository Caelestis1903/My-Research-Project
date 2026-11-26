"""Resource admissions and epoch-fair scalar queue decisions."""
import fcntl
import os
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path
from .guardian import process_identity,alive
from .state import read_json,write_json

LOCK_ROOT=Path.home()/'.cache/mdmt_job_locks'
@contextmanager
def lock(name,blocking=False):
    LOCK_ROOT.mkdir(parents=True,exist_ok=True)
    with (LOCK_ROOT/name).open('a+') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError: raise RuntimeError('Owner lease already held: '+name)
        try: yield stream
        finally: fcntl.flock(stream,fcntl.LOCK_UN)
def gpu_resources(device):
    output=subprocess.check_output(['nvidia-smi',f'--id={device}',
        '--query-gpu=uuid,memory.used,memory.total,power.draw,power.limit','--format=csv,noheader,nounits'],text=True)
    uuid,used,total,power,limit=[v.strip() for v in output.strip().split(',')]
    def number(v):
        try: return float(v)
        except ValueError: return None
    return dict(device=int(device),uuid=uuid,memory_used_mib=float(used),memory_total_mib=float(total),
        power_draw_w=number(power),power_limit_w=number(limit))
def compute_processes():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
        '--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in raw.splitlines():
        uuid,pid,name=[v.strip() for v in line.split(',',2)]
        if 'nvidia-cuda-mps-server' in name: continue
        owner=process_identity(int(pid))
        try: command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        except OSError: command=None
        rows.append(dict(gpu_uuid=uuid,owner=owner,command=command,process_name=name))
    return rows
def unknown_compute(attempt):
    path=Path(attempt)/'RUNTIME_OVERRIDES.json'
    approved=read_json(path).get('shared_compute_owners',[]) if path.exists() else []
    unknown={}
    for row in compute_processes():
        if row['owner'] and row['owner']['pid']==os.getpid(): continue
        if row['command'] and 'overnight_ab_20261103' in row['command'] and str(Path(attempt)) in row['command']: continue
        if any(row['owner']==old['owner'] and row['command']==old['command'] and row['gpu_uuid']==old['gpu_uuid'] for old in approved): continue
        unknown.setdefault(row['gpu_uuid'],[]).append(row)
    return unknown
def available_ram():
    values={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
    return values['MemAvailable'],values['MemTotal']
def resources_allow(resource,peak_bytes,snapshot_bytes=0):
    if resource['memory_used_mib']/resource['memory_total_mib']>=.85: return False
    if resource['power_draw_w'] is not None and resource['power_limit_w'] and resource['power_draw_w']/resource['power_limit_w']>=.85: return False
    if (resource['memory_total_mib']-resource['memory_used_mib'])*(1<<20)<1.2*peak_bytes: return False
    free,total=available_ram()
    return free-max(6*(1<<30),total*.1)>snapshot_bytes*2
def external_reservations(attempt):
    reserved={}; own_root=str(Path(__file__).resolve().parents[2])
    for file in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            pid=int(file.parent.name); cmd=file.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 'python' not in cmd or 'mdmt' not in cmd or 'overnight_ab_20261103' in cmd: continue
            if not any(word in cmd for word in ('coordinator',' run ',' run\n')): continue
            if '--devices' in cmd:
                tail=cmd.split('--devices',1)[1].strip().split()[0]
                for device in tail.split(','):
                    if device.isdigit(): reserved[int(device)]=dict(owner=process_identity(pid),command=cmd)
        except (OSError,IndexError,ValueError): continue
    return reserved
def choose_job(jobs,states,preferred_lane,active_ids):
    eligible=[j for j in jobs if j['job_id'] not in active_ids and
              states[j['job_id']]['status'] in ('READY','PAUSED_RAM','RETRY_PENDING','STATE_LOST_REPLAY')]
    if not eligible: return None
    lanes=[preferred_lane,'B' if preferred_lane=='A' else 'A']
    for lane in lanes:
        current=[j for j in eligible if j['lane']==lane]
        if current:
            return min(current,key=lambda j:(0 if states[j['job_id']].get('commit_index',0)>0 else 1,
                j['seed'],0 if j['dataset']=='kuairand' else 1,0 if j['arm']=='HYPERGATE' else 1))
    return eligible[0]
def accepted(attempt,job):
    prepared=read_json(Path(attempt)/'PREPARED.json')
    path=Path(prepared['report_root'])/'ENGINEERING_ACCEPTANCE.json'
    if not path.exists(): return False
    row=read_json(path).get('families',{}).get(job['family'],{})
    return row.get('status') in ('PASS','PASS_WITH_FIXES','PASS WITH FIXES') and not row.get('blocking_issues') and row.get('source_hash')==job['source_hash']

