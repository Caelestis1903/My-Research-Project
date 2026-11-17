"""Parent progress from the owner's numeric commit ledger; no Torch import or state IPC."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

ROOT=Path('/home/xzf/Code/phase0-3_tvg_rm_20261116')
A=ROOT/'artifacts/26-11-16_augmentation_gradient_momentum'
N=A/'astra_cycles/cycle_04/attempt2'
read=lambda p:json.loads(p.read_text())


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def main():
    manifest=read(N/'run_manifest.json');jobs={}
    for path in (N/'jobs').glob('*.json'):
        j=read(path)
        if j['purpose']=='formal':
            assert j['key']==path.stem and j['source_hash']==manifest['source_hash']
            jobs[j['key']]=j
    assert len(jobs)==20 and {j['seed'] for j in jobs.values()}==set(range(2022,2027))
    epochs={key:0 for key in jobs}
    for line in (N/'epoch_commit_ledger.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['key'] in jobs and r['status']=='RAM_COMMITTED':epochs[r['key']]=max(epochs[r['key']],r['epoch'])
    assert all(0<=e<=50 for e in epochs.values())
    starts={}
    for line in (N/'physical_attempts.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r.get('purpose')=='formal' and r['status']=='STARTED':starts[r['pid']]=r
    live=[]
    for pid,r in starts.items():
        if not alive(pid):continue
        j=jobs[r['key']]
        live.append(dict(pid=pid,dataset=j['dataset'],arm=j['arm'],seed=j['seed'],
                         committed_epoch=epochs[r['key']],device=r['device']))
    complete=set()
    for key in jobs:
        path=N/'runs'/key/'result.json'
        if path.exists():
            try:result=read(path)
            except json.JSONDecodeError:continue
            assert result['job']==jobs[key] and result['epochs']==50 and epochs[key]==50
            complete.add(key)
    live=[r for r in live if starts[r['pid']]['key'] not in complete]
    cards=[];uuid_index={}
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,power.draw,power.limit,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines():
        f=[v.strip() for v in line.split(',')];index=int(f[0]);uuid_index[f[1]]=index
        cards.append(dict(index=index,memory_MiB=float(f[2]),total_MiB=float(f[3]),power_W=float(f[4]),power_limit_W=float(f[5]),utilization_percent=float(f[6])))
    ids={r['pid'] for r in live};cuda={}
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines():
        uuid,pid,memory=[x.strip() for x in line.split(',')]
        if int(pid) in ids:cuda[int(pid)]=dict(gpu=uuid_index[uuid],memory_MiB=float(memory))
    for row in live:row.update(cuda.get(row['pid'],dict(gpu=None,memory_MiB=None)))
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    state=read(A/'AUTONOMOUS_STATE.json');owner=read(N/'PUBLIC_STORAGE_OWNER.json')
    stopped=state.get('manual_stop_requested') or (A/'STOP_AUTONOMOUS').exists()
    assert alive(owner['pid']),'Current RAM owner unavailable'
    counts=dict(COMPLETE=len(complete),ACTIVE=len(live),WAITING=20-len(complete)-len(live))
    value=dict(cycle=4,attempt=N.name,source_hash=manifest['source_hash'],
        status='USER_STOPPED' if stopped else 'TECHNICAL_PAUSE' if (N/'PAUSE_TRAINING').exists() else 'COMPLETE' if len(complete)==20 else 'RUNNING',
        updated_at_utc=datetime.now(timezone.utc).isoformat(),ram_committed_epochs=sum(epochs.values()),total_epochs=1000,
        ram_committed_updates=sum(e*jobs[k]['config']['steps_per_epoch'] for k,e in epochs.items()),total_updates=34500,
        job_status_counts=counts,live_workers=live,actual_cuda_workers=len(cuda),
        actual_cuda_workers_by_gpu=dict(Counter(x['gpu'] for x in cuda.values())),cards=cards,
        host_fraction=1-mem['MemAvailable']/mem['MemTotal'],owner_pid=owner['pid'],
        frontier_semantics='owner numeric RAM_COMMITTED journal; final acceptance separately verifies ACKs and live tensors',
        progress_observation='stdlib ledger/procfs/NVML; no Torch import or state IPC',test_accessed=False)
    temp=N/'CURRENT_PROGRESS.parent.tmp';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(N/'CURRENT_PROGRESS.json')
    with (N/'PARENT_PROGRESS_HISTORY.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    peak=N/'PARENT_CONCURRENCY_VERIFICATION.json'
    if not peak.exists() or len(cuda)>read(peak)['actual_cuda_workers']:peak.write_text(json.dumps(value,indent=2)+'\n')
    state.update(current_cycle=4,current_stage='RUN',cycle04_progress=value,last_update_utc=value['updated_at_utc'],
        training_status=value['status'],completed_jobs=counts['COMPLETE'],planned_jobs=20,
        committed_epochs=value['ram_committed_epochs'],committed_updates=value['ram_committed_updates'],
        committed_optimizer_updates=value['ram_committed_updates'],
        current_cuda_observed_counts=value['actual_cuda_workers_by_gpu'],
        current_formal_worker_counts=dict(Counter(x['device'] for x in live)),
        storage_owner_session_id=str(owner['tool_session']))
    temp=A/'AUTONOMOUS_STATE.parent.tmp';temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');temp.replace(A/'AUTONOMOUS_STATE.json')
    print(json.dumps(value))


if __name__=='__main__':main()
