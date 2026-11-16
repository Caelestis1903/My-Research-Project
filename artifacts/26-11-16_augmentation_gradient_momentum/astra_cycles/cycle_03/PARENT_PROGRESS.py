"""Same-conversation parent observation; no training or admission decisions."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from experiments.nqr_temporal_exposure_20261116 import protocol as p


def main():
    n = p.ATTEMPT
    manifest = p.read_json(n/'run_manifest.json')
    owner = p.request(n, 'status')
    keys = {j['key'] for j in manifest['jobs']}
    jobs = {k:v for k,v in owner['jobs'].items() if k in keys}
    latest = {}
    ledger = n/'epoch_commit_ledger.jsonl'
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            if row['key'] in keys:
                latest[row['key']] = row
    epochs = sum(r['epoch'] for r in latest.values())
    updates = sum(r['epoch']*next(j['config']['steps_per_epoch'] for j in manifest['jobs'] if j['key']==key)
                  for key,r in latest.items())
    statuses = Counter(v['status'] for v in jobs.values())
    statuses['NOT_STARTED'] = len(keys)-len(jobs)
    launches = []
    launch_path = n/'launch_ledger.jsonl'
    if launch_path.exists():
        launches = [json.loads(line) for line in launch_path.read_text().splitlines()]
    live = []
    for row in launches:
        pid = row['pid']
        if not p.alive(pid):
            continue
        try:
            args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        if b'experiments.nqr_temporal_exposure_20261116.runner' not in args or b'worker' not in args:
            continue
        j = next(j for j in manifest['jobs'] if j['key']==row['key'])
        live.append(dict(pid=pid,gpu=row['gpu'],dataset=j['dataset'],arm=j['arm'],seed=j['seed'],
                         committed_epoch=latest.get(j['key'],{}).get('epoch',0),
                         allocated=jobs.get(j['key'],{}).get('allocated',False)))
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,power.draw,power.limit,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    compute_rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                           '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    live_ids = {r['pid'] for r in live}
    cuda_rows = [row for row in compute_rows if int(row.split(',')[1].strip()) in live_ids]
    cuda_ids = {int(row.split(',')[1].strip()) for row in cuda_rows}
    values = {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:','MemAvailable:'))}
    stopped = p.stop_requested()
    paused = (n/'PAUSE_TRAINING').exists()
    value = dict(cycle=3,status='USER_STOPPED' if stopped else 'TECHNICAL_PAUSE' if paused else
                 'COMPLETE' if statuses['COMPLETE']==20 else 'RUNNING',
                 updated_at_utc=datetime.now(timezone.utc).isoformat(),
                 committed_epochs=epochs,total_epochs=1000,committed_updates=updates,total_updates=34500,
                 job_status_counts=dict(statuses),live_workers=live,live_workers_by_gpu=dict(Counter(r['gpu'] for r in live)),
                 actual_cuda_workers=len(cuda_ids),own_cuda_rows= cuda_rows,
                 actual_cuda_workers_by_gpu=dict(Counter(r['gpu'] for r in live if r['pid'] in cuda_ids)),
                 gpu_rows_memoryMiB_powerW_utilPercent=gpu,host_fraction=1-values['MemAvailable']/values['MemTotal'],
                 owner_pid=owner['pid'],owner_retained_states=owner['retained_states'],
                 owner_retained_tensor_bytes=owner['retained_bytes'],test_accessed=False)
    target=n/'CURRENT_PROGRESS.json'; temp=target.with_suffix('.parent.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(target)
    with (n/'PARENT_PROGRESS_HISTORY.jsonl').open('a') as f:
        f.write(json.dumps(value)+'\n')
    peak=p.CYCLE/'PARENT_CONCURRENCY_VERIFICATION.json'
    previous=p.read_json(peak)['actual_cuda_workers'] if peak.exists() else -1
    if len(cuda_ids)>previous:
        temp=peak.with_suffix('.parent.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(peak)
    root=p.ROOT/p.REL/'AUTONOMOUS_STATE.json';state=p.read_json(root)
    state.update(current_cycle=3,current_stage='RUN',current_candidate='NQR-EARLY10',
                 cycle03_progress=value,last_update_utc=value['updated_at_utc'])
    temp=root.with_suffix('.parent.tmp');temp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n');temp.replace(root)
    print(json.dumps(value))


if __name__=='__main__':
    main()
