"""Same-conversation parent observation of the Cycle04 RAM commit frontier."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

from experiments.completion_weight_attribution_20261117 import protocol as p


def main():
    n=p.ATTEMPT;manifest=p.read_json(n/'run_manifest.json')
    jobs={}
    for dataset in p.DATASETS:
        for arm in p.ARMS:
            for seed in p.SEEDS:
                j=p.job(dataset,arm,seed,manifest['source_hash'])
                assert p.read_json(n/'jobs'/f'{j["key"]}.json')==j
                jobs[j['key']]=j
    status=p.request(n,'status');records={k:v for k,v in status['jobs'].items() if k in jobs}
    epochs=sum(r['epoch'] for r in records.values())
    updates=sum(r['epoch']*p.STEPS[jobs[k]['dataset']] for k,r in records.items())
    counts=Counter(r['status'] for r in records.values());counts['NOT_STARTED']=20-len(records)
    live=[]
    for key,record in records.items():
        pid=record.get('worker_pid')
        if not pid:continue
        try:
            if Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
        except FileNotFoundError:continue
        j=jobs[key]
        live.append(dict(pid=pid,dataset=j['dataset'],arm=j['arm'],seed=j['seed'],
                         committed_epoch=record['epoch'],allocated=record.get('allocated',False)))
    cards=[];uuid_index={}
    query=['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,power.draw,power.limit,utilization.gpu','--format=csv,noheader,nounits']
    for line in subprocess.check_output(query,text=True).strip().splitlines():
        fields=[x.strip() for x in line.split(',')]
        index=int(fields[0]);uuid_index[fields[1]]=index
        cards.append(dict(index=index,memory_MiB=float(fields[2]),total_MiB=float(fields[3]),
                          power_W=float(fields[4]),power_limit_W=float(fields[5]),utilization_percent=float(fields[6])))
    ids={r['pid'] for r in live};cuda={}
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines():
        uuid,pid,memory=[s.strip() for s in line.split(',')]
        if int(pid) in ids:cuda[int(pid)]=dict(gpu=uuid_index[uuid],memory_MiB=float(memory))
    for worker in live:worker.update(cuda.get(worker['pid'],dict(gpu=None,memory_MiB=None)))
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
         if line.startswith(('MemTotal:','MemAvailable:'))}
    stopped=p.stop_requested();paused=(n/'PAUSE_TRAINING').exists()
    value=dict(cycle=4,attempt=n.name,source_hash=manifest['source_hash'],status='USER_STOPPED' if stopped else 'TECHNICAL_PAUSE' if paused else
               'COMPLETE' if counts['COMPLETE']==20 else 'RUNNING',
               updated_at_utc=datetime.now(timezone.utc).isoformat(),ram_committed_epochs=epochs,total_epochs=1000,
               ram_committed_updates=updates,total_updates=34500,job_status_counts=dict(counts),live_workers=live,
               actual_cuda_workers=len(cuda),actual_cuda_workers_by_gpu=dict(Counter(v['gpu'] for v in cuda.values())),
               cards=cards,host_fraction=1-mem['MemAvailable']/mem['MemTotal'],owner_pid=status['pid'],
               owner_retained_states=status['retained_states'],owner_retained_tensor_bytes=status['retained_bytes'],
               frontier_semantics='RAM_COMMITTED; final acceptance separately verifies all numeric ACKs',test_accessed=False)
    target=n/'CURRENT_PROGRESS.json';temp=target.with_suffix('.parent.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(target)
    with (n/'PARENT_PROGRESS_HISTORY.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    peak=n/'PARENT_CONCURRENCY_VERIFICATION.json'
    previous=p.read_json(peak)['actual_cuda_workers'] if peak.exists() else -1
    if len(cuda)>previous:
        temp=peak.with_suffix('.parent.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(peak)
    root=p.ROOT/'artifacts/26-11-16_augmentation_gradient_momentum/AUTONOMOUS_STATE.json';state=p.read_json(root)
    state.update(current_cycle=4,current_stage='RUN',cycle04_progress=value,last_update_utc=value['updated_at_utc'])
    temp=root.with_suffix('.parent.tmp');temp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n');temp.replace(root)
    print(json.dumps(value))


if __name__=='__main__':main()
