"""One-shot foreground parent supervision; never starts a session or worker."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from experiments.cell_mass_bce_20261117 import protocol as p

status = p.request(p.ATTEMPT, 'status')
jobs = []
for key, value in status['jobs'].items():
    job = value['job']
    if job['purpose'] != 'formal':
        continue
    jobs.append(dict(key=key, dataset=job['dataset'], seed=job['seed'], status=value['status'],
                     epoch=value['epoch'], worker_pid=value.get('worker_pid'),
                     attempts=value.get('attempts'), allocated=value.get('allocated')))
row = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(), source_hash=p.source_manifest()['sha256'],
           owner=p.owner_binding(p.ATTEMPT), jobs=jobs, completed_epochs=sum(j['epoch'] for j in jobs),
           completed_trajectories=sum(j['status'] == 'COMPLETE' for j in jobs), planned_trajectories=10,
           retained_states=status['retained_states'], retained_bytes=status['retained_bytes'],
           pause_flag=(p.ATTEMPT / 'PAUSE_TRAINING').exists(), user_stop_requested=p.stop_requested(), test_accessed=False)
p.append_json(p.ATTEMPT / 'PARENT_PROGRESS.jsonl', row)
for root in (p.ORIGINAL, p.ROOT):
    path = root / 'artifacts/26-11-16_augmentation_gradient_momentum/AUTONOMOUS_STATE.json'
    d = json.loads(path.read_text())
    d.update(formal_completed_epochs=row['completed_epochs'], formal_completed_trajectories=row['completed_trajectories'],
             formal_planned_epochs=500, formal_planned_trajectories=10,
             current_training_active=any(j['status'] == 'RUNNING' for j in jobs),
             actual_progress=row, updated_at_utc=row['recorded_at_utc'])
    if jobs and not row['pause_flag']:
        d.update(status='ACTIVE', phase='CYCLE05_FORMAL_TRAINING', recovery_in_progress=False)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)
print(json.dumps(dict(time=row['recorded_at_utc'], completed_epochs=row['completed_epochs'],
                      completed_trajectories=row['completed_trajectories'],
                      running=sum(j['status'] == 'RUNNING' for j in jobs),
                      dataset_epochs={d: {j['seed']: j['epoch'] for j in jobs if j['dataset'] == d} for d in p.DATASETS},
                      retained_states=row['retained_states'], retained_bytes=row['retained_bytes'],
                      owner_pid=row['owner']['pid'], pause_flag=row['pause_flag'], user_stop=row['user_stop_requested'])))
