"""Parent resource, phase and prefix audit after complete numeric export."""
from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
N=ROOT/'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/attempt1'


def main():
    read=lambda p:json.loads(p.read_text())
    lines=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
    launches=lines(N/'launch_ledger.jsonl')
    assert len({r['key'] for r in launches})==20
    for row in launches:
        d=row['admission'];c=d['current']
        assert d['eligible'] and not d['reasons'] and not d['pending_allocation_ack']
        assert c['used_bytes']/c['total_bytes']<.85 and c['power']/c['power_limit']<.85
        assert d['host_fraction']<.9 and c['used_bytes']+d['incoming_full_process_bytes']<=d['cuda_usable_bytes']
    allocations=[r for r in lines(N/'admission.jsonl') if r.get('event')=='ALLOCATION_ACK_RESAMPLE']
    assert {r['key'] for r in allocations if r['allocated']}=={r['key'] for r in launches}
    handoff=read(N/'STORAGE_HANDOFF.json');assert len(handoff['epochs'])==1000
    prefix=Counter();phase=Counter();updates=0
    for entry in handoff['epochs']:
        r=read(Path(entry['path']));j=r['job'];e=r['epoch']
        check=r['prefix_check']
        if e<=(10 if j['arm']=='NQR-EARLY10' else 40):
            assert check['status']=='PASS';prefix[j['arm']]+=1
        else:
            assert check is None
        if e==50:
            counts=r['phase_counters'];s=j['config']['steps_per_epoch']
            assert counts['active_updates']==10*s and counts['off_updates']==40*s
            assert counts['lambda_update_sum']==2.5*s
            updates+=r['counts']['optimizer_updates']
            phase.update(counts)
    assert prefix=={'NQR-EARLY10':100,'NQR-LATE10':400}
    assert updates==34500 and phase['active_updates']==6900 and phase['off_updates']==27600
    peak_host=0.
    for path in (N/'runs').glob('*/attempt*/resource_samples.jsonl'):
        with path.open() as f:
            for line in f:
                peak_host=max(peak_host,json.loads(line)['host_ram_fraction'])
    history=lines(N/'PARENT_PROGRESS_HISTORY.jsonl')
    foreground=lines(N/'foreground_runs.jsonl')
    assert foreground[-1]['status']=='COMPLETE'
    with (N/'tables/PHYSICAL_ATTEMPTS.csv').open(newline='') as f:
        physical=list(csv.DictReader(f))
    utc=lambda t:datetime.fromtimestamp(t,timezone.utc).isoformat()
    audit=dict(admissions=len(launches),all_current_memory_power_ram_and_physical_fit_checks_passed=True,
        launches_by_gpu=dict(Counter(r['gpu'] for r in launches)),
        max_prelaunch_memory_fraction=max(r['admission']['current']['used_bytes']/r['admission']['current']['total_bytes'] for r in launches),
        max_prelaunch_power_fraction=max(r['admission']['current']['power']/r['admission']['current']['power_limit'] for r in launches),
        max_prelaunch_host_fraction=max(r['admission']['host_fraction'] for r in launches),
        max_parent_observed_host_fraction=max(r['host_fraction'] for r in history),
        max_measured_host_fraction=peak_host,
        max_observed_concurrent_cuda_workers=max(r['actual_cuda_workers'] for r in history),
        observed_concurrency_receipt='PARENT_CONCURRENCY_VERIFICATION.json',
        allocation_ack_keys=len({r['key'] for r in allocations}),
        foreground_seconds=sum(r['seconds'] for r in foreground),
        physical_worker_seconds=sum(float(r['seconds']) for r in physical),
        physical_status_counts=dict(Counter(r['status'] for r in physical)),
        first_launch_utc=utc(min(r['time'] for r in launches)),last_launch_utc=utc(max(r['time'] for r in launches)),
        foreground_completed_utc=utc(foreground[-1]['time']),
        prefix_comparisons=dict(prefix),new_committed_epochs=1000,new_committed_updates=updates,
        final_phase_counters=dict(phase),test_accessed=False)
    (OUT/'RESOURCE_EXECUTION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':
    main()
