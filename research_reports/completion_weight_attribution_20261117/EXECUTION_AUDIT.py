"""Count returned physical batch records separately from useful epoch commits."""
from collections import Counter
import gzip
import json
from pathlib import Path
import ANALYSIS as a


def main():
    n=a.N
    accepted=json.loads((n/'PARENT_STORAGE_ACCEPTANCE.json').read_text())
    assert accepted['status']=='ACCEPTED' and accepted['verified_epoch_ACKs']==1000
    archive=json.loads((n/'RAW_RUNS_MANIFEST.json').read_text())
    archived={r['archive_path']:r for r in archive['members']}
    committed={}
    for path in (n/'acks').glob('*/epoch_*.json'):
        ack=json.loads(path.read_text());record=json.loads((n/ack['epoch_file']).read_text())
        if record['purpose']=='formal':committed[record['batch_log']]=ack['batch_log_sha256']
    assert len(committed)==1000
    rows=[]
    for path in sorted((n/'runs').glob('*/attempt*/batches_epoch_*.jsonl.gz')):
        relative=str(path.relative_to(n));values=Counter();seconds=0.;complete=True
        try:
            with gzip.open(path,'rt') as stream:
                for line in stream:
                    r=json.loads(line);assert r['test_accessed'] is False
                    values.update(calls=1,rows=r['B'],diagnostics=int(r['ledger']['diagnose']),
                        sweeps=r['extra_autograd_sweeps'],virtual_steps=r['virtual_adam_steps'])
                    seconds+=r['seconds']
        except (EOFError,OSError,json.JSONDecodeError):complete=False
        bound=archived[relative]['sha256']
        if relative in committed:assert complete and bound==committed[relative]
        rows.append(dict(path=relative,key=path.parts[-3],attempt=int(path.parent.name.removeprefix('attempt')),
            epoch=int(path.name.split('_')[-1].split('.')[0]),returned_calls=values['calls'],training_rows=values['rows'],
            diagnostic_points=values['diagnostics'],extra_autograd_sweeps=values['sweeps'],virtual_adam_steps=values['virtual_steps'],
            returned_call_seconds=seconds,committed_file=relative in committed,fully_read=complete,archive_sha256=bound))
    useful=[r for r in rows if r['committed_file']]
    assert sum(r['returned_calls'] for r in useful)==34500
    assert sum(r['training_rows'] for r in useful)==1403149500
    assert sum(r['diagnostic_points'] for r in useful)==80
    assert sum(r['extra_autograd_sweeps'] for r in useful)==240 and sum(r['virtual_adam_steps'] for r in useful)==320
    a.dump_csv('PHYSICAL_BATCH_LEDGER.csv',rows)
    physical=a.physical_attempts(a.read_csv('PHYSICAL_ATTEMPTS.csv'))
    controller_names=('FORMAL_CONTROLLER_INITIAL_PAUSE_01.json','FORMAL_CONTROLLER_RECOVERY_01.json','FORMAL_CONTROLLER_RECOVERY_02.json')
    controllers=[dict(path=name,**{k:v for k,v in json.loads((n/name).read_text()).items() if k in ('status','foreground_wall_seconds')}) for name in controller_names]
    value=dict(status='COMPLETE',source_hash=accepted['source_hash'],design_bundle_sha256=accepted['design_bundle_sha256'],
        useful_epochs=1000,useful_updates=34500,useful_training_rows=1403149500,useful_diagnostic_points=80,
        physical_processes=len(physical),physical_status_counts=dict(Counter(r['status'] for r in physical)),
        physical_batch_files=len(rows),completely_read_files=sum(r['fully_read'] for r in rows),
        physical_returned_calls_lower_bound=sum(r['returned_calls'] for r in rows),
        uncommitted_or_replayed_returned_calls=sum(r['returned_calls'] for r in rows if not r['committed_file']),
        physical_returned_training_rows=sum(r['training_rows'] for r in rows),
        physical_returned_diagnostic_points=sum(r['diagnostic_points'] for r in rows),
        physical_returned_extra_autograd_sweeps=sum(r['extra_autograd_sweeps'] for r in rows),
        physical_returned_virtual_adam_steps=sum(r['virtual_adam_steps'] for r in rows),
        physical_returned_call_seconds=sum(r['returned_call_seconds'] for r in rows),
        known_worker_seconds=sum(float(r['seconds']) for r in physical if r['seconds'] not in ('',None)),
        missing_worker_terminal_durations=sum(r['seconds'] in ('',None) for r in physical),
        supervisor_process_wall_seconds=sum(float(r['supervisor_process_wall_seconds']) for r in physical if r['supervisor_process_wall_seconds'] not in ('',None)),
        controllers=controllers,foreground_segment_seconds=sum(r['foreground_wall_seconds'] for r in controllers),
        physical_time_semantics='controller segments, overlapping worker/process durations and returned-call subset are distinct; do not add them',
        physical_count_semantics='numeric returned records are a lower bound; missing unreturned work is not zero',
        raw_archive_sha256=archive['sha256'],raw_archive_member_count=archive['member_count'],
        cost_field_correction=json.loads((a.OUT/'COST_ACCOUNTING_CORRECTION.json').read_text()),test_accessed=False)
    availability_path=n.parent/'SELECTED_RAM_AVAILABILITY_UPDATE.json'
    availability=json.loads(availability_path.read_text())
    value.update(selected_ram_available_at_parent_acceptance=True,current_selected_ram_available=availability['selected_ram_available'],
        availability_update=dict(path=str(availability_path),sha256=a.sha(availability_path)),final_test_eligible=False)
    (a.OUT/'RESOURCE_EXECUTION_AUDIT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
