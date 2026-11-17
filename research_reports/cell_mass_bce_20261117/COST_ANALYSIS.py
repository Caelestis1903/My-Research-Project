"""Parent cost accounting: deduplicate processes, retain unknowns, no time addition across overlapping scopes."""
from collections import defaultdict
import csv
import json
from pathlib import Path

RAW = Path('/tmp/phase0-3-c05-01a1e9a9/attempt1')
OUT = Path(__file__).resolve().parent


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def csv_read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    acceptance = json.loads((RAW / 'PARENT_STORAGE_ACCEPTANCE.json').read_text())
    assert acceptance['parent_accepted'] and acceptance['verified_epoch_ACKs'] == 500
    events = jsonl(RAW / 'physical_attempts.jsonl')
    by_pid = defaultdict(list)
    for event in events:
        if event.get('pid') is not None:
            by_pid[event['pid']].append(event)
    processes = []
    for pid, rows in by_pid.items():
        starts = [row for row in rows if row['status'] == 'STARTED']
        if not starts:
            continue
        terminals = [row for row in rows if row['status'] == 'TERMINAL_CORE']
        supervisors = [row for row in rows if row['status'] == 'SUPERVISOR_OBSERVED_EXIT']
        assert len(terminals) <= 1 and len(supervisors) <= 1
        start, end, supervisor = starts[0], terminals[0] if terminals else {}, supervisors[0] if supervisors else {}
        processes.append(dict(pid=pid, purpose=start['purpose'], device=start['device'],
                              key=end.get('key') or next((row.get('key') for row in starts if row.get('key')), None),
                              STARTED_records=len(starts), terminal_present=bool(end), supervisor_present=bool(supervisor),
                              start_time=start.get('start_time'), end_time=end.get('end_time'),
                              worker_seconds=end.get('seconds'), supervisor_seconds=supervisor.get('elapsed_wall_seconds'),
                              exit_code=supervisor.get('exit_code'), returned_train_calls=end.get('returned_train_calls'),
                              unknown_partial_call=end.get('unknown_partial_call') if end else None,
                              unreturned_operations=end.get('unreturned_call_operations') if end else None,
                              null_reason=None if end else 'terminal_absent_no_imputation'))
    csv_write(OUT / 'PHYSICAL_PROCESS_SUMMARY.csv', processes)
    costs = csv_read(RAW / 'tables/COSTS.csv')
    calls = [row for row in costs if row['cost_kind'] == 'returned_train_call']
    totals = {}
    for purpose in ('formal', 'engineering_smoke'):
        rows = [row for row in calls if row['purpose'] == purpose]
        totals[purpose] = {name: sum(int(float(row[name])) for row in rows)
                           for name in ('B', 'entries', 'diagnostic_points', 'extra_autograd_sweeps', 'virtual_adam_steps')}
        totals[purpose]['returned_train_calls'] = len(rows)
    assert totals['formal']['returned_train_calls'] >= 17250
    assert totals['engineering_smoke']['returned_train_calls'] == 33
    groups = defaultdict(list)
    for row in costs:
        groups[row.get('purpose') or 'purpose_not_on_component', row['cost_kind']].append(row)
    components = []
    for (purpose, kind), rows in sorted(groups.items()):
        valid = [float(row['seconds']) for row in rows if row.get('seconds') not in ('', None)]
        components.append(dict(purpose=purpose, cost_kind=kind, records=len(rows), measured_seconds_records=len(valid),
                               sum_measured_component_seconds=sum(valid), unknown_seconds_records=len(rows) - len(valid),
                               semantics='Within-kind sum only; overlaps worker/controller/other component scopes; never sum all kinds.'))
    csv_write(OUT / 'COST_COMPONENTS.csv', components)
    progress = jsonl(RAW / 'PARENT_PROGRESS.jsonl')
    maximum_running = max((sum(j['status'] == 'RUNNING' for j in row['jobs']) for row in progress), default=0)
    per_card = {}
    for device in (0, 1):
        own = [r for r in processes if r['purpose'] == 'formal' and r['device'] == device and r['end_time'] is not None]
        changes = [(r['start_time'], 1) for r in own] + [(r['end_time'], -1) for r in own]
        active = peak = 0
        for _, change in sorted(changes, key=lambda r: (r[0], r[1])):
            active += change
            peak = max(peak, active)
        per_card[device] = dict(measured_peak_overlapping_worker_processes=peak,
                               semantics='Process lifetime includes startup/load; actual GPU residency separately in telemetry/allocation ACKs.')
    report = dict(physical_returned_operations=totals, unique_worker_processes=len(processes),
                  processes_missing_terminal=[r['pid'] for r in processes if not r['terminal_present']],
                  processes_missing_supervisor=[r['pid'] for r in processes if not r['supervisor_present']],
                  measured_worker_seconds=sum(r['worker_seconds'] for r in processes if r['worker_seconds'] is not None),
                  unknown_worker_duration_count=sum(r['worker_seconds'] is None for r in processes),
                  measured_supervisor_seconds_overlapping=sum(r['supervisor_seconds'] for r in processes if r['supervisor_seconds'] is not None),
                  maximum_running_jobs_observed_by_parent=maximum_running, per_gpu=per_card,
                  useful_formal_epochs=500, useful_formal_updates=17250,
                  parent_RAM_selected_models=20, parent_RAM_tensor_bytes=acceptance['retained_bytes'],
                  historical_retired_prelaunch='Zero registered jobs/calls/ACKs at retired source; retained original source/setup/controller receipts.',
                  null_policy='Unknown/unreturned work and missing durations remain null, never zero-imputed.',
                  time_policy='Worker, supervisor, controller and component scopes overlap; no aggregate formed by adding them.',
                  test_accessed=False)
    (OUT / 'COST_ACCOUNTING.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
