"""Numerical costs of the retired RAM-lost Cycle04 attempt; no model loading."""
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path

R=Path(__file__).resolve().parent
N=R.parents[1]/'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1'


def main():
    spec=importlib.util.spec_from_file_location('cycle04_analysis',R/'ANALYSIS.py')
    analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
    read=lambda p:[json.loads(line) for line in p.read_text().splitlines()]
    events=read(N/'physical_attempts.jsonl');attempts=analysis.physical_attempts(events)
    batches=Counter();diagnostics=Counter();rows=Counter();batch_seconds=Counter()
    for path in sorted((N/'runs').glob('*/attempt*/batches_epoch_*.jsonl.gz')):
        with gzip.open(path,'rt') as f:
            for line in f:
                x=json.loads(line);k=str(x['attempt']);batches[k]+=1;rows[k]+=x['B']
                diagnostics[k]+=int(x.get('diagnose',False) or bool(x.get('diagnostics')))
                if x.get('seconds') is not None:batch_seconds[k]+=x['seconds']
    loss=json.loads((N.parent/'RAM_STATE_LOSS.json').read_text())
    foreground=read(N/'foreground_runs.jsonl')
    result=dict(retired_attempt='attempt1',source_hash=json.loads((N/'source_receipt.json').read_text())['sha256'],
        terminal_reason='RAM_STATE_UNAVAILABLE',historical_committed_epochs=275,historical_committed_updates=7733,
        available_training_states=0,current_attempt_useful_budget_contribution=0,
        physical_formal_processes=len(attempts),physical_status_counts=dict(Counter(x['status'] for x in attempts)),
        measured_worker_seconds=sum(float(x['seconds']) for x in attempts if x.get('seconds') not in ('',None)),
        missing_worker_terminal_records=sum(x['worker_terminal_missing'] for x in attempts),
        supervisor_process_wall_seconds=sum(float(x['supervisor_process_wall_seconds']) for x in attempts if x['supervisor_process_wall_seconds'] not in ('',None)),
        timing_semantics='supervisor process wall includes loading/exit and overlaps worker seconds; never add these totals',
        logged_returned_training_calls_by_physical_attempt=dict(batches),logged_training_rows_by_physical_attempt=dict(rows),
        logged_batch_seconds_by_physical_attempt=dict(batch_seconds),
        logged_diagnostic_record_presence_by_attempt=dict(diagnostics),
        unknown_partial_formal_process_records=sum(bool(x.get('unknown_partial_call')) for x in attempts),
        unknown_partial_semantics='failure flags are process observations, not an exact count or zero-cost estimate of unreturned Adam calls',
        foreground_runs=foreground,test_accessed=False)
    (R/'RETIRED_ATTEMPT_COSTS.json').write_text(json.dumps(result,indent=2)+'\n')
    analysis.dump_csv('RETIRED_PHYSICAL_ATTEMPTS.csv',attempts)
    print(json.dumps(result))


if __name__=='__main__':main()
