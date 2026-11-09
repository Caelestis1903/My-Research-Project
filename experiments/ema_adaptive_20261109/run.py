"""A0 CLI: source audit, frozen numerical analysis, appendix and release."""
import argparse
import resource
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import analysis
from .audit import Incomplete, ResourceBlock, REPORT, code_identity, read_json, resolve, sha, write_json


def appendix(out):
    decision = read_json(out / 'A0_DECISION.json')
    frame = pd.read_csv(out / 'A0_PREDICTION_SUMMARY.csv')
    select = frame[(frame.dataset == 'kuairand') & (frame.objective == 'gaussian_d0') &
        (frame.feature_panel == 'P_A') & frame.score_panel.isin(['P_B', 'V_A', 'V_B']) &
        frame.scope.isin(['GLOBAL', 'TASK0', 'TASK1']) & frame.model_id.isin(['C_design', 'C_simple', 'C_persist', 'C_carry']) &
        frame.metric.isin(['chosen_minus_F', 'MAE_D', 'chosen_minus_C_design', 'chosen_minus_C_simple', 'chosen_minus_uniform'])]
    columns = ['score_panel', 'scope', 'model_id', 'metric', 'mean', 'sample_sd', 'ci95_low', 'ci95_high']
    table = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---'] * len(columns)) + ' |\n'
    for row in select[columns].itertuples(index=False, name=None):
        table += '| ' + ' | '.join(f'{x:.9g}' if isinstance(x, float) else str(x) for x in row) + ' |\n'
    text = '# A0 numerical appendix\n\n'
    text += f'Machine decision: `{decision["state"]}`. Scientific unit: seed; seeds 2022–2026.\n\n'
    text += 'Each pulse lasts one epoch; adjacent observations are five epochs apart. '
    text += 'New fits use four other seeds’ early training-panel features and labels; held-seed late P_B/Validation are score-only.\n\n'
    text += table + '\n'
    text += 'Complete raw values and identities: A0_S_CONTRASTS.csv, A0_PERSISTENCE_ROWS.csv, A0_PREDICTIVE_OOF.csv, A0_STATIC_ACTIONS.csv.\n'
    text += 'Seed values and paired differences: A0_SEED.csv, A0_PREDICTION_SEED.csv, A0_STATIC_SEED.csv; all numerical stage gates: A0_DECISION.json.\n'
    target = out / 'A0_NUMERICAL_APPENDIX.md'
    target.write_text(text)
    write_json(out / 'REPORT_RECEIPT.json', {'numerical_appendix_sha256': sha(target), 'decision_sha256': sha(out / 'A0_DECISION.json'),
               'final_scientific_report': str(REPORT / 'FINAL_REPORT.md'), 'scientific_narrative_owner': 'orchestrator',
               'new_GPU_seconds': 0, 'new_Adam_steps': 0, 'new_Test_accesses': 0})


def release(out):
    final = REPORT / 'FINAL_REPORT.md'
    if not final.is_file():
        raise Incomplete('Orchestrator FINAL_REPORT.md is required before release')
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'RELEASE_RECEIPT.json'}
    write_json(out / 'RELEASE_RECEIPT.json', {'status': 'RELEASED', 'final_report_sha256': sha(final),
        'implementation': code_identity(), 'artifact_hashes': files, 'model_parameter_files_written': 0,
        'checkpoint_files_written': 0, 'parameter_states_remaining': 0,
        'new_training_runs': 0, 'new_Adam_steps': 0, 'new_GPU_seconds': 0, 'new_Test_accesses': 0})


def cost(out, command, wall, cpu, exit_code):
    path = out / 'COSTS.csv'
    old = pd.read_csv(path).to_dict('records') if path.exists() else []
    output_bytes = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    old.append({'stage': 'A0', 'command': command, 'wall_seconds': wall, 'cpu_seconds': cpu,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'output_bytes': output_bytes, 'new_Adam_steps': 0, 'new_GPU_seconds': 0, 'new_Test_accesses': 0, 'exit_code': exit_code})
    pd.DataFrame(old).to_csv(path, index=False)


def resource_check(out):
    memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    if 1 - memory['MemAvailable'] / memory['MemTotal'] >= .9:
        raise ResourceBlock('Host RAM admission at or above 90%')
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) >= 2 ** 31:
        raise ResourceBlock('A0 output cap exceeded')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'analyze', 'report', 'release'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--stage', choices=['A0'], default='A0')
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    start, cpu = time.monotonic(), time.process_time()
    status = 0
    try:
        resource_check(out)
        study = resolve(args.manifest)
        if args.command == 'audit':
            from .audit import audit
            audit(study, out)
        elif args.command == 'analyze':
            analysis.analyze(study, out)
        elif args.command == 'report':
            appendix(out)
        elif not (REPORT / 'FINAL_REPORT.md').is_file():
            raise Incomplete('Orchestrator FINAL_REPORT.md is required before release')
        resource_check(out)
    except (Incomplete, ResourceBlock, FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
        status = getattr(error, 'exit_code', 5 if isinstance(error, FileNotFoundError) else 3)
        write_json(out / 'EXECUTION_ERROR.json', {'command': args.command, 'error_type': type(error).__name__, 'message': str(error), 'exit_code': status})
        if args.command in ('audit', 'analyze'):
            write_json(out / 'A0_DECISION.json', {'state': 'SOURCE_INCOMPLETE' if isinstance(error, (Incomplete, FileNotFoundError)) else ('RESOURCE_BLOCK' if status == 4 else 'IMPLEMENTATION_ERROR'), 'reason': str(error), 'new_training_triggered': False})
        print(str(error), file=sys.stderr)
    finally:
        cost(out, ' '.join(sys.argv if argv is None else argv), time.monotonic() - start, time.process_time() - cpu, status)
    if status == 0 and args.command == 'release':
        release(out)
    if status == 0:
        print(f'{args.command}: completed; outputs in {out}')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
