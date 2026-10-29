"""Recompute descriptive tables and figures from the complete frozen campaign.

This script never trains, loads model weights or chooses a checkpoint.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
from collections import Counter
import json
import math
import sys

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.mdmt_sota_baselines_2526.config import load_config
from experiments.mdmt_sota_baselines_2526.report import build_report

REPORT = ROOT / 'research_reports/mdmt_baselines_2526'
ARTIFACT = ROOT / 'artifacts/mdmt_baselines_2526'
OUTPUT = REPORT / 'analysis'
METHODS = ('M3OE_FIXED1', 'HYPERGATE', 'AUTOIFS')
DATASETS = ('movielens', 'kuairand')
SEEDS = (2022, 2023, 2024, 2025, 2026)
VIEWS = ('selected', 'fixed_final')
METRICS = ('macro_auc', 'macro_logloss', 'natural_auc', 'natural_logloss')


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(name, rows):
    if not rows:
        raise ValueError('No rows for ' + name)
    with (OUTPUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite independent-seed values are required')
    sd = float(x.std(ddof=1))
    margin = float(t.ppf(.975, len(x)-1) * sd / math.sqrt(len(x)))
    return dict(n=len(x), mean=float(x.mean()), sample_sd=sd,
                ci95_low=float(x.mean()-margin), ci95_high=float(x.mean()+margin),
                positive_seeds=int((x > 0).sum()), negative_seeds=int((x < 0).sum()))


def cell(metric, domain, task):
    values = [c for c in metric['cells'] if c['domain'] == domain and c['task'] == task]
    assert len(values) == 1
    return values[0]


def main():
    check = describe([1, 2, 3, 4, 5])
    assert check['mean'] == 3 and math.isclose(check['sample_sd'], math.sqrt(2.5))
    cfg = load_config()
    authorization = read(REPORT / 'RUN_AUTHORIZATION.json')
    frozen = read(ARTIFACT / 'final_freeze.json')
    assert authorization['status'] == 'USER_APPROVED_RUN'
    assert frozen['status'] == 'FROZEN' and len(frozen['runs']) == 30
    OUTPUT.mkdir(exist_ok=True)
    training_profiles, unsuccessful_profiles = {}, []
    profile_paths = list((REPORT / 'runtime_crash3').glob('*.json'))
    profile_paths += list((REPORT / 'runtime_crash5').glob('*_gdb.json'))
    for path in sorted(profile_paths):
        profile = read(path)
        if profile.get('cli_command') == 'train':
            assert profile['profile'] == 'D1'
            if profile['status'] == 'CLI_EXITED':
                assert profile['entrypoint_exit_code'] == 0
            else:
                recovered = read(Path(profile['run_dir'])/'run.json')
                if not recovered.get('posttraining_recovery'):
                    unsuccessful_profiles.append(str(path.relative_to(ROOT)))
                    continue
                assert profile['status'] == 'LAUNCH_STARTED'
                assert (recovered['dataset'], recovered['method'], recovered['seed']) == ('kuairand','AUTOIFS',2022)
            assert profile['torch_threads'] == {'intra_op':1, 'inter_op':1}
            assert profile['configuration'] == cfg
            key = str(Path(profile['run_dir']).resolve())
            assert key not in training_profiles
            training_profiles[key] = str(path.relative_to(ROOT))
    assert len(training_profiles) == 7

    results, seed_rows, costs, validation, profiles = {}, [], [], [], []
    for dataset in DATASETS:
        data_root = ARTIFACT / 'data' / dataset
        data_manifest = read(data_root / 'DATA_MANIFEST.json')
        features = np.load(data_root / 'x.npy', allow_pickle=False)
        features.setflags(write=False)
        domains = features[:, -1].astype(np.int64)
        columns = data_manifest['features']['columns']
        user_column = columns.index('user_id') if 'user_id' in columns else None
        role_map = {'T0':('fit',), 'F':('P_label',), 'T0_plus_F':('fit','P_label'),
                    'S':('P_cal','P_update'), 'Q':('Q_explore',), 'V':('validation',),
                    'E':('test',), 'legacy':('Q_confirm',)}
        for role, components in role_map.items():
            ids = np.sort(np.concatenate([np.load(data_root/f'rows_{r}.npy') for r in components]))
            for domain in range(3):
                chosen = ids[domains[ids] == domain]
                oov = float(np.mean(features[chosen,user_column] == 0)) if user_column is not None else None
                for task in range(2):
                    positive = sum(cell(data_manifest['roles'][r],domain,task)['positive'] for r in components)
                    profiles.append(dict(dataset=dataset, role=role, domain=domain, task=task,
                                         rows=len(chosen), positive=positive,
                                         positive_rate=positive/len(chosen), user_id_oov_rate=oov))
        for method in METHODS:
            for seed in SEEDS:
                directory = ARTIFACT / 'runs/formal' / dataset / method / f'seed{seed}'
                run = read(directory / 'run.json')
                assert run['status'] == 'TRAIN_COMPLETE_NOT_EVALUATED'
                assert run['config'] == cfg and run['phase'] == 'formal'
                for view in VIEWS:
                    metric = read(directory / f'metrics_E_{view}.json')
                    assert metric['role'] == 'E' and metric['checkpoint_view'] == view
                    assert (metric['method'], metric['dataset'], metric['seed']) == (method, dataset, seed)
                    assert len(metric['cells']) == 6
                    assert math.isclose(np.mean([c['logloss'] for c in metric['cells']]), metric['macro_logloss'], abs_tol=1e-12)
                    assert math.isclose(np.mean([c['auc'] for c in metric['cells']]), metric['macro_auc'], abs_tol=1e-12)
                    results[dataset, method, seed, view] = metric
                    seed_rows.append(dict(dataset=dataset, method=method, seed=seed, view=view,
                                          selected_epoch=run['selected_epoch'], final_epoch=run['final_epoch'],
                                          **{m:metric[m] for m in METRICS},
                                          pooled_auc_task0=metric['pooled_auc_by_task'][0],
                                          pooled_auc_task1=metric['pooled_auc_by_task'][1]))
                for view, field in (('selected', 'selected_validation'), ('fixed_final', 'final_validation')):
                    validation.append(dict(dataset=dataset, method=method, seed=seed, view=view,
                                           epoch=run['selected_epoch'] if view == 'selected' else run['final_epoch'],
                                           **{m:run[field][m] for m in METRICS}))

    # Cost includes every development and formal training trajectory, with
    # AutoIFS search and reuse counted separately from its two Adam calls.
    for path in sorted((ARTIFACT / 'runs').glob('*/*/*/seed*/run.json')):
        run = read(path)
        if run['status'] != 'TRAIN_COMPLETE_NOT_EVALUATED':
            continue
        epochs = run['epochs']
        search = [e for e in epochs if e.get('stage') == 'search']
        reuse = [e for e in epochs if e.get('stage') == 'reuse']
        standard = [e for e in epochs if e.get('stage') == 'train']
        optimizer_calls = sum(e['updates']*2 for e in search) + sum(e['updates'] for e in reuse+standard)
        setup = next((e for e in epochs if e.get('stage') == 'search_setup'), {})
        timing = run['inference_timing']
        key = f"{run['dataset']}/{run['method']}/seed{run['seed']}"
        revision = frozen['runs'][key].get('source_revision', 'single_source') if run['phase']=='formal' else 'R1'
        costs.append(dict(phase=run['phase'], dataset=run['dataset'], method=run['method'], seed=run['seed'],
                          source_revision=revision,
                          cpu_runtime_profile=('D1_measured' if str(path.parent.resolve()) in training_profiles
                                               else 'OMP4_MKL4_launch_effective_counts_unmeasured'),
                          runtime_profile_sidecar=training_profiles.get(str(path.parent.resolve())),
                          native_debugger_training=('runtime_crash5/' in training_profiles.get(str(path.parent.resolve()), '')),
                          selected_epoch=run['selected_epoch'], final_epoch=run['final_epoch'],
                          search_epochs=len(search), reuse_epochs=len(reuse), standard_epochs=len(standard),
                          total_training_epochs=len(search)+len(reuse)+len(standard),
                          batch_updates=run['global_steps'], optimizer_step_calls=optimizer_calls,
                          valid_label_exposures=sum(e['valid_label_exposures'] for e in search+reuse+standard),
                          training_seconds=run['training_seconds'], total_elapsed_seconds=run['total_elapsed_seconds'],
                          logged_epoch_seconds_lower_bound=run.get('logged_epoch_seconds_lower_bound'),
                          original_process_elapsed_seconds=run.get('original_process_elapsed_seconds'),
                          posttraining_recovery=run.get('posttraining_recovery'),
                          posttraining_recovery_seconds=run.get('posttraining_recovery_seconds'),
                          parameters=run['parameter_count'],
                          search_base_parameters=setup.get('base_parameter_count'),
                          search_selector_parameters=setup.get('gate_parameter_count'),
                          reused_frozen_embedding_parameters=(run['parameter_count']-setup['base_parameter_count']-
                                                              setup['gate_parameter_count'] if setup else None),
                          peak_allocated_mib=(run['peak_cuda_bytes']/2**20 if run['peak_cuda_bytes'] is not None else None),
                          inference_p50_ms=timing['p50_ms'], inference_p95_ms=timing['p95_ms'],
                          inference_batch=timing['batch_size'], inference_repetitions=timing['repetitions'],
                          inference_warmup=timing['warmup'], device=timing['device']))

    summaries, paired, paired_summary, cells, tasks = [], [], [], [], []
    for dataset in DATASETS:
        manifest = read(ARTIFACT / 'data' / dataset / 'DATA_MANIFEST.json')
        train_counts = np.array(manifest['roles']['fit']['domain_counts']) + np.array(manifest['roles']['P_label']['domain_counts'])
        for view in VIEWS:
            for method in METHODS:
                for name in METRICS:
                    summaries.append(dict(dataset=dataset, method=method, view=view, metric=name,
                                          **describe([results[dataset, method, seed, view][name] for seed in SEEDS])))
            for method in METHODS[1:]:
                deltas = {m:[] for m in METRICS}
                for seed in SEEDS:
                    candidate = results[dataset, method, seed, view]
                    baseline = results[dataset, 'M3OE_FIXED1', seed, view]
                    delta = {m:candidate[m]-baseline[m] for m in METRICS}
                    paired.append(dict(dataset=dataset, method=method, seed=seed, view=view, **delta))
                    for name in METRICS:
                        deltas[name].append(delta[name])
                for name in METRICS:
                    paired_summary.append(dict(dataset=dataset, method=method, view=view, metric=name, **describe(deltas[name])))
                for task in range(2):
                    task_auc, task_ll, pooled = [], [], []
                    for seed in SEEDS:
                        a = results[dataset, method, seed, view]
                        b = results[dataset, 'M3OE_FIXED1', seed, view]
                        task_auc.append(np.mean([cell(a,d,task)['auc']-cell(b,d,task)['auc'] for d in range(3)]))
                        task_ll.append(np.mean([cell(a,d,task)['logloss']-cell(b,d,task)['logloss'] for d in range(3)]))
                        pooled.append(a['pooled_auc_by_task'][task]-b['pooled_auc_by_task'][task])
                    tasks.append(dict(dataset=dataset, method=method, view=view, task=task,
                                      delta_macro_auc=float(np.mean(task_auc)), delta_macro_logloss=float(np.mean(task_ll)),
                                      delta_pooled_auc=float(np.mean(pooled)),
                                      auc_positive_seeds=sum(x>0 for x in task_auc), ll_negative_seeds=sum(x<0 for x in task_ll)))
                    for domain in range(3):
                        a = [cell(results[dataset,method,s,view],domain,task) for s in SEEDS]
                        b = [cell(results[dataset,'M3OE_FIXED1',s,view],domain,task) for s in SEEDS]
                        da = describe([x['auc']-y['auc'] for x,y in zip(a,b)])
                        dl = describe([x['logloss']-y['logloss'] for x,y in zip(a,b)])
                        cells.append(dict(dataset=dataset, method=method, view=view, domain=domain, task=task,
                                          train_rows=int(train_counts[domain]), evaluation_rows=b[0]['n'],
                                          positive_rate=b[0]['positive']/b[0]['n'],
                                          baseline_auc=float(np.mean([c['auc'] for c in b])),
                                          baseline_logloss=float(np.mean([c['logloss'] for c in b])),
                                          candidate_auc=float(np.mean([c['auc'] for c in a])),
                                          candidate_logloss=float(np.mean([c['logloss'] for c in a])),
                                          **{'delta_auc_'+k:v for k,v in da.items()},
                                          **{'delta_logloss_'+k:v for k,v in dl.items()}))

    for name,rows in [('SEED_METRICS.csv',seed_rows), ('SEED_SUMMARY.csv',summaries),
                      ('PAIRED_SEED_DELTAS.csv',paired), ('PAIRED_SUMMARY.csv',paired_summary),
                      ('CELL_COMPARISONS.csv',cells), ('TASK_COMPARISONS.csv',tasks),
                      ('VALIDATION_STAGE.csv',validation), ('COSTS.csv',costs),
                      ('DATA_PROFILES.csv',profiles)]:
        write_csv(name,rows)
    assert len(seed_rows) == 60 and len(paired) == 40 and len(costs) == 38

    build_report(ARTIFACT / 'runs', REPORT, config=cfg)
    manifest_path = REPORT / 'experiment_manifest.json'
    manifest = read(manifest_path)
    manifest['reviewed_configuration_run_status'] = manifest['run_status']
    manifest['run_status'] = 'FORMAL_COMPLETE_HISTORICAL_E_COMPLETE'
    manifest['user_authorization'] = authorization
    manifest['pre_formal_freeze'] = 'artifacts/mdmt_baselines_2526/PRE_FORMAL_FREEZE.json'
    manifest['final_freeze'] = 'artifacts/mdmt_baselines_2526/final_freeze.json'
    manifest['E_evaluations_complete'] = 60
    source_revisions = dict(Counter(entry.get('source_revision', 'single_source') for entry in frozen['runs'].values()))
    manifest['formal_source_revisions'] = source_revisions
    if 'runtime_io_amendment' in frozen:
        manifest['runtime_io_amendment'] = 'research_reports/mdmt_baselines_2526/RUNTIME_IO_SOURCE_EQUIVALENCE.json'
    incident_path = REPORT / 'RUN_INCIDENTS.json'
    incidents = read(incident_path)['incidents'] if incident_path.exists() else []
    abandoned = [r for r in incidents if not r.get('recovered_existing_training_state', False)]
    manifest['failed_attempts'] = incidents
    manifest['runtime_thread_provenance'] = {
        'review': 'research_reports/mdmt_baselines_2526/RUNTIME_CRASH3_REVIEW_REPORT.md',
        'diagnostics': 'research_reports/mdmt_baselines_2526/CRASH3_DIAGNOSTIC_RAW_RESULTS.json',
        'replay': 'research_reports/mdmt_baselines_2526/CRASH3_REPLAY_EVIDENCE.json',
        'formal_D1_sidecars': list(training_profiles.values()),
        'unsuccessful_D1_training_sidecars': unsuccessful_profiles,
        'note': 'Original run identities are unchanged. Historical effective thread counts were not measured; D4 diagnostic counts are not retroactive measurements.'}
    for entry in manifest['formal_seed_matrix']:
        entry['training_receipt_status'] = entry['status']
        entry['status'] = 'FORMAL_E_COMPLETE'
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    table = REPORT / 'baseline_comparison.md'
    lines, formal = [], False
    for line in table.read_text().splitlines():
        if line.startswith('## '):
            formal = line.startswith('## Formal ')
        if formal:
            line = line.replace('| TRAIN_COMPLETE_NOT_EVALUATED |', '| FORMAL_E_COMPLETE |')
        lines.append(line)
    table.write_text('\n'.join(lines)+'\n')

    summary = dict(status=manifest['run_status'], recorded_utc=datetime.now(timezone.utc).isoformat(),
                   development_trajectories=8, formal_trajectories=30, formal_E_views=60,
                   known_training_seconds_by_phase={p:sum(r['training_seconds'] for r in costs
                                                         if r['phase']==p and r['training_seconds'] is not None)
                                              for p in ('development','formal')},
                   exact_total_training_seconds=(sum(r['training_seconds'] for r in costs)
                                                 if all(r['training_seconds'] is not None for r in costs) else None),
                   measured_training_cost_count=sum(r['training_seconds'] is not None for r in costs),
                   missing_exact_training_cost_count=sum(r['training_seconds'] is None for r in costs),
                   successful_training_seconds_lower_bound=sum(
                       r['training_seconds'] if r['training_seconds'] is not None
                       else r['logged_epoch_seconds_lower_bound'] for r in costs),
                   batch_updates=sum(r['batch_updates'] for r in costs),
                   optimizer_step_calls=sum(r['optimizer_step_calls'] for r in costs),
                   native_process_incident_count=len(incidents),
                   abandoned_training_attempt_count=len(abandoned),
                   total_training_attempts=len(costs)+len(abandoned),
                   crash3_diagnostic_process_seconds=sum(
                       p['process_execution']['elapsed_seconds'] for p in
                       read(REPORT/'CRASH3_DIAGNOSTIC_RAW_RESULTS.json')['profiles'].values()),
                   crash4_gdb_wall_seconds=read(REPORT/'runtime_crash4/gdb_execution.json')['wall_elapsed_seconds'],
                   posttraining_recovery_seconds=sum(r['posttraining_recovery_seconds'] or 0 for r in costs),
                   formal_source_revisions=source_revisions,
                   successful_cpu_runtime_profiles=dict(Counter(r['cpu_runtime_profile'] for r in costs)),
                   native_debugger_training_trajectories=sum(r['native_debugger_training'] for r in costs),
                   abandoned_logged_epoch_seconds_lower_bound=sum(r['logged_epoch_seconds_lower_bound'] for r in abandoned),
                   failure_cost_note='Native process incidents are not independent seeds. An existing recovered training trajectory is counted once; abandoned attempts add a separate logged-time lower bound. Missing exact training time/peak memory is NA.',
                   interval_note='Descriptive paired t intervals across five seeds, df=4, unadjusted for multiple comparisons.',
                   configuration_note='HyperGate remains an unverified-parameter official-core adaptation; historical E is not a fresh confirmation set.')
    (OUTPUT/'EXECUTION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    make_figures(cells)
    print(json.dumps(summary,indent=2))


def make_figures(cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('auc','logloss'):
        fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True)
        for i,dataset in enumerate(DATASETS):
            for j,method in enumerate(METHODS[1:]):
                ax=axes[i,j]
                selected=[c for c in cells if c['dataset']==dataset and c['method']==method and c['view']=='selected']
                selected=sorted(selected,key=lambda c:(c['task'],c['domain']))
                y=np.array([c[f'delta_{metric}_mean'] for c in selected])
                low=np.array([c[f'delta_{metric}_ci95_low'] for c in selected])
                high=np.array([c[f'delta_{metric}_ci95_high'] for c in selected])
                ax.axhline(0,color='black',linewidth=.8)
                ax.errorbar(np.arange(6),y,yerr=np.array([y-low,high-y]),fmt='o',capsize=3)
                ax.set_xticks(np.arange(6),[f"d{c['domain']}t{c['task']}" for c in selected])
                ax.set_title(f'{dataset} / {method} minus M3OE_FIXED1')
                ax.set_ylabel('Delta AUC (higher is better)' if metric=='auc' else 'Delta LogLoss (lower is better)')
                ax.grid(axis='y',alpha=.2)
        fig.suptitle('Selected historical E; paired seed means and unadjusted 95% t intervals')
        fig.tight_layout()
        fig.savefig(OUTPUT/f'cell_delta_{metric}.png',dpi=180)
        fig.savefig(OUTPUT/f'cell_delta_{metric}.pdf')
        plt.close(fig)


if __name__ == '__main__':
    main()
