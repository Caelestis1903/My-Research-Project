"""Foreground parent scientific arithmetic after complete, accepted C05 RUN."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = Path('/tmp/phase0-3-c05-01a1e9a9/attempt1')
TABLES = RAW / 'tables'
U, C, W, S = 'CELL-MASS-BCE-025', 'CLEAN', 'COMPLETION-BCE-025', 'MASS-SCALE-025'
ARMS = (C, S, W, U)
DATASETS = ('movielens', 'kuairand')
ENDPOINTS = ('selected', 'E1', 'E10', 'E30', 'E50')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(name):
    with (TABLES / (name + '.csv')).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    accepted = json.loads((RAW / 'PARENT_STORAGE_ACCEPTANCE.json').read_text())
    assert accepted['parent_accepted'] and accepted['verified_epoch_ACKs'] == 500
    assert accepted['verified_selected_models'] == 20 and accepted['useful_adam_steps'] == 17250
    receipt = json.loads((TABLES / 'TABLE_RECEIPT.json').read_text())
    assert receipt['test_accessed'] is False
    for item in receipt['files'].values():
        assert sha(Path(item['path'])) == item['sha256']
    endpoints, pairs, summaries = read('ENDPOINTS'), read('ALL_PAIRED_SEEDS'), read('ALL_PAIRED_SUMMARY')
    assert (len(endpoints), len(pairs), len(summaries)) == (400, 28900, 5780)
    fields = ('mean', 'sd', 'ci95_low', 'ci95_high')
    for row in summaries:
        assert row['n'] == '5' and row['available'] == 'True'
        for field in fields:
            row[field] = float(row[field])
        for field in ('positive', 'negative', 'zero'):
            row[field] = int(row[field])
    index = {(r['dataset'], r['comparison'], r['endpoint'], r['metric']): r for r in summaries}
    def contrast(dataset, left, right, metric='macro_auc', endpoint='selected', lk='fixed_ema', rk='fixed_ema'):
        symbols = {U: 'U', C: 'C', W: 'W', S: 'S'}
        return index[dataset, f'{symbols[left]}_{lk}-{symbols[right]}_{rk}', endpoint, metric]
    primary = {}
    for dataset in DATASETS:
        auc, loss = (contrast(dataset, U, C, metric) for metric in ('macro_auc', 'macro_logloss'))
        cells = {cell: contrast(dataset, U, C, cell + '_auc') for cell in CELLS}
        floor = min(row['mean'] for row in cells.values())
        primary[dataset] = dict(auc=auc, logloss=loss, cell_auc=cells, minimum_cell_mean_delta=floor,
                                macro_target_met=auc['mean'] >= .003, cell_protection_met=floor >= -.001,
                                qualified=auc['mean'] >= .003 and floor >= -.001,
                                mean_loss_decrease=loss['mean'] < 0)
    positive = sum(v['auc']['mean'] > 0 for v in primary.values())
    outcome = ('NO_CONTINUATION_SIGNAL' if positive == 0 else 'MIXED' if positive == 1 else
               'INTERNAL_SCREEN_QUALIFIED' if all(v['qualified'] for v in primary.values()) else 'POSITIVE_UNQUALIFIED')
    f2 = {d: contrast(d, U, S, 'task1_macro_logloss') for d in DATASETS}
    f3 = {m: contrast('kuairand', U, W, 'd0t0_' + m) for m in ('auc', 'logloss')}
    means = []
    metrics = [name for name in endpoints[0] if name.endswith(('_auc', '_logloss'))] + ['epoch']
    for dataset in DATASETS:
        for arm in ARMS:
            for kind in ('online', 'fixed_ema'):
                for endpoint in ENDPOINTS:
                    rows = [r for r in endpoints if (r['dataset'], r['arm'], r['estimator'], r['endpoint']) == (dataset, arm, kind, endpoint)]
                    assert sorted(int(r['seed']) for r in rows) == list(range(2022, 2027))
                    for metric in metrics:
                        values = [float(r[metric]) for r in rows]
                        means.append(dict(dataset=dataset, arm=arm, estimator=kind, endpoint=endpoint, metric=metric,
                                          n=5, mean=statistics.mean(values), sd=statistics.stdev(values), min=min(values), max=max(values)))
    write('ENDPOINT_MEANS.csv', means)
    selected = [r for r in summaries if r['endpoint'] == 'selected']
    write('SELECTED_PAIRED_SUMMARY.csv', selected)
    relevant = [r for r in summaries if r['metric'] in ('macro_auc', 'macro_logloss', 'task0_macro_auc', 'task1_macro_auc', 'task0_macro_logloss', 'task1_macro_logloss')]
    write('STAGE_CONTRASTS.csv', relevant)
    diagnostic_rows = [r for r in read('DIAGNOSTIC_MEANS') if r['parameter_group'] == 'ALL']
    write('COMMON_STATE_ALL_PARAMETERS.csv', diagnostic_rows)
    for name in ('BASELINE_DOMAIN_CONTEXT.csv', 'BASELINE_CELL_CONTEXT.csv'):
        shutil.copyfile(ROOT / 'research_reports/completion_weight_attribution_20261117' / name, OUT / name)
    fixed_order_checks = 0
    for dataset in DATASETS:
        for seed in range(2022, 2027):
            for endpoint in ENDPOINTS[1:]:
                matched = [r for r in endpoints if (r['dataset'], int(r['seed']), r['endpoint'], r['estimator']) == (dataset, seed, endpoint, 'fixed_ema')]
                assert len(matched) == 4 and len({r['epoch_order_hash'] for r in matched}) == 1
                fixed_order_checks += 1
    appendix = ['# C05 完整数值附录', '', '配对单位为五个 seed；CI 为描述性 t4 区间；LL 绑定同一 AUC 选中状态。', '']
    for dataset in DATASETS:
        for endpoint in ENDPOINTS:
            for kind in ('online', 'fixed_ema'):
                appendix += [f'## {dataset} / {endpoint} / {kind}', '', '| Arm | seed | epoch | Macro AUC | Macro LL | 状态 SHA256 |', '|---|---:|---:|---:|---:|---|']
                for row in endpoints:
                    if (row['dataset'], row['endpoint'], row['estimator']) == (dataset, endpoint, kind):
                        appendix.append('| ' + ' | '.join(row[k] for k in ('arm', 'seed', 'epoch', 'macro_auc', 'macro_logloss', 'state_hash')) + ' |')
                appendix.append('')
    appendix += ['## selected：全部六个 cell', '', '| dataset | arm | estimator | seed | cell | AUC | LL |', '|---|---|---|---:|---|---:|---:|']
    for row in endpoints:
        if row['endpoint'] == 'selected':
            for cell in CELLS:
                appendix.append('| ' + ' | '.join([row[k] for k in ('dataset', 'arm', 'estimator', 'seed')] + [cell, row[cell + '_auc'], row[cell + '_logloss']]) + ' |')
    (OUT / 'NUMERICAL_APPENDIX.md').write_text('\n'.join(appendix) + '\n')
    facts = dict(candidate=U, decision=outcome, primary=primary, F2_task1_loss_U_minus_S=f2,
                 F2_direction_both_datasets=all(v['mean'] < 0 for v in f2.values()),
                 F3_kuairand_d0t0_U_minus_W=f3, F3_rescue_direction=f3['auc']['mean'] > 0 and f3['logloss']['mean'] < 0,
                 new_trajectories=10, reused_numeric_trajectories=30, unique_trajectories=40, paired_n=5,
                 formal_epochs=500, useful_updates=17250, selected_models_verified=20,
                 source_hash=receipt['source_hash'], design_bundle_sha256=receipt['design_bundle_sha256'],
                 table_receipt_sha256=sha(TABLES / 'TABLE_RECEIPT.json'),
                 parent_acceptance_sha256=sha(RAW / 'PARENT_STORAGE_ACCEPTANCE.json'),
                 fixed_stage_matched_epoch_order_groups=fixed_order_checks,
                 selected_ram_available_at_acceptance=True, test_accessed=False, champion_selected=False,
                 scheduled_stop_after_delivery=True, next_cycle_authorized=False,
                 inference_limits='Five paired seeds; exploratory related contrasts; reused controls, no independent replication or mediation identification.')
    (OUT / 'DECISION.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        for left, right in ((U, C), (U, W), (U, S)):
            rows = [contrast(dataset, left, right, endpoint=endpoint) for endpoint in ENDPOINTS]
            ax.errorbar(range(5), [r['mean'] for r in rows], yerr=[[r['mean'] - r['ci95_low'] for r in rows], [r['ci95_high'] - r['mean'] for r in rows]], marker='o', capsize=2, label=f'U-{dict(zip((C,W,S),"CWS"))[right]}')
        ax.axhline(0, color='grey', linewidth=.7)
        ax.set_xticks(range(5), ENDPOINTS)
        ax.set_title(dataset)
        ax.set_ylabel('Paired Macro AUC delta (EMA), mean / t4 CI')
        ax.legend()
    fig.savefig(OUT / 'STAGE_AUC.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        for shift, (right, label) in enumerate(((C, 'C'), (W, 'W'), (S, 'S'))):
            rows = [contrast(dataset, U, right, cell + '_auc') for cell in CELLS]
            xs = [i + (shift - 1) * .2 for i in range(6)]
            ax.errorbar(xs, [r['mean'] for r in rows], yerr=[[r['mean'] - r['ci95_low'] for r in rows], [r['ci95_high'] - r['mean'] for r in rows]], fmt='o', capsize=2, label='U-' + label)
        ax.axhline(0, color='grey', linewidth=.7)
        ax.axhline(-.001, color='red', linestyle=':', linewidth=.7)
        ax.set_xticks(range(6), CELLS)
        ax.set_title(dataset)
        ax.set_ylabel('Selected EMA cell AUC delta')
        ax.legend()
    fig.savefig(OUT / 'CELL_AUC.png', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(decision=outcome, primary=primary, F2=f2, F3=f3), ensure_ascii=False))


if __name__ == '__main__':
    main()
