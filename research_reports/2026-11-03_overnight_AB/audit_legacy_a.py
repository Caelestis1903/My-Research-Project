"""Read-only legacy 30-run audit; never imports or loads model weights."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HISTORY = Path('/home/xzf/Code/phase0-3')
OUT = Path(__file__).resolve().parent
BASE = HISTORY / 'artifacts/mdmt_baselines_2526'
FREEZE = BASE / 'final_freeze.json'
CELLS = {(d, t) for d in range(3) for t in range(2)}


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(HISTORY))


def six_complete(metrics):
    cells = metrics.get('cells', [])
    return (len(cells) == 6 and {(cell['domain'], cell['task']) for cell in cells} == CELLS
            and all(cell.get(name) is not None and math.isfinite(cell[name])
                    for cell in cells for name in ('auc', 'logloss')))


def main():
    freeze = load(FREEZE)
    freeze_sha = sha(FREEZE)
    rows, identities = [], []
    for dataset in ('movielens', 'kuairand'):
        for method in ('M3OE_FIXED1', 'HYPERGATE', 'AUTOIFS'):
            for seed in range(2022, 2027):
                run_path = BASE / f'runs/formal/{dataset}/{method}/seed{seed}/run.json'
                curve_path = run_path.parent / 'epoch_metrics.jsonl'
                run = load(run_path)
                assert (run['dataset'], run['method'], run['seed'], run['phase']) == (dataset, method, seed, 'formal')
                stage = 'reuse' if method == 'AUTOIFS' else 'train'
                candidates = [record for record in run['epochs'] if record.get('stage') == stage]
                by_epoch = {record['epoch']: record for record in candidates}
                assert len(by_epoch) == len(candidates)
                complete = (set(by_epoch) == set(range(1, run['final_epoch'] + 1))
                            and all(six_complete(record['validation']) for record in candidates))
                curve_records = [json.loads(line) for line in curve_path.read_text().splitlines() if line.strip()]
                curve_values = {record['epoch']: record['validation'] for record in curve_records
                                if record.get('stage') == stage}
                assert curve_values == {epoch: record['validation'] for epoch, record in by_epoch.items()}
                best = None
                for record in sorted(candidates, key=lambda record: record['epoch']):
                    if best is None or record['validation']['macro_auc'] > best['validation']['macro_auc']:
                        best = record
                assert best is not None
                selected_epoch = best['epoch']
                key = f'{dataset}/{method}/seed{seed}'
                frozen = freeze['runs'][key]
                assert frozen['status'] == 'COMPLETE'
                matching = [view for view, meta in run['checkpoint_metadata'].items()
                            if meta['stage'] == stage and meta['epoch'] == selected_epoch]
                checkpoint_path = test_path = None
                matched_view = ''
                endpoint = None
                checkpoint_identity = None
                for view in matching:
                    candidate_path = run_path.parent / f'metrics_E_{view}.json'
                    checkpoint = frozen['checkpoints'][view]
                    assert checkpoint['stage'] == stage and checkpoint['epoch'] == selected_epoch
                    candidate_checkpoint = Path(checkpoint['path'])
                    if candidate_path.is_file():
                        metrics = load(candidate_path)
                        assert (metrics['dataset'], metrics['method'], metrics['seed'], metrics['role'], metrics['checkpoint_view']) == (
                            dataset, method, seed, 'E', view)
                        assert six_complete(metrics)
                        saved_validation = run['selected_validation' if view == 'selected' else 'final_validation']
                        assert saved_validation == best['validation']
                        matched_view, test_path, endpoint = view, candidate_path, metrics
                        checkpoint_path, checkpoint_identity = candidate_checkpoint, checkpoint
                        break
                state_available = bool(checkpoint_path and checkpoint_path.is_file())
                if not complete:
                    historical = 'INCOMPLETE_V_CURVE_LEGACY'
                    reason = 'FINAL_STAGE_V_CURVE_DOES_NOT_COVER_OBSERVED_WINDOW'
                elif endpoint is not None:
                    historical = 'RESELECT_ONLY_LEGACY'
                    reason = 'AUC_ARGMAX_MATCHES_FROZEN_CHECKPOINT_AND_COMPLETE_SAME_EPOCH_E_ENDPOINT'
                else:
                    historical = 'MISSING_SELECTED_STATE_LEGACY'
                    reason = 'AUC_ARGMAX_NOT_AMONG_SAVED_SELECTED_OR_FIXED_FINAL_CHECKPOINTS_OR_COMPLETE_E_ENDPOINTS'
                counts = [0, 0, 0]
                for alias in run['config']['roles']['train']:
                    for role in run['data_manifest']['role_map'][alias]:
                        counts = [a + b for a, b in zip(counts, run['data_manifest']['roles'][role]['domain_counts'])]
                assert run['config']['roles']['preprocess_fit'] == 'T0'
                assert counts != ([155949, 318599, 325619] if dataset == 'movielens' else [172782, 1743898, 89452])
                row = dict(dataset=dataset, method=method, seed=seed, action='INCOMPATIBLE',
                    main_eligible=False, population_match=False, preprocess_match=False,
                    main_reason='TRAIN_POPULATION_AND_PREPROCESS_MISMATCH',
                    source_commit=run['source_commits']['repository_main'],
                    source_path=relative(run_path), source_sha256=sha(run_path),
                    curve_path=relative(curve_path), curve_sha256=sha(curve_path),
                    curve_complete=complete, candidate_stage=stage,
                    candidate_window=f'OBSERVED_LEGACY_TRAJECTORY:{stage}:1-{run["final_epoch"]}',
                    observed_final_epochs=run['final_epoch'], selected_epoch=selected_epoch,
                    selected_validation_macro_auc=best['validation']['macro_auc'],
                    historical_reselection=historical, reason=reason,
                    matched_checkpoint_view=matched_view, model_hash='',
                    model_hash_status='NO_TENSOR_HASH_IN_LEGACY_SCALAR_ARTIFACTS',
                    state_available=state_available,
                    checkpoint_path=relative(checkpoint_path) if checkpoint_path else '',
                    checkpoint_file_sha256=sha(checkpoint_path) if state_available else '',
                    checkpoint_identity_sha256=hashlib.sha256(json.dumps(checkpoint_identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                        if checkpoint_identity else '',
                    test_endpoint_path=relative(test_path) if test_path else '',
                    test_endpoint_sha256=sha(test_path) if test_path else '',
                    final_freeze_path=relative(FREEZE), final_freeze_sha256=freeze_sha,
                    train_domain_counts=json.dumps(counts), preprocessing_fit=run['config']['roles']['preprocess_fit'],
                    encoder_hash=run['data_manifest']['features']['encoder_hash'],
                    scaler_hash=run['data_manifest']['features']['scaler_hash'],
                    historical_metric_semantics='RAW_LOGIT_AUC_AND_STABLE_LOGIT_SOFTPLUS_LOGLOSS')
                rows.append(row)
                identities.append(dict(dataset=dataset, method=method, seed=seed,
                    selected_epoch=selected_epoch, final_stage_validation_curve=[dict(epoch=r['epoch'], validation=r['validation'])
                    for r in candidates], checkpoint_identity=checkpoint_identity, test_endpoint=endpoint))
    assert len(rows) == len({(r['dataset'], r['method'], r['seed']) for r in rows}) == 30
    assert all(not row['main_eligible'] for row in rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'LEGACY_A_RESELECTION_AUDIT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'LEGACY_A_RESELECTION_EVIDENCE.json').write_text(json.dumps(identities, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    counts = Counter(row['historical_reselection'] for row in rows)
    report = ['# 旧A三模型30条轨迹重选审计', '',
        '逐条只读核对原formal run.json、完整最终阶段epoch日志、final_freeze身份和六格E端点。未运行训练或前向，未加载模型权重。', '',
        f'30条均有覆盖其历史实际最终阶段的完整V曲线。按该曲线严格最大AUC、并列最早重选：{counts["RESELECT_ONLY_LEGACY"]}条可直接接入同epoch完整历史E端点；{counts["MISSING_SELECTED_STATE_LEGACY"]}条缺对应选中状态/端点；不完整曲线{counts["INCOMPLETE_V_CURVE_LEGACY"]}条。', '',
        '30条主表均INCOMPATIBLE：原train为T0∪F、预处理拟合T0；域人口与本轮完整自然train不同。可重选项只进入OBSERVED_LEGACY_TRAJECTORY历史附录，保留旧LL早停及原raw-logit AUC/softplus LogLoss语义。', '',
        '旧标量收据没有tensor model hash；CSV的model_hash留空，checkpoint_file_sha256标识实际已存在的checkpoint文件，checkpoint_identity_sha256绑定原冻结的阶段/epoch/模型meta和schema。两者分别命名，文件摘要不替代tensor摘要。', '',
        '| 数据 | 方法 | 可重选seed | 缺状态seed |', '|---|---|---|---|']
    for dataset in ('movielens', 'kuairand'):
        for method in ('M3OE_FIXED1', 'HYPERGATE', 'AUTOIFS'):
            group = [row for row in rows if row['dataset'] == dataset and row['method'] == method]
            good = ','.join(str(row['seed']) for row in group if row['historical_reselection'] == 'RESELECT_ONLY_LEGACY') or '—'
            bad = ','.join(str(row['seed']) for row in group if row['historical_reselection'] == 'MISSING_SELECTED_STATE_LEGACY') or '—'
            report.append(f'| {dataset} | {method} | {good} | {bad} |')
    report.extend(['', '逐条路径、SHA256、完整候选窗口、选中epoch、保存视图及E端点见LEGACY_A_RESELECTION_AUDIT.csv；对应V曲线、原冻结checkpoint身份和完整E指标见LEGACY_A_RESELECTION_EVIDENCE.json。重新运行本目录audit_legacy_a.py可从只读历史产物复建此审计。', ''])
    (OUT / 'LEGACY_A_AUDIT.md').write_text('\n'.join(report))
    print(json.dumps(dict(rows=len(rows), historical_reselection=dict(counts), main_eligible=0)))


if __name__ == '__main__':
    main()
