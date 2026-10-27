"""Read existing receipts/logs and export costs; never import training code."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json

REPORT = Path(__file__).resolve().parent
WORKSPACE = REPORT.parents[1]
ROOT = WORKSPACE / 'artifacts/mdmt_relation_basis_v2'
ACTIVE = ROOT / 'revision1'
DATASETS = ('movielens', 'kuairand')
SEEDS = (2022, 2023, 2024, 2025)
METHODS = ('B_REFERENCE', 'DIRECT', 'READ', 'TARGET_ONLY', 'LOWRANK_RELATION', 'FULL_RELATION')
SOURCES = {'movielens':'72f3e825dac422784fd575bd24450e7858e99ce138f6c8261b3ae7636b8b13fc',
           'kuairand':'26708434c81106dd424d52b4de258bee8aad7b35640b1d50b8699290694a2a3c'}
TRAINING_SOURCE = '29d7018f037c06af567d58646c6f81d022cab101a0e87c86c99af295e751cd98'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def relative(path):
    return str(path.relative_to(WORKSPACE))


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def csv_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def export_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def timestamp(path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None


def elapsed(start, end):
    return (datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds() if start and end else None


def epoch_costs(worker):
    result = []
    for name in ('native', 'source'):
        path = worker / 'prefix' / f'{name}_epoch_metrics.jsonl'
        records = lines(path)
        if records:
            result.append(dict(component=name, completed_epochs=len(records),
                               epoch_ids=[r['epoch'] for r in records],
                               observed_epoch_seconds=sum(r['seconds'] for r in records),
                               evidence=relative(path)))
    for path in sorted((ROOT / 'runs/movielens').glob(f'*/{worker.name}/epoch_metrics.csv')):
        records = csv_rows(path)
        if records:
            result.append(dict(component=path.parent.parent.name, completed_epochs=len(records),
                               epoch_ids=[int(r['epoch']) for r in records],
                               observed_epoch_seconds=sum(float(r['seconds']) for r in records),
                               evidence=relative(path)))
    return result


runner = read(REPORT / 'RUNNER_STATE.json', {})
runner_workers = runner.get('revision1', {}).get('workers', {})
e_ledger=read(REPORT/'E_EVALUATION_LEDGER.json',{})
equivalence=read(REPORT/'SOURCE_REVISIONS/REVISION_EQUIVALENCE.json',{})
assert equivalence['training_source_hash']==TRAINING_SOURCE
assert len(equivalence['training_source_files'])==37
assert set((equivalence['old_full_source_hash'],equivalence['new_full_source_hash']))==set(SOURCES.values())
prefixes, runs, epochs, missing = [], [], [], []
for dataset in DATASETS:
    for seed in SEEDS:
        worker = ACTIVE / 'workers' / dataset / f'seed{seed}'
        prefix = read(worker / 'PREFIX_COMPLETE.json')
        timeline = runner_workers.get(f'{dataset}/{seed}', {})
        evaluation=e_ledger.get('workers',{}).get(f'{dataset}/seed{seed}',{})
        if prefix:
            assert prefix['identity']['source_hash'] == SOURCES[dataset]
            native_epochs=lines(worker/'prefix/native_epoch_metrics.jsonl')
            source_epochs=lines(worker/'prefix/source_epoch_metrics.jsonl')
            assert [r['epoch'] for r in native_epochs] == list(range(1,25)), str(worker)
            assert [r['epoch'] for r in source_epochs] == list(range(1,13)), str(worker)
            native = prefix['native']['seconds']
            source = prefix['source']['seconds']
            complete=read(worker/'COMPLETE.json')
            ready_time=timeline.get('ready_at') or timestamp(worker/'READY.json')
            release_time=evaluation.get('released_at') or timeline.get('E_release_at') or timestamp(worker/'EVALUATE.json')
            complete_time=evaluation.get('completed_at') or timeline.get('completed_at') or timestamp(worker/'COMPLETE.json')
            wait=elapsed(ready_time,release_time)
            prefixes.append(dict(dataset=dataset, seed=seed,
                source_hash=prefix['identity']['source_hash'],common_training_source_hash=TRAINING_SOURCE,
                physical_gpu=timeline.get('physical_gpu'), prefix_seconds=prefix['seconds'],
                native_completed_epochs=len(native_epochs),source_completed_epochs=len(source_epochs),
                native_epoch_seconds_sum=sum(r['seconds'] for r in native_epochs),
                source_epoch_seconds_sum=sum(r['seconds'] for r in source_epochs),
                native_seconds=native, source_seconds=source,
                cache_and_prefix_other_seconds=prefix['seconds']-native-source,
                cache_and_prefix_other_scope='Residual of prefix wall minus native and source timers; not separately instrumented',
                native_peak_cuda_bytes=prefix['native'].get('peak_cuda_bytes'),
                native_model_hash=prefix['identity']['native_model_hash'],
                source_function_hash=prefix['identity']['source_function_hash'],
                receipt=relative(worker / 'PREFIX_COMPLETE.json'),
                started_at=timeline.get('started_at'), ready_at=ready_time,
                E_release_at=release_time, completed_at=complete_time,
                E_timestamp_source='E_EVALUATION_LEDGER.json' if evaluation else 'RUNNER_STATE.json; filesystem fallback',
                ready_file_mtime_utc=timestamp(worker/'READY.json'),
                E_release_file_mtime_utc=timestamp(worker/'EVALUATE.json'),
                complete_file_mtime_utc=timestamp(worker/'COMPLETE.json'),
                launch_to_ready_wall_seconds_observed=elapsed(timeline.get('started_at'),ready_time),
                ready_to_release_waiting_seconds_observed=wait,
                release_to_complete_seconds_observed=elapsed(release_time,complete_time),
                final_E_scope='Release command timestamp through COMPLETE write; includes polling/queue, E cache/predictions/export, not pure model inference',
                waiting_measurement_type='Runner timestamps; filesystem mtime fallback if absent. Residence includes READY-to-global-freeze and release queue.',
                worker_wall_minus_waiting_seconds_approx=complete['total_worker_wall_seconds']-wait if complete and wait is not None else None,
                approximate_active_wall_scope='Worker total wall minus observed READY-to-release residence; includes exports, latency, E, cache and bookkeeping, not pure training',
                READY=(worker/'READY.json').exists(), E_complete=(worker/'COMPLETE.json').exists(),
                worker_complete_record=complete))
        for method in METHODS:
            run = ACTIVE / 'runs' / dataset / method / f'seed{seed}'
            receipt = read(run/'run_receipt.json')
            costs = read(run/'costs.json')
            if receipt is None or costs is None:
                partial_epochs=csv_rows(run/'epoch_metrics.csv') if (run/'epoch_metrics.csv').exists() else []
                missing.append(dict(dataset=dataset, seed=seed, method=method,
                                    receipt_present=receipt is not None, costs_present=costs is not None,
                                    observed_epochs=len(partial_epochs),
                                    observed_epoch_seconds=sum(float(r['seconds']) for r in partial_epochs) if partial_epochs else None))
                continue
            assert receipt['identity']['source_hash'] == SOURCES[dataset]
            records = csv_rows(run/'epoch_metrics.csv')
            assert [int(r['epoch']) for r in records] == list(range(1, 21)), str(run)
            assert receipt['final_epoch'] == 20
            assert prefix is not None
            assert costs['common_prefix_seconds'] == prefix['seconds']
            assert costs['adaptation_training_seconds'] == receipt['training_seconds']
            counts = costs['adaptation_parameters']
            row = dict(dataset=dataset, seed=seed, method=method,
                       physical_gpu=timeline.get('physical_gpu'), logical_device=costs['device'],
                       device_name=costs['device_name'], source_hash=receipt['identity']['source_hash'],
                       common_training_source_hash=TRAINING_SOURCE,
                       native_parameters=costs['native_parameters'],
                       frozen_source_hidden_parameters=costs['frozen_source_hidden_parameters'],
                       audit_only_teacher_parameters=costs['audit_only_teacher_parameters'],
                       adaptation_parameters=counts['adaptation'],
                       relation_parameters_subset=counts['relation'],
                       residual_function_parameters_subset=counts['source_functions'],
                       deployment_parameters=costs['deployment_parameters'],
                       selected_epoch=receipt['selected_epoch'], final_epoch=20,
                       global_steps=receipt['global_steps'],
                       adaptation_training_seconds=receipt['training_seconds'],
                       adaptation_epoch_seconds_sum=sum(float(r['seconds']) for r in records),
                       common_prefix_seconds_view_only=prefix['seconds'],
                       single_method_with_prefix_seconds_view_only=costs['complete_training_seconds_with_common_prefix'],
                       diagnostic_seconds=costs['diagnostic_seconds'],
                       benchmark_and_cost_receipt_generation_interval_seconds_observed=(run/'costs.json').stat().st_mtime-(run/'diagnostics.json').stat().st_mtime,
                       benchmark_interval_type='File-timestamp observed duration: diagnostics.json write through latency benchmark, MAC count and costs.json generation; not pure benchmark timer',
                       peak_cuda_bytes=receipt['peak_cuda_bytes'],
                       batch256_linear_MAC=costs['batch256_linear_MAC'], MAC_scope=costs['MAC_scope'],
                       explicit_residual_MAC_per_row=costs['explicit_residual_MAC_per_row'],
                       explicit_lowrank_relation_MAC_per_row=costs['explicit_lowrank_relation_MAC_per_row'],
                       batch256_source_function_calls=costs['batch256_source_function_calls'],
                       batch256_residual_function_calls=costs['batch256_residual_function_calls'],
                       sparse_execution=costs['sparse_execution'],
                       run_receipt=relative(run/'run_receipt.json'), costs_receipt=relative(run/'costs.json'))
            for latency in costs['latency']:
                label=f"batch{latency['batch']}_{latency['kind']}"
                row[label+'_p50_ms']=latency['p50_ms']
                row[label+'_p95_ms']=latency['p95_ms']
                row[label+'_warmup']=latency['warmup']
                row[label+'_measurements']=latency['measurements']
            runs.append(row)
            epochs.extend(dict(dataset=dataset,seed=seed,method=method,source_hash=receipt['identity']['source_hash'],
                               physical_gpu=timeline.get('physical_gpu'), **r) for r in records)

historical = []
observed = {2023:154, 2024:115, 2025:16}
for seed in SEEDS:
    worker=ROOT/'workers/movielens'/f'seed{seed}'
    failed=read(worker/'FAILED.json')
    interrupted=read(worker/'RUNNER_INTERRUPTION.json')
    prefix=read(worker/'PREFIX_COMPLETE.json')
    component_receipts=[]
    if prefix:
        component_receipts.append(dict(component='common_prefix',seconds=prefix['seconds'],
                                       evidence=relative(worker/'PREFIX_COMPLETE.json')))
    else:
        for component in ('native','source'):
            path=worker/'prefix'/f'{component}_receipt.json'
            receipt=read(path)
            if receipt:
                component_receipts.append(dict(component=component,seconds=receipt['seconds'],evidence=relative(path)))
    for path in sorted((ROOT/'runs/movielens').glob(f'*/seed{seed}/run_receipt.json')):
        receipt=read(path)
        component_receipts.append(dict(component=path.parent.parent.name,
                                       seconds=receipt['training_seconds'],evidence=relative(path)))
    epoch_records=epoch_costs(worker)
    covered={c['component'] for c in component_receipts}
    if 'common_prefix' in covered:
        covered.update(('native','source'))
    unreceipted_epoch_seconds=sum(r['observed_epoch_seconds'] for r in epoch_records if r['component'] not in covered)
    component_total=sum(c['seconds'] for c in component_receipts)
    lower_bound=failed['actual_elapsed_seconds'] if failed else component_total+unreceipted_epoch_seconds
    historical.append(dict(attempt='revision0', dataset='movielens',seed=seed,
        physical_gpu=runner.get('workers',{}).get(f'movielens/{seed}',{}).get('physical_gpu'),
        status='FAILED' if failed else 'ABORTED', failure=failed, interruption=interrupted,
        exact_worker_wall_seconds=failed.get('actual_elapsed_seconds') if failed else None,
        observed_process_elapsed_snapshot_seconds=observed.get(seed),
        observed_process_elapsed_scope='ps snapshot before stopping, not final runtime' if seed in observed else None,
        component_receipts=component_receipts, completed_epoch_observations=epoch_records,
        measured_component_receipt_seconds=component_total if component_receipts else None,
        unreceipted_completed_epoch_seconds=unreceipted_epoch_seconds,
        recorded_spent_seconds_lower_bound=lower_bound,
        lower_bound_scope='Exact failed worker elapsed' if failed else 'Nonoverlapping completed component receipts plus uncovered completed epoch timers; interruption overhead unmeasured',
        measured_epoch_seconds=sum(c['observed_epoch_seconds'] for c in epoch_records),
        accounting='Worker wall, component receipt and epoch sums overlap; do not add them',
        evidence=relative(worker)))

diagnostics=read(ROOT/'diagnostic_repair/DIAGNOSTIC_COST_LEDGER.json',{})
kr_failures=[]
kr_archive_index=read(ACTIVE/'attempts/KR_R1_FAILED_ARCHIVE_INDEX.json',{})
for seed in SEEDS:
    archive=ACTIVE/'attempts'/f'KR_seed{seed}_R1_failed'
    worker=archive/'workers/kuairand'/f'seed{seed}'
    failed=read(worker/'FAILED.json')
    if failed is None:
        continue
    prefix=read(worker/'PREFIX_COMPLETE.json')
    native_epochs=lines(worker/'prefix/native_epoch_metrics.jsonl')
    source_epochs=lines(worker/'prefix/source_epoch_metrics.jsonl')
    methods=[]
    for path in sorted((archive/'runs/kuairand').glob(f'*/seed{seed}/run_receipt.json')):
        receipt=read(path)
        observed=csv_rows(path.parent/'epoch_metrics.csv')
        methods.append(dict(method=path.parent.parent.name,training_seconds=receipt['training_seconds'],
                            completed_epochs=len(observed),observed_epoch_seconds=sum(float(r['seconds']) for r in observed),
                            receipt=relative(path),costs_receipt_present=(path.parent/'costs.json').exists()))
    kr_failures.append(dict(dataset='kuairand',seed=seed,status='FAILED_R1_RAW_X_PARITY',
        source_hash=prefix['identity']['source_hash'] if prefix else kr_archive_index.get('source_hash'),
        exact_worker_wall_seconds=failed['actual_elapsed_seconds'],
        native_completed_epochs=len(native_epochs),source_completed_epochs=len(source_epochs),
        prefix_seconds=prefix['seconds'] if prefix else None,
        native_seconds=prefix['native']['seconds'] if prefix else None,
        source_seconds=prefix['source']['seconds'] if prefix else None,
        native_epoch_seconds=sum(r['seconds'] for r in native_epochs),
        source_epoch_seconds=sum(r['seconds'] for r in source_epochs),
        methods=methods,failure=failed,evidence=relative(archive),
        accounting='Exact worker wall includes prefix, B_REFERENCE, export and failed parity; component values overlap and must not be added'))
unmeasured=[dict(attempt='revision1_detached_launch_failure',
    status='STARTUP_PARTIAL_NO_COMPLETED_EPOCH', seconds=None,
    evidence=relative(ACTIVE/'attempts/launch_detached_failure'),
    scope='Partial initialization files and empty stdout retained; no final wall timer'),
    dict(attempt='diagnostic_import_failure',status='IMPORT_FAILED',seconds=None,
    evidence=relative(ROOT/'diagnostic_repair/import_failure.stdout.log'),
    scope='No completed training component; total process time not recorded')]

storage=[]
seen=set()
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or path.suffix not in ('.npy','.npz'):
        continue
    stat=path.stat()
    inode=(stat.st_dev,stat.st_ino)
    duplicate=inode in seen
    seen.add(inode)
    storage.append(dict(path=relative(path),bytes=stat.st_size,
                        unique_physical_file=not duplicate,mtime_utc=timestamp(path),
                        content_kind='row_data_or_predictions'))

totals=dict(completed_costed_runs=len(runs),expected_runs=48,completed_prefixes=len(prefixes),
            expected_prefixes=8,ready_workers=sum(p['READY'] for p in prefixes),
            E_complete_workers=sum(p['E_complete'] for p in prefixes),
            epoch_rows=len(epochs),expected_epoch_rows=960,
            common_prefix_seconds_once=sum(p['prefix_seconds'] for p in prefixes),
            adaptation_training_seconds=sum(r['adaptation_training_seconds'] for r in runs),
            diagnostic_seconds=sum(r['diagnostic_seconds'] for r in runs),
            stored_row_data_logical_bytes=sum(s['bytes'] for s in storage),
            stored_row_data_unique_file_bytes=sum(s['bytes'] for s in storage if s['unique_physical_file']))
totals['recorded_prefix_plus_adaptation_seconds']=totals['common_prefix_seconds_once']+totals['adaptation_training_seconds']
totals['recorded_prefix_plus_adaptation_plus_diagnostic_seconds']=totals['recorded_prefix_plus_adaptation_seconds']+totals['diagnostic_seconds']
totals['revision0_recorded_spent_seconds_lower_bound']=sum(r['recorded_spent_seconds_lower_bound'] for r in historical)
totals['KR_R1_failed_worker_wall_seconds']=sum(r['exact_worker_wall_seconds'] for r in kr_failures)
totals['diagnostic_replay_recorded_spent_seconds_lower_bound']=sum(r.get('whole_diagnostic_seconds',r['measured_component_sum_seconds']) for r in diagnostics.get('attempts',[]))
totals['recorded_retry_spent_seconds_lower_bound']=sum(totals[k] for k in ('revision0_recorded_spent_seconds_lower_bound','KR_R1_failed_worker_wall_seconds','diagnostic_replay_recorded_spent_seconds_lower_bound'))
totals['benchmark_and_cost_receipt_generation_observed_interval_seconds']=sum(r['benchmark_and_cost_receipt_generation_interval_seconds_observed'] for r in runs)
totals['success_worker_total_wall_seconds_including_waiting']=sum(p['worker_complete_record']['total_worker_wall_seconds'] for p in prefixes if p['worker_complete_record'])
totals['success_worker_ready_to_release_waiting_seconds_observed']=sum(p['ready_to_release_waiting_seconds_observed'] for p in prefixes if p['ready_to_release_waiting_seconds_observed'] is not None)
totals['success_worker_wall_minus_observed_waiting_seconds_approx']=sum(p['worker_wall_minus_waiting_seconds_approx'] for p in prefixes if p['worker_wall_minus_waiting_seconds_approx'] is not None)
totals['E_release_to_complete_seconds_observed']=sum(p['release_to_complete_seconds_observed'] for p in prefixes if p['release_to_complete_seconds_observed'] is not None)
totals['source_groups']={value:sum(r['source_hash']==value for r in runs) for value in SOURCES.values()}
if not missing and len(prefixes)==8:
    assert len(epochs)==960
    assert len({(r['dataset'],r['seed'],r['method'],r['epoch']) for r in epochs})==960

ledger=dict(status='COMPLETE_COSTED_DEVELOPMENT_MATRIX' if not missing and len(prefixes)==8 and totals['ready_workers']==8 else 'PARTIAL_LIVE_SNAPSHOT',
    snapshot_utc=datetime.now(timezone.utc).isoformat(), artifact_root=relative(ACTIVE),
    source_hash_by_dataset=SOURCES,common_training_source_hash=TRAINING_SOURCE,
    source_equivalence_evidence=dict(path='SOURCE_REVISIONS/REVISION_EQUIVALENCE.json',training_files=37,
        training_source_hash=TRAINING_SOURCE,changed_files=equivalence['changed_files'],
        diagnostic_scientific_functions_identical=equivalence['diagnostic_scientific_functions_identical'],
        independent_review='RUNTIME_FIX2_REVIEW.md; existing byte-equivalence evidence reused'),
    totals=totals,prefixes=prefixes,runs=runs,missing_runs=missing,
    E_evaluation_ledger=dict(path='E_EVALUATION_LEDGER.json',status=e_ledger.get('status'),
        workers_complete=e_ledger.get('workers_complete'),metric_records=e_ledger.get('selected_final_metric_records'),
        prediction_files=e_ledger.get('prediction_files'),domain_task_cells=e_ledger.get('domain_task_cells')),
    revision0_attempts=historical,diagnostic_replays=diagnostics,
    KR_R1_failed_attempts=kr_failures,KR_R1_failed_archive_index=kr_archive_index,
    startup_and_import_failures=unmeasured,row_data_storage=storage,
    measurement_scopes={
        'native_seconds':'24 epochs including V scoring/selection and RAM state capture; synchronized process wall',
        'source_seconds':'12 pilot passes plus T0 source standardization, teacher conversion check and RAM state capture',
        'common_prefix_seconds':'Native+source timers plus native and source caches, other prefix work; counted once per dataset/seed',
        'adaptation_training_seconds':'20 epochs including V scoring, selection, per-epoch output and RAM selected/final capture; excludes final V/Q exports, functional diagnostics and latency benchmark',
        'diagnostic_seconds':'diagnose() timer, F/S/Q functional probes; excludes latency benchmark',
        'latency':'50 warmup+200 synchronized measurements for raw_X and cached_prefix, batch1/256; P50/P95 do not reconstruct benchmark total wall',
        'benchmark_observed_interval':'costs.json mtime minus diagnostics.json mtime includes latency benchmark, MAC inventory and cost receipt generation; filesystem observation, not pure algorithm timer',
        'unseparated_costs':'Final V/Q exports, latency benchmark total, data loading/preparation, and E evaluation have no independent algorithm timer; worker timeline residual must not be labeled training',
        'waiting':'READY-to-E release residence, excluded from algorithm training; report explicit runner times and file-mtime observations separately',
        'worker_complete':'COMPLETE total_worker_wall_seconds includes RAM waiting and final E work; never sum as training',
        'historical':'Original attempts and diagnostic replays are repeated compute costs, not independent seeds or matrix completion',
        'row_data_storage':'Current .npy/.npz file sizes, deduplicated by filesystem inode for unique bytes; no model state serialization',
        'MAC':'Partial linear-layer inventory with explicit residual/lowrank terms separate; not full FLOPs and not sparse speedup'})
(REPORT/'COST_LEDGER.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
export_csv(REPORT/'RUN_PARAMETERS_AND_COSTS.csv',runs)
export_csv(REPORT/'EPOCH_METRICS.csv',epochs)

by_dataset=[]
for dataset in DATASETS:
    selected=[r for r in runs if r['dataset']==dataset]
    ps=[p for p in prefixes if p['dataset']==dataset]
    prefix_value=f"{sum(p['prefix_seconds'] for p in ps):.3f}" if ps else '未完成'
    training_value=f"{sum(r['adaptation_training_seconds'] for r in selected):.3f}" if selected else '未完成'
    diagnostic_value=f"{sum(r['diagnostic_seconds'] for r in selected):.3f}" if selected else '未完成'
    by_dataset.append(f"| {dataset} | {len(selected)}/24 | {len(ps)}/4 | {prefix_value} | {training_value} | {diagnostic_value} |")
historical_rows=[]
for attempt in historical:
    wall=attempt['exact_worker_wall_seconds']
    snapshot=attempt['observed_process_elapsed_snapshot_seconds']
    scopes=', '.join(f"{r['component']}:{r['completed_epochs']}" for r in attempt['completed_epoch_observations'])
    component=attempt['measured_component_receipt_seconds']
    component_text=f'{component:.3f}' if component is not None else '未记录'
    historical_rows.append(f"| revision0 ML/{attempt['seed']} | {attempt['status']} | {wall if wall is not None else '未记录'} | {snapshot if snapshot is not None else '—'} | {component_text} | {scopes} |")
replays=diagnostics.get('attempts',[])
replay_rows=[f"| {r['attempt']} | {r['status']} | {r['measured_component_sum_seconds']:.3f} | {r.get('whole_diagnostic_seconds','未记录')} |" for r in replays]
kr_failure_rows=[f"| KR/{r['seed']} R1 | {r['exact_worker_wall_seconds']:.6f} | {r['prefix_seconds']:.6f} | {r['native_completed_epochs']} | {r['source_completed_epochs']} | {', '.join(m['method']+':'+str(m['completed_epochs']) for m in r['methods'])} |" for r in kr_failures]
timeline_rows=[]
for p in prefixes:
    wait=p['ready_to_release_waiting_seconds_observed']
    active=p['worker_wall_minus_waiting_seconds_approx']
    ewall=p['release_to_complete_seconds_observed']
    wall=p['worker_complete_record']['total_worker_wall_seconds'] if p['worker_complete_record'] else None
    vals=[f'{v:.3f}' if v is not None else '未完成' for v in (wall,wait,active,ewall)]
    timeline_rows.append(f"| {p['dataset']}/{p['seed']} | {p['E_release_at'] or '未释放'} | {p['completed_at'] or '未完成'} | {' | '.join(vals)} |")
text=f'''# 成本账与参数汇总

状态：`{ledger['status']}`；快照：{ledger['snapshot_utc']}。本报告整理现有记录，不启动训练或追加实验。

修订矩阵目录 `{relative(ACTIVE)}`。ML采用R1来源 `{SOURCES['movielens']}`，KR采用R2来源 `{SOURCES['kuairand']}`；两版训练源码身份均为 `{TRAINING_SOURCE}`。已收集 {len(runs)}/48 个完整 run 成本收据、{len(prefixes)}/8 个共同前缀、{len(epochs)}/960 条适配 epoch 数据。每个已收集 run 的 epoch 恰为 1–20 各一次；未完成项逐条列于 COST_LEDGER.json。READY 为 {totals['ready_workers']}/8，历史 E 导出完成 {totals['E_complete_workers']}/8。

## 修订矩阵的实际计时

| 数据 | 完整 run | 唯一前缀 | 前缀秒（只计一次） | 适配秒 | 功能诊断秒 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(by_dataset)}

已记录前缀与适配合计 {totals['recorded_prefix_plus_adaptation_seconds']:.3f} 秒；加单独功能诊断为 {totals['recorded_prefix_plus_adaptation_plus_diagnostic_seconds']:.3f} 秒。这是各进程计时的组件和，GPU 并发下不等于整个矩阵经过的日历时间。

前缀含 native24、source12、T0 标准化/教师检查和数据缓存。适配计时含每轮 V 评分、选择、日志和 RAM 状态捕获；不含末尾 V/Q 预测导出、功能诊断与 latency 测量。每方法 `single_method_with_prefix_seconds_view_only` 可展示单方法依赖成本，但整个六方法矩阵只计该前缀一次。

参数、所用物理GPU、逻辑设备、selected/final epoch、global steps、峰值显存、每批来源调用、MAC范围及 raw-X/cached 的 batch1/256 P50/P95 见 RUN_PARAMETERS_AND_COSTS.csv。relation 与 residual_function 参数是 adaptation 总数的子集；audit-only teacher 参数不算总部署参数。

Latency 每种路径 50 次 warmup、200 次同步测量。原始收据只有 P50/P95，无独立 latency 总秒；不以分位数反算算法总时间。MAC 是 nn.Linear 部分清单，原收据已明确排除 embedding、LayerNorm、activation、STAR 矩阵乘及部分显式 einsum；显式残差与低秩关系项另列。所有方法 `sparse_execution=false`。

从每run diagnostics.json 写出至 costs.json 写出的文件mtime间隔，累计观察到 {totals['benchmark_and_cost_receipt_generation_observed_interval_seconds']:.3f} 秒；此间隔含部署benchmark、MAC计数与成本收据生成，列于参数成本表。它是文件时间戳观测，不作为精确纯算法计时。

## 原失败/中止 attempt 与重放

| attempt | 状态 | 精确进程秒 | 停止前ps快照秒 | 已完成组件收据秒 | 最终落盘epoch数 |
|---|---|---:|---:|---:|---|
{chr(10).join(historical_rows)}

精确进程秒、组件收据秒与 epoch 秒彼此重叠，不能相加。ps 快照是停止前观测而非最终耗时；例如 seed2025 的最后持久化 native 日志已达24轮，停止前快照仍是15轮/16秒。原四 attempt 全部保留，不纳入修订矩阵48条独立训练结果。

用失败 worker 的精确总秒、其余中止 worker 的不重叠组件收据与尚未覆盖的完整 epoch 计时，原四 attempt 已记录投入至少 {sum(r['recorded_spent_seconds_lower_bound'] for r in historical):.3f} 秒；每项的计算边界均在账本，未计时的中止开销没有填0。

| 诊断重放 | 状态 | native/source/READ组件和秒 | 整体诊断秒 |
|---|---|---:|---:|
{chr(10).join(replay_rows)}

两次重放均使用原 ML seed2022 身份，分别因输出序列化失败和完成定位而记录。组件和与整体诊断秒重叠，不相加。另有诊断 import 失败和 revision1 detached 启动失败，两者日志/初始化目录保留但没有最终总计时；不填0。

## KR R1失败归档与R2重跑

| attempt | 精确worker总秒 | 共同前缀秒（已含于总秒） | native轮数 | source轮数 | 完成适配 |
|---|---:|---:|---:|---:|---|
{chr(10).join(kr_failure_rows)}

四份失败worker及其B_REFERENCE20轮、源码身份、FAILED/epoch/预测/stdout均由归档索引保留在 revision1/attempts/KR_seedNNNN_R1_failed。其精确进程总秒累计 {totals['KR_R1_failed_worker_wall_seconds']:.6f}，包含前缀、B_REFERENCE训练、导出与失败parity流程；不能再叠加表内组件秒。R2原位重跑是修订矩阵的实际KR训练，归档失败轨迹不算额外seed，也不复用为成功前缀。

合并原4attempt、两次诊断重放与KR R1四失败，已记录重复投入至少 {totals['recorded_retry_spent_seconds_lower_bound']:.3f} 秒；缺最终计时的启动/import失败另列而不填0。此成本与成功矩阵组件和分开。

## 等待、导出与存储

每 worker 的 runner 启动、READY、E释放、完成时点及文件 mtime 观测保存在 COST_LEDGER.json。READY 后 RAM 驻留等待不属于算法训练时长；COMPLETE 的 worker wall 含这段等待，不可直接作训练成本。账本另提供 worker wall减READY→release观测等待的近似active wall，仍含导出、benchmark、E与缓存等流程。最终 V/Q 导出、数据准备/加载和 E 评分缺少独立组件秒时，记录为未分离，不能用漏记等同0。

| worker | E释放(+08:00) | COMPLETE(+08:00) | worker总秒（含等待） | READY→释放观察秒 | 减等待近似active秒 | 释放→完成观察秒 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(timeline_rows)}

八worker完整进程计时合计 {totals['success_worker_total_wall_seconds_including_waiting']:.3f} 秒，其中READY→释放观察到 {totals['success_worker_ready_to_release_waiting_seconds_observed']:.3f} 秒；相减的近似active wall为 {totals['success_worker_wall_minus_observed_waiting_seconds_approx']:.3f} 秒。释放→完成观察间隔合计 {totals['E_release_to_complete_seconds_observed']:.3f} 秒，含轮询/排队、E缓存、评分与导出。以上口径彼此重叠，不能再与训练组件和相加；E释放/完成以 E_EVALUATION_LEDGER.json 的实际记录优先，READY以runner记录和READY文件mtime为据。

E账本状态 `{e_ledger.get('status','NOT_RELEASED')}`：{e_ledger.get('workers_complete',0)} worker完成、{e_ledger.get('selected_final_metric_records',0)} 份selected/final指标、{e_ledger.get('prediction_files',0)} 份预测文件、{e_ledger.get('domain_task_cells',0)} 个域任务格。48run保持ML24份R1、KR24份R2；37份共同训练源码字节等价继续使用已独立审查的原证据，没有追加模型运行。

当前数据/预测 .npy/.npz 逻辑大小 {totals['stored_row_data_logical_bytes']:,} 字节，按 inode 去重的唯一文件大小 {totals['stored_row_data_unique_file_bytes']:,} 字节；逐文件大小与mtime在账本。RAM 模型、optimizer 和 checkpoint 未写磁盘。

汇总命令：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/26-10-27_v2/aggregate_costs.py`。完整计时边界见账本 measurement_scopes，原始证据路径随每项保留。
'''
(REPORT/'COST_REPORT.md').write_text(text)
print(json.dumps(dict(status=ledger['status'],**totals)))
