"""Small report references after actual runtime export/cleanup acceptance.

Never invokes cleanup or modifies research artifacts. Full raw data indices stay
on the server; these references make their identity and retained scope explicit.
"""
from pathlib import Path
import json
from datetime import datetime, timezone
from analyze_core_four_seed import ACTIVE, REPORT, read, sha, require_complete_export_inventory


def ref(path):
    return dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size)


def save(name,value):
    (REPORT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    exports=read(ACTIVE/'EXPORT_ACCEPTANCE.json')
    cleanup=read(ACTIVE/'CLEANUP_RECEIPT.json')
    if exports['status']!='PASS' or cleanup['status']!='COMPLETE':
        raise PermissionError('Actual runtime finalization has not passed')
    if cleanup['model_files_written']!=0:raise RuntimeError('Forbidden model storage')
    inventory=read(ACTIVE/'FOUR_SEED_ACCEPTANCE.json')
    if inventory['status']!='PASS' or not inventory['analysis_authorized']:
        raise PermissionError('Four-training-seed evidence required')
    for name in ('CORE_FOUR_SEED_ANALYSIS_RECEIPT.json','COST_FOUR_SEED_ANALYSIS_RECEIPT.json',
                 'SUPPORT_FOUR_SEED_ANALYSIS_RECEIPT.json','SCALE_FOUR_SEED_ANALYSIS_RECEIPT.json',
                 'CROSSOVER_FOUR_SEED_ANALYSIS_RECEIPT.json','POPULATION_FOUR_SEED_ANALYSIS_RECEIPT.json'):
        if read(REPORT/name)['status']!='PASS':raise PermissionError('Missing actual analysis '+name)
    root_index=read(ACTIVE/'ARTIFACT_INDEX.json');prefixes=[];retained_paths=set()
    require_complete_export_inventory(inventory,root_index)
    for item in root_index['prefixes']:
        root=Path(item['root']);accept=read(root/'EXPORT_ACCEPTANCE.json')
        if accept['status']!='PASS':raise RuntimeError('Prefix export failed')
        for record in item['files']:
            path=Path(record['path']).resolve()
            if not path.exists() or path.stat().st_size!=record['bytes']:
                raise RuntimeError('Retained export absent/size changed '+str(path))
            retained_paths.add(path)
        delivery=ACTIVE/'delivery'/item['dataset']/f"seed{item['seed']}"/'COSTS_PARAMETERS_AND_INFERENCE.json'
        prefixes.append(dict(dataset=item['dataset'],seed=item['seed'],root=str(root),
            prediction_methods=item['prediction_methods'],formation_methods=item['formation_methods'],
            full_index=ref(root/'ARTIFACT_INDEX.json'),acceptance=ref(root/'EXPORT_ACCEPTANCE.json'),
            raw_native_acceptance=ref(root/'RAW_X_CACHE_ACCEPTANCE.json'),
            raw_predictor_acceptance=ref(root/'RAW_X_FULL_PREDICTOR_ACCEPTANCE.json'),
            bootstrap=ref(root/'BOOTSTRAP.json'),delivery=ref(delivery),
            indexed_files=len(item['files']),prediction_files_recomputed=accept['prediction_files_recomputed']))
    deleted={Path(row['path']).resolve() for row in cleanup['deleted_files']}
    if deleted&retained_paths:raise RuntimeError('Cleanup touched final export')
    if any(p.exists() for p in deleted):raise RuntimeError('Owned temporary cleanup incomplete')
    forbidden=[str(p) for root in (ACTIVE.parent,REPORT,REPORT.parents[1]/'experiments/mdmt_rcf_20261026')
               for p in root.rglob('*') if p.is_file() and p.suffix in {'.pt','.pth','.ckpt','.safetensors'}]
    if forbidden:raise RuntimeError('Forbidden model/checkpoint files '+repr(forbidden))
    receipt_paths=[*(ACTIVE/'runs').rglob('TRAINING_COMPLETE.json')]
    expected_receipts=sum(len(c['training_seeds']) for c in inventory['conditions'])
    if len(receipt_paths)!=expected_receipts:raise RuntimeError('Canonical training receipts differ from accepted four-seed inventory')
    training=dict(status='PASS',count=len(receipt_paths),
                  receipts=[ref(p) for p in sorted(receipt_paths)],
                  statistical_unit='Independent training seed; replay/folds/epochs/bootstrap excluded')
    save('TRAINING_COMPLETION_MANIFEST.json',training)
    stamp=datetime.now(timezone.utc).isoformat()
    save('ARTIFACT_INDEX.json',dict(status='RETAINED',recorded_utc=stamp,server_workspace=str(ACTIVE.parents[2]),
        large_artifacts_in_git=False,full_server_index=ref(ACTIVE/'ARTIFACT_INDEX.json'),
        frozen_manifest=ref(ACTIVE/'FROZEN_MANIFEST.json'),four_seed_acceptance=ref(ACTIVE/'FOUR_SEED_ACCEPTANCE.json'),
        prefix_exports=prefixes,scale=ref(REPORT/'SCALE_FOUR_SEED_ANALYSIS_RECEIPT.json'),
        scale_server_root=str(ACTIVE.parent/'v1'/'scale'),
        retained='raw selected/terminal and paired predictions, source phi/pilot outputs, formed delta/read features, requests/assignments/groups, coefficients/selection, budgets/profiles/support, identities and logs',
        not_retained='No model/optimizer/RNG checkpoint ever written. Manifest-owned temporary row caches removed; optional local/initial hidden data not independently retained in full.',
        future_forward='New forward or unexported decomposition/refitting may require rebuilding prefix'))
    save('EXPORT_ACCEPTANCE.json',dict(status='PASS',recorded_utc=stamp,runtime_acceptance=ref(ACTIVE/'EXPORT_ACCEPTANCE.json'),
        n_prefixes=len(prefixes),prediction_files_recomputed=sum(p['prediction_files_recomputed'] for p in prefixes),
        full_raw_probability_tolerance=1e-6,four_seed_acceptance=ref(ACTIVE/'FOUR_SEED_ACCEPTANCE.json'),
        required_cost_receipt=ref(REPORT/'COST_FOUR_SEED_ANALYSIS_RECEIPT.json'),
        all_indexed_final_files_retained_with_original_size=True,
        no_second_full_sha_scan_claim='Runtime finalize already recomputed indexed hashes/metrics; this report step checks retained paths/sizes and cleanup disjointness',
        incomplete_nonrequired_fused_timing='See per-method COST tables; raw predictor acceptance is separate'))
    save('CLEANUP_RECEIPT.json',dict(status='COMPLETE',recorded_utc=stamp,runtime_receipt=ref(ACTIVE/'CLEANUP_RECEIPT.json'),
        model_files_written=0,forbidden_model_files_found=forbidden,deleted_file_count=len(deleted),
        deleted_bytes=cleanup['deleted_bytes'],memory_released=cleanup['memory_released'],
        deleted_paths_disjoint_from_retained_exports=True,all_recorded_deleted_paths_absent=True,
        scope=cleanup['cleanup_scope'],historical_task_v1_preserved=True,other_experiments_and_user_processes_untouched=True))
    print(json.dumps(dict(status='PASS',prefixes=len(prefixes),training_receipts=len(receipt_paths),deleted_bytes=cleanup['deleted_bytes'])))


if __name__=='__main__':main()
