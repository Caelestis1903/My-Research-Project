"""Resident per-prefix worker and explicit global-freeze handshake."""
from __future__ import annotations
from pathlib import Path
import gc
import json
import os
import time
import numpy as np
import torch
from .config import METHODS, SEEDS, DATASETS
from .data import Data
from .native import build_native, cache_role, synchronize
from .source_functions import fit_source, cache_phi
from .trainer import train_method, predict, export_prediction
from .diagnostics import diagnose, latency
from .state import configure, source_identity, write_json, state_hash, freeze


def worker(cfg, dataset, seed, artifact_root, device='cuda:0', freeze_file=None):
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError('Dataset/seed outside reviewed matrix')
    root = Path(artifact_root).resolve()/'workers'/dataset/f'seed{seed}'
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        return _resident_worker(cfg, dataset, seed, artifact_root, device, freeze_file)
    except BaseException as error:
        write_json(root/'FAILED.json', dict(status='FAILED', error_type=type(error).__name__,
                                           error=str(error), actual_elapsed_seconds=time.monotonic()-started))
        raise


def _resident_worker(cfg, dataset, seed, artifact_root, device, freeze_file):
    configure()
    artifact_root = Path(artifact_root).resolve()
    root = artifact_root/'workers'/dataset/f'seed{seed}'
    prefix_root = root/'prefix'
    prefix_root.mkdir()
    data = Data(dataset, artifact_root, root/'label_access.jsonl')
    sources = source_identity()
    identity = dict(protocol=cfg['protocol'], dataset=dataset, seed=seed,
                    config_hash=cfg['config_hash'], source_hash=sources['source_hash'],
                    source_main_commit=sources['base_main_commit'], data_role_hash=data.identity,
                    preprocessor_hash=data.manifest['preprocessor_hash'], model_storage='RAM_ONLY')
    write_json(root/'effective_config.json', dict(config=cfg, identity=identity))
    write_json(root/'source_identity.json', sources)
    write_json(root/'data_manifest.json', data.manifest)
    write_json(root/'STATUS.json', dict(status='BUILDING_PREFIX', pid=os.getpid(), identity=identity))
    synchronize(device)
    started = time.monotonic()
    native, native_selected, native_final, native_receipt = build_native(data, seed, cfg, prefix_root, device, identity)
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    identity.update(native_model_hash=native_receipt['selected_model_hash'])
    views = {role:cache_role(native, data, role, prefix_root, device) for role in ('T0', 'F', 'S', 'Q', 'V')}
    source, source_snapshot, source_receipt = fit_source(data, views['T0'], seed, prefix_root, device, identity)
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    for role in ('F', 'S', 'Q', 'V'):
        cache_phi(source, views[role], role, prefix_root, device)
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    prefix_seconds = time.monotonic()-started
    identity.update(source_function_hash=source_receipt['source_function_hash'])
    write_json(root/'PREFIX_COMPLETE.json', dict(identity=identity, seconds=prefix_seconds,
                                                native=native_receipt, source=source_receipt))
    # The native/source selected and terminal states remain CPU RAM objects until E export.
    residents = {}
    for method in METHODS:
        write_json(root/'STATUS.json', dict(status='TRAINING', method=method, pid=os.getpid(), identity=identity))
        resident = train_method(method, data, views, seed, cfg, artifact_root, device, identity)
        model = resident['model'].to(device)
        diagnostics = diagnose(model, data, views, source, device)
        resident['diagnostics'] = diagnostics
        write_json(resident['root']/'diagnostics.json', diagnostics)
        costs = latency(native, source, model, data, views['Q'], device)
        costs['common_prefix_seconds'] = prefix_seconds
        costs['adaptation_training_seconds'] = resident['receipt']['training_seconds']
        costs['diagnostic_seconds'] = diagnostics.get('seconds', 0.)
        costs['complete_training_seconds_with_common_prefix'] = prefix_seconds+resident['receipt']['training_seconds']
        write_json(resident['root']/'costs.json', costs)
        freeze(model).cpu()
        residents[method] = resident
    native.cpu()
    source.cpu()
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    worker_identity = dict(identity,
                           selected={m:r['receipt']['selected_model_hash'] for m, r in residents.items()},
                           final={m:r['receipt']['final_model_hash'] for m, r in residents.items()})
    ready = dict(status='READY_FOR_GLOBAL_FREEZE', pid=os.getpid(), identity=worker_identity,
                 prefix_seconds=prefix_seconds,
                 methods={m:dict(receipt=r['receipt'], diagnostics=r['diagnostics']) for m, r in residents.items()},
                 model_storage='RAM_ONLY', cuda_offloaded=True)
    write_json(root/'READY.json', ready)
    write_json(root/'STATUS.json', dict(status=ready['status'], pid=os.getpid(), identity=worker_identity))
    print(f'READY {dataset} seed={seed} path={root / "READY.json"} GPU_OFFLOADED; waiting for global freeze', flush=True)
    freeze_file = Path(freeze_file or artifact_root/'GLOBAL_FREEZE.json')
    while not freeze_file.exists():
        time.sleep(2)
    manifest = json.loads(freeze_file.read_text())
    data.authorize_final(manifest, worker_identity)
    # The orchestrator releases one worker at a time or a measured safe concurrent set.
    release_file = root/'EVALUATE.json'
    while not release_file.exists():
        time.sleep(2)
    release = json.loads(release_file.read_text())
    if release.get('identity') != worker_identity:
        raise PermissionError('Evaluation release identity differs')
    native.to(device)
    source.to(device)
    write_json(root/'STATUS.json', dict(status='EVALUATING_FROZEN_E', pid=os.getpid(), identity=worker_identity))
    eview = cache_role(native, data, 'E', prefix_root, device)
    cache_phi(source, eview, 'E', prefix_root, device)
    y, mask = data.labels('E', 'final', frozen_manifest=data.outer_frozen)
    for method, resident in residents.items():
        model = resident['model'].to(device)
        for state in ('selected', 'final'):
            snapshot = resident[state]
            model.load_state_dict(snapshot['model'])
            logits = predict(model, eview, device)
            export_prediction(resident['root'], 'E', state, eview, y, mask, logits,
                              snapshot['position']['epoch'], state_hash(snapshot['model']))
        model.cpu()
    write_json(root/'COMPLETE.json', dict(status='FROZEN_HISTORICAL_E_COMPLETE', identity=worker_identity,
                                         frozen_manifest_hash=data.outer_frozen, models_written=0,
                                         total_worker_wall_seconds=time.monotonic()-started))
    write_json(root/'STATUS.json', dict(status='COMPLETE', pid=os.getpid(), identity=worker_identity))
    print(f'COMPLETE {dataset} seed={seed}: six selected/final historical E evaluations exported', flush=True)
    return worker_identity

