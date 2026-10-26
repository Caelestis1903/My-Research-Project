"""Reviewed operational scheduling for the existing resident campaign.

No training function, configuration, model, optimizer, or RNG is changed here.
The one installation target is Campaign.phase in one registered parent PID.
"""
from __future__ import annotations
import hashlib
import json
import marshal
import os
from pathlib import Path
import sys
import time
import traceback

WORKSPACE = Path('/home/xzf/Code/phase0-3')
ARTIFACT_ROOT = WORKSPACE / 'artifacts/mdmt_rcf_20261026/v2_four_seed'
SERVICE_PID = 2088990
CONFIG_HASH = 'cefb6d934c774bd78a438e4fba5c560e934f4c56ab09fbb3dfe389f4f4842fd6'
TARGET = 'Campaign.phase'


def function_fingerprint(function):
    return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _working_set_bytes(campaign, key):
    minimum = campaign.config.get('recovery', {}).get('prefix_working_set_gib', {}).get(key[0], 6)
    observed = campaign.prefix_peak_bytes.get(key[0], 0)
    observed = max(observed, campaign.workers[key].get('last_resources', {}).get('peak_rss_bytes', 0))
    return max(int(minimum * 2**30), int(1.15 * observed))


def _admission(campaign, key, active):
    from .campaign import host_memory_snapshot, gpu_snapshot
    memory = host_memory_snapshot()
    reserve = int(campaign.config.get('recovery', {}).get('host_reserve_gib', 8) * 2**30)
    remaining = 0
    for other in active:
        rss = host_memory_snapshot(campaign.workers[other]['process'].pid).get('rss_bytes', 0)
        remaining += max(0, _working_set_bytes(campaign, other) - rss)
    rss = host_memory_snapshot(campaign.workers[key]['process'].pid).get('rss_bytes', 0)
    additional = max(0, _working_set_bytes(campaign, key) - rss)
    device = campaign.workers[key]['device']
    snapshots = {f"cuda:{row['index']}": row for row in gpu_snapshot()}
    gpu = snapshots.get(device)
    gpu_ok = device == 'cpu' or (gpu is not None and gpu.get('memory_total',0)>0 and gpu.get('power_limit',0)>0
                                  and gpu.get('memory_used',float('inf'))/gpu['memory_total']<.85
                                  and gpu.get('power',float('inf'))/gpu['power_limit']<.85)
    host_ok = memory['memavailable_bytes'] >= reserve + remaining + additional
    return dict(eligible=bool(gpu_ok and host_ok), gpu_eligible=bool(gpu_ok), host_eligible=bool(host_ok),
                device=device, gpu=gpu, available_bytes=memory['memavailable_bytes'],
                system_reserve_bytes=reserve, active_additional_reservation_bytes=remaining,
                candidate_additional_reservation_bytes=additional)


def parallel_phase(self, phase, *, methods_by_dataset=None):
    """Send unchanged stage messages concurrently to eligible resident workers."""
    if phase not in ('s1', 's2', 's3'):
        raise ValueError('Only registered development training stages may be parallelized')
    self.require_review()
    for key, worker in self.workers.items():
        if not worker['process'].is_alive():
            raise RuntimeError(f'Resident worker {key} lost; preserve other RAM and replay explicitly')
        if worker['pending'] is not None:
            raise RuntimeError(f'Resident worker {key} already has a pending command')
    for key in self.workers:
        self._send(key, 'release-data')
        self._recv(key)
    ordered = [key for key in self.workers if key[1] in self.config['seeds']['development']
               and (phase != 's3' or (methods_by_dataset or {}).get(key[0], []))]
    queue = list(ordered)
    active = set()
    completed = {}
    first_error = None
    admissions = {}
    last_progress = 0.
    started = time.monotonic()
    while queue or active:
        changed = False
        if first_error is None:
            # One new dispatch per card per fresh round, followed by completion polling.
            dispatched_devices = set()
            for key in queue[:]:
                device = self.workers[key]['device']
                if device in dispatched_devices:
                    continue
                try:
                    admission = _admission(self, key, active)
                    admissions[str(key)] = admission
                    if not admission['eligible']:
                        continue
                    methods = (methods_by_dataset or {}).get(key[0], [])
                    args = {'methods': methods, 'decision': self.decisions['s2']['datasets'][key[0]]} if phase == 's3' else {}
                    self._send(key, phase, **args)
                    queue.remove(key)
                    active.add(key)
                    dispatched_devices.add(device)
                    changed = True
                except BaseException as error:
                    first_error = dict(key=list(key), error=repr(error), traceback=traceback.format_exc())
                    break
        for key in list(active):
            worker = self.workers[key]
            if not worker['conn'].poll() and worker['process'].is_alive():
                continue
            try:
                result = self._recv(key)
                self._remember(key, result)
                completed[key] = result
            except BaseException as error:
                if first_error is None:
                    first_error = dict(key=list(key), error=repr(error), traceback=traceback.format_exc())
            finally:
                active.remove(key)
                changed = True
        now = time.monotonic()
        if changed or now - last_progress >= 10:
            self.write_status(phase, 'DRAINING_AFTER_FAILURE' if first_error else 'RUNNING_PARALLEL',
                              active=[list(key) for key in ordered if key in active],
                              queued=[list(key) for key in queue],
                              completed=[list(key) for key in ordered if key in completed],
                              admissions=admissions, scheduler='registered_parallel_phase',
                              seconds=now-started, failure=first_error)
            last_progress = now
        if first_error and not active:
            self.write_status(phase, 'FAILED_DRAINED', failure=first_error,
                              completed=[list(key) for key in ordered if key in completed],
                              queued=[list(key) for key in queue], pending_preserved=True)
            raise RuntimeError('Parallel stage stopped after draining other dispatched jobs: ' + str(first_error))
        if not changed and queue and not active:
            # An idle campaign cannot free more host memory by waiting for its own jobs.
            if all(not admissions.get(str(key), {}).get('host_eligible', True) for key in queue):
                self.write_status(phase, 'BLOCKED_HOST_MEMORY', admissions=admissions,
                                  queued=[list(key) for key in queue], pending_preserved=True)
                raise MemoryError('Host reserve prevents dispatch; resident models preserved for an explicit registered recovery')
        if not changed and (queue or active):
            time.sleep(.25)
    results = [completed[key] for key in ordered]
    self.write_status(phase, 'COMPLETE', scheduler='registered_parallel_phase',
                      completed=[list(key) for key in ordered], seconds=time.monotonic()-started)
    return results


def _validate_authorization(token, *, old_hash, new_hash, scheduler_sha, readers_sha):
    exact = dict(service_pid=SERVICE_PID, config_hash=CONFIG_HASH, workspace=str(WORKSPACE),
                 artifact_root=str(ARTIFACT_ROOT), target=TARGET, old_phase_code_hash=old_hash,
                 new_phase_code_hash=new_hash, scheduler_sha256=scheduler_sha, readers_sha256=readers_sha)
    if token.get('verdict') not in ('PASS', 'PASS WITH FIXES') or token.get('blocking_issues', []):
        raise PermissionError('Independent scheduler review has not passed')
    mismatches = [key for key, value in exact.items() if token.get(key) != value]
    if mismatches:
        raise PermissionError('Scheduler authorization mismatch: ' + ', '.join(mismatches))


def install_on_reader_import():
    """One fixed RPC import hook, a no-op in workers and ordinary clients."""
    if os.getpid() != SERVICE_PID:
        return dict(status='NO_OP', reason='Not the registered resident controller PID')
    from .artifacts import _json
    receipt_path = ARTIFACT_ROOT / 'PHASE_SCHEDULER_INSTALL_RECEIPT.json'
    receipt = dict(status='FAIL_CLOSED', service_pid=os.getpid(), target=TARGET,
                   model_files_written=0, config_changed=False, training_functions_changed=False)
    try:
        module = sys.modules.get(__package__ + '.campaign')
        if module is None:
            raise RuntimeError('Existing campaign module is not loaded')
        campaign_class = module.Campaign
        old = campaign_class.phase
        old_hash = function_fingerprint(old)
        new_hash = function_fingerprint(parallel_phase)
        authorization_path = ARTIFACT_ROOT / 'PHASE_SCHEDULER_AUTHORIZATION.json'
        token = json.loads(authorization_path.read_text())
        scheduler_sha = _sha(__file__)
        readers_sha = _sha(Path(__file__).with_name('readers.py'))
        _validate_authorization(token, old_hash=token['old_phase_code_hash'] if old is parallel_phase else old_hash,
                                new_hash=new_hash, scheduler_sha=scheduler_sha, readers_sha=readers_sha)
        service = json.loads((ARTIFACT_ROOT / 'SERVICE.json').read_text())
        if service.get('pid') != SERVICE_PID or service.get('config_hash') != CONFIG_HASH:
            raise PermissionError('Resident service receipt differs')
        # The only object lookup is the exact synchronous normal-RPC frame.
        # No GC scan, other-thread inspection, frame persistence, or arbitrary target exists.
        frame = sys._getframe(1)
        controller = None
        try:
            for _ in range(40):
                if frame is None:
                    break
                if frame.f_code is campaign_class.reload_readers.__code__:
                    possible = frame.f_locals.get('self')
                    if type(possible) is campaign_class:
                        controller = possible
                    break
                frame = frame.f_back
        finally:
            del frame
        if controller is None:
            raise PermissionError('Installation is permitted only inside the exact normal reload-readers RPC')
        if controller.config.get('config_hash') != CONFIG_HASH or controller.root.resolve() != ARTIFACT_ROOT:
            raise PermissionError('Controller configuration or owned artifact root differs')
        expected = {(dataset, seed) for dataset in ('movielens', 'kuairand') for seed in (5201,5202,5203,5204)}
        if set(controller.workers) != expected:
            raise PermissionError('All eight registered prefixes must remain resident')
        if controller.frozen is not None:
            raise PermissionError('Frozen campaign cannot change operational scheduler')
        if any(not worker['process'].is_alive() or worker['pending'] is not None for worker in controller.workers.values()):
            raise PermissionError('Every resident worker must be alive and idle before installation')
        status = json.loads((ARTIFACT_ROOT / 'CAMPAIGN_STATUS.json').read_text())
        if status.get('stage') != 's0' or status.get('status') != 'COMPLETE':
            raise PermissionError('The original prefix stage must have completed before scheduler installation')
        receipt.update(authorization_sha256=_sha(authorization_path), scheduler_sha256=scheduler_sha,
                       readers_sha256=readers_sha, old_phase_code_hash=token['old_phase_code_hash'],
                       new_phase_code_hash=new_hash, all_eight_workers_alive_idle=True,
                       worker_pids={f'{key[0]}:{key[1]}':worker['process'].pid for key,worker in controller.workers.items()},
                       installed_unix=time.time(), only_class_attribute=TARGET,
                       controller_config_hash=controller.config['config_hash'])
        if old is parallel_phase:
            receipt['status']='ALREADY_INSTALLED'
            _json(receipt_path, receipt)
            return receipt
        receipt['status']='VERIFIED_BEFORE_ASSIGNMENT'
        _json(receipt_path, receipt)
        campaign_class.phase = parallel_phase
        receipt['status']='INSTALLED'
        _json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        receipt.update(error=repr(error), installed=False)
        try:
            _json(receipt_path, receipt)
        except OSError:
            pass
        return receipt
