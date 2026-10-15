"""F0-B1 queue construction, search selection, and whole-matrix status ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
import time
import fcntl
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_F0_data import load_b1_bundles, split_hashes  # noqa: E402
from stage_F0_protocol import (  # noqa: E402
    BACKBONES,
    ADDON_SEEDS,
    BATCH_SIZES,
    DEVELOPMENT_SEEDS,
    FORMAL_SEEDS,
    PROTOCOL_VERSION,
    SCOPES,
    WEIGHT_DECAYS,
    LEARNING_RATES,
    all_b1_units,
    canonical_queue_entry,
    config_grid,
    config_id,
    matrix_ledger,
    protocol_hash,
    primary_search_expansion_trigger,
    sha256_json,
    unit_key,
    width_for,
)


IMPLEMENTATION_FILES = (
    "stage_F0_protocol.py", "stage_F0_data.py", "stage_F0_models.py",
    "stage_F0_objectives.py", "stage_F0_queue.py", "stage_F0_run.py",
    "tests/test_stage_F0_protocol.py", "tests/test_stage_F0_data.py",
    "tests/test_stage_F0_models.py", "tests/test_stage_F0_objectives.py",
    "tests/test_stage_F0_queue.py", "tests/test_stage_F0_run.py",
    "tests/conftest.py",
    "stage_S1_data.py", "stage_S1_models.py", "stage_S1_protocol.py",
    "stage_S1_metrics.py", "history_exp/first/p7_models.py",
    "../m3oe/MDMTRec/__init__.py",
    "../m3oe/MDMTRec/basic/__init__.py", "../m3oe/MDMTRec/basic/activation.py",
    "../m3oe/MDMTRec/basic/features.py", "../m3oe/MDMTRec/basic/initializers.py",
    "../m3oe/MDMTRec/basic/layers.py", "../m3oe/MDMTRec/utils/__init__.py",
    "../m3oe/MDMTRec/utils/data.py", "../m3oe/MDMTRec/models/multi_domain/__init__.py",
    "../m3oe/MDMTRec/models/multi_domain/model.py",
)

MAX_CONCURRENT_PYTHON_PROCESSES = 5


class QueueFileLock:
    """Persistent-file advisory lock; the kernel releases it on process death."""

    def __init__(self, path: str, *, nonblocking: bool = True):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.handle = open(self.path, "a+", encoding="utf-8")
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        try:
            fcntl.flock(self.handle.fileno(), flags)
        except BlockingIOError:
            self.handle.close()
            raise

    def write_owner(self, metadata: Mapping[str, object]) -> None:
        self.handle.seek(0)
        self.handle.truncate()
        json.dump(dict(metadata), self.handle, ensure_ascii=False, sort_keys=True)
        self.handle.write("\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        if not self.handle.closed:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class ProcessSlotLease:
    """One of five repository-wide live F0 task-process slots."""

    def __init__(self, lock: QueueFileLock, index: int):
        self.lock = lock
        self.index = int(index)

    def close(self) -> None:
        self.lock.close()


class InheritedProcessSlotLease:
    """A child-side handle for a slot reserved by its supervisor.

    The inherited descriptor refers to the supervisor's open-file description.
    Closing the child's duplicate must not issue LOCK_UN, because that would
    release the supervisor's reservation while it still owns the worker.
    """

    def __init__(self, fd: int, index: int):
        self.fd = int(fd)
        self.index = int(index)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def process_scheduler_root() -> str:
    configured = os.environ.get("F0_PROCESS_SCHEDULER_ROOT")
    if configured:
        path = os.path.abspath(configured)
        os.makedirs(path, exist_ok=True)
        return path
    workspace_id = hashlib.sha256(ROOT.encode("utf-8")).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), "phase0-3-F0-scheduler-" + workspace_id)
    os.makedirs(path, exist_ok=True)
    return path


def acquire_process_slot(*, role: str, blocking: bool = True,
                         poll_seconds: float = 0.1,
                         scheduler_dir: Optional[str] = None) -> ProcessSlotLease:
    """Reserve a shared process slot; training workers also take a GPU lock."""

    root = os.path.abspath(scheduler_dir) if scheduler_dir else process_scheduler_root()
    os.makedirs(root, exist_ok=True)
    while True:
        for index in range(MAX_CONCURRENT_PYTHON_PROCESSES):
            try:
                lock = QueueFileLock(os.path.join(root, "slot{}.lock".format(index)))
            except BlockingIOError:
                continue
            lock.write_owner({"pid": os.getpid(), "host": socket.gethostname(),
                              "role": str(role), "slot": index,
                              "claimed_unix": time.time()})
            return ProcessSlotLease(lock, index)
        if not blocking:
            raise RuntimeError("all five F0 task-process slots are occupied")
        time.sleep(max(0.01, float(poll_seconds)))


def adopt_reserved_process_slot(*, fd: int, index: int, owner_pid: int,
                                role: str, scheduler_dir: Optional[str] = None
                                ) -> InheritedProcessSlotLease:
    """Adopt a wrapper-reserved slot without acquiring or unlocking it.

    The supervisor pre-reserves both worker slots before it starts Python.
    The trajectory child inherits the second descriptor.  This function checks
    that the descriptor is the expected inode, the lock is still held, and its
    durable owner record identifies the live supervisor and requested role.
    """

    slot_index = int(index)
    if slot_index < 0 or slot_index >= MAX_CONCURRENT_PYTHON_PROCESSES:
        raise ValueError("reserved process slot index is outside the five-slot namespace")
    root = os.path.abspath(scheduler_dir) if scheduler_dir else process_scheduler_root()
    path = os.path.join(root, "slot{}.lock".format(slot_index))
    try:
        expected_stat = os.stat(path)
        inherited_stat = os.fstat(int(fd))
    except OSError as exc:
        raise PermissionError("reserved process-slot descriptor is unavailable") from exc
    if (expected_stat.st_dev, expected_stat.st_ino) != (
            inherited_stat.st_dev, inherited_stat.st_ino):
        raise PermissionError("reserved process-slot descriptor does not match its slot inode")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError) as exc:
        raise PermissionError("reserved process-slot owner record is missing or invalid") from exc
    if (int(metadata.get("pid", -1)) != int(owner_pid)
            or str(metadata.get("role", "")) != str(role)
            or int(metadata.get("slot", -1)) != slot_index
            or str(metadata.get("host", "")) != socket.gethostname()):
        raise PermissionError("reserved process-slot owner identity does not match")
    try:
        probe = QueueFileLock(path)
    except BlockingIOError:
        pass
    else:
        probe.close()
        raise PermissionError("reserved process-slot lock is not held")
    return InheritedProcessSlotLease(os.dup(int(fd)), slot_index)


def execution_control_lock_path(output_root: str) -> str:
    return os.path.join(os.path.abspath(output_root), "execution_control",
                        "execution_state.lock")


def execution_hold_path(output_root: str) -> str:
    return os.path.join(os.path.abspath(output_root), "execution_control",
                        "global_hold.json")


def assert_no_execution_hold(output_root: str, *, lock_already_held: bool = False) -> None:
    """Fail closed on an active global hold, including missing release proof."""

    from stage_F0_worker_control import assert_no_global_hold
    assert_no_global_hold(output_root, lock_already_held=lock_already_held)


def _atomic_json(path: str, payload: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def queue_status_path(output_root: str, run_id: str) -> str:
    return os.path.join(os.path.abspath(output_root), "queue_status", str(run_id) + ".json")


def _latest_checkpoint(attempt_dir: str) -> Optional[str]:
    candidates = []
    if os.path.isdir(attempt_dir):
        for name in os.listdir(attempt_dir):
            if name.startswith("resume_epoch_") and name.endswith(".pt"):
                try:
                    completed = int(name[len("resume_epoch_"):-len(".pt")])
                except ValueError:
                    continue
                candidates.append((completed, os.path.join(attempt_dir, name)))
    return max(candidates)[1] if candidates else None


def _claim_next_entry_unlocked(queue: Mapping[str, object], output_root: str, *,
                               host: Optional[str] = None, pid: Optional[int] = None,
                               gpu_uuid: str = "", gpu_index: int = 0,
                               held_run_ids: Iterable[str] = (),
                               recovery_authorizations: Optional[Mapping[str, object]] = None,
                               fresh_only_v15: bool = False,
                               frozen_f140_ids: Optional[set[str]] = None):
    """Atomically claim one available canonical key for a queue worker.

    Returns ``(entry, canonical_lock, resume_path)`` or ``(None, None, None)``.
    Incomplete states are skipped unless the caller supplies a reviewed,
    identity-bound one-use recovery authorization. Existing checkpoint and
    attempt data are never inferred safe merely because a flock is free.
    """

    root = os.path.abspath(output_root)
    protected = {str(value) for value in held_run_ids}
    authorizations = dict(recovery_authorizations or {})
    for source in queue["entries"]:
        entry = dict(source)
        run_id = str(entry["run_id"])
        if fresh_only_v15:
            allowed = frozen_f140_ids or set()
            if run_id not in allowed:
                continue
            from stage_F0_successor_v15 import validate_fresh_entry
            try:
                validate_fresh_entry(root, entry, sorted(allowed))
            except PermissionError:
                continue
        if run_id in protected:
            continue
        status_path = queue_status_path(root, run_id)
        canonical_path = os.path.join(root, "canonical_locks", run_id + ".lock")
        attempt_root = os.path.join(root, run_id)

        def read_previous_status():
            exists = os.path.lexists(status_path) if fresh_only_v15 else os.path.exists(status_path)
            if not exists:
                return {}, False
            try:
                with open(status_path, encoding="utf-8") as handle:
                    previous_payload = json.load(handle)
            except (OSError, ValueError) as exc:
                raise PermissionError("prior canonical status is unreadable; claim blocked: "
                                      + run_id) from exc
            if not isinstance(previous_payload, dict):
                raise PermissionError("prior canonical status is malformed; claim blocked: "
                                      + run_id)
            return previous_payload, True

        previous, has_status = read_previous_status()
        state = str(previous.get("status", ""))
        if state in ("VALIDATION_COMPLETE", "RESOURCE_PREFLIGHT_COMPLETE",
                     "RECOVERY_BLOCKED"):
            continue
        canonical_artifact_existed = os.path.lexists(canonical_path)
        has_history = (has_status or os.path.lexists(attempt_root)
                       or canonical_artifact_existed)
        resume_attempt_id = str(previous.get("attempt_id", entry.get(
            "attempt_id", "attempt-00")))
        resume_path = _latest_checkpoint(os.path.join(
            attempt_root, "attempts", resume_attempt_id))
        if has_history and not _valid_recovery_authorization(
                authorizations.get(run_id), entry=entry, previous=previous,
                resume_path=resume_path):
            continue
        lock = None
        try:
            if fresh_only_v15:
                # This durable administrative marker must precede the first
                # canonical mutation: QueueFileLock.__init__ creates the
                # canonical lock path before it obtains the flock.
                from stage_F0_worker_control import append_phase_event
                phase_log = os.environ.get("F0_PHASE_LOG")
                if phase_log:
                    append_phase_event(phase_log, phase="CANONICAL_MUTATION_STARTED",
                                       worker_id=os.environ.get("F0_WORKER_ID", "direct"),
                                       nonce=os.environ.get("F0_INVOCATION_NONCE", "direct"),
                                       run_id=run_id)
            lock = QueueFileLock(canonical_path)
        except BlockingIOError:
            continue

        try:
            if fresh_only_v15:
                # QueueFileLock has just created this path.  Confirm it is our
                # ordinary, empty inode; all prior history was rejected above.
                stat_path = os.stat(canonical_path, follow_symlinks=False)
                stat_fd = os.fstat(lock.handle.fileno())
                if (os.path.islink(canonical_path) or stat_path.st_ino != stat_fd.st_ino
                        or stat_path.st_dev != stat_fd.st_dev or stat_fd.st_size != 0
                        or os.path.lexists(status_path) or os.path.lexists(attempt_root)):
                    lock.close()
                    continue
            # Re-read under the canonical lock; the writer can only mutate
            # this state while owning the same lock.
            previous, has_status = read_previous_status()
            state = str(previous.get("status", ""))
            if state in ("VALIDATION_COMPLETE", "RESOURCE_PREFLIGHT_COMPLETE",
                         "RECOVERY_BLOCKED"):
                lock.close()
                continue
            has_history = (has_status or os.path.lexists(attempt_root)
                           or canonical_artifact_existed)
            attempt_id = str(previous.get("attempt_id", entry.get("attempt_id", "attempt-00")))
            attempt_dir = os.path.join(root, run_id, "attempts", attempt_id)
            resume_path = _latest_checkpoint(attempt_dir)
            if has_history and not _valid_recovery_authorization(
                    authorizations.get(run_id), entry=entry, previous=previous,
                    resume_path=resume_path):
                lock.close()
                continue
            recoveries = int(previous.get("recovery_count", 0))
            if fresh_only_v15 and has_history:
                lock.close()
                continue
            if has_history:
                if recoveries >= 1:
                    if has_status:
                        _atomic_json(status_path, {**previous, "status": "RECOVERY_BLOCKED",
                                                   "reason": "single recovery allowance exhausted"})
                    lock.close()
                    continue
                recoveries += 1
                if resume_path is None:
                    # A reviewed retry must use a fresh attempt directory and
                    # may never overwrite incomplete prior files.
                    attempt_id = "attempt-01"
                    attempt_dir = os.path.join(root, run_id, "attempts", attempt_id)
                    if os.path.exists(attempt_dir):
                        if has_status:
                            _atomic_json(status_path, {**previous,
                                                       "status": "RECOVERY_BLOCKED",
                                                       "reason": "retry attempt directory already exists"})
                        lock.close()
                        continue
            entry["attempt_id"] = attempt_id
            metadata = {"run_id": run_id, "run_id_full_sha256": entry["run_id_full_sha256"],
                        "canonical_key_json": entry["canonical_key_json"],
                        "pid": int(pid if pid is not None else os.getpid()),
                        "host": str(host or socket.gethostname()),
                        "gpu_uuid": str(gpu_uuid), "gpu_index": int(gpu_index),
                        "attempt_id": attempt_id, "claimed_unix": time.time()}
            if fresh_only_v15:
                from stage_F0_worker_control import linux_process_identity
                metadata["nonce"] = os.environ.get("F0_INVOCATION_NONCE", "")
                metadata["start_ticks"] = int(linux_process_identity(
                    int(pid if pid is not None else os.getpid()))["start_ticks"])
            lock.write_owner(metadata)
            _atomic_json(status_path, {**metadata, "status": "CLAIMED",
                                       "recovery_count": recoveries,
                                       "resume_checkpoint": resume_path})
            return entry, lock, resume_path
        except BaseException:
            lock.close()
            raise
    return None, None, None


def _checkpoint_sha256(path: Optional[str]) -> Optional[str]:
    if path is None or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_recovery_authorization(authorization: object, *, entry: Mapping[str, object],
                                  previous: Mapping[str, object],
                                  resume_path: Optional[str]) -> bool:
    """Recovery requires a reviewed, identity-bound, one-use release record."""

    if not isinstance(authorization, Mapping):
        return False
    canonical_json = str(entry.get("canonical_key_json", ""))
    expected_key_sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    expected_checkpoint_sha = _checkpoint_sha256(resume_path)
    return bool(
        authorization.get("review_verdict") in ("PASS", "PASS WITH FIXES")
        and str(authorization.get("release_id", ""))
        and str(authorization.get("approved_by", ""))
        and str(authorization.get("run_id", "")) == str(entry.get("run_id", ""))
        and str(authorization.get("canonical_key_json_sha256", "")) == expected_key_sha
        and str(authorization.get("attempt_id", "")) == str(previous.get("attempt_id", ""))
        and int(authorization.get("recovery_count", -1)) == int(previous.get("recovery_count", 0))
        and authorization.get("resume_checkpoint_sha256") == expected_checkpoint_sha
    )


def claim_next_entry(queue: Mapping[str, object], output_root: str, *,
                     host: Optional[str] = None, pid: Optional[int] = None,
                     gpu_uuid: str = "", gpu_index: int = 0,
                     held_run_ids: Iterable[str] = (),
                     recovery_authorizations: Optional[Mapping[str, object]] = None,
                     gpu_telemetry_provider=None):
    """Claim one entry under the shared fail-closed execution-state lock.

    Abnormal/incomplete states are skipped unless a separate reviewed,
    canonical-bound recovery authorization is supplied.  The global hold and
    canonical claim are serialized against the supervisor's hold writer.
    """

    output_root = str(Path(output_root).resolve())
    is_v15_production = False
    production_f140_ids = None
    state_lock = QueueFileLock(execution_control_lock_path(output_root),
                               nonblocking=False)
    try:
        assert_no_execution_hold(output_root, lock_already_held=True)
        entries = list(queue.get("entries", []))
        from stage_F0_worker_control import RUNS
        production_root = Path(output_root).resolve() == Path(RUNS).resolve()
        repo_root = Path(output_root).resolve().parents[2]
        candidate_gate_path = Path(output_root).resolve().parent / "execution_isolation_gate.json"
        if candidate_gate_path.is_file() and not candidate_gate_path.is_symlink():
            with candidate_gate_path.open(encoding="utf-8") as gate_handle:
                candidate_gate = json.load(gate_handle)
        else:
            candidate_gate = {}
        isolated_v15_root = (not production_root
                             and candidate_gate.get("schema") == "F0-EXECUTION-ISOLATION-v1.5")
        if production_root or isolated_v15_root:
            # Production claims are bound to the reviewed, fixed queue, not
            # to an arbitrary caller-provided list or recovery authorization.
            from stage_F0_postabnormal_v14 import QUEUE_SHA256, _read_json, sha256_file
            frozen_path = repo_root / "experiments/stage_F0_outputs/queues/F0-B1-ML-search-queue.json"
            if not frozen_path.is_file() or sha256_file(frozen_path) != QUEUE_SHA256:
                raise PermissionError("production claim queue is missing or differs from frozen F140 source")
            frozen_queue = _read_json(frozen_path)
            if entries != frozen_queue.get("entries"):
                raise PermissionError("production claim rows differ from exact frozen queue rows")
            if recovery_authorizations:
                raise PermissionError("production claims prohibit all recovery authorizations")
            gate = candidate_gate
            if gate.get("schema") == "F0-EXECUTION-ISOLATION-v1.5":
                from stage_F0_successor_v15 import (
                    require_live_gpu_memory,
                    validate_current_gate,
                )
                validate_current_gate(gate, root=repo_root, require_release=True,
                                       lock_already_held=True)
                if production_root and gpu_telemetry_provider is not None:
                    raise PermissionError("production claim cannot inject GPU telemetry")
                target_pairs = gate.get("gpu_target_pairs")
                if not isinstance(target_pairs, list):
                    raise PermissionError("v1.5 gate has no reviewed target pairs")
                target = next((row for row in target_pairs
                               if isinstance(row, Mapping)
                               and row.get("index") == int(gpu_index)), None)
                if not isinstance(target, Mapping) or target.get("uuid") != str(gpu_uuid):
                    raise PermissionError("worker physical GPU is outside reviewed v1.5 targets")
                if int(gpu_index) not in (0, 1):
                    raise PermissionError("v1.5 worker index must remain a physical GPU index")
                # This check runs under execution_state.lock and before any
                # per-key canonical lock/status/run mutation.
                from stage_F0_worker_control import append_phase_event
                phase_log = os.environ.get("F0_PHASE_LOG")
                if phase_log:
                    append_phase_event(phase_log, phase="serialized_preclaim_memory_check",
                                       worker_id=os.environ.get("F0_WORKER_ID", "direct"),
                                       nonce=os.environ.get("F0_INVOCATION_NONCE", "direct"),
                                       gpu_index=int(gpu_index), gpu_uuid=str(gpu_uuid))
                require_live_gpu_memory(runs_root=output_root, gpu_index=int(gpu_index),
                                        gpu_uuid=str(gpu_uuid),
                                        boundary="SERIALIZED_PRECLAIM_MEMORY_CHECK",
                                        provider=gpu_telemetry_provider)
                from stage_F0_successor_v15 import frozen_f140_ids
                fresh = set(frozen_f140_ids(root=repo_root))
                production_f140_ids = fresh
                is_v15_production = True
                queued_ids = {str(row.get("run_id", "")) for row in entries
                              if row.get("phase") == "search"}
                if not fresh or not fresh.issubset(queued_ids):
                    raise PermissionError("v1.5 F140 is not a subset of exact frozen queue")
                if not held_run_ids:
                    raise PermissionError("v1.5 production claims require exact H51 hold set")
                from stage_F0_worker_control import require_exact_h51
                held_run_ids = require_exact_h51(held_run_ids)
            elif gate.get("schema") == "F0-EXECUTION-ISOLATION-v1.4":
                from stage_F0_postabnormal_v14 import require_exact_h51, require_live_gpu_memory
                held_run_ids = require_exact_h51(held_run_ids)
                require_live_gpu_memory(runs_root=output_root, gpu_index=int(gpu_index),
                                        gpu_uuid=str(gpu_uuid))
            else:
                raise PermissionError("unknown production execution-gate schema")
        if entries and any(str(entry.get("phase", "")) == "search" for entry in entries):
            from stage_F0_worker_control import require_exact_h51
            held_run_ids = require_exact_h51(list(held_run_ids))
        return _claim_next_entry_unlocked(
            queue, output_root, host=host, pid=pid, gpu_uuid=gpu_uuid,
            gpu_index=gpu_index, held_run_ids=held_run_ids,
            recovery_authorizations=recovery_authorizations,
            fresh_only_v15=is_v15_production,
            frozen_f140_ids=production_f140_ids)
    finally:
        state_lock.close()


def code_hash_inventory(require_tests: bool = True) -> Dict[str, str]:
    result = {}
    for relative in IMPLEMENTATION_FILES:
        path = os.path.join(HERE, relative)
        if not os.path.isfile(path):
            if require_tests:
                raise FileNotFoundError("implementation manifest input is missing: {}".format(path))
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[relative] = digest.hexdigest()
    return result


def code_hash(require_tests: bool = True) -> str:
    return sha256_json(code_hash_inventory(require_tests=require_tests))


def build_search_queue(bundles: Mapping[str, object], code_digest: str,
                       protocol_digest: Optional[str] = None) -> List[Dict[str, object]]:
    digest = protocol_digest or protocol_hash()
    queue: List[Dict[str, object]] = []
    for unit in all_b1_units():
        scope = str(unit["scope"])
        bundle = bundles[scope]
        for config in config_grid():
            for seed in DEVELOPMENT_SEEDS:
                row = canonical_queue_entry(
                    phase="search", intent="search", model=str(unit["backbone"]),
                    scope=scope, seed_role="development", seed=seed,
                    width=unit["width"], split_hashes=split_hashes(bundle),
                    protocol_digest=digest, code_digest=code_digest, config=config)
                row["max_epochs"] = 50
                row["test_access"] = False
                queue.append(row)
    validate_queue(queue, expected_count=288, phase="search")
    return queue


def build_formal_queue(bundles: Mapping[str, object],
                       selected: Mapping[str, Mapping[str, object]],
                       code_digest: str,
                       protocol_digest: Optional[str] = None,
                       primary_validation_deltas: Optional[Sequence[Sequence[float]]] = None
                       ) -> List[Dict[str, object]]:
    digest = protocol_digest or protocol_hash()
    required = {unit_key(model, scope) for model in BACKBONES for scope in SCOPES}
    if set(selected) != required:
        missing = sorted(required - set(selected))
        extra = sorted(set(selected) - required)
        raise ValueError("formal queue requires one frozen selected config per unit; missing={} extra={}".format(
            missing, extra))
    queue: List[Dict[str, object]] = []
    expansion_trigger = (None if primary_validation_deltas is None else
                         primary_search_expansion_trigger(primary_validation_deltas))
    expansion_input_hash = ("" if primary_validation_deltas is None else
                            sha256_json(primary_validation_deltas))
    for unit in all_b1_units():
        scope = str(unit["scope"])
        key = str(unit["unit_key"])
        decision = selected[key]
        if str(decision.get("status", "SELECTED")) != "SELECTED":
            raise ValueError("formal config for {} is not status SELECTED".format(key))
        config = {"lr": float(decision["lr"]),
                  "batch_size": int(decision["batch_size"]),
                  "weight_decay": float(decision["weight_decay"])}
        config_id(config)
        for seed in FORMAL_SEEDS:
            row = canonical_queue_entry(
                phase="formal", intent="formal", model=str(unit["backbone"]),
                scope=scope, seed_role="formal", seed=seed, width=unit["width"],
                split_hashes=split_hashes(bundles[scope]), protocol_digest=digest,
                code_digest=code_digest, config=config)
            row["max_epochs"] = 50
            row["test_access"] = False
            row["selection_artifact_hash"] = str(decision.get("selection_artifact_hash", ""))
            row["seed_expansion_trigger_evaluated"] = expansion_trigger is not None
            row["seed_expansion_trigger"] = expansion_trigger
            row["seed_expansion_input_sha256"] = expansion_input_hash
            queue.append(row)
    if expansion_trigger:
        for unit in all_b1_units():
            scope = str(unit["scope"])
            key = str(unit["unit_key"])
            decision = selected[key]
            config = {"lr": float(decision["lr"]),
                      "batch_size": int(decision["batch_size"]),
                      "weight_decay": float(decision["weight_decay"])}
            for seed in ADDON_SEEDS:
                row = canonical_queue_entry(
                    phase="formal", intent="formal", model=str(unit["backbone"]),
                    scope=scope, seed_role="formal_addon", seed=seed,
                    width=unit["width"], split_hashes=split_hashes(bundles[scope]),
                    protocol_digest=digest, code_digest=code_digest, config=config)
                row["max_epochs"] = 50
                row["test_access"] = False
                row["selection_artifact_hash"] = str(decision.get("selection_artifact_hash", ""))
                row["seed_expansion_trigger_evaluated"] = True
                row["seed_expansion_trigger"] = True
                row["seed_expansion_input_sha256"] = expansion_input_hash
                queue.append(row)
    validate_queue(queue, expected_count=len(queue), phase="formal")
    return queue


def build_resource_queue(bundles: Mapping[str, object], code_digest: str,
                         protocol_digest: Optional[str] = None) -> List[Dict[str, object]]:
    """Register 3 backbones × 2 batch sizes using MDMT/Q and a fixed config."""

    digest = protocol_digest or protocol_hash()
    rows: List[Dict[str, object]] = []
    for model in BACKBONES:
        width = width_for(model, "mdmt")
        for batch_size in BATCH_SIZES:
            config = {"lr": 0.001, "batch_size": int(batch_size), "weight_decay": 0.0}
            row = canonical_queue_entry(
                phase="resource", intent="resource", model=model, scope="mdmt",
                seed_role="resource_preflight", seed=2018, width=width,
                split_hashes=split_hashes(bundles["mdmt"]), protocol_digest=digest,
                code_digest=code_digest, config=config)
            row["max_epochs"] = 1
            row["test_access"] = False
            row["resource_probe"] = "first_search_epoch_and_runtime_peak"
            rows.append(row)
    validate_queue(rows, expected_count=6, phase="resource")
    return rows


def validate_queue(queue: Sequence[Mapping[str, object]], expected_count: int,
                   phase: str) -> None:
    if len(queue) != int(expected_count):
        raise ValueError("{} queue has {} rows, expected {}".format(phase, len(queue), expected_count))
    run_ids = [str(row.get("run_id", "")) for row in queue]
    if any(not value for value in run_ids) or len(set(run_ids)) != len(run_ids):
        raise ValueError("{} queue run_id values are missing or duplicated".format(phase))
    for row in queue:
        if row.get("phase") != phase or row.get("intent") != phase:
            raise ValueError("queue row phase/intent mismatch")
        canonical = str(row.get("canonical_key_json", ""))
        full_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if full_hash != row.get("run_id_full_sha256") or full_hash[:20] != row.get("run_id"):
            raise ValueError("queue canonical run ID hash mismatch")
        if row.get("test_access") is not False:
            raise ValueError("search/formal/resource queue entries must not access test during training")
    if phase == "formal":
        if int(expected_count) not in (48, 96):
            raise ValueError("formal queue supports exactly 48 first-four or 96 expanded trajectories")
        expected_seeds = set(FORMAL_SEEDS)
        if int(expected_count) == 96:
            expected_seeds |= set(ADDON_SEEDS)
        seen = set()
        unit_set = {str(unit["unit_key"]) for unit in all_b1_units()}
        for row in queue:
            key = (str(row.get("unit_key")), int(row.get("seed", -1)))
            if key in seen:
                raise ValueError("formal queue duplicates unit/seed {}".format(key))
            seen.add(key)
            if key[0] not in unit_set or key[1] not in expected_seeds:
                raise ValueError("formal queue contains an unsupported unit or seed {}".format(key))
            expected_role = "formal" if key[1] in FORMAL_SEEDS else "formal_addon"
            if row.get("seed_role") != expected_role:
                raise ValueError("formal queue seed role differs from frozen seed group")
        expected_pairs = {(str(unit["unit_key"]), seed)
                          for unit in all_b1_units() for seed in expected_seeds}
        if seen != expected_pairs:
            raise ValueError("formal queue does not cover the full unit×seed block")
    elif phase == "resource":
        expected = {(model, batch) for model in BACKBONES for batch in BATCH_SIZES}
        observed = {(str(row.get("backbone")), int(row.get("config", {}).get("batch_size", -1)))
                    for row in queue}
        if observed != expected or any(row.get("scope") != "mdmt" for row in queue):
            raise ValueError("resource queue must cover all 3 backbones × 2 batch sizes on MDMT")


def theoretical_dense_forward_flops(backbone: str, scope: str, c: int,
                                    p: int = 113) -> int:
    """Multiply-add estimate from the registered dense Linear paths."""

    model = str(backbone).lower()
    d = 3 if str(scope).lower() == "mdmt" else 1
    t = 2
    c = int(c)
    b = 4 * c
    a = 8 * c
    k = d * t
    if model == "m3oe":
        linears = ((d + 1) * p * a + p * b + a * b + (4 + d + t) * b * c
                   + k * b * 4 + k * (c * c + c))
    elif model == "mmoe":
        linears = 4 * (p * b + b * c) + k * (p * c + c * 4 + c * c + c)
    elif model == "cgc1":
        linears = (4 + k) * (p * b + b * c) + k * (p * c + c * 5 + c * c + c)
    else:
        raise ValueError("unknown F0 backbone: {}".format(backbone))
    return int(2 * linears)


def select_search_configs(results: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    """Select each unit's config only after all 24 development trajectories pass."""

    expected_units = {unit_key(model, scope) for model in BACKBONES for scope in SCOPES}
    grouped: Dict[str, Dict[str, List[Mapping[str, object]]]] = defaultdict(lambda: defaultdict(list))
    seen_trajectory = set()
    for row in results:
        key = str(row.get("unit_key", ""))
        if key not in expected_units:
            raise ValueError("search result includes an unsupported unit_key: {}".format(key))
        if str(row.get("status", "")) not in ("VALIDATION_COMPLETE", "OK"):
            raise ValueError("search trajectory {} is incomplete or invalid".format(row.get("run_id", "")))
        seed = int(row["seed"])
        if seed not in DEVELOPMENT_SEEDS:
            raise ValueError("search result contains non-development seed {}".format(seed))
        config = {"lr": float(row["lr"]), "batch_size": int(row["batch_size"]),
                  "weight_decay": float(row["weight_decay"])}
        cid = config_id(config)
        if str(row.get("config_id", cid)) != cid:
            raise ValueError("search result config_id does not match config fields")
        unique = (key, cid, seed)
        if unique in seen_trajectory:
            raise ValueError("duplicate search trajectory result {}".format(unique))
        seen_trajectory.add(unique)
        score = float(row["selected_validation_macro_auc"])
        if not (score == score and abs(score) != float("inf")):
            raise ValueError("search selected validation AUC is non-finite")
        grouped[key][cid].append({**row, **config, "config_id": cid, "score": score})
    output: Dict[str, Dict[str, object]] = {}
    expected_configs = {config_id(config) for config in config_grid()}
    for key in sorted(expected_units):
        groups = grouped.get(key, {})
        if set(groups) != expected_configs:
            missing = sorted(expected_configs - set(groups))
            extra = sorted(set(groups) - expected_configs)
            raise ValueError("search configs incomplete for {} missing={} extra={}".format(key, missing, extra))
        candidates = []
        model = key.split("|")[3]
        scope = key.split("|")[4]
        for cid, rows in groups.items():
            seeds = [int(row["seed"]) for row in rows]
            if len(rows) != 2 or set(seeds) != set(DEVELOPMENT_SEEDS):
                raise ValueError("config {} in {} needs both development seeds".format(cid, key))
            mean_score = sum(float(row["score"]) for row in rows) / 2.0
            config = {"lr": float(rows[0]["lr"]), "batch_size": int(rows[0]["batch_size"]),
                      "weight_decay": float(rows[0]["weight_decay"])}
            width = width_for(model, scope)
            flops = theoretical_dense_forward_flops(model, scope, width["c"])
            candidates.append({"config_id": cid, **config, "mean_selected_validation_macro_auc": mean_score,
                               "search_seed_scores": {str(int(row["seed"])): float(row["score"])
                                                      for row in rows},
                               "theoretical_dense_forward_flops": flops})
        # Higher validation score, then lower theoretical dense FLOPs, then
        # numeric config tuple (lr, batch, weight_decay).
        candidates.sort(key=lambda row: (-float(row["mean_selected_validation_macro_auc"]),
                                         int(row["theoretical_dense_forward_flops"]),
                                         float(row["lr"]), int(row["batch_size"]),
                                         float(row["weight_decay"])))
        chosen = candidates[0]
        output[key] = {"status": "SELECTED", **chosen,
                       "candidate_order": candidates,
                       "selection_rule": "max_two_development_seed_selected_validation_macro_auc_then_lower_dense_FLOPs_then_numeric_config"}
    if seen_trajectory != {(key, cid, seed) for key in expected_units
                           for cid in expected_configs for seed in DEVELOPMENT_SEEDS}:
        raise AssertionError("search queue coverage is not exactly 12×12×2")
    return output


def write_queue(path: str, queue: Sequence[Mapping[str, object]], *, phase: str,
                include_ledger: bool = True) -> Dict[str, object]:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"stage": "F0", "protocol_version": PROTOCOL_VERSION,
               "phase": phase, "protocol_hash": protocol_hash(),
               "entries": list(queue), "entry_count": len(queue),
               "queue_hash": sha256_json(list(queue)),
               "test_access_during_training": False,
               "resource_policy": {
                   "user_override": "concurrent-process-cap-v1",
                   "max_concurrent_python_processes": MAX_CONCURRENT_PYTHON_PROCESSES,
                   "max_concurrent_test_processes": MAX_CONCURRENT_PYTHON_PROCESSES,
                   "max_training_processes": "min(5, preflight_visible_usable_gpu_count)",
                   "training_processes_per_gpu": 1,
                   "canonical_run_claim": "advisory_flock_full_run_id",
               }}
    if include_ledger:
        payload["matrix_ledger"] = matrix_ledger()
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)
    return payload


def _load_selected(path: str) -> Dict[str, Dict[str, object]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload.get("selected_configs", payload)
    if not isinstance(selected, dict):
        raise ValueError("selected-config artifact must contain a mapping")
    return selected


def main(argv: Optional[Sequence[str]] = None) -> int:
    process_slot = acquire_process_slot(role="F0-queue-builder")
    try:
        return _main_impl(argv)
    finally:
        process_slot.close()


def _main_impl(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("search", "formal", "resource", "ledger"),
                        required=True)
    parser.add_argument("--dataset", default=None, help="MovieLens CSV or containing directory")
    parser.add_argument("--selected", default=None,
                        help="frozen search selection JSON required for --action formal")
    parser.add_argument("--primary-validation-deltas", default=None,
                        help="five first-four validation delta vectors; may trigger the full 2026–29 add-on")
    parser.add_argument("--output", default=None)
    parser.add_argument("--code-hash", default=None,
                        help="frozen implementation+tests digest; defaults to current inventory")
    args = parser.parse_args(argv)
    default_out = os.path.join(HERE, "stage_F0_queue")
    if args.action == "ledger":
        path = args.output or os.path.join(default_out, "matrix_ledger.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(matrix_ledger(), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        return 0
    digest = args.code_hash or code_hash()
    bundles = load_b1_bundles(args.dataset)
    if args.action == "search":
        queue = build_search_queue(bundles, digest)
    elif args.action == "resource":
        queue = build_resource_queue(bundles, digest)
    else:
        if not args.selected:
            parser.error("--selected is required for --action formal")
        deltas = None
        if args.primary_validation_deltas:
            with open(args.primary_validation_deltas, encoding="utf-8") as handle:
                delta_payload = json.load(handle)
            deltas = (delta_payload.get("primary_validation_deltas", delta_payload)
                      if isinstance(delta_payload, Mapping) else delta_payload)
        queue = build_formal_queue(bundles, _load_selected(args.selected), digest,
                                   primary_validation_deltas=deltas)
    path = args.output or os.path.join(default_out, "{}_queue.json".format(args.action))
    write_queue(path, queue, phase=args.action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["IMPLEMENTATION_FILES", "code_hash_inventory", "code_hash",
           "build_search_queue", "build_formal_queue", "build_resource_queue",
           "validate_queue", "theoretical_dense_forward_flops", "select_search_configs",
           "write_queue", "QueueFileLock", "claim_next_entry", "queue_status_path",
           "ProcessSlotLease", "InheritedProcessSlotLease",
           "process_scheduler_root", "acquire_process_slot",
           "adopt_reserved_process_slot", "execution_control_lock_path",
           "execution_hold_path", "assert_no_execution_hold",
           "MAX_CONCURRENT_PYTHON_PROCESSES", "main"]
