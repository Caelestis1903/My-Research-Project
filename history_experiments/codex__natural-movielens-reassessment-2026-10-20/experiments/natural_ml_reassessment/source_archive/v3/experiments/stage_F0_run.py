"""Raw Stage F0-B1 search/formal trajectory runner with exact epoch recovery.

Training never computes test metrics.  The separate selected-checkpoint test
evaluator requires a validation-decision release artifact and is not called by
the search/formal CLI path.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Deterministic CUDA matmul requires this to be present before CUDA context use.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_F0_data import (  # noqa: E402
    file_sha256,
    load_b1_bundles,
    split_hashes,
)
from stage_F0_models import (  # noqa: E402
    build_semantically_paired_model,
    non_embedding_parameter_count,
    parameter_count_manifest,
    validate_frozen_width,
)
from stage_F0_objectives import (  # noqa: E402
    equal_diagnostic_bce,
    evaluate_cells,
    sample_bce,
    split_weighted_sample_loss,
)
from stage_F0_protocol import (  # noqa: E402
    ADAM_BETAS,
    ADAM_EPS,
    ARCH_LR_RATIO,
    EPOCHS,
    FIXED_CHECKPOINT_EPOCHS,
    PROTOCOL_VERSION,
    canonical_queue_entry,
    protocol_hash,
    scope_instance,
    sha256_json,
    width_for,
)
from stage_F0_queue import (  # noqa: E402
    MAX_CONCURRENT_PYTHON_PROCESSES,
    QueueFileLock,
    adopt_reserved_process_slot,
    assert_no_execution_hold,
    claim_next_entry,
    code_hash,
    code_hash_inventory,
    execution_control_lock_path,
    queue_status_path,
    validate_queue,
)
from stage_F0_worker_control import (  # noqa: E402
    EXECUTION_GATE,
    append_phase_event,
    frozen_held_run_ids,
    linux_process_identity,
    validate_execution_gate,
)


def _json_canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32 - 1)


def setup_deterministic_runtime(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32 - 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _capture_rng_state() -> Dict[str, object]:
    result: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": ([state.clone() for state in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_available() else []),
        "stream_derivation": "sha256(seed|dataset|setting|epoch|stream-v1)",
    }
    return result


def _restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda", [])
    if cuda_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def preserve_rng_state():
    state = _capture_rng_state()
    try:
        yield
    finally:
        _restore_rng_state(state)


def _hash_value(digest, value: object) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence")
        for item in value:
            _hash_value(digest, item)
    else:
        digest.update(type(value).__name__.encode("utf-8"))
        digest.update(repr(value).encode("utf-8"))


def stable_state_hash(value: object) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def _cpu_tree(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _sha_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str, value: object) -> None:
    absolute = os.path.abspath(path)
    parent = os.path.dirname(absolute)
    os.makedirs(parent, exist_ok=True)
    temp_path = absolute + ".tmp." + str(os.getpid())
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, absolute)
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_json(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object at {}".format(path))
    return value


def _execution_context() -> Tuple[Optional[str], str, str]:
    return (os.environ.get("F0_PHASE_LOG"),
            os.environ.get("F0_WORKER_ID", "direct"),
            os.environ.get("F0_INVOCATION_NONCE", "direct-" + str(os.getpid())))


def _execution_phase(phase: str, *, canonical: Optional[Mapping[str, object]] = None,
                     gpu: Optional[Mapping[str, object]] = None, **fields) -> None:
    log_path, worker_id, nonce = _execution_context()
    if log_path:
        append_phase_event(log_path, phase=phase, worker_id=worker_id,
                           nonce=nonce, canonical=canonical, gpu=gpu, **fields)


def _write_invocation_receipt(path: Optional[str], payload: Mapping[str, object]) -> None:
    if not path:
        return
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temp = target + ".tmp." + str(os.getpid())
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
    directory_fd = os.open(os.path.dirname(target), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _locked_owner_snapshot(lock: Optional[QueueFileLock]) -> Optional[dict]:
    """Read the exact owner JSON while its flock is still held by this child."""

    if lock is None:
        return None
    handle = lock.handle
    if handle.closed:
        raise PermissionError("cannot freeze an owner record after its lock was released")
    handle.flush()
    handle.seek(0)
    try:
        value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise PermissionError("locked owner record is unreadable at receipt commit") from exc
    if not isinstance(value, dict):
        raise PermissionError("locked owner record is not a JSON object")
    return dict(value)


def _write_completion_receipt(path: Optional[str], *, entry: Mapping[str, object],
                              output_root: str, status_file: str,
                              attempt_dir: str, summary: Mapping[str, object],
                              gpu_lock: Optional[QueueFileLock]) -> None:
    if not path:
        return
    status = _read_json(status_file)
    manifest_path = os.path.join(attempt_dir, "run_manifest.json")
    manifest = _read_json(manifest_path)
    if (summary.get("status") != "VALIDATION_COMPLETE"
            or status.get("status") != "VALIDATION_COMPLETE"
            or manifest.get("status") != "VALIDATION_COMPLETE"):
        raise PermissionError("runner did not commit a validation-complete transaction")
    identity_fields = ("run_id", "run_id_full_sha256", "canonical_key_json", "attempt_id")
    for field in identity_fields:
        expected = str(entry.get(field, ""))
        if expected != str(status.get(field, "")) or expected != str(manifest.get(field, "")):
            raise PermissionError("status/manifest identity mismatch for {}".format(field))
    checkpoint_names = ["final_checkpoint.pt", "selected_checkpoint.pt"]
    checkpoint_names.extend(sorted(
        name for name in os.listdir(attempt_dir)
        if name.startswith("checkpoint_epoch_") and name.endswith(".pt")))
    checkpoint_hashes = {}
    for name in checkpoint_names:
        checkpoint_path = os.path.join(attempt_dir, name)
        if os.path.isfile(checkpoint_path):
            checkpoint_hashes[name] = _sha_file(checkpoint_path)
    if "final_checkpoint.pt" not in checkpoint_hashes or "selected_checkpoint.pt" not in checkpoint_hashes:
        raise PermissionError("completed trajectory is missing final or selected checkpoint")
    from stage_F0_worker_control import sha256_file
    frozen_gpu_owner = _locked_owner_snapshot(gpu_lock)
    frozen_gpu_path = None if gpu_lock is None else os.path.abspath(gpu_lock.path)
    frozen_gpu_sha = None
    if frozen_gpu_owner is not None:
        if int(frozen_gpu_owner.get("pid", -1)) != os.getpid():
            raise PermissionError("locked GPU owner PID differs from the trajectory child")
        frozen_identity = ("run_id", "run_id_full_sha256", "attempt_id")
        for field in frozen_identity:
            if str(frozen_gpu_owner.get(field, "")) != str(entry.get(field, "")):
                raise PermissionError("locked GPU owner differs from trajectory for {}".format(
                    field))
        frozen_gpu_sha = hashlib.sha256(json.dumps(
            frozen_gpu_owner, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    _write_invocation_receipt(path, {
        "schema": "F0-CHILD-RECEIPT-v1", "status": "VALIDATION_COMPLETE",
        "worker_id": _execution_context()[1], "nonce": _execution_context()[2],
        "pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgrp(),
        "timestamp": time.time(), "run_id": entry["run_id"],
        "run_id_full_sha256": entry["run_id_full_sha256"],
        "canonical_key_json": entry["canonical_key_json"],
        "attempt_id": entry["attempt_id"], "status_path": os.path.abspath(status_file),
        "manifest_path": os.path.abspath(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "gpu_lock_owner": frozen_gpu_owner,
        "gpu_lock_path": frozen_gpu_path,
        "gpu_lock_owner_sha256": frozen_gpu_sha,
        "checkpoints": checkpoint_hashes,
        "queue_hash": entry.get("queue_hash"), "output_root": os.path.abspath(output_root),
    })


def _write_jsonl(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_canonical(row) + "\n")
    os.replace(temp_path, path)


def _atomic_torch_save(value: object, path: str) -> None:
    temp_path = path + ".tmp"
    torch.save(value, temp_path)
    os.replace(temp_path, path)


def _module_state_cpu(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _batch_fingerprint(row_ids: np.ndarray, x: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    for arr in (np.asarray(row_ids, dtype=np.int64),
                np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)):
        arr = np.ascontiguousarray(arr)
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(str(tuple(arr.shape)).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _process_peak_rss_bytes() -> Optional[int]:
    try:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError):
        return None
    # Linux reports KiB; macOS reports bytes.
    return peak * (1024 if sys.platform.startswith("linux") else 1)


def build_global_order(global_train_size: int, row_ids: np.ndarray, seed: int,
                       epoch: int, dataset: str = "movielens", setting: str = "natural",
                       stream: str = "main-order-v1") -> np.ndarray:
    """Filter a setting-wide order for a scope without reordering the pool."""

    if int(global_train_size) <= 0:
        raise ValueError("global training pool must be non-empty")
    selected = np.asarray(row_ids, dtype=np.int64)
    if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
        raise ValueError("scope row IDs must be one-dimensional and unique")
    if np.any(selected < 0) or np.any(selected >= int(global_train_size)):
        raise ValueError("scope row IDs fall outside the global training prefix")
    key = "{}|{}|{}|{}|{}".format(int(seed), dataset, setting, int(epoch), stream)
    rng = np.random.default_rng(_seed_from_text(key))
    global_order = rng.permutation(int(global_train_size)).astype(np.int64)
    membership = set(int(value) for value in selected.tolist())
    filtered = np.asarray([value for value in global_order if int(value) in membership], dtype=np.int64)
    if len(filtered) != len(selected) or set(filtered.tolist()) != membership:
        raise AssertionError("scope filter did not preserve every global permutation row")
    return filtered


def runtime_preflight(output_path: Optional[str] = None) -> Dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_rows: List[Dict[str, object]] = []
    if cuda_available:
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            free_bytes = None
            total_bytes = int(properties.total_memory)
            try:
                with torch.cuda.device(index):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
            except Exception:
                free_bytes = None
            gpu_rows.append({"index": index, "name": str(properties.name),
                             "total_memory_bytes": int(total_bytes),
                             "free_memory_bytes": (None if free_bytes is None else int(free_bytes)),
                             "uuid": str(getattr(properties, "uuid", ""))})
    usable_gpu_indices = [int(row["index"]) for row in gpu_rows
                          if row["free_memory_bytes"] is not None
                          and int(row["free_memory_bytes"]) > 0]
    nvidia_smi = {"available": False, "returncode": None, "stdout": "", "stderr": ""}
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,memory.used",
                                     "--format=csv,noheader,nounits"], check=False, capture_output=True,
                                    text=True, timeout=15)
            nvidia_smi = {"available": result.returncode == 0,
                          "returncode": int(result.returncode), "stdout": result.stdout.strip(),
                          "stderr": result.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            nvidia_smi["stderr"] = str(exc)
    disk = None
    if output_path:
        parent = os.path.abspath(output_path)
        while not os.path.exists(parent):
            parent = os.path.dirname(parent)
        usage = shutil.disk_usage(parent)
        disk = {"path": parent, "free_bytes": int(usage.free),
                "total_bytes": int(usage.total), "used_bytes": int(usage.used)}
    try:
        import psutil
        memory = {"total_bytes": int(psutil.virtual_memory().total),
                  "available_bytes": int(psutil.virtual_memory().available)}
    except Exception:
        memory = {"total_bytes": None, "available_bytes": None}
    return {"timestamp_unix": time.time(), "python": sys.version,
            "python_executable": os.path.abspath(sys.executable),
            "platform": platform.platform(), "torch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda or ""),
            "cuda_available": cuda_available, "device_count": device_count,
            "gpus": gpu_rows, "nvidia_smi": nvidia_smi,
            "cpu_count": os.cpu_count(), "ram": memory, "disk": disk,
            "resource_concurrency_policy": {
                "user_override": "concurrent-process-cap-v1",
                "max_concurrent_python_processes": MAX_CONCURRENT_PYTHON_PROCESSES,
                "max_concurrent_test_processes": MAX_CONCURRENT_PYTHON_PROCESSES,
                "max_concurrent_training_processes": min(
                    MAX_CONCURRENT_PYTHON_PROCESSES, len(usable_gpu_indices)),
                "max_training_processes_per_gpu": 1,
                "usable_gpu_count_for_this_preflight": len(usable_gpu_indices),
                "usable_gpu_indices": usable_gpu_indices,
            },
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "amp": False, "tf32": False}


def gpu_memory_preflight(preflight: Mapping[str, object], *, threshold: float = 0.90
                         ) -> Dict[str, object]:
    """Fail closed unless each candidate GPU has readable usage below threshold."""
    observed_at = float(time.time())
    rows = preflight.get("gpus", [])
    if not isinstance(rows, list):
        rows = []
    smi = preflight.get("nvidia_smi", {})
    smi_rows = []
    if isinstance(smi, Mapping) and smi.get("available") is True:
        for line in str(smi.get("stdout", "")).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 5:
                try:
                    row = {"uuid": parts[1], "total_mib": int(parts[2]),
                           "free_mib": int(parts[3]), "used_mib": int(parts[4])}
                    if (not row["uuid"] or row["total_mib"] <= 0 or row["free_mib"] < 0
                            or row["used_mib"] < 0 or row["free_mib"] > row["total_mib"]
                            or row["used_mib"] > row["total_mib"]
                            or row["free_mib"] + row["used_mib"] > row["total_mib"]):
                        raise ValueError("GPU memory values out of range")
                    smi_rows.append(row)
                except ValueError:
                    smi_rows.append(None)
            else:
                smi_rows.append(None)
    devices = []
    eligible = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            devices.append({"index": index, "readable": False,
                            "reason": "malformed CUDA memory row"})
            continue
        total = row.get("total_memory_bytes")
        free = row.get("free_memory_bytes")
        source = "torch.cuda.mem_get_info"
        if (isinstance(total, (int, float)) and isinstance(free, (int, float))
                and total > 0 and 0 <= free <= total):
            used = int(total - free)
            total, free = int(total), int(free)
        else:
            source = "nvidia-smi"
            fallback = smi_rows[index] if index < len(smi_rows) else None
            if isinstance(fallback, Mapping):
                total = int(fallback["total_mib"]) * 1024 * 1024
                free = int(fallback["free_mib"]) * 1024 * 1024
                used = int(fallback["used_mib"]) * 1024 * 1024
            else:
                devices.append({"index": index, "uuid": str(row.get("uuid", "")),
                                "readable": False,
                                "reason": "GPU memory query unavailable"})
                continue
        ratio = float(used) / float(total)
        if (not math.isfinite(ratio) or total <= 0 or used < 0 or free < 0
                or used > total or free > total or not str(row.get("uuid", ""))):
            devices.append({"index": index, "uuid": str(row.get("uuid", "")),
                            "readable": False, "reason": "GPU memory telemetry out of range"})
            continue
        allowed = ratio < float(threshold)
        devices.append({"index": index, "uuid": str(row.get("uuid", "")),
                        "memory_used_bytes": int(used),
                        "memory_free_bytes": int(free),
                        "memory_total_bytes": int(total),
                        "memory_used_fraction": ratio, "source": source,
                        "observed_at_unix": observed_at, "readable": True,
                        "eligible": allowed})
        if allowed:
            eligible.append(index)
    return {"policy": "memory_used / memory_total < 0.90 per target GPU",
            "threshold": float(threshold), "observed_at_unix": observed_at,
            "devices": devices, "eligible_gpu_indices": eligible,
            "go": bool(eligible)}


def resource_preflight_record(entry: Mapping[str, object], bundle,
                             runtime: Mapping[str, object]) -> Dict[str, object]:
    """Build an honest 3-model×2-batch resource preflight row.

    The preflight records hardware values that are already observable and
    leaves training throughput/checkpoint measurements explicitly pending for
    the first real search epoch.
    """

    disk = runtime.get("disk") if isinstance(runtime.get("disk"), Mapping) else {}
    ram = runtime.get("ram") if isinstance(runtime.get("ram"), Mapping) else {}
    batch_size = int(entry["config"]["batch_size"])
    train_rows = int(len(bundle.train_x))
    validation_rows = int(len(bundle.val_x))
    architecture_steps = 1 if str(entry["backbone"]) == "m3oe" else 0
    return {
        "schema_version": "F0-resource-preflight-v1",
        "status": "RESOURCE_PREFLIGHT_COMPLETE",
        "run_id": str(entry["run_id"]),
        "unit_key": str(entry["unit_key"]),
        "backbone": str(entry["backbone"]),
        "scope": str(entry["scope"]),
        "batch_size": batch_size,
        "matching_search_unit": str(entry["unit_key"]),
        "planned_train_rows": train_rows,
        "planned_main_steps_per_epoch": int(math.ceil(train_rows / float(batch_size))),
        "planned_architecture_steps_per_epoch": architecture_steps,
        "planned_validation_rows": validation_rows,
        "planned_validation_batches": int(math.ceil(validation_rows / float(batch_size))),
        "epoch_wall_seconds": None,
        "examples_per_second": None,
        "cpu_ram_total_bytes": ram.get("total_bytes"),
        "cpu_ram_available_bytes": ram.get("available_bytes"),
        "cpu_peak_rss_bytes": _process_peak_rss_bytes(),
        "gpu_preflight_devices": list(runtime.get("gpus", [])),
        "gpu_peak_allocated_bytes": None,
        "gpu_peak_reserved_bytes": None,
        "storage_free_bytes": disk.get("free_bytes"),
        "checkpoint_bytes": None,
        "measurement_status": "PENDING_FIRST_SEARCH_EPOCH",
        "training_started": False,
        "runtime_preflight": dict(runtime),
    }


def validate_run_gate(gate: Mapping[str, object], *, entry: Mapping[str, object],
                      queue_hash: str, dataset_file_hash: str) -> None:
    """Fail closed until data, implementation, GPU and storage gates pass."""

    if str(gate.get("protocol_version", "")) != PROTOCOL_VERSION:
        raise PermissionError("run gate protocol_version mismatch")
    if gate.get("data_audit_verdict", gate.get("data_gate_status")) != "PASS":
        raise PermissionError("MovieLens data audit gate is not PASS")
    if gate.get("implementation_review_verdict", gate.get("review_verdict")) not in (
            "PASS", "PASS WITH FIXES"):
        raise PermissionError("implementation review gate is not PASS/PASS WITH FIXES")
    if gate.get("gpu_preflight_status", gate.get("gpu_preflight")) != "PASS":
        raise PermissionError("GPU preflight gate is not PASS")
    if str(gate.get("dataset_file_sha256", "")) != str(dataset_file_hash):
        raise PermissionError("run gate MovieLens file hash mismatch")
    if str(gate.get("protocol_hash", "")) != str(entry.get("protocol_hash", "")):
        raise PermissionError("run gate protocol hash mismatch")
    if str(gate.get("code_hash", "")) != str(entry.get("code_hash", "")):
        raise PermissionError("run gate implementation/test hash mismatch")
    if str(gate.get("queue_hash", "")) != str(queue_hash):
        raise PermissionError("run gate queue hash mismatch")
    required_bytes = int(gate.get("required_storage_bytes", 0))
    reserved_bytes = int(gate.get("reserved_storage_bytes", 0))
    if required_bytes <= 0 or reserved_bytes < required_bytes:
        raise PermissionError("run gate lacks 1.3x-block storage reservation")


class F0Runner:
    """One queue trajectory.  It writes raw metrics and state, never conclusions."""

    def __init__(self, entry: Mapping[str, object], bundle, attempt_dir: str,
                 device: str = "cuda:0", *, code_digest: Optional[str] = None,
                 protocol_digest: Optional[str] = None,
                 enforce_cuda: bool = True, test_mode: bool = False):
        self.entry = dict(entry)
        self.bundle = bundle
        self.attempt_dir = os.path.abspath(attempt_dir)
        self.device = torch.device(device)
        self.enforce_cuda = bool(enforce_cuda)
        self.test_mode = bool(test_mode)
        self.code_digest = str(code_digest or self.entry.get("code_hash", ""))
        self.protocol_digest = str(protocol_digest or protocol_hash())
        self.phase = str(self.entry.get("phase", ""))
        self.intent = str(self.entry.get("intent", ""))
        self.backbone = str(self.entry.get("backbone", "")).lower()
        self.scope = str(self.entry.get("scope", "")).lower()
        self.unit = str(self.entry.get("unit_key", ""))
        self.seed = int(self.entry["seed"])
        self.config = {"lr": float(self.entry["config"]["lr"]),
                       "batch_size": int(self.entry["config"]["batch_size"]),
                       "weight_decay": float(self.entry["config"]["weight_decay"])}
        self.domain_num = 3 if self.scope == "mdmt" else 1
        self.task_num = 2
        self.physical_domain = scope_instance(self.scope)
        self.width = dict(self.entry["model_width"])
        self.c = int(self.width["c"])
        self.epoch_limit = int(self.entry.get("max_epochs", EPOCHS))
        if self.phase not in ("search", "formal") or self.intent != self.phase:
            raise ValueError("F0Runner accepts only search/formal queue entries")
        if self.scope not in ("mdmt", "sdmt-d0", "sdmt-d1", "sdmt-d2"):
            raise ValueError("unsupported run scope {}".format(self.scope))
        if self.entry.get("test_access") is not False:
            raise PermissionError("training queue entry does not explicitly disable test access")
        if str(self.entry.get("protocol_version")) != PROTOCOL_VERSION:
            raise ValueError("queue protocol version mismatch")
        if str(self.entry.get("protocol_hash")) != self.protocol_digest:
            raise ValueError("queue protocol hash mismatch")
        if str(self.entry.get("code_hash")) != self.code_digest:
            raise ValueError("queue code hash mismatch")
        if not self.test_mode and self.epoch_limit != EPOCHS:
            raise ValueError("research search/formal trajectories require exactly 50 epochs")
        expected_width = width_for(self.backbone, self.scope)
        if not self.test_mode and {k: int(v) for k, v in self.width.items()} != expected_width:
            raise ValueError("queue width differs from the frozen F0-B1 width table")
        expected_hashes = split_hashes(bundle)
        if dict(self.entry.get("split_hashes", {})) != expected_hashes:
            raise ValueError("queue split hashes do not match this natural-pool bundle")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is visible")
        if self.enforce_cuda and self.device.type != "cuda":
            raise RuntimeError("F0-B1 research runs require CUDA; CPU fallback is forbidden")
        if self.device.type == "cuda" and self.device.index is not None \
                and self.device.index >= torch.cuda.device_count():
            raise RuntimeError("requested CUDA index is outside visible devices")
        if len(bundle.train_x) == 0 or len(bundle.val_x) == 0:
            raise ValueError("scope train and validation pools must be non-empty")
        if not self.test_mode and len(bundle.features.columns) != 8:
            raise ValueError("MovieLens schema must contain seven sparse features and one dense feature")
        self.train_x = np.asarray(bundle.train_x, dtype=np.float32)
        self.train_y = np.asarray(bundle.train_y, dtype=np.float32)
        self.train_ids = np.asarray(bundle.train_row_ids, dtype=np.int64)
        self.val_x = np.asarray(bundle.val_x, dtype=np.float32)
        self.val_y = np.asarray(bundle.val_y, dtype=np.float32)
        self.val_ids = np.asarray(bundle.val_row_ids, dtype=np.int64)
        self.global_train_size = int(bundle.preprocessing_metadata.get(
            "global_base_train_size", max(int(self.train_ids.max()) + 1, len(self.train_ids))))
        self.train_id_to_position = {int(row_id): index for index, row_id in enumerate(self.train_ids)}
        self.val_domain_ids = self.val_x[:, -1].astype(np.int64)
        self.train_domain_ids = self.train_x[:, -1].astype(np.int64)
        self.local_to_physical = tuple(int(v) for v in bundle.local_to_physical)
        setup_deterministic_runtime(self.seed, self.device)
        self.model, self.initialization_audit = build_semantically_paired_model(
            bundle.features, backbone=self.backbone, domain_num=self.domain_num,
            task_num=self.task_num, c=self.c, seed=self.seed,
            physical_domain=self.physical_domain,
            # Every scope reconstructs the exact same setting/seed/backbone
            # MDMT source; scope identity only affects the SDMT target init.
            model_seed=_seed_from_text("{}|{}|natural|init-root-v1".format(
                self.seed, self.backbone)))
        self.width_audit = (None if self.test_mode else
                            validate_frozen_width(self.backbone, self.scope, self.model))
        self.model = self.model.to(self.device)
        named = list(self.model.named_parameters())
        self.arch_named = [(name, param) for name, param in named
                           if name.startswith(("_weight_exp_", "_weight_bal_"))]
        self.arch_params = [param for _, param in self.arch_named]
        self.main_named = [(name, param) for name, param in named
                           if not name.startswith(("_weight_exp_", "_weight_bal_"))]
        self.main_params = [param for _, param in self.main_named]
        self.main_optimizer = torch.optim.Adam(
            self.main_params, lr=self.config["lr"], betas=ADAM_BETAS,
            eps=ADAM_EPS, weight_decay=self.config["weight_decay"])
        self.arch_optimizer = (torch.optim.Adam(
            self.arch_params, lr=self.config["lr"] * ARCH_LR_RATIO,
            betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=self.config["weight_decay"])
            if self.arch_params else None)
        self.shares = self._training_domain_shares()
        self.main_steps = 0
        self.arch_steps = 0
        self.examples_processed = 0
        self.arch_examples_processed = 0
        self.completed_epochs = 0
        self.selected_epoch: Optional[int] = None
        self.selected_score = float("-inf")
        self.validation_history: List[Dict[str, object]] = []
        self.metrics_rows: List[Dict[str, object]] = []
        self.resource_epoch_rows: List[Dict[str, object]] = []
        self.batch_fingerprints: List[str] = []
        self.arch_batch_fingerprints: List[str] = []
        self.resume_evidence: List[Dict[str, object]] = []
        self.model_initial_hash = stable_state_hash(self.model.state_dict())
        self.gpu_lock_path: Optional[str] = None
        self.fault_injector = None
        # Real-process control tests install deterministic pre/post receipt
        # barriers here. Production leaves the hook unset.
        self.interruption_receipt_hook = None

    def _training_domain_shares(self) -> Dict[int, float]:
        counts = {d: int((self.train_domain_ids == d).sum()) for d in range(self.domain_num)}
        total = float(sum(counts.values()))
        if total <= 0 or any(count == 0 for count in counts.values()):
            raise ValueError("every local training domain must contain rows")
        return {d: counts[d] / total for d in counts}

    def _assert_class_coverage(self) -> None:
        for split_name, labels, domains in (("validation", self.val_y, self.val_domain_ids),):
            for domain in range(self.domain_num):
                for task in range(self.task_num):
                    cell_y = labels[domains == domain, task]
                    if not len(cell_y) or len(np.unique(cell_y)) < 2:
                        physical = self.local_to_physical[domain]
                        raise ValueError("{} cell d{}t{} is empty or single-class".format(
                            split_name, physical, task))

    def _ordered_positions(self, epoch: int, stream: str) -> Tuple[np.ndarray, np.ndarray]:
        order_ids = build_global_order(self.global_train_size, self.train_ids, self.seed,
                                       epoch, "movielens", "natural", stream)
        positions = np.asarray([self.train_id_to_position[int(row_id)] for row_id in order_ids],
                               dtype=np.int64)
        return order_ids, positions

    def _batch(self, positions: np.ndarray, start: int, batch_size: int):
        selected_positions = positions[start:start + int(batch_size)]
        row_ids = self.train_ids[selected_positions]
        batch_x = self.train_x[selected_positions]
        batch_y = self.train_y[selected_positions]
        x_device = torch.from_numpy(np.ascontiguousarray(batch_x.T)).to(
            device=self.device, dtype=torch.float32)
        y_device = torch.from_numpy(np.ascontiguousarray(batch_y)).to(
            device=self.device, dtype=torch.float32)
        return row_ids, batch_x, batch_y, x_device, y_device

    def _eval_validation(self) -> Dict[str, object]:
        model_state_before = stable_state_hash(self.model.state_dict())
        main_state_before = stable_state_hash(self.main_optimizer.state_dict())
        arch_state_before = (stable_state_hash(self.arch_optimizer.state_dict())
                             if self.arch_optimizer is not None else "absent")
        rng_before = stable_state_hash(_capture_rng_state())
        training_mode = self.model.training
        prediction_parts: List[np.ndarray] = []
        labels_parts: List[np.ndarray] = []
        with preserve_rng_state():
            self.model.eval()
            with torch.no_grad():
                batch_size = int(self.config["batch_size"])
                for start in range(0, len(self.val_x), batch_size):
                    batch_x = self.val_x[start:start + batch_size]
                    tensor = torch.from_numpy(np.ascontiguousarray(batch_x.T)).to(
                        device=self.device, dtype=torch.float32)
                    prediction = self.model(tensor)
                    if not bool(torch.isfinite(prediction).all().detach().cpu()):
                        raise FloatingPointError("non-finite validation prediction")
                    prediction_parts.append(prediction.detach().cpu().numpy().astype(np.float32))
                    labels_parts.append(self.val_y[start:start + batch_size])
        self.model.train(training_mode)
        predictions = np.concatenate(prediction_parts, axis=0)
        labels = np.concatenate(labels_parts, axis=0)
        result = evaluate_cells(labels, predictions, self.val_domain_ids,
                                self.local_to_physical, self.task_num)
        model_state_after = stable_state_hash(self.model.state_dict())
        main_state_after = stable_state_hash(self.main_optimizer.state_dict())
        arch_state_after = (stable_state_hash(self.arch_optimizer.state_dict())
                            if self.arch_optimizer is not None else "absent")
        rng_after = stable_state_hash(_capture_rng_state())
        if (model_state_before, main_state_before, arch_state_before, rng_before) != \
                (model_state_after, main_state_after, arch_state_after, rng_after):
            raise AssertionError("validation evaluation changed model, optimizer, or RNG state")
        return result

    def _train_epoch(self, epoch: int) -> Dict[str, object]:
        self.model.train()
        order_ids, positions = self._ordered_positions(epoch, "main-order-v1")
        batch_size = int(self.config["batch_size"])
        sample_losses: List[torch.Tensor] = []
        equal_losses: List[torch.Tensor] = []
        batch_sizes: List[int] = []
        main_domain_exposure = {d: 0 for d in range(self.domain_num)}
        main_task_positive = {"d{}t{}".format(self.local_to_physical[d], t): 0
                              for d in range(self.domain_num) for t in range(self.task_num)}
        for step, start in enumerate(range(0, len(positions), batch_size)):
            row_ids, batch_x, batch_y, x_device, y_device = self._batch(positions, start, batch_size)
            fingerprint = _batch_fingerprint(row_ids, batch_x, batch_y)
            self.batch_fingerprints.append(fingerprint)
            prediction = self.model(x_device)
            if not bool(torch.isfinite(prediction).all().detach().cpu()):
                raise FloatingPointError("non-finite main training prediction")
            loss = sample_bce(prediction, y_device)
            domain_ids = x_device[-1, :].long()
            equal = equal_diagnostic_bce(prediction, y_device, domain_ids,
                                         self.shares, self.domain_num)
            if not bool(torch.isfinite(loss).detach().cpu()) or \
                    not bool(torch.isfinite(equal).detach().cpu()):
                raise FloatingPointError("non-finite Sample/Equal objective")
            for parameter in self.arch_params:
                parameter.requires_grad_(False)
            for parameter in self.main_params:
                parameter.requires_grad_(True)
            self.main_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in self.main_named:
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().detach().cpu()):
                    raise FloatingPointError("non-finite main gradient {}".format(name))
            self.main_optimizer.step()
            for name, parameter in self.main_named:
                if not bool(torch.isfinite(parameter).all().detach().cpu()):
                    raise FloatingPointError("non-finite updated main parameter {}".format(name))
            count = int(len(batch_y))
            sample_losses.append(loss.detach())
            equal_losses.append(equal.detach())
            batch_sizes.append(count)
            self.main_steps += 1
            self.examples_processed += count
            for d in range(self.domain_num):
                mask = batch_x[:, -1].astype(np.int64) == d
                main_domain_exposure[d] += int(mask.sum())
                physical = self.local_to_physical[d]
                for task in range(self.task_num):
                    key = "d{}t{}".format(physical, task)
                    main_task_positive[key] += int(batch_y[mask, task].sum())

        if not sample_losses:
            raise RuntimeError("epoch has no main optimizer steps")
        sample_loss = split_weighted_sample_loss(sample_losses, batch_sizes)
        equal_loss = split_weighted_sample_loss(equal_losses, batch_sizes)
        arch_ids: List[int] = []
        arch_exposure: Dict[str, object] = {"row_count": 0, "domain_counts": {},
                                          "task_positive_counts": {}}
        if self.arch_optimizer is not None:
            arch_order_ids, arch_positions = self._ordered_positions(epoch, "arch-order-v1")
            arch_count = min(batch_size, len(arch_positions))
            arch_order_ids = arch_order_ids[:arch_count]
            arch_positions = arch_positions[:arch_count]
            arch_ids = [int(value) for value in arch_order_ids.tolist()]
            row_ids, batch_x, batch_y, x_device, y_device = self._batch(
                arch_positions, 0, arch_count)
            fingerprint = _batch_fingerprint(row_ids, batch_x, batch_y)
            self.arch_batch_fingerprints.append(fingerprint)
            for parameter in self.main_params:
                parameter.requires_grad_(False)
            for parameter in self.arch_params:
                parameter.requires_grad_(True)
            self.arch_optimizer.zero_grad(set_to_none=True)
            prediction = self.model(x_device)
            arch_loss = sample_bce(prediction, y_device)
            if not bool(torch.isfinite(arch_loss).detach().cpu()):
                raise FloatingPointError("non-finite architecture objective")
            arch_loss.backward()
            for name, parameter in self.arch_named:
                if parameter.grad is None:
                    raise FloatingPointError("architecture parameter has no gradient: {}".format(name))
                if not bool(torch.isfinite(parameter.grad).all().detach().cpu()):
                    raise FloatingPointError("non-finite architecture gradient {}".format(name))
            self.arch_optimizer.step()
            for name, parameter in self.arch_named:
                if not bool(torch.isfinite(parameter).all().detach().cpu()):
                    raise FloatingPointError("non-finite architecture parameter {}".format(name))
            self.arch_steps += 1
            self.arch_examples_processed += arch_count
            arch_domains = batch_x[:, -1].astype(np.int64)
            arch_exposure = {
                "row_count": arch_count,
                "domain_counts": {"d{}".format(self.local_to_physical[d]): int((arch_domains == d).sum())
                                  for d in range(self.domain_num)},
                "task_positive_counts": {
                    "d{}t{}".format(self.local_to_physical[d], t):
                    int(batch_y[arch_domains == d, t].sum())
                    for d in range(self.domain_num) for t in range(self.task_num)},
                "row_ids": arch_ids,
            }
        for parameter in self.main_params + self.arch_params:
            parameter.requires_grad_(True)
        return {
            "train_sample_bce": float(sample_loss.detach().cpu()),
            "train_equal_diagnostic_bce": float(equal_loss.detach().cpu()),
            "main_order_row_ids": [int(value) for value in order_ids.tolist()],
            "architecture_order_row_ids": arch_ids,
            "main_domain_exposure": {"d{}".format(self.local_to_physical[d]): count
                                      for d, count in main_domain_exposure.items()},
            "main_task_positive_counts": main_task_positive,
            "architecture_exposure": arch_exposure,
            "batch_fingerprints": self.batch_fingerprints[-len(batch_sizes):],
            "batches_in_epoch": len(batch_sizes),
        }

    def _state_payload(self, completed_epochs: int,
                       last_train_record: Optional[Mapping[str, object]]) -> Dict[str, object]:
        next_fingerprint = ""
        if completed_epochs < self.epoch_limit:
            next_ids, next_positions = self._ordered_positions(completed_epochs, "main-order-v1")
            _, bx, by, _, _ = self._batch(next_positions, 0, int(self.config["batch_size"]))
            next_fingerprint = _batch_fingerprint(next_ids[:len(bx)], bx, by)
        model_state = _module_state_cpu(self.model)
        main_state = _cpu_tree(self.main_optimizer.state_dict())
        arch_state = (_cpu_tree(self.arch_optimizer.state_dict())
                      if self.arch_optimizer is not None else None)
        validation_history = copy.deepcopy(self.validation_history)
        return {
            "schema_version": "F0-epoch-checkpoint-v1",
            "run_id": str(self.entry["run_id"]),
            "canonical_key_json": str(self.entry["canonical_key_json"]),
            "attempt_id": str(self.entry["attempt_id"]),
            "phase": self.phase, "unit_key": self.unit, "config_id": str(self.entry["config_id"]),
            "seed": self.seed, "backbone": self.backbone, "scope": self.scope,
            "model_width": self.width, "dataset_file_sha256": str(
                self.bundle.preprocessing_metadata.get("raw_file_sha256", "")),
            "split_hashes": dict(self.entry["split_hashes"]),
            "protocol_hash": self.protocol_digest, "code_hash": self.code_digest,
            "completed_epochs": int(completed_epochs),
            "batch_position": 0 if last_train_record is None else int(last_train_record["batches_in_epoch"]),
            "epoch_boundary_checkpoint": True,
            "iterator_permutation": (None if last_train_record is None else
                                     list(last_train_record["main_order_row_ids"])),
            "architecture_permutation": (None if last_train_record is None else
                                          list(last_train_record["architecture_order_row_ids"])),
            "last_train_record": copy.deepcopy(last_train_record),
            "model_state": model_state,
            "main_optimizer_state": main_state,
            "architecture_optimizer_state": arch_state,
            "architecture_optimizer_absent": self.arch_optimizer is None,
            "rng_state": _capture_rng_state(),
            "stream_seeds": {
                "main": ["{}|movielens|natural|{}|main-order-v1".format(self.seed, e)
                         for e in range(completed_epochs)],
                "architecture": ["{}|movielens|natural|{}|arch-order-v1".format(self.seed, e)
                                 for e in range(completed_epochs) if self.arch_optimizer is not None],
            },
            "next_batch_fingerprint": next_fingerprint,
            "main_steps": self.main_steps,
            "architecture_steps": self.arch_steps,
            "examples_processed": self.examples_processed,
            "architecture_examples_processed": self.arch_examples_processed,
            "selected_epoch": self.selected_epoch,
            "selected_validation_macro_auc": self.selected_score,
            "validation_history": validation_history,
            "metrics_rows": copy.deepcopy(self.metrics_rows),
            "model_state_hash": stable_state_hash(model_state),
            "main_optimizer_state_hash": stable_state_hash(main_state),
            "architecture_optimizer_state_hash": (stable_state_hash(arch_state)
                                                  if arch_state is not None else "absent"),
        }

    def _save_checkpoint(self, name: str, completed_epochs: int,
                         last_train_record: Optional[Mapping[str, object]]) -> str:
        path = os.path.join(self.attempt_dir, name)
        payload = self._state_payload(completed_epochs, last_train_record)
        if name == "selected_checkpoint.pt":
            payload["snapshot_role"] = "selected"
        elif name == "final_checkpoint.pt":
            payload["snapshot_role"] = "final"
        elif name.startswith("checkpoint_epoch_"):
            payload["snapshot_role"] = "fixed"
        elif name.startswith("resume_epoch_"):
            payload["snapshot_role"] = "resume"
        else:
            raise ValueError("unknown checkpoint projection name {}".format(name))
        _atomic_torch_save(payload, path)
        return path

    def _inject_fault(self, commit_point: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(str(commit_point))

    def _validate_snapshot(self, path: str, payload: Mapping[str, object], *,
                           role: str, expected_epoch: int) -> None:
        if not os.path.isfile(path):
            raise RuntimeError("missing committed {} snapshot {}".format(role, path))
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if snapshot.get("snapshot_role") != role:
            raise RuntimeError("{} snapshot role mismatch".format(role))
        identity = ("run_id", "canonical_key_json", "attempt_id", "protocol_hash",
                    "code_hash", "unit_key", "config_id", "seed")
        if any(snapshot.get(key) != payload.get(key) for key in identity):
            raise RuntimeError("{} snapshot identity mismatch".format(role))
        if stable_state_hash(snapshot["model_state"]) != snapshot.get("model_state_hash"):
            raise RuntimeError("{} snapshot model-state hash mismatch".format(role))
        if stable_state_hash(snapshot["main_optimizer_state"]) != \
                snapshot.get("main_optimizer_state_hash"):
            raise RuntimeError("{} snapshot optimizer-state hash mismatch".format(role))
        if role == "selected" and int(snapshot.get("selected_epoch", -1)) != int(expected_epoch):
            raise RuntimeError("selected snapshot epoch does not match frozen selected epoch")
        if role == "fixed" and int(snapshot.get("completed_epochs", -1)) != int(expected_epoch) + 1:
            raise RuntimeError("fixed snapshot is not the expected epoch boundary")

    def _recover_snapshot_projections(self, payload: Mapping[str, object]) -> None:
        """Rebuild only projections whose state is present in the epoch commit."""

        completed = int(payload["completed_epochs"])
        latest_epoch = completed - 1
        selected_epoch = payload.get("selected_epoch")
        selected_path = os.path.join(self.attempt_dir, "selected_checkpoint.pt")
        if selected_epoch is None:
            raise RuntimeError("committed checkpoint has no selected validation epoch")
        selected_epoch = int(selected_epoch)
        if selected_epoch == latest_epoch:
            # The selected projection is derived from this authoritative epoch
            # commit. Rewriting it also repairs a stale projection left by a
            # crash between the resume commit and selected-file replacement.
            selected_payload = dict(payload)
            selected_payload["snapshot_role"] = "selected"
            _atomic_torch_save(selected_payload, selected_path)
        elif not os.path.exists(selected_path):
            raise RuntimeError("selected snapshot is missing and cannot be reconstructed")
        self._validate_snapshot(selected_path, payload, role="selected",
                                expected_epoch=selected_epoch)

        if self.phase == "formal":
            for fixed_epoch in FIXED_CHECKPOINT_EPOCHS:
                if fixed_epoch >= completed:
                    continue
                fixed_path = os.path.join(
                    self.attempt_dir, "checkpoint_epoch_{:02d}.pt".format(fixed_epoch))
                if fixed_epoch == latest_epoch:
                    fixed_payload = dict(payload)
                    fixed_payload["snapshot_role"] = "fixed"
                    _atomic_torch_save(fixed_payload, fixed_path)
                elif not os.path.exists(fixed_path):
                    raise RuntimeError("fixed snapshot is missing and cannot be reconstructed")
                fixed_payload = torch.load(fixed_path, map_location="cpu", weights_only=False)
                self._validate_snapshot(fixed_path, payload, role="fixed",
                                        expected_epoch=fixed_epoch)
                if int(fixed_payload.get("completed_epochs", -1)) != fixed_epoch + 1:
                    raise RuntimeError("fixed snapshot completion boundary mismatch")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("snapshot_role") != "resume":
            raise RuntimeError("resume path must be an epoch-boundary resume snapshot")
        required = {
            "run_id": str(self.entry["run_id"]),
            "canonical_key_json": str(self.entry["canonical_key_json"]),
            "attempt_id": str(self.entry["attempt_id"]),
            "protocol_hash": self.protocol_digest,
            "code_hash": self.code_digest,
            "unit_key": self.unit,
            "config_id": str(self.entry["config_id"]),
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise RuntimeError("resume checkpoint identity mismatch at {}".format(key))
        if payload.get("epoch_boundary_checkpoint") is not True:
            raise RuntimeError("resume checkpoint is not an exact epoch boundary")
        if payload.get("architecture_optimizer_absent") != (self.arch_optimizer is None):
            raise RuntimeError("resume checkpoint architecture-optimizer presence mismatch")
        if payload.get("dataset_file_sha256") != self.bundle.preprocessing_metadata.get("raw_file_sha256", ""):
            raise RuntimeError("resume checkpoint MovieLens file hash mismatch")
        if stable_state_hash(payload["model_state"]) != payload["model_state_hash"]:
            raise RuntimeError("resume checkpoint model state hash mismatch")
        if stable_state_hash(payload["main_optimizer_state"]) != payload["main_optimizer_state_hash"]:
            raise RuntimeError("resume checkpoint main optimizer state hash mismatch")
        arch_state = payload.get("architecture_optimizer_state")
        actual_arch_hash = stable_state_hash(arch_state) if arch_state is not None else "absent"
        if actual_arch_hash != payload["architecture_optimizer_state_hash"]:
            raise RuntimeError("resume checkpoint architecture optimizer state hash mismatch")
        if dict(payload.get("split_hashes", {})) != dict(self.entry["split_hashes"]):
            raise RuntimeError("resume checkpoint split identity mismatch")
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.main_optimizer.load_state_dict(payload["main_optimizer_state"])
        if self.arch_optimizer is not None:
            self.arch_optimizer.load_state_dict(arch_state)
        _restore_rng_state(payload["rng_state"])
        self.completed_epochs = int(payload["completed_epochs"])
        self.main_steps = int(payload["main_steps"])
        self.arch_steps = int(payload["architecture_steps"])
        self.examples_processed = int(payload["examples_processed"])
        self.arch_examples_processed = int(payload["architecture_examples_processed"])
        self.selected_epoch = (None if payload["selected_epoch"] is None
                               else int(payload["selected_epoch"]))
        self.selected_score = float(payload["selected_validation_macro_auc"])
        self.validation_history = copy.deepcopy(payload["validation_history"])
        self.metrics_rows = copy.deepcopy(payload["metrics_rows"])
        if self.completed_epochs > 0 and self.completed_epochs < self.epoch_limit:
            ids, positions = self._ordered_positions(self.completed_epochs, "main-order-v1")
            _, bx, by, _, _ = self._batch(positions, 0, int(self.config["batch_size"]))
            next_fp = _batch_fingerprint(ids[:len(bx)], bx, by)
            if next_fp != payload.get("next_batch_fingerprint"):
                raise RuntimeError("resume next-batch fingerprint mismatch")
        self.resume_evidence.append({
            "checkpoint": os.path.abspath(checkpoint_path),
            "checkpoint_sha256": _sha_file(checkpoint_path),
            "completed_epochs": self.completed_epochs,
            "model_state_hash": stable_state_hash(self.model.state_dict()),
            "main_optimizer_state_hash": stable_state_hash(self.main_optimizer.state_dict()),
            "architecture_optimizer_state_hash": (stable_state_hash(self.arch_optimizer.state_dict())
                                                  if self.arch_optimizer is not None else "absent"),
            "next_batch_fingerprint": (payload.get("next_batch_fingerprint", "")
                                       if self.completed_epochs < self.epoch_limit else ""),
            "exact_epoch_boundary_resume": True,
        })
        self._recover_snapshot_projections(payload)
        # Discard rows that could have been flushed after the last atomic
        # checkpoint but before a process interruption.
        self._flush_metrics()
        self._flush_resource_metrics()

    def _flush_metrics(self) -> None:
        rows = []
        for row in self.metrics_rows:
            item = dict(row)
            item["selected_checkpoint"] = int(
                self.selected_epoch is not None and int(item["epoch"]) == self.selected_epoch)
            rows.append(item)
        _write_jsonl(os.path.join(self.attempt_dir, "metrics_validation.jsonl"), rows)

    def _flush_resource_metrics(self) -> None:
        output_path = os.path.join(self.attempt_dir, "resource_epoch_metrics.jsonl")
        prior_by_epoch = {}
        if os.path.isfile(output_path):
            with open(output_path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    prior = json.loads(line)
                    prior_by_epoch[int(prior["epoch"])] = prior
        rows = []
        for metric in self.metrics_rows:
            epoch = int(metric["epoch"])
            prior = prior_by_epoch.get(epoch, {})
            resume_path = os.path.join(
                self.attempt_dir, "resume_epoch_{:03d}.pt".format(epoch + 1))
            selected_path = os.path.join(self.attempt_dir, "selected_checkpoint.pt")
            fixed_path = os.path.join(
                self.attempt_dir, "checkpoint_epoch_{:02d}.pt".format(epoch))
            row = {
                "epoch": epoch,
                "epoch_wall_seconds": metric.get("epoch_wall_seconds"),
                "examples_per_second": metric.get("examples_per_second"),
                "cpu_peak_rss_bytes": metric.get("cpu_peak_rss_bytes"),
                "gpu_peak_allocated_bytes": metric.get("gpu_peak_allocated_bytes"),
                "gpu_peak_reserved_bytes": metric.get("gpu_peak_reserved_bytes"),
                "storage_free_bytes_after_commit": (
                    int(shutil.disk_usage(self.attempt_dir).free)
                    if os.path.isfile(resume_path) or "storage_free_bytes_after_commit" not in prior
                    else prior["storage_free_bytes_after_commit"]),
                "resume_checkpoint_bytes": (os.path.getsize(resume_path)
                                             if os.path.isfile(resume_path) else
                                             prior.get("resume_checkpoint_bytes")),
                "selected_checkpoint_bytes": (os.path.getsize(selected_path)
                                              if os.path.isfile(selected_path) and
                                              self.selected_epoch == epoch else
                                              prior.get("selected_checkpoint_bytes")),
                "fixed_checkpoint_bytes": (os.path.getsize(fixed_path)
                                           if os.path.isfile(fixed_path) else
                                           prior.get("fixed_checkpoint_bytes")),
                "resource_measurement_status": "MEASURED",
            }
            rows.append(row)
        self.resource_epoch_rows = rows
        _write_jsonl(output_path, rows)

    def _next_batch_fingerprint_at(self, completed_epochs: int) -> str:
        if int(completed_epochs) >= self.epoch_limit:
            return ""
        row_ids, positions = self._ordered_positions(int(completed_epochs), "main-order-v1")
        chosen = positions[:int(self.config["batch_size"])]
        batch_x = self.train_x[chosen]
        batch_y = self.train_y[chosen]
        return _batch_fingerprint(row_ids[:len(chosen)], batch_x, batch_y)

    def _diagnostic_state(self, completed_epochs: int) -> Dict[str, object]:
        gradients = {name: (None if parameter.grad is None else
                            parameter.grad.detach().cpu().clone())
                     for name, parameter in self.model.named_parameters()}
        return {
            "model": stable_state_hash(self.model.state_dict()),
            "main_optimizer": stable_state_hash(self.main_optimizer.state_dict()),
            "architecture_optimizer": (stable_state_hash(self.arch_optimizer.state_dict())
                                       if self.arch_optimizer is not None else "absent"),
            "rng": stable_state_hash(_capture_rng_state()),
            "gradients": stable_state_hash(gradients),
            "requires_grad": stable_state_hash({name: bool(parameter.requires_grad)
                                                  for name, parameter in self.model.named_parameters()}),
            "module_training_modes": stable_state_hash({
                name: bool(module.training) for name, module in self.model.named_modules()}),
            "next_batch_fingerprint": self._next_batch_fingerprint_at(completed_epochs),
        }

    def _capture_live_diagnostic_state(self) -> Dict[str, object]:
        return {
            "model_state": _module_state_cpu(self.model),
            "main_optimizer_state": _cpu_tree(self.main_optimizer.state_dict()),
            "architecture_optimizer_state": (_cpu_tree(self.arch_optimizer.state_dict())
                                              if self.arch_optimizer is not None else None),
            "rng_state": _capture_rng_state(),
            "gradients": {name: (None if parameter.grad is None else
                                  parameter.grad.detach().cpu().clone())
                          for name, parameter in self.model.named_parameters()},
            "requires_grad": {name: bool(parameter.requires_grad)
                              for name, parameter in self.model.named_parameters()},
            "module_training_modes": {name: bool(module.training)
                                      for name, module in self.model.named_modules()},
        }

    def _restore_live_diagnostic_state(self, state: Mapping[str, object]) -> None:
        self.model.load_state_dict(state["model_state"], strict=True)
        self.main_optimizer.load_state_dict(state["main_optimizer_state"])
        if self.arch_optimizer is not None:
            self.arch_optimizer.load_state_dict(state["architecture_optimizer_state"])
        _restore_rng_state(state["rng_state"])
        parameters = dict(self.model.named_parameters())
        for name, parameter in parameters.items():
            gradient = state["gradients"][name]
            parameter.grad = (None if gradient is None else
                              gradient.to(device=parameter.device, dtype=parameter.dtype).clone())
            parameter.requires_grad_(bool(state["requires_grad"][name]))
        for name, module in self.model.named_modules():
            module.training = bool(state["module_training_modes"][name])

    def _probe_positions(self, x: np.ndarray, row_ids: np.ndarray,
                         split_name: str) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        local_domains = np.asarray(x[:, -1], dtype=np.int64)
        positions: List[int] = []
        selected_ids: Dict[str, List[int]] = {}
        for local_domain, physical_domain in enumerate(self.local_to_physical):
            eligible = np.flatnonzero(local_domains == local_domain).tolist()
            eligible.sort(key=lambda position: (
                hashlib.sha256("{}|{}|{}".format(
                    split_name, int(physical_domain), int(row_ids[position])).encode("utf-8")).hexdigest(),
                int(row_ids[position])))
            chosen = eligible[:1024]
            if not chosen:
                raise ValueError("routing probe has no {} rows for physical domain {}".format(
                    split_name, physical_domain))
            positions.extend(chosen)
            selected_ids["d{}".format(physical_domain)] = [int(row_ids[position])
                                                            for position in chosen]
        return np.asarray(positions, dtype=np.int64), selected_ids

    def _routing_probe_structure(self):
        hooks = []
        gate_modules = []
        expert_modules = []
        tower_modules = []
        cell_expert_names: Dict[int, List[str]] = {}
        gate_probabilities: Dict[str, bool] = {}

        if self.backbone == "m3oe":
            for index, module in enumerate(self.model.gate):
                name = "gate.{}".format(index)
                gate_modules.append((name, module))
                gate_probabilities[name] = True
            for group_name in ("expert", "domain_expert", "task_expert"):
                group = getattr(self.model, group_name)
                for index, module in enumerate(group):
                    expert_modules.append(("{}.{}".format(group_name, index), module))
            for task in range(self.task_num):
                for domain in range(self.domain_num):
                    cell = task * self.domain_num + domain
                    cell_expert_names[cell] = [
                        *("expert.{}".format(i) for i in range(self.model.expert_num)),
                        "domain_expert.{}".format(domain), "task_expert.{}".format(task)]
        elif self.backbone == "mmoe":
            for index, module in enumerate(self.model.core.gate):
                name = "core.gate.{}".format(index)
                gate_modules.append((name, module))
                gate_probabilities[name] = False
            for index, module in enumerate(self.model.core.experts):
                expert_modules.append(("core.experts.{}".format(index), module))
            for cell in range(self.domain_num * self.task_num):
                cell_expert_names[cell] = ["core.experts.{}".format(i) for i in range(4)]
        elif self.backbone == "cgc1":
            for index, module in enumerate(self.model.core.gate):
                name = "core.gate.{}".format(index)
                gate_modules.append((name, module))
                gate_probabilities[name] = False
            for index, module in enumerate(self.model.core.shared_experts):
                expert_modules.append(("core.shared_experts.{}".format(index), module))
            for cell, members in enumerate(self.model.core.specific_experts):
                name = "core.specific_experts.{}.0".format(cell)
                expert_modules.append((name, members[0]))
                cell_expert_names[cell] = [name] + [
                    "core.shared_experts.{}".format(i) for i in range(4)]
        else:
            raise ValueError("unsupported backbone routing probe {}".format(self.backbone))

        for index, module in enumerate(self.model.tower):
            tower_modules.append(("tower.{}".format(index), module))
        return gate_modules, expert_modules, tower_modules, cell_expert_names, gate_probabilities

    @staticmethod
    def _pairwise_cosine_summary(values: Sequence[torch.Tensor]) -> Dict[str, object]:
        similarities = []
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                left = values[left_index].float().reshape(len(values[left_index]), -1)
                right = values[right_index].float().reshape(len(values[right_index]), -1)
                similarities.append(torch.nn.functional.cosine_similarity(
                    left, right, dim=1, eps=1e-8).mean())
        if not similarities:
            return {"pair_count": 0, "mean_cosine_similarity": None,
                    "min_cosine_similarity": None, "max_cosine_similarity": None}
        observed = [float(value.detach().cpu()) for value in similarities]
        if not all(math.isfinite(value) for value in observed):
            raise FloatingPointError("routing probe produced non-finite expert similarity")
        return {"pair_count": len(observed),
                "mean_cosine_similarity": float(sum(observed) / len(observed)),
                "min_cosine_similarity": float(min(observed)),
                "max_cosine_similarity": float(max(observed))}

    def _routing_probe_split(self, split_name: str, x: np.ndarray,
                             row_ids: np.ndarray) -> Dict[str, object]:
        positions, selected_ids = self._probe_positions(x, row_ids, split_name)
        sampled_x = np.asarray(x[positions], dtype=np.float32)
        sampled_domains = sampled_x[:, -1].astype(np.int64)
        gate_modules, expert_modules, tower_modules, cell_expert_names, gate_probabilities = \
            self._routing_probe_structure()
        captured: Dict[str, torch.Tensor] = {}
        execution_counts: Dict[str, Dict[str, int]] = {}
        handles = []

        def register(name: str, module: torch.nn.Module, kind: str):
            def capture(_module, _inputs, output):
                if not torch.is_tensor(output) or output.ndim < 1:
                    raise TypeError("routing probe hook {} returned unsupported output".format(name))
                tensor = output.detach().cpu().clone()
                captured[name] = tensor
                counts = execution_counts.setdefault(name, {"calls": 0, "rows": 0})
                counts["calls"] += 1
                counts["rows"] += int(tensor.shape[0])
            handles.append(module.register_forward_hook(capture))

        for name, module in gate_modules:
            register(name, module, "gate")
        for name, module in expert_modules:
            register(name, module, "expert")
        for name, module in tower_modules:
            register(name, module, "tower")

        model_training_modes = {name: module.training for name, module in self.model.named_modules()}
        try:
            self.model.eval()
            with preserve_rng_state(), torch.no_grad():
                tensor = torch.from_numpy(np.ascontiguousarray(sampled_x.T)).to(
                    device=self.device, dtype=torch.float32)
                prediction = self.model(tensor)
                if not bool(torch.isfinite(prediction).all().detach().cpu()):
                    raise FloatingPointError("non-finite formal routing probe prediction")
        finally:
            for handle in handles:
                handle.remove()
            for name, module in self.model.named_modules():
                module.training = bool(model_training_modes[name])

        cells = {}
        for task in range(self.task_num):
            for local_domain, physical_domain in enumerate(self.local_to_physical):
                cell = task * self.domain_num + local_domain
                gate_name = ("gate.{}" if self.backbone == "m3oe" else
                             "core.gate.{}").format(cell)
                gate = captured[gate_name].float()
                if not gate_probabilities[gate_name]:
                    gate = torch.softmax(gate, dim=1)
                row_mask = torch.as_tensor(sampled_domains == local_domain, dtype=torch.bool)
                cell_gate = gate[row_mask]
                if not len(cell_gate):
                    raise AssertionError("routing probe cell has no sampled rows")
                if not bool(torch.isfinite(cell_gate).all()) or bool(torch.any(cell_gate < 0)):
                    raise FloatingPointError("routing gate probabilities are invalid")
                mean_distribution = cell_gate.mean(dim=0)
                entropy = -(cell_gate.clamp_min(1e-12).log() * cell_gate).sum(dim=1).mean()
                top1 = torch.bincount(cell_gate.argmax(dim=1), minlength=cell_gate.shape[1]).float()
                top1 = top1 / float(len(cell_gate))
                similarity = self._pairwise_cosine_summary([
                    captured[name][row_mask] for name in cell_expert_names[cell]])
                cells["d{}t{}".format(physical_domain, task)] = {
                    "physical_domain": int(physical_domain), "task": task,
                    "probe_row_count": int(len(cell_gate)),
                    "gate_mean_distribution": [float(v) for v in mean_distribution.tolist()],
                    "gate_entropy_mean": float(entropy.detach().cpu()),
                    "gate_top1_expert_share": [float(v) for v in top1.tolist()],
                    "expert_output_pairwise_cosine": similarity,
                    "gate_dense_execution": execution_counts[gate_name],
                    "tower_dense_execution": execution_counts["tower.{}".format(cell)],
                }
        if self.backbone == "m3oe":
            fusion = {
                "alpha_domain": float(torch.sigmoid(self.model._weight_exp_d.deep_weights).detach().cpu()),
                "alpha_task": float(torch.sigmoid(self.model._weight_exp_t.deep_weights).detach().cpu()),
                "beta_domain": (float(torch.sigmoid(self.model._weight_bal_d.deep_weights).detach().cpu())
                                if hasattr(self.model, "_weight_bal_d") else None),
                "beta_task": (float(torch.sigmoid(self.model._weight_bal_t.deep_weights).detach().cpu())
                              if hasattr(self.model, "_weight_bal_t") else None),
            }
        else:
            fusion = None
        return {
            "selected_probe_row_ids_by_physical_domain": selected_ids,
            "probe_cap_per_cell": 1024,
            "sampled_row_count": int(len(sampled_x)),
            "physical_domain_by_sample_order": [int(self.local_to_physical[int(d)])
                                                for d in sampled_domains.tolist()],
            "cells": cells,
            "fusion_coefficients": fusion,
            "dense_execution_counts": execution_counts,
            "all_model_paths_executed_densely_on_probe_rows": True,
        }

    def _load_checkpoint_state_for_probe(self, payload: Mapping[str, object], role: str) -> None:
        if payload.get("snapshot_role") != role:
            raise RuntimeError("{} routing snapshot has the wrong checkpoint role".format(role))
        required = {
            "run_id": str(self.entry["run_id"]),
            "canonical_key_json": str(self.entry["canonical_key_json"]),
            "attempt_id": str(self.entry["attempt_id"]),
            "protocol_hash": self.protocol_digest,
            "code_hash": self.code_digest,
            "unit_key": self.unit,
            "config_id": str(self.entry["config_id"]),
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise RuntimeError("{} routing snapshot identity mismatch at {}".format(role, key))
        if stable_state_hash(payload["model_state"]) != payload.get("model_state_hash"):
            raise RuntimeError("{} routing snapshot model hash mismatch".format(role))
        if stable_state_hash(payload["main_optimizer_state"]) != payload.get("main_optimizer_state_hash"):
            raise RuntimeError("{} routing snapshot main optimizer hash mismatch".format(role))
        arch_state = payload.get("architecture_optimizer_state")
        arch_hash = stable_state_hash(arch_state) if arch_state is not None else "absent"
        if arch_hash != payload.get("architecture_optimizer_state_hash"):
            raise RuntimeError("{} routing snapshot architecture optimizer hash mismatch".format(role))
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.main_optimizer.load_state_dict(payload["main_optimizer_state"])
        if self.arch_optimizer is not None:
            self.arch_optimizer.load_state_dict(arch_state)
        _restore_rng_state(payload["rng_state"])

    def _run_formal_routing_probes(self, selected_path: str,
                                   final_path: str) -> Dict[str, object]:
        if self.phase != "formal":
            raise ValueError("formal routing probes are available only for formal runs")
        original = self._capture_live_diagnostic_state()
        original_fingerprint = stable_state_hash({
            "model": original["model_state"],
            "main_optimizer": original["main_optimizer_state"],
            "architecture_optimizer": (original["architecture_optimizer_state"]
                                        if original["architecture_optimizer_state"] is not None
                                        else "absent"),
            "rng": original["rng_state"],
            "gradients": original["gradients"],
            "requires_grad": original["requires_grad"],
            "module_training_modes": original["module_training_modes"],
        })
        rows = {}
        no_op_audits = {}
        timings = {}
        try:
            for role, path in (("selected", selected_path), ("final", final_path)):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if role == "selected" and payload.get("selected_epoch") is None:
                    raise RuntimeError("selected routing snapshot has no selected validation epoch")
                if role == "final" and int(payload.get("completed_epochs", -1)) != self.epoch_limit:
                    raise RuntimeError("final routing snapshot is not at the frozen epoch limit")
                if int(payload.get("completed_epochs", -1)) > self.epoch_limit:
                    raise RuntimeError("routing snapshot exceeds the frozen epoch limit")
                self._load_checkpoint_state_for_probe(payload, role)
                snapshot_epoch = int(payload["completed_epochs"])
                before = self._diagnostic_state(snapshot_epoch)
                started = time.monotonic()
                first = {
                    "train": self._routing_probe_split("train", self.train_x, self.train_ids),
                    "validation": self._routing_probe_split("validation", self.val_x, self.val_ids),
                }
                first_seconds = time.monotonic() - started
                after_first = self._diagnostic_state(snapshot_epoch)
                if before != after_first:
                    raise AssertionError("{} routing probe changed complete diagnostic state".format(role))
                started = time.monotonic()
                second = {
                    "train": self._routing_probe_split("train", self.train_x, self.train_ids),
                    "validation": self._routing_probe_split("validation", self.val_x, self.val_ids),
                }
                second_seconds = time.monotonic() - started
                after_second = self._diagnostic_state(snapshot_epoch)
                if before != after_second:
                    raise AssertionError("repeated {} routing probe changed complete diagnostic state".format(role))
                first_hash = sha256_json(first)
                second_hash = sha256_json(second)
                if first_hash != second_hash:
                    raise AssertionError("repeated {} routing probe output is not identical".format(role))
                rows[role] = {"run_id": payload["run_id"],
                              "attempt_id": payload["attempt_id"],
                              "completed_epochs": snapshot_epoch,
                              "selected_epoch": payload["selected_epoch"],
                              "selected_validation_macro_auc": payload["selected_validation_macro_auc"],
                              "checkpoint_model_state_hash": payload["model_state_hash"],
                              "train": first["train"], "validation": first["validation"]}
                no_op_audits[role] = {
                    "state_before": before, "state_after_first_probe": after_first,
                    "state_after_repeat_probe": after_second,
                    "complete_state_unchanged_after_each_probe": True,
                    "repeated_probe_output_sha256": first_hash,
                    "repeated_probe_outputs_identical": True,
                }
                timings[role] = {"first_train_validation_seconds": first_seconds,
                                 "repeat_train_validation_seconds": second_seconds}
        finally:
            self._restore_live_diagnostic_state(original)
        restored = self._diagnostic_state(self.completed_epochs)
        expected_after_restore = {
            "model": stable_state_hash(original["model_state"]),
            "main_optimizer": stable_state_hash(original["main_optimizer_state"]),
            "architecture_optimizer": (stable_state_hash(original["architecture_optimizer_state"])
                                       if original["architecture_optimizer_state"] is not None else "absent"),
            "rng": stable_state_hash(original["rng_state"]),
            "gradients": stable_state_hash(original["gradients"]),
            "requires_grad": stable_state_hash(original["requires_grad"]),
            "module_training_modes": stable_state_hash(original["module_training_modes"]),
            "next_batch_fingerprint": self._next_batch_fingerprint_at(self.completed_epochs),
        }
        if restored != expected_after_restore:
            raise AssertionError("routing probes failed to restore the pre-probe runner state")
        output = {
            "schema_version": "F0-formal-routing-probe-v1",
            "unit_key": self.unit, "backbone": self.backbone,
            "scope": self.scope, "physical_domains": list(self.local_to_physical),
            "probe_policy": "lowest-sha256(split|physical_domain|row_id) per cell, max 1024",
            "snapshots": rows, "no_op_state_audit": no_op_audits,
            "diagnostic_wall_seconds": timings,
            "runner_state_before_and_after_all_probes_identical": True,
            "runner_state_identity_sha256": original_fingerprint,
        }
        _write_json(os.path.join(self.attempt_dir, "routing_probe_selected.json"), rows["selected"])
        _write_json(os.path.join(self.attempt_dir, "routing_probe_final.json"), rows["final"])
        _write_json(os.path.join(self.attempt_dir, "routing_probe_noop_audit.json"), output)
        return output

    def _run_manifest(self, status: str, **extra) -> Dict[str, object]:
        source_path = str(self.bundle.preprocessing_metadata.get("raw_file", ""))
        return {
            "stage": "F0", "protocol_version": PROTOCOL_VERSION,
            "status": status, "run_id": self.entry["run_id"],
            "canonical_key_json": self.entry["canonical_key_json"],
            "run_id_full_sha256": self.entry["run_id_full_sha256"],
            "attempt_id": self.entry["attempt_id"],
            "phase": self.phase, "intent": self.intent,
            "unit_key": self.unit, "dataset": "movielens", "setting": "natural",
            "backbone": self.backbone, "scope": self.scope,
            "physical_domains": list(self.local_to_physical), "tasks": [0, 1],
            "objective": "Sample BCE per observed row/task, batch means weighted by row count",
            "capacity": "Q", "model_width": self.width,
            "actual_non_embedding_parameter_count": non_embedding_parameter_count(self.model),
            "parameter_manifest": parameter_count_manifest(self.model),
            "parameter_budget_audit": self.width_audit,
            "selected_optimizer_config": self.config,
            "seed_role": self.entry["seed_role"], "seed": self.seed,
            "split_hashes": dict(self.entry["split_hashes"]),
            "dataset_file": source_path,
            "dataset_file_sha256": self.bundle.preprocessing_metadata.get("raw_file_sha256", ""),
            "protocol_hash": self.protocol_digest, "code_hash": self.code_digest,
            "code_hash_inventory": code_hash_inventory(require_tests=False),
            "initial_model_state_hash": self.model_initial_hash,
            "initialization_audit": "initialization_audit.json",
            "main_optimizer": {"name": "Adam", "lr": self.config["lr"],
                               "betas": list(ADAM_BETAS), "eps": ADAM_EPS,
                               "weight_decay": self.config["weight_decay"]},
            "architecture_optimizer": (
                None if self.arch_optimizer is None else {
                    "name": "Adam", "lr": self.config["lr"] * ARCH_LR_RATIO,
                    "betas": list(ADAM_BETAS), "eps": ADAM_EPS,
                    "weight_decay": self.config["weight_decay"], "steps_per_epoch": 1}),
            "architecture_optimizer_absent": self.arch_optimizer is None,
            "epochs_completed": self.completed_epochs,
            "max_epochs": self.epoch_limit,
            "selected_epoch": self.selected_epoch,
            "selected_validation_macro_auc": (None if self.selected_epoch is None
                                               else self.selected_score),
            "test_access_during_training": False,
            "test_metrics_written": False,
            "mixed_precision": False, "amp": False, "tf32": False,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "resume_evidence": list(self.resume_evidence),
            "artifacts": ["run_manifest.json", "metrics_validation.jsonl",
                          "initialization_audit.json", "parameter_manifest.json",
                          "resource_epoch_metrics.jsonl"],
            **extra,
        }

    def _interruption_barrier(self, phase: str, payload: Mapping[str, object]) -> None:
        hook = self.interruption_receipt_hook
        if callable(hook):
            hook(str(phase), dict(payload))

    def _commit_interrupted_receipt(self, *, exc: BaseException,
                                    status_file: str, manifest_path: str,
                                    output_root: str, last_full_checkpoint: Optional[dict],
                                    canonical_lock: QueueFileLock,
                                    gpu_lock: Optional[QueueFileLock]) -> None:
        receipt_path = os.environ.get("F0_INVOCATION_RECEIPT")
        if not receipt_path or not os.environ.get("F0_CHILD_SIGNAL"):
            return
        if gpu_lock is None:
            raise PermissionError("controlled interruption has no held GPU lock")

        child = linux_process_identity(os.getpid())
        parent_pid = os.getppid()
        parent = linux_process_identity(parent_pid)
        worker_id, nonce = _execution_context()[1:]
        signal_number = int(os.environ["F0_CHILD_SIGNAL"])
        repeat_count = int(os.environ.get("F0_CHILD_SIGNAL_REPEAT_COUNT", "0"))
        canonical_owner = _locked_owner_snapshot(canonical_lock)
        gpu_owner = _locked_owner_snapshot(gpu_lock)
        owner_identity = {
            "pid": os.getpid(), "host": socket.gethostname(),
            "pgid": child["pgid"], "start_ticks": child["start_ticks"],
            "supervisor_pid": parent_pid,
            "supervisor_start_ticks": parent["start_ticks"],
            "worker_id": worker_id, "nonce": nonce,
            "run_id": str(self.entry["run_id"]),
            "run_id_full_sha256": str(self.entry["run_id_full_sha256"]),
            "canonical_key_json": str(self.entry["canonical_key_json"]),
            "attempt_id": str(self.entry["attempt_id"]),
            "signal": signal_number, "signal_repeat_count": repeat_count,
            "completed_epochs": int(self.completed_epochs),
            "last_full_checkpoint": last_full_checkpoint,
            "canonical_lock_path": os.path.abspath(canonical_lock.path),
            "canonical_lock_owner_sha256": sha256_json(canonical_owner),
            "gpu_lock_path": os.path.abspath(gpu_lock.path),
            "gpu_lock_owner_sha256": sha256_json(gpu_owner),
        }
        for label, owner in (("canonical", canonical_owner), ("GPU", gpu_owner)):
            for field in ("pid", "run_id", "run_id_full_sha256", "attempt_id"):
                if str(owner.get(field, "")) != str(owner_identity[field]):
                    raise PermissionError("locked {} owner differs at {}".format(label, field))
            for field in ("pgid", "start_ticks", "supervisor_pid", "supervisor_start_ticks",
                          "worker_id", "nonce"):
                if str(owner.get(field, "")) != str(owner_identity[field]):
                    raise PermissionError("locked {} owner process identity differs at {}".format(
                        label, field))

        def make_receipt(identity: Mapping[str, object]) -> dict:
            return {
                "schema": "F0-CHILD-RECEIPT-v1", "status": "INTERRUPTED",
                "transaction_commit": "DURABLE_BEFORE_LOCK_RELEASE",
                "exit_code": 128 + int(identity["signal"]),
                "signal": int(identity["signal"]),
                "signal_repeat_count": int(identity["signal_repeat_count"]),
                "worker_id": worker_id, "nonce": nonce,
                "pid": os.getpid(), "ppid": parent_pid,
                "pgid": int(child["pgid"]),
                "child_start_ticks": int(child["start_ticks"]),
                "supervisor_start_ticks": int(parent["start_ticks"]),
                "run_id": self.entry["run_id"],
                "run_id_full_sha256": self.entry["run_id_full_sha256"],
                "canonical_key_json": self.entry["canonical_key_json"],
                "attempt_id": self.entry["attempt_id"],
                "status_path": os.path.abspath(status_file),
                "manifest_path": os.path.abspath(manifest_path),
                "manifest_sha256": _sha_file(manifest_path),
                "completed_epochs": int(identity["completed_epochs"]),
                "last_full_checkpoint": identity["last_full_checkpoint"],
                "interruption_identity": dict(identity),
                "canonical_lock_owner": canonical_owner,
                "canonical_lock_path": os.path.abspath(canonical_lock.path),
                "canonical_lock_owner_sha256": sha256_json(canonical_owner),
                "gpu_lock_owner": gpu_owner,
                "gpu_lock_path": os.path.abspath(gpu_lock.path),
                "gpu_lock_owner_sha256": sha256_json(gpu_owner),
                "timestamp": time.time(),
            }

        # The resource locks and shared execution-state lock jointly enclose
        # status, manifest, and the receipt's atomic fsync/rename transaction.
        with QueueFileLock(execution_control_lock_path(output_root),
                           nonblocking=False):
            current_status = _read_json(status_file)
            status_payload = {
                **current_status, "status": "INTERRUPTED",
                "error_type": type(exc).__name__, "error": str(exc),
                "completed_epochs": int(self.completed_epochs),
                "last_full_checkpoint": last_full_checkpoint,
                "interruption_identity": owner_identity,
                "updated_unix": time.time(),
            }
            _write_json(status_file, status_payload)
            manifest_payload = self._run_manifest(
                "INTERRUPTED", error_type=type(exc).__name__, error=str(exc),
                last_full_checkpoint=last_full_checkpoint,
                interruption_identity=owner_identity, interrupted_unix=time.time())
            _write_json(manifest_path, manifest_payload)
            draft = make_receipt(owner_identity)
            _execution_phase("interrupted_receipt_precommit",
                             canonical={"run_id": self.entry["run_id"],
                                        "attempt_id": self.entry["attempt_id"]},
                             gpu={"lock_path": os.path.abspath(gpu_lock.path)},
                             child_start_ticks=child["start_ticks"],
                             supervisor_start_ticks=parent["start_ticks"],
                             status_path=os.path.abspath(status_file),
                             manifest_path=os.path.abspath(manifest_path),
                             receipt_path=os.path.abspath(receipt_path))
            self._interruption_barrier("before_receipt_commit", draft)

            # Re-read the idempotent child's repeat counter after the barrier.
            # Signals received during preparation are recorded without entering
            # the cleanup stack again, and status/manifest/receipt stay aligned.
            owner_identity["signal_repeat_count"] = int(os.environ.get(
                "F0_CHILD_SIGNAL_REPEAT_COUNT", str(owner_identity["signal_repeat_count"])))
            status_payload["interruption_identity"] = owner_identity
            status_payload["updated_unix"] = time.time()
            _write_json(status_file, status_payload)
            manifest_payload = self._run_manifest(
                "INTERRUPTED", error_type=type(exc).__name__, error=str(exc),
                last_full_checkpoint=last_full_checkpoint,
                interruption_identity=owner_identity, interrupted_unix=time.time())
            _write_json(manifest_path, manifest_payload)
            receipt = make_receipt(owner_identity)
            _write_invocation_receipt(receipt_path, receipt)
            _execution_phase("interrupted_receipt_committed",
                             canonical={"run_id": self.entry["run_id"],
                                        "attempt_id": self.entry["attempt_id"]},
                             gpu={"lock_path": os.path.abspath(gpu_lock.path)},
                             receipt_path=os.path.abspath(receipt_path),
                             receipt_sha256=sha256_json(receipt),
                             signal_repeat_count=owner_identity["signal_repeat_count"])
            self._interruption_barrier("after_receipt_commit", receipt)

    def run(self, *, resume_from: Optional[str] = None,
            max_epochs: Optional[int] = None,
            gate_verified: bool = False,
            canonical_lock: Optional[QueueFileLock] = None,
            gpu_lock: Optional[QueueFileLock] = None,
            status_file: Optional[str] = None) -> Dict[str, object]:
        if not self.test_mode and not gate_verified:
            raise PermissionError("research training requires validate_run_gate authorization")
        if self.phase == "resource":
            raise ValueError("resource preflight entries do not train trajectories")
        requested_limit = self.epoch_limit if max_epochs is None else int(max_epochs)
        if requested_limit < self.completed_epochs or requested_limit > self.epoch_limit:
            raise ValueError("max_epochs must be within completed_epochs..frozen_epoch_limit")
        output_root = os.path.dirname(os.path.dirname(os.path.dirname(self.attempt_dir)))
        if canonical_lock is None:
            canonical_lock = QueueFileLock(os.path.join(
                output_root, "canonical_locks", str(self.entry["run_id"]) + ".lock"))
        try:
            gpu_index = (int(self.device.index or 0)
                         if self.device.type == "cuda" or gpu_lock is not None else None)
            gpu_uuid = ""
            if self.device.type == "cuda" and torch.cuda.is_available():
                gpu_uuid = str(getattr(torch.cuda.get_device_properties(
                    int(self.device.index or 0)), "uuid", ""))
        except BaseException:
            if gpu_lock is not None:
                gpu_lock.close()
            canonical_lock.close()
            raise
        gpu_identity = {"gpu_index": gpu_index, "gpu_uuid": gpu_uuid}
        try:
            if self.device.type == "cuda" and gpu_lock is None:
                gpu_lock = QueueFileLock(os.path.join(
                    output_root, "gpu_locks", "gpu{}.lock".format(int(self.device.index or 0))))
        except BaseException:
            canonical_lock.close()
            raise
        try:
            child_process_identity = linux_process_identity(os.getpid())
            supervisor_process_identity = linux_process_identity(os.getppid())
            worker_id, invocation_nonce = _execution_context()[1:]
            lock_process_identity = {
                "pid": os.getpid(), "pgid": child_process_identity["pgid"],
                "start_ticks": child_process_identity["start_ticks"],
                "supervisor_pid": os.getppid(),
                "supervisor_start_ticks": supervisor_process_identity["start_ticks"],
                "worker_id": worker_id, "nonce": invocation_nonce,
            }
            with QueueFileLock(execution_control_lock_path(output_root),
                               nonblocking=False):
                canonical_lock.write_owner({
                    "run_id": str(self.entry["run_id"]),
                    "run_id_full_sha256": str(self.entry["run_id_full_sha256"]),
                    "canonical_key_json": str(self.entry["canonical_key_json"]),
                    "pid": os.getpid(), "host": socket.gethostname(),
                    "attempt_id": str(self.entry["attempt_id"]),
                    "claimed_unix": time.time(), **lock_process_identity,
                    **gpu_identity,
                })
                if gpu_lock is not None:
                    gpu_lock.write_owner({
                        "pid": os.getpid(), "host": socket.gethostname(),
                        **lock_process_identity,
                        **gpu_identity, "run_id": str(self.entry["run_id"]),
                        "run_id_full_sha256": str(self.entry["run_id_full_sha256"]),
                        "canonical_key_json": str(self.entry["canonical_key_json"]),
                        "attempt_id": str(self.entry["attempt_id"]),
                        "claimed_unix": time.time(),
                    })
        except BaseException:
            if gpu_lock is not None:
                gpu_lock.close()
            canonical_lock.close()
            raise
        self.gpu_lock_path = None if gpu_lock is None else gpu_lock.path
        manifest_path = os.path.join(self.attempt_dir, "run_manifest.json")
        last_committed_checkpoint = (None if resume_from is None else
                                     os.path.abspath(resume_from))
        committed_epochs = int(self.completed_epochs)
        try:
            os.makedirs(self.attempt_dir, exist_ok=True)
            if os.path.exists(manifest_path) and resume_from is None:
                raise FileExistsError("attempt directory already contains a run manifest")
            if resume_from is not None:
                self._load_checkpoint(resume_from)
                committed_epochs = int(self.completed_epochs)
            else:
                self._assert_class_coverage()
                _write_json(os.path.join(self.attempt_dir, "initialization_audit.json"),
                            self.initialization_audit)
                _write_json(os.path.join(self.attempt_dir, "parameter_manifest.json"),
                            parameter_count_manifest(self.model))
                _write_json(manifest_path, self._run_manifest("RUNNING"))
        except BaseException:
            if gpu_lock is not None:
                gpu_lock.close()
            if canonical_lock is not None:
                canonical_lock.close()
            self.gpu_lock_path = None
            raise
        if status_file:
            with QueueFileLock(execution_control_lock_path(output_root),
                               nonblocking=False):
                status_payload = {}
                if os.path.isfile(status_file):
                    status_payload = _read_json(status_file)
                _write_json(status_file, {**status_payload, "status": "RUNNING",
                                         "run_id": self.entry["run_id"],
                                         "run_id_full_sha256": self.entry["run_id_full_sha256"],
                                         "canonical_key_json": self.entry["canonical_key_json"],
                                         "attempt_id": self.entry["attempt_id"],
                                         "pid": os.getpid(), "host": socket.gethostname(),
                                         "pgid": os.getpgrp(),
                                         "start_ticks": child_process_identity["start_ticks"],
                                         "supervisor_pid": os.getppid(),
                                         "supervisor_start_ticks":
                                             supervisor_process_identity["start_ticks"],
                                         "worker_id": worker_id,
                                         "invocation_nonce": invocation_nonce,
                                         "gpu_index": (int(self.device.index or 0)
                                                       if self.device.type == "cuda" else None),
                                         "updated_unix": time.time()})
        wall_started = time.monotonic()
        last_train_record = None
        try:
            while self.completed_epochs < requested_limit:
                epoch = self.completed_epochs
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                epoch_started = time.monotonic()
                examples_before_epoch = self.examples_processed
                train_record = self._train_epoch(epoch)
                validation = self._eval_validation()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                epoch_wall_seconds = time.monotonic() - epoch_started
                examples_this_epoch = self.examples_processed - examples_before_epoch
                score = float(validation["macro_auc"])
                if not math.isfinite(score):
                    raise FloatingPointError("non-finite validation selection value")
                selected = self.selected_epoch is None or score > self.selected_score
                if selected:
                    self.selected_epoch = epoch
                    self.selected_score = score
                self.completed_epochs = epoch + 1
                self.validation_history.append({"epoch": epoch, "macro_auc": score,
                                                "macro_logloss": float(validation["macro_logloss"])})
                metric_row = {
                    "stage": "F0", "phase": self.phase, "intent": self.intent,
                    "run_id": self.entry["run_id"], "unit_key": self.unit,
                    "config_id": self.entry["config_id"], "seed_role": self.entry["seed_role"],
                    "seed": self.seed, "backbone": self.backbone, "scope": self.scope,
                    "epoch": epoch, "selected_checkpoint": 0,
                    "train_sample_bce": train_record["train_sample_bce"],
                    "train_equal_diagnostic_bce": train_record["train_equal_diagnostic_bce"],
                    "validation_macro_auc": score,
                    "validation_macro_logloss": float(validation["macro_logloss"]),
                    "validation_cells": validation["cells"],
                    "main_optimizer_steps_total": self.main_steps,
                    "architecture_optimizer_steps_total": self.arch_steps,
                    "main_examples_total": self.examples_processed,
                    "architecture_examples_total": self.arch_examples_processed,
                    "main_domain_exposure": train_record["main_domain_exposure"],
                    "main_task_positive_counts": train_record["main_task_positive_counts"],
                    "architecture_exposure": train_record["architecture_exposure"],
                    "batch_fingerprints": train_record["batch_fingerprints"],
                    "epoch_wall_seconds": epoch_wall_seconds,
                    "examples_processed_this_epoch": examples_this_epoch,
                    "examples_per_second": (float(examples_this_epoch / epoch_wall_seconds)
                                             if epoch_wall_seconds > 0 else None),
                    "cpu_peak_rss_bytes": _process_peak_rss_bytes(),
                    "gpu_peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(self.device))
                        if self.device.type == "cuda" else None),
                    "gpu_peak_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(self.device))
                        if self.device.type == "cuda" else None),
                }
                self.metrics_rows.append(metric_row)
                last_train_record = train_record
                resume_path = "resume_epoch_{:03d}.pt".format(self.completed_epochs)
                committed_path = self._save_checkpoint(
                    resume_path, self.completed_epochs, last_train_record)
                last_committed_checkpoint = os.path.abspath(committed_path)
                committed_epochs = int(self.completed_epochs)
                # This checkpoint is the authoritative epoch commit. All
                # metrics/snapshot/manifest files below are recoverable views.
                self._inject_fault("after_epoch_commit")
                self._flush_metrics()
                self._inject_fault("after_metrics_projection")
                for stale_epoch in range(max(1, self.completed_epochs - 3)):
                    stale_path = os.path.join(self.attempt_dir,
                                              "resume_epoch_{:03d}.pt".format(stale_epoch + 1))
                    if stale_epoch + 1 < self.completed_epochs - 2 and os.path.exists(stale_path):
                        os.unlink(stale_path)
                self._inject_fault("after_resume_prune")
                if selected:
                    self._save_checkpoint("selected_checkpoint.pt", self.completed_epochs,
                                          last_train_record)
                self._inject_fault("after_selected_projection")
                if epoch in FIXED_CHECKPOINT_EPOCHS and self.phase == "formal":
                    self._save_checkpoint("checkpoint_epoch_{:02d}.pt".format(epoch),
                                          self.completed_epochs, last_train_record)
                self._inject_fault("after_fixed_projection")
                self._flush_resource_metrics()
                self._inject_fault("after_resource_projection")
                _write_json(manifest_path, self._run_manifest("RUNNING"))
                self._inject_fault("after_manifest_projection")
            self._flush_metrics()
            if self.completed_epochs >= self.epoch_limit:
                final_path = self._save_checkpoint(
                    "final_checkpoint.pt", self.completed_epochs, last_train_record)
                routing_probe_summary = None
                if self.phase == "formal":
                    selected_path = os.path.join(self.attempt_dir, "selected_checkpoint.pt")
                    routing_probe_summary = self._run_formal_routing_probes(
                        selected_path, final_path)
                summary = {
                    "stage": "F0", "status": "VALIDATION_COMPLETE",
                    "run_id": self.entry["run_id"], "unit_key": self.unit,
                    "config_id": self.entry["config_id"], "seed": self.seed,
                    "phase": self.phase, "seed_role": self.entry["seed_role"],
                    "selected_epoch": int(self.selected_epoch),
                    "selected_validation_macro_auc": float(self.selected_score),
                    "final_epoch": self.epoch_limit - 1,
                    "final_validation_macro_auc": float(self.validation_history[-1]["macro_auc"]),
                    "run_manifest": os.path.abspath(manifest_path),
                    "test_access_during_training": False,
                    "routing_probe_status": ("COMPLETE" if routing_probe_summary is not None
                                             else "NOT_REQUIRED_FOR_SEARCH"),
                }
                if routing_probe_summary is not None:
                    summary["routing_probe_artifacts"] = [
                        "routing_probe_selected.json", "routing_probe_final.json",
                        "routing_probe_noop_audit.json"]
                _write_json(os.path.join(self.attempt_dir, "trajectory_summary.json"), summary)
                final_manifest = self._run_manifest(
                    "VALIDATION_COMPLETE", wall_seconds=time.monotonic() - wall_started,
                    selected_epoch=int(self.selected_epoch),
                    selected_validation_macro_auc=float(self.selected_score),
                    final_epoch=self.epoch_limit - 1,
                    final_validation_macro_auc=float(self.validation_history[-1]["macro_auc"]),
                    test_access_during_training=False,
                    routing_probe_status=("COMPLETE" if routing_probe_summary is not None
                                          else "NOT_REQUIRED_FOR_SEARCH"),
                    routing_probe_artifacts=(
                        [] if routing_probe_summary is None else summary["routing_probe_artifacts"]))
                _write_json(manifest_path, final_manifest)
                if status_file:
                    with QueueFileLock(execution_control_lock_path(output_root),
                                       nonblocking=False):
                        _write_json(status_file, {**_read_json(status_file),
                                                  "status": "VALIDATION_COMPLETE",
                                                  "completed_unix": time.time(),
                                                  "selected_epoch": int(self.selected_epoch)})
                _write_completion_receipt(
                    os.environ.get("F0_INVOCATION_RECEIPT"), entry=self.entry,
                    output_root=output_root, status_file=status_file,
                    attempt_dir=self.attempt_dir, summary=summary, gpu_lock=gpu_lock)
                return summary
            partial = self._run_manifest(
                "PARTIAL_RESOURCE_TRAJECTORY", wall_seconds=time.monotonic() - wall_started,
                selected_epoch=self.selected_epoch, selected_validation_macro_auc=self.selected_score,
                test_access_during_training=False)
            _write_json(manifest_path, partial)
            if status_file:
                with QueueFileLock(execution_control_lock_path(output_root),
                                   nonblocking=False):
                    _write_json(status_file, {**_read_json(status_file),
                                              "status": "PARTIAL_RESOURCE_TRAJECTORY",
                                              "updated_unix": time.time()})
            return partial
        except BaseException as exc:
            if last_committed_checkpoint is not None:
                committed_payload = torch.load(last_committed_checkpoint,
                                               map_location="cpu", weights_only=False)
                checkpoint_identity = {
                    "run_id": str(self.entry["run_id"]),
                    "canonical_key_json": str(self.entry["canonical_key_json"]),
                    "attempt_id": str(self.entry["attempt_id"]),
                    "protocol_hash": self.protocol_digest,
                    "code_hash": self.code_digest,
                }
                if (committed_payload.get("snapshot_role") != "resume"
                        or committed_payload.get("epoch_boundary_checkpoint") is not True
                        or any(committed_payload.get(key) != expected
                               for key, expected in checkpoint_identity.items())
                        or int(committed_payload.get("completed_epochs", -1))
                        != committed_epochs):
                    raise RuntimeError(
                        "last full checkpoint identity/epoch failed interruption validation") from exc
                # The resume checkpoint is the epoch commit. Roll in-memory
                # metadata back to that commit before projecting INTERRUPTED;
                # a signal during a later partial epoch must not advance status.
                self.completed_epochs = int(committed_payload["completed_epochs"])
                self.main_steps = int(committed_payload["main_steps"])
                self.arch_steps = int(committed_payload["architecture_steps"])
                self.examples_processed = int(committed_payload["examples_processed"])
                self.arch_examples_processed = int(
                    committed_payload["architecture_examples_processed"])
                self.selected_epoch = (None if committed_payload["selected_epoch"] is None
                                       else int(committed_payload["selected_epoch"]))
                self.selected_score = float(
                    committed_payload["selected_validation_macro_auc"])
                self.validation_history = copy.deepcopy(
                    committed_payload["validation_history"])
                self.metrics_rows = copy.deepcopy(committed_payload["metrics_rows"])
            else:
                # No epoch boundary has been committed. Preserve the initial
                # empty state even if interruption landed during epoch setup.
                self.completed_epochs = 0
                self.main_steps = 0
                self.arch_steps = 0
                self.examples_processed = 0
                self.arch_examples_processed = 0
                self.selected_epoch = None
                self.selected_score = float("-inf")
                self.validation_history = []
                self.metrics_rows = []

            last_full_checkpoint = None
            if last_committed_checkpoint is not None:
                last_full_checkpoint = {
                    "name": os.path.basename(last_committed_checkpoint),
                    "sha256": _sha_file(last_committed_checkpoint),
                    "completed_epochs": int(committed_epochs),
                }
            if status_file:
                try:
                    if (isinstance(exc, KeyboardInterrupt)
                            and os.environ.get("F0_INVOCATION_RECEIPT")
                            and os.environ.get("F0_CHILD_SIGNAL")):
                        self._commit_interrupted_receipt(
                            exc=exc, status_file=status_file,
                            manifest_path=manifest_path, output_root=output_root,
                            last_full_checkpoint=last_full_checkpoint,
                            canonical_lock=canonical_lock, gpu_lock=gpu_lock)
                    else:
                        with QueueFileLock(execution_control_lock_path(output_root),
                                           nonblocking=False):
                            current_status = _read_json(status_file)
                            status_payload = {
                                **current_status, "status": "INTERRUPTED",
                                "error_type": type(exc).__name__, "error": str(exc),
                                "completed_epochs": int(committed_epochs),
                                "last_full_checkpoint": last_full_checkpoint,
                                "updated_unix": time.time(),
                            }
                            _write_json(status_file, status_payload)
                            _write_json(manifest_path, self._run_manifest(
                                "INTERRUPTED", error_type=type(exc).__name__,
                                error=str(exc),
                                last_full_checkpoint=last_full_checkpoint,
                                interrupted_unix=time.time()))
                except BaseException as cleanup_exc:
                    raise RuntimeError(
                        "failed to durably commit INTERRUPTED status, manifest, and receipt") from cleanup_exc
            raise
        finally:
            if gpu_lock is not None:
                gpu_lock.close()
            if canonical_lock is not None:
                canonical_lock.close()
            self.gpu_lock_path = None


_SELECTED_CHECKPOINT_IDENTITY_FIELDS = (
    "run_id", "run_id_full_sha256", "canonical_key_json", "attempt_id", "phase",
    "unit_key", "config_id", "seed_role", "seed", "backbone", "scope",
    "model_width", "split_hashes", "protocol_hash", "code_hash",
    "dataset_file_sha256", "completed_epochs", "selected_epoch",
    "selected_validation_macro_auc", "model_state_hash", "main_optimizer_state_hash",
    "architecture_optimizer_state_hash", "selected_checkpoint_sha256",
)


def frozen_selected_checkpoint_identity(entry: Mapping[str, object],
                                        selected_checkpoint_path: str) -> Dict[str, object]:
    """Return the immutable identity that a validation release must freeze."""

    if os.path.basename(os.path.abspath(selected_checkpoint_path)) != "selected_checkpoint.pt":
        raise PermissionError("test evaluation accepts only selected_checkpoint.pt")
    digest = _sha_file(selected_checkpoint_path)
    payload = torch.load(selected_checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("selected checkpoint must contain a state mapping")
    if payload.get("snapshot_role") != "selected":
        raise PermissionError("only a checkpoint frozen with snapshot_role=selected is accepted")
    checkpoint_fields = (
        "run_id", "canonical_key_json", "attempt_id", "phase", "unit_key", "config_id",
        "seed", "backbone", "scope", "model_width", "split_hashes", "protocol_hash",
        "code_hash", "dataset_file_sha256", "completed_epochs", "selected_epoch",
        "selected_validation_macro_auc", "model_state_hash", "main_optimizer_state_hash",
        "architecture_optimizer_state_hash",
    )
    identity = {"run_id_full_sha256": entry.get("run_id_full_sha256"),
               "seed_role": entry.get("seed_role"),
               "selected_checkpoint_sha256": digest}
    identity.update({key: payload.get(key) for key in checkpoint_fields})
    canonical_digest = hashlib.sha256(
        str(entry.get("canonical_key_json", "")).encode("utf-8")).hexdigest()
    if canonical_digest != entry.get("run_id_full_sha256") or \
            canonical_digest[:20] != entry.get("run_id"):
        raise PermissionError("queue entry canonical run identity is invalid")
    for key in ("run_id", "canonical_key_json", "attempt_id", "phase", "unit_key",
                "config_id", "seed", "backbone", "scope", "model_width", "split_hashes",
                "protocol_hash", "code_hash"):
        if payload.get(key) != entry.get(key):
            raise PermissionError("selected checkpoint does not match queue entry at {}".format(key))
    if payload.get("dataset_file_sha256") != entry.get("dataset_file_sha256",
                                                       payload.get("dataset_file_sha256")):
        raise PermissionError("selected checkpoint dataset identity mismatch")
    if stable_state_hash(payload["model_state"]) != payload.get("model_state_hash"):
        raise ValueError("selected checkpoint model-state hash is invalid")
    if stable_state_hash(payload["main_optimizer_state"]) != payload.get("main_optimizer_state_hash"):
        raise ValueError("selected checkpoint main optimizer-state hash is invalid")
    arch = payload.get("architecture_optimizer_state")
    arch_hash = stable_state_hash(arch) if arch is not None else "absent"
    if arch_hash != payload.get("architecture_optimizer_state_hash"):
        raise ValueError("selected checkpoint architecture optimizer-state hash is invalid")
    if payload.get("selected_epoch") is None or int(payload.get("completed_epochs", -1)) \
            != int(payload["selected_epoch"]) + 1:
        raise ValueError("selected checkpoint is not the frozen selected-epoch snapshot")
    if identity["seed_role"] not in ("formal", "formal_addon") or payload.get("phase") != "formal":
        raise PermissionError("only selected formal checkpoints may be frozen for test")
    identity = {key: identity[key] for key in _SELECTED_CHECKPOINT_IDENTITY_FIELDS}
    return identity


def release_selected_test(bundle, *, release_gate: Mapping[str, object],
                          entry: Mapping[str, object], output_dir: str,
                          selected_checkpoint_path: str,
                          selected_checkpoint_identity: Mapping[str, object],
                          device: str = "cuda:0") -> Dict[str, object]:
    """Evaluate only a checkpoint whose full identity was frozen by validation."""

    required_true = ("all_search_complete", "formal_first_four_complete",
                     "validation_expansion_decision_frozen", "p_trigger_decision_frozen")
    if release_gate.get("status") != "TEST_RELEASED":
        raise PermissionError("test release gate must have status TEST_RELEASED")
    if any(release_gate.get(key) is not True for key in required_true):
        raise PermissionError("test release gate lacks frozen validation decisions")
    if entry.get("seed_role") == "formal_addon" and \
            release_gate.get("formal_addon_complete") is not True:
        raise PermissionError("expanded-seed test release requires the complete ML add-on block")
    if release_gate.get("protocol_hash") != entry.get("protocol_hash") or \
            release_gate.get("code_hash") != entry.get("code_hash"):
        raise PermissionError("test release gate protocol/code hash mismatch")
    if release_gate.get("dataset_file_sha256") != bundle.preprocessing_metadata.get("raw_file_sha256"):
        raise PermissionError("test release gate MovieLens file hash mismatch")
    if release_gate.get("split_hashes") != entry.get("split_hashes"):
        raise PermissionError("test release gate split hash mismatch")
    if entry.get("phase") != "formal" or entry.get("seed_role") not in ("formal", "formal_addon"):
        raise PermissionError("only selected formal checkpoints may enter the test evaluator")
    if os.path.basename(os.path.abspath(selected_checkpoint_path)) != "selected_checkpoint.pt":
        raise PermissionError("test evaluator rejects final/resume/fixed checkpoints")
    identity = dict(selected_checkpoint_identity)
    if set(identity) != set(_SELECTED_CHECKPOINT_IDENTITY_FIELDS):
        raise PermissionError("selected checkpoint identity fields are incomplete or unexpected")
    frozen_identities = release_gate.get("selected_checkpoint_identities")
    if not isinstance(frozen_identities, Mapping) or \
            frozen_identities.get(str(entry.get("run_id"))) != identity:
        raise PermissionError("selected checkpoint identity was not frozen in the release gate")
    current_identity = frozen_selected_checkpoint_identity(entry, selected_checkpoint_path)
    if current_identity != identity:
        raise PermissionError("selected checkpoint file differs from the frozen identity")
    payload = torch.load(selected_checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("dataset_file_sha256") != bundle.preprocessing_metadata.get("raw_file_sha256"):
        raise PermissionError("selected checkpoint MovieLens file hash mismatch")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for deferred test evaluation but unavailable")

    scope = str(entry["scope"])
    domain_num = 3 if scope == "mdmt" else 1
    physical_domain = scope_instance(scope)
    model_seed = _seed_from_text("{}|{}|natural|init-root-v1".format(
        int(entry["seed"]), str(entry["backbone"])))
    model, _audit = build_semantically_paired_model(
        bundle.features, backbone=str(entry["backbone"]), domain_num=domain_num,
        task_num=2, c=int(entry["model_width"]["c"]), seed=int(entry["seed"]),
        physical_domain=physical_domain, model_seed=model_seed)
    validate_frozen_width(str(entry["backbone"]), scope, model)
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(device_obj)

    # Access to fixed-test arrays starts only after every release and checkpoint
    # identity predicate has passed and the selected checkpoint is reconstructed.
    test_x = np.asarray(bundle.test_x, dtype=np.float32)
    test_y = np.asarray(bundle.test_y, dtype=np.float32)
    test_ids = np.asarray(bundle.test_row_ids, dtype=np.int64)
    if len(test_x) == 0 or len(test_x) != len(test_y) or len(test_x) != len(test_ids):
        raise ValueError("fixed test arrays are empty or misaligned")
    model.eval()
    predictions = []
    with torch.no_grad():
        batch_size = int(entry["config"]["batch_size"])
        for start in range(0, len(test_x), batch_size):
            arr = test_x[start:start + batch_size]
            tensor = torch.from_numpy(np.ascontiguousarray(arr.T)).to(
                device=device_obj, dtype=torch.float32)
            value = model(tensor)
            if not bool(torch.isfinite(value).all().detach().cpu()):
                raise FloatingPointError("non-finite selected test prediction")
            predictions.append(value.detach().cpu().numpy().astype(np.float32))
    prediction_array = np.concatenate(predictions, axis=0)
    domains = test_x[:, -1].astype(np.int64)
    metrics = evaluate_cells(test_y, prediction_array, domains,
                             tuple(int(v) for v in bundle.local_to_physical), 2)
    result = {"release_gate_hash": sha256_json(release_gate),
              "selected_checkpoint_identity_hash": sha256_json(identity),
              "selected_checkpoint_sha256": identity["selected_checkpoint_sha256"],
              "run_id": entry["run_id"], "seed": int(entry["seed"]),
              "selected_validation_only": True, "test_access_after_release": True,
              "metrics": metrics}
    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, "metrics_test_selected.json"), result)
    np.savez_compressed(os.path.join(output_dir, "predictions_test_selected.npz"),
                        row_ids=test_ids, physical_domain=np.asarray(
                            [bundle.local_to_physical[int(v)] for v in domains], dtype=np.int64),
                        labels=test_y, prediction=prediction_array)
    return result


def _load_json(path: str) -> object:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_queue(path: str) -> Mapping[str, object]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise ValueError("queue manifest has invalid structure")
    phase = str(payload.get("phase"))
    if phase == "formal":
        expected = len(payload["entries"])
        if expected not in (48, 96):
            raise ValueError("formal queue must contain the full 48 or 96 trajectory block")
    else:
        expected = {"search": 288, "resource": 6}.get(phase)
    if expected is None:
        raise ValueError("unsupported queue phase")
    validate_queue(payload["entries"], expected, phase)
    if payload.get("protocol_hash") != protocol_hash():
        raise ValueError("queue manifest protocol hash mismatch")
    if sha256_json(payload["entries"]) != payload.get("queue_hash"):
        raise ValueError("queue manifest queue hash mismatch")
    return payload


def validate_runtime_control_amendment(execution_gate: Mapping[str, object], *,
                                       frozen_scientific_hash: str,
                                       runtime_code_hash: str) -> None:
    """Bind the amendment's runtime field to the complete 28-file code hash."""

    if execution_gate.get("schema") == "F0-EXECUTION-ISOLATION-v1.5":
        amendment = execution_gate.get("bootstrap_amendment")
        from stage_F0_successor_v15 import (
            control_hash_inventory_v15, sha256_json, _validate_amendment,
        )
        _validate_amendment(amendment,
            scientific_sha256=frozen_scientific_hash,
            runtime_sha256=runtime_code_hash,
            control_sha256=sha256_json(control_hash_inventory_v15()))
        return
    amendment = execution_gate.get("scientific_code_hash_amendment")
    if not isinstance(amendment, Mapping):
        raise PermissionError("control-only scientific-code amendment is missing")
    if (amendment.get("status") != "APPROVED"
            or amendment.get("reviewer_verdict") not in ("PASS", "PASS WITH FIXES")
            or str(amendment.get("frozen_scientific_code_hash", ""))
            != str(frozen_scientific_hash)
            or str(amendment.get("runtime_control_code_hash", ""))
            != str(runtime_code_hash)):
        raise PermissionError("control-only runtime code hash amendment is unreviewed or stale")


def _persist_v15_resource_deferral(args, exc, *, gpu_index: int, gpu_uuid: str) -> int:
    if args.resource_deferral_secret_fd is None:
        raise PermissionError("resource deferral lacks the supervisor secret pipe") from exc
    secret_fd = int(args.resource_deferral_secret_fd)
    secret = b""
    try:
        while len(secret) < 32:
            chunk = os.read(secret_fd, 32 - len(secret))
            if not chunk:
                break
            secret += chunk
    finally:
        os.close(secret_fd)
    if len(secret) != 32:
        raise PermissionError("resource deferral secret pipe was truncated") from exc
    from stage_F0_successor_v15 import (
        INCIDENT_HOLD_SHA256, INCIDENT_NONCE, QUEUE_SHA256, write_deferred_receipt,
    )
    from stage_F0_worker_control import linux_process_identity, sha256_file
    child_identity = linux_process_identity(os.getpid())
    parent_identity = linux_process_identity(os.getppid())
    receipt_path = os.environ.get("F0_INVOCATION_RECEIPT")
    phase_path = os.environ.get("F0_PHASE_LOG")
    if not receipt_path or not phase_path:
        raise PermissionError("resource deferral lacks nonce-specific admin paths")
    with open(phase_path, encoding="utf-8") as phase_handle:
        phases = [json.loads(line).get("phase") for line in phase_handle if line.strip()]
    if "CANONICAL_MUTATION_STARTED" in phases:
        raise PermissionError("canonical mutation began before resource deferral")
    gate_path = Path(args.execution_isolation_gate).resolve()
    release_path = Path(args.output_root).resolve() / "execution_control" / "global_hold_release.json"
    target_pair = {"index": int(gpu_index), "uuid": str(gpu_uuid)}
    core = {
        "schema": "F0-PRECLAIM-RESOURCE-DEFERRED-v1.5",
        "outcome": "RESOURCE_DEFERRED_PRECLAIM", "exit_code": 75,
        "worker_id": str(os.environ.get("F0_WORKER_ID", "")),
        "physical_gpu_index": int(gpu_index), "gpu_uuid": str(gpu_uuid),
        "frozen_target_pair": target_pair,
        "supervisor_pid": int(os.getppid()),
        "supervisor_start_ticks": int(parent_identity["start_ticks"]),
        "child_pid": int(os.getpid()), "child_start_ticks": int(child_identity["start_ticks"]),
        "invocation": int(os.environ.get("F0_INVOCATION_NUMBER", "0")),
        "nonce": str(os.environ.get("F0_INVOCATION_NONCE", "")),
        "expected_receipt_path": os.path.abspath(receipt_path),
        "execution_gate_sha256": sha256_file(gate_path),
        "release_sha256": sha256_file(release_path),
        "queue_sha256": sha256_file(args.queue),
        "source_incident_hold_sha256": INCIDENT_HOLD_SHA256,
        "source_failure_nonce": INCIDENT_NONCE,
        "telemetry": dict(exc.telemetry),
        "decision_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "decision_monotonic": time.monotonic(),
        "boundary": exc.boundary,
        "canonical_claim": None, "claimed_run_id": None,
        "canonical_mutation_started": False, "attempt_created": False,
        "checkpoint_loaded": False,
        "forbidden_flags": {"resume": False, "retry": False,
            "new_attempt": False, "checkpoint_load": False,
            "formal": False, "test": False},
    }
    _execution_phase("resource_deferred_receipt_before", gpu=target_pair,
                     boundary=exc.boundary)
    receipt = write_deferred_receipt(receipt_path, secret, core)
    _execution_phase("resource_deferred_receipt_durable", gpu=target_pair,
                     receipt_sha256=sha256_file(receipt_path),
                     outcome=receipt["outcome"], exit_code=75)
    return 75


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--run-id", default=None,
                          help="run one canonical queue identity")
    selector.add_argument("--claim-next", action="store_true",
                          help="claim the next unclaimed queue identity for this worker")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-gate", required=True)
    parser.add_argument("--execution-isolation-gate", default=str(EXECUTION_GATE))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--held-run-ids-json", default="[]")
    # The shell wrapper reserves this lease before exec. The trajectory child
    # may instead pass the same identity explicitly from its supervisor.
    parser.add_argument("--reserved-process-slot-fd", type=int,
                        default=os.environ.get("F0_RESERVED_PROCESS_SLOT_FD"))
    parser.add_argument("--reserved-process-slot-index", type=int,
                        default=os.environ.get("F0_RESERVED_PROCESS_SLOT_INDEX"))
    parser.add_argument("--reserved-process-slot-owner-pid", type=int,
                        default=os.environ.get("F0_RESERVED_PROCESS_SLOT_OWNER_PID"))
    parser.add_argument("--reserved-process-slot-role",
                        default=os.environ.get("F0_RESERVED_PROCESS_SLOT_ROLE"))
    parser.add_argument("--resource-deferral-secret-fd", type=int, default=None)
    args = parser.parse_args(argv)
    if args.reserved_process_slot_fd is not None:
        if (args.reserved_process_slot_index is None
                or args.reserved_process_slot_owner_pid is None
                or not args.reserved_process_slot_role):
            raise PermissionError("inherited process slot identity is incomplete")
        process_slot = adopt_reserved_process_slot(
            fd=args.reserved_process_slot_fd,
            index=args.reserved_process_slot_index,
            owner_pid=args.reserved_process_slot_owner_pid,
            role=args.reserved_process_slot_role,
            scheduler_dir=os.environ.get("F0_PROCESS_SCHEDULER_ROOT"))
    else:
        raise PermissionError(
            "F0 runner requires a process slot reserved before Python startup")
    atexit.register(process_slot.close)
    execution_gates = validate_execution_gate(
        gate_path=args.execution_isolation_gate, search_gate_path=args.run_gate)
    assert_no_execution_hold(args.output_root)
    search_gate = execution_gates["search_gate"]
    try:
        external_holds = json.loads(args.held_run_ids_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("held run IDs must be a JSON list") from exc
    # Keep the canonical ordered H51 list expected by production claim; list
    # membership is sufficient for queue filtering and preserves its identity.
    held_run_ids = frozen_held_run_ids(
        execution_gates["execution_gate"], search_gate, external_holds)
    queue = _load_queue(args.queue)
    if not queue.get("entries"):
        _write_invocation_receipt(os.environ.get("F0_INVOCATION_RECEIPT"), {
            "schema": "F0-CHILD-RECEIPT-v1", "status": "QUEUE_EMPTY",
            "worker_id": os.environ.get("F0_WORKER_ID", "direct"),
            "nonce": os.environ.get("F0_INVOCATION_NONCE", "direct"),
            "pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgrp(),
            "queue_hash": queue.get("queue_hash"), "held_run_ids": sorted(held_run_ids),
            "output_root": os.path.abspath(args.output_root),
        })
        print(json.dumps({"status": "QUEUE_EMPTY_OR_ALL_CLAIMED",
                          "queue_hash": queue.get("queue_hash")}, sort_keys=True))
        return 0
    frozen_code_hash = str(search_gate.get("code_hash", ""))
    current_code = code_hash(require_tests=True)
    validate_runtime_control_amendment(
        execution_gates["execution_gate"], frozen_scientific_hash=frozen_code_hash,
        runtime_code_hash=current_code)

    _execution_phase("dataset_load_before", dataset_path=args.dataset)
    bundles = load_b1_bundles(args.dataset)
    dataset_hash = str(next(iter(bundles.values())).preprocessing_metadata.get(
        "raw_file_sha256", ""))
    if not dataset_hash:
        raise ValueError("dataset loader did not establish a MovieLens file hash")
    _execution_phase("dataset_load_after", dataset_file_sha256=dataset_hash,
                     scopes=sorted(bundles))
    assert_no_execution_hold(args.output_root)
    candidate = dict(queue["entries"][0])
    if str(candidate.get("code_hash", "")) != frozen_code_hash:
        raise PermissionError("queue scientific code identity differs from frozen run gate")
    validate_run_gate(search_gate, entry=candidate,
                      queue_hash=str(queue["queue_hash"]), dataset_file_hash=dataset_hash)
    _execution_phase("gpu_preflight_before", dataset_file_sha256=dataset_hash)
    preflight = runtime_preflight(args.output_root)
    v15_gate = execution_gates["execution_gate"].get("schema") == "F0-EXECUTION-ISOLATION-v1.5"
    memory_gate = (None if v15_gate else gpu_memory_preflight(preflight))
    _execution_phase("gpu_preflight_after",
                     usable_gpu_indices=list(preflight.get(
                         "resource_concurrency_policy", {}).get("usable_gpu_indices", [])),
                     gpu_memory_gate=memory_gate,
                     cuda_available=bool(preflight.get("cuda_available", False)))
    if (not preflight["cuda_available"] or (not v15_gate and not preflight[
            "resource_concurrency_policy"]["usable_gpu_indices"])):
        raise RuntimeError("F0-B1 queue requires a preflight-usable CUDA device")
    if v15_gate:
        worker_id = os.environ.get("F0_WORKER_ID")
        targets = execution_gates["execution_gate"].get("gpu_target_pairs")
        if worker_id not in ("0", "1") or not isinstance(targets, list):
            raise PermissionError("v1.5 runner requires its physical worker/GPU identity")
        worker_index = int(worker_id)
        selected = next((row for row in targets if isinstance(row, dict)
                         and row.get("index") == worker_index), None)
        if not isinstance(selected, dict):
            raise PermissionError("v1.5 worker is outside the reviewed target subset")
        if worker_index >= int(preflight.get("device_count", 0)):
            raise PermissionError("reviewed physical GPU is unavailable to this child")
        gpu_indices = (worker_index,)
    else:
        memory_eligible = set(memory_gate["eligible_gpu_indices"])
        if not memory_eligible:
            raise RuntimeError("GPU memory preflight is NO-GO: no readable target GPU is below 90%")
        if args.claim_next:
            gpu_indices = tuple(index for index in preflight["resource_concurrency_policy"][
                "usable_gpu_indices"] if index in memory_eligible)
        else:
            selected_device = torch.device(args.device)
            if selected_device.type != "cuda":
                raise RuntimeError("F0-B1 research runs require CUDA; CPU fallback is forbidden")
            requested_gpu = int(selected_device.index or 0)
            if requested_gpu not in preflight["resource_concurrency_policy"]["usable_gpu_indices"]:
                raise RuntimeError("requested CUDA index did not pass GPU preflight")
            if requested_gpu not in memory_eligible:
                raise RuntimeError("requested CUDA index is at or above 90% memory use")
            gpu_indices = (requested_gpu,)
    if not gpu_indices:
        raise RuntimeError("GPU memory preflight is NO-GO: no target GPU is below 90%")

    assert_no_execution_hold(args.output_root)
    if v15_gate:
        selected_target = next((row for row in execution_gates["execution_gate"].get(
            "gpu_target_pairs", []) if isinstance(row, dict)
            and row.get("index") == gpu_indices[0]), None)
        if not isinstance(selected_target, dict):
            raise PermissionError("v1.5 worker target disappeared before child preclaim check")
        from stage_F0_successor_v15 import ResourceDeferred, require_live_gpu_memory
        try:
            require_live_gpu_memory(runs_root=args.output_root, gpu_index=gpu_indices[0],
                gpu_uuid=str(selected_target.get("uuid", "")), boundary="CHILD_PRECLAIM")
        except ResourceDeferred as deferred:
            return _persist_v15_resource_deferral(args, deferred,
                gpu_index=gpu_indices[0], gpu_uuid=str(selected_target["uuid"]))
    gpu_lock = None
    gpu_index = None
    for index in gpu_indices:
        try:
            gpu_lock = QueueFileLock(os.path.join(
                os.path.abspath(args.output_root), "gpu_locks", "gpu{}.lock".format(index)))
            gpu_index = int(index)
            break
        except BlockingIOError:
            continue
    if gpu_lock is None:
        raise RuntimeError("no preflight-visible GPU slot is currently available")
    if v15_gate:
        gpu_uuid = str(selected["uuid"])
    else:
        gpu_uuid = str(preflight["gpus"][gpu_index].get("uuid", ""))
    gpu_lock.write_owner({"pid": os.getpid(), "host": socket.gethostname(),
                          "gpu_index": gpu_index, "gpu_uuid": gpu_uuid,
                          "claimed_unix": time.time()})

    canonical_lock = None
    _execution_phase("canonical_claim_before", gpu={"index": gpu_index,
                                                      "uuid": gpu_uuid},
                     held_run_ids=sorted(held_run_ids))
    try:
        try:
            if args.claim_next:
                entry, canonical_lock, resume_path = claim_next_entry(
                    queue, args.output_root, gpu_uuid=gpu_uuid, gpu_index=gpu_index,
                    held_run_ids=held_run_ids)
            else:
                candidates = [row for row in queue["entries"]
                              if str(row["run_id"]) == str(args.run_id)]
                if len(candidates) != 1:
                    raise ValueError("run_id must select exactly one queue row")
                if str(args.run_id) in held_run_ids:
                    raise PermissionError("requested canonical key is held by the frozen run gate")
                entry, canonical_lock, resume_path = claim_next_entry(
                    {**queue, "entries": candidates}, args.output_root,
                    gpu_uuid=gpu_uuid, gpu_index=gpu_index,
                    held_run_ids=held_run_ids)
        except PermissionError as exc:
            from stage_F0_successor_v15 import ResourceDeferred
            if not isinstance(exc, ResourceDeferred):
                raise
            return _persist_v15_resource_deferral(args, exc,
                gpu_index=int(gpu_index), gpu_uuid=str(gpu_uuid))
        if entry is None or canonical_lock is None:
            _execution_phase("canonical_claim_after", claimed=False,
                             queue_hash=queue["queue_hash"])
            _write_invocation_receipt(os.environ.get("F0_INVOCATION_RECEIPT"), {
                "schema": "F0-CHILD-RECEIPT-v1", "status": "QUEUE_EMPTY",
                "worker_id": os.environ.get("F0_WORKER_ID", "direct"),
                "nonce": os.environ.get("F0_INVOCATION_NONCE", "direct"),
                "pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgrp(),
                "queue_hash": queue["queue_hash"],
                "held_run_ids": sorted(held_run_ids),
                "output_root": os.path.abspath(args.output_root),
            })
            print(json.dumps({"status": "QUEUE_EMPTY_OR_ALL_CLAIMED",
                              "queue_hash": queue["queue_hash"]}, sort_keys=True))
            return 0
        if str(entry["code_hash"]) != frozen_code_hash:
            raise ValueError("queued scientific code hash differs from its frozen gate")
        if str(entry["run_id"]) in held_run_ids:
            raise PermissionError("claim returned a key explicitly held by the run gate")
        _execution_phase("canonical_claim_after", claimed=True,
                         canonical={"run_id": entry["run_id"],
                                    "run_id_full_sha256": entry["run_id_full_sha256"],
                                    "canonical_key_json": entry["canonical_key_json"],
                                    "attempt_id": entry["attempt_id"]},
                         gpu={"index": gpu_index, "uuid": gpu_uuid})
        assert_no_execution_hold(args.output_root)
        bundle = bundles[str(entry["scope"])]
        validate_run_gate(_load_json(args.run_gate), entry=entry,
                          queue_hash=str(queue["queue_hash"]), dataset_file_hash=dataset_hash)
        attempt_dir = os.path.join(os.path.abspath(args.output_root), str(entry["run_id"]),
                                   "attempts", str(entry["attempt_id"]))
        status_file = queue_status_path(args.output_root, str(entry["run_id"]))
        if queue["phase"] == "resource":
            # This records hardware/disk preflight only; it never trains.
            result = resource_preflight_record(entry, bundle, preflight)
            result.update({"run_id_full_sha256": entry["run_id_full_sha256"],
                      "canonical_key_json": entry["canonical_key_json"],
                      "attempt_id": entry["attempt_id"],
                      "resource": preflight,
                      "measurement_fields": ["epoch_wall_seconds", "examples_per_second",
                                              "cpu_peak_rss_bytes", "gpu_peak_allocated_bytes",
                                              "gpu_peak_reserved_bytes", "storage_free_bytes",
                                              "checkpoint_bytes"],
                      "matching_search_unit": entry["unit_key"]})
            os.makedirs(attempt_dir, exist_ok=False)
            _write_json(os.path.join(attempt_dir, "resource_preflight.json"), result)
            with QueueFileLock(execution_control_lock_path(args.output_root),
                               nonblocking=False):
                _write_json(status_file, {**result, "completed_unix": time.time()})
            return 0
        runner = F0Runner(entry, bundle, attempt_dir, device="cuda:{}".format(gpu_index),
                          code_digest=frozen_code_hash, protocol_digest=protocol_hash())
        _execution_phase("training_before",
                         canonical={"run_id": entry["run_id"],
                                    "run_id_full_sha256": entry["run_id_full_sha256"],
                                    "attempt_id": entry["attempt_id"]},
                         gpu={"index": gpu_index, "uuid": gpu_uuid},
                         checkpoint_before=resume_path)
        summary = runner.run(resume_from=resume_path, gate_verified=True,
                             canonical_lock=canonical_lock, gpu_lock=gpu_lock,
                             status_file=status_file)
        # F0Runner owns and releases both locks once it starts.
        canonical_lock = None
        gpu_lock = None
        _execution_phase("training_after",
                         canonical={"run_id": entry["run_id"],
                                    "run_id_full_sha256": entry["run_id_full_sha256"],
                                    "attempt_id": entry["attempt_id"]},
                         gpu={"index": gpu_index, "uuid": gpu_uuid},
                         runner_status=summary.get("status"))
        _execution_phase("completion_validation_before",
                         canonical={"run_id": entry["run_id"],
                                    "run_id_full_sha256": entry["run_id_full_sha256"],
                                    "attempt_id": entry["attempt_id"]})
        _execution_phase("completion_validation_after",
                         canonical={"run_id": entry["run_id"],
                                    "run_id_full_sha256": entry["run_id_full_sha256"],
                                    "attempt_id": entry["attempt_id"]},
                         runner_status=summary.get("status"),
                         receipt_written=summary.get("status") == "VALIDATION_COMPLETE")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        if canonical_lock is not None:
            canonical_lock.close()
        if gpu_lock is not None:
            gpu_lock.close()
        process_slot.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["setup_deterministic_runtime", "preserve_rng_state", "stable_state_hash",
           "build_global_order", "runtime_preflight", "gpu_memory_preflight",
           "validate_run_gate", "F0Runner",
           "frozen_selected_checkpoint_identity", "resource_preflight_record",
           "release_selected_test", "main"]
