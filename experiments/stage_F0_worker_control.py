"""Standard-library-only control primitives for isolated F0 workers.

This module must stay importable without torch, data loaders, or CUDA.  It is
shared by the persistent supervisor and the fresh trajectory bootstrap so the
launch gate, durable events, and global hold have one implementation.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Mapping, Optional


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/stage_F0_outputs"
RUNS = OUTPUT / "runs"
EXECUTION_GATE = OUTPUT / "execution_isolation_gate.json"
SEARCH_GATE = OUTPUT / "run_gate_search.json"
CONTROL_HASH_FILES = (
    "experiments/stage_F0_outputs/runs/run_search_worker.sh",
    "experiments/stage_F0_outputs/runs/run_f0_python_task.sh",
    "experiments/stage_F0_outputs/runs/run_search_claim_worker.py",
    "experiments/stage_F0_outputs/runs/run_search_trajectory_child.py",
    "experiments/stage_F0_outputs/preflight/run_first_search_epoch.py",
    "experiments/stage_F0_worker_control.py",
    "experiments/stage_F0_queue.py",
    "experiments/stage_F0_run.py",
    "experiments/stage_F0_migration.py",
    "experiments/stage_F0_postabnormal_v14.py",
    "experiments/reports/stage_F0_postabnormal_design_v14.md",
    "experiments/reports/stage_F0_postabnormal_v14_implementation_report.md",
    "experiments/tests/test_stage_F0_postabnormal_v14.py",
    "experiments/stage_F0_outputs/runs/WORKER_CONTROL.md",
    "experiments/reports/stage_F0_execution_isolation_design.md",
    "experiments/tests/test_stage_F0_launcher.py",
    "experiments/tests/conftest.py",
    "experiments/tests/test_stage_F0_worker_control.py",
    "experiments/tests/test_stage_F0_queue.py",
    "experiments/tests/test_stage_F0_run.py",
    "experiments/tests/test_stage_F0_migration.py",
    "experiments/tests/test_stage_F0_launcher.py",
)
PRESERVED_INCIDENT_KEYS = (
    "1a42854a5355c9a11835",
    "f62d2f477b73e50f6c84",
    "59474a37e9d12d956633",
)

# v1.2's independently reviewed 47-key set. Keep it intact as the base of the
# v1.3 administrative extension; callers, gates, and release documents cannot
# redefine it.
PRESERVED_V12_H47_RUN_IDS = (
    "005c9b294d0f8b52ce45",
    "15121f20dfe1378922f7",
    "1a42854a5355c9a11835",
    "25d43816fb9ce4f4d53d",
    "29f6299bac7e92919a66",
    "2a23f59a9312908ca612",
    "315de61f4e5ff10f5f00",
    "4e08f1a0bca3565268ca",
    "54ede12c24f035f2d451",
    "551d91a7b7f6b27ecc02",
    "59474a37e9d12d956633",
    "595012293cbdc94c0ca4",
    "60eb188009621135bb11",
    "628036e262d92ca739e7",
    "642c7adca43dae4b95b1",
    "7041ca21e28a9bb5918a",
    "75c6a14dff771de847c2",
    "84439d6c4a31cba6b93d",
    "86c5fcb7eba8a926d942",
    "8d1e0ec1dca0d70b3890",
    "8fcbf0a5af181d8665ce",
    "940341c53cecc7b7842b",
    "99c92debe0e66a15da66",
    "a96d057cf922c5659a9d",
    "b22c602ebecea4764804",
    "b309a36acbf4cc6443e1",
    "b3a55740aadd0e3377a9",
    "b91fb0ec69fc9774d55b",
    "ba2abfcb368b18f35f66",
    "be8ed5c5e9eb7ac86ad2",
    "beae65a8b954527bd35f",
    "c1ff7e2a9cda7e374e94",
    "c5786c1928ea62af8fac",
    "cdae050a20be6fedbda9",
    "ce0ad33341421a4cc007",
    "cfbf6cc0e541e76cfb92",
    "d64a6c2fd169d893d97b",
    "d76e7cd02e6fef25c51b",
    "d95495ccac33c8df96c3",
    "e272a1465754b343fd28",
    "eba87cb747c01a08a2c3",
    "ebce58b03baa75033c0d",
    "ef062a41c6043a48ec55",
    "f077605074cec9f14a04",
    "f62d2f477b73e50f6c84",
    "fa214d1fbc7fa49f676f",
    "fcd3e4341ebe46191500",
)
V13_ADDITIONAL_HELD_RUN_IDS = (
    "7fd2006bdc8f1817a6f7",
    "1f04c7fb623d18f940ac",
)
PRESERVED_H49_RUN_IDS = tuple(sorted(
    set(PRESERVED_V12_H47_RUN_IDS) | set(V13_ADDITIONAL_HELD_RUN_IDS)))
# Existing imports use this name. Its frozen meaning is now H49.
PRESERVED_SIGNAL_CLEANUP_KEYS = PRESERVED_H49_RUN_IDS
_PRESERVED_SIGNAL_CLEANUP_KEY_SET = frozenset(PRESERVED_H49_RUN_IDS)
PRESERVED_H47_SHA256 = "ccbf64945396b542ee2746f4bab37d7d2fcdaea49b00817f096facec45f7f8bf"
PRESERVED_H49_SHA256 = "60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e"
V14_ADDITIONAL_HELD_RUN_IDS = ("0d2e6555b049c012c0e5", "a80a232a3c0068a48a08")
PRESERVED_H51_RUN_IDS = tuple(sorted(set(PRESERVED_H49_RUN_IDS) | set(V14_ADDITIONAL_HELD_RUN_IDS)))
PRESERVED_H51_SHA256 = "e9077aefa1ed48a3237b769b20c0cd9e8ca3f9b0054ee537de1b7a40cd870a86"
PRESERVED_CURRENT_RUN_IDS = PRESERVED_H51_RUN_IDS
PRESERVED_CURRENT_SHA256 = PRESERVED_H51_SHA256


def held_run_ids_sha256(values: object) -> str:
    """Hash a sorted unique held-ID list using the frozen compact JSON form."""

    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError("held run IDs must be a collection of strings")
    items = list(values)
    if any(not isinstance(item, str) for item in items) or len(items) != len(set(items)):
        raise ValueError("held run IDs are malformed or duplicated")
    payload = json.dumps(sorted(items), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_exact_h49(values: object) -> list[str]:
    """Require a caller to supply every independently frozen H49 key exactly once."""
    if not isinstance(values, (list, tuple)):
        raise PermissionError("caller-held IDs are missing or malformed; exact H49 is required")
    items = list(values)
    if (any(not isinstance(item, str) for item in items)
            or len(items) != len(set(items))
            or set(items) != _PRESERVED_SIGNAL_CLEANUP_KEY_SET):
        raise PermissionError("caller-held IDs differ from the frozen H49 protection set")
    return sorted(items)


def require_exact_h51(values: object) -> list[str]:
    """Require the current v1.4 production protection set, retaining H49 separately."""
    if (not isinstance(values, (list, tuple))
            or any(not isinstance(item, str) for item in values)
            or len(values) != len(set(values))
            or set(values) != set(PRESERVED_H51_RUN_IDS)):
        raise PermissionError("current caller/gate held IDs differ from exact frozen H51")
    return list(PRESERVED_H51_RUN_IDS)


def _has_exact_preserved_signal_cleanup_keys(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and all(isinstance(item, str) for item in value)
        and len(value) == len(set(value)) == len(PRESERVED_H49_RUN_IDS)
        and set(value) == _PRESERVED_SIGNAL_CLEANUP_KEY_SET
    )


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def control_hash_inventory(root: os.PathLike | str = ROOT) -> dict[str, str]:
    root_path = Path(root).resolve()
    return {name: sha256_file(root_path / name) for name in CONTROL_HASH_FILES}


def _read_json(path: os.PathLike | str) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object")
    return value


def _atomic_json(path: os.PathLike | str, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp." + uuid.uuid4().hex)
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def validate_execution_gate(*, root: os.PathLike | str = ROOT,
                             gate_path: os.PathLike | str = EXECUTION_GATE,
                             search_gate_path: os.PathLike | str = SEARCH_GATE) -> dict:
    """Verify the new execution identity and refuse all unreviewed launches."""

    gate = _read_json(gate_path)
    if gate.get("schema") == "F0-EXECUTION-ISOLATION-v1.5":
        from stage_F0_successor_v15 import validate_current_gate as validate_v15_gate
        gate_file = Path(gate_path).resolve()
        derived_root = gate_file.parents[2]
        expected_gate = derived_root / "experiments/stage_F0_outputs/execution_isolation_gate.json"
        if gate_file != expected_gate:
            raise PermissionError("v1.5 gate path is not the canonical repository gate")
        return validate_v15_gate(gate, root=derived_root, require_release=True)
    if Path(root).resolve() == ROOT:
        if gate.get("schema") != "F0-EXECUTION-ISOLATION-v1.4":
            raise PermissionError("unknown production execution-gate schema")
        from stage_F0_postabnormal_v14 import validate_current_gate
        validate_current_gate(gate, runs_root=RUNS, require_release=True)
        actual = control_hash_inventory(root)
        if gate.get("control_hashes") != actual:
            raise PermissionError("v1.4 execution control source inventory is stale")
        actual_sha = _sha256_json(actual)
        if gate.get("control_hash_inventory_sha256") != actual_sha:
            raise PermissionError("v1.4 execution control inventory aggregate is stale")
        search = _read_json(search_gate_path)
        if (gate.get("prior_search_gate_sha256") != sha256_file(search_gate_path)
                or gate.get("preserved_scientific_code_hash") != search.get("code_hash")
                or search.get("phase") != "search"
                or search.get("formal_or_test_started") is not False):
            raise PermissionError("v1.4 gate differs from fixed search identity")
        amendment = gate.get("scientific_code_hash_amendment")
        if (not isinstance(amendment, Mapping)
                or amendment.get("frozen_scientific_code_hash") != search.get("code_hash")):
            raise PermissionError("v1.4 control amendment does not bind scientific code hash")
        return {"execution_gate": gate, "search_gate": search}

    gate = _read_json(gate_path)
    if gate.get("schema") != "F0-EXECUTION-ISOLATION-v1":
        raise PermissionError("execution-isolation gate schema missing")
    actual = control_hash_inventory(root)
    frozen = gate.get("control_hashes")
    if not isinstance(frozen, Mapping) or dict(frozen) != actual:
        raise PermissionError("execution-isolation control hash inventory is stale")
    actual_control_hash = _sha256_json(actual)
    if gate.get("control_hash_inventory_sha256") != actual_control_hash:
        raise PermissionError("execution-isolation control inventory aggregate is stale")
    if not _has_exact_preserved_signal_cleanup_keys(
            gate.get("preserved_held_run_ids")):
        raise PermissionError("execution gate must preserve the exact frozen H49 hold set")
    if Path(root).resolve() == ROOT:
        protection = gate.get("v13_protection")
        if not isinstance(protection, Mapping):
            raise PermissionError("v1.3 provenance/status/asset protection binding is missing")
        if (protection.get("schema") != "F0-V13-H49-PROTECTION-v1"
                or protection.get("held_run_ids") != list(PRESERVED_H49_RUN_IDS)
                or protection.get("held_run_ids_sha256") != PRESERVED_H49_SHA256
                or protection.get("recovery_authorized") is not False
                or protection.get("automatic_retry_authorized") is not False
                or protection.get("formal_or_test_authorized") is not False):
            raise PermissionError("v1.3 protection binding is malformed or grants forbidden scope")
        required_v13_hashes = (
            "provenance_correction_sha256", "migration_audit_sha256",
            "completed_asset_inventory_sha256", "incident_asset_inventory_sha256",
            "search_queue_file_sha256", "search_queue_entries_sha256",
            "design_sha256", "implementation_report_sha256", "reviewer_report_sha256",
            "original_source_sha256",
        )
        if any(not isinstance(protection.get(key), str)
               or len(str(protection.get(key))) != 64
               or any(char not in "0123456789abcdef" for char in str(protection.get(key)))
               for key in required_v13_hashes):
            raise PermissionError("v1.3 gate must freeze nonempty source, review, migration, asset, and queue hashes")
        required_v13_paths = {
            "design_path": "experiments/reports/stage_F0_execution_isolation_design.md",
            "implementation_report_path": "experiments/reports/stage_F0_execution_isolation_implementation_report.md",
            "original_source_path": "execution_control/postabnormal_sigsegv_audit_v4_fresh_search.json",
            "provenance_correction_path": "execution_control/postabnormal_sigsegv_audit_v4_provenance_correction.json",
            "migration_audit_path": "execution_control/postabnormal_sigsegv_audit_v5_migration.jsonl",
            "search_queue_path": "experiments/stage_F0_outputs/queues/F0-B1-ML-search-queue.json",
        }
        if any(protection.get(key) != value for key, value in required_v13_paths.items()):
            raise PermissionError("v1.3 gate omits or substitutes a frozen source path")
        if (protection.get("reviewer_report_path") != gate.get("reviewer_report_path")
                or protection.get("reviewer_report_sha256") != gate.get("reviewer_report_sha256")
                or not gate.get("reviewer_report_path") or not gate.get("reviewer_report_sha256")):
            raise PermissionError("v1.3 protection is not bound to the current independent review")
        try:
            from stage_F0_migration import validate_h49_runtime
            from stage_F0_queue import code_hash
            runtime = validate_h49_runtime(
                RUNS,
                correction_sha256=str(protection.get("provenance_correction_sha256", "")),
                migration_sha256=str(protection.get("migration_audit_sha256", "")),
                expected_asset_sha256=str(protection.get("completed_asset_inventory_sha256", "")),
                expected_incident_asset_sha256=str(
                    protection.get("incident_asset_inventory_sha256", "")),
                require_active_hold=False,
            )
            if (protection.get("original_source_path") != "execution_control/postabnormal_sigsegv_audit_v4_fresh_search.json"
                    or protection.get("original_source_sha256") != runtime.get("v4_sha256")
                    or protection.get("design_sha256") != runtime.get("design_sha256")
                    or protection.get("implementation_report_sha256") != runtime.get(
                        "implementation_report_sha256")
                    or protection.get("search_queue_file_sha256") != runtime.get("search_queue_file_sha256")
                    or protection.get("search_queue_path") != "experiments/stage_F0_outputs/queues/F0-B1-ML-search-queue.json"
                    or protection.get("search_queue_entries_sha256") != runtime.get("search_queue_entries_sha256")
                    or protection.get("incident_asset_inventory_sha256") != runtime.get(
                        "incident_asset_inventory_sha256")
                    or protection.get("completed_asset_inventory_sha256") != runtime.get(
                        "asset_inventory_sha256")
                    or protection.get("provenance_correction_sha256") != runtime.get(
                        "correction_sha256")
                    or protection.get("migration_audit_sha256") != runtime.get(
                        "migration_sha256")
                    or str(gate.get("reviewer_report_sha256", ""))
                    != str(protection.get("reviewer_report_sha256", ""))):
                raise PermissionError("v1.3 gate provenance graph differs from validated runtime")
            amendment = gate.get("scientific_code_hash_amendment")
            if not isinstance(amendment, Mapping) or amendment.get(
                    "runtime_control_code_hash") != code_hash(require_tests=False):
                raise PermissionError("runtime implementation inventory differs from gate amendment")
        except Exception as exc:
            raise PermissionError("v1.3 H49 protection evidence failed validation") from exc
    if gate.get("reviewer_verdict") not in ("PASS", "PASS WITH FIXES"):
        raise PermissionError("new execution-isolation reviewer verdict is not PASS")
    report_path = str(gate.get("reviewer_report_path", ""))
    report_hash = str(gate.get("reviewer_report_sha256", ""))
    if not report_path or not report_hash:
        raise PermissionError("new execution-isolation review report is not frozen")
    if sha256_file(Path(root) / report_path) != report_hash:
        raise PermissionError("execution-isolation reviewer report hash mismatch")
    if gate.get("launch_authorized") is not True:
        raise PermissionError("execution-isolation launch authorization is held")
    if gate.get("launch_scope") != "F0-B1-ML search only; fresh UNCLAIMED canonical keys only":
        raise PermissionError("execution-isolation gate scope is not fresh-only search")

    search = _read_json(search_gate_path)
    frozen_search_sha = str(gate.get("prior_search_gate_sha256", ""))
    if not frozen_search_sha or sha256_file(search_gate_path) != frozen_search_sha:
        raise PermissionError("prior search gate hash differs from the frozen execution gate")
    if str(gate.get("preserved_scientific_code_hash", "")) != str(
            search.get("code_hash", "")):
        raise PermissionError("execution amendment does not bind the frozen scientific code hash")
    search_holds = search.get("blocked_search_keys", [])
    frozen_holds = gate.get("preserved_held_run_ids", [])
    if (not isinstance(search_holds, list)
            or any(not isinstance(value, str) for value in search_holds)
            or not set(search_holds).issubset(_PRESERVED_SIGNAL_CLEANUP_KEY_SET)):
        raise PermissionError("new execution gate drops or malforms prior held search keys")
    if search.get("protocol_version") != "F0-TRAIN-ML-v1":
        raise PermissionError("search protocol version mismatch")
    if search.get("data_audit_verdict") != "PASS" or search.get("gpu_preflight_status") != "PASS":
        raise PermissionError("existing data/GPU run gate is not PASS")
    if search.get("formal_or_test_started") is not False:
        raise PermissionError("formal/test state is not frozen as not-started")

    if Path(root).resolve() == ROOT:
        # Runtime validation permits the migration-active state for audit, but
        # an actual launch additionally requires a complete principal release.
        assert_no_global_hold(RUNS)

    amendment = gate.get("scientific_code_hash_amendment")
    if not isinstance(amendment, Mapping):
        raise PermissionError("control-only scientific-code amendment is missing")
    from stage_F0_queue import code_hash
    if (amendment.get("status") != "APPROVED"
            or amendment.get("reviewer_verdict") not in ("PASS", "PASS WITH FIXES")
            or str(amendment.get("frozen_scientific_code_hash", "")) != str(
                search.get("code_hash", ""))
            or amendment.get("control_hash_inventory_sha256") != actual_control_hash
            or amendment.get("runtime_control_code_hash") != code_hash(require_tests=False)):
        raise PermissionError("control-only scientific-code amendment is unreviewed or stale")

    incident_release = gate.get("prior_incident_release")
    if search.get("execution_status") == "PAUSED_ABNORMAL_WORKER_EXIT":
        if not isinstance(incident_release, Mapping):
            raise PermissionError("prior abnormal-worker global hold has no reviewed release")
        if (incident_release.get("reviewer_verdict") not in ("PASS", "PASS WITH FIXES")
                or not incident_release.get("release_id")
                or not incident_release.get("approved_by")
                or incident_release.get("approved_by_role") != "principal"
                or not set(PRESERVED_INCIDENT_KEYS).issubset(
                    set(incident_release.get("preserved_held_run_ids", [])))):
            raise PermissionError("prior incident release is invalid or drops a protected key")
    return {"execution_gate": gate, "search_gate": search}


def frozen_held_run_ids(execution_gate: Mapping[str, object],
                        search_gate: Mapping[str, object],
                        caller_held_run_ids: object) -> list[str]:
    """Return the single held-key set frozen by both gates and the caller."""

    gate_holds = execution_gate.get("preserved_held_run_ids")
    search_holds = search_gate.get("blocked_search_keys", [])
    if (not isinstance(gate_holds, list) or not isinstance(search_holds, list)
            or not isinstance(caller_held_run_ids, list)
            or any(not isinstance(value, str)
                   for value in gate_holds + search_holds + caller_held_run_ids)):
        raise PermissionError("held run IDs are missing or malformed")
    if (len(gate_holds) != len(set(gate_holds))
            or len(caller_held_run_ids) != len(set(caller_held_run_ids))):
        raise PermissionError("held run IDs contain duplicates")
    gate_set = set(gate_holds)
    if gate_set != set(PRESERVED_H51_RUN_IDS):
        raise PermissionError("execution gate differs from the frozen H51 hold set")
    if not set(search_holds).issubset(gate_set):
        raise PermissionError("execution gate drops prior search holds")
    if set(caller_held_run_ids) != gate_set:
        raise PermissionError("caller-held IDs differ from the frozen execution gate")
    if held_run_ids_sha256(gate_holds) != PRESERVED_H51_SHA256:
        raise PermissionError("execution gate H51 hash mismatch")
    return sorted(gate_set)


def execution_lock_path(output_root: os.PathLike | str = RUNS) -> Path:
    return Path(output_root).resolve() / "execution_control" / "execution_state.lock"


def global_hold_path(output_root: os.PathLike | str = RUNS) -> Path:
    return Path(output_root).resolve() / "execution_control" / "global_hold.json"


def global_hold_release_path(output_root: os.PathLike | str = RUNS) -> Path:
    return Path(output_root).resolve() / "execution_control" / "global_hold_release.json"


def _hold_is_released(hold: Mapping[str, object], output_root: os.PathLike | str) -> bool:
    root = Path(output_root).resolve()
    candidate_gate = root.parent / "execution_isolation_gate.json"
    if candidate_gate.is_file() and not candidate_gate.is_symlink():
        try:
            candidate = _read_json(candidate_gate)
            if candidate.get("schema") == "F0-EXECUTION-ISOLATION-v1.5":
                from stage_F0_successor_v15 import validate_current_gate as validate_v15_gate
                repo = root.parents[2]
                validate_v15_gate(candidate, root=repo, require_release=True)
                return True
        except Exception:
            return False
    if Path(output_root).resolve() == RUNS:
        try:
            gate = _read_json(EXECUTION_GATE)
            if gate.get("schema") == "F0-EXECUTION-ISOLATION-v1.5":
                from stage_F0_successor_v15 import validate_current_gate as validate_v15_gate
                validate_v15_gate(gate, root=ROOT, require_release=True)
                return True
            if gate.get("schema") != "F0-EXECUTION-ISOLATION-v1.4":
                return False
            from stage_F0_postabnormal_v14 import ensure_current_release
            ensure_current_release(output_root)
            return True
        except Exception:
            return False
    release_path = global_hold_release_path(output_root)
    if not release_path.is_file():
        return False
    try:
        release = _read_json(release_path)
    except (OSError, ValueError):
        return False
    return bool(
        release.get("reviewer_verdict") in ("PASS", "PASS WITH FIXES")
        and release.get("hold_id") == hold.get("hold_id")
        and release.get("approved_by")
        and release.get("approved_by_role") == "principal"
        and release.get("release_id")
        and release.get("hold_sha256") == sha256_file(global_hold_path(output_root))
        and hold.get("preserved_held_run_ids_sha256") == PRESERVED_H49_SHA256
        and _has_exact_preserved_signal_cleanup_keys(
            hold.get("preserved_held_run_ids"))
        and release.get("preserved_held_run_ids_sha256") == PRESERVED_H49_SHA256
        and _has_exact_preserved_signal_cleanup_keys(
            release.get("preserved_held_run_ids"))
        and _validate_released_v13_runtime(output_root)
    )


def _validate_released_v13_runtime(output_root: os.PathLike | str) -> bool:
    """A released hold still needs exact v1.3 status/source/asset validation."""
    try:
        from stage_F0_migration import validate_h49_runtime
        root = Path(output_root).resolve()
        release = _read_json(global_hold_release_path(root))
        validate_h49_runtime(
            root,
            correction_sha256=str(release.get("provenance_correction_sha256", "")),
            migration_sha256=str(release.get("migration_audit_sha256", "")),
            expected_asset_sha256=str(release.get("completed_asset_inventory_sha256", "")),
            expected_incident_asset_sha256=str(
                release.get("incident_asset_inventory_sha256", "")),
            require_active_hold=False,
        )
        return True
    except Exception:
        return False


def assert_no_global_hold(output_root: os.PathLike | str = RUNS, *,
                          lock_already_held: bool = False) -> None:
    path = global_hold_path(output_root)
    root = Path(output_root).resolve()
    candidate_gate = root.parent / "execution_isolation_gate.json"
    if candidate_gate.is_file() and not candidate_gate.is_symlink():
        try:
            candidate = _read_json(candidate_gate)
        except (OSError, ValueError) as exc:
            raise PermissionError("execution gate is unreadable; claim blocked") from exc
        if candidate.get("schema") == "F0-EXECUTION-ISOLATION-v1.5":
            if not path.is_file() or path.is_symlink():
                raise PermissionError("v1.5 current global hold is missing/non-regular")
            from stage_F0_successor_v15 import assert_no_global_hold as assert_v15_no_hold
            assert_v15_no_hold(root, lock_already_held=lock_already_held)
            return
    # The production search root is v1.4-only.  A missing hold is corruption,
    # not evidence that launch is safe, and current claims must validate the
    # complete release graph while serialized by execution_state.
    if root == Path(RUNS).resolve():
        if not path.is_file():
            raise PermissionError("current global hold is missing; claim blocked")
        gate = _read_json(EXECUTION_GATE)
        if gate.get("schema") == "F0-EXECUTION-ISOLATION-v1.5":
            from stage_F0_successor_v15 import assert_no_global_hold as assert_v15_no_hold
            assert_v15_no_hold(root, lock_already_held=lock_already_held)
            return
        if gate.get("schema") != "F0-EXECUTION-ISOLATION-v1.4":
            raise PermissionError("unknown production execution-gate schema")
        from stage_F0_postabnormal_v14 import ensure_current_release
        ensure_current_release(root)
        return
    if not path.exists():
        return
    try:
        hold = _read_json(path)
    except (OSError, ValueError) as exc:
        raise PermissionError("global execution hold is unreadable; claim blocked") from exc
    if hold.get("active") is not False or not _hold_is_released(hold, output_root):
        raise PermissionError("global execution hold is active; claim blocked")


def record_global_hold(*, output_root: os.PathLike | str = RUNS,
                       event: Mapping[str, object]) -> dict:
    """Durably hold all new claims, serialized against queue claim-next."""

    root = Path(output_root).resolve()
    lock_path = execution_lock_path(root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        hold_path = global_hold_path(root)
        if hold_path.exists():
            try:
                hold = _read_json(hold_path)
            except (OSError, ValueError):
                if root == RUNS:
                    raise PermissionError("current v1.4 global hold is unreadable; refusing replacement")
                hold = {"hold_id": uuid.uuid4().hex, "active": True,
                        "reason": "existing hold record unreadable"}
        else:
            hold = {"hold_id": uuid.uuid4().hex, "active": True,
                    "created_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "created_unix": time.time(), "host": socket.gethostname(),
                    "reason": "execution worker failure"}
        existing_events = hold.get("events", [])
        if not isinstance(existing_events, list) and root == RUNS:
            raise PermissionError("current v1.4 global hold events are malformed")
        events = list(existing_events) if isinstance(existing_events, list) else []
        events.append(dict(event))
        event_value = dict(event)
        event_value.setdefault("preserved_held_run_ids", list(PRESERVED_H51_RUN_IDS))
        event_value.setdefault("preserved_held_run_ids_sha256", PRESERVED_H51_SHA256)
        events[-1] = event_value
        hold.update({"active": True, "events": events,
                     "preserved_held_run_ids": list(PRESERVED_H51_RUN_IDS),
                     "preserved_held_run_ids_sha256": PRESERVED_H51_SHA256,
                     "updated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                     "updated_unix": time.time()})
        _atomic_json(hold_path, hold)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return hold


def append_phase_event(path: os.PathLike | str, *, phase: str, worker_id: str,
                       nonce: str, pid: Optional[int] = None,
                       canonical: Optional[Mapping[str, object]] = None,
                       gpu: Optional[Mapping[str, object]] = None,
                       **fields) -> dict:
    event = {
        "event": "PHASE", "phase": str(phase), "worker_id": str(worker_id),
        "nonce": str(nonce), "pid": int(pid if pid is not None else os.getpid()),
        "ppid": os.getppid(), "pgid": os.getpgrp(), "host": socket.gethostname(),
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="microseconds"),
        "unix_time": time.time(),
    }
    if canonical is not None:
        event["canonical"] = dict(canonical)
    if gpu is not None:
        event["gpu"] = dict(gpu)
    event.update(fields)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True,
                                allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def linux_process_identity(pid: int) -> dict:
    """Read start ticks and process-group identity without psutil."""

    raw = Path("/proc/{}/stat".format(int(pid))).read_text(encoding="ascii")
    right = raw.rfind(")")
    if right < 0:
        raise ValueError("malformed /proc process stat")
    tail = raw[right + 2:].split()  # starts at field 3 (state)
    if len(tail) < 20:
        raise ValueError("short /proc process stat")
    return {"pid": int(pid), "state": tail[0], "ppid": int(tail[1]),
            "pgid": int(tail[2]), "start_ticks": int(tail[19])}


def make_invocation_nonce() -> str:
    return uuid.uuid4().hex


__all__ = [
    "ROOT", "OUTPUT", "RUNS", "EXECUTION_GATE", "SEARCH_GATE",
    "CONTROL_HASH_FILES", "PRESERVED_INCIDENT_KEYS",
    "PRESERVED_V12_H47_RUN_IDS", "V13_ADDITIONAL_HELD_RUN_IDS",
    "PRESERVED_H49_RUN_IDS", "PRESERVED_H47_SHA256", "PRESERVED_H49_SHA256",
    "PRESERVED_H51_RUN_IDS", "PRESERVED_H51_SHA256", "PRESERVED_CURRENT_RUN_IDS",
    "PRESERVED_CURRENT_SHA256", "PRESERVED_SIGNAL_CLEANUP_KEYS", "held_run_ids_sha256",
    "require_exact_h49", "require_exact_h51", "sha256_file",
    "control_hash_inventory", "validate_execution_gate", "frozen_held_run_ids",
    "execution_lock_path",
    "global_hold_path", "global_hold_release_path", "assert_no_global_hold",
    "record_global_hold", "append_phase_event", "linux_process_identity",
    "make_invocation_nonce",
]
