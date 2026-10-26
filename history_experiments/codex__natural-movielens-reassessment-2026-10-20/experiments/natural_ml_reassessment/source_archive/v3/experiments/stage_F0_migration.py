"""Fail-closed v1.3 administrative hold migration for F0 execution control.

This module only changes the two queue-status records and appends execution
control evidence. It never opens a checkpoint for reading or writing and never
calls the runner/recovery path.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import json
import os
import socket
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from stage_F0_worker_control import (
    PRESERVED_H47_SHA256,
    PRESERVED_H49_RUN_IDS,
    PRESERVED_H49_SHA256,
    PRESERVED_V12_H47_RUN_IDS,
    RUNS,
    held_run_ids_sha256,
    linux_process_identity,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]


SCHEMA = "F0-ADMINISTRATIVE-HOLD-MIGRATION-v1.3"
V4_PATH = "execution_control/postabnormal_sigsegv_audit_v4_fresh_search.json"
V4_SHA256 = "a50162f3016841257e8c098cc0040c9ff2be4edb30073c4c2fb42f8ed416f839"
ORIGINAL_HOLD_SHA256 = "e9461efcc99e49bef4b813a3f3cb1c67720bdcd172470df2a73a9e7caca400b9"
EXPECTED_ASSET_INVENTORY_SHA256 = "94039db9a66bfe20e7a4bb2d96b9249c145aa2fcf23fb53ca71bc225ffc718ae"
EXPECTED_COMPLETED_RUN_IDS = (
    "028093a315467d7cf46d", "07710a9213f91c6ad69c", "07fd039feb356252e832",
    "09b7c9f1d1f8932c2bf1", "0ceb319a06ce316cb49b", "0e53caff7b8cca0542ac",
    "1934d9bf8a3666094034", "1f0f71f9a2dd1c234f01", "1f705f86c1ee3846651c",
    "24d8480ca32a4e264cb9", "258e3d1ac516a1085e7c", "2778b13c195202afc98d",
    "278d388ae5e4b83b69ab", "2e921dc7c512f5ab76ce", "34270c398a9ef5904491",
    "37e6a921c1b4531a6bce", "3fb4c73e5c08d6265bb8", "487202d3c5e65bd1b16a",
    "4b1c62798fd5e1d0f1d7", "556d07110eae8b78047e", "5cfa53537fe39eb30f32",
    "5fdc46e6cb9b4e6eb7c4", "653e78731355d7906795", "7175930647197a350f97",
    "730781aac5ae749210ff", "749e8617426c9912a040", "831420788e1c8f6c9137",
    "83a867b7ba406a336d98", "8ee64450d940cea4575b", "9654fbd51197fb5947d6",
    "9e017baa278d28b86cb5", "9f256debdcdb06deae8e", "9f2fdfa7c8670cadad54",
    "a1182293c89d2c0fad85", "a15d0b4d5b0ad472d0dc", "a5c0e3ffb06fd2b86f08",
    "abb17db3d5c34fe8c19a", "ac9d0928766aff133dd9", "af99511ddafee8359546",
    "b15f8a0a152ea2f1c238", "b31091aa9685f951ad4d", "b317b01a512ddf9b33d9",
    "b60376c2c7d61e36eea8", "cbd8481befe4df37ab74", "ce6b3e704d7a8603267a",
    "d431f7a1bf14e11952c7", "dadc19096b9f5d30e28d", "dfbe228b5f81c8257c3e",
    "e1a2c116f3fc8060feeb", "eccb7d1fbb17252a8c1f", "fb23fe17d34c4c4ff24c",
    "ffd63535523b4f285107",
)
EXPECTED_COMPLETED_RUN_IDS_SHA256 = "3d1bb138c30d7dd0ae07d7d3e5730e29d27248c6ecc12ee72afc8a2df6fa7a7a"
EXPECTED_INCIDENT_ASSET_INVENTORY_SHA256 = "10fc3b31d3f462a89f9204db14f7e97a7bce2ad733279bf2724d3311e3e6e15e"
DESIGN_PATH = "experiments/reports/stage_F0_execution_isolation_design.md"
DESIGN_SHA256 = "163f2fbc181712e0186cdb651f8738bb72a27f7a65edf92e4e554e677aac62e7"
IMPLEMENTATION_REPORT_PATH = "experiments/reports/stage_F0_execution_isolation_implementation_report.md"
RELEASE_SCHEMA = "F0-GLOBAL-HOLD-RELEASE-v1.3"
FRESH_ONLY_SCOPE = "F0-B1-ML search; fresh UNCLAIMED only"
MIGRATED_STATUS_AUTHORIZATION = {
    "resume": False, "retry": False, "new_attempt": False,
    "checkpoint_load": False, "formal": False, "test": False,
}
TARGET_SIGSEGV = "7fd2006bdc8f1817a6f7"
TARGET_PEER = "1f04c7fb623d18f940ac"
TARGET_IDS = (TARGET_SIGSEGV, TARGET_PEER)
PEER_RECEIPT_REL = "worker_logs/trajectory_worker0_2_c87c8e65bc7e4b73bed880465ff2dffb.receipt.json"
CORRECTION_NAME = "postabnormal_sigsegv_audit_v4_provenance_correction.json"
MIGRATION_NAME = "postabnormal_sigsegv_audit_v5_migration.jsonl"
CORRECTION_PATH = Path("execution_control") / CORRECTION_NAME
MIGRATION_PATH = Path("execution_control") / MIGRATION_NAME
SEARCH_QUEUE_REL = Path("../queues/F0-B1-ML-search-queue.json")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _compact_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, object], *, fault: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with open(tmp, "wb") as handle:
            payload = json.dumps(dict(value), ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=False).encode("utf-8") + b"\n"
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if fault == "before_rename":
            raise OSError("injected failure before atomic rename")
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            if fault == "before_dir_fsync":
                raise OSError("injected failure before directory fsync")
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _fsync_existing_file_and_parent(path: Path) -> None:
    """Re-establish file and directory durability after an interrupted rename."""

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _append_jsonl(path: Path, event: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(event), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short append to migration ledger")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _status_path(runs_root: Path, run_id: str) -> Path:
    return runs_root / "queue_status" / f"{run_id}.json"


def _search_queue_path(runs_root: Path) -> Path:
    return (runs_root / SEARCH_QUEUE_REL).resolve()


def _artifact_root_for_runs(runs_root: Path) -> Path:
    root = Path(runs_root).resolve()
    if (root.name == "runs" and root.parent.name == "stage_F0_outputs"
            and root.parent.parent.name == "experiments"):
        return root.parent.parent.parent
    return ROOT


def _load_search_queue(runs_root: Path) -> list[dict[str, Any]]:
    value = _read_json(_search_queue_path(runs_root))
    rows = value.get("entries")
    if not isinstance(rows, list) or len(rows) != 288:
        raise PermissionError("frozen search queue must contain exactly 288 entries")
    ids = [str(row.get("run_id", "")) for row in rows]
    if len(ids) != len(set(ids)) or set(TARGET_IDS).difference(ids):
        raise PermissionError("search queue IDs are duplicated or omit v1.3 targets")
    return rows


def _queue_statuses(runs_root: Path, rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, int]]:
    statuses: dict[str, str] = {}
    for row in rows:
        run_id = str(row["run_id"])
        path = _status_path(runs_root, run_id)
        if not path.exists():
            statuses[run_id] = "UNCLAIMED"
            continue
        payload = _read_json(path)
        if str(payload.get("run_id", run_id)) != run_id:
            raise PermissionError(f"status run_id mismatch: {run_id}")
        status = payload.get("status")
        if not isinstance(status, str) or not status:
            raise PermissionError(f"status missing/malformed: {run_id}")
        statuses[run_id] = status
    counts: dict[str, int] = {}
    for status in statuses.values():
        counts[status] = counts.get(status, 0) + 1
    return statuses, counts


def _inventory_for_paths(runs_root: Path, paths: set[Path]) -> tuple[list[dict[str, object]], str]:
    items: list[dict[str, object]] = []
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise PermissionError(f"protected asset is missing or unsupported: {path}")
        rel = "experiments/stage_F0_outputs/runs/" + path.relative_to(runs_root).as_posix()
        items.append({"path": rel, "size": path.stat().st_size,
                      "sha256": sha256_file(path)})
    items.sort(key=lambda item: str(item["path"]))
    # The baseline inventory uses insertion-ordered records path,size,sha256
    # and compact UTF-8 JSON; preserve that frozen representation exactly.
    digest = _sha256_bytes(json.dumps(items, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8"))
    return items, digest


def _tree_files(base: Path) -> set[Path]:
    if not base.is_dir():
        raise PermissionError(f"protected run attempt tree is missing: {base}")
    paths: set[Path] = set()
    for parent, dirs, files in os.walk(base, followlinks=False):
        dirs.sort()
        files.sort()
        if any((Path(parent) / name).is_symlink() for name in dirs):
            raise PermissionError(f"protected attempt tree contains a directory symlink: {base}")
        for name in files:
            paths.add(Path(parent) / name)
    return paths


def _asset_inventory(runs_root: Path, rows: list[dict[str, Any]],
                     completed_run_ids: Optional[list[str] | tuple[str, ...]] = None
                     ) -> tuple[list[dict[str, object]], str]:
    """Hash the immutable original completed trajectories, independent of live statuses."""

    ids = tuple(EXPECTED_COMPLETED_RUN_IDS if completed_run_ids is None else completed_run_ids)
    queue_ids = {str(row.get("run_id", "")) for row in rows}
    if (len(ids) != 52 or len(set(ids)) != 52
            or set(ids) != set(EXPECTED_COMPLETED_RUN_IDS)
            or not set(ids).issubset(queue_ids)):
        raise PermissionError("completed asset keys differ from the independently frozen 52-key set")
    if _sha256_bytes(_compact_json(sorted(ids))) != EXPECTED_COMPLETED_RUN_IDS_SHA256:
        raise PermissionError("frozen completed-run key set hash mismatch")
    paths: set[Path] = set()
    for run_id in ids:
        paths.update(_tree_files(runs_root / run_id))
    return _inventory_for_paths(runs_root, paths)


def _incident_asset_inventory(runs_root: Path) -> tuple[list[dict[str, object]], str]:
    """Hash attempt trees and evidence anchored by v4 and the frozen hold event."""

    runs_root = Path(runs_root).resolve()
    v4 = _read_json(runs_root / V4_PATH)
    paths: set[Path] = set()
    for run_id in TARGET_IDS:
        paths.update(_tree_files(runs_root / run_id / "attempts" / "attempt-00"))
    for section in ("sigsegv_worker", "peer_stopped_worker"):
        worker = v4.get(section)
        if not isinstance(worker, Mapping):
            raise PermissionError("incident asset source record is incomplete")
        keys = (("failure_audit_path",) if section == "sigsegv_worker" else ())
        keys += ("stderr_path", "phase_log_path")
        if section == "peer_stopped_worker":
            keys += ("receipt_path",)
        for key in keys:
            relative = str(worker.get(key, ""))
            prefix = "experiments/stage_F0_outputs/runs/"
            if not relative.startswith(prefix):
                raise PermissionError(f"incident asset path is not repository-relative: {key}")
            path = runs_root / relative.removeprefix(prefix)
            paths.add(path)
        stdout = (Path("worker_logs") /
                  Path(f"trajectory_worker{worker.get('worker_id')}_{worker.get('invocation')}_"
                       f"{worker.get('nonce')}.stdout.log"))
        stdout_path = runs_root / stdout
        if stdout_path.exists():
            paths.add(stdout_path)
    # v4 records the peer's durable INTERRUPTED receipt, while the active hold
    # event is the frozen source for its supervisor failure audit. Bind that
    # audit by canonical key, worker, nonce, repository path, and actual bytes;
    # never consult a released/reusable GPU lock owner for incident identity.
    peer = v4["peer_stopped_worker"]
    peer_nonce = str(peer.get("nonce", ""))
    peer_worker_id = str(peer.get("worker_id", ""))
    hold = _read_json(runs_root / "execution_control" / "global_hold.json")
    hold_events = hold.get("events")
    if not isinstance(hold_events, list):
        raise PermissionError("global hold has no frozen incident event list")
    peer_failure_events = [
        event for event in hold_events
        if isinstance(event, Mapping)
        and event.get("event") == "WORKER_FAILURE_HOLD"
        and event.get("canonical_run_id") == TARGET_PEER
    ]
    if (len(peer_failure_events) != 1
            or peer_failure_events[0].get("nonce") != peer_nonce
            or str(peer_failure_events[0].get("worker_id", "")) != peer_worker_id):
        raise PermissionError("peer failure audit is not uniquely bound by hold event and nonce")
    peer_failure_event = peer_failure_events[0]
    frozen_failure_path = Path(str(peer_failure_event.get("failure_path", "")))
    production_runs = Path(__file__).resolve().parents[1] / "experiments/stage_F0_outputs/runs"
    try:
        peer_failure_relative = frozen_failure_path.relative_to(production_runs)
    except ValueError as exc:
        raise PermissionError("peer failure audit path is outside the frozen F0 runs tree") from exc
    if (peer_failure_relative.parent != Path("worker_logs")
            or not peer_failure_relative.name.startswith(
                f"worker_{peer_worker_id}_failure_")
            or peer_nonce not in peer_failure_relative.name):
        raise PermissionError("peer failure audit path does not bind worker and nonce")
    peer_failure_path = runs_root / peer_failure_relative
    peer_failure_sha = str(peer_failure_event.get("failure_sha256", ""))
    if (len(peer_failure_sha) != 64
            or any(char not in "0123456789abcdef" for char in peer_failure_sha)
            or not peer_failure_path.is_file()
            or sha256_file(peer_failure_path) != peer_failure_sha):
        raise PermissionError("peer durable failure audit is missing or hash-mismatched")
    paths.add(peer_failure_path)
    sig = v4["sigsegv_worker"]
    missing_receipt = runs_root / "worker_logs" / (
        f"trajectory_worker{sig.get('worker_id')}_{sig.get('invocation')}_"
        f"{sig.get('nonce')}.receipt.json")
    if missing_receipt.exists():
        raise PermissionError("SIGSEGV child unexpectedly has a receipt")
    return _inventory_for_paths(runs_root, paths)


def _active_hold(runs_root: Path) -> tuple[Path, dict[str, Any], str]:
    path = runs_root / "execution_control" / "global_hold.json"
    payload = _read_json(path)
    if payload.get("active") is not True:
        raise PermissionError("global execution hold must remain active during v1.3 migration")
    return path, payload, sha256_file(path)


def _probe_process(pid: object, expected_start_ticks: object) -> bool:
    try:
        pid_i = int(pid)
        expected_i = int(expected_start_ticks)
        actual = linux_process_identity(pid_i)
    except (OSError, ValueError, TypeError, ProcessLookupError):
        return False
    return actual["start_ticks"] == expected_i


def _ensure_incident_processes_exited(v4: Mapping[str, Any], runs_root: Path) -> None:
    # v4 freezes PID/PGID while queue status freezes Linux startticks. Combine
    # them so a reused PID cannot be mistaken for the original child.
    for section, run_id in (("sigsegv_worker", TARGET_SIGSEGV),
                            ("peer_stopped_worker", TARGET_PEER)):
        worker = v4.get(section)
        status = _read_json(_status_path(runs_root, run_id))
        if not isinstance(worker, Mapping):
            raise PermissionError("v4 incident identity is incomplete")
        if (int(status.get("supervisor_pid", -1)) != int(worker.get("supervisor_pid", -2))
                or int(status.get("pid", -1)) != int(worker.get("child_pid", -2))
                or int(status.get("pgid", -1)) != int(worker.get("supervisor_pgid", -2))):
            raise PermissionError("v4 and queue-status process identity sources conflict")
        identities = ((status.get("supervisor_pid"), status.get("supervisor_start_ticks")),
                      (status.get("pid"), status.get("start_ticks")))
        for pid, ticks in identities:
            if pid is None or ticks is None:
                raise PermissionError("v4 incident PID/start-ticks are incomplete")
            if _probe_process(pid, ticks):
                raise PermissionError(f"recorded incident process is still alive: pid={pid}")


def _ensure_locks_free(runs_root: Path) -> None:
    paths = [runs_root / "worker_locks" / f"worker_{i}.lock" for i in (0, 1)]
    paths += [runs_root / "gpu_locks" / f"gpu{i}.lock" for i in (0, 1)]
    for run_id in TARGET_IDS:
        paths.append(runs_root / "canonical_locks" / f"{run_id}.lock")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PermissionError(f"execution lock is still held: {path}") from exc
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            os.close(fd)


def validate_provenance(runs_root: Path = RUNS) -> dict[str, Any]:
    """Verify v4 source, actual receipt and source-correction facts."""

    runs_root = Path(runs_root).resolve()
    v4_path = runs_root / V4_PATH
    if sha256_file(v4_path) != V4_SHA256:
        raise PermissionError("frozen v4 audit hash mismatch")
    v4 = _read_json(v4_path)
    if v4.get("schema") != "F0-POSTABNORMAL-WORKER-AUDIT-v1":
        raise PermissionError("frozen v4 audit schema mismatch")
    if v4.get("automatic_retry_authorized") is not False or v4.get("recovery_authorized") is not False:
        raise PermissionError("v4 source unexpectedly authorizes retry/recovery")
    sig = v4.get("sigsegv_worker")
    peer = v4.get("peer_stopped_worker")
    if not isinstance(sig, Mapping) or not isinstance(peer, Mapping):
        raise PermissionError("v4 incident records are incomplete")
    if sig.get("run_id") != TARGET_SIGSEGV or peer.get("run_id") != TARGET_PEER:
        raise PermissionError("v4 incident run IDs differ from frozen H49 extension")
    receipt_path = runs_root / PEER_RECEIPT_REL
    receipt_hash = sha256_file(receipt_path)
    if receipt_hash != "3a3b86cd54b0685ec09506a32706355a26c3465ed41d872c06e641d94bcbacf7":
        raise PermissionError("peer receipt actual hash differs from approved provenance")
    receipt = _read_json(receipt_path)
    actual_uuid = str(receipt.get("gpu_lock_owner", {}).get("gpu_uuid", ""))
    canonical_uuid = str(receipt.get("canonical_lock_owner", {}).get("gpu_uuid", ""))
    peer_queue = _read_json(_status_path(runs_root, TARGET_PEER))
    peer_source_status = peer_queue
    if peer_queue.get("status") == "RECOVERY_BLOCKED":
        migration_path = runs_root / MIGRATION_PATH
        if not migration_path.is_file():
            raise PermissionError("migrated peer status lacks its PREPARED original source")
        migration_events = [json.loads(line) for line in
                            migration_path.read_text(encoding="utf-8").splitlines() if line]
        prepared = next((event for event in migration_events
                         if event.get("event") == "PREPARED"), None)
        original_record = (prepared or {}).get("original_statuses", {}).get(TARGET_PEER)
        if not isinstance(original_record, Mapping):
            raise PermissionError("PREPARED omits original peer status source")
        original_raw = base64.b64decode(
            str(original_record.get("raw_bytes_base64", "")), validate=True)
        if (_sha256_bytes(original_raw) !=
                "61921430b4975ba67720efd19fcb24a6ae4debb5f232f56d536e0e568ad41478"
                or original_record.get("sha256") != _sha256_bytes(original_raw)):
            raise PermissionError("PREPARED original peer status bytes differ from frozen source")
        peer_source_status = json.loads(original_raw)
    peer_uuid = str(peer_source_status.get("gpu_uuid", ""))
    expected_uuid = "2a59fb49-6bef-5320-74ab-ffb24d056908"
    if not (actual_uuid == canonical_uuid == peer_uuid == expected_uuid):
        raise PermissionError("historical peer GPU UUID source conflict is not resolved")
    if peer.get("receipt_sha256") != "3a3b86cd54b0685ec09506a32706355a26c3465ed41d8722f93b2f4":
        raise PermissionError("v4 receipt typo does not match the approved correction source")
    if str(peer.get("gpu", {}).get("uuid", "")) != "2a59fb49-6bef-5320-74ab-bffb24d056908":
        raise PermissionError("v4 GPU UUID typo does not match the approved correction source")
    if receipt.get("status") != "INTERRUPTED" or receipt.get("run_id") != TARGET_PEER:
        raise PermissionError("peer receipt identity/status mismatch")
    if receipt.get("nonce") != peer.get("nonce") or receipt.get("attempt_id") != peer.get("attempt_id"):
        raise PermissionError("peer receipt nonce/attempt identity mismatch")
    if receipt.get("transaction_commit") != "DURABLE_BEFORE_LOCK_RELEASE":
        raise PermissionError("peer receipt transaction is not durably committed")
    manifest_path = (runs_root / TARGET_PEER / "attempts" /
                     str(peer["attempt_id"]) / "run_manifest.json")
    if sha256_file(manifest_path) != "2e85d70d0d97b144fd26cba06be61f9ce3d86bb3879839adde9c866c6377f1d6":
        raise PermissionError("peer attempt manifest hash differs from frozen v1.3 source")
    checkpoint_path = (runs_root / TARGET_PEER / "attempts" /
                       str(peer["attempt_id"]) / "resume_epoch_017.pt")
    if sha256_file(checkpoint_path) != "8843afe2a382b47b7a920986ffbfc0cf996a73b0ab8822bfd138760b2517597c":
        raise PermissionError("peer last complete checkpoint hash differs from frozen v1.3 source")
    sig_status_path = _status_path(runs_root, TARGET_SIGSEGV)
    peer_status_path = _status_path(runs_root, TARGET_PEER)
    sig_status = json.loads(sig_status_path.read_bytes())
    peer_status = json.loads(peer_status_path.read_bytes())
    if not isinstance(sig_status, dict) or not isinstance(peer_status, dict):
        raise PermissionError("target status source is malformed")
    sig_status_hash = sha256_file(sig_status_path)
    peer_status_hash = sha256_file(peer_status_path)
    sig_original_hash = "74cbabc600e19e4e977c3b38d3068419c72c3be0b678cea5857a1867ec66dbbe"
    peer_original_hash = "61921430b4975ba67720efd19fcb24a6ae4debb5f232f56d536e0e568ad41478"
    if sig_status.get("status") == "RUNNING":
        if sig_status_hash != sig_original_hash:
            raise PermissionError("SIGSEGV original status hash changed before migration")
    elif (sig_status.get("status") != "RECOVERY_BLOCKED"
          or sig_status.get("v13_original_status_sha256") != sig_original_hash
          or sig_status.get("v13_original_status_raw_sha256") != sig_original_hash
          or not sig_status.get("v13_migration_id")):
        raise PermissionError("SIGSEGV status lacks exact v1.3 original-byte provenance")
    if peer_status.get("status") == "INTERRUPTED":
        if peer_status_hash != peer_original_hash:
            raise PermissionError("peer original status hash changed before migration")
    elif (peer_status.get("status") != "RECOVERY_BLOCKED"
          or peer_status.get("v13_original_status_sha256") != peer_original_hash
          or peer_status.get("v13_original_status_raw_sha256") != peer_original_hash
          or not peer_status.get("v13_migration_id")):
        raise PermissionError("peer status lacks exact v1.3 original-byte provenance")
    if sig_status.get("attempt_id") != "attempt-00" or peer_status.get("attempt_id") != "attempt-00":
        raise PermissionError("target attempt identity changed")
    if int(sig_status.get("recovery_count", -1)) != 0 or int(peer_status.get("recovery_count", -1)) != 0:
        raise PermissionError("target recovery count changed")
    if sig_status.get("resume_checkpoint") is not None or peer_status.get("resume_checkpoint") is not None:
        raise PermissionError("target has an unexpected resume selection")
    if sig_status.get("status") not in ("RUNNING", "RECOVERY_BLOCKED"):
        raise PermissionError("SIGSEGV target status is neither its frozen source nor migrated state")
    if peer_status.get("status") not in ("INTERRUPTED", "RECOVERY_BLOCKED"):
        raise PermissionError("peer target status is neither its frozen source nor migrated state")
    if sig_status.get("status") == "RECOVERY_BLOCKED" and not sig_status.get("v13_migration_id"):
        raise PermissionError("SIGSEGV target has an unexplained RECOVERY_BLOCKED status")
    if peer_status.get("status") == "RECOVERY_BLOCKED" and not peer_status.get("v13_migration_id"):
        raise PermissionError("peer target has an unexplained RECOVERY_BLOCKED status")
    failure_path = runs_root / str(sig.get("failure_audit_path", "")).removeprefix(
        "experiments/stage_F0_outputs/runs/")
    if (not failure_path.is_file()
            or sha256_file(failure_path) != sig.get("failure_audit_sha256")):
        raise PermissionError("durable SIGSEGV failure audit is missing or hash-mismatched")
    receipt_record = sig.get("receipt")
    if not isinstance(receipt_record, Mapping) or receipt_record.get("exists") is not False:
        raise PermissionError("SIGSEGV source does not prove the receipt is absent")
    missing_receipt = runs_root / "worker_logs" / (
        f"trajectory_worker{sig.get('worker_id')}_{sig.get('invocation')}_"
        f"{sig.get('nonce')}.receipt.json")
    if missing_receipt.exists():
        raise PermissionError("unexpected SIGSEGV child receipt exists")
    _ensure_incident_processes_exited(v4, runs_root)
    return {
        "v4_path": V4_PATH, "v4_sha256": V4_SHA256,
        "peer_receipt_path": PEER_RECEIPT_REL, "peer_receipt_sha256": receipt_hash,
        "wrong_v4_receipt_sha256": str(peer.get("receipt_sha256")),
        "wrong_v4_peer_gpu_uuid": str(peer.get("gpu", {}).get("uuid", "")),
        "actual_peer_gpu_uuid": actual_uuid,
        "peer_queue_status_path": f"queue_status/{TARGET_PEER}.json",
        "peer_queue_status_sha256": (
            peer_original_hash if peer_status.get("status") == "RECOVERY_BLOCKED"
            else peer_status_hash),
        "peer_queue_status_canonical_key_json_sha256": hashlib.sha256(
            str(peer_source_status.get("canonical_key_json", "")).encode("utf-8")).hexdigest(),
        "peer_queue_status_run_id_full_sha256": peer_source_status.get("run_id_full_sha256"),
        "peer_manifest_path": str(manifest_path.relative_to(runs_root)),
        "peer_manifest_sha256": sha256_file(manifest_path),
        "peer_checkpoint_path": str(checkpoint_path.relative_to(runs_root)),
        "peer_checkpoint_sha256": sha256_file(checkpoint_path),
        "target_statuses": {
            TARGET_SIGSEGV: {"status": sig_status.get("status"),
                             "sha256": sig_status_hash},
            TARGET_PEER: {"status": peer_status.get("status"),
                          "sha256": peer_status_hash},
        },
    }


def build_provenance_correction(runs_root: Path = RUNS) -> dict[str, Any]:
    runs_root = Path(runs_root).resolve()
    provenance = validate_provenance(runs_root)
    queue_path = _search_queue_path(runs_root)
    peer_receipt = _read_json(runs_root / PEER_RECEIPT_REL)
    peer_status_path = _status_path(runs_root, TARGET_PEER)
    peer_status_bytes = peer_status_path.read_bytes()
    peer_status = json.loads(peer_status_bytes)
    peer_status_source_sha = provenance["peer_queue_status_sha256"]
    if peer_status.get("status") == "RECOVERY_BLOCKED":
        # The status file now carries v1.3 metadata; the correction remains
        # anchored to the exact original bytes captured in PREPARED.
        events = [json.loads(line) for line in
                  (runs_root / MIGRATION_PATH).read_text(encoding="utf-8").splitlines()
                  if line]
        prepared = next((event for event in events if event.get("event") == "PREPARED"), None)
        original_record = (prepared or {}).get("original_statuses", {}).get(TARGET_PEER, {})
        original_raw = base64.b64decode(str(original_record.get("raw_bytes_base64", "")),
                                        validate=True)
        if (_sha256_bytes(original_raw) != peer_status_source_sha
                or json.loads(original_raw).get("gpu_uuid") != provenance["actual_peer_gpu_uuid"]):
            raise PermissionError("original peer status source is not bound by PREPARED")
        peer_status_bytes = original_raw
        peer_status = json.loads(original_raw)
    receipt_owner = peer_receipt.get("gpu_lock_owner", {})
    canonical_owner = peer_receipt.get("canonical_lock_owner", {})
    return {
        "schema": "F0-POSTABNORMAL-PROVENANCE-CORRECTION-v1.3",
        "created_at_utc": _now(),
        "append_only": True,
        "original_audit": {"path": V4_PATH, "sha256": V4_SHA256,
                           "preserved_byte_for_byte": True,
                           "raw_value": _read_json(runs_root / V4_PATH)},
        "corrections": [
            {"field": "peer_stopped_worker.receipt_sha256",
             "original_value": provenance["wrong_v4_receipt_sha256"],
             "verified_value": provenance["peer_receipt_sha256"],
             "source_path": provenance["peer_receipt_path"],
             "source_sha256": provenance["peer_receipt_sha256"],
             "source_field": "sha256(raw receipt bytes)",
             "source_value": provenance["peer_receipt_sha256"],
             "method": "SHA-256 recomputed from original durable peer receipt"},
            {"field": "peer_stopped_worker.gpu.uuid",
             "original_value": provenance["wrong_v4_peer_gpu_uuid"],
             "verified_value": provenance["actual_peer_gpu_uuid"],
             "sources": [
                 {"path": provenance["peer_receipt_path"],
                  "sha256": provenance["peer_receipt_sha256"],
                  "field": "gpu_lock_owner.gpu_uuid",
                  "value": str(receipt_owner.get("gpu_uuid", ""))},
                 {"path": provenance["peer_queue_status_path"],
                  "sha256": peer_status_source_sha,
                  "field": "gpu_uuid", "value": str(peer_status.get("gpu_uuid", "")),
                  "canonical_key_json_sha256": provenance[
                      "peer_queue_status_canonical_key_json_sha256"],
                  "run_id_full_sha256": provenance["peer_queue_status_run_id_full_sha256"]},
                 {"path": provenance["peer_receipt_path"],
                  "sha256": provenance["peer_receipt_sha256"],
                  "field": "canonical_lock_owner.gpu_uuid",
                  "value": str(canonical_owner.get("gpu_uuid", ""))},
             ],
             "method": "historical receipt lock-owner snapshots plus original queue status"},
        ],
        "impact": "corrects v4 provenance fields only; no canonical identity, metrics, manifest, receipt, checkpoint, or completed asset is changed",
        "verified_by": "v1.3 migration implementation; independent review pending",
    }


def _correction_path(runs_root: Path) -> Path:
    return runs_root / CORRECTION_PATH


def ensure_provenance_correction(runs_root: Path = RUNS) -> tuple[Path, str]:
    path = _correction_path(Path(runs_root).resolve())
    if path.exists():
        value = _read_json(path)
        if value.get("schema") != "F0-POSTABNORMAL-PROVENANCE-CORRECTION-v1.3":
            raise PermissionError("existing provenance correction has wrong schema")
        expected = build_provenance_correction(runs_root)
        comparable = dict(value)
        comparable.pop("created_at_utc", None)
        expected.pop("created_at_utc", None)
        if comparable != expected:
            raise PermissionError("existing provenance correction does not bind exact source paths, hashes, and fields")
        return path, sha256_file(path)
    payload = build_provenance_correction(runs_root)
    # Create exclusively; this artifact is append-only and may never be replaced.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             indent=2, allow_nan=False).encode("utf-8") + b"\n"
        view = memoryview(encoded)
        while view:
            n = os.write(fd, view)
            if n <= 0:
                raise OSError("short write to append-only provenance correction")
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return path, sha256_file(path)


def _lock_file(path: Path, *, nonblocking: bool = True) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(fd, flags)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _acquire_migration_locks(runs_root: Path) -> ExitStack:
    stack = ExitStack()
    paths = [runs_root / "execution_control" / "execution_state.lock"]
    paths.extend(runs_root / "worker_locks" / f"worker_{i}.lock" for i in (0, 1))
    paths.extend(runs_root / "gpu_locks" / f"gpu{i}.lock" for i in (0, 1))
    paths.extend(runs_root / "canonical_locks" / f"{run_id}.lock"
                 for run_id in sorted(TARGET_IDS))
    try:
        for path in paths:
            fd = _lock_file(path, nonblocking=True)
            stack.callback(os.close, fd)
            stack.callback(fcntl.flock, fd, fcntl.LOCK_UN)
    except BaseException:
        stack.close()
        raise
    return stack


def _ensure_h47_statuses(runs_root: Path, rows: list[dict[str, Any]]) -> None:
    statuses, _ = _queue_statuses(runs_root, rows)
    for run_id in PRESERVED_V12_H47_RUN_IDS:
        if statuses.get(run_id) != "RECOVERY_BLOCKED":
            raise PermissionError(f"historical H47 key is not RECOVERY_BLOCKED: {run_id}")


def _validate_initial_counts(runs_root: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    statuses, counts = _queue_statuses(runs_root, rows)
    expected = {"VALIDATION_COMPLETE": 52, "RECOVERY_BLOCKED": 47,
                "UNCLAIMED": 187, "RUNNING": 1, "INTERRUPTED": 1}
    if counts != expected:
        raise PermissionError(f"unexpected pre-migration search status ledger: {counts}")
    if statuses.get(TARGET_SIGSEGV) != "RUNNING" or statuses.get(TARGET_PEER) != "INTERRUPTED":
        raise PermissionError("target statuses differ from frozen v4 incident")
    if set(statuses[run_id] for run_id in TARGET_IDS) != {"RUNNING", "INTERRUPTED"}:
        raise PermissionError("both target source statuses are not present")
    completed_ids = sorted(run_id for run_id, status in statuses.items()
                           if status == "VALIDATION_COMPLETE")
    if completed_ids != list(EXPECTED_COMPLETED_RUN_IDS):
        raise PermissionError("pre-migration completed key identities differ from the frozen 52")
    _ensure_h47_statuses(runs_root, rows)
    return counts


def _status_snapshot(runs_root: Path, run_id: str) -> tuple[bytes, dict[str, Any], str]:
    path = _status_path(runs_root, run_id)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise PermissionError(f"status projection is malformed: {run_id}")
    return raw, payload, _sha256_bytes(raw)


def _validate_prepared_original_status(prepared: Mapping[str, Any], run_id: str
                                       ) -> tuple[bytes, dict[str, Any]]:
    originals = prepared.get("original_statuses")
    if not isinstance(originals, Mapping) or not isinstance(originals.get(run_id), Mapping):
        raise PermissionError(f"PREPARED omits original full status snapshot: {run_id}")
    record = originals[run_id]
    raw = base64.b64decode(str(record.get("raw_bytes_base64", "")), validate=True)
    projection = json.loads(raw)
    if (not isinstance(projection, dict)
            or _sha256_bytes(raw) != record.get("sha256")
            or projection != record.get("projection")
            or record.get("status") != projection.get("status")
            or record.get("canonical_key_json_sha256") != hashlib.sha256(
                str(projection.get("canonical_key_json", "")).encode("utf-8")).hexdigest()
            or record.get("run_id_full_sha256") != projection.get("run_id_full_sha256")
            or projection.get("run_id") != run_id):
        raise PermissionError(f"PREPARED original status byte/projection/fullhash mismatch: {run_id}")
    return raw, projection


def _load_migration_events(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise PermissionError("append-only v1.3 migration audit is missing")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(event, dict) or event.get("schema") != SCHEMA for event in events):
        raise PermissionError("migration ledger contains unknown/malformed records")
    prepared = [event for event in events if event.get("event") == "PREPARED"]
    committed = [event for event in events if event.get("event") == "COMMITTED"]
    if len(prepared) != 1 or len(committed) != 1:
        raise PermissionError("migration audit lacks unique PREPARED/COMMITTED records")
    return events, prepared[0], committed[0]


def _frozen_review_identity(artifact_root: Path) -> tuple[str, str]:
    """Read the current independent review identity from the frozen gate."""

    gate_path = artifact_root / "experiments/stage_F0_outputs/execution_isolation_gate.json"
    gate = _read_json(gate_path)
    review_path = gate.get("reviewer_report_path")
    review_sha = gate.get("reviewer_report_sha256")
    protection = gate.get("v13_protection")
    if (gate.get("reviewer_verdict") not in ("PASS", "PASS WITH FIXES")
            or not isinstance(review_path, str) or not review_path.strip()
            or not isinstance(review_sha, str) or len(review_sha) != 64
            or any(char not in "0123456789abcdef" for char in review_sha)
            or not isinstance(protection, Mapping)
            or protection.get("reviewer_report_path") != review_path
            or protection.get("reviewer_report_sha256") != review_sha):
        raise PermissionError("frozen gate has no matching current reviewer identity")
    relative = Path(review_path)
    if (relative.is_absolute() or ".." in relative.parts
            or not relative.as_posix().startswith("experiments/reports/")):
        raise PermissionError("frozen gate reviewer path is outside the report tree")
    report = artifact_root / relative
    if not report.is_file() or sha256_file(report) != review_sha:
        raise PermissionError("frozen gate reviewer report hash is missing or stale")
    return relative.as_posix(), review_sha


def _release_graph(release: Mapping[str, Any], *, runs_root: Path,
                   hold: Mapping[str, Any], hold_sha: str,
                   prepared: Mapping[str, Any], committed: Mapping[str, Any],
                   correction_sha: str, migration_sha: str,
                   queue_file_sha: str, queue_entries_sha: str,
                   artifact_root: Path,
                   frozen_review_identity: Optional[tuple[str, str]] = None) -> None:
    """Require an identity-complete, fresh-only reviewed hold release."""

    required_nonempty = (
        "release_id", "approved_by", "hold_id", "hold_sha256",
        "source_active_hold_sha256", "design_path", "design_sha256",
        "implementation_report_path", "implementation_report_sha256",
        "reviewer_report_path", "reviewer_report_sha256",
        "original_source_path", "original_source_sha256",
        "provenance_correction_path", "provenance_correction_sha256",
        "migration_audit_path", "migration_audit_sha256",
        "search_queue_path", "search_queue_file_sha256", "search_queue_entries_sha256",
        "completed_asset_inventory_sha256", "incident_asset_inventory_sha256",
        "fresh_only_scope",
    )
    if any(not isinstance(release.get(key), str) or not release[key].strip()
           for key in required_nonempty):
        raise PermissionError("release omits required nonempty provenance identity")
    if (release.get("schema") != RELEASE_SCHEMA
            or release.get("reviewer_verdict") not in ("PASS", "PASS WITH FIXES")
            or release.get("approved_by_role") != "principal"
            or release.get("fresh_only_scope") != FRESH_ONLY_SCOPE
            or release.get("preserved_held_run_ids") != list(PRESERVED_H49_RUN_IDS)
            or release.get("preserved_held_run_ids_sha256") != PRESERVED_H49_SHA256
            or release.get("authorization") != MIGRATED_STATUS_AUTHORIZATION):
        raise PermissionError("release scope, approval, H49 or forbidden authorizations mismatch")
    bindings = {
        "hold_id": hold.get("hold_id"),
        "hold_sha256": hold_sha,
        "source_active_hold_sha256": committed.get("active_hold_sha256"),
        "design_path": DESIGN_PATH,
        "design_sha256": DESIGN_SHA256,
        "implementation_report_path": IMPLEMENTATION_REPORT_PATH,
        "original_source_path": V4_PATH,
        "original_source_sha256": V4_SHA256,
        "provenance_correction_path": str(CORRECTION_PATH),
        "provenance_correction_sha256": correction_sha,
        "migration_audit_path": str(MIGRATION_PATH),
        "migration_audit_sha256": migration_sha,
        "search_queue_file_sha256": queue_file_sha,
        "search_queue_entries_sha256": queue_entries_sha,
        "completed_asset_inventory_sha256": committed.get("completed_asset_inventory_sha256"),
        "incident_asset_inventory_sha256": committed.get("incident_asset_inventory_sha256"),
    }
    for key, expected in bindings.items():
        if release.get(key) != expected:
            raise PermissionError(f"release provenance binding mismatch: {key}")
    if release.get("search_queue_path") != "experiments/stage_F0_outputs/queues/F0-B1-ML-search-queue.json":
        raise PermissionError("release search queue path is not the frozen F0-B1-ML queue")
    if release.get("design_sha256") != sha256_file(artifact_root / DESIGN_PATH):
        raise PermissionError("release design hash does not match current design")
    frozen_review_path, frozen_review_sha = (
        frozen_review_identity if frozen_review_identity is not None
        else _frozen_review_identity(artifact_root))
    if (release.get("reviewer_report_path") != frozen_review_path
            or release.get("reviewer_report_sha256") != frozen_review_sha):
        raise PermissionError("release reviewer identity differs from frozen execution gate")
    implementation_path = artifact_root / IMPLEMENTATION_REPORT_PATH
    if (release.get("implementation_report_sha256") != sha256_file(implementation_path)
            or release.get("reviewer_report_path") == ""):
        raise PermissionError("release implementation report hash mismatch")
    reviewer_path = artifact_root / str(release["reviewer_report_path"])
    reviewer_relative = Path(str(release["reviewer_report_path"]))
    if (reviewer_relative.is_absolute() or ".." in reviewer_relative.parts
            or not reviewer_relative.as_posix().startswith("experiments/reports/")
            or not reviewer_path.is_file()
            or sha256_file(reviewer_path) != release.get("reviewer_report_sha256")):
        raise PermissionError("release independent reviewer report hash mismatch")
    if release.get("active_hold_released") is not True:
        raise PermissionError("release record does not explicitly release the global hold")


def validate_h49_runtime(runs_root: Path = RUNS, *,
                         correction_sha256: Optional[str] = None,
                         migration_sha256: Optional[str] = None,
                         expected_asset_sha256: str = EXPECTED_ASSET_INVENTORY_SHA256,
                         expected_incident_asset_sha256: str = EXPECTED_INCIDENT_ASSET_INVENTORY_SHA256,
                         require_active_hold: bool = True,
                         artifact_root: Optional[Path] = None) -> dict[str, Any]:
    """Validate fixed H49/provenance/assets and the live v1.3 hold graph."""
    return _validate_h49_runtime_impl(
        runs_root, correction_sha256=correction_sha256,
        migration_sha256=migration_sha256,
        expected_asset_sha256=expected_asset_sha256,
        expected_incident_asset_sha256=expected_incident_asset_sha256,
        require_active_hold=require_active_hold, artifact_root=artifact_root,
        historical_only=False)


def _validate_h49_runtime_impl(runs_root: Path = RUNS, *,
                               correction_sha256: Optional[str] = None,
                               migration_sha256: Optional[str] = None,
                               expected_asset_sha256: str = EXPECTED_ASSET_INVENTORY_SHA256,
                               expected_incident_asset_sha256: str = EXPECTED_INCIDENT_ASSET_INVENTORY_SHA256,
                               require_active_hold: bool = True,
                               artifact_root: Optional[Path] = None,
                               historical_only: bool = False) -> dict[str, Any]:
    """Shared verifier; historical mode substitutes only the mutable hold head."""

    runs_root = Path(runs_root).resolve()
    artifact_root = Path(artifact_root or _artifact_root_for_runs(runs_root)).resolve()
    design_path = artifact_root / DESIGN_PATH
    implementation_report_path = artifact_root / IMPLEMENTATION_REPORT_PATH
    if (not design_path.is_file() or sha256_file(design_path) != DESIGN_SHA256
            or not implementation_report_path.is_file()):
        raise PermissionError("design or implementation report source is missing or stale")
    design_sha = sha256_file(design_path)
    implementation_report_sha = sha256_file(implementation_report_path)
    for name, value in (("correction_sha256", correction_sha256),
                        ("migration_sha256", migration_sha256),
                        ("expected_asset_sha256", expected_asset_sha256),
                        ("expected_incident_asset_sha256", expected_incident_asset_sha256)):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise PermissionError(f"required frozen hash is missing or malformed: {name}")
    if tuple(sorted(set(PRESERVED_V12_H47_RUN_IDS) | set(TARGET_IDS))) != tuple(PRESERVED_H49_RUN_IDS):
        raise PermissionError("independent fixed H49 is not H47 plus both v1.3 IDs")
    if held_run_ids_sha256(PRESERVED_V12_H47_RUN_IDS) != PRESERVED_H47_SHA256:
        raise PermissionError("independent H47 hash mismatch")
    if held_run_ids_sha256(PRESERVED_H49_RUN_IDS) != PRESERVED_H49_SHA256:
        raise PermissionError("independent H49 hash mismatch")

    provenance = validate_provenance(runs_root)
    corr_path = _correction_path(runs_root)
    corr_sha = sha256_file(corr_path) if corr_path.is_file() else ""
    if not corr_sha or corr_sha != correction_sha256:
        raise PermissionError("required provenance correction hash differs from frozen binding")
    correction = _read_json(corr_path)
    expected_correction = build_provenance_correction(runs_root)
    comparable = dict(correction)
    comparable.pop("created_at_utc", None)
    expected_correction.pop("created_at_utc", None)
    if comparable != expected_correction:
        raise PermissionError("provenance correction does not bind exact original audit and source fields")

    migration_path = runs_root / MIGRATION_PATH
    migration_sha = sha256_file(migration_path) if migration_path.is_file() else ""
    if not migration_sha or migration_sha != migration_sha256:
        raise PermissionError("required migration audit hash differs from frozen binding")
    events, prep, done = _load_migration_events(migration_path)
    transitions = [event for event in events
                   if event.get("event") == "STATUS_TRANSITION_DURABLE"]
    if (events[0] is not prep or events[-1] is not done
            or len(transitions) != len(TARGET_IDS)
            or {event.get("run_id") for event in transitions} != set(TARGET_IDS)):
        raise PermissionError("migration transaction event order or target transition set is incomplete")
    for event in (prep, done):
        ids = event.get("held_run_ids")
        if (not isinstance(ids, list) or len(ids) != len(set(ids))
                or tuple(sorted(ids)) != tuple(PRESERVED_H49_RUN_IDS)
                or event.get("held_run_ids_sha256") != PRESERVED_H49_SHA256):
            raise PermissionError("migration audit does not freeze the exact independent H49")
    if (done.get("prepared_record_sha256") != _sha256_bytes(_compact_json(prep))
            or prep.get("schema") != SCHEMA
            or prep.get("event") != "PREPARED"
            or prep.get("administrative_only") is not True
            or prep.get("provenance_correction_sha256") != corr_sha
            or done.get("provenance_correction_sha256") != corr_sha
            or done.get("authorization") != MIGRATED_STATUS_AUTHORIZATION):
        raise PermissionError("PREPARED/COMMITTED full provenance or authorization mismatch")
    if (set(done.get("statuses_after", {})) != set(TARGET_IDS)
            or set(done.get("status_projections_after", {})) != set(TARGET_IDS)
            or done.get("held_run_ids") != list(PRESERVED_H49_RUN_IDS)
            or done.get("authorization") != MIGRATED_STATUS_AUTHORIZATION):
        raise PermissionError("COMMITTED full status/H49 projection is incomplete")
    transition_by_id = {str(event.get("run_id")): event for event in transitions}
    for run_id in TARGET_IDS:
        transition = transition_by_id[run_id]
        if (transition.get("new_status") != "RECOVERY_BLOCKED"
                or transition.get("status_file_fsynced") is not True
                or transition.get("status_directory_fsynced") is not True
                or transition.get("new_status_sha256") != done["statuses_after"].get(run_id)):
            raise PermissionError(f"durable transition receipt lacks fsync/hash proof: {run_id}")
    original_hold = prep.get("original_active_hold")
    if not isinstance(original_hold, Mapping):
        raise PermissionError("PREPARED omits original active hold raw snapshot")
    original_hold_raw = base64.b64decode(str(original_hold.get("raw_bytes_base64", "")), validate=True)
    original_hold_path = runs_root / str(original_hold.get("path", ""))
    if (original_hold.get("sha256") != ORIGINAL_HOLD_SHA256
            or prep.get("original_active_hold_sha256") != ORIGINAL_HOLD_SHA256
            or _sha256_bytes(original_hold_raw) != ORIGINAL_HOLD_SHA256
            or json.loads(original_hold_raw) != original_hold.get("projection")
            or not original_hold_path.is_file()):
        raise PermissionError("PREPARED original hold bytes/projection are not preserved")

    rows = _load_search_queue(runs_root)
    queue_path = _search_queue_path(runs_root)
    queue_file_sha = sha256_file(queue_path)
    queue_entries_sha = _sha256_bytes(_compact_json(rows))
    if (prep.get("search_queue_file_sha256") != queue_file_sha
            or prep.get("search_queue_entries_sha256") != queue_entries_sha
            or done.get("search_queue_file_sha256") != queue_file_sha
            or done.get("search_queue_entries_sha256") != queue_entries_sha):
        raise PermissionError("frozen search queue source drifted")
    statuses, counts = _queue_statuses(runs_root, rows)
    for run_id in EXPECTED_COMPLETED_RUN_IDS:
        if statuses.get(run_id) != "VALIDATION_COMPLETE":
            raise PermissionError(f"originally completed trajectory status changed: {run_id}")
    for run_id in PRESERVED_H49_RUN_IDS:
        if statuses.get(run_id) != "RECOVERY_BLOCKED":
            raise PermissionError(f"H49 status is not RECOVERY_BLOCKED: {run_id}")
    allowed_live_statuses = {"UNCLAIMED", "CLAIMED", "RUNNING", "INTERRUPTED",
                             "FAILED", "UNKNOWN", "VALIDATION_COMPLETE", "RECOVERY_BLOCKED"}
    frozen_ids = set(EXPECTED_COMPLETED_RUN_IDS) | set(PRESERVED_H49_RUN_IDS)
    for run_id, status in statuses.items():
        if run_id not in frozen_ids and status not in allowed_live_statuses:
            raise PermissionError(f"fresh queue status is not a known protocol state: {run_id}={status}")

    for run_id in TARGET_IDS:
        original_raw, original_projection = _validate_prepared_original_status(prep, run_id)
        _, current, current_sha = _status_snapshot(runs_root, run_id)
        expected_current = _expected_migrated_status(prep, run_id)
        if current != expected_current:
            raise PermissionError(f"full migrated status projection differs from PREPARED source: {run_id}")
        status_hashes = done.get("statuses_after")
        status_projections = done.get("status_projections_after")
        if (not isinstance(status_hashes, Mapping) or status_hashes.get(run_id) != current_sha
                or not isinstance(status_projections, Mapping)
                or status_projections.get(run_id) != current):
            raise PermissionError(f"COMMITTED full status projection/hash mismatch: {run_id}")
        if (current.get("canonical_key_json") != original_projection.get("canonical_key_json")
                or current.get("run_id_full_sha256") != original_projection.get("run_id_full_sha256")
                or current.get("recovery_count") != original_projection.get("recovery_count")
                or current.get("resume_checkpoint") != original_projection.get("resume_checkpoint")):
            raise PermissionError(f"migration changed canonical identity or recovery state: {run_id}")

    completed_items, completed_sha = _asset_inventory(runs_root, rows)
    if (completed_sha != expected_asset_sha256
            or completed_sha != done.get("completed_asset_inventory_sha256")
            or completed_items != prep.get("completed_asset_inventory")
            or len(completed_items) != 573
            or done.get("completed_asset_file_count") != len(completed_items)):
        raise PermissionError("52 completed trajectory assets differ from frozen byte inventory")
    incident_items, incident_sha = _incident_asset_inventory(runs_root)
    if (incident_sha != expected_incident_asset_sha256
            or incident_sha != prep.get("incident_asset_inventory_sha256")
            or incident_sha != done.get("incident_asset_inventory_sha256")
            or incident_items != prep.get("incident_asset_inventory")
            or done.get("incident_asset_file_count") != len(incident_items)):
        raise PermissionError("incident trees/logs/metrics/checkpoint inventory drifted")

    hold_path = runs_root / "execution_control" / "global_hold.json"
    hold = _read_json(hold_path)
    hold_sha = sha256_file(hold_path)
    if not isinstance(hold.get("active"), bool):
        raise PermissionError("global hold active flag is malformed")
    if require_active_hold and hold.get("active") is not True:
        raise PermissionError("global execution hold is not active")
    if (not historical_only and (hold.get("preserved_held_run_ids") != list(PRESERVED_H49_RUN_IDS)
            or hold.get("preserved_held_run_ids_sha256") != PRESERVED_H49_SHA256)):
        raise PermissionError("global hold does not preserve exact H49")
    if historical_only:
        _verify_historical_h49_hold_and_release(
            runs_root, hold=hold, hold_sha=hold_sha, prepared=prep, committed=done,
            artifact_root=artifact_root)
    elif hold.get("active") is True:
        if hold_sha != done.get("active_hold_sha256"):
            raise PermissionError("COMMITTED record does not bind current active hold hash")
        events_in_hold = hold.get("events", [])
        matches = [event for event in events_in_hold if isinstance(event, Mapping)
                   and event.get("migration_id") == prep.get("migration_id")]
        if len(matches) != 1:
            raise PermissionError("active hold event does not bind this migration provenance")
        _verify_extended_hold_projection(hold, prep, str(prep["migration_id"]), corr_sha)
    else:
        release_path = runs_root / "execution_control" / "global_hold_release.json"
        if not release_path.is_file():
            raise PermissionError("inactive global hold has no reviewed release artifact")
        release = _read_json(release_path)
        _release_graph(release, runs_root=runs_root, hold=hold, hold_sha=hold_sha,
                       prepared=prep, committed=done, correction_sha=corr_sha,
                       migration_sha=migration_sha, queue_file_sha=queue_file_sha,
                       queue_entries_sha=queue_entries_sha, artifact_root=artifact_root)

    # Initial counts are committed as audit facts; live counts may advance.
    if (prep.get("search_queue_counts_before") != {
            "VALIDATION_COMPLETE": 52, "RECOVERY_BLOCKED": 47, "UNCLAIMED": 187,
            "RUNNING": 1, "INTERRUPTED": 1}
            or done.get("search_queue_counts_after") != {
                "VALIDATION_COMPLETE": 52, "RECOVERY_BLOCKED": 49, "UNCLAIMED": 187}):
        raise PermissionError("migration's immutable before/after count receipt is malformed")
    return {"held_run_ids": list(PRESERVED_H49_RUN_IDS),
            "held_run_ids_sha256": PRESERVED_H49_SHA256,
            "counts": counts, "active_hold_sha256": hold_sha,
            "correction_sha256": corr_sha, "migration_sha256": migration_sha,
            "asset_file_count": len(completed_items), "asset_inventory_sha256": completed_sha,
            "incident_asset_file_count": len(incident_items),
            "incident_asset_inventory_sha256": incident_sha,
            "search_queue_file_sha256": queue_file_sha,
            "search_queue_entries_sha256": queue_entries_sha,
            "v4_sha256": V4_SHA256, "peer_receipt_sha256": provenance["peer_receipt_sha256"],
            "design_sha256": design_sha,
            "implementation_report_sha256": implementation_report_sha}


def validate_h49_historical_graph(runs_root: Path = RUNS, *,
                                  correction_sha256: Optional[str] = None,
                                  migration_sha256: Optional[str] = None,
                                  expected_asset_sha256: str = EXPECTED_ASSET_INVENTORY_SHA256,
                                  expected_incident_asset_sha256: str = EXPECTED_INCIDENT_ASSET_INVENTORY_SHA256,
                                  artifact_root: Optional[Path] = None) -> dict[str, Any]:
    """Validate the frozen v1.3 H49 history without treating it as launch proof.

    This public historical API is intentionally distinct from
    ``validate_h49_runtime``. It checks the old PREPARED/COMMITTED records,
    source correction, H49 statuses, protected assets, old migration hold
    projection and v1.3 release/review graph. It does not authorize a current
    launch and does not require the mutable top-level hold to still have its
    v1.3 bytes after later incidents.
    """
    runs_root = Path(runs_root).resolve()
    artifact_root = Path(artifact_root or _artifact_root_for_runs(runs_root)).resolve()
    # The normal validator already performs every historical source,
    # correction, transaction, status, asset, queue and authorization check.
    # Its only live-state dependency is the final hold/release projection.
    # Validate that legacy graph from the frozen ledger and release instead.
    result = _validate_h49_historical_core(
        runs_root, correction_sha256=correction_sha256,
        migration_sha256=migration_sha256,
        expected_asset_sha256=expected_asset_sha256,
        expected_incident_asset_sha256=expected_incident_asset_sha256,
        artifact_root=artifact_root)
    return result


def _validate_h49_historical_core(runs_root: Path, *, correction_sha256: Optional[str],
                                  migration_sha256: Optional[str],
                                  expected_asset_sha256: str,
                                  expected_incident_asset_sha256: str,
                                  artifact_root: Path) -> dict[str, Any]:
    """Historical v1.3 verification extracted from the frozen H49 runtime."""
    # Use the implementation's exact fixed provenance checks, while
    # substituting only the current mutable hold/release check with verification
    # against the embedded PREPARED and COMMITTED historical projections.
    # The common body below deliberately duplicates no acceptance decision:
    # it first validates the fixed source hashes and transaction evidence by
    # invoking a shadow-free helper that does not touch current launch state.
    return _validate_h49_runtime_impl(
        runs_root, correction_sha256=correction_sha256,
        migration_sha256=migration_sha256,
        expected_asset_sha256=expected_asset_sha256,
        expected_incident_asset_sha256=expected_incident_asset_sha256,
        artifact_root=artifact_root, require_active_hold=False,
        historical_only=True)


def _verify_historical_h49_hold_and_release(runs_root: Path, *, hold: Mapping[str, Any],
                                            hold_sha: str, prepared: Mapping[str, Any],
                                            committed: Mapping[str, Any],
                                            artifact_root: Path) -> None:
    """Rebuild and check the v1.3 hold/release heads from immutable evidence."""
    original_record = prepared.get("original_active_hold")
    if not isinstance(original_record, Mapping):
        raise PermissionError("historical v1.3 PREPARED omits original hold bytes")
    raw = base64.b64decode(str(original_record.get("raw_bytes_base64", "")), validate=True)
    if _sha256_bytes(raw) != ORIGINAL_HOLD_SHA256:
        raise PermissionError("historical v1.3 original active hold bytes differ")
    original = json.loads(raw)
    migration_id = str(prepared.get("migration_id", ""))
    event_matches = [event for event in hold.get("events", [])
                     if isinstance(event, Mapping)
                     and event.get("event") == "V13_H49_ABNORMAL_WORKER_STATUS_MIGRATION"
                     and event.get("migration_id") == migration_id]
    if len(event_matches) != 1:
        raise PermissionError("historical v1.3 migration event is absent or duplicated")
    event = dict(event_matches[0])
    if (event.get("preserved_held_run_ids") != list(PRESERVED_H49_RUN_IDS)
            or event.get("preserved_held_run_ids_sha256") != PRESERVED_H49_SHA256
            or event.get("automatic_retry_authorized") is not False
            or event.get("recovery_authorized") is not False
            or committed.get("active_hold_sha256")
            != "a1da4f3d0aa20c7c867b9b01b7c8c509db21408c7cedfed5b924040dfba48b66"):
        raise PermissionError("historical v1.3 committed H49 hold evidence differs")
    report_path = runs_root / "execution_control" / "postmigration_v13_admin_runner_report.json"
    report_sha = sha256_file(report_path) if report_path.is_file() else ""
    if report_sha != "a3bbd3d0bf238462882a3a5b29e1afb8224639478837c8993ca68726cae76af3":
        raise PermissionError("historical v1.3 postmigration runner evidence differs")
    report = _read_json(report_path)
    migration_report = report.get("migration", {})
    if (not isinstance(migration_report, Mapping)
            or migration_report.get("hold_sha256") != committed.get("active_hold_sha256")):
        raise PermissionError("historical v1.3 active hold hash lacks independent runner confirmation")
    release_path = runs_root / "execution_control" / "global_hold_release.json"
    frozen_release = _v14_frozen_source_bytes(
        runs_root, "execution_control/global_hold_release.json")
    if frozen_release is not None:
        release_bytes, release_sha = frozen_release
    else:
        release_bytes = release_path.read_bytes() if release_path.is_file() else b""
        release_sha = _sha256_bytes(release_bytes) if release_bytes else ""
    if release_sha != "239b6749e5d8694b00bd12dc28a91f66b7ea92ac382de6f49ab4729fcbfd4eed":
        raise PermissionError("historical v1.3 principal release bytes differ")
    release = json.loads(release_bytes)
    prelaunch_path = runs_root / "execution_control" / "prelaunch_audit_v13_fresh_search_v2.json"
    if (not prelaunch_path.is_file()
            or sha256_file(prelaunch_path) != "1aecf560bc00e1bb061b11741ec0a5854584525d0d6bc47f6784d3e2c5dcba22"):
        raise PermissionError("historical v1.3 prelaunch source differs")
    prelaunch = _read_json(prelaunch_path)
    prelaunch_execution = prelaunch.get("identity", {})
    if not isinstance(prelaunch_execution, Mapping):
        raise PermissionError("historical v1.3 prelaunch execution projection is malformed")
    released_projection_sha = str(prelaunch_execution.get("planned_released_hold_sha256", ""))
    if (released_projection_sha != "c16f895623a0429314a8cc3fa878a0a1ab8a71b8e7e827d92d9ce754cc698c9c"
            or release.get("hold_sha256") != released_projection_sha
            or release.get("source_active_hold_sha256")
            != committed.get("active_hold_sha256")):
        raise PermissionError("historical v1.3 release does not bind frozen H49 hold heads")
    # The original status/migration projection is rebuilt from PREPARED and
    # the exactly-once H49 event. The released hold bytes were later replaced
    # by the next incident; their frozen digest and complete release graph are
    # retained in the v1.3 prelaunch/release sources.
    historical_released = dict(original)
    historical_released.update({
        "active": False,
        "events": list(original.get("events", [])) + [event],
        "preserved_held_run_ids": list(PRESERVED_H49_RUN_IDS),
        "preserved_held_run_ids_sha256": PRESERVED_H49_SHA256,
        "administrative_migration_id": migration_id,
    })
    # _release_graph checks the full release identity and reviewer graph. The
    # historical released projection hash is independently pinned above.
    rows = _load_search_queue(runs_root)
    queue_file_sha = sha256_file(_search_queue_path(runs_root))
    queue_entries_sha = _sha256_bytes(_compact_json(rows))
    _release_graph(
        release, runs_root=runs_root, hold=historical_released,
        hold_sha=released_projection_sha, prepared=prepared,
        committed=committed,
        correction_sha=str(prepared.get("provenance_correction_sha256", "")),
        migration_sha=sha256_file(runs_root / MIGRATION_PATH),
        queue_file_sha=queue_file_sha, queue_entries_sha=queue_entries_sha,
        artifact_root=artifact_root,
        # The v1.3 release bytes and reviewer artifact are independently
        # hash-pinned above. Do not consult the mutable current v1.4 gate.
        frozen_review_identity=(str(release.get("reviewer_report_path", "")),
                                str(release.get("reviewer_report_sha256", ""))))


def _v14_frozen_source_bytes(runs_root: Path, source_path: str
                             ) -> Optional[tuple[bytes, str]]:
    """Read a v1.3 pointer from the immutable v1.4 snapshot after pointer swap."""
    snapshot_path = runs_root / "execution_control" / "postabnormal_v14_source_snapshot.json"
    if not snapshot_path.is_file():
        return None
    snapshot = _read_json(snapshot_path)
    claimed = snapshot.get("snapshot_sha256")
    unsigned = dict(snapshot)
    unsigned.pop("snapshot_sha256", None)
    if claimed != _sha256_bytes(_compact_json(unsigned)):
        raise PermissionError("v1.4 frozen source snapshot digest is invalid")
    records = snapshot.get("source_files")
    if not isinstance(records, list):
        raise PermissionError("v1.4 source snapshot omits raw pointer records")
    record = next((row for row in records if isinstance(row, Mapping)
                   and row.get("path") == source_path), None)
    if record is None:
        return None
    raw = base64.b64decode(str(record.get("raw_base64", "")), validate=True)
    digest = _sha256_bytes(raw)
    if digest != record.get("sha256") or len(raw) != record.get("size"):
        raise PermissionError("v1.4 frozen historical pointer bytes are invalid")
    return raw, digest


def _make_prepared(runs_root: Path, rows: list[dict[str, Any]],
                   correction_sha: str, migration_id: str,
                   asset_inventory: list[dict[str, object]], asset_sha: str,
                   incident_inventory: list[dict[str, object]],
                   incident_sha: str) -> dict[str, Any]:
    statuses, counts = _queue_statuses(runs_root, rows)
    _validate_initial_counts(runs_root, rows)
    original_statuses: dict[str, dict[str, object]] = {}
    for run_id in TARGET_IDS:
        path = _status_path(runs_root, run_id)
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get("status") != statuses[run_id]:
            raise PermissionError(f"target changed while preparing migration: {run_id}")
        original_statuses[run_id] = {
            "status": payload["status"], "sha256": _sha256_bytes(raw),
            "raw_bytes_base64": base64.b64encode(raw).decode("ascii"),
            "projection": payload,
            "canonical_key_json_sha256": hashlib.sha256(
                str(payload.get("canonical_key_json", "")).encode("utf-8")).hexdigest(),
            "run_id_full_sha256": payload.get("run_id_full_sha256"),
            "attempt_id": payload.get("attempt_id"),
            "recovery_count": payload.get("recovery_count"),
            "resume_checkpoint": payload.get("resume_checkpoint"),
        }
    hold_path, hold, hold_sha = _active_hold(runs_root)
    hold_raw = hold_path.read_bytes()
    if hold_sha != ORIGINAL_HOLD_SHA256:
        raise PermissionError("original active hold bytes differ from approved v1.3 source")
    if _sha256_bytes(hold_raw) != hold_sha or json.loads(hold_raw) != hold:
        raise PermissionError("original active hold snapshot changed while preparing")
    provenance = validate_provenance(runs_root)
    if (len(PRESERVED_V12_H47_RUN_IDS) != 47 or len(PRESERVED_H49_RUN_IDS) != 49
            or set(TARGET_IDS) & set(PRESERVED_V12_H47_RUN_IDS)):
        raise PermissionError("fixed H47/H49 independent source invariant failed")
    return {
        "schema": SCHEMA, "event": "PREPARED", "prepared_at_utc": _now(),
        "migration_id": migration_id, "administrative_only": True,
        "held_run_ids": list(PRESERVED_H49_RUN_IDS),
        "held_run_ids_sha256": PRESERVED_H49_SHA256,
        "v12_h47_run_ids_sha256": PRESERVED_H47_SHA256,
        "source_v4": {"path": V4_PATH, "sha256": V4_SHA256},
        "provenance_correction_path": str(CORRECTION_PATH),
        "provenance_correction_sha256": correction_sha,
        "original_active_hold_sha256": hold_sha,
        "original_active_hold": {
            "path": str(hold_path.relative_to(runs_root)),
            "sha256": hold_sha,
            "raw_bytes_base64": base64.b64encode(hold_raw).decode("ascii"),
            "projection": hold,
        },
        "original_statuses": original_statuses,
        "original_attempt_manifests_and_checkpoints": {
            "peer_manifest_sha256": provenance["peer_manifest_sha256"],
            "peer_checkpoint_sha256": provenance["peer_checkpoint_sha256"],
            "sigsegv_receipt": None,
        },
        "search_queue_counts_before": counts,
        "search_queue_file_sha256": sha256_file(_search_queue_path(runs_root)),
        "search_queue_entries_sha256": _sha256_bytes(_compact_json(rows)),
        "completed_asset_file_count": len(asset_inventory),
        "completed_asset_inventory_sha256": asset_sha,
        "completed_run_ids": list(EXPECTED_COMPLETED_RUN_IDS),
        "completed_asset_inventory": asset_inventory,
        "incident_asset_file_count": len(incident_inventory),
        "incident_asset_inventory_sha256": incident_sha,
        "incident_asset_inventory": incident_inventory,
        "authorization": {"resume": False, "retry": False, "new_attempt": False,
                          "checkpoint_load": False, "formal": False, "test": False},
        "notes": "RECOVERY_BLOCKED changes only outer queue eligibility; original receipt/manifest remain immutable.",
    }


def _migrated_status(original: Mapping[str, Any], run_id: str,
                     migration_id: str) -> dict[str, Any]:
    value = dict(original)
    previous_status = str(original.get("status", ""))
    value["status"] = "RECOVERY_BLOCKED"
    value["previous_status"] = previous_status
    value["blocking_status"] = "V13_ABNORMAL_WORKER_EXIT_HOLD"
    value["v13_migration_id"] = migration_id
    value["v13_original_status_sha256"] = _sha256_bytes(_compact_json(original))
    value["v13_original_status_raw_sha256"] = ""
    value["v13_preserved_receipt_status"] = (
        "MISSING" if run_id == TARGET_SIGSEGV else "INTERRUPTED")
    value["v13_recovery_authorized"] = False
    value["v13_automatic_retry_authorized"] = False
    value["reason"] = "v1.3 administrative hold migration; no resume or retry"
    return value


def _expected_migrated_status(prepared: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    """Rebuild the only permitted full target projection from PREPARED bytes."""

    original_raw, original = _validate_prepared_original_status(prepared, run_id)
    expected = _migrated_status(original, run_id, str(prepared.get("migration_id", "")))
    original_sha = _sha256_bytes(original_raw)
    expected["v13_original_status_raw_sha256"] = original_sha
    expected["v13_original_status_sha256"] = original_sha
    return expected


def _verify_extended_hold_projection(current: Mapping[str, Any],
                                     prepared: Mapping[str, Any],
                                     migration_id: str,
                                     correction_sha: str) -> None:
    """Require exactly one append to the preserved original hold projection."""

    original_record = prepared.get("original_active_hold")
    if not isinstance(original_record, Mapping):
        raise PermissionError("PREPARED omits original hold projection")
    raw = base64.b64decode(str(original_record.get("raw_bytes_base64", "")), validate=True)
    original = json.loads(raw)
    if not isinstance(original, dict) or _sha256_bytes(raw) != ORIGINAL_HOLD_SHA256:
        raise PermissionError("PREPARED original hold bytes are invalid")
    original_events = original.get("events", [])
    current_events = current.get("events", [])
    if (not isinstance(original_events, list) or not isinstance(current_events, list)
            or len(current_events) != len(original_events) + 1
            or current_events[:len(original_events)] != original_events):
        raise PermissionError("extended hold does not append exactly one migration event")
    event = current_events[-1]
    expected_event = {
        "event": "V13_H49_ABNORMAL_WORKER_STATUS_MIGRATION",
        "migration_id": migration_id,
        "source_v4_path": V4_PATH,
        "source_v4_sha256": V4_SHA256,
        "provenance_correction_path": str(CORRECTION_PATH),
        "provenance_correction_sha256": correction_sha,
        "migration_audit_path": str(MIGRATION_PATH),
        "prepared_record_sha256": _sha256_bytes(_compact_json(prepared)),
        "preserved_held_run_ids": list(PRESERVED_H49_RUN_IDS),
        "preserved_held_run_ids_sha256": PRESERVED_H49_SHA256,
        "new_statuses": {run_id: "RECOVERY_BLOCKED" for run_id in TARGET_IDS},
        "automatic_retry_authorized": False,
        "recovery_authorized": False,
        "completed_asset_inventory_sha256": prepared.get("completed_asset_inventory_sha256"),
    }
    if (not isinstance(event, Mapping)
            or any(event.get(key) != value for key, value in expected_event.items())
            or not isinstance(event.get("timestamp_utc"), str)
            or not event.get("timestamp_utc")):
        raise PermissionError("extended hold migration event differs from frozen source graph")
    allowed_updated = {"active", "events", "preserved_held_run_ids",
                       "preserved_held_run_ids_sha256", "administrative_migration_id",
                       "updated_at_utc"}
    expected_current = dict(original)
    expected_current.update({
        "active": True,
        "events": current_events,
        "preserved_held_run_ids": list(PRESERVED_H49_RUN_IDS),
        "preserved_held_run_ids_sha256": PRESERVED_H49_SHA256,
        "administrative_migration_id": migration_id,
    })
    if not isinstance(current.get("updated_at_utc"), str) or not current.get("updated_at_utc"):
        raise PermissionError("extended hold omits its update timestamp")
    expected_current["updated_at_utc"] = current["updated_at_utc"]
    if set(current) != set(expected_current) or dict(current) != expected_current:
        raise PermissionError("extended hold changed fields outside the approved H49 append")


def apply_migration(*, runs_root: Path = RUNS,
                    fault: Optional[str] = None,
                    correction_sha256: Optional[str] = None) -> dict[str, Any]:
    """Apply one exact H49 status migration under serialized execution locks."""

    runs_root = Path(runs_root).resolve()
    if tuple(sorted(PRESERVED_H49_RUN_IDS)) != tuple(PRESERVED_H49_RUN_IDS):
        raise PermissionError("frozen H49 list is not canonical sorted form")
    if held_run_ids_sha256(PRESERVED_H49_RUN_IDS) != PRESERVED_H49_SHA256:
        raise PermissionError("frozen H49 constant fails its approved digest")
    corr_path, corr_sha = ensure_provenance_correction(runs_root)
    if correction_sha256 is not None and corr_sha != correction_sha256:
        raise PermissionError("provenance correction hash differs from caller freeze")
    rows = _load_search_queue(runs_root)
    asset_inventory, asset_sha = _asset_inventory(runs_root, rows)
    if len(asset_inventory) != 573 or asset_sha != EXPECTED_ASSET_INVENTORY_SHA256:
        raise PermissionError("pre-migration 52-key completed asset baseline is not exact")
    incident_inventory, incident_sha = _incident_asset_inventory(runs_root)
    if (EXPECTED_INCIDENT_ASSET_INVENTORY_SHA256
            and incident_sha != EXPECTED_INCIDENT_ASSET_INVENTORY_SHA256):
        raise PermissionError("pre-migration incident-tree/evidence inventory is not exact")
    correction_rel = corr_path.relative_to(runs_root).as_posix()
    migration_path = runs_root / MIGRATION_PATH
    migration_id = "f0-v13-h49-admin-" + uuid.uuid4().hex
    _ensure_locks_free(runs_root)
    with _acquire_migration_locks(runs_root):
        # Once serialized, re-read every input that can change concurrently.
        rows = _load_search_queue(runs_root)
        asset_after_lock, asset_sha_after_lock = _asset_inventory(runs_root, rows)
        if asset_sha_after_lock != asset_sha or asset_after_lock != asset_inventory:
            raise PermissionError("completed asset inventory changed while acquiring locks")
        incident_after_lock, incident_sha_after_lock = _incident_asset_inventory(runs_root)
        if incident_sha_after_lock != incident_sha or incident_after_lock != incident_inventory:
            raise PermissionError("incident asset inventory changed while acquiring locks")
        validate_provenance(runs_root)
        hold_path, hold, hold_sha = _active_hold(runs_root)
        existing_events = []
        if migration_path.exists():
            for line in migration_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                event_value = json.loads(line)
                if not isinstance(event_value, dict) or event_value.get("schema") != SCHEMA:
                    raise PermissionError("migration ledger contains unknown/malformed evidence")
                existing_events.append(event_value)
        prepared_records = [event for event in existing_events if event.get("event") == "PREPARED"]
        committed_records = [event for event in existing_events if event.get("event") == "COMMITTED"]
        if len(prepared_records) > 1 or len(committed_records) > 1:
            raise PermissionError("migration ledger contains duplicate PREPARED/COMMITTED events")
        if committed_records:
            validated = validate_h49_runtime(
                runs_root, correction_sha256=corr_sha,
                migration_sha256=sha256_file(migration_path),
                expected_asset_sha256=asset_sha,
                expected_incident_asset_sha256=incident_sha,
                require_active_hold=False)
            return {"migration_id": prepared_records[0]["migration_id"],
                    "counts": validated["counts"],
                    "hold_sha256": validated["active_hold_sha256"],
                    "correction_sha256": corr_sha,
                    "migration_sha256": validated["migration_sha256"],
                    "asset_file_count": validated["asset_file_count"],
                    "asset_inventory_sha256": validated["asset_inventory_sha256"],
                    "incident_asset_file_count": validated["incident_asset_file_count"],
                    "incident_asset_inventory_sha256": validated[
                        "incident_asset_inventory_sha256"],
                    "idempotent": True}
        if not prepared_records:
            if hold_sha != ORIGINAL_HOLD_SHA256:
                raise PermissionError("original active hold bytes differ from approved v1.3 source")
            prepared = _make_prepared(runs_root, rows, corr_sha, migration_id,
                                      asset_inventory, asset_sha,
                                      incident_inventory, incident_sha)
            # Durable PREPARED evidence is the commit barrier before any status write.
            _append_jsonl(migration_path, prepared)
        else:
            prepared = prepared_records[0]
            migration_id = str(prepared.get("migration_id", ""))
            if (prepared.get("held_run_ids") != list(PRESERVED_H49_RUN_IDS)
                    or prepared.get("held_run_ids_sha256") != PRESERVED_H49_SHA256
                    or prepared.get("provenance_correction_sha256") != corr_sha
                    or prepared.get("completed_asset_inventory_sha256") != asset_sha
                    or prepared.get("completed_asset_inventory") != asset_inventory
                    or prepared.get("incident_asset_inventory_sha256") != incident_sha
                    or prepared.get("incident_asset_inventory") != incident_inventory
                    or prepared.get("original_active_hold_sha256") != ORIGINAL_HOLD_SHA256):
                raise PermissionError("existing PREPARED migration differs from frozen sources")
            original_hold = prepared.get("original_active_hold")
            if not isinstance(original_hold, Mapping):
                raise PermissionError("existing PREPARED migration omits raw original hold bytes")
            original_hold_raw = base64.b64decode(
                str(original_hold.get("raw_bytes_base64", "")), validate=True)
            if (_sha256_bytes(original_hold_raw) != ORIGINAL_HOLD_SHA256
                    or json.loads(original_hold_raw) != original_hold.get("projection")
                    or original_hold.get("sha256") != ORIGINAL_HOLD_SHA256):
                raise PermissionError("existing PREPARED original hold snapshot is corrupt")
            originals = prepared.get("original_statuses")
            if not isinstance(originals, Mapping) or set(originals) != set(TARGET_IDS):
                raise PermissionError("existing PREPARED migration lacks both original statuses")
            for run_id in TARGET_IDS:
                _validate_prepared_original_status(prepared, run_id)
        prepared_sha = _sha256_bytes(_compact_json(prepared))
        if fault == "after_prepared" and not existing_events:
            raise OSError("injected failure after PREPARED; active hold remains unchanged")
        for index, run_id in enumerate(TARGET_IDS):
            path = _status_path(runs_root, run_id)
            raw_before = path.read_bytes()
            record = prepared["original_statuses"][run_id]
            current = json.loads(raw_before)
            if _sha256_bytes(raw_before) == record.get("sha256"):
                original = json.loads(base64.b64decode(
                    str(record["raw_bytes_base64"]), validate=True))
                if current != original:
                    raise PermissionError(f"original status semantics drifted: {run_id}")
                updated = _migrated_status(original, run_id, migration_id)
                updated["v13_original_status_raw_sha256"] = record["sha256"]
                updated["v13_original_status_sha256"] = record["sha256"]
                fault_point = ("before_rename" if fault == f"status{index}_before_rename" else
                               "before_dir_fsync" if fault == f"status{index}_before_dir_fsync" else None)
                _atomic_json(path, updated, fault=fault_point)
                _append_jsonl(migration_path, {
                    "schema": SCHEMA, "event": "STATUS_TRANSITION_DURABLE",
                    "timestamp_utc": _now(), "migration_id": migration_id,
                    "run_id": run_id, "old_status": original["status"],
                    "old_status_sha256": record["sha256"],
                    "new_status": "RECOVERY_BLOCKED",
                    "new_status_sha256": sha256_file(path),
                    "attempt_id": updated.get("attempt_id"),
                    "recovery_count": updated.get("recovery_count"),
                    "resume_checkpoint": updated.get("resume_checkpoint"),
                    "status_file_fsynced": True, "status_directory_fsynced": True,
                })
            elif current == _expected_migrated_status(prepared, run_id):
                # Reconcile a status rename that became durable before its audit append.
                _fsync_existing_file_and_parent(path)
                if not any(event.get("event") == "STATUS_TRANSITION_DURABLE"
                           and event.get("migration_id") == migration_id
                           and event.get("run_id") == run_id for event in existing_events):
                    _append_jsonl(migration_path, {
                        "schema": SCHEMA, "event": "STATUS_TRANSITION_DURABLE",
                        "timestamp_utc": _now(), "migration_id": migration_id,
                        "run_id": run_id, "old_status": record["status"],
                        "old_status_sha256": record["sha256"],
                        "new_status": "RECOVERY_BLOCKED",
                        "new_status_sha256": sha256_file(path),
                        "attempt_id": current.get("attempt_id"),
                        "recovery_count": current.get("recovery_count"),
                        "resume_checkpoint": current.get("resume_checkpoint"),
                        "reconciled_after_interruption": True,
                        "status_file_fsynced": True, "status_directory_fsynced": True,
                    })
            else:
                raise PermissionError(f"status is neither frozen source nor this migration: {run_id}")
            if fault == f"after_status{index}":
                raise OSError(f"injected failure after target status {index}; active hold remains active")
        # Before changing the hold, verify both target statuses and all protected assets.
        for run_id in TARGET_IDS:
            migrated = _read_json(_status_path(runs_root, run_id))
            if (migrated.get("status") != "RECOVERY_BLOCKED"
                    or migrated.get("v13_migration_id") != migration_id
                    or int(migrated.get("recovery_count", -1)) != 0
                    or migrated.get("resume_checkpoint") is not None):
                raise PermissionError(f"target migration status did not verify: {run_id}")
        asset_before_hold, asset_hash_before_hold = _asset_inventory(runs_root, rows)
        if asset_hash_before_hold != asset_sha or asset_before_hold != asset_inventory:
            raise PermissionError("completed assets changed before hold extension")
        current_hold = _read_json(hold_path)
        current_hold_sha = sha256_file(hold_path)
        if current_hold.get("active") is not True:
            raise PermissionError("active hold changed before its H49 extension")
        hold_already_extended = current_hold_sha != ORIGINAL_HOLD_SHA256
        if hold_already_extended and (
                current_hold.get("preserved_held_run_ids") != list(PRESERVED_H49_RUN_IDS)
                or current_hold.get("preserved_held_run_ids_sha256") != PRESERVED_H49_SHA256
                or current_hold.get("administrative_migration_id") != migration_id
                or not any(row.get("migration_id") == migration_id
                           and row.get("event") == "V13_H49_ABNORMAL_WORKER_STATUS_MIGRATION"
                           for row in current_hold.get("events", []) if isinstance(row, Mapping))):
            raise PermissionError("active hold has an unexplained or incomplete H49 extension")
        if hold_already_extended:
            _verify_extended_hold_projection(current_hold, prepared, migration_id, corr_sha)
            # A previous attempt may have renamed the hold but stopped before
            # fsyncing the containing directory. Re-establish durability now.
            _fsync_existing_file_and_parent(hold_path)
        if not hold_already_extended and current_hold_sha != ORIGINAL_HOLD_SHA256:
            raise PermissionError("active hold changed before its H49 extension")
        event = {
            "event": "V13_H49_ABNORMAL_WORKER_STATUS_MIGRATION",
            "timestamp_utc": _now(), "migration_id": migration_id,
            "source_v4_path": V4_PATH, "source_v4_sha256": V4_SHA256,
            "provenance_correction_path": correction_rel,
            "provenance_correction_sha256": corr_sha,
            "migration_audit_path": str(MIGRATION_PATH),
            "prepared_record_sha256": prepared_sha,
            "preserved_held_run_ids": list(PRESERVED_H49_RUN_IDS),
            "preserved_held_run_ids_sha256": PRESERVED_H49_SHA256,
            "new_statuses": {run_id: "RECOVERY_BLOCKED" for run_id in TARGET_IDS},
            "automatic_retry_authorized": False, "recovery_authorized": False,
            "completed_asset_inventory_sha256": asset_sha,
        }
        if not hold_already_extended:
            hold_events = list(current_hold.get("events", []))
            if fault == "before_hold_write":
                raise OSError("injected failure before H49 hold write; current hold remains active")
            updated_hold = dict(current_hold)
            hold_events.append(event)
            updated_hold.update({
                "active": True,
                "events": hold_events,
                "preserved_held_run_ids": list(PRESERVED_H49_RUN_IDS),
                "preserved_held_run_ids_sha256": PRESERVED_H49_SHA256,
                "administrative_migration_id": migration_id,
                "updated_at_utc": _now(),
            })
            _atomic_json(hold_path, updated_hold,
                         fault=("before_rename" if fault == "hold_before_rename" else
                                "before_dir_fsync" if fault == "hold_before_dir_fsync" else None))
        current_hold_after = _read_json(hold_path)
        _verify_extended_hold_projection(current_hold_after, prepared, migration_id, corr_sha)
        new_hold_sha = sha256_file(hold_path)
        # Post-commit verification uses actual disk state and fixed independent H49.
        statuses_after, counts_after = _queue_statuses(runs_root, rows)
        expected_counts = {"VALIDATION_COMPLETE": 52, "RECOVERY_BLOCKED": 49,
                           "UNCLAIMED": 187}
        if counts_after != expected_counts:
            raise PermissionError(f"post-migration counts differ: {counts_after}")
        if any(statuses_after.get(run_id) != "RECOVERY_BLOCKED"
               for run_id in PRESERVED_H49_RUN_IDS):
            raise PermissionError("one or more frozen H49 rows are not blocked after migration")
        if any(statuses_after.get(run_id) in ("RUNNING", "INTERRUPTED")
               for row in rows
               if (run_id := str(row.get("run_id"))) not in PRESERVED_H49_RUN_IDS):
            raise PermissionError("unexpected non-held RUNNING/INTERRUPTED search state")
        assets_post, asset_sha_post = _asset_inventory(runs_root, rows)
        if asset_sha_post != EXPECTED_ASSET_INVENTORY_SHA256 or assets_post != asset_inventory:
            raise PermissionError("52 completed assets changed during administrative migration")
        incidents_post, incident_sha_post = _incident_asset_inventory(runs_root)
        if incident_sha_post != incident_sha or incidents_post != incident_inventory:
            raise PermissionError("incident assets changed during administrative migration")
        queue_path = _search_queue_path(runs_root)
        queue_file_sha = sha256_file(queue_path)
        queue_entries_sha = _sha256_bytes(_compact_json(_load_search_queue(runs_root)))
        if (queue_file_sha != prepared.get("search_queue_file_sha256")
                or queue_entries_sha != prepared.get("search_queue_entries_sha256")):
            raise PermissionError("frozen search queue changed during administrative migration")
        status_projections_after = {
            run_id: _read_json(_status_path(runs_root, run_id)) for run_id in TARGET_IDS}
        committed = {
            "schema": SCHEMA, "event": "COMMITTED", "committed_at_utc": _now(),
            "migration_id": migration_id,
            "prepared_record_sha256": prepared_sha,
            "provenance_correction_path": correction_rel,
            "provenance_correction_sha256": corr_sha,
            "held_run_ids": list(PRESERVED_H49_RUN_IDS),
            "held_run_ids_sha256": PRESERVED_H49_SHA256,
            "statuses_after": {run_id: sha256_file(_status_path(runs_root, run_id))
                               for run_id in TARGET_IDS},
            "status_projections_after": status_projections_after,
            "search_queue_counts_after": counts_after,
            "search_queue_file_sha256": queue_file_sha,
            "search_queue_entries_sha256": queue_entries_sha,
            "active_hold_path": str(hold_path.relative_to(runs_root)),
            "active_hold_sha256": new_hold_sha,
            "completed_asset_file_count": len(assets_post),
            "completed_asset_inventory_sha256": asset_sha_post,
            "incident_asset_file_count": len(incidents_post),
            "incident_asset_inventory_sha256": incident_sha_post,
            "authorization": {"resume": False, "retry": False, "new_attempt": False,
                              "checkpoint_load": False, "formal": False, "test": False},
        }
        _append_jsonl(migration_path, committed)
        return {"migration_id": migration_id, "counts": counts_after,
                "hold_sha256": new_hold_sha, "correction_sha256": corr_sha,
                "migration_sha256": sha256_file(migration_path),
                "asset_file_count": len(assets_post), "asset_inventory_sha256": asset_sha_post,
                "incident_asset_file_count": len(incidents_post),
                "incident_asset_inventory_sha256": incident_sha_post}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--apply-v13-admin-migration", action="store_true",
                        help="perform only the approved administrative status/hold migration")
    args = parser.parse_args(argv)
    if not args.apply_v13_admin_migration:
        raise SystemExit("refusing: explicit --apply-v13-admin-migration is required")
    result = apply_migration(runs_root=args.runs_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
