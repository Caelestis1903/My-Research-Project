"""F0 v1.4 post-abnormal protection and fresh-only execution gate.

This module is deliberately separate from the v1.3 H49 migration code.  The
v1.3 validator remains an immutable historical check; current production
eligibility requires this module's exact H51 graph.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import math
import socket
import subprocess
import tempfile
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

from stage_F0_worker_control import (
    PRESERVED_H49_RUN_IDS,
    PRESERVED_H49_SHA256,
    RUNS,
    _atomic_json,
    _read_json,
    held_run_ids_sha256,
    sha256_file,
)


SCHEMA = "F0-POSTABNORMAL-EXECUTION-CONTROL-v1.4"
SIGSEGV_KEY = "0d2e6555b049c012c0e5"
PEER_KEY = "a80a232a3c0068a48a08"
NEW_INCIDENT_KEYS = (SIGSEGV_KEY, PEER_KEY)
H51_RUN_IDS = tuple(sorted(set(PRESERVED_H49_RUN_IDS) | set(NEW_INCIDENT_KEYS)))
H51_SHA256 = "e9077aefa1ed48a3237b769b20c0cd9e8ca3f9b0054ee537de1b7a40cd870a86"
QUEUE_SHA256 = "43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23"
COMPLETED_KEYS_SHA256 = "06cba20a42bfc3f8dfc20b4df68ac7dc774c66bb34972ab4158a7227db4d7fce"
FRESH_KEYS_SHA256 = "78e529b19e6fb9cc55cdc32524f3243005c0bf88f5dbe4206c44062a645f585b"
EXPECTED_SOURCE_HOLD_SHA256 = "7b2899b8ba587d986d4cdc8c5a7889888d213151f5a70c9393ce14f4a235c4d3"
EXPECTED_STATUS_SHA256 = {
    SIGSEGV_KEY: "809b4dd83609f38df791afb1581f873c801aa3d408b4f51ff1d60c4dc00d4ca0",
    PEER_KEY: "5dc0c4c0d656c7cc3f71772852d7b5508d9d9daa005c4d88c596dc70c5aaf1d5",
}
EXPECTED_MANIFEST_SHA256 = {
    SIGSEGV_KEY: "55404e864d28abe43e346a35b308cb8e2a7e566f55179d81b341af0e64a09c63",
    PEER_KEY: "d8377fafbd11ae7c5b5de209f36f30a6613c4fd060a56ede22bdfdc62fda7ee1",
}
EXPECTED_FAILURE_SHA256 = {
    SIGSEGV_KEY: "51ac190ccfa846b982f5eb0ac0e4a97ee3d144a6f42a1d89b25b72d6c3fa17e2",
    PEER_KEY: "11b2735b9fbfd546ab2272b4f267d4a5358f1f51d62299726baa6dd3b4f1be0b",
}
PEER_RECEIPT_SHA256 = "cced71d99bf329a963916a38b13dd48d0f3818c62faaad1e157a00d6f9b45f1d"
PEER_RECEIPT_REL = "worker_logs/trajectory_worker1_23_1787964e065e44bfbc503c26adca3e97.receipt.json"
INCIDENT_NONCE_LOGS = {
    SIGSEGV_KEY: (
        "worker_logs/trajectory_worker0_24_34de4e7e0970459a84a6457ff8ed490f.stdout.log",
        "worker_logs/trajectory_worker0_24_34de4e7e0970459a84a6457ff8ed490f.stderr.log",
        "worker_logs/trajectory_worker0_24_34de4e7e0970459a84a6457ff8ed490f.phases.jsonl",
    ),
    PEER_KEY: (
        "worker_logs/trajectory_worker1_23_1787964e065e44bfbc503c26adca3e97.stdout.log",
        "worker_logs/trajectory_worker1_23_1787964e065e44bfbc503c26adca3e97.stderr.log",
        "worker_logs/trajectory_worker1_23_1787964e065e44bfbc503c26adca3e97.phases.jsonl",
    ),
}
INCIDENT_NONCES = {
    SIGSEGV_KEY: "34de4e7e0970459a84a6457ff8ed490f",
    PEER_KEY: "1787964e065e44bfbc503c26adca3e97",
}
SOURCE_FILE_PATHS = (
    "execution_control/global_hold.json", "execution_control/global_hold_release.json",
    "../run_gate_search.json", "../execution_isolation_gate.json",
    "execution_control/postabnormal_sigsegv_audit_v4_fresh_search.json",
    "execution_control/postabnormal_sigsegv_audit_v4_provenance_correction.json",
    "execution_control/postabnormal_sigsegv_audit_v5_migration.jsonl",
    "execution_control/prelaunch_audit_v13_fresh_search_v2.json",
    "experiments/reports/stage_F0_execution_isolation_design.md",
    "experiments/reports/stage_F0_execution_isolation_implementation_report.md",
    "experiments/reports/stage_F0_v13_implementation_review_v4.md",
    "experiments/reports/stage_F0_v13_postmigration_review_v1.md",
)
FAILURE_RELS = {
    SIGSEGV_KEY: "worker_logs/worker_0_failure_1791829470_34de4e7e0970459a84a6457ff8ed490f.json",
    PEER_KEY: "worker_logs/worker_1_failure_1791829473_1787964e065e44bfbc503c26adca3e97.json",
}
STATUS_REL = "queue_status"
QUEUE_REL = "../queues/F0-B1-ML-search-queue.json"
CONTROL_DIR = "execution_control"
DESIGN_SHA256 = "c7204144394c734d84a61da2d0a702ccd8223308a35f3bd01119cf217840ebf9"
IMPLEMENTATION_REVIEW_SCHEMA = "F0-POSTABNORMAL-IMPLEMENTATION-REVIEW-v1.4"
IMPLEMENTATION_REPORT_REL = "experiments/reports/stage_F0_postabnormal_v14_implementation_report.md"


def require_live_gpu_memory(*, runs_root: os.PathLike | str, gpu_index: int,
                            gpu_uuid: str) -> dict[str, Any]:
    """Read strict target telemetry for claim while the caller holds state lock."""
    if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
        raise PermissionError("target GPU index is invalid")
    root = _runs_root(runs_root)
    gate_path = root.parent / "execution_isolation_gate.json"
    gate = _read_json(gate_path)
    validate_current_gate(gate, runs_root=root, require_release=True)
    prelaunch = _read_json(root / CONTROL_DIR / "prelaunch_audit_v14_fresh_search.json")
    devices = prelaunch.get("gpu_memory", {}).get("devices", [])
    rows = {row.get("index"): row for row in devices if isinstance(row, Mapping)}
    target = rows.get(gpu_index)
    expected_uuid = target.get("uuid") if isinstance(target, Mapping) else None
    if not isinstance(expected_uuid, str) or not expected_uuid or gpu_uuid != expected_uuid:
        raise PermissionError("claim GPU UUID differs from reviewed prelaunch target")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used",
             "--format=csv,noheader,nounits"], check=False, capture_output=True,
            text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PermissionError("live GPU memory telemetry is unavailable") from exc
    if result.returncode != 0:
        raise PermissionError("live GPU memory telemetry returned nonzero status")
    found = None
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise PermissionError("live GPU memory telemetry row is malformed")
        try:
            row_index, total, used = int(fields[0]), int(fields[2]), int(fields[3])
        except ValueError as exc:
            raise PermissionError("live GPU memory telemetry is nonnumeric") from exc
        if total <= 0 or used < 0 or used > total:
            raise PermissionError("live GPU memory telemetry is out of range")
        if row_index == gpu_index:
            found = {"index": row_index, "uuid": fields[1], "total_mib": total,
                     "used_mib": used, "fraction": used / total,
                     "observed_at_unix": time.time()}
    if found is None or found["uuid"] != expected_uuid or found["fraction"] >= 0.90:
        raise PermissionError("live GPU memory is unreadable, mismatched, or at/above 90%")
    return found


def compact_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                     sort_keys=True, separators=(",", ":"), allow_nan=False)
                          .encode("utf-8")).hexdigest()


def require_exact_h51(values: object) -> list[str]:
    if isinstance(values, (str, bytes, Mapping)):
        raise PermissionError("current execution requires exact frozen H51")
    try:
        normalized = list(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise PermissionError("current execution requires exact frozen H51") from exc
    if (any(not isinstance(value, str) for value in normalized)
            or len(normalized) != len(set(normalized))
            or set(normalized) != set(H51_RUN_IDS)):
        raise PermissionError("current execution requires exact frozen H51")
    return list(H51_RUN_IDS)


def _require_frozen_key_list(values: object, *, expected: tuple[str, ...],
                             digest: str, label: str) -> list[str]:
    if (not isinstance(values, list) or any(not isinstance(v, str) for v in values)
            or values != sorted(values) or len(values) != len(set(values))
            or tuple(values) != expected or compact_sha(values) != digest):
        raise PermissionError(f"{label} exact ordered identity/digest mismatch")
    return values


def _runs_root(path: os.PathLike | str = RUNS) -> Path:
    return Path(path).resolve()


def _relative(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if root not in path.parents and root.parent not in path.parents:
        raise PermissionError("v1.4 source path escapes the F0 runs tree")
    return path


def _tree_inventory(root: Path, run_id: str) -> list[dict[str, object]]:
    base = root / run_id
    if not base.is_dir() or base.is_symlink():
        raise PermissionError("protected run tree is missing or unsafe: " + run_id)
    entries: list[dict[str, object]] = []
    for parent, dirs, files in os.walk(base, followlinks=False):
        dirs.sort()
        files.sort()
        if any((Path(parent) / name).is_symlink() for name in dirs):
            raise PermissionError("protected run tree contains a symlink: " + run_id)
        for name in files:
            path = Path(parent) / name
            if path.is_symlink() or not path.is_file():
                raise PermissionError("protected file is not regular: " + str(path))
            entries.append({"path": path.relative_to(root).as_posix(),
                            "size": path.stat().st_size,
                            "sha256": sha256_file(path)})
    return entries


def _inventory_digest(entries: list[dict[str, object]]) -> str:
    rows = sorted(entries, key=lambda row: str(row["path"]))
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False)
                          .encode("utf-8")).hexdigest()


def _incident_evidence_inventory(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    expected_log_paths = {rel for paths in INCIDENT_NONCE_LOGS.values() for rel in paths}
    for run_id in NEW_INCIDENT_KEYS:
        entries.extend(_tree_inventory(root, run_id))
        failure_path = FAILURE_RELS[run_id]
        failure_file = _relative(root, failure_path)
        failure = _read_json(failure_file)
        nonce = INCIDENT_NONCES[run_id]
        failure_run_id = failure.get("run_id") or failure.get("canonical_claim", {}).get("run_id")
        failure_nonce = (failure.get("nonce") or failure.get("invocation_nonce")
                         or failure.get("canonical_claim", {}).get("nonce"))
        if failure_run_id != run_id or failure_nonce != nonce:
            raise PermissionError("incident failure audit identity/nonce differs: " + run_id)
        entries.append({"path": failure_path, "size": failure_file.stat().st_size,
                        "sha256": sha256_file(failure_file)})
        fields = (("stdout_path", "stdout_sha256", 0),
                  ("native_stderr_path", "native_stderr_sha256", 1),
                  ("phase_events_path", "phase_events_sha256", 2))
        for path_field, sha_field, log_index in fields:
            expected_rel = INCIDENT_NONCE_LOGS[run_id][log_index]
            raw_path = str(failure.get(path_field, ""))
            log_path = _relative(root, expected_rel)
            if (Path(raw_path).name != Path(expected_rel).name
                    or not log_path.is_file() or log_path.is_symlink()
                    or failure.get(sha_field) != sha256_file(log_path)):
                raise PermissionError("incident nonce log is missing or mismatched: "
                                      + expected_rel)
            entries.append({"path": expected_rel, "size": log_path.stat().st_size,
                            "sha256": sha256_file(log_path)})
        for rel in expected_log_paths:
            log = _relative(root, rel)
            if not log.is_file() or log.is_symlink():
                raise PermissionError("required incident nonce log is missing: " + rel)
    peer_receipt = _relative(root, PEER_RECEIPT_REL)
    if not peer_receipt.is_file() or peer_receipt.is_symlink() \
            or sha256_file(peer_receipt) != PEER_RECEIPT_SHA256:
        raise PermissionError("peer durable receipt differs from frozen source")
    entries.append({"path": PEER_RECEIPT_REL, "size": peer_receipt.stat().st_size,
                    "sha256": sha256_file(peer_receipt)})
    sig_receipts = list((root / "worker_logs").glob(
        "trajectory_worker0_*_34de4e7e0970459a84a6457ff8ed490f.receipt.json"))
    if sig_receipts:
        raise PermissionError("SIGSEGV receipt unexpectedly exists")
    # Reject nonce artifacts outside the exact reviewed stdout/stderr/phases
    # trio, including a newly appearing receipt.
    for run_id, nonce in INCIDENT_NONCES.items():
        matches = {path.relative_to(root).as_posix() for path in
                   (root / "worker_logs").glob("*" + nonce + "*") if path.is_file()}
        expected = {rel for rel in INCIDENT_NONCE_LOGS[run_id]}
        expected.add(FAILURE_RELS[run_id])
        if run_id == PEER_KEY:
            expected.add(PEER_RECEIPT_REL)
        if matches != expected:
            raise PermissionError("incident nonce artifact inventory differs: " + run_id)
    entries.sort(key=lambda row: str(row["path"]))
    return entries


def _source_file(root: Path, rel: str) -> dict[str, object]:
    if rel.startswith("experiments/"):
        path = Path(__file__).resolve().parents[1] / rel
    else:
        path = (root / rel).resolve()
        if root.parent not in path.parents and path != root.parent:
            raise PermissionError("v1.4 source path escapes the F0 output tree")
    data = path.read_bytes()
    return {"path": rel, "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data), "raw_base64": base64.b64encode(data).decode("ascii")}


def _queue_and_statuses(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    queue_path = _relative(root, QUEUE_REL)
    if sha256_file(queue_path) != QUEUE_SHA256:
        raise PermissionError("fixed 288-entry search queue hash changed")
    queue = _read_json(queue_path)
    rows = queue.get("entries")
    if not isinstance(rows, list) or len(rows) != 288:
        raise PermissionError("v1.4 requires the exact 288-key search queue")
    ids = [str(row.get("run_id", "")) for row in rows]
    if len(ids) != len(set(ids)) or not set(H51_RUN_IDS).issubset(ids):
        raise PermissionError("search queue identity is malformed or omits held keys")
    statuses: dict[str, dict[str, Any]] = {}
    for rid in ids:
        path = root / STATUS_REL / (rid + ".json")
        if path.exists():
            payload = _read_json(path)
            if payload.get("run_id") != rid or not isinstance(payload.get("status"), str):
                raise PermissionError("queue status identity mismatch: " + rid)
            statuses[rid] = payload
    return queue, statuses


def _historical_v13_runtime(root: Path) -> dict[str, Any]:
    from stage_F0_migration import validate_h49_historical_graph
    result = validate_h49_historical_graph(
        root, correction_sha256="c70c1068c8703db5dbad88dbd1e508cfac43bfebab3927682d0a4047b67ee7b7",
        migration_sha256="2608edd595fcfdc46791eeb73b3aa42b1c82d75b265c7d3abb16146884284f08",
        expected_asset_sha256="94039db9a66bfe20e7a4bb2d96b9249c145aa2fcf23fb53ca71bc225ffc718ae",
        expected_incident_asset_sha256="10fc3b31d3f462a89f9204db14f7e97a7bce2ad733279bf2724d3311e3e6e15e")
    immutable_fields = (
        "held_run_ids", "held_run_ids_sha256", "correction_sha256", "migration_sha256",
        "asset_file_count", "asset_inventory_sha256", "incident_asset_file_count",
        "incident_asset_inventory_sha256", "search_queue_file_sha256",
        "search_queue_entries_sha256", "v4_sha256", "peer_receipt_sha256",
        "design_sha256", "implementation_report_sha256")
    return {name: result[name] for name in immutable_fields}


def _validate_implementation_review(review: Mapping[str, Any], *,
                                    snapshot: Mapping[str, Any],
                                    snapshot_sha256: str,
                                    expected_verdicts: tuple[str, ...] = ("PASS", "PASS WITH FIXES")) -> None:
    repo = Path(__file__).resolve().parents[1]
    from stage_F0_worker_control import control_hash_inventory
    current_control = control_hash_inventory(repo)
    control_sha = compact_sha(current_control)
    report = repo / IMPLEMENTATION_REPORT_REL
    if not report.is_file():
        raise PermissionError("current v1.4 coder implementation report is missing")
    expected = {
        "schema": IMPLEMENTATION_REVIEW_SCHEMA,
        "verdict": expected_verdicts,
        "reviewer_role": "independent reviewer",
        "design_sha256": DESIGN_SHA256,
        "source_snapshot_sha256": snapshot_sha256,
        "coder_report_path": IMPLEMENTATION_REPORT_REL,
        "coder_report_sha256": sha256_file(report),
        "control_hashes": current_control,
        "control_hash_inventory_sha256": control_sha,
        "historical_v13_runtime_sha256": snapshot.get("historical_v13_runtime_sha256"),
        "reviewed_implementation_sha256": current_control.get(
            "experiments/stage_F0_postabnormal_v14.py"),
    }
    bad = [key for key, value in expected.items()
           if (review.get(key) not in value if key == "verdict" else review.get(key) != value)]
    if (not isinstance(review.get("reviewer_identity"), str)
            or not review.get("reviewer_identity")):
        bad.append("reviewer_identity")
    if bad:
        raise PermissionError("independent v1.4 implementation review identity is incomplete/stale: "
                              + ",".join(bad))


def source_snapshot(*, runs_root: os.PathLike | str = RUNS,
                    design_path: os.PathLike | str | None = None) -> dict[str, Any]:
    """Build a read-only, full provenance snapshot before migration."""
    root = _runs_root(runs_root)
    queue, statuses = _queue_and_statuses(root)
    if sha256_file(root / CONTROL_DIR / "global_hold.json") != EXPECTED_SOURCE_HOLD_SHA256:
        raise PermissionError("active source hold differs from frozen v1.4 incident state")
    hold_path = root / CONTROL_DIR / "global_hold.json"
    hold = _read_json(hold_path)
    if hold.get("active") is not True:
        raise PermissionError("v1.4 source snapshot requires the existing active hold")
    old_completed = sorted(rid for rid, payload in statuses.items()
                           if payload.get("status") == "VALIDATION_COMPLETE")
    if len(old_completed) != 97 or compact_sha(old_completed) != COMPLETED_KEYS_SHA256:
        raise PermissionError("completed-key baseline differs from frozen C97")
    fresh = []
    for row in queue["entries"]:
        rid = str(row["run_id"])
        if rid in H51_RUN_IDS or rid in statuses:
            continue
        canonical_lock = root / "canonical_locks" / (rid + ".lock")
        run_dir = root / rid
        if os.path.lexists(canonical_lock) or os.path.lexists(run_dir):
            continue
        fresh.append(rid)
    fresh.sort()
    if len(fresh) != 140 or compact_sha(fresh) != FRESH_KEYS_SHA256:
        raise PermissionError("fresh untouched candidate baseline differs from F140")
    for rid in H51_RUN_IDS:
        payload = statuses.get(rid)
        if payload is None:
            raise PermissionError("held source status is missing: " + rid)
        if rid in EXPECTED_STATUS_SHA256:
            if sha256_file(root / STATUS_REL / (rid + ".json")) != EXPECTED_STATUS_SHA256[rid]:
                raise PermissionError("new held source status differs from frozen bytes: " + rid)
        elif payload.get("status") != "RECOVERY_BLOCKED":
            raise PermissionError("historical H49 status is no longer blocked: " + rid)
        if rid in EXPECTED_MANIFEST_SHA256:
            manifest = root / rid / "attempts" / "attempt-00" / "run_manifest.json"
            if sha256_file(manifest) != EXPECTED_MANIFEST_SHA256[rid]:
                raise PermissionError("new held attempt manifest differs from frozen bytes: " + rid)
    if statuses[SIGSEGV_KEY].get("status") != "RUNNING" or statuses[PEER_KEY].get(
            "status") != "INTERRUPTED":
        raise PermissionError("incident source statuses changed")
    # Validate the independently reviewed historical H49 graph without using
    # it to authorize this current source or any launch.
    old_runtime = _historical_v13_runtime(root)
    old_h49 = {}
    for rid in PRESERVED_H49_RUN_IDS:
        p = root / STATUS_REL / (rid + ".json")
        if not p.is_file():
            raise PermissionError("historical H49 status is missing: " + rid)
        old_h49[rid] = {"raw_sha256": sha256_file(p), "projection": _read_json(p)}
    completed_inventory = []
    for rid in old_completed:
        completed_inventory.extend(_tree_inventory(root, rid))
    incident_inventory = _incident_evidence_inventory(root)
    # Freeze current source gate/release and all historical v1.3 evidence raw.
    frozen_sources = [_source_file(root, rel) for rel in SOURCE_FILE_PATHS]
    if design_path is None:
        repo = Path(__file__).resolve().parents[1]
        design = repo / "experiments/reports/stage_F0_postabnormal_design_v14.md"
    else:
        design = Path(design_path)
    design_bytes = design.read_bytes()
    if hashlib.sha256(design_bytes).hexdigest() != DESIGN_SHA256:
        raise PermissionError("frozen v1.4 design bytes differ from accepted specification")
    snapshot = {
        "schema": "F0-POSTABNORMAL-SOURCE-SNAPSHOT-v1.4",
        "control_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(), "operator": "coder staged snapshot",
        "design_path": "experiments/reports/stage_F0_postabnormal_design_v14.md",
        "design_sha256": hashlib.sha256(design_bytes).hexdigest(),
        "queue_path": "experiments/stage_F0_outputs/queues/F0-B1-ML-search-queue.json",
        "queue_sha256": QUEUE_SHA256,
        "queue_entry_count": 288,
        "excluded_resource_preflight_count": 6,
        "h49_ids": list(PRESERVED_H49_RUN_IDS),
        "h49_sha256": PRESERVED_H49_SHA256,
        "h51_ids": list(H51_RUN_IDS), "h51_sha256": H51_SHA256,
        "completed_ids": old_completed, "completed_ids_sha256": compact_sha(old_completed),
        "completed_inventory": sorted(completed_inventory, key=lambda x: str(x["path"])),
        "completed_inventory_sha256": _inventory_digest(completed_inventory),
        "fresh_ids": fresh, "fresh_ids_sha256": compact_sha(fresh),
        "incident_ids": list(NEW_INCIDENT_KEYS),
        "incident_inventory": sorted(incident_inventory, key=lambda x: str(x["path"])),
        "incident_inventory_sha256": _inventory_digest(incident_inventory),
        "incident_statuses": {rid: {"raw_sha256": EXPECTED_STATUS_SHA256[rid],
                                     "projection": statuses[rid]} for rid in NEW_INCIDENT_KEYS},
        "incident_manifests": EXPECTED_MANIFEST_SHA256,
        "failure_audits": {rid: {"path": FAILURE_RELS[rid],
                                  "sha256": EXPECTED_FAILURE_SHA256[rid]}
                           for rid in NEW_INCIDENT_KEYS},
        "sigsegv_receipt": {"exists": False, "authorized_to_synthesize": False},
        "peer_receipt": {"path": PEER_RECEIPT_REL, "sha256": PEER_RECEIPT_SHA256,
                          "status": "INTERRUPTED", "last_complete_epoch": 32,
                          "checkpoint_sha256": "52a6559c4466a8e19b11c34e0d6ba12f38e09716b8d39e5f9e357a7dbb7076d4"},
        "old_h49_statuses": old_h49,
        "historical_v13_runtime": old_runtime,
        "historical_v13_runtime_sha256": compact_sha(old_runtime),
        "source_files": frozen_sources,
        "hold_active": True,
        "launch_authorized": False,
        "recovery_authorized": False, "retry_authorized": False,
        "new_attempt_authorized": False, "formal_authorized": False,
        "test_authorized": False,
    }
    snapshot["snapshot_sha256"] = compact_sha(snapshot)
    return snapshot


def validate_source_snapshot(snapshot: Mapping[str, Any], *, runs_root: os.PathLike | str = RUNS,
                             allow_live_fresh_progress: bool = False) -> None:
    if snapshot.get("schema") != "F0-POSTABNORMAL-SOURCE-SNAPSHOT-v1.4":
        raise PermissionError("v1.4 source snapshot schema is missing")
    _require_frozen_key_list(snapshot.get("h51_ids"), expected=H51_RUN_IDS,
                             digest=H51_SHA256, label="H51")
    _require_frozen_key_list(snapshot.get("h49_ids"), expected=tuple(PRESERVED_H49_RUN_IDS),
                             digest=PRESERVED_H49_SHA256, label="historical H49")
    _require_frozen_key_list(snapshot.get("fresh_ids"),
                             expected=tuple(sorted(snapshot.get("fresh_ids", []))),
                             digest=FRESH_KEYS_SHA256, label="F140")
    _require_frozen_key_list(snapshot.get("completed_ids"),
                             expected=tuple(sorted(snapshot.get("completed_ids", []))),
                             digest=COMPLETED_KEYS_SHA256, label="C97")
    if snapshot.get("incident_ids") != list(NEW_INCIDENT_KEYS):
        raise PermissionError("source snapshot incident key identity differs")
    repo = Path(__file__).resolve().parents[1]
    design_path = repo / "experiments/reports/stage_F0_postabnormal_design_v14.md"
    if (snapshot.get("design_path") != "experiments/reports/stage_F0_postabnormal_design_v14.md"
            or snapshot.get("design_sha256") != DESIGN_SHA256
            or not design_path.is_file() or sha256_file(design_path) != DESIGN_SHA256):
        raise PermissionError("source snapshot does not bind the accepted v1.4 design bytes")
    unsigned = dict(snapshot)
    claimed_snapshot_sha = unsigned.pop("snapshot_sha256", None)
    if claimed_snapshot_sha != compact_sha(unsigned):
        raise PermissionError("source snapshot digest mismatch: claimed={} computed={}".format(
            claimed_snapshot_sha, compact_sha(unsigned)))
    for incident_id, failure_rel in FAILURE_RELS.items():
        failure = _relative(_runs_root(runs_root), failure_rel)
        if not failure.is_file() or sha256_file(failure) != EXPECTED_FAILURE_SHA256[incident_id]:
            raise PermissionError("frozen failure audit bytes differ: " + incident_id)
    failure_graph = snapshot.get("failure_audits")
    if not isinstance(failure_graph, Mapping):
        raise PermissionError("source snapshot omits failure audit graph")
    for rid in NEW_INCIDENT_KEYS:
        record = failure_graph.get(rid)
        if not isinstance(record, Mapping) or record.get("path") != FAILURE_RELS[rid] or record.get(
                "sha256") != EXPECTED_FAILURE_SHA256[rid]:
            raise PermissionError("source snapshot failure identity differs: " + rid)
    # Every archived raw source record must be internally sound. Mutable
    # pointers are verified against their frozen bytes and only compared to
    # disk during the pre-migration transaction.
    source_records = snapshot.get("source_files")
    if not isinstance(source_records, list) or not source_records:
        raise PermissionError("source snapshot raw source graph is missing")
    source_paths = [str(record.get("path", "")) for record in source_records
                    if isinstance(record, Mapping)]
    if source_paths != list(SOURCE_FILE_PATHS) or len(source_paths) != len(set(source_paths)):
        raise PermissionError("source snapshot raw source path set/order is incomplete or duplicated")
    for record in source_records:
        if not isinstance(record, Mapping):
            raise PermissionError("source snapshot raw source record is malformed")
        try:
            raw = base64.b64decode(str(record.get("raw_base64", "")), validate=True)
        except Exception as exc:
            raise PermissionError("source snapshot raw source bytes are malformed") from exc
        if (len(raw) != record.get("size")
                or hashlib.sha256(raw).hexdigest() != record.get("sha256")):
            raise PermissionError("source snapshot raw source digest mismatch")
        rel = str(record.get("path", ""))
        if rel.startswith("experiments/"):
            source_path = repo / rel
            if not source_path.is_file() or sha256_file(source_path) != record.get("sha256"):
                raise PermissionError("frozen historical source report changed: " + rel)
    root = _runs_root(runs_root)
    historical = _historical_v13_runtime(root)
    if (snapshot.get("historical_v13_runtime") != historical
            or snapshot.get("historical_v13_runtime_sha256") != compact_sha(historical)):
        raise PermissionError("frozen historical v1.3 runtime metadata differs from source graph")
    if snapshot.get("launch_authorized") is not False or snapshot.get("recovery_authorized") is not False:
        raise PermissionError("source snapshot grants execution scope")
    queue, current_statuses = _queue_and_statuses(root)
    completed_ids = snapshot.get("completed_ids", [])
    allowed_fresh = set(snapshot.get("fresh_ids", []))
    if set(current_statuses) - (set(completed_ids) | set(H51_RUN_IDS) | allowed_fresh):
        raise PermissionError("runtime contains status outside frozen C97/H51/F140")
    current_completed = sorted(rid for rid, payload in current_statuses.items()
                               if payload.get("status") == "VALIDATION_COMPLETE")
    if (not set(completed_ids).issubset(current_completed)
            or (not allow_live_fresh_progress and current_completed != list(completed_ids))):
        raise PermissionError("a frozen C97 completed status changed")
    current_completed_inventory: list[dict[str, object]] = []
    for rid in completed_ids:
        current_completed_inventory.extend(_tree_inventory(root, str(rid)))
    current_completed_inventory.sort(key=lambda row: str(row["path"]))
    if (current_completed_inventory != snapshot.get("completed_inventory")
            or _inventory_digest(current_completed_inventory)
            != snapshot.get("completed_inventory_sha256")):
        raise PermissionError("frozen C97 completed assets changed")
    live_fresh = set()
    for row in queue["entries"]:
        rid = str(row["run_id"])
        if rid in current_statuses or rid in H51_RUN_IDS:
            continue
        if (not os.path.lexists(root / "canonical_locks" / (rid + ".lock"))
                and not os.path.lexists(root / rid)):
            live_fresh.add(rid)
    if not live_fresh.issubset(allowed_fresh):
        raise PermissionError("runtime exposes a fresh key outside frozen F140")
    if not allow_live_fresh_progress:
        if live_fresh != allowed_fresh:
            raise PermissionError("pending H51 gate requires every F140 member untouched")
        touched = set(current_statuses) & allowed_fresh
        if touched:
            raise PermissionError("pending H51 gate found an already statused F140 member")
        for rid in allowed_fresh:
            if (os.path.lexists(root / "canonical_locks" / (rid + ".lock"))
                    or os.path.lexists(root / rid)):
                raise PermissionError("pending H51 gate found a touched F140 member: " + rid)
    # The complete incident trees and v1.4 failure evidence remain immutable;
    # only the two external queue-status projections may receive an admin
    # transition. This detects both asset drift and unexpected extra files.
    incident_inventory = _incident_evidence_inventory(root)
    if (incident_inventory != snapshot.get("incident_inventory")
            or _inventory_digest(incident_inventory)
            != snapshot.get("incident_inventory_sha256")):
        raise PermissionError("frozen incident trees/evidence changed")
    old_h49 = snapshot.get("old_h49_statuses")
    if not isinstance(old_h49, Mapping):
        raise PermissionError("source snapshot omits historical H49 statuses")
    for rid in PRESERVED_H49_RUN_IDS:
        record = old_h49.get(rid)
        path = root / STATUS_REL / (rid + ".json")
        if (not isinstance(record, Mapping) or not path.is_file()
                or record.get("raw_sha256") != sha256_file(path)
                or record.get("projection") != _read_json(path)):
            raise PermissionError("historical H49 status changed: " + rid)
    for rel in NEW_INCIDENT_KEYS:
        status_path = root / STATUS_REL / (rel + ".json")
        if not status_path.exists():
            raise PermissionError("incident status disappeared after snapshot freeze")
        current = _read_json(status_path)
        raw_sha = sha256_file(status_path)
        source = snapshot.get("incident_statuses", {}).get(rel, {}).get("projection")
        if raw_sha != EXPECTED_STATUS_SHA256[rel]:
            expected_admin = held_status_projection(source, run_id=rel) if isinstance(
                source, Mapping) else None
            if current != expected_admin:
                raise PermissionError("incident status differs from source or v1.4 admin projection")
    without = dict(snapshot)
    claimed = without.pop("snapshot_sha256", None)
    if claimed != compact_sha(without):
        raise PermissionError("source snapshot digest mismatch")


def validate_current_gate(gate: Mapping[str, Any], *, runs_root: os.PathLike | str = RUNS,
                          require_release: bool = True) -> dict[str, Any]:
    """Validate current H51 launch graph; historical H49 is audit-only."""
    root = _runs_root(runs_root)
    if gate.get("schema") != "F0-EXECUTION-ISOLATION-v1.4":
        raise PermissionError("current execution gate must be v1.4")
    require_exact_h51(gate.get("preserved_held_run_ids"))
    if gate.get("preserved_held_run_ids_sha256") != H51_SHA256:
        raise PermissionError("v1.4 gate H51 digest mismatch")
    if gate.get("search_queue_sha256") != QUEUE_SHA256:
        raise PermissionError("v1.4 gate search queue hash mismatch")
    if gate.get("launch_scope") != "F0-B1-ML search only; fresh UNCLAIMED canonical keys only":
        raise PermissionError("v1.4 gate scope is not fresh-only")
    for name in ("resume", "retry", "new_attempt", "checkpoint_load", "formal", "test"):
        if gate.get("authorization", {}).get(name) is not False:
            raise PermissionError("v1.4 gate authorizes forbidden scope: " + name)
    # This check lives here as well as in validate_execution_gate because
    # queue claim and hold checks call ensure_current_release directly.
    from stage_F0_worker_control import _sha256_json, control_hash_inventory
    repo = Path(__file__).resolve().parents[1]
    current_control = control_hash_inventory(repo)
    if (gate.get("control_hashes") != current_control
            or gate.get("control_hash_inventory_sha256") != _sha256_json(current_control)):
        raise PermissionError("v1.4 current control source inventory is stale")
    search_path = root.parent / "run_gate_search.json"
    if not search_path.is_file() or gate.get("prior_search_gate_sha256") != sha256_file(search_path):
        raise PermissionError("v1.4 gate does not bind the current frozen search gate")
    search_gate = _read_json(search_path)
    if (search_gate.get("phase") != "search"
            or search_gate.get("formal_or_test_started") is not False
            or gate.get("preserved_scientific_code_hash") != search_gate.get("code_hash")):
        raise PermissionError("v1.4 gate differs from frozen scientific search identity")
    amendment = gate.get("scientific_code_hash_amendment")
    if not isinstance(amendment, Mapping):
        raise PermissionError("v1.4 control amendment is missing or invalid")
    if (amendment.get("status") != "APPROVED"
            or amendment.get("reviewer_verdict") not in ("PASS", "PASS WITH FIXES")):
        raise PermissionError("v1.4 control amendment approval metadata is missing or invalid")
    if amendment.get("frozen_scientific_code_hash") != search_gate.get("code_hash"):
        raise PermissionError("v1.4 control amendment does not bind scientific identity")
    from stage_F0_queue import code_hash
    if amendment.get("runtime_control_code_hash") != code_hash(require_tests=True):
        raise PermissionError("v1.4 runtime control amendment is stale")
    snapshot_path = root / CONTROL_DIR / "postabnormal_v14_source_snapshot.json"
    if not snapshot_path.is_file() or sha256_file(snapshot_path) != gate.get("source_snapshot_sha256"):
        raise PermissionError("v1.4 gate source snapshot is missing or changed")
    snapshot = _read_json(snapshot_path)
    validate_source_snapshot(snapshot, runs_root=root,
                             allow_live_fresh_progress=require_release)
    if gate.get("historical_h49_sha256") != PRESERVED_H49_SHA256:
        raise PermissionError("v1.4 gate historical H49 digest mismatch")
    if gate.get("independent_review_verdict") not in ("PASS", "PASS WITH FIXES"):
        raise PermissionError("v1.4 implementation review is not approved")
    review_path = gate.get("independent_review_path")
    if not isinstance(review_path, str) or not review_path:
        raise PermissionError("v1.4 independent review path is missing")
    review = repo / review_path
    if not review.is_file() or sha256_file(review) != gate.get("independent_review_sha256"):
        raise PermissionError("v1.4 independent review bytes differ")
    review_payload = _read_json(review)
    if review_payload.get("verdict") != gate.get("independent_review_verdict"):
        raise PermissionError("v1.4 gate review verdict differs from the reviewed artifact")
    _validate_implementation_review(review_payload, snapshot=snapshot,
        snapshot_sha256=str(gate.get("source_snapshot_sha256")))
    # Recheck historical v1.3 protection independently. This is intentionally
    # not used as authorization for the current H51 release.
    from stage_F0_migration import validate_h49_historical_graph
    old = validate_h49_historical_graph(
        root, correction_sha256="c70c1068c8703db5dbad88dbd1e508cfac43bfebab3927682d0a4047b67ee7b7",
        migration_sha256="2608edd595fcfdc46791eeb73b3aa42b1c82d75b265c7d3abb16146884284f08",
        expected_asset_sha256="94039db9a66bfe20e7a4bb2d96b9249c145aa2fcf23fb53ca71bc225ffc718ae",
        expected_incident_asset_sha256="10fc3b31d3f462a89f9204db14f7e97a7bce2ad733279bf2724d3311e3e6e15e")
    migration_path = root / CONTROL_DIR / "postabnormal_v14_migration.jsonl"
    if not migration_path.is_file():
        raise PermissionError("v1.4 administrative migration is not committed")
    if gate.get("migration_ledger_sha256") != sha256_file(migration_path):
        raise PermissionError("v1.4 migration ledger hash mismatch")
    try:
        migration_events = _read_jsonl(migration_path)
    except (OSError, ValueError) as exc:
        raise PermissionError("v1.4 migration ledger is malformed") from exc
    if (sum(item.get("event") == "PREPARED" for item in migration_events) != 1
            or sum(item.get("event") == "COMMITTED" for item in migration_events) != 1
            or [item.get("run_id") for item in migration_events
                if item.get("event") == "STATUS_TRANSITION_DURABLE"] != list(NEW_INCIDENT_KEYS)
            or sum(item.get("event") == "H51_HOLD_DURABLE"
                   for item in migration_events) != 1):
        raise PermissionError("v1.4 migration ledger lacks one complete H51 transaction")
    hold_path = root / CONTROL_DIR / "global_hold.json"
    hold = _read_json(hold_path)
    require_exact_h51(hold.get("preserved_held_run_ids"))
    if hold.get("preserved_held_run_ids_sha256") != H51_SHA256:
        raise PermissionError("current hold does not bind H51")
    if not any(isinstance(event, Mapping) and event.get("event") == "V14_H51_ADMIN_MIGRATION_COMMITTED"
               and event.get("held_run_ids") == list(H51_RUN_IDS)
               and event.get("held_run_ids_sha256") == H51_SHA256
               for event in hold.get("events", [])):
        raise PermissionError("current hold lacks committed v1.4 H51 event")
    _, current_statuses = _queue_and_statuses(root)
    for rid in H51_RUN_IDS:
        if current_statuses.get(rid, {}).get("status") != "RECOVERY_BLOCKED":
            raise PermissionError("an H51 member is not administratively blocked: " + rid)
    hold_sha = sha256_file(hold_path)
    prepared = next(item for item in migration_events if item.get("event") == "PREPARED")
    committed = next(item for item in migration_events if item.get("event") == "COMMITTED")
    durable_hold = next(item for item in migration_events if item.get("event") == "H51_HOLD_DURABLE")
    if (prepared.get("h51_ids") != list(H51_RUN_IDS)
            or prepared.get("h51_sha256") != H51_SHA256
            or prepared.get("source_snapshot_sha256") != gate.get("source_snapshot_sha256")
            or prepared.get("reviewer_sha256") != gate.get("independent_review_sha256")
            or committed.get("h51_ids") != list(H51_RUN_IDS)
            or committed.get("h51_sha256") != H51_SHA256
            or committed.get("source_snapshot_sha256") != gate.get("source_snapshot_sha256")
            or committed.get("hold_sha256") != durable_hold.get("hold_sha256")
            or gate.get("active_hold_sha256") != committed.get("hold_sha256")):
        raise PermissionError("v1.4 migration event hashes or original active hold differ")
    if gate.get("launch_authorized") is not True and hold_sha != committed.get("hold_sha256"):
        raise PermissionError("pending v1.4 hold bytes differ from committed active projection")
    for rid in NEW_INCIDENT_KEYS:
        status_path = root / STATUS_REL / (rid + ".json")
        projection = held_status_projection(
            snapshot["incident_statuses"][rid]["projection"], run_id=rid)
        transition = next(item for item in migration_events
                          if item.get("event") == "STATUS_TRANSITION_DURABLE"
                          and item.get("run_id") == rid)
        status_sha = sha256_file(status_path)
        if (current_statuses[rid] != projection
                or committed.get("statuses", {}).get(rid) != status_sha
                or transition.get("before_sha256") != EXPECTED_STATUS_SHA256[rid]
                or transition.get("after_sha256") != status_sha
                or transition.get("status_file_fsynced") is not True
                or transition.get("status_directory_fsynced") is not True):
            raise PermissionError("v1.4 status transition does not match frozen projection: " + rid)
    prelaunch_rel = gate.get("prelaunch_audit_path")
    prelaunch_sha = gate.get("prelaunch_audit_sha256")
    if (prelaunch_rel != "execution_control/prelaunch_audit_v14_fresh_search.json"
            or not isinstance(prelaunch_sha, str)):
        raise PermissionError("v1.4 gate does not bind the current prelaunch audit")
    prelaunch_path = root / prelaunch_rel
    if not prelaunch_path.is_file() or sha256_file(prelaunch_path) != prelaunch_sha:
        raise PermissionError("v1.4 prelaunch audit is missing or changed")
    prelaunch = _read_json(prelaunch_path)
    if (prelaunch.get("schema") != "F0-PRELAUNCH-AUDIT-v1.4"
            or prelaunch.get("go") is not True
            or prelaunch.get("h51_ids") != list(H51_RUN_IDS)
            or prelaunch.get("h51_sha256") != H51_SHA256
            or prelaunch.get("source_snapshot_sha256") != gate.get("source_snapshot_sha256")
            or prelaunch.get("migration_ledger_sha256") != gate.get("migration_ledger_sha256")
            or prelaunch.get("active_hold_sha256") != gate.get("active_hold_sha256")
            or prelaunch.get("fresh_ids_sha256") != FRESH_KEYS_SHA256
            or prelaunch.get("fresh_ids") != snapshot.get("fresh_ids")
            or prelaunch.get("completed_ids_sha256") != COMPLETED_KEYS_SHA256
            or prelaunch.get("excluded_resource_preflight_count") != 6
            or prelaunch.get("search_queue_sha256") != QUEUE_SHA256
            or prelaunch.get("control_hash_inventory_sha256") != _sha256_json(current_control)):
        raise PermissionError("v1.4 prelaunch audit is inconsistent with current frozen graph")
    process_count = prelaunch.get("task_python_process_count")
    worker_process_count = prelaunch.get("f0_worker_python_process_count")
    if (not isinstance(process_count, int) or isinstance(process_count, bool)
            or process_count < 0 or process_count > 5
            or not isinstance(worker_process_count, int)
            or isinstance(worker_process_count, bool)
            or worker_process_count < 0 or worker_process_count > 4
            or prelaunch.get("no_residual_f0_processes") is not True
            or prelaunch.get("active_claim_count") != 0
            or prelaunch.get("all_required_locks_available") is not True
            or not isinstance(prelaunch.get("disk_free_bytes"), int)
            or prelaunch.get("disk_free_bytes") <= 0):
        raise PermissionError("v1.4 prelaunch process, lock, or disk check is NO-GO")
    gpu_record = prelaunch.get("gpu_memory", {})
    gpu_rows = gpu_record.get("devices", []) if isinstance(gpu_record, Mapping) else []
    target_gpu_indices = prelaunch.get("target_gpu_indices")
    if (not isinstance(target_gpu_indices, list) or len(target_gpu_indices) != 2
            or any(not isinstance(index, int) or isinstance(index, bool) or index < 0
                   for index in target_gpu_indices)
            or len(set(target_gpu_indices)) != 2
            or not isinstance(gpu_rows, list)):
        raise PermissionError("v1.4 prelaunch GPU memory observations are missing")
    by_index = {row.get("index"): row for row in gpu_rows if isinstance(row, Mapping)}
    for index in target_gpu_indices:
        row = by_index.get(index)
        used, total, fraction, gpu_uuid = ((row or {}).get("memory_used_bytes"),
                                           (row or {}).get("memory_total_bytes"),
                                           (row or {}).get("memory_used_fraction"),
                                           (row or {}).get("uuid"))
        if (not isinstance(row, Mapping) or row.get("readable") is not True
                or row.get("eligible") is not True
                or not isinstance(gpu_uuid, str) or not gpu_uuid
                or not isinstance(used, int) or isinstance(used, bool) or used < 0
                or not isinstance(total, int) or isinstance(total, bool) or total <= 0
                or used > total or not isinstance(fraction, (int, float))
                or not math.isfinite(float(fraction)) or fraction < 0 or fraction >= 0.90
                or not math.isclose(float(fraction), used / total, rel_tol=1e-12, abs_tol=1e-12)):
            raise PermissionError("v1.4 prelaunch GPU memory is not below 90%")
    post_path = gate.get("postmigration_review_path")
    if not isinstance(post_path, str) or not post_path:
        raise PermissionError("v1.4 gate omits independent postmigration GO review")
    post_file = repo / post_path
    if not post_file.is_file() or sha256_file(post_file) != gate.get("postmigration_review_sha256"):
        raise PermissionError("postmigration GO review bytes differ")
    try:
        post_review = _read_json(post_file)
    except (OSError, ValueError) as exc:
        raise PermissionError("postmigration GO review identity is malformed") from exc
    if (post_review.get("verdict") != "GO"
            or post_review.get("source_snapshot_sha256") != gate.get("source_snapshot_sha256")
            or post_review.get("migration_ledger_sha256") != gate.get("migration_ledger_sha256")
            or post_review.get("active_hold_sha256") != committed.get("hold_sha256")
            or post_review.get("h51_ids") != list(H51_RUN_IDS)
            or post_review.get("h51_sha256") != H51_SHA256):
        raise PermissionError("independent postmigration GO does not bind current H51 migration")
    if require_release:
        if gate.get("launch_authorized") is not True:
            raise PermissionError("v1.4 launch authorization is not enabled")
        if hold.get("active") is not False:
            raise PermissionError("v1.4 hold remains active")
        release_path = root / CONTROL_DIR / "global_hold_release_v14.json"
        if not release_path.is_file() or sha256_file(release_path) != gate.get("release_sha256"):
            raise PermissionError("v1.4 principal release is absent or changed")
        release = _read_json(release_path)
        require_exact_h51(release.get("preserved_held_run_ids"))
        if (release.get("release_schema") != "F0-GLOBAL-HOLD-RELEASE-v1.4"
                or release.get("approved_by_role") != "principal"
                or not isinstance(release.get("approved_by"), str)
                or not release.get("approved_by")
                or release.get("principal_intent") != "fresh-only F0-B1-ML search"
                or release.get("hold_sha256") != hold_sha
                or release.get("released_hold_projection_sha256") != hold_sha
                or gate.get("released_hold_projection_sha256") != hold_sha
                or release.get("source_active_hold_sha256") != gate.get("active_hold_sha256")
                or release.get("source_snapshot_sha256") != gate.get("source_snapshot_sha256")
                or release.get("migration_ledger_sha256") != gate.get("migration_ledger_sha256")
                or release.get("preserved_held_run_ids_sha256") != H51_SHA256
                or release.get("independent_review_sha256") != gate.get("independent_review_sha256")
                or release.get("postmigration_review_sha256") != gate.get("postmigration_review_sha256")
                or release.get("prelaunch_audit_sha256") != gate.get("prelaunch_audit_sha256")
                or release.get("fresh_ids_sha256") != FRESH_KEYS_SHA256
                or release.get("launch_scope") != "fresh-only F0-B1-ML search"
                or release.get("authorization") != {
                    "resume": False, "retry": False, "new_attempt": False,
                    "checkpoint_load": False, "formal": False, "test": False}):
            raise PermissionError("v1.4 release identity or source graph is invalid")
        for name in ("resume", "retry", "new_attempt", "checkpoint_load", "formal", "test"):
            if release.get("authorization", {}).get(name) is not False:
                raise PermissionError("v1.4 release authorizes forbidden scope: " + name)
        release_ledger = root / CONTROL_DIR / "postabnormal_v14_release.jsonl"
        if (not release_ledger.is_file()
                or sha256_file(release_ledger) != gate.get("release_ledger_sha256")):
            raise PermissionError("v1.4 append-only release ledger is missing or stale")
        release_events = _read_jsonl(release_ledger)
        expected_events = ["PREPARED", "RELEASE_ARTIFACT_DURABLE", "HOLD_RELEASED", "COMMITTED"]
        if ([item.get("event") for item in release_events] != expected_events
                or any(item.get("release_id") != release.get("release_id")
                       for item in release_events)
                or release_events[1].get("release_sha256") != gate.get("release_sha256")
                or release_events[-1].get("released_hold_projection_sha256") != hold_sha):
            raise PermissionError("v1.4 release ledger does not bind a single complete release graph")
        release_event = [event for event in hold.get("events", [])
                         if isinstance(event, Mapping)
                         and event.get("event") == "V14_PRINCIPAL_FRESH_ONLY_RELEASE"]
        if (len(release_event) != 1
                or release_event[0].get("release_id") != release.get("release_id")
                or release_event[0].get("source_active_hold_sha256") != gate.get("active_hold_sha256")
                or release_event[0].get("h51_ids") != list(H51_RUN_IDS)
                or release_event[0].get("fresh_ids_sha256") != FRESH_KEYS_SHA256):
            raise PermissionError("released hold projection is not identity-bound to principal intent")
    else:
        if (hold.get("active") is not True or gate.get("launch_authorized") is not False
                or gate.get("active_hold_sha256") != hold_sha
                or gate.get("postmigration_review_verdict") != "GO"):
            raise PermissionError("pending v1.4 gate is not bound to reviewed active hold")
    return {"gate": dict(gate), "snapshot": snapshot, "hold": hold,
            "historical_h49": old, "prelaunch": prelaunch,
            "search_gate": search_gate}


def ensure_current_release(runs_root: os.PathLike | str = RUNS) -> dict[str, Any]:
    root = _runs_root(runs_root)
    gate_path = root.parent / "execution_isolation_gate.json"
    gate = _read_json(gate_path)
    validate_current_gate(gate, runs_root=root, require_release=True)
    return gate


def commit_principal_release(*, runs_root: os.PathLike | str = RUNS,
                             gate_path: os.PathLike | str,
                             principal_identity: Mapping[str, Any],
                             fault: str | None = None) -> dict[str, Any]:
    """Commit the identity-bound fresh-only release; never called by migration."""
    root = _runs_root(runs_root)
    gate_file = Path(gate_path)
    identity = dict(principal_identity)
    if (identity.get("approved_by_role") != "principal"
            or not isinstance(identity.get("approved_by"), str)
            or not identity.get("approved_by")
            or identity.get("principal_intent") != "fresh-only F0-B1-ML search"
            or not isinstance(identity.get("request_nonce"), str)
            or len(identity["request_nonce"]) < 16
            or identity.get("authorization") != {
                "resume": False, "retry": False, "new_attempt": False,
                "checkpoint_load": False, "formal": False, "test": False}):
        raise PermissionError("principal release intent/identity is incomplete or too broad")
    control = root / CONTROL_DIR
    locks = [control / "execution_state.lock"]
    locks += [root / "worker_locks" / f"worker_{i}.lock" for i in (0, 1)]
    locks += [root / "gpu_locks" / f"gpu{i}.lock" for i in (0, 1)]
    locks += [root / "canonical_locks" / f"{rid}.lock" for rid in NEW_INCIDENT_KEYS]
    handles = []
    ledger = control / "postabnormal_v14_release.jsonl"
    try:
        for path in locks:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise PermissionError("release lock is busy: " + str(path)) from exc
            handles.append(handle)
        if ledger.is_file():
            prior_release_events = _read_jsonl(ledger, repair_partial=True)
            commits = [item for item in prior_release_events if item.get("event") == "COMMITTED"]
            if commits:
                release_path = control / "global_hold_release_v14.json"
                hold_path = control / "global_hold.json"
                gate = _read_json(gate_file)
                expected_id = "f0-v14-release-" + identity["request_nonce"]
                expected_events = ["PREPARED", "RELEASE_ARTIFACT_DURABLE",
                                   "HOLD_RELEASED", "COMMITTED"]
                if ([item.get("event") for item in prior_release_events] != expected_events
                        or any(item.get("release_id") != expected_id
                               for item in prior_release_events)
                        or prior_release_events[0].get("principal_identity_sha256")
                            != compact_sha(identity)
                        or prior_release_events[2].get("released_hold_projection_sha256")
                            != prior_release_events[1].get("released_hold_projection_sha256")
                        or prior_release_events[3].get("release_sha256")
                            != prior_release_events[1].get("release_sha256")
                        or prior_release_events[3].get("released_hold_projection_sha256")
                            != prior_release_events[1].get("released_hold_projection_sha256")):
                    raise PermissionError("committed release ledger stages/identity are inconsistent")
                if (len(commits) != 1 or commits[0].get("release_id") != expected_id
                        or not release_path.is_file()
                        or sha256_file(release_path) != commits[0].get("release_sha256")):
                    raise PermissionError("committed release no longer matches exact artifact")
                release = _read_json(release_path)
                if ({key: release.get(key) for key in identity}
                        != dict(identity)
                        or release.get("release_id") != expected_id
                        or release.get("released_hold_projection_sha256")
                        != commits[0].get("released_hold_projection_sha256")):
                    raise PermissionError("committed release identity/projection changed")
                hold_path = control / "global_hold.json"
                hold = _read_json(hold_path)
                if hold.get("active") is True:
                    if sha256_file(hold_path) != release.get("source_active_hold_sha256"):
                        raise PermissionError("committed release active source hold changed")
                    recovered = dict(hold)
                    recovered["active"] = False
                    recovered.setdefault("released_at_utc", release["approved_at_utc"])
                    recovered["events"] = list(hold.get("events", [])) + [{
                        "event": "V14_PRINCIPAL_FRESH_ONLY_RELEASE",
                        "release_id": expected_id,
                        "approved_by": release["approved_by"],
                        "approved_by_role": "principal",
                        "approved_at_utc": release["approved_at_utc"],
                        "source_active_hold_sha256": release["source_active_hold_sha256"],
                        "source_snapshot_sha256": release["source_snapshot_sha256"],
                        "migration_ledger_sha256": release["migration_ledger_sha256"],
                        "h51_ids": list(H51_RUN_IDS), "h51_sha256": H51_SHA256,
                        "fresh_ids_sha256": FRESH_KEYS_SHA256,
                    }]
                    recovered["updated_at_utc"] = release["approved_at_utc"]
                    recovered_bytes = json.dumps(recovered, ensure_ascii=False, sort_keys=True,
                        indent=2, allow_nan=False).encode("utf-8") + b"\n"
                    if hashlib.sha256(recovered_bytes).hexdigest() != commits[0].get(
                            "released_hold_projection_sha256"):
                        raise PermissionError("committed release cannot reconstruct hold projection")
                    if gate.get("launch_authorized") is False:
                        validate_current_gate(gate, runs_root=root, require_release=False)
                        _atomic_json(control / "global_hold_release.json", release)
                        gate.update({"launch_authorized": True,
                            "release_sha256": commits[0]["release_sha256"],
                            "released_hold_projection_sha256": commits[0][
                                "released_hold_projection_sha256"],
                            "release_ledger_sha256": sha256_file(ledger)})
                        _atomic_json(gate_file, gate)
                        _fsync_file_parent(gate_file)
                    elif (gate.get("launch_authorized") is not True
                          or gate.get("release_sha256") != commits[0].get("release_sha256")
                          or gate.get("released_hold_projection_sha256")
                            != commits[0].get("released_hold_projection_sha256")
                          or gate.get("release_ledger_sha256") != sha256_file(ledger)):
                        raise PermissionError("committed release gate binding changed")
                    release_pointer = control / "global_hold_release.json"
                    if (not release_pointer.is_file()
                            or sha256_file(release_pointer) != sha256_file(release_path)
                            or _read_json(release_pointer) != release):
                        raise PermissionError("committed release pointer differs from immutable artifact")
                    if gate_file.resolve() != (root.parent / "execution_isolation_gate.json").resolve():
                        raise PermissionError("release reentry gate is not the canonical execution gate")
                    # A gate-authorized replay is still pending until the hold
                    # transition. Revalidate the complete pending graph just
                    # before that transition; only the view's authorization
                    # bit is lowered, while the persisted release bindings are
                    # independently verified above and remain untouched.
                    pending_gate = dict(gate)
                    pending_gate["launch_authorized"] = False
                    validate_current_gate(pending_gate, runs_root=root, require_release=False)
                    snapshot_path = control / "postabnormal_v14_source_snapshot.json"
                    snapshot = _read_json(snapshot_path)
                    migration_events = _read_jsonl(control / "postabnormal_v14_migration.jsonl")
                    migration_prepared = next((item for item in migration_events
                        if item.get("event") == "PREPARED"), None)
                    if migration_prepared is None:
                        raise PermissionError("release reentry has no prepared migration identity")
                    _verify_frozen_source_pointers(
                        snapshot, root=root,
                        migration_id=migration_prepared.get("migration_id"),
                        snapshot_sha256=gate.get("source_snapshot_sha256"),
                        expected_mutable={
                            "../execution_isolation_gate.json": gate,
                            "execution_control/global_hold_release.json": release,
                        })
                    _atomic_json(hold_path, recovered)
                elif sha256_file(hold_path) != commits[0].get(
                        "released_hold_projection_sha256"):
                    raise PermissionError("committed release no longer matches exact hold")
                if (gate.get("release_sha256") != commits[0].get("release_sha256")
                        or gate.get("released_hold_projection_sha256")
                            != commits[0].get("released_hold_projection_sha256")):
                    raise PermissionError("committed release no longer matches exact gate")
                for path in (release_path, hold_path, ledger, gate_file):
                    _fsync_file_parent(path)
                validate_current_gate(gate, runs_root=root, require_release=True)
                return {"status": "COMMITTED", "release_id": expected_id,
                        "release_sha256": commits[0]["release_sha256"],
                        "released_hold_sha256": commits[0][
                            "released_hold_projection_sha256"], "idempotent": True}
        gate = _read_json(gate_file)
        graph = validate_current_gate(gate, runs_root=root, require_release=False)
        hold_path = control / "global_hold.json"
        active_hold = graph["hold"]
        active_sha = sha256_file(hold_path)
        if active_hold.get("active") is not True or active_sha != gate.get("active_hold_sha256"):
            raise PermissionError("release requires the exact active reviewed H51 hold")
        prepared_id = "f0-v14-release-" + identity["request_nonce"]
        release_path = control / "global_hold_release_v14.json"
        released_at = identity.get("approved_at_utc")
        if not isinstance(released_at, str) or not released_at.endswith("Z"):
            raise PermissionError("principal release timestamp must be explicit UTC")
        released = dict(active_hold)
        released["active"] = False
        released.setdefault("released_at_utc", released_at)
        released["events"] = list(active_hold.get("events", [])) + [{
            "event": "V14_PRINCIPAL_FRESH_ONLY_RELEASE",
            "release_id": prepared_id, "approved_by": identity["approved_by"],
            "approved_by_role": "principal", "approved_at_utc": released_at,
            "source_active_hold_sha256": active_sha,
            "source_snapshot_sha256": gate["source_snapshot_sha256"],
            "migration_ledger_sha256": gate["migration_ledger_sha256"],
            "h51_ids": list(H51_RUN_IDS), "h51_sha256": H51_SHA256,
            "fresh_ids_sha256": FRESH_KEYS_SHA256,
        }]
        released["updated_at_utc"] = released_at
        released_bytes = json.dumps(released, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False).encode("utf-8") + b"\n"
        released_sha = hashlib.sha256(released_bytes).hexdigest()
        release = {
            "release_schema": "F0-GLOBAL-HOLD-RELEASE-v1.4",
            "release_id": prepared_id, **identity,
            "source_active_hold_sha256": active_sha,
            "released_hold_projection_sha256": released_sha,
            "hold_sha256": released_sha,
            "source_snapshot_sha256": gate["source_snapshot_sha256"],
            "migration_ledger_sha256": gate["migration_ledger_sha256"],
            "independent_review_sha256": gate["independent_review_sha256"],
            "postmigration_review_sha256": gate["postmigration_review_sha256"],
            "prelaunch_audit_sha256": gate["prelaunch_audit_sha256"],
            "preserved_held_run_ids": list(H51_RUN_IDS),
            "preserved_held_run_ids_sha256": H51_SHA256,
            "fresh_ids_sha256": FRESH_KEYS_SHA256,
            "launch_scope": "fresh-only F0-B1-ML search",
            "active_hold_released": True,
        }
        control.mkdir(parents=True, exist_ok=True)
        for label, path in (("execution_gate", gate_file),
                            ("hold_release_pointer", control / "global_hold_release.json")):
            if path.exists():
                archive = control / f"v14_pointer_snapshot_{label}_{sha256_file(path)}.json"
                _create_immutable_json(archive, {
                    "schema": "F0-V14-POINTER-SNAPSHOT-v1",
                    "pointer_path": str(path), "raw_sha256": sha256_file(path),
                    "raw_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
                })
        if ledger.exists():
            events = _read_jsonl(ledger, repair_partial=True)
        else:
            events = []
        existing_commits = [item for item in events if item.get("event") == "COMMITTED"]
        if existing_commits and (len(existing_commits) != 1
                                 or existing_commits[0].get("release_id") != prepared_id):
            raise PermissionError("a different principal release is already committed")
        if not any(item.get("event") == "PREPARED" for item in events):
            _append_durable_jsonl(ledger, {"schema": SCHEMA, "event": "PREPARED",
                "release_id": prepared_id, "source_active_hold_sha256": active_sha,
                "source_snapshot_sha256": gate["source_snapshot_sha256"],
                "principal_identity_sha256": compact_sha(identity)})
            events.append({"schema": SCHEMA, "event": "PREPARED", "release_id": prepared_id,
                "source_active_hold_sha256": active_sha,
                "source_snapshot_sha256": gate["source_snapshot_sha256"],
                "principal_identity_sha256": compact_sha(identity)})
        release_sha = _create_immutable_json(release_path, release)
        if fault == "after_release_artifact":
            raise OSError("injected failure after durable release artifact")
        if not any(item.get("event") == "RELEASE_ARTIFACT_DURABLE" for item in events):
            _append_durable_jsonl(ledger, {"schema": SCHEMA,
                "event": "RELEASE_ARTIFACT_DURABLE", "release_id": prepared_id,
                "release_sha256": release_sha, "released_hold_projection_sha256": released_sha})
            events.append({"schema": SCHEMA, "event": "RELEASE_ARTIFACT_DURABLE",
                "release_id": prepared_id, "release_sha256": release_sha,
                "released_hold_projection_sha256": released_sha})
        if not any(item.get("event") == "HOLD_RELEASED" for item in events):
            _append_durable_jsonl(ledger, {"schema": SCHEMA, "event": "HOLD_RELEASED",
                "release_id": prepared_id, "released_hold_projection_sha256": released_sha})
            events.append({"schema": SCHEMA, "event": "HOLD_RELEASED",
                "release_id": prepared_id, "released_hold_projection_sha256": released_sha})
        _atomic_json(control / "global_hold_release.json", release)
        _fsync_file_parent(control / "global_hold_release.json")
        if not any(item.get("event") == "COMMITTED" for item in events):
            _append_durable_jsonl(ledger, {"schema": SCHEMA, "event": "COMMITTED",
                "release_id": prepared_id, "release_sha256": release_sha,
                "released_hold_projection_sha256": released_sha,
                "gate_sha256": sha256_file(gate_file)})
        gate.update({"launch_authorized": True, "release_sha256": release_sha,
                     "released_hold_projection_sha256": released_sha,
                     "release_ledger_sha256": sha256_file(ledger)})
        _atomic_json(gate_file, gate)
        _fsync_file_parent(gate_file)
        if fault == "after_gate_authorization":
            raise OSError("injected failure after gate authorization with hold still active")
        current_hold = _read_json(hold_path)
        if current_hold.get("active") is True:
            if sha256_file(hold_path) != active_sha:
                raise PermissionError("active hold changed during release transaction")
            _atomic_json(hold_path, released)
            _fsync_file_parent(hold_path)
        elif current_hold != released:
            raise PermissionError("inactive hold is not the exact released projection")
        if fault == "after_hold_release":
            raise OSError("injected failure after inactive hold projection")
        validate_current_gate(_read_json(gate_file), runs_root=root, require_release=True)
        return {"status": "COMMITTED", "release_id": prepared_id,
                "release_sha256": release_sha, "released_hold_sha256": released_sha}
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def append_v14_hold(*, runs_root: os.PathLike | str = RUNS,
                    event: Mapping[str, object]) -> dict[str, Any]:
    """Append a failure hold, preserving prior events and exact H51 protection."""
    root = _runs_root(runs_root)
    path = root / CONTROL_DIR / "global_hold.json"
    hold = _read_json(path)
    events = hold.get("events")
    if not isinstance(events, list):
        raise PermissionError("existing hold events are malformed")
    item = dict(event)
    item["preserved_held_run_ids"] = list(H51_RUN_IDS)
    item["preserved_held_run_ids_sha256"] = H51_SHA256
    events.append(item)
    hold.update({"active": True, "events": events,
                 "preserved_held_run_ids": list(H51_RUN_IDS),
                 "preserved_held_run_ids_sha256": H51_SHA256,
                 "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    _atomic_json(path, hold)
    return hold


def held_status_projection(payload: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    """Create a status projection with only the approved administrative fields."""
    if run_id not in NEW_INCIDENT_KEYS:
        raise PermissionError("v1.4 may administer only the two new incident keys")
    original = dict(payload)
    original_status = "RUNNING" if run_id == SIGSEGV_KEY else "INTERRUPTED"
    if original.get("status") != original_status:
        raise PermissionError("incident source status changed")
    result = dict(original)
    result.update({
        "status": "RECOVERY_BLOCKED",
        "administrative_status": "RECOVERY_BLOCKED",
        "administrative_reason": "F0 v1.4 H51 post-abnormal permanent recovery/retry exclusion",
        "administrative_migration_schema": SCHEMA,
        "administrative_migration_id": "f0-v14-h51-postabnormal",
        "resume_authorized": False, "retry_authorized": False,
        "new_attempt_authorized": False, "checkpoint_load_authorized": False,
        "formal_authorized": False, "test_authorized": False,
    })
    return result


def _expected_h51_hold(snapshot: Mapping[str, Any], migration_id: str,
                       snapshot_sha256: str) -> dict[str, Any]:
    records = snapshot.get("source_files")
    if not isinstance(records, list):
        raise PermissionError("frozen source snapshot omits original hold bytes")
    rec = next((row for row in records if isinstance(row, Mapping)
                and row.get("path") == "execution_control/global_hold.json"), None)
    if not isinstance(rec, Mapping):
        raise PermissionError("frozen source snapshot has no original hold record")
    raw = base64.b64decode(str(rec.get("raw_base64", "")), validate=True)
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SOURCE_HOLD_SHA256:
        raise PermissionError("frozen original active hold bytes do not match fixed source")
    hold = json.loads(raw)
    hold["active"] = True
    hold["preserved_held_run_ids"] = list(H51_RUN_IDS)
    hold["preserved_held_run_ids_sha256"] = H51_SHA256
    hold.setdefault("events", []).append({
        "event": "V14_H51_ADMIN_MIGRATION_COMMITTED",
        "migration_schema": SCHEMA, "migration_id": migration_id,
        "held_run_ids": list(H51_RUN_IDS), "held_run_ids_sha256": H51_SHA256,
        "source_snapshot_sha256": snapshot_sha256,
        "automatic_retry_authorized": False, "recovery_authorized": False,
    })
    return hold


def _verify_frozen_source_pointers(snapshot: Mapping[str, Any], *, root: Path,
                                   migration_id: str | None = None,
                                   snapshot_sha256: str | None = None,
                                   expected_mutable: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    records = snapshot.get("source_files")
    if not isinstance(records, list):
        raise PermissionError("frozen source pointer inventory is malformed")
    repo = Path(__file__).resolve().parents[1]
    for record in records:
        if not isinstance(record, Mapping):
            raise PermissionError("frozen source pointer row is malformed")
        rel = str(record.get("path", ""))
        path = repo / rel if rel.startswith("experiments/") else (root / rel).resolve()
        if not path.is_file():
            raise PermissionError("frozen source pointer disappeared: " + rel)
        current_sha = sha256_file(path)
        if current_sha == record.get("sha256"):
            continue
        expected = (expected_mutable or {}).get(rel)
        if expected is not None and _read_json(path) == dict(expected):
            continue
        if rel == "execution_control/global_hold.json" and migration_id and snapshot_sha256:
            expected = _expected_h51_hold(snapshot, migration_id, snapshot_sha256)
            if _read_json(path) == expected:
                continue
        raise PermissionError("frozen source pointer changed before/within migration: " + rel)


def apply_administrative_migration(*, runs_root: os.PathLike | str = RUNS,
                                   snapshot_path: os.PathLike | str,
                                   gate_path: os.PathLike | str,
                                   reviewer_path: os.PathLike | str,
                                   reviewer_sha256: str,
                                   fault: str | None = None) -> dict[str, Any]:
    """Execute the narrowly scoped H51 status migration after source review.

    The caller must provide a frozen source snapshot, a reviewed v1.4 gate,
    and a separate reviewer verdict. This function does not recover or launch.
    """
    root = _runs_root(runs_root)
    snapshot_file = Path(snapshot_path)
    control = root / CONTROL_DIR
    ledger = control / "postabnormal_v14_migration.jsonl"
    prior_events: list[dict[str, Any]] = []
    lock_names = [root / CONTROL_DIR / "execution_state.lock"]
    lock_names += [root / "worker_locks" / f"worker_{i}.lock" for i in (0, 1)]
    lock_names += [root / "gpu_locks" / f"gpu{i}.lock" for i in (0, 1)]
    lock_names += [root / "canonical_locks" / f"{rid}.lock" for rid in NEW_INCIDENT_KEYS]
    control.mkdir(parents=True, exist_ok=True)
    lock_handles = []
    try:
        for path in lock_names:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise PermissionError("v1.4 migration lock is busy: " + str(path)) from exc
            lock_handles.append(handle)
        snapshot = _read_json(snapshot_file)
        validate_source_snapshot(snapshot, runs_root=root)
        reviewer_file = Path(reviewer_path)
        reviewer = _read_json(reviewer_file)
        if sha256_file(reviewer_file) != reviewer_sha256:
            raise PermissionError("independent review file hash differs")
        _validate_implementation_review(reviewer, snapshot=snapshot,
            snapshot_sha256=sha256_file(snapshot_file))
        gate = _read_json(Path(gate_path))
        if (gate.get("schema") != "F0-EXECUTION-ISOLATION-v1.4"
                or gate.get("source_snapshot_sha256") != sha256_file(snapshot_file)
                or gate.get("independent_review_sha256") != reviewer_sha256
                or gate.get("independent_review_verdict") != reviewer.get("verdict")
                or Path(gate.get("independent_review_path", "")).resolve()
                    != reviewer_file.resolve()
                or gate.get("launch_authorized") is not False
                or gate.get("preserved_held_run_ids") != list(H51_RUN_IDS)
                or gate.get("preserved_held_run_ids_sha256") != H51_SHA256):
            raise PermissionError("migration gate is not the reviewed pending-release H51 gate")
        if ledger.exists():
            prior_events = _read_jsonl(ledger, repair_partial=True)
        prepared_id_for_sources = next((item.get("migration_id") for item in prior_events
                                        if item.get("event") == "PREPARED"), None)
        committed_rows = [item for item in prior_events if item.get("event") == "COMMITTED"]
        try:
            _verify_frozen_source_pointers(snapshot, root=root,
                migration_id=prepared_id_for_sources, snapshot_sha256=sha256_file(snapshot_file))
        except PermissionError as exc:
            if committed_rows:
                raise PermissionError("committed v1.4 migration source pointer changed") from exc
            raise
        if committed_rows:
            prepared_rows = [item for item in prior_events if item.get("event") == "PREPARED"]
            if len(committed_rows) != 1 or len(prepared_rows) != 1:
                raise PermissionError("migration ledger has duplicate or missing transaction heads")
            committed = committed_rows[0]
            hold_path = control / "global_hold.json"
            hold = _read_json(hold_path)
            if (committed.get("source_snapshot_sha256") != sha256_file(snapshot_file)
                    or committed.get("h51_ids") != list(H51_RUN_IDS)
                    or committed.get("h51_sha256") != H51_SHA256
                    or hold != _expected_h51_hold(snapshot, prepared_rows[0]["migration_id"],
                                                  sha256_file(snapshot_file))
                    or committed.get("hold_sha256") != sha256_file(hold_path)):
                raise PermissionError("committed v1.4 migration no longer matches exact hold projection")
            for rid in NEW_INCIDENT_KEYS:
                status_path = root / STATUS_REL / (rid + ".json")
                if (_read_json(status_path) != held_status_projection(
                        snapshot["incident_statuses"][rid]["projection"], run_id=rid)
                        or committed.get("statuses", {}).get(rid) != sha256_file(status_path)):
                    raise PermissionError("committed v1.4 status projection changed: " + rid)
                _fsync_file_parent(status_path)
            _fsync_file_parent(hold_path)
            _fsync_file_parent(ledger)
            return {"status": "COMMITTED", "ledger_sha256": sha256_file(ledger),
                    "idempotent": True}
        prepared_rows = [item for item in prior_events if item.get("event") == "PREPARED"]
        if len(prepared_rows) > 1:
            raise PermissionError("v1.4 migration ledger contains duplicate PREPARED events")
        hold_path = control / "global_hold.json"
        if prepared_rows:
            prepared = prepared_rows[0]
            if (prepared.get("schema") != SCHEMA
                    or prepared.get("h51_ids") != list(H51_RUN_IDS)
                    or prepared.get("h51_sha256") != H51_SHA256
                    or prepared.get("source_snapshot_sha256") != sha256_file(snapshot_file)
                    or prepared.get("reviewer_sha256") != reviewer_sha256):
                raise PermissionError("existing v1.4 PREPARED source graph differs")
            hold_now = _read_json(hold_path)
            expected_hold = _expected_h51_hold(snapshot, prepared["migration_id"],
                                               sha256_file(snapshot_file))
            raw_hold = base64.b64decode(next(
                row["raw_base64"] for row in snapshot["source_files"]
                if row.get("path") == "execution_control/global_hold.json"), validate=True)
            if hold_now not in (json.loads(raw_hold), expected_hold):
                raise PermissionError("existing H51 hold is neither source nor exact partial commit")
        else:
            if sha256_file(hold_path) != EXPECTED_SOURCE_HOLD_SHA256:
                raise PermissionError("source hold changed before PREPARED migration")
            prepared = {"schema": SCHEMA, "event": "PREPARED", "h51_ids": list(H51_RUN_IDS),
                        "h51_sha256": H51_SHA256,
                        "migration_id": "f0-v14-h51-admin-" + uuid.uuid4().hex,
                        "source_snapshot_sha256": sha256_file(snapshot_file),
                        "reviewer_path": str(reviewer_path), "reviewer_sha256": reviewer_sha256,
                        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            _append_durable_jsonl(ledger, prepared)
            prior_events.append(prepared)
            if fault == "after_prepared":
                raise OSError("injected v1.4 failure after PREPARED")
        for rid in NEW_INCIDENT_KEYS:
            status_path = root / STATUS_REL / (rid + ".json")
            current = _read_json(status_path)
            source_projection = snapshot["incident_statuses"][rid]["projection"]
            admin_projection = held_status_projection(source_projection, run_id=rid)
            current_sha = sha256_file(status_path)
            if current_sha != EXPECTED_STATUS_SHA256[rid] and current != admin_projection:
                raise PermissionError("incident status is not source or exact partial migration projection")
            if current_sha == EXPECTED_STATUS_SHA256[rid] and current != source_projection:
                raise PermissionError("incident source bytes/projection disagree before migration")
        before = {rid: _read_json(root / STATUS_REL / (rid + ".json"))
                  for rid in NEW_INCIDENT_KEYS}
        source_projections = snapshot.get("incident_statuses", {})
        if not isinstance(source_projections, Mapping):
            raise PermissionError("source snapshot omits incident status projections")
        after = {}
        for rid in NEW_INCIDENT_KEYS:
            source_record = source_projections.get(rid, {})
            source_projection = source_record.get("projection") if isinstance(
                source_record, Mapping) else None
            if not isinstance(source_projection, Mapping):
                raise PermissionError("source snapshot incident status projection is malformed")
            updated = held_status_projection(source_projection, run_id=rid)
            after[rid] = updated
            status_path = root / STATUS_REL / (rid + ".json")
            current = _read_json(status_path)
            current_sha = sha256_file(status_path)
            if current_sha == EXPECTED_STATUS_SHA256[rid]:
                if current != source_projection:
                    raise PermissionError("incident source bytes/projection disagree")
                _atomic_json(status_path, updated)
                if fault == "after_status_write_" + rid:
                    raise OSError("injected v1.4 failure after status rename/fsync")
                _fsync_file_parent(status_path)
            elif current == updated:
                _fsync_file_parent(status_path)
            else:
                raise PermissionError("incident status is neither source nor exact v1.4 projection")
            transitioned = [item for item in prior_events
                            if item.get("event") == "STATUS_TRANSITION_DURABLE"
                            and item.get("run_id") == rid]
            if len(transitioned) > 1:
                raise PermissionError("v1.4 migration has duplicate status transition records")
            if not transitioned:
                event = {"schema": SCHEMA, "event": "STATUS_TRANSITION_DURABLE",
                         "migration_id": prepared["migration_id"],
                         "run_id": rid,
                         "before_sha256": EXPECTED_STATUS_SHA256[rid],
                         "after_sha256": sha256_file(status_path),
                         "status_file_fsynced": True,
                         "status_directory_fsynced": True,
                         "reconciled_after_interruption": current == updated}
                _append_durable_jsonl(ledger, event)
                prior_events.append(event)
            if fault == "after_status_" + rid:
                raise OSError("injected v1.4 failure after status transition")
        hold = _read_json(hold_path)
        hold_events = hold.get("events")
        if not isinstance(hold_events, list):
            raise PermissionError("active hold event list is malformed")
        v14_events = [item for item in hold_events if isinstance(item, Mapping)
                      and item.get("event") == "V14_H51_ADMIN_MIGRATION_COMMITTED"]
        if len(v14_events) > 1:
            raise PermissionError("active hold contains duplicate v1.4 migration events")
        if not v14_events:
            if fault == "before_hold_write":
                raise OSError("injected v1.4 failure before H51 hold projection")
            hold["active"] = True
            hold["preserved_held_run_ids"] = list(H51_RUN_IDS)
            hold["preserved_held_run_ids_sha256"] = H51_SHA256
            hold.setdefault("events", []).append({
                "event": "V14_H51_ADMIN_MIGRATION_COMMITTED",
                "migration_schema": SCHEMA, "migration_id": prepared["migration_id"],
                "held_run_ids": list(H51_RUN_IDS),
                "held_run_ids_sha256": H51_SHA256,
                "source_snapshot_sha256": sha256_file(snapshot_file),
                "automatic_retry_authorized": False, "recovery_authorized": False,
            })
            _atomic_json(hold_path, hold)
            if fault == "after_hold_write":
                raise OSError("injected v1.4 failure after H51 hold projection")
        elif (hold.get("preserved_held_run_ids") != list(H51_RUN_IDS)
              or hold.get("preserved_held_run_ids_sha256") != H51_SHA256
              or hold.get("active") is not True):
            raise PermissionError("existing v1.4 hold transition differs from exact H51")
        if hold != _expected_h51_hold(snapshot, prepared["migration_id"],
                                      sha256_file(snapshot_file)):
            raise PermissionError("active H51 hold differs from exact frozen-source projection")
        # A partial migration may encounter a hold rename whose directory
        # durability was interrupted; reestablish both file and parent fsync
        # before the ledger is allowed to record H51_HOLD_DURABLE.
        _fsync_file_parent(hold_path)
        hold_durable = [item for item in prior_events if item.get("event") == "H51_HOLD_DURABLE"]
        if not hold_durable:
            h_event = {"schema": SCHEMA, "event": "H51_HOLD_DURABLE",
                       "migration_id": prepared["migration_id"],
                       "hold_sha256": sha256_file(hold_path)}
            _append_durable_jsonl(ledger, h_event)
            prior_events.append(h_event)
        if fault == "after_hold":
            raise OSError("injected v1.4 failure after H51 hold")
        # Fail-closed completion proof: all protected status rows must have the
        # exact administrative projection and no extra run may be touched.
        for rid in NEW_INCIDENT_KEYS:
            current = _read_json(root / STATUS_REL / (rid + ".json"))
            if current != after[rid] or current.get("status") != "RECOVERY_BLOCKED":
                raise PermissionError("v1.4 status projection did not commit exactly")
        if hold.get("active") is not True or hold.get("preserved_held_run_ids") != list(H51_RUN_IDS):
            raise PermissionError("v1.4 active H51 hold failed final verification")
        validate_source_snapshot(snapshot, runs_root=root)
        _verify_frozen_source_pointers(snapshot, root=root,
            migration_id=prepared["migration_id"], snapshot_sha256=sha256_file(snapshot_file))
        committed = {"schema": SCHEMA, "event": "COMMITTED",
                     "migration_id": prepared["migration_id"],
                     "h51_ids": list(H51_RUN_IDS), "h51_sha256": H51_SHA256,
                     "source_snapshot_sha256": sha256_file(snapshot_file),
                     "hold_sha256": sha256_file(hold_path),
                     "statuses": {rid: sha256_file(root / STATUS_REL / (rid + ".json"))
                                  for rid in NEW_INCIDENT_KEYS},
                     "committed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _append_durable_jsonl(ledger, committed)
        return {"status": "COMMITTED", "ledger_sha256": sha256_file(ledger),
                "hold_sha256": committed["hold_sha256"], "idempotent": False}
    finally:
        for handle in reversed(lock_handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def _append_durable_jsonl(path: Path, item: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _read_jsonl(path, repair_partial=True)
    payload = json.dumps(dict(item), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("short v1.4 migration ledger write")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _read_jsonl(path: Path, *, repair_partial: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        if not repair_partial:
            raise PermissionError("append-only ledger has an unterminated tail")
        boundary = raw.rfind(b"\n") + 1
        prefix = raw[:boundary]
        try:
            parsed_prefix = [json.loads(line) for line in prefix.splitlines() if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PermissionError("append-only ledger has malformed complete records") from exc
        if any(not isinstance(row, dict) for row in parsed_prefix):
            raise PermissionError("append-only ledger record is not an object")
        fd = os.open(path, os.O_WRONLY)
        try:
            os.ftruncate(fd, boundary)
            os.fsync(fd)
        finally:
            os.close(fd)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        raw = prefix
    try:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError("append-only ledger contains a malformed complete record") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise PermissionError("append-only ledger record is not an object")
    return rows


def _fsync_file_parent(path: Path) -> None:
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


def _create_immutable_json(path: Path, payload: Mapping[str, object]) -> str:
    """Create a new evidence artifact without replacing any existing bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False).encode("utf-8") + b"\n"
    fd, temp_name = tempfile.mkstemp(prefix="." + path.name + ".tmp.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o644)
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short immutable evidence write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise FileExistsError(
                    "refusing to overwrite different immutable evidence: " + str(path))
            _fsync_file_parent(path)
            return sha256_file(path)
        _fsync_file_parent(path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
    return sha256_file(path)


def write_source_snapshot(*, runs_root: os.PathLike | str = RUNS,
                          path: os.PathLike | str | None = None) -> dict[str, Any]:
    snapshot = source_snapshot(runs_root=runs_root)
    target = Path(path) if path is not None else (
        _runs_root(runs_root) / CONTROL_DIR / "postabnormal_v14_source_snapshot.json")
    digest = _create_immutable_json(target, snapshot)
    return {"path": str(target), "sha256": digest, "snapshot": snapshot}


def write_incident_audit(*, runs_root: os.PathLike | str = RUNS,
                         path: os.PathLike | str | None = None) -> dict[str, Any]:
    """Persist the raw v1.4 incident identity audit, never a training result."""
    snapshot = source_snapshot(runs_root=runs_root)
    audit = {
        "schema": "F0-POSTABNORMAL-WORKER-AUDIT-v1.4",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design_sha256": snapshot["design_sha256"],
        "queue_sha256": QUEUE_SHA256,
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "h49_ids": list(PRESERVED_H49_RUN_IDS), "h49_sha256": PRESERVED_H49_SHA256,
        "h51_ids": list(H51_RUN_IDS), "h51_sha256": H51_SHA256,
        "baseline_counts_before": {"VALIDATION_COMPLETE": 97, "RECOVERY_BLOCKED": 49,
                                    "UNCLAIMED": 140, "RUNNING": 1, "INTERRUPTED": 1},
        "baseline_counts_after": {"VALIDATION_COMPLETE": 97, "RECOVERY_BLOCKED": 51,
                                   "UNCLAIMED": 140},
        "sigsegv_worker": {"run_id": SIGSEGV_KEY, "nonce": "34de4e7e0970459a84a6457ff8ed490f",
                            "exit_code": -11, "cause": "UNKNOWN",
                            "status_sha256": EXPECTED_STATUS_SHA256[SIGSEGV_KEY],
                            "manifest_sha256": EXPECTED_MANIFEST_SHA256[SIGSEGV_KEY],
                            "failure_sha256": EXPECTED_FAILURE_SHA256[SIGSEGV_KEY],
                            "receipt_exists": False},
        "peer_stopped_worker": {"run_id": PEER_KEY, "nonce": "1787964e065e44bfbc503c26adca3e97",
                                 "exit_code": 143, "status_sha256": EXPECTED_STATUS_SHA256[PEER_KEY],
                                 "manifest_sha256": EXPECTED_MANIFEST_SHA256[PEER_KEY],
                                 "failure_sha256": EXPECTED_FAILURE_SHA256[PEER_KEY],
                                 "receipt_path": PEER_RECEIPT_REL,
                                 "receipt_sha256": PEER_RECEIPT_SHA256,
                                 "receipt_status": "INTERRUPTED", "last_complete_epoch": 32,
                                 "checkpoint_sha256": "52a6559c4466a8e19b11c34e0d6ba12f38e09716b8d39e5f9e357a7dbb7076d4"},
        "completed_ids_sha256": snapshot["completed_ids_sha256"],
        "completed_inventory_sha256": snapshot["completed_inventory_sha256"],
        "fresh_ids_sha256": snapshot["fresh_ids_sha256"],
        "incident_inventory_sha256": snapshot["incident_inventory_sha256"],
        "source_hold_sha256": EXPECTED_SOURCE_HOLD_SHA256,
        "launch_authorized": False, "recovery_authorized": False,
        "retry_authorized": False, "formal_authorized": False,
        "test_authorized": False,
    }
    audit["audit_sha256"] = compact_sha(audit)
    target = Path(path) if path is not None else (
        _runs_root(runs_root) / CONTROL_DIR / "postabnormal_sigsegv_audit_v6_fresh_search.json")
    digest = _create_immutable_json(target, audit)
    return {"path": str(target), "sha256": digest, "audit": audit}


__all__ = ["SCHEMA", "SIGSEGV_KEY", "PEER_KEY", "NEW_INCIDENT_KEYS", "H51_RUN_IDS",
           "H51_SHA256", "QUEUE_SHA256", "COMPLETED_KEYS_SHA256", "FRESH_KEYS_SHA256",
           "require_exact_h51", "source_snapshot", "validate_source_snapshot",
           "write_source_snapshot", "write_incident_audit", "validate_current_gate",
           "ensure_current_release", "append_v14_hold",
           "held_status_projection", "apply_administrative_migration"]
