"""Versioned, fail-closed F0 v1.5 successor execution control.

This module deliberately imports only the Python standard library.  It keeps
the successor evidence and transaction separate from the historical v1.4
release graph.  Production admission remains closed until a reviewed v1.5
terminal graph has been durably published.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import hashlib
import hmac
import json
import math
import os
import secrets
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/stage_F0_outputs"
RUNS = OUTPUT / "runs"
CONTROL = RUNS / "execution_control"
SCHEMA = "F0-EXECUTION-ISOLATION-v1.5"
OBSERVATION_STAGES = frozenset(("LOCKED_RELEASE_DECISION",
                                "PRE_HOLD_RELEASE_DECISION"))
SOURCE_SCHEMA = "F0-SUCCESSOR-SOURCE-v1.5"
PRELAUNCH_SCHEMA = "F0-PRELAUNCH-AUDIT-v1.5"
RELEASE_SCHEMA = "F0-GLOBAL-HOLD-RELEASE-v1.5"
LEDGER_SCHEMA = "F0-SUCCESSOR-RELEASE-LEDGER-v1.5"
DEFER_SCHEMA = "F0-PRECLAIM-RESOURCE-DEFERRED-v1.5"
SCOPE = "F0-B1-ML search only; fresh UNCLAIMED canonical keys only"
PER_DEVICE_RULE = "assigned physical GPU UUID must match the reviewed target and satisfy 10*used_bytes < 9*total_bytes at each fresh decision"
APPROVAL_RECORD = "experiments/stage_F0_outputs/runs/execution_control/bootstrap_amendment_remediation_v1.json"
APPROVAL_RECORD_SHA256 = "0bdc330cd594b5c8ac519888fc93182047f1302eeee6a43ce2e8cfc8fc165138"
REMEDIATION_CANDIDATE = "experiments/stage_F0_outputs/execution_isolation_gate_bootstrap_remediation_v1_no_go.json"
REMEDIATION_CANDIDATE_SHA256 = "7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef"
DESIGN_PATH = "experiments/reports/stage_F0_v15_successor_control_design_v2.md"
DESIGN_SHA256 = "8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316"
DESIGN_REVIEW_PATH = "experiments/reports/stage_F0_v15_successor_control_design_review_v2.md"
DESIGN_REVIEW_SHA256 = "9dd314ddd4b1ae7622436e00195888ab682089e65bf68649816afaeaeef392eb"
QUEUE_REL = "experiments/stage_F0_outputs/queues/F0-B1-ML-search-queue.json"
SEARCH_GATE_REL = "experiments/stage_F0_outputs/run_gate_search.json"
CANONICAL_GATE_REL = "experiments/stage_F0_outputs/execution_isolation_gate.json"
HOLD_REL = "experiments/stage_F0_outputs/runs/execution_control/global_hold.json"
RELEASE_REL = "experiments/stage_F0_outputs/runs/execution_control/global_hold_release.json"
RELEASE_LEDGER_REL = "experiments/stage_F0_outputs/runs/execution_control/postbootstrap_v15_release.jsonl"
OBSERVATION_LEDGER_REL = "experiments/stage_F0_outputs/runs/execution_control/postbootstrap_v15_runtime_observations.jsonl"
V14_RELEASE_SHA256 = "4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63"
V14_RELEASE_LEDGER_SHA256 = "9bdbb2bc801c33a36a6f950e66f6a4a13ba11faa21581a683545e4c9103dc71e"
V14_MIGRATION_LEDGER_SHA256 = "b91d912c3d7f9be9b896e19c6fa7744e92872c2ea870decedef00a5ac1087e30"
V14_SOURCE_SNAPSHOT_SHA256 = "717ea4baeda17e89f684004ded3b230aff30826f6649b4ebb5d592c0d2984533"
V14_RELEASE_POINTER_SHA256 = "4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63"
EXPECTED_MIGRATION_SOURCE_SHA256 = "9e4c903277505a0e513f3e5f673ce3e263371722bf807a4a0cd7c4b2e4d103a0"
SEARCH_GATE_SHA256 = "958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e"
QUEUE_SHA256 = "43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23"
INCIDENT_HOLD_SHA256 = "e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51"
INCIDENT_HOLD_ID = "cd2dbf50-3a50-4b81-b976-38f1b03991a3"
INCIDENT_NONCE = "8fc4a33fc7574a368312b4965e24a283"
INCIDENT_FAILURE_SHA256 = "1ee4f54f83dd4fa6f9b6615e1e7f074be2178ea5e160263aa2aeae14296ed1f4"
INCIDENT_STDERR_SHA256 = "2798981cb0b8c4c650c3cb2aa427db201aeb6038ffa2000e8ad75759056dab9a"
SCIENTIFIC_SHA256 = "b656a1bfc931259c4a0ec81f150fce518f973aa178363536f676bed1672e3f19"
EXPECTED_V14_SOURCE_SHA256 = "5380ef1c3698f8b4ddfe34fce1bd8a4de15c04ad62f91a156bfcb273bb53298d"
MAX_GPU_INDEX = 1
MAX_TASK_PYTHON_PROCESSES = 5
MEMORY_FRESH_SECONDS = 60.0

# A distinct, deterministic v1.5 inventory.  Do not reuse v1.4's frozen list.
V15_CONTROL_FILES = (
    "experiments/stage_F0_successor_v15.py",
    "experiments/stage_F0_postabnormal_v14.py",
    "experiments/stage_F0_migration.py",
    "experiments/stage_F0_worker_control.py",
    "experiments/stage_F0_queue.py",
    "experiments/stage_F0_run.py",
    "experiments/stage_F0_outputs/runs/run_search_claim_worker.py",
    "experiments/stage_F0_outputs/runs/run_search_trajectory_child.py",
    "experiments/stage_F0_outputs/runs/run_search_worker.sh",
    "experiments/stage_F0_outputs/runs/run_f0_python_task.sh",
    "experiments/stage_F0_outputs/preflight/run_first_search_epoch.py",
    "experiments/tests/test_stage_F0_successor_v15.py",
    "experiments/tests/conftest.py",
    "experiments/tests/test_stage_F0_postabnormal_v14.py",
    "experiments/tests/test_stage_F0_migration.py",
    "experiments/tests/test_stage_F0_launcher.py",
    "experiments/tests/test_stage_F0_queue.py",
    "experiments/tests/test_stage_F0_run.py",
    "experiments/tests/test_stage_F0_worker_control.py",
    DESIGN_PATH,
)


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _read_json(path: os.PathLike | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PermissionError("expected a JSON object: " + str(path))
    return value


def _repo_path(root: os.PathLike | str, relative: str) -> Path:
    base = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or not relative or ".." in rel.parts:
        raise PermissionError("evidence path escapes repository root")
    path = base / rel
    if path != base and base not in path.parents:
        raise PermissionError("evidence path escapes repository root")
    current = base
    for part in rel.parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise PermissionError("evidence path contains a symlink: " + relative)
    return path


def control_hash_inventory_v15(root: os.PathLike | str = ROOT) -> dict[str, str]:
    base = Path(root).resolve()
    if len(V15_CONTROL_FILES) != len(set(V15_CONTROL_FILES)):
        raise RuntimeError("v1.5 control inventory contains duplicate paths")
    output: dict[str, str] = {}
    for name in V15_CONTROL_FILES:
        path = _repo_path(base, name)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("v1.5 control inventory input is missing/non-regular: " + name)
        output[name] = sha256_file(path)
    return output


def runtime_hash_inventory(root: os.PathLike | str = ROOT) -> dict[str, str]:
    """Return the frozen scientific/runtime code domain used by the queue."""
    import sys
    exp = str(Path(root).resolve() / "experiments")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    from stage_F0_queue import code_hash_inventory
    inventory = code_hash_inventory(require_tests=True)
    if not inventory:
        raise PermissionError("complete runtime inventory is empty")
    return dict(sorted(inventory.items()))


def runtime_hash(root: os.PathLike | str = ROOT) -> str:
    return sha256_json(runtime_hash_inventory(root))


def _assert_frozen_artifacts(root: os.PathLike | str, *,
                             allow_released_hold: bool = False,
                             allow_gate_replaced: bool = False) -> dict[str, str]:
    expected = {
        DESIGN_PATH: DESIGN_SHA256,
        DESIGN_REVIEW_PATH: DESIGN_REVIEW_SHA256,
        APPROVAL_RECORD: APPROVAL_RECORD_SHA256,
        REMEDIATION_CANDIDATE: REMEDIATION_CANDIDATE_SHA256,
        QUEUE_REL: QUEUE_SHA256,
        SEARCH_GATE_REL: SEARCH_GATE_SHA256,
        "experiments/stage_F0_outputs/execution_isolation_gate.json":
            "f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64",
        HOLD_REL: INCIDENT_HOLD_SHA256,
        "experiments/stage_F0_outputs/runs/execution_control/global_hold_release_v14.json": V14_RELEASE_SHA256,
        "experiments/stage_F0_outputs/runs/execution_control/global_hold_release.json": V14_RELEASE_POINTER_SHA256,
        "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_release.jsonl": V14_RELEASE_LEDGER_SHA256,
        "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_migration.jsonl": V14_MIGRATION_LEDGER_SHA256,
        "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_source_snapshot.json": V14_SOURCE_SNAPSHOT_SHA256,
        "experiments/stage_F0_postabnormal_v14.py": EXPECTED_V14_SOURCE_SHA256,
        "experiments/stage_F0_migration.py": EXPECTED_MIGRATION_SOURCE_SHA256,
        "experiments/stage_F0_outputs/runs/worker_logs/worker_1_failure_1791870436_8fc4a33fc7574a368312b4965e24a283.json": INCIDENT_FAILURE_SHA256,
        "experiments/stage_F0_outputs/runs/worker_logs/trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.stderr.log": INCIDENT_STDERR_SHA256,
        "experiments/stage_F0_outputs/runs/worker_logs/trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.phases.jsonl": "49ddf8f36f62da00fb3f19d1105ae5f390070c393b0744581e757142393e7cba",
        "experiments/stage_F0_outputs/runs/worker_logs/trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.stdout.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    base = Path(root).resolve()
    actual: dict[str, str] = {}
    for relative, digest in expected.items():
        path = base / relative
        if relative == CANONICAL_GATE_REL and (allow_released_hold or allow_gate_replaced):
            if path.is_file() and not path.is_symlink() and sha256_file(path) == digest:
                actual[relative] = digest
                continue
            archive = (base / "experiments/stage_F0_outputs/runs/execution_control" /
                       ("v15_preimage_gate_" + digest + ".bin"))
            if not archive.is_file() or archive.is_symlink() or sha256_file(archive) != digest:
                raise PermissionError("exact v1.4 canonical gate preimage archive is absent")
            actual[relative] = digest
            continue
        if relative == HOLD_REL and allow_released_hold:
            if not path.is_file() or path.is_symlink():
                raise PermissionError("released hold is not an ordinary file")
            archive = (base / "experiments/stage_F0_outputs/runs/execution_control" /
                       ("v15_preimage_hold_" + INCIDENT_HOLD_SHA256 + ".bin"))
            if not archive.is_file() or archive.is_symlink() or sha256_file(archive) != INCIDENT_HOLD_SHA256:
                raise PermissionError("exact active-hold preimage archive is absent")
            actual[relative] = INCIDENT_HOLD_SHA256
            continue
        if relative == RELEASE_REL and (allow_released_hold or allow_gate_replaced):
            archive = (base / "experiments/stage_F0_outputs/runs/execution_control" /
                       ("v15_preimage_release_pointer_" + digest + ".bin"))
            if path.is_file() and not path.is_symlink() and sha256_file(path) == digest:
                actual[relative] = digest
                continue
            if not archive.is_file() or archive.is_symlink() or sha256_file(archive) != digest:
                raise PermissionError("exact v1.4 release-pointer preimage archive is absent")
            actual[relative] = digest
            continue
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise PermissionError("frozen v1.4/incident evidence drift: " + relative)
        actual[relative] = digest
    hold = _read_json(base / HOLD_REL)
    if hold.get("hold_id") != INCIDENT_HOLD_ID:
        raise PermissionError("current v1.4 failure hold identity changed")
    if allow_released_hold:
        if (hold.get("active") is not False
                or hold.get("source_incident_hold_sha256") != INCIDENT_HOLD_SHA256):
            raise PermissionError("released hold is not the exact v1.5 source projection")
    elif hold.get("active") is not True or sha256_file(base / HOLD_REL) != INCIDENT_HOLD_SHA256:
        raise PermissionError("current v1.4 failure hold is not the frozen active source")
    failure = _read_json(base / "experiments/stage_F0_outputs/runs/worker_logs/worker_1_failure_1791870436_8fc4a33fc7574a368312b4965e24a283.json")
    if (failure.get("nonce") != INCIDENT_NONCE or failure.get("canonical_claim") is not None
            or failure.get("receipt_sha256") is not None):
        raise PermissionError("incident no-claim/no-receipt identity changed")
    receipt = base / "experiments/stage_F0_outputs/runs/worker_logs/trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.receipt.json"
    if os.path.lexists(receipt):
        raise PermissionError("incident receipt absence changed")
    return actual


def _validate_amendment(amendment: object, *, scientific_sha256: str,
                        runtime_sha256: str, control_sha256: str) -> None:
    if not isinstance(amendment, Mapping):
        raise PermissionError("v1.5 amendment is missing or malformed")
    if amendment.get("status") != "APPROVED":
        raise PermissionError("v1.5 amendment status must be APPROVED")
    if amendment.get("reviewer_verdict") not in ("PASS", "PASS WITH FIXES"):
        raise PermissionError("v1.5 amendment reviewer verdict is not approved")
    if amendment.get("frozen_scientific_code_hash") != scientific_sha256:
        raise PermissionError("v1.5 amendment scientific digest mismatch")
    if amendment.get("runtime_control_code_hash") != runtime_sha256:
        raise PermissionError("v1.5 amendment complete runtime digest mismatch")
    if amendment.get("control_hash_inventory_sha256") != control_sha256:
        raise PermissionError("v1.5 amendment control inventory digest mismatch")
    if amendment.get("approval_record_sha256") != APPROVAL_RECORD_SHA256:
        raise PermissionError("v1.5 amendment is not bound to frozen remediation approval")
    if amendment.get("remediation_candidate_sha256") != REMEDIATION_CANDIDATE_SHA256:
        raise PermissionError("v1.5 amendment is not bound to frozen no-go candidate")


def _candidate_core_projection(gate: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"implementation_review_path", "implementation_review_sha256",
                "staged_go_path", "staged_go_sha256", "approved_candidate_core_sha256",
                "launch_authorized", "transaction_state", "release_artifact_path",
                "release_sha256", "released_hold_sha256", "committed_ledger_sha256"}
    return {key: value for key, value in gate.items() if key not in excluded}


def _validate_review_graph(gate: Mapping[str, Any], *, root: Path,
                           runtime_inventory: Mapping[str, str],
                           control_inventory: Mapping[str, str]) -> dict[str, Any]:
    report_rel, report_sha = gate.get("implementation_report_path"), gate.get("implementation_report_sha256")
    review_rel, review_sha = gate.get("implementation_review_path"), gate.get("implementation_review_sha256")
    if not all(isinstance(value, str) and value for value in
               (report_rel, report_sha, review_rel, review_sha)):
        raise PermissionError("v1.5 implementation report/review binding is absent")
    report = _repo_path(root, report_rel)
    review_file = _repo_path(root, review_rel)
    if (not report.is_file() or report.is_symlink() or sha256_file(report) != report_sha
            or not review_file.is_file() or review_file.is_symlink()
            or sha256_file(review_file) != review_sha):
        raise PermissionError("v1.5 implementation report/review bytes differ")
    review = _read_json(review_file)
    candidate_core_sha = sha256_json(_candidate_core_projection(gate))
    if gate.get("approved_candidate_core_sha256") != candidate_core_sha:
        raise PermissionError("v1.5 gate candidate core digest differs")
    expected = {
        "verdict": ("PASS", "PASS WITH FIXES"),
        "design_path": DESIGN_PATH, "design_sha256": DESIGN_SHA256,
        "implementation_report_path": report_rel,
        "implementation_report_sha256": report_sha,
        "approved_candidate_core_sha256": candidate_core_sha,
        "runtime_hashes": dict(runtime_inventory),
        "runtime_hash_inventory_sha256": sha256_json(runtime_inventory),
        "control_hashes": dict(control_inventory),
        "control_hash_inventory_sha256": sha256_json(control_inventory),
        "changed_source_hashes": dict(control_inventory),
        "unresolved_blockers": [], "scientific_preservation_confirmed": True,
    }
    for key, value in expected.items():
        actual = review.get(key)
        if key == "verdict":
            if actual not in value:
                raise PermissionError("v1.5 independent review verdict is not approved")
        elif actual != value:
            raise PermissionError("v1.5 independent review binding mismatch: " + key)
    if not isinstance(review.get("reviewer_identity"), str) or not review["reviewer_identity"].strip():
        raise PermissionError("v1.5 independent reviewer identity is missing")
    if gate.get("implementation_review_sha256") != sha256_file(review_file):
        raise PermissionError("v1.5 final gate review hash differs")
    staged_rel, staged_sha = gate.get("staged_go_path"), gate.get("staged_go_sha256")
    if (not isinstance(staged_rel, str)
            or not staged_rel.startswith("experiments/stage_F0_outputs/runs/execution_control/staged_go_v15_")
            or not staged_rel.endswith(".json")
            or not isinstance(staged_sha, str) or len(staged_sha) != 64):
        raise PermissionError("v1.5 gate has no canonical staged-GO binding")
    staged_path = _repo_path(root, staged_rel)
    if (not staged_path.is_file() or staged_path.is_symlink()
            or staged_path.stat().st_mode & 0o222 or sha256_file(staged_path) != staged_sha):
        raise PermissionError("v1.5 staged-GO record bytes differ")
    staged = _read_json(staged_path)
    if (staged.get("schema") != "F0-V15-STAGED-GO-v1"
            or staged.get("verdict") != "GO"
            or staged.get("production_release_authorized") is not False
            or staged.get("candidate_core_sha256") != candidate_core_sha
            or staged.get("implementation_review_sha256") != review_sha
            or staged.get("implementation_report_sha256") != report_sha
            or staged.get("source_snapshot_sha256") != gate.get("source_snapshot_sha256")
            or staged.get("prelaunch_audit_sha256") != gate.get("prelaunch_audit_sha256")
            or staged.get("design_sha256") != DESIGN_SHA256
            or staged.get("reviewer_identity") != review.get("reviewer_identity")):
        raise PermissionError("v1.5 staged-GO graph does not bind the approved candidate")
    return review


def fresh_gpu_eligible(row: Mapping[str, Any]) -> bool:
    """Strict exact integer memory test; malformed rows are ineligible."""
    if row.get("readable") is not True or row.get("eligible") is False:
        return False
    total, used = row.get("memory_total_bytes"), row.get("memory_used_bytes")
    if (isinstance(total, bool) or isinstance(used, bool)
            or not isinstance(total, int) or not isinstance(used, int)
            or total <= 0 or used < 0 or used > total):
        return False
    if not isinstance(row.get("uuid"), str) or not row.get("uuid"):
        return False
    fraction = row.get("memory_used_fraction")
    if fraction is not None:
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            return False
        if not math.isfinite(float(fraction)) or not math.isclose(
                float(fraction), used / total, rel_tol=1e-9, abs_tol=1e-9):
            return False
    return 10 * used < 9 * total


def validate_target_rows(rows: Sequence[Mapping[str, Any]],
                         targets: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    by_index: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        by_index.setdefault(index, []).append(row)
    if not targets:
        raise PermissionError("v1.5 target set must be nonempty")
    accepted: dict[int, str] = {}
    seen_uuids: set[str] = set()
    for target in targets:
        index, target_uuid = target.get("index"), target.get("uuid")
        if (isinstance(index, bool) or not isinstance(index, int)
                or index not in (0, 1) or not isinstance(target_uuid, str) or not target_uuid):
            raise PermissionError("v1.5 target identity is invalid")
        matches = by_index.get(index, [])
        if len(matches) != 1:
            raise PermissionError("v1.5 target GPU row is absent or ambiguous")
        row = matches[0]
        if row.get("uuid") != target_uuid or not fresh_gpu_eligible(row):
            raise PermissionError("v1.5 target GPU identity/memory is not eligible")
        if target_uuid in seen_uuids:
            raise PermissionError("v1.5 physical GPU UUID is duplicated")
        seen_uuids.add(target_uuid)
        accepted[index] = target_uuid
    return accepted


def observation_is_fresh(*, started_monotonic: float, now_monotonic: float | None = None,
                         max_age: float = MEMORY_FRESH_SECONDS) -> bool:
    end = time.monotonic() if now_monotonic is None else float(now_monotonic)
    start = float(started_monotonic)
    return math.isfinite(start) and math.isfinite(end) and 0 <= end - start <= max_age


class ResourceDeferred(PermissionError):
    """A valid memory-threshold NO-GO before any canonical mutation."""
    def __init__(self, telemetry: Mapping[str, Any], boundary: str):
        super().__init__("v1.5 assigned GPU is at or above the 90 percent memory limit")
        self.telemetry = (dict(telemetry) if isinstance(telemetry, Mapping)
                          else {"devices": [dict(row) for row in telemetry]})
        self.boundary = boundary


def query_gpu_memory(*, gpu_index: int, expected_uuid: str,
                     provider=None) -> dict[str, Any]:
    """Query one physical GPU. No utilization or other-process data is read."""
    if isinstance(gpu_index, bool) or gpu_index not in (0, 1) or not expected_uuid:
        raise PermissionError("assigned physical GPU identity is invalid")
    started = time.monotonic()
    utc = _dt.datetime.now(_dt.timezone.utc).isoformat()
    if provider is None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--id=" + str(gpu_index),
                 "--query-gpu=index,uuid,memory.total,memory.used",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                check=False, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PermissionError("assigned GPU telemetry is unreadable") from exc
        if result.returncode != 0:
            raise PermissionError("assigned GPU telemetry command failed")
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise PermissionError("assigned GPU query returned ambiguous rows")
        fields = [item.strip() for item in lines[0].split(",")]
        if len(fields) != 4:
            raise PermissionError("assigned GPU telemetry row is malformed")
        try:
            index, gpu_uuid = int(fields[0]), fields[1]
            total = int(fields[2]) * (1 << 20)
            used = int(fields[3]) * (1 << 20)
        except ValueError as exc:
            raise PermissionError("assigned GPU integer telemetry is unreadable") from exc
    else:
        row = dict(provider(gpu_index))
        index, gpu_uuid = row.get("index"), row.get("uuid")
        total, used = row.get("memory_total_bytes"), row.get("memory_used_bytes")
    if index != gpu_index or gpu_uuid != expected_uuid:
        raise PermissionError("assigned GPU UUID/index identity drift")
    row = {"index": gpu_index, "uuid": gpu_uuid, "readable": True,
           "memory_total_bytes": total, "memory_used_bytes": used,
           "observed_at_utc": utc, "observation_started_monotonic": started}
    if (isinstance(total, bool) or isinstance(used, bool) or not isinstance(total, int)
            or not isinstance(used, int) or total <= 0 or used < 0 or used > total):
        raise PermissionError("assigned GPU integer telemetry is out of range")
    row["memory_used_fraction"] = used / total
    row["eligible"] = 10 * used < 9 * total
    if not observation_is_fresh(started_monotonic=started):
        raise PermissionError("assigned GPU observation aged out before decision")
    return row


def require_live_gpu_memory(*, runs_root: os.PathLike | str, gpu_index: int,
                            gpu_uuid: str, boundary: str,
                            provider=None) -> dict[str, Any]:
    telemetry = query_gpu_memory(gpu_index=gpu_index, expected_uuid=gpu_uuid,
                                 provider=provider)
    if not fresh_gpu_eligible(telemetry):
        raise ResourceDeferred(telemetry, boundary)
    return telemetry


def fresh_member_paths(output_root: os.PathLike | str, run_id: str) -> dict[str, Path]:
    root = Path(output_root).resolve()
    return {
        "status": root / "queue_status" / (run_id + ".json"),
        "lock": root / "canonical_locks" / (run_id + ".lock"),
        "run": root / run_id,
    }


def _canonical_key_identity_is_valid(entry: Mapping[str, Any]) -> bool:
    run_id = entry.get("run_id")
    canonical = entry.get("canonical_key_json")
    full_sha = entry.get("run_id_full_sha256")
    if (not isinstance(run_id, str) or not run_id
            or not isinstance(canonical, str) or not isinstance(full_sha, str)
            or len(full_sha) != 64
            or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != full_sha
            or run_id != full_sha[:20]):
        return False
    try:
        parsed = json.loads(canonical)
    except (TypeError, ValueError):
        return False
    return (isinstance(parsed, dict)
            and json.dumps(parsed, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")) == canonical)


def key_is_untouched(output_root: os.PathLike | str, entry: Mapping[str, Any]) -> bool:
    """Probe only: every prior path, including a dangling symlink, is history."""
    run_id = entry.get("run_id")
    if not _canonical_key_identity_is_valid(entry):
        return False
    return not any(os.path.lexists(path) for path in fresh_member_paths(output_root, run_id).values())


def frozen_f140_ids(root: os.PathLike | str = ROOT) -> list[str]:
    snapshot = _read_json(Path(root) / "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_source_snapshot.json")
    values = snapshot.get("fresh_ids")
    if (not isinstance(values, list) or len(values) != 140
            or any(not isinstance(item, str) for item in values)
            or values != sorted(set(values))
            or sha256_json(values) != "78e529b19e6fb9cc55cdc32524f3243005c0bf88f5dbe4206c44062a645f585b"):
        raise PermissionError("frozen F140 identity list is missing or changed")
    return values


def validate_fresh_entry(output_root: os.PathLike | str,
                         entry: Mapping[str, Any], f140_ids: Sequence[str]) -> None:
    rid = entry.get("run_id")
    canonical = entry.get("canonical_key_json")
    full_sha = entry.get("run_id_full_sha256")
    if not isinstance(rid, str) or rid not in set(f140_ids):
        raise PermissionError("requested key is outside frozen F140")
    if not _canonical_key_identity_is_valid(entry):
        raise PermissionError("canonical key identity differs from frozen run ID")
    if not key_is_untouched(output_root, entry):
        raise PermissionError("F140 key has prior status/lock/run history")


def _read_jsonl_exact(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    if not os.path.lexists(path):
        if missing_ok:
            return []
        raise PermissionError("required append-only ledger is absent")
    if path.is_symlink() or not path.is_file():
        raise PermissionError("append-only ledger is not an ordinary file")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise PermissionError("truncated append-only ledger tail; preserve for review")
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise PermissionError("malformed append-only ledger; preserve for review") from exc
        if not isinstance(value, dict):
            raise PermissionError("append-only ledger event is not an object")
        rows.append(value)
    return rows


def _append_jsonl_durable(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PermissionError("refusing symlink ledger")
    # Validate the existing complete tail before appending; never repair/truncate.
    _read_jsonl_exact(path, missing_ok=True)
    data = canonical_json_bytes(dict(payload), newline=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json_create(path: os.PathLike | str, payload: Mapping[str, Any], *, mode: int = 0o444) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    _fsync_dir(target.parent)
    return sha256_bytes(raw)


def durable_bytes_create(path: os.PathLike | str, raw: bytes, *, mode: int = 0o444) -> str:
    """Publish exact bytes once; used for immutable preimage archives."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags, mode)
    except FileExistsError as exc:
        raise PermissionError("immutable artifact already exists: " + str(target)) from exc
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(target.parent)
    except BaseException:
        raise
    return sha256_bytes(raw)


def _atomic_replace_bytes(path: Path, raw: bytes, *, mode: int = 0o600,
                          after_replace=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    if after_replace is not None:
        after_replace()
    _fsync_dir(path.parent)


@contextlib.contextmanager
def execution_state_lock(runs_root: os.PathLike | str, *, already_held: bool = False):
    """Acquire once; callers holding the lock pass already_held=True."""
    lock_path = Path(runs_root).resolve() / "execution_control" / "execution_state.lock"
    if already_held:
        yield lock_path
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _admission_resource_locks(runs_root: Path):
    """Serialize release against every frozen worker and GPU lock."""
    held: list[int] = []
    try:
        lock_paths = [runs_root / "worker_locks" / ("worker_{}.lock".format(index))
                      for index in (0, 1)]
        lock_paths += [runs_root / "gpu_locks" / ("gpu{}.lock".format(index))
                       for index in (0, 1)]
        for path in lock_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise PermissionError("worker/GPU lock path is a symlink: " + str(path))
            fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(fd)
                raise PermissionError("a reviewed worker/GPU resource is currently locked") from exc
            held.append(fd)
        yield
    finally:
        for fd in reversed(held):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _durability_fence(gate_path: Path, *, lock_already_held: bool) -> None:
    # No nested flock on the terminal path: the outer caller owns serialization.
    if not lock_already_held:
        raise RuntimeError("durability fence must be called inside execution_state_lock")
    fd = os.open(gate_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(gate_path.parent)


def _durability_fence_path(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("durability fence target is absent or non-regular: " + str(path))
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def validate_current_gate(gate: Mapping[str, Any], *, root: os.PathLike | str = ROOT,
                          require_release: bool = True,
                          lock_already_held: bool = False,
                          terminal_source_only: bool = False) -> dict[str, Any]:
    """Validate the complete v1.5 source and terminal graph; no v1.4 fallback."""
    base = Path(root).resolve()
    if gate.get("schema") != SCHEMA:
        raise PermissionError("v1.5 gate schema mismatch")
    if gate.get("launch_scope") != SCOPE:
        raise PermissionError("v1.5 scope is not fresh-only search")
    if gate.get("authorization") != {"resume": False, "retry": False,
            "new_attempt": False, "checkpoint_load": False, "formal": False, "test": False}:
        raise PermissionError("v1.5 forbidden authorization flags are not all false")
    authorized = gate.get("launch_authorized") is True
    frozen = _assert_frozen_artifacts(base, allow_released_hold=authorized)
    search_path = base / SEARCH_GATE_REL
    queue_path = base / QUEUE_REL
    if sha256_file(search_path) != SEARCH_GATE_SHA256 or sha256_file(queue_path) != QUEUE_SHA256:
        raise PermissionError("v1.5 fixed search identity changed")
    search = _read_json(search_path)
    if search.get("phase") != "search" or search.get("formal_or_test_started") is not False:
        raise PermissionError("v1.5 search gate is outside frozen search-only state")
    if search.get("code_hash") != SCIENTIFIC_SHA256:
        raise PermissionError("v1.5 scientific identity differs from frozen search")
    if gate.get("prior_search_gate_sha256") != SEARCH_GATE_SHA256:
        raise PermissionError("v1.5 gate does not bind frozen search gate")
    if gate.get("search_queue_sha256") != QUEUE_SHA256:
        raise PermissionError("v1.5 gate does not bind frozen search queue")
    runtime_inventory = runtime_hash_inventory(base)
    runtime_digest = sha256_json(runtime_inventory)
    control_inventory = control_hash_inventory_v15(base)
    control_digest = sha256_json(control_inventory)
    if gate.get("runtime_hashes") != runtime_inventory or gate.get("runtime_hash_inventory_sha256") != runtime_digest:
        raise PermissionError("v1.5 complete runtime inventory is stale")
    if gate.get("control_hashes") != control_inventory or gate.get("control_hash_inventory_sha256") != control_digest:
        raise PermissionError("v1.5 control inventory is stale")
    if gate.get("source_frozen_artifacts") != frozen:
        raise PermissionError("v1.5 gate source incident graph differs from frozen artifacts")
    source_rel = gate.get("source_snapshot_path")
    source_sha = gate.get("source_snapshot_file_sha256")
    source_identity = gate.get("source_snapshot_sha256")
    prelaunch_rel = gate.get("prelaunch_audit_path")
    prelaunch_sha = gate.get("prelaunch_audit_sha256")
    if not all(isinstance(value, str) for value in
               (source_rel, source_sha, source_identity, prelaunch_rel, prelaunch_sha)):
        raise PermissionError("v1.5 source/prelaunch graph bindings are incomplete")
    if source_rel != "experiments/stage_F0_outputs/runs/execution_control/postbootstrap_v15_source_snapshot.json":
        raise PermissionError("v1.5 source snapshot path is not the canonical versioned artifact")
    if (not prelaunch_rel.startswith(
            "experiments/stage_F0_outputs/runs/execution_control/prelaunch_audit_v15_fresh_search_")
            or not prelaunch_rel.endswith(".json")):
        raise PermissionError("v1.5 prelaunch path is outside the canonical versioned namespace")
    source_file = _repo_path(base, source_rel)
    prelaunch_file = _repo_path(base, prelaunch_rel)
    if (not source_file.is_file() or source_file.is_symlink()
            or source_file.stat().st_mode & 0o222
            or sha256_file(source_file) != source_sha):
        raise PermissionError("v1.5 immutable source snapshot bytes differ")
    source_snapshot = _read_json(source_file)
    validate_source_snapshot(source_snapshot, root=base,
                             allow_released_hold=gate.get("launch_authorized") is True,
                             allow_gate_replaced=gate.get("launch_authorized") is False)
    if source_snapshot.get("snapshot_sha256") != source_identity:
        raise PermissionError("v1.5 source snapshot identity mismatch")
    if (not prelaunch_file.is_file() or prelaunch_file.is_symlink()
            or prelaunch_file.stat().st_mode & 0o222
            or sha256_file(prelaunch_file) != prelaunch_sha):
        raise PermissionError("v1.5 immutable prelaunch audit bytes differ")
    prelaunch = _read_json(prelaunch_file)
    targets = validate_prelaunch_audit(prelaunch, root=base,
             allow_released_hold=gate.get("launch_authorized") is True,
             allow_gate_replaced=gate.get("launch_authorized") is False,
             source_snapshot_sha256=source_identity)
    if prelaunch.get("source_snapshot_sha256") != source_identity:
        raise PermissionError("v1.5 prelaunch audit is not bound to current source snapshot")
    expected_pairs = [{"index": index, "uuid": device_uuid}
                      for index, device_uuid in sorted(targets.items())]
    if gate.get("gpu_target_pairs") != expected_pairs:
        raise PermissionError("v1.5 gate targets differ from reviewed prelaunch audit")
    if gate.get("prelaunch_target_pairs") != expected_pairs:
        raise PermissionError("v1.5 gate prelaunch target identity differs")
    amendment = gate.get("bootstrap_amendment")
    _validate_amendment(amendment, scientific_sha256=SCIENTIFIC_SHA256,
                        runtime_sha256=runtime_digest, control_sha256=control_digest)
    for name, digest in (("design_path", DESIGN_PATH), ("design_sha256", DESIGN_SHA256),
                         ("design_review_path", DESIGN_REVIEW_PATH),
                         ("design_review_sha256", DESIGN_REVIEW_SHA256)):
        if gate.get(name) != digest:
            raise PermissionError("v1.5 gate design/review binding mismatch: " + name)
    review = _validate_review_graph(gate, root=base,
                                    runtime_inventory=runtime_inventory,
                                    control_inventory=control_inventory)
    pending = gate.get("launch_authorized") is False
    if terminal_source_only:
        if not authorized or gate.get("transaction_state") != "COMMITTED":
            raise PermissionError("terminal source graph is not an authorized COMMITTED gate")
        return {"execution_gate": dict(gate), "search_gate": search,
                "control_inventory": control_inventory, "runtime_inventory": runtime_inventory}
    if pending:
        if require_release:
            raise PermissionError("v1.5 pending gate cannot authorize launch")
        return {"execution_gate": dict(gate), "search_gate": search,
                "control_inventory": control_inventory, "runtime_inventory": runtime_inventory}
    if gate.get("launch_authorized") is not True:
        raise PermissionError("v1.5 launch authorization is malformed")
    if not require_release:
        raise PermissionError("authorized v1.5 graph requires terminal release validation")
    runs_root = base / "experiments/stage_F0_outputs/runs"
    with execution_state_lock(runs_root, already_held=lock_already_held):
        result = validate_terminal_graph(gate, root=base, lock_already_held=True)
    return {"execution_gate": dict(gate), "search_gate": search,
            "control_inventory": control_inventory, "runtime_inventory": runtime_inventory,
            **result}


def validate_terminal_graph(gate: Mapping[str, Any], *, root: os.PathLike | str,
                            lock_already_held: bool = False) -> dict[str, Any]:
    if not lock_already_held:
        raise RuntimeError("terminal graph must be validated under execution_state.lock")
    base = Path(root).resolve()
    gate_path = base / CANONICAL_GATE_REL
    if not gate_path.is_file() or gate_path.is_symlink():
        raise PermissionError("canonical v1.5 terminal gate is not an ordinary file")
    if sha256_file(gate_path) != sha256_bytes(json.dumps(dict(gate), ensure_ascii=False,
            sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"):
        raise PermissionError("canonical gate bytes differ from validated terminal gate")
    if gate.get("launch_authorized") is not True or gate.get("transaction_state") != "COMMITTED":
        raise PermissionError("terminal gate is not COMMITTED and authorized")
    validate_current_gate(gate, root=base, require_release=True,
                          lock_already_held=True, terminal_source_only=True)
    release_path = base / RELEASE_REL
    release_artifact_rel = gate.get("release_artifact_path")
    release_artifact = _repo_path(base, release_artifact_rel) if isinstance(release_artifact_rel, str) else None
    if (not release_path.is_file() or release_path.is_symlink() or release_artifact is None
            or not release_artifact.is_file() or release_artifact.is_symlink()):
        raise PermissionError("v1.5 release pointer/artifact is absent")
    if sha256_file(release_path) != gate.get("release_sha256") or sha256_file(release_artifact) != gate.get("release_sha256"):
        raise PermissionError("v1.5 release pointer/artifact digest differs")
    release = _read_json(release_artifact)
    hold_path = base / HOLD_REL
    if not hold_path.is_file() or hold_path.is_symlink() or sha256_file(hold_path) != gate.get("released_hold_sha256"):
        raise PermissionError("current hold does not equal exact released projection")
    hold = _read_json(hold_path)
    if hold.get("active") is not False or hold.get("successor_release_id") != release.get("release_id"):
        raise PermissionError("current hold is not the matching inactive v1.5 projection")
    ledger_path = base / RELEASE_LEDGER_REL
    rows = _read_jsonl_exact(ledger_path)
    expected_events = ["PREPARED", "RELEASE_ARTIFACT_DURABLE", "PENDING_GATE_DURABLE",
                       "RELEASE_POINTER_DURABLE", "HOLD_RELEASED", "COMMITTED"]
    if ([row.get("event") for row in rows] != expected_events
            or any(row.get("schema") != LEDGER_SCHEMA for row in rows)):
        raise PermissionError("v1.5 terminal ledger is not the exact ordered six-phase graph")
    transaction_id = rows[0].get("transaction_id")
    if (not isinstance(transaction_id, str) or not transaction_id
            or any(row.get("transaction_id") != transaction_id for row in rows)):
        raise PermissionError("v1.5 terminal ledger has mixed transaction identities")
    prepared = rows[0]
    release_sha = sha256_file(release_artifact)
    hold_sha = sha256_file(hold_path)
    pending_sha = prepared.get("pending_gate_sha256")
    if (release.get("release_id") != transaction_id
            or release.get("nonce") != prepared.get("nonce")
            or release.get("source_snapshot_sha256") != prepared.get("source_snapshot_sha256")
            or release.get("prelaunch_audit_sha256") != prepared.get("prelaunch_audit_sha256")
            or release.get("targets") != prepared.get("targets")
            or release.get("pending_gate_sha256") != pending_sha
            or rows[1].get("release_sha256") != release_sha
            or rows[2].get("pending_gate_sha256") != pending_sha
            or rows[3].get("release_sha256") != release_sha
            or rows[4].get("released_hold_sha256") != hold_sha
            or rows[5].get("release_sha256") != release_sha
            or rows[5].get("pending_gate_sha256") != pending_sha
            or rows[5].get("released_hold_sha256") != hold_sha
            or release_sha != gate.get("release_sha256")
            or hold_sha != gate.get("released_hold_sha256")):
        raise PermissionError("COMMITTED event differs from terminal release graph")
    if transaction_id != uuid.uuid5(
            uuid.NAMESPACE_URL, "f0-v15:" + str(prepared.get("nonce", ""))).hex:
        raise PermissionError("terminal PREPARED transaction ID/nonce binding differs")
    if (prepared.get("source_incident_hold_sha256") != INCIDENT_HOLD_SHA256
            or prepared.get("source_failure_nonce") != INCIDENT_NONCE
            or prepared.get("historical_v14_release_sha256") != V14_RELEASE_SHA256
            or prepared.get("historical_v14_release_ledger_sha256") != V14_RELEASE_LEDGER_SHA256
            or prepared.get("v14_migration_ledger_sha256") != V14_MIGRATION_LEDGER_SHA256
            or prepared.get("search_queue_sha256") != QUEUE_SHA256):
        raise PermissionError("terminal PREPARED frozen-source identity differs")
    ledger_sha = sha256_file(ledger_path)
    if ledger_sha != gate.get("committed_ledger_sha256"):
        raise PermissionError("terminal gate COMMITTED ledger digest mismatch")
    _durability_fence(gate_path, lock_already_held=True)
    return {"release": release, "hold": hold, "ledger_sha256": ledger_sha,
            "terminal_acknowledgement": True}


def build_source_snapshot(*, root: os.PathLike | str = ROOT,
                          allow_released_hold: bool = False,
                          allow_gate_replaced: bool = False) -> dict[str, Any]:
    """Side-effect-free snapshot; no status, lock, or run artifact is created."""
    base = Path(root).resolve()
    frozen = _assert_frozen_artifacts(base, allow_released_hold=allow_released_hold,
                                      allow_gate_replaced=allow_gate_replaced)
    from stage_F0_postabnormal_v14 import validate_source_snapshot
    runs_root = base / "experiments/stage_F0_outputs/runs"
    historical_path = base / "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_source_snapshot.json"
    historical = _read_json(historical_path)
    validate_source_snapshot(historical, runs_root=RUNS, allow_live_fresh_progress=False)
    h51_statuses = _capture_post_migration_h51(runs_root)
    _validate_protected_snapshot_at_root(historical, runs_root,
                                         allow_live_fresh_progress=False,
                                         expected_h51=h51_statuses)
    payload = {
        "schema": SOURCE_SCHEMA,
        "design_sha256": DESIGN_SHA256,
        "design_review_sha256": DESIGN_REVIEW_SHA256,
        "approval_record_sha256": APPROVAL_RECORD_SHA256,
        "remediation_no_go_candidate_sha256": REMEDIATION_CANDIDATE_SHA256,
        "source_incident_hold_sha256": INCIDENT_HOLD_SHA256,
        "source_incident_hold_id": INCIDENT_HOLD_ID,
        "source_failure_nonce": INCIDENT_NONCE,
        "failure_record_sha256": INCIDENT_FAILURE_SHA256,
        "historical_v14_release_sha256": V14_RELEASE_SHA256,
        "historical_v14_release_ledger_sha256": V14_RELEASE_LEDGER_SHA256,
        "historical_v14_migration_ledger_sha256": V14_MIGRATION_LEDGER_SHA256,
        "historical_v14_source_snapshot_sha256": V14_SOURCE_SNAPSHOT_SHA256,
        "search_gate_sha256": SEARCH_GATE_SHA256,
        "queue_sha256": QUEUE_SHA256,
        "frozen_artifacts": frozen,
        "historical_protection_snapshot": historical,
        "post_migration_h51_statuses": h51_statuses,
    }
    payload["snapshot_sha256"] = sha256_json(payload)
    return payload


def _capture_post_migration_h51(runs_root: Path) -> dict[str, dict[str, Any]]:
    import stage_F0_postabnormal_v14 as v14
    root = Path(runs_root).resolve()
    _queue, statuses = v14._queue_and_statuses(root)
    if not set(v14.H51_RUN_IDS).issubset(statuses):
        raise PermissionError("post-migration H51 status membership is incomplete")
    records: dict[str, dict[str, Any]] = {}
    for rid in v14.H51_RUN_IDS:
        path = root / v14.STATUS_REL / (rid + ".json")
        row = statuses[rid]
        if row.get("status") != "RECOVERY_BLOCKED" or path.is_symlink() or not path.is_file():
            raise PermissionError("post-migration H51 member is not durably recovery-blocked: " + rid)
        records[rid] = {"sha256": v14.sha256_file(path), "projection": row}
    return records


def _validate_protected_snapshot_at_root(snapshot: Mapping[str, Any], runs_root: Path,
                                        *, allow_live_fresh_progress: bool,
                                        expected_h51: Mapping[str, Any]) -> None:
    """Compare C97/H51/F140/incident projections without replaying old release logic."""
    import stage_F0_postabnormal_v14 as v14
    root = Path(runs_root).resolve()
    queue, statuses = v14._queue_and_statuses(root)
    frozen_completed = snapshot.get("completed_ids")
    if not isinstance(frozen_completed, list) or frozen_completed != sorted(set(frozen_completed)):
        raise PermissionError("v1.5 frozen C97 identity list is invalid")
    current_frozen_completed = sorted(
        rid for rid in frozen_completed
        if isinstance(statuses.get(rid), Mapping)
        and statuses[rid].get("status") == "VALIDATION_COMPLETE")
    if current_frozen_completed != frozen_completed:
        raise PermissionError("v1.5 protected C97 status membership changed")
    completed_inventory = []
    for rid in frozen_completed:
        completed_inventory.extend(v14._tree_inventory(root, rid))
    completed_inventory.sort(key=lambda row: str(row["path"]))
    if (completed_inventory != snapshot.get("completed_inventory")
            or v14._inventory_digest(completed_inventory) != snapshot.get("completed_inventory_sha256")):
        raise PermissionError("v1.5 protected C97 asset inventory changed")
    old_h49 = snapshot.get("old_h49_statuses")
    if not isinstance(old_h49, Mapping):
        raise PermissionError("historical H49 status inventory is missing")
    for rid in v14.PRESERVED_H49_RUN_IDS:
        status_path = root / v14.STATUS_REL / (rid + ".json")
        record = old_h49.get(rid)
        if (not status_path.is_file() or not isinstance(record, Mapping)
                or v14.sha256_file(status_path) != record.get("raw_sha256")
                or v14._read_json(status_path) != record.get("projection")):
            raise PermissionError("protected historical H49 status changed: " + rid)
    if list(expected_h51) != list(v14.H51_RUN_IDS):
        raise PermissionError("post-migration H51 status inventory membership/order mismatch")
    for rid in v14.H51_RUN_IDS:
        row = statuses.get(rid)
        path = root / v14.STATUS_REL / (rid + ".json")
        record = expected_h51.get(rid)
        if (not isinstance(row, Mapping) or row.get("status") != "RECOVERY_BLOCKED"
                or not isinstance(record, Mapping) or not path.is_file() or path.is_symlink()
                or v14.sha256_file(path) != record.get("sha256")
                or row != record.get("projection")):
            raise PermissionError("post-migration protected H51 status changed: " + rid)
    incident_inventory = v14._incident_evidence_inventory(root)
    if (incident_inventory != snapshot.get("incident_inventory")
            or v14._inventory_digest(incident_inventory) != snapshot.get("incident_inventory_sha256")):
        raise PermissionError("v1.5 protected incident asset inventory changed")
    f140 = set(snapshot.get("fresh_ids", []))
    if (len(f140) != 140 or v14.compact_sha(sorted(f140)) != v14.FRESH_KEYS_SHA256):
        raise PermissionError("v1.5 protected F140 identity set is invalid")
    queue_rows = {str(row.get("run_id", "")): row for row in queue.get("entries", [])}
    for rid in sorted(f140):
        row = queue_rows.get(rid)
        if not isinstance(row, Mapping) or not _canonical_key_identity_is_valid(row):
            raise PermissionError("frozen F140 queue identity is missing or malformed: " + rid)
        status_path = root / v14.STATUS_REL / (rid + ".json")
        lock_path = root / "canonical_locks" / (rid + ".lock")
        run_path = root / rid
        touched = any(os.path.lexists(path) for path in (status_path, lock_path, run_path))
        if touched and not allow_live_fresh_progress:
            raise PermissionError("pending source snapshot found touched F140 member: " + rid)
        if not touched:
            continue
        if (status_path.is_symlink() or not status_path.is_file()):
            raise PermissionError("F140 progress lacks an ordinary claim status: " + rid)
        status = _read_json(status_path)
        for field in ("run_id", "run_id_full_sha256", "canonical_key_json"):
            if status.get(field) != row.get(field):
                raise PermissionError("F140 status identity differs from frozen queue row: " + rid)
        if (status.get("attempt_id") != "attempt-00"
                or status.get("recovery_count") != 0
                or status.get("resume_checkpoint") is not None):
            raise PermissionError("F140 progress attempted recovery/resume: " + rid)
        if lock_path.is_symlink() or not lock_path.is_file():
            raise PermissionError("F140 progress lacks an ordinary canonical lock: " + rid)
        owner = _read_json(lock_path)
        for field in ("pid", "start_ticks", "nonce", "run_id_full_sha256", "canonical_key_json", "attempt_id"):
            if owner.get(field) != status.get(field):
                raise PermissionError("F140 status/lock owner identity differs: " + rid)
        state = status.get("status")
        if state == "CLAIMED":
            from stage_F0_worker_control import linux_process_identity
            try:
                process = linux_process_identity(int(status.get("pid", -1)))
            except (OSError, ValueError, PermissionError) as exc:
                raise PermissionError("F140 claimed peer is not a live identified process: " + rid) from exc
            if process.get("start_ticks") != status.get("start_ticks"):
                raise PermissionError("F140 claimed peer PID was reused: " + rid)
        elif state == "VALIDATION_COMPLETE":
            attempt = run_path / "attempts" / "attempt-00"
            manifest_path = attempt / "run_manifest.json"
            summary_path = attempt / "trajectory_summary.json"
            if (not attempt.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink()
                    or not summary_path.is_file() or summary_path.is_symlink()):
                raise PermissionError("completed F140 run tree is incomplete: " + rid)
            manifest, summary = _read_json(manifest_path), _read_json(summary_path)
            for value in (manifest, summary):
                if (value.get("status") != "VALIDATION_COMPLETE"
                        or value.get("run_id") != rid
                        or value.get("run_id_full_sha256") != row.get("run_id_full_sha256")
                        or value.get("canonical_key_json") != row.get("canonical_key_json")
                        or value.get("attempt_id") != "attempt-00"):
                    raise PermissionError("completed F140 run identity/status differs: " + rid)
        else:
            raise PermissionError("F140 progress state is not a fresh-only claim/completion: " + rid)
    for row in queue.get("entries", []):
        rid = str(row.get("run_id", ""))
        if rid not in f140:
            continue
        touched = (rid in statuses or os.path.lexists(root / "canonical_locks" / (rid + ".lock"))
                   or os.path.lexists(root / rid))
        if touched and not allow_live_fresh_progress:
            raise PermissionError("pending source snapshot found touched F140 member: " + rid)


def _prelaunch_lock_inventory(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = [runs_root / "execution_control" / "execution_state.lock"]
    candidates.extend(runs_root / "worker_locks" / f"worker_{i}.lock" for i in (0, 1))
    candidates.extend(runs_root / "gpu_locks" / f"gpu{i}.lock" for i in (0, 1))
    for path in candidates:
        if not os.path.lexists(path):
            rows.append({"path": str(path.relative_to(runs_root)), "exists": False,
                         "active": False, "sha256": None})
            continue
        if path.is_symlink() or not path.is_file():
            raise PermissionError("prelaunch lock path is non-regular: " + str(path))
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PermissionError("prelaunch found an active execution/worker/GPU lock: " + str(path)) from exc
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        rows.append({"path": str(path.relative_to(runs_root)), "exists": True,
                     "active": False, "sha256": sha256_file(path)})
    return rows


def _scheduler_root_for_release() -> Path:
    configured = os.environ.get("F0_PROCESS_SCHEDULER_ROOT")
    if configured:
        raw = Path(configured)
    else:
        scheduler_hash = hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:16]
        raw = Path(os.environ.get("TMPDIR", "/tmp")) / ("phase0-3-F0-scheduler-" + scheduler_hash)
    if raw.is_symlink():
        raise PermissionError("F0 process scheduler root is a symlink")
    return raw.resolve()


def _assert_no_competing_task_slots() -> dict[str, Any]:
    scheduler = _scheduler_root_for_release()
    own_index_text = os.environ.get("F0_RESERVED_PROCESS_SLOT_INDEX")
    own_fd_text = os.environ.get("F0_RESERVED_PROCESS_SLOT_FD")
    own_owner = os.environ.get("F0_RESERVED_PROCESS_SLOT_OWNER_PID")
    own_role = os.environ.get("F0_RESERVED_PROCESS_SLOT_ROLE")
    own_index = None
    if own_index_text is not None:
        try:
            own_index = int(own_index_text)
            own_fd = int(own_fd_text or "-1")
            owner_pid = int(own_owner or "-1")
        except ValueError as exc:
            raise PermissionError("reserved task slot environment is malformed") from exc
        if not 0 <= own_index < MAX_TASK_PYTHON_PROCESSES or owner_pid != os.getpid():
            raise PermissionError("reserved task slot is not owned by this release process")
        path = scheduler / ("slot{}.lock".format(own_index))
        if path.is_symlink() or not path.is_file():
            raise PermissionError("reserved task slot file is missing or non-regular")
        fd_stat, path_stat = os.fstat(own_fd), path.stat()
        if (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
            raise PermissionError("reserved task slot descriptor differs from scheduler inode")
        owner = _read_json(path)
        if (owner.get("pid") != owner_pid or owner.get("slot") != own_index
                or owner.get("role") != own_role or owner.get("host") != socket.gethostname()):
            raise PermissionError("reserved task slot owner identity is invalid")

    active: list[int] = []
    for index in range(MAX_TASK_PYTHON_PROCESSES):
        path = scheduler / ("slot{}.lock".format(index))
        if not os.path.lexists(path):
            continue
        if path.is_symlink() or not path.is_file():
            raise PermissionError("scheduler slot is non-regular")
        fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active.append(index)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    if len(active) > MAX_TASK_PYTHON_PROCESSES:
        raise PermissionError("active F0 task process count exceeds five slots")
    competing = [index for index in active if index != own_index]
    if competing:
        raise PermissionError("another F0 task process/worker occupies scheduler slots")
    if own_index is not None and own_index not in active:
        raise PermissionError("current release task does not hold its reserved slot flock")
    return {"scheduler_root": str(scheduler), "active_slots": sorted(active),
            "active_task_process_count": len(active)}


def _assert_f140_untouched_for_nonterminal(repo_root: Path, runs_root: Path) -> int:
    queue = _read_json(_repo_path(repo_root, QUEUE_REL))
    fresh_ids = set(frozen_f140_ids(repo_root))
    if len(fresh_ids) != 140:
        raise PermissionError("nonterminal release recovery requires exact frozen F140")
    entries = {str(row.get("run_id")): row for row in queue.get("entries", [])}
    for rid in fresh_ids:
        entry = entries.get(rid)
        if not isinstance(entry, Mapping) or not _canonical_key_identity_is_valid(entry):
            raise PermissionError("nonterminal release recovery has a malformed F140 key")
        paths = fresh_member_paths(runs_root, rid)
        if any(os.path.lexists(paths[key]) for key in ("status", "lock", "run")):
            raise PermissionError("nonterminal release recovery found touched F140 member: " + rid)
    return len(fresh_ids)


def _assert_current_release_resources(runs_root: Path, repo_root: Path) -> dict[str, Any]:
    free = shutil.disk_usage(runs_root).free
    if isinstance(free, bool) or not isinstance(free, int) or free <= 0:
        raise PermissionError("current release disk-free evidence is invalid or exhausted")
    slots = _assert_no_competing_task_slots()
    untouched = _assert_f140_untouched_for_nonterminal(repo_root, runs_root)
    return {"disk_free_bytes": free, "slots": slots, "f140_untouched_count": untouched}


def validate_source_snapshot(snapshot: Mapping[str, Any], *, root: os.PathLike | str = ROOT,
                             allow_released_hold: bool = False,
                             allow_gate_replaced: bool = False) -> None:
    if snapshot.get("schema") != SOURCE_SCHEMA:
        raise PermissionError("v1.5 source snapshot schema mismatch")
    without = dict(snapshot)
    claimed = without.pop("snapshot_sha256", None)
    if claimed != sha256_json(without):
        raise PermissionError("v1.5 source snapshot digest mismatch")
    base = Path(root).resolve()
    frozen = _assert_frozen_artifacts(base, allow_released_hold=allow_released_hold,
                                      allow_gate_replaced=allow_gate_replaced)
    historical = snapshot.get("historical_protection_snapshot")
    if not isinstance(historical, Mapping):
        raise PermissionError("v1.5 source snapshot omits the historical protection graph")
    from stage_F0_postabnormal_v14 import validate_source_snapshot as validate_v14_snapshot
    validate_v14_snapshot(historical, runs_root=RUNS,
                          allow_live_fresh_progress=allow_released_hold)
    _validate_protected_snapshot_at_root(
        historical, base / "experiments/stage_F0_outputs/runs",
        allow_live_fresh_progress=allow_released_hold,
        expected_h51=snapshot.get("post_migration_h51_statuses", {}))
    expected_static = {
        "design_sha256": DESIGN_SHA256,
        "design_review_sha256": DESIGN_REVIEW_SHA256,
        "approval_record_sha256": APPROVAL_RECORD_SHA256,
        "remediation_no_go_candidate_sha256": REMEDIATION_CANDIDATE_SHA256,
        "source_incident_hold_sha256": INCIDENT_HOLD_SHA256,
        "source_incident_hold_id": INCIDENT_HOLD_ID,
        "source_failure_nonce": INCIDENT_NONCE,
        "failure_record_sha256": INCIDENT_FAILURE_SHA256,
        "historical_v14_release_sha256": V14_RELEASE_SHA256,
        "historical_v14_release_ledger_sha256": V14_RELEASE_LEDGER_SHA256,
        "historical_v14_migration_ledger_sha256": V14_MIGRATION_LEDGER_SHA256,
        "historical_v14_source_snapshot_sha256": V14_SOURCE_SNAPSHOT_SHA256,
        "search_gate_sha256": SEARCH_GATE_SHA256,
        "queue_sha256": QUEUE_SHA256,
        "frozen_artifacts": frozen,
    }
    if any(snapshot.get(key) != value for key, value in expected_static.items()):
        raise PermissionError("v1.5 source snapshot frozen identity differs from current artifacts")


def build_prelaunch_audit(*, root: os.PathLike | str = ROOT, nonce: str,
                          target_rows: Sequence[Mapping[str, Any]],
                          disk_free_bytes: int, process_count: int,
                          observation_started_monotonic: float) -> dict[str, Any]:
    if not isinstance(nonce, str) or len(nonce) < 16:
        raise ValueError("prelaunch nonce must contain at least 16 characters")
    if isinstance(disk_free_bytes, bool) or not isinstance(disk_free_bytes, int) or disk_free_bytes <= 0:
        raise PermissionError("prelaunch disk_free_bytes must be a positive integer")
    if isinstance(process_count, bool) or not isinstance(process_count, int) or not 0 <= process_count <= MAX_TASK_PYTHON_PROCESSES:
        raise PermissionError("task Python process count exceeds the five-process maximum")
    frozen = validate_target_rows(target_rows, [row for row in target_rows if row.get("selected") is True])
    snapshot = build_source_snapshot(root=root)
    payload = {
        "schema": PRELAUNCH_SCHEMA, "nonce": nonce, "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "design_sha256": DESIGN_SHA256, "design_review_sha256": DESIGN_REVIEW_SHA256,
        "approval_record_sha256": APPROVAL_RECORD_SHA256,
        "remediation_no_go_candidate_sha256": REMEDIATION_CANDIDATE_SHA256,
        "search_gate_sha256": SEARCH_GATE_SHA256, "queue_sha256": QUEUE_SHA256,
        "gpu_target_pairs": [{"index": i, "uuid": u} for i, u in sorted(frozen.items())],
        "gpu_memory": {"devices": [dict(row) for row in target_rows]},
        "disk_free_bytes": disk_free_bytes, "task_python_process_count": process_count,
        "active_claim_count": 0, "no_residual_evidence": True,
        "worker_lock_inventory": _prelaunch_lock_inventory(
            Path(root).resolve() / "experiments/stage_F0_outputs/runs"),
        "formal_or_test_started": False, "launch_authorized": False,
        "observation_started_monotonic": float(observation_started_monotonic),
    }
    _validate_prelaunch_device_inventory(payload)
    payload["audit_sha256"] = sha256_json(payload)
    return payload


def _validate_prelaunch_device_inventory(audit: Mapping[str, Any]) -> dict[int, str]:
    devices = audit.get("gpu_memory", {}).get("devices", [])
    if not isinstance(devices, list):
        raise PermissionError("prelaunch GPU device inventory is malformed")
    by_index: dict[int, Mapping[str, Any]] = {}
    audit_mono = audit.get("observation_started_monotonic")
    if (isinstance(audit_mono, bool) or not isinstance(audit_mono, (int, float))
            or not math.isfinite(float(audit_mono))):
        raise PermissionError("prelaunch observation monotonic timestamp is invalid")
    audit_utc = audit.get("created_at_utc")
    try:
        audit_utc_value = _dt.datetime.fromisoformat(str(audit_utc).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PermissionError("prelaunch audit UTC timestamp is malformed") from exc
    for row in devices:
        if not isinstance(row, Mapping):
            raise PermissionError("prelaunch GPU row is not an object")
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index not in (0, 1) or index in by_index:
            raise PermissionError("prelaunch GPU index is missing or ambiguous")
        by_index[index] = row
        if row.get("readable") is True:
            if not isinstance(row.get("uuid"), str) or not row.get("uuid"):
                raise PermissionError("readable prelaunch GPU lacks UUID")
            eligible = fresh_gpu_eligible(row)
            if row.get("eligible") is not eligible:
                raise PermissionError("prelaunch GPU eligibility differs from strict memory rule")
            for field in ("observation_started_monotonic", "observed_at_utc"):
                if field not in row:
                    raise PermissionError("prelaunch GPU row omits observation time")
            row_mono = row.get("observation_started_monotonic")
            if (isinstance(row_mono, bool) or not isinstance(row_mono, (int, float))
                    or row_mono > audit_mono or audit_mono - row_mono > MEMORY_FRESH_SECONDS):
                raise PermissionError("prelaunch GPU monotonic observation is stale")
            try:
                row_utc = _dt.datetime.fromisoformat(str(row["observed_at_utc"]).replace("Z", "+00:00"))
                utc_delta = (audit_utc_value - row_utc).total_seconds()
            except ValueError as exc:
                raise PermissionError("prelaunch GPU UTC observation is malformed") from exc
            if not 0.0 <= utc_delta <= MEMORY_FRESH_SECONDS:
                raise PermissionError("prelaunch GPU UTC observation is stale or future-dated")
        else:
            if row.get("eligible") is not False or not isinstance(row.get("reason"), str) or not row["reason"]:
                raise PermissionError("unreadable prelaunch GPU needs an explicit ineligible reason")
        if row.get("selected") is not (row.get("eligible") is True):
            raise PermissionError("initial prelaunch target set differs from all eligible devices")
    if set(by_index) != {0, 1}:
        raise PermissionError("prelaunch audit must inventory both physical GPU indices")
    selected = {i: str(row.get("uuid")) for i, row in by_index.items()
                if row.get("selected") is True}
    if not selected:
        raise PermissionError("no independently eligible GPU exists in prelaunch inventory")
    return selected


def validate_prelaunch_audit(audit: Mapping[str, Any], *, root: os.PathLike | str = ROOT,
                             max_age_monotonic: float | None = None,
                             allow_released_hold: bool = False,
                             allow_gate_replaced: bool = False,
                             source_snapshot_sha256: str | None = None) -> dict[int, str]:
    if audit.get("schema") != PRELAUNCH_SCHEMA or audit.get("launch_authorized") is not False:
        raise PermissionError("v1.5 prelaunch audit is not a nonauthorizing v1.5 audit")
    disk_free = audit.get("disk_free_bytes")
    process_count = audit.get("task_python_process_count")
    if (isinstance(disk_free, bool) or not isinstance(disk_free, int) or disk_free <= 0
            or isinstance(process_count, bool) or not isinstance(process_count, int)
            or not 0 <= process_count <= MAX_TASK_PYTHON_PROCESSES
            or audit.get("active_claim_count") != 0
            or audit.get("no_residual_evidence") is not True
            or not isinstance(audit.get("worker_lock_inventory"), list)):
        raise PermissionError("v1.5 prelaunch resource/process/lock evidence is incomplete")
    static_identity = {
        "design_sha256": DESIGN_SHA256, "design_review_sha256": DESIGN_REVIEW_SHA256,
        "approval_record_sha256": APPROVAL_RECORD_SHA256,
        "remediation_no_go_candidate_sha256": REMEDIATION_CANDIDATE_SHA256,
        "search_gate_sha256": SEARCH_GATE_SHA256, "queue_sha256": QUEUE_SHA256,
        "formal_or_test_started": False,
    }
    if any(audit.get(key) != value for key, value in static_identity.items()):
        raise PermissionError("v1.5 prelaunch static design/review/approval identity differs")
    if not isinstance(audit.get("nonce"), str) or len(audit["nonce"]) < 16:
        raise PermissionError("v1.5 prelaunch nonce is invalid")
    expected_lock_paths = {"execution_control/execution_state.lock",
                           "worker_locks/worker_0.lock", "worker_locks/worker_1.lock",
                           "gpu_locks/gpu0.lock", "gpu_locks/gpu1.lock"}
    lock_rows = audit["worker_lock_inventory"]
    if ({row.get("path") for row in lock_rows if isinstance(row, Mapping)} != expected_lock_paths
            or len(lock_rows) != len(expected_lock_paths)
            or any(not isinstance(row, Mapping) or row.get("active") is not False
                   or (row.get("exists") is True and not isinstance(row.get("sha256"), str))
                   or (row.get("exists") is False and row.get("sha256") is not None)
                   for row in lock_rows)):
        raise PermissionError("v1.5 prelaunch lock inventory is malformed or reports an active lock")
    without = dict(audit)
    claimed = without.pop("audit_sha256", None)
    if claimed != sha256_json(without):
        raise PermissionError("v1.5 prelaunch audit digest mismatch")
    if max_age_monotonic is not None and not observation_is_fresh(
            started_monotonic=float(audit.get("observation_started_monotonic", -1)),
            now_monotonic=max_age_monotonic):
        raise PermissionError("v1.5 runtime observation is older than the 60 second bound")
    if source_snapshot_sha256 is None:
        if allow_released_hold or allow_gate_replaced:
            raise PermissionError("transactional prelaunch validation requires source snapshot identity")
        source_snapshot_sha256 = build_source_snapshot(root=root)["snapshot_sha256"]
    if audit.get("source_snapshot_sha256") != source_snapshot_sha256:
        raise PermissionError("v1.5 prelaunch audit source graph changed")
    pairs = audit.get("gpu_target_pairs")
    if not isinstance(pairs, list):
        raise PermissionError("v1.5 target pair list is missing")
    rows = audit.get("gpu_memory", {}).get("devices", [])
    selected = _validate_prelaunch_device_inventory(audit)
    validated = validate_target_rows(rows, pairs)
    if selected != validated or audit.get("gpu_target_pairs") != [
            {"index": index, "uuid": gpu_uuid} for index, gpu_uuid in sorted(selected.items())]:
        raise PermissionError("v1.5 prelaunch initial selected/eligible target projection differs")
    return validated


def create_deferred_receipt(secret: bytes, core: Mapping[str, Any]) -> dict[str, Any]:
    if len(secret) != 32:
        raise ValueError("deferral HMAC secret must be exactly 256 bits")
    required = {
        "schema": DEFER_SCHEMA, "outcome": "RESOURCE_DEFERRED_PRECLAIM",
        "canonical_claim": None, "claimed_run_id": None,
        "canonical_mutation_started": False, "attempt_created": False,
        "checkpoint_loaded": False,
    }
    payload = dict(core)
    for key, value in required.items():
        if payload.get(key) != value:
            raise PermissionError("deferral receipt missing exact field: " + key)
    required_fields = ("worker_id", "physical_gpu_index", "gpu_uuid", "frozen_target_pair",
        "supervisor_pid", "supervisor_start_ticks", "child_pid", "child_start_ticks",
        "invocation", "nonce", "expected_receipt_path", "execution_gate_sha256",
        "release_sha256", "queue_sha256", "source_incident_hold_sha256",
        "source_failure_nonce", "decision_utc", "decision_monotonic", "forbidden_flags")
    if any(key not in payload for key in required_fields):
        raise PermissionError("deferral receipt omits required invocation identity")
    if (not isinstance(payload.get("worker_id"), str)
            or payload.get("worker_id") not in ("0", "1")
            or payload.get("physical_gpu_index") not in (0, 1)
            or payload.get("physical_gpu_index") != int(payload["worker_id"])
            or not isinstance(payload.get("gpu_uuid"), str) or not payload["gpu_uuid"]
            or payload.get("frozen_target_pair") != {"index": payload["physical_gpu_index"],
                                                       "uuid": payload["gpu_uuid"]}):
        raise PermissionError("deferral receipt physical target identity is invalid")
    for key in ("supervisor_pid", "supervisor_start_ticks", "child_pid", "child_start_ticks", "invocation"):
        item = payload.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise PermissionError("deferral receipt process/invocation identity is invalid")
    if (not isinstance(payload.get("nonce"), str) or not payload["nonce"]
            or not isinstance(payload.get("expected_receipt_path"), str)
            or not os.path.isabs(payload["expected_receipt_path"])):
        raise PermissionError("deferral receipt nonce/path identity is invalid")
    for key in ("execution_gate_sha256", "release_sha256", "queue_sha256",
                "source_incident_hold_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise PermissionError("deferral receipt digest identity is invalid: " + key)
    if (not isinstance(payload.get("source_failure_nonce"), str)
            or not isinstance(payload.get("decision_utc"), str)
            or isinstance(payload.get("decision_monotonic"), bool)
            or not isinstance(payload.get("decision_monotonic"), (int, float))
            or not math.isfinite(float(payload["decision_monotonic"]))
            or payload.get("forbidden_flags") != {"resume": False, "retry": False,
                "new_attempt": False, "checkpoint_load": False, "formal": False, "test": False}):
        raise PermissionError("deferral receipt timing/forbidden flags are invalid")
    if payload.get("exit_code") != 75 or payload.get("boundary") not in (
            "CHILD_PRECLAIM", "SERIALIZED_PRECLAIM_MEMORY_CHECK"):
        raise PermissionError("deferral outcome/decision boundary is invalid")
    row = payload.get("telemetry", {})
    if (not isinstance(row, Mapping) or row.get("readable") is not True
            or isinstance(row.get("index"), bool) or not isinstance(row.get("index"), int)
            or row.get("index") not in (0, 1)
            or isinstance(row.get("memory_total_bytes"), bool)
            or isinstance(row.get("memory_used_bytes"), bool)
            or not isinstance(row.get("memory_total_bytes"), int)
            or not isinstance(row.get("memory_used_bytes"), int)
            or row.get("memory_total_bytes", 0) <= 0
            or not 0 <= row.get("memory_used_bytes", -1) <= row["memory_total_bytes"]
            or not isinstance(row.get("uuid"), str) or not row.get("uuid")):
        raise PermissionError("invalid telemetry cannot authenticate resource deferral")
    if (row.get("index") != payload.get("physical_gpu_index")
            or row.get("uuid") != payload.get("gpu_uuid")
            or isinstance(row.get("observation_started_monotonic"), bool)
            or not isinstance(row.get("observation_started_monotonic"), (int, float))
            or not isinstance(row.get("observed_at_utc"), str)):
        raise PermissionError("deferral receipt telemetry timestamps/target are invalid")
    observed_mono = float(row["observation_started_monotonic"])
    decision_mono = float(payload["decision_monotonic"])
    if not 0.0 <= decision_mono - observed_mono <= MEMORY_FRESH_SECONDS:
        raise PermissionError("deferral telemetry observation is stale at decision time")
    try:
        observed_utc = _dt.datetime.fromisoformat(row["observed_at_utc"].replace("Z", "+00:00"))
        decision_utc = _dt.datetime.fromisoformat(payload["decision_utc"].replace("Z", "+00:00"))
        utc_delta = (decision_utc - observed_utc).total_seconds()
    except (TypeError, ValueError) as exc:
        raise PermissionError("deferral telemetry UTC timestamps are malformed") from exc
    if not 0.0 <= utc_delta <= MEMORY_FRESH_SECONDS:
        raise PermissionError("deferral telemetry UTC observation is stale or future-dated")
    fraction = row.get("memory_used_fraction")
    if (fraction is not None and (isinstance(fraction, bool)
            or not isinstance(fraction, (int, float)) or not math.isfinite(float(fraction))
            or not math.isclose(float(fraction), row["memory_used_bytes"] /
                                row["memory_total_bytes"], rel_tol=1e-9, abs_tol=1e-9))):
        raise PermissionError("deferral telemetry fraction disagrees with integer byte counts")
    if 10 * row["memory_used_bytes"] < 9 * row["memory_total_bytes"]:
        raise PermissionError("eligible telemetry cannot produce resource deferral")
    payload["hmac_sha256"] = hmac.new(secret, canonical_json_bytes(payload), hashlib.sha256).hexdigest()
    return payload


def verify_deferred_receipt(secret: bytes, receipt: Mapping[str, Any], *,
                            expected: Mapping[str, Any]) -> None:
    if len(secret) != 32:
        raise PermissionError("deferral secret has invalid length")
    value = dict(receipt)
    mac = value.pop("hmac_sha256", None)
    expected_mac = hmac.new(secret, canonical_json_bytes(value), hashlib.sha256).hexdigest()
    if not isinstance(mac, str) or not hmac.compare_digest(mac, expected_mac):
        raise PermissionError("resource deferral HMAC is invalid")
    for key, exact in expected.items():
        if value.get(key) != exact:
            raise PermissionError("resource deferral identity mismatch: " + key)
    core_expected = create_deferred_receipt(secret, value)
    if core_expected != dict(receipt):
        raise PermissionError("resource deferral receipt core is incomplete or malformed")


def new_deferral_secret() -> bytes:
    return secrets.token_bytes(32)


def inspect_claim_outcome(output_root: os.PathLike | str, entries: Sequence[Mapping[str, Any]],
                          *, allowed_ids: set[str] | None = None) -> dict[str, Any]:
    """Read-only claim inventory; accepts distinct authenticated in-progress owners."""
    root = Path(output_root).resolve()
    claims = []
    changed = []
    for entry in entries:
        rid = str(entry.get("run_id", ""))
        paths = fresh_member_paths(root, rid)
        present = {name: os.path.lexists(path) for name, path in paths.items()}
        if any(present.values()):
            changed.append(rid)
        status_path = paths["status"]
        if present["status"]:
            if status_path.is_symlink() or not status_path.is_file():
                raise PermissionError("claim status path is not a regular file")
            status = _read_json(status_path)
            if status.get("status") == "CLAIMED":
                claims.append({"run_id": rid, "status": status, "paths": paths})
    allowed = allowed_ids or set()
    if any(item["run_id"] not in allowed for item in claims):
        raise PermissionError("unattributed in-progress peer claim exists")
    return {"claims": claims, "changed_run_ids": sorted(changed)}


def capture_f140_state(output_root: os.PathLike | str, entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    root = Path(output_root).resolve()
    state: dict[str, str] = {}
    for entry in entries:
        rid = str(entry.get("run_id", ""))
        if not rid:
            raise PermissionError("F140 source row lacks run ID")
        for kind, path in fresh_member_paths(root, rid).items():
            if os.path.lexists(path):
                if path.is_symlink():
                    digest = "SYMLINK:" + os.readlink(path)
                elif path.is_file():
                    digest = sha256_file(path)
                elif path.is_dir():
                    inventory = []
                    for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
                        rel = child.relative_to(path).as_posix()
                        if child.is_symlink():
                            inventory.append([rel, "SYMLINK", os.readlink(child)])
                        elif child.is_file():
                            inventory.append([rel, "FILE", sha256_file(child)])
                        elif child.is_dir():
                            inventory.append([rel, "DIRECTORY"])
                        else:
                            inventory.append([rel, "OTHER"])
                    digest = "TREE:" + sha256_json(inventory)
                else:
                    digest = "OTHER"
                state[rid + ":" + kind] = digest
    return state


def _authenticated_peer_progress(root: Path, run_id: str, status: Mapping[str, Any],
                                 entry: Mapping[str, Any]) -> bool:
    if status.get("status") not in ("CLAIMED", "VALIDATION_COMPLETE"):
        return False
    pid = status.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    lock_path = fresh_member_paths(root, run_id)["lock"]
    if lock_path.is_symlink() or not lock_path.is_file():
        return False
    try:
        with lock_path.open("r", encoding="utf-8") as handle:
            owner = json.load(handle)
    except (OSError, ValueError):
        return False
    identity_fields = ("run_id", "run_id_full_sha256", "canonical_key_json", "attempt_id",
                       "pid", "start_ticks", "nonce")
    if (not isinstance(owner, dict)
            or any(owner.get(key) != status.get(key) for key in identity_fields)
            or status.get("run_id") != run_id
            or status.get("run_id_full_sha256") != entry.get("run_id_full_sha256")
            or status.get("canonical_key_json") != entry.get("canonical_key_json")
            or status.get("attempt_id") != "attempt-00"
            or status.get("recovery_count") != 0 or status.get("resume_checkpoint") is not None):
        return False
    nonce = owner.get("nonce")
    if not isinstance(nonce, str) or not nonce or not _canonical_key_identity_is_valid(entry):
        return False

    def lock_is_held(path: Path) -> bool:
        if path.is_symlink() or not path.is_file():
            return False
        try:
            fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    return False
            finally:
                os.close(fd)
        except OSError:
            return False

    gpu_index = status.get("gpu_index")
    gpu_uuid = status.get("gpu_uuid")
    if (isinstance(gpu_index, bool) or gpu_index not in (0, 1)
            or not isinstance(gpu_uuid, str) or not gpu_uuid):
        return False
    try:
        gate_path = root.parent / "execution_isolation_gate.json"
        gate = _read_json(gate_path)
        target_pairs = gate.get("gpu_target_pairs", [])
    except (OSError, PermissionError, ValueError):
        return False
    if not any(isinstance(row, Mapping) and row.get("index") == gpu_index
               and row.get("uuid") == gpu_uuid for row in target_pairs):
        return False
    worker_id = str(gpu_index)
    worker_lock = root / "worker_locks" / ("worker_{}.lock".format(worker_id))
    worker_control = root / "worker_locks" / ("worker_{}.control".format(worker_id))
    gpu_lock = root / "gpu_locks" / ("gpu{}.lock".format(gpu_index))

    def owner_record(path: Path) -> dict[str, Any] | None:
        if path.is_symlink() or not path.is_file():
            return None
        try:
            value = _read_json(path)
        except (OSError, ValueError, PermissionError):
            return None
        return dict(value) if isinstance(value, Mapping) else None

    def worker_owner_matches(supervisor_pid: int, supervisor_pgid: int,
                             started: Mapping[str, Any]) -> bool:
        """Join the supervisor-owned lock/control record, not the child nonce."""
        if worker_control.is_symlink() or not worker_control.is_file():
            return False
        try:
            fields = worker_control.read_text(encoding="ascii").split()
            if len(fields) != 6:
                return False
            control_pid, control_pgid, control_ticks = map(int, fields[:3])
            control_nonce, control_host, control_worker_id = fields[3:]
            owner_value = owner_record(worker_lock)
            from stage_F0_worker_control import linux_process_identity
            supervisor_now = linux_process_identity(supervisor_pid)
        except (OSError, ValueError, PermissionError):
            return False
        if owner_value is None:
            return False
        invocation_host = started.get("host")
        return bool(
            supervisor_pid != pid and control_pid == supervisor_pid
            and control_pgid == supervisor_pgid == supervisor_pid
            and control_ticks > 0 and control_nonce
            and isinstance(invocation_host, str) and invocation_host.strip()
            and control_host == invocation_host
            and control_host == owner_value.get("host")
            and control_worker_id == worker_id
            and owner_value.get("pid") == supervisor_pid
            and owner_value.get("pgid") == supervisor_pgid
            and owner_value.get("start_ticks") == control_ticks
            and owner_value.get("nonce") == control_nonce
            and owner_value.get("worker_id") == worker_id
            and supervisor_now.get("pid") == supervisor_pid
            and supervisor_now.get("pgid") == supervisor_pgid
            and supervisor_now.get("start_ticks") == control_ticks
            and started.get("pid") == supervisor_pid
            and started.get("pgid") == supervisor_pgid
            and started.get("worker_id") == worker_id
            and started.get("nonce") == nonce
            and status.get("nonce") == nonce
            and control_nonce != nonce)

    def gpu_owner_matches(started: Mapping[str, Any]) -> bool:
        """Accept the bootstrap owner schema and validate optional later fields."""
        value = owner_record(gpu_lock)
        if value is None:
            return False
        invocation_host = started.get("host")
        if not isinstance(invocation_host, str) or not invocation_host.strip():
            return False
        status_host = status.get("host")
        if status_host is not None and status_host != invocation_host:
            return False
        # The runner initially publishes only these fields while holding the
        # GPU flock. Status schemas may omit host; use the same authenticated
        # CHILD_STARTED invocation host that is joined to worker lock/control.
        # Richer canonical identity may be written after claim.
        required = {"pid": pid, "host": invocation_host,
                    "gpu_index": gpu_index, "gpu_uuid": gpu_uuid}
        if not all(value.get(key) == wanted for key, wanted in required.items()):
            return False
        claimed = value.get("claimed_unix")
        if (isinstance(claimed, bool) or not isinstance(claimed, (int, float))
                or not math.isfinite(float(claimed)) or claimed <= 0):
            return False
        # Fields that the runner may add after initial publication strengthen
        # attribution, but are not required during the bootstrap claim window.
        optional = {"worker_id": worker_id, "nonce": nonce,
                    "start_ticks": status.get("start_ticks"), "run_id": run_id,
                    "run_id_full_sha256": entry.get("run_id_full_sha256"),
                    "canonical_key_json": entry.get("canonical_key_json"),
                    "attempt_id": "attempt-00"}
        return all(key not in value or value.get(key) == expected
                   for key, expected in optional.items())
    # Match one supervisor invocation and, for active claims, the live child
    # start-time. A completed peer requires both its start and completion rows.
    logs = root / "worker_logs"
    for log in (logs / ("audited_worker_{}.jsonl".format(worker_id)),):
        try:
            rows = _read_jsonl_exact(log)
        except (PermissionError, OSError):
            continue
        started = next((row for row in rows if row.get("event") == "CHILD_STARTED"
            and row.get("nonce") == nonce and row.get("child_pid") == pid
            and row.get("child_start_ticks") == status.get("start_ticks")
            and row.get("worker_id") == worker_id
            and isinstance(row.get("invocation"), int) and row.get("invocation") > 0), None)
        if started is None:
            continue
        if status.get("status") == "CLAIMED":
            try:
                from stage_F0_worker_control import linux_process_identity
                child_now = linux_process_identity(int(pid))
                supervisor_now = linux_process_identity(int(started.get("pid", -1)))
            except (OSError, ValueError, PermissionError):
                continue
            if (child_now.get("start_ticks") == status.get("start_ticks")
                    and child_now.get("ppid") == started.get("pid")
                    and supervisor_now.get("pid") == started.get("pid")
                    and supervisor_now.get("start_ticks") == started.get("supervisor_start_ticks",
                                                                          supervisor_now.get("start_ticks"))
                    and supervisor_now.get("pgid") == started.get("pgid")
                    and lock_is_held(lock_path) and lock_is_held(worker_lock)
                    and lock_is_held(gpu_lock)
                    and worker_owner_matches(int(started.get("pid", -1)),
                                             int(started.get("pgid", -1)), started)
                    and gpu_owner_matches(started)):
                return True
            continue
        attempt = root / run_id / "attempts" / "attempt-00"
        manifest_path, summary_path = attempt / "run_manifest.json", attempt / "trajectory_summary.json"
        if (not manifest_path.is_file() or manifest_path.is_symlink()
                or not summary_path.is_file() or summary_path.is_symlink()):
            continue
        manifest, summary = _read_json(manifest_path), _read_json(summary_path)
        if any(value.get("status") != "VALIDATION_COMPLETE"
               or value.get("run_id") != run_id
               or value.get("run_id_full_sha256") != entry.get("run_id_full_sha256")
               or value.get("canonical_key_json") != entry.get("canonical_key_json")
               or value.get("attempt_id") != "attempt-00" for value in (manifest, summary)):
            continue
        completion = next((row for row in rows if row.get("event") == "CHILD_VALIDATION_COMPLETE"
               and row.get("nonce") == nonce
               and row.get("invocation") == started.get("invocation")
               and row.get("child_pid") == pid and row.get("run_id") == run_id), None)
        receipt_path_value = started.get("receipt_path")
        if completion is None or not isinstance(receipt_path_value, str):
            continue
        receipt_path = Path(receipt_path_value)
        if receipt_path.is_symlink() or not receipt_path.is_file():
            continue
        try:
            receipt_sha = sha256_file(receipt_path)
            if completion.get("receipt_sha256") != receipt_sha:
                continue
            import run_search_claim_worker as claim_worker
            repo_root = root.parents[2] if root.name == "runs" else ROOT
            receipt = claim_worker._validate_completion_receipt(
                receipt_path, child_pid=int(pid), nonce=nonce, worker_id=worker_id,
                supervisor_pid=int(started.get("pid", -1)),
                worker_pgid=int(started.get("pgid", -1)),
                queue_path=repo_root / QUEUE_REL, output_root=root)
        except (OSError, ValueError, PermissionError, TypeError):
            continue
        if (receipt.get("run_id") == run_id
                and receipt.get("run_id_full_sha256") == entry.get("run_id_full_sha256")
                and receipt.get("canonical_key_json") == entry.get("canonical_key_json")
                and receipt.get("attempt_id") == "attempt-00"):
            return True
    return False


def verify_deferred_no_mutation(output_root: os.PathLike | str,
                                entries: Sequence[Mapping[str, Any]], *,
                                before: Mapping[str, str], child_pid: int,
                                child_start_ticks: int, nonce: str) -> None:
    root = Path(output_root).resolve()
    after = capture_f140_state(root, entries)
    own_prefix = None
    # Any current status/owner attributable to this child's full identity is a
    # forbidden canonical mutation, even if it appeared and then was altered.
    for entry in entries:
        rid = str(entry.get("run_id", ""))
        status_path = fresh_member_paths(root, rid)["status"]
        lock_path = fresh_member_paths(root, rid)["lock"]
        if status_path.is_file() and not status_path.is_symlink():
            try:
                status = _read_json(status_path)
            except PermissionError:
                raise
            if status.get("pid") == child_pid and status.get("nonce") == nonce:
                raise PermissionError("deferred child wrote a canonical status")
        if lock_path.is_file() and not lock_path.is_symlink():
            try:
                owner = _read_json(lock_path)
            except PermissionError:
                raise
            if (owner.get("pid") == child_pid and owner.get("nonce") == nonce
                    and owner.get("start_ticks") == child_start_ticks):
                raise PermissionError("deferred child wrote a canonical lock owner")
    before_ids = {key.split(":", 1)[0] for key in before}
    after_ids = {key.split(":", 1)[0] for key in after}
    entry_by_id = {str(row.get("run_id")): row for row in entries}
    changed_ids = {key.split(":", 1)[0] for key in set(before) | set(after)
                   if before.get(key) != after.get(key)}
    for rid in changed_ids:
        entry = entry_by_id.get(rid)
        if entry is None:
            raise PermissionError("changed F140 member is outside the frozen queue identity")
        status_path = fresh_member_paths(root, rid)["status"]
        if not status_path.is_file() or status_path.is_symlink():
            raise PermissionError("changed F140 activity has no regular authenticated status")
        status = _read_json(status_path)
        if not _authenticated_peer_progress(root, rid, status, entry):
            raise PermissionError("changed F140 activity is not an authenticated live peer progress")
    # Existing peer claims may advance while this child defers. Their owner
    # and supervisor CHILD_STARTED identities must remain consistent.
    for key in set(before) & set(after):
        if before[key] == after[key] or not key.endswith(":status"):
            continue
        rid = key.split(":", 1)[0]
        status_path = fresh_member_paths(root, rid)["status"]
        if status_path.is_file() and not status_path.is_symlink():
            status = _read_json(status_path)
            entry = entry_by_id.get(rid)
            if entry is None or not _authenticated_peer_progress(root, rid, status, entry):
                raise PermissionError("changed peer progress is not independently authenticated")


def write_deferred_receipt(path: os.PathLike | str, secret: bytes,
                           core: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    if os.path.lexists(target):
        raise PermissionError("deferral receipt path already exists")
    receipt = create_deferred_receipt(secret, core)
    durable_json_create(target, receipt, mode=0o600)
    return receipt


def record_runtime_observation(runs_root: os.PathLike | str, *, transaction_id: str,
                               stage: str, target: Mapping[str, Any],
                               telemetry: Mapping[str, Any], go: bool,
                               source_snapshot_sha256: str,
                               prepared_identity_sha256: str,
                               prelaunch_audit_sha256: str,
                               reason: str,
                               root: os.PathLike | str = ROOT) -> str:
    """Append-only resource observation; intentionally outside COMMITTED hashes."""
    if stage not in OBSERVATION_STAGES:
        raise PermissionError("unknown runtime decision boundary")
    if not isinstance(go, bool) or not isinstance(transaction_id, str) or not transaction_id:
        raise PermissionError("runtime observation transaction/decision is invalid")
    if (not isinstance(target, Mapping) or not isinstance(telemetry, Mapping)
            or target.get("index") != telemetry.get("index")
            or target.get("uuid") != telemetry.get("uuid")):
        raise PermissionError("runtime observation target differs from telemetry identity")
    eligible = fresh_gpu_eligible(telemetry)
    expected_reason = ("ASSIGNED_GPU_BELOW_90_PERCENT" if eligible
                       else "ASSIGNED_GPU_AT_OR_ABOVE_90_PERCENT")
    if go is not eligible or reason != expected_reason:
        raise PermissionError("runtime observation reason/GO differs from integer threshold")
    row = {"schema": "F0-V15-RUNTIME-OBSERVATION-v2", "transaction_id": transaction_id,
           "stage": stage, "target": dict(target), "telemetry": dict(telemetry),
           "go": bool(go), "source_snapshot_sha256": source_snapshot_sha256,
           "prepared_identity_sha256": prepared_identity_sha256,
           "prepared_identity_kind": "CANONICAL_PREPARED_REQUEST",
           "prelaunch_audit_sha256": prelaunch_audit_sha256,
           "reason": reason,
           "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
           "monotonic": time.monotonic()}
    raw = canonical_json_bytes(row)
    _append_jsonl_durable(Path(runs_root).resolve() / "execution_control" /
                          "postbootstrap_v15_runtime_observations.jsonl", row)
    return sha256_bytes(raw)


def validate_observation_ledger(path: os.PathLike | str, *,
                                expected_identity: Mapping[str, Any] | None = None
                                ) -> list[dict[str, Any]]:
    rows = _read_jsonl_exact(Path(path), missing_ok=True)
    graph_identity = None
    for row in rows:
        target, telemetry = row.get("target"), row.get("telemetry")
        total = telemetry.get("memory_total_bytes") if isinstance(telemetry, Mapping) else None
        used = telemetry.get("memory_used_bytes") if isinstance(telemetry, Mapping) else None
        try:
            timestamp = _dt.datetime.fromisoformat(str(row.get("utc")))
            observed_timestamp = _dt.datetime.fromisoformat(
                str(telemetry.get("observed_at_utc"))) if isinstance(telemetry, Mapping) else None
        except (TypeError, ValueError):
            timestamp, observed_timestamp = None, None
        mono = row.get("monotonic")
        observed_mono = (telemetry.get("observation_started_monotonic")
                         if isinstance(telemetry, Mapping) else None)
        fraction = telemetry.get("memory_used_fraction") if isinstance(telemetry, Mapping) else None
        def digest_ok(value: Any) -> bool:
            return (isinstance(value, str) and len(value) == 64
                    and all(c in "0123456789abcdef" for c in value))
        if (row.get("schema") != "F0-V15-RUNTIME-OBSERVATION-v2"
                or not isinstance(row.get("transaction_id"), str)
                or row.get("stage") not in OBSERVATION_STAGES
                or not isinstance(target, Mapping) or not isinstance(telemetry, Mapping)
                or isinstance(target.get("index"), bool) or target.get("index") not in (0, 1)
                or not isinstance(target.get("uuid"), str)
                or telemetry.get("index") != target.get("index")
                or telemetry.get("uuid") != target.get("uuid")
                or not isinstance(row.get("go"), bool)
                or not digest_ok(row.get("source_snapshot_sha256"))
                or not digest_ok(row.get("prepared_identity_sha256"))
                or row.get("prepared_identity_kind") != "CANONICAL_PREPARED_REQUEST"
                or not digest_ok(row.get("prelaunch_audit_sha256"))
                or row.get("reason") not in ("ASSIGNED_GPU_BELOW_90_PERCENT",
                                                "ASSIGNED_GPU_AT_OR_ABOVE_90_PERCENT")
                or isinstance(total, bool) or not isinstance(total, int) or total <= 0
                or isinstance(used, bool) or not isinstance(used, int) or not 0 <= used <= total
                or telemetry.get("readable") is not True
                or telemetry.get("eligible") is not (10 * used < 9 * total)
                or not isinstance(telemetry.get("observed_at_utc"), str)
                or observed_timestamp is None or observed_timestamp.tzinfo is None
                or observed_timestamp.utcoffset() != _dt.timedelta(0)
                or isinstance(observed_mono, bool)
                or not isinstance(observed_mono, (int, float))
                or not math.isfinite(float(observed_mono))
                or timestamp is None or timestamp.tzinfo is None
                or timestamp.utcoffset() != _dt.timedelta(0)
                or isinstance(mono, bool) or not isinstance(mono, (int, float))
                or not math.isfinite(float(mono))):
            raise PermissionError("v1.5 runtime observation ledger contains invalid identity or truncated data")
        eligible = 10 * used < 9 * total
        expected_reason = ("ASSIGNED_GPU_BELOW_90_PERCENT" if eligible
                           else "ASSIGNED_GPU_AT_OR_ABOVE_90_PERCENT")
        if row["go"] is not eligible or row["reason"] != expected_reason:
            raise PermissionError("runtime observation reason/GO differs from integer GPU threshold")
        if fraction is not None:
            if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                    or not math.isfinite(float(fraction))
                    or not math.isclose(float(fraction), used / total,
                                        rel_tol=1e-9, abs_tol=1e-9)):
                raise PermissionError("runtime observation fraction differs from integer telemetry")
        utc_delta = (timestamp.astimezone(_dt.timezone.utc)
                     - observed_timestamp.astimezone(_dt.timezone.utc)).total_seconds()
        mono_delta = float(mono) - float(observed_mono)
        if (utc_delta < -1.0 or utc_delta > 60.0
                or mono_delta < 0.0 or mono_delta > 60.0):
            raise PermissionError("runtime observation is stale or decision time precedes query")
        identity = (row["transaction_id"], row["source_snapshot_sha256"],
                    row["prelaunch_audit_sha256"], row["prepared_identity_sha256"],
                    row["prepared_identity_kind"])
        if graph_identity is None:
            graph_identity = identity
        elif identity != graph_identity:
            raise PermissionError("runtime observations span different immutable transaction graphs")
        if expected_identity is not None:
            expected = (expected_identity.get("transaction_id"),
                        expected_identity.get("source_snapshot_sha256"),
                        expected_identity.get("prelaunch_audit_sha256"),
                        expected_identity.get("prepared_identity_sha256"),
                        "CANONICAL_PREPARED_REQUEST")
            if identity != expected:
                raise PermissionError("runtime observation differs from current PREPARED source graph")
    return rows


def _record_live_decision(runs_root: Path, prepared: Mapping[str, Any],
                          telemetry_rows: Sequence[Mapping[str, Any]], *, stage: str) -> bool:
    path = runs_root / "execution_control" / "postbootstrap_v15_runtime_observations.jsonl"
    prepared_sha = sha256_json(dict(prepared))
    expected_identity = {"transaction_id": str(prepared["transaction_id"]),
        "source_snapshot_sha256": str(prepared["source_snapshot_sha256"]),
        "prelaunch_audit_sha256": str(prepared["prelaunch_audit_sha256"]),
        "prepared_identity_sha256": prepared_sha}
    validate_observation_ledger(path, expected_identity=expected_identity)
    all_go = True
    for telemetry in telemetry_rows:
        target = {"index": telemetry.get("index"), "uuid": telemetry.get("uuid")}
        eligible = fresh_gpu_eligible(telemetry)
        decision_reason = ("ASSIGNED_GPU_BELOW_90_PERCENT" if eligible
                           else "ASSIGNED_GPU_AT_OR_ABOVE_90_PERCENT")
        if stage not in ("LOCKED_RELEASE_DECISION", "PRE_HOLD_RELEASE_DECISION"):
            raise PermissionError("unknown runtime decision boundary")
        record_runtime_observation(runs_root, transaction_id=str(prepared["transaction_id"]),
            stage=stage, target=target, telemetry=telemetry,
            go=eligible,
            source_snapshot_sha256=str(prepared["source_snapshot_sha256"]),
            prepared_identity_sha256=prepared_sha,
            prelaunch_audit_sha256=str(prepared["prelaunch_audit_sha256"]),
            reason=decision_reason)
        all_go = all_go and eligible
    validate_observation_ledger(path, expected_identity=expected_identity)
    return all_go


def prepare_release_identity(*, nonce: str, principal: str, approval_utc: str,
                             prelaunch_sha256: str, source_snapshot_sha256: str,
                             targets: Sequence[Mapping[str, Any]],
                             root: os.PathLike | str = ROOT) -> dict[str, Any]:
    """Build the immutable PREPARED identity, without writing or releasing."""
    if not isinstance(nonce, str) or len(nonce) < 16:
        raise ValueError("release nonce must be at least 16 characters")
    if not isinstance(principal, str) or not principal.strip():
        raise ValueError("named principal identity is required")
    target_pairs = [{"index": row.get("index"), "uuid": row.get("uuid")} for row in targets]
    if len(target_pairs) == 0 or len({(x["index"], x["uuid"]) for x in target_pairs}) != len(target_pairs):
        raise PermissionError("release target pairs must be nonempty and unique")
    if any(x["index"] not in (0, 1) or not x["uuid"] for x in target_pairs):
        raise PermissionError("release target pair is invalid")
    _assert_frozen_artifacts(root)
    return {
        "schema": LEDGER_SCHEMA, "event": "PREPARED", "transaction_id": uuid.uuid5(
            uuid.NAMESPACE_URL, "f0-v15:" + nonce).hex,
        "nonce": nonce, "approved_by": principal.strip(), "approved_by_role": "principal",
        "intent": SCOPE, "approval_utc": approval_utc,
        "per_device_resource_rule": PER_DEVICE_RULE,
        "prepared_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "authorization": {"resume": False, "retry": False, "new_attempt": False,
                           "checkpoint_load": False, "formal": False, "test": False},
        "source_incident_hold_sha256": INCIDENT_HOLD_SHA256,
        "source_failure_nonce": INCIDENT_NONCE,
        "source_snapshot_sha256": source_snapshot_sha256,
        "prelaunch_audit_sha256": prelaunch_sha256,
        "targets": target_pairs,
        "historical_v14_release_sha256": V14_RELEASE_SHA256,
        "historical_v14_release_ledger_sha256": V14_RELEASE_LEDGER_SHA256,
        "v14_migration_ledger_sha256": V14_MIGRATION_LEDGER_SHA256,
        "search_queue_sha256": QUEUE_SHA256,
    }


def _canonical_indented_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False).encode("utf-8") + b"\n"


def _event_index(rows: Sequence[Mapping[str, Any]], event: str) -> int | None:
    hits = [index for index, row in enumerate(rows)
            if row.get("event") == event and row.get("schema") == LEDGER_SCHEMA]
    if len(hits) > 1:
        raise PermissionError("v1.5 transaction contains duplicate " + event)
    return hits[0] if hits else None


def _transaction_rows(path: Path, transaction_id: str) -> list[dict[str, Any]]:
    rows = _read_jsonl_exact(path, missing_ok=True)
    if not rows:
        return []
    if rows[0].get("event") != "PREPARED" or rows[0].get("schema") != LEDGER_SCHEMA:
        raise PermissionError("v1.5 ledger does not begin with PREPARED")
    if rows[0].get("transaction_id") != transaction_id:
        raise PermissionError("v1.5 ledger belongs to a different nonce/transaction")
    allowed = ["PREPARED", "RELEASE_ARTIFACT_DURABLE", "PENDING_GATE_DURABLE",
               "RELEASE_POINTER_DURABLE", "HOLD_RELEASED", "COMMITTED"]
    seen: list[str] = []
    for row in rows:
        if row.get("schema") != LEDGER_SCHEMA or row.get("transaction_id") != transaction_id:
            raise PermissionError("v1.5 ledger contains a foreign transaction event")
        name = row.get("event")
        if name not in allowed or len(seen) >= len(allowed) or name != allowed[len(seen)]:
            raise PermissionError("v1.5 ledger event order/identity is invalid")
        seen.append(name)
    return rows


def _append_once(path: Path, rows: list[dict[str, Any]], event: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(event.get("event"))
    index = _event_index(rows, name)
    if index is not None:
        if rows[index] != event:
            raise PermissionError("v1.5 durable event differs from exact expected event")
        return rows
    _append_jsonl_durable(path, event)
    rows.append(event)
    return rows


def _check_exact_or_source(path: Path, source_sha: str | None, expected: bytes,
                           *, label: str) -> str:
    if not os.path.lexists(path):
        if source_sha is None:
            return "absent"
        raise PermissionError(label + " unexpectedly disappeared")
    if path.is_symlink() or not path.is_file():
        raise PermissionError(label + " is not an ordinary file")
    actual = sha256_file(path)
    expected_sha = sha256_bytes(expected)
    if actual == expected_sha:
        return "expected"
    if source_sha is not None and actual == source_sha:
        return "source"
    raise PermissionError(label + " differs from both exact transaction states")


def _released_hold_projection(source_hold: Mapping[str, Any], prepared: Mapping[str, Any],
                              source_doc: Mapping[str, Any]) -> dict[str, Any]:
    historical = source_doc.get("historical_protection_snapshot")
    if not isinstance(historical, Mapping):
        raise PermissionError("source snapshot lacks historical protection identity")
    completed_ids = historical.get("completed_ids")
    fresh_ids = historical.get("fresh_ids")
    if not isinstance(completed_ids, list) or not isinstance(fresh_ids, list):
        raise PermissionError("source snapshot lacks frozen C97/F140 identities")
    from stage_F0_postabnormal_v14 import H51_RUN_IDS
    released = dict(source_hold)
    release_event = {
        "event": "V15_PRINCIPAL_FRESH_ONLY_RELEASE",
        "successor_release_id": prepared["transaction_id"],
        "source_incident_hold_sha256": INCIDENT_HOLD_SHA256,
        "source_failure_sha256": INCIDENT_FAILURE_SHA256,
        "historical_v14_release_sha256": V14_RELEASE_SHA256,
        "historical_v14_release_ledger_sha256": V14_RELEASE_LEDGER_SHA256,
        "frozen_C97_sha256": sha256_json(completed_ids),
        "frozen_H51_sha256": sha256_json(list(H51_RUN_IDS)),
        "fresh_F140_sha256": sha256_json(fresh_ids),
        "target_pairs": prepared.get("targets"),
        "timestamp_utc": prepared["released_at_utc"],
    }
    events = list(released.get("events", []))
    if any(isinstance(row, Mapping) and row.get("event") == release_event["event"]
           for row in events):
        raise PermissionError("source hold already contains a v1.5 release event")
    events.append(release_event)
    released["events"] = events
    released.update({"active": False,
        "successor_release_id": prepared["transaction_id"],
        "historical_released_at_utc": source_hold.get("released_at_utc"),
        "released_at_utc": prepared["released_at_utc"],
        "source_incident_hold_sha256": INCIDENT_HOLD_SHA256})
    return released


def commit_principal_release(*, runs_root: os.PathLike | str = RUNS,
                             prepared: Mapping[str, Any], gate_pending: Mapping[str, Any],
                             release_payload: Mapping[str, Any],
                             live_observation: Mapping[str, Any],
                             fault_after: str | None = None) -> dict[str, Any]:
    """Durably publish or exactly resume one identity-bound v1.5 transaction.

    This is the reviewed principal release API. Implementation qualification
    exercises it only in an isolated root; callers must independently enforce
    staged review and obtain explicit principal approval before production use.
    """
    runs = Path(runs_root).resolve()
    repo = runs.parents[2] if runs.name == "runs" else runs
    if (prepared.get("schema") != LEDGER_SCHEMA or prepared.get("event") != "PREPARED"
            or prepared.get("approved_by_role") != "principal" or prepared.get("intent") != SCOPE
            or prepared.get("source_incident_hold_sha256") != INCIDENT_HOLD_SHA256
            or prepared.get("source_failure_nonce") != INCIDENT_NONCE
            or not isinstance(prepared.get("prepared_at_utc"), str)):
        raise PermissionError("v1.5 PREPARED identity is malformed or not principal-authorized")
    if not isinstance(prepared.get("nonce"), str) or len(prepared["nonce"]) < 16:
        raise PermissionError("v1.5 transaction nonce is invalid")
    txid = str(prepared.get("transaction_id", ""))
    if txid != uuid.uuid5(uuid.NAMESPACE_URL, "f0-v15:" + prepared["nonce"]).hex:
        raise PermissionError("v1.5 transaction ID is not derived from its nonce")
    if release_payload.get("schema") != RELEASE_SCHEMA or release_payload.get("release_id") != txid:
        raise PermissionError("v1.5 release artifact identity differs from PREPARED")
    if not isinstance(gate_pending, Mapping) or gate_pending.get("schema") != SCHEMA:
        raise PermissionError("v1.5 pending gate schema is invalid")
    if gate_pending.get("launch_authorized") is not False or gate_pending.get("authorization") != prepared.get("authorization"):
        raise PermissionError("v1.5 pending gate is authorized or has different forbidden flags")
    if gate_pending.get("source_snapshot_sha256") != prepared.get("source_snapshot_sha256"):
        raise PermissionError("v1.5 pending gate differs from PREPARED source identity")
    if gate_pending.get("gpu_target_pairs") != prepared.get("targets"):
        raise PermissionError("v1.5 pending gate targets differ from PREPARED target pairs")
    if gate_pending.get("prepared_release_id") != txid:
        raise PermissionError("v1.5 pending gate is not bound to exact PREPARED identity")

    control_dir = runs / "execution_control"
    control_dir.mkdir(parents=True, exist_ok=True)
    ledger = control_dir / "postbootstrap_v15_release.jsonl"
    hold_path = control_dir / "global_hold.json"
    gate_path = runs.parent / "execution_isolation_gate.json"
    release_pointer = control_dir / "global_hold_release.json"
    release_path = control_dir / "global_hold_release_v15.json"
    pending_path = control_dir / ("execution_isolation_gate_v15_pending_" + prepared["nonce"] + ".json")
    final_archive = control_dir / ("execution_isolation_gate_v15_released_" + prepared["nonce"] + ".json")
    pending_bytes = _canonical_indented_bytes(gate_pending)
    release_bytes = _canonical_indented_bytes(release_payload)
    expected_release_fields = {
        "nonce": prepared["nonce"], "approved_by": prepared["approved_by"],
        "approved_by_role": "principal", "principal_intent": SCOPE,
        "per_device_resource_rule": PER_DEVICE_RULE,
        "approval_utc": prepared["approval_utc"],
        "source_incident_hold_sha256": INCIDENT_HOLD_SHA256,
        "source_failure_nonce": INCIDENT_NONCE,
        "source_snapshot_sha256": prepared["source_snapshot_sha256"],
        "prelaunch_audit_sha256": prepared["prelaunch_audit_sha256"],
        "targets": prepared["targets"], "authorization": prepared["authorization"],
        "pending_gate_sha256": sha256_bytes(pending_bytes),
    }
    if any(release_payload.get(key) != value for key, value in expected_release_fields.items()):
        raise PermissionError("v1.5 immutable release does not bind exact principal/evidence identity")

    with execution_state_lock(runs), contextlib.ExitStack() as resource_stack:
        existing = _transaction_rows(ledger, txid)
        if existing:
            immutable_prepared = existing[0]
            for key in ("nonce", "transaction_id", "approved_by", "approved_by_role",
                        "approval_utc", "prepared_at_utc", "source_snapshot_sha256",
                        "prelaunch_audit_sha256", "targets", "authorization", "intent"):
                if immutable_prepared.get(key) != prepared.get(key):
                    raise PermissionError("reentry differs from immutable PREPARED identity: " + key)
            if (immutable_prepared.get("pending_gate_sha256") != sha256_bytes(pending_bytes)
                    or immutable_prepared.get("release_sha256") != sha256_bytes(release_bytes)):
                raise PermissionError("reentry pending gate/release artifact bytes differ")
        # A fully published exact graph is a read-only acknowledgement. It does
        # not append observations, touch pointers, or reevaluate stale telemetry.
        if existing and _event_index(existing, "COMMITTED") is not None:
            canonical = _read_json(gate_path)
            if (canonical.get("launch_authorized") is True
                    and canonical.get("transaction_state") == "COMMITTED"):
                terminal = validate_terminal_graph(canonical, root=repo, lock_already_held=True)
                committed = existing[_event_index(existing, "COMMITTED")]
                expected_release_sha = sha256_bytes(release_bytes)
                expected_pending_sha = sha256_bytes(pending_bytes)
                expected_hold_sha = sha256_file(hold_path)
                if (committed.get("release_sha256") != expected_release_sha
                        or committed.get("pending_gate_sha256") != expected_pending_sha
                        or committed.get("released_hold_sha256") != expected_hold_sha):
                    raise PermissionError("terminal acknowledgement identity differs from requested release")
                expected_final = dict(gate_pending)
                expected_final.update({"launch_authorized": True,
                    "transaction_state": "COMMITTED",
                    "release_artifact_path": str(release_path.relative_to(repo)),
                    "release_sha256": expected_release_sha,
                    "released_hold_sha256": expected_hold_sha,
                    "committed_ledger_sha256": terminal["ledger_sha256"]})
                if canonical != expected_final:
                    raise PermissionError("terminal gate is not the exact requested pending-gate projection")
                return {"status": "COMMITTED_IDEMPOTENT", "gate_sha256": sha256_file(gate_path),
                        "ledger_sha256": terminal["ledger_sha256"]}
        # Only a nonterminal transaction needs a fresh assigned-device reading.
        # Exact terminal acknowledgement above verifies the complete graph and
        # durability under lock without consulting telemetry or writing logs.
        resource_stack.enter_context(_admission_resource_locks(runs))
        _assert_current_release_resources(runs, repo)
        pairs = validate_target_rows(
            [{"index": row.get("index"), "uuid": row.get("uuid"), "readable": True,
              "eligible": True, "memory_total_bytes": 1000, "memory_used_bytes": 0}
             for row in prepared.get("targets", [])], prepared.get("targets", []))
        approved_pairs = [{"index": index, "uuid": device_uuid} for index, device_uuid in sorted(pairs.items())]
        if gate_pending.get("gpu_target_pairs") != approved_pairs:
            raise PermissionError("v1.5 pending gate targets differ from live prepared target pairs")

        def fresh_decision(stage: str) -> list[dict[str, Any]]:
            while True:
                observed = [query_gpu_memory(gpu_index=index, expected_uuid=device_uuid)
                            for index, device_uuid in sorted(pairs.items())]
                all_go = _record_live_decision(runs, prepared, observed, stage=stage)
                if any(not observation_is_fresh(started_monotonic=float(
                        row["observation_started_monotonic"])) for row in observed):
                    continue
                if not all_go:
                    raise ResourceDeferred(observed, stage)
                validate_target_rows(observed, prepared.get("targets", []))
                return observed

        locked_observation = fresh_decision("LOCKED_RELEASE_DECISION")
        if not existing:
            if _assert_frozen_artifacts(repo) is None:
                raise PermissionError("frozen source evidence unavailable")
            if sha256_file(hold_path) != INCIDENT_HOLD_SHA256:
                raise PermissionError("source incident hold changed before PREPARED")
            # Validate the real pending graph before the first durable mutation.
            validate_current_gate(gate_pending, root=repo, require_release=False)
            if not gate_path.is_file() or gate_path.is_symlink():
                raise PermissionError("canonical source gate is not an ordinary file")
            if not release_pointer.is_file() or release_pointer.is_symlink():
                raise PermissionError("historical v1.4 release pointer is absent or non-regular")
            preimages = {"gate": sha256_file(gate_path),
                         "release_pointer": sha256_file(release_pointer),
                         "hold": sha256_file(hold_path)}
            if preimages["gate"] != "f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64":
                raise PermissionError("canonical gate is not the frozen v1.4 source preimage")
            if preimages["hold"] != INCIDENT_HOLD_SHA256:
                raise PermissionError("active hold is not the frozen incident source preimage")
            for label, path, digest in (("gate", gate_path, preimages["gate"]),
                                        ("hold", hold_path, preimages["hold"]),
                                        ("release_pointer", release_pointer,
                                         preimages["release_pointer"])):
                archive = control_dir / ("v15_preimage_" + label + "_" + digest + ".bin")
                if os.path.lexists(archive):
                    if archive.is_symlink() or not archive.is_file() or sha256_file(archive) != digest:
                        raise PermissionError("existing content-addressed preimage archive conflicts")
                else:
                    durable_bytes_create(archive, path.read_bytes())
            frozen = _assert_frozen_artifacts(repo)
            source_rel_for_hold = gate_pending.get("source_snapshot_path")
            source_path_for_hold = (_repo_path(repo, source_rel_for_hold)
                                    if isinstance(source_rel_for_hold, str) else None)
            if (source_path_for_hold is None or not source_path_for_hold.is_file()
                    or source_path_for_hold.is_symlink()):
                raise PermissionError("source snapshot is unavailable for released-hold binding")
            prepared_event = dict(prepared)
            prepared_event.update({"source_preimages": preimages,
                "control_inventory_sha256": sha256_json(control_hash_inventory_v15(repo)),
                "runtime_inventory_sha256": sha256_json(runtime_hash_inventory(repo)),
                "frozen_artifacts": frozen,
                "pending_gate_sha256": sha256_bytes(pending_bytes),
                "release_sha256": sha256_bytes(release_bytes),
                "released_at_utc": prepared["prepared_at_utc"]})
            hold_projection = _released_hold_projection(
                _read_json(hold_path), prepared_event, _read_json(source_path_for_hold))
            prepared_event["released_hold_sha256"] = sha256_bytes(
                _canonical_indented_bytes(hold_projection))
            _append_jsonl_durable(ledger, prepared_event)
            existing = [prepared_event]
            if fault_after == "PREPARED":
                raise RuntimeError("fault injection after PREPARED append")
        prepared_event = existing[0]
        for key in ("nonce", "transaction_id", "approved_by", "approval_utc", "prepared_at_utc",
                    "source_snapshot_sha256", "prelaunch_audit_sha256", "targets", "authorization"):
            if prepared_event.get(key) != prepared.get(key):
                raise PermissionError("reentry differs from immutable PREPARED identity: " + key)
        if (prepared_event.get("pending_gate_sha256") != sha256_bytes(pending_bytes)
                or prepared_event.get("release_sha256") != sha256_bytes(release_bytes)):
            raise PermissionError("reentry pending gate/release artifact bytes differ")
        preimages = prepared_event.get("source_preimages")
        if not isinstance(preimages, Mapping) or preimages.get("hold") != INCIDENT_HOLD_SHA256:
            raise PermissionError("PREPARED source preimages are incomplete")
        for label, digest in (("gate", preimages.get("gate")), ("hold", preimages.get("hold")),
                              ("release_pointer", preimages.get("release_pointer"))):
            archive = control_dir / ("v15_preimage_" + label + "_" + str(digest) + ".bin")
            if not archive.is_file() or archive.is_symlink() or sha256_file(archive) != digest:
                raise PermissionError("PREPARED exact byte preimage archive changed")
        hold_now = _read_json(hold_path)
        released_state = hold_now.get("active") is False
        if released_state and hold_now.get("successor_release_id") != txid:
            raise PermissionError("an inactive hold belongs to a different release identity")
        frozen_now = _assert_frozen_artifacts(repo, allow_released_hold=released_state,
                                              allow_gate_replaced=True)
        if frozen_now != prepared_event.get("frozen_artifacts"):
            raise PermissionError("frozen historical incident graph changed during reentry")
        if (sha256_json(control_hash_inventory_v15(repo)) != prepared_event.get("control_inventory_sha256")
                or sha256_json(runtime_hash_inventory(repo)) != prepared_event.get("runtime_inventory_sha256")):
            raise PermissionError("v1.5 source/runtime inventory changed during transaction")
        runtime_now = runtime_hash_inventory(repo)
        control_now = control_hash_inventory_v15(repo)
        runtime_sha = sha256_json(runtime_now)
        control_sha = sha256_json(control_now)
        _validate_amendment(gate_pending.get("bootstrap_amendment"),
                            scientific_sha256=SCIENTIFIC_SHA256,
                            runtime_sha256=runtime_sha, control_sha256=control_sha)
        for path_field, sha_field, expected_path, expected_sha in (
                ("design_path", "design_sha256", DESIGN_PATH, DESIGN_SHA256),
                ("design_review_path", "design_review_sha256", DESIGN_REVIEW_PATH,
                 DESIGN_REVIEW_SHA256)):
            if gate_pending.get(path_field) != expected_path or gate_pending.get(sha_field) != expected_sha:
                raise PermissionError("v1.5 review/design identity changed during transaction")
        _validate_review_graph(gate_pending, root=repo,
                               runtime_inventory=runtime_now,
                               control_inventory=control_now)
        source_rel = gate_pending.get("source_snapshot_path")
        prelaunch_rel = gate_pending.get("prelaunch_audit_path")
        source_file = _repo_path(repo, source_rel) if isinstance(source_rel, str) else None
        prelaunch_file = _repo_path(repo, prelaunch_rel) if isinstance(prelaunch_rel, str) else None
        if (source_file is None or prelaunch_file is None or not source_file.is_file()
                or source_file.is_symlink() or not prelaunch_file.is_file() or prelaunch_file.is_symlink()
                or sha256_file(source_file) != gate_pending.get("source_snapshot_file_sha256")
                or sha256_file(prelaunch_file) != gate_pending.get("prelaunch_audit_sha256")):
            raise PermissionError("immutable source/prelaunch graph changed during transaction")
        allow_released = released_state
        allow_gate_replaced = os.path.lexists(ledger) and sha256_file(gate_path) != preimages["gate"]
        source_doc = _read_json(source_file)
        validate_source_snapshot(source_doc, root=repo, allow_released_hold=allow_released,
                                 allow_gate_replaced=allow_gate_replaced)
        prelaunch_doc = _read_json(prelaunch_file)
        validate_prelaunch_audit(prelaunch_doc, root=repo,
                                 allow_released_hold=allow_released,
                                 allow_gate_replaced=allow_gate_replaced,
                                 source_snapshot_sha256=prepared_event.get("source_snapshot_sha256"))
        if source_doc.get("snapshot_sha256") != prepared_event.get("source_snapshot_sha256"):
            raise PermissionError("source snapshot identity changed during transaction")
        def maybe_fault(name: str) -> None:
            if fault_after == name:
                raise RuntimeError("fault injection at " + name)

        def durable_immutable(path: Path, raw: bytes, label: str) -> None:
            if os.path.lexists(path):
                if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
                    raise PermissionError(label + " has conflicting durable bytes")
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                try: os.fsync(fd)
                finally: os.close(fd)
                _fsync_dir(path.parent)
            else:
                durable_bytes_create(path, raw)

        durable_immutable(release_path, release_bytes, "release artifact")
        maybe_fault("RELEASE_ARTIFACT_RENAME")
        rows = _transaction_rows(ledger, txid)
        rows = _append_once(ledger, rows, {"schema": LEDGER_SCHEMA,
            "event": "RELEASE_ARTIFACT_DURABLE", "transaction_id": txid,
            "release_sha256": sha256_bytes(release_bytes)})
        maybe_fault("RELEASE_ARTIFACT_DURABLE")

        durable_immutable(pending_path, pending_bytes, "pending gate archive")
        current_gate_state = _check_exact_or_source(gate_path, preimages.get("gate"), pending_bytes,
                                                    label="canonical gate")
        final_bytes = final_archive.read_bytes() if final_archive.is_file() and not final_archive.is_symlink() else None
        if current_gate_state == "source":
            _atomic_replace_bytes(gate_path, pending_bytes,
                                  after_replace=lambda: maybe_fault("PENDING_GATE_RENAME"))
        elif current_gate_state != "expected":
            if final_bytes is None or gate_path.read_bytes() != final_bytes:
                raise PermissionError("canonical gate is an unrecognized transaction state")
        if current_gate_state == "expected":
            _durability_fence_path(gate_path)
        maybe_fault("PENDING_GATE_RENAME_DURABLE")
        rows = _transaction_rows(ledger, txid)
        rows = _append_once(ledger, rows, {"schema": LEDGER_SCHEMA,
            "event": "PENDING_GATE_DURABLE", "transaction_id": txid,
            "pending_gate_sha256": sha256_bytes(pending_bytes)})
        maybe_fault("PENDING_GATE_DURABLE")

        pointer_state = _check_exact_or_source(release_pointer, preimages.get("release_pointer"),
                                                release_bytes, label="release pointer")
        if pointer_state != "expected":
            _atomic_replace_bytes(release_pointer, release_bytes,
                                  after_replace=lambda: maybe_fault("RELEASE_POINTER_RENAME"))
        else:
            _durability_fence_path(release_pointer)
        rows = _transaction_rows(ledger, txid)
        rows = _append_once(ledger, rows, {"schema": LEDGER_SCHEMA,
            "event": "RELEASE_POINTER_DURABLE", "transaction_id": txid,
            "release_sha256": sha256_bytes(release_bytes)})
        maybe_fault("RELEASE_POINTER_DURABLE")

        hold_source = control_dir / ("v15_preimage_hold_" + INCIDENT_HOLD_SHA256 + ".bin")
        source_hold = hold_source.read_bytes()
        hold = json.loads(source_hold)
        released_hold = _released_hold_projection(hold, prepared_event, _read_json(source_file))
        released_bytes = _canonical_indented_bytes(released_hold)
        if sha256_bytes(released_bytes) != prepared_event.get("released_hold_sha256"):
            raise PermissionError("released hold projection differs from immutable PREPARED identity")
        _assert_current_release_resources(runs, repo)
        immediate_observation = fresh_decision("PRE_HOLD_RELEASE_DECISION")
        _assert_current_release_resources(runs, repo)
        hold_state = _check_exact_or_source(hold_path, INCIDENT_HOLD_SHA256,
                                            released_bytes, label="active hold")
        if hold_state == "source":
            _atomic_replace_bytes(hold_path, released_bytes,
                                  after_replace=lambda: maybe_fault("HOLD_RENAME"))
        else:
            _durability_fence_path(hold_path)
        maybe_fault("HOLD_RENAMED_DURABLE")
        rows = _transaction_rows(ledger, txid)
        rows = _append_once(ledger, rows, {"schema": LEDGER_SCHEMA,
            "event": "HOLD_RELEASED", "transaction_id": txid,
            "released_hold_sha256": sha256_bytes(released_bytes)})
        maybe_fault("HOLD_RELEASED")

        pending_sha = sha256_bytes(pending_bytes)
        release_sha = sha256_bytes(release_bytes)
        hold_sha = sha256_bytes(released_bytes)
        rows = _transaction_rows(ledger, txid)
        committed_event = {"schema": LEDGER_SCHEMA, "event": "COMMITTED",
            "transaction_id": txid, "release_sha256": release_sha,
            "released_hold_sha256": hold_sha, "pending_gate_sha256": pending_sha}
        rows = _append_once(ledger, rows, committed_event)
        ledger_sha = sha256_file(ledger)
        maybe_fault("COMMITTED")

        final_gate = dict(gate_pending)
        final_gate.update({"launch_authorized": True, "transaction_state": "COMMITTED",
            "release_artifact_path": str(release_path.relative_to(repo)),
            "release_sha256": release_sha, "released_hold_sha256": hold_sha,
            "committed_ledger_sha256": ledger_sha})
        final_bytes_expected = _canonical_indented_bytes(final_gate)
        durable_immutable(final_archive, final_bytes_expected, "final authorized gate archive")
        maybe_fault("FINAL_GATE_ARCHIVE")
        current = _check_exact_or_source(gate_path, pending_sha, final_bytes_expected,
                                         label="canonical gate before final publication")
        if current != "expected":
            _atomic_replace_bytes(gate_path, final_bytes_expected,
                after_replace=lambda: maybe_fault("FINAL_GATE_RENAME_BEFORE_DIR_FSYNC"))
        else:
            # Visible-after-rename recovery performs the same terminal fence.
            _durability_fence(gate_path, lock_already_held=True)
        maybe_fault("FINAL_GATE_DURABLE")
        terminal = validate_terminal_graph(final_gate, root=repo, lock_already_held=True)
        return {"status": "COMMITTED", "gate_sha256": sha256_file(gate_path),
                "ledger_sha256": terminal["ledger_sha256"]}


def assert_no_global_hold(output_root: os.PathLike | str, *,
                          lock_already_held: bool = False) -> None:
    root = Path(output_root).resolve()
    hold_path = root / "execution_control" / "global_hold.json"
    gate_path = root.parent / "execution_isolation_gate.json"
    if root != RUNS.resolve():
        # Isolated fixture roots still require exact v1.5 released proof.
        if not hold_path.is_file() or not gate_path.is_file():
            raise PermissionError("v1.5 hold/gate missing")
        hold = _read_json(hold_path)
        gate = _read_json(gate_path)
        if hold.get("active") is True:
            raise PermissionError("v1.5 global hold is active")
        validate_current_gate(gate, root=root.parents[2], require_release=True,
                              lock_already_held=lock_already_held)
        return
    if not gate_path.is_file() or gate_path.is_symlink():
        raise PermissionError("canonical execution gate is missing or non-regular")
    gate = _read_json(gate_path)
    if gate.get("schema") != SCHEMA:
        raise PermissionError("v1.5 dispatch refuses non-v1.5 or unknown production gate")
    hold = _read_json(hold_path)
    if hold.get("active") is True:
        raise PermissionError("v1.5 global hold is active")
    validate_current_gate(gate, root=ROOT, require_release=True,
                          lock_already_held=lock_already_held)


__all__ = ["SCHEMA", "SOURCE_SCHEMA", "PRELAUNCH_SCHEMA", "RELEASE_SCHEMA", "LEDGER_SCHEMA",
           "DEFER_SCHEMA", "SCOPE", "ROOT", "RUNS", "V15_CONTROL_FILES",
           "canonical_json_bytes", "sha256_bytes", "sha256_file", "sha256_json",
           "control_hash_inventory_v15", "runtime_hash_inventory", "runtime_hash",
           "validate_current_gate", "validate_terminal_graph", "assert_no_global_hold",
           "build_source_snapshot", "validate_source_snapshot", "build_prelaunch_audit",
           "validate_prelaunch_audit", "fresh_gpu_eligible", "validate_target_rows",
           "observation_is_fresh", "key_is_untouched", "fresh_member_paths",
           "create_deferred_receipt", "verify_deferred_receipt", "new_deferral_secret",
           "inspect_claim_outcome", "record_runtime_observation", "validate_observation_ledger",
           "prepare_release_identity", "commit_principal_release", "execution_state_lock",
           "durable_json_create"]
