"""Read-only, fail-closed revalidation of the archived A1 wave-2 target pair."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

import stage_S3_orchestrator as orchestrator
from stage_S1_orchestrator import validate_raw_prediction_metrics
from stage_S3_protocol import (
    ACCESS, LEGACY_QUEUE_SHA256, MODEL_SOURCE_SHA256, S3_SEEDS, STAGE, build_fresh_matrix,
    canonical_dataset, fresh_formal_key,
)


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(path: str) -> str:
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise FileNotFoundError("archive tree missing: {}".format(root))
    files = []
    for current, directories, names in os.walk(root):
        directories.sort()
        for name in sorted(names):
            path = os.path.join(current, name)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            files.append((relative, _file_hash(path)))
    return orchestrator._digest(files)


def _receipt_payload(receipt: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in receipt.items() if key != "receipt_digest"}


def validate_receipt(receipt: Mapping[str, object]) -> None:
    if receipt.get("schema") != "stage_S3_A1_wave2_revalidation_v1":
        raise ValueError("wave-2 receipt schema mismatch")
    if receipt.get("reviewed_model_sha256") != MODEL_SOURCE_SHA256:
        raise ValueError("wave-2 receipt forward source hash mismatch")
    if receipt.get("sample_full_pairing") != "PENDING_SAMPLE_FULL_REPLACEMENT":
        raise ValueError("wave-2 Sample target must remain pending until FULL replacement")
    if receipt.get("live_promotion_performed") is not False:
        raise ValueError("read-only archive validation must not promote artifacts")
    expected = orchestrator._digest(_receipt_payload(receipt))
    if receipt.get("receipt_digest") != expected:
        raise ValueError("wave-2 receipt digest mismatch")


def _queue_task(queue: Mapping[str, object], key: str) -> Mapping[str, object]:
    tasks = queue.get("tasks", [])
    matches = [task for task in tasks if isinstance(task, Mapping)
               and str(task.get("formal_run_key", "")) == key]
    if len(matches) != 1:
        raise ValueError("historical queue does not identify exactly one archived task: " + key)
    return matches[0]


def _member_run_dir(archive_root: str, objective: str) -> str:
    member = os.path.join(archive_root, objective)
    # The A1 wave-2 archive is frozen with the S3 attempt wrapper structure.
    run = os.path.join(member, "run")
    if not os.path.isdir(run):
        raise FileNotFoundError("archived {} run directory missing".format(objective))
    return run


def _validate_member(run_dir: str, task: Mapping[str, object],
                     current_source: Mapping[str, str], *, allow_legacy_empty_object: bool,
                     require_formal: bool) -> Dict[str, object]:
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest = orchestrator._read_json(manifest_path)
    reasons = []
    if manifest.get("stage") != STAGE or manifest.get("formal_run_key") != task.get("formal_run_key"):
        reasons.append("queue_task_identity_mismatch")
    try:
        manifest_dataset = canonical_dataset(str(manifest.get("dataset", "")))
        task_dataset = canonical_dataset(str(task.get("dataset", "")))
    except Exception:
        manifest_dataset, task_dataset = "", "identity-invalid"
    if manifest_dataset != task_dataset \
            or manifest.get("setting") != task.get("setting") \
            or int(manifest.get("seed", -1)) != int(task.get("seed", -2)):
        reasons.append("run_identity_mismatch")
    if manifest.get("shared_access") != ACCESS or manifest.get("objective_weighting") != task.get(
            "objective_weighting"):
        reasons.append("objective_or_access_mismatch")
    code_hashes = manifest.get("code_hashes", {})
    model_hash = code_hashes.get(
        "m3oe/MDMTRec/models/multi_domain/model.py", "") if isinstance(code_hashes, Mapping) else ""
    recorded_forward = manifest.get("forward_model_source_sha256", "")
    if model_hash != MODEL_SOURCE_SHA256 \
            or recorded_forward not in ("", MODEL_SOURCE_SHA256):
        reasons.append("model_source_hash_mismatch")
    # These children predate the A1 coder recovery. Their frozen source proof is
    # the reviewed forward-model hash; a new report/orchestrator byte hash must
    # not be retroactively imputed as their training source.
    reasons.extend(orchestrator._artifact_hash_reasons(run_dir, manifest))
    nonfinite = os.path.join(run_dir, "nonfinite_inventory.json")
    legacy_empty_object = False
    if not os.path.isfile(nonfinite):
        reasons.append("nonfinite_inventory_missing")
    else:
        with open(nonfinite, "r", encoding="utf-8") as stream:
            legacy_empty_object = json.load(stream) == {}
        failure = orchestrator._nonfinite_inventory_failure(
            nonfinite, allow_empty_object=allow_legacy_empty_object)
        if failure:
            reasons.append(failure)
    reasons.extend(orchestrator.validate_batch_trace(task, run_dir, manifest))
    try:
        import csv
        _fields, test_rows = orchestrator._read_csv(os.path.join(run_dir, "metrics_test.csv"))
        _fields, summary_rows = orchestrator._read_csv(os.path.join(run_dir, "metrics_summary.csv"))
        with np.load(os.path.join(run_dir, "predictions_test.npz"), allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
        reasons.extend(validate_raw_prediction_metrics(arrays, manifest, task,
                                                       test_rows, summary_rows))
    except Exception as exc:
        reasons.append("prediction_metric_validation_failed:{}".format(type(exc).__name__))
    if reasons:
        raise ValueError("archived run {} failed: {}".format(
            task.get("formal_run_key"), sorted(set(reasons))))
    return {
        "formal_run_key": str(task["formal_run_key"]),
        "manifest_sha256": _file_hash(manifest_path),
        "artifact_hashes": dict(manifest.get("artifact_hashes", {})),
        "forward_model_source_sha256": str(model_hash),
        "empty_object_inventory_interpreted_as_empty": bool(legacy_empty_object),
        "status": "VALID_RETAINED",
    }


def revalidate_wave2_archive(*, queue_path: str, raw_root: str,
                             tasks: Sequence[Mapping[str, object]],
                             current_source_root: str, process_inventory=False,
                             require_formal: bool = True) -> Dict[str, object]:
    """Revalidate both immutable wave-2 children and return an in-memory receipt.

    Queue bytes and archive bytes are fingerprinted before and after. No
    canonical path, queue, lock, archive, or receipt file is written.
    """
    if not os.path.isfile(queue_path):
        raise FileNotFoundError(
            "historical Stage S3 queue is required to prove archive task identity")
    queue_hash_before = _file_hash(queue_path)
    if queue_hash_before != LEGACY_QUEUE_SHA256:
        raise ValueError("historical Stage S3 queue SHA-256 does not match the frozen queue")
    queue = orchestrator._read_json(queue_path)
    if queue.get("stage") != STAGE:
        raise ValueError("historical queue stage identity mismatch")
    key_equal = fresh_formal_key("equal", "kuairand", "natural", 2022)
    key_sample = fresh_formal_key("sample", "kuairand", "natural", 2022)
    equal_task, sample_task = _queue_task(queue, key_equal), _queue_task(queue, key_sample)
    supplied = {str(task.get("formal_run_key")): task for task in tasks}
    for task in (equal_task, sample_task):
        candidate = supplied.get(str(task.get("formal_run_key")))
        if candidate is None or str(candidate.get("dataset")) != str(task.get("dataset")) \
                or str(candidate.get("seed")) != str(task.get("seed")):
            raise ValueError("frozen task matrix does not match the historical queue")
    archive_root = os.path.join(os.path.abspath(raw_root), ".stage_S3_archives",
                                "s3_kuairand_natural_seed2022", "wave_2")
    archive_digest_before = _tree_digest(archive_root)
    if process_inventory is True:
        raise RuntimeError("process inventory callback required for active-process audit")
    if callable(process_inventory):
        active = process_inventory()
        if active:
            raise RuntimeError("active Stage S3 child prevents archive revalidation")
    current = orchestrator.source_inventory(current_source_root)
    members = {
        "equal": _validate_member(_member_run_dir(archive_root, "equal"), equal_task, current,
                                  allow_legacy_empty_object=True, require_formal=require_formal),
        "sample": _validate_member(_member_run_dir(archive_root, "sample"), sample_task, current,
                                   allow_legacy_empty_object=True, require_formal=require_formal),
    }
    queue_hash_after = _file_hash(queue_path)
    archive_digest_after = _tree_digest(archive_root)
    if queue_hash_before != queue_hash_after or archive_digest_before != archive_digest_after:
        raise RuntimeError("archive or historical queue changed during read-only revalidation")
    receipt = {
        "schema": "stage_S3_A1_wave2_revalidation_v1",
        "historical_queue_path": os.path.abspath(queue_path),
        "historical_queue_sha256": queue_hash_before,
        "launch_group_id": "s3_kuairand_natural_seed2022",
        "wave": 2, "archive_path": archive_root,
        "archive_tree_digest": archive_digest_before,
        "member_checks": members,
        "reviewed_model_sha256": MODEL_SOURCE_SHA256,
        "sample_full_pairing": "PENDING_SAMPLE_FULL_REPLACEMENT",
        "live_promotion_performed": False,
        "source_inventory_digest": orchestrator.source_inventory_digest(current),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = orchestrator._digest(_receipt_payload(receipt))
    validate_receipt(receipt)
    return receipt


def promote_wave2_archive(*, receipt: Mapping[str, object],
                          tasks: Sequence[Mapping[str, object]],
                          raw_root: str) -> Dict[str, object]:
    """Copy the independently revalidated pair into canonical dirs as one transaction."""
    validate_receipt(receipt)
    archive_root = os.path.abspath(str(receipt["archive_path"]))
    if _tree_digest(archive_root) != receipt.get("archive_tree_digest"):
        raise RuntimeError("wave-2 archive changed after receipt creation")
    by_key = {str(task.get("formal_run_key")): task for task in tasks}
    keys = (
        fresh_formal_key("equal", "kuairand", "natural", 2022),
        fresh_formal_key("sample", "kuairand", "natural", 2022),
    )
    selected = [by_key.get(key) for key in keys]
    if any(task is None for task in selected):
        raise ValueError("frozen fresh matrix is missing an archived target task")
    canonical = [os.path.abspath(str(task["run_dir"])) for task in selected]
    if any(os.path.lexists(path) for path in canonical):
        raise RuntimeError("canonical target path already exists; refusing archive promotion")
    stage_parent = tempfile.mkdtemp(prefix=".s3-wave2-promotion-",
                                    dir=os.path.abspath(raw_root))
    staged = []
    promoted = False
    try:
        for objective, task in zip(("equal", "sample"), selected):
            source = _member_run_dir(archive_root, objective)
            destination = os.path.join(stage_parent, str(task["worker_id"]), "run")
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copytree(source, destination)
            staged.append(destination)
        if _tree_digest(archive_root) != receipt.get("archive_tree_digest"):
            raise RuntimeError("wave-2 archive changed while staging promotion")
        import stage_S3_orchestrator as orchestrator
        orchestrator._atomic_pair_promote(staged, canonical)
        promoted = True
        trees = [_tree_digest(path) for path in canonical]
        promotion = {
            "schema": "stage_S3_A1_wave2_promotion_v1",
            "revalidation_receipt_digest": receipt["receipt_digest"],
            "archive_tree_digest": receipt["archive_tree_digest"],
            "canonical_run_keys": list(keys),
            "canonical_paths": canonical,
            "canonical_tree_digests": trees,
            "sample_full_pairing": "PENDING_SAMPLE_FULL_REPLACEMENT",
            "live_promotion_performed": True,
        }
        promotion["promotion_digest"] = orchestrator._digest(promotion)
        return promotion
    except Exception:
        # Preserve staged bytes for forensic recovery if a pair move was interrupted.
        raise
    finally:
        if promoted and os.path.isdir(stage_parent):
            shutil.rmtree(stage_parent)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Read-only Stage S3 A1 archive revalidation")
    parser.add_argument("--queue", default=os.path.join(
        os.path.dirname(__file__), "stage_S3_queue_manifest.json"))
    parser.add_argument("--raw-root", default=os.path.join(
        os.path.dirname(__file__), "stage_S3_raw_results"))
    args = parser.parse_args()
    tasks = build_fresh_matrix(args.raw_root)
    receipt = revalidate_wave2_archive(
        queue_path=args.queue, raw_root=args.raw_root, tasks=tasks,
        current_source_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        process_inventory=False)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
