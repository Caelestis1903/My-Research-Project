"""Post-run Stage S2 collection compatibility adapter.

The formal S2 manifests are already hash-bound and must remain immutable.  A
fresh S2 manifest uses the human-readable value (for example, ``MovieLens-1M``)
at its top-level ``dataset`` field while the reused S1 manifest uses the
canonical key (``movielens``).  This module fixes that reporting-only mismatch
in copied in-memory dictionaries and then delegates every other comparison and
every artifact check to the existing Stage S2 collector.

It deliberately does not import or launch the runner.  The only files it
writes are derived structural tables and the explicit normalization audit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import stage_S2_orchestrator as orchestrator
import stage_S2_report as report
from stage_S2_protocol import BACKBONE, canonical_dataset


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_S2_RAW_ROOT = os.path.join(HERE, "stage_S2_raw_results")
DEFAULT_S1_RAW_ROOT = os.path.join(HERE, "stage_S1_raw_results")
NORMALIZATION_AUDIT_JSON = "stage_S2_normalization_audit.json"
NORMALIZATION_AUDIT_CSV = "stage_S2_normalization_audit.csv"


class DatasetNormalizationError(ValueError):
    """Raised when the approved dataset-only normalization is not legal."""

    def __init__(self, message: str, audit: Optional[Mapping[str, object]] = None):
        super().__init__(message)
        self.audit = dict(audit or {})


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str).encode("utf-8")).hexdigest()


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON object expected: {}".format(path))
    return value


def _write_json(path: str, value: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _manifest_identity(manifest: Mapping[str, object], side: str) -> Dict[str, str]:
    """Validate one manifest's dataset identity without changing it."""

    raw_dataset = manifest.get("dataset")
    if not isinstance(raw_dataset, str) or not raw_dataset.strip():
        raise ValueError("{}_dataset_missing".format(side))
    try:
        canonical = canonical_dataset(raw_dataset)
    except (TypeError, ValueError) as exc:
        raise ValueError("{}_dataset_unknown:{}".format(side, raw_dataset)) from exc

    dataset_key = manifest.get("dataset_key")
    if not isinstance(dataset_key, str) or dataset_key != canonical:
        raise ValueError("{}_dataset_key_not_canonical".format(side))
    try:
        if canonical_dataset(dataset_key) != canonical:
            raise ValueError("{}_dataset_key_cross_dataset".format(side))
    except (TypeError, ValueError) as exc:
        if str(exc).startswith(side + "_"):
            raise
        raise ValueError("{}_dataset_key_unknown".format(side)) from exc

    dataset_label = manifest.get("dataset_label")
    if not isinstance(dataset_label, str) or not dataset_label.strip():
        raise ValueError("{}_dataset_label_missing".format(side))
    try:
        label_canonical = canonical_dataset(dataset_label)
    except (TypeError, ValueError) as exc:
        raise ValueError("{}_dataset_label_unknown:{}".format(side, dataset_label)) from exc
    if label_canonical != canonical:
        raise ValueError("{}_dataset_label_cross_dataset".format(side))
    return {
        "raw_dataset": raw_dataset,
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "canonical_dataset": canonical,
    }


def normalize_dataset_pair(
    equal_manifest: Mapping[str, object],
    sample_manifest: Mapping[str, object],
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    """Return normalized copies and an auditable representation record.

    Only the copied top-level ``dataset`` value is changed.  In particular,
    ``dataset_key`` and ``dataset_label`` are checked but never rewritten.
    """

    equal_copy = copy.deepcopy(dict(equal_manifest))
    sample_copy = copy.deepcopy(dict(sample_manifest))
    audit: Dict[str, object] = {
        "normalization_field": "dataset",
        "equal_raw_dataset": equal_manifest.get("dataset", ""),
        "sample_raw_dataset": sample_manifest.get("dataset", ""),
        "equal_manifest_pre_hash": _json_hash(equal_manifest),
        "sample_manifest_pre_hash": _json_hash(sample_manifest),
        "normalization_status": "FAIL",
        "strict_comparison_reasons": [],
        "non_dataset_mismatch_reasons": [],
        "non_dataset_bypass_count": 0,
    }
    try:
        equal_identity = _manifest_identity(equal_manifest, "equal")
        sample_identity = _manifest_identity(sample_manifest, "sample")
        if equal_identity["dataset_label"] != sample_identity["dataset_label"]:
            raise ValueError("dataset_label_not_equal")
        if equal_identity["canonical_dataset"] != sample_identity["canonical_dataset"]:
            raise ValueError("dataset_canonical_value_mismatch")
        canonical = equal_identity["canonical_dataset"]
    except ValueError as exc:
        audit["failure"] = str(exc)
        raise DatasetNormalizationError(str(exc), audit) from exc

    equal_copy["dataset"] = canonical
    sample_copy["dataset"] = canonical
    audit.update({
        "equal_dataset_key": equal_identity["dataset_key"],
        "sample_dataset_key": sample_identity["dataset_key"],
        "equal_dataset_label": equal_identity["dataset_label"],
        "sample_dataset_label": sample_identity["dataset_label"],
        "canonical_dataset": canonical,
        "equal_manifest_post_hash": _json_hash(equal_copy),
        "sample_manifest_post_hash": _json_hash(sample_copy),
        "normalization_status": "OK",
    })
    return equal_copy, sample_copy, audit


# Capture the unchanged comparator once.  The temporary runtime wrapper below
# calls this exact implementation after normalizing only copied dictionaries.
STRICT_MANIFEST_COMPARATOR = orchestrator.compare_equal_sample_manifests


def compare_normalized_manifests(
    equal_manifest: Mapping[str, object],
    sample_manifest: Mapping[str, object],
    *,
    source_artifact_hashes: Optional[Mapping[str, object]] = None,
) -> Tuple[List[str], Dict[str, object]]:
    """Apply the approved normalization and then the frozen strict comparator."""

    try:
        equal_copy, sample_copy, audit = normalize_dataset_pair(equal_manifest, sample_manifest)
        reasons = list(STRICT_MANIFEST_COMPARATOR(equal_copy, sample_copy))
    except DatasetNormalizationError as exc:
        reasons = ["dataset_normalization_failed:" + str(exc)]
        audit = dict(exc.audit)
    non_dataset_reasons = [reason for reason in reasons
                           if reason != "pairing_field_mismatch:dataset"]
    audit["strict_comparison_reasons"] = sorted(set(reasons))
    audit["non_dataset_mismatch_reasons"] = sorted(set(non_dataset_reasons))
    # There is intentionally no bypass list.  Every reason returned by the
    # frozen comparator remains a hard failure; this field makes that property
    # explicit in the audit for independent review.
    audit["non_dataset_bypass_count"] = 0
    audit["source_artifact_hashes"] = copy.deepcopy(dict(source_artifact_hashes or {}))
    return reasons, audit


def _task_key(task: Mapping[str, object]) -> str:
    return str(task.get("formal_run_key") or task.get("source_formal_run_key") or "")


def _task_run_dir(task: Mapping[str, object]) -> str:
    return os.path.abspath(str(task.get("run_dir") or task.get("source_run_dir") or ""))


def snapshot_raw_artifacts(tasks: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    """Hash every file below each of the 80 immutable run directories."""

    snapshot: Dict[str, Dict[str, object]] = {}
    for task in tasks:
        key = _task_key(task)
        root = _task_run_dir(task)
        if not key or not os.path.isdir(root):
            raise RuntimeError("raw artifact run directory missing: {}".format(root))
        if key in snapshot:
            raise RuntimeError("duplicate raw artifact key: {}".format(key))
        file_hashes: Dict[str, str] = {}
        for current, directories, filenames in os.walk(root):
            directories.sort()
            filenames.sort()
            for filename in filenames:
                path = os.path.join(current, filename)
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                file_hashes[relative] = _file_hash(path)
        snapshot[key] = {
            "formal_run_key": key,
            "provenance_type": str(task.get("provenance_type", "")),
            "dataset": str(task.get("dataset", "")),
            "setting": str(task.get("setting", "")),
            "seed": int(task.get("seed", 0)),
            "run_type": str(task.get("run_type", "")),
            "artifact_count": len(file_hashes),
            "file_hashes": file_hashes,
        }
    return {key: snapshot[key] for key in sorted(snapshot)}


def _block_source_hashes(
    equal_tasks: Sequence[Mapping[str, object]],
    reuse_tasks: Sequence[Mapping[str, object]],
    snapshot: Mapping[str, Mapping[str, object]],
) -> Dict[Tuple[str, str, int], Dict[str, object]]:
    sample_by_block = {
        (str(task["dataset"]), str(task["setting"]), int(task["seed"])): task
        for task in reuse_tasks if task.get("run_type") == "joint"
    }
    result: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for task in equal_tasks:
        block = (str(task["dataset"]), str(task["setting"]), int(task["seed"]))
        sample_task = sample_by_block.get(block)
        if sample_task is None:
            continue
        equal_key = _task_key(task)
        sample_key = _task_key(sample_task)
        result[block] = {
            "equal": copy.deepcopy(snapshot.get(equal_key, {})),
            "sample": copy.deepcopy(snapshot.get(sample_key, {})),
        }
    return result


def _block_from_manifest(manifest: Mapping[str, object]) -> Tuple[str, str, int]:
    raw_key = str(manifest.get("dataset_key", manifest.get("dataset", "")))
    try:
        dataset = canonical_dataset(raw_key)
    except (TypeError, ValueError):
        dataset = raw_key
    return dataset, str(manifest.get("setting", "")), int(manifest.get("seed", 0))


@contextmanager
def _normalized_comparator_patch(
    records: List[Dict[str, object]],
    source_by_block: Mapping[Tuple[str, str, int], Mapping[str, object]],
) -> Iterator[None]:
    """Temporarily adapt the globals used by the existing strict validator."""

    original = orchestrator.compare_equal_sample_manifests

    def wrapped(equal_manifest: Mapping[str, object], sample_manifest: Mapping[str, object]) -> List[str]:
        block = _block_from_manifest(equal_manifest)
        reasons, audit = compare_normalized_manifests(
            equal_manifest, sample_manifest,
            source_artifact_hashes=source_by_block.get(block, {}))
        audit["block"] = "{}|{}|{}".format(block[0], block[1], block[2])
        audit["equal_formal_run_key"] = equal_manifest.get("formal_run_key", "")
        audit["sample_formal_run_key"] = sample_manifest.get("formal_run_key", "")
        records.append(audit)
        return reasons

    orchestrator.compare_equal_sample_manifests = wrapped  # type: ignore[assignment]
    try:
        yield
    finally:
        orchestrator.compare_equal_sample_manifests = original  # type: ignore[assignment]


def _validate_normalization_records(
    records: Sequence[Mapping[str, object]],
    expected_blocks: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    expected_keys = {_task_key(task) for task in expected_blocks}
    if len(records) != len(expected_keys):
        raise RuntimeError("normalization audit block count {} != {}".format(
            len(records), len(expected_keys)))
    formal_keys = {str(record.get("equal_formal_run_key", "")) for record in records}
    if formal_keys != expected_keys:
        raise RuntimeError("normalization audit formal-key support mismatch")
    normalized: List[Dict[str, object]] = []
    for record in records:
        if record.get("normalization_status") != "OK":
            raise RuntimeError("dataset normalization failed: {}".format(record))
        reasons = list(record.get("strict_comparison_reasons", []))
        if reasons:
            raise RuntimeError("strict Equal/Sample comparison failed: {}".format(reasons))
        if list(record.get("non_dataset_mismatch_reasons", [])):
            raise RuntimeError("non-dataset mismatch was not accepted: {}".format(record))
        if int(record.get("non_dataset_bypass_count", 0)) != 0:
            raise RuntimeError("non-dataset comparison bypass detected")
        normalized.append(dict(record))
    normalized.sort(key=lambda item: str(item.get("equal_formal_run_key", "")))
    return normalized


def inspect_formal_normalizations(
    *,
    s2_raw_root: str = DEFAULT_S2_RAW_ROOT,
    s1_raw_root: str = DEFAULT_S1_RAW_ROOT,
) -> List[Dict[str, object]]:
    """Inspect the 16 Equal/Sample mappings without writing any output."""

    equal_tasks = orchestrator.build_tasks(s2_raw_root)
    reuse_tasks = orchestrator.build_reuse_matrix(s1_raw_root)
    sample_by_block = {
        (str(task["dataset"]), str(task["setting"]), int(task["seed"])): task
        for task in reuse_tasks if task.get("run_type") == "joint"
    }
    records: List[Dict[str, object]] = []
    for equal_task in equal_tasks:
        block = (str(equal_task["dataset"]), str(equal_task["setting"]), int(equal_task["seed"]))
        sample_task = sample_by_block[block]
        equal_manifest = _read_json(os.path.join(_task_run_dir(equal_task), "run_manifest.json"))
        sample_manifest = _read_json(os.path.join(_task_run_dir(sample_task), "run_manifest.json"))
        reasons, audit = compare_normalized_manifests(equal_manifest, sample_manifest)
        audit["block"] = "{}|{}|{}".format(*block)
        audit["equal_formal_run_key"] = equal_manifest.get("formal_run_key", equal_task["formal_run_key"])
        audit["sample_formal_run_key"] = sample_manifest.get(
            "formal_run_key", sample_task["source_formal_run_key"])
        audit["strict_comparison_reasons"] = sorted(set(reasons))
        records.append(audit)
    return _validate_normalization_records(records, equal_tasks)


def _unique_row_count(rows: Sequence[Mapping[str, object]], fields: Sequence[str], label: str) -> None:
    keys = [tuple(row.get(field, "") for field in fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate {} row key".format(label))


def _assert_collection_contract(result: Mapping[str, object]) -> None:
    raw_rows = list(result.get("raw_rows", []))
    paired_rows = list(result.get("paired_rows", []))
    inventory = list(result.get("inventory", []))
    if len(raw_rows) != 2016:
        raise RuntimeError("post-run raw cardinality {} != 2016".format(len(raw_rows)))
    if len(paired_rows) != 2016:
        raise RuntimeError("post-run paired cardinality {} != 2016".format(len(paired_rows)))
    if len(inventory) != 80:
        raise RuntimeError("post-run run inventory {} != 80".format(len(inventory)))
    _unique_row_count(raw_rows,
                      ("dataset", "setting", "seed", "physical_domain", "task", "checkpoint", "method"),
                      "raw")
    _unique_row_count(paired_rows,
                      ("comparison", "dataset", "setting", "seed", "physical_domain", "task", "checkpoint"),
                      "paired")
    invalid = [row for row in paired_rows if int(row.get("pairing_valid", 0)) != 1]
    if invalid:
        raise RuntimeError("post-run invalid pair rows: {}".format(invalid[:2]))
    fresh = [row for row in inventory if row.get("provenance_type") == "fresh"]
    reused = [row for row in inventory if row.get("provenance_type") == "reused"]
    if len(fresh) != 16 or any(row.get("status") != "OK" for row in fresh):
        raise RuntimeError("post-run fresh inventory contract failed")
    if len(reused) != 64 or any(row.get("status") != "OK" for row in reused):
        raise RuntimeError("post-run reused inventory contract failed")
    audit = result.get("audit", {})
    if not isinstance(audit, Mapping) or audit.get("expected_count") != 64 \
            or audit.get("valid_count") != 64 or audit.get("rejected_count") != 0 \
            or not bool(audit.get("all_reuse_ok")):
        raise RuntimeError("post-run 64-key reuse audit contract failed")
    pairing_audit = result.get("pairing_audit", {})
    if not isinstance(pairing_audit, Mapping) or not bool(pairing_audit.get("valid")):
        raise RuntimeError("post-run pairing audit contract failed")
    manifest = result.get("manifest", {})
    if isinstance(manifest, Mapping):
        expected_values = {
            "raw_row_count": 2016,
            "expected_raw_row_count": 2016,
            "paired_row_count": 2016,
            "expected_paired_row_count": 2016,
            "invalid_paired_row_count": 0,
        }
        for field, expected in expected_values.items():
            if manifest.get(field) != expected:
                raise RuntimeError("post-run summary cardinality mismatch: {}".format(field))


def _write_normalization_audit(output_root: str, payload: Mapping[str, object]) -> None:
    json_path = os.path.join(output_root, NORMALIZATION_AUDIT_JSON)
    _write_json(json_path, payload)
    csv_path = os.path.join(output_root, NORMALIZATION_AUDIT_CSV)
    fields = (
        "block", "equal_formal_run_key", "sample_formal_run_key",
        "equal_raw_dataset", "sample_raw_dataset", "canonical_dataset",
        "equal_dataset_key", "sample_dataset_key", "equal_dataset_label",
        "sample_dataset_label", "equal_manifest_pre_hash", "equal_manifest_post_hash",
        "sample_manifest_pre_hash", "sample_manifest_post_hash",
        "normalization_status", "strict_comparison_reasons",
        "non_dataset_mismatch_reasons", "non_dataset_bypass_count",
        "source_artifact_hashes",
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in payload.get("blocks", []):
            writer.writerow({
                field: json.dumps(record.get(field), sort_keys=True)
                if isinstance(record.get(field), (dict, list)) else record.get(field, "")
                for field in fields
            })


def collect_postrun(
    *,
    s2_raw_root: str = DEFAULT_S2_RAW_ROOT,
    s1_raw_root: str = DEFAULT_S1_RAW_ROOT,
    output_root: str = HERE,
    require_complete: bool = True,
    write_audit: bool = True,
) -> Dict[str, object]:
    """Run the existing collector with only the approved in-memory adapter."""

    equal_tasks = orchestrator.build_tasks(s2_raw_root)
    reuse_tasks = orchestrator.build_reuse_matrix(s1_raw_root)
    all_tasks = list(equal_tasks) + list(reuse_tasks)
    if len(all_tasks) != 80:
        raise RuntimeError("post-run source inventory requires 80 runs")
    before = snapshot_raw_artifacts(all_tasks)
    source_by_block = _block_source_hashes(equal_tasks, reuse_tasks, before)
    normalization_records: List[Dict[str, object]] = []

    with _normalized_comparator_patch(normalization_records, source_by_block):
        result = report.collect(
            s2_raw_root=s2_raw_root,
            s1_raw_root=s1_raw_root,
            output_root=output_root,
            require_complete=require_complete,
            write_audit=write_audit,
        )

    after = snapshot_raw_artifacts(all_tasks)
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("raw artifact hashes changed during post-run collection")

    blocks = _validate_normalization_records(normalization_records, equal_tasks)
    if require_complete:
        _assert_collection_contract(result)
    mapping_counts = Counter(str(record.get("canonical_dataset", "")) for record in blocks)
    audit_payload: Dict[str, object] = {
        "schema": "stage_S2_postrun_normalization_audit_v1",
        "adapter_source": os.path.abspath(__file__),
        "adapter_source_hash": _file_hash(__file__),
        "normalization_field": "dataset",
        "allowed_normalization_count": sum(
            str(record.get("equal_raw_dataset")) != str(record.get("sample_raw_dataset"))
            for record in blocks),
        "expected_block_count": 16,
        "block_count": len(blocks),
        "mapping_counts": dict(sorted(mapping_counts.items())),
        "non_dataset_bypass_count": sum(
            int(record.get("non_dataset_bypass_count", 0)) for record in blocks),
        "raw_artifact_snapshot_unchanged": unchanged,
        "raw_artifact_snapshot_pre": before,
        "raw_artifact_snapshot_post": after,
        "blocks": blocks,
        "source_hash_scope": "80 immutable raw run directories; 16 fresh + 64 reused",
        "collection_contract": {
            "raw_row_count": len(result.get("raw_rows", [])),
            "paired_row_count": len(result.get("paired_rows", [])),
            "run_inventory_count": len(result.get("inventory", [])),
            "invalid_pair_row_count": sum(
                int(row.get("pairing_valid", 0)) != 1
                for row in result.get("paired_rows", [])),
        },
        "scientific_analysis": "forbidden; structural collection only",
    }
    if audit_payload["allowed_normalization_count"] != 16:
        raise RuntimeError("expected 16 dataset-only normalizations")
    if audit_payload["mapping_counts"] != {"kuairand": 4, "movielens": 12}:
        raise RuntimeError("unexpected dataset normalization mapping counts")
    if audit_payload["non_dataset_bypass_count"] != 0:
        raise RuntimeError("non-dataset comparison bypass detected")

    os.makedirs(output_root, exist_ok=True)
    _write_normalization_audit(output_root, audit_payload)
    summary_path = os.path.join(output_root, "stage_S2_summary_manifest.json")
    if os.path.isfile(summary_path):
        summary = _read_json(summary_path)
        summary["normalization_audit"] = {
            "path": NORMALIZATION_AUDIT_JSON,
            "schema": audit_payload["schema"],
            "adapter_source_hash": audit_payload["adapter_source_hash"],
            "allowed_normalization_count": audit_payload["allowed_normalization_count"],
            "non_dataset_bypass_count": audit_payload["non_dataset_bypass_count"],
            "raw_artifact_snapshot_unchanged": audit_payload["raw_artifact_snapshot_unchanged"],
        }
        _write_json(summary_path, summary)
    result["normalization_audit"] = audit_payload
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S2 post-run structural collector adapter")
    parser.add_argument("--s2_raw_root", default=DEFAULT_S2_RAW_ROOT)
    parser.add_argument("--s1_raw_root", default=DEFAULT_S1_RAW_ROOT)
    parser.add_argument("--output_root", default=HERE)
    parser.add_argument("--dry_run", action="store_true",
                        help="write derived outputs only to a temporary directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="stage_s2_postrun_collect_") as output_root:
            result = collect_postrun(
                s2_raw_root=args.s2_raw_root,
                s1_raw_root=args.s1_raw_root,
                output_root=output_root,
            )
            print(json.dumps({
                "dry_run": True,
                "raw_row_count": len(result.get("raw_rows", [])),
                "paired_row_count": len(result.get("paired_rows", [])),
                "run_inventory_count": len(result.get("inventory", [])),
                "normalization_audit": result["normalization_audit"]["schema"],
                "allowed_normalization_count": result["normalization_audit"][
                    "allowed_normalization_count"],
                "mapping_counts": result["normalization_audit"]["mapping_counts"],
                "non_dataset_bypass_count": result["normalization_audit"][
                    "non_dataset_bypass_count"],
                "raw_artifact_snapshot_unchanged": result["normalization_audit"][
                    "raw_artifact_snapshot_unchanged"],
            }, sort_keys=True))
        return 0
    result = collect_postrun(
        s2_raw_root=args.s2_raw_root,
        s1_raw_root=args.s1_raw_root,
        output_root=args.output_root,
    )
    print(json.dumps({
        "raw_row_count": len(result.get("raw_rows", [])),
        "paired_row_count": len(result.get("paired_rows", [])),
        "run_inventory_count": len(result.get("inventory", [])),
        "normalization_audit": NORMALIZATION_AUDIT_JSON,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DatasetNormalizationError", "normalize_dataset_pair", "compare_normalized_manifests",
    "snapshot_raw_artifacts", "inspect_formal_normalizations", "collect_postrun", "main",
]
