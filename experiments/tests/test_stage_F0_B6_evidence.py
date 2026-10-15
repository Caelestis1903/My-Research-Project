from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import stage_F0_B6_evidence as evidence

_COVERAGE = (
    "current_gate_hold_release", "v14_historical_controls",
    "incident_failure_stream_evidence", "audit_logs_and_incident_slice",
    "C97_H51_state_runs_checkpoints", "F140_status_locks_runs_attempts",
    "queue_and_search_gate", "source_runtime_control_and_tests",
    "frozen_data_split_manifests",
)


def _collect(**kwargs):
    """Add four isolated read-only quiescence fixtures, then call the real collector."""
    roots = kwargs["scope_roots"]
    paths = {root_id: list(values) for root_id, values in kwargs["scope_paths"].items()}
    required = set(kwargs["required_paths"])
    expected_absent = set(kwargs["expected_absent"])
    quiescence_paths = {}
    scope_coverage = {}
    f140_members = []
    f140_root_id = next(iter(roots))
    for root_id, root_value in roots.items():
        root = Path(root_value)
        qdir = root / "__b6_quiescence"
        if root.is_dir():
            qdir.mkdir(exist_ok=True)
        fixtures = {
            "processes": b"synthetic process snapshot\n",
            "claims": b"synthetic claim snapshot\n",
            "slots": b"synthetic slot snapshot\n",
            "proc_locks": b"synthetic /proc/locks snapshot\n",
        }
        for category, data in fixtures.items():
            if qdir.is_dir() and not (qdir / category).exists():
                (qdir / category).write_bytes(data)
            relative = "__b6_quiescence/" + category
            quiescence_paths.setdefault(category, root_id + ":" + relative)
            required.add(root_id + ":" + relative)
        coverage_dir = root / "__b6_coverage"
        if root.is_dir():
            coverage_dir.mkdir(exist_ok=True)
        for category in _COVERAGE:
            if category == "F140_status_locks_runs_attempts" and root_id == f140_root_id:
                for index in range(140):
                    member_id = hashlib.sha256(
                        ("synthetic-F140-member-{}".format(index)).encode()).hexdigest()[:20]
                    role_paths = {
                        "status": "queue_status/" + member_id + ".json",
                        "lock": "canonical_locks/" + member_id + ".lock",
                        "run": member_id,
                        "attempt": member_id + "/attempts/attempt-00",
                    }
                    bound_paths = {}
                    for role, relative in role_paths.items():
                        key = root_id + ":" + relative
                        bound_paths[role] = key
                        if relative not in paths[root_id]:
                            paths[root_id].append(relative)
                        required.add(key)
                        expected_absent.add(key)
                    f140_members.append({"member_id": member_id, "paths": bound_paths})
            elif category != "F140_status_locks_runs_attempts":
                relative = "__b6_coverage/" + category + ".json"
                references = [root_id + ":" + relative]
                if coverage_dir.is_dir() and not (root / relative).exists():
                    (root / relative).write_text("synthetic " + category + " evidence\n", encoding="utf-8")
                required.add(root_id + ":" + relative)
                if "__b6_coverage" not in paths[root_id]:
                    paths[root_id].append("__b6_coverage")
                scope_coverage[category] = references
        if "__b6_quiescence" not in paths[root_id]:
            paths[root_id].append("__b6_quiescence")
    scope_coverage["F140_status_locks_runs_attempts"] = {
        "schema": evidence._F140_COVERAGE_SCHEMA,
        "members": sorted(f140_members, key=lambda item: item["member_id"])}
    kwargs["scope_paths"] = paths
    kwargs["required_paths"] = required
    kwargs["expected_absent"] = expected_absent
    kwargs["quiescence_paths"] = quiescence_paths
    kwargs["scope_coverage"] = scope_coverage
    kwargs["repo_path"] = kwargs.get("repo_path", next(iter(roots.values())))
    return evidence.collect_postincident_baseline(**kwargs)


def _case(tmp_path, *, baseline_id="baseline-a", paths=None, required=None,
          expected_absent=(), protected=(), quiescence=()):
    source = tmp_path / "production-synthetic"
    source.mkdir(parents=True)
    archive_parent = tmp_path / "evidence" / "B6"
    archive_parent.mkdir(parents=True)
    (source / "gate.json").write_bytes(b'{"launch_authorized":false}\n')
    (source / "history.log").write_bytes(b"legacy non-json line\n{\"event\":\"CHILD_STARTED\"}\n")
    (source / "nested").mkdir()
    (source / "nested" / "status.json").write_text('{"status":"UNCLAIMED"}\n', encoding="utf-8")
    policy = tmp_path / "policy.md"
    policy.write_text("synthetic policy fixture\n", encoding="utf-8")
    selected = paths or ["gate.json", "history.log", "nested"]
    keys = {"prod:" + path for path in selected}
    result = _collect(
        baseline_id=baseline_id,
        scope_roots={"prod": source},
        scope_paths={"prod": selected},
        archive_parent=archive_parent,
        forbidden_roots=[source],
        required_paths=required if required is not None else keys,
        expected_absent={"prod:" + path for path in expected_absent},
        protected_paths={"prod:" + path for path in protected},
        policy_path=policy,
        collector_command=["collect", "--read-only"],
        environment={"python": "project-env", "mode": "synthetic"},
        history_missing=["pre-edit inventory", "original runner preimage/diff"],
        residual_risks=["synthetic fixture does not restore historical evidence"],
    )
    return result, source, archive_parent


def test_stable_inventory_archives_raw_bytes_and_never_grants_authority(tmp_path):
    result, source, _ = _case(tmp_path, quiescence=["gate.json"])
    assert result["status"] == "COLLECTED_UNREVIEWED"
    assert result["execution_authorization"] == "NO_GO"
    doc = evidence.validate_postincident_manifest(result["manifest_path"])
    assert doc["historical_B6_status"] == "OPEN_FAIL_IRRECOVERABLE_GAP"
    assert doc["postincident_baseline_status"] == "COLLECTED_UNREVIEWED"
    assert doc["forward_disposition_status"] == "NOT_GRANTED"
    assert doc["historical_no_side_effects_proven"] is False
    assert doc["historical_full_equality_proven"] is False
    assert doc["original_runner_byte_identity_proven"] is False
    assert doc["stable_double_read"] is True
    assert doc["quiescence_observations"]["lock_acquired"] is False
    assert set(doc["quiescence_observations"]["observations"]) == {
        "processes", "claims", "slots", "proc_locks"}
    assert doc["quiescence_observations"]["observations"]["proc_locks"]["sha256"]
    source_log = source / "history.log"
    log_row = next(row for row in doc["inventory"] if row["path"] == "prod:history.log")
    archive_log = Path(result["archive_path"]) / log_row["archive_rel_path"]
    assert archive_log.read_bytes() == source_log.read_bytes()
    assert b"legacy non-json line" in archive_log.read_bytes()
    assert len(doc["inventory_sha256"]) == 64


def test_manifest_rejects_empty_policy_categories_even_when_f140_is_bound(tmp_path):
    result, _, _ = _case(tmp_path)
    manifest = Path(result["manifest_path"])
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    for category in _COVERAGE:
        if category != "F140_status_locks_runs_attempts":
            doc["scope_coverage"][category] = []
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="no required evidence"):
        evidence.validate_postincident_manifest(manifest)


def test_f140_coverage_binds_140_members_to_all_realistic_absence_paths(tmp_path):
    result, _, _ = _case(tmp_path)
    doc = evidence.validate_postincident_manifest(result["manifest_path"])
    coverage = doc["scope_coverage"]["F140_status_locks_runs_attempts"]
    members = coverage["members"]
    assert coverage["schema"] == evidence._F140_COVERAGE_SCHEMA
    assert len(members) == 140
    assert len({member["member_id"] for member in members}) == 140
    assert len(doc["absence_paths"]) >= 560
    assert all(set(member["paths"]) == {"status", "lock", "run", "attempt"}
               for member in members)
    assert all(member["paths"]["status"].endswith(
                   "queue_status/" + member["member_id"] + ".json")
               for member in members)
    assert all(member["paths"]["lock"].endswith(
                   "canonical_locks/" + member["member_id"] + ".lock")
               for member in members)
    assert len({member["paths"]["lock"] for member in members}) == 140


def test_f140_incomplete_or_misassigned_role_is_rejected(tmp_path):
    result, _, _ = _case(tmp_path / "incomplete")
    manifest = Path(result["manifest_path"])
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    del doc["scope_coverage"]["F140_status_locks_runs_attempts"]["members"][0]["paths"]["attempt"]
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="each F140 member must bind"):
        evidence.validate_postincident_manifest(manifest)

    result, _, _ = _case(tmp_path / "misassigned")
    manifest = Path(result["manifest_path"])
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    member = doc["scope_coverage"]["F140_status_locks_runs_attempts"]["members"][0]
    member["paths"]["status"] = member["paths"]["lock"]
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="status path does not match"):
        evidence.validate_postincident_manifest(manifest)


def test_manifest_rejects_coverage_path_reused_across_policy_categories(tmp_path):
    result, _, _ = _case(tmp_path)
    manifest = Path(result["manifest_path"])
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["scope_coverage"]["v14_historical_controls"] = list(
        doc["scope_coverage"]["current_gate_hold_release"])
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="reused across categories"):
        evidence.validate_postincident_manifest(manifest)


def test_manifest_rejects_quiescence_categories_reusing_one_absent_f140_path(tmp_path):
    result, _, _ = _case(tmp_path)
    manifest = Path(result["manifest_path"])
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    absent = doc["scope_coverage"]["F140_status_locks_runs_attempts"]["members"][0]["paths"]["status"]
    doc["quiescence_observations"]["path_ids"] = {
        category: absent for category in ("processes", "claims", "slots", "proc_locks")}
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="four distinct paths"):
        evidence.validate_postincident_manifest(manifest)


def test_manifest_rejects_missing_quiescence_observation_section(tmp_path):
    result, _, _ = _case(tmp_path)
    manifest = Path(result["manifest_path"])
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    del doc["quiescence_observations"]
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="quiescence schema invalid"):
        evidence.validate_postincident_manifest(manifest)


def test_validator_rejects_archive_hardlinked_to_source_inode(tmp_path):
    result, source, _ = _case(tmp_path)
    manifest = Path(result["manifest_path"])
    doc = evidence.validate_postincident_manifest(manifest)
    row = next(item for item in doc["inventory"] if item["path"] == "prod:gate.json")
    archived = Path(result["archive_path"]) / row["archive_rel_path"]
    archived.unlink()
    os.link(source / "gate.json", archived)
    with pytest.raises(evidence.EvidenceError, match="archive inode aliases recorded source inode"):
        evidence.validate_postincident_manifest(manifest)


def test_expected_absence_is_explicit_but_unexpected_missing_path_is_rejected(tmp_path):
    result, _, _ = _case(tmp_path, paths=["gate.json", "missing.status"],
                         required={"prod:gate.json"}, expected_absent=["missing.status"])
    doc = evidence.validate_postincident_manifest(result["manifest_path"])
    assert result["status"] == "COLLECTED_UNREVIEWED"
    absent = next(row for row in doc["inventory"] if row["path"] == "prod:missing.status")
    assert absent["kind"] == "absent"
    assert absent["path"] in doc["absence_paths"]

    other = tmp_path / "unexpected"
    other.mkdir()
    archive = tmp_path / "archive-unexpected"
    archive.mkdir()
    policy = tmp_path / "policy-unexpected.md"
    policy.write_text("policy", encoding="utf-8")
    rejected = _collect(
        baseline_id="missing-required", scope_roots={"r": other},
        scope_paths={"r": ["must-exist"]}, archive_parent=archive,
        forbidden_roots=[other], required_paths={"r:must-exist"}, expected_absent=set(),
        protected_paths=set(), policy_path=policy, collector_command=["test"],
        environment={}, history_missing=["unknown"], residual_risks=[])
    assert rejected["status"] == "REJECTED"
    with pytest.raises(evidence.EvidenceError, match="baseline rejected"):
        evidence.validate_postincident_manifest(rejected["manifest_path"])
    rejected_doc = json.loads(Path(rejected["manifest_path"]).read_text(encoding="utf-8"))
    assert next(row for row in rejected_doc["inventory"]
                if row["path"] == "r:must-exist")["kind"] == "absent"
    assert any(error.startswith("MISSING_REQUIRED:") for error in rejected_doc["errors"])
    assert rejected_doc["execution_authorization"] == "NO_GO"


def test_expected_absent_path_that_exists_is_rejected(tmp_path):
    result, _, _ = _case(tmp_path, paths=["gate.json"], expected_absent=["gate.json"])
    assert result["status"] == "REJECTED"
    with pytest.raises(evidence.EvidenceError, match="baseline rejected"):
        evidence.validate_postincident_manifest(result["manifest_path"])
    doc = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert "EXPECTED_ABSENCE_PATH_EXISTS" in doc["errors"][0]


def test_symlink_is_archived_as_target_text_without_following(tmp_path):
    result, source, _ = _case(tmp_path, paths=["gate.json", "outside-link"])
    external = tmp_path / "external-secret"
    external.write_bytes(b"do not archive target contents")
    link = source / "outside-link"
    link.symlink_to(external)
    # The fixture's initial collection ran before this link was created; use a fresh run.
    result = _collect(
        baseline_id="symlink-case", scope_roots={"prod": source},
        scope_paths={"prod": ["outside-link"]}, archive_parent=tmp_path / "evidence" / "B6",
        forbidden_roots=[source], required_paths={"prod:outside-link"}, expected_absent=set(),
        protected_paths=set(), policy_path=tmp_path / "policy.md", collector_command=["test"],
        environment={}, history_missing=["old preimage"], residual_risks=[])
    doc = evidence.validate_postincident_manifest(result["manifest_path"])
    row = next(row for row in doc["inventory"] if row["path"] == "prod:outside-link")
    assert row["kind"] == "symlink"
    archived = Path(result["archive_path"]) / row["archive_rel_path"]
    assert archived.read_text(encoding="utf-8") == str(external)
    assert b"do not archive target contents" not in archived.read_bytes()


def test_protected_symlink_and_symlinked_root_fail_closed(tmp_path):
    result, source, archive_parent = _case(tmp_path)
    (source / "protected-link").symlink_to(source / "gate.json")
    with pytest.raises(evidence.EvidenceError, match="protected control path is a symlink"):
        _collect(
            baseline_id="protected-link", scope_roots={"prod": source},
            scope_paths={"prod": ["protected-link"]}, archive_parent=archive_parent,
            forbidden_roots=[source], required_paths={"prod:protected-link"}, expected_absent=set(),
            protected_paths={"prod:protected-link"}, policy_path=tmp_path / "policy.md",
            collector_command=["test"], environment={}, history_missing=[], residual_risks=[])
    alias = tmp_path / "root-alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(evidence.EvidenceError, match="symlink path component"):
        _collect(
            baseline_id="symlink-root", scope_roots={"prod": alias},
            scope_paths={"prod": ["gate.json"]}, archive_parent=archive_parent,
            forbidden_roots=[source], required_paths={"prod:gate.json"}, expected_absent=set(),
            protected_paths=set(), policy_path=tmp_path / "policy.md", collector_command=["test"],
            environment={}, history_missing=[], residual_risks=[])


def test_broken_symlink_is_not_misclassified_as_absent(tmp_path):
    _, source, archive_parent = _case(tmp_path)
    (source / "broken").symlink_to(source / "does-not-exist")
    result = _collect(
        baseline_id="broken-symlink", scope_roots={"prod": source},
        scope_paths={"prod": ["broken"]}, archive_parent=archive_parent,
        forbidden_roots=[source], required_paths={"prod:broken"}, expected_absent=set(),
        protected_paths=set(), policy_path=tmp_path / "policy.md", collector_command=["test"],
        environment={}, history_missing=[], residual_risks=[])
    doc = evidence.validate_postincident_manifest(result["manifest_path"])
    row = next(row for row in doc["inventory"] if row["path"] == "prod:broken")
    assert row["kind"] == "symlink"
    assert row["symlink_target"].endswith("does-not-exist")


def test_source_hardlink_alias_is_refused(tmp_path):
    _, source, archive_parent = _case(tmp_path)
    os.link(source / "gate.json", source / "gate-copy.json")
    with pytest.raises(evidence.EvidenceError, match="hard-linked source refused"):
        _collect(
            baseline_id="hardlink", scope_roots={"prod": source},
            scope_paths={"prod": ["gate.json", "gate-copy.json"]}, archive_parent=archive_parent,
            forbidden_roots=[source], required_paths={"prod:gate.json", "prod:gate-copy.json"},
            expected_absent=set(), protected_paths=set(), policy_path=tmp_path / "policy.md",
            collector_command=["test"], environment={}, history_missing=[], residual_risks=[])


def test_archive_inside_or_overlapping_scope_is_refused(tmp_path):
    source = tmp_path / "scope"
    source.mkdir()
    (source / "item").write_text("x", encoding="utf-8")
    policy = tmp_path / "policy.md"
    policy.write_text("policy", encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="overlaps a scoped root|inside a forbidden"):
        _collect(
            baseline_id="archive-alias", scope_roots={"r": source}, scope_paths={"r": ["item"]},
            archive_parent=source / "new-evidence", forbidden_roots=[source],
            required_paths={"r:item"}, expected_absent=set(), protected_paths=set(),
            policy_path=policy, collector_command=["test"], environment={},
            history_missing=[], residual_risks=[])


def test_two_reads_detect_drift_and_preserve_rejected_observations(tmp_path, monkeypatch):
    source = tmp_path / "scope"
    source.mkdir()
    item = source / "item"
    item.write_bytes(b"before")
    archive = tmp_path / "archive"
    archive.mkdir()
    policy = tmp_path / "policy.md"
    policy.write_text("policy", encoding="utf-8")
    real_scan = evidence._scan_once
    count = {"value": 0}

    def drift_between_reads(**kwargs):
        count["value"] += 1
        rows = real_scan(**kwargs)
        if count["value"] == 1:
            item.write_bytes(b"after")
        return rows

    monkeypatch.setattr(evidence, "_scan_once", drift_between_reads)
    result = _collect(
        baseline_id="drift", scope_roots={"r": source}, scope_paths={"r": ["item"]},
        archive_parent=archive, forbidden_roots=[source], required_paths={"r:item"},
        expected_absent=set(), protected_paths=set(), policy_path=policy,
        collector_command=["test"], environment={}, history_missing=[], residual_risks=[])
    assert result["status"] == "REJECTED_UNSTABLE"
    with pytest.raises(evidence.EvidenceError, match="unstable"):
        evidence.validate_postincident_manifest(result["manifest_path"])
    doc = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert doc["stable_double_read"] is False
    assert "DOUBLE_READ_UNSTABLE" in doc["errors"]
    first = next(row for row in doc["inventory"] if row["path"] == "r:item")
    second = next(row for row in doc["second_read_inventory"] if row["path"] == "r:item")
    assert first["sha256"] != second["sha256"]
    assert (Path(result["archive_path"]) / first["archive_rel_path"]).read_bytes() == b"before"
    assert (Path(result["archive_path"]) / second["archive_rel_path"]).read_bytes() == b"after"


def test_archive_corruption_and_manifest_digest_tampering_are_rejected(tmp_path):
    result, _, _ = _case(tmp_path)
    doc = evidence.validate_postincident_manifest(result["manifest_path"])
    row = next(row for row in doc["inventory"] if row["kind"] == "regular")
    raw = Path(result["archive_path"]) / row["archive_rel_path"]
    raw.write_bytes(b"corrupted after collection")
    with pytest.raises(evidence.EvidenceError, match="archive raw SHA mismatch"):
        evidence.validate_postincident_manifest(result["manifest_path"])

    second, _, _ = _case(tmp_path / "second")
    manifest = Path(second["manifest_path"])
    altered = json.loads(manifest.read_text(encoding="utf-8"))
    altered["inventory_sha256"] = "0" * 64
    manifest.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="digest-mismatched"):
        evidence.validate_postincident_manifest(manifest)


def test_execution_go_or_historical_pass_never_validate(tmp_path):
    result, _, _ = _case(tmp_path)
    manifest = Path(result["manifest_path"])
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["execution_authorization"] = "GO"
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="non-authorization"):
        evidence.validate_postincident_manifest(manifest)


def test_deterministic_inventory_and_manifest_refuse_overwrite(tmp_path):
    result, source, archive_parent = _case(tmp_path, baseline_id="first")
    doc = evidence.validate_postincident_manifest(result["manifest_path"])
    # A distinct archive id changes only archive namespace, not source inventory.
    second = _collect(
        baseline_id="second", scope_roots={"prod": source},
        scope_paths={"prod": ["gate.json", "history.log", "nested"]},
        archive_parent=archive_parent, forbidden_roots=[source],
        required_paths={"prod:gate.json", "prod:history.log", "prod:nested"},
        expected_absent=set(), protected_paths=set(), policy_path=tmp_path / "policy.md",
        collector_command=["collect", "--read-only"],
        environment={"python": "project-env", "mode": "synthetic"},
        history_missing=["pre-edit inventory", "original runner preimage/diff"],
        residual_risks=["synthetic fixture does not restore historical evidence"])
    doc2 = evidence.validate_postincident_manifest(second["manifest_path"])
    assert doc["inventory_sha256"] == doc2["inventory_sha256"]
    with pytest.raises(evidence.EvidenceError, match="already exists"):
        _collect(
            baseline_id="first", scope_roots={"prod": source},
            scope_paths={"prod": ["gate.json"]}, archive_parent=archive_parent,
            forbidden_roots=[source], required_paths={"prod:gate.json"}, expected_absent=set(),
            protected_paths=set(), policy_path=tmp_path / "policy.md", collector_command=["test"],
            environment={}, history_missing=[], residual_risks=[])


def test_unreadable_missing_root_and_malformed_manifest_fail_closed(tmp_path):
    with pytest.raises(evidence.EvidenceError, match="path component is absent"):
        _collect(
            baseline_id="missing-root", scope_roots={"r": tmp_path / "absent"},
            scope_paths={"r": ["item"]}, archive_parent=tmp_path, forbidden_roots=[],
            required_paths={"r:item"}, expected_absent=set(), protected_paths=set(),
            policy_path=tmp_path / "not-policy", collector_command=["test"], environment={},
            history_missing=[], residual_risks=[])
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="malformed"):
        evidence.validate_postincident_manifest(bad)
