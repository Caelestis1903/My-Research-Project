"""Current-conversation entry points for the frozen history-information matrix."""

import argparse
import csv
import io
import json
import shlex
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from . import analysis
from .selection import (
    ARTIFACTS, CELLS, DATASETS, FAMILIES, PLAN, READOUTS, ROOT, SEEDS, STAGES, STUDY,
    assert_open, conditional_modules, digest, file_hash, load, materialize, pending_baseline_repairs, read_plan,
    read_queue, stage_decision, write_json,
)

PUBLISHED_REPORT = "research_reports/2026-11-03_overnight_AB/"
PUBLISHED_PROTOCOL = ROOT / "research_plans/26-11-03_overnight_AB/protocol.json"
HISTORY_TABLE = "artifacts/conditional_history_shared_update_20261122/tables/A_HISTORY/endpoints.csv"
PUBLISHED_MODELS = {"M3oE": "A0", "HyperGate": "HYPERGATE", "AutoIFS": "AUTOIFS"}
TEST_POLICY = "TEST_POLICY.json"
FINAL_FREEZE = "FINAL_FREEZE.json"
TEST_ACCESS = "TEST_ACCESS.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_blob(relative, revision, expected=None):
    """Read the pinned bytes, and compare the local copy when it exists."""
    pinned = subprocess.check_output(
        ["git", "-C", str(ROOT.parent / "phase0-3_git"), "rev-parse", f"{revision}:{relative}"],
        text=True,
    ).strip()
    if expected is not None and pinned != expected:
        raise ValueError(f"unexpected pinned blob: {relative}")
    content = subprocess.check_output(
        ["git", "-C", str(ROOT.parent / "phase0-3_git"), "cat-file", "blob", pinned],
    )
    path = ROOT / relative
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"actual source differs from pinned blob: {relative}")
    from hashlib import sha256

    return content, {"path": relative, "commit": revision, "git_blob": pinned,
                     "sha256": sha256(content).hexdigest()}


def csv_content(content):
    return list(csv.DictReader(io.StringIO(content.decode("utf-8"))))


def published_inventory(plan, bundles):
    from experiments.overnight_ab_20261103.config import job_config

    commit = plan["repository"]["published_model_source_sha"]
    table, source = checked_blob(PUBLISHED_REPORT + "ENDPOINTS.csv", commit,
                                 "a3d7aa3e2035f0b7ff55406e260fc59c880ba254")
    _, reuse_audit = checked_blob(PUBLISHED_REPORT + "A_REUSE_AUDIT.csv", commit,
                                  "3ea43b298e5fb832bfdc595f286e5384fa5244a7")
    protocol_content, protocol_source = checked_blob(
        str(PUBLISHED_PROTOCOL.relative_to(ROOT)), commit, "44a5238ecbf7fad0970e9f605cdac1824cb727f7",
    )
    protocol = json.loads(protocol_content)
    selection = protocol["selection"]
    if (selection["mode"], selection["metric"], selection["first_epoch"], selection["last_epoch"],
        selection["autoifs_stage"], selection["use_test_for_selection"], selection["cells"]) != (
        "strict_max_earliest_tie", "macro_auc", 1, 50, "reuse", False, list(CELLS),
    ):
        raise ValueError("published protocol selected identity differs")
    endpoints = defaultdict(list)
    for row in csv_content(table):
        if row["role"] == "PUBLISHED_BASELINE" and row["checkpoint"] == "selected":
            endpoints[row["dataset"], int(row["seed"]), row["arm"]].append(row)
    slots = []
    for dataset, bundle in bundles.items():
        if protocol["datasets"][dataset]["raw_sha256"] != bundle.identity["raw_sha256"]:
            raise ValueError(f"published raw population differs: {dataset}")
        supports = {}
        for domain in range(3):
            mask = bundle.val_x[:, -1] == domain
            for task in range(2):
                supports[f"d{domain}t{task}"] = int(mask.sum()), int(bundle.val_y[mask, task].sum())
        for seed in SEEDS:
            slots.append({"model": "M3oE", "dataset": dataset, "seed": seed,
                          "status": "MATCH_CURRENT_B0", "arm": "B0"})
            for name, arm in PUBLISHED_MODELS.items():
                if name == "M3oE":
                    continue
                pair = endpoints[dataset, seed, arm]
                expected_stage = "reuse" if arm == "AUTOIFS" else "train"
                expected_config = digest(job_config(protocol, arm, dataset))
                failures = []
                validation = next((row for row in pair if row["split"] == "validation"), {})
                test = next((row for row in pair if row["split"] == "test"), {})
                if len(pair) != 2 or not validation or not test:
                    failures.append("MISSING_MATCHED_SELECTED_PAIR")
                if not failures:
                    match_fields = ("epoch", "selection_id", "model_hash", "source_hash", "trajectory_id")
                    if any(not validation[key] or validation[key] != test[key] for key in match_fields):
                        failures.append("VALIDATION_AND_TEST_SELECTED_STATE_MISMATCH")
                    for row in (validation, test):
                        expected_id = digest(dict(
                            trajectory_id=row["trajectory_id"], estimator="online", selection=selection,
                        ))
                        expected_endpoint = digest(dict(
                            selection_id=expected_id, checkpoint="selected", stage=expected_stage,
                            epoch=int(row["epoch"]), model_hash=row["model_hash"], split=row["split"],
                        ))
                        if (row["estimator"] != "online" or row["stage"] != expected_stage
                                or row["protocol_id"] != protocol["protocol_id"]
                                or row["data_protocol"] != protocol["data_protocol"]
                                or not 1 <= int(row["epoch"]) <= 50
                                or row["config_hash"] != expected_config
                                or row["selection_id"] != expected_id or row["endpoint_id"] != expected_endpoint
                                or any(analysis.number(row[f"{cell}_auc"]) is None for cell in CELLS)):
                            failures.append(f"{row['split'].upper()}_ENDPOINT_IDENTITY")
                    if any(tuple(int(float(validation[f"{cell}_{key}"])) for key in ("count", "positive"))
                           != support for cell, support in supports.items()):
                        failures.append("VALIDATION_POPULATION_SUPPORT_MISMATCH")
                slot = {"model": name, "dataset": dataset, "seed": seed, "arm": arm,
                        "status": "BASELINE_REPAIR_REQUIRED" if failures else "NUMERIC_REUSE",
                        "identity_failures": failures, "readout": "online",
                        "config": job_config(protocol, arm, dataset), "config_hash": expected_config,
                        "data_hash": bundle.data_hash, "raw_sha256": bundle.identity["raw_sha256"]}
                if not failures:
                    slot.update({name: validation[name] for name in (
                        "model_hash", "source_hash", "selection_id", "trajectory_id", "source_sha256",
                    )})
                    slot.update(epoch=int(validation["epoch"]), stage=expected_stage,
                                validation_endpoint_id=validation["endpoint_id"],
                                test_endpoint_id=test["endpoint_id"],
                                row_population_hash=bundle.identity["split_row_fingerprints"]["validation"])
                slots.append(slot)
    return {"models": list(PUBLISHED_MODELS), "source": source, "audit_source": reuse_audit,
            "protocol_source": protocol_source, "slots": slots}


def reuse_inventory(plan, roster):
    main = plan["repository"]["evidence_main_sha"]
    values, history_source = checked_blob(
        HISTORY_TABLE, main, "d420cdf0043f88b66a6bb886e0c73d72175023a8",
    )
    rows = csv_content(values)
    if len(rows) != 6480:
        raise ValueError("unexpected D0 source endpoint count")
    derived = []
    for stem, count in (("seed", 1050), ("summary", 210)):
        path = PLAN.parent / f"evidence/validation_interaction_{stem}.csv"
        if len(analysis.csv_rows(path)) != count:
            raise ValueError(f"unexpected D0 derived count: {path}")
        derived.append({"path": str(path.relative_to(ROOT)), "sha256": file_hash(path), "rows": count})
    history = {}
    for row in rows:
        if row["arm"] not in {"A00", "A10", "A20", "A21", "A22"} or row["checkpoint_rule"] != "selected":
            continue
        key = (row["dataset"], int(row["seed"]), row["arm"], row["readout"], row["split"])
        if row["cell"] == CELLS[0]:
            if key in history:
                raise ValueError(f"duplicated historical endpoint identity: {key}")
            history[key] = dict(
                reference_id=row["arm"], canonical_trajectory_id=row["trajectory_id"],
                attempt_id=row["physical_attempt"], dataset=row["dataset"], seed=int(row["seed"]),
                arm=row["arm"], readout=row["readout"], endpoint="selected", split=row["split"],
                epoch=row["epoch"], source_commit=main, source_hash=row["source_hash"],
                config_hash=row["config_hash"], data_hash=row["data_hash"],
                selection_protocol="HISTORICAL_STRICT_MAX_EARLIEST", model_hash=row["model_hash"],
                numeric_source=history_source["path"], numeric_sha256=history_source["sha256"],
                numeric_available=True, predictions_available=False, ram_get_verified=False,
                optimizer_available=False, shadow_complete=False, reuse_status="NUMERIC_REUSE",
                rebuild_reason=("LIVE_STATE_UNAVAILABLE" if row["arm"] in {"A00", "A10", "A21"} else ""),
                logical_trajectory_count=0, physical_run_count=0,
            )
    result = list(history.values())
    for slot in roster["slots"]:
        if slot["model"] == "M3oE":
            continue
        for split in ("validation", "test"):
            result.append(dict(
                reference_id=slot["model"], canonical_trajectory_id=slot.get("trajectory_id"),
                dataset=slot["dataset"], seed=slot["seed"], arm=slot["arm"], readout="online",
                endpoint="selected", split=split, epoch=slot.get("epoch"),
                source_commit=roster["source"]["commit"], source_hash=slot.get("source_hash"),
                config_hash=slot["config_hash"], data_hash=slot["data_hash"],
                row_population_hash=slot.get("row_population_hash") if split == "validation" else "",
                selection_protocol="AUC_SELECTED_V1", model_hash=slot.get("model_hash"),
                numeric_source=roster["source"]["path"], numeric_sha256=roster["source"]["sha256"],
                numeric_available=slot["status"] == "NUMERIC_REUSE", predictions_available=False,
                ram_get_verified=False, optimizer_available=False, shadow_complete=False,
                reuse_status=slot["status"], rebuild_reason=";".join(slot["identity_failures"]),
                logical_trajectory_count=0, physical_run_count=0,
            ))
    return result, {"source": history_source, "rows": len(rows), "derived": derived}


def past_parent_store():
    path = ROOT / "artifacts/conditional_history_shared_update_20261122/attempt1/guardian.json"
    if not path.is_file():
        return {"receipt": str(path), "status": "ARCHIVED_RECEIPT_UNAVAILABLE"}
    info = load(path)
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(1)
        try:
            client.connect(info["address"])
        except (FileNotFoundError, ConnectionRefusedError):
            return {"receipt": str(path), "pid": info["pid"], "status": "ARCHIVED_GUARDIAN_NO_LISTENER"}
    raise ValueError("historical parent guardian has a listener and needs actual state identity audit")


def audit(root, path):
    from . import sources
    from .protocol import RunSpec

    assert_open(root)
    if Path(path).resolve() != PLAN:
        raise ValueError("audit requires this study's approved manifest")
    plan = read_plan()
    root.mkdir(parents=True, exist_ok=True)
    status = root / "AUDIT_RECEIPT.json"
    if (root / "S1" / "QUEUE.json").exists():
        raise ValueError("S0 manifest frozen by the S1 queue")
    if load(root / TEST_POLICY)["study_id"] != STUDY:
        raise ValueError("current-scope Test policy has the wrong study")
    try:
        sources.ensure_shared_sources()
        from . import scheduler

        scheduler.register_stop_controls(root)
        source = sources.audit_sources()
        from . import groups, objectives

        cache = root / "train_validation_inputs"
        bundles = {dataset: sources.prepare_train_validation_cache(cache, dataset) for dataset in DATASETS}
        data = {dataset: bundle.identity for dataset, bundle in bundles.items()}
        scopes = {}
        for dataset, bundle in bundles.items():
            scopes[dataset] = {}
            configurations = (("m3oe", "H0"), ("hypergate", "HG_B0"), ("autoifs", "REPAIR_AUTOIFS"))
            expected = {"movielens": (773470, 290318, 483152), "kuairand": (1614254, 952334, 661920)}
            for backbone, arm in configurations:
                if backbone != "m3oe" and not source["published_source_verified"]:
                    continue
                spec = RunSpec(dataset=dataset, seed=SEEDS[0], arm=arm)
                model, optimizer, ema = objectives.build_training(spec, bundle, "cpu")
                _, identity = groups.registry(model, bundle)
                if backbone == "m3oe":
                    counts = identity["counts"]
                    identity["s_dense_numel"] = sum(counts[group] for group in ("STAR_SKIP", "EXPERT"))
                    identity["n_numel"] = sum(count for group, count in counts.items() if group.startswith(("D", "T")))
                    actual = (sum(counts.values()), identity["n_numel"],
                              sum(counts.values()) - identity["n_numel"])
                    if actual != expected[dataset]:
                        raise ValueError(f"frozen parameter counts differ: {dataset}/{actual}")
                scopes[dataset][backbone] = identity
                del model, optimizer, ema
        roster = published_inventory(plan, bundles)
        reuse, d0 = reuse_inventory(plan, roster)
        parent_live = past_parent_store()
    except Exception as exc:
        write_json(status, {"study_id": STUDY, "status": "TECHNICAL_PAUSED",
                            "blocker": f"{type(exc).__name__}: {exc}"})
        raise
    for filename, content in (("SOURCE_MANIFEST.json", source), ("DATA_MANIFEST.json", data),
                              ("PARAMETER_SCOPE_MANIFEST.json", scopes), ("PUBLISHED_ROSTER.json", roster)):
        write_json(root / filename, content)
    analysis.write_csv(root / "REUSE_REGISTER.csv", reuse, fields=(
        "reference_id", "canonical_trajectory_id", "attempt_id", "dataset", "seed", "arm",
        "readout", "endpoint", "split", "epoch", "source_commit", "source_hash", "config_hash",
        "data_hash", "preprocess_hash", "row_population_hash", "selection_protocol", "model_hash",
        "numeric_source", "numeric_sha256", "numeric_available", "predictions_available",
        "ram_receipt_id", "ram_get_verified", "optimizer_available", "shadow_complete",
        "reuse_status", "rebuild_reason", "logical_trajectory_count", "physical_run_count",
    ))
    return write_json(status, {
        "study_id": STUDY, "status": "COMPLETE", "parent_live_reuse": [],
        "parent_state": parent_live,
        "conditional_sources": {
            "N": "VERIFIED",
            "HyperGate": "VERIFIED" if source["published_source_verified"] else "UNRESOLVED_IDENTITY",
            "Gaussian": "VERIFIED" if source["gaussian_source_verified"] else "UNRESOLVED_IDENTITY",
        },
        "d0": d0, "published_numeric_slots": sum(slot["status"] == "NUMERIC_REUSE" for slot in roster["slots"]),
    })


def check_review(root):
    from . import sources

    source = load(root / "SOURCE_MANIFEST.json")
    review = load(root / "REVIEW_VERDICT.json")
    if (review.get("study_id") != STUDY or review.get("source_hash") != source["source_hash"]
            or review.get("design_sha256") != file_hash(root / "DESIGN_SPECIFICATION.md")
            or (review.get("verdict") != "PASS" and not (
                review.get("verdict") == "PASS WITH FIXES" and review.get("blocking_issues_corrected") is True))):
        raise PermissionError("accepted independent review for the current sources and design is required")
    sources.ensure_shared_sources()
    for entry in source["files"]:
        if file_hash(entry["actual_path"]) != entry["sha256"]:
            raise ValueError(f"reviewed implementation identity changed: {entry['actual_path']}")
    return review


def check_conditionals(root, queue):
    modules = conditional_modules(queue)
    if not modules:
        return None
    from . import sources
    from .kernel_checks import conditional_contracts

    source = load(root / "SOURCE_MANIFEST.json")
    identity = dict(study_id=STUDY, stage=queue["stage"], queue_hash=queue["queue_hash"],
                    source_hash=source["source_hash"])
    path = root / queue["stage"] / "CONDITIONAL_CHECKS.json"
    results = {}
    module = None
    gaussian_identity = {}
    try:
        if any(job["source_hash"] != source["source_hash"] for job in queue["jobs"]):
            raise ValueError("conditional queue jobs differ from the audited source")
        prior = load(path) if path.exists() else None
        if prior is not None and (any(prior.get(key) != value for key, value in identity.items())
                                  or prior.get("status") != "PASS"
                                  or set(prior["modules"]) != set(modules)):
            raise ValueError("conditional check receipt differs from the current queue or source")
        for module in modules:
            expected_source = {"source_hash": source["source_hash"]}
            if module == "Gaussian":
                gaussian = source["gaussian_source"]
                commit = source["published_commit"]
                gaussian_identity = {"gaussian_commit": commit, "gaussian_source_hash": gaussian["source_hash"]}
                current = sources.audit_gaussian_source(ROOT.parent / "phase0-3_git", commit)
                gaussian_identity["gaussian_source_status"] = current["status"]
                if (current["status"] != "VERIFIED" or not current["files"]
                        or gaussian["status"] not in ("VERIFIED", "UNRESOLVED_IDENTITY")
                        or (gaussian["status"] == "VERIFIED" and gaussian != current)):
                    raise ValueError(f"Gaussian source does not match its pinned closure: {current['blocker']}")
                gaussian_identity["gaussian_source_hash"] = current["source_hash"]
                expected_source["gaussian_source_hash"] = current["source_hash"]
            report = conditional_contracts(module, root) if prior is None else prior["modules"][module]
            if (report["result"] != "PASS" or "support" not in report
                    or any(report["source"].get(key) != value for key, value in expected_source.items())):
                raise ValueError(f"conditional check did not pass on the audited source: {module}")
            results[module] = report
        if prior is None:
            write_json(path, identity | {"status": "PASS", "modules": results}, immutable=True)
    except Exception as exc:
        failure = identity | gaussian_identity | {"status": "TECHNICAL_PAUSED", "module": module, "modules": results,
                              "blocker": f"{type(exc).__name__}: {exc}"}
        failed_path = write_json(path.with_name("CONDITIONAL_CHECKS_FAILURE.json"), failure)
        return failure | {"artifact": str(failed_path)}
    return None


def policy_hash(root, *, require_permission):
    policy = load(root / TEST_POLICY)
    if (policy.get("study_id") != STUDY or policy.get("holdout_identity") != "HISTORICALLY_EXPOSED"
            or policy.get("global_freeze_before_new_test") is not True
            or policy.get("actual_selected_ram_get_hash_required") is not True
            or policy.get("honor_existing_active_scope_stop_freeze_ram_gates") is not True
            or policy.get("new_test_endpoints") != ["selected"]
            or policy.get("adaptive_changes_after_test") is not False):
        raise PermissionError("invalid current-scope Test policy")
    if require_permission and (policy.get("current_task_evaluation_allowed") is not True
                               or not policy.get("authorization_source") or not policy.get("authorization_scope")):
        raise PermissionError("TEST_PENDING_AUTHORIZATION")
    return file_hash(root / TEST_POLICY)


def _selected(supervisor, point):
    from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.protocol import state_digest

    actual = supervisor.selected_get(point["job_key"], point["readout"])
    identity = actual["identity"]
    expected = {
        "job_key": point["job_key"], "readout": point["readout"], "epoch": point["epoch"],
        "config_hash": point["config_hash"], "data_hash": point["data_hash"],
        "source_hash": point["source_hash"], "group_hash": point["group_hash"],
    }
    if point.get("buffers_hash"):
        expected["buffers_hash"] = point["buffers_hash"]
    seen = dict(identity)
    seen.setdefault("group_hash", seen.get("scope_hash"))
    if (actual["model_hash"] != point["model_hash"]
            or any(seen.get(key) != value for key, value in expected.items())):
        raise ValueError(f"selected RAM get/hash mismatch: {point['job_key']}/{point['readout']}")
    if "model_state" not in actual:
        raise ValueError("selected RAM returned no model state")
    if state_digest(actual["model_state"]) != actual["model_hash"]:
        raise ValueError("selected RAM model bytes differ from the reported hash")
    return actual


def selected_device(identity):
    import torch

    uuid = identity["gpu_uuid"].removeprefix("GPU-").lower()
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        if str(properties.uuid).removeprefix("GPU-").lower() == uuid:
            return f"cuda:{index}"
    raise ValueError(f"selected trajectory GPU is unavailable: {identity['gpu_uuid']}")


def retained_jobs(root, stages):
    decisions = stage_decision(root, "S2")
    keep = {(data, arm) for choice in decisions["family_representatives"].values()
            for data, arm in choice["original_by_dataset"].items()}
    return [job for stage in stages for job in read_queue(root, stage)["jobs"]
            if stage != "S2" or (job["dataset"], job["arm"]) in keep]


def freeze(root, supervisor):
    from .protocol import RunSpec

    assert_open(root)
    review = check_review(root)
    authorization = policy_hash(root, require_permission=False)
    transfer = root / "S3d" / "RUN_RECEIPT.json"
    if transfer.exists() and load(transfer).get("conditional_pause"):
        raise ValueError("S3d remains TECHNICAL_PAUSED pending Gaussian acceptance and completion")
    stages = STAGES if any(pending_baseline_repairs(root).values()) else STAGES[:-1]
    decisions = stage_decision(root, stages[-1])
    rows = {}
    for stage in stages:
        stage_decision(root, stage)
        accepted, _ = analysis.accepted_stage(root, stage)
        rows.update(accepted)
    jobs = retained_jobs(root, stages)
    frozen = []
    selected = set()
    for job in jobs:
        spec = RunSpec.from_dict(job)
        for readout in spec.readouts:
            row = rows[spec.dataset, spec.seed, spec.arm, readout, "selected"]
            point = {"job_key": spec.job_key, **{name: row[name] for name in (
                "readout", "epoch", "model_hash", "config_hash", "data_hash", "source_hash", "group_hash",
                "scope_hash", "buffers_hash",
            )}}
            actual = _selected(supervisor, point)
            point.update(spec=job, actual_get_hash=actual["model_hash"], device=actual["identity"].get("device"),
                         gpu_uuid=actual["identity"]["gpu_uuid"], ram_packet_hash=actual["ram_hash"],
                         owner_generation=actual["identity"]["owner_generation"])
            frozen.append(point)
            selected.add((spec.dataset, spec.seed, spec.arm, readout))
            del actual
    roster = load(root / "PUBLISHED_ROSTER.json")
    for slot in roster["slots"]:
        if slot["status"] == "NUMERIC_REUSE":
            continue
        if slot["model"] == "M3oE":
            alternatives = ("B0",)
        elif slot["model"] == "HyperGate":
            alternatives = ("HG_B0", "REPAIR_HG")
        else:
            alternatives = ("REPAIR_AUTOIFS",)
        slot["arm"] = next(arm for arm in alternatives
                           if (slot["dataset"], slot["seed"], arm, "online") in selected)
        slot["status"] = "CURRENT_FROZEN_SELECTED"
    identity_files = (
        "SOURCE_MANIFEST.json", "DATA_MANIFEST.json", "PARAMETER_SCOPE_MANIFEST.json", "PUBLISHED_ROSTER.json",
        "REVIEW_VERDICT.json", "DESIGN_SPECIFICATION.md", "STOP_CONTROLS.json",
        *(f"{stage}/DECISIONS.json" for stage in stages),
    )
    comparisons = [row for row in analysis.comparison_contract("TEST", decisions, analysis.EndpointTable(rows))
                   if all((row[0], seed, arm, readout) in selected
                          for arm, readout, _ in row[3] for seed in SEEDS)]
    result = {
        "study_id": STUDY, "policy_hash": authorization, "holdout_identity": "HISTORICALLY_EXPOSED",
        "adaptive_changes_after_test": False, "readout_endpoint": "selected",
        "identity_files": {name: file_hash(root / name) for name in identity_files},
        "algorithm_plan_sha256": file_hash(PLAN),
        "seeds": list(SEEDS), "review_verdict": review["verdict"],
        "published_roster": roster, "stage_gates": decisions["stage_gates"] | (
            {} if "BASELINE_REPAIR" in stages else {"BASELINE_REPAIR": {"status": "NOT_TRIGGERED"}}
        ),
        "triggered_arms": decisions["triggered_arms"], "not_triggered_arms": decisions["not_triggered_arms"],
        "Cstar": decisions["Cstar"], "family_representatives": decisions["family_representatives"],
        "endpoints": frozen,
        "comparison_contract": comparisons,
        "protection_references": list(read_plan()["selection"]["cell_protection_references"]),
    }
    result["freeze_hash"] = digest(result)
    path = write_json(root / FINAL_FREEZE, result, immutable=True)
    supervisor.release_selected(sorted({point["job_key"] for point in frozen}))
    return path


def _pinned_test_rows(frozen):
    roster = frozen["published_roster"]
    origin = roster["source"]
    raw, source = checked_blob(origin["path"], origin["commit"], origin["git_blob"])
    if source != origin:
        raise ValueError("frozen published endpoint source changed")
    indices = {row["endpoint_id"]: row for row in csv_content(raw)
               if row["split"] == "test" and row["role"] == "PUBLISHED_BASELINE" and row["checkpoint"] == "selected"}
    for slot in roster["slots"]:
        if slot["status"] != "NUMERIC_REUSE":
            continue
        row = indices[slot["test_endpoint_id"]]
        if (row["model_hash"] != slot["model_hash"] or row["selection_id"] != slot["selection_id"]
                or row["config_hash"] != slot["config_hash"] or row["dataset"] != slot["dataset"]
                or int(row["seed"]) != slot["seed"]):
            raise ValueError("pinned published Test endpoint disagrees with freeze")
        yield {**row, "config_id": slot["model"], "readout": "online", "endpoint": "selected",
               "source_type": "NUMERIC_REUSE", "data_hash": slot["data_hash"]}


def _test_outputs(root, rows, frozen):
    analysis.write_csv(root / "TEST_ENDPOINTS.csv", rows, immutable=True)
    normalized = analysis.normalize_endpoints(rows, "test")
    for slot in frozen["published_roster"]["slots"]:
        if slot["status"] == "CURRENT_FROZEN_SELECTED":
            key = (slot["dataset"], slot["seed"], slot["arm"], "online", "selected")
            normalized[(slot["dataset"], slot["seed"], slot["model"], "online", "selected")] = normalized[key]
    table = analysis.EndpointTable(normalized)
    comparisons = [
        (dataset, readout, name, terms, ("selected",))
        for dataset, readout, name, terms, _ in frozen["comparison_contract"]
    ]
    candidates = {name: values["original_by_dataset"] for name, values in frozen["family_representatives"].items()}
    if frozen["Cstar"]:
        candidates["Cstar"] = frozen["Cstar"]["arms"]
    result = {}
    for candidate_id, candidate in candidates.items():
        result[candidate_id] = {}
        for readout in READOUTS:
            result[candidate_id][readout] = analysis.strong_method_gates(
                table, candidate, readout, frozen["published_roster"]["models"],
            )
            for data in DATASETS:
                for baseline in frozen["published_roster"]["models"]:
                    comparisons.append((data, readout, f"{candidate[data]}_vs_{baseline}",
                                        [(candidate[data], readout, 1), (baseline, "online", -1)], ("selected",)))
    write_json(root / "TEST_METHOD_GATES.json", result, immutable=True)
    analysis.write_comparisons(root / "test", table, comparisons)


def _check_test_stop(root):
    from .scheduler import stop_status

    control = stop_status(root)
    if control is not None:
        raise PermissionError(f"current study stop control: {control[0]}")


def evaluate_test(root, supervisor):
    if (root / TEST_ACCESS).exists():
        raise PermissionError("a frozen Test access was already claimed")
    check_review(root)
    _check_test_stop(root)
    frozen = load(root / FINAL_FREEZE)
    from . import objectives, sources
    from .protocol import RunSpec

    checksum = digest({key: value for key, value in frozen.items() if key != "freeze_hash"})
    authorization = policy_hash(root, require_permission=True)
    if (checksum != frozen["freeze_hash"] or frozen["study_id"] != STUDY
            or frozen["policy_hash"] != authorization or frozen["algorithm_plan_sha256"] != file_hash(PLAN)
            or frozen["readout_endpoint"] != "selected" or frozen["adaptive_changes_after_test"] is not False):
        raise PermissionError("frozen Test identity verification failed")
    for name, expected in frozen["identity_files"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"frozen identity file changed: {name}")
    for point in frozen["endpoints"]:
        actual = _selected(supervisor, point)
        if actual["model_hash"] != point["actual_get_hash"] or actual["ram_hash"] != point["ram_packet_hash"]:
            raise ValueError("current selected RAM differs from freeze")
        if (actual["identity"].get("gpu_uuid") != point["gpu_uuid"]
                or actual["identity"].get("owner_generation") != point["owner_generation"]):
            raise ValueError("selected trajectory device or RAM owner identity changed after freeze")
        selected_device(actual["identity"])
        del actual
    claim = {"study_id": STUDY, "access_id": uuid4().hex, "claimed_at": now(), "freeze_hash": checksum,
             "policy_hash": authorization, "status": "TEST_ACCESS_CLAIMED", "split": "test",
             "actual_selected_ram_get_hash_verified": True, "new_test_endpoints": ["selected"]}
    _check_test_stop(root)
    write_json(root / TEST_ACCESS, claim, immutable=True)
    try:
        _check_test_stop(root)
        rows = list(_pinned_test_rows(frozen))
        cache = root / "train_validation_inputs"
        for dataset in DATASETS:
            _check_test_stop(root)
            split = sources.load_authorized_test(dataset, claim | {"dataset": dataset}, artifact_root=root)
            test_population_hash = sources.source.split_row_fingerprint(split.x, split.y, split.row_ids)
            bundle = sources.load_train_validation(dataset, cache)
            for point in frozen["endpoints"]:
                spec = RunSpec.from_dict(point["spec"])
                if spec.dataset != dataset:
                    continue
                state = _selected(supervisor, point)
                device = selected_device(state["identity"])
                model, optimizer, ema = objectives.build_training(spec, bundle, device)
                del optimizer, ema
                model.load_state_dict(state["model_state"], strict=True)
                if spec.backbone == "autoifs":
                    objectives.restore_inference_meta(model, state["identity"]["model_meta"])
                metrics, work = objectives.evaluate(spec, model, split.x, split.y, device)
                if metrics["macro_auc"] is None or any(metrics[f"{cell}_auc"] is None for cell in CELLS):
                    raise ValueError(f"frozen Test lacks complete support: {spec.job_key}")
                rows.append(dict(dataset=dataset, seed=spec.seed, config_id=spec.arm,
                                 logical_run_id=spec.job_key, split="test", readout=point["readout"],
                                 endpoint="selected", selected_epoch=point["epoch"], model_hash=point["model_hash"],
                                 config_hash=spec.config_hash, source_hash=spec.source_hash, data_hash=spec.data_hash,
                                 group_hash=spec.group_hash, buffers_hash=point["buffers_hash"],
                                 source_type="ACTUAL_SELECTED_RAM", evaluation_forwards=work["forwards"],
                                 evaluation_rows=work["rows"], evaluation_seconds=work["seconds"],
                                 test_population_hash=test_population_hash, **metrics))
                del model, state
        _test_outputs(root, rows, frozen)
    except Exception as exc:
        write_json(root / "TEST_RECEIPT.json", claim | {"status": "EVALUATION_FAILED",
                                                       "failure": f"{type(exc).__name__}: {exc}"})
        raise
    return write_json(root / "TEST_RECEIPT.json", claim | {
        "status": "COMPLETE", "completed_at": now(), "rows": len(rows),
        "endpoints_sha256": file_hash(root / "TEST_ENDPOINTS.csv"),
    }, immutable=True)


def command_parser():
    parser = argparse.ArgumentParser(prog=f"python -m experiments.{STUDY}.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").add_argument("--plan", type=Path, default=PLAN)
    commands.add_parser("check").add_argument("--scope", choices=("scientific_contracts",), required=True)
    for command in ("materialize", "run", "analyze"):
        stage = commands.add_parser(command)
        stage.add_argument("--stage", choices=STAGES, required=True)
        if command == "materialize":
            stage.add_argument("--decisions", type=Path)
    commands.add_parser("freeze")
    commands.add_parser("evaluate").add_argument("--split", choices=("test",), required=True)
    commands.add_parser("supervise")
    return parser


def execute(root, arguments, supervisor=None):
    command = arguments.command
    if command == "audit":
        return {"status": "COMPLETE", "artifact": str(audit(root, arguments.plan))}
    if command == "check":
        from . import sources

        sources.ensure_shared_sources()
        from .kernel_checks import algebra_checks, gate_check, optimizer_checks
        from .tests.orchestration_check import check

        orchestration = check()
        write_json(root / "ORCHESTRATION_CHECKS.json", orchestration)
        path = write_json(root / "IMPLEMENTATION_CHECKS.json", {
            "study_id": STUDY, "status": "PASS", "orchestration": orchestration,
            "algebra": algebra_checks(), "optimizer": optimizer_checks(), "test_gate": gate_check(),
        })
        return {"status": "COMPLETE", "artifact": str(path)}
    if command == "materialize":
        result = materialize(root, arguments.stage, arguments.decisions)
        if result is None:
            return {"status": "NOT_TRIGGERED", "jobs": 0, "reason": "PUBLISHED_ROSTER_MATCHED"}
        return {"status": "COMPLETE", "jobs": len(result["jobs"]), "queue_hash": result["queue_hash"],
                "artifact": str(root / arguments.stage / "QUEUE.json")}
    if command == "analyze":
        result = analysis.analyze(root, arguments.stage)
        directory = root / arguments.stage
        if result["status"] == "TECHNICAL_PAUSED":
            directory /= "clean"
        return {"status": result["status"], "artifact": str(directory / "DECISIONS.json")}
    if supervisor is None:
        raise RuntimeError(f"{command} requires the existing foreground supervise process")
    if command == "run":
        assert_open(root)
        if (root / arguments.stage / "DECISIONS.json").exists():
            raise ValueError(f"{arguments.stage} already has an accepted decision")
        check_review(root)
        from .scheduler import stop_status

        control = stop_status(root)
        if control is not None:
            return {"status": control[0], "control": control[1]}
        queue = read_queue(root, arguments.stage)
        blocker = check_conditionals(root, queue)
        control = stop_status(root)
        if control is not None:
            return {"status": control[0], "control": control[1]}
        if blocker is not None and not (arguments.stage == "S3d" and blocker["module"] == "Gaussian"):
            return blocker
        if arguments.stage in STAGES[2:]:
            prior = STAGES[:STAGES.index(arguments.stage)]
            keep = retained_jobs(root, prior) + queue["jobs"]
            supervisor.release_selected(sorted({digest(job) for job in keep}))
        return supervisor.run(queue, conditional_pause=blocker) if blocker else supervisor.run(queue)
    if command == "freeze":
        return {"status": "COMPLETE", "artifact": str(freeze(root, supervisor))}
    if command == "evaluate":
        return {"status": "COMPLETE", "artifact": str(evaluate_test(root, supervisor))}
    raise ValueError(f"unsupported foreground command: {command}")


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def supervise(root, parser):
    from . import sources

    sources.ensure_shared_sources()
    from .runner import Supervisor

    supervisor = Supervisor(root, emit=emit)
    try:
        emit({"status": "READY", "study_id": STUDY})
        for line in sys.stdin:
            if not line.strip():
                continue
            command_id = uuid4().hex
            try:
                arguments = parser.parse_args(shlex.split(line))
                if arguments.command not in {"run", "freeze", "evaluate"}:
                    raise ValueError("foreground commands are run, freeze, and evaluate")
                emit({"command_id": command_id, "command": arguments.command, **execute(root, arguments, supervisor)})
            except (Exception, SystemExit) as exc:
                emit({"command_id": command_id, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        supervisor.close()


def main():
    parser = command_parser()
    arguments = parser.parse_args()
    read_plan()
    if arguments.command == "supervise":
        supervise(ARTIFACTS, parser)
    else:
        emit(execute(ARTIFACTS, arguments))


if __name__ == "__main__":
    main()
