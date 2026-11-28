"""C1 roster, frozen source identities and exact historical A1 control links."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from experiments.conditional_history_shared_update_20261122.protocol import write

from . import c0_audit, scheduler
from .a_step import a_config
from .c_step import prefix_step
from .macro_pair_sampler import PAIR_COUNT, MacroPairSampler, ordered_ids_hash
from .memory_bank import BANK_CAP, CANDIDATE_CAP, CLIP, TEACHER_EPOCH, TEMPERATURE, load_candidates
from .protocol import (DATASETS, GA31_CALLBACK_SHA256, GA31_OBJECTIVE, MAIN, PACKAGE, READOUTS, ROOT, SEEDS,
                       STUDY, RunSpec, algorithm_digest as original_algorithm_digest, digest,
                       load_train_validation, read, sha, v3, verify_sources)
from .ram_state import Journal, lock

REPORTS = ROOT / "research_reports" / STUDY
SNAPSHOT_RECEIPT = "A_B_SOURCE_SNAPSHOT.json"
V3_MANIFEST_PATH = "research_reports/2026-11-02_mdmt_information_momentum_v3/IMPLEMENTATION_SOURCE_MANIFEST.json"
NUMERICAL = ("a_step", "macro_pair_sampler", "c0_audit", "memory_bank", "c_step")
C0_MODELS = {"B0": "C0", "P0": "C1"}


def c1_jobs():
    return [
        RunSpec("C1", dataset, seed, arm, a_config("P0", dataset),
                dict(memory_mode=mode, teacher_epoch=TEACHER_EPOCH, unique_pair_capacity=BANK_CAP,
                     capacity_unit="active_sampling_condition", candidate_capacity=CANDIDATE_CAP,
                     pair_draws=PAIR_COUNT, margin_clip=[-CLIP, CLIP], temperature=TEMPERATURE))
        for dataset in DATASETS for seed in SEEDS for arm, mode in (("C2", "random"), ("C3", "correct"))
    ]


def algorithm_digest(job, source):
    return digest(dict(main=MAIN, source_manifest=source["source_manifest"], ga31=source["ga31"],
                       numerical={name: source["runtime"][name] for name in NUMERICAL},
                       a1_warmup=source["a1_warmup"], config_hash=job.config_hash))


def _guard(artifacts):
    if marker := scheduler.stop_status(artifacts):
        raise ValueError(f"{marker['status']} at {marker['path']}")


def _source_snapshot(artifacts, a_roster, b_roster):
    receipt_path = artifacts / SNAPSHOT_RECEIPT
    receipt = read(receipt_path)
    commit = receipt["commit"]
    c0_audit.require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "Invalid A1/B1 archive commit ID")
    git = ["git", "-C", str(ROOT)]
    subprocess.run([*git, "merge-base", "--is-ancestor", MAIN, commit], check=True)

    expected = {}
    for stage, roster in (("A1", a_roster), ("B1", b_roster)):
        source = roster["source"]
        c0_audit.require(roster["study"] == STUDY and roster["stage"] == source["stage"] == stage
                         and source["main"] == MAIN and roster["test_status"] == "SEALED"
                         and source["source_hash"] == digest({k: v for k, v in source.items() if k != "source_hash"})
                         and source["ga31"]["callback_sha256"] == GA31_CALLBACK_SHA256,
                         f"Original {stage} roster source identity mismatch")
        files = {f"experiments/{STUDY}/{name}.py": value for name, value in source["runtime"].items()}
        files.update({f"research_reports/{STUDY}/{arm}_EXECUTION_DESIGN.md": value
                      for arm, value in source["design"].items()})
        files.update({
            f"artifacts/{STUDY}/{stage}/ROSTER.json": sha(artifacts / stage / "ROSTER.json"),
            f"artifacts/{STUDY}/SOURCE_AUDIT.json": source["source_audit"],
            f"artifacts/{STUDY}/DATA_AUDIT.json": source["data_audit"],
            V3_MANIFEST_PATH: source["source_manifest"],
            str((v3.HERE / GA31_OBJECTIVE).relative_to(ROOT)): GA31_CALLBACK_SHA256,
        })
        c0_audit.require(all(expected.get(path, value) == value for path, value in files.items()),
                         "A1/B1 frozen source files disagree")
        expected.update(files)
    for path, value in expected.items():
        c0_audit.require(receipt["files"].get(path) == value
                         and hashlib.sha256(subprocess.check_output([*git, "show", f"{commit}:{path}"])).hexdigest() == value,
                         f"Archived A1/B1 source bytes differ: {path}")
    return dict(commit=commit, receipt_sha256=sha(receipt_path), files=expected)


def verify_gates(artifacts, *, expected=None):
    c0 = artifacts / "C0"
    paths = (c0 / "STATUS.json", c0 / "MANIFEST.json", c0 / "SUMMARY.json", c0 / "decisions.json",
             REPORTS / "C0_ANALYSIS.md", REPORTS / "C0_RESEARCH_REVIEW.md")
    checks = {path.name: sha(path) for path in paths}
    if expected is not None:
        c0_audit.require(checks == expected, "Frozen C0 scientific gate or manifest bytes changed")
        return checks
    status, manifest, summary, decisions = (read(path) for path in paths[:4])
    inventory = {f"A1-{d}-{a}-seed{s}": dict(dataset=d, seed=s, arm=a, state="COMPLETE", reason=None)
                 for d in DATASETS for s in SEEDS for a in C0_MODELS}
    c0_audit.require(status["status"] == "COMPLETE" and status["complete_input_accounting"] is True
                     and status["stop"] is None and status["manifest_sha256"] == checks["MANIFEST.json"]
                     and len(status["completed_jobs"]) == len(inventory) == 20
                     and set(status["completed_jobs"]) == set(inventory) == set(manifest["jobs"])
                     and status["input_inventory"] == manifest["inventory"] == inventory
                     and summary["complete_input_accounting"] is True
                     and summary["expected_trajectories"] == summary["observed_trajectories"] == 20
                     and manifest["study"] == STUDY and manifest["stage"] == "C0"
                     and manifest["source_sha256"] == sha(PACKAGE / "c0_audit.py")
                     and manifest["design_sha256"] == sha(REPORTS / "C0_EXECUTION_DESIGN.md")
                     and all(type(row["test_accesses"]) is int and row["test_accesses"] == 0
                             for row in (status, manifest, decisions)),
                     "C0 requires intact complete 20-job source and input accounting")
    bank = decisions["C1_bank"]
    c0_audit.require(decisions["status"] == "COMPLETE"
                     and decisions["C1"] == "SCIENTIFIC_GATE_PASS_CAPACITY_APPROVED"
                     and bank["unique_pair_capacity"] == BANK_CAP
                     and bank["capacity_unit"] == "active_sampling_condition"
                     and bank["pair_draws_per_step"] == PAIR_COUNT
                     and "**裁决 PASS。" in paths[-1].read_text(),
                     "C1 requires the passing independent C0 scientific review and approved capacity")
    return checks


def _current_source(artifacts, a_roster, snapshot):
    shared = verify_sources(artifacts, "B1")
    names = (*NUMERICAL, "c1_protocol")
    runtime = dict(shared["runtime"], **{name: sha(PACKAGE / f"{name}.py") for name in names})
    design = dict(shared["design"], **{name: sha(REPORTS / f"{name}_EXECUTION_DESIGN.md") for name in ("A", "C0", "C1")})
    a1 = a_roster["source"]
    c0_audit.require(all(shared[k] == a1[k] for k in ("main", "source_audit", "source_manifest", "data_audit", "ga31"))
                     and runtime["macro_pair_sampler"] == a1["runtime"]["macro_pair_sampler"]
                     and design["A"] == a1["design"]["A"], "C1 foundation differs from the original A1 source")
    identity = dict({k: v for k, v in shared.items() if k not in ("stage", "source_hash", "runtime", "design")},
                    stage="C1", runtime=runtime, design=design, historical_source_snapshot=snapshot,
                    a1_warmup={name: a1["runtime"][name] for name in ("a_step", "macro_pair_sampler")})
    return dict(identity, source_hash=digest(identity))


def _registered_datasets(artifacts):
    data = read(artifacts / "DATA_AUDIT.json")["datasets"]
    audit = read(artifacts / "SOURCE_AUDIT.json")
    return {name: dict(data_hash=row["data_hash"], raw_sha256=row["raw_sha256"],
                       train_row_hash=row["split_row_id_hashes"]["train"],
                       validation_row_hash=row["split_row_id_hashes"]["validation"],
                       train_domain_counts=row["train_domain_counts"],
                       scope_hash=audit["canonical_optimizer_and_ema"][name]["parameter_manifest_hash"])
            for name, row in data.items()}


def _record(job, source):
    return dict(job.record(), config_hash=job.config_hash, algorithm_hash=algorithm_digest(job, source))


def job_journal(artifacts, roster, job):
    data = roster["datasets"][job.dataset]
    binding = dict(study=STUDY, stage=job.stage, dataset=job.dataset, seed=job.seed, arm=job.arm,
                   source_hash=roster["source"]["source_hash"], config_hash=job.config_hash,
                   algorithm_hash=algorithm_digest(job, roster["source"]), data_hash=data["data_hash"],
                   scope_hash=data["scope_hash"], job_key=job.key)
    return Journal(artifacts / job.stage / "runs" / job.key, binding)


def _endpoint(result, checkpoint, readout):
    found = [row for row in result["endpoints"]
             if row["checkpoint"] == checkpoint and row["readout"] == readout]
    c0_audit.require(len(found) == 1, f"Missing or duplicate A1 endpoint {checkpoint}/{readout}")
    return found[0]


def _receipt(loaded, archived):
    return loaded == {k: v for k, v in archived.items() if k != "read_verify_decode_seconds"}


def _validate_endpoint(row, journal, scores, readout, checkpoint, epoch, model_hash):
    c0_audit.require(isinstance(model_hash, str) and re.fullmatch(r"[0-9a-f]{64}", model_hash) is not None
                     and row == dict(journal.binding, split="validation", readout=readout, checkpoint=checkpoint,
                                     epoch=epoch, model_hash=model_hash, **scores),
                     f"A1 {readout}/{checkpoint} differs from committed Validation metrics or source")


def _control(artifacts, roster, job, audit_job, cache, published, *, bundle=None):
    source = roster["source"]
    registered = [record for record in roster["jobs"] if record["key"] == job.key]
    c0_audit.require(registered == [dict(job.record(), config_hash=job.config_hash,
                                       algorithm_hash=original_algorithm_digest(job, source))],
                     f"Original A1 full config or algorithm identity differs: {job.key}")
    directory = artifacts / "A1" / "runs" / job.key
    c0_audit.require((directory / "JOURNAL.lock").is_file(), "Completed A1 job has no existing journal lock")
    completion_bytes = (directory / "COMPLETE_VERIFIED.json").read_bytes()
    journal = c0_audit.runner.job_journal(artifacts, roster, job)
    journal_bytes = journal.path.read_bytes()
    result, metadata = json.loads(completion_bytes), json.loads(journal_bytes)
    c0_audit.require(result["job"] == job.record()
                     and result["binding"] == metadata["binding"] == journal.binding == audit_job["binding"]
                     and type(result["test_accesses"]) is int and result["test_accesses"] == 0
                     and metadata["complete"] == dict(result=result, required=result["compact_models"])
                     and all(f"fixed_ema-{label}" in metadata["compact"] for label in c0_audit.SNAPSHOTS),
                     f"Original verified A1 completion or journal binding differs: {job.key}")
    n_train = cache["split_rows"]["train"]
    steps = math.ceil(n_train / job.config["batch_size"])
    prepared = c0_audit.validation_records(journal, metadata, result, n_train, steps)
    epochs = [read(directory / f"epoch_{epoch:02d}.json") for epoch in range(1, 51)]
    c0_audit.require(len(result["endpoints"]) == len(READOUTS) * 3,
                     f"Original A1 completion requires six Validation endpoints: {job.key}")

    compact, selected = {}, {}
    for readout in READOUTS:
        per_epoch = [row["validation"][readout] for row in epochs]
        for score in per_epoch:
            cells = [score[f"{c}_auc"] for c in c0_audit.CELLS]
            losses = [score[f"{c}_logloss"] for c in c0_audit.CELLS]
            c0_audit.require(all(math.isfinite(v) for v in (*cells, *losses, score["macro_logloss"]))
                             and score["macro_auc"] == float(np.mean(cells)),
                             f"Malformed original A1 {readout} Validation metrics")
        epoch = max(range(1, 51), key=lambda i: per_epoch[i - 1]["macro_auc"])
        key = f"{readout}-selected"
        model_hash = epochs[-1]["selected_model_hashes"][readout]
        expected = dict(kind="compact_model", readout=readout, epoch=epoch, model_hash=model_hash)
        c0_audit.require(result["compact_models"][key] == expected
                         and result["selected"][readout] == epochs[-1]["selected_epochs"][readout] == epoch
                         and epochs[-1]["selected_metrics"][readout] == per_epoch[epoch - 1],
                         f"Original A1 {readout} selection differs from earliest strict maximum")
        row = selected[readout] = _endpoint(result, "selected", readout)
        _validate_endpoint(row, journal, per_epoch[epoch - 1], readout, "selected", epoch, model_hash)
        compact[key] = _compact(journal, metadata, key, dict(expected, global_step=epoch * steps),
                                verify_live=bundle is not None)
        for checkpoint_epoch in (TEACHER_EPOCH, 50):
            checkpoint = f"E{checkpoint_epoch}"
            record = epochs[checkpoint_epoch - 1]
            row = _endpoint(result, checkpoint, readout)
            model_hash = _endpoint(dict(endpoints=published), checkpoint, readout)["model_hash"]
            if readout == "fixed_ema" and checkpoint_epoch == TEACHER_EPOCH:
                model_hash = prepared["snapshots"]["E10"]["expected"]["model_hash"]
            if record["selected_epochs"][readout] == checkpoint_epoch:
                c0_audit.require(model_hash == record["selected_model_hashes"][readout],
                                 f"A1 {readout}/{checkpoint} differs from the committed same-epoch model")
            _validate_endpoint(row, journal, record["validation"][readout], readout, checkpoint,
                               checkpoint_epoch, model_hash)

    c0_audit.require(result["endpoints"] == published,
                     f"Original A1 published Validation endpoints differ: {job.key}")
    c0_audit.require(_receipt(compact["fixed_ema-selected"], audit_job["ram"]["selected"])
                     and selected["fixed_ema"]["epoch"] == audit_job["selected_epoch"],
                     f"C0 selected EMA receipt differs: {job.key}")
    e10_key = "fixed_ema-E10"
    e10 = _compact(journal, metadata, e10_key, prepared["snapshots"]["E10"]["expected"],
                   verify_live=bundle is not None and job.arm == "P0", fp32=True)
    c0_audit.require(_receipt(e10, audit_job["ram"]["E10"]),
                     f"Original E10 fixed EMA receipt differs: {job.key}")
    link = dict(logical_arm=C0_MODELS[job.arm], original_job=job.record(), original_binding=journal.binding,
                completion_sha256=hashlib.sha256(completion_bytes).hexdigest(),
                journal_sha256=hashlib.sha256(journal_bytes).hexdigest(),
                final_epoch_receipt=epochs[-1]["ram_receipt"], selected=selected, readbacks=compact,
                validation_endpoints=result["endpoints"])
    if job.arm == "B0":
        return link, None
    compact[e10_key] = e10
    teacher = dict(original_job_key=job.key, epoch=TEACHER_EPOCH, global_step=steps * TEACHER_EPOCH,
                   model_hash=e10["identity"]["model_hash"],
                   online_hash=_endpoint(result, "E10", "online")["model_hash"], original_ram_sha256=e10["sha256"])
    prefix = _prefix(job, journal, epochs, steps, n_train, cache["split_row_fingerprints"]["train"], bundle=bundle)
    return link, dict(teacher=teacher, prefix=prefix)


def _compact(journal, metadata, key, fields, *, verify_live, fp32=False):
    receipt = metadata["compact"][key]
    expected = dict(journal._expected(metadata, receipt), key=key, **fields)
    c0_audit.require(all(receipt["identity"].get(k) == value for k, value in expected.items()),
                     f"Original A1 compact receipt differs: {key}")
    if verify_live:
        model, verified = journal.read_model(key, expected=fields)
        c0_audit.require(verified == receipt and
                         (not fp32 or all(not t.is_floating_point() or t.dtype == torch.float32 for t in model.values())),
                         f"Original A1 compact readback differs: {key}")
    return receipt


def _prefix(job, journal, epochs, steps, n_train, fingerprint, *, bundle=None):
    """Bind the original committed P0 row and current-pair receipts for every warmup step."""
    writer_by_epoch = {epoch: epochs[epoch - 1]["ram_receipt"]["identity"]["attempt"]
                       for epoch in range(1, TEACHER_EPOCH + 1)}
    by_step = {}
    traces, inits = {}, set()
    for writer in sorted(set(writer_by_epoch.values())):
        root = journal.directory / "attempts" / writer
        identity_path, trace_path = root / "IDENTITY.json", root / "batches.jsonl"
        identity = read(identity_path)
        inits.add(identity["init_model_hash"])
        data = identity["data"]
        c0_audit.require(data["data_hash"] == journal.binding["data_hash"]
                         and data["split_row_fingerprints"]["train"] == fingerprint
                         and (bundle is None or data == bundle.identity), f"A1 P0 prefix data differs: {job.key}")
        with trace_path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                c = row["context"]
                if writer_by_epoch.get(c["epoch"]) != writer:
                    continue
                s = c["global_step"]
                c0_audit.require(s not in by_step, f"Duplicate committed A1 P0 prefix step {s}")
                by_step[s] = row
        traces[writer] = dict(trace_sha256=sha(trace_path), identity_sha256=sha(identity_path))
    c0_audit.require(len(inits) == 1 and set(by_step) == set(range(1, TEACHER_EPOCH * steps + 1)),
                     f"Missing A1 P0 initialization or warmup receipts: {job.key}")
    loader = torch.Generator().manual_seed(job.seed + 11000) if bundle is not None else None
    batches, orders = [], []
    width = job.config["batch_size"]
    for epoch in range(1, TEACHER_EPOCH + 1):
        epoch_hash = epochs[epoch - 1]["order_hash"]
        if bundle is not None:
            order = torch.randperm(n_train, generator=loader).numpy()
            rows = bundle.train_row_ids[order]
            c0_audit.require(v3.state_digest(rows) == epoch_hash, f"Committed A1 P0 order mismatch E{epoch:02d}")
        orders.append(epoch_hash)
        for batch, offset in enumerate(range(0, n_train, width), 1):
            step = (epoch - 1) * steps + batch
            row = by_step[step]
            count = min(width, n_train - offset)
            diagnostics = row["diagnostics"]
            context = dict(dataset=job.dataset, seed=job.seed, epoch=epoch, global_step=step, batch=batch,
                           last_batch=offset + count == n_train, split_id="original")
            c0_audit.require(all(row["context"].get(k) == v for k, v in context.items())
                             and diagnostics["train_identity"] == fingerprint
                             and diagnostics["ordered_bce_row_hash"] == row["row_id_hash"]
                             and (bundle is None or row["row_id_hash"] == ordered_ids_hash(rows[offset:offset + count]))
                             and row["rows"] == count and row["counts"]["pair_draws"] == PAIR_COUNT
                             and diagnostics["w_over_q"] == diagnostics["w"] / diagnostics["q"],
                             f"Committed A1 P0 batch receipt differs at logical step {step}")
            batches.append(prefix_step(row["row_id_hash"], diagnostics))
    prefix = dict(init_model_hash=inits.pop(), steps_per_epoch=steps, order_hashes=orders, batches=batches, traces=traces)
    return dict(prefix, hash=digest(prefix))


def _verify_archive(path, receipt):
    c0_audit.require(Path(receipt["path"]) == path and path.stat().st_size == receipt["bytes"]
                     and sha(path) == receipt["sha256"], f"Missing or mismatched C0 archive {path}")


def _verified_manifest(artifacts, gates):
    payload = (artifacts / "C0" / "MANIFEST.json").read_bytes()
    c0_audit.require(hashlib.sha256(payload).hexdigest() == gates["MANIFEST.json"],
                     "Frozen C0 manifest bytes changed")
    return json.loads(payload)


def _snapshots(audit_job):
    return [dict(label=label, model_hash=audit_job["ram"][label]["identity"]["model_hash"])
            for label in c0_audit.SNAPSHOTS]


def _validate_seed_inputs(manifest, dataset, seed, inputs):
    key = f"{dataset}_{seed}"
    job_key = f"A1-{dataset}-P0-seed{seed}"
    audit_job = manifest["jobs"][job_key]
    c0_audit.require(inputs["candidate"] == manifest["candidates"][key]
                     and inputs["margins"] == audit_job["margins"],
                     f"C1 seed {key} candidate or P0 margin receipt differs from the frozen C0 manifest")
    e10 = audit_job["ram"]["E10"]
    identity = e10["identity"]
    teacher = dict(original_job_key=job_key, epoch=TEACHER_EPOCH, global_step=identity["global_step"],
                   model_hash=identity["model_hash"], original_ram_sha256=e10["sha256"])
    c0_audit.require(identity["job_key"] == job_key and identity["epoch"] == TEACHER_EPOCH
                     and inputs["snapshots"] == _snapshots(audit_job)
                     and all(inputs["teacher"].get(k) == v for k, v in teacher.items()),
                     f"C1 seed {key} P0 E10 snapshot or teacher identity differs from the frozen C0 manifest")


def _verify_original_roster(artifacts, a_roster, manifest):
    audit_roster = manifest["roster"]
    c0_audit.require(audit_roster["sha256"] == sha(artifacts / "A1" / "ROSTER.json")
                     and audit_roster["source"] == a_roster["source"]
                     and audit_roster["datasets"] == a_roster["datasets"]
                     and audit_roster["jobs"] == [row for row in a_roster["jobs"] if row["arm"] in C0_MODELS],
                     "C0 C1 controls differ from the audited original A1 roster")


def verify_seed_inputs(artifacts, *, expected, dataset, seed, inputs):
    """Bind the records used at bank preparation to the verified original audit and P0 prefix."""
    artifacts = Path(artifacts)
    gates = verify_gates(artifacts, expected=expected)
    manifest = _verified_manifest(artifacts, gates)
    _validate_seed_inputs(manifest, dataset, seed, inputs)
    a_roster = read(artifacts / "A1" / "ROSTER.json")
    _verify_original_roster(artifacts, a_roster, manifest)
    job = RunSpec("A1", dataset, seed, "P0", a_config("P0", dataset), {})
    published = [row for row in read(artifacts / "A1" / "ENDPOINTS.json") if row["job_key"] == job.key]
    _, source = _control(artifacts, a_roster, job, manifest["jobs"][job.key], manifest["caches"][dataset], published)
    _verify_prefix_inputs(inputs, source, job.key)


def _verify_prefix_inputs(inputs, original, key):
    c0_audit.require(inputs.get("teacher") == original["teacher"] and inputs.get("prefix") == original["prefix"],
                     f"C1 teacher or warmup prefix differs from the committed A1 P0 evidence: {key}")


def _collect_inputs(artifacts, a_roster, manifest, *, verify_live=False):
    _verify_original_roster(artifacts, a_roster, manifest)
    published = read(artifacts / "A1" / "ENDPOINTS.json")
    c0 = artifacts / "C0"
    links, seed_inputs = {}, {}
    for dataset in DATASETS:
        _guard(artifacts)
        cache = manifest["caches"][dataset]
        identity = a_roster["datasets"][dataset]
        bundle = None
        if verify_live:
            bundle = load_train_validation(dataset, artifacts / "DATA_AUDIT.json")
            fingerprint = bundle.identity["split_row_fingerprints"]["train"]
            c0_audit.require(bundle.data_hash == identity["data_hash"] == cache["data_hash"]
                             and bundle.identity["split_row_id_hashes"]["train"] == identity["train_row_hash"]
                             and bundle.identity["split_row_id_hashes"]["validation"] == identity["validation_row_hash"]
                             and cache["scope_hash"] == identity["scope_hash"]
                             and cache["split_rows"]["train"] == len(bundle.train_y)
                             and cache["split_row_fingerprints"]["train"] == fingerprint,
                             f"Original A1/C0 Train/Validation/scope binding differs: {dataset}")
            sampler = MacroPairSampler(dataset, bundle.train_x[:, -1], bundle.train_y, bundle.train_row_ids, fingerprint)
        for seed in SEEDS:
            key = f"{dataset}_{seed}"
            candidate = manifest["candidates"][key]
            if verify_live:
                _verify_archive(c0 / "candidates" / f"{key}.npz", candidate)
            for arm in C0_MODELS:
                _guard(artifacts)
                job = RunSpec("A1", dataset, seed, arm, a_config(arm, dataset), {})
                audit_job = manifest["jobs"][job.key]
                if verify_live:
                    _verify_archive(c0 / "margins" / f"{job.key}.npz", audit_job["margins"])
                endpoints = [row for row in published if row["job_key"] == job.key]
                link, input_record = _control(artifacts, a_roster, job, audit_job, cache, endpoints, bundle=bundle)
                links[job.key] = dict(link, audit=dict(candidate_sha256=candidate["sha256"],
                                                       margins_sha256=audit_job["margins"]["sha256"]))
                if arm == "P0":
                    inputs = dict(input_record, candidate=candidate, margins=audit_job["margins"],
                                  snapshots=_snapshots(audit_job))
                    _validate_seed_inputs(manifest, dataset, seed, inputs)
                    if verify_live:
                        load_candidates(sampler, seed, inputs)
                    seed_inputs[key] = inputs
    return links, seed_inputs


def freeze_roster(artifacts, *, create=False):
    """Verify inputs before freezing the immutable initial twenty-trajectory C1 roster."""
    artifacts = Path(artifacts)
    _guard(artifacts)
    a_roster, b_roster = (read(artifacts / stage / "ROSTER.json") for stage in ("A1", "B1"))
    snapshot = _source_snapshot(artifacts, a_roster, b_roster)
    gates = verify_gates(artifacts)
    manifest = _verified_manifest(artifacts, gates)
    source = _current_source(artifacts, a_roster, snapshot)
    datasets = _registered_datasets(artifacts)
    c0_audit.require(datasets == a_roster["datasets"] == b_roster["datasets"], "C1 frozen data or scope differs from A1/B1")
    jobs = c1_jobs()
    fields = dict(study=STUDY, stage="C1", source=source, datasets=datasets,
                  jobs=[_record(job, source) for job in jobs], gates=gates, test_status="SEALED", C4="NOT_TRIGGERED")
    path = artifacts / "C1" / "ROSTER.json"
    if create:
        with lock(artifacts / "SUPERVISION.lock", nonblocking=True):
            _guard(artifacts)
            status = read(artifacts / "B1" / "STAGE_STATUS.json")
            c0_audit.require(status["stage"] == "B1"
                             and status["status"] in ("COMPLETE", "TECHNICAL_PAUSED", "PAUSE_TRAINING", "USER_STOPPED"),
                             "C1 shared integration requires a terminal B1 supervisor state")
            if not path.exists():
                links, seed_inputs = _collect_inputs(artifacts, a_roster, manifest, verify_live=True)
                _guard(artifacts)
                verify_gates(artifacts, expected=gates)
                roster = dict(fields, controls=links, seed_inputs=seed_inputs)
                roster["roster_hash"] = digest(roster)
                write(path, roster)
                return roster, jobs
    roster = read(path)
    c0_audit.require(all(roster.get(key) == value for key, value in fields.items())
                     and roster["roster_hash"] == digest({k: v for k, v in roster.items() if k != "roster_hash"})
                     and set(roster["seed_inputs"]) == {f"{dataset}_{seed}" for dataset in DATASETS for seed in SEEDS},
                     "Reviewed C1 source, data, configuration or roster differs")
    links, seed_inputs = _collect_inputs(artifacts, a_roster, manifest)
    c0_audit.require(roster.get("controls") == links, "C1 control links differ from verified original A1 records")
    for dataset in DATASETS:
        for seed in SEEDS:
            key = f"{dataset}_{seed}"
            inputs = roster["seed_inputs"][key]
            _validate_seed_inputs(manifest, dataset, seed, inputs)
            _verify_prefix_inputs(inputs, seed_inputs[key], key)
    return roster, jobs
