"""Shared supervised Train/Validation worker and raw artifact publisher."""
import argparse
import copy
import json
import math
import os
import resource
import signal
import time
import traceback
from pathlib import Path

import torch

from experiments.conditional_history_shared_update_20261122.history_controls import make_history_optimizer
from experiments.conditional_history_shared_update_20261122.protocol import append, read, v3, write
from experiments.conditional_history_shared_update_20261122.runner import add_counts, batch_inputs, select

from .freshness_optimizer import FreshnessAdapter
from .protocol import (B_HISTORY_PROVENANCE, READOUTS, STUDY, algorithm_digest, load_train_validation,
                       scope_digest, stage_jobs, verify_sources)
from .ram_state import Journal, lock
from . import scheduler


def make_training_parts(model, job, bundle, *, artifacts=None, roster=None):
    if job.stage == "A1":
        from .a_step import make_training_parts as a_parts
        return a_parts(model, job, bundle)
    if job.stage == "C1":
        from . import c1_protocol, c_step

        return c_step.make_training_parts(
            model, job, bundle, roster["seed_inputs"][f"{job.dataset}_{job.seed}"],
            verify_inputs=lambda record: c1_protocol.verify_seed_inputs(
                artifacts, expected=roster["gates"], dataset=job.dataset, seed=job.seed, inputs=record),
        )
    cfg = dict(job.config)
    optimizer = v3.make_optimizer(model, cfg) if job.arm == "B0" else make_history_optimizer(model, cfg)
    return optimizer, FreshnessAdapter(**job.adapter)


def freeze_roster(artifacts, stage, *, create=False):
    if stage == "C1":
        from .c1_protocol import freeze_roster as freeze_c1

        return freeze_c1(artifacts, create=create)
    source_identity = verify_sources(artifacts, stage)
    data = read(artifacts / "DATA_AUDIT.json")["datasets"]
    audit = read(artifacts / "SOURCE_AUDIT.json")
    jobs = stage_jobs(stage)
    if stage == "A1":
        cells = {f"d{d}t{t}" for d in range(3) for t in range(2)}
        for dataset, row in data.items():
            support = row["train_macro6_cell_support"]
            if set(support) != cells or any(not v["positive"] or not v["negative"] for v in support.values()):
                raise ValueError(f"{dataset} lacks audited Train support for all six Macro AUC cells")
    roster = dict(study=STUDY, stage=stage, source=source_identity,
                  datasets={name: dict(data_hash=row["data_hash"], raw_sha256=row["raw_sha256"],
                                       train_row_hash=row["split_row_id_hashes"]["train"],
                                       validation_row_hash=row["split_row_id_hashes"]["validation"],
                                       train_domain_counts=row["train_domain_counts"],
                                       scope_hash=audit["canonical_optimizer_and_ema"][name]["parameter_manifest_hash"])
                            for name, row in data.items()},
                  jobs=[dict(job.record(), config_hash=job.config_hash,
                             algorithm_hash=algorithm_digest(job, source_identity)) for job in jobs],
                  test_status="SEALED")
    if stage == "B1":
        roster.update(B1_parent=B_HISTORY_PROVENANCE, B3="B1_OBSERVER", gamma_confirm="NOT_TRIGGERED",
                      EXP082="HISTORICAL_TEST_PROVENANCE",
                      context_models=["M3oE", "HyperGate", "AutoIFS"])
    if stage == "A1":
        roster.update(A2="NOT_TRIGGERED", C0_models=dict(arms=["B0", "P0"], fixed_ema_epochs=[5, 10]))
    path = artifacts / stage / "ROSTER.json"
    if create:
        with lock(artifacts / "SUPERVISION.lock", nonblocking=True):
            if not path.exists():
                write(path, roster)
    if read(path) != roster:
        raise ValueError(f"Reviewed {stage} source, data, configuration or roster differs")
    return roster, jobs


def job_journal(artifacts, roster, job):
    if job.stage == "C1":
        from .c1_protocol import job_journal as c1_journal

        return c1_journal(artifacts, roster, job)
    data = roster["datasets"][job.dataset]
    binding = dict(study=STUDY, stage=job.stage, dataset=job.dataset, seed=job.seed, arm=job.arm,
                   source_hash=roster["source"]["source_hash"], config_hash=job.config_hash,
                   algorithm_hash=algorithm_digest(job, roster["source"]), data_hash=data["data_hash"], scope_hash=data["scope_hash"], job_key=job.key)
    return Journal(artifacts / job.stage / "runs" / job.key, binding)


def _scope_manifest(model):
    from experiments.conditional_information_momentum_20261102_v3.optimizers import shared_name
    return [dict(name=n, shape=list(p.shape), history=shared_name(n)) for n, p in model.named_parameters()]


def _epoch_files(directory, keep):
    for file in directory.glob("epoch_*.json"):
        if int(file.stem.split("_")[1]) > keep:
            file.unlink()


def _epoch_file(journal, snapshot, receipt):
    write(journal.directory / f'epoch_{snapshot["epoch"]:02d}.json',
          dict(snapshot["epoch_metrics"], ram_receipt=receipt))


def _restore(state, model, optimizer, ema, adapter, loader, device, *, memory=False):
    if memory:
        adapter.restore_epoch(state)
    model.load_state_dict(state["online"])
    model.train(state["model_mode"])
    ema.load_state_dict(state["ema"])
    ema.train(state["ema_mode"])
    optimizer.load_state_dict(state["optimizer"])
    if not memory:
        adapter.load_state_dict(state["adapter"])
    loader.set_state(state["loader"])
    v3.restore_rng(state["rng"], device)


def _endpoint(job, binding, kind, checkpoint, epoch, model_hash, metrics):
    return dict(binding, split="validation", readout=kind, checkpoint=checkpoint, epoch=epoch,
                model_hash=model_hash, **metrics)


def _retention_required(job):
    return job.stage == "A1" and job.arm in ("B0", "P0")


def _retain_epoch_model(journal, attempt, snapshot):
    key = f'fixed_ema-E{snapshot["epoch"]:02d}'
    return journal.commit_model(attempt, key, "fixed_ema", snapshot["epoch"], snapshot["ema"],
                                snapshot["global_step"])


def _receipt_fields(receipt):
    return {key: receipt["identity"][key] for key in ("kind", "readout", "epoch", "model_hash")}


def _committed_c1_rows(journal, epochs, steps):
    metadata = read(journal.path)
    writer_epochs = {}
    for epoch in range(1, epochs + 1):
        record = read(journal.directory / f"epoch_{epoch:02d}.json")
        receipt = record["ram_receipt"]
        identity = receipt["identity"]
        if (identity["kind"] != "epoch" or record["epoch"] != identity["epoch"] or identity["epoch"] != epoch
                or not record["global_step"] == identity["global_step"] == epoch * steps
                or any(identity.get(k) != v for k, v in journal._expected(metadata, receipt).items())
                or (epoch == epochs and receipt != metadata["epochs"][0])):
            raise ValueError(f"Invalid committed C1 epoch receipt E{epoch:02d}")
        writer_epochs.setdefault(identity["attempt"], set()).add(epoch)
    by_step = {}
    for writer, allowed in writer_epochs.items():
        with (journal.directory / "attempts" / writer / "batches.jsonl").open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                context = row["context"]
                epoch = context["epoch"]
                if epoch not in allowed:
                    continue
                step, batch = context["global_step"], context["batch"]
                if (not 1 <= batch <= steps or step != (epoch - 1) * steps + batch or step in by_step
                        or context["dataset"] != journal.binding["dataset"]
                        or context["seed"] != journal.binding["seed"] or context["split_id"] != "original"):
                    raise ValueError(f"Invalid committed C1 batch receipt at logical step {step}")
                by_step[step] = row
    for step in range(1, epochs * steps + 1):
        if step not in by_step:
            raise ValueError(f"Missing committed C1 batch receipt at logical step {step}")
        yield by_step.pop(step)


def _runtime_worker(artifacts, roster, job, gpu, invocation, supervisor, diagnostic_through):
    journal = job_journal(artifacts, roster, job)
    device = torch.device("cuda:0")
    stopping = False

    def request_stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    def stopped():
        return stopping or os.getppid() != supervisor or scheduler.stop_status(artifacts)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    with journal.writer():
        return _worker(artifacts, journal, roster, job, gpu, device, invocation, diagnostic_through, stopped)


def _worker(artifacts, journal, roster, job, gpu, device, invocation, diagnostic_through, stopped):
    attempt = journal.new_attempt(gpu)
    out = journal.directory / "attempts" / attempt["id"]
    out.mkdir(parents=True)
    started = time.monotonic()
    step = completed = physical_steps = physical_epochs = peak_gpu = usable = batch = 0
    epoch = 1
    phase = "SETUP"
    allocation = None
    allocation_number = 0
    observer_seconds = 0.
    outcome = {}
    cfg = dict(job.config)
    memory = job.stage == "C1"
    if memory:
        from .memory_bank import TEACHER_EPOCH, replay_exposure

    def capture_residency(site):
        nonlocal peak_gpu, usable, allocation, allocation_number
        torch.cuda.synchronize(device)
        _, usable = torch.cuda.mem_get_info(device)
        telemetry = scheduler.gpu_sample()
        process_bytes = scheduler.residency(telemetry, os.getpid(), gpu)
        peak_gpu = max(peak_gpu, process_bytes)
        event = dict(epoch=epoch, global_step=step, batch=batch, site=site, gpu=gpu,
                     process_bytes=process_bytes, peak_process=peak_gpu, cuda_usable=usable,
                     torch_peak_reserved=torch.cuda.max_memory_reserved(device), telemetry=telemetry)
        allocation = event
        allocation_number += 1
        write(invocation / "allocations" / f"{allocation_number:04d}.json", event)
        return event

    try:
        write(out / "MANIFEST.json", dict(job=job.record(), binding=journal.binding, attempt=attempt,
                                         diagnostic_through=diagnostic_through, invocation=str(invocation)))
        if stopped():
            outcome = dict(status="BOUNDARY_STOP")
            return outcome
        bundle = load_train_validation(job.dataset, artifacts / "DATA_AUDIT.json")
        if bundle.data_hash != journal.binding["data_hash"]:
            raise ValueError("Input identity differs from frozen trajectory")
        v3.setup(job.seed, device)
        frozen = v3.load_protocol()["datasets"][job.dataset]["source"]
        model = v3.make_model(bundle, frozen).to(device)
        ema = copy.deepcopy(model).requires_grad_(False)
        optimizer, adapter = make_training_parts(model, job, bundle, artifacts=artifacts, roster=roster)
        scope = getattr(optimizer, "parameter_manifest", None) or _scope_manifest(model)
        if scope_digest(scope) != journal.binding["scope_hash"]:
            raise ValueError("Parameter manifest differs from the audited M3oE scope")
        init_model_hash = v3.state_digest(model.state_dict())
        if memory:
            adapter.verify_init(init_model_hash)
        write(out / "IDENTITY.json", dict(init_model_hash=init_model_hash, parameter_manifest=scope,
                                         encoder_hash=bundle.features.encoder_hash, scaler_hash=bundle.features.scaler_hash,
                                         data=bundle.identity, gaussian_field_order=frozen["perturbable_sparse_fields"]))
        loader = torch.Generator().manual_seed(job.seed + 11000)
        selected, selected_epochs, selected_values, counts, endpoints, retained = {}, {}, {}, {}, [], {}
        state, verified = journal.recover()
        if state is not None:
            if memory:
                epoch, step = state["epoch"] + 1, state["global_step"]
                phase = "C1_RESTORE"
            _restore(state, model, optimizer, ema, adapter, loader, device, memory=memory)
            completed, step = state["epoch"], state["global_step"]
            selected = state["selected_models"]
            selected_epochs, selected_values = state["selected_epochs"], state["selected_metrics"]
            counts, endpoints, retained = state["counts"], state["endpoints"], state["retained"]
        if not diagnostic_through:
            _epoch_files(journal.directory, completed)
            if state is not None:
                _epoch_file(journal, state, verified)
                if _retention_required(job) and completed in (5, 10):
                    retained[str(completed)] = _receipt_fields(_retain_epoch_model(journal, attempt, state))
        del state
        write(out / "RESUME.json", dict(last_verified=verified, completed=completed, global_step=step))
        epoch = completed + 1
        phase = "SETUP"
        torch.cuda.reset_peak_memory_stats(device)
        capture_residency("setup")
        probe_seen = False

        def probe_residency():
            nonlocal probe_seen
            if not probe_seen:
                capture_residency("first_two_state_probe")
                probe_seen = True

        if isinstance(adapter, FreshnessAdapter):
            adapter.on_probe = probe_residency
        memory_seen = False
        last_epoch = min(cfg["epochs"], diagnostic_through or cfg["epochs"])
        if diagnostic_through and diagnostic_through <= completed:
            raise ValueError("Post-OOM measurement requires a pending epoch")
        for epoch in range(completed + 1, last_epoch + 1):
            if stopped():
                outcome = dict(status="BOUNDARY_STOP")
                return outcome
            epoch_started = time.monotonic()
            model.train()
            order = torch.randperm(len(bundle.train_y), generator=loader).numpy()
            losses, real_losses, epoch_counts = 0., 0., {}
            epoch_observer = 0.
            for batch, offset in enumerate(range(0, len(order), cfg["batch_size"]), 1):
                if stopped():
                    outcome = dict(status="BOUNDARY_STOP")
                    return outcome
                phase = "TRAIN"
                ids = order[offset:offset + cfg["batch_size"]]
                x, y, rows = batch_inputs(bundle, ids, device)
                step += 1
                context = dict(study="EMA02V2", dataset=job.dataset, split_id="original", seed=job.seed, epoch=epoch,
                               global_step=step, batch=batch, last_batch=offset + len(ids) == len(order),
                               role="NEW_SCIENTIFIC_TRAIN")
                step_started = time.monotonic()
                value = adapter.step(model, optimizer, ema, x, y, rows, cfg, context)
                seconds = time.monotonic() - step_started
                physical_steps += 1
                observation = value["diagnostics"].get("observer_seconds", 0.)
                observer_seconds += observation
                epoch_observer += observation
                add_counts(counts, value["counts"])
                add_counts(epoch_counts, value["counts"])
                losses += len(ids) * value["loss"]
                real_losses += len(ids) * value["real_loss"]
                append(out / "batches.jsonl", dict(context=context, row_id_hash=v3.state_digest(rows), rows=len(ids),
                                                    loss=value["loss"], real_loss=value["real_loss"],
                                                    step_seconds=seconds, scientific_step_seconds=seconds - observation,
                                                    diagnostics=value["diagnostics"], counts=value["counts"]))
                if physical_steps == 1:
                    capture_residency("first_completed_batch")
                if memory and epoch > TEACHER_EPOCH and not memory_seen and value["counts"]["pair_forwards"]:
                    capture_residency("first_memory_batch")
                    memory_seen = True
                del x, y, value
            order_hash = None
            if memory:
                order_hash = v3.state_digest(bundle.train_row_ids[order])
                if epoch == TEACHER_EPOCH:
                    phase = "E10_MEMORY_INSTALL"
                boundary = adapter.finish_epoch(epoch, step, order_hash, model, optimizer, ema, loader, device)
                if boundary is not None:
                    write(out / "E10_MEMORY.json", boundary)
                    capture_residency("e10_memory_install")
            phase = "VALIDATION"
            validation, validation_cost = {}, {}
            for kind, estimator in (("online", model), ("fixed_ema", ema)):
                scores, cost = v3.evaluate(estimator, bundle.val_x, bundle.val_y, device)
                if scores["macro_auc"] is None or not math.isfinite(scores["macro_auc"]):
                    raise ValueError("Full Validation six-cell Macro AUC is undefined")
                select(epoch, kind, estimator, scores, selected, selected_epochs, selected_values)
                validation[kind], validation_cost[kind] = scores, cost
                capture_residency(f"validation_{kind}")
                if epoch in (10, cfg["epochs"]):
                    endpoints.append(_endpoint(job, journal.binding, kind, f"E{epoch}", epoch,
                                               v3.state_digest(estimator.state_dict()), scores))
            physical_epochs += 1
            if diagnostic_through:
                continue
            phase = "RAM_COMMIT"
            record = dict(epoch=epoch, global_step=step,
                          order_hash=order_hash if memory else v3.state_digest(bundle.train_row_ids[order]),
                          rows=len(order), loss=losses / len(order), real_loss=real_losses / len(order),
                          epoch_counts=epoch_counts, wall_seconds=time.monotonic() - epoch_started,
                          observer_seconds=epoch_observer, validation=validation, validation_cost=validation_cost,
                          selected_epochs=selected_epochs,
                          selected_model_hashes={n: v3.state_digest(v) for n, v in selected.items()},
                          selected_metrics=selected_values, residency=allocation)
            if memory and boundary is not None:
                record["memory"] = boundary
            snapshot = v3.cpu_copy(dict(epoch=epoch, global_step=step, permutation=order, cursor=len(order),
                                        epoch_metrics=record, loader=loader.get_state(), rng=v3.rng_state(device),
                                        online=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                                        adapter=adapter.state_dict(), model_mode=model.training, ema_mode=ema.training,
                                        selected_models=selected, selected_epochs=selected_epochs,
                                        selected_metrics=selected_values, sampler_context=context,
                                        counts=counts, endpoints=endpoints, retained=retained))
            receipt = journal.commit_epoch(attempt, snapshot)
            _epoch_file(journal, snapshot, receipt)
            completed = epoch
            if _retention_required(job) and epoch in (5, 10):
                retained[str(epoch)] = _receipt_fields(_retain_epoch_model(journal, attempt, snapshot))
            del snapshot
        measurement = (dict(complete=True, epoch=epoch, cuda_usable=usable, peak_process=peak_gpu)
                       if physical_epochs else None)
        if diagnostic_through:
            outcome = dict(status="DIAGNOSTIC_COMPLETE", measurement=measurement)
            return outcome
        if completed != cfg["epochs"]:
            raise ValueError("Worker exhausted its training loop before the required final epoch")
        if memory:
            if stopped():
                outcome = dict(status="BOUNDARY_STOP")
                return outcome
            phase = "C1_EXPOSURE_REPLAY"
            steps = math.ceil(len(bundle.train_y) / cfg["batch_size"])
            exposure = replay_exposure(adapter.sampler, adapter.bank, _committed_c1_rows(journal, completed, steps))
            e10 = read(journal.directory / f"epoch_{TEACHER_EPOCH:02d}.json")
            if e10["memory"]["prefix"] != adapter.boundary or e10["memory"]["bank"]["bank_hash"] != adapter.bank["bank_hash"]:
                raise ValueError("Committed C1 E10 boundary differs from the verified adapter")
        required = {}
        for kind in READOUTS:
            key = f"{kind}-selected"
            selected_epoch = selected_epochs[kind]
            selected_step = selected_epoch * math.ceil(len(bundle.train_y) / cfg["batch_size"])
            receipt = journal.commit_model(attempt, key, kind, selected_epoch, selected[kind], selected_step)
            required[key] = _receipt_fields(receipt)
            endpoints.append(_endpoint(job, journal.binding, kind, "selected", selected_epoch,
                                       required[key]["model_hash"], selected_values[kind]))
        if _retention_required(job):
            for e in (5, 10):
                required[f"fixed_ema-E{e:02d}"] = retained[str(e)]
        result = dict(job=job.record(), binding=journal.binding, epoch=completed, global_step=step,
                      counts=counts, endpoints=endpoints, selected=selected_epochs, compact_models=required,
                      source_provenance=B_HISTORY_PROVENANCE if job.stage == "B1" and job.arm != "B0" else None,
                      test_accesses=0)
        if memory:
            result["memory"] = dict(e10["memory"], exposure=exposure,
                                    e10_ram_packet_bytes=e10["ram_receipt"]["size"])
        journal.complete(attempt, result, required)
        write(journal.directory / "COMPLETE_VERIFIED.json", result)
        outcome = dict(status="COMPLETE", measurement=measurement)
        return outcome
    except Exception as exc:
        failure = traceback.format_exc()
        oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        if oom:
            try:
                capture_residency("failed_allocation")
            except (OSError, RuntimeError, ValueError):
                pass
        outcome = dict(status="CUDA_OOM" if oom else "TECHNICAL_FAILURE", traceback=failure,
                       failed_phase=phase, failed_epoch=epoch, failed_batch=batch, failed_step=step,
                       failed_allocation_bytes=scheduler.failed_allocation(exc) if oom else None,
                       measurement=dict(complete=False, epoch=epoch, cuda_usable=usable, peak_process=peak_gpu))
        return outcome
    finally:
        receipt = dict(attempt=attempt, **outcome, completed=completed, physical_train_steps=physical_steps,
                       physical_validation_epochs=physical_epochs, observer_seconds=observer_seconds,
                       worker_seconds=time.monotonic() - started, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       diagnostic_through=diagnostic_through)
        write(out / "ATTEMPT.json", receipt)
        write(invocation / "OUTCOME.json", receipt)


def publish_tables(artifacts, stage, roster, jobs, status):
    output = artifacts / stage
    endpoints, costs, compact = [], [], []
    is_c1 = stage == "C1"
    completed = 0
    exposure, boundaries = {}, {}
    for job in jobs:
        journal = job_journal(artifacts, roster, job)
        tags = dict(logical_arm=job.arm, linked_control=False) if is_c1 else {}
        if result := journal.completed():
            completed += 1
            endpoints.extend(dict(row, **tags) for row in result["endpoints"])
            for key, expected in result["compact_models"].items():
                _, receipt = journal.read_model(key, expected=expected)
                compact.append(dict(receipt, **tags))
            if is_c1:
                memory = result["memory"]
                exposure[job.key] = dict(binding=journal.binding, rows=memory["exposure"])
                boundaries.setdefault((job.dataset, job.seed), {})[job.arm] = {
                    key: memory[key] for key in ("prefix", "bank", "e10_ram_packet_bytes")
                }
        attempts = [read(path) for path in sorted((journal.directory / "attempts").glob("*/ATTEMPT.json"))]
        costs.extend(dict(job=job.key, **attempt, **tags) for attempt in attempts)
    if is_c1:
        for key, control in roster["controls"].items():
            tags = dict(logical_arm=control["logical_arm"], linked_control=True)
            endpoints.extend(dict(row, **tags) for row in control["validation_endpoints"])
            compact.extend(dict(row, **tags) for row in control["readbacks"].values())
            original = artifacts / "A1" / "runs" / key / "attempts"
            for path in sorted(original.glob("*/ATTEMPT.json")):
                costs.append(dict(job=key, **read(path), **tags))
        write(output / "LINKED_CONTROLS.json", roster["controls"])
        write(output / "EXPOSURE.json", exposure)
        write(output / "E10_BOUNDARIES.json", [
            dict(dataset=dataset, seed=seed, arms=arms,
                 prefix_matches=arms["C2"]["prefix"] == arms["C3"]["prefix"] if arms.keys() >= {"C2", "C3"} else None)
            for (dataset, seed), arms in sorted(boundaries.items())
        ])
    selected = [row for row in endpoints if row["checkpoint"] == "selected"]
    six_cell = [dict(dataset=row["dataset"], seed=row["seed"], arm=row["arm"], readout=row["readout"],
                     epoch=row["epoch"], domain=d, task=t, auc=row[f"d{d}t{t}_auc"],
                     logloss=row[f"d{d}t{t}_logloss"], count=row[f"d{d}t{t}_count"],
                     **({k: row[k] for k in ("stage", "job_key", "logical_arm", "linked_control")} if is_c1 else {}))
                for row in selected for d in range(3) for t in range(2)]
    for name, rows in (("ENDPOINTS", endpoints), ("SELECTED", selected), ("SIX_CELL", six_cell),
                       ("COST", costs), ("VERIFIED_MODEL_READBACKS", compact)):
        write(output / f"{name}.json", rows)
    write(output / "STAGE_STATUS.json",
          dict(status, completed=completed if is_c1 else len(selected) // len(READOUTS), jobs=len(jobs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("A1", "B1", "C1"))
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--freeze-roster", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--gpu")
    parser.add_argument("--invocation", type=Path)
    parser.add_argument("--supervisor", type=int)
    parser.add_argument("--diagnostic-through", type=int)
    args = parser.parse_args()
    if not args.artifacts.is_absolute():
        parser.error("--artifacts requires an absolute path")
    artifacts = args.artifacts.resolve()
    if args.worker and (args.supervisor != os.getppid() or not args.invocation or not args.gpu):
        parser.error("Workers must be launched by the foreground scheduler")
    roster, jobs = freeze_roster(artifacts, args.stage, create=args.freeze_roster)
    if args.freeze_roster:
        print(f"{len(jobs)} frozen {args.stage} trajectories")
        return 0
    if args.worker:
        job = next(job for job in jobs if job.key == args.worker)
        result = _runtime_worker(artifacts, roster, job, args.gpu, args.invocation,
                                 args.supervisor, args.diagnostic_through)
        return 1 if result["status"] in ("TECHNICAL_FAILURE", "CUDA_OOM") else 0
    status = scheduler.supervise(artifacts, args.stage, roster, jobs)
    return 0 if status["status"] in ("COMPLETE", "USER_STOPPED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
