"""Canonical J_ORIG training, end-epoch transactions and isolated S1 forks."""

import copy
import math
import os
import resource
import signal
import time
import traceback

import numpy as np
import torch

from experiments.conditional_history_shared_update_20261122.protocol import digest, read, sha, write
from experiments.conditional_information_momentum_20261102_v3 import protocol as v3
from experiments.conditional_information_momentum_20261102_v3.guardian import request
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.protocol import tensor_bytes

ANCHORS = (10, 25, 40)
KINDS = ("online", "fixed_ema")
COUNTERS = ("training_forwards", "parameter_backward_calls", "optimizer_steps", "ema_updates")
FORK_COST_FIELDS = (*COUNTERS, "eval_forwards", "eval_rows", "prediction_seconds", "metric_seconds", "seconds")


def owner_identity(owner):
    info = read(owner / "guardian.json")
    if request(owner, "status")["pid"] != info["pid"]:
        raise RuntimeError("RAM guardian PID changed")
    return digest([info["address"], info["pid"]])


def register(owner, key, identity):
    result = request(owner, "register", key=key, job=identity, pid=os.getpid())
    if result["state"] is not None or result["attempt"] != 1:
        raise RuntimeError(f"RAM storage key already used: {key}")


def verify(owner, receipt):
    state = request(owner, "get", key=receipt["key"])["state"]
    if (state is None or state["epoch"] != receipt["sequence"]
            or v3.state_digest(state["packet"]) != receipt["state_hash"]):
        raise RuntimeError(f"RAM readback hash mismatch: {receipt['key']}")
    if state["packet"]["owner"] != receipt["owner"]:
        raise RuntimeError("RAM owner changed")
    return state["packet"]


def commit(owner, key, sequence, path, packet, costs):
    receipt = dict(key=key, sequence=sequence, epoch=packet["epoch"],
                   owner=packet["owner"], state_hash=v3.state_digest(packet),
                   bytes=tensor_bytes(packet) + packet["permutation"].nbytes,
                   status="RAM_INTENT")
    write(path, receipt)
    state = dict(epoch=sequence, packet=packet, counts=packet["counts"],
                 physical_worker_seconds=costs)
    ack = request(owner, "commit", key=key, state=state, epoch_file=str(path))
    if ack != dict(status="RAM_COMMITTED", epoch=sequence):
        raise RuntimeError(f"Unexpected RAM acknowledgment: {ack}")
    verify(owner, receipt)
    receipt["status"] = "RAM_VERIFIED"
    write(path, receipt)
    return receipt


def recover(owner, generation, out, job):
    """Recover an interrupted publication only from a fresh, exact owner get."""
    states = request(owner, "status")["jobs"]
    candidates = []
    for path in out.glob("attempts/*/RAM_E*.json"):
        receipt = read(path)
        record = states.get(receipt["key"])
        if (receipt["owner"] == generation and record
                and record["epoch"] == receipt["sequence"] and not record.get("state_released")):
            candidates.append((receipt["epoch"], path, receipt))
    if not candidates:
        return None, None
    _, path, receipt = max(candidates)
    packet = verify(owner, receipt)
    if packet["job"] != job:
        raise RuntimeError("Resume configuration or data identity changed")
    if receipt["status"] == "RAM_INTENT":
        receipt["status"] = "RAM_VERIFIED"
        write(path, receipt)
    return packet, receipt


def modes(model, ema):
    return [[module.training for module in net.modules()] for net in (model, ema)]


def capture(model, opt, ema, loader, order, device, **metadata):
    return dict(**metadata,
                model=v3.cpu_copy(model.state_dict()),
                optimizer=v3.cpu_copy(opt.state_dict()),
                ema=v3.cpu_copy(ema.state_dict()),
                modes=modes(model, ema),
                rng=v3.rng_state(device), loader=loader.get_state().clone(),
                permutation=order.copy(), cursor=len(order))


def restore(packet, model, opt, ema, loader, device):
    if packet["cursor"] != len(packet["permutation"]):
        raise RuntimeError("Incomplete epoch transaction")
    model.load_state_dict(packet["model"])
    opt.load_state_dict(v3.cpu_copy(packet["optimizer"]))
    ema.load_state_dict(packet["ema"])
    for net, saved in zip((model, ema), packet["modes"], strict=True):
        for module, mode in zip(net.modules(), saved, strict=True):
            module.training = mode
    loader.set_state(packet["loader"])
    v3.restore_rng(packet["rng"], device)
    if (not v3.equal(model.state_dict(), packet["model"])
            or not v3.equal(opt.state_dict(), packet["optimizer"])
            or not v3.equal(ema.state_dict(), packet["ema"])
            or not v3.equal(v3.rng_state(device), packet["rng"])
            or not torch.equal(loader.get_state(), packet["loader"])):
        raise RuntimeError("Model, optimizer, EMA, RNG or loader restore differs from RAM")


def next_order(size, loader):
    return torch.randperm(size, generator=loader).numpy()


def preview_order(size, loader):
    clone = torch.Generator().set_state(loader.get_state())
    return next_order(size, clone)


def batch_plan(size, batch_size):
    return [(offset, min(offset + batch_size, size)) for offset in range(0, size, batch_size)]


def batch_inputs(bundle, indices, device):
    return (torch.as_tensor(bundle.train_x[indices], device=device).T,
            torch.as_tensor(bundle.train_y[indices], device=device))


def select(selected, kind, epoch, metrics, model):
    score = metrics["macro_auc"]
    if score is None or not math.isfinite(score) or epoch < 1:
        raise RuntimeError("Full Validation selection requires a finite post-epoch Macro AUC")
    if kind not in selected or score > selected[kind]["metrics"]["macro_auc"]:
        selected[kind] = dict(epoch=epoch, metrics=metrics, model=v3.cpu_copy(model.state_dict()))


def add_counts(total, changes):
    for name, value in changes.items():
        total[name] = total.get(name, 0) + value


def process_memory():
    from .scheduler import sample
    telemetry = sample()
    matches = [row["bytes"] for row in telemetry["processes"] if row["pid"] == os.getpid()]
    if not matches:
        raise RuntimeError("Full CUDA process residency is unavailable")
    return max(matches), telemetry


def diagnose(owner, packet, parent_receipt, model, opt, ema, live_loader, bundle, panels, out, attempt, device,
             physical_diagnostics):
    from . import diagnostics

    epoch, job = packet["epoch"], packet["job"]
    epoch_tag = f"E{epoch:02d}"
    receipt_path = out / f"anchor_{epoch_tag}.json"
    previous = read(receipt_path) if receipt_path.exists() else None
    anchor_key = f"D2:{job['logical_id']}:{packet['owner']}:anchor:{epoch_tag}:{attempt}"
    anchor = dict(packet)
    for field in ("selected", "epochs", "next_prediction"):
        del anchor[field]
    anchor["ledger_hash"] = digest(packet["epochs"])
    anchor_hash = v3.state_digest(anchor)
    if previous and previous["owner"] == packet["owner"] and previous["status"] == "RAM_INTENT":
        existing_key = request(owner, "status")["jobs"].get(previous["key"])
        if existing_key and existing_key["epoch"] == previous["sequence"]:
            verify(owner, previous)
            previous["status"] = "RAM_VERIFIED"
            write(receipt_path, previous)
        else:
            previous = None
    if previous and previous["owner"] == packet["owner"]:
        if (previous["state_hash"] != anchor_hash
                or v3.state_digest(verify(owner, previous)) != anchor_hash):
            raise RuntimeError("Stored end-epoch anchor differs from verified baseline")
        anchor_receipt = previous
    else:
        register(owner, anchor_key, dict(job=job, role="ANCHOR", epoch=epoch))
        anchor_receipt = commit(owner, anchor_key, 1, receipt_path, anchor, 0)
    path = out / f"forks_{epoch_tag}.json"
    if path.exists():
        existing = read(path)
        if (existing["metadata"]["anchor_hash"] == anchor_hash
                and existing["metadata"]["anchor_key"] == anchor_receipt["key"]):
            return anchor_receipt
        if previous and previous["owner"] == packet["owner"]:
            raise RuntimeError("Existing anchor fork outputs have a different identity")
        attempt_copy = out / "attempts" / existing["metadata"]["attempt_id"] / path.name
        write(attempt_copy, existing)

    loader = torch.Generator().set_state(packet["loader"])
    order = preview_order(len(bundle.train_y), loader)
    batch_size = job["config"]["batch_size"]
    if len(order) < 2 * batch_size:
        raise RuntimeError("S1 anchor requires two complete batches")
    first_two = [order[:batch_size], order[batch_size:2 * batch_size]]
    if v3.state_digest(order) != packet["next_prediction"]:
        raise RuntimeError("Anchor preview disagrees with the baseline's next-permutation hash")
    fork_key = f"D2:{job['logical_id']}:{attempt}:fork:{epoch_tag}:{time.monotonic_ns()}"
    register(owner, fork_key, dict(job=job, role="FORK", epoch=epoch, anchor_hash=anchor_hash))
    attempt_dir = out / "attempts" / attempt
    fork_path = attempt_dir / f"fork_RAM_{epoch_tag}.json"

    def record_cost(row):
        physical_diagnostics["costs"].append(dict(row, fork_key=fork_key))
        add_counts(physical_diagnostics["counts"], {name: row[name] for name in FORK_COST_FIELDS if name in row})
        write(attempt_dir / "FORK_COSTS.json", physical_diagnostics)

    try:
        fork_receipt = commit(owner, fork_key, 1, fork_path, anchor, 0)
        saved = verify(owner, fork_receipt)
        fork_model, fork_opt, fork_ema = copy.deepcopy((model, opt, ema))
        fork_loader = torch.Generator()
        restore(saved, fork_model, fork_opt, fork_ema, fork_loader, device)
        del saved
        rows = [bundle.train_row_ids[ids] for ids in first_two]
        metadata = dict(dataset=job["dataset"], seed=job["seed"], epoch=epoch, source_hash=job["source_hash"],
                        data_hash=job["data_hash"], config_hash=job["config_hash"], logical_id=job["logical_id"],
                        attempt_id=attempt, anchor_hash=anchor_hash, anchor_key=anchor_receipt["key"],
                        fork_key=fork_key, baseline_key=parent_receipt["key"],
                        loader_hash=v3.state_digest(packet["loader"]), next_permutation_hash=packet["next_prediction"],
                        ledger_hash=anchor["ledger_hash"], batch_positions=[[0, batch_size], [batch_size, 2 * batch_size]],
                        batch_index_hashes=[v3.state_digest(ids) for ids in first_two],
                        row_id_hashes=[v3.state_digest(ids) for ids in rows])
        batches = tuple(batch_inputs(bundle, ids, device) for ids in first_two)
        try:
            result = diagnostics.forks(fork_model, fork_opt, fork_ema, batches, panels, metadata, device,
                                       record_physical=record_cost)
        finally:
            v3.restore_rng(packet["rng"], device)
        if (not v3.equal(model.state_dict(), packet["model"])
                or not v3.equal(opt.state_dict(), packet["optimizer"])
                or not v3.equal(ema.state_dict(), packet["ema"])
                or modes(model, ema) != packet["modes"]
                or not v3.equal(v3.rng_state(device), packet["rng"])
                or not torch.equal(live_loader.get_state(), packet["loader"])
                or v3.state_digest(preview_order(len(order), live_loader)) != packet["next_prediction"]):
            raise RuntimeError("Fork execution changed baseline state or loader preview")
        torch.cuda.synchronize(device)
        measured, telemetry = process_memory()
        result["resources"] = dict(process_gpu_bytes=measured, telemetry=telemetry)
        result["ordered_row_ids"] = [ids.tolist() for ids in rows]
        write(path, result)
    finally:
        request(owner, "release", key=fork_key)
        request(owner, "failed", key=fork_key, failure="FORK_STORAGE_RELEASED")
    return anchor_receipt


def epoch_step(model, opt, ema, loader, bundle, job, epoch, device, counts, allocate, physical):
    from . import blocks, diagnostics

    cfg = job["config"]
    model.train()
    started = time.monotonic()
    order = next_order(len(bundle.train_y), loader)
    positions = batch_plan(len(order), cfg["batch_size"])
    row_ids = bundle.train_row_ids[order]
    visited = np.empty_like(row_ids)
    batch_rows = []
    before = counts.copy()
    loss = 0.0
    gpu_events = []
    for start, end in positions:
        indices = order[start:end]
        x, y = batch_inputs(bundle, indices, device)
        point = torch.cuda.Event(enable_timing=True)
        end_point = torch.cuda.Event(enable_timing=True)
        point.record()
        value = blocks.step(model, opt, ema, ((x, y),), cfg, detailed=False)
        end_point.record()
        gpu_events.append((point, end_point))
        add_counts(counts, {name: value[name] for name in COUNTERS})
        add_counts(physical, {name: value[name] for name in COUNTERS})
        loss += value["loss"] * (end - start)
        visited[start:end] = bundle.train_row_ids[indices]
        batch_rows.append(dict(start=start, end=end, rows=end-start,
                               row_id_hash=v3.state_digest(visited[start:end])))
        allocate()
        del x, y
    if not np.array_equal(row_ids, visited) or len(np.unique(order)) != len(order):
        raise RuntimeError("Canonical and actual row coverage differ")
    validation_started = time.monotonic()
    validation = {}
    for kind, estimator in zip(KINDS, (model, ema)):
        point = torch.cuda.Event(enable_timing=True)
        end_point = torch.cuda.Event(enable_timing=True)
        point.record()
        validation[kind], eval_counts = diagnostics.evaluate(estimator, bundle, device)
        end_point.record()
        gpu_events.append((point, end_point))
        change = {f"{kind}_{name}": value for name, value in eval_counts.items()}
        add_counts(counts, change)
        add_counts(physical, change)
    torch.cuda.synchronize(device)
    gpu_seconds = sum(start.elapsed_time(end) for start, end in gpu_events) / 1000
    physical["gpu_interval_seconds"] = physical.get("gpu_interval_seconds", 0) + gpu_seconds
    return order, dict(epoch=epoch, train_loss=loss / len(order), validation=validation,
                       canonical_rows_hash=v3.state_digest(row_ids), actual_rows_hash=v3.state_digest(visited),
                       permutation_hash=v3.state_digest(order), batches=batch_rows, rows=len(order),
                       counts={k: v - before.get(k, 0) for k, v in counts.items()},
                       wall_seconds=time.monotonic() - started, gpu_interval_seconds=gpu_seconds,
                       validation_seconds=time.monotonic() - validation_started)


def worker(art, owner, job, attempt, device, stop):
    from . import diagnostics, source

    generation = owner_identity(owner)
    out = art / "runs" / job["logical_id"]
    run_attempt = out / "attempts" / attempt
    key = f"D2:{job['logical_id']}:{attempt}:baseline"
    start = time.monotonic()
    physical = {}
    physical_diagnostics = dict(counts={}, costs=[])
    measured = []
    allocation_published = False
    stopped = False
    status = "FAILED"
    failure = None
    completed = 0
    recovered_epoch = 0
    receipt = None
    prior_receipt = None
    device = torch.device(device)

    def signal_stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    def allocate():
        nonlocal allocation_published
        if not allocation_published:
            torch.cuda.synchronize(device)
            used, telemetry = process_memory()
            measured.append(used)
            request(owner, "allocated", key=key, peak_memory_bytes=used)
            write(run_attempt / "ALLOCATION.json",
                  dict(key=key, process_gpu_bytes=used, telemetry=telemetry))
            allocation_published = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal_stop)
    register(owner, key, dict(job=job, attempt=attempt, role="BASELINE"))
    try:
        dataset = job["dataset"]
        bundle, identity = source.load_data(dataset)
        if identity["data_hash"] != job["data_hash"]:
            raise RuntimeError("Worker Train/Validation identity differs from S0")
        panels = diagnostics.prepare(bundle, dataset, art)
        if panels.manifest["panel_hash"] != job["panel_hash"]:
            raise RuntimeError("Worker Validation panels differ from frozen identities")
        v3.setup(job["seed"], device)
        model = source.make_model(bundle, dataset).to(device)
        ema = copy.deepcopy(model).requires_grad_(False)
        opt = source.optimizer(model, job["config"])
        loader = torch.Generator().manual_seed(job["seed"] + 11000)
        packet, prior_receipt = recover(owner, generation, out, job)
        selected, records, counts, anchors = {}, [], {}, {}
        if packet is not None:
            restore(packet, model, opt, ema, loader, device)
            completed = recovered_epoch = packet["epoch"]
            selected, records, counts = packet["selected"], packet["epochs"], packet["counts"].copy()
            write(run_attempt / "RESUME.json", dict(previous=prior_receipt, epoch=completed,
                  permutation_hash=packet["next_prediction"], restored=True))
        for boundary in ANCHORS:
            path = out / f"anchor_E{boundary}.json"
            if boundary < completed and path.exists():
                candidate = read(path)
                if candidate["owner"] != generation:
                    raise RuntimeError("Recovered baseline is missing a same-owner anchor")
                verify(owner, candidate)
                anchors[str(boundary)] = candidate
        torch.cuda.reset_peak_memory_stats(device)
        sequence = 0
        while True:
            if stopped or stop():
                status = "PAUSED_RESOURCE"
                break
            if completed in ANCHORS and str(completed) not in anchors:
                anchors[str(completed)] = diagnose(owner, packet, receipt or prior_receipt, model, opt, ema,
                                                    loader, bundle, panels, out, attempt, device,
                                                    physical_diagnostics)
                fork_resources = read(out / f"forks_E{completed:02d}.json")["resources"]
                measured.append(fork_resources["process_gpu_bytes"])
                if stopped or stop():
                    status = "PAUSED_RESOURCE"
                    break
            if completed == job["config"]["epochs"]:
                status = "COMPLETE"
                break
            del packet
            epoch = completed + 1
            expected_order_hash = v3.state_digest(preview_order(len(bundle.train_y), loader)) if completed in ANCHORS else None
            order, record = epoch_step(model, opt, ema, loader, bundle, job, epoch, device, counts, allocate, physical)
            if expected_order_hash is not None:
                fork_metadata = read(out / f"forks_E{completed:02d}.json")["metadata"]
                if not (record["permutation_hash"] == expected_order_hash == fork_metadata["next_permutation_hash"]):
                    raise RuntimeError("Baseline next training permutation differs from anchor prediction")
            record.update(job=job, attempt_id=attempt)
            for kind, estimator in zip(KINDS, (model, ema)):
                select(selected, kind, epoch, record["validation"][kind], estimator)
            records = records + [record]
            packet = capture(model, opt, ema, loader, order, device, job=job, owner=generation, epoch=epoch,
                             counts=counts.copy(), epochs=records, selected=selected,
                             next_prediction=v3.state_digest(preview_order(len(order), loader)) if epoch in ANCHORS else None)
            sequence += 1
            receipt = commit(owner, key, sequence, run_attempt / f"RAM_E{epoch:02d}.json",
                             packet, time.monotonic()-start)
            completed = epoch
            if prior_receipt:
                request(owner, "release", key=prior_receipt["key"])
                prior_receipt = None
        if status == "COMPLETE":
            receipt = receipt or prior_receipt
            saved = verify(owner, receipt)
            if saved["job"] != job or set(anchors) != set(map(str, ANCHORS)):
                raise RuntimeError("Baseline completion is missing verified identities")
            result = dict(job=job, attempt_id=attempt, epochs=completed, selected_epochs={}, selected_hashes={},
                          selected_validation={}, anchors=anchors, checkpoint=receipt, test_evaluations=0,
                          costs=dict(counts), epoch_hashes={})
            for kind, best in saved["selected"].items():
                result["selected_epochs"][kind] = best["epoch"]
                result["selected_hashes"][kind] = v3.state_digest(best["model"])
                result["selected_validation"][kind] = best["metrics"]
            for record in saved["epochs"]:
                path = out / f"epoch_{record['epoch']:02d}.json"
                write(path, record)
                result["epoch_hashes"][path.name] = sha(path)
            result["fork_hashes"] = {f"forks_E{epoch:02d}.json": sha(out / f"forks_E{epoch:02d}.json") for epoch in ANCHORS}
            write(out / "result.json", result)
    except BaseException as exc:
        status = "FAILED"
        failure = dict(error=type(exc).__name__, traceback=traceback.format_exc(),
                       cuda_oom=isinstance(exc, torch.cuda.OutOfMemoryError))
        raise
    finally:
        costs = dict(job=job, attempt_id=attempt, status=status, last_verified_epoch=completed,
                     resumed_epoch=recovered_epoch, wall_seconds=time.monotonic()-start,
                     counts=physical, diagnostic_counts=physical_diagnostics["counts"],
                     process_gpu_bytes=max(measured, default=0), failure=failure,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.cuda.is_initialized() else 0,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if torch.cuda.is_initialized() else 0,
                     max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        write(out / "attempts" / f"{attempt}.json", costs)
        request(owner, "failed", key=key, failure=status)
        if status == "COMPLETE" and receipt["key"] != key:
            request(owner, "release", key=key)
