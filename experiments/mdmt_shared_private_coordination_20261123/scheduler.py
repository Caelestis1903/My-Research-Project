"""Foreground S1 supervision of the ten registered trajectories."""

import csv
import fcntl
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from experiments.conditional_history_shared_update_20261122.protocol import append, digest, read, sha, write
from experiments.conditional_history_shared_update_20261122.scheduler import sample

from .training import ANCHORS, KINDS, owner_identity, v3, verify

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
STUDY = PACKAGE.name
ARTIFACTS = ROOT / "artifacts" / STUDY
REPORTS = ROOT / "research_reports" / STUDY
MODULE = f"experiments.{STUDY}.cli"
DATASETS = ("movielens", "kuairand")
SEEDS = (2022, 2023, 2024, 2025, 2026)


def stopping(art):
    siblings = (ROOT, ROOT.parent / "phase0-3", ROOT.parent / "phase0-3_tvg_rm_20261116")
    controls = [art, *(root / "artifacts" / STUDY for root in siblings)]
    return (any((directory / name).exists() for directory in controls
                for name in ("USER_STOPPED", "S0_PAUSE", "S1_PAUSE"))
            or any((root / "artifacts/26-11-16_augmentation_gradient_momentum/STOP_AUTONOMOUS").exists()
                   for root in siblings))


def admit(card, host, physical=None):
    if not (host < .90 and card["used"] < .85 * card["total"]
            and card["watts"] < .85 * card["limit"]):
        return False
    if physical is None:
        return True
    usable, worker = physical
    return 0 < worker <= usable - card["used"]


def startup_pending(path, pid, measurements):
    return (not path.exists() or read(path)["telemetry"]["time"] > measurements["time"]
            or not any(row["pid"] == pid for row in measurements["processes"]))


def jobs(authority):
    from .source import config

    result = []
    for seed in SEEDS:
        for dataset in DATASETS:
            cfg = config(dataset)
            item = dict(stage="S1", dataset=dataset, seed=seed, arm="J_ORIG", split="original",
                        config=cfg, config_hash=digest(cfg), source_hash=authority["source_hash"],
                        panel_hash=authority["panels"][dataset], data_hash=authority["datasets"][dataset])
            result.append(dict(item, logical_id=digest(item)))
    return result


def verified_result(art, owner, generation, job, owner_jobs):
    out = art / "runs" / job["logical_id"]
    path = out / "result.json"
    if not path.exists():
        return None
    result = read(path)
    if result["job"] != job or result["epochs"] != 50:
        raise RuntimeError("S1 result differs from its registered logical configuration")
    receipt = result["checkpoint"]
    if receipt["owner"] != generation or receipt["key"] not in owner_jobs:
        write(out / "attempts" / result["attempt_id"] / "lost_RAM_result.json", result)
        return None
    packet = verify(owner, receipt)
    if packet["job"] != job or packet["epoch"] != 50 or len(packet["epochs"]) != 50:
        raise RuntimeError("Final baseline packet differs from the scientific run identity")
    for kind in KINDS:
        best = packet["selected"][kind]
        if (v3.state_digest(best["model"]) != result["selected_hashes"][kind]
                or best["epoch"] != result["selected_epochs"][kind]
                or best["metrics"] != result["selected_validation"][kind]):
            raise RuntimeError("Selected-validation readback failed")
    del packet
    if set(result["anchors"]) != set(map(str, ANCHORS)):
        raise RuntimeError("Anchor index is incomplete")
    if set(result["epoch_hashes"]) != {f"epoch_{epoch:02d}.json" for epoch in range(1, 51)}:
        raise RuntimeError("Epoch receipt index is incomplete")
    if set(result["fork_hashes"]) != {f"forks_E{epoch:02d}.json" for epoch in ANCHORS}:
        raise RuntimeError("Fork receipt index is incomplete")
    for filename, expected in (result["epoch_hashes"] | result["fork_hashes"]).items():
        if sha(out / filename) != expected:
            raise RuntimeError(f"Scalar evidence changed: {filename}")
    from .diagnostics import FORKS

    for epoch in ANCHORS:
        anchor = result["anchors"][str(epoch)]
        if anchor["owner"] != generation:
            raise RuntimeError("Anchor refers to another RAM owner")
        packet = verify(owner, anchor)
        fork = read(out / f"forks_E{epoch:02d}.json")
        metadata = fork["metadata"]
        realized = [(branch["arm"], branch["scope"], branch["orientation"])
                    for branch in fork["purity"]["fork_states"]]
        if len(realized) != len(FORKS) or set(realized) != set(FORKS):
            raise RuntimeError("Fork receipt does not contain all 19 registered branches")
        if (packet["job"] != job or packet["epoch"] != epoch or metadata["anchor_hash"] != anchor["state_hash"]
                or metadata["anchor_key"] != anchor["key"] or fork["panel_hash"] != job["panel_hash"]
                or metadata["ledger_hash"] != packet["ledger_hash"]
                or metadata["loader_hash"] != v3.state_digest(packet["loader"])):
            raise RuntimeError("Forks do not bind to the verified baseline anchor and frozen panels")
        next_epoch = read(out / f"epoch_{epoch+1:02d}.json")
        hashes = [v3.state_digest(np.asarray(ids, dtype=np.int64)) for ids in fork["ordered_row_ids"]]
        actual = next_epoch["batches"][:2]
        if (metadata["next_permutation_hash"] != next_epoch["permutation_hash"]
                or metadata["row_id_hashes"] != hashes or hashes != [row["row_id_hash"] for row in actual]
                or metadata["batch_positions"] != [[row["start"], row["end"]] for row in actual]):
            raise RuntimeError("Fork inputs differ from the next actual baseline epoch")
    write(out / "COMPLETE_VERIFIED.json", dict(status="RAM_VERIFIED", owner=generation, result_sha256=sha(path),
                                               anchor_count=len(ANCHORS)))
    return result


def csv_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def publish(art, results):
    anchors, endpoints, costs = [], [], []
    for result in results:
        job = result["job"]
        meta = {key: job[key] for key in
                ("dataset", "seed", "arm", "logical_id", "source_hash", "data_hash", "config_hash")}
        out = art / "runs" / job["logical_id"]
        forks = {epoch: read(out / f"forks_E{epoch:02d}.json") for epoch in ANCHORS}
        accepted = {fork["metadata"]["fork_key"] for fork in forks.values()}
        attempts = out / "attempts"
        statuses = {}
        for path in sorted(attempts.glob("*.json")):
            attempt = read(path)
            statuses[attempt["attempt_id"]] = attempt["status"]
            costs.append(dict(meta, attempt_id=attempt["attempt_id"], cost_kind="baseline_attempt",
                              epoch=attempt["last_verified_epoch"], readout="online+fixed_ema",
                              status=attempt["status"], resumed_epoch=attempt["resumed_epoch"],
                              wall_seconds=attempt["wall_seconds"], max_rss_bytes=attempt["max_rss_bytes"],
                              peak_allocated_bytes=attempt["peak_allocated_bytes"],
                              peak_reserved_bytes=attempt["peak_reserved_bytes"],
                              process_gpu_bytes=attempt["process_gpu_bytes"], **attempt["counts"]))
        for journal in sorted(attempts.glob("*/FORK_COSTS.json")):
            status = statuses.get(journal.parent.name, "WORKER_RECEIPT_MISSING")
            for row in read(journal)["costs"]:
                if row["fork_key"] not in accepted:
                    costs.append(meta | row | dict(cost_kind="fork_attempt", status=status,
                                                 readout=row["readout"] or "online+fixed_ema"))
        for epoch, fork in forks.items():
            receipt = result["anchors"][str(epoch)]
            anchors.append(dict(meta, epoch=epoch, receipt=receipt,
                                fork_sha256=result["fork_hashes"][f"forks_E{epoch:02d}.json"]))
            for row in fork["costs"]:
                scalars = {key: value for key, value in row.items() if key != "steps"}
                costs.append(meta | scalars | dict(cost_kind="fork", readout=row.get("readout") or "online+fixed_ema",
                                               ram_anchor_bytes=receipt["bytes"],
                                               process_gpu_bytes=fork["resources"]["process_gpu_bytes"]))
        for epoch in range(1, 51):
            record = read(out / f"epoch_{epoch:02d}.json")
            for kind in KINDS:
                endpoints.append(dict(meta, attempt_id=record["attempt_id"], epoch=epoch, readout=kind,
                                      split="validation", selected=epoch == result["selected_epochs"][kind],
                                      model_hash=result["selected_hashes"][kind] if epoch == result["selected_epochs"][kind] else "",
                                      **record["validation"][kind]))
    write(art / "ANCHOR_INDEX.json", dict(status="RAM_VERIFIED", anchors=anchors))
    csv_rows(art / "BASELINE_ENDPOINTS.csv", endpoints)
    csv_rows(art / "COSTS.csv", costs)


def supervise(art, owner, authority):
    from .training import request

    art.mkdir(exist_ok=True, parents=True)
    with (art / "S1_SUPERVISION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return dispatch(art, owner, authority, request)
        except Exception as error:
            status_path = art / "S1_RUN.json"
            if not status_path.exists() or read(status_path)["status"] == "RUNNING":
                write(status_path, dict(status="TECHNICAL_BLOCKED", time=time.time(), error=repr(error)))
            raise


def dispatch(art, owner, authority, request):
    generation = owner_identity(owner)
    if stopping(art):
        raise RuntimeError("D2 stop or pause control is active")
    frozen = jobs(authority)
    manifest = dict(stage="S1", jobs=frozen)
    manifest_path = art / "S1_RUN_MANIFEST.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise RuntimeError("Registered S1 matrix or source identity changed")
    write(manifest_path, manifest)
    write(art / "S1_RUN.json", dict(status="RUNNING", owner=generation, time=time.time()))
    owner_jobs = request(owner, "status")["jobs"]
    results = {}
    queue = []
    for job in frozen:
        logical_id = job["logical_id"]
        result = verified_result(art, owner, generation, job, owner_jobs)
        if result:
            results[logical_id] = result
        else:
            queue.append(job)

    events = art / "S1_ADMISSIONS.jsonl"
    policy_path = art / "S1_PHYSICAL_CAPACITY.json"
    policy = (read(policy_path) if policy_path.exists()
              else dict(oom=False, observed={}, oom_attempts=[]))
    profiles = {dataset: max(policy["observed"].get(dataset, 0), authority["profiles"][dataset])
                for dataset in DATASETS}
    live = {}
    disposition = None
    failures = []
    interrupted = False

    def stop_signal(_sig, _frame):
        nonlocal interrupted
        interrupted = True

    signals = {sig: signal.signal(sig, stop_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while queue or live:
            measurements = sample()
            for logical_id, entry in list(live.items()):
                proc, job, card, attempt, log, launched = entry
                profile = profiles[job["dataset"]]
                current = max((row["bytes"] for row in measurements["processes"] if row["pid"] == proc.pid),
                              default=0)
                if current > profile:
                    profiles[job["dataset"]] = current
                    policy["observed"] = profiles
                    write(policy_path, policy)
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                del live[logical_id]
                out = art / "runs" / logical_id
                attempt_path = out / "attempts" / f"{attempt}.json"
                if not attempt_path.exists():
                    journal = attempt_path.with_name(attempt) / "FORK_COSTS.json"
                    write(attempt_path, dict(job=job, attempt_id=attempt, status="PROCESS_EXIT_WITHOUT_RECEIPT",
                                             last_verified_epoch=None, resumed_epoch=None, counts={},
                                             diagnostic_counts=read(journal)["counts"] if journal.exists() else {},
                                             wall_seconds=time.monotonic()-launched, process_gpu_bytes=current,
                                             peak_allocated_bytes=None, peak_reserved_bytes=None, max_rss_bytes=None,
                                             failure=dict(cuda_oom=False, returncode=code)))
                cost = read(attempt_path)
                if cost:
                    profile = max(profile, cost["process_gpu_bytes"])
                    profiles[job["dataset"]] = max(profiles[job["dataset"]], profile)
                    policy["observed"] = profiles
                    write(policy_path, policy)
                append(events, dict(event="exit", logical_id=logical_id, attempt_id=attempt,
                                    returncode=code, gpu=card, telemetry=measurements))
                if not code and cost and cost["status"] == "COMPLETE":
                    result = verified_result(art, owner, generation, job, request(owner, "status")["jobs"])
                    if result is None:
                        raise RuntimeError("Completed worker has no verified state")
                    results[logical_id] = result
                else:
                    queue.append(job)
                    if cost and cost["failure"] and cost["failure"]["cuda_oom"]:
                        policy["oom"] = True
                        policy["observed"] = profiles
                        policy["oom_attempts"].append(dict(logical_id=logical_id, attempt_id=attempt,
                                                           dataset=job["dataset"], gpu=card))
                        write(policy_path, policy)
                        disposition = "PAUSED_RESOURCE"
                        failures.append(str(attempt_path))
                    elif not interrupted and not stopping(art) and disposition is None:
                        disposition = "TECHNICAL_BLOCKED"
                        failures.append(str(attempt_path))
            if interrupted or stopping(art):
                disposition = "PAUSED_RESOURCE"
            if disposition:
                for proc, *_ in live.values():
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGTERM)
                if live:
                    time.sleep(1)
                    continue
                break
            starting = set()
            for proc, job, card, attempt, _log, _launched in live.values():
                path = art / "runs" / job["logical_id"] / "attempts" / attempt / "ALLOCATION.json"
                if startup_pending(path, proc.pid, measurements):
                    starting.add(card)
            for card in measurements["devices"]:
                if stopping(art) or interrupted or card["index"] in starting:
                    continue
                choice = None
                for job in queue:
                    physical = (authority["usable"].get(card["uuid"], 0), profiles[job["dataset"]]) if policy["oom"] else None
                    if admit(card, measurements["host_ram_used"], physical):
                        choice = job
                        break
                if choice is None:
                    continue
                logical_id = choice["logical_id"]
                attempt = uuid.uuid4().hex
                log_path = art / "runs" / logical_id / "attempts" / attempt / "worker.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("a")
                launched = time.monotonic()
                proc = subprocess.Popen(
                    [sys.executable, "-m", MODULE, "_worker", "--artifacts", str(art), "--guardian", str(owner),
                     "--logical-id", logical_id, "--attempt-id", attempt, "--parent", str(os.getpid())],
                    cwd=ROOT, env=os.environ | {"CUDA_VISIBLE_DEVICES": card["uuid"]},
                    stdout=log, stderr=subprocess.STDOUT)
                live[logical_id] = (proc, choice, card["index"], attempt, log, launched)
                queue.remove(choice)
                append(events, dict(event="admitted", logical_id=logical_id, dataset=choice["dataset"],
                                    attempt_id=attempt, gpu=card["index"], pid=proc.pid, telemetry=measurements,
                                    physical_fit=physical))
                break
            if queue and not live:
                disposition = "PAUSED_RESOURCE"
                break
            if live:
                time.sleep(1)
    except BaseException:
        disposition = "TECHNICAL_BLOCKED"
        raise
    finally:
        for proc, *_ in live.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc, _job, _card, _attempt, log, _launched in live.values():
            proc.wait()
            log.close()
        for sig, handler in signals.items():
            signal.signal(sig, handler)
        if disposition:
            write(art / "S1_RUN.json", dict(status=disposition, time=time.time(), owner=generation,
                                          completed=sorted(results), pending=[job["logical_id"] for job in queue],
                                          failures=failures))
    if disposition:
        return read(art / "S1_RUN.json")
    publish(art, [results[job["logical_id"]] for job in frozen])
    result = dict(status="COMPLETE", owner=generation, completed=sorted(results), time=time.time(),
                  trajectories=len(results), anchors=len(results) * len(ANCHORS),
                  forks=len(results) * len(ANCHORS) * 19, test_evaluations=0)
    write(art / "S1_RUN.json", result)
    return result
