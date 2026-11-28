"""Foreground, per-device dispatch using measured host and GPU telemetry."""
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from uuid import uuid4

from experiments.conditional_history_shared_update_20261122.protocol import append, read, write
from experiments.conditional_history_shared_update_20261122.scheduler import MIB, smi

from .protocol import ROOT, STUDY
from .ram_state import lock

WORKSPACES = (ROOT.with_name("phase0-3"), ROOT.with_name("phase0-3_tvg_rm_20261116"))
HISTORY = "history_information_20261124"
SENTINELS = ("USER_STOPPED", "PAUSE_TRAINING")
MODULE = "experiments.three_directions_20261128.runner"


def stop_controls(artifacts):
    scopes = [Path(artifacts), ROOT / "artifacts" / STUDY]
    scopes.extend(root / "artifacts" / study for root in WORKSPACES for study in (HISTORY, STUDY))
    return tuple(dict(status=status, path=str(scope / status))
                 for status in SENTINELS for scope in dict.fromkeys(scopes))


def stop_status(artifacts):
    for control in stop_controls(artifacts):
        if Path(control["path"]).exists():
            return control
    return None


def _proc_memory():
    fields = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal:", "MemAvailable:"))}
    return fields["MemTotal"], fields["MemAvailable"]


def _psutil_memory():
    import psutil
    mem = psutil.virtual_memory()
    return mem.total, mem.available


def _free_memory():
    lines = subprocess.check_output(["free", "-b"], text=True).splitlines()
    columns = lines[0].split()
    values = next(line.split()[1:] for line in lines[1:] if line.startswith("Mem:"))
    return int(values[columns.index("total")]), int(values[columns.index("available")])


def host_memory():
    errors = []
    for source, measure in (("/proc/meminfo", _proc_memory), ("psutil", _psutil_memory), ("free -b", _free_memory)):
        try:
            total, available = measure()
            if not 0 <= available <= total or total <= 0:
                raise ValueError("Invalid host memory total/available")
            return dict(source=source, total=total, available=available,
                        used_fraction=(total - available) / total, errors=errors)
        except (OSError, ValueError, KeyError, IndexError, ImportError, StopIteration, subprocess.SubprocessError) as exc:
            errors.append(dict(source=source, error=str(exc)))
    raise RuntimeError(f"Host available memory could not be measured: {errors}")


def eligible(gpu, host_fraction, *, usable=None, full_process=None, physical_fit=False):
    values = (host_fraction, gpu["used"], gpu["total"], gpu["watts"], gpu["limit"])
    if not all(math.isfinite(v) and v >= 0 for v in values) or not gpu["total"] or not gpu["limit"]:
        return False
    if host_fraction >= .90 or gpu["used"] >= .85 * gpu["total"] or gpu["watts"] >= .85 * gpu["limit"]:
        return False
    return not physical_fit or bool(usable and full_process and gpu["used"] + full_process <= usable)


def gpu_sample():
    devices, errors = [], []
    for raw in smi("--query-gpu=index,uuid,memory.used,memory.total,power.draw,power.limit"):
        try:
            i, uuid, used, total, watts, limit = raw
            devices.append(dict(index=int(i), uuid=uuid, used=int(used) * MIB,
                                total=int(total) * MIB, watts=float(watts), limit=float(limit)))
        except ValueError:
            errors.append(dict(query="GPU", raw=raw))
    processes = [dict(pid=int(pid), uuid=uuid, bytes=int(used) * MIB)
                 for pid, uuid, used in smi("--query-compute-apps=pid,gpu_uuid,used_memory")]
    if not devices:
        raise RuntimeError(f"No Nvidia GPU resource telemetry is available: {errors}")
    return dict(time=time.time(), devices=devices, processes=processes, errors=errors)


def sample():
    return dict(gpu_sample(), host=host_memory())


def residency(measurement, pid, gpu):
    values = [p["bytes"] for p in measurement["processes"] if p["pid"] == pid and p["uuid"] == gpu]
    if not values:
        raise RuntimeError(f"NVML process residency is unavailable for PID {pid} on {gpu}")
    return max(values)


def failed_allocation(error):
    match = re.search(r"Tried to allocate ([\d.]+) (GiB|MiB|KiB|bytes)", str(error))
    if match:
        number, unit = match.groups()
        return math.ceil(float(number) * {"GiB": 2**30, "MiB": 2**20, "KiB": 2**10, "bytes": 1}[unit])
    return None


def _quiescent(gpu, telemetry, active):
    return (not any(row["gpu"] == gpu["uuid"] for row in active)
            and not any(p["uuid"] == gpu["uuid"] for p in telemetry["processes"]))


def launch_plan(gpu, telemetry, active, queue, capacity):
    """Choose one job for the card and mark isolated post-OOM measurement when required."""
    uuid = gpu["uuid"]
    host_fraction = telemetry["host"]["used_fraction"]
    if not eligible(gpu, host_fraction):
        return None
    running = [row for row in active if row["gpu"] == uuid]
    if any(row["diagnostic"] or not row["allocated"] for row in running):
        return None
    gpu_capacity = capacity.get(uuid)
    if not gpu_capacity or not gpu_capacity["oom"]:
        return (queue[0], None) if queue else None
    for job in queue:
        receipt = gpu_capacity["measurements"].get(job.workload)
        target = gpu_capacity["needed_epoch"].get(job.workload, 1)
        if receipt and receipt["complete"] and receipt["epoch"] >= target:
            if eligible(gpu, host_fraction, usable=receipt["cuda_usable"], full_process=receipt["peak_process"],
                        physical_fit=True):
                return job, None
        elif _quiescent(gpu, telemetry, active):
            return job, target
    return None


def supervise(artifacts, stage, roster, jobs):
    from .runner import job_journal, publish_tables

    output = artifacts / stage
    output.mkdir(exist_ok=True)
    active = []
    telemetry_path = output / "DISPATCH.jsonl"
    capacity_path = artifacts / "CAPACITY.json"
    halt = None

    def stop_on_signal(sig, _frame):
        nonlocal halt
        halt = dict(status="USER_STOPPED", signal=signal.Signals(sig).name)

    def request_boundary():
        for row in active:
            if not row.get("stop_sent") and row["child"].poll() is None:
                row["child"].terminate()
                row["stop_sent"] = True

    with lock(artifacts / "SUPERVISION.lock", nonblocking=True):
        old_handlers = {sig: signal.signal(sig, stop_on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            write(output / "STOP_CONTROLS.json", dict(controls=stop_controls(artifacts)))
            halt = halt or stop_status(artifacts)
            queue = [] if halt else [job for job in jobs if not job_journal(artifacts, roster, job).completed()]
            capacity = read(capacity_path) if capacity_path.exists() else {}
            if queue:
                from torch import cuda
                if not cuda.is_available():
                    raise RuntimeError("CUDA is unavailable to the supervised runner")
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                devices = sample()["devices"]
                if visible is not None:
                    tokens = visible.split(",")
                    devices = [gpu for gpu in devices if str(gpu["index"]) in tokens or gpu["uuid"] in tokens]
                if not devices:
                    raise RuntimeError("No CUDA-visible device matches NVML telemetry")
                visible_uuids = {d["uuid"] for d in devices}
            while queue or active:
                marker = stop_status(artifacts)
                if marker and (halt is None or marker["status"] == "USER_STOPPED"):
                    halt = marker
                if halt:
                    request_boundary()
                for row in list(active):
                    child, invocation = row["child"], row["invocation"]
                    allocations = sorted((invocation / "allocations").glob("*.json"))
                    for path in allocations[row["allocations"]:]:
                        allocation = read(path)
                        row["allocated"] |= allocation["site"] == "first_completed_batch"
                        append(telemetry_path, dict(event="ALLOCATION_ACK", job=row["job"].key,
                                                    gpu=row["gpu"], site=allocation["site"], telemetry=sample()))
                    row["allocations"] = len(allocations)
                    if child.poll() is None:
                        continue
                    active.remove(row)
                    outcome = read(invocation / "OUTCOME.json") if (invocation / "OUTCOME.json").is_file() else {}
                    status = outcome.get("status")
                    job, gpu = row["job"], row["gpu"]
                    device = capacity.setdefault(gpu, dict(oom=False, measurements={}, needed_epoch={}))
                    measured = outcome.get("measurement")
                    if status == "CUDA_OOM":
                        device["oom"] = True
                        target = outcome["failed_epoch"]
                        device["needed_epoch"][job.workload] = max(target, device["needed_epoch"].get(job.workload, 1))
                        device["measurements"].pop(job.workload, None)
                        write(capacity_path, capacity)
                    elif measured and measured["complete"]:
                        device["measurements"][job.workload] = measured
                        write(capacity_path, capacity)
                    append(telemetry_path, dict(event="EXIT", job=job.key, gpu=gpu, returncode=child.returncode,
                                                invocation=str(invocation), outcome=outcome))
                    if halt:
                        continue
                    if status == "CUDA_OOM" and not row["diagnostic"]:
                        # An isolated retry at the same failing boundary identifies a capacity blocker.
                        queue.insert(0, job)
                    elif status == "DIAGNOSTIC_COMPLETE":
                        queue.insert(0, job)
                    elif status != "COMPLETE" or child.returncode != 0:
                        halt = dict(status="TECHNICAL_PAUSED", reason=status or "WORKER_EXIT",
                                    job=job.key, invocation=str(invocation))
                if halt:
                    if not active:
                        break
                    time.sleep(.25)
                    continue
                launched = False
                if queue:
                    # Re-read after every launch and after every observed first-allocation acknowledgement.
                    telemetry = sample()
                    for gpu in telemetry["devices"]:
                        if gpu["uuid"] not in visible_uuids:
                            continue
                        plan = launch_plan(gpu, telemetry, active, queue, capacity)
                        if plan is None:
                            continue
                        halt = halt or stop_status(artifacts)
                        if halt:
                            break
                        job, target = plan
                        invocation = output / "dispatch" / uuid4().hex
                        invocation.mkdir(parents=True)
                        argv = [sys.executable, "-m", MODULE, "--artifacts", str(artifacts), "--stage", stage,
                                "--worker", job.key, "--gpu", gpu["uuid"], "--invocation", str(invocation),
                                "--supervisor", str(os.getpid())]
                        if target is not None:
                            argv += ["--diagnostic-through", str(target)]
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"])
                        with (invocation / "worker.log").open("w") as log:
                            child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
                        active.append(dict(child=child, gpu=gpu["uuid"], job=job, invocation=invocation,
                                           allocated=False, allocations=0, diagnostic=target is not None))
                        queue.remove(job)
                        append(telemetry_path, dict(event="LAUNCH", job=job.key, gpu=gpu["uuid"], pid=child.pid,
                                                    diagnostic_through=target, telemetry=telemetry))
                        launched = True
                        break
                if not launched:
                    if queue and not active and all(
                            capacity.get(gpu["uuid"], {}).get("oom", False)
                            and _quiescent(gpu, telemetry, active)
                            and eligible(gpu, telemetry["host"]["used_fraction"])
                            for gpu in telemetry["devices"] if gpu["uuid"] in visible_uuids):
                        halt = dict(status="TECHNICAL_PAUSED", reason="MEASURED_CUDA_CAPACITY")
                    time.sleep(.5)
            result = dict(halt or dict(status="COMPLETE"), stage=stage, test_accesses=0)
            publish_tables(artifacts, stage, roster, jobs, result)
            return result
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            result = dict(status="TECHNICAL_PAUSED", stage=stage, reason=str(exc), test_accesses=0)
            write(output / "STAGE_STATUS.json", result)
            return result
        finally:
            request_boundary()
            for row in active:
                row["child"].wait()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
