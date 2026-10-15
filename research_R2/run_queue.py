"""Stage R2 queue driver.

Three subcommands, each scoped to a single Stage R2 phase.  The driver
explicitly does NOT auto-progress through phases -- the caller invokes
``r2a`` until completion, then ``decide``, then ``r2b`` (or skips the
latter).  This matches the Stage R2 protocol's fail-closed posture:
missing or invalid runs must stop the dataset's queue before any
future schedule is queued.

Concurrency
-----------
* At most **two** Python sub-processes are alive at any moment.
* The R2-A scheduling is fixed order per dataset:

      B0(2022) -> B0(2023) -> G1 replay -> B1(2022) -> B1(2023)
                                        -> B2(2022) -> B2(2023).

* When ``--dataset both`` is selected, the two datasets proceed in lock
  step, paired at each scheduling position.  KuaiRand and MovieLens
  run simultaneously (one process each) so the wall-time roughly
  matches a single dataset's duration.

Subcommands
-----------
``r2a``    Process the entire R2-A queue (B0 -> G1 -> B1 -> B2).
``decide`` Compute the validation-only decision.  Fails closed when
           any required run is missing / invalid.
``r2b``    Process R2-B (B3/B4) only when the dataset's decision is YES.

State machine
-------------
Per-run state is read from ``run_manifest.json``.  Possible values
emitted by this driver:

    * ``PLANNED`` -- queued but not yet picked up.
    * ``RUNNING`` -- a Python child process has been spawned.
    * ``COMPLETE`` -- run_formal terminated successfully.
    * ``FAILED``   -- run_formal aborted with a regular exception.
    * ``INVALID``  -- run_formal aborted with a FloatingPointError,
                      or any G1-3 violation.
    * ``NOT_SCHEDULED`` -- R2-B arm refused because decision=NO.

CLI examples::

    # Run the full R2-A queue on both datasets, in lock-step pairs.
    venv/Scripts/python.exe research_R2/run_queue.py r2a --dataset both

    # Compute the per-dataset decision manifest.
    venv/Scripts/python.exe research_R2/run_queue.py decide \\
        --dataset kuairand --runs-root research_R2/runs

    # If the decision was YES, run R2-B on both datasets.
    venv/Scripts/python.exe research_R2/run_queue.py r2b --dataset both

Hidden flag (orchestration self-test only)::

    --self-test   reduces every run to epochs=1 via the run_formal
                  command-line.  Production entries never expose this
                  flag; it is consumed only when ``$R2_SELF_TEST`` is
                  set in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
R2_SRC = os.path.join(HERE, "src")
PROJECT_ROOT = os.path.dirname(HERE)
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")
M3OE_ROOT = os.path.join(PROJECT_ROOT, "m3oe")
for p in (R2_SRC, PROJECT_ROOT, EXPERIMENTS_ROOT, M3OE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from stage_E1_instrumentation import write_json  # noqa: E402

from ao04_v12_protocol import (  # noqa: E402
    BUDGET_PER_DATASET,
    DATASETS,
    DATASET_SETTINGS,
    R2A_ARMS,
    R2B_ARMS,
    SEEDS,
    arm_supported,
    dataset_supported,
)


HERE_DIR = os.path.abspath(os.path.dirname(__file__))
PYTHON_EXE = sys.executable


# ---------------------------------------------------------------------------
# Data structures.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueueKey:
    dataset: str  # 'kuairand' or 'movielens'
    arm: str  # 'B0'..'B4'
    seed: int  # 2022 or 2023

    def relative_run_dir(self, runs_root: str) -> str:
        """Return absolute per-run directory under ``runs_root``.

        Matches Specification \u00a77 layout:

            <runs_root>/<dataset>/<setting>/m3oe/<arm>/seed<seed>/attempt1/
        """

        return os.path.join(runs_root, self.dataset,
                            DATASET_SETTINGS[self.dataset][0],
                            "m3oe", self.arm,
                            "seed{}".format(int(self.seed)), "attempt1")

    @property
    def config_path(self) -> str:
        if self.dataset == "kuairand":
            return os.path.join(HERE_DIR, "configs",
                                 "kuairand_natural_m3oe.json")
        if self.dataset == "movielens":
            return os.path.join(HERE_DIR, "configs",
                                 "movielens_d0_large_m3oe.json")
        raise ValueError("unknown dataset {}".format(self.dataset))

    @property
    def setting(self) -> str:
        return DATASET_SETTINGS[self.dataset][0]


@dataclass
class RunRecord:
    """In-memory bookkeeping for a single run."""

    key: QueueKey
    status: str = "PLANNED"
    rc: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    log_path: Optional[str] = None

    def is_terminal(self) -> bool:
        return self.status in ("COMPLETE", "FAILED", "INVALID",
                               "NOT_SCHEDULED")


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _read_manifest(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as h:
            return json.load(h)
    except Exception:
        return {}


def _read_status(run_dir: str) -> str:
    return str(_read_manifest(run_dir).get("status", "")).upper()


def _is_completed(run_dir: str) -> bool:
    return _read_status(run_dir) == "COMPLETE"


def _is_valid(run_dir: str) -> bool:
    return _is_completed(run_dir)


def _spawn_run_formal(key: QueueKey, *, runs_root: str, epochs: int = 50,
                       init_dir: Optional[str] = None,
                       extra_argv: Sequence[str] = ()
                       ) -> subprocess.Popen:
    """Spawn run_formal.py for one (dataset, arm, seed)."""

    abs_run_dir = key.relative_run_dir(runs_root)
    init_dir_str = (init_dir or os.path.abspath(os.path.join(HERE_DIR, "init")))
    # The stub runner exists only for the orchestration self-test.  It is
    # honoured solely when the caller explicitly armed self-test mode via
    # the R2_SELF_TEST environment variable — a leaked R2_STUB_FORMAL in
    # a production shell must never silently replace real training.
    self_test_armed = bool(os.environ.get("R2_SELF_TEST"))
    stub_path = os.environ.get("R2_STUB_FORMAL")
    if self_test_armed and stub_path and os.path.exists(stub_path):
        runner = stub_path
    else:
        runner = os.path.join(HERE_DIR, "run_formal.py")
    argv = [
        PYTHON_EXE,
        runner,
        "--config", os.path.abspath(key.config_path),
        "--arm", key.arm,
        "--seed", str(int(key.seed)),
        "--run-dir", abs_run_dir,
        "--init-dir", init_dir_str,
        "--epochs", str(int(epochs)),
    ]
    argv.extend(list(extra_argv))
    print("[queue] spawn: {}".format(" ".join(argv)))
    proc = subprocess.Popen(argv)  # noqa: S603
    return proc


def _spawn_g1_replay(key: QueueKey, *, runs_root: str, write_to: str
                      ) -> subprocess.Popen:
    argv = [
        PYTHON_EXE,
        os.path.join(HERE_DIR, "run_g1_replay.py"),
        "--config", os.path.abspath(key.config_path),
        "--run-dir", key.relative_run_dir(runs_root),
        "--seed", str(int(key.seed)),
        "--write-to", os.path.abspath(write_to),
    ]
    print("[queue] spawn (G1): {}".format(" ".join(argv)))
    return subprocess.Popen(argv)  # noqa: S603


def _wait_all(procs: Mapping[Any, subprocess.Popen]) -> Dict[Any, int]:
    """Wait for every process to exit and return their return codes."""

    rcs: Dict[Any, int] = {}
    for k, proc in procs.items():
        rc = proc.wait()
        rcs[k] = int(rc)
    return rcs


def _wait_pair(proc_a: subprocess.Popen, proc_b: Optional[subprocess.Popen]
                ) -> Tuple[int, int]:
    """Wait for two processes; the queue runs them in lock-step."""

    rc_a = proc_a.wait()
    rc_b = proc_b.wait() if proc_b is not None else 0
    return rc_a, rc_b


# ---------------------------------------------------------------------------
# Schedule definitions.
# ---------------------------------------------------------------------------

def _r2a_schedule(dataset: str) -> List[Tuple[str, str]]:
    """Return ordered list of (phase, tag) for R2-A.

    Phase = 'run' for train_val executions; 'g1' for the G1 replay.
    Tag  = a human label (e.g. ``B0/seed2022``).

    Order is fixed by Specification §9/§10: both B0 runs first, then the
    G1 replay gate, then B1/B2 — B1/B2 may only start after G1 PASSes.
    """

    schedule: List[Tuple[str, str]] = []
    for seed in SEEDS:
        schedule.append(("run", "B0/seed{}".format(seed)))
    schedule.append(("g1", "G1-replay"))
    for arm in R2A_ARMS:
        if arm == "B0":
            continue
        for seed in SEEDS:
            schedule.append(("run", "{}/seed{}".format(arm, seed)))
    return schedule


def _r2b_schedule(dataset: str) -> List[Tuple[str, str]]:
    schedule: List[Tuple[str, str]] = []
    for arm in R2B_ARMS:
        for seed in SEEDS:
            schedule.append(("run", "{}/seed{}".format(arm, seed)))
    return schedule


def _key_for(dataset: str, arm: str, seed: int) -> QueueKey:
    if not dataset_supported(dataset):
        raise ValueError("unsupported dataset {}".format(dataset))
    if not arm_supported(arm):
        raise ValueError("unsupported arm {}".format(arm))
    if int(seed) not in SEEDS:
        raise ValueError("seed must be 2022 or 2023")
    return QueueKey(dataset=dataset, arm=arm, seed=int(seed))


# ---------------------------------------------------------------------------
# Status emission: queue.log / queue_state.json.
# ---------------------------------------------------------------------------

def _write_queue_state(state_path: str, state: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
    write_json(state_path, dict(state))


def _log_event(queue_state: Dict[str, Any], kind: str, detail: str) -> None:
    """Append to the ordered audit event log.

    ``per_run`` is serialised with sorted keys, so dict order cannot
    serve as execution-order evidence; this append-only log preserves
    the true schedule order (run_start/run_end/g1_start/g1_end).
    """

    log = queue_state.setdefault("event_log", [])
    log.append({"seq": len(log), "at": _now_iso(),
                "kind": kind, "detail": detail})


def _update_queue_record(records: Dict[str, RunRecord], key: QueueKey
                          ) -> RunRecord:
    rec = records.setdefault(str(key), RunRecord(key=key))
    return rec


# ---------------------------------------------------------------------------
# Subcommand: r2a.
# ---------------------------------------------------------------------------

def _r2a_for_one_dataset(dataset: str, *,
                          runs_root: str, init_dir: str,
                          epochs: int, queue_state: Dict[str, Any],
                          dry_run: bool
                          ) -> Tuple[bool, Dict[str, RunRecord]]:
    """Run R2-A on a single dataset.  Returns (continue?, records)."""

    records: Dict[str, RunRecord] = {}
    schedule = _r2a_schedule(dataset)
    print("[queue] R2-A dataset={} schedule={}".format(dataset, schedule))

    for phase, tag in schedule:
        if phase == "run":
            arm, _, seed_str = tag.partition("/seed")
            seed = int(seed_str)
            key = _key_for(dataset, arm, seed)
            rec = records[str(key)] = RunRecord(key=key, status="PLANNED")
            queue_state.setdefault("per_run", {})[str(key)] = {
                "dataset": dataset, "arm": arm, "seed": seed,
                "status": rec.status,
            }
            queue_state["status"] = "RUNNING"
            queue_state["active_phase"] = "r2a"
            queue_state["active_step"] = "{}/seed{}".format(arm, seed)
            _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                                queue_state)
            if dry_run:
                print("[queue] dry-run: skipping spawn {}/seed{}".format(
                    arm, seed))
                rec.status = "DRY"
                continue
            _log_event(queue_state, "run_start", tag)
            proc = _spawn_run_formal(key, runs_root=runs_root, epochs=epochs, init_dir=init_dir)
            rec.started_at = _now_iso()
            rec.status = "RUNNING"
            queue_state["per_run"][str(key)]["status"] = "RUNNING"
            _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                                queue_state)
            rc = proc.wait()
            rec.finished_at = _now_iso()
            rec.rc = int(rc)
            # Refresh status from manifest written by run_formal.
            manifest = _read_manifest(key.relative_run_dir(runs_root))
            rec.status = str(manifest.get("status", "UNKNOWN")).upper() or "UNKNOWN"
            _log_event(queue_state, "run_end",
                       "{} status={} rc={}".format(tag, rec.status, rc))
            if rec.status == "COMPLETE":
                # SUCCESS path -> next step.
                pass
            else:
                # FAILED / INVALID -> stop this dataset's queue, mark INVALID.
                print("[queue] {} arm={} seed={} -> status={} rc={}".format(
                    dataset, arm, seed, rec.status, rc))
                queue_state["status"] = "INVALID"
                queue_state["active_step"] = "stopped-after-failed-run"
                queue_state["per_run"][str(key)] = {
                    "dataset": dataset, "arm": arm, "seed": seed,
                    "status": rec.status, "rc": int(rc),
                    "started_at": rec.started_at,
                    "finished_at": rec.finished_at,
                }
                _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                                    queue_state)
                return False, records
            queue_state["per_run"][str(key)] = {
                "dataset": dataset, "arm": arm, "seed": seed,
                "status": "COMPLETE",
                "rc": int(rc),
                "started_at": rec.started_at,
                "finished_at": rec.finished_at,
            }
            _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                                queue_state)
        elif phase == "g1":
            # G1 replay (per seed pair) – emit two replay files (one per
            # seed), but both must PASS for the dataset queue to continue.
            for seed in SEEDS:
                key = _key_for(dataset, "B0", seed)
                rec = records[str(key)] = RunRecord(key=key, status="PLANNED")
                if dry_run:
                    print("[queue] dry-run: skipping G1 replay B0/seed{}".format(
                        seed))
                    rec.status = "DRY"
                    continue
                write_to = os.path.abspath(
                    os.path.join(key.relative_run_dir(runs_root), "g1_replay.json"))
                _log_event(queue_state, "g1_start",
                           "B0/seed{}".format(seed))
                proc = _spawn_g1_replay(key, runs_root=runs_root, write_to=write_to)
                rec.started_at = _now_iso()
                rec.status = "RUNNING"
                rc = proc.wait()
                rec.finished_at = _now_iso()
                rec.rc = int(rc)
                # Status = PASS iff rc == 0.
                try:
                    with open(write_to, encoding="utf-8") as h:
                        rep = json.load(h)
                except Exception:
                    rep = {}
                rec_status = str(rep.get("status", "FAIL")).upper()
                rec.status = "COMPLETE" if rec_status == "PASS" else "INVALID"
                _log_event(queue_state, "g1_end",
                           "B0/seed{} status={}".format(seed, rec.status))
                queue_state["per_run"][str(key) + "/g1"] = {
                    "dataset": dataset, "arm": "B0", "seed": seed,
                    "status": rec.status, "rc": int(rc),
                    "report": str(write_to),
                }
                _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                                    queue_state)
                if rec.status != "COMPLETE":
                    queue_state["status"] = "INVALID"
                    queue_state["active_step"] = (
                        "stopped-after-g1-replay-failure")
                    _write_queue_state(
                        os.path.join(runs_root, "queue_state.json"),
                        queue_state)
                    return False, records
        else:
            raise RuntimeError("unknown schedule phase {!r}".format(phase))

    # All runs + G1 PASSED -> dataset R2-A queue is done.
    queue_state["status"] = "COMPLETE"
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)
    return True, records


def cmd_r2a(args: argparse.Namespace) -> int:
    """Implementation of the ``r2a`` subcommand."""

    epochs = _resolve_epochs(args)
    datasets = _select_datasets(args.dataset)
    runs_root = os.path.abspath(args.runs_root)
    init_dir = os.path.abspath(args.init_dir)

    queue_state: Dict[str, Any] = {
        "phase": "r2a", "created_at": _now_iso(),
        "status": "PLANNED",
        "datasets": datasets,
        "per_run": {},
        "epochs": int(epochs),
    }
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)

    if len(datasets) == 1:
        ok, _ = _r2a_for_one_dataset(
            datasets[0], runs_root=runs_root, init_dir=init_dir,
            epochs=epochs, queue_state=queue_state, dry_run=args.dry_run)
        return 0 if ok else 2

    if args.dry_run:
        # The paired path has no dry-run branch; refuse rather than
        # silently spawning real training under a dry-run flag.
        print("[queue] --dry-run supports single-dataset r2a only")
        return 2

    # Two-dataset mode: pair them at each scheduling position.  At most
    # two Python child processes are alive at any moment.  Both datasets
    # use the identical schedule, so a position-wise zip pairs the same
    # (phase, arm, seed) across datasets: (B0,2022), (B0,2023), G1,
    # (B1,2022), (B1,2023), (B2,2022), (B2,2023).
    print("[queue] R2-A both-datasets, lock-step pair scheduling")
    queue_state["status"] = "RUNNING"
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)
    ok_a = True
    ok_b = True
    paired_schedule_all = list(zip(_r2a_schedule(datasets[0]),
                                    _r2a_schedule(datasets[1])))

    # Execute in order, but each pair concurrently.  We preserve the
    # gate order (run ... run, gate, run, gate, run, gate, ...).
    for slot_a, slot_b in paired_schedule_all:
        # If one side is a 'run', spawn both; if one side is 'g1', spawn
        # both g1 replays.
        if slot_a[0] == "run":
            arm_a, _, seed_a = slot_a[1].partition("/seed")
            arm_b, _, seed_b = slot_b[1].partition("/seed")
            key_a = _key_for(datasets[0], arm_a, int(seed_a))
            key_b = _key_for(datasets[1], arm_b, int(seed_b))
            ok_a_step, ok_b_step = _paired_run(
                key_a, key_b,
                epochs=epochs,
                init_dir=init_dir,
                runs_root=runs_root,
                queue_state=queue_state)
            ok_a = ok_a and ok_a_step
            ok_b = ok_b and ok_b_step
            if not (ok_a_step and ok_b_step):
                _write_queue_state(
                    os.path.join(runs_root, "queue_state.json"),
                    queue_state)
                return 2
        elif slot_a[0] == "g1":
            for seed in SEEDS:
                key_a = _key_for(datasets[0], "B0", seed)
                key_b = _key_for(datasets[1], "B0", seed)
                step_ok_a, step_ok_b = _paired_g1(key_a, key_b, runs_root,
                                                    queue_state)
                ok_a = ok_a and step_ok_a
                ok_b = ok_b and step_ok_b
                if not (step_ok_a and step_ok_b):
                    queue_state["status"] = "INVALID"
                    queue_state["active_step"] = (
                        "stopped-after-g1-replay-failure")
                    _write_queue_state(
                        os.path.join(runs_root, "queue_state.json"),
                        queue_state)
                    return 2
        else:
            raise RuntimeError("unknown slot phase")

    queue_state["status"] = "COMPLETE"
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)
    return 0


def _paired_run(key_a: QueueKey, key_b: QueueKey, *,
                 epochs: int, init_dir: str, runs_root: str,
                 queue_state: Mapping[str, Any]) -> Tuple[bool, bool]:
    """Run two (arm, seed) train_val steps concurrently.

    Returns (ok_a, ok_b).  Spawns up to two child processes; never
    exceeds the documented cap.  Failure on either side stops both
    queues.
    """

    queue_state["per_run"][str(key_a)] = {"dataset": key_a.dataset,
                                            "arm": key_a.arm,
                                            "seed": key_a.seed,
                                            "status": "RUNNING"}
    queue_state["per_run"][str(key_b)] = {"dataset": key_b.dataset,
                                            "arm": key_b.arm,
                                            "seed": key_b.seed,
                                            "status": "RUNNING"}
    queue_state["active_step"] = "{}/{}/seed{} | {}/{}/seed{}".format(
        key_a.dataset, key_a.arm, key_a.seed,
        key_b.dataset, key_b.arm, key_b.seed)
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)

    proc_a = _spawn_run_formal(key_a, runs_root=runs_root, epochs=epochs, init_dir=init_dir)
    proc_b = _spawn_run_formal(key_b, runs_root=runs_root, epochs=epochs, init_dir=init_dir)
    _log_event(queue_state, "run_start",
               "{}/seed{} | {}/seed{}".format(key_a.arm, key_a.seed,
                                              key_b.arm, key_b.seed))
    rc_a, rc_b = _wait_pair(proc_a, proc_b)
    ok_a = _refresh_run_status(key_a, rc_a, queue_state, runs_root)
    ok_b = _refresh_run_status(key_b, rc_b, queue_state, runs_root)
    _log_event(queue_state, "run_end",
               "{} ok_a={} ok_b={}".format(queue_state["active_step"],
                                           ok_a, ok_b))
    return ok_a, ok_b


def _paired_g1(key_a: QueueKey, key_b: QueueKey, runs_root: str,
                queue_state: Mapping[str, Any]) -> Tuple[bool, bool]:
    queue_state["active_step"] = "G1-replay {}/seed{} | {}/seed{}".format(
        key_a.dataset, key_a.seed, key_b.dataset, key_b.seed)
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)
    write_to_a = os.path.abspath(
        os.path.join(key_a.relative_run_dir(runs_root), "g1_replay.json"))
    write_to_b = os.path.abspath(
        os.path.join(key_b.relative_run_dir(runs_root), "g1_replay.json"))
    proc_a = _spawn_g1_replay(key_a, runs_root=runs_root, write_to=write_to_a)
    proc_b = _spawn_g1_replay(key_b, runs_root=runs_root, write_to=write_to_b)
    _log_event(queue_state, "g1_start",
               "B0/seed{} | B0/seed{}".format(key_a.seed, key_b.seed))
    rc_a, rc_b = _wait_pair(proc_a, proc_b)

    def _g1_ok(write_to: str) -> bool:
        try:
            with open(write_to, encoding="utf-8") as h:
                rep = json.load(h)
        except Exception:
            return False
        return str(rep.get("status", "FAIL")).upper() == "PASS"

    ok_a = _g1_ok(write_to_a)
    ok_b = _g1_ok(write_to_b)
    _log_event(queue_state, "g1_end",
               "B0/seed{} | B0/seed{} ok_a={} ok_b={}".format(
                   key_a.seed, key_b.seed, ok_a, ok_b))
    queue_state["per_run"][str(key_a) + "/g1"] = {
        "dataset": key_a.dataset, "arm": "B0", "seed": key_a.seed,
        "status": "COMPLETE" if ok_a else "INVALID",
        "rc": int(rc_a), "report": write_to_a,
    }
    queue_state["per_run"][str(key_b) + "/g1"] = {
        "dataset": key_b.dataset, "arm": "B0", "seed": key_b.seed,
        "status": "COMPLETE" if ok_b else "INVALID",
        "rc": int(rc_b), "report": write_to_b,
    }
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)
    return ok_a, ok_b


def _refresh_run_status(key: QueueKey, rc: int, queue_state: Mapping[str, Any],
                         runs_root: str) -> bool:
    """Update queue_state with the post-run manifest status.  Return True iff COMPLETE."""

    run_dir = key.relative_run_dir(runs_root)
    manifest = _read_manifest(run_dir)
    status = str(manifest.get("status", "UNKNOWN")).upper() or "UNKNOWN"
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    queue_state["per_run"][str(key)] = {
        "dataset": key.dataset, "arm": key.arm, "seed": key.seed,
        "status": status, "rc": int(rc),
        "manifest": manifest_path,
    }
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)
    return status == "COMPLETE"


# ---------------------------------------------------------------------------
# Subcommand: decide.
# ---------------------------------------------------------------------------

def cmd_decide(args: argparse.Namespace) -> int:
    """Compute the validation-only R2-A decision for one dataset."""

    dataset = args.dataset
    if dataset not in DATASETS:
        print("[queue] unsupported dataset {}".format(dataset))
        return 2
    runs_root = os.path.abspath(args.runs_root)
    decisions_root = os.path.abspath(args.decisions_root)
    os.makedirs(decisions_root, exist_ok=True)

    # All six R2-A runs (B0/B1/B2 × 2022/2023) must exist and be COMPLETE.
    run_dirs: Dict[Tuple[str, int], str] = {}
    missing = []
    for arm in R2A_ARMS:
        for seed in SEEDS:
            key = _key_for(dataset, arm, seed)
            run_dir = key.relative_run_dir(runs_root)
            run_dirs[(arm, int(seed))] = run_dir
            manifest = _read_manifest(run_dir)
            status = str(manifest.get("status", "")).upper()
            if status != "COMPLETE":
                missing.append((arm, int(seed), status))
    if missing:
        print("[queue] decide: refusing -- missing/invalid runs:")
        for arm, seed, status in missing:
            print("  arm={} seed={} status={}".format(arm, seed, status))
        # Write a stub manifest under decisions/ capturing the refusal.
        ref_path = os.path.join(decisions_root,
                                 "r2a_decision_{}_REFUSED.json".format(dataset))
        write_json(ref_path, {"dataset": dataset,
                              "status": "REFUSED",
                              "reason": "missing-or-invalid-runs",
                              "missing": [{"arm": a, "seed": s,
                                             "status": st}
                                            for a, s, st in missing],
                              "created_at": _now_iso()})
        return 2

    # Defer the actual call to the decision module.  We re-import to
    # avoid pulling sys.path into the parent at import time.
    sys.path.insert(0, R2_SRC)
    from ao04_v12_decision import compute_r2a_decision
    write_to = os.path.join(decisions_root,
                             "r2a_decision_{}.json".format(dataset))
    code_hashes = {}
    for name in (
        "ao04_v12_model.py",
        "ao04_v12_protocol.py",
        "ao04_v12_runner.py",
        "ao04_v12_decision.py",
        "ao04_v12_test_evaluator.py",
        "ao04_v12_static_check.py",
        "ao04_v12_hpost.py",
    ):
        code_hashes[os.path.join("src", name)] = _file_sha256(
            os.path.join(R2_SRC, name))
    for name in ("run_formal.py", "run_g1_replay.py", "run_queue.py"):
        code_hashes[name] = _file_sha256(os.path.join(HERE_DIR, name))
    result = compute_r2a_decision(dataset, run_dirs,
                                    write_to=write_to,
                                    code_hashes=code_hashes)
    print("[queue] decide: dataset={} decision={} signal={} "
          "valid={}".format(dataset, result.decision,
                              result.signal_class, result.valid))
    return 0


def _file_sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Subcommand: r2b.
# ---------------------------------------------------------------------------

def _read_decision(dataset: str, decisions_root: str) -> Dict[str, Any]:
    path = os.path.join(decisions_root,
                         "r2a_decision_{}.json".format(dataset))
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as h:
            return json.load(h)
    except Exception:
        return {}


def _r2b_for_one_dataset(dataset: str, *, runs_root: str, init_dir: str,
                          decisions_root: str,
                          epochs: int, queue_state: Dict[str, Any]
                          ) -> int:
    decision = _read_decision(dataset, decisions_root=decisions_root)
    decision_value = str(decision.get("decision", "")).upper() or "NO"
    if decision_value != "YES":
        print("[queue] R2-B: dataset={} decision={} -> NOT SCHEDULED".format(
            dataset, decision_value))
        queue_state["phase"] = "r2b"
        queue_state["status"] = "NOT_SCHEDULED"
        queue_state["active_step"] = "decision was not YES"
        _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                            queue_state)
        return 0

    queue_state["phase"] = "r2b"
    queue_state["status"] = "RUNNING"
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)
    for arm in R2B_ARMS:
        for seed in SEEDS:
            key = _key_for(dataset, arm, seed)
            proc = _spawn_run_formal(key, runs_root=runs_root, epochs=epochs, init_dir=init_dir)
            proc.wait()
            ok = _refresh_run_status(key, proc.returncode, queue_state,
                                      runs_root)
            if not ok:
                queue_state["status"] = "INVALID"
                _write_queue_state(
                    os.path.join(runs_root, "queue_state.json"),
                    queue_state)
                return 2
    queue_state["status"] = "COMPLETE"
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)
    return 0


def cmd_r2b(args: argparse.Namespace) -> int:
    epochs = _resolve_epochs(args)
    datasets = _select_datasets(args.dataset)
    runs_root = os.path.abspath(args.runs_root)
    init_dir = os.path.abspath(args.init_dir)
    decisions_root = os.path.abspath(getattr(args, "decisions_root",
                                             "research_R2/decisions"))
    queue_state: Dict[str, Any] = {"phase": "r2b", "status": "PLANNED",
                                     "datasets": datasets, "per_run": {}}
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)
    if len(datasets) == 1:
        return _r2b_for_one_dataset(datasets[0], runs_root=runs_root,
                                      init_dir=init_dir,
                                      decisions_root=decisions_root,
                                      epochs=epochs,
                                      queue_state=queue_state)
    # Two datasets: paired mode.  up to two concurrent.
    # Check decision manifests BEFORE spawning anything.
    for ds in datasets:
        dec = _read_decision(ds, decisions_root=decisions_root)
        if str(dec.get("decision", "")).upper() != "YES":
            print("[queue] R2-B: dataset={} decision != YES -> NOT SCHEDULED".format(
                ds))
            queue_state["status"] = "NOT_SCHEDULED"
            queue_state["active_step"] = "decision != YES: refused"
            _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                                queue_state)
            return 0
    rc_per_dataset: Dict[str, int] = {}
    paired_schedule = list(zip(*[_r2b_schedule(d) for d in datasets]))
    queue_state["status"] = "RUNNING"
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)
    for slot_a, slot_b in paired_schedule:
        arm_a, _, seed_a = slot_a[1].partition("/seed")
        arm_b, _, seed_b = slot_b[1].partition("/seed")
        key_a = _key_for(datasets[0], arm_a, int(seed_a))
        key_b = _key_for(datasets[1], arm_b, int(seed_b))
        queue_state["per_run"][str(key_a)] = {"dataset": key_a.dataset,
                                                "arm": key_a.arm,
                                                "seed": key_a.seed,
                                                "status": "RUNNING"}
        queue_state["per_run"][str(key_b)] = {"dataset": key_b.dataset,
                                                "arm": key_b.arm,
                                                "seed": key_b.seed,
                                                "status": "RUNNING"}
        _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                            queue_state)
        proc_a = _spawn_run_formal(key_a, runs_root=runs_root, epochs=epochs, init_dir=init_dir)
        proc_b = _spawn_run_formal(key_b, runs_root=runs_root, epochs=epochs, init_dir=init_dir)
        rc_a, rc_b = _wait_pair(proc_a, proc_b)
        ok_a = _refresh_run_status(key_a, rc_a, queue_state, runs_root)
        ok_b = _refresh_run_status(key_b, rc_b, queue_state, runs_root)
        if not ok_a or not ok_b:
            queue_state["status"] = "INVALID"
            _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                                queue_state)
            return 2
    queue_state["status"] = "COMPLETE"
    _write_queue_state(os.path.join(runs_root, "queue_state.json"),
                        queue_state)
    return 0


# ---------------------------------------------------------------------------
# Helpers shared by subcommands.
# ---------------------------------------------------------------------------

def _resolve_epochs(args: argparse.Namespace) -> int:
    if not getattr(args, "self_test", False):
        # Formal mode requires exactly 50 epochs for every run; the
        # queue refuses the hidden 1-epoch flag.
        return 50
    if not os.environ.get("R2_SELF_TEST"):
        raise SystemExit(
            "--self-test requires R2_SELF_TEST=1 in the environment; "
            "refusing to shorten formal runs otherwise")
    return 1


def _select_datasets(arg: str) -> List[str]:
    if arg == "both":
        return ["kuairand", "movielens"]
    if arg in DATASETS:
        return [arg]
    raise ValueError("unknown dataset {!r}".format(arg))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ao04-queue",
                                 description="Stage R2 queue driver.")
    sub = p.add_subparsers(dest="subcommand", required=True)

    pa = sub.add_parser("r2a", help="Run the R2-A queue.")
    pa.add_argument("--dataset", choices=("kuairand", "movielens", "both"),
                     required=True)
    pa.add_argument("--runs-root", default="research_R2/runs")
    pa.add_argument("--init-dir", default="research_R2/init")
    pa.add_argument("--dry-run", action="store_true",
                     help="Skip run_formal spawn but emit queue_state.json.")

    pd = sub.add_parser("decide",
                        help="Compute the validation-only R2-A decision manifest.")
    pd.add_argument("--dataset", choices=("kuairand", "movielens"), required=True)
    pd.add_argument("--runs-root", default="research_R2/runs")
    pd.add_argument("--decisions-root", default="research_R2/decisions")

    pb = sub.add_parser("r2b", help="Run the R2-B queue (B3/B4).")
    pb.add_argument("--dataset", choices=("kuairand", "movielens", "both"),
                     required=True)
    pb.add_argument("--runs-root", default="research_R2/runs")
    pb.add_argument("--init-dir", default="research_R2/init")
    pb.add_argument("--decisions-root", default="research_R2/decisions")

    p.add_argument("--self-test", action="store_true",
                    help="Internal orchestration self-test: every run uses epochs=1.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.subcommand == "r2a":
        return cmd_r2a(args)
    if args.subcommand == "decide":
        return cmd_decide(args)
    if args.subcommand == "r2b":
        return cmd_r2b(args)
    raise ValueError("unknown subcommand")


if __name__ == "__main__":
    raise SystemExit(main())
