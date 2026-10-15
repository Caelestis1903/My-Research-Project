"""Stage R2 single-run formal entry point.

This is a thin orchestration module.  It does not implement any
algorithm itself; it invokes the already-reviewed components in
``research_R2/src`` and persists the immutable run-level artefacts.

CLI::

    venv/Scripts/python.exe research_R2/run_formal.py \\
        --config research_R2/configs/kuairand_natural_m3oe.json \\
        --arm B0 \\
        --seed 2022 \\
        --run-dir runs/kuairand/natural/m3oe/B0/seed2022/attempt1 \\
        [--epochs 1]            # hidden flag: only smoke / self-test
        [--init-dir research_R2/init]

Behaviour:

    1. Read the JSON config (every locked field stays at its protocol
       value; the orchestrator never overrides them).
    2. Ensure the (dataset, seed) paired init artefacts exist.  If not,
       run ``paired_init_phase`` once (one B0 construction + state save).
    3. Walk the run state machine: PLANNED -> RUNNING -> COMPLETE/FAILED/INVALID.
       The manifest is rewritten with each transition.
    4. Run ``train_val_loop`` with the strict-loaded B0 state and the
       supplied run_dir; the runner writes a FAILED/INVALID manifest on
       error.
    5. Run ``state_writer`` to persist selected / final / common
       checkpoints, optimizer state, run manifest, and CSVs.

Non-empty run_dir is refused (mirrors Stage S1).  Every sub-process
stdout / stderr are redirected to the run_dir for audit; nothing in the
training path touches any test fixture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH_R2 = HERE
R2_SRC = os.path.join(RESEARCH_R2, "src")
PROJECT_ROOT = os.path.dirname(RESEARCH_R2)
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")
M3OE_ROOT = os.path.join(PROJECT_ROOT, "m3oe")
for p in (R2_SRC, PROJECT_ROOT, EXPERIMENTS_ROOT, M3OE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from stage_E1_instrumentation import write_json  # noqa: E402

from ao04_v12_protocol import (  # noqa: E402
    ARM_SET,
    BUDGET_PER_DATASET,
    DATASETS,
    DATASET_SETTINGS,
    DEFAULT_DEVICE,
    EPOCHS,
    SEEDS,
    arm_supported,
    dataset_supported,
)
from ao04_v12_runner import (  # noqa: E402
    paired_init_phase,
    state_writer,
    train_val_loop,
)


# Hidden flag: lets the orchestration layer self-test the queue ordering
# without performing the formal 50-epoch run.  Never documented in normal
# help; the queue driver also rejects this flag outside --self-test runs.
SMOKE_EPOCHS = 1


# ---------------------------------------------------------------------------
# Manifest helpers.
# ---------------------------------------------------------------------------

def _manifest_path(run_dir: str) -> str:
    return os.path.join(run_dir, "run_manifest.json")


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _write_status(run_dir: str, status: str, *,
                   reason: Optional[str] = None,
                   started_at: Optional[str] = None,
                   finished_at: Optional[str] = None,
                   extra: Optional[Dict[str, Any]] = None,
                   keep_existing: bool = True) -> Dict[str, Any]:
    """Rewrite the run manifest with a new status, preserving earlier fields.

    If a manifest already exists (e.g. FAILED/INVALID written by the
    underlying runner), ``keep_existing=True`` merges those fields into
    the new payload so the state machine is auditable.
    """

    path = _manifest_path(run_dir)
    existing: Dict[str, Any] = {}
    if keep_existing and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as h:
                existing = json.load(h)
        except Exception:
            existing = {}
    payload: Dict[str, Any] = dict(existing)
    payload["status"] = status
    payload["status_updated_at"] = _now_iso()
    if reason is not None:
        payload["status_reason"] = reason
    if started_at is not None:
        payload["started_at"] = started_at
    if finished_at is not None:
        payload["finished_at"] = finished_at
    if extra:
        for key, value in extra.items():
            payload[key] = value
    os.makedirs(os.path.abspath(run_dir), exist_ok=True)
    write_json(path, payload)
    return payload


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ao04-run-formal",
                                 description="Stage R2 single-run formal entry point.")
    p.add_argument("--config", required=True,
                    help="Path to a frozen JSON config (e.g. configs/kuairand_natural_m3oe.json).")
    p.add_argument("--arm", required=True, choices=tuple(sorted(ARM_SET)),
                    help="Which arm to train (B0/B1/B2/B3/B4).")
    p.add_argument("--seed", required=True, choices=list(SEEDS),
                    type=int, help="Formal Stage R2 seed (2022 or 2023).")
    p.add_argument("--run-dir", required=True,
                    help="Output directory; non-empty directories are refused.")
    p.add_argument("--init-dir", default=os.path.join(RESEARCH_R2, "init"),
                    help="Directory that holds paired-init artefacts "
                         "(default: research_R2/init).")
    p.add_argument("--epochs", type=int, default=EPOCHS,
                    help=f"Number of epochs to run (default: {EPOCHS}). "
                         "Orchestration self-test only: any value other "
                         f"than {EPOCHS} requires R2_SELF_TEST=1 in the "
                         "environment.")
    p.add_argument("--resume-from", default=None,
                    help="Path to a previous last_resume.pt for the same "
                         "run (budget: at most one resume per run).  When "
                         "given, the non-empty run_dir refusal is skipped.")
    return p.parse_args()


def _load_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError("config not found: {}".format(path))
    with open(path, encoding="utf-8") as h:
        cfg = json.load(h)
    return cfg


def _refuse_overwrite(run_dir: str) -> None:
    """Stage S1's anti-overwrite policy: non-empty canonical dir is refused."""

    if not os.path.isdir(run_dir):
        return
    skip = {"failed_attempts", "stdout.log", "stderr.log"}
    present = [n for n in os.listdir(run_dir) if n not in skip]
    if present:
        manifest_path = _manifest_path(run_dir)
        prior_status = "<missing>"
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as h:
                    prior = json.load(h)
                prior_status = prior.get("status", "<missing>")
            except Exception:
                prior_status = "<unreadable>"
        raise FileExistsError(
            "refusing to overwrite run_dir (status={}): {}".format(
                prior_status, run_dir))


# ---------------------------------------------------------------------------
# Paired-init discovery / construction.
# ---------------------------------------------------------------------------

def _paired_state_path(init_dir: str, dataset: str, seed: int) -> str:
    return os.path.join(init_dir, dataset, "seed{}".format(seed), "init_state.pt")


def _ensure_paired_init(dataset: str, seed: int, init_dir: str,
                         dataset_path: Optional[str] = None
                         ) -> Tuple[str, str]:
    """Return (init_state_path, init_manifest_path).  Run paired_init if needed.

    The init artefact is shared by all five arms of the same
    (dataset, seed).  Re-running the init phase when the artefact
    already exists would rename the underlying model unobserved; we
    skip if ``init_state.pt`` and ``init_manifest.json`` both exist.
    """

    state_path = _paired_state_path(init_dir, dataset, seed)
    manifest_path = os.path.join(os.path.dirname(state_path), "init_manifest.json")
    if os.path.isfile(state_path) and os.path.isfile(manifest_path):
        return state_path, manifest_path
    os.makedirs(init_dir, exist_ok=True)
    _, artifacts = paired_init_phase(
        dataset=dataset, seed=seed, init_dir=init_dir, dataset_path=dataset_path)
    return artifacts.init_state_path, artifacts.init_manifest_path


# ---------------------------------------------------------------------------
# Main formal flow.
# ---------------------------------------------------------------------------

def _redirect_to_run_log(run_dir: str) -> Tuple[Any, Any]:
    """Buffer stdout / stderr in memory; flushed to <run_dir>/{stdout.log, stderr.log}
    only AFTER train_val_loop has accepted the run_dir.

    ``train_val_loop`` refuses any non-empty run_dir, so writing
    stdout / stderr to disk eagerly would trip its anti-overwrite
    guard.  We capture everything in memory here and persist once the
    runner has accepted the directory.
    """

    import io
    os.makedirs(os.path.abspath(run_dir), exist_ok=True)
    prior_out = sys.stdout
    prior_err = sys.stderr
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    sys.stdout = buf_out
    sys.stderr = buf_err
    return (prior_out, prior_err, buf_out, buf_err)


def _flush_logs_to_run_dir(run_dir: str, buf_out, buf_err,
                            prior_out: Any, prior_err: Any,
                            status_label: str = "COMPLETE") -> None:
    """Persist the buffered stdout / stderr and restore the real streams."""

    try:
        sys.stdout = prior_out
        sys.stderr = prior_err
    except Exception:
        pass
    stdout_path = os.path.join(run_dir, "stdout.log")
    stderr_path = os.path.join(run_dir, "stderr.log")
    try:
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(buf_out.getvalue())
    except Exception:
        pass
    try:
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(buf_err.getvalue())
    except Exception:
        pass
    sys.stdout = prior_out
    sys.stderr = prior_err


def _drop_log_files(run_dir: str) -> None:
    """Remove orchestrator-side stdout/stderr logs so the runner sees an
    empty directory.

    Used both at the very start (to discard any left-over artefacts
    from a previous crash) and around the train_val_loop call when its
    anti-overwrite guard would otherwise trip on the buffered outputs.
    """

    for name in ("stdout.log", "stderr.log", "failed_attempts"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _restore_stdio(prior_out: Any, prior_err: Any) -> None:
    """Restore the original stdout / stderr if we swapped them."""

    sys.stdout = prior_out
    sys.stderr = prior_err


def main() -> int:
    args = _parse_args()
    if not arm_supported(args.arm):
        print("[run_formal] error: unsupported arm {!r}".format(args.arm))
        return 2
    cfg = _load_config(args.config)
    dataset = str(cfg["dataset"])
    if not dataset_supported(dataset):
        print("[run_formal] error: unsupported dataset {!r}".format(dataset))
        return 2
    if dataset not in DATASET_SETTINGS:
        print("[run_formal] error: missing setting for {}".format(dataset))
        return 2
    seed = int(args.seed)

    run_dir = os.path.abspath(args.run_dir)
    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None:
        resume_from = os.path.abspath(resume_from)
        if not os.path.isfile(resume_from):
            print("[run_formal] error: --resume-from not found: {}".format(
                resume_from))
            return 2
    os.makedirs(run_dir, exist_ok=False) if not os.path.isdir(run_dir) else None
    if resume_from is None:
        try:
            _refuse_overwrite(run_dir)
        except FileExistsError as exc:
            print("[run_formal] refuse overwrite:", exc)
            return 1

    epochs = int(args.epochs)
    smoke = epochs != EPOCHS
    if smoke:
        # A shortened run is an orchestration self-test ONLY.  It must
        # be explicitly armed via the environment so a stray CLI flag can
        # never silently truncate a formal run.
        if not os.environ.get("R2_SELF_TEST"):
            print("[run_formal] error: --epochs {} != {} requires "
                  "R2_SELF_TEST=1 in the environment (self-test only); "
                  "refusing.".format(epochs, EPOCHS))
            return 2
        print("[run_formal] WARNING: self-test mode — running {} epoch(s) "
              "instead of {}; artefacts are NOT formal evidence.".format(
                  epochs, EPOCHS))

    # The per-run state machine is *implicit* inside run_manifest.json:
    # train_val_loop writes FAILED/INVALID on error and state_writer writes
    # COMPLETE on success.  ``PLANNED`` and ``RUNNING`` live in the queue's
    # queue_state.json (see run_queue.py) and are emitted before this
    # orchestrator's first heavy action.  We refuse to write run_manifest
    # pre-emptively because ``train_val_loop``'s anti-overwrite guard
    # would reject any non-empty run_dir.

    # Defensive: discard any orphan orchestrator-side log files so the
    # anti-overwrite guard below doesn't trip on something the earlier
    # attempt (or a crashed run) left behind.
    _drop_log_files(run_dir)

    prior_out, prior_err, buf_out, buf_err = _redirect_to_run_log(run_dir)
    rc = 0
    try:
        print("[run_formal] PLAN+RUN  dataset={} arm={} seed={} epochs={} smoke={}".format(
            dataset, args.arm, seed, epochs, smoke))

        # 1. paired_init (ensure it exists; idempotent across arms).
        paired_state_path, init_manifest_path = _ensure_paired_init(
            dataset=dataset, seed=seed, init_dir=os.path.abspath(args.init_dir),
            dataset_path=cfg.get("dataset_path"))
        print("[run_formal] paired_init: state={} manifest={}".format(
            paired_state_path, init_manifest_path))

        # 2. train_val_loop.
        config_for_runner = dict(cfg)
        # --epochs is authoritative: the frozen configs carry epochs=50,
        # so a mere "if absent" fill would silently discard a self-test
        # override.  Non-default epochs only reach this point in
        # R2_SELF_TEST mode (checked above).
        config_for_runner["epochs"] = epochs
        config_for_runner["device"] = str(cfg.get("device", DEFAULT_DEVICE))

        from stage_S1_data import load_stage_s1_bundle
        bundle = load_stage_s1_bundle(
            dataset, cfg.get("dataset_path"),
            DATASET_SETTINGS[dataset][0], seed)

        try:
            train_val_result = train_val_loop(
                arm=args.arm, dataset=dataset, seed=seed,
                run_dir=run_dir, config=config_for_runner,
                paired_state_path=paired_state_path,
                bundle=bundle,
                # The runner only accepts a non-50 epoch count in smoke
                # mode (smoke_max_batches not None).  10**9 is larger
                # than any real batch count, so no batch is truncated;
                # it purely unlocks the smoke channel of
                # _assert_locked_config for the self-test.
                smoke_max_batches=(10 ** 9 if smoke else None),
                resume_from=resume_from)
        except Exception as exc:
            # train_val_loop already wrote the FAILED/INVALID manifest.
            print("[run_formal] train_val_loop failed:", type(exc).__name__, str(exc))
            traceback.print_exc()
            _flush_logs_to_run_dir(run_dir, buf_out, buf_err,
                                    prior_out, prior_err,
                                    status_label="FAILED")
            rc = 3
            return rc

        try:
            state_writer(
                arm=args.arm, dataset=dataset, seed=seed,
                run_dir=run_dir, config=config_for_runner,
                bundle=bundle, train_val=train_val_result,
                paired_state_path=paired_state_path,
                test_arrays_present=True)
        except Exception as exc:
            print("[run_formal] state_writer failed:", type(exc).__name__, str(exc))
            traceback.print_exc()
            _write_status(run_dir, "FAILED",
                           reason="state_writer: {}".format(exc),
                           finished_at=_now_iso())
            _flush_logs_to_run_dir(run_dir, buf_out, buf_err,
                                    prior_out, prior_err,
                                    status_label="FAILED")
            rc = 4
            return rc

        # Stamp finished_at onto the run manifest without losing
        # state_writer's fields.  _write_status merges with existing data.
        _write_status(run_dir, "COMPLETE",
                       reason="run-completed",
                       finished_at=_now_iso(),
                       extra={"epochs_completed":
                              int(train_val_result.epochs_completed),
                              "epochs_requested": epochs,
                              "smoke_completed": bool(smoke),
                              "resumed_from": train_val_result.resumed_from,
                              "queued_at": getattr(args, "_queued_at", None),
                              "paired_init_manifest": init_manifest_path,
                              "paired_init_state": paired_state_path})
        print("[run_formal] COMPLETE   status=COMPLETE run_dir={}".format(run_dir))
        # Now that the run_dir contains all of state_writer's artefacts,
        # it is safe to release our buffered stdout/stderr.
        _flush_logs_to_run_dir(run_dir, buf_out, buf_err,
                                prior_out, prior_err,
                                status_label="COMPLETE")
        rc = 0
        return rc
    finally:
        # Belt-and-braces restore in case an outer exception escapes.
        try:
            _restore_stdio(prior_out, prior_err)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
