# Frozen test queue intake

This is a mechanical intake for parent-frozen, explicitly authorized, read-only
test evaluations. A frozen report or a ready stage alone does not authorize a
test. No test runs unless an immutable release JSON appears in one of these
directories with `status: "RELEASED"` and `test_authorized: true`:

```text
ART/runner/released_test_queues/N17/<release_id>.json
ART/runner/released_test_queues/AO04/<release_id>.json
ART/runner/released_test_queues/transfer/<release_id>.json
```

Publish each release atomically (write a temporary file, then rename it to the
final `.json` name). Never edit or replace a published release. A new release
needs a new `release_id`; transfer releases are one file per study and scope.
Only files in the three directories above are considered releases.

## Schema v1

Required top-level fields:

```json
{
  "schema": "natural_movielens_test_release_v1",
  "family": "N17",
  "release_id": "N17_formal_20261020_a",
  "status": "RELEASED",
  "test_authorized": true,
  "parent_freeze_path": "/absolute/path/to/frozen_test_manifest.json",
  "parent_freeze_sha256": "<64 lowercase hexadecimal characters>",
  "tasks": []
}
```

For `transfer`, include top-level `study` and `scope`; every task must match
those values. Do not include top-level `study` or `scope` for N17/AO04.

Each task requires:

```json
{
  "run_id": "frozen_test_task_id",
  "study": "N17",
  "scope": "formal",
  "config_path": "/absolute/path/to/task_config.json",
  "run_dir": "/absolute/path/to/task_output_directory",
  "command": [
    "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
    "-m", "experiments.natural_ml_extensions.n17_readout", "test",
    "--run-dir", "/absolute/path/to/task_output_directory",
    "--device", "cuda:0"
  ],
  "receipt": "/absolute/path/to/task_output_directory/test_receipt.json",
  "receipt_checks": {"status": "<exact completion status>"},
  "required_files": ["/absolute/path/to/task_config.json"],
  "artifacts": ["/absolute/path/to/task_output_directory/test_metrics.csv"]
}
```

The command is an argv array (never a shell string), uses the frozen Python
environment, and contains exactly one `--device cuda:0`. The coordinator maps
that logical device to its selected physical GPU through
`CUDA_VISIBLE_DEVICES`; the frozen argv is recorded in the ledger and event
log. If `--config` or `--run-dir` occurs in the argv, its value must match the
task fields. `config_path`, `run_dir`, `receipt`, all artifact paths, and all
required input paths are absolute. The receipt and listed artifacts must be
inside `run_dir`; artifacts list only the test outputs and excludes the
receipt. The config and run directory must exist at intake.

Optional task fields:

- `profile_group`: memory-profile key; defaults to `TEST_<FAMILY>`.
- `artifact_sha256`: map from declared absolute artifact paths to frozen
  lowercase SHA256 digests. When present, every digest is checked.

`receipt_checks` gives exact top-level key/value checks against the completion
receipt and must include `status`. The coordinator also requires every listed
artifact to exist. It does not interpret test metrics. `required_files` are
checked before launch. Use a unique `run_id` across all test releases; the
ledger prevents a later release from reusing a test identity.

## Intake boundaries and idempotence

The coordinator scans all three family directories at these safe boundaries:

- after N17 checkpoint and precision diagnostics;
- after AO04 diagnostics;
- after the 160 core offline diagnostics;
- after every transfer study/scope diagnostic closure.

Each scan considers every authorized release in family order N17, AO04,
transfer, including a late release for an earlier family. Each release uses
the existing GPU scheduler, per-card five-process limit, measured memory
profile, 90% admission ceiling, shared event log, and `RUN_LEDGER.csv`.
N17 readout commands pass through the shared-split prepare lock wrapper.

Test ledger rows use `execution_kind=test` and `test__<run_id>_attemptNN`,
separate from training, diagnostics, and controls. A valid receipt plus all
declared artifacts is the completion check. A completed test attempt is
re-audited and skipped on later scans. A valid receipt found without a test
ledger row is recorded as complete with `UNKNOWN_MONITOR_LOST` for the OS exit
code; the coordinator does not invent an exit code or completion time. Failed
or incomplete attempts remain in the ledger, and only the release-declared
receipt and artifacts are moved to `runner/failed_attempts/tests/` before a
retry. Training directories and states are never recursively archived by this
test path.

The release file and its parent freeze are SHA-checked on discovery, before
execution, and when auditing completion. A changed file at an already-seen
release path is an error; publish a new release ID instead. No test tasks are
queued until the exact authorized release file is present.
