# Runtime Crash 5 implementation report

Status: **IMPLEMENTED; focused review pending.** No GDB command, CUDA workload, training, or E evaluation was run. The Crash5 design requires both independent review PASS and explicit orchestrator authorization before the one full-process launch.

## Deliverables

- `run_runtime_crash5_gdb.sh` runs the unchanged Crash3 D1 `launch` wrapper around the original formal AUTOIFS/KuaiRand/2023 training CLI. It retains the configured full search/reuse budget and `cuda:1`; it does not resume the archived rewind checkpoint or override the config.
- It refuses a non-empty canonical output directory and refuses to overwrite any of its three evidence files. The previously failed partial run remains in `artifacts/mdmt_baselines_2526/attempts/formal_kuairand_AUTOIFS_seed2023_attempt01_SIGSEGV`; preserve that archive and the 26 completed runs unchanged.
- One GDB invocation uses `-q -nx -batch`, disables core dumps, stops without passing SIGSEGV, and requests all-thread backtrace depth 40, registers, the current instruction, and shared libraries. It does not install/download symbols or issue another `continue`/`run`.
- The complete GDB stdout/stderr is written to `runtime_crash5/gdb_transcript.txt`; the existing D1 launcher writes the exact CLI/config/source/data sidecar. `gdb_execution.txt` records UTC start/end, external elapsed seconds, GDB exit code, and an inferior outcome parsed from the transcript. A detected SIGSEGV makes the shell script exit nonzero even if GDB itself exits zero. An unrecognized inferior termination is also nonzero and must be inspected rather than treated as success.

## Authorized runner command

Run this only after the focused independent review is PASS, the orchestrator explicitly authorizes this one launch, and the canonical output directory has been removed or is empty after the partial archive is confirmed. Before launch, verify the current package is exact reviewed R2 and the original config, source commits, prepared-data identity, seed, phase, device, and output location still match the archived run. The script itself performs no archive operation.

```bash
bash research_reports/mdmt_baselines_2526/run_runtime_crash5_gdb.sh
```

Afterward, inspect both the transcript's inferior termination/signal and the canonical `run.json`/checkpoints. GDB exit code zero alone is not evidence of successful training. If the inferior signals, identity fails, or the process stops before normal completion, preserve the transcript and stop; do not retry or advance the remaining seeds.

## Exact comparison of the original 14 epochs

Compare the original archived JSONL to the first 14 completed scientific records in the replacement JSONL. The command requires the original order (search epochs 0–9, reuse epochs 1–4), compares every JSON field exactly, and removes only the explicit per-epoch `seconds` field:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python - \
  artifacts/mdmt_baselines_2526/attempts/formal_kuairand_AUTOIFS_seed2023_attempt01_SIGSEGV/epoch_metrics.jsonl \
  artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2023/epoch_metrics.jsonl <<'PY'
import json
import sys
from pathlib import Path

expected = [("search", epoch) for epoch in range(10)] + [("reuse", epoch) for epoch in range(1, 5)]

def first14(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("stage") in {"search", "reuse"}:
                records.append(row)
    records = records[:14]
    assert [(row["stage"], row["epoch"]) for row in records] == expected, path
    return [{key: value for key, value in row.items() if key != "seconds"} for row in records]

old, new = first14(sys.argv[1]), first14(sys.argv[2])
assert old == new, "one or more first-14 scientific fields differ"
print("PASS: all 14 ordered search/reuse records match exactly except seconds")
PY
```

This comparison is necessary but not sufficient for replacing the failed matrix cell: the full-budget run must also complete normal pruning/bookkeeping and produce the expected selected/fixed-final checkpoints and completed run status. Do not interpret a matching prefix as evidence that a crash was fixed.

## Verification performed

- `bash -n research_reports/mdmt_baselines_2526/run_runtime_crash5_gdb.sh`: **PASS**.
- CPU-only invocation-context check through the existing `_launch_context` confirmed the exact formal method/dataset/seed/device/output argv resolves to the intended canonical run and KuaiRand manifest without entering the experiment CLI.
- No GDB, CUDA, training, checkpoint load, or E command was run. Existing package, launcher, config, source pair, archived partial run, and completed trajectories were not modified.

## Focused GDB signal-format correction

The signal classifier matches `received signal SIGSEGV` and `terminated with signal SIGSEGV`, covering both `Program received ...` and `Thread 1 "python" received ...` GDB transcript forms. The remaining normal-exit and unrecognized-transcript branches are unchanged. CPU transcript fixtures for both signal forms, normal exit, and an unrecognized ptrace failure all classify as expected; `bash -n` passes. No GDB or training was run for this correction.

## Reviewed continuation callers: 2024–2026

The approved design continuation permits these three original, never-started formal seeds to run **once each, sequentially**, with the same full config, D1 launcher, GDB arguments, R2 package, and `cuda:1` as seed 2023. This section supplies their exact CLI argv and distinct paths; it does not run them. Root must explicitly authorize launches. Preserve the 2023 result and all prior runs.

| Seed | Literal CLI argv after `--sidecar PATH --` | Canonical output directory | D1 sidecar | GDB transcript | External execution receipt |
|---:|---|---|---|---|---|
| 2024 | `--config experiments/mdmt_sota_baselines_2526/default_config.json train --method AUTOIFS --dataset kuairand --seed 2024 --phase formal --device cuda:1 --output artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2024` | `artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2024` | `research_reports/mdmt_baselines_2526/runtime_crash5/AUTOIFS_kuairand_seed2024_gdb.json` | `research_reports/mdmt_baselines_2526/runtime_crash5/gdb_transcript_seed2024.txt` | `research_reports/mdmt_baselines_2526/runtime_crash5/gdb_execution_seed2024.txt` |
| 2025 | `--config experiments/mdmt_sota_baselines_2526/default_config.json train --method AUTOIFS --dataset kuairand --seed 2025 --phase formal --device cuda:1 --output artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2025` | `artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2025` | `research_reports/mdmt_baselines_2526/runtime_crash5/AUTOIFS_kuairand_seed2025_gdb.json` | `research_reports/mdmt_baselines_2526/runtime_crash5/gdb_transcript_seed2025.txt` | `research_reports/mdmt_baselines_2526/runtime_crash5/gdb_execution_seed2025.txt` |
| 2026 | `--config experiments/mdmt_sota_baselines_2526/default_config.json train --method AUTOIFS --dataset kuairand --seed 2026 --phase formal --device cuda:1 --output artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2026` | `artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2026` | `research_reports/mdmt_baselines_2526/runtime_crash5/AUTOIFS_kuairand_seed2026_gdb.json` | `research_reports/mdmt_baselines_2526/runtime_crash5/gdb_transcript_seed2026.txt` | `research_reports/mdmt_baselines_2526/runtime_crash5/gdb_execution_seed2026.txt` |

For each row, substitute only its `SEED`, `RUN_DIR`, `SIDECAR`, `TRANSCRIPT`, and `EXECUTION` values into this same GDB invocation. Keep all debugger actions and launch arguments unchanged; the row's literal CLI argv must be passed after `--sidecar "$SIDECAR" --`:

```bash
SEED=2024
RUN_DIR=artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2024
SIDECAR=research_reports/mdmt_baselines_2526/runtime_crash5/AUTOIFS_kuairand_seed2024_gdb.json
TRANSCRIPT=research_reports/mdmt_baselines_2526/runtime_crash5/gdb_transcript_seed2024.txt
EXECUTION=research_reports/mdmt_baselines_2526/runtime_crash5/gdb_execution_seed2024.txt
ulimit -c 0
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s.%N)"
set +e
PYTHONFAULTHANDLER=1 /usr/bin/gdb -q -nx -batch \
  -ex 'set pagination off' -ex 'set confirm off' \
  -ex 'set disable-randomization off' \
  -ex 'handle SIGSEGV stop print nopass' \
  -ex run -ex 'thread apply all bt 40' -ex 'info registers' \
  -ex 'x/i $pc' -ex 'info sharedlibrary' -ex quit \
  --args /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python \
  research_reports/mdmt_baselines_2526/runtime_crash3_tool.py launch \
  --sidecar "$SIDECAR" -- \
  --config experiments/mdmt_sota_baselines_2526/default_config.json train \
  --method AUTOIFS --dataset kuairand --seed "$SEED" --phase formal --device cuda:1 \
  --output "$RUN_DIR" > "$TRANSCRIPT" 2>&1
GDB_EXIT_CODE=$?
ENDED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s.%N)"
ELAPSED_SECONDS="$(awk -v start="$START_EPOCH" -v end="$END_EPOCH" 'BEGIN { printf "%.3f", end-start }')"
printf 'started_at_utc=%s\nended_at_utc=%s\nexternal_elapsed_seconds=%s\ngdb_exit_code=%s\ntranscript=%s\nsidecar=%s\n' \
  "$STARTED_AT" "$ENDED_AT" "$ELAPSED_SECONDS" "$GDB_EXIT_CODE" "$TRANSCRIPT" "$SIDECAR" \
  > "$EXECUTION"
```

Before the next row, confirm the inferior exited normally (GDB exit code alone is insufficient), the run identity is exact, scientific outputs are finite, original stopping rules and pruning/postprocessing completed, the expected checkpoints exist, and the run receipt truthfully says complete. Any signal, identity/numeric mismatch, or incomplete run stops the queue; do not retry or recover it. These three seeds have no prior same-seed trajectory: the seed-2023 fourteen-epoch comparison does not apply, and do not assume its stopping epoch for seeds 2024–2026.

CPU-only `_launch_context` checks of all three row argv resolved the matching formal AUTOIFS/KuaiRand seed and canonical output, loaded the KuaiRand manifest, and did not invoke the experiment CLI. No GDB, CUDA, or training was run for this continuation.
