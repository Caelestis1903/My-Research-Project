#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

EVIDENCE="research_reports/mdmt_baselines_2526/runtime_crash5"
CANONICAL="artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2023"
SIDECAR="$EVIDENCE/AUTOIFS_kuairand_seed2023_gdb.json"
TRANSCRIPT="$EVIDENCE/gdb_transcript.txt"
EXECUTION="$EVIDENCE/gdb_execution.txt"
PYTHON="/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python"

if [[ ! -x /usr/bin/gdb || ! -x "$PYTHON" ]]; then
  echo "Required GDB or project Python executable is unavailable." >&2
  exit 2
fi
if [[ -e "$CANONICAL" ]] && [[ -n "$(find "$CANONICAL" -mindepth 1 -print -quit)" ]]; then
  echo "Canonical formal output is not empty; preserve/archive it before this one launch." >&2
  exit 2
fi
mkdir -p "$EVIDENCE"
for path in "$SIDECAR" "$TRANSCRIPT" "$EXECUTION"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite existing evidence: $path" >&2
    exit 2
  fi
done

ulimit -c 0
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s.%N)"
if PYTHONFAULTHANDLER=1 /usr/bin/gdb -q -nx -batch \
  -ex 'set pagination off' \
  -ex 'set confirm off' \
  -ex 'set disable-randomization off' \
  -ex 'handle SIGSEGV stop print nopass' \
  -ex run \
  -ex 'thread apply all bt 40' \
  -ex 'info registers' \
  -ex 'x/i $pc' \
  -ex 'info sharedlibrary' \
  -ex quit \
  --args "$PYTHON" \
  research_reports/mdmt_baselines_2526/runtime_crash3_tool.py launch \
  --sidecar "$SIDECAR" \
  -- --config experiments/mdmt_sota_baselines_2526/default_config.json train \
  --method AUTOIFS --dataset kuairand --seed 2023 --phase formal --device cuda:1 \
  --output "$CANONICAL" \
  > "$TRANSCRIPT" 2>&1; then
  GDB_EXIT_CODE=0
else
  GDB_EXIT_CODE=$?
fi
ENDED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s.%N)"
ELAPSED_SECONDS="$(awk -v start="$START_EPOCH" -v end="$END_EPOCH" 'BEGIN { printf "%.3f", end-start }')"

if grep -Eq 'received signal SIGSEGV|terminated with signal SIGSEGV' "$TRANSCRIPT"; then
  INFERIOR_OUTCOME="SIGSEGV"
  SCRIPT_EXIT_CODE=1
elif grep -Eq 'Inferior .* exited normally' "$TRANSCRIPT"; then
  INFERIOR_OUTCOME="EXITED_NORMALLY"
  SCRIPT_EXIT_CODE="$GDB_EXIT_CODE"
elif grep -Eq 'Inferior .* exited with code' "$TRANSCRIPT"; then
  INFERIOR_OUTCOME="EXITED_WITH_CODE"
  SCRIPT_EXIT_CODE=1
else
  INFERIOR_OUTCOME="NOT_IDENTIFIED_IN_TRANSCRIPT"
  SCRIPT_EXIT_CODE=1
fi

{
  printf 'started_at_utc=%s\n' "$STARTED_AT"
  printf 'ended_at_utc=%s\n' "$ENDED_AT"
  printf 'external_elapsed_seconds=%s\n' "$ELAPSED_SECONDS"
  printf 'gdb_exit_code=%s\n' "$GDB_EXIT_CODE"
  printf 'inferior_outcome=%s\n' "$INFERIOR_OUTCOME"
  printf 'script_exit_code=%s\n' "$SCRIPT_EXIT_CODE"
  printf 'transcript=%s\n' "$TRANSCRIPT"
  printf 'launcher_sidecar=%s\n' "$SIDECAR"
} > "$EXECUTION"

printf 'GDB exit=%s; inferior=%s; transcript=%s; execution=%s\n' \
  "$GDB_EXIT_CODE" "$INFERIOR_OUTCOME" "$TRANSCRIPT" "$EXECUTION"
exit "$SCRIPT_EXIT_CODE"
