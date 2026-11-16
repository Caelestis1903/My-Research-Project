# Cycle03 core implementation report

Status: **CORE IMPLEMENTED; finite CPU checks PASS; runtime integration pending.**

Fresh native core coder `01a1e4bb-52a5-7142-97be-15c25dfbe75e`, original current conversation, `fork_context=false`, explicit file handoff, no descendants. Ponytail FULL and Stop That Shit CHANGE applied. Design authority is the seven-file bundle `fe784adac9005f29b230e019b5e0fcd9c98dc842c1ac4ebe12849ba90512f3bb`, accepted as COMPLETE_APPROVED_FOR_IMPLEMENTATION. Astra Harvey and designer Copernicus remain closed.

Read W/O AGENTS, O/start.md decoded GB18030, W/buchong.md and execution runbook; all seven approved Cycle03 files; relevant frozen C2 core/numerical helpers and the populated-Adam R1 restoration correction. Both workspaces' stop controls were ACTIVE, manual stop false, termination null and all four sentinels absent.

The exact five changed paths are:

- `/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/nqr_temporal_exposure_20261116/__init__.py`
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/nqr_temporal_exposure_20261116/core.py`
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/nqr_temporal_exposure_20261116/checks.py`
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/CORE_IMPLEMENTATION_REPORT.md`
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/CORE_CPU_CHECKS.json`

The 86-line core exposes the exact public `schedule(arm, epoch)` and frozen positional `train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False)` signatures. Its only numerical call is the imported frozen C2 train_step, exactly once per successful adapter call. It validates the fixed scientific config, both windows and coefficients, source data/selection identities, explicit purpose/clock pairing and required real-worker row IDs before delegation. The local inner copy changes only arm to NQR-025, objective to natural_quadratic_ranking_v1 and effective mixture_lambda. Caller config/context and old globals remain unchanged.

Both external temporal arms remain in the outer config and returned schedule. All fifteen frozen C2 result fields, including the original ledger/diagnostic objects, pass through unchanged. The only additions are the schedule with explicit clock_kind and the eight specified phase_increments. Core imports no new protocol/runtime and implements no QR, CLEAN, backward, Adam, EMA, ledger or diagnostic operation.

Finite CPU verification passed:

- All 100 schedule values, exact ten-epoch windows and lambda sum; 61 invalid config/context/row-ID cases rejected before any delegation or state/RNG mutation.
- Eight active/off × fresh/populated × dataset identity cases, two updates each: active versus direct frozen C2, off versus direct source CLEAN, diagnostic on/off, exact model/EMA/Adam/gradient/None/activity/RNG/private-loader identities.
- Both virtual boundary sequences use four actual updates, global steps1–4, with populated Adam crossing2→3. They preserve source None versus explicit-zero behavior and accumulate exactly two active/two off updates with observed phase counters.
- Ten support/tail cases cover mixed/empty/one-class/singleton support and actual B21927/40052. Ledger fields remain source-owned; nulls, coefficient/task mass, true ordered row hashes and synthetic-check-only None row IDs pass. Scalar mass check uses absolute1e-15; all state and gradient comparisons are exact.
- Three owned Python files compile in memory. Frozen C2 87-file and C1 79-file hashes match. Five protected source-tree inventories, including all existing pycache bytes/membership, match before/after.

The original literal QR/graph/math checks and R1 restore proof are reused by exact receipt hashes in CORE_CPU_CHECKS.json; their full suites were not rerun. The first local run found an incorrect assumption in the new receipt reader: historical CORE_CPU_CHECKS uses CORE_CPU_PASS_INTEGRATION_PENDING while each of its five groups is PASS. The reader now checks those actual fields. The subsequent core run and focused tightened-mass rerun passed. Exact executed check commands, exit codes and results are in CORE_CPU_CHECKS.json; every Python invocation used the required interpreter and PYTHONDONTWRITEBYTECODE=1.

`checks.check_runtime_contracts()` is implemented behind lazy imports and a seven-file/runtime-report readiness gate. It checks the shared config APIs, absolute new-root isolation, pure formal clock arithmetic and admission predicates, and two actual populated engineering boundary/BytesIO/restore fixtures. The restore fixtures call the new runtime's real restore helper, test step/exp_avg/exp_avg_sq storage against both saved snapshots and another restored route, and compare the exact uninterrupted suffix and observed phase/activity counters. They never set Adam/global counters to virtual scientific epochs.

**Concrete integration pending:** at this handoff, README.md and RUNTIME_IMPLEMENTATION_REPORT.md do not exist. The runtime coder's four files remain exclusively theirs. Runtime integration was not executed or claimed complete. Once those files are final, the parent runs from W:

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.nqr_temporal_exposure_20261116.checks
```

Parent integrates both coder reports and all seven files before any new full-source freeze, then obtains a fresh distinct reviewer verdict. The parent alone subsequently launches the new foreground N RAM owner after code PASS; the four real GPU smoke cases and distinct resource signoff precede formal20. Formal clocks and retained source two-key counts remain runtime responsibilities under specification5.3/7.1/7.5.

Owned source file SHA-256 values:

| File | SHA-256 |
| --- | --- |
| __init__.py | `0740474419393ac1702d19d5793b6c2ed379c80b74f1498d9c34a09f49f31250` |
| core.py | `89f6f9342ab1d1663c9b34984ba5fa0859bff7409322a33a1b39875267b2565c` |
| checks.py | `1c18a2d0ab51548842bc3994e4ad92b0ff271e4dd27e62f10ba62bf3d4c55959` |

Frozen C2 remains `ddaf5d97b676c4c8ba06c40cb2a7aa2209ae27148950b848329e1dc8bb669bae`; frozen C1 remains `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1`. No new full source receipt was frozen. No GPU query, IPC/owner contact, research trajectory, Test access, analysis conclusion, old source/rootstate/history/journal/result write, Git/publication, dependency or other-coder edit occurred. All finite tensor state stayed in process RAM; BytesIO restore checks await runtime integration. The existing TV/C1/C2 owners were not contacted.

Parent closes this core-coder context permanently after file handoff; do not reuse/resume it. Runtime integration is the remaining implementation gate. No GPU or storage-owner verification was performed.
