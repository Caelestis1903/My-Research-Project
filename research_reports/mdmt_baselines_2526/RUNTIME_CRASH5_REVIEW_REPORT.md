# REVIEW REPORT — fifth native crash / one full-process gdb diagnostic

2026-10-29. **Latest focused continuation verdict: PASS — the three original seed2024/2025/2026 callers satisfy the approved continuation clause.** Each may run once sequentially on orchestrator instruction, with the next gated by the prior run's actual complete checks. This review runs nothing and does not release E. Native root cause remains unresolved. See §5; all prior review history is preserved.

**Earlier seed2023 verdict: PASS — C5-1 corrected and independently verified.** That pre-run review made the one original full-process gdb diagnostic eligible after explicit orchestrator authorization, without releasing the remaining3 or E.

**Initial verdict: PASS WITH FIXES — C5-1 MINOR signal classification correction required before launch.** The initial review withheld launch eligibility pending that correction and final re-review.

## 1. Requirement-by-requirement verification

Read `DESIGN_RUNTIME_CRASH5_ADDENDUM.md`, sealed `run_runtime_crash5_gdb.sh`, `CODER_RUNTIME_CRASH5_REPORT.md`, archive preservation receipt and actual archived metadata/log/source files. Reviewed against the existing unchanged D1 launcher; no package/model implementation redesign.

| Requirement | Independent evidence | Result |
|---|---|---|
| Original formal identity | Actual archived run is TRAINING/AUTOIFS/KuaiRand/2023/formal; full config equals current default and approved pair, source_commits/prepared-data identity equal approved originals. | PASS |
| Preserve partial and successful work | Actual archive has21 files, rewind exists, selected/fixed-final absent; source18 files match R2. Preservation receipt records21/21 original pre/post-rename direct byte equality. Script has no archive/run/checkpoint write beyond delegating the new canonical CLI. | PASS |
| Current frozen source | Independently decoded approved R2 source and directly compared all18 actual package and all18 archived snapshot file bytes. No source/hash/config change. | PASS |
| Full original process | Exact literal CLI uses existing Crash3 D1 launcher, original config/AUTOIFS/KuaiRand/2023/formal/cuda:1/canonical path; no epoch cap, rewind resume, seed substitution or new backend. | PASS |
| One invocation and evidence preservation | Script refuses nonempty canonical output and existing sidecar/transcript/execution receipt; one gdb run, no continue/retry loop. `ulimit -c 0`; all-thread bt40/registers/instruction/libraries retained, stdout/stderr combined. | PASS |
| Original14 scientific records | Actual original ordered search0–9/reuse1–4,5908 updates, no completed pruning or saved selected/final. | PASS |
| Exact prefix comparison | Extracted actual literal comparison into CPU temporary fixtures: accepts identical scientific objects with changed seconds only; rejects1e-8 loss, component or V-cell changes, missing14th epoch and wrong stage/epoch order. All-object equality entails maximum macro AUC/LL differences0; tolerance is not relaxed. | PASS |
| No premature matrix completion | Coder note explicitly requires actual full pruning/bookkeeping, both checkpoints, canonical completion and identity checks in addition to prefix. Remaining3/E not released. | PASS |
| Inferior signal regardless of debugger exit | Program-prefixed SIGSEGV correctly returns signal/nonzero; thread-prefixed SIGSEGV is not classified correctly, although it still returns nonzero and stops. See C5-1. | FIX REQUIRED |

Independent `bash -n` passed. Canonical target was absent at review. No gdb, CUDA, checkpoint load, training, E, dependency installation or hash was run by reviewer.

## 2. C5-1 — MINOR, required signal classification fix

Initial transcript parser matches only `Program received signal SIGSEGV` or `Program terminated with signal SIGSEGV`. Gdb can print `Thread 1 "python" received signal SIGSEGV, Segmentation fault.` in a multithreaded inferior. An independent CPU fixture executing only the actual parser fragment with GDB_EXIT_CODE=0 classified that as NOT_IDENTIFIED_IN_TRANSCRIPT/exit1 rather than SIGSEGV/exit1. The safety stop is retained, but the required native outcome is misclassified.

Minimal correction: match `received signal SIGSEGV|terminated with signal SIGSEGV` independently of Program/Thread prefix. No new verifier or debugger run is needed. Recheck Program, Thread, normal exit and ptrace/unknown fixtures before final PASS.

## 3. Corrections and remaining boundaries

No implementation change made by reviewer; only this report written. The full scientific prefix must still be compared on the real new logs after the explicitly authorized diagnostic; CPU fixtures are not experiment results. If signal, mismatch or incomplete/identity failure occurs, retain all evidence and stop without a second attempt.

Only complete normal training/pruning/saving/postprocessing plus exact14-prefix and existing run/checkpoint identities can replace the failed seed2023 cell. Preserve the original132.920945704s failed-process cost separately and retain debugger overhead as actual replacement runtime provenance. No root-cause fix, additional seed repetition, future salvage, remaining3 release or freeze/E bypass is inferred from success.

## 4. Focused final correction review

The sealed script now matches `received signal SIGSEGV|terminated with signal SIGSEGV` without a Program-only prefix. Independently reran `bash -n` and extracted the corrected actual parser into CPU temporary fixtures with debugger exit0. Program SIGSEGV -> SIGSEGV/exit1; Thread SIGSEGV -> SIGSEGV/exit1; normal inferior -> EXITED_NORMALLY/exit0; ptrace failure -> NOT_IDENTIFIED_IN_TRANSCRIPT/exit1. **C5-1 closed.** No debugger was launched for these fixtures. No other implementation defect remains identified within this literal-command/comparison scope.

Prior identity/source/archive/14-prefix evidence in §1 remains applicable. All algorithms, formulas, tensor/mask/gradient semantics and baseline training paths stay the exact reviewed R2 bytes; the external shell changes no mathematics or configuration. Only diagnostic instrumentation/native-debugger scheduling differs. No additional implementation correction was made by reviewer.

**Final execution boundary:** root must explicitly authorize one original seed2023 from-initialization full-budget D1/cuda:1 gdb process using this script. Inspect debugger and inferior outcomes separately, preserve native frames/registers/instruction/stage on signal and stop on any signal, numerical/identity mismatch or incomplete process. Never treat debugger exit0 alone as success. Do not automatically rerun when command/debugger/resource failure occurs.

After a normal full inferior completion, independently confirm actual14 ordered scientific objects equal the archive except seconds, maximum macro AUC/LL differences0, complete pruning, both selected/fixed-final checkpoints, original source/config/data identities and normal completed/test-locked bookkeeping before accepting this replacement matrix cell. No real comparison/replacement success is claimed by this pre-run report. The remaining3 require subsequent orchestrator evidence assessment; E retains all existing30-run/mixed-R1-R2 checks.

## 5. Focused continuation review — original seeds2024/2025/2026

Read the designer's appended continuation clause and coder's three literal caller rows/command. **PASS, no blocking issue found.** Independently parsed all three row argv with the existing CPU-only `_launch_context`: method AUTOIFS, dataset KuaiRand, matching seed2024/2025/2026, formal phase, unchanged full default config, cuda:1 and corresponding canonical seed directory all agree. Canonical paths plus the three distinct per-seed D1 sidecar/transcript/execution paths are unique (12 distinct paths total).

Compared the actual gdb action tokens in the2023 shell command against the continuation literal: identical flags/actions, same unchanged D1 launcher, no epoch override or new profile. Per-process debugger/external elapsed cost and GDB exit are recorded with separate transcript and D1 sidecar. The caller instructions explicitly require actual normal inferior termination independently of debugger exit, finite scientific records, original identity/stop rules, full pruning/postprocessing, both checkpoints and truthful completed receipt before the next seed. Any signal, numeric/identity mismatch or incomplete run stops the queue, without automatic salvage/retry.

These are the three never-started cells of the original30 matrix, each once, with no prior same-seed record. No14-epoch equality claim or seed2023 stopping epoch is applied to them. This focused review introduces no source/config/model change, rerun matrix or verifier; prior frozen R2/D1 implementation evidence remains applicable. No GDB, CUDA, training or E was executed. E remains locked until actual complete30 and the existing mixed-R1-R2 freeze pass. Under-debugger timing remains disclosed runtime provenance; clean instrumentation does not establish a native-fault fix.
