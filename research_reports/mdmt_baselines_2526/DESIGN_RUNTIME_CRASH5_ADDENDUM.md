# DESIGN — one full-process native diagnostic for crash5

2026-10-29 · MDMT-SOTA-2526-v1 · **DESIGN ONLY: no replacement/retry has been authorized or started.** Existing boundaries apply; no package/config/source revision, dependency, hash or runtime workaround.

1. **Question/evidence.** Can gdb capture the fault in its actual full-training process, instead of another fresh-checkpoint approximation? Inspected fifth stderr/execution receipt and `training.py:_fit_autoifs/_pruning_summary`. Native SIGSEGV occurred at `_pruning_summary:379`, `tensor(data.x[rows],device)`, under D1. Search0–9 and reuse1–4 are complete (14 scientific epoch records;5908 updates), but pruning did not finish. Only rewind9.pt exists; selected/final were still RAM and are lost. Crash4's checkpoint recovery is inapplicable.26 complete entries, this partial2023,3 never started, E0 are preserved; recorded failed-process wall132.920945704s remains extra cost, not another seed result.

2. **Rationale/hypothesis.** A fault depending on accumulated process state is plausible but unproved. One initialization-to-completion process under a native debugger directly covers training, pruning, validation and saving. A shorter rewind replay cannot reproduce the same unknown state or restore lost selector/selected weights. This is the smallest available diagnostic reaching the actual path without inventing model/source changes; it costs one replacement attempt, not a parameter experiment. A clean run will still NOT prove the cause fixed.

3. **Gate/ownership.** Pipeline coder supplies only a literal launch command or short external shell script plus exact14-epoch comparison, reusing unchanged `runtime_crash3_tool.py launch`. Independent reviewer checks command, original identity, preservation and comparison conditions; all blocking issues must be corrected. **Reviewer PASS alone does not launch: orchestrator must explicitly authorize this one full-process diagnostic.** No new1k-line verifier, Python package edits, extra profiles or diagnostic grid.

4. **Preservation before an authorized launch.** Archive partial2023 once under a unique owned attempt directory, retain original run.json/log/rewind/source snapshot byte-for-byte and compare copied original bytes directly. Preserve26 successful trajectories. Verify current package is exact reviewedR2; original config, external commits, data identity,2023 seed,phase=formal,cuda:1,D1 and canonical output location remain the same. Use new diagnostic sidecar/log paths; the canonical output must be absent/empty. No rewind resume or seed/config substitution.

5. **Literal command template (do not execute during design).** Run from repository root, with a new owned `runtime_crash5` evidence directory. Existing launcher sets D1 before computation. The inferior executes the original full config, including search10/reuse max50/patience1, not an altered14-epoch cap:

```bash
/usr/bin/gdb --batch \
  -ex 'set pagination off' -ex 'set confirm off' \
  -ex 'set disable-randomization off' \
  -ex 'handle SIGSEGV stop print nopass' \
  -ex run -ex 'thread apply all bt 40' -ex 'info registers' \
  -ex 'x/i $pc' -ex 'info sharedlibrary' -ex quit \
  --args /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python \
  research_reports/mdmt_baselines_2526/runtime_crash3_tool.py launch \
  --sidecar research_reports/mdmt_baselines_2526/runtime_crash5/AUTOIFS_kuairand_seed2023_gdb.json \
  -- --config experiments/mdmt_sota_baselines_2526/default_config.json train \
  --method AUTOIFS --dataset kuairand --seed 2023 --phase formal --device cuda:1 \
  --output artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2023
```

Capture full debugger stdout/stderr and inferior/external wall status. Inferior SIGSEGV must be recognized from transcript/status even if gdb exits0. No continue/relaunch after a signal, attaching to others, core dump, symbol installation, extra CUDA synchronization/launch-blocking or allocator switches. If ptrace/gdb/native symbols prevent useful capture, record the exact limitation and stop rather than change environment again.

6. **Fidelity/acceptance.** Compare original14 ordered search/reuse scientific records with the new first14, exact equality of every non-time scientific field: stages/epochs, train losses/components, updates/exposures, temperature/ticket, six V cells and aggregates. Ignore only explicit timing/process diagnostic fields. Macro V AUC/LL max absolute differences must both0. If logs indicate a mismatch or altered stopping/selection, stop further work; do not relax tolerances. R2/config/data/external sources must match throughout. Debugger may perturb scheduling, not authorize changed numerical results.

7. **Outcomes/stop rule.** Native fault: retain native C/library frames, instruction/registers and Python/log stage; stop all launches and return to evidence-based design. Numerical/identity mismatch: stop, no second attempt. Normal exit: only if all original training/pruning/postprocessing finish, exact14-epoch comparison passes, selected/final checkpoints and canonical completion pass existing checks, may this single trajectory replace the failed2023 matrix cell. It is not a fresh independent repeat and not a repaired-runtime claim. No automatic launch of the remaining3 seeds, no automatic salvage/retry if another crash follows; orchestrator assesses the actual evidence and existing authorization separately.

8. **Freeze/report.** No change to existing R1/R2 source equivalence or complete30 freeze. E remains locked. Keep debugger overhead timing as actual runtime provenance, failed cost separately, never combine failures into performance repetitions or conceal gdb runtime in comparisons. Failure to obtain a useful native location yields a concrete blocker: recurrent native process fault not localized with the available full-state reproducer; identify any missing symbol/debugger artifact precisely. No fifth guessed patch or dependency change follows automatically.

9. **Evidence/current boundary.** Strong evidence for repeated native failures and lost unsaved states; weak evidence for a specific cause. This document only proposes the finite full-process diagnostic. No commands, training or replacement execution occurred during design.

10. **Reviewed continuation after the actual2023 replacement.** Orchestrator reports the2023 gdb run completed normally (146.609s wall), all14 original scientific records/config/data/source/seed matched exactly, pruning and both checkpoints/completion passed;27/30 entries are now complete. This supports extending the same **instrumentation**, not a runtime repair. Subject to the same reviewer checking the three literal callers and the orchestrator's launch instruction, the original never-started AUTOIFS/KuaiRand seeds2024,2025,2026 may each run **once, sequentially**, with identical full protocol/R2/D1/cuda:1/gdb arguments. Only substitute the registered seed and its matching canonical output directory plus unique sidecar/stdout/stderr paths; leave the2023 script and all package/config sources unchanged. After each run, verify normal inferior termination, finite scientific outputs, original identity/stopping rules, complete pruning/postprocessing/checkpoints and truthful completion before launching the next. There is no prior same-seed trajectory for these three: do not invent a replay-equality claim or apply2023's14-epoch count to their stopping behavior. Any signal or identity/numeric failure stops the queue, with no automatic recovery, debugger continuation or retry. Record actual per-run costs as under-debugger measurements. This finite clause supersedes only item7's pending decision about those three starts; all other stop rules remain. Historical E may be read only after the unchanged complete30 R1/R2 freeze and original guards actually PASS. Root cause remains unknown/NOT_REPRODUCED; instrumentation success is not a fix.
