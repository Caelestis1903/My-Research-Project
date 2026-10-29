# Development Runner Log — MDMT-SOTA-2526-v1

Runner record updated 2026-10-29 (Asia/Shanghai). Raw results only.

## Scope

- Methods: HYPERGATE, AUTOIFS
- Datasets: MovieLens, KuaiRand
- Seeds: 2022, 2023
- Planned trajectories: 8
- Current status: 8 complete; 0 not run
- User authorization: `research_reports/mdmt_baselines_2526/RUN_AUTHORIZATION.json`

## Completed runs

| Method | Dataset | Seed | Status | Selected epoch | Final epoch | Train sec | Total sec | Peak CUDA bytes | Q selected macroLL / macroAUC / naturalAUC | Q fixed-final macroLL / macroAUC / naturalAUC |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| HYPERGATE | movielens | 2022 | TRAIN_COMPLETE_NOT_EVALUATED | 4 | 14 | 60.413 | 62.660 | 93294592 | 0.471111 / 0.812700 / 0.812706 | 0.467720 / 0.823046 / 0.822275 |
| AUTOIFS | movielens | 2022 | TRAIN_COMPLETE_NOT_EVALUATED | 1 | 2 | 25.293 | 25.889 | 228471808 | 0.478704 / 0.804029 / 0.804856 | 0.478091 / 0.804379 / 0.805170 |
| HYPERGATE | kuairand | 2022 | TRAIN_COMPLETE_NOT_EVALUATED | 10 | 20 | 305.640 | 313.951 | 124982272 | 0.459294 / 0.715623 / 0.743505 | 0.457881 / 0.719700 / 0.746728 |
| AUTOIFS | kuairand | 2022 | TRAIN_COMPLETE_NOT_EVALUATED | 1 | 2 | 117.069 | 118.687 | 487964672 | 0.462080 / 0.697753 / 0.741513 | 0.463314 / 0.697713 / 0.741724 |
| HYPERGATE | movielens | 2023 | TRAIN_COMPLETE_NOT_EVALUATED | 2 | 12 | 41.180 | 42.034 | 93294592 | 0.475071 / 0.807629 / 0.808052 | 0.464647 / 0.821057 / 0.820085 |
| AUTOIFS | movielens | 2023 | TRAIN_COMPLETE_NOT_EVALUATED | 2 | 3 | 26.208 | 26.770 | 228471808 | 0.474926 / 0.807952 / 0.808334 | 0.476827 / 0.808470 / 0.808840 |
| HYPERGATE | kuairand | 2023 | TRAIN_COMPLETE_NOT_EVALUATED | 8 | 18 | 273.898 | 277.294 | 124982272 | 0.461117 / 0.709398 / 0.740336 | 0.459172 / 0.715476 / 0.746047 |
| AUTOIFS | kuairand | 2023 | TRAIN_COMPLETE_NOT_EVALUATED | 3 | 4 | 136.535 | 138.135 | 488751104 | 0.462513 / 0.697397 / 0.740927 | 0.462140 / 0.697923 / 0.740803 |

Full configs and cell-level raw metrics are in [DEVELOPMENT_RAW_RESULTS.json](DEVELOPMENT_RAW_RESULTS.json). Epoch-level V metrics, checkpoints, predictions, and Q metric files are in `artifacts/mdmt_baselines_2526/runs/development/`. Stdout/stderr are in `artifacts/mdmt_baselines_2526/run_logs/`.

## Execution notes

- All completed runs use `cuda:1`; GPU0 was left untouched.
- Training/Q stderr files are empty; no runtime errors recorded.
- Runtime and CUDA peak allocation are emitted by the reviewed trainer.
- All eight planned development trajectories completed; no runs are missing.
- This report records raw metrics only and does not include scientific interpretation.

## Formal RUN record (partial; stopped after native crash)

- Frozen matrix: 3 methods × 2 datasets × seeds 2022–2026 = 30 fresh trajectories, authorized by root after `PRE_FORMAL_FREEZE.json`.
- Complete canonical training runs: 30/30, including the reviewed posttraining completion recovery for AUTOIFS/KuaiRand/2022 and the full-process GDB replacements for AUTOIFS/KuaiRand/2023–2026. No canonical partial failed run remains; one failed partial attempt is archived; no matrix trajectory remains unstarted. Five native-crash attempts are recorded; prior failed attempts remain in the failure history and are not extra matrix results.
- Completed groups: MovieLens all 15 trajectories; KuaiRand M3OE_FIXED1 all five seeds; KuaiRand HYPERGATE seeds 2022–2026; KuaiRand AUTOIFS seeds2022–2025.
- R2 bounded replay: HYPERGATE × KuaiRand seed2022 completed 31 epochs, selected=21/final=31, training 471.180s, total 474.653s, peak CUDA allocation 124,982,272 bytes, 52,235 steps. First five full training records matched attempt01 exactly after excluding only `seconds`. Evidence: [RUNTIME_IO_REPLAY_EVIDENCE.json](RUNTIME_IO_REPLAY_EVIDENCE.json).
- Stop condition: KuaiRand × HYPERGATE seed2025 attempt01 exited 139 after epoch19 and remains archived at `artifacts/mdmt_baselines_2526/attempts/formal_kuairand_HYPERGATE_seed2025_attempt01_SIGSEGV/`. The approved D1 attempt02 completed 26 epochs (selected=16/final=26), training 395.497s, total 398.959s, peak CUDA allocation 124,982,272 bytes, 43,810 steps. First19 training records matched exactly excluding seconds; evidence is [CRASH3_REPLAY_EVIDENCE.json](CRASH3_REPLAY_EVIDENCE.json). After the reviewed seed2022 posttraining completion recovery, AUTOIFS/KR seed2023 first failed and was archived intact, then completed on the authorized full-process GDB trajectory; seeds2024 and2025 have also completed under the reviewed D1 GDB callers, while seed2026 remains unstarted.
- GPU1 was the only device used; GPU0 was left untouched. GPU1 was checked before launches; peak allocation and epoch timing are recorded per run.
- No global freeze, E evaluation, or result-report CLI was run because the matrix is incomplete.
- Per-run full config, V epoch metrics, AutoIFS search/reuse records, checkpoint metadata, timing/cost, source/data identities, logs, failed attempts and missing seeds are in [FORMAL_RAW_RESULTS.json](FORMAL_RAW_RESULTS.json). Raw artifacts are preserved under `artifacts/mdmt_baselines_2526/runs/formal/` and `artifacts/mdmt_baselines_2526/attempts/`.

- Original AUTOIFS × KuaiRand seed2022 attempt exited139 at `training.py:539` after search10/reuse1–3 (epoch-time lower bound121.66783051693346s; original process wall129.106386328s). The three original checkpoints were present and preserved. After the separately reviewed one-time posttraining completion recovery, the canonical run is `TRAIN_COMPLETE_NOT_EVALUATED`, completion mode `CHECKPOINT_POSTTRAIN_RECOVERY`, selected2/final3/global_steps5486. Original `training_seconds`, `total_elapsed_seconds`, and `peak_cuda_bytes` remain null; recovery duration/inference timing are separately identified in the canonical manifest and recovery receipt. The original native-crash attempt remains in failed-attempt history and is not counted as a separate matrix result.


- AUTOIFS × KuaiRand seed2023 initial D1 attempt PID1520107 exited139/SIGSEGV after132.921s; its17-record partial (search0–9/reuse1–4) is preserved in the archive with21/21 files byte-equal per [AUTOIFS_KR2023_ARCHIVE_PRESERVATION.json](AUTOIFS_KR2023_ARCHIVE_PRESERVATION.json). The single reviewer-approved full-process GDB run PID1524176 ended normally (GDB exit0, inferior normal, external146.609s), completed full pruning/bookkeeping, selected3/final4, saved both checkpoints, and finished `TRAIN_COMPLETE_NOT_EVALUATED`. All first14 ordered scientific records matched archived records exactly excluding `seconds`; maximum macro AUC and macro LL differences were both0. R2 source/config/data identities and checkpoint embedded identities passed. Raw verification: [CRASH5_FULL_PROCESS_RAW_RESULTS.json](CRASH5_FULL_PROCESS_RAW_RESULTS.json). At that stage, seed2026, freeze and E evaluation remained unstarted.

- Canonical counts have been reconciled against the current run manifests: 29 canonical complete, including the seed2022 existing-state recovery, the seed2023 full-process GDB replacement, and the seed2024–2025 full-budget GDB runs; zero canonical partial; one archived failed attempt for seed2023; zero never started. All30 formal training identities are complete. Freeze and E evaluation were not yet started in this RUN segment.

- AUTOIFS × KuaiRand seed2024 completed under the reviewed full-process GDB caller: PID1527332, GDB/inferior exit normal, 131.425s external wall; selected1/final2, 10 search + 2 reuse epochs, full six-cell pruning summary and both checkpoints. Training120.594s, total122.380s, peak CUDA487,964,672B. D1/cuda:1, source/config/data identities and all scientific values checked. Detailed raw metrics remain in canonical `run.json`/epoch JSONL and per-seed GDB transcript/sidecar.

- AUTOIFS × KuaiRand seed2025 completed under the reviewed full-process GDB caller: PID1528077, GDB/inferior exit normal, 126.258s external wall; selected1/final2, 10 search + 2 reuse epochs, full six-cell pruning summary and both checkpoints. Training119.095s, total120.781s, peak CUDA487,964,672B. D1/cuda:1, source/config/data identities and scientific-value finiteness passed. Detailed per-cell values are retained in canonical run JSON and epoch JSONL. No seed2026/freeze/E started yet.

- AUTOIFS × KuaiRand seed2026 completed under the reviewed full-process GDB caller: PID1529065, GDB/inferior exit normal, 125.839s external wall; selected1/final2, 10 search + 2 reuse epochs, full six-cell pruning summary and both checkpoints. Training118.782s, total120.494s, peak CUDA487,964,672B. D1/cuda:1, source/config/data identities and scientific-value finiteness passed. Detailed per-cell values are retained in canonical run JSON and epoch JSONL. All30 canonical training manifests are now TRAIN_COMPLETE_NOT_EVALUATED; no E has been evaluated yet.

## Final formal RUN reconciliation — 2026-10-29

This section supersedes interim statements above that the formal matrix, freeze, E evaluations, or mechanical report were still pending. Final counts were refreshed from all 30 canonical `run.json` files.

- Canonical matrix: 30/30 `TRAIN_COMPLETE_NOT_EVALUATED` training manifests; zero canonical partials; zero unstarted identities. One failed partial seed2023 attempt remains archived and separate from the matrix. Five native process incidents remain recorded separately; the cause remains unresolved.
- AUTOIFS × KuaiRand/2022 retains `completion_mode=CHECKPOINT_POSTTRAIN_RECOVERY`; the original `training_seconds`, `total_elapsed_seconds`, and `peak_cuda_bytes` remain null. The original failed process is not counted as a second matrix trajectory.
- Runtime revisions remain limited to the approved runtime-only R2 amendment. Seven trajectories have measured D1 profile receipts; 23 have unspecified or unmeasured profile. All formal training/evaluation used `cuda:1`.
- The explicit amended freeze passed with 30 entries at `artifacts/mdmt_baselines_2526/final_freeze.json` before evaluation.
- Historical E reassessment: 60/60 views completed successfully (30 selected, 30 fixed-final). Each view has finite six-cell metrics and a prediction CSV whose row count matches `valid_requests`; D1 sidecars report CLI exit code 0. This partition was already exposed and is recorded as reassessment.
- The authorized mechanical report CLI completed: 456 per-cell rows, 120 aggregate rows, and 240 paired rows. Outputs include `aggregate_results.csv`, `per_cell_results.csv`, `paired_deltas.csv`, `baseline_comparison.md`, and `runtime_cost.csv`.
- `FORMAL_RAW_RESULTS.json` now includes all 30 canonical manifests read from disk and all 60 E metric records, prediction paths/counts, execution receipts, sidecars, and logs. Raw predictions and other run artifacts remain under `artifacts/mdmt_baselines_2526/runs/`.

This is an execution record only; scientific analysis is assigned to the orchestrator.
