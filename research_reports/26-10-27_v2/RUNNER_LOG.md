# RUNNER LOG — MDMT Relation-Basis v2

State: running. Reviewer verdict: PASS, blocking findings: 0. No scientific interpretation is recorded here.

| Item | Value |
|---|---|
| Experiment | MDMT-RELATION-BASIS-v2 stage 1 |
| Local Git checkout | unavailable; fixed source main `8eb461f5e0abc9c06385a4e8239e9621cae87ef8` |
| Environment | `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`, Python 3.13.5, PyTorch 2.9.0+cu126 |
| Seeds | 2022, 2023, 2024, 2025 |
| Matrix | 2 datasets × 6 methods × 4 seeds = 48 adaptation trajectories; 8 shared prefixes |
| Storage | models/optimizer/RNG/checkpoints RAM-only; logs and row-data caches on disk |

## Commands and run records

- Read-only audit: `python -m experiments.mdmt_relation_basis_v2 audit`; exit 0. Full stdout: `artifacts/mdmt_relation_basis_v2/logs/runner/audit.stdout.json`.
- Read-only plan: `python -m experiments.mdmt_relation_basis_v2 plan`; exit 0. Effective config hash `c08a1ffded211086145c300cffe1807ec014246ef443ce399fd59e11177d0b73`. Full stdout: `artifacts/mdmt_relation_basis_v2/logs/runner/plan.stdout.json`.
- MovieLens preparation: complete, exit 0. Raw SHA matches; full encoding, labels, fit-only preprocessing, and row-plan verification are true. Data hash `e0719739c5b83dd15e469b5d159025cd357ff9eca7586bd2cea55879cbbf8655`. Full stdout/stderr: `artifacts/mdmt_relation_basis_v2/logs/runner/prepare_movielens.stdout.log`.
- GPU at launch: GPU0 RTX 3090, 17652 MiB used, 231.65/350 W; GPU1 RTX 3090, 18 MiB used, 25.13/350 W. No process was interrupted.

MovieLens seed2022 started on physical GPU1 through `CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`, PID 98499, exec session 80834. Native epoch 1/24 validation LL 0.608143255, 1.09 seconds; epochs 2–5 completed in 0.91–0.92 seconds each, epoch5 validation LL 0.532320027. At the observed snapshot GPU1 memory was 531 MiB, power 125.93/350 W; GPU0 was 17652/24576 MiB and 233.45/350 W. MovieLens seed2023 started on physical GPU0, logical `cuda:0`, exec session 19827. No process was interrupted. Workers stay resident through READY until orchestrator release.

## Status

| Dataset | Seed | Prefix | Six adaptations | READY | E evaluation |
|---|---:|---|---|---|---|
| MovieLens | 2022 | running, PID 98499 | pending | pending | locked |
| MovieLens | 2023 | running | pending | pending | pending | locked |
| MovieLens | 2024 | pending | pending | pending | locked |
| MovieLens | 2025 | pending | pending | pending | locked |
| KuaiRand | 2022 | pending | pending | pending | locked |
| KuaiRand | 2023 | pending | pending | pending | locked |
| KuaiRand | 2024 | pending | pending | pending | locked |
| KuaiRand | 2025 | pending | pending | pending | locked |


## Runtime diagnostic failure and pause

Seed2022 PID98499 exited after the READ adaptation `run_receipt.json` was written. `diagnostics.latency()` raised `ValueError: Raw-X reconstruction mismatch` at `experiments/mdmt_relation_basis_v2/diagnostics.py:141`; worker `READY.json` was not produced, and RAM model/optimizer states were lost on process exit. No model files were written. Logs and already-exported B_REFERENCE/DIRECT/READ V/Q metric JSON remain in place.

At the orchestrator pause instruction, the following owned workers were recorded before stopping: seed2023 PID98833 on GPU0, elapsed 2:34, B_REFERENCE epoch19/20 after native24 and source12; seed2024 PID99295 on GPU0, elapsed 1:55, B_REFERENCE epoch7/20 after native24; seed2025 PID100310 on GPU1, elapsed 0:16, native epoch15/24. The seed2025 worker had been launched immediately before the pause message arrived; it is recorded as partial. Resource snapshot was GPU0 18638/24576 MiB, 232.05/350W; GPU1 1463/24576 MiB, 305.05/350W. No other user's process was touched.

The three unfinished workers were stopped at the orchestrator's direction because their shared diagnostics path follows the observed implementation failure. Their original logs/artifacts are preserved; none reached READY and no E evaluation was released. New workers and KuaiRand preparation are paused pending diagnosis/fix/review. No scientific results are reported from these partial runs.

## Revision 1 — repaired runtime numerics, independent run root

- Independent runtime-fix review: PASS, blocking 0 (`RUNTIME_FIX_REVIEW.md`). Source identity verified in the project environment: `72f3e825dac422784fd575bd24450e7858e99ce138f6c8261b3ae7636b8b13fc`. Config hash remains `c08a1ffded211086145c300cffe1807ec014246ef443ce399fd59e11177d0b73`.
- New artifact root: `/home/xzf/Code/phase0-3/artifacts/mdmt_relation_basis_v2/revision1`. All previous attempt and diagnostic artifacts remain under the original root and are excluded from revision1 completion accounting.
- Revision1 read-only audit and plan both exited 0. Audit confirmed MovieLens raw SHA and reused prepared dataset via `data/movielens` symlink; MovieLens `VALIDATED.json` exists. KuaiRand raw SHA verified, prepared data absent. Full outputs: `revision1/logs/runner/audit.stdout.json`, `plan.stdout.json`, `audit_pre_run.stdout.json`.
- Launch bookkeeping: detached shell launch at 17:21:48 did not leave a live Python worker; its empty log and partial initialization directory are preserved under `revision1/attempts/launch_detached_failure/` and excluded from run matrix. Resident worker is now held by exec session.
- MovieLens seed2022 started 2026-10-27 17:22:39 +08:00, PID 110427, exec session 29662, physical GPU1 via `CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`; config/source/data identity recorded in its worker directory. Started native epoch 1–14, with epoch1 validation LL 0.608143255 and epoch14 0.536372448; epoch durations 1.03–1.24 seconds observed. At snapshot GPU1 2339/24576 MiB and 282.77/350 W. Worker remains active in BUILDING_PREFIX.
- MovieLens seed2023 started 2026-10-27 17:23:02 +08:00, exec session 20639, physical GPU0 via `CUDA_VISIBLE_DEVICES=0`, logical `cuda:0`; worker directory records PID/status and is active. Launch snapshot GPU0 was 17652/24576 MiB and 213.00/350 W; another user's existing workload remains untouched.
- Both workers use `native epochs=24`, `source passes=12`, six methods × 20 adaptation epochs with prescribed warmup; no early-stop. E remains locked. READY workers will remain resident until global freeze.
- Accounting must include prior failed/aborted attempts and diagnostics as separate historical cost ledgers, plus all revision1 worker start/READY/E-release/completion times and resource snapshots.

### Revision1 live worker update (2026-10-27 17:25 +08:00)

| Dataset/seed | PID / exec session | Physical GPU | OS process start (+08:00) | Current STATUS | Persisted training log paths |
|---|---:|---:|---|---|---|
| MovieLens/2022 | 110427 / 29662 | 1 | 17:22:42 | TRAINING, LOWRANK_RELATION | `workers/movielens/seed2022/prefix/{native,source}_epoch_metrics.jsonl`; `runs/movielens/*/seed2022/{epoch_metrics.jsonl,stdout.log}` |
| MovieLens/2023 | 110791 / 20639 | 0 | 17:23:05 | TRAINING, B_REFERENCE | same path pattern under seed2023 |
| MovieLens/2024 | 111281 / 43607 | 1 | 17:23:51 | TRAINING, READ | same path pattern under seed2024 |
| MovieLens/2025 | 111787 / 47584 | 0 | 17:24:31 | BUILDING_PREFIX | same path pattern under seed2025; supervisor stdout `revision1/logs/workers/movielens/seed2025.stdout.log` |

Outer Python process output for seeds 2022–2024 was streamed in the held exec sessions; per-epoch native/source metrics and per-method stdout/metrics JSONL are persisted by the implementation. The later seed2025 outer process stdout is also redirected to the supervisor log above. At 17:25:24, GPU0 was 19590/24576 MiB and 206.60/350 W; GPU1 was 1888/24576 MiB and 204.99/350 W. All workers remain alive.

Seed2022 READ completed training and diagnostic before the worker proceeded to TARGET_ONLY. The persisted READ `costs.json` records parity `passed=true`, `finite=true` for batch1 and batch256. Batch1 max/mean absolute difference: 2.384185791015625e-7 / 1.7881393432617188e-7. Batch256: 2.384185791015625e-6 / 2.3182292352430522e-7, maximum at `[205,1]`, raw `-4.291540622711182`, cached `-4.291538238525391`, allowed `6.578999710083008e-6`; row identity hashes and cached reference are in `costs.json`.

### Revision1 readiness update (2026-10-27 17:27 +08:00)

MovieLens seed2024 reached `READY_FOR_GLOBAL_FREEZE` at 2026-10-27 17:26:51.519596291 +08:00 (PID111281, physical GPU1, total process elapsed about 3:00). `READY.json` is present under `revision1/workers/movielens/seed2024/`; worker remains alive/offloaded and waiting for global freeze. No E release/evaluation has occurred. Other three MovieLens seeds remain live and training. At 17:27:04, GPU0 was 18646/24576 MiB and 202.22/350 W; GPU1 was 987/24576 MiB and 125.82/350 W.

MovieLens seed2022 reached `READY_FOR_GLOBAL_FREEZE` at 2026-10-27 17:26:24.574308014 +08:00 (PID110427, physical GPU1; elapsed about 3:42). Its worker remains alive/offloaded waiting for global freeze; E remains locked. At 17:27:37, seeds2023 and 2025 were still training READ and DIRECT respectively; seed2024 remained READY.

MovieLens seed2023 completed READ epoch20/20 and its READ diagnostics/costs, then advanced to TARGET_ONLY. READ cross-batch parity passed for both batch sizes: batch1 max_abs 2.384185791015625e-7, batch256 max_abs 1.1920928955078125e-6 (duplicate entries identical). The full latency procedure remained CPU-active for about 100 seconds after epoch20 before writing `costs.json`; no error occurred.

### Revision0 final-file-count correction

The earlier pause snapshot rows above are observation-time snapshots, not final artifact counts or completed elapsed times. Reviewing the final persisted logs gives:

| Aborted MovieLens seed | Native metrics rows / epoch seconds | Source metrics rows / epoch seconds | Adaptation log rows / epoch seconds | Snapshot note |
|---|---:|---:|---:|---|
| 2023 | 24 / 61.80100833499455 | 12 / 23.4452955809993 | B_REFERENCE 20 / 56.14455756299867 | snapshot had reported B_REFERENCE 19/20 |
| 2024 | 24 / 63.69695874000172 | 12 / 17.61051732099986 | B_REFERENCE 11 / 30.315028934997827 | snapshot had reported B_REFERENCE 7/20 |
| 2025 | 24 / 22.691445335003664 | no source metrics file | none | snapshot had reported native 15/24 at observation time |

These final-file counts are cost/accounting evidence for the failed/aborted attempts only. They do not count as revision1 completion runs. Process elapsed values in the prior pause record remain approximate observations at the time of the stop request; use the final cost audit for wall-clock totals.

### KuaiRand preparation launched concurrently with remaining MovieLens workers

At the orchestrator's scheduling update, all four MovieLens seeds had been started in the planned order; MovieLens/2022 was fully READY, revision1 source/config identities remained fixed, and all E access remained globally locked. The unused GPU1 snapshot showed 1860 MiB and 195.52/350 W while two MovieLens workers were already offloaded; this justified beginning the independent KuaiRand data preparation and prioritizing its seed2022 on GPU1 without waiting for the remaining MovieLens latency work. No ML process was stopped or moved, and no method, seed, budget, or selection setting changed. KuaiRand preparation command was started at 2026-10-27 17:33:42 +08:00; full stdout is directed to `revision1/logs/runner/prepare_kuairand.stdout.log`. Start KR seed2022 on physical GPU1 after the prepared-data verification passes; add only seeds 2023–2025 if measured per-card memory and power stay below 85%.

KuaiRand preparation passed all raw identity and data-role checks. `VALIDATED.json` timestamp 2026-10-27 17:34:48.715282022 +08:00; data hash `853a9f37795042f04e987909cea6fd09922445faf907176688ee34ced20f028c`; verified encoding, labels, fit-only preprocessing and row plan all true. Full preparation stdout: `revision1/logs/runner/prepare_kuairand.stdout.log`.

KuaiRand seed2022 started on physical GPU1 (PID119863, exec session60362); KuaiRand seed2023 started on physical GPU0 (PID120153, exec session10685). Both use the fixed revision1 config/source and are live in BUILDING_PREFIX. seed2022 native epochs1/2 took 4.24/3.99s. After KR seed2023 startup, measured GPU0 was 20422/24576 MiB (83.1%) at 204.31/350 W; GPU1 was 3629/24576 MiB (14.8%) at 271.18/350 W. Current per-card headroom does not support another concurrent worker without exceeding the <85% power/memory rule; reassess after resident workers offload or resource use changes. Worker stdout logs are `revision1/logs/workers/kuairand_seed2022.stdout.log` and `...seed2023.stdout.log`.

Per the explicit strict `<85%` resource rule, the added KR workers were scheduled without an extra power margin. KR seed2024 started on physical GPU1 at 17:36:28 +08:00 (PID120841/session25968) after a measured 3.6 GiB / 284.84 W snapshot; the next snapshot was 5.4 GiB / 296.45 W (84.7%), still below both limits and with ample VRAM headroom, so KR seed2025 started on physical GPU1 at 17:36:45 +08:00 (PID121077/session17247). After both starts, all four KR seeds were live: seed2022 PID119863/GPU1, seed2023 PID120153/GPU0, seed2024 PID120841/GPU1, seed2025 PID121077/GPU1. At 17:37:02, GPU0 was 20416 MiB / 209.42 W and GPU1 was 5912 MiB / 293.15 W; both metrics remained below 85%. All four are still in prefix construction at this snapshot. No existing ML process was stopped; E remains locked.

At 2026-10-27 17:38:09 +08:00: KuaiRand seed2022 native24/24 and pilot11/12; seed2023 native16/24; seed2024 native14/24; seed2025 native12/24. GPU0 20416 MiB / 204.36 W, GPU1 5112 MiB / 279.06 W. All four KR processes remained live. MovieLens seeds2022/2024 READY; seeds2023/2025 still in LOWRANK_RELATION adaptation. No worker errors observed.

### KuaiRand seed2022 runtime failure

KuaiRand seed2022 PID119863 exited with `FAILED.json` timestamp 2026-10-27 17:39:25.189910731 +08:00 and `actual_elapsed_seconds=255.44353696500184`. Native24 and source12 completed; B_REFERENCE adaptation20 completed and its run receipt, V/Q outputs, and diagnostics preamble were preserved. Latency diagnostics rejected the batch256 FP32 comparison and no `costs.json` or `READY.json` was produced. Exact exception and full traceback are in `revision1/logs/workers/kuairand_seed2022.stdout.log` and `revision1/workers/kuairand/seed2022/FAILED.json`: max_abs `1.341104507446289e-6`, mean_abs `3.2409298000857234e-7`, max `[70,0]`, cached `-0.07308383285999298`, raw `-0.07308517396450043`, allowed `1.0950089827179907e-6`, finite true, passed false, row hash `14d410146cfe116b0cfee23654a29eee64cbadcb4f0180c7edd9c1e0870cbcce`, row range82–5659. No implementation or seed changes were made. Other ML/KR workers were left running.

At 2026-10-27 17:40:19 +08:00, the failed KR/2022 worker had released its GPU allocation. Other active statuses: MovieLens/2023 TRAINING FULL_RELATION (LOWRANK_RELATION costs already written); MovieLens/2025 still completing LOWRANK_RELATION diagnostics; KuaiRand/2023 source3/12; KuaiRand/2024 B_REFERENCE epoch6/20; KuaiRand/2025 source8/12. GPU0 was 19234 MiB / 205.61 W, GPU1 2083 MiB / 180.09 W. No additional seed was launched and no active worker stopped.

KuaiRand seed2024 and seed2025 also exited FAILED at B_REFERENCE latency after adaptation epoch20/20. Neither produced `costs.json` or `READY.json`; both complete per-method metric/receipt directories and `FAILED.json` plus full supervisor stdout remain. Exact values: seed2024 actual elapsed242.213809573s, max_abs1.6689300537109375e-6, mean_abs3.2575917430222034e-7, max index[139,0], cached−2.302630662918091/raw−2.3026323318481445, allowed_at_max3.993419861793518e-6; seed2025 elapsed240.327865493s, max_abs1.430511474609375e-6, mean_abs3.116001607850194e-7, max index[134,1], cached−4.102777004241943/raw−4.102778434753418, allowed_at_max6.333610105514526e-6. Both report finite=true, passed=false, batch256, row hash `14d410146cfe116b0cfee23654a29eee64cbadcb4f0180c7edd9c1e0870cbcce`, row range82–5659. Failure records and worker status are preserved; no retry or worker replacement was launched.

KuaiRand seed2023 also exited FAILED during B_REFERENCE latency after adaptation epoch20/20. `FAILED.json` timestamp 2026-10-27 17:43:48.723414071 +08:00, actual_elapsed_seconds497.100701077. Native24 and source12 completed; B_REFERENCE receipt/evaluation artifacts and diagnostics preamble were retained, but no costs/READY. Its batch256 error: finite=true/passed=false, max_abs1.8477439880371094e-6, mean_abs3.36891389451921e-7, max index[66,1], cached−0.9073416590690613/raw−0.9073398113250732, allowed_at_max2.1795441567897796e-6; row hash `14d410146cfe116b0cfee23654a29eee64cbadcb4f0180c7edd9c1e0870cbcce`, range82–5659. Full traceback is in `revision1/logs/workers/kuairand_seed2023.stdout.log`. No retry or replacement worker was launched.

At 17:44:16 +08:00 all four KuaiRand workers had failed at B_REFERENCE latency; none has been restarted. The four MovieLens workers remain alive: seeds2022/2024 `READY_FOR_GLOBAL_FREEZE`; seeds2023/2025 in `TRAINING FULL_RELATION`. GPU0 18646 MiB / 220.07 W, GPU1 987 MiB / 34.58 W. E remains locked.

MovieLens seed2023 reached `READY_FOR_GLOBAL_FREEZE` at 2026-10-27 17:45:03.595198594 +08:00 (PID110791, physical GPU0). All six methods completed their 20-epoch budgets and diagnostics/cost exports; the worker remains alive/offloaded waiting for freeze. E has not been released.

MovieLens seed2025 reached `READY_FOR_GLOBAL_FREEZE` at 2026-10-27 17:46:46.060350174 +08:00 (PID111787, physical GPU0). All four MovieLens seeds are now READY and remain resident/offloaded awaiting the global freeze; no E release/evaluation occurred. KuaiRand seed2022/2023/2024/2025 all failed during B_REFERENCE latency and remain recorded as failed. No replacement/retry has been launched; awaiting reviewed diagnostic change and orchestrator run instruction.

### R1 KuaiRand failures archived before R2 retry

Verified the four failed PIDs had exited. Moved each failed worker directory, every existing KuaiRand method/seed run directory (B_REFERENCE only for these attempts), and its full supervisor stdout into its distinct `revision1/attempts/KR_seedNNNN_R1_failed/` tree. Prepared KuaiRand data, all four MovieLens directories/runs and the four live ML PIDs were not moved. Cross-seed index: `revision1/attempts/KR_R1_FAILED_ARCHIVE_INDEX.json`; each seed archive has its own `ARCHIVE_INDEX.json`. Index preserves each old source identity, exact failed JSON/error and actual elapsed cost. R1 elapsed seconds: 2022 255.443536965; 2023 497.100701077; 2024 242.213809573; 2025 240.327865493.

R2 independent review PASS/blocking0 (`RUNTIME_FIX2_REVIEW.md`); current source hash verified as `26708434c81106dd424d52b4de258bee8aad7b35640b1d50b8699290694a2a3c`, common training-source hash `29d7018f037c06af567d58646c6f81d022cab101a0e87c86c99af295e751cd98`. Four MovieLens workers remain READY with original R1 identities. R2 KR retries use the original revision1 root, the same seeds/budgets/config, and `--freeze-file revision1/GLOBAL_FREEZE_FP32_STANDARD.json`; each new worker's own source identity must show R2. No E release.

### R2 KuaiRand retry launch

After verifying the four failed R1 PIDs had exited, archived their worker directories, every existing method/seed run directory and per-seed outer stdout into `revision1/attempts/KR_seedNNNN_R1_failed/`. Cross-seed and per-seed archive indexes preserve old source identities, exact FAILED JSON, logs, and actual elapsed seconds. Prepared data and all MovieLens artifacts/processes remain at their original paths. R2 full source identity was rechecked in the project environment: `26708434c81106dd424d52b4de258bee8aad7b35640b1d50b8699290694a2a3c`; common training-source identity from reviewed equivalence is `29d7018f037c06af567d58646c6f81d022cab101a0e87c86c99af295e751cd98`.

R2 KuaiRand seed2022 started 2026-10-27 17:53:52 +08:00, PID133279/session43529, physical GPU1; seed2023 started 17:54:10 +08:00, PID133567/session84096, physical GPU1. Both command lines use the existing explicit freeze file `revision1/GLOBAL_FREEZE_FP32_STANDARD.json` and R2 source identities verified in their `source_identity.json`. Both outer stdout files are fresh logs in `revision1/logs/workers/`. Seed2022 reached native epoch6/24. After adding seed2023, the 17:54:19 GPU snapshot was GPU1 4525/24576 MiB, 306.16/350 W (power above 85%); no additional KR seed was started at that snapshot. Existing workers continue unchanged. E remains locked.

R2 KuaiRand seed2024 started 17:56:01 +08:00 (PID134831/session24477) and seed2025 started 17:56:18 (PID135061/session46393), both physical GPU1 and with the prescribed `--freeze-file revision1/GLOBAL_FREEZE_FP32_STANDARD.json`. All four R2 seed identities are verified at full source hash `26708434c81106dd424d52b4de258bee8aad7b35640b1d50b8699290694a2a3c`; the four original ML workers remain alive with their R1 identities. At 17:56:25 GPU1 was 6882/24576 MiB and 299.43/350 W, slightly above 85% power after all four finite seeds had started; no additional worker is pending. E remains locked.

R2 live progress at 2026-10-27 17:58:22 +08:00: KR2022 B_REFERENCE epoch13/20; KR2023 B_REFERENCE epoch7/20; KR2024 native20/24; KR2025 native17/24. GPU1 5544 MiB / 297.43 W (below 85% at this reading); all four seed processes are already active, so no extra run was launched. No R2 parity failure observed yet; first B_REFERENCE costs/parity record pending completion of its 20 epochs and latency pass.

### First R2 raw-X parity pass

KuaiRand seed2022 B_REFERENCE diagnostics/costs completed at 2026-10-27 17:59:03.572877455 +08:00. Batch1 and batch256 both report `finite=true`, `passed=true`. Batch1 max_abs 2.384185791015625e-7 and max_normalized_error 0.017048146622162228. Batch256 max_abs 1.341104507446289e-6, mean_abs3.2409298000857234e-7, worst_index[70,0], max_normalized_error0.13284827281899145, rowhash `14d410146cfe116b0cfee23654a29eee64cbadcb4f0180c7edd9c1e0870cbcce`, row range82–5659. The worker continued into DIRECT adaptation; R2 full source identity is in its worker manifest.

R2 KuaiRand seed2024 B_REFERENCE parity passed; `costs.json` timestamp 18:00:27 +08:00. Batch1 max_abs2.384185791015625e-7, max_normalized_error0.016706735321012935; batch256 max_abs1.6689300537109375e-6, max_normalized_error0.13212864813725997, worst_index[188,1], rowhash `14d410146cfe116b0cfee23654a29eee64cbadcb4f0180c7edd9c1e0870cbcce`, range82–5659. Worker advanced to DIRECT. Seed2025 B_REFERENCE epoch20/20 was awaiting costs at this snapshot; seeds2022/2023 were in READ.

R2 KuaiRand seed2025 B_REFERENCE parity passed; batch1 and batch256 both finite/pass true. Batch1 max_abs4.76837158203125e-7/max_normalized_error0.03232060363303882; batch256 max_abs1.430511474609375e-6/max_normalized_error0.13518647392828106, worst_index[150,0], rowhash `14d410146cfe116b0cfee23654a29eee64cbadcb4f0180c7edd9c1e0870cbcce`, row range82–5659. Worker advanced to DIRECT. The first three R2 KR workers to complete B_REFERENCE parity all passed.

R2 KuaiRand additional diagnostics: seed2023 READ costs passed batch1 and batch256 (`max_abs` 4.76837158203125e-7 / 1.9073486328125e-6); worker advanced to TARGET_ONLY. Seed2025 DIRECT costs passed both batches (`max_abs` 2.384185791015625e-7 / 1.3113021850585938e-6); worker advanced to READ. No new errors. At 18:01:29 all four R2 workers remained active.

R2 KuaiRand seed2024 READ parity passed and worker advanced to TARGET_ONLY. Batch1 max_abs4.76837158203125e-7/max_normalized_error0.029656928150981664; batch256 max_abs1.6689300537109375e-6/max_normalized_error0.1271261737018286; both finite/pass true. Seed2025 READ epoch20/20 had entered diagnostics at this point.

R2 KuaiRand seed2025 READ parity passed and worker advanced to TARGET_ONLY. Batch1 max_abs2.384185791015625e-7/max_normalized_error0.01675107971261928; batch256 max_abs1.430511474609375e-6/max_normalized_error0.13548558847320527; both finite/pass true. At 18:03:16, seed2022 LOWRANK_RELATION epoch19/20, seed2023 LOWRANK_RELATION epoch15/20, seed2024 TARGET_ONLY epoch12/20 and seed2025 TARGET_ONLY epoch5/20.

R2 KuaiRand seed2022 reached `READY_FOR_GLOBAL_FREEZE` at 2026-10-27 18:05:52.500936347 +08:00 (PID133279, physical GPU1, source hash `26708434c81106dd424d52b4de258bee8aad7b35640b1d50b8699290694a2a3c`). All six method budgets and diagnostics completed; worker remains resident/offloaded waiting on the explicit `GLOBAL_FREEZE_FP32_STANDARD.json` path. No E release.

R2 KuaiRand seed2023 reached `READY_FOR_GLOBAL_FREEZE` at 2026-10-27 18:06:03.773488449 +08:00 (PID133567, physical GPU1, R2 source). All six method budgets and diagnostics/cost exports completed; worker remains resident/offloaded waiting on the explicit compatibility freeze file. No E release.

R2 KuaiRand seed2024 reached `READY_FOR_GLOBAL_FREEZE` at 2026-10-27 18:06:52.907894554 +08:00 (PID134831, GPU1). All six method budgets and diagnostics completed; FULL_RELATION parity batch1/batch256 both passed with max_abs4.76837158203125e-7 / 1.430511474609375e-6. Worker remains resident/offloaded waiting for global freeze; no E release.

## Revision1 matrix readiness and raw-record audit

All eight resident workers are now `READY_FOR_GLOBAL_FREEZE`; their processes remain alive/offloaded and wait for their freeze/release handshake. E has not been released. Read-only inventory confirmed for every dataset/seed: native24/24, source12/12, six run receipts, six costs records, and exactly 20 adaptation epochs in each of the six methods (48/48 trajectory receipts total). `READY.json` lists all six methods. MovieLens workers retain R1 full source `72f3e825dac422784fd575bd24450e7858e99ce138f6c8261b3ae7636b8b13fc`; KuaiRand retries use R2 full source `26708434c81106dd424d52b4de258bee8aad7b35640b1d50b8699290694a2a3c`; the reviewed common training-source hash is `29d7018f037c06af567d58646c6f81d022cab101a0e87c86c99af295e751cd98`. All workers preserve their exact source identity. Config hash remains `c08a1ffded211086145c300cffe1807ec014246ef443ce399fd59e11177d0b73`.

| Dataset/seed | PID / GPU | Started (+08:00) | READY (+08:00) | Observed full worker wall |
|---|---:|---|---|---:|
| MovieLens/2022 | 110427 / GPU1 | 17:22:42 | 17:26:24.574308014 | 3m42s |
| MovieLens/2023 | 110791 / GPU0 | 17:23:05 | 17:45:03.595198594 | 21m58s |
| MovieLens/2024 | 111281 / GPU1 | 17:23:51 | 17:26:51.519596291 | 3m00s |
| MovieLens/2025 | 111787 / GPU0 | 17:24:31 | 17:46:46.060350174 | 22m15s |
| KuaiRand/2022 R2 | 133279 / GPU1 | 17:53:52 | 18:05:52.500936347 | 12m00s |
| KuaiRand/2023 R2 | 133567 / GPU1 | 17:54:10 | 18:06:03.773488449 | 11m54s |
| KuaiRand/2024 R2 | 134831 / GPU1 | 17:56:01 | 18:06:52.907894554 | 10m52s |
| KuaiRand/2025 R2 | 135061 / GPU1 | 17:56:18 | 18:06:55.640076871 | 10m38s |

R1 KuaiRand failed attempts and their actual runtime costs remain separately preserved under `revision1/attempts/KR_seedNNNN_R1_failed/`, including the per-seed and cross-seed indexes. No extra seeds, retried successful ML workers, or E evaluations were run. Final Q/evaluation release and compatible freeze remain the orchestrator's next stage.

Structured raw metrics/cost report: `research_reports/26-10-27_v2/RUNNER_RAW_RESULTS.json`. It contains all 48 completed run receipts, full per-role/per-state V/Q metric cells (including per-domain/per-task AUC/loss), costs/latency records, per-worker identity, and a separate archive of four failed R1 KR attempts. It makes no research interpretation and records E as unreleased.


## Final historical E evaluation (2026-10-27)

Orchestrator authorization: `STAGE1_REPORT.md` and `STAGE_DECISION.json` were complete; reviewed canonical freeze manifest hash `e7964525d8a176867abc3c36d68d8e25f903a8e5f42ef20bca6dee2c5f5167c2` and compatible views were present. Runner did not select methods, alter configs, or train additional models.

Command for each worker: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.mdmt_relation_basis_v2 --artifact-root /home/xzf/Code/phase0-3/artifacts/mdmt_relation_basis_v2/revision1 evaluate --dataset DATASET --seed SEED`. CLI release stdout and timestamps are in `E_RELEASE_*.log`; worker stdout remains in the original `artifacts/mdmt_relation_basis_v2/revision1/logs/workers/` files.

| Worker | Released | Completed | PID | Total worker wall (s) | Status |
|---|---|---|---:|---:|---|
| movielens/seed2022 | 2026-10-27T18:11:43+08:00 | 2026-10-27T18:11:51.356108+08:00 | 110427 | 2947.528055 | COMPLETE |
| movielens/seed2023 | 2026-10-27T18:12:01+08:00 | 2026-10-27T18:12:34.672155+08:00 | 110791 | 2967.183246 | COMPLETE |
| movielens/seed2024 | 2026-10-27T18:11:44+08:00 | 2026-10-27T18:11:52.152184+08:00 | 111281 | 2878.691131 | COMPLETE |
| movielens/seed2025 | 2026-10-27T18:12:02+08:00 | 2026-10-27T18:12:35.824260+08:00 | 111787 | 2882.496091 | COMPLETE |
| kuairand/seed2022 | 2026-10-27T18:10:52+08:00 | 2026-10-27T18:11:16.748776+08:00 | 133279 | 1042.810649 | COMPLETE |
| kuairand/seed2023 | 2026-10-27T18:10:53+08:00 | 2026-10-27T18:11:21.221212+08:00 | 133567 | 1028.355090 | COMPLETE |
| kuairand/seed2024 | 2026-10-27T18:10:54+08:00 | 2026-10-27T18:11:23.133398+08:00 | 134831 | 918.194701 | COMPLETE |
| kuairand/seed2025 | 2026-10-27T18:10:55+08:00 | 2026-10-27T18:11:22.505337+08:00 | 135061 | 899.575515 | COMPLETE |

All eight workers exited naturally after writing `COMPLETE.json`; no model files were written. E output inventory: 96 selected/final metric JSON files, 96 prediction NPZ files, and 576 domain-task metric cells (6 per metric record). Each record includes macro AUC, macro log loss, natural log loss, pooled AUC by task, and per-domain/task counts, AUC, and log loss.

`python -m experiments.mdmt_relation_basis_v2 --artifact-root .../revision1 report --report-root research_reports/26-10-27_v2` refreshed `RESULTS_SUMMARY.json` and `SEED_METRICS.csv`; the full raw E records are included in `RUNNER_RAW_RESULTS.json`. E timestamp/cost index: `E_EVALUATION_LEDGER.json`.
