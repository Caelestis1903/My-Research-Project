# EXPERIMENT EXECUTION REPORT

## 1. Experiment ID

`AUG_HISTORY_20261105`, Route C, MovieLens-only `C_FULL`, output `artifacts/aug_history_20261105/attempt_C1`.

## 2. Git/worktree state

- Worktree: `/home/xzf/Code/phase0-3_aug_history_20261105`
- Branch: `26-11-05`
- HEAD: `933cb2f02790722af2fa4ea704f993b79df29afa`
- Implementation source hash: `d78a938330e5eb8334213c7f0beed66926956c35e1cff90c180260d12fd331e2`
- Manifest SHA-256: `76345194a8d09c8cadf3544e39c35174a452abb3827f071250da7f259c45749d`
- Review acceptance: `PASS`, MovieLens scope; SHA-256 `7347215828f1fb094b17c0a9e6fc9a38ccbb314675f0f553a023235e2ee4db22`.
- Dataset hash: MovieLens `9ec469c348ddfb359019f8072a866d2e9e832f3bc87cb9adbbfa76a2c6ecc42f`.

## 3. Config

- Scope: `movielens` only.
- Parent config: fixed1, 773,470 parameters; batch 40,960; 50 epochs; 20 steps/epoch; FP32; Adam lr 0.01, betas (0.9, 0.999), eps 1e-8, coupled weight decay 1e-4; κ 0.05, λ 0.5, ρ 0.9828205985452511.
- Perturbation fields: user_id, movie_id, gender, occupation, zip, cate_id; target domain d0.
- Numeric configurations resolved and run:

| Candidate | gamma | Refresh K | Mode | Seeds |
|---|---:|---:|---|---|
| C1_NEG | -2.0 | 20 | task | 2022–2026 |
| C2_POS | +2.0 | 20 | task | 2022–2026 |
| C3_WEAK | +0.5 | 20 | task | 2022–2026 |
| C4_CADENCE | +0.5 | 40 | task | 2022–2026 |
| C_MERGED | +0.5 | 20 | merged | 2022–2026 |
| C_ROTATED | +0.5 | 20 | rotated | 2022–2026 |

- W12 recorded C2_POS (choice hash `71f2cd789f5779d1c74aaef967358f6adace505ac643fe5f68cccf4a6b0fa145`).
- W123 recorded C3_WEAK (choice hash `866d3ce36fe9c9376e35db350296744b888ebcda76afd4a8d38faed937500ec8`).
- C_WINNER recorded C3_WEAK, gamma +0.5/K20/task (choice hash `7ffc4cb56f12ab06ba9878d5fcd27beff3e2052eb17e9d0bf1442f68e5b97cfe`).
- The validation decision recorded `selected_candidate=C_MERGED`, `search_winner=C3_WEAK`, and `STOP_AND_REPORT`; `test_consulted=false`.

## 4. Seeds

Five seeds for every numeric configuration: 2022, 2023, 2024, 2025, 2026. The repository's default extra seed 2026 was included as specified by the frozen C matrix.

## 5. Run status

- Preflight: PASS.
- C suite validation and GPU smoke: PASS.
- Cross-stage continuation from frozen A decision: PASS; authorized action was MovieLens C_FULL.
- Train: `C_TRAIN_COMPLETE`; 30/30 jobs complete, each with exactly 50 epoch records. Total: 1,500 epochs, 30,000 Adam steps, 3,000 online/EMA validation records.
- Select: `HPO_FROZEN`.
- Decide: `DECIDED` on validation, before test evaluation.
- Evaluate: `EVALUATED` for frozen endpoints.
- Report: `C_REPORTED`; 330 endpoint rows, 16,830 paired rows, 3,366 summary rows.
- Final continuation: `STOP_AND_REPORT`, no further commands.
- KuaiRand C: `DEFERRED_NOT_RUN`.
- RAM guardian remains retained pending explicit release from the orchestrator.

The coordinator `progress.json` remained stale during adopted worker execution; final counts above are from per-job `runs/<job_id>/epochs/E*.json` and `TRAIN_COMPLETE.json` records. There are 30 worker logs and 30 completion markers.

## 6. Raw metrics

The exact per-seed, per-domain/task, and per-cell validation/test AUC and logloss records are in `C/ENDPOINTS.csv`; full validation histories are in `C/VALIDATION_CURVES.csv`; paired per-seed and summary metrics with SD and 95% CI are in `C/PAIRED_SEED.csv` and `C/PAIRED_SUMMARY.csv`. The table below gives unweighted five-seed means for fixed-EMA selected test endpoints, extracted from `C/ENDPOINTS.csv`.

### Selected fixed-EMA test metrics (mean across seeds)

| Candidate | Macro AUC | Macro logloss | d0 macro AUC | d1 macro AUC | d2 macro AUC | Task 0 macro AUC | Task 1 macro AUC | d0 macro LL | d1 macro LL | d2 macro LL | Task 0 macro LL | Task 1 macro LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3_WEAK | 0.779087147 | 0.496417590 | 0.781901994 | 0.791764949 | 0.763594500 | 0.770796729 | 0.787377566 | 0.496813858 | 0.482556522 | 0.509882391 | 0.580728440 | 0.412106741 |
| C_MERGED | 0.778689413 | 0.496180076 | 0.781472435 | 0.791264134 | 0.763331670 | 0.770410598 | 0.786968228 | 0.496627501 | 0.482783081 | 0.509129645 | 0.580920611 | 0.411439541 |
| C_ROTATED | 0.779068244 | 0.496316116 | 0.781649654 | 0.791813498 | 0.763741579 | 0.770783409 | 0.787353078 | 0.496865937 | 0.482651092 | 0.509431319 | 0.580746359 | 0.411885873 |

Per-seed macro AUC / macro logloss, in seed order 2022, 2023, 2024, 2025, 2026:

| Candidate | Macro AUC by seed | Macro logloss by seed |
|---|---|---|
| C3_WEAK | 0.779266950, 0.778480060, 0.779860801, 0.778664359, 0.779163567 | 0.495711744, 0.497982365, 0.493682181, 0.495160340, 0.499551322 |
| C_MERGED | 0.778799971, 0.778630239, 0.779474538, 0.778208888, 0.778333428 | 0.497329426, 0.495621340, 0.494793563, 0.495678835, 0.497477216 |
| C_ROTATED | 0.779612180, 0.778397131, 0.778879779, 0.778808871, 0.779643257 | 0.495165413, 0.497316564, 0.494470793, 0.495022318, 0.499605494 |

Validation HPO summary fields (`score`, `score_sd`, measured cost seconds) from `C/HPO_SUMMARY.csv`:

| Candidate | gamma/K | Complete | Score | Score SD | Measured cost seconds |
|---|---|---|---:|---:|---:|
| C1_NEG | -2 / 20 | yes | 0.7594828673 | 0.0004044094 | 2745.5490 |
| C2_POS | +2 / 20 | yes | 0.7595265456 | 0.0003905533 | 2609.4607 |
| C3_WEAK | +0.5 / 20 | yes | 0.7596043456 | 0.0003881442 | 2833.0154 |
| C4_CADENCE | +0.5 / 40 | yes | 0.7595231039 | 0.0005089833 | 2663.8349 |
| C_MERGED | +0.5 / 20 | yes | 0.7595028385 | 0.0004027518 | 2390.0882 |
| C_ROTATED | +0.5 / 20 | yes | 0.7595242563 | 0.0003579009 | 2207.7495 |

### Recorded costs

Totals from 30 rows in `C/COSTS.csv`: 30,000 Adam steps; 3,000 validation records; 1,200,250,500 original rows; 424,181,280 extra-view rows; 13,836.4489 training seconds; 118.5459 probe seconds; 1,494.7029 validation seconds; 15,449.6977 GPU/wall seconds.

The source fields `gpu_seconds` and `wall_seconds` both sum worker training, probe, and validation timers. They include concurrent GPU waiting and do not measure machine elapsed time or exclusive CUDA kernel time. IPC and frozen evaluation are outside that sum.

## 7. Log and artifact locations

- Per-job full worker logs: `artifacts/aug_history_20261105/attempt_C1/runs/<job_id>/worker.log` (30 files).
- Per-epoch raw records: `artifacts/aug_history_20261105/attempt_C1/runs/<job_id>/epochs/E001.json` … `E050.json` (1,500 records total).
- Per-job completion markers: `runs/<job_id>/TRAIN_COMPLETE.json` (30 files).
- Validation test log: `artifacts/aug_history_20261105/attempt_C1/validation_tests.log`.
- GPU engineering smoke receipt: `artifacts/aug_history_20261105/attempt_C1/GPU_SMOKE.json`.
- Admission, IPC, and epoch transaction logs: `ADMISSIONS.jsonl`, `IPC_COSTS.jsonl`, `epoch_commit_ledger.jsonl` under the attempt directory.
- Frozen/raw report artifacts: `C/selection_frozen.json`, `C/DECISION.json`, `C/EVALUATED.json`, `C/ENDPOINTS.csv`, `C/HPO_SUMMARY.csv`, `C/HPO_SELECTION.json`, `C/VALIDATION_CURVES.csv`, `C/PAIRED_SEED.csv`, `C/PAIRED_SUMMARY.csv`, `C/COSTS.csv`, `C/REPORT.md`.

## 8. Runtime warnings/errors

- Runtime errors: none.
- Nonfatal PyTorch TF32 setter deprecation warning during validation.
- Nonfatal NumPy read-only-array-to-tensor warning during evaluation.
- No failed or missing jobs. All 30 jobs have 50 epoch records and a `TRAIN_COMPLETE.json` marker.

## 9. Commands and exit status

All commands used `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`.

```text
python -m experiments.aug_history_20261105.run preflight --manifest research_plans/26-11-5/experiment_manifest.json --route C --datasets movielens --output artifacts/aug_history_20261105/attempt_C1
exit 0 — PREFLIGHT_PASS

python -m experiments.aug_history_20261105.run validate --manifest research_plans/26-11-5/experiment_manifest.json --route C --suite C --gpu-smoke --devices 0 1 --output artifacts/aug_history_20261105/attempt_C1
exit 0 — validation PASS; GPU_SMOKE PASS

python -m experiments.aug_history_20261105.run continue --manifest research_plans/26-11-5/experiment_manifest.json --route C --decision artifacts/aug_history_20261105/attempt_A1/A/DECISION.json --output artifacts/aug_history_20261105/attempt_C1 --resume
exit 0 — CONTINUATION_RESOLVED

python -m experiments.aug_history_20261105.run train --manifest research_plans/26-11-5/experiment_manifest.json --route C --datasets movielens --seeds 2022 2023 2024 2025 2026 --devices 0 1 --output artifacts/aug_history_20261105/attempt_C1 --resume
exit 0 — C_TRAIN_COMPLETE

python -m experiments.aug_history_20261105.run select --manifest research_plans/26-11-5/experiment_manifest.json --route C --split validation --output artifacts/aug_history_20261105/attempt_C1
exit 0 — HPO_FROZEN

python -m experiments.aug_history_20261105.run decide --manifest research_plans/26-11-5/experiment_manifest.json --route C --basis validation --output artifacts/aug_history_20261105/attempt_C1
exit 0 — DECIDED; test_consulted=false

python -m experiments.aug_history_20261105.run evaluate --manifest research_plans/26-11-5/experiment_manifest.json --route C --frozen artifacts/aug_history_20261105/attempt_C1/C/selection_frozen.json --device cuda:0 --output artifacts/aug_history_20261105/attempt_C1
exit 0 — EVALUATED

python -m experiments.aug_history_20261105.run report --manifest research_plans/26-11-5/experiment_manifest.json --route C --output artifacts/aug_history_20261105/attempt_C1
exit 0 — C_REPORTED

python -m experiments.aug_history_20261105.run continue --manifest research_plans/26-11-5/experiment_manifest.json --route C --decision artifacts/aug_history_20261105/attempt_C1/C/DECISION.json --output artifacts/aug_history_20261105/attempt_C1 --resume
exit 0 — STOP_AND_REPORT
```

## 10. Key SHA-256 hashes

| Artifact | SHA-256 |
|---|---|
| `PREPARED.json` | `5995419b2b1d52e6840cb4a930b478ea859e5c39090f627659319ec8cc8273b7` |
| `GPU_SMOKE.json` | `5ccf4307c4fa36fb730971e9f3c24ed6588c01ce317e8a2f27ee9c5afeb15ae8` |
| `C/selection_frozen.json` | `214e857f1ff9b59356abad9a27f58231abc16590836221224054817ab96063c1` |
| `C/DECISION.json` | `39b7555c3fc0680b16b261c64efe8013e112bdf7206a5e0a24eec00d498fef54` |
| `C/EVALUATED.json` | `1cb083056b698f54bf918294f9779ff4dbd6cf103449eeb7a80f6e59c8b4b302` |
| `C/ENDPOINTS.csv` | `c74a65284cdb4bda8eab94ed73f590716667e6556ae0ef62907712807e3b79ba` |
| `C/HPO_SUMMARY.csv` | `83366c8f7759b6cf076468c750b6b80bd5ca978fe9bd483912189db4abb45f92` |
| `C/HPO_SELECTION.json` | `41a8b61254fafe36a3d6856e3deea8ab07978283ddb6b5b6e8ea8ec2fe696d40` |
| `C/VALIDATION_CURVES.csv` | `28d06bbeef2e7195b9d2df0b4d316dbbc7acf1d4d0fd3a096c76d40caa11f5b8` |
| `C/PAIRED_SEED.csv` | `c81333a2a18328eded4be3dae9b836e0f13598f88829e94aece7326d30676d1b` |
| `C/PAIRED_SUMMARY.csv` | `39da6dc1eecd19435d960f125560fd93585d11baa0853b78f25118f6c143b9b5` |
| `C/COSTS.csv` | `f07d448c2790421b9d653aa6baaf0101ebe1d3c65975f83e88d0f34580ba51d3` |
| `validation_tests.log` | `bd709405a8fe884f213c9c50431d604d9d418b9336b005c0eecb3c48a5a5b543` |
| `ADMISSIONS.jsonl` | `12ff3560c4d2ad3d53f21825e713614cad9466d7d449533115a3988bf6512739` |
| `IPC_COSTS.jsonl` | `f423d6dce053d93771cf4c9eeb5062518f59f3e347dda39e040aaf4ddb2ce32c` |
| `epoch_commit_ledger.jsonl` | `4bf35f9eee2de073c38c717fd719b2e6a44451d115f902927c8b4dcd69850ef1` |

GPU smoke receipt fields: `batch_size=40960`; `gamma_zero_steps=2`; `conditional_steps=21`; `last_valid_statistic_step=21`; `nonzero_statistics=true`; `old_teacher_step=20`; peak GPU allocation 1,168,113,664 bytes. Purpose is `ENGINEERING_ONLY`.

## 11. Authorized guardian release

- Authorization ID: `C_RELEASE_20261106_01` (one-time; consumed).
- Command executed exactly once: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.aug_history_20261105.run release --route C --manifest research_plans/26-11-5/experiment_manifest.json --output artifacts/aug_history_20261105/attempt_C1`
- Numeric exit code: `0`.
- CLI result: `{"status": "RELEASED"}`.
- Stored guardian PID: `118013`.
- Read-only check: `ps -p 118013 -o pid=,stat=,args=` returned no process row (ps exit code `1`); guardian PID is absent.
