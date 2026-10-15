# Stage S6 Phase B execution report

## 1. Experiment ID and execution scope

**Experiment:** S6 Phase B — user-requested diagnostic completion of the F/T/HT/DH optimizer matrix under the frozen Sample objective.

Phase A remains recorded as STOP in the existing S6 analysis. Phase B was run as the user-authorized extension documented in [stage_S6_phaseB_extension_scope.md](stage_S6_phaseB_extension_scope.md); these runs do not alter the Phase A gate record. This report contains execution and audit records only, with no scientific interpretation.

**Stages:** S6 KuaiRand/natural and S6ML MovieLens/d2_large. The S6 Phase B source inventory and the S6ML Phase B source inventory each froze successfully. The 21 inherited source pins and the training run source identity were checked against the canonical Phase A manifests before launch; the live run-source hash matched for all four Phase A seeds in each dataset. Phase B collectors later verified producer-critical source identity and paired F/T/HT/DH manifests within each seed.

A standalone S6 reviewer markdown was not present in `experiments/reports`. In accordance with the direct user request and orchestration authorization, execution proceeded using the existing frozen implementation, Phase A structural reports, and fail-closed run validators. No algorithm or training code was changed by the runner.

## 2. Git/worktree state

Git state is unavailable: `git rev-parse --show-toplevel` returned “fatal: not a git repository” at `D:\CODE\phase0-3`.

## 3. Config and commands

The frozen S6 design was used: `objective_weighting=sample`; M3OE; KuaiRand `natural` and MovieLens `d2_large`; arms F/T/HT/DH; seeds 2022, 2023, 2024, 2025; 50 epochs; batch size 40960; main learning rate 0.01; architecture learning rate 0.001; weight decay 1e-5; Adam betas (0.9, 0.999); CUDA device 0. The existing orchestrators supplied the remaining locked parameters and generated append-only run manifests.

Commands, launched from `D:\CODE\phase0-3`:

```powershell
.\venv\Scripts\python.exe experiments\stage_S6_orchestrator.py --phase B --reviewer-pass --python .\venv\Scripts\python.exe
.\venv\Scripts\python.exe experiments\stage_S6ML_orchestrator.py --phase B --reviewer-pass --python .\venv\Scripts\python.exe
.\venv\Scripts\python.exe experiments\stage_S6_report.py --phase B
.\venv\Scripts\python.exe experiments\stage_S6ML_report.py --phase B
```

KuaiRand was run to completion before MovieLens began. Within each seed wave, exactly two training Python processes (HT and DH) ran concurrently. Both Phase B queues record `max_training_processes=2`; all eight waves proved positive process overlap and `max_live_observed=2`.

## 4. Environment

- Python: `D:\CODE\phase0-3\venv\Scripts\python.exe`
- PyTorch: `2.9.0+cu126`; CUDA available: `True`; PyTorch CUDA: `12.6`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB; driver 596.49
- Shell: PowerShell on Windows

The venv launcher printed `Failed to find real location of D:\Programs\anaconda3\python.exe` on orchestrator and report invocations. The venv process continued, training and collectors exited successfully, and all formal validators passed. This warning is recorded as an environment compatibility warning; it did not result in a failed run.

## 5. Run outcomes and retries

Each wave below executed both listed arms together. Exit codes are HT,DH. Each run’s formal validation passed; wave overlap validation passed; promotion was ready; retries: none.

| Dataset/stage | Seed | Wave | Exit codes | Max live | Positive overlap | Longest run (s) | Validation | Retry |
|---|---:|---|---|---:|---:|---:|---|---|
| KuaiRand / S6 | 2022 | `s6_HTDH_seed2022` | 0, 0 | 2 | 7892.674 s | 7892.8 | PASS/PASS | none |
| KuaiRand / S6 | 2023 | `s6_HTDH_seed2023` | 0, 0 | 2 | 6525.973 s | 6526.1 | PASS/PASS | none |
| KuaiRand / S6 | 2024 | `s6_HTDH_seed2024` | 0, 0 | 2 | 6513.368 s | 6514.1 | PASS/PASS | none |
| KuaiRand / S6 | 2025 | `s6_HTDH_seed2025` | 0, 0 | 2 | 6530.257 s | 6530.3 | PASS/PASS | none |
| MovieLens / S6ML | 2022 | `s6ml_HTDH_seed2022` | 0, 0 | 2 | 562.769 s | 562.8 | PASS/PASS | none |
| MovieLens / S6ML | 2023 | `s6ml_HTDH_seed2023` | 0, 0 | 2 | 560.996 s | 561.1 | PASS/PASS | none |
| MovieLens / S6ML | 2024 | `s6ml_HTDH_seed2024` | 0, 0 | 2 | 565.773 s | 565.8 | PASS/PASS | none |
| MovieLens / S6ML | 2025 | `s6ml_HTDH_seed2025` | 0, 0 | 2 | 565.231 s | 565.2 | PASS/PASS | none |

Across all waves: 16/16 new HT/DH runs promoted; 8/8 seed waves completed; 0 retries; 0 failed run validations; 0 missing Phase B runs; no runtime error prevented completion.

## 6. Validation results

- Both Phase B queue states list all four seeds as completed, with four attempts/waves each, exit codes `[0, 0]`, `retry_class=none`, `overlap_valid=true`, `ready_for_promotion=true`, and no validation failures.
- The orchestrators validated every Phase B HT/DH run and each within-phase seed block.
- Both Phase B collectors revalidated all Phase A F/T and Phase B HT/DH runs, cross-phase `PAIR_FIELDS`, `batch_order_hash`, and producer-critical source identity for every seed. Both manifests record `pairing_valid=true` and `cross_phase_pairing_audit` verified.
- S6 config-lock audit: `config_lock_passed=true`, 0 failures.
- Final collector status for S6 and S6ML: `STRUCTURALLY_VALIDATED`.

### M1 design-gate audit (separate from structural validation)

`STRUCTURALLY_VALIDATED` reports successful artifact, configuration, pairing, and run-structure validation. It does **not** mean every design identification gate passed. The focused post-run audit records the following M1 outcomes; see [stage_S6_phaseB_audit_review.md](stage_S6_phaseB_audit_review.md) for the independent audit record and exact margin values.

| Dataset | m conjunction | v conjunction | M1 status |
|---|---:|---:|---|
| KuaiRand / S6 | 4/8 PASS; 4/8 FAIL | 8/8 PASS; 0/8 FAIL | **FAIL** |
| MovieLens / S6ML | 8/8 PASS; 0/8 FAIL | 8/8 PASS; 0/8 FAIL | **PASS** |

KuaiRand per-seed/epoch flags (epochs 0 and 1):

| Seed | Epoch | m HT margin | m DH margin | m conjunction | v conjunction |
|---|---:|---:|---:|---|---|
| 2022 | 0 | -0.034231 | -0.015792 | FAIL | PASS |
| 2022 | 1 | +0.161342 | -0.125357 | FAIL | PASS |
| 2023 | 0 | +0.005663 | +0.008741 | PASS | PASS |
| 2023 | 1 | -0.157011 | +0.239659 | FAIL | PASS |
| 2024 | 0 | -0.013301 | +0.000869 | FAIL | PASS |
| 2024 | 1 | +0.009808 | +0.152827 | PASS | PASS |
| 2025 | 0 | +0.009947 | +0.010285 | PASS | PASS |
| 2025 | 1 | +0.073282 | +0.621579 | PASS | PASS |

The KuaiRand M1 failure is retained as an unmet design gate. It does not change the successful structural validation or run records above, and structural validation must not be reported as all-design/M1 compliance.

## 7. Raw matrix coverage and files

All 16 canonical runs per dataset (4 arms × 4 seeds) are represented: Phase A F/T products were reused; Phase B supplied the missing HT/DH products. Each raw results table includes 7 checkpoint labels × 6 domain/task cells × 16 runs = 672 rows, with per-cell AUC and LogLoss. Paired results contain all 6 arm contrasts across the available cells and checkpoints. No metric values were filtered or omitted by the collectors.

**KuaiRand / S6** — raw root `experiments/stage_S6_raw_results`; outputs in `experiments/stage_S6_outputs`:

- `stage_S6_runs.csv`: 16 rows
- `stage_S6_raw_results.csv`: 672 rows (per-cell AUC and LogLoss)
- `stage_S6_paired_results.csv`: 1008 rows
- `stage_S6_aggregates.csv`: 252 rows
- `stage_S6_gradient_steps.csv`: 40,000 rows
- `stage_S6_probe.csv`: 1,600 rows
- `stage_S6_moment_audit.csv`: 28 rows
- `stage_S6_config_lock_audit.csv`: 0 failures
- `reports/stage_S6_run_report.md` and `stage_S6_summary_manifest.json`

**MovieLens / S6ML** — raw root `experiments/stage_S6ML_raw_results`; outputs in `experiments/stage_S6ML_outputs`:

- `stage_S6ML_runs.csv`: 16 rows
- `stage_S6ML_raw_results.csv`: 672 rows (per-cell AUC and LogLoss)
- `stage_S6ML_paired_results.csv`: 1008 rows
- `stage_S6ML_aggregates.csv`: 252 rows
- `stage_S6ML_gradient_steps.csv`: 4,800 rows
- `stage_S6ML_probe.csv`: 1,600 rows
- `stage_S6ML_moment_audit.csv`: 28 rows
- `reports/stage_S6ML_run_report.md` and `stage_S6ML_summary_manifest.json`

Per-run stdout, process intervals, manifests, metrics, checkpoints, and audit artifacts are under the canonical run directories. Append-only attempt logs and concurrency receipts are under each raw root’s `.attempts` directory; Phase B queue state and process records are `stage_S6_queue_manifest_phaseB.json.state.json` and `stage_S6ML_queue_manifest_phaseB.json.state.json`. Queue manifests are `stage_S6_queue_manifest_phaseB.json` and `stage_S6ML_queue_manifest_phaseB.json`.

Existing Phase A output artifacts were preserved before collection overwrote the standard output filenames:

- `experiments/stage_S6_outputs_pre_phaseB_20261007`
- `experiments/stage_S6ML_outputs_pre_phaseB_20261007`

## 8. Runtime errors and missing runs

- Runtime errors: none.
- Retries: none.
- Missing runs: none.
- Non-blocking environment warning: broken base-interpreter redirector message described above; all invoked commands returned exit code 0.
