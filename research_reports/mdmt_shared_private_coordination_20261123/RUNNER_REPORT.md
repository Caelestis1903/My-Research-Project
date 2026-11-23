# Experiment execution report

## 1. Experiment ID

`mdmt_shared_private_coordination_20261123` (D2), S0 engineering smoke and S1 execution.

## 2. Git and worktree

Branch `26-11-23_v2` at `f66bf17cba2980618e6f2edd1c7381b73c35cac5`. The worktree included untracked study implementation and artifacts. The independent review and runtime gate bound implementation SHA-256 `f45b68ef687179f756f1397314b47d46359fbd8ecbdd9312ef6da3704d819202` and S1 source identity `c21201f21edb338dd7640c2cd1262988c0beded1559f5cf8a8eacbf61ce01213`. [Implementation review](IMPLEMENTATION_REVIEW.json) · [Resource audit](../../artifacts/mdmt_shared_private_coordination_20261123/S0/RESOURCE_AUDIT.json)

## 3. Configuration

Project Python `phase0-3-env`, PyTorch `2.9.0+cu126`, CUDA 12.6, and two NVIDIA RTX 3090s. S0 ran on physical GPU 0. S1 used both physical GPUs with independent measured admission.

S1 ran canonical fixed-fusion M3oE `J_ORIG` with native Adam and natural BCE on the original split for 50 epochs. Both datasets used a batch size of 40,960 and a learning rate of 0.01. Each run retained end-E10, E25 and E40 anchors with 19 named forks per anchor. Online and fixed-EMA selections each used the earliest strict maximum of full Validation Macro AUC. [Registered config, data and seed identities](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN_MANIFEST.json) · [Fork count erratum](DESIGN_ERRATUM.md)

| Dataset | Weight decay | Fixed EMA ρ | Batches per epoch | Natural tail rows |
| --- | ---: | ---: | ---: | ---: |
| MovieLens | 0.0001 | 0.9828205985452511 | 20 | 21,927 |
| KuaiRand | 0.00001 | 0.995 | 49 | 40,052 |

## 4. Seeds

S0 used 2022 for both datasets. S1 used 2022, 2023, 2024, 2025 and 2026 for each dataset.

## 5. Run status and costs

S0 **PASS**. Both datasets matched the donor exactly for first and natural-tail probabilities, online state, native Adam state and EMA state. Both authenticated RAM get/hash receipts and exact next-step replays passed with the accepted owner. Full CUDA process profiles were 1,388,314,624 bytes for MovieLens and 1,675,624,448 bytes for KuaiRand. [GPU smoke](../../artifacts/mdmt_shared_private_coordination_20261123/S0/GPU_SMOKE.json) · [Independent S0 acceptance](S0_SMOKE_REVIEW.md)

S1 **COMPLETE** on 2026-11-23. The interval from first admission to final publication was 1,807.5 seconds (13:41:32–14:11:39 UTC). All ten first attempts exited with code 0. The supervisor verified 500 baseline epochs, 30 anchors, 570 forks and ten E50 checkpoint packets with both selected states. The 30 anchor packets and ten final checkpoint packets were still present in the same live RAM owner at report verification. The owner ledger recorded 562 commits, including both released S0 smoke fixtures and 30 released S1 fork clones. Test evaluations totaled zero. [Stage status](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN.json) · [RAM anchor index](../../artifacts/mdmt_shared_private_coordination_20261123/ANCHOR_INDEX.json)

Five admissions ran on GPU 0 and five on GPU 1. Each admission in the ledger met the per-card current 85% memory and power thresholds and the host 90% RAM threshold. The scheduler recorded full-process residency profiles of 1,413,480,448 bytes for MovieLens and 1,847,590,912 bytes for KuaiRand. [Admissions and exits](../../artifacts/mdmt_shared_private_coordination_20261123/S1_ADMISSIONS.jsonl) · [CUDA capacity record](../../artifacts/mdmt_shared_private_coordination_20261123/S1_PHYSICAL_CAPACITY.json)

| Recorded cost | Baseline attempts | Accepted diagnostic forks |
| --- | ---: | ---: |
| Training forwards | 17,250 | 1,140 |
| Parameter backward calls | 17,250 | 1,140 |
| Native Adam steps | 17,250 | 990 |
| EMA ticks | 17,250 | 1,140 |
| Evaluation forwards | 5,000 | 6,900 |
| Worker wall time summed across attempts, s | 11,837.6 | |
| Baseline CUDA event intervals summed across attempts, s | 6,931.5 | |
| Fork phase time summed across records, s | | 2,657.1 |

Baseline worker wall time includes in-worker diagnostics. The cost table contains ten baseline attempt rows and 2,520 accepted diagnostic phase rows. [Full cost ledger](../../artifacts/mdmt_shared_private_coordination_20261123/COSTS.csv)

## 6. Raw metrics

The selected endpoints below report full Validation Macro AUC and Macro LogLoss. E50 training loss is the recorded natural BCE over E50. The baseline CSV contains both readouts for all 500 epochs, including per-domain, per-task, six-cell, traffic-weighted and pooled metrics. Each anchor's `forks_E*.json` contains full Validation, V_A and V_B endpoint records, six-cell supports, pair transitions, cell responses, update counters and state hashes. [Baseline endpoints](../../artifacts/mdmt_shared_private_coordination_20261123/BASELINE_ENDPOINTS.csv) · [Raw runs and diagnostic receipts](../../artifacts/mdmt_shared_private_coordination_20261123/runs/)

| Dataset | Seed | GPU | Worker s | E50 train BCE | Online epoch | Online AUC | Online LogLoss | EMA epoch | EMA AUC | EMA LogLoss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MovieLens | 2022 | 0 | 792.2 | 0.417756 | 29 | 0.756911 | 0.531181 | 26 | 0.760026 | 0.526644 |
| MovieLens | 2023 | 0 | 784.1 | 0.419484 | 17 | 0.756404 | 0.524254 | 18 | 0.759714 | 0.523377 |
| MovieLens | 2024 | 1 | 745.4 | 0.414568 | 24 | 0.756061 | 0.526846 | 20 | 0.759673 | 0.525143 |
| MovieLens | 2025 | 0 | 734.6 | 0.419440 | 26 | 0.756669 | 0.525911 | 28 | 0.759084 | 0.527043 |
| MovieLens | 2026 | 1 | 708.0 | 0.418282 | 17 | 0.757282 | 0.524295 | 22 | 0.759060 | 0.527621 |
| KuaiRand | 2022 | 1 | 1,689.6 | 0.486320 | 44 | 0.688511 | 0.431608 | 33 | 0.692638 | 0.430601 |
| KuaiRand | 2023 | 0 | 1,613.1 | 0.488531 | 18 | 0.676205 | 0.435294 | 50 | 0.677979 | 0.432980 |
| KuaiRand | 2024 | 1 | 1,671.9 | 0.488908 | 40 | 0.693477 | 0.426897 | 50 | 0.695994 | 0.426760 |
| KuaiRand | 2025 | 0 | 1,515.0 | 0.489939 | 50 | 0.685375 | 0.431793 | 50 | 0.686161 | 0.429821 |
| KuaiRand | 2026 | 1 | 1,583.8 | 0.490831 | 43 | 0.687823 | 0.427230 | 47 | 0.689500 | 0.428142 |

## 7. Logs and live storage identity

The S1 supervisor log and all worker logs are retained. Each logical run's `attempts/<attempt_id>/` contains `worker.log`, `ALLOCATION.json`, the epoch RAM receipts and the physical fork journal. The attempt outcome is in the adjacent `attempts/<attempt_id>.json`. [S1 supervisor log](../../artifacts/mdmt_shared_private_coordination_20261123/S1-run-20261123T134129.log) · [Worker directories](../../artifacts/mdmt_shared_private_coordination_20261123/runs/)

The accepted owner stayed live in the foreground runner's tool terminal. Its PID was `191092` and its generation was `b027c4c8cc666b9221efe49505ef24f6bb5f8bfc550f949ff9a2839597e93a84`. [Accepted S0 log](../../artifacts/mdmt_shared_private_coordination_20261123/S0/smoke-pty-20261123T133556.log) · [RAM commit ledger](../../artifacts/mdmt_shared_private_coordination_20261123/ram_pty_20261123T133555/epoch_commit_ledger.jsonl)

## 8. Runtime errors

The first S0 command exited successfully after the numerical smoke passed. Its owner became unreachable when that command session closed, and the post-exit live check returned `ConnectionRefusedError [Errno 111]`. The runner repeated the reviewed smoke with a fresh D2 owner in an interactive terminal kept open across S0 and S1. The retry passed the independent live-owner gate. S1 recorded zero failed attempts, retries or CUDA OOMs. PyTorch emitted the frozen donor TF32 setter deprecation warning. [Archived first smoke and live-owner failure](../../artifacts/mdmt_shared_private_coordination_20261123/S0/attempts/smoke-20261123T133331/)

## 9. Missing runs

None in S0/S1. The S1 manifest and completion receipt contain all ten registered logical runs. The baseline CSV contains 1,000 readouts and 20 selected endpoints. The runner cross-checked all 500 epoch and 30 fork file hashes against the corresponding run receipts. [S1 manifest](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN_MANIFEST.json) · [S1 completion receipt](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN.json)
