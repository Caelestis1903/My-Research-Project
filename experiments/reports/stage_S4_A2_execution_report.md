# Stage S4 Amendment A2 — Runner Execution Report (raw facts only)

## Experiment ID
Stage S4, Amendment A2 (`experiments/stage_S4_design.md` Amendment A2 + original spec).

## A2 queue SHA-256 and frozen source hashes

- A2 queue manifest: `experiments/stage_S4_queue_manifest_A2.json`
  - SHA-256: `c7353a2e7186e8102bc1fd9a1a3d60332526110585259cea161275ccbd696202`
- A2 queue state: `experiments/stage_S4_queue_manifest_A2.json.state.json`
- Queue properties at freeze (recomputed):
  - `tasks`: 16
  - `waves`: 8 (ordered `FT×2022..2025` then `NU×2022..2025`)
  - `source_inventory`: 21 entries including M3OE forward `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1`
  - `source_inventory_digest`: `8dc07f3df68fb3f8186c1c2585a736f6302edfc4a53e301d3dde04c3149f7e07`
  - `aborted_predecessor.queue_sha256`: `c6a8f1322213bd001f6994d42ff009e460e869e6d0624903de08e12088600d66`
  - `aborted_predecessor.state_sha256`: `540aeb37729a749585dd018adb3e242caf05c83a5d88dfb647a51675a5f2d554`
  - `aborted_predecessor.attempt_inventory_sha256`: `3bb4e8859533c597790215f433876f89a40a88714cf23503731478c05d670c6a` (56 files)
  - `amendment`: `"A2"`
  - `max_attempts_per_wave`: 2; `max_training_processes`: 2

| Source file | SHA-256 |
| --- | --- |
| `experiments/stage_S4_protocol.py` | `a40d62059410bbdf4054fb80cb786baf0e13ad9b6bf6f5bc6560f2474ddfffb2` |
| `experiments/stage_S4_gradient_controls.py` | `e695bbfafafb9b238d5cb691fde5113bc7e87f2b7699e766fb84d71167090645` |
| `experiments/stage_S4_run.py` | `99b171cba6a64e3adb4dab141bb9fb8f7c0ed091248c459ca586ce04914ca107` |
| `experiments/stage_S4_orchestrator.py` | `e44b44551c0dfc6136f13070227b885f41b8c0218cf3c3a8349837d9261c9801` |
| `experiments/stage_S4_report.py` | `97e1f97f600d9f5c99a7fbee7fe42b8b7157a22090b9f122172606e0364d9695` |
| `experiments/stage_S3_run.py` | `8db00e535f56160ae7888010bf917d275b1a88863605a918d0b24c4a9bf25e8d` |
| `m3oe/MDMTRec/models/multi_domain/model.py` (forward) | `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1` |

## Commands used

```
'/c/Users/86199/.conda/envs/EDDAenv/python.exe' experiments/stage_S4_orchestrator.py --a2 --reviewer-pass --freeze-only
'/c/Users/86199/.conda/envs/EDDAenv/python.exe' experiments/stage_S4_orchestrator.py --a2 --reviewer-pass   (background, run_in_background=true, disable_timeout=true)
'/c/Users/86199/.conda/envs/EDDAenv/python.exe' experiments/stage_S4_report.py
```

All commands executed from `D:\CODE\phase0-3`. Pre-flight used the same Python interpreter to compute SHA-256 hashes; no other training process was launched.

## Per-wave timing, launch PIDs, observed max-live process count, overlap

Orchestrator background bash task `bash-i2963cu4`, parent PID 9436, exit_code 0.

| Wave | Attempt | Start (local) | End (local) | Wall time (s) | F/T or N/U PIDs | `max_live_observed` | Overlap (s) | `overlap_valid` | Exit codes | `retry_class` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s4_FT_seed2022 | 1 | 2026-09-26 13:42:37 | 2026-09-26 16:03:26 | 8449 | F=2628, T=2244 | 2 | 8440.53 | true | [0, 0] | none |
| s4_FT_seed2023 | 1 | 2026-09-26 16:03:37 | 2026-09-26 17:46:29 | 6172 | F=18500, T=5676 | 2 | 6168.97 | true | [0, 0] | none |
| s4_FT_seed2024 | 1 | 2026-09-26 17:46:40 | 2026-09-26 20:10:56 | 8656 | F=30840, T=34136 | 2 | 8653.16 | true | [0, 0] | none |
| s4_FT_seed2025 | 1 | 2026-09-26 20:11:19 | 2026-09-26 23:02:49 | 10290 | F=10200, T=2460 | 2 | 10286.73 | true | [0, 0] | none |
| s4_NU_seed2022 | 1 | 2026-09-26 23:03:12 | 2026-09-27 00:51:55 | 6524 | N=29376, U=40504 | 2 | 6521.06 | true | [0, 0] | none |
| s4_NU_seed2023 | 1 | 2026-09-27 00:52:06 | 2026-09-27 02:38:02 | 6356 | N=13912, U=20464 | 2 | 6353.77 | true | [0, 0] | none |
| s4_NU_seed2024 | 1 | 2026-09-27 02:38:13 | 2026-09-27 04:24:46 | 6393 | N=20380, U=17432 | 2 | 6390.83 | true | [0, 0] | none |
| s4_NU_seed2025 | 1 | 2026-09-27 04:24:57 | 2026-09-27 06:11:28 | 6392 | N=42792, U=23568 | 2 | 6389.68 | true | [0, 0] | none |

Total orchestrator wall time (freeze-to-final-exit): 16:28:53 (16 h 28 min 53 s) — freeze at 2026-09-26 13:42:25, orchestrator exit at 2026-09-27 06:11:28 (first process start 2026-09-26 13:42:37; last process end 2026-09-27 06:11:28 → 16:28:51 end-to-end training span).

Observed GPU utilisation: two python training processes on `cuda:0` for the duration of every wave; peak resident memory per process ~6.0–7.8 GB (the 7.8 GB peak was observed during wave `s4_FT_seed2025` and is consistent with the in-flight step plus Adam state memory). No other python training process observed on `cuda:0` during the run.

Process-interval overlap fact (per state file): every wave records `overlap_valid: true` and `max_live_observed: 2`, confirming exactly two concurrent training processes per wave as required.

## Per-run status table (16 keys)

| Formal run key | Seed | Arm | Status | Validation | Exit code | Run dir (canonical) |
| --- | --- | --- | --- | --- | --- | --- |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2022\|F | 2022 | F | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\F\seed2022\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2022\|T | 2022 | T | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\T\seed2022\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2022\|N | 2022 | N | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\N\seed2022\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2022\|U | 2022 | U | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\U\seed2022\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2023\|F | 2023 | F | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\F\seed2023\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2023\|T | 2023 | T | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\T\seed2023\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2023\|N | 2023 | N | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\N\seed2023\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2023\|U | 2023 | U | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\U\seed2023\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2024\|F | 2024 | F | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\F\seed2024\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2024\|T | 2024 | T | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\T\seed2024\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2024\|N | 2024 | N | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\N\seed2024\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2024\|U | 2024 | U | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\U\seed2024\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2025\|F | 2025 | F | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\F\seed2025\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2025\|T | 2025 | T | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\T\seed2025\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2025\|N | 2025 | N | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\N\seed2025\joint` |
| s4\|equal\|kuairand\|natural\|m3oe\|seed2025\|U | 2025 | U | SUCCESS | valid (no reasons) | 0 | `experiments\stage_S4_A2_raw_results\U\seed2025\joint` |

## Canonical counts and block counts

- Canonical runs present under `experiments\stage_S4_A2_raw_results\`: 16/16 (F, T, N, U × {2022, 2023, 2024, 2025}).
- Strict four-arm blocks (seeds with all four arms present and validated): 4/4 (one per seed).
- Promotion directories: 8 (`experiments\stage_S4_A2_raw_results\.promotions\s4_{FT,NU}_seed{2022..2025}\<id>\F|T|N|U`).
- `.attempts` tree: 8 waves × 1 attempt = 8 attempt dirs; `attempt_inventory_sha256` not separately recomputed by the runner for the A2 attempts tree (orchestrator-owned).

## Retry / failure records

- `retry_class`: `none` for every wave; `retry_reason`: "successful processes; validation failures are not transient" for every wave.
- `terminated_peer_pids`: `[]` for every wave.
- No crash or transient fault observed; no failure.json written by the orchestrator.

## Collector outputs (under `experiments/stage_S4_A2_outputs/`)

| File | Rows (data) | Bytes | SHA-256 |
| --- | --- | --- | --- |
| `stage_S4_runs.csv` | 16 | 3332 | `926256e815b7e9089c9bade7f2e4abbf0c5f1c31723e5307efe6b5797b04c3d0` |
| `stage_S4_raw_results.csv` | 672 | 321830 | `85c97beb9bdaa30ac3b586d0d36d01d7b9318da09e067a46a709ce5a282f75e0` |
| `stage_S4_paired_results.csv` | 1008 | 242689 | `549d57b1ed6b68c134fbf7569448b6119c9b65807c56d8e06a364b3ffc3be6bc` |
| `stage_S4_gradient_steps.csv` | 40000 | 23966491 | `7b51de01f7b5adc40a58ac63e0bd8e924ebb5f6130e984755ec88dd1be101043` |
| `stage_S4_probe.csv` | 1600 | 1118667 | `7c6cf918329d9f98290882111272fe9abbf4ddf731d858c31ff20f83add80dac` |
| `stage_S4_aggregates.csv` | 252 | 33206 | `b06005f25aeb2b507014b3a933a95e36f8c2abb179fb509655073644db5f7b4a` |
| `stage_S4_summary_manifest.json` | (n/a) | 1348 | `393557f5df4f2d28bb1da13a25c8e8e4a38cf47c6ada15b4455c47a4ef07c6ab` |

`stage_S4_summary_manifest.json` reports:
- `amendment`: `"A2"`
- `stage`: `"S4"`
- `queue_sha256`: `c7353a2e7186e8102bc1fd9a1a3d60332526110585259cea161275ccbd696202`
- `aborted_predecessor_sha256`: `c6a8f1322213bd001f6994d42ff009e460e869e6d0624903de08e12088600d66`
- `four_arm_blocks`: 4
- `pairing_valid`: true
- `cardinalities`: `{aggregates: 252, gradient_steps: 40000, paired_results: 1008, probe: 1600, raw_results: 672, runs: 16}`
- `status`: `STRUCTURALLY_VALIDATED`
- `aborted_attempts_excluded`: true

Collector-written run report:
- `experiments/stage_S4_A2_outputs/reports/stage_S4_A2_run_report.md` (no SHA computed by runner; file written by collector).

## Locations of all artifacts

- Queue: `experiments/stage_S4_queue_manifest_A2.json`
- State: `experiments/stage_S4_queue_manifest_A2.json.state.json`
- Attempts tree: `experiments/stage_S4_A2_raw_results/.attempts/s4_{FT,NU}_seed{2022..2025}/attempt1/`
- Promotion tree: `experiments/stage_S4_A2_raw_results/.promotions/s4_{FT,NU}_seed{2022..2025}/<id>/`
- Canonical runs: `experiments/stage_S4_A2_raw_results/{F,T,N,U}/seed{2022..2025}/joint/`
- Collector outputs: `experiments/stage_S4_A2_outputs/` (CSVs + summary_manifest + `reports/stage_S4_A2_run_report.md`)
- Runner execution report: this file at `experiments/reports/stage_S4_A2_execution_report.md`

End of runner execution report.