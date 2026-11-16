# Cycle02 engineering smoke — runner report

- **Status:** PASS (6/6 cases); no missing or failed runs.
- **Experiment/source:** `Cycle02 natural quadratic ranking engineering smoke`; source `ddaf5d97b676c4c8ba06c40cb2a7aa2209ae27148950b848329e1dc8bb669bae`.
- **Design bundle:** `9e5265df52809e45ac186d5c5a4a2b80ac96e4c952ded35eb92f76c1efcccfb4`; review PASS, zero blockers.
- **Git:** detached HEAD `556b4ae354fa1465c0bd33f1744eec8d93c59dd2` in `/home/xzf/Code/phase0-3_tvg_rm_20261116`; existing tracked root-state modification preserved.
- **Configuration:** three approved arms × two datasets; seed 2022; lambda 0.25; FP32; 40,960 full rows plus natural tails (MovieLens 21,927; KuaiRand 40,052).
- **Device:** all six cases ran on GPU 1 (RTX 3090); GPU 0 was skipped whenever current power exceeded 85%. Each case used current memory/power and host RAM readings, full-process physical-fit admission, and a post-allocation resample.
- **Validation:** natural online and fixed-EMA validation values are transcribed per case below; no cross-case summaries.

| Dataset | Arm | Seed | Online macro AUC | Online macro log loss | EMA macro AUC | EMA macro log loss | Device | Full/tail rows | Recovery |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| movielens | NQR-025 | 2022 | 0.578131139 | 0.624481564 | 0.577520301 | 0.643751249 | 1 | 40960/21927 | True |
| movielens | BRIER-OPPORTUNITY-025 | 2022 | 0.566266954 | 0.627426806 | 0.579574543 | 0.644187750 | 1 | 40960/21927 | True |
| movielens | MEAN-GAP-025 | 2022 | 0.577950123 | 0.624466988 | 0.577560615 | 0.643752807 | 1 | 40960/21927 | True |
| kuairand | NQR-025 | 2022 | 0.518943353 | 0.487695445 | 0.520044460 | 0.675051888 | 1 | 40960/40052 | True |
| kuairand | BRIER-OPPORTUNITY-025 | 2022 | 0.518306108 | 0.492902987 | 0.523219676 | 0.675675207 | 1 | 40960/40052 | True |
| kuairand | MEAN-GAP-025 | 2022 | 0.518946305 | 0.487695399 | 0.520040023 | 0.675051886 | 1 | 40960/40052 | True |

## Raw metric and log files

Complete per-cell, per-domain, per-task and aggregate raw metrics: [SMOKE_RUNNER_REPORT.json](/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/SMOKE_RUNNER_REPORT.json).

Admission samples: [smoke_admission.jsonl](/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/attempt1/smoke_admission.jsonl). Worker stdout/stderr logs and checksummed case/epoch records are listed in the JSON report.

## Run accounting

- RAM owner PID `1944719` and its original foreground session `52263` remain parent-owned and live.
- No runtime errors. Worker logs contain a PyTorch TF32 backend deprecation warning.
- No model checkpoint/tensor files were written to disk; snapshots used the existing RAM/BytesIO path.
- Formal 30-job training did not start; separate smoke/resource review remains pending.
