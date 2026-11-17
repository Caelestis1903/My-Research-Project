# Cycle04 Attempt2 Formal Runner Report

**Status:** PAUSED (technical resource measurement failure).

## Execution record

- Native dispatch receipt: `RUNNER_DISPATCH_FORMAL.json` (PASS; gpt-6-luna/high).
- Controller PID: `2482923`; tool session: `43477`; exit code: `0`.
- Foreground wall time: `11688.9056` seconds.
- Source SHA-256: `d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f`.
- Design bundle SHA-256: `2acde7e9e6764f6243920cba3f538e2c4cd49b36b6ddfa2351addba3b9fc62b7`.
- Test access: false.

## Raw formal accounting

- Formal trajectories: 13/20 completed at 50 epochs; 7 incomplete.
- Acknowledged formal epochs: 771.
- Acknowledged useful Adam updates: 29021.
- Engineering smoke acknowledgments excluded from formal totals.

## Technical pause

- Paused trajectory key: `926a61e2c285299f6c3433b05b917384f93b0811dc98759de243c3dd85b9a33a`.
- Recorded at: 2026-11-17T06:03:08.405273+00:00.
- Trace: `protocol.TechnicalPause: NVML measurement failed code=3` at worker profile `E*_B*_after`; profile cleanup encountered the same measurement failure.
- Raw log: `runs/926a61e2c285299f6c3433b05b917384f93b0811dc98759de243c3dd85b9a33a/foreground.log`.
- Runner reported `PAUSED`; no retry or algorithm changes were made.

## Per trajectory

| Key | Dataset | Arm | Seed | ACK epochs | ACK useful Adam updates | State | Result path |
|---|---|---|---:|---:|---:|---|---|
| `053b80c9771678c1d81928302842a73f9f8f0f38b7279101eee286d56db0673b` | movielens | MASS-SCALE-025 | 2022 | 50 | 1000 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/053b80c9771678c1d81928302842a73f9f8f0f38b7279101eee286d56db0673b/result.json` |
| `09d5f15bdfa3940a13db15351620e4f300cb22f82109387755071e8d9a364f3d` | movielens | MASS-SCALE-025 | 2024 | 50 | 1000 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/09d5f15bdfa3940a13db15351620e4f300cb22f82109387755071e8d9a364f3d/result.json` |
| `1af0b0505e1d334a17681dff5dda2f310deec074c4fcb534ea05cf26bb773455` | movielens | COMPLETION-BCE-025 | 2022 | 50 | 1000 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/1af0b0505e1d334a17681dff5dda2f310deec074c4fcb534ea05cf26bb773455/result.json` |
| `200ab75fcc8adc4a6cd6ceff96a1ccc65096e7c4d1ebee0b6c53e2d5edabf1a5` | kuairand | COMPLETION-BCE-025 | 2022 | 50 | 2450 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/200ab75fcc8adc4a6cd6ceff96a1ccc65096e7c4d1ebee0b6c53e2d5edabf1a5/result.json` |
| `2df253901fbd915c33bb3f5a2c6152666931bbd1bc093b16f9aee1828dd1932c` | movielens | COMPLETION-BCE-025 | 2026 | 0 | 0 | not_started_or_no_ack | — |
| `36d45cdfe51e115f07ddd7dac653693b07f84ea50ee57b87224b076a45f5bc1d` | kuairand | COMPLETION-BCE-025 | 2024 | 50 | 2450 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/36d45cdfe51e115f07ddd7dac653693b07f84ea50ee57b87224b076a45f5bc1d/result.json` |
| `3a0ec7ec3099176ca1a2776a2a7d81a5ecd760e3666014f8680b750d39126795` | kuairand | MASS-SCALE-025 | 2023 | 50 | 2450 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/3a0ec7ec3099176ca1a2776a2a7d81a5ecd760e3666014f8680b750d39126795/result.json` |
| `40e2f2b566be11aace762c43aafec94887239533c871aa990e2dbe3a0a1503a3` | movielens | COMPLETION-BCE-025 | 2024 | 50 | 1000 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/40e2f2b566be11aace762c43aafec94887239533c871aa990e2dbe3a0a1503a3/result.json` |
| `5696c58a7c592fca412476a4bed827b3fd6f2c689096e8bdcd558d3dd3359cb2` | kuairand | COMPLETION-BCE-025 | 2023 | 50 | 2450 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/5696c58a7c592fca412476a4bed827b3fd6f2c689096e8bdcd558d3dd3359cb2/result.json` |
| `64ad311b996e93992d959e04c2b358126404faa5002e43aa7ae8f92b4be300e9` | movielens | MASS-SCALE-025 | 2023 | 50 | 1000 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/64ad311b996e93992d959e04c2b358126404faa5002e43aa7ae8f92b4be300e9/result.json` |
| `67510a0c232f831ab36e16984ee074646cb59e69be1b16d94cba3a4fbac9a011` | movielens | COMPLETION-BCE-025 | 2025 | 1 | 20 | paused_in_progress | — |
| `6cd8a644cf38d3c04562b27b47c861685d956866eb71b1e79787245802d5276f` | movielens | COMPLETION-BCE-025 | 2023 | 50 | 1000 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/6cd8a644cf38d3c04562b27b47c861685d956866eb71b1e79787245802d5276f/result.json` |
| `8a38a215216804b09d08558ef78402941bf9192600e22efa98a722448e3f54d9` | kuairand | COMPLETION-BCE-025 | 2025 | 50 | 2450 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/8a38a215216804b09d08558ef78402941bf9192600e22efa98a722448e3f54d9/result.json` |
| `926a61e2c285299f6c3433b05b917384f93b0811dc98759de243c3dd85b9a33a` | kuairand | COMPLETION-BCE-025 | 2026 | 40 | 1960 | paused_in_progress | — |
| `9b81123228e5056da60e3c51ad0ac52f97588cf53c76217831b8034eb7e5a2d8` | kuairand | MASS-SCALE-025 | 2026 | 39 | 1911 | paused_in_progress | — |
| `c6481fe6067fe4a819b8e1249a1ccf9ef3c6491b61e70042af7d1d85c76636a0` | movielens | MASS-SCALE-025 | 2025 | 1 | 20 | paused_in_progress | — |
| `dfe465e04b744abf6867b4888223611c767a56cf194cba73b62a530c33dc0615` | kuairand | MASS-SCALE-025 | 2024 | 50 | 2450 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/dfe465e04b744abf6867b4888223611c767a56cf194cba73b62a530c33dc0615/result.json` |
| `ea3f3584e6de6d4f328882f42055264e6306665c82c874d0245577eb1d29148d` | movielens | MASS-SCALE-025 | 2026 | 0 | 0 | not_started_or_no_ack | — |
| `eb5a63846ea11f9ae277c85be08707cdf6413ab6246890ea702cbd159431569f` | kuairand | MASS-SCALE-025 | 2022 | 50 | 2450 | complete_50_epochs | `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/runs/eb5a63846ea11f9ae277c85be08707cdf6413ab6246890ea702cbd159431569f/result.json` |
| `f8df5a89010729c008e138d2f74668f9698d391c6fe32f93041a3537493c6649` | kuairand | MASS-SCALE-025 | 2025 | 40 | 1960 | paused_in_progress | — |

This is a raw execution report only. Runner handoff and table export were not run because the formal run did not complete.
