# Cycle04 Engineering Runner Report

Outcome: `ENGINEERING_PENDING_INDEPENDENT_REVIEW`  

This is a factual mechanical RUN report. It contains no scientific interpretation.

- Foreground runner PID 2430254, tool session 71379, exit code 0.
- Totals: 32 returned train calls; 8 diagnostic points; 24 extra autograd sweeps; 32 virtual Adam steps; 0 scientific epochs.
- All four case receipts report `ENGINEERING_VERIFIED_PENDING_REVIEW`; all exactness and immutable-prefix flags are true; Test remained sealed.
- Case workers exited and were observed exited. The parent RAM owner PID 2429109 remains active. All four temporary keys remain retained; none were released.

## Cases

| Case | GPU | Dataset / arm | Seed | Returned calls | Worker PID(s) | Case / profile / proof |
|---|---:|---|---:|---:|---|---|
| `case1_gpu0_movielens` | 0 | movielens / COMPLETION-BCE-025 | 2022 | 8 | 2430339 | [`case.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case1_gpu0_movielens/attempt1/case.json), [`case_profile.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case1_gpu0_movielens/attempt1/case_profile.json), [`verified_plan.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case1_gpu0_movielens/attempt1/verified_plan.json) |
| `case2_gpu0_kuairand` | 0 | kuairand / MASS-SCALE-025 | 2022 | 8 | 2430558 | [`case.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case2_gpu0_kuairand/attempt1/case.json), [`case_profile.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case2_gpu0_kuairand/attempt1/case_profile.json), [`verified_plan.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case2_gpu0_kuairand/attempt1/verified_plan.json) |
| `case3_gpu1_movielens` | 1 | movielens / MASS-SCALE-025 | 2022 | 8 | 2430340 | [`case.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case3_gpu1_movielens/attempt1/case.json), [`case_profile.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case3_gpu1_movielens/attempt1/case_profile.json), [`verified_plan.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case3_gpu1_movielens/attempt1/verified_plan.json) |
| `case4_gpu1_kuairand` | 1 | kuairand / COMPLETION-BCE-025 | 2022 | 8 | 2430591 | [`case.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case4_gpu1_kuairand/attempt1/case.json), [`case_profile.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case4_gpu1_kuairand/attempt1/case_profile.json), [`verified_plan.json`](artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/smoke/case4_gpu1_kuairand/attempt1/verified_plan.json) |

## Raw records

- Physical attempts: [`artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/physical_attempts.jsonl`] (20 rows).
- Costs: [`artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/costs.jsonl`] (52 rows).
- Resource observations: [`artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/resource_observations.jsonl`].
- Commit ledger: [`artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/epoch_commit_ledger.jsonl`].

Parent supervision acceptance is recorded in `PARENT_ENGINEERING_SUPERVISION_ACCEPTANCE.json`: direct get/hash/record validation accepted all four retained states (logical step 4, storage ordinal 2, counts 4/4; exact ACK continuation and scientific identities). All four keys remain unreleased. Independent resource review remains pending; no signoff was written.
