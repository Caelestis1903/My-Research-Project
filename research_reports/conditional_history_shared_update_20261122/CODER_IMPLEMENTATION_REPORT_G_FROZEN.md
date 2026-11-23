# G_FROZEN control identity audit

**TECHNICAL_BLOCKED.** The 11/22 plan registered ten A21 + Gaussian trajectories contingent on reusable controls with the same Gaussian background. The two official five-seed D0 controls use a different random draw on the first actual batch for every seed. The available row-keyed D0 controls cover two MovieLens seeds and zero KuaiRand seeds. No G_FROZEN training or Test evaluation was dispatched.

| Dataset | Exact row-keyed D0 selected controls | Missing seeds | Official D0 control |
|---|---|---|---|
| MovieLens | 2022, 2024 | 2023, 2025, 2026 | GAUSSIAN_A1, alias G-D0, five seeds with GA31-R3 noise |
| KuaiRand | None | 2022–2026 | G-D0, five seeds with GA31-R3 noise. Row-keyed EMA02 A1 used D2 |

## Engineering evidence

The audit read original training traces and frozen Validation identities. The official rows retain the upstream run ID, selected epoch, model/config/source hash, selection/endpoint ID, role and reuse flags. It matched both population identities to S0, checked the ten dataset/seed identities, checked all five MovieLens selections against the pinned source audit and recomputed the five KuaiRand selections from 50-epoch Validation curves. Both controls use a single D0 sparse-embedding view, a 0.5 replacement fraction, the registered noise scale and fixed EMA.

For each original seed the audit reconstructed the first 40,960 indices with the training loader, verified the archived original batch hash and D0 support, then evaluated the prescribed EMA02 SplitMix64/Box–Muller normal on those original D0 row IDs. It compared the resulting FP32 SHA-256 with the *actual archived GPU draw* recorded in the historical run. The GA31-R3 source used a step-keyed device-local torch.Generator and drew one matrix per field. The two generators yielded different hashes on all ten fixtures. For seed 2022 the MovieLens batch had 7,957 D0 rows, with hash prefixes 1ad918736bdd2ea6 (historical) and 0663dfdae605713e (registered). KuaiRand had 3,474 D0 rows and prefixes 249f53a2b35cd7ca and 84d669140754a037.

The EMA02, information-momentum v3 and history-information A1 endpoint tables contain the same (epoch, model hash) for the same two development seeds on each dataset. They supply two usable MovieLens seed identities. KuaiRand's target_domain in this lineage was 2. The five-seed completion Gaussian alias points to the original AIG-M source records. The later 11/06 Gaussian D0 replay also calls the GA31 loss with implicit views. The 11/10 and 11/11 trajectory adapters import this source. These paths cannot supply additional EMA02-row-keyed D0 controls.

## Handoff

The trigger remains TRIGGERED, and the supplement is technically paused under the original runbook S4 and resources §2. An exact matched table would require acquiring archived controls for three MovieLens seeds and five KuaiRand seeds or an explicitly revised control matrix. The ten candidate-only jobs cannot supply those eight controls. S5 may document this technical gap alongside the independently resolved A/B results.

| Artifact | SHA-256 |
|---|---|
| experiments/conditional_history_shared_update_20261122/gaussian_control_audit.py | bfcfac3d739987a2751b41ba96090343235ef0ad3d13780c714dee5aff6494ef |
| artifacts/conditional_history_shared_update_20261122/SOURCE_G_CONTROL_AUDIT.json | 10a8b2ed10b944ce21292526cc9a0920c3dfcc850acc2199d17ccaa68ed002c3 |
| artifacts/conditional_history_shared_update_20261122/G_FROZEN_TRIGGER.json | dd2c2d1589113ddf2f1bec9f757431659586d4962a501e1dbaff27bc6dc84179 |

Reproduce from the project root with `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.conditional_history_shared_update_20261122.gaussian_control_audit`. The receipt lists 29 immutable Git path/checksum pairs, both source archive identities, all ten compared draw fingerprints and the alternative selected endpoints. The audit required CPU only.
