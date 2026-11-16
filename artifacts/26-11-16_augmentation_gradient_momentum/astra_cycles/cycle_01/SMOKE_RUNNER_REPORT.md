# EXPERIMENT EXECUTION REPORT — GPU SMOKE ONLY

- **Experiment ID:** `26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01`
- **Git/worktree:** detached `c4dc36bd38bd4f7a2a3319377a8289ba8185532d`; dirty; runner made no source edits.
- **Review/source:** PASS; `2ee2bdddb67861e65449cb3577c1d835f13395000a90b4ac101508c855f7b1a1`; design `f682ad676b8a2a84b21ebb3fd63aef41348ee49f45d838a9cbca2af2b10e9754`.
- **Config:** GPU 1 / RTX 3090, FP32, batch 40,960, one step per dataset and arm; seed 2022.
- **Seeds:** 2022 (smoke uses the fixed implementation seed; no formal seed matrix was run).
- **Run status:** PASS.

| Dataset | Arm | Rows | Seconds | Loss | Clean loss | View loss | Peak allocated bytes | Peak reserved bytes | Board power peak W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| movielens | AXIS-EVEN | 40960 | 0.913642 | 0.695795655251 | 0.695795357227 | 0.69579654932 | 4028229632 | 4133486592 | 125.210 |
| movielens | AXIS-IID2 | 40960 | 0.566016 | 0.695796370506 | 0.695795357227 | 0.695799469948 | 4066142208 | 4162846720 | 195.610 |
| movielens | ISOTROPIC-EVEN | 40960 | 0.578966 | 0.695795357227 | 0.695795357227 | 0.695795357227 | 4066142208 | 4162846720 | 195.610 |
| kuairand | AXIS-EVEN | 40960 | 0.667826 | 0.719164907932 | 0.719164907932 | 0.719164907932 | 4743774208 | 5012193280 | 162.150 |
| kuairand | AXIS-IID2 | 40960 | 0.693099 | 0.719164967537 | 0.719164907932 | 0.719164967537 | 4922294272 | 5192548352 | 206.310 |
| kuairand | ISOTROPIC-EVEN | 40960 | 0.652886 | 0.719164967537 | 0.719164907932 | 0.719164967537 | 4922294272 | 5192548352 | 205.360 |

**Full/tail source identities:**

- movielens: PASS; full=40960 rows, null/source exact=True/True; tail=21927 rows, null/source exact=True/True.
- kuairand: PASS; full=40960 rows, null/source exact=True/True; tail=40052 rows, null/source exact=True/True.

- **Runtime errors:** none. One PyTorch TF32 API deprecation warning; smoke status remained PASS.
- **Missing runs:** none for smoke. Formal matrix not run (outside task scope).
- **Complete log:** `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/GPU_SMOKE.log`
- **Raw receipt:** `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/SMOKE_REPORT.json`
- **Storage handoff:** `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/STORAGE_HANDOFF.json` (owner PID 1570727, tool session 59365, 0 states / 0 bytes).
- **Parent acceptance:** accepted; `PARENT_STORAGE_ACCEPTANCE.json` confirms PID 1570727 / session 59365, 0 retained states / 0 bytes.
- **Preservation:** old inventories match before/after smoke; historical guardian PID 1025015 observed alive; Test not accessed.
