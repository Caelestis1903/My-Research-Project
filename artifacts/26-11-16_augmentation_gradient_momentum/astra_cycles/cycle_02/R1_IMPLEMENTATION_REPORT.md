# R1 implementation report — Cycle02

Status: **R1 repaired; focused CPU checks PASS; fresh distinct REVIEW pending.**

Actual R1 coder: `01a1e41b-a08d-7180-9788-cc62bee1e57e` (`CODEX_THREAD_ID`). Applied ponytail FULL and stop-that-shit CHANGE. File handoffs only; no descendants.

Read current W/O AGENTS, O/start.md decoded GB18030, W/buchong.md, execution runbook, all seven accepted Cycle02 design files, both original coder reports, and independent IMPLEMENTATION_REVIEW.md/json including actual Q2/Q3 evidence. The prior FAIL and its source-candidate receipt are preserved.

The source change is exactly two one-line replacements:

- [runner.py:257](/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/runner.py:257): load `cpu_copy(state['optimizer'])` through the existing helper. Every restore now receives fresh Adam step/moment tensors. This fixes the shared restore used by repeated smoke comparisons and continuation recovery.
- [protocol.py:293](/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/protocol.py:293): `len(set(coder_ids)) >= 2`; reviewer exclusion still covers every contributing coder. Verdict, blockers, CPU, source and design checks remain intact. There is no contributor/revision cap.

The focused fixture uses the reviewer's same small CPU Scores model, natural synthetic four-row labels and source Adam. After one useful ordinary update populates step/exp_avg/exp_avg_sq, the real repaired `smoke_identities` passes diagnostic-on/off, lambda-zero/source and all-single-class/source comparisons. A wraps-only spy records all seven actual restores without modifying arguments or behavior. The entire saved baseline hashes to `2ad631b88978821e8cb642d2ba7ad2f455c7f896f733265e85df79beb2fb1a87` before and after; the full final state matches that hash, Adam restores step **1 -> 1**, and live step/both moment tensors have distinct storage from the saved baseline.

The separate temporary in-memory review/hash fixture accepts all three distinct coders and rejects a reviewer equal to each one. It writes no review receipt. Both changed Python files pass in-memory syntax compilation. Exact executed focused commands/results are in [R1_CPU_CHECKS.json](/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/R1_CPU_CHECKS.json).

| Identity | SHA256 / result |
| --- | --- |
| Actual repaired 87-file source | `ddaf5d97b676c4c8ba06c40cb2a7aa2209ae27148950b848329e1dc8bb669bae` |
| Prior FAIL 87-file source, preserved receipt | `d3f518b8d23cbafa9be0b954de995d0edfeaf12ebab3836cd4a723d3eda9d0a0` |
| Accepted 7-file design, unchanged | `9e5265df52809e45ac186d5c5a4a2b80ac96e4c952ded35eb92f76c1efcccfb4` |
| Protected C1 79-file source, unchanged before/after | `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1` |
| Existing 11 cycle files | Before/after hashes match, including both original reports, FAIL review and old source candidate |

The fresh reviewer must list these contributing coder IDs and be distinct from all of them:

- Original core: `01a1e3f5-7f1c-7ad2-9512-5c86d72c033e`.
- Original runtime: `01a1e3f5-7f2c-7262-945c-10b1626cfb40`.
- R1: `01a1e41b-a08d-7180-9788-cc62bee1e57e`.

Writes were confined to runner.py, protocol.py, this report and R1_CPU_CHECKS.json. Both workspace stop controls were clear at the before/after checks. Algorithm, source Adam, precision, protocol budget and frozen dependencies are unchanged. No prepare/source-receipt/manifest operation, owner contact, GPU query, IPC, research run, Test, git or publication occurred; no state/tensor files were written.

The unchanged five core groups and 13 admission fixtures were not rerun. Parent closes this coder permanently, then obtains fresh distinct review of the actual new source before engineering GPU smoke; the prescribed six-case smoke and separate resource signoff remain subsequent gates.

