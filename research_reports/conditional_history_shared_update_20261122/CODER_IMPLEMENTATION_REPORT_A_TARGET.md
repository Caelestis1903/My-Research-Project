# A_TARGET implementation handoff

Stage is IMPLEMENT. The approved 11-22 design and direction A §9 specify a read-only actual-trajectory probe. The frozen A_HISTORY decision selected static A21 on MovieLens. Both datasets use the same history form. The two measured intensities use A21 and A40_HISTORY_L025. The runner processes ten accepted parent trajectories per intensity and four retained pre-update packets per trajectory. This consumes zero full training replays and adds zero scientific seeds.

## Mechanism and evidence

The implementation is in `a_localization.py`, `a_localization_stage.py`, and `a_localization_checks.py` under `experiments/conditional_history_shared_update_20261122`.

- Every parent job is matched to its reviewed source manifest and complete result. On the runner, each packet must match the retained guardian generation, recorded packet SHA, parent job, group and data identities, real batch and row IDs, permutation cursor, global step, and regenerated static PAIR sample. Each actual step is applied on a private model and optimizer from the recorded post-forward input gradients. Online model, EMA, and complete optimizer including observation shadows must reproduce all three recorded hashes. The code checks native None and per-parameter step advancement, the pair clock, the reconstructed Q sum and both numerator and applied-update closure.
- The full S_dense H_SHADOW uses the actual post-step mR plus all six .9 auxiliary shadows, with the same WD, parameter clocks, lr and v. Each STAR_SKIP/EXPERT group has one replacement of the full numerator and six individual history-source replacements using the corresponding corrected Q and M9 terms. A single-source action keeps every other numerator term. Unchanged parameters and model buffers come from the actual H step. S99 real/WD residuals do not enter the selected A21 numerator.
- Each of the 14 group/source replacements has raw and norm-matched variants. A uses the reference group's L2 norm as the target. All coordinates share one scale. The raw norm, applied norm, mismatch, and update cosine are recorded. Both-zero preserves the update, zero reference scales the alternative to zero, and a zero alternative with a nonzero reference records an explicit unmatchable status. The 29th action is the raw full H_SHADOW diagnostic. Each scored branch uses the same pre-step EMA as the actual branch.
- Each successful action retains online/fixed EMA × P_A/P_B/V_A/V_B × six cells, six-cell Macro, population AUC, sampled-pair AUC, exact tie/correction/break transition tables, and natural LL. Descriptive reliability includes row-fold sign agreement, sampled-pair/population differences, and transition closure. The gate uses fixed-EMA V_B population AUC. V_B is the direction B independent confirmation fold. The action menu is fixed before observing any panel. The committed rows are accepted by the prior reviewed `results.localization_evidence`.
- One per-anchor scalar record plus a digest-bound receipt survives a stop. Reuse verifies the source hash and retained parent receipt. All model, state and counterfactual tensors are transient in RAM. The foreground supervisor uses measured per-device memory/power, host RAM, physical-fit correction, resampling per startup, and first complete-anchor allocation protection. Initial full-process residency comes from the completed A_HISTORY worker measurements. Once a diagnostic worker allocates, current measured residency updates that profile.
- The complete two-intensity materialization requires 80 accepted anchors with exactly 29 actions each, each measured action with four panels and six receiver cells in both readouts. It writes 320 group/seed/epoch/dose/scale rows in `tables/A_TARGET/localization.json` and `A_TARGET_RUN.json` with all 80 input digests and `localization_sha256`. An unmatchable norm remains explicit. The gate input consumes only whole-group raw/norm-matched rows.

## Engineering check

In the registered project Python, `python -m experiments.conditional_history_shared_update_20261122.a_localization_checks` passed. A five-step FP32 synthetic model used both shared groups, changing PAIR sources, an empty pair, and native missing gradients. An independently accumulated ordinary mixed first moment matched the group shadow update. Six single-source differences reconstructed the full displacement. The check validated packet immutability through the entire probe, 29 action identities, 240 panel forwards, group norm controls, the three zero-norm cases and 80 synthetic gate-input rows over the registered anchor/seed grid. No scientific packet, full training or Test data was evaluated as part of implementation.

## Review and run contract

The implementation source hash for this handoff is `225beded64c94932a2d32d36184f0290a2de64bc04c368785c09f92020e89e95`. A separate reviewer can recompute it using `a_localization_stage.source_id()`. The reviewer receipt at `research_reports/conditional_history_shared_update_20261122/A_TARGET_REVIEW_RECEIPT.json` binds the verdict, empty blocking issues, source_hash and design_sha256. Only the independent reviewer can issue this receipt. Scientific CLI actions check it before packet access or scheduling.

The runner commands below each stay in the foreground. Each arm command verifies its parent stage has completed. The combined table is available only after both commands complete.

```bash
PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
MOD=experiments.conditional_history_shared_update_20261122.a_localization_stage
"$PY" -m "$MOD" run --arm A21
"$PY" -m "$MOD" run --arm A40_HISTORY_L025
"$PY" -m "$MOD" combine
```

The measured interventions are one-step state-local effects. Persistent training outcomes belong to the registered scope arms if the dose gate opens them. This implementation report contains no scientific outcomes.
