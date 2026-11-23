# D2 S1 independent scientific review

**PASS.** The raw Validation results reproduce `COMPLETE_NO_GO_S1`. KuaiRand was the sole qualifying dataset for `P_AXIS` on V_A. On V_B neither dataset had a positive aggregate contrast against J_ACC, and neither had two anchors positive against both registered confirming controls. S2 and S3 stay closed. [Design specification](DESIGN_SPECIFICATION.md) · [Registered statistics and gates](../../research_plans/11-23/07_METRICS_AND_STATISTICS.md) · [Scope decision](../../artifacts/mdmt_shared_private_coordination_20261123/SCOPE_DECISION.json)

## Verification

- **Population, repeats and controls.** I matched the ten completed logical IDs to the manifest, verified the ten completion-result hashes and the 500 epoch and 30 fork hashes, and found all E10/E25/E40 anchors for seeds 2022–2026 on each dataset. Each anchor contains the 19 named forks in the erratum, giving 570. Earliest strict maximum Macro AUC reproduces the selected epoch independently for online and fixed EMA on all ten trajectories. The raw receipts record zero Test evaluations. [Run manifest](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN_MANIFEST.json) · [Anchor index](../../artifacts/mdmt_shared_private_coordination_20261123/ANCHOR_INDEX.json) · [Design erratum](DESIGN_ERRATUM.md) · [Runner report](RUNNER_REPORT.md)
- **Panels and sampled pairs.** V_A and V_B row IDs are disjoint and cover full Validation. Every cell has both classes. Each dataset/panel/cell has 4,096 unique frozen pair IDs and row-ID pairs within its manifest panel. All 52,920 raw transition tables match the corresponding pair/row hashes, and the nine grade bins sum to the published upward, downward and unchanged counts. For example, KuaiRand full Validation post-U d0t1 reproduces 1,091 upward and 1,212 downward transitions, and d1t0 reproduces 1,270 and 1,203. These counts include repeated observations of the same frozen pairs across mappings, anchors and seeds. [MovieLens panel manifest](../../artifacts/mdmt_shared_private_coordination_20261123/PANELS_movielens.json) · [KuaiRand panel manifest](../../artifacts/mdmt_shared_private_coordination_20261123/PANELS_kuairand.json) · [Pair transitions](../../artifacts/mdmt_shared_private_coordination_20261123/PAIR_TRANSITIONS.csv)
- **V_A selection.** I recomputed Macro AUC as the equal mean of six cells, averaged AB and BA at each anchor, reduced the anchors within seed, then used the five paired seeds. The only scope/dataset to clear all three positive means and the −0.001 cell guard was KuaiRand `P_AXIS`. Its differences against U_ONLY, reversal and J_ACC were +9.618, +7.309 and +30.862 ×10⁻⁶. The equal-dataset ranking score was −12.473 ×10⁻⁶, consistent with qualification on one dataset. [Numerical readout](S1_NUMERIC_READOUT.md#registered-selection-and-confirmation)
- **V_B confirmation.** The means, sample-SD-based paired intervals and signs below reproduce the reports. Both datasets passed the V_B cell guard, with worst confirming cell means of −35.739 ×10⁻⁶ for MovieLens and −265.792 ×10⁻⁶ for KuaiRand. The only jointly positive KuaiRand anchor was E25. MovieLens had zero and was ineligible after V_A. MovieLens J_ACC was negative in all five seeds at E25 and E40. Both datasets’ aggregate confirming intervals have positive upper bounds, so the other-dataset exclusion rule was clear. The primary confirmation conditions still failed. [Raw forks](../../artifacts/mdmt_shared_private_coordination_20261123/runs/) · [Scope decision](../../artifacts/mdmt_shared_private_coordination_20261123/SCOPE_DECISION.json)

| V_B online difference | Mean ×10⁻⁶ | Paired 95% CI ×10⁻⁶ | Positive seeds |
| --- | ---: | ---: | ---: |
| MovieLens vs reversal | +11.265 | [−19.429, +41.960] | 4/5 |
| MovieLens vs J_ACC | −1.147 | [−103.616, +101.322] | 2/5 |
| KuaiRand vs reversal | +52.746 | [+3.875, +101.616] | 4/5 |
| KuaiRand vs J_ACC | −82.005 | [−251.364, +87.354] | 1/5 |
| Equal datasets vs reversal | +32.006 | [+9.704, +54.307] | 5/5 |
| Equal datasets vs J_ACC | −41.576 | [−137.970, +54.818] | 1/5 |

The equal-dataset intervals use the five same-seed two-dataset averages. KuaiRand’s five J_ACC differences were −32.037, −311.229, +52.101, −73.067 and −45.793 ×10⁻⁶ in seed order. [Paired seeds](../../artifacts/mdmt_shared_private_coordination_20261123/PAIRED_SEEDS.csv)

## Scientific interpretation

- **H1 · MEDIUM for local six-cell heterogeneity.** Full Validation MovieLens mean post-U Macro ΔAUC was −313.073 ×10⁻⁶ with five negative seed means and six negative cell means. KuaiRand had opposing d0 and d1 cell means and a wide Macro interval. On KuaiRand, d1 exceeded d0 in mean absolute post-U logit shift in both tasks in 14 of 15 seed–anchor blocks, and in mean absolute pair-margin shift in both tasks in all 15, after averaging AB and BA. This repeated pattern supports the bounded local claim. The blocks share five training replicates, and the full panel shares rows with V_A/V_B. [Literal post-U response](S1_NUMERIC_READOUT.md#literal-online-post-u-response-and-after-p-change)
- **H2 · STRONG for the registered NO_GO and MEDIUM for the local reversal contrast.** KuaiRand’s positive reversal contrast recurred in the means at all three anchors and had four positive seed averages on V_B. Its V_B AB/BA effects of +125.284/−19.793 ×10⁻⁶ show sensitivity to batch assignment. Full J_ACC also updates P_CELL, which the frozen sequential arms hold fixed. Reversal matches group clocks, and J2 uses two full Adam updates, so these controls identify different contrasts. The data leave batch composition, training stage, accumulation and EMA smoothing as competing explanations for parts of the pattern. Evidence for extra ranking gains from the lower V_B LogLoss is **WEAK**. Online J_ACC Macro LogLoss improved on both datasets while mean Macro AUC declined. S1 supplies no estimate of a 50-epoch sequential effect or H3 qualification. [Endpoint contrasts](S1_NUMERIC_READOUT.md#endpoint-auc-and-logloss) · [Scientific analysis](S1_REPORT.md)
- **Domain size as the main explanation · WEAK.** The smallest MovieLens training domain has the highest selected fixed-EMA baseline AUC, while the largest KuaiRand domain has the highest baseline AUC. Baseline AUC describes ranking difficulty under this model and population, with different label supports across domains and tasks. KuaiRand d0 gained and d1/d2 lost against U_ONLY. Against J_ACC, d0 lost and the smallest domain d2 gained. MovieLens lost in all three domains against U_ONLY. The small-domain gain/large-domain decline pattern depends on the chosen domain and comparator. [Selected baselines](S1_NUMERIC_READOUT.md#selected-j_orig-validation-endpoints-and-support) · [Domain and task contrasts](S1_NUMERIC_READOUT.md#endpoint-auc-and-logloss)

## Findings and corrections

**MINOR wording clarification.** The S1 analysis describes KuaiRand E25/E40 as recovery after U. On full Validation, E25 partially recovered a mean post-U deficit (−200.027 then +58.953 ×10⁻⁶). E40 added +62.265 ×10⁻⁶ to an already positive post-U mean of +20.516 ×10⁻⁶. The numerical table is correct. Across all three anchors, the after-P mean was +34.533 ×10⁻⁶, with two positive seeds and a paired 95% CI [−113.536, +182.602] ×10⁻⁶. This review records the wording correction. Zero critical or major findings. [S1 analysis](S1_REPORT.md) · [Numerical readout](S1_NUMERIC_READOUT.md#literal-online-post-u-response-and-after-p-change)

The runner receipt’s `COMPLETE` records successful S1 execution. The scientific disposition is `COMPLETE_NO_GO_S1`. No scope can be reselected from V_B. The proper-subset S2 clock mismatch remains documented and is moot for this disposition. S2 training and S3 scoring are ineligible, and Test remains sealed. [Final report](FINAL_REPORT.md) · [Scope decision](../../artifacts/mdmt_shared_private_coordination_20261123/SCOPE_DECISION.json)

## Evidence identity and limits

The review used project Python for read-only recomputation from retained scalar Validation and run records at `26-11-23_v2@f66bf17cba2980618e6f2edd1c7381b73c35cac5`. Pair transitions were checked against frozen pair IDs and the raw grade tables. Individual logits and fresh RAM readbacks were outside this scientific review.

| Reviewed input | SHA-256 |
| --- | --- |
| [S1_RUN.json](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN.json) | `79f961d44bac89657dacb380eb5aaff7253ca58b6c2555031c04dbff1bdc0000` |
| [SCOPE_DECISION.json](../../artifacts/mdmt_shared_private_coordination_20261123/SCOPE_DECISION.json) | `fc3ec521e3777dde08551d03aad24c958f44506fd393ed2f0a7f0d6ffa8a7188` |
| [S1_RUN_MANIFEST.json](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN_MANIFEST.json) | `124c89bc9ee07459d4139b6b31639ad1ffa2be5828e15b794af5cea985ed8043` |
| [ANCHOR_INDEX.json](../../artifacts/mdmt_shared_private_coordination_20261123/ANCHOR_INDEX.json) | `2049c251f1dbd9a60d10791351e4c89d586ff2588e012c4652a5e781275ecabd` |
| [PANELS_movielens.json](../../artifacts/mdmt_shared_private_coordination_20261123/PANELS_movielens.json) | `def51cdc0e37397c12235372b6c687d2de9fb2552285bd4eb1b13579e819a068` |
| [PANELS_kuairand.json](../../artifacts/mdmt_shared_private_coordination_20261123/PANELS_kuairand.json) | `3c27736a8835f0d2d1498c389d4eab53c8e89555b74e9243789d3ff673740f00` |
| [Registered statistics](../../research_plans/11-23/07_METRICS_AND_STATISTICS.md) | `c62822d26dd467d4667c509a55170f7adfa1d569f55cebb3984363cdc8eecaa5` |
| [DESIGN_SPECIFICATION.md](DESIGN_SPECIFICATION.md) | `7c1df7c873468de90b17ff53e32deab0721001bdd53dab2600de3b924467a74b` |
| [DESIGN_ERRATUM.md](DESIGN_ERRATUM.md) | `45b42e2d47dc10341ebe898e86d42a137861587fd4024a9a61627c1ecab792e5` |
| [S1_REPORT.md](S1_REPORT.md) | `c6b62b7a96f698271fa0d3a2c44b39a489ee79f132411eb307dc7de866019323` |
| [FINAL_REPORT.md](FINAL_REPORT.md) | `9de3b8ccd3f2f830e0017c4614eb66ee1f7f013f8978949124907ef1978473f7` |
| [S1_NUMERIC_READOUT.md](S1_NUMERIC_READOUT.md) | `4c15b60cbf57d31d748c6754364ca704e0c5d50f7837ee14d663419f152afa4a` |
| [RUNNER_REPORT.md](RUNNER_REPORT.md) | `f6d6751669a1de4bbb3127d5a309a1a54b67394e5c07cf134e5e0d513db8f2f4` |

## Orchestrator publication note

The published S1 report applies the review’s E25/E40 wording correction and incorporates its 14/15 and 15/15 KuaiRand checks. The final report adds the review and RAM closure links. The numerical readout, registered decision and raw evidence retain the reviewed hashes above.

| Published report | SHA-256 |
| --- | --- |
| [S1_REPORT.md](S1_REPORT.md) | `1f947087ef851dbc768c6c9a506d75224bba3b6d7f2189be64dfd3732700fe24` |
| [FINAL_REPORT.md](FINAL_REPORT.md) | `b9fee710f5c698ba99f6ae50b241522d7b0012847cf94f34b9eb87e59aee9af7` |
