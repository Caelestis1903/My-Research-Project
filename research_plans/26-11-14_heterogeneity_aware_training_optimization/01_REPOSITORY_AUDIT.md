# Repository Audit

**Audit snapshot:** 2026-11-14  
**Repository:** Caelestis1903/My-Research-Project  
**main HEAD:** 01979f6588935f713029b2b5e04557de9b8c153f  
**Remote branches enumerated:** 46, including main  
**Important limitation:** GitHub compare responses cap some large diffs at 300 files. For those branches the branch-level ahead/behind and returned changed-file sample are recorded; the cap is not misrepresented as a complete file inventory. Key relevant reports and implementation roots were opened directly.

## 1. Main context files

| File | Audit result / role |
|---|---|
| EXP_AI_CONTEXT.md | Read; latest research handoffs, evaluation rules, accepted findings |
| experiments/Experimental_Global_Summary.md | Read; consolidated findings and dated handoffs |
| experiments/Experimental_Stable_Findings.md | Read; stable evidence registry, including SF038–SF048 |
| experiments/Experimental_Open_Questions.md | Read; older unresolved questions, with newer handoffs taking priority |
| experiments/Experimental_Experiment_Index.md | Read; indexed research stages through EXP060; not 60 independent algorithms |
| MDMT_ALGORITHM_METRICS_LEDGER.md | Read; per-cell and macro metrics with protocol identity |
| AGENT.md | Read; current code and engineering contract |
| start.md | Not present on current main at audit time |
| AGENTS.md | Not present on current main; an AGENTS.md copy exists on older research branches |

Main root also contains research_plans/, research_reports/, experiments/, history_experiments/, MDMTRec/ and historical data/utility assets. Main report archive includes 2026-11-02 through 2026-11-06 core reports; later 11-09 to 11-11 research branches are not fully merged into main.

## 2. All remote branches: tip SHA, author timestamp (UTC), comparison to main

Ahead/behind means commits unique to branch / commits unique to main. 0/behind indicates the branch is an ancestor or stale branch; it is not an unreviewed new experiment.

| Branch | Tip SHA (short) | Tip date UTC | Ahead / behind |
|---|---|---|---:|
| 10-24 | b64b2922 | 2026-10-24 14:13 | 1 / 163 |
| 26-10-25 | 619371d9 | 2026-10-25 05:57 | 2 / 163 |
| 26-10-26 | dc32f60a | 2026-10-26 06:41 | 0 / 160 |
| 26-10-27 | 03c33809 | 2026-10-27 10:33 | 2 / 148 |
| 26-10-28 | 97151c74 | 2026-10-27 17:17 | 3 / 148 |
| 26-10-31 | ee6e81fd | 2026-10-31 11:27 | 1 / 143 |
| 26-11-01_v2 | ee4cf051 | 2026-11-01 07:21 | 2 / 134 |
| 26-11-01_v3 | 6f5742c2 | 2026-11-01 11:59 | 2 / 121 |
| 26-11-01 | a6f496b5 | 2026-11-01 03:56 | 1 / 135 |
| 26-11-02_v2 | 38a53bc1 | 2026-11-02 10:39 | 3 / 117 |
| 26-11-02_v3 | a06f9d3e | 2026-11-03 08:29 | 3 / 104 |
| 26-11-02 | e59efa2f | 2026-11-02 00:32 | 4 / 118 |
| 26-11-03_history_information | f9c2a8ef | 2026-11-03 11:11 | 1 / 102 |
| 26-11-06 | 40705dfe | 2026-11-05 16:24 | 13 / 95 |
| 26-11-07 | ffa3de06 | 2026-11-06 16:45 | 17 / 95 |
| 26-11-09_v2 | a5f461de | 2026-11-09 14:10 | 1 / 3 |
| 26-11-09 | 1391affa | 2026-11-09 03:32 | 18 / 95 |
| 26-11-11 | fb120946 | 2026-11-11 13:06 | 19 / 95 |
| 2026-10-25_15-56-04 | 8a4b3b29 | 2026-10-25 12:31 | 3 / 163 |
| baseline/mdmt-sota-2526 | d2fdd272 | 2026-10-29 09:53 | 5 / 148 |
| codex/add-mdmt-frontier-plan-2026-10-18 | 9f6d8968 | 2026-10-18 11:01 | 4 / 165 |
| codex/add-research-plans-2026-10-17 | 7a7cf543 | 2026-10-17 08:03 | 13 / 176 |
| codex/conditional-paths-2026-10-19 | eab817b3 | 2026-10-20 03:24 | 19 / 163 |
| codex/execute-2026-10-15 | b17cb270 | 2026-10-16 05:36 | 15 / 183 |
| codex/execute-2026-10-16-v2 | c31b531e | 2026-10-16 12:00 | 36 / 183 |
| codex/execute-2026-10-16-v3 | 2d04c6de | 2026-10-17 05:19 | 42 / 176 |
| codex/execute-2026-10-16 | aa39e7b2 | 2026-10-16 09:23 | 24 / 183 |
| codex/execute-2026-10-17-run | 5550afbc | 2026-10-18 07:00 | 79 / 176 |
| codex/execute-2026-10-18 | 14cf09a3 | 2026-10-18 11:45 | 2 / 164 |
| codex/execute-n19-2026-10-18 | c4403506 | 2026-10-18 16:12 | 24 / 164 |
| codex/m3oe-reproduction-2026-10-21 | c5618e28 | 2026-10-21 06:50 | 4 / 163 |
| codex/natural-movielens-reassessment-2026-10-20 | f0b1084b | 2026-10-20 17:00 | 69 / 163 |
| codex/research-plan-2026-10-16-v3 | 0698fe31 | 2026-10-16 15:02 | 9 / 177 |
| codex/2026-10-16-v2-research-plan | 04ea9617 | 2026-10-16 10:08 | 2 / 180 |
| docs/mdmt-ap-codex-plan-20261027 | b176d6aa | 2026-10-27 01:15 | 0 / 153 |
| main | 01979f65 | 2026-11-11 05:47 | 0 / 0 |
| planning/ga31-v2-reassessment | d857e4cb | 2026-10-31 13:23 | 10 / 143 |
| research/26-11-01-gradient-augmentation-momentum-plan | f2444e72 | 2026-11-01 05:12 | 16 / 135 |
| research/26-11-04-gaussian-domain-coverage | c8675ae2 | 2026-11-04 05:08 | 7 / 95 |
| research-plan/26-10-27-v2 | ed94caa8 | 2026-10-27 08:02 | 13 / 152 |
| research-plan/26-11-2-v3-conditional-information-momentum | 2c4ab338 | 2026-11-02 14:03 | 0 / 105 |
| research-plan/26-11-03-mdmt-history-information | 75f698d1 | 2026-11-03 09:14 | 0 / 103 |
| research-plan/26-11-03-overnight-ab-auc | 9e682f3e | 2026-11-04 01:45 | 4 / 95 |
| research-plan/2026-10-19-modulation-path-generalization | 3bd84f10 | 2026-10-19 04:46 | 5 / 164 |
| research-plan/2026-10-26-v2-mechanisms | e9e14ab9 | 2026-10-26 17:42 | 2 / 156 |
| review/ga31-r3-codex-ready-20261031 | 2f5f3058 | 2026-10-31 14:32 | 0 / 136 |

## 3. Relevant branch groups and changed content

### 3.1 Historical optimization and augmentation
- 26-10-31: 63 changed files, including experiments/gradient_aug_20261031/ and research_reports/2026-10-31/.
- 26-11-01_v2, 26-11-01, 26-11-01_v3: successive gradient/augmentation/momentum and AIG-M implementations; experiments/gradient_aug_20261031_v2/, experiments/gradient_aug_momentum_20261101/, experiments/gradient_aug_aig_m_20261101/.
- 26-11-02: GATS B1/B2/B3/C2; its full compare is capped at 300 files.
- 26-11-02_v2: EMA02V2 implementation experiments/ema_view_design_20261102_v2/.
- 26-11-02_v3: conditional information / PAIR / ECCM implementation experiments/conditional_information_momentum_20261102_v3/.
- 26-11-03_history_information: experiments/mdmt_history_information_20261103/, including mixing, objectives, optimizers and tests.
- research/26-11-04-gaussian-domain-coverage: Gaussian coverage and overnight A+B implementations.

### 3.2 EMA research
- 26-11-06 and 26-11-07: AUG_HISTORY implementation and complete artifacts.
- 26-11-09 / 26-11-09_v2: PUB migration and later research; 26-11-09_v2 is three commits behind main and one ahead.
- 26-11-11: 19 commits ahead / 95 behind main; contains 11-10 causal attribution and 11-11 next-phase research. The full compare is capped at 300 files; latest reports/plans were read directly.

### 3.3 Earlier research/planning branches
- 10-15 through 10-21 execution branches contain extensive S0–S8 gradient/objective, optimizer state, and model-trajectory work. These branches share substantial artifacts and must not be counted as independent replications.
- 10-18/10-19 conditional path and frontier planning, 10-25/10-27 relation-basis and BASP-A, and 10-31 GA31/R3 plans were included in branch comparison.
- Several branches are stale plan snapshots with zero unique commits; they are not independent experimental evidence.

## 4. Directly opened reports / implementation evidence

- PAIR/ECCM report: research_reports/2026-11-02_mdmt_information_momentum_v3/REPORT.md on 26-11-02_v3, SHA 9326005c10ef9a550c0f35bb7dc4547424dcdb9a.
- Conditional information / real history report: research_reports/2026-11-03_mdmt_history_information/REPORT.md on 26-11-03_history_information, SHA 274032ce44c46d99eff5f5f6f4bda86b14cce5a3.
- Five-seed completion: research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md on 26-11-11, SHA 8580593b9c8607376fe2ad74adb2eaac1e35f03c.
- Overnight A+B report and Gaussian coverage report on 26-11-11.
- AUG_HISTORY and EMA mechanism reports in main archive.
- 11-10 EMA causal attribution report and 11-11 next-phase context/plan on 26-11-11.
- AGENT.md on main; start.md and AGENTS.md are not present on main.

## 5. Audit completeness and boundaries

The audit enumerated all 46 remote branches, retrieved tip SHAs and timestamps, compared each branch to main, and identified branch groups and implementation/report paths. The most relevant later reports and source implementations were read directly. GitHub compare truncates some large branch diffs at 300 files; this is not represented as a complete file inventory. This audit did not read every historical binary artifact, every log line, or every source file in every legacy branch. Where a report was not directly opened, the ledger identifies it as a source-indexed claim rather than claiming a full raw-artifact reanalysis.

No historical branch, code, report, or experiment artifact was merged into main by this planning task.
