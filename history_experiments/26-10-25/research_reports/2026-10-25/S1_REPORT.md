> 本次仅发布研究报告与设计文档（含报告统计图）；运行代码、CSV/JSON结果、原始数据、日志、逐行预测和checkpoint均保留在服务器。正文数值与研究结论未改。
> 原文SHA256：`0ac4210b10227e7366d75fac955cabd38cf6aecd6422aac1de0e53f7c5fa4586`。

# S1 mechanism report — development evidence only

All eight accepted warmups completed10epochs, 192 H1 logical updates and80 H5 updates completed. Two superseded mean64 attempts are retained and excluded from accepted carrier comparison. The corrected mean bridge uses raw expert mean. All execution exitcodes are0. S1 is one optimization seed (3101); evidence strength is WEAK and does not decide joint-MVP admission.

| Dataset | GLOBAL51 Q AUC / LL | DIRECT51 Q AUC / LL | LPS51 Q AUC / LL | CAL_PROB_SELECT Q AUC / LL | CAL_OFF Q AUC / LL |
|---|---|---|---|---|---|
| ML | .807386 / .476041 | .807372 / .476025 | .807393 / .476038 | .807802 / .475651 | .808250 / .474664 |
| KR | .683375 / .471091 | .683445 / .471119 | .683325 / .471134 | .683755 / .471055 | .683645 / .469663 |

FACT: in the concatK4 frozen predictor, LPS−DIRECT AUC is approximately +.000021 (ML) and−.000120 (KR). The probability-based controls are competitive; CAL_OFF has lower risk on both datasets. These are frozen READ_FROZEN policies fitted on P_label/P_cal, scored on Q_explore; not complete training effects. INFERENCE: calibration or the additional meta-label information is a plausible explanation competing with local knowledge selection. HYPOTHESIS: a joint content/reader trajectory may differ; S2 must proceed regardless of this weak frozen signal.

C8 H1 best measured actions across12 anchors: ML [1,5,3,5,1,1,2,4,3,7,1,7]; KR [7,4,6,4,5,1,1,1,1,1,3,3]. Maximum observed macro utility relative BASE is .00059434/.00618027. At epoch10, H5 winners are action1 forML (+.00097125) and action4 forKR (+.00012794). These limited-panel maxima are optimistic hindsight opportunities, not deployable guarantees. Different H1/H5 winners and tiny frozen LPS gains contradict an automatic short-term-to-long-term success interpretation.

Static calibration uses eight complete5epoch branches per dataset, fixed epoch15 validation LL. ML selects action5=TRANSFER_1_TO_2; KR selects action0=BASE. All scores are in tables/static_scores.json. Temporal bootstrap uses these shared fallback templates; state-coordinate correction and original packet preservation are documented in S1_IMPLEMENTATION_AMENDMENT.md. FREEZE_WARMUP is not evaluated on its selection12 anchors.

During8 truly prequential warmup scoring anchors, mean measured regret is ML STATIC/HIST/ridge/NN .000378, LAST .000279; KR STATIC/HIST/ridge/NN .001168, LAST .000810. Equal regrets are expected while the prescribed8-error readiness requirement is being accumulated; they do not demonstrate state equivalence after deployment. Whole40-anchor forward-tail scoring belongs to S2.

Domain size is not established as a main explanatory variable (WEAK). Natural domains simultaneously differ in labels, representation and baseline difficulty. Domain/task support and six-cell absolute metrics remain in manifests/data-role files and each frozen_policy_metrics.json; no assumption that small domains are difficult, nor any small-improves/large-declines trade-off, is used to choose further jobs.

Next: all48 core joint trajectories and mandatory aux0 factorial. Competitive CAL triggers a bounded full closed-loop reserve package after reserving mandatory aux costs; if the full package cannot fit, mark PENDING_BUDGET and do not claim that LPS beat calibration. No outer role has been accessed. New independent confirmation remains UNSEEN_CONFIRM_PENDING_DATA.

Raw artifacts: tables/S1_warmups.csv; tables/policy_regret.csv; diagnostics/<accepted_run>/frozen_policy_metrics.json and Q_explore_cache.npz; runs/<native>/attempt_001/utility_H1_*.npz; diagnostics/<native>/utility_H5_011.npz; S1_QUEUE_execution.jsonl; STATIC_QUEUE_execution.jsonl; S1_DIAGNOSTICS_EXIT.json.

Additional exact-cache diagnostics: `tables/S1_nested_panels.csv` reports nested256/1024 from the identical branches, not independent experiments. Mean winner MC-SE on H1 P_cal is ML .000312/.000148 and KR .001746/.000503. Winner conditioning remains optimistic. `S1_local_response.csv` shows both K4 frozen LPS policies chose depth0; different actions relative GLOBAL arise from the honest value split, not demonstrated within-cell local routing. `S1_target_key_support.csv` reports soft/hard target responsibility, explicitly not owner fit support. `S1_proxy_comparison.csv` compares Brier versus exact BCE action averages descriptively; the proxy never controls training. These checks strengthen the need for full closed-loop comparison, not a claim of local-policy success.

`tables/S1_owner_unit_support.csv` supplements the target support table with all-fit owner soft/hard responsibility, frozen-key entropy/temperature, and first2048 owner-row unit output norms/cosines at epoch10. Finite support or distinct unit outputs alone does not establish transferable specialization; the matched K1/K4 complete trajectories remain the relevant quality test.
