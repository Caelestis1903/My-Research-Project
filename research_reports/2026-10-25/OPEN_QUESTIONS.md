# 开放问题与证据边界

由本轮研究问题记录转换为Markdown；不是原始实验数据。

## Q_CALIBRATION

- **fact**：S1 CAL_OFF and CAL_PROB_SELECT are competitive; S2 CAL_OFF independently selected along OWN_OFF.
- **evidence**：diagnostics/S1_movielens_concat_K4_3101/frozen_policy_metrics.json；diagnostics/S1_kuairand_concat_K4_3101/frozen_policy_metrics.json；tables/S2_Q_explore.json
- **alternatives**：extra meta-label calibration；foreign-unit content；reader/actor fitting capacity
- **test**：Locked two-dataset CAL_PROB_SELECT closed-loop vs DIRECT51 and CAL_OFF with matched roles and main exposure.
- **status**：PENDING_BUDGET
- **budget_source**：S2_RESERVE
- **evidence_strength**：MEDIUM for repeated ML CAL_OFF improvement; WEAK for universal calibration benefit or CAL_PROB closed-loop

## Q_AUX

- **fact**：All four mandatory aux0 trajectories stopped by budget before epoch24.
- **evidence**：tables/reserve_account.json
- **alternatives**：prototype specialization due to aux；capacity or router advantage independent of aux
- **test**：Complete K1/K4 x aux0/.1 at seed3101 from epoch5 and fixed epoch24; no epoch10 aux-contaminated fork.
- **status**：AUX_ATTRIBUTION_PENDING_BUDGET
- **budget_source**：S2_RESERVE
- **evidence_strength**：WEAK; no complete factorial

## Q_ROUTER_CAPACITY

- **fact**：K4 has51 action outputs; K1 has7 despite identical42-dimensional input and equal source-bank parameter budget. Complete K4-vs-K1 ranking contrast is positive in only1/4 pairs; no content package promoted.
- **evidence**：manifests/actions.json
- **alternatives**：multiple transferable units；larger action-head capacity or stronger residual selection
- **test**：If complete content contrasts justify it, K1_H31 versus default K1 and K4_DIRECT under matched budget.
- **status**：NOT_TRIGGERED_CONTENT_NOT_PROMOTED
- **budget_source**：S2_RESERVE
- **evidence_strength**：WEAK; not identified

## Q_STATE_GATE

- **fact**：All four BASE40 traces complete. State actual actions change HIST zero times in forward14; occasional nonzero gamma still rejected by the margin. Raw state gains are negative in3/4 traces at gamma1.
- **evidence**：tables/policy_regret.csv
- **alternatives**：state adds no predictive information；finite-panel noise masks small gain；state proposal useful but empirical margin conservative
- **test**：Freeze existing predictors/rules and compare actual versus raw proposals on a preregistered later stream with independent risk panels; do not retune on scored labels.
- **status**：CURRENT_STATE_VERSION_NOT_SUPPORTED
- **budget_source**：next version, not automatic extension
- **evidence_strength**：MEDIUM for this fixed decision pipeline; no universal claim about state information

## Q_DOMAIN_SIZE

- **fact**：Only three natural domains per dataset; size co-varies with label prevalence, representation and native difficulty.
- **evidence**：tables/S1_owner_unit_support.csv；tables/S2_cell_contrasts.csv
- **alternatives**：support size；baseline difficulty；task-specific response or training stage
- **test**：Hold domain identity/labels/test support fixed and manipulate fit support size with paired seeds; compare size-matched domains of different native difficulty.
- **status**：OPEN_NEXT_VERSION
- **budget_source**：not authorized in current finite matrix
- **evidence_strength**：WEAK for causal size explanation

## Q_CONFIRMATION

- **fact**：Old Q_confirm and test were previously analyzed; no independent new artifact verified.
- **evidence**：manifests/exposure.json；manifests/historical_identity.json
- **alternatives**：benchmark-specific effect；generalizable mechanism
- **test**：Obtain a verified new artifact with exposure history locked before training or selection.
- **status**：UNSEEN_CONFIRM_PENDING_DATA
- **evidence_strength**：No unseen-confirm evidence

## Q_SIMPLE_FEEDBACK

- **fact**：CUM_MEAN, EMA and FREEZE_WARMUP have positive mean forward14 measured utility relative HIST on all four common BASE traces.
- **evidence**：tables/S2_forward14_summary.csv；tables/C8_cross_panel_agreement.csv
- **alternatives**：simple persistent action preference；conservative margin suppresses useful signal；one-step proxy does not translate to full trajectory
- **test**：Pre-register complete24epoch CUM_MEAN/EMA or FREEZE against STATIC8 and HIST with identical roles and seeds, report actual closed-loop LL/AUC and cost; do not promote from trace alone.
- **status**：PENDING_BUDGET
- **budget_source**：S2_RESERVE exhausted by mandatory aux; next version only
- **evidence_strength**：MEDIUM for same-state trace effect; WEAK for training benefit
