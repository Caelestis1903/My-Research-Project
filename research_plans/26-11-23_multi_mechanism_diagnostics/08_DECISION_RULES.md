# 08 — 决策规则

## 1. 状态定义

- PASS：预注册阶段目标达到，证据与身份审计通过。
- NO_GO：实验正常完成，但当前具体机制/动作在当前协议下未达到目标。
- TECHNICAL_BLOCKED：代码、数据、资源、状态恢复或执行身份问题导致不能作科学判断。
- INCONCLUSIVE：运行完成但关键支持、独立确认或统计信息不足。
- NOT_TRIGGERED：预注册触发条件未满足，不执行后续阶段。

## 2. M1 gate

S1 PASS 条件：
1. P-consistency step 正确执行，U 参数和 U Adam state bitwise unchanged；
2. L_cons 在至少 4/5 seeds、至少两个 anchors 下降；
3. 相对 P-BCE 与 shuffled control 的梯度一致性改善具有一致方向；
4. V_A 上 pairwise correction 或六格 AUC 有正向关联，且无 cell mean < −0.001。

S2 PASS 条件：
1. V_B 上独立复现一致性改善；
2. 预测排序收益不由单一 seed/anchor 驱动；
3. 相对 P-BCE/shuffled controls 的差异保持；
4. 代理变化与实际排序响应方向一致。

若 L_cons 改善但 AUC/pair correction 无响应，结论为 MECHANISM_PRESENT_NO_RANKING_TRANSLATION，当前方法 NO_GO，不否认梯度可塑现象。

## 3. M2 gate

S1 PASS 条件：
- task-specific score 对局部 pair-margin change 的预测优于 task-agnostic 与 shuffled controls；
- 在至少 4/5 seeds 方向一致，且跨至少两个 anchors；
- 不由 loss/gradient norm/label frequency 单一代理解释；
- 有足够样本支持和稳定 score。

S2 PASS 条件：
- V_B 独立确认排序响应；
- 同一任务条件影响在两个数据集或同一数据集多个域任务单元中有可解释边界；
- cell guard 通过。

若仅有梯度对齐相关而无排序预测，NO_GO 作为 AUC 方法。

## 4. M3 gate

仅当触发条件满足才运行。PASS 需要 interaction residual 在 V_B/独立 fork 中方向复现，且相对 A-only/B-only/A/A、B/B controls 有增量解释。否则 NO_GO 或 INCONCLUSIVE。不得扩展到全量样本对。

## 5. M4/M5

- M4 只有在具体字段/关系有可解释、可干预的任务条件错误依赖时，才建议形成独立设计。
- M5 只有在字段级语义保持约束有明确来源时，才建议形成增强设计。
- 缺少语义支持时 NO_GO，不得构造虚假反事实。

## 6. S3 short trajectory

PASS：
- 5 epochs 连续动作保持预期机制读数；
- 相对匹配预算控制不消失；
- 至少 4/5 seeds 同方向；
- 六格 cell guard 不破；
- 不只是额外 optimizer steps 或样本暴露。

初期负向后期正向或反之必须完整报告，不得只选 E5。

## 7. S4

只有 S3 PASS、算法规格冻结、独立 Reviewer PASS 才触发。正式 AUC mean gain ≥ +0.003、每 cell mean delta ≥ −0.001；LogLoss 单独评价。Test freeze 后只运行一次正式评价。没有新算法规格则 NOT_TRIGGERED。

## 8. STOP 语义

NO_GO 仅针对实际测试的动作和条件，不自动关闭 M1/M2/M3 整个研究家族。TECHNICAL_BLOCKED 不能写成科学失败；INCONCLUSIVE 不能写成成功或失败。


## 9. CGPA full-training gate

CGPA is eligible only if M1:
- passes the S1 mechanism execution and signal checks;
- independently passes S2 on V_B;
- passes the 5-epoch S3 trajectory from E25;
- retains gains against P-BCE-prestep budget control;
- passes cell guard and has no evidence that the result is merely an extra-step effect.

If eligible, run 2 datasets × 5 seeds × 50 epochs. Hyperparameter selection is light (consistency lr multiplier 0.1/0.3/1.0; inner steps fixed 1) and uses five-seed Validation aggregate. Freeze before Test. If CGPA does not exceed the matched P-BCE-prestep control and published baseline gates, do not claim algorithm gain.

M2/M3 results do not automatically become full training algorithms. A separate explicit design amendment and reviewer approval is required.
