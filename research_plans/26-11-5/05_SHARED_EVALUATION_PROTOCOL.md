# 05 五 seed 调参、评价与结果冻结

## 1. 强制基本规则

所有进入科学比较的配置使用 seeds `[2022,2023,2024,2025,2026]`，每 seed 完整 50 epoch。可以做 CPU 合成测试或少量 batch 工程 smoke；其 AUC 不得进入筛选、决定参数或否决方法。禁止 successive halving、单 seed 初筛、只补正向 seed、按早期 AUC 提前终止某候选。

每数据集独立选一组候选超参数，必须全五 seed 共用该组。不能为每个 seed 分别选择最优超参数再平均。两个数据集可以有不同较优值，但必须显式报告为数据集级调参。

## 2. 两层选择不能混淆

对配置 h、seed s：
\[
e^*_{h,s}=\min\arg\max_{e=1,...,50} V_{h,s,e}^{macro\ AUC}.
\]

首先是每个 seed 的 checkpoint 选择；精确并列取最早，不用 test/LL 改选。

随后计算配置级验证得分：
\[
S(h)=\frac15\sum_s V_{h,s,e^*_{h,s}}^{macro\ AUC}.
\]

调参用该均值，而不是平均五条曲线后选一个共同 epoch，也不是五个 seed 各选不同 h。所有候选都有完整五 seed 才能作最终排名。

### 预先冻结的选择算法

每个候选还计算相对同轨迹/同协议 F=Gaussian+原固定 EMA 的六格验证配对均差。优先在各格均差≥−.001的候选中最大化 S；若没有满足者，仍选 S 最大者但标 `VALIDATION_CELL_TRADEOFF`，不把整个路线自动归零。另保留 unconstrained_score_winner，避免保护筛选掩盖宏收益。精确同分以实测运行成本更低者优先，再以较小改动（|log2半衰期倍率|/|γ|）优先，最后按 candidate_id 字典序。浮点排序只使用完整精度；显示用舍入值不得参与判断。

该选择规则不要求相对 F 达到 .003 才能被选中。低收益是结果，不是拒绝完成搜索的理由。

## 3. 训练探针和 validation 的分工

A/C 控制器只读 train；不能读取 validation/test 来更新系数、协方差或梯度。validation 只用于 checkpoint、有限候选选型和预定义阶段调度。B 仅使用当前训练 batch。训练探针允许同一训练样本后来正常参加训练，但不改变原训练人口或每步监督权重。

所有训练中 test loader 默认禁用。阶段搜索结束、五 seed 齐全、候选和 checkpoint hash 已冻结、`selection_frozen.json` 写出后，才可评价约定的 test 端点。test 不参与选超参数、重选 checkpoint 或触发追加搜索。

C 的四候选顺序搜索每一步都等待当前候选五 seed 完成；三次选择读取 validation CSV。中间没有 test 访问。

## 4. 报告必须保留的信息

每 seed 每 endpoint：dataset、candidate_id、family、配置完整 JSON/hash、source/data/split/preprocessing hash、seed、trajectory_id、estimator、selected epoch、model hash、六格 AUC/未经缩放 LogLoss、Macro、Traffic、Pooled、训练/验证/探针耗时、GPU peak、RAM peak。

Macro=六格等权。Traffic=按真实评价人口加权的六格指标；Pooled AUC=每任务跨域合并计算后对两个任务平均。三者不能都简称总 AUC。selected 主报告，E10/E50 为共同阶段描述；同一轨迹的 online/EMA/E10/E50 都不是额外 seed。

输出至少：
- `VALIDATION_CURVES.csv`：全部候选五 seed 50 epoch；
- `HPO_SUMMARY.csv`：每候选五 seed均值、每格差、成本、完整性；
- `HPO_SELECTION.json`：每路线/数据集选择、规则、候选集合和输入文件hash；
- `selection_frozen.json`：被报告估计器的 checkpoint 身份；
- `ENDPOINTS.csv`、`PAIRED_SEED.csv`、`PAIRED_SUMMARY.csv`；
- `DECISION.json`、`REPORT.md`、`COSTS.csv`、`RUN_MANIFEST.json`。

## 5. 正式模型比较与组件比较

正式 baseline 为已有发表模型的工程适配：M3oE fixed1、HyperGate、AutoIFS。使用源 A+B 报告的逐 seed 数表且逐项核对统一人口、预处理与 AUC 选点；不据旧 LogLoss 选点绝对值排名。HyperGate 的部分参数仍有 UNVERIFIED_PROPOSAL 标记，继承披露，不能冒称作者严格复现。

必须分别列：完整候选 vs 每个发表模型，以及完整候选 vs F 的组件增量。以当前台账最强发表模型作为冻结主参考：ML M3、KR HyperGate；同时报告三模型，不能在 test 后改换一个较弱对象冒充主胜出。

五 seed 配对均差 ΔMacro AUC≥.003 为合格，六格均差各≥−.001单独判断。保留最新报告中的 .002 候选等级和 >0 正向胜出描述；它们不替代 .003 合格线。LogLoss 以五 seed 均差<0 独立判断，AUC 不因 LL 未改善失去资格。无需全部 seed 同向，不新增统计显著性门。

## 6. 比较机制信息

B 要同时报告 D* vs F、U*、M*、S*（各家族同预算独立调参），以及 D* 与同一 (h,K) 的 M/S 匹配差。不能只选对 D 最有利的控制配置。A 同理比较分组自适应与全局自适应、固定分组和全局固定。C 的两个机制对照使用 C* 的 γ/K，标“匹配参数消融”，不称各对照全局最优。

## 7. test 历史外露与研究阶段

当前 test 已在历史研究中使用；本轮不是新外部独立确认。默认下一路线调度仅用 validation 加已归档现象。若人为依据本轮 test 改变路线或预算，必须在 `decision_basis.test_consulted=true` 记录使用的结果、时间和理由。随后同 test 结果只能称适应性研究比较，不能写成完全独立确认。

原结果只读；未运行候选标 NOT_RUN，1–4 seed 标 INCOMPLETE，工程/成本异常分别标 ENGINEERING_BLOCKED/COST_BLOCKED，不写性能失败。
