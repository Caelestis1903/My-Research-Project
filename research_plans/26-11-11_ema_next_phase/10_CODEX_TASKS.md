# 10｜可直接交付 Codex 的任务清单

唯一入口：[00_EXECUTION_RUNBOOK.md](00_EXECUTION_RUNBOOK.md)。算法由04、12–14固定；门由09固定；Codex负责实现和执行这些定义。

## 1. 准备和冻结

- [ ] 创建源SHA隔离worktree，取得本规划的实际commit，记录source/plan/code/environment身份。
- [ ] 读取source `start.md`（GB18030编码）、本目录与三份main上下文；保留五seed、RAM-only和每卡准入约束。
- [ ] 复用自然数据cache、train-only预处理、源P_A/P_B/V_mech；校验row/split/domain/task/feature哈希。Test未打开。
- [ ] 将manifest展开为明确job/config/readout/comparison DAG；核算740核心与2490最大epoch上限，所有条件行有dependency和gate。
- [ ] 为所有候选best声明RAM consumers、A fork历史、C上一epoch权重、KD真实父F初值；输出状态/存储上界。

产物：SOURCE_RECEIPT、PREPARED、RESOLVED_PROTOCOL、DATA/PANEL/GROUP_MANIFEST、EXPANDED_JOBS、READOUT_SCHEDULE、RESOURCE_ESTIMATE。

## 2. 实现

- [ ] 新包适配source session；不改旧model/loss/optimizer代码。
- [ ] 实现A_MATCH与A_RAW，native→patch→F顺序、未用m仍更新、v/step保留、EMBED真实mask。
- [ ] 实现B稀疏Fbank、两态S/all、logit/prob及S-only；每个best同时pin实际父F。
- [ ] 实现C固定pair、PA拟合下一epoch应用、全局/六格收缩和同alpha参数控制。
- [ ] 实现冻结教师train软标签、归一化KD、fresh Adam、同预算LABEL与Fteacher、e0选点。
- [ ] 实现完整V评价、source metrics/严格最早最大、四类配对比较、09纯函数门。
- [ ] 实现单owner DAG、五seed barrier、条件后继、RAM事务/恢复/释放与预算计数。

## 3. 独立验收与工程smoke

- [ ] coder提交实现说明；由不同reviewer agent审阅公式、时钟、source一致性、Test隔离与state生命周期。
- [ ] 仅执行06列出的结论相关检查；无动作等价、A同状态对齐、B参数重心、C未来时钟、KD退化/对照、合法选点、单次恢复。
- [ ] 两数据真实batch GPU smoke，seed2022仅工程使用；测完整epoch/20读出峰值，并据此更新资源预留。
- [ ] critical问题修复后复核；绑定实施hash的PASS与smoke receipt齐全，再允许execute。

## 4. 自动科研执行

- [ ] 五seed父轨迹各50epoch，B/C所有曲线完整；完成后先冻结B配置供A附带读出。
- [ ] A两alpha×两个起点×五seed真实6epoch，读取E25/26、E40/41，自动计算G_A。
- [ ] 按门运行RAW/续训；保留共同前缀best；A不通过时转B，不追加复杂梯度网格。
- [ ] 用已完成父曲线裁定B/C，选一个教师或跳过KD。
- [ ] KD四候选、两LABEL控制都齐五seed后选λ/LR，再补Fteacher五seed控制。
- [ ] 每dataset选择一个Clean主候选，条件触发Gaussian继承评价；每个数据集单独裁定。
- [ ] 全局冻结前不调用Test；已触发任务完整，所有NOT_TRIGGERED附数值理由。

## 5. 最终评价与报告

- [ ] freeze绑定候选/控制、所有selected状态hash、发表模型端点、门输入、完整五seed覆盖。
- [ ] 只评价冻结的Test清单；正式baseline/内部控制分表；O/F复用须hash/协议一致。
- [ ] 给出六格AUC/LL、Macro/Traffic/Pooled、逐seed配对、CI、online/F、组件增量、简单替代和预算。
- [ ] A：真实反馈/online增量未转成F增量/起点边界；B：两态余量/1pass替代/蒸馏保留；C：差异/动作/未来价值/统一对照分别裁定。
- [ ] 区分AUC合格和LL代价；给ACCEPT/REVISE/STOP/NEXT及未触发/阻塞状态，负面结果完整归档。
- [ ] 写 `FINAL_REPORT.md`、`QUANTITATIVE_APPENDIX.md`、`DECISIONS.json`、`ENDPOINTS.csv.gz`、`PAIRED.csv.gz`、`COSTS.csv.gz`、`PROVENANCE.md`、`ARTIFACT_INDEX.json`。
- [ ] 校验数表与报告，再释放owned RAM，写RELEASE_RECEIPT；确认无模型/检查点文件。
- [ ] 实验代码/报告在本任务实验worktree分支交付；不自动覆盖main历史结果。提供实际commit与完成/未触发清单。

## 6. Codex不得自行决定的事项

不得改变主路线次序、源数据/采样、算法数学、两个A起点、H/scope候选、C时钟/动作集合、KD预算、五seed、正式阈值、LL裁定或Test冻结。依09的有限条件自动选择是允许且必要的；新增算法、更多HPO或另一个研究问题需要下一次研究计划，不能在本次队列临时发明。

数据路径、可用GPU、eval microbatch、并发和实现文件的合理组织属于工程适配；保持科学语义并记录实际身份即可。
