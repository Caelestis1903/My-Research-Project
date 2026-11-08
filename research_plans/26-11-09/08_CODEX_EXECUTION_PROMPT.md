# 08｜可独立交给 Codex 的执行提示词

你正在执行 My-Research-Project 仓库中的 MDMT 自适应 EMA 研究。先读取本目录 00–07、09 和 experiment_manifest.json，再严格按阶段门执行。**本提示词可脱离当前ChatGPT会话独立使用。**

## 一、任务范围

目标：检验参数更新收益的时间持续性，并比较：
A. 全局共享参数收益驱动EMA；
B. task0/task1两份共享参数历史；
C. 单一共享历史加task条件化插值。

固定EMA已有收益，不得重新验证或否定。不得重跑11/05 A/B/C已有方案。不得以域样本规模作为难度、优化难度或EMA需求代理，也不得围绕此关系设计实验。不要扩展到全部12参数组；本轮只研究S。

本任务是工程实现和后续实验执行，不是重新构思方向。不得修改历史训练代码、历史报告或main总上下文。

## 二、开始前必须完成

1. 确认仓库、main、26-11-07分支head、merge-base和ahead/behind。
2. 逐项读取main总上下文、26-11-07的11/07机制报告与源代码、11/05增强历史报告和main的11/06规划。
3. 核验所有源文件真实存在并记录blob SHA；不得假定26-11-07已合并main。
4. 检查11/07原始逐窗数据与预测文件是否存在、非空、hash匹配；不能用报告摘要伪造逐窗数据。
5. 输出SOURCE_AUDIT.md、AUDIT_INVENTORY.json、EVIDENCE_REUSE.csv。

## 三、阶段A0：零GPU离线审计

重算Fast−F、Slow−F相邻pulse窗口收益序列；评估lag-1符号持续性、相关、e→e+5前瞻预测、P_A/P_B与V_A/V_B迁移，并与C_design、C_simple、M0/M2/M3和固定身份/阶段参照比较。seed为唯一统计单位。明确区分当前收益、未来收益、预测能力、动作价值和完整模型收益。

根据06的A0门输出唯一状态：
- PERSISTENCE_SUPPORTED；
- STATIC_PREFERENCE_ONLY；
- NO_ACTIONABLE_SIGNAL；
- SOURCE_INCOMPLETE。

若NO_ACTIONABLE_SIGNAL，停止新增训练并报告固定F为当前简单选择；不得为了继续任务而制造新控制指标。若STATIC_PREFERENCE_ONLY，只设计静态任务偏好对照，不称为动态C_persist。若SOURCE_INCOMPLETE，先恢复/重算归档数据，不得当作阴性。

## 四、阶段A1：新seed前瞻性验证

仅在A0触发后运行KuaiRand Clean/Gaussian D0，seeds 2027–2031，每条50epoch。保持11/07 M3oE fixed1、自然人口、预处理、Adam、Gaussian与数据顺序完全一致。A1只验证信号迁移，不使用Test选动作。

所有方案共享同一online Adam轨迹，parallel shadow：
- online与固定F；
- A全局S状态；
- B task0/task1两份S状态；
- C单一F_S与task λ；
- 每epoch one-epoch Fast/Slow pulse observer；
- 统一Fast/Fixed/Slow动作参照。

controller只用P_A训练面板过去收益；P_B/V_A/V_B只审计迁移。epoch e观测只能决定e+1动作。不得改变Adam、loss、loader、Gaussian RNG或训练样本。

## 五、阶段B闭环比较

严格按03数学定义实现A/B/C。必须同时实现固定F、统一Fast/Slow和简单即时收益参照。所有动作记录决策时间、信号来源、信号年龄、rho/λ、P_A效用、P_B/V_A/B审计效用。A/B/C分别可开关，不能互相覆盖。

B路由必须使用真实task ID，验证同一S坐标按task读取不同历史；显式报告这改变了共享参数原始语义。C的λ为凸插值，只能增加当前online权重，不得宣称等价B或能实现Slow。

## 六、冻结、评价和Test

Validation每epoch六格Macro AUC，最早严格最大值选checkpoint。Test只在算法和超参冻结后一次性评价。报告六格、Macro、Traffic、Pooled AUC/LL及每seed配对增量。固定F为内部对照；发表模型baseline仅使用已发表M3oE/HyperGate/AutoIFS，ML/KR最强对照遵循同协议当前台账。

五seed规则、AUC +0.003强收益、+0.002候选收益、逐格平均下降不超过0.001、LogLoss独立评价均按06执行。不得用单seed或所有seed同向作为否决门。

## 七、工程交付

实现路径和文件结构见05。必须完成单元测试、数值一致性、Adam轨迹不变、task路由、state恢复、panel无泄漏、Test冻结和资源检查。先完成CPU/单batch smoke与review；review通过后才允许A1训练。输出实现审查、manifest解析、资源预算、训练/控制器日志、自动分析脚本、最终报告与释放收据。

## 八、停止与失败处理

- A0无持续性且无静态偏好：停止。
- A1不能迁移：不进入动态B；可保留固定F或静态候选。
- A/B/C都不超过F：停止自适应控制路线。
- 工程/资源失败：分别记录，不得包装为科学阴性。
- 不增加未预注册的指标、参数组、数据集或seed筛选；若任何偏离会改变研究决策，先更新计划与manifest并保留版本记录。

**第一条命令必须是audit，不得直接train。**
