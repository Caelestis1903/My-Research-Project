# 01｜当前研究依据与停止重复的事项

返回 [执行入口](00_EXECUTION_RUNBOOK.md)。证据版本与逐项来源见 [15](15_SOURCE_EVIDENCE.md)；所有新增算法结果由后续实验产生。

## 1. 当前结论

**现有证据已经足以进入新算法判别。** 下一步最有价值的动作，是检验共享 EMBED 的当前方向能否产生真实反馈增量，以及历史预测的非线性组合优势能否用少数状态表达。继续扩展原 fast/slow rho 控制器的优先级低于这两个问题。

最新归因分支 SHA 为 `5ef45b18b973ed11be6fbb5f654562349998de60`；本次 main 读取基准为 `5cf92af0d3a8ed77688762616248166b9b68266d`。最新报告的“下一方向 C / 备选 D”在本轮任务命名中分别对应 **路线 A / 路线 B**，不能按旧字母导入算法。

## 2. 已接受事实

| 五 seed，独立 selected Test | F−O Macro AUC | F−O Macro LL |
|---|---:|---:|
| MovieLens Clean | +0.0038488449 | +0.001248644 |
| KuaiRand Clean | +0.0046163994 | −0.001225342 |
| MovieLens Gaussian D0 | +0.0038568018 | +0.005334282 |
| KuaiRand Gaussian D0 | +0.0077669275 | +0.000327687 |

来源：[最新最终报告 §9](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。历史与本轮 O/F 端点复用相同训练身份，不合并为十 seed。LL 正值是代价；ML Gaussian 五 seed LL 均恶化，不能用排序提升掩去。

共享修正 S 是主要来源；EMBED 在四条件中有稳定重要贡献；STAR/Skip 在 KR 也重要；N 的独立贡献小或不稳定。EMBED recent×old 在两种参数背景、四条件均表现正交互，较早历史单独为负并不等于删除它有益。完整 h/fused 切口和精确 pair 纠正−破坏净量支持共享表示到排序变化的因果链。细融合项与路径积分的解释边界，保留最新报告审阅的限定，不影响主结论。

固定面板 late 的历史 logit 组合相对 F AUC 增量分别为 ML-C +0.0007681、KR-C +0.0021033、ML-G +0.0007821、KR-G +0.0026413；对应 LL 均下降。它是每 optimizer step、同 rho、同初始化质量的预测 EMA；尚无独立 selected 全 Test 结论。该面板证据足以安排 B 的完整可执行读出，不必再重跑全 step q 证明相同事实。

Clean、EMBED、E20/E35 的同状态同范数 current-preconditioned，在 V_B 平均 AUC 为 ML +0.0003336、KR +0.0003559；移除二阶预调节约为 −0.0123103、−0.0112120。保留 v、明确 WD 和实际步幅，是 A 的直接依据；该实验没有反馈到下一步训练。

## 3. 历史失败按实际环节区分

| 实验 | 实际检验及结果 | 当前归类 | 新方案的实质区别 |
|---|---|---|---|
| 11/05 B 增强响应方向 EMA | D−F selected Test AUC：ML +1.04e−7，KR −4.946e−6；方向被启用 | 有动作，但函数增量极小；缺旧 Q/innovation 投影状态，无法补算覆盖率 | A 改真实 online 更新；B 改参数均值之外的预测表达 |
| 11/05 A 分组/全局 rho | ML GROUP/GLOBAL−F −0.000750/−0.000728；KR −0.000175/−0.000213 | 参数读出确实变，早期优势未转 selected 收益 | 不再扩张原分组半衰期网格 |
| 11/05 C 漂移条件增强 | ML merged−F −0.000667190；任务版−旋转 +0.000018904；KR未运行 | 真实反馈已做，当前方向机制增量不足；KR仍为未执行 | 当前主路线不改增强协方差，不把未跑当阴性 |
| 11/07 pulse/预测 | KR Gaussian 逐格时间尺度有局部交叉空间；复杂预测器相对简单规则增量弱 | 局部动作及未来预测，未形成持久闭环训练 | C 直接学习同一有限预测动作的训练排序风险，不预测旧 rho 标签 |
| 11/09 A0 | C_persist 部分误差下降，动作却与 C_design 相同；30条 task 静态政策全选 Fast | 信号/动作/价值分离；A1及闭环B/C未触发 | C 必须报告系数差、实际排序差和优于统一策略的增量 |
| 11/09 PUB | 三个迁移五 seed 均明显低于普通 Adam；solver 正常 | 真正性能失败；原始系数尺度、coupled WD 和更新半径共同作用 | A 在本分支本步匹配 native Adam 的实际范数，不接着修 PUB |
| 11/11 新桥接 | ML有限菜单 oracle 空间小；KR机制 oracle约+0.0012，训练可见规则未利用；rho拟合规则全选 Fast | ML空间小；KR有空间但当前信号不可操作 | 不把全部失败归因指标不准确，也不把 KR 写成没有空间 |

来源报告与实现列在 [15](15_SOURCE_EVIDENCE.md)。`C_design/C_simple` 的 all/fast 退化范围不能扩展到 carry/direct；最新 [术语修正](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/research_reports/2026-11-10_ema_causal_attribution/FINDINGS_CLARIFICATIONS.md) 已精确区分。

## 4. 正式发表模型与内部控制

继承 main [指标台账](../../MDMT_ALGORITHM_METRICS_LEDGER.md) 中 **11/04 FULLTRAIN_NATURAL80_10_10 / AUC_SELECTED_V1** 五 seed 结果；不使用更早按 LL 选点的数据排序。

| 数据集 | M3oE fixed1 online AUC/LL | HyperGate AUC/LL | AutoIFS AUC/LL | 最强发表模型（AUC） |
|---|---|---|---|---|
| ML | .775190 / .494891 | .771412 / .498554 | .772425 / .496094 | M3oE |
| KR | .671907 / .437172 | .682113 / .440590 | .678597 / .432992 | HyperGate |

三者是已发表架构的仓库工程适配，各自保留训练配置；HyperGate部分超参有 UNVERIFIED_PROPOSAL 标签，AutoIFS含 search/reuse 阶段。这些标签原样保留。新表要列每个发表模型，主性能门相对固定的上述最强者计算；不按每 seed 挑不同最强者形成 oracle。

KR Clean F=.676523，距 HyperGate 约−.005590；因此“对 F 有增量”和“超过最强发表模型 +.003”是两个真实不同的目标。不得把面板 q 差额加到 selected Test 上伪造预期性能。

## 5. 已有资产能复用什么

| 资产 | 用途 | 本轮动作 |
|---|---|---|
| 最新七份报告/数表、1,604条汇总、46个来源索引 | 接受既有结论，设定动作和门 | 直接复用；本次复算所有行的五 seed 均值/SD/t4区间，无差异 |
| 统一发表模型逐 seed 六格端点 | 正式外部比较 | 校验身份后复用，不重训三套基线 |
| 源模型、registry、Adam/EMA、数据/评价、guardian | 新包实现底座 | 固定 SHA，新增 adapter，不修改旧实现 |
| 数据/P_A/P_B/V_mech 索引与 hash | 同人口、同 pair 的必要观察 | 原 cache 存在则复用，缺失按冻结构建器重建 |
| 原 online/Adam/F/历史/selected 参数 | 不能由 CSV恢复；已 RELEASE | 重建本轮必要公共父轨迹，并在 RAM 生命周期内完成消费者 |
| 旧全 step q 数组 | 固定面板机制结论 | 读取已有数表；不充当新样本可调用模型 |

本次 [SOURCE_AUDIT.json](SOURCE_AUDIT.json) 复算的是已保存 seed 汇总，不声称重新训练或重新生成所有原始干预。数值审计无须再次占用 GPU。

## 6. 研究边界和决策

小域不是困难域，直接继承，不提出相关假设、分析或实验。域 ID 与 source 数据映射一致；不以样本数决定 alpha、rho、难度或噪声强度。当前 Gaussian 真正目标为 D0；嵌套 source 元数据旧 `augmented_domain=2` 仅为来源记录，不得改掉实际 target_domain=0。

本轮不重新论证固定 EMA有效性、共享来源、recent×old互补、年龄匹配延迟；不重跑旧控制器网格；不开展 PUB WD=0修复矩阵；不因为类似方法已有论文而排除 B。必须解决的是 **A的真实反馈、B的可部署表达与C的信息增量**；跨更多 backbone、神经元级完整分解和无限历史最优集成保留为后续问题。
