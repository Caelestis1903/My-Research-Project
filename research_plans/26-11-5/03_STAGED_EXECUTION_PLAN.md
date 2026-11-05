# 03 分阶段执行规划

本文件区分“方案完成”和“正式训练”。A/B/C 本次均已设计；实际按 B 优先、结果驱动执行，绝不将未运行路线写成失败。

## 阶段0：共同工程准备

读取07总入口、09工程合同、source index和manifest。准备固定源提交、main最新研究上下文、数据指纹；检查工作区，建立独立执行worktree，禁止覆盖用户未提交改动。实现共享loss/model/selection/state接口及B必须依赖的最小模块；A/C的公共接口可以一并准备，但不为通用性改写整个训练工程。

完成无训练性能含义的单元测试：源模型等价、实际2N分母、RNG隔离、参数布局、Adam反事实纯性、标准EMA退化、RAM恢复、候选五seed门和test冻结门。可以用合成数据与少量GPU batch检查资源，所有输出标ENGINEERING_ONLY，不按AUC淘汰方法。

产物：SOURCE_RECEIPT.json、RESOLVED_PROTOCOL.json、DATA_MANIFEST.json、IMPLEMENTATION_REPORT.md、REVIEW_REPORT.md、STAGE0_CHECKS.json。main上的设计文档不是“正式算法已经实现”的证据。

## 阶段1：B完整五seed及有限调参

每数据5条G-D0在线轨迹，每条维护15个非反馈影子：3个全局时间尺度、4个D、4个M、4个S。源在线训练和每个影子均运行完整50epoch，候选在θ0初始化，保留全部V曲线及RAM选点状态。

顺序：train → 五seed HPO selection → 冻结checkpoint/候选 → 用validation生成NEXT_ACTION → 统一test评价和正式报告。调度默认不读取本轮test。HPO所有候选都五seed，不能先用2022挑h/K后只补赢家。

B报告必须给出总收益、相对F组件增量、相对U/M/S的比较、匹配h/K消融、六格AUC/LL和训练/验证/探针成本。不同数据集分别得到决策，不强制共同赢家或共同超参数。

## 阶段2：预定义A/C继续

调用`continue --decision .../B/DECISION.json`，按04的有序规则只派发被触发的数据集和路线。

A触发：按照A1–A4实现和验收全部固定/自适应/全局对照。默认重放原G-D0在线轨迹，每数据5条，15影子完整有限搜索；不是读取B最终checkpoint直接补EMA。已有A从t0合法维护且完整评价时才可复用。

C触发：按照C1–C4实现和验收；先C1/C2各五seed，冻结validation选择，再C3五seed，再C4五seed；最后两个匹配控制各五seed。C改变online轨迹，不能共享为一个模型的多个shadow。γ0普通Gaussian对照经身份核验优先复用B/F。

被暂缓路线的数学、实现、搜索和命令文件仍保留，状态DEFERRED_NOT_RUN。不得删除其方案，也不得因本轮没有调度就称之为否定。

## 阶段3：收敛、剩余路线和可选组合

A若找到更简单的有效时间尺度方案，选择它；若没有足够增量且还有增强生成问题，则在相同数据集启动已经定义好的C。C后默认停止本轮并报告，不重新循环B→A→C或按test追加未定义网格。

B明显有价值且成本可接受时，可保留B并停止；这不迫使训练全部路线。B+C只有在独立结果提供明确理由且额外预算获准时启用；固定C教师不变、B仅作为另一个非反馈最终估计器，见C3。默认manifest关闭该组合。

任何额外候选或额外搜索都要创建新manifest版本并写理由，不让执行器临时展开。完成一个数据集的5seed结果可以保留局部结论，无需等另一数据集同样成功才承认。

## 机器可读状态

每route/dataset保留状态：DESIGN_COMPLETE → IMPLEMENTING → ENGINEERING_PASS → RUNNING → WAITING_FIVE_SEEDS → HPO_FROZEN → ENDPOINTS_FROZEN → EVALUATED → DECIDED。DEFERRED_NOT_RUN、ENGINEERING_BLOCKED、COST_BLOCKED独立记录，不等于性能阴性。

自动代理无法安全跨任务执行时，仍写完整DECISION.json和对应继续命令；下一次Codex只执行文件中已冻结动作，不再要求研究者重设计。

## 结果回填

正式运行完成后更新新研究报告、主指标台账的同协议分组及必要的AI上下文摘要；保留来源、seed、六格和总指标。本次规划发布不提前写新性能、不篡改历史结果。
