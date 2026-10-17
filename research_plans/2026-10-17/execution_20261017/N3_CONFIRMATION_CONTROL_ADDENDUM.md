# N3 确认与精度补充的机械清单细则

DESIGN COMPLETE。只将已经冻结的 N3_DESIGN_ADDENDUM 转成执行清单，不增加实验或改变候选规则。主 agent 完整收集 16 个开发锚后调用一次 `n17_local_readout.py --nominate`；之后候选文件 SHA 不变。

`n17_local_control.py prepare` 接受该文件和明确的三种动作：confirmation（2036–2039、选定阶段、双路径、32draw）、development_supplement（2032–2035，同候选，同阶段，draw32–63）、confirmation_supplement（2036–2039，同候选同阶段，draw32–63）。只计算已冻结子块；与原始 drawstream 定义一致，补充跳过前32而非新造/重复计数。开发补充须原冻结候选 need64=true；确认补充须确认 gate need64=true 且 candidate SHA 一致。若无定义候选，记录 NO_DEFINED_CANDIDATE，不造确认矩阵。补充不调用 nominate，不重新筛选。

清单只引用已完成 N1 checkpoint 及其原优化器/LR；保留路径、seed、阶段及 checkpoint SHA。生成清单不自动启动，也不更改正在运行的 source/queue。只有一个有 PENDING 的调度器可准入；额外清单须等待原队列 PENDING=0，或仅暂停原 scheduler 的准入、健康子进程继续，待新增清单的**全部预期任务**已准入且 PID 可见后恢复。不存在的 status 文件不等于 PENDING=0。

最小结果核验：正式非engineering配置、receipt COMPLETE、配置/源 SHA、锚点身份与未变标记、恰好预期draw区间、每draw两个q的行ID记录、每候选两macro/12cell读数、有限 AUC/LL/J/更新范数。方向未定义时允许只有full并单独记录；不以补零造norm/direction。该核验只检查实际影响比较的缺失/重复/身份错误。

审查：主 agent 将脚本逐项对照本文件及原 N3，语法和实际产物字段核对，不提前读取开发子集提名。`independent_review=false`。运行前须另记录 PASS_SELF_REVIEW 及冻结源 SHA。
