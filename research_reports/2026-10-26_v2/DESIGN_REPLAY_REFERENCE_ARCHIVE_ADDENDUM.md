# DESIGN SPECIFICATION — 旧两种子重放诊断引用归档

日期：2026-10-26。状态：DESIGN_COMPLETE，待独立reviewer核准操作。仅处理额外技术一致性门槛，不改变算法、不作效果结论、不修改任何原始指标。

## 1. 问题与准许范围

恢复版Campaign._remember新增了“旧v1整个metrics字典必须Python精确相等”的训练阻断条件。原四种子恢复设计要求native selected、epoch24、pilot身份hash核对，不要求所有新训练小头的所有浮点汇总都逐位相等；旧v1两种子结论及gate早已SUPERSEDED。这个额外条件现使S1进入DRAINING_AFTER_FAILURE，但模型仍在RAM。

已存在的纯数据审计 `v2_four_seed/replay_comparisons/movielens_5202_NUMERIC_DIFFERENCES.json` 记录若干汇总差异。不得把它们全部断言为机器舍入：ETA_DIRECT_MATCHED最大单cell差约6.18e-5，应如实保留。旧v1只有指标汇总，没有完整可用旧p，不能声称已经比较旧新逐行预测或证明全部旧小头逐位一致。新v2使用完整四种子和统一的已审查实现；不把旧v1指标混进四种子矩阵。

允许将旧summary目录作为完整历史引用**原子改名归档**，并记录理由，使已加载controller的这个额外exact比较不再阻止新四种子矩阵。此操作不是删除失败证据：所有旧文件原字节、目录清单和SHA保留，所有既有EXACT失败收据保留；旧新差异继续独立审计。新四种子实用阈值、原生hash门槛、P_cal选择规则及新模型raw-X≤1e-6验收完全不变。不得修改旧summary内容来迁就新结果，不得把新指标拷回旧路径，不得以本操作绕过真正的实现/协议错误。

## 2. 恢复前最低技术条件

1. 等当前scheduler完成drain，所有自有worker没有pending；记录service/worker PID及存活状态，禁止重启、重训、关闭模型或向busy pipe发送第二命令。
2. 原5201/5202×两数据集的native selected、epoch24和pilot hash对旧v1均严格相等。检查新四种子全部角色、训练参数、数据/分割、选点及方法清单仍是已审查版本；数据mmap前后dtype/shape/行序/值hash验收仍有效。哈希或真实协议不一致不能按本操作放行。
3. 对旧四context，把新已完成方法的Q预测从允许的row-data cache按row/domain/y复算汇总，与旧summary逐项记录有符号差及绝对差。记录可取得的选择candidate/pass/lambda/off与P_cal依据；旧字段缺失记MISSING_OLD，不猜造相等。只做技术重放差异审计，不提前写效果解释。
4. 区分三类身份：原生/pilot严格身份已核对；小头是按同一登记算法重新训练、结果可数值不同；新v2四种子全由同一实现完成且内部比较合法。不把这三类压成“完全相同训练状态”。对差异可能来源只列已证实事实；尚未定位的底层浮点路径不写成已确证原因。
5. 若存在新实现错误的明确路径（数据行/标签错位、改变目标/梯度/归一化/模型输入、错误选择、实际参数不同、非有限数、错误缓存覆盖），停止对应路径交coder真实修复与独立review。仅历史浮点字典不同、而上述协议/身份检查通过，不是再次重训或改算法的充分理由。

此处不要求把两种子旧gate重新认证为有效；它已被用户四种子要求废止。主线实用gate只由新四种子完成结果计算。

## 3. 原子归档与审查收据

由主编排者执行明确、可回查的数据文件操作：

- 源目录固定为 `artifacts/mdmt_rcf_20261026/v1/summaries`，目标固定为同一v1根内 `summaries_historical_two_seed`。目标必须不存在；不递归覆盖，不操作其他目录。
- 操作前写迁移清单，列全部相对文件名、字节数、SHA256、原位置、目标位置、SUPERSEDED_TWO_SEED_PROVISIONAL状态、触发的额外exact门槛和批准本设计的review收据。记录旧EXACT失败文件路径及其SHA。
- 同文件系统一次rename，保持文件原字节；操作后重新逐文件核SHA及文件清单完全相等，写迁移完成收据。原路径不再放summary文件；在迁移收据和新ARTIFACT_INDEX中明确指向真实归档位置。若需要路径说明，放v1根的独立说明文件，不能伪造一个空/相等summary。
- 归档不代表PASS：旧exact audit原失败状态保留，新增摘要写 `HISTORICAL_METRICS_NOT_BITWISE_IDENTICAL / DIAGNOSTIC_ONLY`，明确原生hash验证仍PASS，旧小头数值差全部可查。

当前cached controller的 `_remember` 在旧path不存在时按原代码跳过该额外比较，不需要第二次代码热更新、任意注入或重启。独立reviewer须确认这是对已废止历史引用的明确归档，而非变更当前新v2数据、四种子判据或训练模型。

## 4. 幂等恢复和最终materiality检查

归档后正常重发S1；先核worker内全部13个已完成S1方法及训练完成收据。已有methods只汇总，不重训，不增加训练/label-access计数；确实未执行的方法才按原流程执行。记录重发前后method清单、模型hash、训练完成时间/receipt身份，证明不是把重训藏在“恢复”。新 `_remember` 产生的四种子summary必须来自各自当前worker结果，不能从旧summary填充。

S2/S3只按新四种子gate及已登记必跑KR控制继续。四种子齐备后、正式结论和freeze前，对任何涉及旧seed差异的方法做最小决策敏感性记录：是否涉及实际晋级比较、选点规则是否保持、旧新变化是否足以越过已登记阈值/保护。若差异可能改变某项历史可比性，明确该项历史精确重放主张不成立或敏感，不能改指标/阈值/候选或挑seed使其一致。

新v2四种子矩阵的有效性由其统一协议、真实四seed训练、合法P_cal选点、raw-X与导出复算决定，不由旧两种子字典是否相等决定。若敏感性检查揭示实际新协议错误则必须修复；若只是历史浮点结果不同，保留差异与边界，不为追求位级等同无限重训。任何最终效果报告都只在相关方法四seed完成后由主编排者撰写。
