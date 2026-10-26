# DESIGN SPECIFICATION — ML5201丢失推理收据的有界补全

日期：2026-10-27。状态：DESIGN_COMPLETE，待coder实施及独立reviewer PASS。本文件只追加既有DESIGN_DELIVERY_OBSERVATION_PRESERVATION_ADDENDUM.md的特定context补测白名单；不修改任何既有设计、配置或冻结清单原字节。

原05§10要求逐arm成本。MovieLens seed5201唯一完整benchmark=True收据因摘要覆盖而丢失，原九方法补测范围会使该context另外七个已登记方法缺一seed的推理计时，包括最终仍需讨论的PILOT_PROB。这些是可补的收据缺项，不能与编译器parity拒绝混为一类。

**唯一范围追加：** 专用delivery观察授权token仅对`movielens:5201`允许从当前已冻结的全部16方法中补测真正缺失的条目；额外七方法固定为`LOCAL`、`ETA_DIRECT_MATCHED`、`PILOT_PROB`、`PILOT_LOGIT`、`FULL_SOURCE`、`FULL_TARGET`、`RANDOM_FEATURE`。其余七个worker维持原REQUIRED九方法白名单，并复用其已验证真实快照，不扩任何其他context或方法。

沿用原正常reload-readers固定typed delivery-only入口及身份/源码/独立PASS授权校验；只扩该token中唯一context的允许方法集合，不新增通用command、任意执行接口或宏化参数。每条目先查同身份有效MEASURED收据/快照，有则直接复用；因覆盖丢失或从未获得测量收据的条目仅补测一次。记录本次追加授权版本、方法、是否复用/补测、实际结果及原始收据path/SHA；重复同授权须幂等，不重复计时。

任何方法若已有明确的实际parity失败记录，保留原NOT_MEASURED及reason/error，不重试；其余context非REQUIRED的已知失败也完全保持。新补测如真实失败，记录NOT_MEASURED及原始原因，允许保留相应latency比较边界。原九REQUIRED方法仍是两数据集四种子的硬性计时交付验收；额外七条是一次有界补全义务，不把“尝试过”虚报成MEASURED。

原FrozenDeployment/compiler、4096自然原序raw-X、3次warmup+10次实际计时、GPU同步/传输范围及1e-6 parity阈值一律不变。不得改变模型、p/p_terminal、选择、lambda、RNG/optimizer、四种子决策或FROZEN_MANIFEST；不训练、不重启、不重做export/bootstrap，不为额外方法修复编译器或更换设备/精度以追求计时成功。

最低验收：独立review核token白名单差异仅`movielens:5201`上述七方法；有效已有收据复用、明确parity失败跳过、补测一次且重复幂等；其余worker方法集合不变。原观察器冻结状态前后指纹校验和测量收据保留机制继续执行。实际RUN后逐条公开复用/MEASURED/NOT_MEASURED结果，不能填造丢失的旧计时数字。
