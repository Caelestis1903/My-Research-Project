# 实现、独立审查与修订历史

实现遵循DESIGN→IMPLEMENT→REVIEW→RUN→ANALYZE。研究算法由coder按设计实现，另一个reviewer独立审查，机械运行交runner，科学结论由主编排者形成。工程PASS仅表示相应规范的实现验收，不代表科研假设成立。效果分析前已核每个实际比较条件四个独立训练seed；136条主线方法完成收据与8条独立SCALE训练均保留。

## 已落实的科学语义

- fixed-fusion M3oE的h为四个共享专家归一化后拼接，B输入[h,z]；自然Sample BCE训练与六格Macro评价分开。
- 两折B_OOF只产生需求代理；A与后继已使用P_form标签，不称全链OOF。需求在真实target群体计算，消去bias/temperature响应，不把target需求外推为source真值。
- δ隐藏复制source pilot并加入配对微扰，末层精确零；target梯度只进后继。native、owner、B、原pilot及normalizer冻结，最终reader只读δ。
- 每槽query、重要权、辅助source行计划与空query步骤按规范执行；R约束所有foreign支持。训练末做全量可行检查及确定性缩放，而非把dual当保证。
- POOLED独立使用完整合格foreign域；LOCAL_ACTIVE有充分活跃容量且无source anchor。K1_WIDE重新训练、重新读取，不用简单删槽代替必要性控制。
- reader在P_read拟合，P_cal仅选择完整六头政策，包含精确B_refit OFF。FULL_SOURCE_ALL明确额外使用P_form；FULL系数无逐维box，其他系数按规范有界。
- 冻结后统一打开test_exposed/legacy，未据外层结果改seed、预算、群体或S4决定。

原始梯度、权限、None/Adam状态、query权重、约束缩放、连续两步与内存resume下一步、reader梯度/政策选择及source/profile检查见独立收据：`S0_IMPLEMENTATION_REVIEW.md`、`S1_C_IMPLEMENTATION_REVIEW.md`、`FORMATION_IMPLEMENTATION_REVIEW.md`、`RECOVERY_IMPLEMENTATION_REVIEW.md`。仅执行能改变当前训练或科学判断的最少验证。

## 必须保留的修订与异常

| 问题与重要性 | 实际处理 | 结论边界 |
|---|---|---|
| 功能干预从饱和概率反推logit会失真，Critical | coder改用原始logit，reviewer独立反例与复审PASS后才正式形成 | 不靠概率clip掩盖数值问题 |
| FULL原L-BFGS在真实fixture未收敛，Critical | 精确Hessian预条件变换，保持目标、无界系数和容差；实际fixture对比复审PASS | ftol成功不等于所有原坐标KKT均≤1e-8，实际残差保留 |
| 首服务意外消失，RAM状态丢失 | 同身份重放；旧预测/时间/hash保留 | 重放不是新增独立seed，不能从日志声称恢复Adam |
| 旧四worker触发主机OOM，内核杀KR5202 | 保留原运行证据；终止本任务受影响残余worker，新四seed配置将明确白名单逐行数据mmap外移，保留主机余量 | 未缩小physical batch，模型/Adam/RNG不落盘 |
| 用户增加至少四训练seed要求 | designer补充设计、coder实现四seed门槛、reviewer独立复审，再运行5201–5204 | 原两seed推论标SUPERSEDED，不进入新结论 |
| 旧小头不全位级复现，Relevant | native selected/epoch24/pilot原四上下文精确匹配；保留小头差值与旧ETA缺失选点；旧summary原字节/SHA归档解除额外整字典guard | 未确证唯一根因；BLAS/线程环境是候选解释，未选择性只重跑负结果 |
| 阶段调度串行 | 版本化正常RPC安装已审查并发派发器，逐卡85%及主机余量守卫，阶段barrier和错误drain保留 | 模型/政策hash不变；并发墙时不是独占GPU速度 |
| NATIVE terminal原记录误重复selected | 观察器接实际保存的epoch24 logits，selected p与政策不变 | 真实终态另报，不伪造权重或前向 |
| ML5201 FULL_ALL raw预测差2.1977e-6>1e-6，Critical交付问题 | 独立设计/实现/审查，只重现原完整role、原设备、40960或16384分块及FP32舍入顺序；136收据、全部RAM状态、冻结清单不变 | 不放宽1e-6，不改训练或所选p；真实各prefix验收另见最终收据 |
| 观察器marshal函数身份不稳定 | CODE_SEMANTIC_V1比较实际bytecode、参数、常量等，排除纯路径/表示差异；独立正反例复审PASS | 不是允许任意旧代码，固定源SHA与typed RPC守卫仍保留 |

冻结训练清单SHA为`09def5e2f43786f9c2875e5ae466845a10ec7db47a3c6edd28379827b72e927a`。prefix、各训练阶段和后加观察器分别记录自身源码身份；当前磁盘源码并不冒称所有早期运行的加载版本。`PRE_RAW_OBSERVER_REPAIR_IDENTITY.json`和`RAW_OBSERVER_INSTALL_ACCEPTANCE.json`记录136完成收据、模型/Adam/RNG/模式/normalizer/政策前后保持一致。

## 实际八context交付验收

S0/S1/S2及触发S3全部实际条件均5201–5204四seed完成，S4两数据按冻结Q门槛NO_START，独立SCALE八条训练/外层复算PASS。八prefix所有136个完整raw-X预测器实际PASS、最大概率误差均0；5328个原始预测文件与指标重算PASS。CORE核算有29376指标行、44064配对汇总；SUPPORT有7776分层指标行，原始KR user/item真实缺失不伪造。跨角色功能profile以相同登记准则核算，并保留ML test唯一四seed交叉与方向反转。

原summary以benchmark=False覆盖True计时记录是实际交付缺陷。独立设计/实现/审查PASS后，normal reload-readers仅导入七份经SHA/政策/formation/row/env/recipe复核的原快照，另对ML5201遗失的16个已有方法各执行一次原3warmup+10/4096的计时。没有重训练、重做export/bootstrap或更改预测；全部八份preservation receipt为COMPLETE，完整模型/终态/Adam/RNG/梯度/模式/normalizer/政策/p前后不变。

现有135/136方法实例MEASURED，72项必需实例齐备；KR5201 RANDOM_FEATURE融合parity5.7095e-6超过1e-6，保留原NOT_MEASURED且不重试。其原模型完整raw-X输出实际PASS，不能将融合计时拒绝混成性能输出失败。非关键compiler缺项仅限制相应latency比较，不增网格或放宽容差。

审查链为DESIGN_DELIVERY_OBSERVATION_PRESERVATION_ADDENDUM、CODER_IMPLEMENTATION_REPORT_DELIVERY_PRESERVATION、DELIVERY_PRESERVATION_IMPLEMENTATION_REVIEW，以及ML5201补全的独立design/coder/review三件套。最终七快照复核见DELIVERY_SNAPSHOT_FINAL_INDEPENDENT_AUDIT.json。原collector加载源码9795…与期间动态磁盘SHA分开记录，数值compiler/benchmark AST原样；当前观察器source617f1841bf2f4b4208b82000d62779fc3b10ea2e55347501ab3d0ce8419d178b，未倒写为早期训练来源。

所有新增代码在`experiments/mdmt_rcf_20261026/`，报告在本目录，数据/日志在`artifacts/mdmt_rcf_20261026/`。无模型或checkpoint落盘。清理前最后一轮CLI要求的全量SHA/指标复核已实际8/8 PASS，再次重算5328预测文件。随后实际仅删除当前v2缓存manifest拥有的17591文件/101602861308字节，八worker与service均退出、RAM模型释放；raw预测、φ、δ、需求、选择、预算、日志、代码与hash完整保留，旧v1未删除。本轮artifact/code/report范围没有模型/checkpoint文件。实际收据为本目录EXPORT_ACCEPTANCE.json、CLEANUP_RECEIPT.json，完整原始收据由路径/SHA索引。
