# R1 Charter Change Log — v1.1 → v1.2

日期2026-10-11（Asia/Shanghai）。范围仅R1 Limited Revision，依据 `promt/r1_xiuding.md`。独立Reviewer最终 **PASS（设计合同）**，P1/P2/P3全部 **CLOSED**，见 [最终复核](R1_to_R2_Independent_Review.md) §8；执行准备度 **NOT READY**。本文件记录设计变化，没有实现/训练/性能评估/实际checkpoint搜索。原v1.1、Architecture Audit、Baseline Protocol和旧设计报告全部保留。

## 1. 保持不变

核心AO-04问题；B0 canonical S1设备安全适配器/正确两optimizer分组；D3/T2/N4真实图、global四scalar及原初值1；KuaiRand natural域/任务定义、字节hash、80/10/10 split-first、train-only encoder/OOV；Sample样本×task mean BCE；embedding16/FCN512,256,256,64/batch40960；完整50epoch和shadow patience10；原main Adam .01、arch Adam .001、wd1e-5/betas/eps及原更新节律；新q主Adam规则；2022/2023配对；validation earliest macro max选模；test最终评价；旧guards、参数/状态/RNG/log要求与最多10完整attempt/500epochs、精确resume边界均继承。不得声称改baseline来提高方法。

## 2. 逐项修订与理由

| 项 | v1.1 | v1.2 | 理由与解释限制 |
|---|---|---|---|
| 正式对照 | Bcap共享gate温度 | B4逐格delta，b_dt=sigmoid(raw_bD+delta_dt)，other内部均匀 | 更直接检验own/other调节竞争解释；不声称亲和被识别 |
| 新参数 | Bcap6theta主Adam | B4恰6delta、初值0、同B1主Adam，原raw_bD arch不变 | 初始前向与B0等价、参数数量明确；不改变原scalar协议 |
| 可辨识性 | Bcap温度可吸收进Linear | B4 raw与delta共同shift冗余，最多新增5个b向量可辨识方向 | 6trainable不等6函数类自由度；记录delta均值/中心化与有效b，限制机制说法 |
| Bcap身份 | 正式五臂之一 | 仅以后可选研究，不进入本预算 | 替换而非叠加，不扩大方法数 |
| 训练次序 | 五臂2seed全部完成再筛查 | A固定B0/B1/B2×2=6attempt；B满足预注册val-only门才B3/B4×2最多4 | 节省没有额外任务增量时的控制成本，保持10attempt/500epochs硬上限 |
| 追加信号 | 实用.003 | 实用>=.003；探索>=.001且<.003；两seed均正、旧B1−B0 guards全过 | 探索与实用分开，禁止放宽保护线或看test追加 |
| 任务增量门 | 所有臂后解释B1−B2 | 在A追加门中固定两T_s>0且mean>=.001 | 不显示增量时预算停止复杂化，不是证明B2等效/充分 |
| validity | 固定完整矩阵 | A六份有效完整配对才门控；failed/missing不能入门 | 防止缺失/故障被解释为阴性或择优追加 |
| guard主体 | 多臂同时报告 | 追加门明确对B1−B0；B2所有指标照报，自身信号另用同guards | 避免把所有对照性能必须过线误称有效性要求 |
| test时机 | 全十run选模及扫描锁后评 | A门YES则等B全完；NO先锁停止决定；再统一最终test | 条件性计划仍test隔离，不可test重开B或调臂 |
| Hpost | 与五臂计划并列、扫描需B0复现 | 默认disabled、执行前可预登记开启，不新增训练、不强制B1之前扫描 | 冻结后处理是独立问题；权重缺失不虚构、也不阻止重建训练本身 |
| 结论 | 有限信号/非平凡性 | 描述A、初步B、实用C、剩余机会D四层 | B不推D；B阶段未运行时竞争解释未排除；两seed最多WEAK |
| 历史checkpoint | 审计缺失/未来可找 | 未来一次有界搜索、根/扩展名/60秒/20000条/32候选/512MiB固定 | 不重做全R1或无边界全盘搜索；找不到转已预算的B0重建 |
| 旧checkpoint复用 | 当前runtime匹配重建优先 | 保留匹配重建默认；找到旧环境权重不自动替换paired B0 | 维持runtime公平，历史数字不能拼接；想变更复用须另修review |
| 缺三R0 work报告 | 旧启动表列九份完整要求 | 如实记录文档缺口，单独不阻断已验明重建 | 限定补充修订，不假称阅读；真正hard gates是源码/data/依赖/前向/review |
| 文件与实施 | 旧规范拟隔离AO04文件 | 建议v12隔离src/config，当前不创建代码 | 保留所有旧文档/源码与S0–S8，独立实现review后才未来run |

探索性追加不是降低实用性能通过线：只在同向>=.001且B1−B2明确增量、旧guards不放宽时允许所测解释对照。若B1/B2都改善却接近，停止的是额外复杂化预算，保留静态非均匀信号、机制解释未测，不能判AO-04无效。

## 3. 审核关注点与闭环

初稿已纳入预审攻击：B4主Adam delta均值可快速调global比率、与格异质性混合；共同shift冗余；B1−B2低功效不达标不是等效证据；六有效A run才追加；test不能重开计划。必要只读源码发现S1 fit末尾自动test、write_outputs绑test/state，未来须隔离train/validation-state writer与最终test evaluator，tripwire证明解封前零新test预测/指标；不直接调用旧fit/write_outputs。B0保存未训练初始tensor/RNG/sampler供配对，不从训练好的B0迁移初始化；last_resume与selected用途分开。B4用未sigmoid raw，不双sigmoid、参数ID分组、不机械套B1的other质量约束。正式blocking issue与修订闭环由新 `R1_to_R2_Independent_Review.md`记录；独立Reviewer实际复核后给出最终PASS（仅设计）。

三份新增文件是完整合同、变更记录和未来实施计划。当前0training/0smoke/0prediction/0baseline-search，设计审核不授予执行权限。本次完成后停止，等待R2正式授权。

初审blocking P1已在Charter/Plan明确修订：禁止裸调用S1自动test入口，拆train_val/state writer/final test，未来Implementation Review动态tripwire保证Adecision前及B期间全局解封前test prediction/metric=0；testlabels只preflight完整性/hash。同时修复P2“其他四raw”笔误为总四raw（raw_bD+另三）、补齐精确参数账，以及P3所有stage阈值用未四舍五入metrics且无额外容差。独立Reviewer §8实际复核确认P1/P2/P3全部CLOSED、最终PASS。此后仅更新三文件审核状态与本段闭环元数据，不改科学内容；复核快照允许该元数据更新。执行NOT READY，当前无实施/运行，通过设计不代替未来实现验收或人类R2授权。
