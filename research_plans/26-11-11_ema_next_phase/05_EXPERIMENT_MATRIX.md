# 05｜完整实验矩阵与成本

返回 [执行入口](00_EXECUTION_RUNBOOK.md)。算法定义见 [04](04_ALGORITHM_SPECIFICATIONS.md)、[12–14](12_ROUTE_A.md)，进入门见 [09](09_EVALUATION_AND_DECISION.md)。本文件的训练量均为**未来正常执行预算**。

## 1. ID 与展开规则

`{d}`为`movielens/kuairand`，`{s}`为2022–2026，`{o}`为`clean/gaussian_d0`。每个实际实验ID包括dataset、objective、seed、family、所有科学超参和parent ID；SHA与epoch不能只藏在文件名。下面每个集合按笛卡尔积精确展开，不允许Codex临时添臂。

| 实验ID模板 | 方法/范围 | 对照与依赖 | seed与训练预算 | 进入条件 |
|---|---|---|---|---|
| `P.C.{d}.{s}` | 普通源Adam，O/F；父轨迹 | 冻结source；S0 | 每数据5×50epoch | 必做，重建当前状态用于新实验 |
| `A.M.{d}.{s}.e{20,35}.a{05,10}` | EMBED A_MATCH | 同seed父相同时刻Adam/F；完整RAM fork | 40个6epoch短支，合计240epoch | 父五seed完整 |
| `A.R.{d}.{s}.e{20,35}` | EMBED A_RAW β1=0 | 同时刻父F、获选MATCH | 至多20个6epoch短支，120epoch | 本dataset G_A通过 |
| `A.FULL.{d}.{s}.{MATCH,RAW}` | E20启动算法，E26→E50续训 | 父F/O；公共前缀完整 | 每dataset最多2×5×24=240新增epoch | 对应短程门通过；MATCH仅一个alpha |
| `B.SPLIT.{d}.{s}.H{H}.scope{S,all}.mode{L,P}` | 两态预测 | 父F，B_SHARED_ONLY，另一scope | 12读出配置×两数据×5seed；0额外SGD | 父轨迹逐epoch采集 |
| `B.SHARED.{d}.{s}` | F_S + online_N | F与O | 10条完整读出曲线；0SGD | 同上 |
| `C.G.{d}.{s}.mode{PRED,PARAM}` | 主H，训练可见统一alpha | F，同H固定B，最佳B | 两策略×10父轨迹，0SGD | 同上，H固定40/138 |
| `C.CELL.{d}.{s}.s{05,10}.mode{PRED,PARAM}` | 六格收缩 | 对应GLOBAL与同alpha参数/预测 | 四策略×10父轨迹，0SGD | 同上；晋级依G_C |
| `AB.{d}.{s}.{MATCH,RAW}` | 冻结父B配置作用于A历史 | F、父B、A_F完整2×2 | A短支/续训附带读出；0额外SGD | 父G_B过才采集；A门过才保留完整AB |
| `KD.B.{d}.{s}.l{025,05}.lr{001,003}` | B/C冻结教师蒸馏 | 父F、LABEL/Fteacher | 每data4×5×5=100epoch | 教师进入门通过 |
| `KD.L.{d}.{s}.lr{001,003}` | label-only继续训练 | 同初始化/额外步数/optim | 每data2×5×5=50epoch | 同KD.B一起必做 |
| `KD.F.{d}.{s}.winner` | 固定F教师 | 与KD.B赢家相同λ/LR | 每data5×5=25epoch | KD.B参数选五seed冻结后 |
| `P.G.{d}.{s}` | Gaussian D0父轨迹 | 源Gaussian O/F | 至多10×50=500epoch | 本dataset G_G通过且需要新状态 |
| `G.CAND.{d}.{s}.winner` | 仅该dataset Clean主候选 | 同Gaussian父F/O，必要匹配控制 | A/AB另5×30；B/C无SGD；KD另3×5×5 | 继承Clean超参，0新HPO |

候选同轨迹上的online/F、多个H与scope不增加训练seed。短支不是50epoch完整评价；续训合计使用前缀20+短支6+剩余24=50epoch。KD使用已训练教师轨迹再增加5epoch，计算预算必须分列。

## 2. 必做核心量与有限上限

| 阶段 | ML epoch上限 | KR epoch上限 | 总epoch | Adam更新 |
|---|---:|---:|---:|---:|
| Clean父轨迹 | 250 | 250 | 500 | 17250 |
| A_MATCH短程 | 120 | 120 | 240 | 8280 |
| **必做核心** | **370** | **370** | **740** | **25530** |
| 条件A_RAW短程 | 60 | 60 | 120 | 4140 |
| 条件MATCH+RAW续训 | 240 | 240 | 480 | 16560 |
| 条件KD七臂 | 175 | 175 | 350 | 12075 |
| 条件Gaussian父轨迹 | 250 | 250 | 500 | 17250 |
| Gaussian获胜算法的额外训练上限 | 150 | 150 | 300 | 10350 |
| **所有条件均触发的保守上限** | **1245** | **1245** | **2490** | **85905** |

Gaussian的额外上限是A/AB的30epoch续训；若为KD则每dataset75epoch而非150：Gaussian父轨迹按冻结教师配置选点并提供F初值，KD_B/LABEL/Fteacher三个学生使用Clean目标、未扰动输入、冻结λ/LR各跑五seed×5epoch，条件单列gaussian_d0_parent__clean_student；若为B/C则0。这里不把不同互斥主候选的Gaussian额外成本相加。工程smoke、失败epoch和一次恢复重放另列物理账本；不扩展逻辑矩阵或增加统计n。

## 3. 无SGD读出成本也必须计数

父Clean完整V每epoch的前向上限：O/F=2、B_SHARED_ONLY=1、B_SPLIT=3H×2scope×2态=12、C匹配参数=GLOBAL1+两s各2=5，共**20个全batch等效前向**。C预测组合直接复用B主H/S的两态logits；logit/prob共用组件前向。20是去重前上界，状态相同按hash复用。

10父轨迹×50epoch因此最多10000个full-V等效前向；另有P_A/P_B小面板的同样必要读出和C固定pair风险。C_CELL_PARAM每个domain两task各用一个状态，总计6次调用、约2个全batch等效前向，记录实际调用/行数与时间，不能误算1次。

A短支/续训默认O/F两读出；若父B已通过G_B，再加该唯一冻结B的2次前向，不增加12配置网格。A_RAW按同一规则；不会因为算子成本较低而隐去读出耗时。

KD每条5epoch的online/F各读出，e0复用初始化指标；缓存B/C教师原train软标签最多2个全train前向，F教师最多1个。正式Test只按freeze列表计数，没有任何优化器更新。

## 4. 必须保存的结果

所有曲线：每seed六格AUC/LL、Macro/Traffic/Pooled、epoch/step、prediction/state/config hash、实际人口数、valid mask、运行时间、显存与RAM、同源控制ID。

A额外：native/applied EMBED实际范数、匹配误差、方向cosine、fallback计数、online/F短程差、E20与E35各自末两epoch差；不保存全部梯度。

B额外：H/h/a、重心重构误差、scope、logit/prob、两态预测差与固定pair净纠正/破坏；不每step保存参数或q。

C额外：fit_epoch/application_epoch、P_A/P_B身份、全局/六格alpha、每cell支持与退化、是否改变rank、未来风险与AUC；参数匹配控制独立分数。

KD额外：教师精确状态/选点、学生初始父F、soft target cache hash、归一化loss两项、额外曝光/步数、LABEL/Fteacher配对差、保留率。所有候选原始标量包括负面结果，不能只留赢家。

## 5. GPU、RAM与存储的估算依据

原因果smoke（含旧面板观察）Clean每epoch ML13.3294s、KR42.3881s；Gaussian16.7385s、47.3646s。原20轨迹1000epoch的worker epoch耗时合计29280.57s；都不是本轮纯训练速度。完整V旧4500读出累计2893.289s可作读出量级参考。[来源](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/research_reports/2026-11-10_ema_causal_attribution/PROVENANCE.md)

用旧Clean epoch量级估算：核心740epoch约5.73累计worker小时；全2490epoch上限以Clean单价粗估19.27小时，Gaussian与20读出的变化必须另校准。建议核心预留**6–12累计worker小时**、全触发预留**20–36累计worker小时**的量级，两个可用GPU的墙钟由实际竞争与准入决定。此估计不是每日硬停条件；smoke将训练、观察、CPU风险与IPC分项后生成当前RESOURCE_ESTIMATE，不能把含观察的旧时间再全量相加重复计费。

参数数ML773470、KR1614254，一份FP32参数2.95/6.16MiB；B稀疏历史峰值5/11份F即14.75/67.74MiB；20个A前缀anchor的θ/m/v/F核心约364.3MiB。父候选best保守每个最多3份参数（两态+真实父F），C条件参数可保存两态和alpha在读取时重建。最终按tensor_bytes计算guardian4GiB与外部7GiB、host90%/GPU85%，具体生命周期见08。

teacher两个任务FP32 soft targets每行8bytes：ML每seed6,401,336bytes、KR16,049,056bytes；两teacher×五seed×两数据总224,503,920bytes（约214.1MiB）。预测输出总预算2GiB；完整V/Test原logits在RAM现场归约，不持久化所有epoch/网格。

## 6. 完成、失败与未触发

任务完成必须有正式COMMITTED收据和五seed覆盖；只有文件存在不算完成。已触发短支/续训/蒸馏不因seed1–2表现差取消其余。数值非有限、来源不符或资源不足分别失败/阻塞；重试规则见08。每个未触发行写出门ID及实际输入数值，不使用“效果可能不好”作为状态。
