# DESIGN SPECIFICATION — EMA_CAUSAL_20261110

设计日期：2026-11-10。状态：DESIGN COMPLETE；实现、独立代码审查与实验尚待执行。

本文件将本目录00–11的冻结规范落实为实现范围与验收合同。公式、表schema和条件判别细节分别以03/05、07、08为准；不另设算法、超参或性能门。原来源固定为 `ffa3de067fcda5a1293af83a465d741e069493f1`。设计时来源checkout仍在下载；源码对应关系须在实现前按下述Stage0核对。

## 1. 责任与边界

- 新增 `experiments/ema_causal_20261110/`、该包必要测试和 `research_reports/2026-11-10_ema_causal_attribution/`；原模型、loss、Adam、数据管线、报告只读。
- coder与reviewer必须是不同子agent，沿用主agent模型；runner仅执行并记录原始结果。主agent负责科学分析。实现必须经过独立PASS或修复全部阻塞项后的PASS WITH FIXES才能正式运行。
- 用户当前请求授权实现、完整运行、报告与推送新分支；规划中的DESIGN_ONLY描述的是旧交付状态，不阻止本轮执行。
- start.md的五seed覆盖规则优先：只用2022–2026。不存在性能淘汰或提前停止。
- 自然人口、行序与抽样比例不变。冻结Gaussian D0作为指定源训练边界执行；不另造数据、不增加其他增强或重采样。
- 单卡分别在显存和功耗均低于85%、host RAM低于90%且新任务预算可容纳时准入。计划max-workers=2为初始配置；若提高并发，仍须满足4GiB模型payload和2GiB观察输出约束，并记录解析后的资源配置。
- 参数、梯度、Adam、EMA、R、恢复RNG均只驻RAM。禁止checkpoint、参数NPZ、临时文件、tmpfs绕行；允许原始数据cache、有限预测、数表、日志、代码与报告。

## 2. 唯一正式矩阵及源训练

按ML-C→KR-C→ML-G→KR-G四块，每块五seed，每条50epoch：20轨迹、1000epoch、34,500真实Adam步。ML每epoch20步，KR49步；batch40960，尾批ML21927/KR40052。

M3oE fixed1、embedding16、FCN[512,256,256,64]、4 shared experts；四个fusion系数均1。Adam lr=.01、betas=(.9,.999)、eps=1e-8、coupled WD ML1e-4/KR1e-5；FP32，关闭AMP/TF32/foreach/fused/amsgrad。loader使用原seed+11000生成器。Clean维持原归约顺序；Gaussian沿源noise key、同view、D0 replacement lambda=.5，kappa ML.05/KR.1；KR旧nested augmented_domain=2不生效。

F从theta0初始化，每个真实Adam后原FP32 `mul_(rho).add_(theta, alpha=1-rho)`一次。rho ML=.9828205985452511/KR=.995。O和F分别用完整Validation六格等权Macro AUC严格最大值选点，并列保留最早epoch。所有干预仅作用独立scratch，不反馈训练。

## 3. 最小实现分工

| 责任 | 必须实现并复用的源组件 |
|---|---|
| 来源/协议/数据 | protocol与audit/prepare；复用overnight data_bridge，训练bundle include_test=False；核对raw/cache/row/split/encoder/scaler身份 |
| 状态数学 | ParameterLayout、build_path_groups；history与interventions实现实际R、四时间窗口、组坐标复制和descriptor身份 |
| 采集 | trajectory维持源worker训练语义；每step面板q、稀疏事件、pulse、选点、有限来源包 |
| 内存与调度 | state/guardian；不可变packet增量IPC、epoch事务、四条件块finalize、恢复与资源计数 |
| 真实读出 | 源readouts概率/logit及LL；参数替换、两完整cut、五融合项删除、有限积分和来源诊断 |
| 数表 | pairs精确Fenwick转移、固定pair诊断；analysis按seed统计、固定H桥接；ARTIFACT_INDEX登记统一schema |
| 入口与交付 | run.py所有00列出的子命令；连续队列；freeze→独立Test→report→release |

允许合并模块，不增加第二套模型/数据/指标框架。coder必须报告实际文件、复用接口、最小检查结果和尚未满足的验收项；不得分析科学结果。

## 4. Stage0必须核对的身份

ML raw SHA=`f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`；train/V/Test=800167/100021/100021。
KR raw SHA=`a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`；train/V/Test=2006132/250767/250767。

源cache可复用；不存在则按源prepare_cache重建，不能替换人口。ML/KR参数数773470/1614254；组S=483152/661920，每Dd=74960/295632，每Tt=16576，每DTdt=5381。domain embedding按行；gate/tower索引3*t+d。两数据核对结构支持cell(d,t)→S/Dd/Tt/DTdt；full-functional与训练detach图分别检查。

记录源commit、实际plan commit/hash、代码hash、环境版本及main三份上下文。主agent已确认规划main commit为 `5cf92af0d3a8ed77688762616248166b9b68266d`，目标新分支 `26-11-10`；规划引用的旧main SHA不是本规划提交SHA。源码checkout未完成时不以本地历史相似文件替代固定source。

## 5. 必须预采，不能释放后补算

1. 每epoch完整V O/F与各自选点；F-selected同时保留同刻O、四old分量和所需面板状态。
2. step0及每真实step的P_A/P_B/V_mech p/z；FP64 q_p/q_z从真实theta0预测初始化，同rho更新。只流式消费，不保存全部step预测。
3. 四窗口ML[20,40,80,160]、KR[69,138,277,553]；h_eff=min(h,t)。R_actual=F32.double−O32.double；old=rho^h_eff*R_(t−h_eff)，recent=R−old。每epoch末尾均可能成为tF，提前编译全部t−h捕获事件与引用释放。
4. 固定E5/10/20/35/50及tF参数包；同时保留窗口起点真实在线预测和训练目标摘要。重复锚点复用。
5. 每E5/10/…/50从前一epoch开始F初始化的fast=rho²/slow=sqrt(rho) pulse；终点现场计算一次F/P_A六格全256×256 softplus风险完整梯度，归约all/S/N后释放。
6. E35/E50额外H步fast/slow，从t−H的F开始、首更新t−H+1；用于有条件H2配平。
7. Clean E20/E35最后真实step仅保留pretheta、data_grad、mhat、vhat四FP32向量；WD梯度和实际位移按需推导。

每个anchor仅保留O32、R64、四old64与登记的有限状态。另维护一份理想FP64 R递推审计舍入，不替换真实F。全组端点用原O/F坐标复制，时间状态仅最终合成后一次转FP32。

## 6. 五seed块内实际归因

- A：完整V六格3×3错误/平局/正确转移，O(N log N)、int64、不枚举P×N。固定面板最多32768均匀无放回pairs，同seed/同数据跨目标复用；保留真实mask及原4096历史pairs。纠正、破坏、平局、分层人口占比、重叠均报告。
- B核心：每固定anchor/tF执行O/S-only/N-only/F二因素。五seed齐全后，以晚期E35/E50先seed内后seed间V_A平均删除效应及交互、epsilon=.0002，确定S/N细分；S为EMBED/STAR_SKIP/EXPERT，N为D_ALL/T_ALL/DT_ALL；最多12额外状态、限E10/E35/E50/tF。V_B独立列明。
- C核心：四h的O/recent-only/old-only/F。有限C2只取冻结候选的最多两个重要互斥组、主H窗、O/F两个背景；按03选择，不能混背景相加。
- D1：O/F/q_p/sigmoid(q_z)比较限定固定面板、完全同权重；核对相同pair纠正与破坏。
- D2：STAR+skip的h替换覆盖全部gate和experts；fused_dt替换对应tower输入。O→F及至多一个按五seedV_A正删除效应选定的F−X→F contrast，各cut四态，00/11复现源端点。
- D2b：E35/tF，实际g/e/d/t构成content/routing/interaction/domain/task五项；在真实u_F上逐项删除并真实前向。五seedV_A按冻结顺序选最多一个正项，在同一X contrast重新计算该项并删除，第六态连接X→功能→同pair。
- D3：冻结有界行/对，full-functional scratch导数，alpha=0/.25/.5/.75/1真实读出，积分8→16→32封顶。积分未闭合保留实际接口证据并记录具体残差。
- D4：实际历史窗口训练目标、验证排序损害及recent恢复同pair的重叠；不先命名噪声、负迁移或过拟合。

触发STAR_CROSS、H2_MATCH、ADAM_SOURCE严格采用08§6.1全部布尔条件，五seed完成后、活参数释放前执行。依赖缺失=DEPENDENCY_INCOMPLETE，不得写NOT_TRIGGERED。未触发不得作为科学阴性。

## 7. H桥接与有限判别

每条件独立比较rho7与mechanism7，均含F及all/S/N×两个动作。只实现08固定的C_design_adapted、C_simple_adapted与无拟合carry。ridge目标为P_A未来真实效用，SSE+1*非截距系数平方，训练折标准化；不调参。

LOSO其他四seed origins5/10/15→targets10/15/20拟合，留出seed origins25/30/35/40/45→targets30/35/40/45/50评分；P_B/V仅评分。静态动作、F、oracle、当前直接选择、未来预测分别报告；科学n=5。pulse终点梯度特征必须为完整256×256风险，不能沿用4096抽样替代。

H2触发后仅E35/E50、all+最多一个S/N共同scope，P_A选动作、相同H与最小公共L2半径、V_B评分；零半径记无信息。ADAM_SOURCE仅Clean E20/E35五seed，两个范数匹配虚拟方向，真实新增Adam=0。

## 8. 资源、恢复与状态生命周期

每epoch提交新不可变packet与active恢复状态，再原子发布标量/有限预测和commit ledger；旧anchor不得每epoch重拷贝。GPU读出scratch独立，当前块五seed完成后执行全部核心和触发归因，再仅保留每seed三份端点Osel/Fsel/OatF。四块完成、全局机制冻结后才能独立加载Test。

4GiB模型payload计入FP64、IPC与提交瞬时副本；2GiB输出按06的保存清单计账；真实RSS另受RAM90%约束。fullV预测与激活仅即时RAM使用；不长期保存逐pair大表。每任务一次worker重启；guardian丢失最多一次同seed确定性重放，核对已有hash并去重，实际成本另列。

## 9. 最小必要验收与审查放行

| 检查 | 放行条件 |
|---|---|
| 源纯度 | 两数据Clean/Gaussian含尾批观察开/关：online、梯度、Adam、F、RNG、loader/buffer一致；不得改训练语义 |
| 数学 | 常量/交替短序列验证符号指数初始化；recent+old相对L2≤1e-10；q递推与显式展开一致 |
| 分组 | 坐标互斥完备、真实domain行与cell映射、训练/full函数结构支持正确 |
| pair核 | ties/全同分/完全纠正破坏算例与暴力一致；真实人口计数总和PN、AUC闭合≤1e-12 |
| 图函数 | 两cut端点复现源；分解恒等式≤1e-10；融合FP64代数闭合≤1e-10且源FP32舍入单列；完整导数有限差分核对 |
| 积分 | abs residual≤1e-5+.01*abs(actual delta margin)，最多32点；不足时保留具体未闭合身份 |
| 恢复 | 一次小规模真实worker故障恢复与后续状态一致；guardian丢失验证计划重放及去重路径 |
| 成本 | 原batch、完整V计数、逐step面板、JVP/fusion真实计时与RAM实测；生成完整有限预算 |
| 数据与Test | 数据身份符合；训练bundle不载Test；freeze未满足不能evaluate |

仅执行上述影响研究结论的检查，不加泛化硬化或新ablation。独立reviewer必须查看本规范、实现与CODER IMPLEMENTATION REPORT，并明确逐项缺失是否阻塞；空表、占位统计或只实现CLI不能放行。

## 10. 冻结、报告及完成

统一ARTIFACT_INDEX将07所有逻辑表映射物理路径、schema、hash和行数。EXPECTED_OUTPUTS完整列核心/条件任务；只有已提交epoch数据进入分析。五seed配对原值、均值、样本SD、95% t区间系数2.7764451052及正负数，先seed内阶段聚合，AUC/LL分开。

四块完成后主agent依据数表写Q1–Q5和H1/H2/H3，标证据STRONG/MEDIUM/WEAK；指出重复现象、矛盾、竞争解释、实际域任务边界与唯一主要下一阶段方向。域人数用于人口/成本，不作为困难度代理；本轮不新增域规模实验。至少形成参数/时间X→完整函数→同pair纠正/破坏→净AUC的链，或同权历史预测组合的充分主解释。

冻结所有解释与选择后独立Test仅五态：Osel、Fsel、OatF、S-only-atF、N-only-atF。报告分离独立选点性能与同刻机制。最后检查20轨迹/34,500正式步、核心与全部触发任务、来源/选择/成本/恢复收据，完成FINAL_REPORT及小数表后释放RAM并写RELEASE_RECEIPT。

推送仅本次代码、报告、必要小数表与规范；不推模型、原始数据、大预测数组。不得用旧报告数值填新结果，或在训练结束但功能归因缺失时宣布完成。
