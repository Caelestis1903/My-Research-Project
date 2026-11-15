# STAGE 2 DESIGN SPECIFICATION — 自然人口任务视图混合

状态：**设计冻结，等待 Stage1 最终 decision 与支持表确认后实施。** 本文只授权下述有条件的 NDP-A 初始矩阵，不授权 HPO、Test、NDP-B 或新投影机制。适用计划为 `research_plans/26-11-15_auxiliary_objective_attribution/` 的04–09与runbook；通用协议继承本目录 `DESIGN_SPECIFICATION.md`。

## 1. Stage1触发依据与进入条件

Stage1冻结训练源码hash：`a78bce2f713cf62f9af32548b50bc360b134420ff9ed6ac6990dca22d76a2ad0`。已完成40条新轨迹、2000epochs、69000次Adam/EMA更新，另复用10条CLEAN。当前可用的selected fixed-EMA配对原值来自 `artifacts/auxiliary_objective_attribution_20261115/attempt1/tables/PAIRED_SUMMARY.csv`：

| 对比：验证Macro AUC | MovieLens均值 | KuaiRand均值 |
|---|---:|---:|
| Z-PAIR−CLEAN | −.001546555 | −.019672372 |
| Z-EXAMPLE−CLEAN | −.001995199 | −.009074927 |
| B−Z-PAIR | +.000193271 | −.001884694 |
| C−Z-EXAMPLE | +.000352080 | −.000113225 |

MovieLens C−Z-EXAMPLE为4/5正，五个cell均值为正，95%配对区间为[−.000394935,.001099094]；该区间跨零按原计划不增加否决门。其E1–10/E11–30/E31–50阶段AUC均值分别为−.000914/−.000057/+.000032，selected为+.000352，**扰动正增量证据为WEAK**。KuaiRand相同对比阶段均值为+.000342/+.000554/+.000592，而selected为−.000113，选点与阶段结果不一致。全部B/C仍低于CLEAN。

KuaiRand Z-PAIR−CLEAN早期为+.013559（5/5正），中/晚期为−.012878/−.021348（各0/5正）；旧pilot的早期正值在零扰动下也出现。Stage1已经观测到辅助目标大幅改变实际domain/task/label系数质量，尤其KuaiRand pair正例与小人口域的辅助质量。该权重缺陷是已测对象；扰动方向是否有净收益仍须由自然权重下的直接验证回答。

按预设的五seed均值>0定义、MovieLens五个cell正向及已测目标权重缺陷，设计**一次NDP-A可否证检验**：保留较简单的task-BCE视图方向，移除额外BCE质量和人工cell覆盖，检查自然row-task权重下是否产生验证AUC增量。MovieLens的信号较弱且选点/阶段不一致，KuaiRand没有selected正均值；当前没有据此认定扰动有效。

**实施前条件：** 主agent完成Stage1的support/weight汇总与最终decision，独立审查确认完整性/支持解释，最终decision明确选择进入NDP-A。Stage2运行清单保存该decision、支持表和Stage1源收据的路径及hash。正在流式导出的Stage1支持表不重跑；若最终结果改变进入判断，本文不触发实现或训练。

## 2. 唯一初始候选与新增预算

| 臂 | MovieLens | KuaiRand | 预算 |
|---|---|---|---|
| NDP-A，lambda=.25、alpha=.03 | seeds2022–2026 | seeds2022–2026 | 10条新训练，每条50epochs |
| CLEAN，自然混合delta=0的匹配对照 | 复用5条已审计轨迹 | 复用5条已审计轨迹 | 0条新增，须通过第5节等价门 |

新增 **10条、500epochs、17250次Adam及EMA更新**：MovieLens每条1000步，KuaiRand每条2450步。完整Stage2比较含20条轨迹。CLEAN既是clean参考也是NDP-A数学零delta对照，不以不同名字重复派发。

不新增pair、投影、RANDOM/GAUSSIAN、NDP-C独立臂，不搜索alpha/lambda，不缩减数据人口、batch或训练预算。NDP-B仅在实测full-batch成本阻止NDP-A运行时另行设计，当前不实现；不以主观预估成本提前换成子集。

## 3. 数学目标和自然权重

设自然minibatch有B行，每行两个task。令概率BCE为ell，clean概率为p_it，task t的专属扰动视图概率为p_it^(t)。优化：

`L_NDP = (1/(2B)) * sum_i sum_t [.75*ell(p_it,y_it) + .25*ell(p_it^(t),y_it)]`

等价可实现形式：

`L_clean = mean_probability_BCE(model(x), y)`

`L_view = .5 * [mean_i BCE(p_view0[i,0], y[i,0]) + mean_i BCE(p_view1[i,1], y[i,1])]`

`L_total = .75*L_clean + .25*L_view`

其中view0的另一task输出仅供drift诊断，不加入该视图的训练loss；view1同理。所有B行均出现在每个task的混合中，**没有acceptance剔除、pair采样、label平衡、cell二次归一化或额外辅助均值**。

每个原始row-task的clean-view系数为`.75/(2B)`，augmented-view系数为`.25/(2B)`，合计仍为`1/(2B)`。总质量为1。任意(domain,task,label)分层的总系数质量精确等于该层自然entry数量除以2B。分层aug/clean系数质量比恒为1/3，合计相对自然目标的质量比为1。最后不足40960行的batch使用实际B，不能沿用固定40960作分母。

lambda=.25是该凸混合的占比，与Stage1 `L_clean+.25*(2*mean_selected_BCE)` 的系数含义不同。NDP-A−C的差异包含支持、归一化和目标共同变化；主归因对比为NDP-A−CLEAN/零delta，而不是把全部NDP-A−C变化命名为单一权重效应。

## 4. 全自然batch的task视图

只复用已审查的Stage1 `core.forward`、`core.sensitivity`：同一实际数值模型，临时probe重接gate输入以取得完整输入Jacobian，普通参数训练保留历史gate detach。embedding边界与C一致：捕获模型完整embedding输出，不新增field mask或修改embedding结构。

对每个自然行i、task t：

`g_it = (sigmoid(z_it)-y_it) * grad_e(z_it)`

`r_i = RMS(e_i)`

`delta_it = .03*r_i*g_it/(norm(g_it)+1e-12)`

沿用已冻结数值阈值：`norm(g_it)>1e-10`且`r_i>1e-12`时使用该delta；否则将该行该task的delta设为**精确零**，但保留该entry的完整混合权重。记录fallback原因/数量；不得跳过该行或重归一剩余行。无pair-margin cap、无projection、无非目标task rejection/backtracking。

生成的每个delta detach；探针图和hooks在训练forward前释放，参数`.grad`不被探针填充。一个full-batch probe可同时提取两个task的逐行梯度，无需每行单独运行或构造巨大跨行Jacobian。只保留detached embedding/logits/gradients/deltas；zero/near-zero方向不能导致NaN。

训练过程：一次zero_grad；full-batch敏感度提取；两套detached task deltas；clean forward与各task专属view forward；一次combined backward；一次Adam step；一次after-Adam EMA。不得为了省显存悄悄改为多个Adam step、microbatch更新或额外optimizer。若三个训练forward图的full-batch峰值无法容纳，记录实际资源阻塞再交主agent决定，不能直接改实验语义。

## 5. CLEAN作为零delta counterpart的等价门

数学上delta全零时`L_view=L_clean`、`L_total=L_clean`。FP32下不应仅凭公式就把不同求和/反传顺序的轨迹视为完全相同。实现须提供明确的零delta工程路径，**当两套视图delta全为零时归约到canonical source clean BCE图**，从而保持clean梯度、Adam/WD和EMA数值路径。该路径用于正确性检查，不增加性能配置或搜索项。

最小证据：

1. 同模型/输入的zero-view与clean输出一致；显式2B逐entry凸混合的loss及参数梯度，在已冻结FP32数值容差内等于source clean。
2. 实际实现的zero-delta canonical路径与直接source clean在相同初始化、loader/RNG下得到**完全相同的model、Adam、EMA和更新计数**；小步连续训练与RAM恢复均匹配，selected checkpoint状态/epoch规则一致。
3. 两数据集真实full-batch工程检查中，zero路径单步状态hash与source clean单步hash一致；初始化/数据/软件/optimizer/EMA/validation选择规则与已审计CLEAN记录相同。

等价门通过后，在Stage2复用收据明确记录CLEAN同时承担zero counterpart身份与对应代码/测试证据。若显式混合检查或canonical路径不匹配，先修正该具体差异；若保持实验定义仍无法复用，主agent须记录新增zero臂预算后重新冻结矩阵。不得默认把CLEAN标为匹配而略过检查，也不得默默增加10条zero训练。

## 6. 文件责任和冻结源码

只在以下位置新增：

- `experiments/auxiliary_objective_attribution_20261115/stage2/`：必要的NDP-A core、独立runner/protocol、compact checks与表格导出。模块数取满足责任的最小值。
- `artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1/`：独立运行清单、guardian、轨迹日志及tables；失败重试使用attemptN，不覆盖旧尝试。
- 本报告目录：Stage2实现报告、独立review、分析代码/数表/报告和decision。

不得改Stage1的`__init__.py/core.py/protocol.py/runner.py/checks.py`或其历史outputs，不改main vendor。Stage1 source_manifest只列上述五个冻结文件及main utilities，添加stage2子包不改变旧hash。新Stage2 source_manifest必须列出本次实际训练/检查源文件、所有实际import的Stage1 helpers和其main utilities，单独记录report/export源码hash。允许直接复用Stage1 `protocol.ROOT`，避免在深一层package中错误推导repository root。

| 责任 | 复用/新增位置 | 义务 |
|---|---|---|
| 模型、数据、setup/RNG/EMA/evaluate | main既有conditional_information package | 字节身份与Stage0/1一致，原文件不改 |
| forward/Jacobian | frozen Stage1 core.forward/sensitivity | 保留probe gate重接与训练gate detach，hooks清理 |
| data_cache/optimizer/immutable/hash | frozen Stage1 protocol helpers | 只读复用已核验data-only缓存，独立Stage2 config |
| GPU/RAM准入、RAM状态IPC | Stage1现有helper/main guardian | 独立Stage2 attempt；不复用Stage1 job配置/worker臂列表 |
| NDP-A目标与ledger | stage2新core | 全2B支持、总权重1、task-specific视图 |
| 独立worker/dispatch | stage2新runner | 10条完整矩阵、source/config/selection/hash绑定、val-only |
| paired统计导出 | 可importStage1纯paired/write_csv helper | 不调用会重扫Stage1巨大support流的export；Stage2独立结果集合 |

Stage2运行清单记录Stage1 final decision和receipt hash、自己reviewed source/config/data/selection/design hash、seed池、10条新预算、CLEAN复用索引、RAM_ONLY与Test=false。Stage2实现及独立review均只针对本阶段已授权范围。

## 7. 紧凑日志与指标

NDP-A没有adaptive acceptance，支持由自然loader唯一确定。每batch保留原始行序hash、B及2B、12个(domain,task,label)计数、clean/view/total BCE numerator和denominator、各层clean/view系数质量与loss贡献；断言总质量1、每task各占.5、自然分层质量一致。零方向entry保留在计数/分母内。

每task记录实际非零/zero方向数和原因，delta norm及target/non-target logit drift的count/mean/分位数/max，按cell汇总必要的fallback率。**不重复写81920个entry的行ID/norm数组**：batch raw-row hash、epoch order hash、冻结raw row IDs/loader身份和全支持断言已能重建自然支持。每batch ledger保留gzip JSONL，epoch汇总另存JSON并记录日志hash。

每epoch对online/fixed EMA分别输出unperturbed validation六格AUC/未缩放LogLoss、Macro、Traffic、Pooled、per-task/domain及worst-cell；保存50epoch曲线、E10/30/50、独立selected epoch/state hash、初始/最终模型/optimizer hash、更新计数。成本包括探针/构造/训练/validation时间、wall/throughput、peak allocated/reserved和RAM。GPU异步计时采用必要同步或CUDA event测量并注明口径。

## 8. 最小正确性、smoke与独立review门

复用仍与本阶段数值函数一致的Stage1 Jacobian/finite-difference检查证据；新增compact检查覆盖实际新责任：

- 显式2B凸混合loss/参数梯度与实现一致，lambda=.25时clean/view总质量=.75/.25；最后短batch、缺域/单标签cell仍全量计数，近零方向仅delta置零。
- 两task视图只进入各自target loss，另一task仅做drift诊断；delta detach、无参数probe grad、普通gate detach不变。
- 第5节zero counterpart的loss/梯度/单步Adam/EMA/小步resume及selected身份匹配。
- 每batch ledger分母/自然分层质量、更新计数、loader/RNG独立和恢复状态匹配。
- 在两数据集真实first natural full batch做NDP-A单步backward与zero/clean等价smoke，记录实际峰值、时间和finite参数；实际四行probe复用冻结check helper即可，不扩大数值矩阵。

工程full-batch smoke属于IMPLEMENT的基本shape/device/backward与资源检查，可由coder在最终review前完成。与coder不同的独立reviewer检查设计、实现、coder报告、smoke结果及最终源码hash；只有smoke通过，且review为PASS或所有阻塞项修正后的PASS WITH FIXES，才派发10条正式训练。review/source hash与smoke hash必须对应最终训练实现。

## 9. 运行、恢复与后续Test状态

训练协议完全继承Stage1：FP32、自然80/10/10、batch40960、50epochs、seed2022–2026、single coupled Adam lr.01、原WD与固定EMA、每epochval最早严格最大选点，online/EMA独立。逐卡显存及功耗均<85%、RAM<90%时准入；每卡首batch峰值未知只准一个allocating worker，已知后继续按该卡实际资源并发。不中断无关任务。

model/optimizer/EMA/RNG/loader/best/selected状态仅留独立RAM guardian，完整epoch提交与immutable失败日志沿用已有模式；不写磁盘模型或checkpoint。**完成训练后保留NDP-A selected状态**，供门控后可能的HPO配置复用或冻结Test评估；本阶段不得调用release清空状态。

Stage1复制的CLEAN evidence只含数值与state hash，不等于已持有其模型权重。Stage2验证不需要旧CLEAN权重；当前不增加baseline训练，也不更改/释放旧pilot guardian。若后续Stage3/4触发，先只读检查旧guardian是否保留与复用result完全匹配的CLEAN selected状态；若不可用，另行记录必要的CLEAN恢复/完整重跑预算并走相应设计review。不能从hash/指标重建权重，不能把最终epoch模型替代selected模型，不能因此提前打开Test。

## 10. 五seed分析和有界下一步

主比较为NDP-A fixed-EMA−CLEAN fixed-EMA（同时为数学zero counterpart）。另报告NDP-A online−clean online、NDP-A fixed-EMA−clean online总收益，保持EMA贡献分离。Stage1 C/Z-EXAMPLE仅作目标重构前后上下文，不替代NDP-A的clean主比较。

每数据集分别报告五seed原始值/配对差、mean/sample SD、95%t区间`mean±2.776445*SD/sqrt(5)`和正向seed数；六格、task/domain、Traffic/Pooled、worst cell、selected与E10/30/50均完整保留。CI和seed方向数描述证据，不设排零/全seed正的新硬门。AUC主判，LogLoss独立。

Stage2完整五seed矩阵结束并独立审查后：

1. 若NDP-A相对CLEAN/zero的验证Macro AUC在两数据集均无正均值，停止当前task-view增强路线；HPO、Test、条件消融为NOT_TRIGGERED。
2. 若至少一个数据集出现正均值，结合另一数据集、六格和训练阶段代价，按原计划决定是否进入一次有界lambda研究；没有自动成功结论。此时另行冻结Stage3设计，lambda候选仅{.1,.25,.5}，每值两数据集完整五seed，当前已跑.25可在身份一致时复用；alpha=.03不默认再开搜索。
3. 若仅LogLoss下降，记录概率损失改善，不以此替代AUC晋级。若非目标task出现具体损失，先利用现有six-cell/drift数据定位；不自动开启projection/soft-protection矩阵。

主agent负责科学分析与STRONG/MEDIUM/WEAK证据标签，runner只记录原值。正式发表模型+0.003与每格−.001门属于未来冻结Test阶段，不能作为本Stage2内部验证的额外淘汰门。最终Stage2 decision记录完整配置/源码/数据/选点hash、所有五seed原值与配对结果、自然权重核验、zero/CLEAN复用证据、Test=false、下一阶段触发状态及独立review结论。
