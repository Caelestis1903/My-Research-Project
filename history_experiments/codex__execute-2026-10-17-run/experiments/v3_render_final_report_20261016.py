from pathlib import Path
import json,re,time,hashlib
import pandas as pd
O=Path('/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016');load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# This writer only runs after primary scientific review and all frozen test checks.
dec=load(O/'final_scientific_decision.json');assert dec['status']=='FINAL_APPROVED_AFTER_ALL_REQUIRED_RESULTS';ver=load(O/'analysis/test_all_cohorts_primary_verification/receipt.json');assert ver['status']=='PASS';q=load(O/'question_registry.json');assert all(x['status'] in ['RESOLVED','FALSIFIED','BOUNDARY','NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN'] for x in q['questions']);assert len(q['questions'])>=46
coh=load(O/'test_summary_verifier_manifest.json');base=load(O/'analysis/test_baseline7_summary/test_seed_summary.json');bs=pd.read_csv(O/'analysis/test_baseline7_summary/seed_test_readouts.csv');on=pd.read_csv(O/'archive/formal/final_baseline_onsets_and_selection_56arm_rows.csv')
def stat(g,label,metric):return next(x for x in base if (x['group'],x['readout'],x['metric'])==(g,label,metric))
def fm(x):return f'{x:+.6f}'
groups=list(load(O/'test_baseline7_summary_manifest.json')['groups'])
selected=['|组|n|SN AUC|SB AUC|AUC差距 SB−SN|95%seed CI|正/负|LL差距|','|---|---:|---:|---:|---:|---|---|---:|']
for g in groups:
 x=stat(g,'selected_macro_auc','macro_auc');ll=stat(g,'selected_macro_auc','macro_logloss');selected.append(f"|{g}|{x['n']}|{x['SN_mean']:.6f}|{x['SB_mean']:.6f}|{fm(x['mean'])}|[{fm(x['ci_low'])},{fm(x['ci_high'])}]|{x['positive']}/{x['negative']}|{fm(ll['mean'])}|")
selected='\n'.join(selected)
onlines=['|组|seed|diagnostic AUC下降 onset SN/SB|confirmed onset SN/SB|','|---|---:|---|---|']
for g in groups:
 for seed in range(2022,2026):
  z=on[(on.group==g)&(on.seed==seed)].set_index('arm')
  fmt=lambda v:'未检出' if pd.isna(v) else str(int(v))
  onlines.append(f"|{g}|{seed}|{fmt(z.loc['SN','diagnostic_AUC_degradation_onset'])}/{fmt(z.loc['SB','diagnostic_AUC_degradation_onset'])}|{fmt(z.loc['SN','diagnostic_confirmed_overfitting_onset'])}/{fmt(z.loc['SB','diagnostic_confirmed_overfitting_onset'])}|")
onlines='\n'.join(onlines)
per=['|组|边际|SN test AUC|SB test AUC|差距|正/负|','|---|---|---:|---:|---:|---|']
for g in groups:
 for v in ['d0_auc','d1_auc','d2_auc','t0_auc','t1_auc']:
  x=stat(g,'selected_macro_auc',v);per.append(f"|{g}|{v}|{x['SN_mean']:.6f}|{x['SB_mean']:.6f}|{fm(x['mean'])}|{x['positive']}/{x['negative']}|")
per='\n'.join(per)
appendix=(O/'Newly_Exposed_Questions_WORKING.md').read_text();pos=appendix.index('## Q28');appendix=appendix[pos:];appendix=re.sub(r'^## Q','### Q',appendix,flags=re.M)
# Keep the six original fields; final actual obligations are clarified without deleting observations.
appendix+='\n最终完成范围补充：Q34已完成254归档快照756800行刷新（compact1016组），Q38最终964/1000次尾部移动；Q29/Q6/Q22规定V7已完成，C的AUC阳性与联合loss失败按Q46保留；全部冻结test读出与数值重建完成。旧“下一实验/待执行”只在提出更强新主张时适用，均不构成本轮遗留必做项。\n'
lit=(O/'V10_existing_work_coverage_boundary.md').read_text();table=lit[lit.index('| 工作 |'):lit.index('\n\nD3/SAF')]
classification=dec['classification'];assert classification in ['MECHANISM-CONFIRMED','EFFECT-CONFIRMED_MECHANISM-UNRESOLVED','BOUNDARY-CONFIRMED','OVERFITTING_ONLY','OPTIMIZATION-SPEED_ONLY','SELECTION-DEPENDENT','COVERED_BY_PRIOR_WORK','NULL','NOT_IDENTIFIABLE']
sections=[]
def add(title,text):sections.append('## '+title+'\n\n'+text.strip()+'\n')
add('1. Executive conclusion',f'''
本轮最终知道的是：在同一Sample经验目标及既有stop-gradient图导数字段下，改变q会改变有限batch估计器与训练路径；性能影响依赖模型配置、优化器/LR、域任务、训练阶段、预算及selection规则。不存在当前证据支持的统一“小域=困难域”、稳定课程收益或唯一必要中介。全部规定实验和触发分支已按实际结果闭合；方法化决定为 **{dec['method_decision']}**。

- **core phenomenon，MEDIUM**：固定参考SGD.1与Adam.01的共同早期诊断AUC四seed负，校准后存在明确反例；下面给出冻结test实际结果。
- **mechanism**：同均值不同协方差身份STRONG；C的局部selection AUC路径与whole-state映射影响MEDIUM，长期必要性和唯一中介未识别。
- **boundary**：ORDER长预算反向、V8指标/阶段不单调、ML配置束/窗口差异、state floor及健康/差距冲突均保留。
- **selection/overfitting**：二者会修饰结论；共同轮次/早期反例阻止普遍selection-only/late-only解释；onset仅操作性分类。
- **optimizer**：无动量SGD足以提供Adam通用必要性的反例；calibrated interaction触发并完成全部V11，fixed-reference interaction不触发。
- **novelty/method**：一般sampling/order/memory已有覆盖；没有唯一MDMT必要机制或新方法四seedprimary endpoint且全单元无害的证据。

{dec['test_interpretation']}

四seed配对20kbootstrap与精确符号随机化分别报告；n4双侧最小p=.125，不能将四个同号写p<.05。历史test曾被使用，本轮条件冻结后的读数不宣称全新pristine holdout。
''')
add('2. Historical correction','''
S5–S8的已闭合结果属于Equal-specific条件下的Adam历史/late-overfitting rescue；换成Sample objective后原selected rescue不再保留，late-only damper不优于旧best的路线已STOP。本轮不复活该路线，也不把同目标q问题重命名为旧方法成功。历史结果用于纠错与范围限定，不计为本轮独立复现；见[V0](V0_decision.json)。
''')
add('3. Objective equivalence',r'''
\[
J(\theta)=\sum_d p_d\frac1{N_d}\sum_{i\in d}\frac{\ell_{i0}(\theta)+\ell_{i1}(\theta)}2,\qquad
\widetilde G^*(\theta)=\sum_d p_d\widetilde G_d(\theta).
\]
\[
\mathbb E[\widehat g_{SN}]=\mathbb E[\widehat g_{SB}]=\sum_d q_d(p_d/q_d)\widetilde G_d=\widetilde G^*.
\]

routing输入保留`emb.detach()`，故G̃是该相同计算图的导数字段，不声称普通全routing导数。SN取q=p、权重1；SB取均匀域q、权重p/q。共同weight_decay=1e−5独立于前向数据J；不能以数据J平台证明全局收敛。模型/数据/标签/初始化/损失与图都匹配；估计器分布和covariance不同。

16个固定θ的完整1,614,254坐标及512批/q审计，直接完整经验图梯度分解残差最大约3.35e−7；covariance trace比SB/SN为2.3155–2.7522。可控exact对照同路径距离0，而随机估计路径不同。代数身份与MC不拒绝分开；点状95%覆盖155/160不是FWER证明。PERM的无条件固定θ边际不能直接当适应性历史下条件无偏性。详见[V1](V1_decision.json)、[Q28及问题全集](question_registry.json)。
''')
add('4. Trajectory vs selection vs overfitting',f'''
宏AUC为六个物理域×任务AUC的等权均值，不是pooled AUC；sample-count和train-p加权单元均值作为独立备选指标保留。定义Δ(t)=AUC_SB(t)−AUC_SN(t)，Δ_selected使用各臂冻结selection规则选轮；两者是不同估计量。固定SGD.1共同pre50诊断AULC差−.003043四负，固定Adam.01共同pre27−.006980四负；校准SGD.3 pre37仅−.000493且1正3负。固定SGD.1的原v2/v3选轮差异已以19600batch损失/16参数快照逐值身份和评价人口/重选分解解释（Q32），并非训练实现改变。

以下为selection macroAUC选轮后的**实际冻结test**；LL使用同一所选轮次。各组最终实际预算不同，不能把自选端点当同训练时钟。

{selected}

[七组test宏观差距图](figures/final_test_selected_macro_gaps.png)（同目录PDF）；[六单元图](figures/final_test_all_six_cells.png)和[全部画图数值](archive/formal/final_test_selected_all_six_cell_plot_values.csv)。

所有固定轮5/10/15/20/30/40/50/100（实际有支持才输出）、final、四selection及common-SN/common-SB选轮、AULC10/20/50/共同pre均在[14cohort测试索引](test_summary_verifier_manifest.json)。final_epoch与selected_fixed_budget是同一端点别名，不能双计；固定100的n2/n3按预算资格有条件描述，不写成四seed无条件效应。1000重选区间是冻结拟合轨迹条件下的validation row不确定性，与test row和训练seed区间不同。

{onlines}

“未检出”不等于无过拟合；onset扫描没有已证明的同时FWER。完整56臂记录含selection onset与选轮，见[CSV](archive/formal/final_baseline_onsets_and_selection_56arm_rows.csv)。

学习速度另列：B0/B2同轮处理样本相同但49/25步；1225步四seed交互−.003588、CI跨零，2450步仅seed25 AUC交互+.007950。2448首次hit记录中1082达到、1366右删失；没有把censor记预算+1或虚构并行墙钟速度。V4自身B0/B2最终50/50/50/100，V5/V8的350属于另一个匹配问题，不能改写V4预算。见[V4](V4_final_budget_and_learning_speed_analysis.md)。
''')
add('5. Sampling decomposition','''
|项|实际control|observation|decision/残余不确定性|
|---|---|---|---|
|exposure|每轮总examples/更新数匹配；q与p/q冻结，逐域任务实际draw计数|SN/SB域曝光不同，同目标期望仍相同|总q干预可识别；曝光不是独立于采样高阶矩的单一中介|
|repetition|WR/PERM保持每q的物理域序列/配额|全部40最终训练4200臂轮/205800批完成；PERM仍重复|固定E,N有R=E−U≥max(0,E−N)，SB小域E>N不能零重复同配额，必要性不可唯一识别|
|coverage|unique/first-seen/cumulative/repeat hist逐批审计|最终350 sampler1400比较臂轮、6300coverage行、50400遥测行；共同50累计coverage全1|覆盖和重复代数耦合，不是独立必要性反事实|
|order|精确同batch multiset/每row次数，仅执行次序置换|E50 ORDER AUC交互+.005816四正，最终均值−.006792；seed25最终交互−.041913|MEDIUM阶段路径敏感性；IID非数据依赖置换不是curriculum；若所有batch等大小且IID，置换保持联合规律，条件路径差异不能自动当总体策略收益。实际964/1000臂轮移动短尾，content与size-clock未分离|
|composition/weights|每批实际域计数、p/q矩、ESS、short-tail位置|SN ESS/n=1；SB约.4357、最大权重2.60785；新公式误差1.024e−15|STRONG身份；不证明scalar variance唯一原因；批内域组成仍随q改变|

Adam PERM最终匹配50/100/100/50，SGD WR/PERM/ORDER50/50/50/350。SGD ORDER自选AUC交互+.007723四正，却SN绝对AUC四负；不是稳定两臂改善。PERM早期/最终CI跨±.003，未用于等效零或必要性排除，额外seed规则不触发。22个最终350来源的规定中间前缀全部完成并逐值等于父前缀。

[V5 SGD](V5_SGD_final_matched_analysis.md)、[V5 Adam](V5_Adam_final_matched_analysis.md)、[最终批次审计](V5_all40_final_batch_weight_tail_primary_verification.json)。
''')
add('6. Gradient mechanism','''
完整均值与covariance trace见第3章。[完整冻结谱图](figures/final_frozen_covariance_spectra.png)及[PDF](figures/final_frozen_covariance_spectra.pdf)展示全部16固定θ的两q谱与累计解释率；[所有CPRE/ALL的81920个已存特征值](archive/formal/final_V1_all_frozen_eigenvalues_81920.csv)逐条可导出。曲线为各固定θ条件结果，不是8个独立训练seed或总体谱；末端近零值有浮点误差。SGD E10 effective-rank比SB/SN=.7681–.8987四seed低于1、anisotropy比1.0290–1.1376四seed高于1，描述重复性MEDIUM；Adam rank比.8878–1.0061有反例。经验谱秩≤511是512批向量的有限样本谱，不是模型潜在总体谱。

固定θ中心化IID lag1/2/5 ALL cosine，Adam约−.000484/+.003080/−.002916、SGD−.005441/+.002653/+.000987。真实移动θ校准E10的C前6批相邻cosine四seed均负，SGD SN−.9849..−.9610、SB−.9751..−.9359；无Adam也可强交替。固定θ来自参考LR、移动轨迹来自校准LR，不能当同θ配对。轮内lag1/2/5仅5/4/1对；跨轮原lag1实际44步（Q35），不称整epoch时间相关。

CPRE和两任务完整梯度身份均完成。p加权task cosine SGD .1111–.2880、Adam .2738–.4262均正，局部域/块可负；不支持“所有任务普遍冲突”。q同时改变trace/shape/direction/高阶矩和后续θ，尚无独立trace-matched shape必要性干预，Q21不可识别。几何与downstream的联系只有共同干预路径及局部块作用，不能以相关性声称谱中介已确认。

最后254归档快照、756800遥测行恢复corrected m并保留旧mixed范数，compact实际1016组；前缀/克隆不是新seed。8个实际Adam/AdaTask块状态oracle已验证恢复公式。大raw本地保留并按SHA索引。[V6详表/谱](V6_geometry_observational_analysis.md)、[最终矩遥测](Q34_final_all_training_moment_refresh_primary_verification.json)。
''')
add('7. Shared/private','''
源码及六单元梯度支持确认C/E共享，R的gates/towers为domain-task-private，P混合。不能把R再私有化包装成阻断共享R的实验。CPRE距离/协方差是correlation，必要性需要真正干预。

V7全部12同θ/状态局部、7条件完成。**C在E10/E40的selection AUC四seed均衰减≥50%，筛选为阳性。** E10的selection loss/train J衰减均四正；E40两者各2正2负、平均衰减均负。冻结规范额外要求loss/J方向一致，故完整条件未满足、full-training NOT_TRIGGERED。没有事后改变AUC50%/四seed/两阶段门槛，也不能写所有局部阴性。P/R/E没有AUC合格阶段。

E10 selection的C相对SB AUC四seed改善，diagnostic仅2正2负；E40缩小正AUC gap会降低本来有利的SB AUC。MEDIUM局部C路径效应，WEAK长期健康恢复或唯一必要性；Q6/Q22不可识别，Q46保留指标冲突。每θ每q一个预定大batch而非额外独立模型。最能证伪局部到长期外推的是另行预注册AUC-only容量/初值/预算匹配C私有化与全单元健康；本轮不事后放宽联合门槛。[完整V7](V7_local_counterfactual_analysis.md)。
''')
add('8. Domain/task factors',f'''
训练N为d0=172782、d1=1743898、d2=89452。实际train正例率(task0/task1)分别.090889/.041214、.377832/.256083、.616867/.484807；diagnostic人口与训练标签率不同。每个draw同时贡献两task标签；p/q、实际域/正例曝光、重复和unique均已记录。六个KR baseline家族中SN固定50的d0常比最小d2更低AUC，因此小域不是困难域的定义。

2x/4x物理行别名复制保持WR empirical distribution和耦合更新完全相同，未引入新独立支持，不识别真实新样本N因果。V8三强度最终50/200/50/350，seed23/25的1200/2100批物理SHA、六单元支持/标签与共同阶段曝光身份完成。目标d2 SN平均AUC随mask下降，但E50 mild loss比clean略好，q差距随阶段/强度不单调；跨severity是不同J_c、只有一个固定嵌套mask，不是纯自然困难度。CPRE范数仅移动θ前6批，未测逐域范数就不能编造“小域梯度更大”。

以下给全部baseline的冻结selected test域/任务边际，完整CI、LL与六cell逐seed同目录可追溯。

{per}

[全部test单元](analysis/test_baseline7_summary/seed_cell_test_readouts.csv)、[单元统计](analysis/test_baseline7_summary/test_cell_seed_summary.json)、[576支持/曝光/训练性能上下文](analysis/V8_final_domain_task_context/domain_task_context.csv)、[384稀疏CPRE范数](analysis/V8_final_domain_task_context/sparse_block_gradient_context.csv)。

**最终test反例：** strong own-final d2交互+.003823四正，诊断却−.013904四负；test绝对SN−.030257/SB−.026434，两臂均受损。mild selected宏交互+.006399四正，但目标d2两臂也受损。此评价人口/阶段反例按Q8保留，不反向增加训练条件。

域规模为主要解释的证据WEAK，因果N驱动不可识别。V5 E50 ORDER SB平均d0+.006111/d1−.000469/d2+.003281有局部大小取舍，但早期/最终反例均存在；V8与ML同样无普遍小域改善/大域下降。只有3物理域，不能伪造高自由度回归或把域/任务当独立seed。
''')
add('9. Optimizer','''
Adam不是通用必要条件：固定无动量SGD.1早期及selected有四seed负向AUC。校准Adam.003相对SGD.3共同22的q-gap交互−.013910四负、CI[−.017713,−.010107]，实际触发V11；固定参考.01/.1不触发，为LR边界。interaction本身不等于历史必要性。

V11四seed三状态六臂最终250/150/100/200；全部12最终+27历史配对和156000次重选检查完成。v-only原始22交互+.012912四正，SN绝对−.012157四负，主要靠SN下降；联合4窗口下SN均值变为+.001180、2正2负，说明估计量依赖cohort窗口（Q44）。共同E50两臂AUC均四seed改善，交互不稳定。β2=0最终两q AUC=.5且LL/J很高，为破坏性floor；m-only长期学习，AUC/loss不总同向。

12个以native SN θ为锚的同θ影子历史/范数对照及1740方向行完成；结论条件于该锚，不是SB自然路径上的间接效应。全局等范数不能固定CPRE块幅度（swap_v的R/P/E约1.48/1.35/1.18倍），不识别纯方向或唯一m/v机制。局部E10 m/full-state映射可改善两q绝对AUC但gap仍负。MEDIUM条件状态映射作用，长期唯一必要性未识别。当前不继续Adam-specific新方法路线；不宣称所有状态方法、LR/eps调参都无效。

AdaTask同目标组件固定.003的最终50/100/50/50仍有q敏感性；最终gap相对native更负可同时伴两臂绝对AUC改善、LL交互四seed改善。不将gap一项当整体方法输赢。[V11完整](V11_fullstate_final_diagnostic_analysis.md)、[V10](V10_final_vs_calAdam_analysis.md)。
''')
add('10. Cross-environment',f'''
|Question|KuaiRand|MovieLens第二环境|Final status|
|---|---|---|---|
|SB−SN exists|固定SGD.1/Adam.01等明确配置早期四负；calSGD阶段敏感|SGD.3早期自身16均值+.001391、CI跨零|配置边界；无稳定同向宏观损害跨环境复现|
|pre-overfit|calSGD自身37−.000493；共同16+.0000776|共同16+.001391、CI跨零|不同窗口不能直接相减（Q41）|
|repetition|WR/PERM完整四seed控制；必要性受E,N结构约束|预设深层机制复制未触发|不宣称已跨环境复制该中介|
|order|相同multiset完整训练，阶段反转/短尾clock边界|条件分支未触发|KR局部路径敏感性不能无条件外推|
|geometry|完整固定θ谱/块/任务方向；非唯一必要性|条件分支未触发|不编造第二环境几何证据|
|shared block|C局部AUC阳性但联合loss门槛失败|条件分支未触发|长期必要性未识别|

KR每轮49步、ML5步（batch40960），同epoch不等同进度。ML四seed同时改变训练子集与优化随机性（同seed SN/SB人口一致），不是固定人口的纯optimizer方差；validation/encoder/scaler固定。KR p约(.0861,.8693,.0446)、ML(.184,.016,.8)，大小比19.5/50与Σp²排序也不同，不能将一个imbalance标量当唯一因果。冻结test提供同规则读数，不反向新增预冻结未触发训练。**实际test边界：** {dec['cross_environment_test']} [V9](V9_final_cross_environment_analysis.md)。
''')
add('11. Existing work',table+'''

文献截止启动日2026-10-16，九项原文/公式与路径见[literature_source_receipt](literature_source_receipt.json)。D3/SAF完整目标改变为STRONG公式边界，不等于性能方法无效；固定AdaTask组件残差不是完整论文全调参失败。一般重要性采样方差、顺序与optimizer memory已有覆盖，当前没有独特MDMT必要机制已识别的证据；Q12/Q13/Q25为BOUNDARY/MEDIUM，不宣称检索穷尽或普遍新颖性。
''')
add('12. Final decision',f'''
**唯一总分类：{classification}**

**方法决定：{dec['method_decision']}。** {dec['method_reason']}

[V13十项终局证据](V13_final_method_decision.md)逐项说明：pre-overfit/非普遍selection-only有配置内支持，环境有明确边界；但非纯速度的普遍排除、唯一必要mediator、主要替代的独立排除、真正新方法primary endpoint/四seed稳定/全域任务无损未同时成立。NO-GO不以C局部AUC失败为理由，也不以低功效null证明不存在机制。

```mermaid
flowchart LR
  D[数据 模型 标签] --> S[N p 正例率 基线]
  Q[q与p/q干预] --> Z[曝光 重复 覆盖 顺序]
  Q --> G[图梯度分布t]
  S --> Z
  Z -. 未单独分离 .-> G
  G --> U[更新与状态t]
  A[优化器 LR batch时钟] --> U
  U --> T[theta t+1]
  T --> G1[下一步图梯度分布]
  T --> M[阶段AUC loss J]
  D --> M
  M --> K[冻结validation选轮]
  K --> E[固定test评估]
```

这是时间展开的候选因果/流程图：q的总干预与部分局部映射已受控，不能把图中每条中介边都当作已经独立识别。

[V12六CF终局](V12_final_causal_synthesis.md)保留同J改q、同q改optimizer/域内sampler、同multiset改order、同θ改块、2B/exact。额外2026–2029未触发，因为核心关闭没有依赖覆盖±.003的关键等效零；不是随意减少seed。除按冻结条件未触发分支外，没有剩余必做实验；归档/推送完成后才结束用户任务。
''')
add('13. Unlimited new-question section','# Newly Exposed Questions\n\n全部Q28–Q46共19项逐条保留，不限制新增问题数量；每项保留原始observation、未解释原因、阻塞范围、已完成最小实验、boundary、下一最小判别实验。\n\n'+appendix)
add('14. Final scientific statement',f'''
{dec['final_statement']}

全部Q1–Q{len(q['questions'])}状态、证据等级、最小实验与剩余边界见[机器可读登记](question_registry.json)和[逐问题终局表](final_question_closure.md)。最终test核验实际覆盖{ver.get('numeric_values_checked','见收据')}个数值，最大误差{ver.get('max_error','见收据')}；14cohort和13500原生模型轮次以及36精确前缀按真实来源归档，未冒充独立重复。运行/失败修复/原始产物索引见[执行报告](execution_report.md)，主要表图见[结果报告](experiment_results.md)，证据/竞争解释见[机制报告](mechanism_analysis.md)。
''')
f=O/'FINAL_REPORT.md';f.write_text('# 2026-10-16 v3 实验终局报告\n\n'+'\n'.join(sections));print('FINAL_REPORT14chapters written',len(sections))
