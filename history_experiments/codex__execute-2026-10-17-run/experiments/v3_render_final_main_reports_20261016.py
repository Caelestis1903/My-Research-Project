from pathlib import Path
import json,time,re,hashlib
import pandas as pd
O=Path('/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016');load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();dec=load(O/'final_scientific_decision.json');assert dec['status']=='FINAL_APPROVED_AFTER_ALL_REQUIRED_RESULTS';archive=load(O/'final_test_archive_primary_verification.json');assert archive['status']=='PASS_ALL156_TEST_ARCHIVES_AND_RAW_INDEX';ver=load(O/'analysis/test_all_cohorts_primary_verification/receipt.json');assert ver['status']=='PASS';q=load(O/'question_registry.json');assert len(q['questions'])==46 and all(x['status'] in ['RESOLVED','FALSIFIED','BOUNDARY','NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN'] for x in q['questions'])
phases=[('V0','阅读/历史纠错/冻结定义','V0_decision.json','规划d18f194；旧Equal/late-rescue不复活'),('V1','同目标与完整图梯度审计','V1_decision.json','16θ×512batch/q；1,614,254坐标及exact控制'),('V2','固定参考轨迹','V2_fixedSGD_and_interaction_sensitivity_analysis.md','Adam.01/SGD.1各四seed两q，固定轮/早期/选轮独立'),('V3','Selection规则与基线','V3_final_baseline_selection_analysis.md','7家族28对112000次选择检查；冻结test完成'),('V4','校准/2B/足预算/学习速度','V4_final_budget_and_learning_speed_analysis.md','16pilot；正式训练与延长；2448hit/censor，20等步记录'),('V5','域内采样与顺序','V5_SGD_final_scientific_obligations_primary_certificate.json','40最终训练4200臂轮205800批；350来源22前缀完成'),('V6','谱/块/任务/稀疏时间与矩遥测','V6_geometry_observational_analysis.md','320几何行、8完整任务梯度；254快照756800矩行刷新'),('V7','共享/私有块局部反事实','V7_local_gate_primary_decision.json','12局部×7条件；C AUC筛选阳性，联合loss门槛失败，条件完整训练未触发'),('V8','名义规模/输入困难度/标签曝光','V8_final_diagnostic_obligations_primary_certificate.json','2x/4x精确别名；三强度4seed；576单元/384稀疏范数；3300物理batch文件'),('V9','第二环境','V9_final_cross_environment_analysis.md','ML四seed50；宏观负效应未稳定复现，深层机制分支未触发'),('V10','文献/目标/AdaTask','V10_final_vs_calAdam_analysis.md','9原文审计；同目标组件4seed50/100/50/50'),('V11','完整状态与影子反事实','V11_final_diagnostic_report_primary_review.json','最终250/150/100/200；12最终+27历史配对、156000选择检查'),('V12','六CF因果闭环','V12_final_causal_synthesis.md','CF1–6全部完成，结构NI与条件未触发明确'),('V13','十项方法化判断','V13_final_method_decision.md',dec['method_decision']),('V14','终局问题/报告/归档','FINAL_REPORT.md','46问题全部终态、19新问题六字段、14章；推送核对后任务结束')]
ledger={}
for phase,title,evidence,detail in phases:ledger[phase]=dict(status='COMPLETE',required=True,report_or_receipt=evidence,actual_scope=detail,independent_review=False)
ledger['V7']['conditional_full_training']='NOT_TRIGGERED_BY_COMPLETE_FROZEN_GATE_C_AUC_POSITIVE_BUT_LOSS_CONFLICT';ledger['V9']['conditional_mediator_replication']='NOT_TRIGGERED_BY_PREDECLARED_DIAGNOSTIC_GATE';(O/'execution_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
phase_table='|阶段|已完成内容|实际范围|证据|\n|---|---|---|---|\n'+'\n'.join(f'|{v}|{t}|{d}|[{e}]({e})|' for v,t,e,d in phases)
execution=f'''# 实验执行报告：规定流程已完成

全部规划V0–V14、已触发分支、冻结test与数值核验、问题闭合及14章报告已完成。唯一总分类为 **{dec['classification']}**；方法决定 **{dec['method_decision']}**。最终远程提交须以Git实际push及ls-remote回读为准，不以“已启动”替代完成。

本轮使用隔离目录`/home/xzf/Code/phase0-3-20261016-v3-git`，分支`codex/execute-2026-10-16-v3`，保护原工作区未提交改动与缓存。原始start.md为GB18030，已保存[source_start_utf8.md](source_start_utf8.md)；其§11优先约束为主控完成设计/实现/自审/分析，子agent只机械执行进程，故independent_review=false。没有伪称独立review。

## 环境和执行约束

使用既有`phase0-3-env`：Python3.13、PyTorch2.9/CUDA12.6、2×RTX3090、32CPU、约64GiB RAM。无必要依赖重装或数据重下载。原始数据完整性KR SHA a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f、ML SHA f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4；每source config另含完整人口/编码/预处理指纹。训练种子2022–2025，q/模型/任务/初值/目标/超参数按冻结设计。没有根据test改训练条件。

持久Popen/start_new_session队列保存真实PID/argv/日志/状态；每GPU≤5且显存<90%，所有GPU前序状态纳入调度，CPU阶段按2前缀/8summary等资源约束。未使用ionice/nice或进程优先级变动。完整模型/预测/批次留本地artifact盘；GPU最终状态与真实退出记录见[状态快照](final_actual_GPU_queue_state_archive.json)。直接工具分析未捕获OS子PID时如实记null和真实tool session，不伪造PID，见[记录](primary_direct_analysis_execution_receipts.json)。

{phase_table}

## 实际预算和停止条件

V4 native Adam50/100/50/50、SGD B0/B2各50/50/50/100；V5 Adam50/100/100/50；V5 SGD50/50/50/350；V8三强度50/200/50/350；V10 AdaTask50/100/50/50；V11三状态共同250/150/100/200。按train最近5/前5相对改善≤.005或selection确认过拟合停止，两臂及预定匹配组共同延长；实际350并非硬上限，seed25此前300不满足而继续。未用diagnostic/test决定预算。实用平台不等于全局收敛。

V7不是全部阴性：C AUC在10/40轮四seed≥50%衰减，第二阶段LL/J方向冲突，原联合条件完整门槛未满足。其长期架构对照按条件未触发，并非遗漏必做。额外2026–2029未触发，因为核心关闭不依赖±.003关键null等效。所有NI结论均已有规定最小实验及结构/联合干预限制。

## 冻结test、核验与归档

[test全局冻结](test_readout_frozen.json)先于本轮任何test指标，固定46问题定义、14初始/最终cohort、所有producer/config/model/源SHA、四选择规则及窗口。156来源目录、79个不同paired分析目录、160个cohort-pair记录有重叠，不能当独立seed。按字节完全相同config/model分组，120原生来源实际13500模型轮次test推断，36精确前缀引用2200已推断模型轮次；所有156来源94200条epoch×cell记录（含复用）已归档。

首个AdaTask23 SN100→50前缀全部50×6单元从实际test标签与预测以独立AUC/LL公式核验，最大误差8.33e−17。[全14组数值重建](analysis/test_all_cohorts_primary_verification/receipt.json)核验{ver['numeric_values_checked']}数值，最大误差{ver['max_error']}，包含全部选择重采样、实际n1–4/预算资格、final别名、所有原生预测hash和既定交互。历史test已暴露，不声称pristine holdout。

[156来源归档核验](final_test_archive_primary_verification.json)、[原始本地产物hash索引](final_test_local_raw_hash_index.json)、[精确主控核验源码](primary_final_audit_source_archive.json)。上传CSV/JSON/图PDF/PNG和可复现源码，不上传数据集、模型权重、预测NPZ、冗余日志或缓存。Q34 raw155.7MiB本地SHA索引、compact1016组归档。

## 异常、修复与未执行分支

早期V4首次启动的工程故障及最小修复保留于[V4_attempt1_postmortem](V4_attempt1_postmortem.json)和[V4_startup_fix_review](V4_startup_fix_review.json)，没有覆盖历史或把失败标完成。主控test前缀独立校验首次因DataFrame列顺序比较失败；修正校验的列对齐后全部300单元PASS，未改冻结实验源码、数据或指标，也未重复推断/前缀生产。14组汇总首次在JSON输出边界因NumPy int64类型失败，原冻结统计源码保持原SHA；独立兼容wrapper仅将标量转为Python原生类型，14个失败目录完整保留，重跑实际全部exit0，42份原有CSV/JSON逐字节相同；见[修复核验](test_summary_serialization_repair_actual_primary_verification.json)。正式最终test任务及汇总修复重跑均记录实际exit0；队列完成状态回读，持久launcher的returncode保持null，不伪称直接等待到队列OS退出码。动态Q28–Q46共19项完整保留。

没有剩余规定必做实验。未触发条件分支、结构不可识别和更强未来假设逐项写明；不以“future work”掩盖当前必做缺口。最终验收以[FINAL_REPORT](FINAL_REPORT.md)、[问题全集](question_registry.json)、Git实际远程验证记录为准。
'''
(O/'execution_report.md').write_text(execution)
base=load(O/'analysis/test_baseline7_summary/test_seed_summary.json');groups=list(load(O/'test_baseline7_summary_manifest.json')['groups']);rows=[]
for g in groups:
 for label in ['common_epoch50','final_epoch','selected_macro_auc']:
  for metric in ['macro_auc','macro_logloss']:
   x=next(x for x in base if (x['group'],x['readout'],x['metric'])==(g,label,metric));rows.append(dict(group=g,readout=label,metric=metric,**{k:x[k] for k in ['n','SN_mean','SB_mean','mean','ci_low','ci_high','positive','negative']}))
df=pd.DataFrame(rows);df.to_csv(O/'archive/formal/final_baseline_key_test_readouts.csv',index=False)
table='|组|端点|指标|n|SN|SB|SB−SN|95%seed CI|正/负|\n|---|---|---|---:|---:|---:|---:|---|---|\n'
for x in df.itertuples():table+=f'|{x.group}|{x.readout}|{x.metric}|{x.n}|{x.SN_mean:.6f}|{x.SB_mean:.6f}|{x.mean:+.6f}|[{x.ci_low:+.6f},{x.ci_high:+.6f}]|{x.positive}/{x.negative}|\n'
inter=load(O/'analysis/test_all_cohorts_primary_verification/test_contrast_summary.json');chosen=[]
for x in inter:
 if x['cohort'] in ['V4_batch_final','V4_optimizer_final','V5_Adam_final','V5_SGD_final','V8_final','V10_final','V11_final'] and x['variable']=='macro_auc' and (x['readout'] in ['common_epoch50','final_epoch','selected_macro_auc'] or x['readout'].startswith('common_pre_AULC_')):chosen.append(x)
t2='|cohort|条件|端点|n|SN绝对变化|SB绝对变化|q-gap交互|95%CI|正/负|\n|---|---|---|---:|---:|---:|---:|---|---|\n'
for x in chosen:t2+=f"|{x['cohort']}|{x['condition']}|{x['readout']}|{x['n']}|{x['SN_condition_minus_baseline_mean']:+.6f}|{x['SB_condition_minus_baseline_mean']:+.6f}|{x['mean']:+.6f}|[{x['ci_low']:+.6f},{x['ci_high']:+.6f}]|{x['positive']}/{x['negative']}|\n"
results=f'''# 实验结果报告：真实全流程读数

唯一总分类 **{dec['classification']}**，方法决定 **{dec['method_decision']}**。{dec['test_interpretation']}

## 七组baseline的冻结test

宏AUC为六个物理domain×task单元AUC等权均值，不是pooled AUC；按test样本数和train p加权的单元指标另列。

{table}

每组四seed真实值、逐域/任务、六cell、AUC/LL以及AULC/四选择/common-SN/common-SB选择规则均见[14cohort清单](test_summary_verifier_manifest.json)。上表final与fixed-budget为别名；最终预算可随seed不同，不能把own-final当统一epoch。固定100可能只有预算条件n2/n1支持，Q45要求不写四seed。n4精确双侧符号随机化最小p=.125；CI为20k训练seed配对bootstrap，1000选择bootstrap是另一个条件不确定性。

[宏观test图PNG](figures/final_test_selected_macro_gaps.png)/[PDF](figures/final_test_selected_macro_gaps.pdf)、[六cell图PNG](figures/final_test_all_six_cells.png)/[PDF](figures/final_test_all_six_cells.pdf)、[原始画图数值](archive/formal/final_test_selected_all_six_cell_plot_values.csv)。六cell/域/任务不是独立训练seed；图中正负方向需分别看AUC与LL。

## 既定反事实的冻结test交互

{t2}

上述只是全量表中预定义共同早期、E50、最终与AUC自选的宏AUC摘录；所有LL/域任务/六cell及每seed绝对变化、支持交集/排除key均在[完整交互](analysis/test_all_cohorts_primary_verification/test_contrast_summary.json)、[逐seed表](analysis/test_all_cohorts_primary_verification/seed_test_contrasts.csv)。不同训练预算/选择池/阶段不能混作同一因果对照；差距缩小还须结合SN/SB绝对健康，不能自动叫修复。

## 诊断轨迹、训练目标与机制读数

固定SGD.1共同pre50 AUC−.003043四负，fixedAdam.01 pre27−.006980四负，calAdam.003 pre22−.013951四负，calSGD.3 pre37−.000493为1正3负。V5 ORDER E50交互+.005816四正，但seed25 FINAL350交互−.041913；自选AUC交互+.007723四正伴SN绝对四负。V7 C局部AUC阳性与E40 LL/J冲突并列；V11 β2=0最终floor、v-only健康/差距阶段冲突保留。全部具体CI、每域任务/六cell和trainJ参见下列阶段报告。

|读数|完整报告|
|---|---|
|固定SGD与参考interaction|[V2](V2_fixedSGD_and_interaction_sensitivity_analysis.md)|
|selection与7组baseline|[V3](V3_final_baseline_selection_analysis.md)|
|LR/batch/足预算/20等步/2448hit与censor|[V4](V4_final_budget_and_learning_speed_analysis.md)|
|WR/PERM/ORDER最终|[SGD](V5_SGD_final_matched_analysis.md)、[Adam](V5_Adam_final_matched_analysis.md)|
|均值/trace/谱/CPRE/任务/稀疏时间|[V6](V6_geometry_observational_analysis.md)|
|12局部×7条件|[V7](V7_local_counterfactual_analysis.md)|
|规模/输入proxy/正例率/曝光/576cell上下文|[V8](V8_final_matched_diagnostic_analysis.md)|
|第二环境|[V9](V9_final_cross_environment_analysis.md)|
|AdaTask固定同目标组件|[V10](V10_final_vs_calAdam_analysis.md)|
|完整状态与1740方向记录|[V11](V11_fullstate_final_diagnostic_analysis.md)|

所有实际停止依据、各seed预算、56臂onset与选择epoch在[执行报告](execution_report.md)和[onset CSV](archive/formal/final_baseline_onsets_and_selection_56arm_rows.csv)。完整证据分类与竞争解释见[机制报告](mechanism_analysis.md)；14章最终规范见[FINAL_REPORT](FINAL_REPORT.md)。
''';(O/'experiment_results.md').write_text(results)
mechanism=f'''# 科研机制分析：证据、反例与边界

**{dec['classification']}；{dec['method_decision']}。** {dec['final_statement']}

## 已确认的观察与证据强度

- **STRONG：** 同前向经验J及同stop-gradient图导数字段期望身份；完整固定θ/批次权重/模型与产物SHA；同q三severity行/顺序/曝光耦合；R已私有/C和E共享/P混合。STRONG身份不等于泛化机制STRONG。
- **MEDIUM：** 明确配置内早期q影响，optimizer/LR交互与顺序阶段敏感性，whole-state映射局部/条件影响，以及C在E10/E40的selection AUC局部路径。test按冻结规则另行确认，见[数值核验](analysis/test_all_cohorts_primary_verification/receipt.json)。
- **WEAK或不可识别：** 唯一长期必要中介、纯content-order、coverage独立于repetition、shape独立于trace、全局等范数下纯方向、N主要因果驱动、普遍困难域规律、小域改善/大域下降取舍，以及新方法全面收益。

## 反事实与竞争解释

[V12终局综合](V12_final_causal_synthesis.md)分别列同J改q、同q改optimizer/sampler、同multiset改order、同θ改块、2B/exact六项。目标/图不一致已反证；泛指所有效应只由Adam、selection、late-overfitting造成均有共同早期反例。不能因此排除速度/clock、参数路径、高阶估计器分布、输入/表示law、coverage/repetition耦合与多通路共享效应。

固定E,N下R=E−U≥max(0,E−N)，SB小域E>N使零重复同资源对照不可行；PERM保留重复并改变依赖结构。ORDER1000臂轮964次移动短尾，不能把纯内容与batch-size-clock隔离。大batch同时改变步数。名义2x/4x别名没有新独立支持。V8 mask改变J_c且只有一个固定嵌套mask，每seed并非独立mask重复。

C的AUC筛选阳性必须保留：10/40轮四seed衰减≥50%；E40 selectionLL和trainJ各2正2负、平均衰减负，原联合一致门槛未满足，条件完整架构训练未触发。局部gap缩小不代表绝对健康；selection与diagnostic也不同。V11固定22的v-only缩gap主要来自SN下降；E50两臂AUC改善而gap交互不稳；β2=0的floor不能当机制被阻断；globalnorm=1仍改变CPRE块幅度。

四seedCI、20kbootstrap与精确sign检验分开，n4双侧最小p=.125；部分固定100只按真实n/seed预算资格描述。历史test暴露、ML训练子集seed混合、onset非FWER、移动θ前6批稀疏lag，以及routing detach均限制外推。没有用宽CI证明等效零来关闭核心，所以额外2026–2029分支未触发。

## Domain/task与模型阶段

KR的中等d0在多baseline中比最小d2更低AUC，N不能代替难度；正例率与train/validation人口不同，loss高低不能独立排序难度。V8目标域SN平均AUC随mask下降，但mild E50 loss略改善、q效应不单调；C/P/R/E稀疏范数也不单调。V5 E50局部小中域SB改善/最大域略损，但早期/最终、V8/ML反例阻止统一取舍；证据WEAK。只有3域，不以伪回归识别N主因。

ML共同16与KR各自37是不同估计量，跨环境数据/输入维度/标签/步钟/采样子集共同变化，不归于唯一dataset/N/difficulty。预设宏观负效应未稳定复现，深层机制复制条件未触发；最终test不反向新增训练。

## 文献与方法判断

九篇原文审计区分目标等价、干预和机制。D3/SAF完整权重一般改J，不能直接覆盖同J问题；固定AdaTask组件残余不能推出最优完整方法失败。一般importance sampling/variance、order与optimizer memory已有先例，没有识别唯一MDMT必要机制就不能宣布创新。详见[文献边界](V10_existing_work_coverage_boundary.md)。

[V13十项终局](V13_final_method_decision.md)：当前没有新方法在真实primary endpoint、四seed稳定且所有域任务无损的完整证据，方法决定为{dec['method_decision']}。这不等于所有未来机制方法都失败。后续最能区分解释的设计为固定尾部时钟的顺序控制、可行E≤N资源下支持/重复控制、健康/进度匹配状态，以及另行预注册AUC-only C私有化；这些更强新问题不掩盖本轮必做项，后者已全部完成。

全部46问题终态与19新问题六字段见[终局问题表](final_question_closure.md)、[registry](question_registry.json)及[最终报告第13章](FINAL_REPORT.md)。
''';(O/'mechanism_analysis.md').write_text(mechanism)
print('Main3 rewritten andclean15phasefinalledgercomplete; selectedtest42rows+fullinteractions included')
