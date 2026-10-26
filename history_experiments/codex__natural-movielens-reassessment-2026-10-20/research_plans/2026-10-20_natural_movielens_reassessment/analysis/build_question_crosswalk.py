"""Preserve every old question verbatim; append the new population assessment."""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'question_registry';OUT.mkdir(exist_ok=True)
sources={
 'N17_QUESTION_REGISTRY_FINAL.json':Path('/home/xzf/Code/phase0-3-20261017-git/research_plans/2026-10-17/execution_20261017/QUESTION_REGISTRY_FINAL.json'),
 'N19_OPEN_PROBLEMS_ORIGINAL.md':Path('/home/xzf/Code/phase0-3-n19-git/research_plans/2026-10-18/N19/OPEN_PROBLEMS.md'),
 'CM19_OPEN_PROBLEMS_ORIGINAL.md':Path('/home/xzf/Code/phase0-3-20261019-git/research_plans/2026-10-19/execution/OPEN_PROBLEMS.md'),
}
index=[]
for name,path in sources.items():
    shutil.copyfile(path,OUT/name)
    index.append(dict(source_path=str(path),copied_path=str(OUT/name),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
pd.DataFrame(index).to_csv(OUT/'preserved_source_index.csv',index=False)
old=json.loads((OUT/'N17_QUESTION_REGISTRY_FINAL.json').read_text())

# These are current assessments, not replacements of historical states. All
# as-yet-unfinished natural stages explicitly remain pending.
assessments={
 1:('SCOPE_CONFIRMED','MEDIUM','V2;AC-2D','自然 fixed Adam final SB−SN −.004084 四负；learned AC-2D selected −.000047 两正两负。存在性保留，非普遍性能损害。','N17;core mechanisms'),
 2:('MODIFIER_NOT_SOLE_CAUSE','MEDIUM','V2;AC-2D','自然 V2 Adam selected +.000479 而 final −.004084；固定轮次也有 q 差异，不能一概归为选模假象。','N17;AC-V9 trajectory'),
 3:('HISTORICAL_FALSIFICATION_RETAINED','MEDIUM','historical V2/V9;new V2','Adam必要性的历史反例保留。自然 SGD .1 selected −.004258，但其有限LR低基线不等于所有SGD。','AC-V9 full trajectory'),
 4:('NOT_IDENTIFIED','WEAK','historical S5/V3','m/v/history 联合状态与即时非线性不是 m 单独必要性。','AC-S5;core mechanisms;N17 N3'),
 5:('NOT_IDENTIFIED','WEAK','AC-2D controls','固定 N/step 的人口对照可排除部分单标量解释，但支持、标签特征、重放及几何仍成组变化。','N17 matching'),
 6:('NOT_IDENTIFIED','WEAK','historical V3/N17','命名为 shared 的块不等于必要中介；局部干预与长期拓扑需要分开。','N17 N3'),
 7:('SIZE_ONLY_EXPLANATION_CHALLENGED','MEDIUM','AC-2D','同187500行、250步时 Equal损害已显著减小；全支持匹配250明显强于匹配N。N相关但不单独解释全部。','N17;N19;CM19'),
 8:('SIZE_IS_NOT_DIFFICULTY','MEDIUM','V2;AC-2D','自然最大 d2 的 SN baseline AUC 低于 d0/d1；不能以小域替代困难域。困难度由固定模型/任务/阶段定义。','all remaining baseline cells'),
 9:('NOT_IDENTIFIED','WEAK','AC-2D controls','自然比例比人工接近均衡，但改变了实际支持/标签分布；不是失衡量的单因素实验。','N17;transfer'),
10:('STAGE_SELECTION_BOUNDARY','MEDIUM','V2;AC-2D','selected/final 以及宏、train-p、eval-p cell 加权指标分别保留，不按结果择优。','all remaining stages'),
11:('CROSS_ENVIRONMENT_GENERALIZATION_RESTRICTED','MEDIUM','V2;AC-2D;N18','历史人工ML不能当第二个自然总体；新自然部分配置不复现原强负均值。KR数值仍有效。','N17;transfer;AO04'),
12:('HISTORICAL_LITERATURE_SCOPE','WEAK','historical N17 N6','原文献比较只在其已读来源和当时形式化范围成立；本轮不凭自然性能宣称新颖性。','none; no new exhaustive literature claim'),
13:('NOVEL_MECHANISM_NOT_ESTABLISHED','WEAK','historical N17/N18;new finite results','新自然现象未自动形成可识别的新机制或方法。先区分标准重要性采样、优化器条件效应和端点效用。','mechanisms;transfer'),
14:('NO_METHOD_CLAIM','WEAK','V2;AC-2D','自然结果存在阶段与损失冲突，未确立普遍占优候选；历史 NO_GO 不因单个正均值解除。','all remaining stages'),
15:('IDENTITY_RETAINED','STRONG','data protocol;core fixed-point design','同一人口同alpha时重要性校正保持同经验目标；不同人口的经验目标本身不同。','core finite/MC numerical checks'),
16:('LATE_ONLY_EXPLANATION_UNSUPPORTED','MEDIUM','historical V2/V9;new core common points','早期、后期与自选效应需分别描述；不能把所有 q 差异称过拟合之后才出现。','AC-V9 trajectory;N17'),
17:('FINITE_BUDGET_BOUNDARY','MEDIUM','AC-2D','自然1000步与250步的算法差异不一致；arch频率比也变，不能称纯时钟因果。','N17 H100/conditional H200'),
18:('HISTORICAL_CONDITIONAL_RESULT_RETAINED','MEDIUM','historical V3 ORDER','同multiset置换顺序可改变有限训练结果的原观察保留，不推为课程学习必然有益。','none; no natural ORDER grid prescribed'),
19:('STRUCTURAL_IDENTIFICATION_BOUNDARY','STRONG','historical V3','固定曝光 E 和独立覆盖 U 时重复数 R=E−U，不能三者完全独立。','N17 matching describes coupled changes'),
20:('STRUCTURAL_IDENTIFICATION_BOUNDARY','STRONG','historical V3','coverage 与 exposure/repetition 的约束仍在；支持量对照不等于独立控制全部三者。','N17 matching'),
21:('NOT_IDENTIFIED','WEAK','historical Stage2C/N17','标量trace不能保证解释方向风险；完整坐标协方差与性能中介仍须分开。','core mechanisms;N17 N3'),
22:('NOT_IDENTIFIED','WEAK','historical N17','C/R 局部敏感性不等于 q→长期风险的必要中介。','N17 N3'),
23:('CONDITIONAL_INTERACTION_RETAINED','MEDIUM','V2;AC-V9','Adam/SGD、LR与阶段之间结果不同；跨协议交互不能单归优化器类别。','AC-V9 trajectory;N17 LR grid'),
24:('NO_UNIQUE_ENVIRONMENT_ATTRIBUTION','WEAK','AC-2D controls','同N自然比例已改变差距，但特征/标签支持仍改变；不能唯一定位dataset identity。','N17;transfer controls'),
25:('HISTORICAL_LITERATURE_SCOPE','WEAK','historical N17 N6','已有工作覆盖范围按原审查保留；实测端点与机制新颖性分开。','none'),
26:('OBJECTIVE_DISTINCTION_RETAINED','STRONG','historical V3/N17','改变目标的D3/SAF与保持alpha的q干预并非相同估计对象。','none'),
27:('HISTORICAL_CONDITIONAL_RESULT_RETAINED','MEDIUM','historical V3 AdaTask','固定分量optimizer状态不数学消除q依赖；不把未做自然AdaTask复制写成新阴性。','none'),
28:('DETACH_BOUNDARY_RETAINED','STRONG','historical Stage2C;core mechanism design','原stop-gradient路由贡献和重新连接的函数风险Jacobian是不同对象。','core mechanisms'),
29:('FUNCTIONAL_PARTITION_RETAINED','STRONG','archived source;core partitions','参数名与真实前向支持分列，A_arch与main更新块分开。','core mechanisms'),
30:('LR_GRID_BOUNDARY','WEAK','historical N17;new N17 design','有限网格边界赢家不能证明全局最优；禁止结果后无限扩网格。','N17 all pilots/formals'),
31:('OPTIMIZATION_SEED_SEPARATION','STRONG','new N17 manifest','pilot2030/2031与formal2022–2025分离；同一数据总体与旧test仍复用，不是独立总体确认。','N17 actual receipts'),
32:('HISTORICAL_ARITHMETIC_RETAINED','STRONG','historical V3 fixed SGD','旧数值变化的选择/评价口径解释保留，不重写旧数据；自然比较单独列。','none'),
33:('TUNING_ESTIMAND_SEPARATED','STRONG','new N17 design','common-LR、own-LR及完整q×LR四格分别报告；不由仅SN调参推q算法优劣。','N17 pilots/formals'),
34:('HISTORICAL_TELEMETRY_FIX_RETAINED','STRONG','historical V3','旧一阶/二阶遥测修订保留，不把原混合范数继续当m单独因果。','new true optimizer telemetry'),
35:('OBSERVATION_CADENCE_BOUNDARY','MEDIUM','new complete-step core observations','训练实际step与读取间隔分列，原稀疏lag不是逐步状态因果。','core mechanisms'),
36:('MULTIPLICITY_BOUNDARY','STRONG','analysis protocol','四seed点态t区间及众多cell比较未提供全族错误率/总体确认。','none'),
37:('NO_CURRICULUM_CLAIM','STRONG','historical ORDER design','IID批次排序实验不等价于已学到有效课程策略。','none'),
38:('ORDER_TAIL_CONFOUND_RETAINED','STRONG','historical V3','ORDER移动短尾批次的混合干预保留，不能单归几何。','none'),
39:('HISTORICAL_PREFIX_RESULT_RETAINED','STRONG','historical V3','历史前缀数值复用的配置内事实保留；新运行不靠不同预算等价假设省略训练。','none'),
40:('JOINT_STATE_INTERVENTION_BOUNDARY','WEAK','historical S5/V3','状态消融会共同改变健康、幅度、有效进度；不能说分离了m必要性。','AC-S5;AC-S6;core diagnostics'),
41:('WINDOW_ALIGNMENT_REQUIRED','MEDIUM','historical V9;new common points','自然E10差+.000761四正、E40−.016065三负、selected−.001615三负，说明阶段必须对齐；完整窗口补读中。','AC-V9 trajectory'),
42:('NORM_DIRECTION_NOT_SEPARATED_BY_GLOBAL_NORM','STRONG','historical N17','全局等范数不能分离子块方向、相对尺度与交互。','N17 N3'),
43:('NEW_POPULATION_FIXED','STRONG','DATA_PROTOCOL_AUDIT;frozen population hashes','新优化seed不改变人口；旧V2/V9随seed改变人工子集。新旧种子方差组成不同。','none'),
44:('HISTORICAL_DESTRUCTIVE_BOUNDARY','MEDIUM','historical V3','破坏性控制改变共同早期窗口的原边界保留；无自然对应运行不假报复制。','none'),
45:('COMPLETE_BUDGET_REQUIRED','STRONG','new manifests;runner ledger','新所有四seed按完整预算，H200仅预定平台门控；未完成时不做正式结论。','N17;transfer'),
46:('AUC_LOSS_CONFLICT_RETAINED','MEDIUM','V2;AC-2D','自然AC-2D final EN−SN AUC+.000973但LL+.012372四差；排序改善不保证概率风险改善。','N17 N3;transfer'),
47:('METRIC_ARITHMETIC_AND_POPULATION_SEPARATED','STRONG','core prediction recomputation','同预测的评价权重差可精确分解；跨人口模型差仍非权重单因素干预，加权cell AUC不是pooled。','N17 metric tables'),
48:('PENDING_NATURAL_SYMMETRIC_GRID','WEAK','historical N17;new design','旧ML own/common selected负而own E100正；完整自然q×LR与独立pilot尚须执行。','N17'),
49:('PENDING_NATURAL_COUNT_DECOMPOSITION','WEAK','historical N17','KR count trace占比极小的局部结果保留；新ML须完整坐标及quota长期对照，不由trace预测AUC。','N17 N2/quota'),
50:('PENDING_NATURAL_LOCAL_INTERVENTIONS','WEAK','historical N17','旧局部C效果小且方向混合；双路径双bank、norm/direction交互及MC精度门控按新设计执行。','N17 N3'),
51:('INSTANTANEOUS_NOT_HISTORY_NECESSITY','WEAK','historical N17;new diagnostic designs','同theta/m/v的非线性均值更新差与长期历史必要性是不同问题。','core mechanisms;N17 N3'),
52:('HISTORICAL_LITERATURE_SCOPE','WEAK','historical N17 N6','保留既有8篇来源限定结论，不冒称覆盖所有后续文献或建立新颖性。','none'),
53:('KR_PRECISION_STILL_INCONCLUSIVE','WEAK','historical N17 N1','KR无动量SGD四/八种子原CI仍不足以证明±.003等价；新增ML不能解决KR精度，不按显著性无限补seed。','none; KR not rerun'),
}
assert len(old['questions'])==len(assessments)==53
rows=[];merged=[]
for question in old['questions']:
    number=int(question['id'][1:]);status,strength,evidence,answer,pending=assessments[number]
    old_eval=question.get('new_evaluation',question.get('final_evaluation',{}))
    if not old_eval and number>=47:old_eval=question.get('final_disposition',{})
    row=dict(id=question['id'],question=question['question'],historical_status=old_eval.get('status',question.get('status','see preserved original JSON')),
        current_status=status,evidence_strength=strength,current_evidence=evidence,
        current_assessment=answer,required_natural_work_remaining=pending,overall_natural_study_complete=False)
    rows.append(row);merged.append(dict(historical_record=question,natural_reassessment=row))
pd.DataFrame(rows).to_csv(OUT/'N17_question_crosswalk.csv',index=False)
(OUT/'N17_question_crosswalk.json').write_text(json.dumps(dict(status='INTERIM_NOT_FINAL',
    all_53_historical_records_preserved=True,historical_fields_overwritten=False,questions=merged),ensure_ascii=False,indent=2)+'\n')
print('53 original questions preserved; new assessment INTERIM, unfinished stages explicitly pending')
