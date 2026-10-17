from pathlib import Path
import json,re,time
O=Path('/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016');load=lambda p:json.loads(p.read_text());q=load(O/'question_registry.json');dec=load(O/'final_scientific_decision.json');assert dec['status']=='FINAL_APPROVED_AFTER_ALL_REQUIRED_RESULTS'
minimum={
'Q4':('四seed三状态最终250/150/100/200；12最终+27历史配对、12影子同θ/范数控制与1740方向行均完成。','整体更新映射与步幅/进度/块间尺度联合变化，状态最终预算与native不同且有floor；没有唯一长期历史必要性识别。'),
'Q5':('Adam/SGD全部WR/PERM及SGD ORDER四seed最终训练/采样身份、完整几何与任务方向均完成。','重复/覆盖/条件依赖联动，shape与trace及后续θ共同变化，不能唯一归因非Adam中介。'),
'Q6':('12规定同θ局部×7条件全完成；C E10/E40 AUC阳性，E40 LL/J冲突，原联合完整训练条件未触发。','局部单draw直接作用不等于长程架构必要性；没有事后放宽loss一致条件。'),
'Q9':('真实域任务N/正例率/曝光/六cell支持审计、2x/4x别名与三mask强度完整四seed控制完成。','3自然域的N/p/输入/标签率/基线难度共同变化，不能单独识别task/domain imbalance。'),
'Q19':('全部四seed两optimizer WR/PERM最终采样/coverage/repeat身份及性能完成，SB小域E>N已实计。','固定E,N使R=E−U≥max(0,E−N)，不存在同配额同支持的零重复控制，PERM残差不能排除必要性。'),
'Q20':('4200最终臂轮的unique/repeat/exposure、first-seen与累计coverage及采样操纵核验完成。','固定曝光与支持下repeat和unique coverage受代数约束，当前设计不能独立改变coverage而保持repeat。'),
'Q21':('16固定θ×512/q完整谱、320移动θ/块记录、8完整任务梯度及下游控制分析完成。','q同时改变trace/shape/direction/高阶矩/θ；没有独立trace-matched shape操纵，相关谱不识别额外因果解释力。'),
'Q22':('同Q6全部12局部及共享支持审计完成，C AUC阳性、原完整联合条件未触发；R已私有。','C局部路径不是唯一长期必要性；已私有R不能伪装成共享R私有化反事实。'),
'Q42':('12同θshadow/raw与全局等范数控制、1740方向/块幅度记录和12检查点replay完成。','全局一个范数约束不固定CPRE块尺度；swap_v的R/P/E仍约1.48/1.35/1.18，纯角度作用无法隔离。')}
for x in q['questions']:
 if 'next_action' in x:
  x['historical_pre_final_next_action']=x['next_action']
  x['next_action']='本轮规定且已触发的实验和冻结test均已完成；更强新主张仅按保留的最小实验另行预注册。'
 if x['id']=='Q3':
  x['previous_evidence_strength']=x['evidence_strength'];x['evidence_strength']='MEDIUM';x['strength_scope']='明确配置/四seed反例的经验强度为MEDIUM；不是跨配置总体效应STRONG或p<.05。'
 if x['id']=='Q23':x['strength_scope']='STRONG指冻结触发判据数值成立的可追溯性，不是总体optimizer机制因果强度。'
 if x['id']=='Q34':x['conclusion']='原exp_avg是sqrt(||m||²+||v||²)，8个实际Adam/AdaTask状态块独立核验后，最终254归档快照756800行全部刷新，compact1016组；序列化范数公式最大相对误差3.324e-11，原mixed与corrected m最大相对差99.973%。训练更新/预测/指标不受影响，无需重训。'
 if x['id']=='Q45':x['final_actual_support_verification']='冻结v2已实际核验全部14cohort，n1–4按真实支持交集，部分固定100仍仅预算资格条件描述；未补造已停止seed。'
 if x['id'] in minimum:
  x['completed_minimal_experiment'],x['why_current_design_cannot_identify']=minimum[x['id']];x['blocks_limited_final_conclusion']=False;x['blocks_stronger_claim']='唯一或长期必要中介/独立因果N等未识别强主张，范围见当前问题。'
 if x['id']=='Q14':x.update(status='RESOLVED',decision=dec['method_decision'],evidence_strength='MEDIUM',evidence='V13_final_method_decision.md; final_scientific_decision.json',conclusion=dec['method_reason'],remaining_uncertainty='不等于所有未来方法/调参不可能；本轮不从test提出新候选。',next_action='本轮不进行新方法开发；规定实验/报告/归档推送全部完成后结束。')
assert all(x['status'] in ['RESOLVED','FALSIFIED','BOUNDARY','NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN'] for x in q['questions']);q['historical_field_scope']='observation/origin/test_access等字段保留问题提出时的历史语境，不表示最终尚未完成；冻结test现已全部完成，旧next_action已单独保留。';q['status']='FINAL_ALL_46_QUESTIONS_TERMINAL';q['time_finalized']=time.time();q['new_questions_count']=len(q['questions'])-27;q['scientific_classification']=dec['classification'];q['method_decision']=dec['method_decision'];(O/'question_registry.json').write_text(json.dumps(q,indent=2)+'\n')
titles={}
for line in (O.parent/'QUESTION_REGISTRY.md').read_text().splitlines():
 if re.match(r'\| Q\d+ \|',line):
  parts=line.split('|');titles[parts[1].strip()]=parts[2].strip()
for line in (O/'Newly_Exposed_Questions_WORKING.md').read_text().splitlines():
 m=re.match(r'## (Q\d+)：(.*)',line)
 if m:titles[m[1]]=m[2]
answers={'Q1':'明确配置内同目标q改变路径/早期性能；不普遍外推。','Q2':dec['selection_answer'],'Q3':'Adam通用必要性被无动量SGD反例反驳。','Q4':'状态映射有条件影响，唯一长期历史机制未识别。','Q5':'非Adam候选路径未唯一分离，最小采样/几何控制完成。','Q6':'C局部AUC阳性但联合loss失败；长期必要性未识别。','Q7':'精确别名改变名义池而不改变WR规律，未操纵新独立支持。','Q8':'目标AUC对mask敏感，loss/阶段/q不单调；不是自然困难度唯一因果。','Q9':'N/p/标签/输入/基线共同变化，不能单独归于imbalance。','Q10':'规则/评价人口/预算池会改变结论，所有预定规则均保留。','Q11':'第二环境没有稳定同向宏观损害，保留配置束/阶段边界。','Q12':'D3/SAF目标不同、AdaTask固定组件有残差，一般主题部分覆盖。','Q13':'没有识别独特必要中介；既不宣称穷尽覆盖，也不强称创新。','Q14':'十项GO条件未同时满足；NO_GO。','Q15':'同forward J及同图导数字段期望身份成立；保留detach边界。','Q16':'明确配置在操作性pre阶段已有差值，不能归为全部late-only。','Q17':'LR/batch/预算与clock改变响应，既不全确认也不全排除速度解释。','Q18':'同multiset的整体顺序干预改变阶段响应，长预算反向保留。','Q19':'固定E,N零重复不可行，PERM不能单独识别必要重复。','Q20':'R=E−U使覆盖和重复不能在当前资源下独立识别。','Q21':'谱描述有信息，shape独立于trace的必要性未识别。','Q22':'同Q6；C局部路径阳性不等于长期必要；R原已私有。','Q23':'calibrated共同22交互满足预定触发；fixed reference未触发。','Q24':'窗口/步钟/数据标签/训练子集等配置束边界已分解，非单一N。','Q25':'通用sampling/order/memory部分覆盖，不构成MDMT创新证明。','Q26':'完整D3/SAF学习权重一般改变J，不是同目标机制等价控制。','Q27':'固定同目标AdaTask组件未消除q敏感性，不推出全论文失败。'}
lines=['# 全部问题终局闭合表','','全部46项均为允许终态；不是把未做实验标不可识别。所有Q28–Q46共19项的原始观察、六字段与下一最小实验完整保留于[FINAL_REPORT第13章](FINAL_REPORT.md)。机器可读的每项证据和边界见[question_registry](question_registry.json)。','','|ID|问题|状态|证据强度|回答/边界|','|---|---|---|---|---|']
for x in q['questions']:
 answer=answers.get(x['id'],'原始观察及当前边界按终局报告第13章逐项保留。');lines.append(f"|{x['id']}|{titles.get(x['id'],x.get('question',''))}|{x['status']}|{x['evidence_strength']}|{answer}|")
lines+=['','## 不可识别项目的已完成最小实验及原因','']
for x in q['questions']:
 if x['status']=='NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN':
  assert x['id'] in minimum;lines+=['### '+x['id'], '', '- 已完成：'+x['completed_minimal_experiment'],'- 当前不可识别原因：'+x['why_current_design_cannot_identify'],'- 阻塞范围：只阻塞更强唯一/必要/独立因果主张，不阻塞有限配置边界结论。','']
lines+=['额外2026–2029的触发条件未满足：没有用覆盖±.003的关键null等效关闭核心。C的AUC阳性与loss冲突是Q46的真实反例，不是以不显著替代机制不存在。所有条件未触发分支和最小控制都已记录；最终Git推送和远程验证后结束任务。']
(O/'final_question_closure.md').write_text('\n'.join(lines)+'\n');print('All46terminal; NIminimumactualproofs explicit; all19newpreserved')
