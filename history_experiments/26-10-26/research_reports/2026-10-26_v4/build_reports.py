"""Assemble compact research reports from immutable prediction/decision data."""
from pathlib import Path
import collections,csv,hashlib,json,shutil,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/xzf/Code/phase0-3/artifacts/mdmt_dfd_v4/v4_0')
TABLES=HERE/'tables';EVIDENCE=HERE/'evidence';FIG=HERE/'figures'
for p in [TABLES,EVIDENCE,FIG]:p.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def write(name,text):(HERE/name).write_text(text.strip()+'\n')
def md(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def csvrows(p):return list(csv.DictReader(Path(p).open()))
def fp(v,n=9):return f'{v:+.{n}f}'
def f(v,n=9):return f'{v:.{n}f}'
def copy(src,dst):shutil.copyfile(src,dst)

freeze=read(ROOT/'FROZEN_EVALUATION_SPEC.json');execution=read(ROOT/'FINALIZATION_EXECUTION_REPORT.json')
decision=read(ROOT/'DECISION_P2.json')['datasets']['movielens'];refs=decision['comparisons']
methods=['NATIVE','LOCAL','H_DIRECT','ETA_DIRECT','UNIFORM','GEOM','SHUFFLE','CONST_UNIFORM','LOCAL_CAP','DFD']
roles=['Q_explore','validation','test_exposed','Q_legacy_confirm_exposed']
for ds in ['movielens','kuairand']:
    for name in ['macro.csv','six_cells.csv','paired_seed_statistics.json']:
        copy(ROOT/'analysis/p1'/ds/name,TABLES/f'P1_{ds}_{name}')
for role in roles:
    for state in ['selected','read_pass10_or_fixed']:
        for name in ['macro.csv','six_cells.csv','paired_seed_statistics.json']:
            copy(ROOT/'analysis/final/movielens'/role/state/name,TABLES/f'P2_{role}_{state}_{name}')
    for ds in ['movielens','kuairand']:
        for name in ['macro.csv','six_cells.csv','paired_seed_statistics.json']:
            copy(ROOT/'analysis/final'/ds/role/'prefix'/name,TABLES/f'PREFIX_{ds}_{role}_{name}')
for name in ['macro.csv','six_cells.csv','paired_seed_statistics.json']:
    copy(ROOT/'analysis/p2/movielens/Q_explore/selected'/name,TABLES/f'P2_REGISTERED_Q_{name}')
for path in (ROOT/'analysis/support').iterdir():
    if path.suffix in ['.csv','.json']:copy(path,TABLES/f'SUPPORT_{path.name}')
for path in (ROOT/'analysis/costs').iterdir():
    if path.suffix in ['.csv','.json']:copy(path,TABLES/f'COST_{path.name}')
for name in ['DECISION_P1_FINAL.json','DECISION_P2.json','FROZEN_EVALUATION_SPEC.json','FINALIZATION_EXECUTION_REPORT.json','INTERMEDIATE_CLEANUP.json','REVIEW_OUTER_DEMAND_GROUP_AUDIT.json','FINAL_BENCHMARKS_COMPLETE.json']:
    if (ROOT/name).exists():copy(ROOT/name,EVIDENCE/name)
for ds in ['movielens','kuairand']:
    copy(HERE/'reviewer_p0'/f'{ds}_DATA_ROLE_MANIFEST.json',EVIDENCE/f'{ds}_DATA_ROLE_MANIFEST.json')

bootstrap_rows=[]
for p in sorted((ROOT/'analysis').rglob('*bootstrap1000.json')):
    a=read(p);kind='P1' if '/p1/' in str(p) else 'P2';role=a.get('role','P_form_OOF')
    dataset=a.get('dataset',p.parent.name if kind=='P1' else p.parents[2].name)
    seed=a.get('seed',int(p.name.split('_')[0][4:]));unit=a.get('cluster',a.get('unit'))
    state=p.parent.name if kind=='P2' else 'OOF'
    for ref,c in a['comparisons'].items():
        for metric,ci in c['percentile95'].items():
            invalid=c.get('invalid_resamples',c.get('invalid_resamples',0))
            if isinstance(invalid,dict):invalid=invalid[metric]
            bootstrap_rows.append(dict(stage=kind,dataset=dataset,role=role,state=state,seed=seed,reference=ref,metric=metric,lower=ci[0],upper=ci[1],replicates=1000,unit=unit,invalid_resamples=invalid,source=str(p.relative_to(ROOT))))
with (TABLES/'BOOTSTRAP_CI.csv').open('w') as h:
    w=csv.DictWriter(h,fieldnames=list(bootstrap_rows[0]));w.writeheader();w.writerows(bootstrap_rows)

def point(role,method,seed):return read(ROOT/'final_evaluation/movielens'/f'seed{seed}'/'p2'/method/'selected'/f'{role}.metrics.json')
def pair(role,ref):
    ll=[];aa=[]
    for s in [5101,5102]:
        a,b=point(role,'DFD',s),point(role,ref,s);ll.append(a['macro_logloss']-b['macro_logloss']);aa.append(a['macro_auc']-b['macro_auc'])
    return ll,aa
rawrows=[]
for method in methods:
    a,b=point('Q_explore',method,5101),point('Q_explore',method,5102)
    rawrows.append([method,f(a['macro_logloss']),f(a['macro_auc']),f(b['macro_logloss']),f(b['macro_auc'])])
gaterows=[]
for c in refs:
    gaterows.append([c['reference'],fp(c['ll_gains'][0]),fp(c['ll_gains'][1]),fp(c['mean_ll_gain']),fp(float(np.mean(c['auc_deltas']))),f(c['worst_mean_cell_ll_delta']),'PASS' if c['PASS'] else 'FAIL'])
role_rows=[]
for ref in ['NATIVE','H_DIRECT','ETA_DIRECT','UNIFORM','GEOM','SHUFFLE','LOCAL_CAP']:
    vals=[]
    for role in roles:
        ll,aa=pair(role,ref);vals.append(fp(float(np.mean(ll))))
    role_rows.append([ref,*vals])

write('FULL_PIPELINE_REPORT.md',f'''# 完整流水线：MovieLens P2 与冻结后角色比较

ML两开发seed的十个规定输出全部完成（20/20）。native来自本seed预定validation LL-selected状态，pilot固定12pass，五种形成库固定12pass；软reader为P_form20→P_read10，整组六头只由P_cal选OFF/pass0–10。H_DIRECT/ETA_DIRECT为完整强概率对照，CONST_UNIFORM通过固定simplex近优验收。没有因为先完成某个对照而取消剩余输出。

## 注册开发主表：Q_explore，所选预测器

{md(['输出','5101 LL','5101 AUC','5102 LL','5102 AUC'],rawrows)}

## 九项配对晋级，逐项公开

G=reference LL−DFD LL（正有利），ΔAUC=DFD−reference；最差格列严格先对同一cell跨seed平均再取max。

{md(['reference','G5101','G5102','平均G','平均ΔAUC','max mean ΔLL cell','全部条件'],gaterows)}

仅LOCAL、CONST_UNIFORM、LOCAL_CAP三项整体通过；其余六项失败，**P2 FAIL，P3/P4 NO_START**。全部AUC保护通过；H_DIRECT、ETA_DIRECT同时超过.003的逐格LL保护，最差cell均为domain0/task0。非退化条件为真，不能把负结果归为全OFF或责任恒定。

DFD对UNIFORM的G为+.000002130/+.000052687，均值+.000027408，远低于注册.0005资源尺度。对GEOM、SHUFFLE方向不稳定；对两直接概率头的LL两seed都更坏。DFD胜LOCAL/LOCAL_CAP时仍未稳定超越NATIVE；LOCAL自身在Q上的LL比NATIVE两seed更坏，胜过它部分是在弥补额外pilot阶段的退化。

## 冻结后的伴随角色

下面全为平均ΔLL=DFD−reference，负有利。test/legacy均为历史暴露数据，不重新触发训练，也不覆盖开发失败。

{md(['reference','Q_explore','validation','test_exposed','legacy exposed'],role_rows)}

test_exposed中DFD对H_DIRECT的ΔLL为+.010269921/+.007682428，对ETA_DIRECT为+.011033980/+.007546972，且AUC相对两者也两seed下降。与UNIFORM的LL方向在四角色两seed均略好，但规模仍小，AUC在Q/legacy不一致；不能宣布等效，也不能据这些小收益翻转原晋级规则。GEOM的平均LL在四角色均低于DFD。SHUFFLE的配对LL跨seed/角色反向，特定样本需求对应关系没有稳定增量。

Q上DFD相对H_DIRECT的AUC两seed略高，test上却两seed更低；风险/排序与角色变化必须同时报告。两个直接概率对照不受五候选概率凸包约束，也实质使用同一目标标签；该竞争解释仍成立。ETA_DIRECT含source pilot特征，H_DIRECT只额外读完整h/native logits，native本身已有多域训练历史。

所有完整六格、domain/task、natural LL、pooled AUC和seed t95见[tables](tables/)。每seed/每角色/所选与固定read10状态均按原始用户1000次共同权重bootstrap，所有cell和macro重新计算；与正式4seed确认无关。fixed read10表的NATIVE/LOCAL/常量项保留固定定义，原生epoch24另见PREFIX表；不能把不同阶段终点当成共同训练时点。

![九对照开发增量与暴露基准伴随结果](figures/P2_comparisons.svg)

## 机制信息边界

PF OOF分组、source责任/支持、全部raw预测及权重摘要存在。Q/outer冻结μ分组遗漏，见[缺项登记](PROCESS_REVISIONS.md)及审查附录，不能声称该机制分层已完成。它不改变本页注册门控或直接对照的原始结果。P3函数替换/U_BUDGET/POOL未触发，均未运行，不由删除某路径或参数距离补造功能证据。
''')

write('ATTRIBUTION_REPORT.md',f'''# 归因：需求可读、函数形成与完整增量

## H1：可预测需求

ML的PF OOF需求策略在两个seed均胜常量，六格均非负，P1通过；KR完整h版本在两个seed都输常量和LOGIT，P1失败。证据分别是**MEDIUM**的有限开发支持/反例，不能用ML替代KR，也不能把KR结果解释为所有条件需求不存在。LOGIT在KR仍有两seed LL收益，但伴随AUC下降。详细数值和不确定性在[P1报告](P1_DEMAND_REPORT.md)。

## H2：需求跨支持指导内容形成

ML责任确实变化，DFD每行/任务两项和2最大误差约5.96e-8；source0/task0的责任SD为.5488/.5703（seed5101/5102），不是恒定权重或全OFF。来源函数独立受各自source标签监督，初始化微扰/row plan在匹配臂相同，新的Adam状态、缺源None语义均经实际验收。责任质量、ESS、clip率、优化器step/moment摘要完整保留在原始JSON和[支持表](tables/COST_formation_support.csv)。

责任两项相关−1由和2约束必然产生，不能当互补知识证据。ESS是加权支持集中度，不是独立样本数。正权weighted BCE在充分容量/优化下仍有相同source Bayes目标；函数身份和非零权重并不保证不同功能。

对UNIFORM的开发平均LL收益只有.0000274，GEOM/SHUFFLE未被稳定超过。因此特定需求形成的独立收益为**WEAK/本版未获支持**。不能把source外推错、容量有限、优化路径或凸包约束中的任何一项指定为已经识别的根因。尤其Q/outer固定μ分层缺失，外层需求迁移原因保留为MISSING信息，未用ETA_DIRECT微调输出替代。

## H3：完整方法的实际价值

ML完整流水线未胜两个强直接概率头；最小域task0的LL代价还超保护门槛。H3本版为**VERSION_NOT_SUPPORTED**，有限反例证据**MEDIUM**。DFD胜LOCAL、LOCAL_CAP与常量读取仍可保留，普通来源监督/条件读取、追加标签拟合或概率范围约束已足以竞争。P3与P4均NO_START，不声称已验证标签预算桥接、pooled来源必要性、外部领先或四seed重复性。

## 规模、难度与阶段

ML P1三域LL/AUC均改善。P2相对NATIVE，最小域0的LL在两个seed恶化，较大域1/2改善；相对H_DIRECT，小域与最大域的AUC都可改善而中域下降，LL三个域都变差。没有稳定可推广的“小域改善/大域下降”规律。KR最小域2并非最低基线AUC域。规模与域语义/基率/支持混杂，作为主因的证据**WEAK**。

四个原生模型的epoch24在test LL均差于自己的合法selected；ML Q/legacy的晚期AUC却升而validation/test下降。不同角色的人口支持与模型阶段提供竞争解释，不能把晚期较稳等同最佳质量，也不能据观察断言Adam或共享机制的普遍根因。

## 最有判别力的下一项问题

先在真正未暴露的自然窗口锁定比较H_DIRECT、ETA_DIRECT与普通UNIFORM来源读取，维持目标标签与完整流程计费；检验来源函数是否仍有强简单方法之外的增量。若没有，停止进一步细分形成。当前不再追加K、元学习、正则、seed或人工领域比例搜索。该问题是未来建议，本轮未执行。
''')

research=read(ROOT/'analysis/costs/research_actual_recorded.json');standalone=csvrows(ROOT/'analysis/costs/standalone_known_components.csv')
costrows=[]
for method in methods:
    rr=[x for x in standalone if x['method']==method];costrows.append([method,*[f(float(x['reconstructed_known_seconds']),3) for x in rr]])
benchrows=[[x['dataset'],x['arm'],f(x['median_seconds']*1000,4),f(x['p95_seconds']*1000,4),x['parameters']] for x in execution['benchmarks']]
write('COST_REPORT.md',f'''# 成本与信息使用

## 实际研究执行账

四条必需native24轨迹、四套pilot/P1以及ML二十个P2逻辑输出全部完成，无P3/P4。记录4062个不重叠训练/选择/开发leaf，共 **{research['known_training_and_development_leaf_seconds']:.6f}秒**；其活跃区间并集{research['leaf_active_interval_union_seconds']:.6f}秒。按36个实际进程首末记录计算，活动跨度和{research['process_activity_spans_sum_seconds']:.6f}秒、跨度并集{research['process_activity_spans_union_seconds']:.6f}秒。并发值定义不同，不能相加或称GPU小时/能耗。

主要leaf：native训练2029.606604秒、原生validation24.929963、缓存131.458023、pilot104.264690、η/LOGIT423.246784、P1_DIRECT224.112090、形成371.622553、reader P_form1334.271729、P_read514.452157；所有组件/数量在[CSV](tables/COST_training_components.csv)。冻结后184项raw-X预测/重建计26.581971秒，metric/I/O计25.192602秒，另记51.774572秒。分析bootstrap为CPU信息处理，另有实际秒数收据。

部分首次setup、责任/候选缓存、审查、包装器与I/O未独立journal；不将缺失当0。进程跨度覆盖部分间隙，但仍不是完整启动/收尾wall；精确研究总量为UNKNOWN，而非伪造一个精确总耗时。训练前包装器错误保留日志，均在训练前重试原配置；不是额外seed或负结果重跑。

## 独立部署重训的已记录组件账

同一prefix在实际账只计一次；下表按每方法必要依赖重新加回native/cache/pilot/η/库/reader的已记录组件，不加无关诊断。H_DIRECT不需要pilot；ETA_DIRECT不需要OOF/m/τ/形成；CONST_UNIFORM需要UNIFORM库但不需要其神经reader；DFD/SHUFFLE承担η folds/refit。表中组件是在共享GPU负载下测量，**不是各法独占GPU最短重训时间**，未记录部分仍UNKNOWN。

{md(['方法','5101已记录秒','5102已记录秒'],costrows)}

## 开发/选型与标签账

P1有144个神经fold-head拟合和48个常量target/task/fold拟合；每行只进入一折训练，η两fold20对应每个标签20训练passes。DFD P_form训练label-pass为η-fold20+refit20+reader20=60；SHUFFLE同样依赖需求。UNIFORM/GEOM/LOCAL_CAP/H_DIRECT为20，ETA_DIRECT refit20；各神经readout另P_read10。常量在PF∪PR的唯一行做凸拟合，不伪称同SGD更新数。统计/选择/重复读取另记，不是新增独立标签。开发账与独立部署/研究实际账重叠，不能相加。

## 完整模型推理实测

GPU0 RTX3090，seed5101，自然fit前4096行编码X已在GPU，完整raw-X native+下游模型，FP32网络、锁定概率精度，无AMP/TF32；5warmup后30次CUDA同步。ML10结构与KR2前缀均测，30次仅为延时重复。

{md(['数据','方法','P50 ms','P95 ms','实际注册参数'],benchrows)}

ML native实测772974参数，KR1614254；DFD额外200304，训练期η另50502，LOCAL_CAP额外299388。DFD P50约5.709ms，H_DIRECT约2.535ms；该实现的较高延时没有获得完整风险增量支持。参数数本身不作速度证据。完整sample/peak/设备负载与predictor hash均保留。

候选贡献数与实际执行需分开：LOCAL实际计算3个source pilot/task再gather，贡献1；soft OFF仍评5候选、跳gate。表中的静态candidate_functions字段不能当实测活跃量，亦无稀疏推理承诺。常量simplex系数作为固定数据保存，不全由nn.Parameter计数表示。

## 最终存储

按用户要求，模型/Adam与大中间缓存删除，原生数据及其他`/home_new`目录不动。清理时8,954,541,940→634,400,615字节，移除8,320,141,325字节；之后追加的分析表仍为信息文件。没有.pt/.pth或模型symlink。保留行级预测、日志、hash、normalizer/选择/支持信息和两个约10MB的固定责任数据包（不是神经参数）。最终预测可直接重算指标；重新前向需重训，不能声称权重永久留存。见[清理收据](evidence/INTERMEDIATE_CLEANUP.json)。
''')

audits={ds:read(EVIDENCE/f'{ds}_DATA_ROLE_MANIFEST.json') for ds in ['movielens','kuairand']}
countrows=[]
for role in ['fit','P_form','P_read','P_cal','P_update','Q_explore','validation','test_exposed','Q_legacy_confirm_exposed']:
    vals=[]
    for ds in ['movielens','kuairand']:
        x=audits[ds]['roles'][role];vals.append(x['n'] if isinstance(x,dict) else len(x))
    countrows.append([role,*vals])
checkpointrows=[]
for ds in ['movielens','kuairand']:
    for seed in [5101,5102]:
        ep=read(ROOT/'prefix'/ds/f'seed{seed}'/'epoch_metrics.json');chosen=min([x for x in ep if x['epoch']>=12],key=lambda x:(x['macro_logloss'],x['epoch']))
        pinfo=freeze['prefixes'][f'{ds}:{seed}'];checkpointrows.append([ds,seed,chosen['epoch'],pinfo['components']['native_hash'],pinfo['components']['epoch24_hash']])
write('DATA_AND_ROLE_AUDIT.md',f'''# 数据、角色、源码与选点身份

两raw CSV完整SHA与规范一致：ML `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`，KR `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。prepared x/y、fit-only词表/scaler/OOV、原始域/标签与固定整数角色协议均经独立完整核对，未改原始行或比例。ML8字段，KR34字段，完整schema/正负数/逐角色行hash在两个[evidence数据manifest](evidence/)。

自然fit域计数ML124759/254879/260495、KR138225/1395118/71561。角色如下；P_label是历史父池，不重复计入原子角色。

{md(['角色','MovieLens行数','KuaiRand行数'],countrows)}

fit用于native/pilot/source，P_form用于η和reader预拟合，P_read拟合最终读取，P_cal选完整六头，validation只选native。P_update主算法不用。Q_explore开发，test/legacy已暴露；无新数据确认，也没有6101–6104正式训练。任务ML为rating>3、rating>4；KR保留实际is_click/long_view，不从名称假设嵌套。

ML原始用户/物品ID可恢复，KR processed表不含可信原始ID，禁止用UNK或profile编码替代。ML原始用户OOV比例在validation约76.16%、Q约0%、test约19.32%；支持分布是竞争解释，不是原因识别。完整预测的support分析与原始行索引保留在artifact中，KR相关分层为MISSING并用row-bootstrap退路说明限制。

## 冻结选择与状态

{md(['数据','seed','selected epoch','native权重hash','epoch24权重hash'],checkpointrows)}

选择严格为validation macro LL的epoch12–24最小值/最早平局；24epoch全跑完，pilot/F固定pass12，reader/direct所选与pass10分别保存预测。选点、更新/库形成和终点不可混为同一阶段。prefix保留原FP32 sigmoid再FP64评价；P2保留原已锁FP64概率路径，未事后改变AUC ties。

fixed-fusion M3oE：exp_d/exp_t/bal_d/bal_t全1，4shared experts，fcn[512,256,256,64]。数据自然两任务Sample BCE，full40960物理/逻辑batch、短末批，coupled Adam .001/.9/.999/eps1e-8/wd1e-5；η/read wd0，无AMP/TF32，无复制/噪声/AutoML。h确为4shared专家逐个LN后拼接256，非STAR同维向量。canonical原始初始化后拷贝固定骨干，两schema真实前向/权限已审。

源码来自当前main独立分支；vendor7文件与pinned c5618e28 Git blob逐项相等，历史v3五报告与固定8a4b3b29逐项相等。前缀source1273530d…、P1需求77c80737…、P2 fc97ec1d…与最终all-source ca1ac6b7…分别准入，完整SHA在审查/冻结收据。未合并旧实验分支，未恢复有意移除的learned fusion/DARTS。

最终freeze `{freeze['freeze_hash']}`：46实际variants、184角色导出、252临时完整状态身份。全行raw-X重建最大误差{execution['maximum_raw_x_reconstruction_error']:.12g}≤1e-6，模型状态不变。模型与优化器最终按用户要求删除，identity/预测信息留存。Q/outer μ机制分组遗漏另列MAJOR，184完整性不能代替该分组完整性。
''')

missing='Q/outer原refitη的固定μ分组未导出；清理后不能由hash/摘要唯一恢复，不能用P_read后的ETA_DIRECT替代。'
write('OPEN_QUESTIONS.md',f'''# 开放问题、证据边界与缺项

1. **MAJOR / MISSING：{missing}** 它限制外层需求/支持迁移解释，不改变已存OOF/P2门控输入。两coder/reviewer已只读复核并补审查记录；依据start最小必要规则，不为补该机制信息重训关闭版本，不宣称全部机制规范完成。
2. PF同行标签折外不等于用户/物品隔离。ML H1支持只在当前表示、候选、人口与n=2开发seed内。
3. KR full-h需求估计失败、LOGIT仍有LL收益，原因可能是容量/估计方差/硬动作/优化或支持差异，当前没有单因素识别。
4. source外推没有无偏需求真值。低clip率、ESS充足、非恒定责任不证明η在source上有目标语义。
5. 正权weighted BCE不保证不同Bayes函数；不同参数/责任不能证明功能原型。DFD与UNIFORM增量小，GEOM/SHUFFLE可竞争，形成独有性未成立。
6. H_DIRECT/ETA_DIRECT强于完整混合风险，凸包约束、直接概率拟合和训练信息是竞争解释；不能把全部收益或失败称为单调校准。
7. η/reader重复P_form监督不同于新增独立标签；P3 U_BUDGET/POOL未触发，因此未完成标签预算与pooled必要性归因，不能假称已排除。
8. 部分域/任务、Q/validation/test和selected/终态方向反向，必须保留。规模与域身份、基率、编码支持和优化stage混杂，未识别为主因。
9. n=2训练区间较宽；1000bootstrap重复、六格、角色或fold不是额外seed。KR缺原始用户ID，只能作独立性未证的row退路。
10. test/旧Q_confirm已经暴露；冻结后评分仍是历史基准复评。P4未触发，未训练新6101–6104、STEM或AutoIFS，不宣称正式四seed支持或外部领先。
11. 真实完整推理已测，但训练成本来自共享GPU负载；部分setup/I/O等分量未单独journal，不把已记录秒数冒充精确独占GPU总成本或能耗。
12. 用户明确不留模型，原始完整张量已删除。指标/预测可复算，未来重新推理需按源代码重训；哈希不能代替模型。缺失的机制导出不会被摘要伪造。
13. 旧RHD四个固定锚均缺完整工件，新增分叉更新0；旧日志的保守动作频率与晚期减损是观察关联，不能识别频率或共享/私有写入原因。
14. 文献近邻PRP等仍有既有全文/复现缺口。本版性能主张已经未获支持，不用模块命名、非零梯度或首创叙事替代证据。

最有区分力的下一项工作是在真正新、自然且封存的数据窗口，对强直接概率与普通来源读取作完整同标签流程比较，先重建来源监督的独立增量；不继续当前版本的无边界超参/K/seed搜索。本轮不执行该未来研究。
''')

# Small standalone vector figures; all numerical summaries remain in CSV.
plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(9,3.3))
p1=read(ROOT/'DECISION_P1_FINAL.json')['datasets']
for ax,(ds,title) in zip(axs,[('movielens','MovieLens'),('kuairand','KuaiRand')]):
    vv=[x['gain_const'] for x in p1[ds]['rows']];ax.bar(['5101','5102'],vv,color=['#287d8e','#d78d35']);ax.axhline(0,color='black',lw=.7);ax.axhline(.0001,color='gray',ls='--',lw=.8);ax.set_title(title);ax.set_ylabel('CONST LL - DEMAND LL');ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3));ax.set_xlabel('Training seed (n=2)')
fig.tight_layout();fig.savefig(FIG/'P1_demand.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4.3))
for ax,role,title in zip(axs,['Q_explore','test_exposed'],['Registered development comparison','Historically exposed companion (descriptive)']):
    for j,ref in enumerate([c['reference'] for c in refs]):
        if role=='Q_explore':v=next(c['ll_gains'] for c in refs if c['reference']==ref)
        else:v=[-x for x in pair(role,ref)[0]]
        ax.plot(v,[j-.1,j+.1],'o',ms=4,color='#287d8e');ax.plot(np.mean(v),j,'D',ms=4,color='black')
    ax.set_yticks(range(9),[c['reference'] for c in refs]);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8)
    if role=='Q_explore':ax.axvline(.0005,color='#aa5555',ls='--',lw=.8)
    ax.set_xlabel('Reference LL - DFD LL (positive favors DFD)');ax.set_title(title);ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(FIG/'P2_comparisons.svg');plt.close(fig)

# Retained-file index has no row-level observations or tensor parameters inline.
index=[]
for rec in freeze['predictors']:
    base=ROOT/'final_evaluation'/rec['dataset']/f"seed{rec['seed']}"/rec['stage']/rec['arm']/rec['variant']
    for role in freeze['roles']:
        receipt=base/f'{role}.receipt.json';j=read(receipt)
        index.append(dict(dataset=rec['dataset'],seed=rec['seed'],stage=rec['stage'],arm=rec['arm'],variant=rec['variant'],role=role,predictor_hash=rec['predictor_hash'],prediction_path=str((base/f'{role}.npz').relative_to(ROOT)),prediction_sha256=j['prediction_sha256'],receipt_path=str(receipt.relative_to(ROOT)),receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()))
(HERE/'ARTIFACT_INDEX.json').write_text(json.dumps(dict(artifact_root=str(ROOT),freeze_hash=freeze['freeze_hash'],exports=index,p1_oof=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted((ROOT/'prefix').glob('*/*/p1_oof.npz'))],formation_information=read(ROOT/'information_data/INDEX.json'),raw_arrays_and_models_uploaded=False,model_retention='Deleted by user instruction',missing_outer_mu_groups=missing),ensure_ascii=False,indent=2)+'\n')

write('FINAL_REPORT.md',f'''# DFD-v4.0 完整阶段研究报告

**结论：当前DFD-v4没有获得强简单对照之外的完整方法支持，按预注册规则结束。** ML需求可预测但P2形成/完整增量未晋级；KR在P1即失败。所有已触发训练、184个冻结预测导出、重建与12结构推理计时已完成；P3/P4及正式/外部训练均NO_START。另有一项规定机制信息导出遗漏，下面明确披露，不把核心实验完成概括为全部机制信息无缺失。

## H1/H2/H3

- **H1：ML有限支持，KR当前版本未支持。** ML CONST−DEMAND LL平均+.001089350，两seed六格都非负；KR平均−.000400568，两seed只有2/6、1/6格非负。两数据不合并成一个平均结论，证据**MEDIUM**。
- **H2：特定需求形成未获支持。** ML对UNIFORM的平均LL增量仅+.000027408，GEOM/SHUFFLE未被稳定超过；责任确实变化、非退化，但不能据此证明功能。独有机制证据**WEAK**。
- **H3：完整实践增量未成立。** ML九对照仅三项通过；对H_DIRECT、ETA_DIRECT开发LL平均更坏+.001639660/+.001814021，并触发最差cell保护。固定exposed test中相应LL均值更坏+.008976175/+.009290476，AUC两seed也更低。当前版本为**VERSION_NOT_SUPPORTED**，有限反例证据**MEDIUM**；不是所有需求/迁移方法不可能有效的证明。

## 稳定现象与反例

DFD对UNIFORM的LL在四角色两seed均略好，但尺度小、AUC不稳定；相对SHUFFLE有seed反转，GEOM四角色平均LL均更好。普通条件读取胜常量、胜LOCAL或LOCAL_CAP仍可保留，强直接概率头足以竞争，native自身还是必要参照。test相对NATIVE的DFD LL一好一坏，不能用平均小改善宣称稳定成功。

ML相对H_DIRECT的AUC在Q两seed略高，test却两seed更低，说明指标与人口角色不能择一。四个native的test epoch24 LL均差于合法selected；ML Q/legacy的晚期AUC反而上升，不能把晚期排序或少恶化改写为最佳泛化。域规模未被识别为主要解释，未见可普遍推广的小域改善/大域下降规律。

## 数据、重复与适用范围

自然ML/KR三域两任务、fit-only编码，原生Sample目标/full40960/coupledAdam/FP32，4native24完整轨迹；ML选epoch13/13，KR14/16。每个数据独立决定，seed是重复单位，n=2描述性t区间与每seed1000次配对bootstrap分列，未把六格、fold或角色当额外seed。ML用原始用户，KR无可信原始用户ID而用row退路。

test/legacy历史暴露，冻结hash `{freeze['freeze_hash']}` 之后才统一评分。没有真正新数据确认、没有6101–6104正式复评或STEM/AutoIFS领先主张。完整macro/natural/pooled/六格、配对差、区间、selected/固定终态、身份与修订均附在分报告、[tables](tables/)和[工件索引](ARTIFACT_INDEX.json)。

## 执行可信度、缺项与存储

DESIGN→IMPLEMENT→独立REVIEW→RUN→主agent分析全部完成；实际raw-X重建最大{execution['maximum_raw_x_reconstruction_error']:.12g}≤1e-6，184/184导出及12/12结构计时通过。训练源码/数据/行计划身份冻结，修复和过程偏差在[PROCESS_REVISIONS](PROCESS_REVISIONS.md)，未选择性重跑负结果。

**MAJOR机制信息缺口：{missing}** 该项清理前漏检，限制outer需求分层/迁移解释，不改变已保存H1/P2门控输入或主要负结论。按start最小必要原则记录MISSING，不为补机制分组重新训练已关闭版本，也不把ETA_DIRECT微调结果冒充原η。

遵从用户要求，模型/优化器和大中间缓存已删，未改动其他`/home_new`目录；清理移除8.32GB，无.pt/.pth或模型symlink。保留日志、原始预测、指标、源码/角色/选择/责任信息。指标可从预测复算；重做模型前向需重训。

## 成本与下一项可区分问题

已记录训练/开发leaf5674.226502秒，共享GPU进程跨度和/并集另列，缺失setup等不伪称0或精确总量。ML DFD完整raw-X P50约5.709ms，H_DIRECT2.535ms；更高实现成本没有获得完整风险收益支持。三个成本账重叠，不相加，不称GPU小时/能耗。

下一项优先在真正封存的自然数据窗口，比较强直接概率与普通来源读取，先重建来源监督的独立增量；不继续堆K、元优化或额外seed追逐当前版本阳性。该未来研究本轮未执行。

详见[报告索引](REPORT_INDEX.md)、[P1](P1_DEMAND_REPORT.md)、[完整流水线](FULL_PIPELINE_REPORT.md)、[归因](ATTRIBUTION_REPORT.md)、[成本](COST_REPORT.md)、[开放问题](OPEN_QUESTIONS.md)。
''')

write('REPORT_INDEX.md','''# DFD-v4.0 报告与工件入口

本轮已按注册门槛正常结束：4个native24前缀与全P1，ML20个P2输出完成；KR P2及两数据P3/P4 NO_START。184/184冻结导出、12/12完整推理测量和用户要求的临时模型清理完成。外层固定μ分组有明确MAJOR缺项，不能声称全部机制信息完整。

- [最终结论](FINAL_REPORT.md)
- [数据与角色](DATA_AND_ROLE_AUDIT.md)
- [P1完整结果](P1_DEMAND_REPORT.md)，[ML详情](P1_MOVIELENS_ANALYSIS.md)，[KR详情](P1_KUAIRAND_ANALYSIS.md)
- [完整P2与冻结伴随角色](FULL_PIPELINE_REPORT.md)
- [归因与证据强度](ATTRIBUTION_REPORT.md)
- [成本/标签/推理/存储](COST_REPORT.md)
- [开放问题与缺项](OPEN_QUESTIONS.md)
- [过程修订](PROCESS_REVISIONS.md)，[旧RHD有界诊断](RHD_BOUNDED_DIAGNOSTIC.md)
- [设计规格](DESIGN_SPECIFICATION.md)，[训练独立审查](IMPLEMENTATION_REVIEW.md)，[最终化独立审查及缺项补审](FINALIZATION_REVIEW.md)
- [184个冻结预测、4个P1 OOF及形成责任信息的路径/身份](ARTIFACT_INDEX.json)
- [完整聚合数值、所有六格、seed/cluster区间、角色与终态](tables/)
- [冻结、实际阶段决定、审计和清理机器收据](evidence/)

服务器信息产物根：`/home/xzf/Code/phase0-3/artifacts/mdmt_dfd_v4/v4_0/`。大原始数组、行级预测和模型不上传GitHub；模型已按用户要求清理。Git分支只发布新源码、精简报告、聚合数值与审计身份。原workspace中的新源码/报告为独立有效checkout的链接，原失效.git未修改。

分析可用源码 `analysis_p1.py`、`analysis_pipeline.py`、`analysis_final.py`、`analysis_support.py`、`analysis_costs.py`、`build_reports.py` 从保留的预测和信息复算。训练入口在 `experiments/mdmt_dfd_20261026/cli.py`；本次冻结/评分/清理由独立 `finalize.py` 执行。所有重推须先恢复/重训模型，不会因摘要或hash生成虚假预测。
''')
print('Reports assembled',len(bootstrap_rows),'bootstrap CI rows',len(index),'prediction index rows',flush=True)
