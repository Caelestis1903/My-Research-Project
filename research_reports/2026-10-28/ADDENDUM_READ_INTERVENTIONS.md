# DESIGN ADDENDUM — 固定状态读取干预与reader再适配

本附录细化原02的同状态动作、03的函数替换、06第4/9节要求，补齐既有验收；不增加主实验臂、训练seed、选参机会或P1—P5轨迹预算。只实施下述有限诊断。

## 1. 适用对象与固定口径

P1的FIXED_READ、RELATION_ONLY、CLUSTER_RELATION在每dataset/development-seed完成后，对V-selected及epoch20-final状态均执行。后续触发且实际保留的显式foreign读取方法沿用同一诊断；没有显式foreign的NATIVE_RAW/TARGET_MATCH/STATIC_LOCAL/CLUSTER_ONLY标NOT_APPLICABLE。selected与final若完整状态相同，复用一次实测并保留两个状态标签，不重复续训。

输入状态是既有完整RAM快照；不重建或改变其主训练结果。F_probe及Q原有行身份、任务标签、mask和自然权重保持。以一个`(target domain d, source domain s, active source leaf j)`为一路径p；它覆盖该target的全部需求叶、两个任务对应读出，每任务/需求叶风险仍分别导出。只考虑s≠d的实际已定义叶，不把UNKNOWN/inactive当负收益。

所有固定干预保留原q、实测G/low-rank政策、先验m、alpha、NULL、其余路径、目标头和source函数。下面只改变指明的预测分量，不刷新teacher、不重测关系、不重分配gate，不把干预后的alpha解释为新学习政策。

## 2. 固定状态动作：所有路径都测

先一次计算原始预测、z_ref、q、alpha、各path的c及冻结pilot根phi；缓存同一请求的这些量，批量完成下列评分，避免为每动作重复运行native。

- `KEEP_FIXED`：原状态输出。
- `ALL_NULL_FIXED`：所有foreign修正为0，输出z_ref。
- `OFF_PATH(p)`：该target上将路径p的c置0，其余路径alpha及c不变；删除的alpha质量视为归入NULL，NULL输出z_ref，所以无需重归一化。
- `ROOT_PATH(p)`：该target上把该source叶的f_sj换为冻结标准化pilot phi_s，仍用原w_dt,s及原alpha；动态树时移除该叶路径上的全部可训练source残差，根明确为phi而非任意中间父。owner风险路径不被改动。
- `FIXED_RELATION_FIXED`：保留原q/source/w/目标函数，改foreign权重为当前质量守恒m/2，NULL=.5；这是同函数库的确定读取政策比较，不更新模型。

每个有效p均测OFF和ROOT，包括无实测依赖的路径，以保留完整性。导出F_probe及Q每域任务、macro/natural LL、AUC、覆盖组支持和相对KEEP_FIXED差；逐需求叶用原状态责任加权risk，原始责任及row_id同时保留。固定移除损害仅说明当前预测对该路径的依赖；ROOT改善/损害说明当前读出下残差函数的作用。

## 3. F-only确定一个再适配路径

每状态只选一个p*做reader再适配，避免对全部路径进行短训搜索。排序依据是本状态F_probe的自然域权重LL损害：

`score(p) = sum_d pi_F(d) * mean_valid[BCE(OFF_PATH(p)) - BCE(KEEP_FIXED)]`，其中pi_F由完整自然F各域行数确定，不使用probe的等域抽样比例。由于p只作用一个target，其他域项为0。两个任务使用原主损失的valid标签均值，不单挑有利任务。

取score最大且严格为正的路径；并列按稳定`(d,s,leaf_id)`顺序。该排序只读F和既有预测，不读Q/V/S/E。若没有正score，输出`NO_POSITIVE_FIXED_DEPENDENCE`，保留第2节全部评分，不执行无明确对象的H步分支。这里不设置新显著性、幅度或支持阈值；状态身份与p*必须在读取本状态Q诊断结果前写入INTERVENTION_PLAN。

## 4. 三分支、原Adam、相同H=8

对p*从同一个完整RAM状态fork三臂：`KEEP_READAPT`、`OFF_PATH_READAPT(p*)`、`ROOT_PATH_READAPT(p*)`。三者各执行恰好8个F主batch，沿该状态原sampler下一游标继续确定计划；final状态也可生成后续permutation数据，不将其计作主训练epoch21。三臂使用相同有序row_id、尾批规则、mask和随机流。无需owner/probe/shadow batch，因为本诊断冻结其参数且只训练reader。

唯一可更新参数是模型原有的全部foreign reader w；所有w有相同更新权限，允许其他来源读出替代被移除路径。source兄弟原本共享w时继续共享。a/local、need trunk/mu/中心/路由、owner、source U/V、native/pilot全部冻结，G/alpha/teacher也冻结。各branch使用原状态w对应Adam m/v/step、原lr=.001/betas/eps，不重置；每步只用原自然`L_rec`，不加shadow、gain、CL、owner或anchor；每步执行原w半径2投影。其余参数grad=None，不推进其Adam状态。

OFF分支的被删路径始终不参与预测；该w如仍服务兄弟/其他保留路径可正常更新。ROOT分支使用冻结phi值但w可更新。KEEP也继续同8步，禁止拿续训后的OFF/ROOT对不续训KEEP。没有可部署训练参数或实际梯度的情况据实记录，不造零参数或补训练步。八步不选最佳中间状态、不按Q加步、不调整学习率、不刷新G来补偿干预。

这三臂回答“在冻结目标/路由/函数、既有w参数与Adam历史、同8步F机会下，其他读取或pilot根能否恢复风险”。它不能代替完整主矩阵TARGET_MATCH等对照，也不能单凭仍有损害宣称source或结构在所有适配方式下不可替代。

## 5. 评价、状态隔离与成本

先完成F排序和三分支8步，锁定三branch状态，再统一读取一次本状态的Q诊断标签，为第2节全部固定动作及三续训分支评分。Q不用于选p*、选步数、选checkpoint或提交赢家。比较至少同时提供`OFF_READAPT - KEEP_READAPT`、`ROOT_READAPT - KEEP_READAPT`，并保留各自固定干预相对KEEP_FIXED差，展示依赖与恢复能力是否一致。

不读取S或E，不把三分支并入主resident/selected/final。保存前后主状态模型/buffer/Adam/RNG/全部游标与teacher/G版本直接相等的验收；候选完成后恢复母状态，释放候选RAM。每个branch有显式状态名、原epoch/step、干预path、行计划、实际reader更新次数、loss和成本，无新hash或磁盘模型文件。

输出写入FUNCTION_INTERVENTIONS.csv及INTERVENTION_PLAN.json，保存必要row_id/预测/标签/原责任/路径分量数据以复算；COST_LEDGER分别列fixed_forward诊断和三branch训练/评价秒数、GPU峰值及实际额外24个reader step。它们属于诊断成本，不计新seed或完整适配轨迹；也不修改主模型20epoch+3×8主时钟。某方法未训练/某阶段未触发标NOT_TRIGGERED；无正依赖时短训标上述具体原因，不将缺测补0。

最小新增验收仅覆盖：OFF质量转NULL的手算logit、ROOT替换的明确函数、两种干预在训练8步期间持续生效、仅w有梯度、三臂同row计划和Adam初态、母状态无污染、F确定p*且Q在锁定后访问。已有loss/角色/state测试直接复用。

本附录不增加逐路径短训搜索、全模型修复、额外超参或E指导选择；未被选为p*的路径只有固定状态依赖证据，其8步可替代性记录UNMEASURED。
