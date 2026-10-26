# REVIEW REPORT — CORE 首批 AC-2D

日期：2026-10-20。独立 reviewer。范围为 `CORE_DESIGN_SPECIFICATION.md` 中数据、64个AC-2D配置、对应160个离线诊断任务与执行gate；不覆盖AC-V9/S5/S6的56个未审生产配置。已独立阅读设计、coder报告与actual代码；未运行正式训练或正式512-batch研究诊断，未分析模型性能结果。仅修改本报告。

## 1. Verdict

**最终 Verdict：PASS（2026-10-20修复复查），仅AC-2D首批及其已审离线诊断。**

AC-2D训练公式、真实模型、两optimizer、完整预算与测试时序符合设计。初审为PASS WITH FIXES，以下第3节保留原始发现；其阻塞项全部修复并独立复查关闭，见第7节。此处不授予其它core阶段PASS。

## 2. Requirement-by-requirement verification

|要求|actual核对|结论|
|单一自然入口|`data.load_population`调用归档F0 loader，固定population seed=2022；完整自然800167行、155949/318599/325619；年龄域、标签和80/10/10原顺序保持；schema只从base train拟合|通过|
|人工固定控制|直接原`role_reversal_sample(...,'d2_large',2022,row_ids=...)`，34500/3000/150000；没有将optimizer seed传给人口采样|通过|
|自然等N控制|按自然p最大余数，36543/74656/76301；RandomState(20221020+d)无放回、排序原行，总187500|通过|
|schema/held-out/身份|三data manifest的schema及六个held-out array SHA相同；三个NPZ的实际SHA与记录逐一相符；base/selected身份显式区分，控制人口未被标作完整自然train pool|通过|
|真实归档模型/step|Stage2D、S1模型、S3分区、MDMTRec模型的归档SHA独立与原树Git blob对比相同；所有370归档经source_hashes校验。adapter继承原Sampler与实际step，包装仅作只读观测；原gate detach、task-major routing、四个learned fusion保留|通过|
|alpha/q/loss|SN/SB Sample alpha=p，EN/EB Equal alpha=uniform；N q=p、B q=uniform；p来自实际人口；逐行alpha/q乘任务平均BCE，再按实际B均值；无自归一化或clip|通过|
|优化器与梯度|原main C/P和arch A两组互斥Adam，LR .01/.001、betas .9/.999、eps1e-8、coupled WD1e-5；每epoch独立arch采样batch；每步只执行相应optimizer；完整主干/原routing梯度保持|通过|
|配置与预算|实际manifest120条，其中AC-2D64，四条件各16：三人口full50 + 全自然matched250。自然50epoch为1000main/50arch；其余三条件250main/50arch；matched尾批23660，自然尾批21927；完整自然matched仍用全自然p/q|通过|
|真实同batch检查|读取actual basiccheck源码与产物：四臂均21真实行，同batch逐域固定分母参考loss、全raw梯度、fresh主Adam实际参数误差均0；same-q sampler一致，分组互斥；独立risk clone预测相等。该检查没有覆盖persistent arch一步，其实现由继承原源码及本次独立代码核对确认|通过本批范围|
|epoch/选模/test|保持0..49为完成epoch索引；E10/20/30/40/50对应9/19/29/39/49；所有50轮参与validation严格大于、最早平局；先selection_frozen后加载selected/final/common零更新test；test不影响选点或训练|通过|
|日志|六cell逐epoch AUC/LL、宏/train-p/eval-p/target/non-target及每task cell聚合；完整曝光/unique/repeat histogram/row batch hash；每步raw/WD/effective梯度、m/v、bias correction、delta分区摘要；state含model/tau、两optimizer、RNG、两sampler与曝光|通过；中断工件见C2/C3|
|离线全坐标MC|SN/SB路径×四条件×四seed×初始化/E10/E40/selected/final，共160任务；每theta每估计器512批B40960；全773474坐标Welford，不挑坐标，C/P/A分区trace与mean标准误|通过|
|有限pool与数学|32真实行/域的96行pool；逐行双任务均值gradient；q_d/32概率、alpha/q权重；exact weighted mean与`Var(weighted row)/B`；512 multinomial实现IID均值MC；有限pool与全人口MC字段明确分开|通过|
|Adam诊断边界|同一baseline state、有限pool full/target梯度，构造一次synthetic history event再current event，切换m/v/current八格，coupled WD、bias correction按t+2；明确不是跨轨迹history因果；只对C汇总delta|通过，解释必须保持该限定|
|risk Jacobian|deep clone只把gate输入detach改成reattach；先验证预测逐值完全相同，再分别求training derivative/risk梯度全坐标差；不修改正式model|通过|
|执行gate|run逐配置检查verdict、无blocking、stage覆盖及当前source SHA；queue按device独立进程；固定stable源码列表，不glob未来controlled/v9文件|通过；放行仅AC-2D|

## 3. Bugs/deviations found

### C1 — MAJOR / BLOCKING：完成progress的SHA过期（coder已修，待最终合并签署）

初始 `stage2d.py:165–170` 在计算run_manifest.artifact_hashes后才覆盖progress为COMPLETE，导致其artifact SHA必定与最终文件不同。coder已将COMPLETE progress提前至hash收集前，actual复读确认；训练逻辑无变化。sha_index应在run_manifest写定后生成，原manifest不得反向引用即将覆盖的sha_index。

### C2 — MAJOR / BLOCKING：恢复selected预测工件可能旧/缺

位置：`stage2d.py` 的save_state调用顺序、恢复分支及selected存档。last_resume保存早于selected_state与validation_selected写入；若这段中断，恢复分支可重建selected_state，却未重建validation_selected，最终可能把上一selected预测留在新selected标签下。应在该恢复分支依据恢复state零更新重算并保存validation预测，保持原evaluate的RNG保护，或提供等价校验/重建。不需要修改选模规则或优化算法。

### C3 — MAJOR / BLOCKING：完整resume payload覆盖失去原子性

位置：adapter `save_state` 最后直接torch.save到last_resume。父类先原子写入的payload不含adapter追加的draw_counts/config等；随后直接覆盖会在中断时留下半文件，无法按当前恢复入口继续。应对追加后的完整payload使用临时文件与原子replace。

### C4 — MINOR / NONBLOCKING：身份与解释限制

当前core run config review标记仍是REQUIRED_EXTERNAL_PASS，实际授权由外部gate强制执行；最终执行账本应保存gate路径/SHA并注明探索性复现。数据每次从原CSV重建；运行期间应保持已冻结原始bytes，跨run核对config记录的raw SHA与data manifest。离线Adam八格是有限pool synthetic-history即时更新范数，不能把它写成真实双轨迹history干预或性能修复。以上不构成新算法授权。

## 4. Corrections made

reviewer未改coder代码。C1由coder修正，actual已核；C2/C3已通知coder。将保留本节原始发现并追加最终修复核对。

## 5. Remaining risks

本报告不声称已执行64个完整训练或160个正式诊断。已有same-batch检查是fresh主Adam单步，不是persistent/arch全路径数值重放。全source identity包括已冻结core与v3归档，未来不能修改这些归档；新增controlled/v9 wrapper不会进入AC-2D固定source集合。历史test被重复使用，分析应标探索性复现。

## 6. Whether experiments are safe to run

**最终：是，仅64个AC-2D正式配置及已审160个离线诊断任务。** C1/C2/C3及SHA索引边界修复已关闭。独立gate的stages只含AC-2D；其余56配置不能运行。正式训练与离线诊断分别列账，runner仅记录原始工件。

## 7. 修复复查与最终签署

再次读取actual源码及coder修复记录；完成只读语法compile和AST保存字段比对。未运行训练、optimizer step或正式机制诊断。

|初审发现|修复后actual核对|结论|
|C1 progress SHA|先原子写COMPLETE progress，再计算run_manifest.artifact_hashes，再写run_manifest，fit最后写sha_index|关闭|
|C2 selected恢复|恢复saved model/tau/两optimizer/RNG/sampler后，若selected_epoch等于saved epoch，零更新调用继承的preserve_rng evaluate('val')，以临时NPZ→replace重建对应selected validation预测；恢复selected_state|关闭|
|C3完整resume原子提交|独立构建原producer全部state字段及5个新增字段，一次resume.tmp.pt→last_resume.pt；AST核对原字段遗漏0，新增仅budget/config/draw_counts/model_class/population|关闭|
|复查SHA索引边界|最终artifact_hashes显式排除sha_index.json，避免重复完成/恢复路径把旧sha_index纳入后又覆盖它；sha_index排除自身，索引已写定run_manifest|关闭|

已核source冻结列表为显式10个稳定wrapper与370个归档文件，无glob未来controlled.py/v9.py；实际Stage2D/F0/S1/model均从本次core归档导入。当前复查关键源码SHA：`stage2d.py=986eb3cf65e549b168565ba651ded5f7856e8acc487a717ea4e9c185e960a994`；`bootstrap.py=bba81680c4860476790d040b16b6960ae7d1cfc511bf126dede6db7286bc2bf8`；`mechanism.py=2f1d13d62508dadbddcfeafdaaf271ec6b9571faf21e0c0591da8d974fe653bb`。

最终无blocking issues。放行gate保存于 `/home_new/xzf_natural_movielens_reassessment/review_gate.json`，包含本报告及设计SHA、实际380项source_hashes、verdict=PASS、blocking_issues=[]、stages=[AC-2D]。gate不授权AC-V9/S5/S6。same-batch数值检查对应修复前保存工件代码版本；修复只涉及持久化/恢复及索引，actual step/gradient/loss与该已核版本相同。
