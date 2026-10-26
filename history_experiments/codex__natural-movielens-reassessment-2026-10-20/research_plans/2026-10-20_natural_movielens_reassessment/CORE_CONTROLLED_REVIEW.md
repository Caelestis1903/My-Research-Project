# REVIEW REPORT — AC-S5 / AC-S6

日期：2026-10-20。独立review。设计：`CORE_DESIGN_SPECIFICATION.md`；实现报告：`CORE_IMPLEMENTATION_REPORT.md` 后续56配置交付部分。实际审查 `controlled.py/controlledcheck.py/freeze_configs.py`、对应历史归档、冻结配置及科学正确性检查工件。本批只覆盖AC-S5 32和AC-S6 16配置。未运行正式训练、正式诊断或分析研究性能；reviewer未修改算法代码。

## 1. Verdict

**PASS。无blocking issues。** 仅放行48条AC-S5/S6配置；不沿用至AC-V9。

## 2. Requirement-by-requirement verification

|要求|独立实际核对|结论|
|原算法与来源|S5/S6归档文件逐一与原commit `3e6642820b7b2f17f130a5ccd9d382a16e023811` Git blob相同；Mixin经MRO分别dispatch到原S5ML/S6ML `_controlled_step`，未替换为统一简化算法|通过|
|模型/分区/优化器|初始化S3→S2→S1实际learned M3OE；原routing detach、C/P/A分区和不重叠两Adam保持；main .01/arch .001、betas .9/.999、eps1e-8、coupled WD1e-5；target物理d2|通过|
|Equal/Sample|S5原compute_equal_coefficients按自然p得1/(3p_d)；S6原sample table权重1。目标项保持mask后按完整batch分母mean，未改成target域内均值；绕过S4构造器的Equal覆盖，保留Sample|通过|
|history/current数学|F=(FULL,FULL)、T=(TARGET,TARGET)、HT=(TARGET,FULL)、DH=(FULL,TARGET)；原方法从同次forward求gf/gt及private FULL梯度，按history仅执行一次persistent optimizer.step，HT/DH以预步state算出的current delta覆盖C或A，private P始终FULL；none-gradient、bias correction、WD及原残差断言保留|通过|
|实际同state检查|读取actual checker与最终JSON，六臂使用非零persistent m/v基线、同真实21行，与对应未改producer比较参数/m/v/step全部0误差，独立arch参数0误差。检查源码确实调用实际原method，而非互比两个新封装|通过|
|真实loader|RecordedDataset只记录索引；base仍原TorchDataset+shuffle DataLoader，num_workers0/drop_lastFalse；full每epoch无放回800167；不是Stage2D IID-WR|通过|
|匹配预算|完整自然support/p保持，main截187500，实际batch `[40960,40960,40960,40960,23660]`；arch persistent iterator使用无cap的全自然loader；没有将main cap带入arch|通过|
|持久arch及恢复|state保存创建/首次next torch RNG与消耗batch数；在preserve_rng范围重建同一loader cursor，随后恢复训练RNG；checker验证next batch完全相同、环境RNG不变。完整state含model/tau/两optimizer、draw counts、selection、none/zero-target counters与arch iterator；初始化即有原子resume|通过|
|矩阵/完整预算|S5四臂×两预算×四seed=32，S6 F/DH×两预算×四seed=16；自然full50为1000main/50arch，matched250为250main/50arch。无early stop；shadow bad_epochs仅记录|通过|
|选模/test|原顺序main→arch→preserve_rng validation；全validation六cell最早严格最大，50轮全部保留；先selection_frozen后零更新selected/final/E10/20/30/40/50 test；0..49完成epoch语义明确|通过|
|日志/状态|六cell AUC/LL及宏/train-p/eval-p/target/non-target/逐task聚合，实际步数、row-ID batch SHA、主曝光unique/repeat、原gradient/probe、原moment_state dump保留；完整state原子写，选点及common恢复可重建。最终progress先写定，manifest/sha_index排除自引用|通过|
|冻结与输出|frozen_core_configurations实际120条，其中32/16及预算正确；这48条source集合与当前381项一致，也与controlled_same_batch检查stamp一致；仅新增controlled.py进入依赖，不改变已签AC2D稳定源；输出独立root|通过|

## 3. Bugs/deviations found

无CRITICAL或MAJOR问题。MINOR解释限制：`optimizer_full_observations`中的raw gradient对共享参数表示实际安装的history梯度；HT/DH的applied delta来自覆盖后的current路径，应结合原gradient_step/probe中两种梯度/两种delta阅读。普通shuffle loader的q=p是人口边际，不能写成IID-WR。部分历史stage字符串仍为S5ML/S6ML，人口身份以config.dataset_config与新stage字段为准。

## 4. Corrections made

无reviewer代码修正。本轮只写review报告和独立gate。

## 5. Remaining risks

检查工件formal_training_epochs_run=0，不代表完整训练稳定或任何科学效果。arch iterator roundtrip与单batch非零state检查覆盖科学必要语义；没有新增泛化工程测试。历史test重复使用，最终报告需标探索性复现。恢复只从已原子提交的epoch边界继续，截断下一不完整epoch的CSV；没有保存任意batch中途续跑状态。

## 6. Whether experiments are safe to run

**是，仅AC-S5/AC-S6共48配置，按原完整50epoch与预定匹配预算。** 独立gate：`/home_new/xzf_natural_movielens_reassessment/review_gate_controlled.json`，verdict=PASS，blocking_issues=[]，stages=[AC-S5,AC-S6]，source_hashes为 `bootstrap.source_hashes(('controlled.py',))`。
