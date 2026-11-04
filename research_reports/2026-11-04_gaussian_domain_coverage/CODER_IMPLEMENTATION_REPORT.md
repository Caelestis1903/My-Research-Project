# CODER IMPLEMENTATION REPORT

状态：IMPLEMENT COMPLETE，2026-11-04。依据本目录已完成并获主agent批准的DESIGN_SPECIFICATION。旧Gaussian源码、旧实验报告和原始数据未修改；实现限定在新实验包。

## 实现

- `objectives.py`：单域直接调用原vendored GA31-R3 `source_losses`；双域分别按原key和各域行数抽取原Gaussian视图，两个任务的BCE差值统一除以实际`2B`。保留原RMS detach、domain_indicator及局部Generator规则。一次backward、普通Adam step、固定EMA更新。
- `worker.py`：原fixed1模型factory及1614254参数身份，原seed初始化、seed+11000独立loader、完整49步/epoch；online/fixedEMA各自validation六格MacroAUC严格最大、并列最早；selected/E10/E50全状态只驻RAM。export/restore保存模型、EMA、Adam、loader、Python/NumPy/CPU/CUDA RNG、阶段和选点。
- `run.py`：固定10job准备、数据/旧端点审计、独立review hash门、原batch smoke、逐GPU资源派发、原guardian epoch提交和合法RAM恢复。每条50epoch且2450步完成后冻结V选点，再按同hash评价V/Test。保留逐batch原行号hash、各域噪声key/hash、view行号hash、更新计数、逐epoch损失/六格验证、物理成本及失败日志。
- `config.py`：固定配置、10job/source/data身份、来源闭包manifest。`reporting.py`复用既有stats/write_csv，导出绝对数、逐seed配对、均值/样本SD/t95区间、独立宏收益等级/六格保护/LogLoss裁定、成本、曲线和运行manifest。
- 原`overnight_ab`的guardian/state/selection/evaluation/data_bridge/resource gate及owner锁直接复用。原Gaussian代码直接从history/vendor导入，不复制、不换为EMA02V2随机流。

## 已执行必要检查

项目解释器：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

`python -m compileall -q experiments/gaussian_domain_coverage_20261104`：通过。

`python -m pytest experiments/gaussian_domain_coverage_20261104/tests -q`：审查修复后 **14 passed in 2.39s**。覆盖：d0/d2单域与源loss/梯度/噪声精确一致；双域覆盖、d1排除、非等域样本预算和尾批、双任务；原key噪声共享；lambda/kappa关闭和空目标；RMS detach；一次Adam/EMA及普通Adam状态；两epoch RAM export/restore后的模型/EMA/optimizer/loader/RNG/selection一致；validation并列最早与阶段限制；10×50矩阵；guardian丢失后的完整重放链及已封存DONE释放RAM。

`python -m experiments.gaussian_domain_coverage_20261104.run prepare`：通过。核对原CSV、cache manifest SHA、九个缓存文件SHA、六数组/row_ids/encoder/scaler、域映射、原A1普通Adam+Gaussian+ema995配置、AB表SHA及150个复用V/Test端点来源checksum；新job恰10。审查修复后重新prepare，source hash为`c8c76da2cb644559ea1cdf69d56e75fef00ae1fb70e214529708a023884b096d`，以当前`manifest`输出供review绑定。旧prepare保留为`PREPARED_BEFORE_REVIEW_FIX.json`；断言核对新旧数据、配置、复用端点身份完全相同，训练尚未启动。

## 独立审查修复

reviewer识别出guardian RAM丢失后旧commit链与重置的launch编号冲突。现统一以每个job最新`commit_index=1`开始的连续链作为正式轨迹；archive和summarize共用`committed_chain`，共享ledger保持append-only。确认旧worker退出后，将该job旧标量目录移到`replay_history/<job>/replay_N`，新guardian可从新`launch_1`开始，原日志、废弃epoch与物理成本全部保留并导出成本表。科学job身份不增加seed。

同时修复worker在DONE写入与RAM release之间退出的完成路径：reconcile先核对DONE各文件SHA，再释放该job残留guardian状态，保持已封存结果并完成shutdown。没有修改旧guardian源码或新增研究配置。

本阶段未执行GPU smoke和正式训练；两者由独立review通过后进入RUN，smoke日志不会用于性能选择。未进行研究结果分析。

## 入口与交接

先由独立reviewer核对设计、实现和本报告，并写`INDEPENDENT_REVIEW_RECEIPT.json`。随后运行`smoke --devices 0,1`、`run --devices 0,1`、`summarize`，均从新worktree根目录执行上述模块入口。全部轨迹完整50epoch，不按早期seed结果停止。

新包README给出完整CLI；`prepare`已冻结10job和数据/源码身份。报告目录中的中文研究报告、训练后原始标量归档、runner报告与台账增量由后续RUN/ANALYZE阶段生成。
