# Runner/objective/optimizer static audit

本文件为主 agent 静态检查记录，不是独立 REVIEW PASS 或新训练复现。

## 数据路径与身份

实际数据根目录固定为 `/home/xzf/data`。`stage_S1_data.load_stage_s1_bundle` 接受显式 `dataset_path`，否则使用公共默认值。本次只修改公共默认值及 `stage_S1_orchestrator._dataset_paths` 的两个 fallback：MovieLens `/home/xzf/data/ml-1m`，KuaiRand `/home/xzf/data/kuairand/processed`。历史 run config、checkpoint、hold 文件未改。旧 E/S0 启动器仍包含旧默认路径，不在本次训练矩阵中；若未来调用，必须显式传当前路径，不能直接复用旧命令。

CPU 完整加载由 `dataset_preflight.py` 实测：两套 CSV 可读、特征/标签有限、split 行互斥，原始哈希见 `dataset_preflight_*.json`。MovieLens 此次验证 AO04 的 d0_large/2022；S6ML 的 d2_large 需另做相应配对验证，不将 d0_large 结果冒充该设定。KuaiRand 保留全局 row-order 80/10/10，不重新打乱。

历史 `physical_domain` NPZ 字段的取值为 0/1/2，而原 CSV tab 为 0/1/4。`stage_S1_data._load_kuairand_base` 明确 `domain_lookup`，但 bundle.local_to_physical=(0,1,2)。因此该字段实际携带 local ID。审计先按 row_ids 绑定原 CSV，再逐行确认 {tab0:0,tab1:1,tab4:2}，未按样本数猜域。

## 调用链与目标

`stage_S6_run.main → S6Runner → S5Runner/S4Runner/S3Runner/S2Runner/S1Runner`。S6 初始化绕过 S4 的 Equal 强制设置并安装 Sample coefficient table。S5 则为 Equal/KuaiRand/natural/d1 target。

`stage_S3_run._objective_components → stage_S3_gradient_access.loss_components`：先取每样本两任务 BCE 均值，Sample 的 domain weight=1；Equal 使用固定训练比例权重 1/(3p_d)。full 为 batch 固定分母 mean；target 在同一个分母内乘域 mask；non_target=full-target。不是先过滤 target 再重取 mean。batch 不含 target 时显式报错，不能宣称所有缺域边界都被自动支持。

S1 model 初始化后按 `is_architecture_parameter` 分 main 与 arch；main Adam 包含 C/P，arch Adam 包含四个 architecture scalars。S3 `partition_parameters` 以名字穷尽 C/P/A_arch 并检查交集；本次未构造模型，因此不把静态断言当作动态 reachability oracle 已运行。

C 包含 embedding、shared STAR/skip/star MLP、shared/domain/task experts；P 包含 slot/gate/tower；A_arch 单独。domain expert 名字不保证它只从同名域获得梯度。实际 reachability 仍需有效模型与原状态的动态审查，当前为未验证。

S5/S6 `_controlled_step`：main 时 names=C，private=P；arch 时 names=A_arch，private 为空。一次 forward 取得 FULL/TARGET 梯度，P 固定 FULL。用 pre-step persistent state 计算 df/dt，再将 history source 的梯度安装到 optimizer.step() 写入 moments，必要时把应用到参数的 delta 替换为 current source 的反事实 delta。F=(FULL,FULL)，T=(TARGET,TARGET)，HT=(TARGET,FULL)，DH=(FULL,TARGET)。不能将 DH 称作 fresh target Adam，也不能只检查 raw cosine 来证明实际 applied update。

## 选点和状态

S1 `fit` 每 epoch 执行 main train → persistent arch iterator update → validation（保存/恢复 RNG），按 six-cell macro AUC 严格 > 更新最优（并列取最早）。固定 50 epochs，shadow patience 记录不提前截断。常规路径只保留 selected、final、common 9/19/29/39/49 的模型状态，不是每 epoch 完整 optimizer dump。

`stage_S6_R2/s6r2_common` 中 R0–R4 用 validation 选 epoch；test 查表仅接受实际保存的 selected/final/common 对应 epoch，查不到必须 NA。A-grid 的 alternative-rule 支持不能升级为 exact alternative selected test。

B1 dense-checkpoint 的 full C/P/A optimizer、RNG、iterator 要求尚未实现或验证；原 S6 不能直接当作已满足 dense/full-state 条件。S8 历史完整状态重训与 continuation 也不能替代丢失的 S6 原工件。

## AO04 身份与指标

`ao04_v12_runner` 训练/validation，`ao04_v12_test_evaluator` 独立 test 入口，检查 global unlock 并加载 selected/final/common。此次没有调用模型 evaluator 或加载任何 checkpoint；只从保存 NPZ 重算指标。`safe_logloss` 源码冻结 epsilon=1e-12；另外保留 1e-7 诊断列。

28 个 config→AO04 source hash 检查一致；四个可读取完整 epoch CSV 的 selected epoch 与 manifest 一致。该检查不覆盖缺失的五个完成 run，也不等于所有训练依赖已独立重放。

G1 旧 receipt 引用的工件当前 31 absent、17 raw hash match、3 raw mismatch。3 项均为 metrics_epoch.csv，LF→CRLF 后精确恢复旧 SHA（`csv_newline_diagnostics.json`），说明此三项差异仅为换行转换；原始 bytes 未写回。缺失的 checkpoint 不能以旧 receipt 的 PASS 替代。
