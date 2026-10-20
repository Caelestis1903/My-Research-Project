# CORE CODER IMPLEMENTATION REPORT

日期：2026-10-20。本文报告实现与基本科学正确性检查，不解释研究结果。

## 当前交付范围

**首批可独立审查：数据、完整 AC-2D 64 配置、AC-2D 离线机制诊断、队列和严格 review gate。** AC-V9 8、AC-S5 32、AC-S6 16 共 56 配置已全部列入 120-run manifest，后续实现独立审查前不可运行。首批审查不得被解释为这 56 配置的运行授权。

所有新代码位于 `experiments/natural_ml_reassessment/`；所有大工件位于 `/home_new/xzf_natural_movielens_reassessment/`。历史文件只读。本次归档包括 core 与 v3 的 Python dependency tree、历史 `history_exp/first` 的 p7_models 依赖；运行无 sibling sys.path 依赖。

## 对应设计逐项说明

1. `data.py:load_population` 唯一基础入口是归档 `stage_F0_data.load_movielens_natural`，不调用人工采样。优化 seed 不入 loader；bundle population seed 固定 2022。自然训练实测 800167，域计数 155949/318599/325619；保留原物理年龄域、两个 rating 阈值、全局顺序 80/10/10、全自然 base-train 编码与 scaler。
2. 人工固定对照直接调用原 `role_reversal_sample(...,'d2_large',2022)`：34500/3000/150000。自然等 N 按最大余数与 `RandomState(20221020+d)` 生成：36543/74656/76301，总 187500。所有变体的 validation/test x/y/row IDs 和 schema 已逐数组 hash 核对相同。NPZ 保存全部 split IDs/features/labels；JSON 保存 raw CSV SHA、encoder/scaler SHA、各数组 SHA 和 NPZ SHA。
3. 人口唯一身份为 `dataset_config`。人工 training_pool=`fixed_d2_large_subsample`，等 N=`natural_proportion_matched_N`。`setting=natural` 是本次派生入口身份，不表示人工控制人口是原始自然人口。历史 DataBundle 的 `base_split_row_id_hashes` 实际返回所选人口 IDs；新增明确的 `base_train_row_ids_hash`、`base_train_size` 与 `selected_train_row_ids_hash`、`selected_train_size` 防止混淆，不修改历史 DataBundle 类。
4. `stage2d.Runner` 继承真实 Stage2D Runner、原 Sampler、原 `step`；仅替换数据入口、MovieLens partition count 验证和源/输出身份。原实际 StageS1M3OEAdapter、gate routing detach、learned four scalars、两组不重叠 Adam、LR .01/.001、WD 1e-5、betas .9/.999、eps 1e-8 全保留。`step` 在 `super().step` 前后增加 read-only 观测，无梯度/参数替代。
5. 四臂 `alpha=(p 或 uniform)` 与 `q=(p 或 uniform)` 均由实际训练人口计数计算；原 loss 为固定实际 batch 分母的 `mean_i[(alpha[d]/q[d])*mean_task BCE_i]`。原 IID with replacement 双 uniform 耦合 sampler 未重写。
6. AC-2D 为三人口各 16 配置加完整自然匹配预算 16，共 64。全自然 full50 为 50×20 main + 50 arch；人工与等 N 为 50×5 main + 50 arch；自然 matched250 为每宏 epoch 40960×4+23660，250 main/50 arch。完整自然 alpha/q 基于全部 800167 行；没有以子集 alpha/q 替代。
7. Stage2D 生命周期封装沿用原逐 epoch 训练/独立 arch/evaluate/strict earliest validation selection/冻结后 test 结构。每 epoch 验证六 cell；selected/final/E10/20/30/40/50 六 cell test。附宏、train-p、eval-p、target d2、non-target 宏以及逐 task 聚合，均为 cell 指标加权，不是 pooled AUC。每步 raw/WD/effective gradient、实际 delta、persistent m/v、step、bias-correction 分区观测；曝光、unique、repeat histogram 和精确 row-ID batch hash 记录。
8. 初始化、common、selected/final 保存完整 model/tau、两个 optimizer、RNG、两个 sampler RNG、逐行曝光计数、选点史和配置。中断续跑重建原采样链、校验 hash、恢复完整状态并清理不完整 epoch 的追加观测。最终 checkpoint/prediction SHA 索引独立保存。
9. `mechanism.py` 为每个 SN/SB 路径锚、每个人口/预算、每 seed 的初始化/E10/E40/selected/final，显式建立 160 个离线任务；训练计数与诊断计数分开。每固定 theta 使用 512 独立实际 B40960 batch 估计全参数均值和全坐标 covariance trace/C-P-A block trace，在线 Welford，不挑坐标。96 条真实训练行（32/域）构成 exact finite stratified pool，逐行计算完整梯度，对 SN/SB alpha/q identity 精确求和，解析 batch trace 与独立 512 组 multinomial pool MC 误差一起报告。有限 pool 不是完整人口的 exact oracle，两者字段分开。
10. Adam counterfactual grid 从同一个 theta/optimizer baseline 产生 FULL/TARGET synthetic history，分别切换 m/v/current 的 8 种来源；不拼接不同轨迹，不把它当作跨轨迹 history 因果。原 gate detach 的训练导数与在深拷贝中仅 reattach gate input 的预测风险 Jacobian 分开计算，必须先逐预测核对完全相同。

## 原源身份核实

完整实际 SHA、绝对原来源、bytes 见 `source_archive/inventory.json`，关键 git commit/blob 核对见 `source_archive/provenance.json`。关键源全部 matches_git_blob=true：

| Producer | 最后文件提交 | SHA256 |
|---|---|---|
| Stage2D | 66c4cb33d4c3677a38f65010e06b6e7a3cacfc17 | 115c37e1e865a8e2da7802bcf7a90c1edecc3eec5dabfe783e165457b7706b91 |
| F0 loader | 3e6642820b7b2f17f130a5ccd9d382a16e023811 | e39ab4ea8ab1282f2ec939a987d02674a52d91a3423314c3786fca34e2d1df42 |
| S1 data | 528a5fb945a3007747ea289f35ef160ec1795284 | 8a0045bf3f1868b94ea1a223a817104b4985ed39449006a2d031f978065b81ee |
| S1 adapter | 3e6642820b7b2f17f130a5ccd9d382a16e023811 | 1f6fa05358b047b048b115681cd62ac2f833ffd1d0892ad68b64b91216caf691 |
| S5ML run | 3e6642820b7b2f17f130a5ccd9d382a16e023811 | 825316be2a479eee55618d03029d54ef2b15eb78153cc3e285809e81a5a63fe2 |
| S6ML run | 3e6642820b7b2f17f130a5ccd9d382a16e023811 | 32e7ac0511788296569cb4ad789f1f804c0dd527f1610dc3cb041b01062b1874 |
| V3 run | 0f83141bcdf033200fba678965ec808e69983e8c | af6ca583e8224063472c635a5b2d3c53809dbc99864f82319418d687ed006c7b |

S6ML 实际源码 `_validate_args` 和 `_output_formal_protocol` 都要求 main LR=.01、arch=.001。其构造链有明确注释并直接走 S3Runner.__init__，绕过 S4Runner.__init__ 对 Equal 的强制覆盖，因而保持 Sample。后续封装必须同样保持此链与 `_controlled_step` 的持久 optimizer.step + overwrite 公式。上述是源码核实，不是结果分析。

## 已执行检查与命令

工作目录 `/home/xzf/Code/phase0-3-natural-git`，Python `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`，环境 `PYTHONDONTWRITEBYTECODE=1`。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_ml_reassessment.data
PYTHONDONTWRITEBYTECODE=1 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_ml_reassessment.manifest
PYTHONDONTWRITEBYTECODE=1 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_ml_reassessment.basiccheck
```

检查产物 `/home_new/xzf_natural_movielens_reassessment/basicchecks/ac2d_same_batch.json`：四臂同实际 batch 显式逐域固定分母 loss、完整 raw gradient 与独立 Adam 实际参数更新对齐；检验 main/arch 参数互斥与同 q 同 seed 的原 sampler stream 相同；risk clone 预测值严格相同。检查只做单 batch，不调用 fit，正式训练 epoch 数为 0。

## Review 后的运行命令

runner 强制读取 review JSON，其 `verdict` 必须为 PASS 或 PASS WITH FIXES、`blocking_issues=[]`、`stages` 覆盖实际 stage，且 `source_hashes` 与当前逐文件完全一致；默认 `/home_new/xzf_natural_movielens_reassessment/review_gate.json`。这不是自行给出的 reviewer verdict。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_ml_reassessment.queue --stage AC-2D --devices 0 1 --review-report /home_new/xzf_natural_movielens_reassessment/review_gate.json
```

完整可续跑 run manifest：`/home_new/xzf_natural_movielens_reassessment/core_manifest.json`。其每一配置含完整执行 argv 和预算；该文件列全 120，不代表尚未审查 stage 可以运行。离线机制命令为：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_ml_reassessment.mechanism --run-dir RUN_DIR --checkpoint epoch09.pt --output DIAGNOSTIC_JSON --device cuda:0
```

## 未解决 blocking issues

首批 AC-2D 仍需独立 review verdict；coder 无权授予 PASS。AC-V9/S5/S6 生产封装与对应同 batch oracle 尚在实现，不能以首批通过替代其审查。未运行正式训练或正式 512-batch 机制诊断，本文不提供任何研究效果结论。

## 独立审查修复记录

2026-10-20 R1：reviewer 指出最终 run_manifest 对 progress.json 的 SHA 在其从 TRAINING 改为 COMPLETE 前计算，会使该索引失效。已把 COMPLETE progress 的原子写入移动到 artifact hash 收集之前。run_manifest 的 artifact_hashes 不包含其自身与 stdout.log；随后 sha_index 只索引写定文件、不包含自身，run_manifest 不反向索引 sha_index，因此无相互递归索引。此修复不改变任何训练算法、预算或选点。
