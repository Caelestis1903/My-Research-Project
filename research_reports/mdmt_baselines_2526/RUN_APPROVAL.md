# RUN 审批材料

**2026-10-29 用户已明确批准：“批准进入run阶段”。** 执行授权与当前 GPU 快照见 [RUN_AUTHORIZATION.json](RUN_AUTHORIZATION.json)。以下保留原审批材料；reviewed config 数值不变，开始按既定顺序执行 RUN。配置中的历史 `PENDING_APPROVAL` 字段由原严格加载器固定，本次授权单独记录，不改动已审查算法或超参数。

状态：**IMPLEMENTED_AND_REVIEWED_UNIFIED_ADAPTATION_PENDING_USER_APPROVAL**。最终独立复审 **PASS**，21 项 CPU 合成测试通过；真实数据训练尚未启动。用户要求进入 RUN 前批准，审批范围包含试运行、开发及五种子正式实验。没有启动 runner。

## 待批准的实验范围

协议 `MDMT-SOTA-2526-v1`；正式种子固定为 2022、2023、2024、2025、2026，不依测试结果更换。

1. 按任务顺序先在 MovieLens 上运行 HyperGate/AutoIFS seed 2022，再在 KuaiRand 上运行同种子；身份与预算完全相同的试运行可计入开发，不能重复计为独立实验。
2. 两个新基线 × 两份数据 × 开发种子 2022/2023，最多 8 条开发轨迹。
3. 开发后冻结模型、源版本、数据、配置和选点规则；三方法 `HYPERGATE`、`AUTOIFS`、`M3OE_FIXED1` × 两份数据 × 五种子，正式 30 条独立轨迹。正式 2022/2023 是冻结后重新训练，但已被开发看过；只有 2024–2026 是未参与开发的种子。
4. 训练只访问 T0∪F，T0 拟合预处理器，V macro LogLoss 选 checkpoint，Q 供开发报告。S、legacy 不训练；历史 E 仅在整个正式矩阵身份冻结后显式评估。记录失败、异常和实际成本，不把未完成矩阵标作完成。

## 具体训练设置

| 方法 | 完整训练阶段 | 优化设置 | 预算上限 |
|---|---|---|---|
| M3OE_FIXED1 | 固定融合主干端到端训练；不恢复 AutoML/DARTS | Adam lr .001，weight decay 1e-5，batch 40960 | 24 epoch，V 在 12–24 选点 |
| AutoIFS | search → 第 9 索引轮开始前 rewind → final selector/gate embedding → 硬剪枝 → reuse | search lr 3e-4、wd 3e-5；gate lr 3e-6、wd 0；reuse lr 1e-4、wd 3e-4；batch 4096 | search 10 epoch；reuse 最多 50 epoch，V macroLL 首次不改善停止 |
| HyperGate | 全部三种门控及域/任务辅助损失 | Adam lr .001、wd 0，batch 1024，eval 512 | 最多 1000 epoch，V macroLL patience 10 |

本轮已重建并逐值核验训练池：MovieLens 688143，KuaiRand 1725271。保留尾批，不做域重采样。

按上述行数计算，每条轨迹每 epoch 的更新次数分别为：M3OE_FIXED1 17/43，AutoIFS 169/422，HyperGate 673/1685（顺序为 MovieLens/KuaiRand）。这些是更新次数预算，不是训练时间估计；真实时间和显存尚未测。HyperGate 的 1000 轮是 RecStudio 默认上限，实际受 patience 10 限制，审批时须明确接受此上限。

## 配置来源与适配限制

- HyperGate 父源固定 `fdc82e3f3feee23694a8123f91092615f977900e`，使用其 RecStudio gitlink `f08cca607e998e8be4c5e915a393346baffa71f6`，不追随 `--remote` 最新分支。
- AutoIFS 固定 `65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e`；保持官方 forward 和四信息流，机械修复四组偏置输出维度错误；统一数据输入与 V macroLL 选点均需记录。MovieLens rank 2、KuaiRand rank 16。无许可证源码采用外部固定检出，不纳入提交。
- **HyperGate 完整论文超参数未公开核实。** 保留三门控、augmentation、anchor、detach 和辅助损失；expert [256,256]、gate [256]、tower [256,128]、dropout .3 来自固定 RecStudio 的 MMoE 配置；embed 16 沿用统一数据适配。缺项补全 `aug_embed_dim=16`、`aug_layers=[64]`、`aug_dropout=0`、`lambda_d=.1`、`lambda_t=.1` 明确标为 `UNVERIFIED_PROPOSAL`，并非论文参数。
- 接受此配置，实验身份为“官方核心的统一数据适配”，不能宣称 HyperGate 原论文数值复现。若获得作者完整配置，只能在 RUN 前通过设计修订和复审替换，不能看测试集后修改。
- M3oE 比较项明确为固定融合 `M3OE_FIXED1`，并非原论文 learned-fusion M3oE；旧 BASP/Relation 分阶段结果不混入新的同协议主表。

## 审批前就绪证据

- [可执行完整配置](../../experiments/mdmt_sota_baselines_2526/default_config.json)：包含全部超参数、三模型、数据权限、2022–2026 种子与输出路径。
- [复现与分阶段命令](reproduction_guide.md)：空环境命令、固定外部源码、CPU 检查、数据重建、8 条开发与 30 条正式轨迹、冻结和最终评估。
- [源码清单](SOURCE_MANIFEST.json) 和 [审计及适配说明](SOURCE_AUDIT.md)：固定官方 commit/gitlink、原始与适配训练协议、AutoIFS 四组偏置最小修复、HyperGate 官方初始化与配置缺口。
- [独立审查](REVIEW_REPORT.md)：最终 FIX2 **PASS**；初审和 FIX1 FAIL 历史保留，F1–F9 已关闭。独立完整 CPU 测试为 **21 passed in 4.04s**，包括原始实现 parity、阶段转换、恢复、冻结绑定、参数/源码漂移拒绝、实际数值报表及运行记录。
- [数据就绪回执](DATA_READINESS.json)：两份完整数据已在新目录重建；T0 预处理、编码/标签/角色逐值核验，八角色×六格支持与历史清单一致。历史 SHA 仅作已有出处，未重新计算哈希。
- [预运行 audit](PRE_RUN_AUDIT.json) 与 [环境/资源](ENVIRONMENT.json)：官方源版本匹配，owned arrays 齐全。两张 24GB RTX3090 当前仍被其他进程占用且利用率高；批准后由 runner 重新检查资源，不能把当前状态视为已有可用训练资源。
- [当前结果清单](experiment_manifest.json) 与 [比较表](baseline_comparison.md)：正式 30 条全部为 NOT_RUN，真实指标为空。没有用合成测试填入性能数值。
- 交付分支 `baseline/mdmt-sota-2526`，基于 `26-10-28@97151c74544c3651feb805d1898bc9ce1b8d12d3`；代码、测试、参考元数据和报告交付，数组、checkpoint 与逐行预测保留本地。

**待用户决定：是否接受上述 HyperGate 未核实参数的官方核心统一适配配置，并批准按本文件的开发顺序及 30 条正式五种子矩阵进入 RUN？** 只有收到明确批准后才可派 runner。当前 AUC/LogLoss、实际 GPU 训练成本和论文数值表格资格均未测。
