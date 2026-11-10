# 10｜不可变来源、关键数值核验与可复用资产

状态：**DESIGN_ONLY；来源审计已完成，新实验未执行。** 本文件把规划中依赖的事实绑定到固定 Git commit、原文件和原实现。实施与运行遵循 [00_EXECUTION_RUNBOOK.md](00_EXECUTION_RUNBOOK.md)、[05_IMPLEMENTATION_SPEC.md](05_IMPLEMENTATION_SPEC.md) 和 [06_SCHEDULER_AND_RESOURCES.md](06_SCHEDULER_AND_RESOURCES.md)。

## 1. 来源快照

| 标识 | 分支 / 位置 | 固定 commit | 作用 |
|---|---|---|---|
| MAIN | main | a646c5a29b74d4f3d731a8ae2c13038b06f5250f | 规划审计时的总研究上下文与评价规则 |
| AUG | 26-11-06 | 40705dfe0bb4fe614bdca835c0bae567348c87d0 | AUG_HISTORY 原实验、公式与控制器实现 |
| MECH | 26-11-07 | ffa3de067fcda5a1293af83a465d741e069493f1 | 本次主要代码来源；EMA_MECHANISM 原实验与协议 |
| ADAPT | 26-11-09 | 1391affa5f08556c79b4cb2bfad15228a8b9129f | 持续性和动作价值复核 |
| PUB | 26-11-09_v2 | a5f461de714232a416a2e95ca6fdc57fa7ce7ae8 | PUB 更新迁移参考及其失败边界 |

规划发布时 main 可以因新增文档产生新 commit；这不会把上述来源改成浮动分支。Codex Stage 0 需记录实际 HEAD、上述来源是否仍可访问、计划内容 hash 及选用实现 hash。只有与本任务有关的差异才需要处理；不得因后来多了一份无关报告重做全部研究审计。

main 的归档策略是整合研究结论，原 CSV/JSON、图、运行产物和实现仍留在来源分支。[MAIN 的最新 handoff](https://github.com/Caelestis1903/My-Research-Project/blob/a646c5a29b74d4f3d731a8ae2c13038b06f5250f/EXP_AI_CONTEXT.md) 明确了这一点。实施应从 MECH 固定 commit 建立隔离 worktree，将本套规划带入工作树；保留来源报告和原实验目录只读，在新的实验包内扩展。**不执行整个旧实验分支合并，不覆盖 main 的原报告，也不假定 main 已有所有可 import 的模块。**

## 2. 关键文件与 Git blob 身份

以下 blob 是 GitHub 返回的文件对象 SHA，不是 SHA-256 内容摘要。运行收据中的 SHA-256 与 Git blob SHA 分列，不能混用。

| 来源 | 文件 | Git blob SHA |
|---|---|---|
| MAIN | [EXP_AI_CONTEXT.md](https://github.com/Caelestis1903/My-Research-Project/blob/a646c5a29b74d4f3d731a8ae2c13038b06f5250f/EXP_AI_CONTEXT.md) | d604ffefbbfc3534c3523a3039e24a9a742bcb06 |
| MAIN | [Experimental_Global_Summary.md](https://github.com/Caelestis1903/My-Research-Project/blob/a646c5a29b74d4f3d731a8ae2c13038b06f5250f/experiments/Experimental_Global_Summary.md) | 572c667b3473385375bc05ec2abee61685b3fe8e |
| MAIN | [MDMT_ALGORITHM_METRICS_LEDGER.md](https://github.com/Caelestis1903/My-Research-Project/blob/a646c5a29b74d4f3d731a8ae2c13038b06f5250f/MDMT_ALGORITHM_METRICS_LEDGER.md) | e8af866e1e64d9285d20a49cdb3f2b0b675a4934 |
| MECH | [FINAL_REPORT.md](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md) | fc1127c4b0888d39ae8670139c2d10365d6a8401 |
| MECH | [ENDPOINTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/ENDPOINTS.csv) | 76c792ac4439abb7f846e458534f07f5396264cb |
| MECH | [RESOLVED_PROTOCOL.json](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json) | b2a4803ced45ba254ef3744ffbe7f994cf22e927 |
| MECH | [GROUP_MAP.json](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/GROUP_MAP.json) | 18f9ca5cbc71574e75d19b7dec6f40cff68c0856 |
| MECH | [SOURCE_RECEIPT.json](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/SOURCE_RECEIPT.json) | 7f4ec2004eeb7c2edd22f1252555555f4294e781 |
| MECH | [RELEASE_RECEIPT.json](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/RELEASE_RECEIPT.json) | a2c0958cbc3735d6fbc65c47954746f1369aaf9a |
| MECH | [COSTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/COSTS.csv) | 54a65579be681c5d47b408aaae7a3f1f2944d914 |
| AUG | [REPORT.md](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/REPORT.md) | f225fc77c474d01f0147b6ce9821ba0e314b0df6 |
| ADAPT | [FINAL_REPORT.md](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md) | b2503688a574a53f76852104ed6ed241459e0eaa |
| ADAPT | [A0_DECISION.json](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/artifacts/ema_adaptive_20261109/attempt1/A0_DECISION.json) | e4e51b832fb5771e8d1c17b7d6b03b8e838cc052 |
| ADAPT | [A0_STATIC_POLICIES.csv](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/artifacts/ema_adaptive_20261109/attempt1/A0_STATIC_POLICIES.csv) | 84e0deb12feb1be73ceb75a39da4a40c77d6e615 |
| PUB | [FINAL_REPORT.md](https://github.com/Caelestis1903/My-Research-Project/blob/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md) | 9198e79c6f63422dfd414f67edada6bf233433f9 |

额外完整阅读的来源：

- [AUG B 设计](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/DESIGN_SPECIFICATION.md)、[A 设计](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/DESIGN_SPECIFICATION_A.md)、[C 设计](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/DESIGN_SPECIFICATION_C.md)。
- [MECH 设计规范](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/DESIGN_SPECIFICATION.md)、[实现审查](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/IMPLEMENTATION_REVIEW.md)、[资源与复用规范](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_plans/26-11-6/07_RESOURCES_AND_REUSE.md)。
- [ADAPT 设计规范](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive/DESIGN_SPECIFICATION.md)、[分析实现](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/experiments/ema_adaptive_20261109/analysis.py)。
- [运行环境 start.md](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/start.md)、[AUG 分组与支持审计](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/aug_history_20261105/tests/test_a.py)。

## 3. 关键数值核验及口径

### 3.1 固定 EMA 的 selected Test 收益

复核规则：

1. 只取 MECH ENDPOINTS.csv 中 split=test 且 checkpoint=selected。
2. 按 dataset/objective/seed 连接 online 与 aliases 含 F 的唯一 physical readout。ML 的 F=H2；不把 F/H2 算两条训练。
3. 逐 seed 先计算 F−online，再对五个 seed 求均值、样本 SD、正向数；LL 独立求差。
4. 每个 cell 同样先配对再平均。online 与 F 各自的 selected epoch 可以不同，不将这些数值用作同刻因果归因。

| 条件 | ΔAUC 均值（未四舍五入计算值） | 配对样本 SD | 最差格的平均 ΔAUC |
|---|---:|---:|---:|
| ML Clean | +0.003848844882298508 | 0.0003743962265102343 | +0.003144672674492277 |
| ML Gaussian D0 | +0.003856801819326505 | 0.0006452382716306681 | +0.002609411157897523 |
| KR Clean | +0.004616399443180885 | 0.0030487727242198923 | +0.0013737173121483258 |
| KR Gaussian D0 | +0.007766927498394161 | 0.004257500054894625 | +0.0010385352456231845 |

对应平均 ΔLL 为 +0.001248643761276891、+0.005334281672863239、−0.001225341663044000、+0.000327687211976502。这些数值与原报告及 MAIN 台账相符。[原始端点](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/ENDPOINTS.csv)

晚期 E31–50 年龄匹配 AUC 增量为 ML Clean +0.002993±0.000422、ML Gaussian +0.003153±0.000325、KR Clean +0.004503±0.001214、KR Gaussian +0.003746±0.000840；前述 ± 是原报告的 seed 统计，不与本表 selected 配对 SD 混用。四组年龄匹配区间均支持历史组合的额外价值。[原报告](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)

### 3.2 真实 EMA 时间尺度

| 项目 | MovieLens | KuaiRand |
|---|---:|---:|
| 每 optimizer step 的固定 $\rho$ | 0.9828205985452511 | 0.995 |
| optimizer steps/epoch | 20 | 49 |
| 50 epoch 总 steps | 1000 | 2450 |
| 半衰期 $\log(0.5)/\log(\rho)$，steps | 40 | 138.282572986061 |
| 半衰期，epochs | 2 | 2.822093326246 |
| 稳态平均年龄 $\rho/(1-\rho)$，steps | 57.209245684958 | 199 |
| 90% / 95% / 99% 几何权重窗口，steps | 133 / 173 / 266 | 460 / 598 / 919 |

前四个协议值来自 [MECH 解析协议](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json) 与 [FixedShadowBank](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/shadows.py)，时间尺度由上述公式重算。它们是参数平均的权重性质，不是已经发现的“有效收益记忆长度”。本次窗口按真实 optimizer step 和这些半衰期构造，不搜索更优 $\rho$，也不把 epoch 数直接当相同历史长度。

### 3.3 真实参数分组

| 参数组 | ML 坐标数 | KR 坐标数 | 分组依据 |
|---|---:|---:|---|
| S | 483152 | 661920 | 普通字段 embedding、shared_weight/bias、skip_conn、star_mlp、共享 expert |
| 每个 Dd | 74960 | 295632 | slot_weight/bias[d]、domain_expert[d]、domain embedding 的实际第 d 行 |
| 每个 Tt | 16576 | 16576 | task_expert[t] |
| 每个 DTdt | 5381 | 5381 | gate/tower[3*t+d] |
| 全模型 | 773470 | 1614254 | 1×S+3×D+2×T+6×DT，互斥且完备 |

固定融合系数全为1时，cell(d,t) 的数据函数支持为 S、Dd、Tt、DTdt，共24个 cell/group 支持关系。通用 MLP 内部的 domain_specific 字符串不能决定专有归属；domain embedding 也不能整表归 S。coupled WD 或 Adam 历史引起的参数位移不等同于当前 cell 的数据梯度支持。[GROUP_MAP](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/GROUP_MAP.json)；[真实 registry](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/aug_history_20261105/group_registry.py)；[逐 cell 支持检查](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/aug_history_20261105/tests/test_a.py)

Stage 0 复用这一分组后，核对两个数据集的参数量、真实 lookup、alias 和 full-functional 支持；只做会影响归因正确性的有限检查。不能因单次 ReLU 导致某组零梯度，就改写结构支持关系。

### 3.4 自适应历史原表核查

- AUG B/A 的 HPO_SUMMARY.csv 各有30行，即两数据 × 15候选；各候选 seed_ids 均为2022–2026。它们证明当时比较范围与五 seed 完整性，不是新研究要重复运行的矩阵。[B 原表](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/artifacts/aug_history_20261105/attempt1/B/HPO_SUMMARY.csv)；[A 原表](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/artifacts/aug_history_20261105/attempt_A1/A/HPO_SUMMARY.csv)
- ADAPT A0_DECISION.json 的 state=NO_ACTIONABLE_SIGNAL、new_training_triggered=false，pulse_duration_epochs=1、observation_spacing_epochs=5。[原决策](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/artifacts/ema_adaptive_20261109/attempt1/A0_DECISION.json)
- ADAPT 的 KR Gaussian 静态策略原表含30条主条件记录，动作均为1，即 Fast，覆盖五个留出 seed × 两个训练面板方向 × GLOBAL/TASK0/TASK1。它直接支持“已选静态 task 策略没有区别于统一 Fast”，不支持“所有任务历史机制均无作用”。[原静态策略](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/artifacts/ema_adaptive_20261109/attempt1/A0_STATIC_POLICIES.csv)

## 4. 可复用代码映射与适配界线

下表全部以 MECH 固定 commit 为实施来源。新文件名和接口扩展由 [05](05_IMPLEMENTATION_SPEC.md) 定义；此处只列仓库中实际存在的模块，不把拟建 CLI 当成现成能力。

| 已有模块 / 接口 | 可复用内容 | 必须注意 |
|---|---|---|
| [aug_history_20261105/model_adapter.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/aug_history_20261105/model_adapter.py) | build_model；ModelAdapter.forward_logits | 参数输入、raw logits 与 full graph 钩子复用；正式训练保留原 gate.detach |
| [aug_history_20261105/registry.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/aug_history_20261105/registry.py) | ParameterLayout、names/shapes/offsets、flatten、copy_to | flatten 会 detach；路径积分带图参数按布局切片，不能误用该 flatten |
| [aug_history_20261105/group_registry.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/aug_history_20261105/group_registry.py) | build_path_groups、坐标 mask、source_cells、counts | 不凭名称新增分组；域表按行；记录 layout/group hash |
| [ema_mechanism_20261106/readouts.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/readouts.py) | read_predictions、PredictionBatch(p,z)、evaluate_full_validation、group_hybrid、panel_predictions | 正式 AUC 使用源 FP32 p 和 ties；raw z 用于 margin 解释，不能替换正式 p 评价 |
| [ema_mechanism_20261106/shadows.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/shadows.py) | FixedShadowBank.after_adam | 从 θ0 初始化，每真实 Adam step 后一次，无 warmup/偏差校正；不可整套照搬旧候选池 |
| [ema_mechanism_20261106/worker.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/worker.py) | 源损失、Adam、loader、RNG、export_session/restore_session | 采集与读出插件必须保持 online 参数/Adam/loader/Gaussian RNG 不变 |
| [ema_mechanism_20261106/observers.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/observers.py) | 旧面板、几何和功能指标语义 | 新全量转移账本与完整 R 因果分解需新增；旧统计不能冒充完整归因 |
| [ema_mechanism_20261106/state.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/state.py) | RAM 状态事务、原子标量写入、身份与幂等模式 | 历史参数只驻 RAM；丢失状态须按源初始化重放 |
| [overnight_ab_20261103/data_bridge.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/overnight_ab_20261103/data_bridge.py) | prepare_cache、cached_bundle(include_test=False) | 数据身份核验后才复用；训练 bundle 不载入 Test |
| [overnight_ab_20261103/selection.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/overnight_ab_20261103/selection.py) | 原 Validation 选点逻辑 | 最早严格最大值，不按显示舍入、LL 或 Test 改点 |
| [overnight_ab_20261103/state.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/overnight_ab_20261103/state.py) | CPU RAM 深拷贝、state_digest | 诊断 scratch 与可恢复训练状态不可互相覆盖 |

训练模型的关键计算路径见 [vendor_ga31/reference.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/reference.py)。STAR 与 skip 共同形成 h；h 同时进入共享/域/任务 expert 和 gate；gate 训练输入 detach。对 h 的功能替换必须覆盖所有出边，否则不是完整表示切口。第二切口是融合 expert 向 tower 的输入。完整切口与状态构造规范见 [03](03_CAUSAL_EXPERIMENT_DESIGN.md)。

## 5. 数据、模型与环境身份

| 项目 | MovieLens | KuaiRand |
|---|---|---|
| 原 CSV | /home_new/xzf/datasets/ml-1m/ml-1m.csv | /home_new/xzf/datasets/kuairand/processed/kuai_rand.csv |
| 原 CSV SHA-256 | f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4 | a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f |
| train / Validation / Test | 800167 / 100021 / 100021 | 2006132 / 250767 / 250767 |
| train d0 / d1 / d2 | 155949 / 318599 / 325619 | 172782 / 1743898 / 89452 |
| 域定义 | 年龄{1,18} / {25} / {35,45,50,56} | tab0 / tab1 / tab4 |
| 任务 | rating>3 / rating>4 | click / long_view |
| batch / 尾批人数 | 40960 / 21927 | 40960 / 40052 |
| coupled WD | 1e−4 | 1e−5 |
| Gaussian κ / λ / target | .05 / .5 / 0 | .1 / .5 / 0 |

共同配置：M3oE fixed1、embedding16、FCN [512,256,256,64]、4 shared experts，exp_d=exp_t=bal_d=bal_t=1；普通 Adam lr=.01、betas=(.9,.999)、eps=1e−8，foreach/fused/amsgrad=False；FP32、AMP/TF32/clipping关闭；50 epoch，无性能早停。自然80/10/10原行序划分，train-only编码/缩放，KR先split再筛tab0/1/4。loader每epoch使用 seed+11000 的独立CPU Generator。[解析协议](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json)；[worker](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/worker.py)

**Gaussian 目标域以实际调用为准。** 旧 nested source metadata 的 KuaiRand augmented_domain=2 不是本实验的实际目标；MECH configs.target_domain 和 worker.train_batch 的 source_losses 实参都是0。本轮使用 Gaussian D0，不能因旧字段误切 D2。两任务共用同一次 Gaussian view、replacement λ=.5；noise key 沿用 GA31-R3 的 dataset/seed/train/step/gaussian/1，不插入新的诊断 candidate ID。

源解释器为 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python；已记录环境 Python3.13.5、PyTorch2.9.0+cu126。原缓存路径为 /home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache。Stage 0 只把这些当已知来源位置，核对运行服务器上的真实路径、版本、raw/split/encoder/scaler/数组身份后复用；不能假定规划环境就是训练服务器，也不能不经身份核对自动改换人口或依赖版本。[来源收据](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/SOURCE_RECEIPT.json)

## 6. 资产可复用性与必要新采集

| 资产 | 当前已知状态 | 本轮处理 |
|---|---|---|
| 原始报告、CSV/JSON、协议、数据身份 | 固定 commit 中可读 | 直接作为已知证据，不重新跑实验确认 |
| 已审计训练、EMA、读出、参数 registry | 来源分支可读，main不保证有实现 | 在隔离 worktree 复用，必要改动集中于新实验包 |
| 旧模型、Adam、EMA、anchor 状态 | 释放收据为 RELEASED，释放参数payload 2,550,757,800 bytes；未落盘 | **不可从报告恢复**；必须重新采集20条同协议轨迹的所需状态 |
| 原服务器有限预测 NPZ | 旧流程曾保存于本地 predictions，仓库主要保留标量及身份；当前是否仍在需查 | 存在且身份匹配可复用旧诊断；不能当成参数 checkpoint，不能替代新干预预测 |
| 旧当前/未来 pulse 数值 | ADAPT 已审计复用，参数状态不在其中 | 可支持历史失败分类和指标语义，不重复计 seed |
| 新 $R_t$、$R_{t-h}$、参数组/时间混合状态 | 历史没有完整留存 | 新采集，在 RAM 稀疏事件快照中保留，及时消费 |
| 精确逐 step 的历史预测 EMA | 旧锚点预测不能补成完全等价序列 | 新训练时每step维护固定面板概率/logit EMA；从 θ0 分数初始化 |
| 全量排序转移账本、完整切口替换和路径闭合 | 历史未覆盖本次完整 R 问题 | 新模块输出有界表和预测；不存巨量成对矩阵或参数向量 |

释放事实来自 [MECH 收据](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/RELEASE_RECEIPT.json) 和 [AUG 最终报告的状态释放记录](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/REPORT.md)。现存文件可复用到什么层次必须按对象内容判断，不能用“已有实验产物”模糊替代参数状态。

## 7. 资源证据与本轮预算的关系

旧 MECH 的实测完成量为20条轨迹、1000 epoch、34500次正式 Adam；4500次完整 Validation 读出、5200新增pulse、2400小面板梯度。累计 worker 计时约训练13226.080秒、Validation2893.289秒、pulse553.674秒、梯度301.436秒、IPC2411.616秒，队列墙钟9980.239秒。峰值 CUDA allocated/reserved 约1.134/1.481 GB，worker/guardian RSS约3.923/3.633 GB，guardian参数payload约2.551 GB，观察输出约1.97 GB。[原成本](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/COSTS.csv)；[原报告](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)

这些计时来自共享 GPU 环境，worker累计值含并发等待，不能称为独占GPU耗时或直接保证本轮墙钟。新计划增加的全量干预、面板每步预测、切口替换和积分成本按 [06](06_SCHEDULER_AND_RESOURCES.md) 独立估算并在首批工程校准中记录；不以旧2.77小时直接承诺完成时间。

[start.md](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/start.md) 的约束包括：模型训练状态和 checkpoint 不落盘；每GPU显存及功耗均低于85%才准入新进程，整机RAM不得超过90%。本轮主动沿用旧计划的 **4 GiB guardian payload、2 GiB观察输出**作为保守预算；这两个数值来自旧研究计划，不冒称全局 AGENTS 的通用硬件上限。用条件分块、稀疏历史事件、及时释放临时状态满足预算；不得偷偷写参数临时文件或降低正式 batch/精度来规避。

## 8. 源码与指标的易错点

1. **AUC 的分数类型。** 正式六格 AUC 使用源 FP32 概率 p，按真实 ties 计半分。raw logit z 的严格单调性在数学实数中成立，但浮点 sigmoid 饱和可形成 ties；不能把 z-AUC 偷换为正式 p-AUC。margin功能分析同时记录 p 与 z 的对应变化。
2. **LL 的精确语义。** 源规则是 p 转float64后使用 max(log(p),−100) 与 max(log1p(−p),−100)，不是随意新增 epsilon-clamp。比较必须同口径。[源评价](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/readouts.py)
3. **参数导数语义。** 训练 gate.detach 不改变前向值，却截断 h→gate 的导数。完整函数归因用 scratch full graph；正式训练保持原样。单一一阶内积不是实际预测收益。[read_predictions](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/readouts.py)
4. **脉冲时钟。** 旧 Fast/Slow 是从窗口起点F初始化的一epoch实验，不是从θ0常驻，也不是新在线训练轨迹；预测e+5的新pulse不等于预测持续闭环控制。[ADAPT报告](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)
5. **旧特征命名。** M2 的 gradient_amplitude 是P_A完整功能rank风险梯度组RMS，不是实际训练Adam梯度；旧 global_preference 是对应单组读出的六格平均，不是全参数联合Fast模型。[解析协议 feature_blocks](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json)
6. **历史阴性与未运行。** AUG 的KR C、ADAPT A1/B/C、PUB+EMA均未触发，不能记作已运行负结果；其状态在原报告原样保留。
7. **旧研究的规模段落。** 只读取数据身份与实际计算需求；本轮不继承任何将域人数转写为困难度的解释或试验。固定边界“小域不是困难域”不进入待检验列表。

## 9. 本次证据完整性验收

Codex 在 Stage 0 输出来源清单，至少含：来源标识、固定commit、文件路径、实际文件hash、用途、读取状态、复用范围、是否含参数状态。对核心来源缺失，先通过上述固定commit恢复只读文件；不依赖浮动分支别名猜测内容。

后续报告每项主要结论要能追到本轮具体表的 seed/condition/anchor/intervention/state identity，并与本文件旧证据分开。旧结论标“已有证据”，新结果标“本轮实测”，推论标“机制解释”，未运行模块标“未运行/不触发”。**文档中没有预填任何新的机制赢家、参数贡献比例或研究方向胜出结果。**


