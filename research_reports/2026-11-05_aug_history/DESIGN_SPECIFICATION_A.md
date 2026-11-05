# DESIGN SPECIFICATION A：实际路径分组 EMA

状态：DESIGN_COMPLETE，可交独立 coder。B 冻结的 validation 决策对 MovieLens、KuaiRand 均触发 `RUN_A/B_SIMPLE_SUFFICIENT`。本规格映射既有 A1–A4、04/05/08/09 与 manifest，不增加算法、候选或测试矩阵。

## 1. 身份与实施边界

PLAN_REF=`2645f1f724f52f5ed5dea4b479335c0dc12fabcd`，骨干 SOURCE_REF=`c8675ae2923c15b6965884b140b5b646690cd95e`。B 实施提交为 `79690772dedefafa24b80d1817aca8c61907be5b`，source hash=`9baac157d4574ccc2ec02a9db42cd365ec918346c32b1b3ace2dcc6f65893f2a`，manifest hash=`76345194a8d09c8cadf3544e39c35174a452abb3827f071250da7f259c45749d`，selection hash=`fce00de767c5e95b0cb627f53eb5851d43a6c1e64a8c4786db73e57c0ff202a4`。

`artifacts/aug_history_20261105/attempt1/` 及其 B 决策/冻结/端点保持只读；B 评价完成前不修改源码。新增 A 代码必然改变整包 source hash，使用独立 `artifacts/aug_history_20261105/attempt_A1/`，由新源码生成 PREPARED/SOURCE_RECEIPT/RUN_MANIFEST、A 工程检查和独立 REVIEW_GATE。保留 B 的 Git 提交和旧源码收据，不把新 A 代码标成 B 的旧 hash。

继续来源记录在 A 输出的 `CONTINUATION.json`：B DECISION 和 selection_frozen 的路径、文件 hash、selection hash、B prepared/source/manifest 身份及允许的两数据集 A_FULL 动作。读取 B 时按其自身 PREPARED+冻结文件互相核对；不调用要求“当前源码=B旧源码”的 `prepared(attempt1)`，不回写 B 文件。A preflight 仍严格核验当前 A 源码、原 manifest、同源数据与新输出身份。

新增 `group_registry.py`、`grouped_ema.py`、`ranking_probe.py`；对现有 protocol/worker/run/selection/report 作必要路线分派。复用现有模型、概率 BCE/Gaussian、Adam、validation、RAM guardian 和资源调度；不改 vendor，不改已验收的 B 数学路径，不增加依赖。新增 A 必须经过独立 coder→reviewer→runner；A REVIEW_GATE 绑定新 source hash。C 保持 DESIGN_COMPLETE_NOT_IMPLEMENTED，直至 A 决策触发。

## 2. 原父训练与完整矩阵

两数据固定 G-D0，使用原自然人口与 train-only 预处理、原 CSV/cache 身份；KR 显式 target=0。batch40960、50 epoch、FP32、普通 coupled Adam、原 κ/λ/ρ0/wd/loader/Gaussian RNG 均不改变。每数据 seeds 2022–2026，每 seed 从 θ0 重放 online 和全部 15 影子；B 的终点或标量不能生成 A 历史。

| 家族 | 配置 | 数量 |
|---|---|---:|
| U | F、U_FAST(h=.5)、U_SLOW(h=2) | 3 |
| A_FIXED | S_FAST、S_SLOW、D_FAST、T_FAST；中性配置直接复用 F | 4 |
| A_GROUP | η=.05/.2 × K20/40，独立 12 维 q | 4 |
| A_GLOBAL | 相同 η/K 网格，独立标量 q | 4 |

配置 ID 完全使用 manifest。固定半衰期 profile 按 (S,D,T,DT)：S_FAST=(.5,2,2,2)，S_SLOW=(2,.5,.5,.5)，D_FAST=(1,.5,2,1)，T_FAST=(1,2,.5,1)。固定 profile 初始化即展开，不读取域样本量改变系数。

A 完整预算为 10 条 online、500 epoch、17250 个真实 Adam step、150 个影子-seed实例、7500 次影子 validation+500 次 online validation。所有候选每 epoch 完整验证，不预筛、不早停。B RAM 在完成冻结评价/报告后可释放；A 使用自己的 guardian 状态。

## 3. 精确坐标分组

以现有 `ParameterLayout` 的 named_parameters 顺序和 alias 去重为基础；新增独立 GroupRegistry/hash，输出逐数据 `GROUP_MAP.json`。每条记录含 parameter_name、shape、flat ranges/row indices、group、numel、source_cells。组顺序固定 `S,D0,D1,D2,T0,T1,DT00,DT01,DT10,DT11,DT20,DT21`；可有 INACTIVE。

| 实际 source 参数/坐标 | 归属 | 允许的数据监督来源 |
|---|---|---|
| `shared_weight`、`shared_bias`、`skip_conn.*`、`star_mlp.*`、`expert.*` | S | 全六格 |
| 普通 feature 的 `embedding.embed_dict.<feature_index>.weight` 整表 | S | 全六格；不按本次 ID 出现域分类 |
| `slot_weight.<d>`、`slot_bias.<d>`、`domain_expert.<d>.*` | Dd | (d,0)、(d,1) |
| domain_indicator embedding 的实际 lookup 行 d，整行 16 坐标 | Dd | (d,0)、(d,1) |
| `task_expert.<t>.*` | Tt | (0,t)、(1,t)、(2,t) |
| `gate.<3*t+d>.*`、`tower.<3*t+d>.*` | DTdt | 仅 (d,t) |

源 `MLP_N.domain_specific` 是通用内部层名，按父模块归组。固定 fusion=1 时其它域/任务专家即使被计算，其不允许来源的数据梯度仍为零；coupled L2/Adam 历史产生的位移不属于这项数据梯度检查。

tower/gate 的 0/1/2/3/4/5 分别映射 DT00/DT10/DT20/DT01/DT11/DT21，不能与上述 q 向量的 DT 列表简单 zip。domain_indicator 列通过 `bundle.features.columns.index(...)` 定位，再核验对应 SparseFeature/实际 embed_dict lookup；不能把序号写死。

已通过只读源模型构造核对：

| 项目 | MovieLens | KuaiRand |
|---|---:|---:|
| domain_indicator 列索引/参数名 | 7 / `embedding.embed_dict.7.weight` | 33 / `embedding.embed_dict.33.weight` |
| 域表 shape；合法训练 lookup 值 | [3,16]；0/1/2 | [3,16]；0/1/2 |
| S 坐标数 | 483152 | 661920 |
| 每个 Dd 坐标数 | 74960 | 295632 |
| 每个 Tt 坐标数 | 16576 | 16576 |
| 每个 DTdt 坐标数 | 5381 | 5381 |
| INACTIVE / 全参数坐标数 | 0 / 773470 | 0 / 1614254 |

生产仍从模型和 schema 生成以上映射并核对数目。若合法输入永不使用的域表额外行存在，则标 INACTIVE、固定 ρ0；其余新增未知参数失败，不默默归 S。所有坐标恰好覆盖一次，域表不再整体归 S，alias 不重复更新。

## 4. 探针与 q 更新的实现合同

`sample_rank_probe(train_bundle, excluded_row_ids, seed, step)` 只接受 train bundle。每格在排除当前原 batch 的 raw row IDs 后，从正/负 train 池独立无放回抽各最多16行；不足用全部，不复制。源 train 数组位置与 raw row ID 分开保存。六格最多192个 cell-row实例，可对跨任务重复原行去重，保留每格正负索引。缺一类跳过该格；全空保留 q，记录原因。

独立 SHA256 key 使用 `AH1105|dataset|seed|step|A_RANK|domain|task`。同一步所有待刷新 A 候选共用这份行抽样；不含候选 ID/η/K，不消费初始化、loader、Gaussian 或训练全局 RNG。可采用按 key 重建 Generator 的无状态实现；快照记录键规范/最近 probe step/hash，任何持久 Generator 则完整保存其状态。

每候选从 θ0 初始化 φ，自适应 q=0。`ρ_b(q)=ρ0**(2**(-q_b))`。刷新在 k=K,2K,...，位于本步真实 Adam 后、正式 EMA 前；K20 首次20，K40 首次40。各候选使用自身旧 φ_(k−1)/q；只共享 probe 行。

`virtual_parameters` 返回同名同形 OrderedDict：`ψ_b=ρ_b(q)**K * detach(φ_old,b)+(1−ρ_b(q)**K)*detach(θ_after,b)`。θ/φ 无图，仅 q 可微。全局版本仅一个 q 广播至活动坐标；INACTIVE 固定原 ρ0。参数化 forward 复用现有 `ModelAdapter.forward_logits(..., params=ψ, probe_input_full_graph=False)`；它已通过 tower hook+functional_call 读取 raw logits，保留原 gate 输入 detach，不启用 C 的 full-input derivative，也不从概率反推 logit。

排序目标为有效格等权平均，每格对全部正负配对平均 `softplus(z_neg−z_pos)`；不以样本数加权格。只调用 `autograd.grad(rank_loss,q)`，不调用 rank_loss.backward，不写在线参数 `.grad`，不新增 optimizer step。梯度有限且非零时 `q_new=clamp(q−η*g/max(norm_inf(g),1e-12),−1,1)`；g=0/无有效格保持 q，非有限值报数值错误。更新结果 detach，不能累积跨步图。

随后按新 q 做本步一次普通分组 EMA：`φ_b ← ρ_b*φ_old,b+(1−ρ_b)*θ_after,b`，buffers 复制。非刷新步沿用 q。F/全部 q0 使用源原地 `mul_(ρ0).add_(θ,alpha=1−ρ0)` 次序，保证 F 退化与原源路径一致；固定 profile 仅更新 φ、不跑探针。所有影子每 batch 各一次；U 与 F 可直接复用现有统一 EMA。

新 `GroupRegistry`、`grouped_update`、`virtual_parameters`、`ranking_objective`、`controller_step` 和 `GroupedEMAState` 接口按 A2 实现。现有 worker 的 A 分支跳过 B 的 auxiliary gradients/counterfactual/Q；原 loss→一次 backward→一次 Adam 保持不变，额外把当前 batch raw row IDs 传给共享 probe。controller 的 autograd 仅包住 ψ/q 探针，分组参数更新 no_grad，避免把现有 B bank 的整段 no_grad 直接套在 q 求导外层。

## 5. RAM 状态、计时与必要验收

沿用已工作的 guardian 事务、worker 回收和恢复逻辑，不另建恢复框架。A snapshot 带 route=A、新 schema/source/manifest/data/layout/group hash、完整 online/Adam/loader/global RNG，以及全部候选 φ/buffers/q/η/K/最近与下次刷新步/controller 次数/probe 身份、selection/best score/selected/E10/E50、累计候选成本与 commit_index。恢复时先核对 route/身份再还原；不能重置 q、重新初始化 shadow、漏还原选择器或把刷新时钟变成 B 的 k1 时钟。参数与 optimizer/ψ 不写磁盘。

临时 ψ 图逐候选释放，独立 scratch 评价影子，原始 batch/FP32 不变。分候选记录真实 controller/probe、EMA update 和 validation 时间；固定候选 probe=0。共享抽样成本单列；HPO 成本比较不能让全部自适应候选只继承相同 B response 时间。GPU/RAM 并发仍使用已有85%显存+功耗、90% RAM规则。

最小必需检查仅覆盖 A2 和公共合同新增行为：

- GroupRegistry 全覆盖/互斥/alias、两数据上述坐标数、域表实际 lookup、六个 gate/tower 路径。单 cell 数据梯度检查禁止来源为零；有限组合法输入确认主要允许路径有可响应实例，不因单次 ReLU=0判路径不存在。
- F/q0 精确退化；固定 profile 小张量手算；常量 θ 不变；group/global q 维度正确、非零 q 梯度实际影响 φ、步幅≤η和边界、空 probe 保持、非有限值拒绝。
- probe train-only、排除当前 raw IDs、无放回/缺类/共享行；原 mode/θ/Adam/buffers/loader/Gaussian RNG 与正式数据梯度不受污染。raw logits sigmoid 与源概率匹配，gate detach保持。
- K20/40 后 Adam 前 EMA 的时序；两种 q 控制器的 RAM 恢复与连续运行一致（q/φ/时钟/选点/RNG）；15影子从θ0、每步各一次、online hash不变。复用已验收的 HPO 五seed门/test冻结门/epoch幂等检查，不扩为无关测试。
- A 原 batch GPU smoke 含源 online 一致性和实际 q 探针路径，标 ENGINEERING_ONLY；只判正确性/资源。reviewer 对新源码与 coder 报告独立给 PASS 或已修完阻塞的 PASS WITH FIXES 后才派发完整矩阵。

## 6. 路线选择、冻结与后续动作

选择逻辑复用05：每 candidate/seed 按50次 validation macro AUC最早严格最大值选点，所有候选五seed齐全后才按同配置五seed均值排名；先六格相对本轮 A/F 保护，再分数、实测成本、静态配置偏离和 candidate_id。保持 unconstrained winner 与 tradeoff 标记，不用 test 或 LL 改选，不强制各 seed 同向。

家族集合：A_group*=四 GROUP，A_global*=四 GLOBAL，A_fixed*=四固定profile+F，U*=F/U_FAST/U_SLOW；F参与两个家族集合但只训练/保存一个估计器。固定 profile 的偏离按其明确的 log2 半衰期向量，adaptive 候选保持共同的初始q0/边界，不能把缺少 `half_life_ratio` 当坏配置或借 test 定义偏离。

对四家族赢家依04按同一 validation 规则比较；简单候选与最优相差≤.0002且成本更低时可优先简单版本，记录选择对象、精确分差和成本。所选方案相对本轮 F 有正 validation 增量且六格保护通过→RETAIN_A_OR_B_AND_REPORT；否则且 C 未执行→RUN_C。不把+.003当 A 阶段新增正增量门，不因 F 赢得简化就把整个方向写成失败。

正式 test 端点冻结至少包含 F、U*、A_group*、A_global*、A_fixed*、A_group*同η/K的 GLOBAL 控制和 online，各 selected/E10/E50；重名去重。在 A 的 HPO_SELECTION/selection_frozen/validation DECISION 全部冻结后评价 test；不回写 B 的动作，也不按 A test 改派 C。

报告所有候选五seed HPO、完整V曲线、q/probe刷新标量、端点和配置身份、六格/Macro/Traffic/Pooled AUC/LL、配对seed/均值/SD/CI、成本。完整方法对三个源发表模型与组件对本轮 F 分开；分组对 GLOBAL、固定profile、U 及匹配η/K控制分开。A/B 成果均保留，C 未触发为 DEFERRED_NOT_RUN。

## 7. 公开 CLI 与跨阶段流程

新源码将 `resolve/preflight` 增加路线选择（默认 B 保持既有调用）；job/prepared/worker/guardian、review_gate、验证、select/decide/evaluate/report 使用明确 route，不将 A 结果写进硬编码 B 目录。private worker 也从 prepared job或显式参数取得 A，不能落回 parser 的 B 默认值。`continue` 按传入 DECISION 的所属 attempt 与冻结身份读取上阶段，写动作收据到当前 A 输出；允许输出已解析人工命令，仍须完成授权 A 训练。

```bash
PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
PLAN=research_plans/26-11-5/experiment_manifest.json
OUT=artifacts/aug_history_20261105/attempt_A1
BDEC=artifacts/aug_history_20261105/attempt1/B/DECISION.json
$PY -m experiments.aug_history_20261105.run preflight --manifest "$PLAN" --route A --output "$OUT"
$PY -m experiments.aug_history_20261105.run validate --manifest "$PLAN" --route A --suite A --gpu-smoke --output "$OUT"
# 独立审查通过、A REVIEW_GATE绑定本次source后继续。
$PY -m experiments.aug_history_20261105.run continue --manifest "$PLAN" --route A --decision "$BDEC" --output "$OUT" --resume
$PY -m experiments.aug_history_20261105.run train --manifest "$PLAN" --route A --datasets movielens kuairand --seeds 2022 2023 2024 2025 2026 --output "$OUT" --resume
$PY -m experiments.aug_history_20261105.run select --manifest "$PLAN" --route A --split validation --output "$OUT"
$PY -m experiments.aug_history_20261105.run decide --manifest "$PLAN" --route A --basis validation --output "$OUT"
$PY -m experiments.aug_history_20261105.run evaluate --manifest "$PLAN" --route A --frozen "$OUT/A/selection_frozen.json" --output "$OUT"
$PY -m experiments.aug_history_20261105.run report --manifest "$PLAN" --route A --output "$OUT"
$PY -m experiments.aug_history_20261105.run continue --manifest "$PLAN" --route A --decision "$OUT/A/DECISION.json" --output "$OUT" --resume
```

continue 已自动完成训练时，train 对完成且身份一致的 jobs 跳过。C 被触发后读取 C1–C4、由独立 coder/reviewer 实施验收并在新阶段身份下运行规定矩阵；不以未实施直接结束本次授权任务。只在04预定义停止状态或具体工程/成本阻塞时报告对应状态。

本阶段不重复 B 训练，不追加 q 网格、B+C 组合、额外机制排除实验或恢复框架；A 新增源码需要新 preflight 和审查身份，旧 B 冻结产物继续按旧身份保存。
