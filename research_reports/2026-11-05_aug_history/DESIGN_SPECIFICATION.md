# DESIGN SPECIFICATION：11/05 增强响应与参数历史实验

状态：DESIGN_COMPLETE。本规格将既有冻结方案映射到固定源码；首轮实现 B，A/C 按冻结决策触发后实施和审查。算法、候选、停止规则均继承 `research_plans/26-11-5/`，不新增搜索或消融。

## 1. 权威、身份与改动范围

- PLAN_REF：`2645f1f724f52f5ed5dea4b479335c0dc12fabcd`；规划位于 main 的 `research_plans/26-11-5/`。
- SOURCE_REF：`c8675ae2923c15b6965884b140b5b646690cd95e`。
- 执行 worktree：`/home/xzf/Code/phase0-3_aug_history_20261105`；执行分支：`26-11-05`。
- 入口：`07_FINAL_CODEX_ENTRY.md`，公共合同 `09_SHARED_ENGINEERING_CONTRACT.md`，数学和验收以 B1–B4 为准；选择、决策、搜索分别以 05、04、08 和 manifest 为准。
- 读取当前 main 的 `EXP_AI_CONTEXT.md`、`experiments/Experimental_Global_Summary.md`、`MDMT_ALGORITHM_METRICS_LEDGER.md` 的最新五 seed/发表模型/11/04 handoff。下方历史门不覆盖当前规则。
- 新代码仅在 `experiments/aug_history_20261105/`；对应报告、原始标量与必要台账增量属于交付范围。父训练代码、历史报告和原始 CSV 保持原身份。
- DESIGN → 独立 CODER → 独立 REVIEW → RUN → 主代理 ANALYZE。reviewer 必须审查本文件、实际实现和实施报告，通过后才启动正式训练。非机械运行子代理继承主代理模型。

## 2. 固定训练协议

| 项目 | MovieLens | KuaiRand |
|---|---:|---:|
| 原始 CSV | `/home_new/xzf/datasets/ml-1m/ml-1m.csv` | `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv` |
| raw SHA-256 | `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4` | `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f` |
| train / validation / test | 800167 / 100021 / 100021 | 2006132 / 250767 / 250767 |
| train d0 / d1 / d2 | 155949 / 318599 / 325619 | 172782 / 1743898 / 89452 |
| 任务 | rating>3 / rating>4 | is_click / long_view |
| 参数量 | 773470 | 1614254 |
| 每 epoch / 全预算 Adam 步数 | 20 / 1000 | 49 / 2450 |
| Gaussian κ / λ | .05 / .5 | .1 / .5 |
| 增强域 | d0 | d0 |
| 原 EMA ρ0 | .9828205985452511 | .995 |
| coupled weight decay | 1e-4 | 1e-5 |
| 可扰动字段数 | 6 | 33 |

共同：FULLTRAIN_NATURAL80_10_10，原始行序划分、train-only 编码和缩放；KR 先划分再筛 tab 0/1/4。3 域、2 任务、fixed1，四融合系数为 1，4 shared experts，embedding16，FCN `[512,256,256,64]`。batch40960，50 epoch，不早停。普通 `torch.optim.Adam(lr=.01, betas=(.9,.999), eps=1e-8)`；foreach/fused/amsgrad/decoupled-weight-decay、AMP、TF32、clipping 均关闭。seed 固定 2022–2026。

项目 Python 必须使用 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。运行收据保存实际 Python/PyTorch/CUDA/依赖版本与代码、manifest、数据、布局 hash，不升级训练依赖后沿用原身份。

## 3. 已核对的源集成路径

1. 数据使用 `mdmt_history_information_20261103.vendor.v3.vendor.ema02.data.load_dataset` 与 vendored `audited_data._load_*_base`。核对 raw、数组、split 行内容/行号、encoder/scaler hash 和人口；原始 CSV 可直接生成准确的自然人口，无需缺失的 `mdmt_next_20261024_strict_v1`。
2. 可复用已有只读 `.npy` 缓存 `/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache`，但按源 resolved protocol 核验缓存内容，而不只信任 manifest。训练 bundle 排除 test 数组；数据身份核对不计算 test 性能。
3. `history.load_protocol()` 依赖源 `research_plans/26-11-03/experiment_queue.yaml`；该文件已恢复。旧 YAML 的开发 seed 门和旧 A1 的 KR target=2 不进入本轮调度；本轮显式解析 manifest，KR target=0。
4. 模型工厂：vendored `gradient_aug_20261031_v2.views.make_model(bundle, dataset_source_conf)`。源输入 `[F,N]`、概率输出 `[N,2]`；gate 索引 `3*t+d`，训练 gate 输入 detach。所有参数进入布局与 EMA，包括 embedding、heads；不得只对共享专家做投影。
5. 原 `gaussian_domain_coverage_20261104/worker.py` 是 KR 专用：工厂配置、参数量、2450 步/人口断言均硬编码 KR。新 worker 适配两个数据集，不能直接套用其 `create_session`。
6. 原 coverage `objectives.losses` 的单域分支把 dataset 硬编码成 `kuairand`。B 的 loss 直接调用 vendored `gradient_aug_20261031_v2.objectives.losses`，传实际 dataset、target=0、`num_views=1/view_family=gaussian/pairing=iid`，保持源 scalar 运算次序。从返回的同图 `real`/`augmented` 派生每任务增强项，不重算正式 total。
7. 复用 `overnight_ab_20261103.evaluation`、`selection.consider`、`state` 和 RAM guardian 的既有行为，逐项适配新 family/job/session 身份、15 影子和 schema。旧 guardian 的进程启动入口指向旧 run，需适配新入口或显式复用支持的请求协议。
8. 正式发表模型引用源 A+B 五 seed、同自然人口和 AUC 选点端点。`research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv` 的固定 Git blob 为 `a3d7aa3e2035f0b7ff55406e260fc59c880ba254`。主参照固定 ML=M3oE fixed1、KR=HyperGate，同时列三发表模型；HyperGate 保留 UNVERIFIED_PROPOSAL 来源标记。

## 4. B 的精确批次语义

实际原 batch 大小为 N。源概率 BCE 给出 clean 矩阵 `real:[N,2]` 和 d0 单一 Gaussian 前向的 `augmented:[N0,2]`。

`L_a,t = λ * (augmented[:,t] - real[mask,t]).sum() / (2*N)`。

`LossTerms.total` 使用源返回 scalar；`clean` 与 `aug_by_cell[(0,t)]` 保留计算图。总梯度不含 L2，coupled L2 在 Adam 内只加一次。Gaussian 的样本/字段 RMS detach；不改 ID、dense、domain embedding 或标签；两任务共享一次增强前向、同一基础噪声。

刷新步精确执行：

1. `zero_grad(set_to_none=True)`，保存旧参数、旧 Adam state 与 param_groups 的只读视图/独立快照。
2. 一次源 clean/aug 前向得到 total 和两个增强项。
3. 两次 `autograd.grad(..., create_graph=False, retain_graph=True, allow_unused=True)` 提取 a0/a1，不写 `.grad`。
4. 正式 `total.backward()` 恰好一次，正式 g 从 `.grad` 读取；不把分解后的梯度重排写回。
5. 同一个旧 θ/m/v/逐参数 step 下，预测 U(g)、U(g−a0)、U(g−a1)、U(g−a0−a1)。使用正式 g 的 active mask；None 与显式 zero 保持不同，减去增强项不能将 zero 改成 None。
6. 得到 `d0=U(g)-U(g-a0)`、`d1=U(g)-U(g-a1)`、`dM=U(g)-U(g-a0-a1)`。U 返回源 FP32 新参数减旧参数的实际存储位移；counterfactual 不提交参数、moment、step、RNG。正式 optimizer 恰好 step 一次。
7. Q 由当前真实更新前响应生成；本步更新后的 θ 用于 15 影子各一次更新，buffers 从 online 复制。释放临时图与反事实状态。

非刷新步只有源损失、一次 backward/step 和既有 Q 的影子更新。生产反事实可用纯函数或对独立克隆 tensor 调源 Adam 计算；独立克隆模型+`torch.optim.Adam` 为验收真值。保留逐参数 bias correction、coupled L2 与 eps 的源顺序，拒绝不支持的优化器选项。

## 5. 子空间、影子与随机性

对 D 的非零列用 FP64 范数归一化；FP64 Gram 特征分解，保留 `λ > max(1e-12,1e-10*λmax)`。D 秩≤2，M 秩≤1；共线/反向共线降秩。Q 转 FP32 前后检查正交性。非有限值、严重负特征值、正交失败为工程错误。响应全为精确零或目标域缺失时，本次刷新将 Q 置空，直到下次刷新使用 F 递推。

每个 K 的刷新时钟为 `k>=1 and (k-1)%K==0`。K10/K20 共用每 10 步一次响应计算，K20 只在自己的时钟接收当前 Q。S 使用独立流生成的每 seed 固定坐标 Rademacher s，`Q_S=diag(s)Q_D`；不每步重抽、不用列置换代替方向消融。

每影子从 θ0 初始化、独立维护 φ。方向半衰期比 h 对应 `ρA=ρ0**(1/h)`，先保存 `innovation=θ_after−φ_old` 再更新：

`φ_new = ρ0*φ_old + (1−ρ0)*θ_after + (ρ0−ρA)*Q*(Q.T*innovation)`。

h=1 或空 Q 走源标准 EMA 路径，保证退化；U 用全参数统一 `ρ0**(1/h)`。方向/协方差 detach，无跨步图，无 P×P 矩阵。

基础 loader 为 seed+11000；基础 Gaussian 完全沿用源 `view_key/derive_seed`，键不加入 candidate/h/K。新增随机流为 SHA256 `AH1105|dataset|seed|global_step|stream|domain|task`，S 固定流不消费训练全局 RNG。验证、functional 读影子和探针保持训练模式、requires_grad、RNG、buffers；临时 hooks 用 try/finally 释放。

## 6. 有限矩阵、选择和阶段终止

每数据每 seed 只训练一个 G-D0 online。每条轨迹维护 F、U_FAST、U_SLOW，以及 D/M/S 各 h=.5/2×K10/20，共 15 影子。所有候选 50 次完整 validation，不按性能删候选。B 总计 10 条 online、500 epoch、17250 个真实 Adam step、150 个影子-seed实例、7500 次影子 validation+500 次 online validation。

每候选每 seed 用 validation 六格等权 AUC 最早严格最大值选择 epoch。所有候选五 seed 齐全后，每数据各家族按五个 selected validation AUC 均值选同一超参数；优先六格相对 F 配对均差均≥−.001 的候选。若无保护候选则仍选最高均分并标 VALIDATION_CELL_TRADEOFF，保留 unconstrained winner。精确并列按实测成本、较小父配置偏离、candidate_id 排序，不用显示舍入值或 LL 改选。

冻结各家族赢家、F、D* 同 h/K 的 M/S、online 的 selected/E10/E50 身份后才访问 test。所有候选 selected/E10/E50 参数在候选五 seed 汇总前留 RAM；可按相同 hash 去重，不能仅保留当时领先者。

B 决策逐数据使用 validation，顺序严格按 04：不完整/身份错误→WAIT_OR_REPAIR；资源/数值故障→对应 BLOCKED；D*≤F→C；D*相对 F 有任一格<−.001→A；合法无增强参考下仍有格代价且归档 online 代价支持→C；D*比 U/M/S 最强者优势≤.0002、或 direction_gain≤0、或 multi_response_gain≤0→A；其余保留 B 并报告。direction/multi gain 同时取家族赢家和匹配 h/K 控制差的较小值。online_cost_supported 只来自既有同协议共同 E50 或明确归档观察，缺证据为 unknown。

A/C 未触发为 DEFERRED_NOT_RUN；触发后完整读取四份独立规范、实施、独立审查、验收，再运行其冻结矩阵。A 需要从 θ0 重放 G-D0，不能从 B 终点补历史；C 各 γ/K 是独立轨迹，先四点依赖搜索、后两个匹配控制，每点五 seed 全预算。continue 可输出已解析人工命令与决策，不能据本轮 test 改派或扩网。完整任务执行到本决策树的预定义终止状态。

## 7. 最小实现与公开接口

依 09 提供 `protocol.resolve`、`model_adapter.build_model/forward_logits`、`objectives.loss_terms`、`registry.ParameterLayout`、`shadow_bank`、`worker.create_session/advance_epoch/export_session/restore_session`、`selection`、`run`、`report`，及 B 的 `adam_counterfactual/responses/subspace/directed_ema`。复用现有 helper；不加依赖、不改骨干层次。B 不需要 A/C 控制器和训练排序探针，延期接口在 validate 中明确 DESIGN_COMPLETE_NOT_IMPLEMENTED，不能假 PASS。

布局按源 named_parameters 的完整固定顺序记录 names/shapes/offsets/dtype/alias/hash；B 在完整参数空间操作。A 的活动 coordinate_groups 在 A 触发时依据真实计算图实现验收，不能现在用参数名猜分组。若 B 提供公共 logits 读出，从 tower 的原始输出按 `3*t+d` 与域 mask 拼 `[N,2]`，不得从饱和概率反推 logit；不改变概率训练前向。A/C 的 full-input derivative 与参数/δ 功能在对应路线触发后完成。

CLI 必须支持入口的 `preflight/validate/train/select/decide/evaluate/continue/report`、manifest/output、路线、数据、seeds、resume；--help 与 parse 检查通过。all suite 对 A/C 报设计完成未实施，B/公共全部通过即可 B 正式训练。退出码：0 完成、2 配置/数学不一致、3 工程阻塞、4 成本阻塞、5 等待五 seed 或依赖。

## 8. RAM 事务与资源调度

模型、Adam、影子、Q、selected/E10/E50 只存在 RAM 或 RAM guardian/IPC，磁盘仅代码、标量、JSON/hash、日志、报告；禁止参数/优化器 checkpoint，包括临时文件。快照包含 schema/source/manifest/data/layout/job 身份、online 参数/buffers、Adam 全状态、所有 φ/Q/时钟/固定符号、全部 RNG、loader 状态、epoch/batch/global_step、选择器/端点、累计成本与 commit_index。

完成 epoch+validation 后先生成/提交 RAM 状态，再提交对应标量索引；同 job/epoch/commit_index 幂等，不重复计训练，冲突 hash 报错。RAM 完整时准确恢复；丢失时同 seed 从 θ0 重放，校验既有顺序/hash 并追加 attempt 记录。不能从磁盘指标复原模型，不释放未冻结候选端点。

两张卡各自检查：单卡显存和功耗都低于各自上限 85%，且系统 RAM<90% 时追加可容纳原 batch 的并发训练；记录现有其他作业和本任务峰值。OOM 首先减少并驻 shadow/并发、顺序验证，保留原 batch、FP32 和完整候选。cost 分开记录训练、响应/probe、validation、IPC、GPU 秒、墙钟、GPU/RAM 峰值和故障重放。

## 9. 必需验收及证据

仅执行与本轮科学结论直接相关的公共+B 验收，完成后进入真实训练。

| 检查 | 接受标准 |
|---|---|
| 源兼容 | 两数据参数量正确；同初始/RNG/batch 下概率、loss、正式 grad、Adam update 与源一致；15 非反馈影子不改变 online hash |
| loss/输入 | toy FP64 `g=g0+a0+a1`；源 total 不重组；尾批实际 2N；空 d0 分支；Gaussian 不触动 domain/dense/label、RMS detach |
| Adam | 初始无状态、已有历史、逐参数不等 step、None vs zero、两数据 wd、不同 shape/unused 参数均对齐独立克隆 Adam；真实 state/RNG 调用前后不变 |
| Q/S | 零列、秩1/2、共线/反向共线、列置换/变号投影不变；S 保持秩/support/绝对值/正交性，方向投影可变；非有限值拒绝 |
| EMA/时钟 | h1/空 Q 退化 F；常量 θ 不变；正交手工例；15 候选各一次；K10/K20 从 k1 正确刷新，无未来 Q；恢复保留 Q 与时钟 |
| 选择/隔离 | validation 精确并列取最早；不足五 seed 拒绝 HPO；冻结前拒绝 test；共用一个 dataset 级配置；RNG/mode/buffer 恢复 |
| 状态/成本 | 连续与 RAM 恢复匹配；幂等 epoch；参数不落盘；同轨迹估计器不当 seed；原 batch GPU smoke 与资源/计时收据 |

FP32 默认 atol=1e-7、rtol=1e-5，完全同算子路径优先逐位；FP64 投影/矩阵 identity 容差1e-10。失败定位具体算子或协议，不自动放宽。smoke 标 ENGINEERING_ONLY，不以其 AUC 选型。

## 10. 交付和现有阻塞清单

输出 manifest 所列 SOURCE_RECEIPT、RESOLVED_PROTOCOL、DATA_MANIFEST、RUN_MANIFEST、实施/审查/工程检查报告，全部候选 VALIDATION_CURVES/HPO_SUMMARY/HPO_SELECTION、冻结选点、ENDPOINTS/PAIRED_SEED/PAIRED_SUMMARY、DECISION、COSTS 与中文 REPORT。每 seed/endpoint 保留配置、数据/代码/预处理/split、model hash、selected epoch、六格与 Macro/Traffic/Pooled AUC/原尺度 LL、阶段和成本。

分析单位为 seed：均值、样本标准差、配对差和 seed 级 95% CI；AUC≥+.003、逐格≥−.001、LL均差<0 分别裁定，不要求全部 seed 同向或区间排除零。报告总收益与相对 F 的组件增量，D 对 U/M/S 的独立调参及匹配控制差。分析种子一致性、域/任务、域大小与独立难度证据、model/训练阶段、矛盾结果、竞争解释及最能区分解释的下一实验，证据标 STRONG/MEDIUM/WEAK，不将小域当困难域。

静态核对发现的两个 KR 硬编码和源 YAML 依赖已在第 3 节明确。父代理确认两 raw checksum 与正式 baseline blob 已核对；已有自然人口缓存可用。B 不依赖缺失 strict_v1，也不依赖释放的历史 RAM 参数。尚待实施后的 runtime 数据身份/模型兼容/原 batch smoke 与独立审查；这些检查通过即可执行 B。A/C 尚未实施，按明确触发条件继续；不预填性能和赢家。

本轮不追加旧基线重训、谱分析、额外网格、B+C 组合或无关工程重构；需要保留的风险是 GPU 共享资源和 RAM 丢失可能增加重放成本，按既定恢复与成本记录处理。
