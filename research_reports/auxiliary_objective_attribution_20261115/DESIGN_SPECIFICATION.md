# DESIGN SPECIFICATION — 零扰动辅助目标归因

设计冻结：2026-11-15。范围为计划 Stage 0/1；Stage 2 尚未授权实现。权威入口为 `research_plans/26-11-15_auxiliary_objective_attribution/00_EXECUTION_RUNBOOK.md` 及同目录 01–09、manifest。用户规定仅使用 seeds **2022–2026**，不保存任何模型/优化器/checkpoint 文件。

## 1. 审计结论与完整矩阵

详见 `STAGE0_AUDIT.md/json`。当前 main 为 `e5738e52294f064c23a9b953fe74abb84de713df`；来源 pilot 为 `26-11-15@52b48c9aab066d56fa91a060864c7c9abf12e50c`，训练实现提交 `bf36e454adc2d0c569218a7993834f6d56cfc97f`，59 文件收据 hash 为 `a67f446902642f0ce777ac7736474b00a1400902184ccf0d461581ace1b9a491`。52 个公共 utility 文件与 main 逐字节相同。

| 臂 | MovieLens | KuaiRand | 处理 |
|---|---|---|---|
| CLEAN | 2022–2026 | 2022–2026 | 复用已审计的 10 条完整轨迹 |
| B | 2022–2026 | 2022–2026 | 重跑：旧日志没有逐行候选/accepted support |
| C | 2022–2026 | 2022–2026 | 同上 |
| Z-PAIR | 2022–2026 | 2022–2026 | 新运行 |
| Z-EXAMPLE | 2022–2026 | 2022–2026 | 新运行 |

新增 **40 条 × 50 epochs = 2000 epochs**，比原计划 20 条增加匹配 B/C 的 20 条；全部新增训练应有 **69000** 次 Adam 和 EMA 更新。MovieLens 每轨迹 1000 步，KuaiRand 每轨迹 2450 步。完整分析为 50 条轨迹（含复用 CLEAN）。不重跑 A、RANDOM、GAUSSIAN。

## 2. 最小实现边界

只新增 `experiments/auxiliary_objective_attribution_20261115/`、本报告目录和 `artifacts/auxiliary_objective_attribution_20261115/`。不得修改历史代码、vendor、报告、数据和根指标台账，不合并 pilot 分支。模块数按实际需要决定，不强制把建议文件列表逐个建成抽象。

新代码必须在 main 独立运行：模型、数据、优化器/RNG/EMA/评价 utility 从 main 的 `experiments.conditional_information_momentum_20261102_v3` 导入。可把已审计 pilot 的必要适配函数复写到新模块并标注来源固定提交，保持 B/C 公式及数值运算顺序；禁止运行时 import 另一工作区的 pilot 包。已有 RAM guardian 可直接 import 复用。

训练前将已审计 10 条 CLEAN 的 `result.json`、各 50 份 epoch 验证 JSON、batch 日志、各 data manifest 和来源 source/run/review/completion 收据复制到新 artifact 的 `reused/`，保存原始字节及 SHA256 映射。分析通过统一结果加载接口读取新运行与复用记录。原件只读；仓库报告引用固定提交的 GitHub 路径与新 `reused/` 相对路径，避免仅引用本机绝对路径。

数据缓存是数据文件，可显式传入既有 data-only cache 路径并只读核验；路径属于执行配置，不作为跨工作区源码依赖。无缓存时使用 main 数据 loader。核验 raw hash、数组 hash、原始行 ID/split/encoder/scaler hash；不重建已存在且核验通过的缓存。Test 数组仅允许身份核验，禁止预测或指标计算。

## 3. 冻结训练协议

- M3oE fixed1：3 domains × 2 tasks；embedding dim16；FCN `[512,256,256,64]`；4 experts；四个固定融合系数均为 1，无 DARTS/learned fusion/DBO。
- 自然人口及既有 natural80/10/10 split、train-only preprocessing；不改变域比例和标签。
- FP32，batch **40960**，drop_last=False，**50 epochs**；模型和 optimizer 每个 dataset/seed 初始化相同。
- `setup(seed, device)`；loader 为独立 CPU `torch.Generator().manual_seed(seed+11000)`，每 epoch `randperm`。
- 一个 Adam：lr=.01、betas=(.9,.999)、eps=1e-8、coupled WD ML=1e-4 / KR=1e-5、foreach/fused/amsgrad=False。
- EMA 初始复制初始模型；每次 Adam 后更新一次；rho ML=.9828205985452511 / KR=.995；buffers 复制。不得更改普通训练 gate 输入 detach。
- 每 epoch clean validation；online 与 fixed_ema 分别按六格等权 Macro AUC 的最早严格最大值选择。记录全部 50 epoch 与共同 epochs10/30/50。

## 4. 候选采样、有效性和零扰动

严格沿用 pilot 的键：`dataset|seed|epoch|global_step|stream` 经 SHA256 前 8 bytes little-endian 转整数建立 numpy `default_rng`。pair stream=`pairs`，example stream=`examples`。同族 B/Z-PAIR、C/Z-EXAMPLE 在相同 batch context 使用相同候选索引与次序；不得在 perturbation/诊断分支消费全局 numpy/torch RNG。

按 d=0..2、t=0..1 循环，k=min(128,n_positive,n_negative)。pair 先独立不放回取 k 个正、k 个负，交错排列 `[positive,negative]`；example 在该域自然行中不放回取 2k，不做标签平衡。空 cell 跳过，全部空时 M=0。

通过临时 embedding/tower hooks 获得 `[M,input_dim]` embedding 和 tower `task*3+domain` 的 logit。灵敏度探针临时把 gate 输入接回 `star_mlp+skip_conn`，只计算数值函数 Jacobian；训练前移除 hooks 并保持原 gate detach。`autograd.grad(logit[:,t].sum(), embedding)` 提取逐行梯度；不积累参数 `.grad`，不更新状态。

**B/Z-PAIR：** 不投影。令 g 为 target-logit 梯度、r=RMS(e)，正行 sign=-1、负行 sign=+1。raw delta=sign*.03*r*g/(norm(g)+1e-12)。pair margin m=z_pos-z_neg；predicted change c=(g_pos·delta_pos)-(g_neg·delta_neg)。两行 norm(g)>1e-10、r>1e-12，且 abs(m)>1e-7、abs(c)>1e-12 才有效；cap=min(1,abs(m)/(abs(c)+1e-12))，同 pair 的两行 delta 同乘 cap。无 A 的 projection/drift rejection/backtracking。

**C/Z-EXAMPLE：** g_bce=(sigmoid(z_target)-y_target)*g；delta=.03*r*g_bce/(norm(g_bce)+1e-12)；norm(g_bce)>1e-10 且 r>1e-12 的行有效；无 pair margin cap。

**Z 的关键顺序：先按其匹配 B/C 计算假想非零方向、cap 和有效 mask，再把 accepted delta 替换为精确零。** 不能先置零再执行 pair-change validity，否则会改变 support 定义。所有 delta 在训练 augmented forward 前 detach。

候选支持必须一致；模型状态在训练后分岔，adaptive accepted masks 可以不同。主 estimand 为同一候选 sampler/acceptance rule 下扰动算子的整体增量。不得把它称为逐步固定 accepted-support 因果效应。保存 trace 可在后续重放/比较 accepted 索引；本轮不冻结并强制共用某一臂的 accepted trace。

## 5. 目标与有效权重

保持 pilot 的概率 BCE 原语及 reduction 数值路径：

`L_clean = F.binary_cross_entropy(model(x), y)`

`L_aux = 2 * F.binary_cross_entropy(p_aug[target_entry], y[target_entry])`

`L_total = L_clean + .25 * L_aux`；M=0 时 L_aux 为同 device/dtype 标量零。仅 target task 计辅助损失，不向另一 task 加 loss。

每个自然 row-task 系数为 1/(2B)；每个 accepted 辅助 entry 为 .5/M，其相对单个 clean entry 的系数为 **B/M**。M>0 时总系数质量 1.5，M=0 时为1。B=40960、M=1536 时相对比为26.6666667。这只是可计算权重，不是性能结论。

每 step：一次 zero_grad；探针 `autograd.grad`；生成 detached delta；clean/aux forward；一次 combined backward；一次 Adam step；一次 after-Adam EMA。ledger 用 detached per-entry BCE 统计，不以诊断重排训练目标或修改归一化。

## 6. 必需数据日志

每 batch 保存 B、M、epoch/step、候选与 accepted 原始 row IDs/局部索引/task/domain/标签、pair membership、keep mask 与 skip reason；原始/截断/实际 delta norm、cap/skip/drift 诊断；clean/aux numerator/denominator、clean/aux/total loss；按 domain/task/label 的数量、系数质量与损失贡献；unique row 数与跨 task 重复次数。索引列表和 mask 可用标准 gzip JSONL，避免重复复制完整 feature/data。每 epoch 汇总权重和支持，必须保留可核对逐 batch 的原始数据。

每 dataset/seed/family 比较候选 exact identity、accepted intersection/union/Jaccard、每 cell/label accepted counts 与权重差；M=0 时 overlap 规则显式定义，不能除零。gradient clean/aux norm 与 cosine 在每 epoch 第一 batch 作固定诊断即可，使用不改变 `.grad`/RNG 的 `autograd.grad`，不是阶段门。

保留各 epoch 未缩放 validation 六格 AUC/LogLoss、Macro、Traffic、Pooled、最差 cell、选点 epoch/state hash、初始/最终状态 hash、行序 hash、optimizer/EMA 更新数、实际 wall/throughput、GPU peak allocated/reserved、RAM 和探针/训练计时。训练权重/optimizer/RNG/loader/selected 模型只留 RAM guardian。

## 7. 最小正确性门

集成小组 CPU 测试覆盖：

1. Z accepted delta bitwise zero；相同 selected 输入的 zero/no-delta forward 一致；zero aux 等于同 row/task clean BCE。
2. total loss 及参数梯度等于显式 clean+.25*aux；M=0 精确 clean；一次 optimizer/EMA 更新。
3. 候选 sampler 可复现、stream 独立、全局 RNG 不动；domain/task/label/pair 次序、row IDs、counts 和 ledger 分母/质量正确。
4. probe/ordinary logits 输出一致；实际固定模型逐 row 梯度与 logit-sum 梯度一致，加入无关行不改变目标行梯度；hooks 在正常/异常路径移除；训练 gate detach 保留；delta detach。
5. 采用同 FP32 初始化系数的 float64 副本，在光滑点做有限差分，步长1e-5、FD rtol1e-3/atol1e-7；实际 FP32 梯度 vs 同系数 FP64 probe 为 rtol1e-2/atol1e-4。沿用 pilot 审查后规则，不重复旧的大位移探针错误。
6. RAM snapshot/restore 包含 model、Adam、EMA、Python/numpy/torch/CUDA RNG、loader、epoch/step、best/selected 状态和更新计数；小批无中断与恢复续跑状态及 selected identity 一致。

每数据集以真实自然 first full batch 对 B/C/Z 两族做 full-batch backward smoke，保留 shape/device/finite loss/一次更新/显存计时证据；实际源模型的四行 Jacobian 检查合并在 smoke 中。只做与上述义务相对应的检查。通过独立 reviewer 的 PASS 或已修正阻塞项的 PASS WITH FIXES 后才能开始正式训练。

## 8. 运行、恢复、审查

runner 只执行冻结矩阵和记录原始数据，不分析科学结论。每卡分别按显存<85% 且功耗<85% 满足时准入，host RAM<90%；首 full-batch 峰值未知时该卡最多一个 allocating worker，已知后按实测 peak/当前资源继续并发，无需等两卡同时空闲。不中断无关进程，不降低 batch/epoch/seed/data budget。

RAM guardian 每 epoch 保存完整继续与 selected state；失败尝试日志不可覆盖，retry 建 attemptN+1 并验证源码/config/data identity；不得把不同源码的部分轨迹拼接。若 guardian 状态丢失，只能以新 attempt 完整重跑，不能声称恢复成功。完成 Stage1 后先生成 paired 原始统计，再由独立 reviewer 核对结果身份、完成量和决策；主 agent 承担科学分析。

## 9. 分析和停止门

主对比按 dataset/seed 配对：Z-PAIR−CLEAN、Z-EXAMPLE−CLEAN、B−Z-PAIR、C−Z-EXAMPLE、Z-PAIR−Z-EXAMPLE。分别报告 online/fixed EMA，selected 和 E10/E30/E50，并分离 candidate+EMA−clean online 总收益、candidate+EMA−clean EMA 组件增量、candidate online−clean online。

每指标给五个原始差值、均值、sample SD、95% t区间 `mean ± 2.776445*SD/sqrt(5)`、正向 seed 数；六格及 per-task/domain、Traffic/Pooled、最差 cell 一并展示。AUC 与 LogLoss 独立裁定，不用梯度统计代替排序结果。各 dataset 单独判断，不用跨数据集平均掩盖反号。

按计划09的决策树逐条解释：两种 zero 都有害且两个扰动增量非正时停止该族；zero 中性/正、扰动负时停止当前扰动配方；稳定正扰动增量或 ledger 识别的具体可修正目标权重问题才允许考虑 Stage2。已知旧权重公式本身不自动触发 Stage2；若全部效果小/不确定且不改变路线，停止。均值>0 是计划的正增量定义，不新增 all-seed-positive 或 CI排零硬门。支持变化仅在能改变 Stage2 选择时触发最多一个针对性 intention-to-augment 对照，先单独设计审查。

Stage1决策前 **不实现/运行 NDP、HPO、Test 或额外消融**。后续未触发记 NOT_TRIGGERED；工程阻塞单独记 ENGINEERING_BLOCKED。报告证据强度 STRONG/MEDIUM/WEAK、矛盾、替代解释及下一项最有判别力的实验。域大小仅描述自然人口，不作为困难度代理。

输出包括 run/source/data/config/selection receipts、逐轨迹结果和50epoch曲线、压缩支持/权重ledger、`COSTS.csv`、`ENDPOINTS.csv`、`PAIRED_SEEDS.csv`、`PAIRED_SUMMARY.csv`、`decision.json` 和中文报告；decision 包含五seed原始与配对值、六格差、权重/overlap、所有hash、Test=false、各阶段状态和独立审查结论。
