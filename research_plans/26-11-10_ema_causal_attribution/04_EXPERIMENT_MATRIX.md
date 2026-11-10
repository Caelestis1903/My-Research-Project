# 实验矩阵、冻结配置与评价人口

状态：**DESIGN_ONLY**。所有数值超参来自已审计来源或明确标为本次诊断预算，不进行常规HPO。完整工程配置由未来 `prepare` 从本文及源协议生成、冻结为 `RESOLVED_PROTOCOL.json`。

## 1. 唯一正式训练矩阵：20条轨迹

| 条件ID | 数据集 | 训练目标 | seed | epoch | step/epoch | 总真实Adam步 | 研究角色 |
|---|---|---|---|---:|---:|---:|---|
| ML-C | MovieLens | Clean | 2022–2026 | 50 | 20 | 5,000 | 主要机制发现 |
| KR-C | KuaiRand | Clean | 2022–2026 | 50 | 49 | 12,250 | 主要机制发现 |
| ML-G | MovieLens | Gaussian D0 | 2022–2026 | 50 | 20 | 5,000 | 训练目标边界 |
| KR-G | KuaiRand | Gaussian D0 | 2022–2026 | 50 | 49 | 12,250 | 训练目标边界 |
| 合计 | 两数据 | 两目标 | 每条件5个 | 1,000 epoch | — | **34,500** | **20条独立训练身份** |

`online/F/所有hybrid/时间分解/预测EMA`共享各自父训练轨迹，不增加独立seed或训练数量。与旧轨迹同seed确定性重放也不能当新增独立确认。旧状态已释放，需要本轮新采集，但这20条的目的在于取得尚未保存的参数/历史证据。

无需新训 M3oE/HyperGate/AutoIFS 性能基线矩阵；论文模型身份及既有结果只作历史参照。本次内部控制为同轨迹 O/F。

## 2. 冻结的来源训练配置

来源为 [26-11-07 的协议](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json)与其 `worker.py` 实际调用。

| 项 | 冻结值 |
|---|---|
| backbone | M3oE fixed1；embedding16；FCN[512,256,256,64]；4 shared experts |
| fusion | exp_d=exp_t=bal_d=bal_t=1；不恢复可学习融合或AutoML |
| optimizer | `torch.optim.Adam`，coupled WD；lr=.01；betas=(.9,.999)；eps=1e-8 |
| WD | ML1e-4；KR1e-5 |
| batch | 40960；保留真实tail batch；不改训练batch以省归因成本 |
| precision | FP32；AMP/TF32关闭；foreach/fused/amsgrad关闭 |
| steps | ML20/epoch、50epoch1000；KR49/epoch、50epoch2450 |
| F | theta0初始化；每次真实Adam后更新一次；ML rho=.9828205985452511，KR rho=.995 |
| buffers | 复制online；源图LayerNorm，无BatchNorm/Dropout；实际审计复核 |
| loader | 原样本顺序规则；独立loader generator=seed+11000；沿用noise key |
| loss | 源Clean `real.mean(1).mean()`运算顺序；不改变归约顺序 |
| Gaussian | 仅D0；lambda=.5替换；kappa ML.05/KR.1；对非routing稀疏embedding字段施扰动 |
| 模型选择 | 每个O/F各自完整Validation六格等权Macro AUC严格最大，完全并列取最早epoch |
| 终止 | 满50epoch；无科学性能提前停止；没有单seed晋级/否决门 |

Gaussian 的损失是 `(sum(real_bce)+lambda*(sum(aug_d0_bce)-sum(real_d0_bce)))/(2*N_actual)`。禁止把lambda改成额外加和损失。嵌套旧元数据 `configs.kuairand.source.augmented_domain=2` 不控制本轮；实际 source loss target 参数必须为0，并记录生效值。

## 3. 数据身份及人口

| 数据集 | train/Validation/Test | train域人数[d0,d1,d2] | 实际tail batch |
|---|---|---|---:|
| MovieLens | 800167 / 100021 / 100021 | [155949,318599,325619] | 21927 |
| KuaiRand | 2006132 / 250767 / 250767 | [172782,1743898,89452] | 40052 |

以上仅作数据身份和资源核对，**不赋予任何困难度含义**。自然人口、原行序80/10/10、train-only编码/缩放均继承源实现；KR原始文件先切分再筛既有tab映射，不重新排序或分层。ML任务为rating>3与rating>4；KR任务与域映射从数据manifest读取，不凭旧聊天或列名猜测。

原始文件默认位置可被显式参数覆盖，但SHA与cache身份必须一致：

- ML `/home_new/xzf/datasets/ml-1m/ml-1m.csv`，SHA256 `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`。
- KR `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`，SHA256 `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。

优先复用通过hash核对的本地cache；没有cache但原始文件身份正确时，使用源 `prepare_cache` 在本轮数据目录重建。缺失原文件或hash不符为 `DATA_IDENTITY_BLOCKED`，不能替换成近似数据继续执行。

## 4. 观测矩阵：同一轨迹多个读出

| 观测 | 时点 | 状态/次数上限 | 人口 | 是否新增训练 |
|---|---|---|---|---|
| O/F曲线与选点 | 每epoch1–50 | 2×50×20=2000完整V读出 | 完整V；P_A/P_B/V_mech | 否 |
| 在线预测及历史预测EMA | 初始化+每真实step | q_p/q_z两个流式累积；在线逐步分数 | 固定面板 | 否 |
| B核心S/N | 五固定anchor+tF | 每时刻2额外状态，O/F复用 | 完整V+面板 | 否 |
| C核心recent/old | 同上 | 每时刻4h×2额外状态，退化去重 | 完整V+面板 | 否 |
| B细化 | E10/E35/E50+tF | 每时刻至多12额外状态 | 完整V+面板 | 否 |
| C重要组×时间 | 同上 | 至多2组、h=round(H)、2背景；端点复用 | 完整V+面板 | 否 |
| D图接口四态 | 同上 | O→F及最多1主contrast；2cut；每cut2mixed | 完整V+面板 | 否 |
| D专家融合分量 | E35+tF | 五项各删除，再将top1项在主组/时间contrast上做一次删除；6状态封顶 | 完整V+面板 | 否 |
| D参数路径积分 | 同上 | 8→16→32点；固定有界诊断行 | 诊断面板 | 否 |
| D STAR交叉项 | 有条件、同上 | 一个来源明确的交叉项删除/注入 | 完整V+面板 | 否 |
| H1/H2/H3桥接 | E5,E10,…,E50 | 两套7动作，F共享；既有pulse | 面板；核心anchor完整V结果复用 | 否 |
| H2时域/幅度判别 | 有条件执行；E35/E50预采状态 | H步fast/slow两个向量；至多2个同scope等半径对照、4状态/anchor | P_A选动作；V_B评并记录完整V | 否 |
| Adam来源诊断 | 有条件，仅Clean E20/E35 | 每dataset五seed；2反事实方向+实际状态 | 固定训练/验证面板 | **真实Adam新增0** |
| 冻结后Test | 全部机制冻结后 | 每seed5逻辑状态，按state_hash去重 | 完整Test六格 | 否 |

核心锚点的逐组/逐时间/接口结果是同刻机制比较，**不为每个hybrid重新选择其最好epoch**。只在预定时刻观察的hybrid不能冒充完成50epoch选模的算法候选。

### 4.1 tF 与五固定锚点

固定锚点 E=[5,10,20,35,50]。早期摘要=E5/E10均值；中期=E20；晚期=E35/E50均值；先在seed内聚合，再跨seed。tF是同seed F独立Validation选点时刻，用于把阶段机制连接到已接受的selected表现；不以“F−O最大”选时点。

E与tF重合时复用状态和预测，但输出两个逻辑用途。Online自己的tO保留用于正式selected Test，不将tO与tF差异混入同刻归因。

### 4.2 Test的唯一预定状态集

全部研究分组、动作菜单、解释模板和分析结果在 `MECHANISM_FREEZE.json` 中冻结后，才启动独立Test evaluator，读出：

1. O在自己的tO；
2. F在自己的tF；
3. O在F的tF；
4. S-only在F的tF；
5. N-only在F的tF。

1对2是正式独立选点性能比较；3/4/5/2组成选定时刻的机制四态。后者用于确认冻结的粗组结论及六格边界，不重新挑细组、时间窗口或机制假设。Test读取以后不新增归因分组、特征或超参。

完整时间与图接口机制以Validation证据为主，Test不承担所有细粒度机制的重复搜索。所有已确定细节都在Test之前完成。

## 5. 固定面板与数据角色

| 面板 | 来自哪里 | 规模 | 用途 | 禁止用途 |
|---|---|---|---|---|
| P_A | 源互斥训练行池A | 每格256正+256负，另每域512自然行 | 可在训练时取得的信号、历史指标 | 冒充独立验证人口 |
| P_B | 源互斥训练行池B | 同上 | 训练侧跨行审计、未来动作独立评分 | 用同一origin未来信息回填当前信号 |
| V_mech | 固定真实Validation行 | 每域min(4096,实际行数) | 因果机制与精确流式预测EMA | 当作完整Validation的性能数值 |
| V_A/V_B | V_mech冻结拆分 | 按源分割及有效mask | 定位/跨行确认、菜单选评交换 | 作为训练可见label喂控制器 |
| 完整V | 全部Validation | 见数据表 | 正式选点；准确六格干预AUC/LL | Test替代或反向调训练 |
| Test | 完整真实Test | 见数据表 | 冻结后的预定5状态 | 选组、选特征、选窗、重新设计 |

源面板若在当前机器不存在，可从相同数据行身份和源随机规则确定性重建并核对manifest；不得默默重抽。缺一类别的cell记missing与人数，不能填AUC=0.5代替。模型训练仍使用完整原始train；诊断面板划分不会改变训练目标。

## 6. 新任务配置样例（未来prepare生成，当前不是实现）

```yaml
study_id: EMA_CAUSAL_20261110
status: DESIGN_ONLY
source_commit: ffa3de067fcda5a1293af83a465d741e069493f1
main_at_design: a646c5a29b74d4f3d731a8ae2c13038b06f5250f
seeds: [2022, 2023, 2024, 2025, 2026]
conditions: [movielens/clean, kuairand/clean,
             movielens/gaussian_d0, kuairand/gaussian_d0]
epochs: 50
batch_size: 40960
source_training_semantics: unchanged
anchors_epochs: [5, 10, 20, 35, 50]
extra_anchor: fixed_ema_own_validation_selected
time_windows_half_lives: [0.5, 1.0, 2.0, 4.0]
prediction_ema:
  panels: [P_A, P_B, V_mech]
  frequency: every_real_optimizer_step_including_initialization
  primary_space: probability
  secondary_space: logit
fine_group_extra_state_cap_per_anchor: 12
group_time_group_cap: 2
functional_cuts: [star_plus_skip, fused_tower_input]
functional_contrast_cap: 2
fusion_component_extra_state_cap: 6
fusion_component_anchors: [35, fixed_ema_own_validation_selected]
h2_match_capture_epochs: [35, 50]
h2_match_pulse_horizon: rounded_true_half_life_steps
h2_match_extra_state_cap: 4
path_integral_nodes: [8, 16, 32]
diagnostic_pairs_per_cell: 32768
legacy_training_pairs_per_cell: 4096
mechanism_triage_resolution: 0.0002
test_states: [online_own_selected, F_own_selected,
              online_at_F_selected, S_only_at_F_selected, N_only_at_F_selected]
training_feedback: false
adaptive_controller_training: false
guardian_payload_cap_bytes: 4294967296
observation_output_cap_bytes: 2147483648
model_or_optimizer_disk_write: false
max_workers: 2
max_worker_restart_per_job: 1
max_same_seed_replay_after_guardian_loss: 1
```

这是本次预设的研究菜单与资源预算，不是宣称已存在runner或已产出结果。Codex须把配置解析、类型和依赖检查实现于新包，而不能把旧manifest改名后直接运行。

## 7. 统一评价规则的落地

- 五seed配对均值为正式统计单位；同seed多读出、重复运行、多个epoch均不扩充n。
- 只有未来正式算法晋级适用 `mean ΔAUC≥.003`；机制删除/接口变化不要求达到该门槛。
- 正式算法六格平均下降不得低于−.001；违反时标记实际代价，保留宏收益，不抹去数据。
- AUC与LL分别报告；LL使用未训练缩放的真实人口概率损失。
- 不加“必须5/5正”“CI下界>0”“某个seed某格不能下降”等新的硬门。
- 参数/接口干预效应可用小效应量形成机制判断；`.0002`只是区分值得细化的量级，不是新科学一票否决线。

## 8. 允许的停止与完成

运行中只因身份冲突、数值错误、状态丢失超出恢复预算或实测资源不足暂停。优先降低观察并行度、evaluation microbatch与派生文件冗余，不降低训练batch、seed数或完整人口AUC。

所有核心条件完成五seed后，若没有额外可操作空间，机制研究仍可成功结束并选择方向D。科学阴性不是工程失败；机制交互显著也不是必须继续拆分到唯一参数的理由。细化/可选来源诊断未触发时给出真实原因和已有证据即可。


