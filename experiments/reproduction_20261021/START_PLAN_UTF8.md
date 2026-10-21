## 实验验证原则

本项目的目标是**验证算法/机制本身的实验效果与研究结论**，而不是证明整个代码工程不存在任何潜在问题。

执行实验时遵循「**最小必要验证**」原则：

1. **优先完成核心实验，不要主动扩大测试范围。**
2. 发现问题时，首先判断：  
   **这个问题是否可能实际改变训练过程、模型输出、指标比较或最终研究结论？**
3. 只有存在明确的「问题 → 实验结果变化 → 结论变化」路径时，问题才属于**必须处理的问题**。
4. 对不会影响当前实验结论的问题（如代码风格、非关键边界情况、未使用模块、一般性工程风险、理论上可能但无证据的异常），**无需额外验证或修复**。
5. 不要为了提高“代码可靠性”而进行与当前研究问题无关的大量防御性测试、重构或额外 ablation。
6. 如果已有实验结果能够回答当前研究问题，**不要因为追求更高程度的确定性而无限追加实验**。
7. 对发现的问题分为：
   - **Critical**：可能改变实验结论 → 必须修复并重新实验。
   - **Relevant**：可能影响结果但尚未证明 → 只做最小验证。
   - **Irrelevant**：不影响当前结论 → 记录即可，不阻塞实验。
8. **不要预设研究假设或为了证明某个结论成立而设计额外测试。** 实验应由研究问题和已有证据驱动。
9. 在无法证明某个问题会影响结论时，默认采用**最小成本验证**，而不是扩大测试矩阵。
10. 最终报告应重点回答：**算法是否有效、现象是否稳定、结论是否被实验支持，以及是否存在足以推翻结论的关键问题。**
11.严禁在非代码训练阶段派发任何子agent，除了机械的运行实验进程外的所有任务应当只由主agent进行.当实验进入到运行训练阶段，派发子agent：luna_worker进行执行。
12.每个显卡设备允许并发多个实验进程，最多不超过5个。只要显卡设备的GPU的显存占用不超过90%就应当尽可能满足并发上限，不要求多张显卡同时满足占用低于90%才进入训练。
13.整个实验必须完成所有规定的流程，除非是满足设定的提前最终停止条件，否则严禁在实验未完全进行完毕前停止。

# M3oE 历史数据生成 / 梯度调幅方法：忠实复现、跨数据集验证与机制消融方案

> 用途：作为 Linux 服务器 Codex 的实验执行参考。本文是实验计划，**不是已执行的实验结果**。
>
> 证据来源：用户提供的 `layers(1).py`、`model(2).py`、`ctr_trainer(2).py`、`ctr_trainer加入梯度修正(2).py`；`Caelestis1903/My-Research-Project` 的 `PROJECT_ARCHITECTURE.md`、`m3oe/examples/run_movielens.py`、`m3oe/examples/run_kuaishou.py`、`experiments/reports/history_report/P3_kuairand.md`，以及 S3/S4/E1 机制分析。

## 0. 关键前提

- 当前实际观察：两套历史改动**仅在 MovieLens、seed=2022** 试过并有单次正向结果；**尚未在 KuaiRand 正式验证**。历史日志中一些运行缺少可确认的配置标记，不能以其最高 AUC 充当某一方法的可靠复现目标。
- 不得把三个上传文件组成的 DBO 风格版本误称为「完整数据生成版本」。`PROJECT_ARCHITECTURE.md` 中的历史数据生成方案还包括 **VUG、随机替换+软标签、分阶段训练和掩码**。必须找到当时真正运行过的完整 `mask` 源码快照与启动参数。
- 上传的梯度调幅 trainer 需要配套 `forward(x, aug_mask=..., noise_std=...)` 的模型文件；上传的 `model(2).py` 没有此接口。不要混装。
- 原研究有意去除 learned fusion 权重和 AutoML/DARTS，以避免干扰实验：这是**设计条件，不是错误**。若某个历史结果确实包含 AutoML，作为 `historical-original` 独立条件保留，不得悄悄混入当前 fixed-fusion 统一对照。
- GitHub main 的 `m3oe/` 当前只有 M3oE 主干模型和训练入口；历史 `mask` / 梯度调幅完整代码须从旧工作区、归档或历史提交中找回，不能根据文档凭空重写后宣称「原样复现」。

## 1. 实验目标（分别给出结论）

1. **效果存在性**：两个历史方法在 Linux 上重跑 MovieLens seed2022 是否得到与历史同量级结果？
2. **稳定性**：在 seeds 2022–2026 上收益的方向、均值和分散程度如何？
3. **迁移性**：KuaiRand-Pure 采用清楚定义的语义适配后，方法是否仍有效？
4. **独立贡献**：数据生成套件的哪些机制、梯度调幅的增量分别有效？
5. **解释/组合**：收益是否依赖 Equal objective、梯度贡献错配、Adam 的历史状态？两套方法叠加是否互补？

## 2. Stage 0：源码快照与 seed2022 忠实复现（先做）

### 2.1 源码交付

独立打包三个执行单元，绝不覆盖同一个正在使用的 `MDMTRec/`：

```
mdmt_repro/
  sources/
    base_fixed/          # 与旧结果相容的 fixed-fusion M3oE 基线
    data_gen_original/   # 当年真正跑出 MovieLens seed2022 成果的完整版本
    grad_original/       # 当年真正跑出 MovieLens seed2022 成果的完整版本
  data/
    movielens/
    kuairand/
  runs/
    <dataset>/<arm>/<seed>/
  receipts/
```

每个源码单元至少有：`MDMTRec/basic`、`MDMTRec/models`、`MDMTRec/trainers`、`MDMTRec/utils`、`examples/run_*.py`、实际启动命令、超参数、依赖版本及历史配置和日志（若有）。记录文件 SHA256、Git commit；源快照只读，修复在单独拷贝或 commit 中进行。

### 2.2 复现判定

- 用**历史完全相同的** MovieLens 文件与划分、任务定义、随机种子 2022、训练 epoch、batch size、优化器、loss、固定融合系数、训练阶段与增强强度。
- 严格记录当前原始 seed2022 历史 AUC/LogLoss 的来源与运行配置。如果历史输出未标注对应版本，不要求重现某个无法确认的最大 AUC。
- 在 Linux 上不强求跨 OS/GPU/PyTorch 的 bitwise 一致；验证该版本训练/评估流程确实一致，以及 AUC/LogLoss 是否接近历史量级、改善方向是否一致。不能为了追历史分数调整学习率、seed、checkpoint 或查看测试集选参。
- 最小必要诊断：1 个 batch 的前向、backward 和一次 step；1 个短运行；随后完整 50-epoch 上限/原早停策略。过多与结论无关的防御性测试不要做。
- 若发生致命错误，做**最小修复并保留 patch/diff**；若修复改变数学定义/训练轨迹，应命名为 `ported/fixed` 而非 `original`，不能合并统计。

### 2.3 已知必须核查的点

- 上传梯度训练器调用 `aug_mask`、`noise_std`，必须找到对应原模型。
- 上传梯度训练器的空域分支写 `logloss_list.append(...)`，但当前变量是 `task_losses`；发生空域时是运行错误，不是调幅效果。
- 数据生成旧代码如果硬编码 `movie_id` 下标，迁移 KuaiRand 时不能照抄：现有 `run_kuaishou.py` 特征里 **没有直接将 `video_id` 作为输入字段**，按列号复用可能掩到 `is_live_streamer` 等其他特征。
- 旧 `run_movielens.py` / `run_kuaishou.py` 使用相对路径，应从对应源码的 `examples/` 目录运行，并给出服务器上的绝对 `--dataset_path`；预先创建 `model_para/`、`output/`。
- 必须核对 `model.forward()` 的输出协议：某些方法训练态返回 `(pred, complexity, specificity)` 或 `(pred, specificity)`，评估态只返回 `pred`。
- 检查旧 trainer 到 epoch 上限时是否加载验证最佳权重；原样复现阶段保留原行为，后续统一协议才单独调整为相同的最佳 validation checkpoint 规则。

## 3. Stage 1：稳定性 × 数据集迁移（正式首轮）

### 3.1 主矩阵

| 方法臂 | MovieLens-1M | KuaiRand-Pure | 对照目的 |
| --- | --- | --- | --- |
| B：统一 fixed-fusion M3oE | seeds 2022–2026 | seeds 2022–2026 | 同源对照基线 |
| D：完整历史数据生成套件 | seeds 2022–2026 | seeds 2022–2026 | 套件整体是否有效 |
| G：完整历史梯度调幅套件 | seeds 2022–2026 | seeds 2022–2026 | 套件整体是否有效 |

名义 **2 数据集 × 3 臂 × 5 seeds = 30** 次正式训练；`2022` 如不能证明历史严格一致，就在统一协议下重跑。若 D/G 的固定参数、模型输入或优化设置不同，须增设与其**源版本一一对应的基线**，而不是假定 B 能同时充当可归因基线。

### 3.2 统一数据与指标

- MovieLens：原 3 个年龄域；两任务 click/like；保持历史 split、编码和特征顺序以复现。
- KuaiRand：复用已有经过审计的 `tab={0,1,4}`、click/long_view、按时间排序后的 8:1:1 split。所有方法臂使用**同一数据与特征映射**，保存 train/val/test 行身份和每域任务标签统计。
- KuaiRand 的生成策略只允许**语义等价适配**：先明确扰动/生成的是哪个 item 或 embedding 维度，确保其语义与 MovieLens 的 `movie_id` 处理目标一致。如果数据管线需增加 `video_id`，三个方法臂必须同时增加并另设协议标签；不得一边不改基线，一边给生成版新增输入字段。
- 主指标：每个 `(domain, task)` 的 test AUC/LogLoss，macro AUC/LogLoss，最差 cell 和其性能变化；单独列 pooled AUC，不能与 macro 混称。
- 单位：**seed**。报告均值、标准差、配对增益与 seed 级置信区间；不要把 task/epoch 当独立重复数。
- 选参/选 checkpoint **只看验证集**。测试集用于冻结配置后的最终评估；不要用测试集寻找 `gen_mode`、`tau` 或掩码比例。

### 3.3 第一轮判断

- D?B：**历史生成套件的整体净收益**，不是“VUG 生成单因素收益”。
- G?B：**历史梯度套件的整体净收益**，不是“gradient scaling 单因素收益”。
- 只要存在稳定的正向组、反向组或强烈数据集依赖，都保留在结论中，不强行只报告平均值。

## 4. Stage 2：消融并确定真正有用的机制（不进行无意义全组合穷举）

### 4.1 G：最先做梯度调幅开关对照

在完全相同的 V2 model、样本复制、特异性权重、优化器、数据 batch 和 seed 下：

- **G0**：`use_grad_norm_balance=False`，其他机制照旧。
- **G1**：`use_grad_norm_balance=True`，其余相同（相当于 Stage1 的 G）。

额外最多 **2 数据集 × 5 seeds = 10** 组 G0；`G1?G0` 才可以归因为实际 gradient balancing 增量。

随后按必要性拆：G 的 specificity-only 权重、稀缺域复制/噪声增强、grad_tau=0/0.25/0.5/0.75/1、norm 是全参数还是仅共享参数。`tau=0` 与关闭调幅应基本一致，若不一致，重点检查 `param.grad=None`、weight decay、Adam 状态被改变，不要当成有效收益。

### 4.2 D：完整数据生成套件拆分

先做“逐个拿掉”：

1. 原版完整 D；
2. D 去掉 DBO 权重，保留真实生成、掩码、训练阶段；
3. D 去掉 VUG Phase3，保留 Phase2 + DBO / mask；
4. D 去掉 Phase2 随机特征/软标签，保留后期 VUG + DBO；
5. D 去掉自适应掩码，其他保持；
6. D 将生成样本改成原样复制/不做 embedding 变换（负对照）。

如某一机制是另一个机制的依赖，必须做**可运行、语义明确**的条件消融，不能让缺失组件成为 no-op，却报告为有效因子试验。

根据第一轮定位到的有效组件，再验证最有信息价值的 2~3 组交互（如 `VUG×DBO`、`VUG×mask`、`G×generated samples`），最后在有效机制上调整超参数。**禁止从混合历史日志的最高 test AUC 反推有效配置**。

### 4.3 哪些结果最重要

- **AUC 稳定提升但 LogLoss 不改善**：检查优化目标、预测校准和 checkpoint，而不是直接宣布算法有效。
- **收益只有 MovieLens**：如实报告迁移边界，分析数据/标签/增强语义差异。
- **只有 D/G 套件有效、单因子无效**：研究交互，而不是删除“无单独效果”的组件后失去整体收益。
- **G1?G0 不显著，G?B 显著**：说明正结果来自 G 的其他模块，不能归因于梯度调幅。

## 5. Stage 3：Equal / Sample 与两者组合

在 Stage2 筛出的可靠组件上，对比：

| Objective | B 基线 | D 最有效成分 | G 最有效成分 | D+G |
| --- | --- | --- | --- | --- |
| Equal | ? | ? | ? | ? |
| Sample | ? | ? | ? | ? |

- 优先复用现有 S2/S3 实验中已核对的 **Equal / Sample coefficient 定义**，不要在训练中悄悄改变损失尺度、step 数或优化器。
- **生成样本不应悄悄改变“原始自然训练分布的 domain share”**。Sample 的 `p_d` 应预先按真实训练数据定义，生成项另记并独立控制，或显式声明另一个目标函数。
- 记录原始每组梯度 norm、cosine、按目标系数加权后的贡献、**真实 Adam 更新方向**、一二阶矩、生成样本梯度与真实样本梯度夹角。
- 如果联合收益存在，用相同 baseline 计算交互项：`I = (D+G?B) ? (D?B) ? (G?B)`。不能把单 Seed 交互当普适结论。

## 6. 执行规则与输出

每个正式 run 单独文件夹：

```
runs/<dataset>/<arm>/seed2022/
  config.json          # effective 参数、loss 公式和源 SHA
  data_manifest.json   # 数据及 split/feature 身份
  epoch_metrics.csv    # validation 六格 + macro + epoch
  test_metrics.csv     # 冻结 checkpoint 后一次测试
  model_best.pth
  optimizer_state.pth  # 若做后续历史/恢复分析
  stdout.log
```

阶段分析报告必须给出：逐 seed 原始数值、配对差、所有域任务 AUC/LogLoss、checkpoint 身份、缺陷与修订历史、支持的机制解释、替代解释和没有解决的问题（**数量不设上限**）。

**工程测试范围**：只做直接影响模型正确性和结论可信度的最小测试（shape/device/单 batch backward/空域分支与 eval 协议）。不要进行无关防御性单测或大规模重构。先让真实训练跑起来。

## 7. 服务器上如何运行

1. 将完整历史源码分别上传到 `~/mdmt_repro/sources/`，不要覆盖现有主仓库；从上传文件和原始存档计算 SHA256。
2. `nvidia-smi`、`python -c 'import torch; print(torch.__version__,torch.cuda.is_available())'` 核对实际 GPU/PyTorch。
3. 在每个版本独立 Python 环境中安装原依赖；若老 PyTorch 无法在当前服务器安装，记录新环境及版本差异，保持方法公式不变。
4. 从该版本的 `examples/` 目录运行，确保 `output/`、`model_para/` 存在；显式传入服务器绝对数据路径、seed、模型参数。
5. 先做 MovieLens seed2022 忠实复现；然后 2023–2026；最后用冻结版本迁移 KuaiRand。
6. 如 GPU OOM，不要默默减小 batch size（会改变每个 Adam step 的域组成、Equal 子集均值、训练更新次数）；先排查是否能在原 batch size 下运行。必须改时另立配置并统一所有对照。

### 提交给 Codex 的紧凑指令

> 请先根据本计划执行 Stage 0+1，不直接开展组合/超参搜索。对两份真实历史算法源码分别建立不可覆盖的原版快照，核对模型、trainer、入口、参数和历史 seed2022 日志是否配套；只做必要的兼容修复并保留 diff。首先在服务器用完整 MovieLens 数据重跑 seed2022，随后 seeds 2023–2026；再在 KuaiRand-Pure 同源 split 和任务设置下执行五 seeds。所有对照必须在对应方法版本相同的 fixed-fusion 基线上进行，不恢复被有意移除的 AutoML/DARTS。留存逐域任务 AUC/LogLoss、验证 checkpoint、训练配置和原始日志。对于 KuaiRand 的 movie_id 相关生成/掩码，明确合法的语义映射，不要按原始列下标套用。仅处理阻断训练或影响结论的错误，避免无关防御性测试。完成后整理配对结果、稳定性、迁移性和所有未解决问题，然后根据证据选择 Stage 2 的消融项目。
