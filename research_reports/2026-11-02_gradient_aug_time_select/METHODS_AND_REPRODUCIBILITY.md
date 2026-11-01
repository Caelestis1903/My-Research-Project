# 方法、执行身份与复现

本轮 GATS02 按 `start.md` 指向的 [唯一执行入口](../../research_plans/26-11-2/06_Codex完整执行提示词.md) 及 [设计裁决](DESIGN_SPECIFICATION.md) 实施。阶段顺序为 DESIGN → IMPLEMENT → 独立 REVIEW PASS → CUDA Stage0 PASS → RUN → 协调者 ANALYZE。实现者与审查者不同，runner 只执行和记录数值。设计、实现、审查和运行报告均保留，不将后续分析回写成预注册内容。

## 执行身份

- 工作基线 main：`0bf3e970ca47da44b129e67bb4c84a653ba6a20b`；历史证据：`26-11-01_v3@6f5742c2085c1e49d1767b0a7079bb9f28160eda`。
- 首次attempt的训练源码 commit：`ac2ec5cc70be7dcb880a7dde1015e4634726fd96`；source manifest SHA-256：`ff5136395313436fa8b96f7bf585245eb0c6e7f2607c1caad40f4d4013bea130`。53 个实际执行源码文件的 Git blob 与该 commit 逐一相同，见 [执行核验](EXECUTION_SOURCE_VERIFICATION.json)。该attempt在开发训练后、统一测试前发生接口错误，不能将其称为完成全部实验；恢复执行身份另行记录。最终发布commit追加报告与分析文件，实际训练身份由各attempt收据分别确定。
- Python 3.13.5、PyTorch 2.9.0+cu126，显式解释器 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`；环境快照和 GPU 状态见 [预检](ENVIRONMENT_PREFLIGHT.json) 与原始运行 environment.json。
- [来源收据](SOURCE_FETCH_RECEIPT.json)、[依赖核验](DEPENDENCY_SOURCE_VERIFICATION.json)、[数据核验](DATA_VERIFICATION.json)、[历史基线来源](historical_aigm_baseline/SOURCE_RECEIPT.json) 分别记录读取内容与身份。旧 MASK/Polyak 配套原始代码的可获得性边界见 [证据回顾](EVIDENCE_REVIEW.md)，本轮不声称逐行重现它们。

## 数据、模型与选择

两个真实 CSV 分别为 `/home_new/xzf/datasets/ml-1m/ml-1m.csv` 和 `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。自然原行序 80/10/10；KuaiRand 全局划分后筛选 tab 0/1/4。仅 train 拟合 encoder/scaler，OOV=0；未重采样。原始 SHA-256、六个数组、行号、encoder/scaler 与原合同一致。

| 数据 | train 总数 | d0 | d1 | d2 | 增强目标域 | 每 epoch 更新 |
|---|---:|---:|---:|---:|---|---:|
| MovieLens | 800167 | 155949 | 318599 | 325619 | d0 | 20 |
| KuaiRand | 2006132 | 172782 | 1743898 | 89452 | d2 | 49 |

模型为 fixed1 MDMTRec/M3oE：3 域×2 任务、4 shared experts、embedding16、fcn_dims=[512,256,256,64]，融合系数固定为1。ML 参数773470，KR参数1614254。任务分别为 rating>3/rating>4 与 is_click/long_view。标准 Adam，lr=.01、betas=(.9,.999)、eps=1e-8、coupled weight decay ML1e-4/KR1e-5；batch40960、50 epoch、FP32，AMP/TF32关闭。增强目标损失使用原 batch 大小归一化。

开发种子2022/2024；条件确认种子2023/2025/2026。每种估计器独立按 validation 六格 macro AUC 选择最早严格最优 epoch，保留 selected、E10、E50。SWA从E31开始，E10为N/A；其selected端点可能只有少于20个epoch末参数参与平均。统一测试前冻结全部候选、配置、端点和模型/optimizer hash。测试值不参与本轮门槛、训练或选点。相同测试划分已参与历史研究，因此这里的统一测试不构成首次未见测试集。

## 干预与控制

| 臂 | 干预 | 必要比较 |
|---|---|---|
| A0 | 真实 BCE；online、fixed EMA、delayed5 EMA、E31起SWA | 同轨迹估计器比较 |
| A1 | 目标域 sparse embedding Gaussian，κ ML.05/KR.1，λ=.5 | 对A0；是B的固定基线 |
| B1 | clean loss平均排名决定每行每字段覆盖率p=.05+.9(1-r) | R1：随机位置，逐步逐字段匹配数量与实际扰动能量 |
| B2 | 每任务、E/P全参数组分别移除增强差分对同任务clean梯度的反向分量 | R2：训练范数统计冻结组固定缩放q |
| B3 | E1–5关闭，E6–50 λ=25/45 | R3：固定seed1729随机关闭5epoch，同活跃次数与λ直方图 |
| C2 | A1轨迹上梯度范数100步CV决定rho，每100步更新 | 同轨迹fixed EMA与全rho序列随机排列replay |

固定rho：ML exp(log(.5)/40)=.9828205985452511，KR .995。A1/B/控制维护fixed、adaptive和独立replay三个shadow；A0维护fixed、delayed、SWA三个shadow。shadow不反馈online。Replay从相同初始化重新执行整个online更新序列，逐step模型hash、最终Adam、数据行序和view hash必须相同；它是额外物理计算，不能增加独立种子数。

B2逐任务的target clean及aug-clean损失均除以原B，投影后两个任务只平均一次。FP64 reduction得到组系数，更新为FP32，U组差分须精确零或None；weight decay不进入参考梯度。R2的q为两开发种子全部训练step/task的保留范数之和除以原差分范数之和，完全不看validation/test。它没有匹配逐步变化或两任务合成后的抵消，B2与R2差异仍可能包含这些因素及Adam预条件效应。

R1匹配每字段总数量和能量，不匹配每行覆盖数/能量异质性。B3与A1的“等剂量”专指Σepochλ=25，不等于Σλ²或实际扰动能量相等；R3匹配活跃次数与λ直方图，但只使用一个预先冻结的随机日程。rho随机排列同时改变时序关联、自相关与warm-up位置。

## 门槛、预算与统计

主门为开发两seed Δmacro AUC均正且均值≥.001；探索门≥.0003且均正。LogLoss均值恶化≥.002或worst/最大域任务保护下降达到阈值则阻止晋级。匹配控制须两seed均正、平均ΔAUC≥.0005且无保护损害；近似平局优先更简单控制。原始比较、区间与状态以原始decision JSON为准，不用测试结果改写裁决。

20基础唯一轨迹；最多8附加，控制优先顺序R2→R1→R3；最多12新增确认（包括不能复用时的新增基线），总上限40。完整匹配控制后才可晋级；最多1个B进入C1，最多2个B/C进入确认。预算为上限，无需填满。τ2公式未获明确合同，因此只允许触发后补充设计/实现/复审；未触发不猜测公式。η.5/delay10需要注册触发和各自控制预算。

seed是统计重复单位。开发n=2、确认n=3、合并n=5分开；表格报告配对原值、均值、样本SD、配对t95%区间和正向seed数。任务、epoch、训练步、数据行和replay均不能增加n；这些区间未作多重比较修正，主要用于呈现种子不确定性。macro为六格等权；traffic为按样本数加权的格指标；pooled先合并预测再计算每任务AUC后平均，二者不可混称。

## 成本与已知记录缺口

`costs.csv`为原始每唯一轨迹的物理成本（包含重算pass），`physical_pass_costs.csv`另分首遍和重算。forward、梯度遍历、optimizer更新及shadow更新单独计数；五次梯度遍历不称为五倍FLOPs。每kind验证成本和独立选点均计入。并发worker秒可相加，不能当作队列墙钟。

独立复审发现 `mask_fields`/`perturbation_energy` 只在B1/R1 mask通道累积。A1/B2/B3/R2/R3的原始零为**未测量**，最终展示表改为N/A并保留原始列；A0的energy为结构性零，mask_fields为N/A。此缺口不影响指标、R1数量/能量匹配、R2的q或rho重算一致性；不补造实际能量。详见 [诊断记账复审](DIAGNOSTIC_ACCOUNTING_REVIEW.md)。B1/R1比较只取primary，能量为λ缩放前；actual perturbation RMS=√(energy/(target row exposures×fields×16))。日志中的embedding_rms是clean embedding尺度。

正式训练前的首次wrapper启动因输出目录预先存在stdout/stderr而被空目录检查拒绝，尚未执行训练；修正外层日志位置后以同一配置启动。首次错误文件随后被重定向覆盖，其描述只能作为消息转录，不能称原始traceback保留。GPU外部任务未终止或操作。

首次attempt完成20条基础科学轨迹、36个物理pass后，在历史数值基线核验中访问不存在的顶层encoder_hash而退出，尚未统一test。它的原始fatal traceback保留，RAM状态随进程退出丢失；不能从数值日志恢复模型。此前审查未用真实落盘schema覆盖此接口，恢复须经设计附录、coder修复、独立复审、真实48端点接口预检及新Stage0。重新训练原配置用于重建RAM，其物理成本全部累计，不增加独立seed或科学重复数；绝对10/12小时deadline沿用首次正式训练开始时间。训练失败、OOM、协调失败和重复计算分别记账，见 [失败attempt运行报告](FAILED_ATTEMPT_RUNNER_REPORT.md)。

所有model/Adam/shadow/RNG/selected状态只在RAM及RAM IPC中保存。磁盘仅数值日志、概率/行号数组、配置、hash和源码。队列冻结统一测试后释放RAM；复现需从头运行，不能从磁盘权重续训。
