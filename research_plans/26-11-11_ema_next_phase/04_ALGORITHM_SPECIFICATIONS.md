# 04｜统一算法、数据与状态契约

返回 [执行入口](00_EXECUTION_RUNBOOK.md)。A、B、C的完整公式和伪代码分别位于 [12](12_ROUTE_A.md)、[13](13_ROUTE_B.md)、[14](14_ROUTE_C.md)，是本规范的组成部分。运行中的新动作只能来自这些定义。

## 1. 固定协议

| 项目 | MovieLens | KuaiRand |
|---|---:|---:|
| 原自然train行数 | 800167 | 2006132 |
| optimizer steps / epoch | 20 | 49 |
| 训练epoch / batch | 50 / 40960 | 50 / 40960 |
| Adam lr / beta1 / beta2 / eps | .01 / .9 / .999 / 1e−8 | 同左 |
| coupled WD | 1e−4 | 1e−5 |
| rho（每optimizer step） | 0.9828205985452511 | .995 |
| B窗口H | 20,40,80 | 69,138,277 |
| C主窗口H | 40 | 138 |
| Gaussian目标 / 替换比 / kappa | D0 / .5 / .05 | D0 / .5 / .1 |
| FP32参数数P | 773470 | 1614254 |
| 共享参数数S | 483152 | 661920 |

seed均为2022–2026。以上来自冻结 `RESOLVED_PROTOCOL`、真实registry与最新protocol；数据/特征/任务/域映射通过源hash确认。模型fixed1、训练gate输入detach、LayerNorm结构保持。无AMP/TF32、无梯度裁剪；Adam foreach/fused/amsgrad=False。新候选不得改变sampling、split、loss_scale或训练batch。

Clean源目标为原自然batch中两个任务BCE均值。Gaussian沿源目标D0按0.5替换对应损失，保留原batch分母、噪声stream和kappa，不新增增强方法。蒸馏主实验为Clean，公式见13。若09触发Gaussian教师边界，学生仍用相同Clean硬/软标签目标和未扰动输入，条件名为gaussian_d0_parent__clean_student；不引入新的增强蒸馏损失。

## 2. 符号与算法身份

- θ：online参数；F：本条真实轨迹上的固定FP32 EMA；S/N：真实共享/其余坐标；G：S中的EMBED，明确排除domain embedding。
- A的alpha表示应用更新的方向混合；C的alpha表示两态logit混合。代码字段分别 `direction_alpha` / `history_alpha`，禁止共用裸 `alpha` 配置键。
- B的a=ρ^h是旧历史质量；蒸馏lambda=`distill_weight`；源Gaussian lambda=`replacement_ratio`；WD=`weight_decay`。
- `A_MATCH`、`A_RAW`、`B_SPLIT`、`B_SHARED_ONLY`、`C_GLOBAL_PRED`、`C_GLOBAL_PARAM`、`C_CELL_PRED`、`C_CELL_PARAM`、`B_DISTILL`、`KD_LABEL`、`KD_FTEACHER`、`AB_READOUT`为唯一算法族名。O/F是内部控制，不计独立训练seed。
- `scope=S`的两个B状态共享N=F_N；`B_SHARED_ONLY`则是S=F_S、N=θ_N，这两个对象不同。

## 3. 真实图与评价图

训练所有候选使用原 `model_adapter.build_model` / `forward_logits` 和原loss路径；不启用归因专用 `probe_input_full_graph=True`。完整前向值可通过原raw-logit accessor读取。对于同一模型状态，probability与logit来自同次真实forward。

A只在源Adam step之后、EMA之前改写G坐标；B/C只读，在每个指定epoch端点用scratch模型评价；KD更新单独学生。观察器必须恢复train/eval标记、不改online参数/optimizer，不推进训练RNG。buffer逐项继承：源模型没有BatchNorm/Dropout，若发现运行时结构或buffer集合与锁不符，记录SOURCE_MISMATCH，不能擅自加入BN重估。

评价AUC沿用源FP32概率分数与tie规则；logit混合结果经FP32 sigmoid得到评价分数，概率混合也转FP32。LL沿用源clipping/实际评价人口定义。保存raw logit仅作机制诊断，不悄悄用更高精度logit AUC替换源指标。

## 4. 选点是完整预测对象

每个readout在E1–E50完整V曲线上按严格`>`更新best，tie保留最早epoch，不用epsilon改变选点。KD在其预定e0–e5曲线上独立选点。每个候选best保存RAM中可重现的整体预测对象，而不只是epoch：

| 对象 | 必须保留 |
|---|---|
| O/F/A_F/B_SHARED_ONLY/全局参数C | 完整参数、buffer、布局、指标和身份 |
| B_SPLIT | 两态S或all、共用N/buffer、H/h/a、mode、epoch |
| C预测/参数 | 相同两态、上一epoch已拟合且本epoch应用的αG/αc、s、mode |
| KD学生 | online/F自身选点参数，teacher/config/初始化/extra_steps身份 |

A完整算法复制公共前缀E1–20曲线与best，不能从E21重新计算一个人为更弱的对照。B、C若初始两态相同可按state hash去重物理读出，但逻辑配置仍保留曲线。全局先冻结五seed共用超参，再对每seed应用其own-selected端点。

## 5. 数据可见性

train保持原split；P_A/P_B是训练行的预定观测面板，仍参加原SGD。C只用P_A上一epoch拟合；P_B仅未来审计，不更新系数。V用于五seed配置选择和checkpoint选择；不得直接拟合C系数或生成soft目标。Test禁止参与代码分支、HPO、教师权重及训练；最终freeze后才加载。

C固定pair从源P_A/P_B的有效正负行产生，每cell至多4096个唯一pair，seed+panel+cell的固定hash排序/有界采样，生成一次缓存；不能在训练循环内重复全量pair抽样。细节见14。完整V/Test仍精确使用全人口。

## 6. 所有随机性和状态顺序

源模型初始化、loader generator（source seed+11000）、Gaussian噪声由源接口继承。新pair/诊断用独立named stream，不消耗源训练流。主seed字段不变；子stream的hash/key单独记录，不计新seed。

A步骤：zero_grad → 源forward/loss → backward → 捕获G步前参数/梯度 → 原Adam更新m/v/θ → A改写G → F.after_adam → capture历史/计数。C拟合发生在epoch末观察时，得到的系数只能下一epoch应用。KD教师先冻结并eval生成train目标，学生独立fresh Adam；teacher无梯度也不更新。

所有RAM fork包含θ、optimizer per-tensor step/m/v、F时钟、paramgroups、loader/batch cursor、Python/NumPy/CPU/CUDA RNG、best对象、B稀疏历史及C上一epoch系数、当前配置与源身份。不得引用父轨迹未来梯度或仅复制θ后宣称同状态fork。

## 7. 新组合的精确定义

父轨迹五seed完成后先冻结最佳B的scope/H/mode及G_B结果，再开始A短程。B已通过G_B时，A短程保留这个固定H的稀疏F历史，并附带该唯一B配置的全V曲线（最多2次额外前向/epoch）；无需采集A上的12配置网格。只有A也通过G_A才在其E20分支续训中保留AB。E1–20与父轨迹B相同，E21–26来自该实际A短支，E27–50来自续训；父E20 RAM包已带源三个H所需的历史快照，可裁剪到已冻结H。这样AB拥有完整50epoch曲线，不追加SGD或事后补造历史。

E20提交时，把O/F以及全部12个B配置在E1–20的prefix-best预测对象建立不可变RAM引用；B前缀只需两态/N/buffer/权重及身份，不保留供KD使用的同epoch父F副本。父轨迹随后更新current-best不能释放这些前缀引用。待五seed父轨迹选定B后，各seed只保留该唯一配置的前缀对象；E35 anchor不复制任何C或B的prefix-best目录，只保留其实际短支所需状态。A前缀O/F同理。

AB不搜索新H、scope、mode，不派生额外teacher家族；Gaussian若主候选为AB，只用已冻结A+B组合及同Gaussian父F/父B对照。所有参数快照只驻RAM，结束前完成对应Test。
