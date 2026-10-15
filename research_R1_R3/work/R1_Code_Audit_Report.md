# R1 代码与 baseline 只读审计

审计时间：2026-10-11。范围：仅 R1；未修改算法、未启动训练、未运行正式预测或指标评估。进行了从历史 feature schema 构造当前模型的 CPU 参数清点，不使用历史模型权重。机器可读证据见同目录 `R1_Code_Audit_Inventory.json`（代码、R0报告、16个run的config/manifest/parameter manifest、原始CSV的绝对路径/大小/SHA256、逐run缺失工件、当前依赖）。

## 1. 材料与版本边界

- 已读取任务书 `promt/stage_r1.md`，以及 `research_context/r0/` 六份正式报告。R0 第一候选为 AO-04；R0 原始置零阳性/阴性解释已被 R1 任务书明确覆盖。
- R0 写的 `research_R0/work/lit_opportunity_review.md`、`exp_evidence_review.md`、`skeptic_review.md` 在当前可访问项目树未找到；不得声称已读九份。六份正式报告实际搬移位置为 `research_context/r0/`，不是报告文本中的 `research_R0/`。
- 在项目可访问树检索本地 AGENTS.md 无命中；本次执行用户消息提供的 AGENTS.md。根目录 `git status --short` 返回 not a git repository，不能提供可靠 commit。采用 SHA256 文件清单。
- 可访问树 `*.pth/*.pt/*.ckpt` 检索未返回模型文件；检索有两处拒绝访问：`experiments/history_exp/data_archive/experiments/stage_S3_raw_results/target_only_shared/sample/kuairand/natural/m3oe/seed2022/joint` 与对应 `equal/.../joint`。该结论不代表全磁盘或拒绝访问位置无权重。
- 历史审查 `experiments/reports/history_report/stage_S1_review.md` 含多轮修复：最终 PASS 7（715–773行）；早期 FAIL 不是最终审查结论。当前历史报告与raw metrics均可访问，不能沿用R0“历史材料不可读”的旧限制；但模型工件缺失是本次核实的新限制。

## 2. 实际模型结构与计算图

构造器：`m3oe/MDMTRec/models/multi_domain/model.py:75–136`。正式历史实验使用 `experiments/stage_S1_models.py:134–216` 的 StageS1M3OEAdapter，继承同一构造器并修复原forward最终CPU零张量分配（原model.py:199–208）；不能直接把旧 examples trainer 当canonical runner。

设域数D=3、任务T=2、共享专家N=4，输入embedding维K：KuaiRand K=544（34个稀疏特征×16），MovieLens K=113（7个稀疏×16 + age dense1）。STAR结构为共享矩阵W与域slot矩阵V_d按元素相乘；样本域d使用 x(W⊙V_d)+b+b_d，随后512→256的MLP，加K→256 skip。MLP层为Linear+LayerNorm+ReLU，无BatchNorm、无Dropout。下游专家均256→64。

- N=4个共享专家、D=3个域专家、T=2个任务专家；专家个数是 N+D+T=9，不是D×T个完整backbone。
- D×T=6个共享专家gate，每个Linear(256,4)+softmax；D×T=6个预测塔，每个64→64+LayerNorm+ReLU→1+sigmoid。
- 所有域专家/任务专家对所有样本的同一个STAR表示计算输出。域专家不是仅在其源域样本上执行；所有专家均dense执行。

精确融合（原model.py:160–194；实际adapter stage_S1_models.py:155–181）：

```
aD = sigmoid(_weight_exp_d.deep_weights)
aT = sigmoid(_weight_exp_t.deep_weights)
bD = sigmoid(_weight_bal_d.deep_weights)
bT = sigmoid(_weight_bal_t.deep_weights)
R_d = bD*E_d + (1-bD)/(D-1)*sum_{j!=d}(E_j)
Q_t = bT*E_t + (1-bT)/(T-1)*sum_{u!=t}(E_u)
F_{t*D+d} = sum_n(g_{t*D+d,n}(detach(emb))*S_n) + aD*R_d + aT*Q_t
p_{t*D+d} = sigmoid(tower_{t*D+d}(F_{t*D+d}))
```

四个`deep_weights`各只有一个元素，是跨域跨任务跨样本的全局标量。own域系数=aD*bD，每个other域系数=aD*(1-bD)/2，总other质量=aD*(1-bD)。任务T=2时other任务唯一。所有raw初始化值为1，实际sigmoid值≈0.7310586，并非融合系数1（stage_S1_models.py:208–213；model.py:14–26,45–47）。索引task-major：0=t0d0、1=t0d1、2=t0d2、3=t1d0、4=t1d1、5=t1d2；输出仅保留样本实际域对应的每任务概率，形状[B,2]。

`Weights.forward()`每调用递减tau，且同一前向多次调用标量；正式softmax_type=3为sigmoid，不依赖tau，所以这些递减不改变上述有效权重。tau不是state_dict中的Parameter/buffer；今后若更改softmax_type会变成不同机制，而不是当前baseline。

gate可随样本、目标域、任务改变共享专家混合；`emb.detach()`阻断gate输入路径向STAR的梯度，gate参数本身仍可学习。gate仅看共享专家N个通道，不看D个域专家，故不能直接为other域j单独分配系数。任务专家、逐格非线性塔、域条件STAR及共享专家可以间接补偿域混合，不能推断它们完全无法表达任何亲和；但没有源域级、任务条件化的独立域融合权重。

AO-04保质量重分配应写为 aD[bD E_d+(1-bD)Σ_{j≠d}q_{d,t,j}E_j]，q在other域上和为1；原q=1/2，保持bD own-vs-other含义。置零不重归一改变总other质量，是不同实验。冻结后处理、训练期可学习权重也必须分开。D=3时other仅两个，连续比例比删除20%/50%机械分组更合理。

## 3. 精确参数量（当前构造器加载核实）

从S1 seed2022 manifest中的feature schema构造模型，逐命名tensor numel与历史parameter_manifest完全一致；无训练、无forward、无baseline权重加载。

| 模块 | KuaiRand | MovieLens |
|---|---:|---:|
| embedding | 44,752 | 196,992 |
| shared_weight | 278,528 | 57,856 |
| shared_bias | 512 | 512 |
| 3个slot_weight | 835,584 | 173,568 |
| 3个slot_bias | 1,536 | 1,536 |
| skip_conn | 140,032 | 29,696 |
| star_mlp | 131,840 | 131,840 |
| 4个shared expert | 66,304 | 66,304 |
| 3个domain expert | 49,728 | 49,728 |
| 2个task expert | 33,152 | 33,152 |
| 6个gate | 6,168 | 6,168 |
| 6个tower | 26,118 | 26,118 |
| 4个alpha/beta标量 | 4 | 4 |
| **总计** | **1,614,258** | **773,474** |

每个专家16,576，每gate1,028，每tower4,353。当前MovieLens examples default expert_num=4（run_movielens.py:217），正式S1/S6ML也是4。R0 Minimum Pilot Questions 的“MovieLens N=1无gate测量对象”不符合这些当前配置，须勘误；不能用论文或其它旧配置覆盖代码。

## 4. Sample/Equal与训练协议

当前canonical S1 Sample：`stage_S1_run.py:114–142,657–680`直接`binary_cross_entropy(pred,target,reduction=mean)`，即样本×任务平均。按全数据解释为 Σ_d p_d mean_t BCE_d,t；训练中不再乘一次p_d，无balanced sampler。

S2 Equal：`stage_S2_objectives.py:58–98,178–220`采用全训练集固定p_d=n_d/Σn、w_d=1/(3p_d)，普通shuffle，mean_i(w[d_i]*mean_t BCE)。总体每域质量1/3，每格1/6，固定权重不按batch归一；实际单batch值不必恰等于6格均值。S3/S6继承`stage_S3_gradient_access.py:103–145`的相同Sample/Equal拆分。

不要混用历史口径：S0 `stage_S0_run.py:602`记录Sample为w_d=D*p_d后sum域任务均值，是与当前标准化目标不同的历史尺度。原`ctr_trainer.py:116–124`是sum六格BCE（批内Equal尺度）；`ctr_trainer.py:50`用`~ bool`筛选main参数，Python得到-1/-2均为truthy，导致四scalar也进入main optimizer，随后还由第二optimizer更新。canonical S1 `stage_S1_run.py:618–633`明确分离main与arch，修复该legacy路径。因此“原始M3OE”需要同时写出采用哪个trainer。

canonical超参：`stage_S1_protocol.py:51–86`：D3/T2/N4，embedding16，fcn[512,256,256,64]，batch40960，50epoch，Adam main lr.01、arch lr.001、betas(.9,.999)、wd1e-5，num_workers0，shuffle_train=True，drop_last=False，无scheduler。main每batch更新；四scalar每epoch仅更新一次，来自train_loader一批而非validation（stage_S1_run.py:696–732）。新q参数属于main还是architecture及更新频率必须预先规定，否则只是不同优化预算对比。

S1验证逐格macro AUC（不加权）最大、严格`>`保留最早tie；50epoch全完成，shadow_patience10只记录假设停止点、不break（stage_S1_run.py:740–794）。正式真实patience10若在R2重新采用，需新B0重跑，不得声称与历史全50epoch模型选择协议完全一致。历史测试在selection冻结后才评selected/final/common9/19/29/39/49，不能以后择最有利测试checkpoint。

历史干预残留：S3 TARGET_ONLY_SHARED为梯度访问，S4–S6 HT/DH为共享梯度/Adam历史组合；S6 F是FULL current+FULL history，才是无目标域限制的Sample参考（stage_S6_run.py:83–105,119–145,189起）。不能以T/HT/DH替代B0；也不能把原生Adam历史状态等同“额外Momentum算法”。建议对R2用独立入口/独立目录，锁定forward与objective，避免继承受控干预臂。

## 5. 数据、seed与依赖

数据代码：`stage_S1_data.py:389–495` MovieLens，`498–619` KuaiRand；全局原始行顺序80/10/10，词表和age scaler仅全局base train拟合，OOV=0。KuaiRand map tab0/1/4→d0/1/2，click与stored_long_view；expected train n=172782/1743898/89452。MovieLens age分域{1,18}/{25}/{35,45,50,56}，标签rating>3/rating>4，训练L/M/S=150000/34500/3000，按RandomState(10*seed+physical_domain)抽样（stage_S1_data.py:91–123），验证/测试保持原始全局split。

本次实际读取CSV字节并SHA256核对（不重新解析/预测）：KuaiRand 416370411 bytes，`a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`；MovieLens 75514352 bytes，`f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`。KuaiRand与抽查历史manifest数据hash一致。没有重新生成split fingerprints，不能声称本次独立复算了split/encoder一致性。

seed=2022/2023/2024/2025；torch/cuda/numpy/random设seed，cudnn deterministic=True/benchmark=False（stage_S1_run.py:104–111），未要求torch.use_deterministic_algorithms，跨runtime不能承诺位级复现。

S1历史Python3.8.20/Torch2.0.0/CUDA11.7/cuDNN8500，EDDAenv；S6/S6ML历史与当前均Python3.13.5/Torch2.9.0+cu126/CUDA12.6/cuDNN91002。当前numpy2.1.3/pandas2.2.3/sklearn1.6.1，cuda_available=True。requirements.lock含大量本机file://构建路径，不是完全可移植依赖锁；其版本更接近当前/S6，不能拿来证明S1 EDDAenv已严格恢复。

## 6. 四seed checkpoint真实可用性

逐目录检查S1 KuaiRand natural m3oe joint、S1 MovieLens d0_large m3oe joint、S6/F KuaiRand joint、S6ML/F MovieLens joint，每族四seed，共16runs。目录均存在、manifest状态OK，但每run宣称的7个checkpoint全部缺失，实际目录无.pth；optimizer/architecture optimizer/rng等被宣称的二进制工件也缺失。结果CSV和predictions_test.npz存在不能逆推出模型weights。

| 族 | selected epoch，seed2022/23/24/25 | 可加载selected权重 |
|---|---|---|
| S1 KuaiRand natural | 37 / 42 / 46 / 26 | 0/4 |
| S1 MovieLens d0_large | 6 / 6 / 7 / 6 | 0/4 |
| S6 KuaiRand F | 30 / 47 / 41 / 24 | 0/4 |
| S6ML MovieLens F | 7 / 8 / 8 / 12 | 0/4 |

以上是历史manifest记录，不是本次重新选模。S1、S6ML已记录code hashes均匹配当前对应文件；S6仅stage_S6_report.py发生hash变化，训练model/data/runner哈希未检出变化。无git情况下不能进一步归因报告变更内容。完整缺失与drift逐run清单在Inventory。

**R1判定：历史checkpoint严格复用 NOT AVAILABLE；baseline数值重新复现 NOT PERFORMED。** 可用数据、代码、配置支持拟定匹配baseline重建计划；但正式推理比例扫描需要先找回完整且hash匹配的权重，或走设计→实现→审查→runner重新训练，并在验证/逐格结果复核后才宣称复现。任何重训练采用不同runtime、实际earlystop、q更新预算均必须说明差异，不能直接拼接历史test结果作为paired B0。

## 7. 给Charter的阻塞/注意项

1. 必须指定canonical B0为Sample、N4、四全局scalar、S1设备安全forward、明确当前或历史runtime与50epoch选择方式。
2. 冻结推理重分配所需historical weights当前不可用；将其列为条件臂，不能假定已经完成。
3. B1/B2的q须归一化保总other质量；parameter开销与optimizer归属/频率预注册。D3下target-domain q有3个独立自由度，task-target q有6个独立自由度；若用每pair两logits存储则分别6/12个参数（存在softmax平移冗余）。
4. gate只混共享专家的结构事实支持该实验有可区分的操纵对象，不能由此宣称存在AUC收益或排除非线性补偿。
5. 仅R1审计完成，未开始R2/R3；本报告不包含算法有效性结论。
