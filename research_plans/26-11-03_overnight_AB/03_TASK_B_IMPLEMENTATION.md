# 03 任务B：C_MIX＋P01的唯一组合定义及简单替代

**研究决定：保留一次有明确阶段互补性的组合实验，不预设正收益相加。** 主组合是`C_MIX_P01_SERIAL_V1`；替代是`C_MIX_GAUSSIAN_V1`。不添加第三个组合，不因已有类似思想放弃本次研究。

## 1. 代码事实：三个阶段不能混为一谈

| 组件 | 数据/梯度阶段 | 在线优化状态 | 参数与EMA |
|---|---|---|---|
| C_MIX | 真实BCE反向；条件插值只构造probe；投影当前梯度 | 修改进入Adam一阶矩的当前项；二阶矩保持未投影梯度 | 共享参数更新一次；最后固定非反馈EMA |
| P01 | 真实BCE＋静态PAIR排序辅助目标；抽样2048对 | ECCM维护条件边界历史h；真实项进入m；混合梯度进入v | 共享参数用m与h组合更新；非共享普通Adam用混合梯度；最后EMA |
| Gaussian A1 | 一个既定目标域的单IID高斯view，真实/增强损失插值 | 原Adam；单独使用时不修动量公式 | 普通更新后固定EMA |

P01不是“将Adam的旧m投影一下”，C_MIX也不是“把mix loss直接加进训练”。不能因名字带MIX就把C改成M的lambda_mix=.1目标。固定EMA作用于更新后的参数，不反馈在线θ，也不替代上述m/h。

### 共享参数精确集合S

两源optimizer的`shared_name`相同：

```python
name in ('shared_weight', 'shared_bias') or name.startswith(
    ('skip_conn.', 'star_mlp.', 'expert.'))
```

采用model.named_parameters()原顺序。embedding、domain/task-specific experts、gates、towers不因语义上“共享”就擅自加入S。记录每个name/shape/参数个数及分组hash；S与ordinary集合不交且并集覆盖全部参数。S由融合优化器更新，ordinary由唯一torch.optim.Adam更新；绝不将S再次交给该Adam。

## 2. 固定公共底座与随机性

两数据均50epoch、batch40960尾批保留，FP32、无AMP/TF32；Adam lr=.01、β1=.9、β2=.999、eps1e−8，coupled wd ML1e−4/KR1e−5。源M3 fixed1架构、初始化、torch shuffle generator(seed+11000)、全部数据完全不变。ML固定EMA rho=.9828205985452511；KR rho=.995，初始EMA=θ0，参数做EMA、buffers按源update_ema复制，非反馈。

每个训练batch的context固定为`study='EMA02V2', split_id='original', dataset,seed,epoch,global_step`。保持源随机流，不把新arm名字加入随机key；C的donor、PAIR的condition/pair_indices、Gaussian的A1|gaussian使用各自原stream。日志/诊断不得消耗全局训练随机流。主组合先完成原PAIR训练forward/backward，再按原C流程产生probe；保留源probe的train-mode/buffer行为，不擅自改为eval-mode。

源函数定位：

- C：`history.objectives.train_step/collect_history_probes`；`history.mixing.build_mix_batches`；`history.optimizers.project_history`。
- P：`pair.objectives.train_step`；`pair.boundary_sampler.sample/forward_logits`；`pair.optimizers.PairOptimizer.step`。
- G：`history.vendor.v3.vendor.ema02.objectives.losses`及原`source_losses`。

其中history=`experiments.mdmt_history_information_20261103`，pair=`experiments.conditional_information_momentum_20261102_v3`。新代码只import纯函数或在新包引用原生单组件训练接口，原包文件不改。

## 3. C_MIX原定义：必须保留的细节

设真实数据梯度为g_t，w为coupled weight_decay，u_t=g_t+wθ_t。源代码投影的是**含coupled L2的u_t**，不是单独g_t。

```
v_t = .999 v_(t-1) + .001 u_t^2
D_t = 1 / (sqrt(v_t / (1-.999^t)) + 1e-8)
c_t = Repair_D(u_t; probes, kappa=1)
m_t = .9 m_(t-1) + .1 c_t
θ_(t+1) = θ_t - lr * D_t * m_t/(1-.9^t)
```

probe域从`(global_step-1)%3`开始循环取本batch首个非空域；该域两个任务各自构造clean anchor与条件插值anchor，tau=.1，沿用原donor优先级/退化处理。两probe梯度只对S求导，detach；不累加到p.grad，不额外增加训练loss。

`Repair_D`直接调用原`project_history`：在sqrt(D)坐标中，仅当clean/mixed两梯度cosine≥.2且范数有效时纳入两条半空间约束；对sqrt(D)u做欧氏投影后映回，kappa=1。原活跃集枚举/KKT容差/伪逆精度和空约束直接返回规则不改。每步只轮转一个域、两个任务，最多4条约束；不扩成六格逐个probe。

## 4. P01原定义：静态PAIR＋ECCM

P01固定lambda_pair=.05、dynamic=False、每步抽2048个正负对。仅使用同域真实样本及真实任务logit；不从C虚拟probe建立PAIR，不用sigmoid概率做logit差。

源sample从实际存在的domain×task×joint-label正负边界中选择i。最多24个活跃条件，不假设每个域四种联合标签都存在；ML缺失的01状态无需补造。

```
w_i = (n_domain/(2*B)) * (n_positive_joint/n_positive_task)
      * (n_negative_joint/n_negative_task)
q_i = .25/K + .75*w_i/sum(w)         # dynamic=False时的原式化简
L_pair_i = mean(softplus(-(z_positive-z_negative)))
L_train = L_real + lambda_pair * (w_i/q_i) * L_pair_i
r_i = grad_S(L_pair_i)
```

令a_t=lambda_pair*(w_i/q_i)*r_i，混合含正则梯度b_t=grad_S(L_train)+wθ，真实分量u_t=b_t−a_t。缺少活跃pair时a_t=0、只用真实目标，不额外读取旧h项。

所有本batch活跃条件的n_j增加1；只有抽中的h_i更新`h_i <- (1-.01/q_i)h_i + (.01/q_i)r_i`。未抽中h不更新，但其活跃时钟仍增加。源约束w/q≤4、.01/q≤.96保留，不随意重新归一化w。

```
m_t = .9 m_(t-1) + .1 u_t
v_t = .999 v_(t-1) + .001 b_t^2
H_t = lambda_pair * sum_active_j[w_j * h_j/(1-.99^n_j)]
θ_(t+1) = θ_t - lr * D_t * (m_t/(1-.9^t) + H_t)
```

这里v用b，不是u，更不是h；非共享参数普通Adam接收L_train梯度。保留原`PairOptimizer.step`先更新h/active，再计算本步numerator的顺序。pair=None时H_t=0，旧历史表保留不清零。

## 5. 主组合唯一数学定义

**组合名称：C_MIX_P01_SERIAL_V1。** 只让C修复P01分离出的当前真实分量u；PAIR历史h本身不再做第二次C投影。

给定上节同一batch的b、a、u、r和C原probe：

```
(1) 用P01原采样与目标得到 L_train、r_i、全部p.grad。
(2) 得到 b_t=p.grad_S+wd*θ；u_t=b_t-lambda_pair*(w_i/q_i)*r_i。
(3) v_t=.999*v_old+.001*b_t^2；计算D_t。只更新一次v。
(4) c_t=project_history(u_t,D_t,probes,kappa=1,gamma=.2)。
(5) m_t=.9*m_old+.1*c_t。
(6) 按原P01时钟更新h_i，计算H_t。
(7) S参数一次更新：θ -= lr*D_t*(m_t/(1-.9^t)+H_t)。
(8) ordinary Adam对非S参数按原混合p.grad更新一次（自带一次coupled wd）。
(9) update_ema(ema,model,rho)一次；global_step只加一次。
```

步骤6与3–5没有互相依赖，但H必须使用本次更新后的h。为了与源P01顺序一致，可在步骤3前更新h；数值结果不应因此改变。实现冻结采用“先原P更新h/active，再v、C、m、numerator、参数”的顺序。

**禁止的替代实现：**先HistoryOptimizer.step再PairOptimizer.step；将完整PAIR梯度同时加进m和H；再次投影H或最终numerator；用c²更新v；将coupled wd加两次；改变h的beta或active时钟；用(1−lambda)缩小真实BCE；为匹配范数额外归一化；把EMA混入onlineθ。

开关关系：

| C开关 | P开关 | 必须调用/得到 |
|---:|---:|---|
| 0 | 0 | 原history A0f普通Adam＋固定EMA；复用A0端点 |
| 1 | 0 | 原C_MIX train_step与HistoryOptimizer；不经融合近似实现 |
| 0 | 1 | 原P01 train_step与PairOptimizer，lambda=.05 |
| 1 | 1 | 上述融合目标与新SerialCPOptimizer |

关闭项直接走原生分支，保留浮点操作顺序，避免为组合工程重训已有单组件。融合的二阶矩因P的加入而变化，C的投影度量也随之变化；这是明确定义的组合交互，不伪称两个组件数值上完全独立。

## 6. 新文件与接口

| 新文件 | 必须实现的接口/职责 |
|---|---|
| `component_configs.py` | `native_c_config(dataset)`, `native_p_config(dataset)`, `native_g_config(dataset)`；嵌套参数转原生dict，禁止kappa串用 |
| `component_steps.py` | `step(model,opt,ema,x,y,row_ids,cfg,context,diagnostic=False)->{loss,real_loss,pred,diagnostics,counts}`；单组件直派原函数，双组件按本文 |
| `optimizers.py` | `SerialCPOptimizer(model,cfg)`：zero_grad、step(pair,r,probes)、state_dict、load_state_dict；普通Adam仅非S |
| `worker_b.py` | 基于history.run.worker移植epoch循环，替换step/optimizer工厂，接入新guardian；不调用旧stage决策 |
| `diagnostics.py` | 低频、无额外训练forward的梯度/历史/更新统计，detach计算 |

SerialCPOptimizer状态至少包括：schema_version、config_hash、parameter_manifest、global_step、每共享参数(step,m,v)、ordinary Adam全部状态、h表、active/selected计数、pair_clock（兼容保存）、demand（静态仍保留）、RNG由worker保存。加载时检查名字/顺序/shape/模式/配置身份，不能只恢复θ而丢moment或EMA。

B worker训练循环沿用原source：50epoch、固定loader、逐epochonline/fixedEMA V曲线和各自selected，E10、E50；new optimizer的state_dict进入同一RAM原子提交。已完成的单组件通过01桥接复用，不调用旧worker中的D0/S1 gate。

## 7. 完整2×2与配对分析

主组合每个数据集均为：

| 格 | arm | 估计器 | 每数据seed数 | 默认动作 |
|---|---|---|---:|---|
| 00 | A0 | fixedEMA | 5 | 复用 |
| 10 | C_MIX | fixedEMA | 5 | 历史/补评复用或等待 |
| 01 | P01 | fixedEMA | 5 | 历史/补评复用或等待 |
| 11 | C_MIX_P01_SERIAL_V1 | fixedEMA | 5 | 新训 |

因此两数据共40个消融槽位，新训练默认10条。在线端点从同轨迹顺便记录，不计新增独立seed。A0 online同时为完整方案总收益参照及A的M3适配端点。

对每个seed计算selected test宏AUC：`CP-A0_ema`, `CP-C`, `CP-P`, `C-A0_ema`, `P-A0_ema`和`I=CP-C-P+A0_ema`；最后跨五seed求均值/样本SD/描述性t区间。六格和LL同样输出配对差；I不是新合格门，也不因I≤0否定组合总增益。由于各arm各自V选点，I是“所定义训练＋选点流程”的交互，不声称固定参数下局部因果机制证明。

单组件独立增益和组合增量各自有明确reference。完整方案`CP_fixedEMA-A0_online`另行裁定；不能将该总收益解释为C或P独立贡献。正式对比再分别给出CP对M3、HG、AI。

## 8. 唯一简单替代：C_MIX＋Gaussian

替代arm为`C_MIX_GAUSSIAN_V1`，四格A0、C_MIX、GAUSSIAN_A1、C_MIX_GAUSSIAN_V1，全为同rho固定EMA，各数据2022–2026。前三格优先复用；新增最多10条。

G参数：lambda_aug=.5；ML target_domain0、gaussian.kappa=.05；KR target_domain2、gaussian.kappa=.1；一个IID高斯view；原embedding相对尺度噪声、detach、row-id RNG均沿用A1。不改变已有目标域设置，不重新用域大小解释它。

训练标量直接调用源`ema02.objectives.losses`，传原A1配置和原context。其语义为：

`L_G = [sum(real_BCE) + .5*(sum(target_aug_BCE)-sum(target_real_BCE))]/(2B)`。

这不是将全部真实loss乘.5。然后一次backward；按原C构造clean/mixed probes。对共享参数：`u=grad(L_G)+wdθ`，v由未投影u²更新，C投影u进入m，一次更新；private普通Adam接收L_G梯度，最后一次EMA。没有PAIR梯度/条件历史。C probe仍用clean与条件插值，不为配合Gaussian再设计新的probe。

实现可复用原HistoryOptimizer(current_gradient_repair)，只需要新objective组合入口。两个开关关闭时分别直派原A0/C/A1，避免参数混淆。**配置必须区分`c_mix.kappa=1`和`gaussian.kappa=.05/.1`**；禁止一个平面kappa覆盖另一个。

启动替代的机械规则：主组合存在经工程测试确认的不可执行/内存阻塞且已完成一次针对性修复后仍阻塞时，将B活动矩阵切换为完整CG两数据五seed；CP已完成部分留档，不做性能裁决。A继续。普通代码笔误应修复，不立即换算法。主组合能正常运行时默认不追加CG；仅在启动前设置`extra_after_core=true`，且A和CP全部完成、资源仍可用时运行第二组合。中途test值不触发切换或追加。

## 9. 有针对性的自批判与决策

**不存在的前提不能作为动机。** 不假设C在训练中直接优化mix loss，不假设P只是修复Adam旧m，不假设固定EMA已改写在线moment，不假设所有联合标签状态存在；这些都与实际实现不符。组合动机只基于已存在的当前梯度约束与条件PAIR历史这两个计算位置，以及已观察到的候选信号。

**真正会影响选择的不确定性。** P引入的mixed v会改变C投影几何；H项可能重新引入C排除的方向；两组件可能改善同一错误而收益饱和；PAIR梯度尺度可能抵消C的修正。它们会影响组合是否值得保留，但不需要再增加一整套防御性矩阵：完整2×2、源语义验收和随训练记录的低频统计已能支持本轮决策。

低频统计取step1和每20step：raw/repaired梯度范数比、投影距离/约束数、real-moment项与H项范数、更新方向与已算probe的点积、pair w/q及active/selected计数、v与总更新范数、EMA-online距离、训练成本。只使用已有tensor，不增加one-step lookahead/新forward；不设统计阈值性能停机。若只是机制仍不完全解释但五seedAUC达标，保留算法并推进，不要求先证明所有机制。

**更简单替代的决策价值。** CG去掉24条件历史表和ECCM时钟，仅在源A1梯度上加C；因此工程与状态复杂度更低。它是预定义替代，不是断言比CP更好。A0+固定EMA、已有C/P/G也始终保留为内部效果/成本参照；不能因组合更复杂就自动优先它。

## 10. 有限工程验收

必须测试：四个开关分支与原生调用一致；单次共享/非共享更新且L2一次；v输入是未投影b；m不重复含PAIR；h仅选中更新、n对所有active更新；pair=None不除零、不读H；C空约束原样返回；kappa参数隔离；模型/optimizer/EMA/RNG/loader/selection RAM恢复等价；probe autograd不污染混合p.grad。采用小张量和短步轨迹，不增加科学seed预筛选。两个数据真实完整batch做数值/内存smoke，指标不用于性能否决。
