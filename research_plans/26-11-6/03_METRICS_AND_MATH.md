# 03｜EMA 收益指标与数学规范

本文件定义本轮研究的数学对象、观测量、脉冲干预和指标正负方向。实现接口、随机流和恢复约束见 [05_IMPLEMENTATION_SPEC.md](05_IMPLEMENTATION_SPEC.md)；统计比较与决策见 [06_EVALUATION_AND_DECISIONS.md](06_EVALUATION_AND_DECISIONS.md)。本轮全部读出与观测不反馈到在线训练。

## 1. 符号、张量与效用方向

| 符号 | 定义与形状 |
|---|---|
| \(a\) | 训练轨迹类型：clean 或 gaussian_d0 |
| \(s\) | 独立训练 seed，固定为 2022、2023、2024、2025、2026 |
| \(k\) | 完成一次真实 Adam 更新后的全局步数，同时也是 EMA 更新计数 |
| \(e\) | 完成的 epoch，范围 1…50 |
| \(c=(d,t)\) | 六个域–任务对，\(d\in\{0,1,2\},t\in\{0,1\}\) |
| \(\theta_k\) | 在线模型的可训练参数；概念上展平为 \(\mathbb R^P\) |
| \(F_k\) | 历史协议固定 EMA，全参数 \(\mathbb R^P\) |
| \(F_k^{(H)}\) | 从同一 \(\theta_0\) 初始化、持续维护的 half-life 为 H epoch 的非反馈 EMA |
| \(P_b\) | 参数组 b 的坐标投影；实现为索引，不构造 \(P\times P\) 矩阵 |
| \(x,y,z,p\) | 输入 \(x:[n_{\rm fields},N]\)，标签 \(y:[N,2]\)，raw logits \(z:[N,2]\)，原模型 FP32 sigmoid 概率 \(p:[N,2]\) |
| \(\mathcal P_c\) | cell c 的固定正负样本配对索引，形状 [4096,2] |
| \(U_c^{\rm AUC}\) | 原仓库概率 AUC，越大越好 |
| \(U_c^{\rm LL}\) | \(-\mathrm{LogLoss}_c\)，仅为统一收益符号；结果表仍报告正的 LogLoss |

参数组顺序固定为：

\[
(S,D0,D1,D2,T0,T1,DT00,DT01,DT10,DT11,DT20,DT21).
\]

必须满足 \(P_bP_{b'}=0\;(b\ne b')\)、\(\sum_bP_b=I\)，每个可训练坐标恰好归入一组。分组来自实际计算路径，包含 domain embedding 按行拆分；不能按名字猜测，也不能把监督路径归属等同于“参数的实际 Adam 位移只由单一 cell 决定”。具体映射和源代码证据见 01、05。

本轮 fixed1 骨干的完整预测函数支持为 cell \((d,t)\) 可达 \(S,Dd,Tt,DTdt\) 四组，六格合计 24 个可达 c,b 组合；其余 48 个组合保留在完整效应矩阵及工程检查中。M0–M3 主预测分析使用冻结的这 24 个可达组合，避免结构性零膨胀评价。支持 mask 来自计算图和固定 fusion 配置，不按实测效应是否为零筛选；完整导数恢复 gate 输入后，组级支持仍相同。

除明确涉及不同训练设置外，下面的比较均固定 dataset、seed、在线轨迹、参数初始化、评价人口与评价时间。宏指标是六格等权均值，不以 traffic 权重替代。

## 2. 固定 EMA：精确性质与有效时间尺度

### 2.1 更新、展开与初始化

当前实现每次真实 Adam 更新后执行

\[
F_k=\rho_FF_{k-1}+(1-\rho_F)\theta_k,\qquad F_0=\theta_0.
\]

因此

\[
F_k=\rho_F^k\theta_0+
(1-\rho_F)\sum_{j=1}^k\rho_F^{k-j}\theta_j.
\tag{1}
\]

全部权重之和为 1。现有实现从模型参数初始化，没有 warmup，也没有零初始化；不得额外除以 \(1-\rho_F^k\)。非参数 buffer 按源规则从 online 复制。此模型使用 LayerNorm，没有 BatchNorm，因此本轮不额外加入 BN 重估实验。

式 (1) 描述“怎样平均”，不直接回答“为何 AUC 提高”。

### 2.2 半衰期、平均年龄与固定候选

固定系数的半衰期和稳态平均历史年龄分别为

\[
h_{\rm step}=\frac{\log(1/2)}{\log\rho},
\qquad
\mu_{\rm age,step}=\frac{\rho}{1-\rho}.
\tag{2}
\]

从 \(\theta_0\) 初始化时，有限 k 的平均年龄为

\[
\mu_{\rm age,step}(k)=
\frac{\rho(1-\rho^k)}{1-\rho}.
\tag{3}
\]

平均年龄与半衰期不是同一个量。每 epoch 有 \(K\) 次真实更新时，

\[
H=\frac{h_{\rm step}}K,\qquad
\rho(H)=2^{-1/(HK)}.
\tag{4}
\]

本轮 batch=40960、保留尾 batch，MovieLens 的 K=20，KuaiRand 的 K=49。以下数值基于 [冻结源协议](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/aug_history_20261105/protocol.py) 和 [历史 manifest](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_plans/26-11-5/experiment_manifest.json)。

| 读出 | MovieLens：每步 \(\rho\) | KuaiRand：每步 \(\rho\) | 用途 |
|---|---:|---:|---|
| H=1 epoch | 0.9659363289248456 | 0.9859537217481000 | 预注册较快历史 |
| H=2 epoch | 0.9828205985452511 | 0.9929520238904295 | 预注册中间历史 |
| H=4 epoch | 0.9913730874626621 | 0.9964697807211363 | 预注册较慢历史 |
| 历史 F | 0.9828205985452511 | 0.9950000000000000 | 继承已有可靠对照 |
| F 的实际半衰期，epoch | 2.0000000000 | 2.8220933262 | 解释原始 \(\rho\) |
| F 的稳态平均年龄，epoch | 2.8604622842 | 4.0612244898 | 解释滞后，不替代 half-life |

实现用式 (4) 计算完整精度，不从表格的舍入值反推。MovieLens 的 H=2 与 F 同一初始化、同一更新序列，可别名共享；KuaiRand 保留 F 和三个 H，共四个持续影子。不得按域样本量推导不同 half-life；本轮考察的是实际收益偏好。

持久影子从 \(\theta_0\) 开始，整个训练过程都以其固定 \(\rho\) 更新。后文脉冲影子每次从当前 F 重新初始化，即使 \(\rho\) 恰好相同，也不能与持续影子合并。

### 2.3 初值与时间比较

早期曲线同时保存初值权重 \(\rho^k\)，允许长 half-life 存在更强初值滞后。不能把早期数据删除后才声称所有时间尺度一致。

两数据集比较 H 而不是直接比较 \(\rho\)。H 按实际更新次数/epoch 定义；本轮不改为“每行样本等权时间”递推，因为尾 batch 与普通 batch 的 EMA 更新权重在历史实现中相同。

## 3. 创新量与实际 Adam 位移

定义

\[
\delta_k=\theta_k-\theta_{k-1},\quad
r_k=\theta_k-F_{k-1},\quad
\epsilon_k=\theta_k-F_k.
\tag{5}
\]

直接代入得到

\[
r_k=\delta_k+\epsilon_{k-1}
=\delta_k+\rho_Fr_{k-1},
\tag{6}
\]

\[
\epsilon_k=\rho_Fr_k,\qquad
F_k-F_{k-1}=(1-\rho_F)r_k.
\tag{7}
\]

所以 \(r_k\) 已经包含累计历史位移，既不是当前梯度，也不是单步 Adam 更新。以 \(-\eta g_k\) 替代 \(\delta_k\) 会忽略 Adam 状态及源实现的 coupled weight decay。观测时若已经更新 F，只能把 \(\theta_k-F_k\) 记为 \(\epsilon_k\)，不能误命名为创新 \(r_k\)。

起始 \(\epsilon_0=0\)，故 \(r_1=\delta_1\)；若统一使用式 (6) 的递推记号，约定 \(r_0=0\)。

范数与夹角描述分歧幅度和时间结构，不自动具有收益方向。跨组比较使用 RMS 等明确归一化，并保留未归一化量；最终以预测读出收益判断其信息价值。

实现还在每步 F 更新前即时 reduce 各组 \(\|P_br_k\|_2^2\)、\(\|P_br_k\|_2/\sqrt{|b|}\) 及组能量占比 \(\|P_br_k\|_2^2/\|r_k\|_2^2\)，聚合为每 epoch 的均值、样本 SD 和最大值。总能量为零时占比记 missing。能量占比受组维数影响，因此必须与 RMS 同报；不把能量最大的组预设为收益最大的组。只保留标量汇总，不持久保存每步 P 维创新或梯度。

## 4. 方差、漂移和收益：本轮使用的理论边界

### 4.1 低通是性质，不是性能结论

忽略初始化瞬态，固定参数 EMA 的传递函数为

\[
H(z)=\frac{1-\rho}{1-\rho z^{-1}},
\quad
|H(e^{i\omega})|^2=
\frac{(1-\rho)^2}{1+\rho^2-2\rho\cos\omega}.
\tag{8}
\]

\[
|H(e^{i0})|=1,\qquad
|H(e^{i\pi})|=\frac{1-\rho}{1+\rho}.
\tag{9}
\]

\(\omega\) 是参数序列在时间上的角频率，单位 rad/EMA-update，不是梯度之间的夹角。参数 EMA 的直流增益为 1；频域动量论文中 \(m_k=u m_{k-1}+v g_k\) 的增益 \(v/(1-u)\) 可以大于 1，那是梯度递推的性质，不能直接当作本轮参数 EMA 的解释。相关工作见 08。

### 4.2 相关扰动下的平均效应

用局部解释模型 \(\theta_k=\mu+\xi_k\)，假设 \(\xi_k\) 零均值、二阶平稳，\(\Gamma_\ell=\operatorname{Cov}(\xi_k,\xi_{k-\ell})\)，\(w_i=(1-\rho)\rho^i\)，则

\[
\Sigma_F=\sum_{i,j\ge0}w_iw_j\Gamma_{i-j}.
\tag{10}
\]

独立扰动特例：

\[
\Sigma_F=\frac{1-\rho}{1+\rho}\Sigma_\theta.
\tag{11}
\]

对自相关为 \(q^{|\ell|}\) 的标量投影，

\[
\frac{\operatorname{Var}(F)}{\sigma^2}
=\frac{1-\rho}{1+\rho}\frac{1+\rho q}{1-\rho q}.
\tag{12}
\]

因此不能把相邻参数当作独立估计。真实参数还含训练漂移；原始参数轨迹的样本方差不是已经辨认出的“梯度噪声方差”。本轮主要用预测函数变化和受控读出识别效用，不需要保存每一步全参数频谱。

### 4.3 滞后与方差的竞争结构

若局部均值 \(\mu_k=\mu_0+kv\)，则稳态滞后

\[
b=E[F_k]-\mu_k=-\frac{\rho}{1-\rho}v.
\tag{13}
\]

进一步在以 \(\mu_k\) 为局部最优点、正定曲率矩阵 \(C\) 的二次近似中，

\[
E[L(F)-L(\theta)]\approx
\tfrac12b^\top Cb+
\tfrac12\operatorname{tr}\{C(\Sigma_F-\Sigma_\theta)\}.
\tag{14}
\]

式 (14) 给出可区分的解释：平均可以降低某些扰动代价，也可以错过有益漂移。它不证明本仓库符合全部近似条件，更不证明所有高频变化有害。不同 cell 对同一参数变化的梯度、曲率和排序边界不同，是研究历史偏好差异的合理依据；是否存在可利用的差异由实验决定。

对实际光滑 cell loss，一个更直接的局部式子为

\[
L_c(F)-L_c(\theta)
\approx \nabla L_c(\theta)^\top h+
\tfrac12h^\top\nabla^2L_c(\theta)h,\quad h=F-\theta.
\tag{15}
\]

本轮不把完整 Hessian 计算列为必做。功能一阶指标与真实有限改变量的比较已经可以判断一阶代理是否有用。

## 5. 同轨迹收益与 checkpoint 效应

对固定 epoch e、cell c、训练类型 a：

\[
G_c^{a,v}(e)=U_c(v^a_e)-U_c(\theta^a_e),
\tag{16}
\]

其中 v 可以为 F 或某个持续 \(F^{(H)}\)。这是同阶段读出效应。

正式选点定义

\[
e_v^*=\operatorname*{arg\,max}_{e=1,\dots,50}
\mathrm{MacroAUC}_{\rm Validation}(v_e),
\tag{17}
\]

精确并列取最早 epoch。选点后的交付差为

\[
G_{\rm selected}=U(v_{e_v^*})-
U(\theta_{e_\theta^*}).
\tag{18}
\]

式 (16) 与 (18) 回答不同问题。预注册共同 epoch 的比较、完整 epoch 曲线及各自选点比较都要保留；不能将各自最佳点之间的差全部归为同阶段平滑效应。追加候选 H 带来的选择自由度由 06 的独立机制分区/交叉评价处理，不通过增加 Test 调参解决。

平均年龄相近的旧 online 对照直接从同一固定 Validation 人口的已有曲线读取。在 e≥5 的锚点，令 \(k=eK\)，

\[
a_k=\frac{\rho_F(1-\rho_F^k)}{(1-\rho_F)K},\quad
\ell(e)=\lfloor a_k+0.5\rfloor,\quad j(e)=e-\ell(e).
\tag{18a}
\]

比较 \(U(F_e)-U(\theta_{j(e)})\)，同时保留 \(U(F_e)-U(\theta_e)\)。这是以最近整数 epoch 匹配平均年龄的延迟对照，不声称与 EMA 历史分布完全匹配；不按性能另选 j。j<1 时记历史不足。它只需完整 epoch 指标，不要求保存旧 online 参数，也不新增训练。

## 6. 一个 epoch 的受控历史脉冲

### 6.1 操作与精确等价

脉冲终点固定为 e=5,10,…,50。epoch e−1 结束后，

\[
\psi_{\rm start}^{f}=\psi_{\rm start}^{s}=F_{e-1}.
\tag{19}
\]

在下一个完整 epoch 内，对同一在线 \(\theta_k\) 更新

\[
\psi_k^v=\rho_v\psi_{k-1}^v+
(1-\rho_v)\theta_k,\quad
\rho_f=\rho_F^2,\quad\rho_s=\sqrt{\rho_F}.
\tag{20}
\]

epoch e 末端定义

\[
q_e^v=\psi_e^v-F_e,\qquad
\eta_e^{v,b}=F_e+P_bq_e^v.
\tag{21}
\]

其他参数均保留 F，buffer 统一使用 F_e 的共同 buffer。参数 EMA 按坐标可分离，所以 \(\eta_e^{v,b}\) 精确等同于：从同一 F_{e−1} 出发，仅 b 组在该一个 epoch 采用快/慢 \(\rho_v\)，其他组采用 \(\rho_F\)。归纳证明只需把式 (20) 分别限制到 b 及其补集。

这不是任意交换两个独立训练模型的参数；它是同一固定轨迹上的确定性历史估计干预。它仍然是非线性模型中的条件组效应，不应解释为该组全部历史的唯一因果贡献。

令 \(q_k^v=\psi_k^v-F_k\)，还有精确数值校验式

\[
q_k^v=\rho_vq_{k-1}^v+
(\rho_F-\rho_v)r_k,\qquad q_{\rm start}^v=0.
\tag{22}
\]

式 (22) 表明脉冲改变的是整个窗口内创新的加权吸收。单步吸收、一个 epoch 脉冲、从训练开始持续的 H 读出是三个不同对象，不混合命名。

### 6.2 必须保存的收益方向

对每个 cell、组、终点和评价 panel：

\[
G_{c,b,e}^{f}=U_c(\eta_e^{f,b})-U_c(F_e),
\tag{23}
\]

\[
G_{c,b,e}^{s}=U_c(\eta_e^{s,b})-U_c(F_e),
\tag{24}
\]

\[
D_{c,b,e}=U_c(\eta_e^{f,b})-
U_c(\eta_e^{s,b})
=G^f_{c,b,e}-G^s_{c,b,e}.
\tag{25}
\]

D>0 只说明 fast 优于 slow；两者可能都差于 F。保持 F 是明确的第三种诊断结果。AUC 和 −LogLoss 各自计算，不合并成一个未定义分数。

### 6.3 组间配合与非加性

已保存的 \(\psi^f,\psi^s\) 同时提供全组脉冲读出：

\[
D_{c,\mathrm{all},e}=U_c(\psi^f_e)-U_c(\psi^s_e),
\tag{26}
\]

\[
R_{c,e}=D_{c,\mathrm{all},e}
-\sum_bD_{c,b,e}.
\tag{27}
\]

也保存各自相对 F 的非加性

\[
R^v_{c,e}=
[U_c(\psi^v_e)-U_c(F_e)]
-\sum_bG^v_{c,b,e},\quad v\in\{f,s\}.
\tag{27a}
\]

应满足 \(R_{c,e}=R^f_{c,e}-R^s_{c,e}\)，可作为分析数值检查。

R 表示本组划分和 F 背景下的非加性，不能分摊成某一组的责任。一次脉冲共有 27 个读出：F、12 个 fast hybrid、12 个 slow hybrid、all-fast、all-slow。只需两个额外 P 维影子和一个顺序复用的读出模型。

如果非加性会实质改变路径归因，阶段 4 只在保留的 e20/e40 向量上，对预定的至多两个关键组补第二背景：

\[
\eta^{f|s,b}=\psi^s+P_b(\psi^f-\psi^s),\qquad
\eta^{s|f,b}=\psi^f-P_b(\psi^f-\psi^s).
\tag{28}
\]

对应的两种边际为 \(U(\eta^{f|s,b})-U(\psi^s)\) 和 \(U(\psi^f)-U(\eta^{s|f,b})\)。它们回答在其他组全 slow 或全 fast 时该组的边际作用。默认不穷举组合、不计算完整 Shapley。

## 7. 精确 AUC 与 LogLoss：排序和置信度分开解释

### 7.1 概率 AUC 与排序翻转

源评价以原模型 FP32 sigmoid 概率为排序输入。即使 logits 在数学上与概率单调对应，也不能用 logit AUC 替代源概率 AUC，因为数值饱和可能产生额外 ties。

对 cell c 固定评价行，i 为正样本、j 为负样本，\(m_{ij}=p_i-p_j\)，

\[
B(m)=\mathbf1(m>0)+\tfrac12\mathbf1(m=0),
\quad
\mathrm{AUC}=\frac1{n_+n_-}\sum_{i,j}B(m_{ij}).
\tag{29}
\]

两个读出 a、b 的 pair 改变量

\[
v_{ij}=B(m^a_{ij})-B(m^b_{ij}),
\]

\[
F^+=\frac1M\sum_{i,j}\max(v_{ij},0),\quad
F^-=\frac1M\sum_{i,j}\max(-v_{ij},0),
\tag{30}
\]

\[
\Delta\mathrm{AUC}=F^+-F^-,\qquad M=n_+n_-.
\tag{31}
\]

tie 变化保留 1/2。\(F^++F^-\) 反映总排序变化，净值可能因纠正和损害相抵而很小。不存在“改变量大所以有益”的推断。

完整评价 AUC 用源精确排序函数，复杂度为 \(O(N\log N)\)，不枚举 V_mech 的全部正负组合。训练与 V_mech 各 cell 的 4096 对固定均匀配对只用于有界翻转诊断和平滑 ranking 指标。输出分别命名 exact_panel_auc、paired_sample_auc、sampled_net_flip；式 (31) 在全 pair 理论定义或同一抽样 pair 集合内部成立，抽样差不声称精确等于完整 AUC 差。不得无权重地用近边界挑选替代总体 AUC。

### 7.2 忠实保留源 LogLoss

将原 FP32 概率转 float64 后，源 LogLoss 为

\[
\mathrm{LL}=-\frac1N\sum_i
\left[
y_i\max(\log p_i,-100)
+(1-y_i)\max(\log(1-p_i),-100)
\right].
\tag{32}
\]

实现保留原 \(\log1p(-p)\) 运算，p=0/1 的对数下限为 −100。不能换成 sklearn 默认 epsilon clamp、对模型概率再次 sigmoid，或用 logit BCE 代替正式 LL。参见 [源 metrics.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/metrics.py)。

研究用光滑 BCE 可以在 logits 上写为

\[
\ell(z,y)=\operatorname{softplus}(z)-yz,\qquad
\Delta(-\ell)\approx(y-p)\delta z
-\tfrac12p(1-p)(\delta z)^2.
\tag{33}
\]

式 (33) 仅解释置信度效应和局部近似，不取代式 (32)。

理想实数运算下，单格 logits 的正仿射变换 z→az+b，a>0，保留排序却一般改变 LogLoss。因此 AUC 提升、LL 不提升可以明确解释为排序与概率质量的不同变化；先查看实际 logits、概率与翻转，不因两指标不同步自动怀疑已成立的 AUC 收益。

### 7.3 排序的时间变化量：不把同刻恒等式当机制

固定同一组正负 pair，令 `B(p_pos-p_neg)` 在正、零、负时分别为 1、1/2、0。对于模型读出 m∈{online,F}，从已保存的每 epoch 源概率计算

\[
T_{c,e}^{m}=\frac1{|\mathcal P_c|}
\sum_{(i,j)\in\mathcal P_c}
\left|B(p^m_{i,e}-p^m_{j,e})-B(p^m_{i,e-1}-p^m_{j,e-1})\right|,
\qquad e\ge2.
\]

这是 pair 排序分数的时间变化率；ties 的半步变化计1/2。同步记录正变化和负变化，分别说明排序改进与损害。P_A/P_B用已有4096 pairs；V_mech在preflight从每格自然人口的正负配对全集均匀无放回抽取4096个固定pairs，只保存索引；不足时用全部可用pairs、缺类为NA。V不参与梯度计算。所有epoch复用同一pair ID，零额外前向。

按既定阶段先在seed内汇总；early实际使用e2–10，不虚构e0预测。并列报告 online/F 的 T、同刻净AUC收益与年龄匹配online差。排序变化少既可能来自抑制损害，也可能来自延后改进，不预设为有益。单时刻“纠正减损害”只是 pair AUC 差的代数分解，不能单独当作波动降低的证据；时间量与收益的对应、延迟对照和局部历史干预共同约束解释。

## 8. 固定观测 panel

训练探针 P_A、P_B 按原训练 row identity 完全不相交，均保持参与正常 online 训练；不从训练人口扣除。每个 panel 包含两种用途：

| 用途 | 固定规模 | 使用方式 |
|---|---:|---|
| 单格排序 panel | 每 cell 256 正、256 负 | AUC 探针与 4096 个均匀正负配对 |
| 自然人口 panel | 每域 512 行 | 概率 LL、预测 RMS、自然人口功能变化 |
| V_mech | Validation 每域自然抽取 4096 行 | 独立于训练探针的机制标签；不参与 online 训练 |

同一 panel 的 row ID、各 cell 标签索引、pair ID 从预注册 RNG namespace 确定，在所有 epoch、读出、指标和参数组中固定。P_A/P_B 使用独立 namespace；训练 Gaussian 随机流保持源 key，不复用 probe RNG。

V_mech 另按稳定 hash(dataset,row_id) 的最低位分成 \(V_A,V_B\)，hash 不包含模型 seed，使同一 validation 行在所有 seed 始终落入同一半。它不修改自然训练、完整 Validation 人口或正式选点。完整 V_mech 用于描述性效应；H 偏好和主预测评价使用两半交叉读出，见第 10 节。两半正负类别不足时依 05 的 missing 规则处理，不依模型结果重新分配行。

抽样细节、缺失类别处理及身份 hash 由 05 固定。某格只有单一类别时 AUC 为 missing，不用 0 或 0.5 填充。角色不同的 row 集合与分母不能混用：rank panel 的类别分层适合 class-conditional 排序比较，不能用它的 LL 冒充自然人口 LL。

每个 epoch 保存 online/F 在 P_A、P_B 的有界 raw logits 和源概率；V_mech 的相应值从完整 Validation 评估流中提取。AUC/flip 使用源概率，pairloss/梯度使用 raw logits，不能从落盘 z 重新 sigmoid 后宣称 bitwise 重建原概率。脉冲终点只在这些有界 panel 上评估 27 个读出，不额外在完整 Validation 穷举 12 组。

## 9. 四类候选收益指标

全部候选先在 P_A 计算；P_B 对 M1/M3 做独立训练样本复测，并用精确风险变化检验 M2_P_A 的样本迁移，不额外计算 P_B 梯度。预测验证标签取 V_mech。低成本几何量没有标签；功能量和直接探针量的标签仅来自训练人口。

### M0：参数几何量及固定设计变量

在脉冲终点，定义

\[
d_b=P_b(\theta_e-F_e),\quad
\Delta_b=P_b(\psi^f_e-\psi^s_e),\quad p_b=|b|.
\]

基础几何特征冻结为

\[
x^{(0,1)}_b=\|d_b\|_2/\sqrt{p_b},
\quad
x^{(0,2)}_b=\|\Delta_b\|_2/\sqrt{p_b},
\tag{34}
\]

\[
x^{(0,3)}_b=
\frac{d_b^\top\Delta_b}{\|d_b\|_2\|\Delta_b\|_2}.
\tag{35}
\]

任一范数为零，cos 记 missing 并保存 zero flag；不能把未定义 cos 当作方向冲突。上述量只读取已有四份末端状态，无需新增保存上一步完整参数或梯度。每组相同几何特征可对应六格不同结果，这本身是待检验的信息限制。

固定设计/阶段变量为 dataset、augmentation、cell、group、epoch。它们构成简单预测参照，不能先按结果挑选有利阶段。

### M1：预测函数差异幅度

在 P_A 的 cell 自然人口行集 \(\mathcal N_c\) 上：

\[
R_c^{f,b}=
\sqrt{\frac1{|\mathcal N_c|}
\sum_{i\in\mathcal N_c}
[z_i(\eta^{f,b})-z_i(F)]^2},
\tag{36}
\]

同样计算 \(R_c^{s,b}\) 与 \(R_c^{f-s,b}\)。M1 的三个 RMS 均无收益正负方向；它们只说明参数改动是否在该 cell 的预测函数上可见，不能把“变化大”直接当作应吸收。

### M2：完整预测函数的排序效用方向导数

固定温度 \(\tau=1\)，无需搜索：

\[
L_{{\rm rank},c}(\vartheta)=
\frac1{4096}\sum_{(i,j)\in\mathcal P_c}
\operatorname{softplus}
\left(-\frac{z_i(\vartheta)-z_j(\vartheta)}{\tau}\right).
\tag{37}
\]

在 F、P_A 和相同 pair ID 上计算完整函数梯度

\[
g_c^{\rm full}=\nabla_\vartheta
L_{{\rm rank},c}(\vartheta)\big|_{\vartheta=F}
\in\mathbb R^P.
\]

核心标量为

\[
\gamma_{c,b}^{f-s}
=-(g_c^{\rm full})^\top
P_b(\psi^f-\psi^s).
\tag{38}
\]

同时保存

\[
\gamma_{c,b}^{f-F}=-(g_c^{\rm full})^\top P_b(\psi^f-F),
\quad
\gamma_{c,b}^{s-F}=-(g_c^{\rm full})^\top P_b(\psi^s-F).
\tag{39}
\]

正值表示该方向在一阶近似下减少 ranking loss。一次 cell 梯度与所有组位移作内积即可，主分析每终点共六次梯度计算，无需每组各做一次 backward，也不构造 Hessian。内积累加用确定的参数顺序和足够精度，梯度算完即释放，不落盘。

**必须区分完整函数导数和源训练代理。** 源模型 gate 输入使用 emb.detach()。真实参数反事实会改变 embedding 及其 gate 输入，因此 M2 主指标通过

    ModelAdapter.forward_logits(..., probe_input_full_graph=True)

获取与实际前向函数有限差分一致的导数；训练始终使用原 detach 语义。必须同时保存 probe_input_full_graph=False 得到的 source_detach 标量作为代理对照，输入、pair、归一化必须完全一致。每个终点为六格×两种导数语义，共 12 次梯度归约。不能把两种导数混写为同一个“梯度收益”。

同时从已计算梯度归约 `grad_rank_rms_full(c,b)=||P_b g_c^full||₂/sqrt(|b|)`，以及 source_detach 对应值，立即释放向量。前者作为无符号简单参照进入 06 的 C_simple；它是训练探针排序风险梯度的 RMS，不是原训练 BCE/Adam 梯度。全 fast/F、全 slow/F 的 P_A 六格 Macro AUC 差也已由同一组读出得到，作为两个全局历史偏好参照。两项均不增加 forward/backward。

完整函数梯度只在独立 scratch 观测调用中求取，不使用训练 optimizer，不累积到 online .grad，不改变 Adam 状态或训练图。必要数值测试用同一输入下的中心有限差分检验式 (38) 的小步方向，而不是要求有限一个 epoch 脉冲严格满足线性近似。

F、\(\psi^f,\psi^s\) 的历史更新均已 detach。求 \(g_c^{\rm full}\) 只对当前 F 处的预测函数求导，不展开 EMA 历史，不对生成脉冲位移的训练过程反传。

### M3：训练人口上的直接反事实收益

在 P_A 的 rank panel 上计算原概率 AUC，以及固定配对的式 (37)。分别保存

\[
D^{\rm AUC}_{A,c,b}=
\mathrm{AUC}_{A,c}(\eta^{f,b})-
\mathrm{AUC}_{A,c}(\eta^{s,b}),
\tag{40}
\]

\[
D^{\rm rank}_{A,c,b}=
L_{{\rm rank},A,c}(\eta^{s,b})-
L_{{\rm rank},A,c}(\eta^{f,b}).
\tag{41}
\]

两者正号都表示 fast 更优。相对 F 的 fast、slow 两个收益同样保存。自然人口 panel 上另报正式 LL 收益与功能变化，不混合采样分母。

P_B 使用完全相同公式得到独立复测。主预测输入固定为 P_A；P_B 只用于采样稳定性，不另拟合 A/B 均值版本，不按结果挑选更好的 panel。A/B 的差反映探针样本敏感性；两个 panel 不等于两个 seed，也不能把二者合并后再冒称独立验证。

M3 在训练探针上是真实读出收益，在 V_mech/未来窗口上是否有效仍须检验；它不因“直接”二字自动胜过 M2 或 M0。原始 AUC 在小变化下可能没有 pair 翻转，平滑 rankloss 可以补充功能信息，但不能用其提高冒充 AUC 提高。

### 四类指标的输出合同

| 家族 | 主输出单位 | 是否自带收益符号 | 参数组/cell |
|---|---|---|---|
| M0 | 参数 RMS、cos、zero flag | 否 | 每 b；复制 cell 仅为对应分析 |
| M1 | logit RMS | 否 | 每 c,b |
| M2 | 一阶预测的 ranking loss 减少量 | 是 | 每 c,b，full/source 明确区分 |
| M3 | 实际 AUC 差、ranking loss 减少量；另报自然 LL | 是 | 每 panel,c,b |

每个记录携带 dataset、seed、a、epoch、group、panel hash、pair hash、layout/group hash、读出状态 hash、\(\rho_F,\rho_f,\rho_s\)、正负样本数、missing flags。单位和正号不依赖后续结果变化。

## 10. 当前对应、未来预测与独立性

当前跨人口对应：用 P_A 的 M0…M3 与同一 e 的 V_mech 脉冲收益比较。这检验样本迁移与描述效用，不称为时间预测。不同 H 的局部历史偏好采用留一 seed 与 validation 行二分的交叉评价：其他四 seed 的 \(V_A\) 在同一预注册 epoch 选择统一 H 或各 cell H，在留出 seed 的 \(V_B\) 评价；再交换 A/B。候选集合不变，每个 seed 先汇总两个数据折。这样区分额外选择自由度与在未用于选择的行上保留的效用；条件读出仍是机制诊断，不宣称已实现一个单模型算法。

对指定验证面板 \(V\in\{V_{\rm mech},V_A,V_B\}\)，未来目标为：

\[
Y_{c,b,e}^{V,\rm AUC}
=D_{V,c,b,e+5}^{\rm AUC}.
\tag{42}
\]

e+5 时从 F_{e+4} 重新开始一个新脉冲。该目标是未来再次实施相同历史吸收改变的收益，不是旧脉冲五个 epoch 后的训练影响。由于无反馈，旧脉冲本来就不会改变未来 \(\theta\) 或 F。

特征仅使用 e 及此前的训练观测。固定训练 origins 为 {5,10,15}，未来评估 origins 为 {25,30,35,40,45}；每个目标均为 origin+5。每个 dataset 单独执行三重隔离：

1. 外层留出一个 seed；拟合只用其他四 seed。
2. 拟合只用早期 origins，评价只用较晚 origins；拟合目标最晚 e20，评价特征最早 e25。
3. 拟合的未来收益标签只来自 \(V_A\)，留出 seed 的未来评价标签只来自 \(V_B\)；交换 A/B 再做一次。

标准化与固定 ridge=1 只用对应拟合折。每个 seed 先合并两个行分区折，再五 seed 汇总；十个模型/行分区组合不是十个独立 seed。保留完整 V_mech 的描述性效应，但不把它混入上述样本外预测标签。不同 dataset 分别拟合主分析，再比较跨数据集现象；epoch/cell/group/panel/shadow 均不增加独立 seed 数。

模型细节、简单对照、预测误差、符号一致性和选择 regret 的判定由 06 固定。相同时刻相关系数不替代式 (42) 的预测结果；预测价值可以成立而完整因果机制仍在其他层次。

为保留 F 这一合法选项，ridge 实际拟合两个目标列。下面以 \(V\) 表示当前折对应的 \(V_A\) 或 \(V_B\)，

\[
Y^{V}_{c,b,e}=
\left[
G^f_{V,c,b,e+5},
G^s_{V,c,b,e+5}
\right]\in\mathbb R^2,
\tag{42a}
\]

由 \(\widehat D=\widehat Y_f-\widehat Y_s\) 得到主报告。样本矩阵的 target 形状为 [N_records,2]，主 D 为 [N_records]。离线三选一诊断为

\[
\widehat a=\operatorname*{arg\,max}
\{0,\widehat Y_f,\widehat Y_s\},
\tag{42b}
\]

并列优先 F，其他并列按 06 的固定顺序处理。无需拟合的 M2/M3 同样使用各自 fast−F、slow−F 两列比较 0；M2 原始单位是 ranking loss 一阶减少量，不能直接称为 AUC 预测值。不同家族的主要预测误差比较通过同一个固定形式 ridge 标定，避免单位差异制造优劣。该选择只用于离线计算 regret，不在训练期间更换 F 或控制增强。

## 11. Gaussian 与 EMA 的四格交互

对 \(a\in\{\mathrm{clean},\mathrm{gaussian\_d0}\}\)，

\[
G_{{\rm EMA},c}^{a}(e)=
U_c(F_e^{a})-U_c(\theta_e^{a}),
\tag{43}
\]

\[
I_c(e)=
G_{{\rm EMA},c}^{\rm gaussian\_d0}(e)
-G_{{\rm EMA},c}^{\rm clean}(e).
\tag{44}
\]

两个训练目标对应不同在线轨迹；各轨迹内部 online/F 则完全共用训练。总组合差
\(U(F^{\rm gaussian})-U(\theta^{\rm clean})\)
不能直接命名为 EMA 贡献。

- I>0：该评价口径下 Gaussian 增加 EMA 的边际收益。
- I≈0：在观测精度和效应量范围内，两者近似可加。
- I<0：Gaussian 降低 EMA 的边际收益；组合总收益仍可能为正。

必须报告每 seed 配对交互、六格及 Macro，并区分相同 epoch 与各自选点后的交互。I>0 不自动证明“EMA 消除了增强噪声”，也不自动意味着超过最强发表基线。

现有 Gaussian 在 sparse embedding 上使用 detached 字段 RMS 尺度、目标 D0、共享任务噪声视图及 replacement loss；本轮继承这些语义。条件固定的小扰动近似 \(E\ell(e+\sigma\xi)\approx\ell(e)+\frac{\sigma^2}2\operatorname{tr}H_e\ell\) 可启发解释，但不能替代真实有限尺度、非线性 MDMT 模型的实验，也不改变训练目标。

## 12. 本文件允许形成的结论

1. 同轨迹读出差可以形成明确的历史估计效应结论；不要求先解释全部优化噪声。
2. 非重叠短窗口脉冲可以定位可重复的“快吸收、慢吸收、保留 F”偏好及组背景依赖。
3. 跨 cell 的不同效应是 MDMT 收益结构证据；在未参与拟合的验证行、留出 seed 和未来窗口仍保留的信息，进一步支持本研究范围内可利用的指标价值，不要求先验证新人口才接受这一结论。
4. 某指标无预测增量，可以排除该指标在本协议中的必要性；不因此否定已有 F 收益。
5. 参数低通、几何一致性、预测变化幅度、真实收益及未来预测是不同层次。按实际观测收敛结论，不从任一层次自动推出其余层次。
