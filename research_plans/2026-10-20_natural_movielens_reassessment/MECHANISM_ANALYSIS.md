# 机制分析（阶段性版本，规定诊断尚未全部完成）

本文件先记录已完成的精确权重计算及可支持的解释边界。160个core固定点任务、N17 N2/N3及quota训练、S6、AO04和N19/CM19诊断仍待完整闭环；此时不能报告它们已证实某种机制。

## 1. 目标、proposal和训练导数

固定经验人口及模型状态，α表示目标的域权重，q表示抽取域的proposal，域内均匀有放回抽样。单行损失对两个任务取均值，重要性权重w_d=α_d/q_d。

`SN: α=p,q=p`；`SB: α=p,q=1/3`；`EN: α=1/3,q=p`；`EB: α=1/3,q=1/3`。同α的SN/SB或EN/EB具有相同的经验目标值及预期训练导数；跨人口时域内经验均值和Sample的p均可能变化，不能再称同一经验目标。

这里的g是原实现包含routing detach的训练自动微分场。其恒等式是`E_q[(α_d/q_d)g_i]=Σ_d α_d μ_d`，不把它偷换为重新连接路由后的全函数风险导数。原图的训练导数与相同前向预测下重新连接的risk Jacobian须分列；也不能把含A_arch的全部参数诊断当成主optimizer实际同时更新的参数块。

同目标代数身份的证据为 **STRONG**。固定96行oracle仅证明该有限pool的数值身份；全人口MC512是另一个估计对象。两者均不是直接的AUC或长期更新结论。

### 1.1 同目标trace差异仍取决于域内梯度几何

对任意固定参数分区，记单行训练导数的域均值为$\mu_d$、域内协方差为$\Sigma_d$，$G=\sum_d\alpha_d\mu_d$。在IID有放回、固定B、无batch内额外耦合的定义下：

$$\operatorname{tr}\operatorname{Cov}(\hat G_q)=\frac1B\left[\sum_d\frac{\alpha_d^2}{q_d}\left(\operatorname{tr}\Sigma_d+\|\mu_d\|^2\right)-\|G\|^2\right].$$

可分为域内项$B^{-1}\sum_d\alpha_d^2\operatorname{tr}\Sigma_d/q_d$与域计数项$B^{-1}[\sum_d\alpha_d^2\|\mu_d\|^2/q_d-\|G\|^2]$。两项均依赖实际梯度几何，不能用标量ESS替换。相同theta、同alpha时，Sample目标的SB−SN trace差为$B^{-1}\sum_d(3p_d^2-p_d)E_d\|g\|^2$；Equal目标的EB−EN差为$B^{-1}\sum_d[1/3-1/(9p_d)]E_d\|g\|^2$。系数有不同符号，单凭p更均衡不能保证某个q总是减方差，更不能由trace直接推出AUC方向。

上述是条件数学身份（**STRONG**），不是尚未完成的MC实测结果。含训练BN等batch耦合的其他模型、无放回shuffle、短尾batch以及原系统quota的整数余数，不能未经核对直接套该IID公式。N17的quota与计数分解需按实际实现和保存的计数评价；这里没有把理想整数固定计数当作其逐batch精确身份。

## 2. 人工人口确实放大了标量权重差异

从实际冻结F0人口manifest精确计算。以下ESS/B=`(E_q w)^2/E_q(w²)`，是IID标量权重诊断，**不是实际梯度协方差、独立覆盖数或optimizer更新方差**。

|人口|SN|SB|EN|EB|
|---|---:|---:|---:|---:|
|人工固定2022，187500行|1.000000|.494478|.130086|1.000000|
|自然完整，800167行|1.000000|.920507|.891103|1.000000|
|自然比例匹配N，187500行|1.000000|.920507|.891104|1.000000|

人工EN中最小物理域d1的w=20.833333；自然EN三个域分别约1.710318/.837173/.819124。标量权重二阶矩由人工EN的7.687198降至自然EN的1.122205。自然p约.194896/.398166/.406939；最大/最小份额比约2.09，而人工为50。

来源：[逐域精确系数](analysis/sampling_weights/exact_domain_coefficients.csv)、[二阶矩/ESS](analysis/sampling_weights/scalar_weight_summary.csv)、[重现脚本](analysis/analyze_sampling_weights.py)。此表使用base训练人口，不能替代扣train-meta后的N19/CM19 main p；也不是S5/S6无放回shuffle loader的完整协方差公式。证据 **STRONG**（冻结人口下的算术事实）。

## 3. 权重标量不足以解释全部新旧性能差异

AC-2D同N=187500、同250main步和50arch步下，EN−SN selected从人工−.018376变为自然比例匹配N的−.001267；支持总量相同已改变原强负差距，所以不能只归因总N。另一方面，EB在这两人口的α=q=uniform、w恒为1，selected宏AUC却由.704953变为.737580（约+.03263）。因此“仅p/q标量权重波动足以解释所有人口差异”的解释不成立。

这不排除实际梯度/更新噪声：域内特征、标签和支持改变仍会改变g的均值与协方差。它也不识别哪一种支持属性是唯一原因。两人口的具体数据、学习轨迹及辅助优化路径均需保留。性能条件证据 **MEDIUM**；唯一机制归因 **WEAK**。详见 [AC-2D性能分析](AC2D_PERFORMANCE_ANALYSIS.md)。

自然V2 fixed Adam final SB−SN仍为−.004084、四seed负，而selected +.000479、两正两负；AC-2D同类比较更小且阶段不同。自然比例更均衡不等于q效应消失，也不保证Adam历史状态是必要条件。V9无动量SGD的阶段读数保留为必要反例和边界，完整固定窗口尚待补读。

## 4. 规定机制任务的解释规则

|证据|它能回答什么|不得据此声称|
|---|---|---|
|core完整坐标MC512及有限96行oracle|特定θ/人口/目标/q/B下的训练导数均值及trace；有限pool身份另列|trace下降必然提高AUC；有限pool等同完整人口；非零MC均值差范数即有偏|
|同基线m/v/current网格|同θ和指定历史下的瞬时更新映射差异|历史m或v在完整轨迹中单独必要；不同历史路径已经被唯一中介识别|
|N17 quota与IID、精确域均值分解|计数项相对域内噪声，以及完整预算下quota×q差异|相同标量方差必然同结果；计数项小就全局排除所有数据分配机制|
|N17双路径锚/独立bank的C子块、norm/direction|固定θ及native state的一步局部作用与交互，MC误差单列|局部敏感块即长期必要中介；draw/bank/相关cell等于独立训练seed或总体|
|S5/S6完整历史/当前控制|原persistent Adam、同批次输入下history/current联合协议对完整风险的影响|HT/DH已经只改变m；共享梯度幅度、v、时钟和健康状态均被分别排除|
|AO04 q/rho Jacobian及有限扰动|固定θ、固定小验证探针上的分支可识别性/敏感性|小探针的单类AUC可填.5；B0局部反事实是重新训练的B1；有导数即有完整性能收益|
|N19容量/路由/信用与P2/P6日志|实际分配、固定meta上的有限动作和估计器实现|slot矩阵谱即整个共享乘性operator谱；one-hot选择优于局部坐标即优于当前非线性mixture|
|CM19 path与共同state局部counterfactual|实现内中和动作在固定探针/检查点是否影响预测，局部optimizer作用|精确0是全函数永久吸收态；路径恢复必然有竞争性能收益；不同模型中和尺度可直接排名|

MC误差范数/RMS是误差尺度，不是自动的置信区间；向量差范数有有限MC的正值基底。固定B=40960的诊断不代表每个训练短尾batch。N3只按结果前的MC精度门控加至最多64draw/bank，不按效应符号补抽样。

## 5. 既有CM19精度与尺度反例

原人工ML名义AdamW的float32乘数为1，12/12个fixed/STAR配对的final模型canonical SHA与global Adam WD0相同；KR实际乘数.9999999403953552且0/12相同。自然ML保持原lr/WD/dtype，不能把未实现的正收缩效果写为已经检验过。

旧KR fixed selected的固定development探针modulation差为0，但一个seed在final出现极小浮点级非零；独立STAR在更大的首步L2/data梯度范数比下仍保持路径活跃。精确观察、数值可表示性、尺度解释和端点效用分别判断。见 [历史重新聚合](HISTORICAL_NUMERIC_RECONSTRUCTION.md)，新自然路径尚待完成。

## 6. S5两预算：已完成的历史/当前控制

自然matched250/full50各16个完整run均已完成，1344个预测cell已复算；Sample S6仍待完成。共享C/A的持久状态/覆盖公式保留，同seed两预算共八run初始SHA相同，每预算内四臂的300/1050个实际main/arch批次SHA相同。原公式残差、实际预算和全坐标遥测已有原始证据，见[S5阶段分析](ACS5_PHASE_ANALYSIS.md)。这项实施身份为 **STRONG**，并非一阶历史必要性的识别。

HT−F selected宏AUC−.009759、四负，LL−.007626、四改善；target AUC−.008006四负，与旧KR/人工ML的救援方向相反。C块E41–50，HT实际位移范数为F的2.512–2.778倍，但持久m/v范数均更小，四seed一致。这与状态尺度及FULL当前位移放大兼容，参数路径和两种状态的联合变化没有被拆开；不能据此单独认定v、m或共享C为唯一中介。稳定条件现象为 **MEDIUM**，唯一因果机制为 **WEAK**。

matched250的T/HT/DH相对F没有“两较小域改善、最大域下降”的严格模式；DH−HT却在selected3/4、final2/4出现该模式。full50 HT−F final则4/4出现该模式：宏AUC+.005965，但target AUC−.006671四负、LL+.046714四差，non-target AUC+.012283四正。DH full50 final宏AUC+.016059、LL−.087078均四seed改善，六cell AUC均四正；其selected宏AUC仍−.003070、只有一正。因此取舍必须按候选/参照与阶段定义，同时报告其相对F的整体下降，不能把所有对照压成不存在取舍或普遍小域受益的单一故事。

同人口改变预算后，HT−F末轮AUC交互+.015140、DH−F+.023150，均四正；但F本身full50−matched250末轮AUC−.023105，DH只有+.000045且LL+.022672四差。相对救援与参照后期风险升高共同存在，不能改称取得更高selected最优值。两预算主/arch比5:1→20:1，仍未独立识别纯时钟。

## 7. 需要在最终版本补齐

完成后逐人口/模型/任务/域/θ阶段更新上表的实际数值、方向、四seed一致性及竞争解释。所有“待完成”只有在对应raw receipt、完整预算与冻结评价存在时才能改为完成；不能以本文件的解释框架代替实验。当前整项研究状态仍为IN_PROGRESS。
