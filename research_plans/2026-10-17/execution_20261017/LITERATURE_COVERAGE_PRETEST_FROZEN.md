# N6 原文差异表（训练前冻结）

2026-10-17直接访问以下八篇原文。此表是具体已检索范围，不声称穷尽；历史v3 NO_GO不改写。

|原文与定位|目标/公式及采样单位|优化器、度量与假设|与本研究的对应/边界|
|---|---|---|---|
|[D3, AAAI2024](https://ojs.aaai.org/index.php/AAAI/article/view/28699/29354), Eq15–19|attention导出c，c'=clip((c−mean(c))/(2std(c))+.5)；w_cpl=λ1+α1c'、w_spf=λ2+α2g_sp；逐样本BCE乘两权重；同时改变融合模型|推荐CTR的AUC/LL；权重是学习模型/当前batch函数，目标随其变化；不是固定q重要性修正的恒定目标对照|一般不满足q*w=p的恒等约束；完整D3不是本轮同J机制等价控制，也不能由本轮结果否定完整D3效果。本文印刷Eq15为ΣAlogA，不能自动替换成负号后声称逐式复现。|
|[SAF, AAAI2026](https://ojs.aaai.org/index.php/AAAI/article/view/38491/42453), Eq3,11–14|RFF近似的加权HSIC：n^-2 ||(H diag(w) Z_A)^T(H Z_B)||²_F；交替更新特征/专家样本权重，再以w_f*w_ex加权MDMT loss|MDMT AUC/LL，跨域任务融合；全局与batch特征/权重EMA，交替优化；改变目标和模型|学权/特征去相关不同于固定Sample J的p/q修正；本轮保留原routing detach，但不实现近似SAF冒充完整方法。|
|[AdaTask](https://arxiv.org/html/2211.15055), Eq5–7 / Alg2|每task每坐标独立m_k,v_k，θ更新为各task归一化更新之和；学习目标本身可以保持固定|AdaGrad/RMSProp/Adam适配；AU/rAU与各task效果；共享参数的累积状态支配解释|比D3/SAF更接近状态控制；v3固定配置组件未消除q敏感性，不等于论文全方法失效。本轮N1主比较只Adam/无动量SGD，不把组件表现当同预算全面基线结论。|
|[Importance Sampling for Minibatches, JMLR2018](https://www.jmlr.org/papers/volume19/16-241/16-241.pdf), Def2,Thm3,§5–6|划分非空buckets，每bucket恰取1样本，按包含概率修正；研究采样相关性和ESO界|dfSDCA/正则ERM；凸、光滑loss及强凸正则的线性收敛范围，优化误差/复杂度|支持采样设计/相关结构已被研究。本轮domain quota是每域多WR及随机整数舍入，不是直接复现该bucket算法；M3OE非凸detach字段不满足其全套收敛假设。|
|[Not All Samples Are Created Equal, ICML2018](https://proceedings.mlr.press/v80/katharopoulos18a/katharopoulos18a.pdf), Eq2–8,20|逐样本概率p_i、权重1/(Np_i)；方差trace控制局部SGD距离进展，q∝梯度范数；用可算上界近似|SGD、计算代价和收敛；可导网络局部界及输出层近似的条件；不是AUC非劣保证|同均值不同方差及重要性采样不构成新发现；这里固定域proposal、非凸MDMT排序/LL取舍尚需实测。detach图可用向量矩身份，但不能自动套普通完整梯度的全局最优距离解释。|
|[Optimizer Memory Makes Shuffle Order a First-Order Source of Fine-Tuning Noise](https://arxiv.org/html/2606.29554), Eq7,10,13–17 / AppA|固定有限窗口、同multiset置换，lifted-state θ端点差=ηΛ+O(η²)；非零Λ依赖预热状态和固定β时钟|memoryless SGD与固定时钟momentum/AdamW；局部参数/读数差及置换方差；平滑、非退化、有限窗口条件|本轮为WR的q干预、coupled WD Adam及detach字段，不是该AdamW定理直接应用；顺序/optimizer-memory通用主张已有覆盖，v3随机ORDER不等于课程。|
|[Exploring Variance Reduction in Importance Sampling for Efficient DNN Training, SIAM](https://epubs.siam.org/doi/10.1137/25M1726339), Prop4.1 / Eq5.1–5.2|EMS=N*tr(V_uniform)/tr(V_IS)，来自每样本梯度矩估计；SGD η_EMS=η*(EMS/N)，Adam建议平方根缩放|梯度估计均值和trace匹配；DNN训练/误差/成本；trace等价不保证形状、非线性更新或AUC等价|ESS仅权重矩，不能替代EMS。N1公平LR响应面先检验配置解释；只有后续候选仍需噪声适配才按论文实现EMS基线，不扩张矩阵。|
|[Multiple Importance Sampling for Stochastic Gradient Estimation](https://arxiv.org/html/2407.15525), Eq3,5–8 / §5|多个面向参数梯度的proposal；按MIS分配组合；OMIS解Aα=b，估计值Σα_j。balance heuristic相当于混合proposal|SGD向量梯度估计、在线近似/输出层重要性及矩EMA；估计质量、收敛和成本|blockwise proposal思想已存在。只有明确未覆盖的MDMT结构命题才讨论新贡献；本轮N2四臂不称OMIS，N3更新替换不称采样策略。|

本轮可检验缺口是冻结数据/图下的q×LR×预算×评价人口取舍，以及域计数和共享C局部干预；是否稳定、是否有机制及方法价值均待N1–N3数据。GO-EFFECT、GO-MECHANISM、GO-METHOD分别判断。新训练test未读。

状态：N6训练前原文差异表COMPLETE；最终结论须在实测后补充。
