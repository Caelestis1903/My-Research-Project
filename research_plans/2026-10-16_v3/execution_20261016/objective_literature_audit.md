# v3 已有方法的 objective 审计（实验判定尚未完成）

设本轮目标 J=Σ_d p_d mean_{i∈d}(mean_t BCE)。IPW 通过 q_d·(p_d/q_d)=p_d 保留目标。任何新增样本权重 a_i 若未满足逐样本目标测度身份，一般会改变目标；只做权重均值归一化不足以保证身份。

[D3](https://ojs.aaai.org/index.php/AAAI/article/download/28699/29354) Eq19 使用 complexity 与 specificity 两权重之积乘交叉熵；Eq16–18 由 attention/融合信号生成这些权重，而非逆采样概率。因此完整 DBO 一般改变经验目标。将可变权重退化成常数可恢复自然采样的未加权损失，但同时关掉了所研究的权重机制；不能将该退化写成完整 D3 机制复现。

[SAF](https://ojs.aaai.org/index.php/AAAI/article/download/38491/42453) §3.6 Eq14 使用底层与 expert 层的学习样本权重之积；Eq11–13 通过 HSIC 与移动统计更新权重。此权重不被约束为 p/q，完整方法的 expected objective 一般与本轮不同。其输入/架构、权重学习和自适应状态也共同改变。不能用相同“sample weighting”术语宣称机制已等价，也不运行未经授权的近似 SAF 替身。论文关于稀缺域的结果只限定其数据环境。

[AdaTask](https://arxiv.org/abs/2211.15055) 分离 task-wise 累积梯度和自适应状态。本轮保留 L=L_task0/2+L_task1/2 与其梯度和，再改变 optimizer map；L2 各 task 分配一半。它是可进行相同目标对照的已发表优化机制。其能否消除 v3 sampling effect 必须等正式四种子训练和完整 readout，不能由公式推定。

一般的同目标 sampling/variance 效应、顺序效应、optimizer memory 效应已有文献覆盖；本轮 MDMT 参数块的必要性与跨环境边界仍需反事实证据。完整文献清单与来源见 literature_source_receipt.json。最终 covered/partially-covered 判断等待实验闭环，不以论文场景不同作为创新证据。
