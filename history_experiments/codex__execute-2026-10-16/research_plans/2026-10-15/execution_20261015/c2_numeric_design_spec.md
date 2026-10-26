# C2.1 固定参数估计器 oracle（0 training）

2026-10-16，主 agent DESIGN。用户最新禁止子 agent，后续实现、SELF_REVIEW、CPU RUN、分析均由主 agent顺序完成；不称独立review，不签发原F0或C2训练GO。

依据原规划§4.3与§10.1，在B1持久化调度器等待GPU期间进行独立数值前置检查，不修改冻结B1源码、数据或旧checkpoint。

- 两数据集：KuaiRand/natural与MovieLens/d2_large；seeds 2022–2025。加载现有真实训练split，仅从每域固定取16个训练row作为有限oracle总体（不按label选择）。每seed记录真实row IDs、完整数据hash、schema/split、实际初值、源码。有限总体不代表全训练分布或selected效果。
- 原M3OE结构、初始化与C/P/A分区，eval、固定θ、不训练。检查LayerNorm、无随机dropout/跨样本BatchNorm条件及softmax_type=3；验证分批预测与整批预测一致。每row×task取实际autograd梯度，任务均值后形成每row梯度。真实θ在RUN前后必须不变。
- α∈{Equal=1/3, Sample=全训练p}，q∈{Natural=p, Balanced=1/3}，另加stress q=(.0001,.9998,.0001)；域内在16个row上均匀有放回。损失与梯度使用固定分母和α/q。先精确枚举有限总体，核对full-coordinate期望与α目标，不能只凭Monte Carlo近似证明等价。
- B∈{16,128}的Monte Carlo，10000个独立batch；记录loss均值/方差、预先固定规则选取的每C/P最多32个非零支持坐标与全部A坐标的均值/covariance、norm、缺域概率、unique exposure/reuse。full-gradient covariance trace按每row全参数梯度精确计算，明确不构造P×P全矩阵。坐标选择为各分区非零支持上的等距位置，并记录名称/index。
- Monte Carlo均值与精确loss/梯度对照，报告SE与最大标准化残差；有限样本检验容差为8SE+数值容差。理论期望full-gradient相对残差≤1e-10；模型聚合autograd与单row聚合在float32规范下相对残差≤1e-4。MC未满足时保持INCOMPLETE并诊断，不修改抽样定义以获得PASS。
- batch-cell normalized Equal明确为(1/3)Σ存在域(mean-row BCE)，缺域贡献0；精确期望是(1/3)Σ[1-(1-q_d)^B]μ_d。旧原始Σ_{d,t}mean BCE是此定义的6倍。不得把这两者与fixed-p weighting默认为同目标。
- Adam一步仅在detached坐标副本上，使用相同θ、fresh zero moments、step0、main/C/P lr=.01、A lr=.001、betas(.9,.999)、WD=1e-5、eps=1e-8。用真实torch.Adam对MC样本和目标期望梯度的formula oracle复核；记录E[Δ(g)]与Δ(E[g])，以及不同q的差异。不构造异轨迹m/v，不作optimizer-history因果归因。
- 输出`C2_numeric_oracle_20261016/`中的source/data/row manifest、精确/MC统计表、分区/选定坐标covariance、cell描述、Adam数值核对、RUN receipt。原始逐row全梯度不入Git，关键统计可归档。

验收仅限有限总体与固定θ的目标/更新数值一致性。任何差异都不是选中AUC改善证据，不能证明课程、动量或小域困难。C2训练仍须原规划前置条件与后续主agent判定；B1及总目标不因oracle完成而自动完成。
