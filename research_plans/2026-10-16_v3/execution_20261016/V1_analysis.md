# V1 objective门控：PASS（不是本轮终局）

16个固定theta审计覆盖Adam/SGD、四种子及epoch1/5/10/20/40的规定快照；每臂512个独立40960行批次。完整模型1614254个参数坐标均参与，非投影。完整训练集分域梯度与独立全行遍历的最大差为3.3525e-7，满足预先数值容差；输入数据fingerprint与源checkpoint配置逐项相同。权重身份与group-gradient身份通过。此处“梯度”明确指既有routing stop-gradient计算图，Q28限制保留。

160个block/arm/snapshot的MC均值偏差均通过预注册保守插值方差门；其中155个处于逐点bootstrap95%噪声尺度内，最大比例1.228。不能把非拒绝写成精确相等证明，保守界也不是已证明的有限样本覆盖保证。相同期望的依据同时包括每行权重代数、代码的逐行独立性和完整经验梯度身份。

所有16个快照的全坐标SB/SN covariance trace比为2.316–2.752。证据STRONG，仅限定这些固定theta和生产批次；还不能说更大的variance造成AUC下降，或排除方向、曝光、重复、coverage、order。epoch10的四种子与seed2022的跨epoch快照分开解释，不把16个快照当16个独立训练种子。

四种子固定子集诊断中，两个exact目标轨迹的最大差均0，随机SN/SB终点距离分别0.667/0.626/0.546/0.489。它支持消除随机估计器时的数值身份，不能替代全数据训练的泛化或因果中介证据。

Q29梯度支持审计四种子完全一致：R gate与tower各只激活1个cell；STAR slot激活同域2个task；C及E的各参数tensor均可被6个cell使用。故R本身已是cell-private，P混合了两种sharing角色，E不能因名字含domain/task就被叫作private。此结构事实为STRONG；其性能必要性仍需V7。

V1通过，当前没有满足最终提前终止条件。继续16项LR pilots（仅selection validation）与16项50epoch主轨迹。完整数据与哈希见analysis/V1_*.csv、各audits/receipt.json及deterministic/、support/。巨大原始梯度使用producer哈希并独立检查形状；其余紧凑数值产物重新计算哈希，未伪称对100GB矩阵二次完整哈希。
