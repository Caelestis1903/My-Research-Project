# AIGM01 实施报告

实现依据：[DESIGN_SPECIFICATION.md](DESIGN_SPECIFICATION.md)。新增代码仅位于 `experiments/gradient_aug_aig_m_20261101/`；GAM01、GA31-R3、历史报告、原始数据与计划保留原样。所有Python检查使用 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

## 实际实现

| 文件 | 责任与实际行为 |
|---|---|
| `protocol.py` | 复用GAM01来源协议并冻结新study、精确有限配置、独立组件开关、完整配置/运行hash、四种固定shadow、纯RAM snapshot/restore、模型/optimizer身份与源码manifest。 |
| `optimizer.py` | 按模型原参数顺序保留E/P/U静态标签；E/P更新前使用旧未校正q与不含wd的真实g聚合u，固定βδ=.95，仅组标量λ变化；m含coupled wd、q不含wd，v使用当前r+λd，分子使用分别偏差校正的m/q。U无q且使用标准Adam运算次序。 |
| `objectives.py` | 直接复用GAM01真实目标、Gaussian视图与clean/delta拆分。A0/A1一次backward；SAD/AIG两次autograd.grad、全部梯度完成后只更新一次；增强关闭/λmax0/κ0不抽view。固定SAD直接调用历史类。 |
| `run.py` | S1/S2/S3/S4分阶段16/12/8/8条开发轨迹；每条50epoch、原batch40960。逐阶段冻结配置与validation决策；S95 E1两个开发seed训练u的线性q50/q75冻结AIG阈值。S5按数据集独立晋级，严格身份一致的2022/2024完整RAM轨迹明确缓存复用。所有正式模型/epoch身份冻结后统一test。 |
| `analyze.py` | 仅输出原值配对、mean/sample SD、seed级t95%区间、预设增益与保护项、最终规则裁决，不撰写研究解释。开发n=2与正式n=5分开；正式未获准候选记STOP_DEVELOPMENT。 |
| `checks.py`、`tests/test_semantics.py` | 小张量公式/退化/None/显式零/U标准Adam、四shadow不干扰与RAM恢复、开发门与近似平局顺序；真实CUDA Stage0复用既有源模型/Adam/SAD检查并增加本轮AIG和四shadow检查。 |
| `README.md` | 提供review后Stage0、完整有限实验及离线数表重算命令，说明RAM生命周期、缓存与CUDA-event计时边界。 |

参数顺序没有按组重排：SAD与AIG均消费原 `model.parameters()` 顺序的g/d；E/P/U仅作平行标签聚合。参数组按真实 `named_parameters()` 规则生成并逐run写入数值身份文件；真实Stage0直接验证U组所有d为None或精确零。

关闭AIG且不启用SAD时使用普通混合梯度Adam；beta_delta=0为普通混合Adam哨兵。λmin=λmax时AIG保持固定SAD公式；β=.9恒λ的历史代数等价检查沿用有效源检查。当前d=0但已有q不清空，q按固定β衰减；g/d都None不更新state，单侧活动另一侧按零，显式零仍更新。

四个shadow从θ0初始化，每个online step后更新，buffers直接复制；不反馈online训练。所有模型/optimizer/q/shadow/RNG/loader/选择状态仅RAM，通过既有BytesIO/Pipe传输。各自selected和共同E10/E50快照保留RAM，过时selected释放。磁盘输出仅数值CSV/JSON、hash、配置及stdout/stderr。

## 选择、缓存与成本

S1从四基础配置×四shadow选同数据集统一lr/wd；S2从三增强配置×四shadow选正式增强参考，G2始终保留为同κ/.5机制对照。S3和S4均κ=.1、同S1基础配置。近似平局严格小于.0005，按规格固定顺序；选择使用两个开发seed均值，不逐seed调参。

AIG开发门对最强S2 A1+EMA：Δselected validation macro AUC均值≥.001且2/2正，LL/worst/最大域任一任务保护为空。每个候选×shadow保留全部门输入与配对差；只在满足门的选项内选择晋级配置。两个数据集独立决定，名义总预算64/69/74；缓存复用单列并减少实际唯一训练数。cache key包含完整配置、dataset/seed、源码commit/manifest、完整数据身份、RNG语义、epoch/batch/precision/指标定义，且实际比较完整RAM轨迹身份。

完整有限模板及ledger schema在训练前生成；后续lr/wd与阈值是预注册解析项，解析后重新生成实际config hash，recipe hash不会冒充运行配置hash。trial ledger保存PLANNED/COMPLETE/FAILED/CACHE_REUSED、完整超参、源码/数据/raw/split/encoder/scaler身份、epoch/step、初始化/行序/selected模型hash、缓存来源；launch/failure账本保留尝试。

调度复用逐卡GPU资源查询及RAM transport：一张卡自己的显存/功耗均低于85%且剩余显存可容纳原batch峰值时继续并发；首次真实batch分配回执后再调度。OOM只允许原batch降并发重试，失败保留账本，不能缩epoch/batch/seed。保留既有外部GPU进程。

成本分别记录样本曝光、增强曝光和view draw、forward、梯度遍历、一次参数更新、四shadow更新、各模型validation forward/时间/选择次数、test forward/时间、训练worker秒、总wall秒、allocated/reserved峰值和RAM快照tensor bytes。CUDA-event训练区间覆盖单batch输入传输到online/shadow更新，包含host调度间隙；validation/test以每batch纯模型forward区间求和，CPU metric计算另在wall时间中。AIG的日志标量名为 `reference_fixed_mixed_loss`，不把按参数组变化λ的更新伪称为同一个标量目标。

## 实际检查结果

1. `python -m compileall -q experiments/gradient_aug_aig_m_20261101`：通过。
2. `python -m pytest -q experiments/gradient_aug_aig_m_20261101/tests/test_semantics.py`：**3 passed，1.45s**。FP32/FP64动态λ四步手算（含wd、q/v偏差修正和中途d=0）、固定λ连续SAD对照、None/显式零、U与标准Adam逐位一致；关闭/κ0/λ0无view及beta0哨兵；四shadow递推/无干扰、完整RAM恢复下一步、loader/view hash；开发增益/LL保护/双seed正向/.0005平局顺序。
3. 另用临时数值CSV执行完整 `build_tables`：两个dataset×三个正式逻辑臂×五seed×selected/E10/E50，EMA配对差固定+.002，导出差值与裁决符合预期，未晋级AIG输出STOP_DEVELOPMENT；临时目录由标准库自动释放。
4. CPU数值收据位于本目录 `formula_checks.json`、`decision_checks.json`、`checkpoint_restore_checks.json`。唯一warning为历史setup使用的PyTorch TF32设置API弃用提示，沿用冻结环境设置。

下一步是独立reviewer检查本规格、实际代码和本报告。review通过后真实CUDA Stage0使用原batch40960、两数据集seed2022；源检查加本轮AIG共每数据集19次实际optimizer更新，验证U的d、四shadow、真实模型RAM恢复下一步与随机流，不评价test。Stage0生成PASS收据及该源码manifest hash；完整all入口必须同时验证PASS、manifest hash与执行commit一致才启动开发训练。

后续科学分析由协调者在全部授权运行与数表完成后撰写。本轮没有新增依赖、计划外训练臂或动态β/AIG-SWA实现。
