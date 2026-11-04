# DESIGN SPECIFICATION — KuaiRand Gaussian领域覆盖

设计状态：COMPLETE。日期：2026-11-04。本规范落实 `start.md`、`buchong.md` 已定义的实验，不增加研究配置。主agent已批准本合同后进入实现。

## 1. 范围与冻结来源

- 代码基准：`research-plan/26-11-03-overnight-ab-auc`，commit `9e682f3e3f608c7d63c2e9306b8657b16d822541`；已阅读main `652ee0235fec2d0d4cfc67b4a6ef9977519619bf` 的上下文、全局摘要和指标台账，以及11/03五seed补全报告、11/02v2规划和history源码。
- 用户入口位于历史工作目录 `/home/xzf/Code/phase0-3/`，两文件编码均为GB18030。`start.md` SHA256=`9bd35b6ef18a800d100f4c0434c97539c7b8d28e01e93b74b5df430272d7e84d`；`buchong.md` SHA256=`b9f802dda819d589b153faedc4ab3167b531c3b68485af6e7547ef58841ae7ac`。
- 新代码只写 `experiments/gaussian_domain_coverage_20261104/`；报告只写本目录；台账新增独立增量，不覆盖旧代码、旧报告或并行实验。
- 分支：`research/26-11-04-gaussian-domain-coverage`，完整结果提交推送，main不自动合并。
- 唯一算法因素：Gaussian增强的领域集合。自然Sample、Gaussian机制、模型、数据、Adam和固定EMA均保持来源设置。不搜索参数、不新增数据集、不重训G-D2或发表模型、不按早期seed性能停止。

| 配置 | target_domains | seeds | epoch | 动作 |
|---|---|---|---:|---|
| G-D2 | [2] | 2022,2023,2024,2025,2026 | 各50 | 复用已有端点 |
| G-D0 | [0] | 2022,2023,2024,2025,2026 | 各50 | 新训练5条 |
| G-D02 | [0,2] | 2022,2023,2024,2025,2026 | 各50 | 新训练5条 |

新增预算恰为10条逻辑轨迹、500epoch、24500次Adam更新。恢复不增加独立seed；只有同源码/数据/优化/评价身份已有合法结果时才减少物理训练。不能新增G-D2正式轨迹来验证实现。

## 2. 数据、模型与优化不变量

数据协议 `FULLTRAIN_NATURAL80_10_10`：原CSV全局原行序80/10/10切分，KuaiRand再筛tab0/1/4，预处理只拟合train，不重采样、不重编码、不按规模排序。

| 模型域 | 原tab | train行数 |
|---|---|---:|
| d0 | 0 | 172782 |
| d1 | 1 | 1743898 |
| d2 | 4 | 89452 |

train=2006132行，validation=test=250767行；batch40960，每epoch49步，尾批保留，50epoch2450步。任务0/1固定为is_click/long_view。d1最大；规模不替代难度定义。

CSV绝对路径 `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`，raw SHA256=`a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。绑定六数组、row_ids、encoder/scaler等全部来源身份，不能只比行数。

可只读复用缓存 `/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache/kuairand/`；其 `manifest.json` SHA256=`33a9b0f2866a0c2440481df4bd3315b041b4f85042a06c6a4652972be44fb08d`。prepare核对原CSV、cache文件哈希及manifest中的历史数组/划分身份；训练view不含test数组。缓存只有数据，不包含模型状态。

完整来源身份在已有vendored `vendor_ga31/resolved_protocol.json`。其中train_x/y哈希分别为 `ad562a0708316156dff5734a67fc0512ba1831c48c2436e9dee633c41288fa1a` / `48a4db0faa150b89fb84634c4326deecef36178993d414916aff3507a7b4d024`；encoder=`58ff742e14eabc33ed001ad6ec2e4f74c6be375ab4b2a68028277b7b9bb3e33a`，scaler=`44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`。

模型为来源fixed1 M3oE/MDMTRec，1614254个参数，embedding16、4共享专家、FCN[512,256,256,64]、融合系数1，无DARTS/DBO。直接复用来源factory与训练gate detach。33个非路由sparse字段允许扰动，domain_indicator、离散ID、标签均不改。

Adam lr=.01、coupled weight_decay=1e-5、betas=(.9,.999)、eps=1e-8，foreach/fused/amsgrad=False，全部参数恰覆盖一次。用普通 `torch.optim.Adam`，不添加自定义历史优化器。FP32，AMP/TF32关闭，源setup(seed)及线程/初始化语义保留。

固定EMA rho=.995，初始精确复制theta0、requires_grad=False；每batch在唯一Adam step后更新一次，buffer从online复制，无bias correction，不反馈Gaussian构造。loader单独 `torch.Generator().manual_seed(seed+11000)`，每epoch来源torch.randperm，原始batch行序与row_ids记录。

## 3. 源码选择与Gaussian随机流裁决

历史复用的Gaussian不是后来EMA02V2 row-keyed A1重训结果。它来自GAM01/AIGM的GA31-R3源路径。为了只改变覆盖集合，必须保留历史Gaussian抽样方式。

现有main依赖已包含字节相同的原source，无需复制顶层旧实验包：

`experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/`

| 相对该目录文件 | SHA256 | 责任 |
|---|---|---|
| objectives.py | `28eaa19444cb650ca46d28c4e22829771612920add6c27188b828d934ea5479c` | 原单域Sample损失 |
| views.py | `cc83584fa43a57ffd8c9c176e14a668a5dc835e902086e378a8db15e2a75f44b` | draw_view/apply_view/make_model |
| protocol.py | `1f6e345cf50d417285fde8100c1c99bf4ef3a71bddf4e256c3b3640186247ccd` | 原view_key/derive_seed |
| vendor_ga31/ | 执行manifest逐文件登记 | 原模型、特征、setup、数据身份和指标 |

上述前三文件已与本地顶层 `experiments/gradient_aug_20261031_v2/` 对应文件逐字节哈希核对一致。原G-D2源码保持只读。

原key：`GA31-R3|kuairand|{seed}|train|{global_step}|gaussian|1`。SHA256前8字节little-endian并与63bit掩码生成本地CUDA Generator种子；每个字段依次draw `torch.randn([n_domain,16])`，堆为 `[n_domain,33,16]`。不消耗训练全局随机流。

新增多域路径对每个目标域分别调用原 `draw_view`，每次使用同一个原key及该域行数。这样D0在G-D0/G-D02中的相同行序具有相同噪声，D2在G-D02中具有历史G-D2的同一步噪声。保留域内来源batch顺序，不能先拼接两个域再抽一个不同shape的噪声，也不能额外加入arm/domain/study key。两个域的局部生成器复用原种子，域间噪声前缀的对应关系随之保留并记录。

原噪声没有row_id键；row_ids用于数据身份及日志，不为本轮更换成EMA02V2的SplitMix64/Box-Muller规则。用户要求尽可能沿用确定性流，本合同以保持已复用G-D2来源规则为准。

## 4. 完整target_domains数据流与损失

接口可保留 `target_domain` 单值兼容入口，再归一化为唯一、固定顺序的tuple/list。正式配置仅[0]和[0,2]，单域[2]仅用于工程等价检查。配置身份显式包含target_domains。

单域时直接调用上述未修改 `source_losses`，给原target整数及 `num_views=1,view_family='gaussian',pairing='iid',mix_lambda=.5,kappa=.1`。这保留原mask、下游接口、算术路径与噪声。

双域时：

1. 原输入x为[field,B]、y为[B,2]。online model(x)只做一次，得到原概率并计算逐元素BCE `real`。
2. 对d按固定[0,2]遍历，独立mask `x[-1]==d`。空mask不抽噪声、不forward；非空mask用该域原始x/y生成一个Gaussian视图，两个任务都计算BCE。
3. 扰动保持原 `apply_view`：`h + kappa*sqrt(mean(h**2,field_dim)).detach()*noise`；只在perturb_slices改变embedding。RMS尺度detach，原embedding主路径保留梯度，增强不新增高阶求导或教师probe。
4. 累加两个域的增强BCE总和与对应real总和之差，统一除以 `real.numel()==2*B`：

   `loss = (real.sum() + .5*sum_d(aug_d.sum()-real[mask_d].sum())) / real.numel()`。

选中行每个任务clean/aug权重各.5/(2B)，未选中行1/(2B)。不能对域均值等权相加，不能增加域采样、对两个域分别optimizer.step，不能让d1产生增强视图。

lambda=0/kappa=0/无目标行时直接原A0标量路径，不抽增强随机数、不做额外forward。尾批分母使用实际B。训练只调用一次backward、一次Adam.step、一次EMA更新，global_step只增1。返回clean概率/real_loss、混合loss、每域target/extra-view行数与更新计数，保留完整标量日志。

## 5. 最小实现结构与运行

新增包可只设 `__init__.py`、`objectives.py`、`run.py` 和 `tests/`，按实际需要拆出worker/reporting；不新增公共框架。复用 `overnight_ab_20261103` 的state、selection、evaluation、data_bridge、queue资源函数与guardian通信；若原CLI硬编码旧实验，仅在新包加薄适配，旧文件不改。

Session遵循现有create/advance/export/restore责任：固定job/config/data/source身份，online、EMA、普通Adam、loader、Python/NumPy/CPU/CUDA RNG、global_step/epoch、selected/E10/E50状态、累计成本。新worker可沿用AB worker_b的epoch交易结构，但其optimizer和train_step必须指向本规范，不能进入C/P/HistoryOptimizer分支。

最低CLI：`prepare`、`smoke`、`run --devices 0,1`、`status`、`summarize`（或report）；guardian内部入口按实际需要提供。全部项目Python均用：

`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gaussian_domain_coverage_20261104.run <command>`。

prepare冻结10个job、全部超参和target集合、数据/源码/复用文件身份、评价协议及裁定阈值。RUN前独立review PASS或已完成blocking修复的PASS WITH FIXES，并绑定实际实现及依赖源码hash。

每张GPU独立在显存占比<85%且功耗占比<85%时派发；另满足已测峰值×1.2空闲量和RAM预留max(6GiB,total10%)。满足条件时继续并发，不等待两卡同时空闲、不设固定每卡1worker上限。真实batch smoke测得峰值后逐个加进程，使用原有owner锁防重复派发。现有其他GPU任务保持运行，不能终止或抢占；本任务已有共享授权按当前owner身份记录，外部明确设备预留继续尊重。

每完整epoch以authenticated RAM guardian提交完整状态后ACK；持久化只写配置、数据缓存、标量、日志和报告。模型/optimizer/EMA/RNG tensor不能写磁盘checkpoint、pickle或参数memmap；BytesIO IPC保留。资源不足等待，不降batch、不减epoch。worker死亡优先从最后RAM epoch恢复；评价失败只重试评价；主机/guardian丢失RAM才同身份重放，成本记录失败物理工作。已封存合法DONE结果不重跑。

## 6. 十项必要工程验收

只运行以下影响本次算法或评价的检查，不增加研究训练臂。coder可做这些语义/basic检查；reviewer独立读实现及coder报告。

1. **覆盖**：混合域小batch中G-D0只有d0、G-D02只有d0/d2产生视图；每域行数及row_ids匹配，d1额外视图数0。
2. **双任务**：两任务均进入源概率BCE，分别改变任一任务标签都会相应改变增强项；输出shape与路由保留，domain_indicator及非扰动字段不变。
3. **原预算**：非均衡域数量和尾批例子，直接按所有原始行展开的0.5clean+0.5aug目标与实现值/梯度一致，不按目标域数量改分母。
4. **关闭退化**：lambda=0、kappa=0、全目标域空时与原A0 loss/grad一致，无额外增强forward/RNG。
5. **单域兼容**：[0]及[2]与原source_losses对应target的loss/参数梯度、噪声key/shape一致；[2]不运行正式50epoch。
6. **随机流**：D0单域/双域中的d0噪声一致，D2源/双域中的d2噪声一致；日志/validation不消耗训练或loader RNG。不得用新row-keyed算法代替历史draw_view。
7. **更新与detach**：每batch仅一次backward/Adam/EMA/global_step；RMS扰动尺度detach，原embedding仍有监督梯度，teacher无grad，Adam decay只施加一次。
8. **RAM恢复**：同seed连续两步（或两epoch）与一步→export/restore→一步的online/EMA/Adam/loader/RNG/global_step/selection一致；跨设备恢复使用AB已有active CUDA RNG映射。
9. **选点及指标**：六格validation AUC严格最大、精确并列最早、epoch1–50；LL和test不参与。V/Test相同model hash；Macro/Traffic/Pooled分开，未缩放LL、两个任务分别pooled再平均。
10. **真实原batch smoke与完成门**：KuaiRand batch40960对[0]和[0,2]做有限forward/backward/EMA，loss/grad有限、RAM export/restore可用，记录显存；smoke后正式seed重新初始化。确认精确10job×50epoch，源码审查hash生效、状态不落盘、六格/endpoint完整。smoke性能不用于筛选。

## 7. 评价与历史端点身份

继承最新AB `AUC_SELECTED_V1`：每seed固定EMA independently按validation六格等权Macro AUC最大选checkpoint，精确并列取最早。online视图可同步记录作为诊断，不增加轨迹n。selected为主裁定，E10/E50预定义阶段诊断保留。所有配置在训练前冻结；每条轨迹完整50epoch后，先封存选点及model hash，再评价test并写endpoint checksum，之后释放该job RAM；符合最新AB逐seed冻结/评价协议。

历史统一输入：`research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv`，本地SHA256=`eb260009b1172f3fb4ed25a12cc1836c06a1ffb24bf583cea1e7e08de4943fce`。原FINAL_REPORT SHA256=`592b9ee82d30ac1a1dda398bea10dee9bce16b0a87f50cc24424308d1ca3bcab`。

按dataset=kuairand、split=test/validation、checkpoint=selected/E10/E50及以下实际身份导入，不能按显示别名直接过滤：

| 本轮标签 | AB arm | estimator | n | selected epochs，2022→2026 |
|---|---|---|---:|---|
| G-D2 | GAUSSIAN_A1 | fixed_ema | 5 | 28,22,36,47,36 |
| 固定EMA内部控制 | A0 | fixed_ema | 5 | 33,50,50,50,47 |
| M3oE正式模型 | A0 | online | 5 | 44,18,40,50,43 |
| HyperGate正式模型 | HYPERGATE | online | 5 | 21,10,43,28,30 |
| AutoIFS正式模型 | AUTOIFS | online | 5 | 6,32,2,6,39 |

每个导入端点保留原source/trajectory/selection/model/endpoint身份、原路径及文件SHA；真实新训练使用新的身份，不伪装成源hash。

G-D2/A0源端点文件：历史根目录下 `artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720/test_metrics.csv`，SHA256=`eccb79a4aad2952ceb6dfdd80f49789a1e313d2f69d1999529271b306f1b5a80`。其 `formal_configuration_frozen.json` SHA256=`86ef0e2313887ccd91aa712b81d6585e4123fc7f8ece80720695de1c3a1b5fbd`，`data_manifest.json` SHA256=`b7758a878467d73f83d8aa05b35a5a22435bad65de1e5b5dc560469501e6b68a`。配置身份明确KuaiRand A1使用augmentation、普通Adam、kappa.1/lambda.5、ema995，原code_hash=`d297bea130cae8fedaeef47ae35239fd9af4f4d7f3076416eaf50670bcdae02c`。

G-D2五seed selected model hash：2022=`e1da2fa7a28d3a916ffb9c35e9acf003c58c7e8cb96bfeb96ffa253612b43113`；2023=`c348148846c7ef638e493622ca91c71cc5c34553dfc50f104bafe6eee2ef81f4`；2024=`3eec5aa5a12c2300a1ced1adcb642658d4370dd48a9704c0456d5ca9d98f31d7`；2025=`f9695d2a655ae67a87318890b81f6bfa980c4cb764196ecf0bf99d39aea4c685`；2026=`2e8b55e9eb3d4ee1a19468fb391b0e9c01f77a9df6b2585a905252eab5e90b3a`。

HG/AutoIFS真实原始路径为 `artifacts/mdmt_overnight_ab_20261103/attempt1/runs/kuairand.{HYPERGATE|AUTOIFS}.s{seed}/selection_frozen.json` 及同目录端点，沿AB已有审计接入。旧10/29的T0/F训练与LL选点结果不导入。复用审计只核对本次实际输入与既有证据，不重做历史研究。

## 8. 汇总、裁定与交付

全部配置每seed导出六格AUC/LogLoss、各域/任务、Macro、Traffic（sample_weighted_cell）、Pooled（pooled_task_mean_auc/pooled_logloss）、count/positive、选中epoch和同hash V/Test端点。五seed统计用样本SD(ddof1)；配对CI为mean±2.7764451051977987*SD/sqrt(5)，seed为唯一独立单位。

主比较固定EMA selected：G-D0−G-D2、G-D02−G-D2、G-D02−G-D0，以及每个Gaussian分别减A0 fixedEMA、M3oE、HG、AutoIFS。同源M3/A0 online只计一次。每项保留五seed原值、差值、均值/SD、CI及方向数。阶段比较按同一E10/E50，不以test择优。

正式baseline只有M3oE、HG、AutoIFS；固定EMA是内部控制。最强发表模型按既有五seed平均Test Macro AUC为HG约.682113。Gaussian对最强者的五seed平均ΔAUC：>=.003强算法；>=.002候选算法；>0且<.002正向胜出；<=0如实记录。阈值用完整精度，正seed数及CI不增性能门。

六格保护独立裁定，每个比较各域任务五seed平均ΔAUC>=−.001；宏收益等级与保护结果分列。LogLoss以meanΔLL<0独立报告，不替代AUC。

必须产物：`REPORT.md`完整中文分析、`ENDPOINTS.csv`、`ABSOLUTE_SUMMARY.csv`、`PAIRED_SEED.csv`、`PAIRED_SUMMARY.csv`、`DECISIONS.json`、`RUN_MANIFEST.json`、`COSTS.csv`、validation曲线/训练标量、源码manifest、coder实施报告、独立review报告/receipt、runner原始运行报告、台账增量。原始运行记录归档须可复算且只包含数据/标量/日志，不归档模型权重或guardian认证密钥。

成本按10条逻辑轨迹与物理失败/恢复分别记，复用成本明确为历史；每条新轨迹应有2450原batch更新和100306600原行曝光，G-D0增强行曝光8639100，G-D02增强行曝光13111700。online/EMA、E10/E50和重复评价不增加n或重复计训练成本。

主agent在全部10条新轨迹和评价完成后分析：回答双域是否优于D2/D0，d0两任务相对HG缺口变化，未增强d1两任务影响，三发表模型中最佳者是否被超过；区分完整方案收益、相对固定EMA组件增量、域/任务及AUC/LL取舍。分析seed一致性、阶段、基线表现、规模与难度，标记STRONG/MEDIUM/WEAK；结合反例给出竞争解释，不把域大小直接写成因果。只有结果实际改变下一步选择才提出一个最能区分解释的后续实验，本轮不追加。

完成条件是全部50epoch/端点/配对/审查与推送齐备；GitHub报告提交包含具体分支及SHA，不以计划或工程检查替代正式训练。
