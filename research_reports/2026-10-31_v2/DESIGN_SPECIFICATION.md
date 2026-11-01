# GA31-R3：设计与执行规格

本设计在新实现与正式训练前冻结。执行方案为 `research_plans/26-10-31_v2/` 的 R3-r1，规划来源 `main@a08d1045390f55b3047017ed8e0487d7198c1ea4`；以 `04_实验矩阵与决策树.md`、`05_CODEX实施合同.md` 和 `phase1_protocol.json` 为合同。实际底座冻结为 `ee6e81fdfd37284e6447e8149cb93d04736dba02:experiments/gradient_aug_20261031/`。

用户已修正本轮入口为26-10-31_v2。旧GA31-v2四臂和历史报告只读；本轮ID为GA31-R3。00文件表及05接口段残留的“25–45”是修订前文字；按04、JSON和05任务文本一致的 **35次必要、45次上限** 执行。05是入口，03仅为候选库，不运行C3–C7。

## 1. 责任范围与实现文件

只新增 `experiments/gradient_aug_20261031_v2/`、`research_reports/2026-10-31_v2/`、本轮独立 `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_*` 的代码、日志和数值记录。协调者修正start入口、固定来源并发布。coder负责实现与必要验收，独立reviewer审查后才允许runner运行；coder/runner不作研究结论。

main不包含旧实验包；新结果须能独立发布。将已经固定且身份核对过的必要源文件放入新包的 `vendor_ga31/`，仅调整包内导入并保留来源记录；新模块通过明确命名空间复用该副本，不修改历史源包。不复制旧输出、旧四臂分析或无关测试。新代码复用源 `data.load_dataset`、`metrics.metrics/evaluate`、fixed1主干、field_slices、RAM snapshot/restore/IPC及GPU状态/调度逻辑。新parser不调用旧protocol的40run硬断言；新coordinator不调用旧freeze_selection/evaluate_all的40run断言。不得全局monkey-patch源模块。

| 文件 | 最小职责 |
| --- | --- |
| `source_bridge.py`（可合并入protocol） | 读取冻结源resolved_protocol，继承model/datasets，记录本地文件blob/SHA256及来源commit；导出必要源接口 |
| `views.py`/`model.py` | 源fixed1主干与实际field_slices；显式标准噪声或单字段选择输入；不新增参数、不改变state_dict |
| `objectives.py` | K视图的原样本预算scalar，目标子batch额外forward，正式每batch一次backward/Adam.step |
| `protocol.py` | 新合同解析、SHA256随机键、A/B臂配置、validation基线选择、ANTI门和有效run集合 |
| `run.py` | audit/check/all；单个RAM生命周期执行A→冻结基线→离线probe/ANTI门→B→全体冻结→统一test→数值表 |
| `probe.py` | 有界A-online、离线14梯度/5反事实、统计及只读验收；不训练控制器 |
| `analyze.py` | 已保存数值的seed配对表、门槛判定与报告素材；不读取模型文件 |
| `tests/` | 下述必要语义和调度门验收，不扩展无关工程测试 |

coder拥有 `experiments/gradient_aug_20261031_v2/` 全包（包括确定性配对与决策 `analyze.py`）；协调者拥有 `research_reports/2026-10-31_v2/analyze_results.py`、图表与科学 `REPORT.md`。协调者已写 `SOURCE_FETCH_RECEIPT.json`，设计与实现复用该收据，不覆盖。不新增依赖。基本数据、模型和评价使用源包现有接口；只为本轮真实差异编写代码。

## 2. 固定数据、主干和训练

数据使用源resolved_protocol中的全部raw/六数组/split/encoder/scaler身份、feature顺序和自然人口；仅实际CSV路径重定位为 `/home_new/xzf/datasets/ml-1m/ml-1m.csv`、`/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。ML train/val/test=800167/100021/100021，train域计数155949/318599/325619；KR=2006132/250767/250767，域计数172782/1743898/89452。split_seed2022、仅train拟合编码/scaler、OOV0、全局原始行序80/10/10；KR先切分再过滤tab0/1/4。

fixed1 MDMTRec：3域2任务、4 experts、embedding16、fcn_dims=[512,256,256,64]，四个融合Python标量均为1。保持STAR、LayerNorm、gate输入detach和sigmoid输出[B,2]。ML773470参数、KR1614254参数。无learned fusion/DARTS、BatchNorm/Dropout、新网络或优化器状态。

五个配对seed2022–2026，seed_all后构造同源模型，同seed全部臂初始张量和参数顺序相同。自然无放回shuffle，CPU train/val/test generator分别seed+11000/+33000/+44000；每epoch记录原row_id顺序哈希并配对核验。batch40960、drop_last=false、num_workers0、50epoch、无早停，ML20步/epoch/1000步每run，KR49/2450。FP32，关闭AMP、TF32、scheduler、gradient clipping/accumulation、参数EMA；cudnn.benchmark=false。

一个耦合L2 `torch.optim.Adam`：lr=.01、betas=(.9,.999)、eps=1e-8、weight_decay=1e-5、amsgrad/foreach/fused=false，仅下表A臂显式覆盖。c只乘数据scalar；None梯度保留PyTorch原义，不补零。评价LogLoss始终为真实输入未缩放BCE。

每epoch完整真实validation，六格macro AUC严格`>`选模，平局最早、排除E0。train曲线为更新前batch在线预测，明确标training-online；不与固定checkpoint真实评估混称。

## 3. 正式矩阵、基线和阶段顺序

| 阶段/臂 | 数据 | c | weight_decay | eps | 视图 | 五seed run/真实Adam步 |
| --- | --- | --- | --- | --- | --- | --- |
| A B-ML | ML | .1 | 1e-5 | 1e-8 | 无 | 5/5000 |
| A L2-ML | ML | 1 | 1e-4 | 1e-8 | 无 | 5/5000 |
| A B-KR | KR | 1 | 1e-5 | 1e-8 | 无 | 5/12250 |
| B M1 | ML及KR | 冻结基线 | 冻结基线 | 冻结基线 | 单字段零向量，K1 | 10/17250 |
| B G-IID2 | ML及KR | 冻结基线 | 冻结基线 | 冻结基线 | 独立高斯K2 | 10/17250 |
| B G-ANTI2 | 每个通过方差门的数据集 | 冻结基线 | 冻结基线 | 冻结基线 | ξ和−ξ，K2 | 0/5/10，最多17250步 |

必要35run/56750步；两数据集均触发ANTI时45run/74000步。只ML触发时40run/61750步，只KR触发时40run/69000步。G-IID2与M1必做，不受任何探针性能或ANTI门影响。B臂从相同seed初始模型重新训练，不从A selected续训。

A15run全部完成后，ML按五seed各自selected的validation配对Δ=L2-ML−B-ML判断：若|meanΔ|≤.0005、至少4/5的|Δ|≤.001，且validation保护项无触发，优先L2-ML；否则按selected validation macro AUC五seed均值最大，差≤1e-6平局取L2-ML。KR固定B-KR。保护项为最大训练域任一任务、pooled任务均值、worst-cell AUC的mean≤−.002且≥4/5下降，或macro LL mean≥+.002且≥4/5恶化。写 `baseline_frozen.json`，含全部seed值、保护项、optimizer参数和理由。test、在线时间信号及小panel单步效用均不参与基线选择。

随后只在最终所选普通基线的E1/E10/selected状态执行离线probe；冻结ANTI门和B任务集合到 `candidate_frozen.json` 后运行B。A非工作臂的完整指标仍保留。所有A/B有效run完成后冻结selected和共同E10/E50；统一test不得提前。

## 4. 视图、目标与随机流

令a_i=1[d_i=d*]，d*=ML0/KR2。每个原样本两个任务均可用。基线Lr=mean_i,t BCE_real；视图scalar为

`L_K = (sum(real_BCE) + .5*(sum_v sum(target_aug_BCE_v)/K - sum(target_real_BCE)))/(B*T)`。

每个原样本权重和1，不改变原始采样/标签/人口。完整真实batch一次forward、目标子batchK次forward、合成scalar一次backward/step；缺目标跳过视图。λ0/关闭增强直接真实路径，高斯κ0和显式恒等视图精确折叠以免重复forward的FP32求和差影响验收。

沿实际sparse-then-dense field_slices。ML eligible6字段，KR继承完整33字段；domain Embedding、dense、ID、路由、标签、row_id不变。M1逐目标行独立均匀抽一个eligible字段，将该16维段直接置零，不用padding ID，不反缩放其余字段。Gaussian每field使用clean Embedding的 `e+.1*detach(RMS(e))*ξ`，RMS无floor，Embedding本身保留梯度。IID2的ξ1/ξ2独立；ANTI2共用ξ1，第二视图严格−ξ1，两个view各自前向，不平均输入。

`derive_seed(key)=int.from_bytes(sha256(key.encode('utf-8')).digest()[:8],'little') & ((1<<63)-1)`。训练key=`GA31-R3|dataset|seed|train|global_step|family|view`，family=gaussian/fieldmask，view从1开始，step从1开始。IID2与ANTI2第一高斯key相同；ANTI第二不另抽随机。离线key=`GA31-R3|dataset|seed|probe|anchor_step|batch_index|family|view`。时间key=`GA31-R3|dataset|seed|temporal|epoch|original_batch_index|gaussian|view`。各用独立设备Generator，不消耗正式全局/loader RNG，记录key/seed和噪声或字段选择哈希。

## 5. A-online：真实轨迹上的有限只读诊断

A全部15run的ML E10/E20、KR E10/E40，每窗口原始batch1–4，在该batch正式Adam.step前观察参数；推荐在 `opt.zero_grad(set_to_none=True)` 后、正式loss/backward前调用，避免污染正式`.grad`。固定训练参考B_ref为独立seed314159 permutation前1024行，跨所有seed/臂/窗口/步相同，记录row_id哈希。

同B_ref、同B_ref*T分母计算target真实梯度及两份独立Gaussian target梯度，共3次autograd.grad。δ_tv=g_target_aug_v−g_target_real。保存两份δ范数、均值范数、条件波动、带符号S_t=dot(δ_t1,δ_t2)、相邻步C=dot(meanδ_prev,meanδ_current)、epoch/原batch/global_step/target数/键及参考行身份。跨步C只在同一预注册窗口相邻观测计算；窗口末释放暂存向量。S/C负值原样保留；零范数或负信号的归一化量为null+原因。

使用独立图autograd.grad，不额外optimizer.step、不写`.grad`。每次核对参数/buffers、Adam、已有`.grad`、model.training、所有捕获的Python/NumPy/Torch CPU/CUDA RNG及loader状态执行前后相同；诊断中若构造临时模型消耗RNG则隔离恢复。总计最多120观测/360额外梯度/0额外训练step。只输出数值和哈希，不写梯度大数组。全部A臂保留记录，主要时间解释只使用最后冻结工作基线。

## 6. A-probe：固定参数的多视图与实际Adam比较

所选A基线每数据集5seed，在E1/E10/selected去重状态、2个固定1024训练batch执行。固定行来自seed314159 permutation前2048行，前1024与online参考一致；最多60锚点-batch。缺目标记unavailable，不改为均衡采样。

每锚点-batch固定θ/Adam历史：1次真实full梯度gr，1次真实target梯度gt，4个独立正高斯、对应4个负高斯、4个独立FieldMask target梯度，共14次autograd.grad，最大840次。全部target项除原B*T；δ_v=gt_aug−gt。raw梯度排除c/decay。按任务只记clean/aug BCE和预测差，不能为任务诊断额外突破14次梯度预算。

统计使用完整参数向量的CPU FP64点积/范数，可直接复用源向量函数；None只在统计拼接中作零。独立正高斯组和MASK组分别报告meanδ范数、V=sum||δ−meanδ||²/(K−1)、S=||meanδ||²−V/K、两个独立半组均值cosine、目标数、各任务BCE/预测变化。S保留负值。相关负伙伴组单独计算V_minus与cross_cov=sum dot(δplus−meanplus,δminus−meanminus)/(K−1)，不得作为IID样本混入独立S/V。

Vanti=(Vplus+Vminus+2cross_cov)/4，Viid=(Vplus+Vminus)/4；以4个配对均值的直接样本方差交叉核验Vanti。方差负到−1e-12以下为实现错误，微小负舍入截0且记录；cross_cov和S不截。Viid≤1e-24*max(1,||gr||²)则该观察NA。ANTI门只用E10/selected去重观察：每seed先各自平均有效Vanti和Viid，再取R_s=meanVanti/meanViid；每seed至少一个有效锚点，必须五seed均有效。跨seed meanR≤.8且≥4/5 R<1才触发该数据集五次G-ANTI2。E1不入门。记录有效观察、每seedR和门结论，不据test或单步BCE改变门。

每观察最多5个独立联合deepcopy((model,optimizer))分支：real=gr，G-single=gr+.5δ1，G-IID2=gr+.25(δ1+δ2)，G-ANTI2=gr+.25(δ1+δminus1)，M-single=gr+.5δmask1。每支提交c*g给真实源配置Adam，耦合L2只在optimizer内加入一次。保留实际计算图连接性的None/非None；不能把全参数写零。每支一步，最多300次反事实step。记录实际更新norm/cosine、与real更新差、m/v变化、per-param step及源身份。

两真实validation panel按域固定permutation种子271828+100*d，每域n=min(1024,floor(Nval_d/2))，panel1取前n、panel2下一n，互不重叠。分别保存更新前/后六格BCE、六格macro及按完整validation人口比例加权BCE；不把panel内均衡人口当全validation流量权重。效用仅作诊断，不控制M1/IID2/ANTI门。

probe读取CPU快照后恢复源model+Adam到设备，再每分支联合deepcopy，核对clone参数与optimizer绑定。保存/恢复全局RNG、训练模式；源快照及loader/RNG执行前后相等。向量按观察释放，不落盘；在线/离线/正式训练计算分别计数。

## 7. 优化器诊断、RAM及并发

A阶段增加实际优化器尺度诊断：ω||θ_pre||/(c||g_data||)，以及ε/(sqrt(v_hat)+ε)的分布和按实际参数更新平方能量加权摘要。可仅在每epoch最后一个真实update测量并注明时点：backward后记录pre参数和数据梯度，step后使用实际v/step bias correction及实际参数差。零分母写null/原因。不依据近零坐标数量推断epsilon主导，不额外计算训练梯度或优化器step。

一个长期存活父coordinator保留所有A/B CPU model+Adam+RNG/loader快照，E1/E10/E50/selected去重；过时非固定selected释放。父进程不初始化CUDA，Linux fork worker继承只读数据；BytesIO/Pipe传状态，不触发模型共享文件或临时checkpoint。无`.pt/.pth/.pkl`及改扩展名权重文件。RAM状态在全部test和数值表生成后释放并写释放记录。冷启动all无法恢复旧内存；进程丢状态需从同seed完整重跑受影响run并保留失败attempt。

复用逐卡显存/功耗调度：各卡两项均低于85%时增加worker，启动后等首次原batch分配完成再读新状态；现有外部作业不停止。保留真实数据和快照所需CPU RAM。真实OOM先减少并发、同batch重试；不能改batch、seed或50epoch。单独运行仍OOM等具体阻塞写ledger。阶段间复用调度器时progress应注明stage与累计计数，不能将单阶段完成误写全实验完成。

## 8. 必要验收与运行接口

Python统一使用 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。入口 `python -m experiments.gradient_aug_20261031_v2.run --mode audit|check|all --protocol research_plans/26-10-31_v2/phase1_protocol.json --run-dir <new-attempt> --devices 0,1`；all不分裂RAM生命周期。源和数据身份、实际硬件/软件版本记录后，reviewer PASS或PASS WITH FIXES且blocking修复才正式all。

只做影响当前比较的检查：

1. 同初始化/同batch源fixed1与新无增强模型的预测、Sample loss、梯度、实际Adam一步；参数量/键/顺序/特征slice和数据全部身份匹配。
2. 手算B*T与原样本系数和1；K视图恒等、κ0、λ0/关闭MASK回真实；缺域/缺目标/batch1；domain/dense/标签/ID不变，MASK整段置零且无逆缩放。
3. IID/ANTI第一noise严格相同、ANTI第二互为负，clean RMS一致；key确定性及全局/loader RNG独立。
4. coupled-Adam完整尺度匹配控制：c=.1/ω1e-5/eps1e-8与c1/ω1e-4/eps1e-7，m'=m/c、v'=v/c²，step及None一致，同步下一步参数在CPU FP32 atol1e-6/rtol1e-5容差内一致。
5. online临时epoch1四步on/off同seed同序列比较真实model/grad/Adam/RNG/loader；离线真实状态联合copy只读及5分支绑定正确；RAM epoch边界恢复下一步一致。工程短run仅seed2022、每数据集总计至多1epoch，无test、不按效果调参数。
6. 合成门测试明确覆盖L2实用替代优先、validation保护、均值/平局规则、R的先平均后取比、NA与缺seed、E10/selected去重、IID2无条件执行和0/1/2数据触发后的35/40/45run/精确更新预算。
7. 原batch40960单步CUDA容纳；统一test只能接受all生命周期已冻结的有效run集合；selected step来自实际selected快照，50epoch/曝光/顺序一致。

发现能改变训练或结论的Critical问题修复后重跑受影响run；Relevant仅最小核验，无关工程问题不扩展矩阵。未执行的候选不以假数据填充。

## 9. 评价、分析、工件及完成条件

全体selected/共同E10/E50清单和实际run集合冻结后写 `selection_frozen.json`，再一次统一test。重合epoch复用模型评估；每run仍输出selected/E10/E50三个语义行。selected身份包含dataset/arm/seed/attempt/epoch/global_step、source和validation值。

逐seed、六格AUC/真实LogLoss、每域两任务均值、每任务三域均值、macro、worst-cell、traffic加权cell、真正pooled逐任务跨域AUC再均值、总体LL全部保留。配对表含L2-ML−B-ML、M1−冻结基线、IID2−冻结基线、触发时ANTI2−冻结基线与ANTI2−IID2，三个checkpoint和validation/test均报告；均值、样本SD、正负数、df4描述性配对t95%区间以seed为单位。

研究继续门：selected Δmacro AUC≥.001且≥4/5正向；validation/test方向分别列，保护项同第3节。ANTI独有价值须同时超过普通基线和同K IID2；只对基线有效保留相应条件方法。触发保护记TRADEOFF并保留实际宏收益；只E50收益不称selected GO。未触发ANTI明确NOT_TRIGGERED，不当算法阴性。B-IID/M1不因中途seed表现删臂。

保存配置/来源/数据manifest、split行号、init/order/view/state摘要、launch/run/failure ledger、epoch/per-cell/view/runtime、optimizer-scale、increment_probe/temporal_probe/temporal_summary/counterfactual_update、baseline_frozen/candidate_frozen/selection_frozen、paired_effects、DECISIONS和REPORT。source_manifest记录固定源commit与blob及规划commit；不保存模型状态。日志含所有超参、actual optimizer参数、forward签名、实际模型模块和硬件软件。

REPORT由主协调者分析：算法是否有效、种子与阶段稳定性、AUC/LL/域任务交换、基线难度与域规模是否能解释结果；明确STRONG/MEDIUM/WEAK、矛盾、竞争解释、最有区分力的后续证伪实验。成本分正式Adam/额外视图/在线360以内梯度/离线840以内梯度与300以内反事实step，报告曝光/forward/backward/autograd/显存/worker秒/总时长。时间信号用于未来C4投入判断，不自动训练动量。

完成为规定35–45run各50epoch、全部有界探针、冻结后统一test、完整分析与轻量工件、独立审查及新结果分支发布。实际触发前不能声称必做45run；具体数据身份/单卡OOM/不可修复非有限阻塞则写事实和影响。完成本轮后停止，不自动扩展C3–C7、mask率/高斯强度网格、额外seed或共享控制。
