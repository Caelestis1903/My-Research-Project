# DESIGN SPECIFICATION — EMA 收益机制研究

设计审核结论：**APPROVED FOR IMPLEMENTATION**。冻结计划可由现有纯组件实现，未发现需要改变科学矩阵的矛盾。本文补齐两个实现者的共享接口；数值由 `research_plans/26-11-6/experiment_manifest.json` 驱动，数学、评价和生命周期分别以该目录 03、06、05 为准。

设计日期：2026-11-06。源码基底 `40705dfe0bb4fe614bdca835c0bae567348c87d0`；计划来源 `9fb59d5a53bf0efa555caeaca03e1ffa3812b686`。独立工作区 `/home/xzf/Code/phase0-3_ema_mechanism_20261106`，交付新分支 `26-11-06_v2`。主代理负责统筹和科学判断，独立 coder 实现，另一 reviewer 审查通过后 runner 才运行。最新五 seed、域规模/难度排除及新分支要求覆盖旧默认。

## 1. 科学责任与完整范围

研究 H1 历史组合/有效滞后与排序时间变化，H2 六格/时间尺度/路径偏好，H3 train-only 指标对独立未来 pulse 收益的描述与预测，H4 Gaussian 对 EMA 收益结构的交互。Q1–Q5 按计划 02/manifest 对应关系作答；不设计新自适应 EMA，不修改 online、Adam 或 Gaussian，不重跑 A/B/C、发表模型或旧性能确认实验。

L1 已有经验收益直接继承，L2 分布、L3 受控只读干预、L4 预测价值分开形成结论。seed 为唯一独立科学单位；全部结论标 STRONG/MEDIUM/WEAK，并说明具体效应、条件、重复现象、冲突证据及竞争解释。域身份用于六格分解；本轮不设置域规模/难度解释或相关实验。

有限矩阵：MovieLens、KuaiRand × clean、gaussian_d0 × seeds `[2022,2023,2024,2025,2026]`，20 条逻辑轨迹。每条50 epoch，无 early stop；ML 每epoch20步、KR49步，共1000 epoch/34500 Adam。观测一次采齐 S1–S3；S4 无新增训练。已有同轨迹重放不增加科学seed。

## 2. 不可改变的训练与数据语义

- 自然人口 `FULLTRAIN_NATURAL80_10_10`，源原始行序 split、train-only编码/缩放，KR先split再筛tab0/1/4。ML train/validation/test=800167/100021/100021；KR=2006132/250767/250767。
- batch40960，尾批ML21927、KR40052；fixed1 M3oE，embedding16、FCN[512,256,256,64]、4 shared experts，参数ML773470、KR1614254。
- 源 `history.setup(seed,device)` 初始化，独立 CPU loader generator `seed+11000`，每epoch一次原 `randperm`；训练 gate 输入保留 detach。
- `torch.optim.Adam` lr=.01、betas=(.9,.999)、eps=1e-8、coupled wd ML1e-4/KR1e-5，foreach/fused/amsgrad=False，无 AMP/TF32/clipping。
- 必须直接调用冻结 source `objectives.losses`。clean `num_views=0` 保留 `real.mean(1).mean()` 归约顺序；G-D0 `num_views=1`、target0、lambda=.5、kappa ML.05/KR.1，实际尾批分母2N，replacement loss，两个任务共享同一view。
- Gaussian detached字段embedding RMS，只扰动源非routing sparse字段，噪声key保留 `GA31-R3|dataset|seed|train|step|gaussian|1`。observer/panel/H 不进入训练RNG身份。
- 训练/分析 bundle 始终 `include_test=False`。preflight 可核验Test文件字节hash与封存元数据，不向训练/分析暴露Test标签/指标。

## 3. 复用边界与实际入口

直接引用已核对的以下组件，保留原源码：

| 组件 | 必须复用的语义 |
|---|---|
| `aug_history_20261105/model_adapter.py` | `build_model(bundle,config)`，`ModelAdapter.forward_logits(...,params=...,probe_input_full_graph=...)`，raw tower路由 |
| `aug_history_20261105/registry.py` | `ParameterLayout` 参数名/shape/offset/alias/dtype身份、复制；其 `.flatten()` detach，不用于功能导数图 |
| `aug_history_20261105/group_registry.py` | `build_path_groups`；domain embedding按行；12组及计数hash |
| 冻结 vendor `objectives.py/views.py` | clean/Gaussian目标与随机语义 |
| `overnight_ab_20261103/data_bridge.py` | 原 `prepare_cache`、`cached_bundle(include_test=False)` |
| `overnight_ab_20261103/evaluation.py/selection.py/state.py` | 源概率指标、六格严格选点、RAM深拷贝/hash/资源计数 |
| 已有 guardian/IPC | `io.BytesIO`+内存IPC、所有权、原子事务的现有模式 |

不能调用旧 `aug_history_20261105.protocol.resolve/worker` 冒充新协议：它们绑定D0/.5/15候选/A/B/C。新包只借用纯组件，独立组织新worker/guardian。

已找到可候选复用cache `/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache`；preflight 核验原CSV SHA、cache字节及数组/row/split/encoder/scaler身份完全匹配才使用。缺失则用原构建器在新任务cache重建，冲突不覆盖。原CSV为 `/home_new/xzf/datasets/ml-1m/ml-1m.csv` 和 `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。cache资源与2GiB观察输出分开记账。

## 4. EMA、pulse、组与更新次序

组顺序固定 `S,D0,D1,D2,T0,T1,DT00,DT01,DT10,DT11,DT20,DT21`。全部可训练坐标恰归一组，domain embedding按lookup行拆分；tower/gate内部索引`3*t+d`转换为展示cell顺序d0t0,d0t1,d1t0,d1t1,d2t0,d2t1。cell(d,t)只支持 S,Dd,Tt,DTdt，共24可达、48结构零；后者保留检查但不计主预测均值，不能按实测效应筛行。

所有EMA从theta0初始化，每次真实Adam后更新一次 `phi=rho*phi+(1-rho)*theta_after`，无warmup/偏差校正。H为1/2/4 epoch，`rho=2**(-1/(H*S))`。ML F精确rho=.9828205985452511，与H2物理去重、alias F→H2；KR F=.995另保留，实际half-life约2.8220933262epoch。物理shadow数ML3/KR4。source F精确rho优先，不从舍入表反算。

单步顺序：source loss→backward→唯一Adam step→展平theta_after→在F尚未更新时计算innovation `r=theta_after-F_before` 并即时按组归约→更新全部物理常驻EMA一次→若pulse活跃更新fast/slow→更新global_step。r既非梯度也非单步Adam位移；归约组L2平方、RMS、能量占比，持久只保存epoch均值/样本SD/最大值及anchor最后一步量。总能量0记missing。

pulse终点e=5,10,…,50；在epoch e−1完成F更新后从F复制两份状态，接下来恰好S步使用 `rho_fast=rho_F**2`、`rho_slow=sqrt(rho_F)`。终点27读出为F、12个fast hybrid、12个slow hybrid、全fast、全slow。hybrid实现copy F后仅复制目标组坐标，buffer统一F；等价于该窗口仅这组改变rho。F输出复用，不多存。所有读出顺序复用一份scratch，无online反馈。

epoch顺序：完成训练→完整V顺序评价online与物理H/F并独立选点→同次V前向提取V_mech→online/F训练panel读出→终点pulse读出与M0–M3→E20/E40仅保留F/fast/slow三向量→必要时启动下一pulse→恢复训练RNG→RAM先提交，再原子发布预测/标量/receipt。selected/E10/E50模型状态留RAM直至冻结Test结束。

## 5. 面板与原始预测

P_A/P_B先用 `SHA256(EMA_MECHANISM_20261106|train_partition|dataset|seed|original_row_id)` 最低位分成互不交池，再按计划role namespace抽样；不从online训练删行。每panel每cell256正+256负、固定4096个无放回正负pair；每域512自然行，natural用于LL/函数RMS。V_mech每域4096自然Validation行，两个任务同一批行；再用seed无关`SHA256(dataset|row_id)`最低位分V_A/V_B。固定V pair每格4096，不足用全部可用、缺类NA。配额不足工程阻塞，不更换人口。

面板key由SHA256派生独立generator，不触动训练RNG；相同dataset/seed的clean/G共用panel身份。row IDs、labels、role/pair索引与hash只存一次。V_A/V_B同一原始行跨seed保持同侧，完整V选点不受分区影响。

单次原forward同时取得原模型FP32概率p与tower raw logit z，hook在finally移除并恢复模式。官方AUC使用原p及ties；LL使用转float64后的源 `max(log(p),-100)`、`max(log1p(-p),-100)`。禁止重sigmoid(z)代替p、替换LL clamp或把rank-risk写成正式LL。

有界NPZ `allow_pickle=False`、禁object；train rank p/z形状[6,512]，natural p/z [3,512,2]；V p/z [3,4096,2]。regular online/F每epoch三panel p/z，其余H只V p；pulse新增26状态三panel，P_A/P_B p/z、V只p。完整Validation/Test预测不落盘，参数/梯度/optimizer/RNG checkpoint绝不落盘。

## 6. M0–M3与方向

M0在终点按组计算gap=`theta-F`、delta=`fast-slow`的RMS与cos；零范数cos=null，保存zero/missing flag。M1在自然训练面板对应cell计算fast−F、slow−F、fast−slow三项logit RMS，无收益符号。

M2在F及P_A固定pair上用tau1的平均 `softplus(-(z_pos-z_neg))` 排名风险。建立 `flat=F.detach().clone().requires_grad_(True)`，按布局切片建立带图params，调用 `probe_input_full_graph=True`。每cell一次grad与12组位移即时内积，保存 `-g·(fast-F)`、`-g·(slow-F)`、`-g·(fast-slow)` 及组grad RMS；全参数梯度立即释放。用完全相同F/panel/pairs改为False再做source_detach代理。full/source前向值相同，导数可不同；full导数须有限差分吻合。两语义共每anchor12次小面板梯度归约，全研究2400；P_B不求导。

M3在P_A/P_B rank panel计算精确原p AUC差及ranking risk降低量，另在natural panel报LL。fast−F、slow−F分别保存，fast−slow为前两项之差。正效用统一为AUC增加或LL/rank-risk减少。F是第三个合法选项；fast胜slow不表示fast胜F。

固定pair的纠正/损害/平局分解与精确面板AUC分开。相邻epoch e≥2用固定pair的 `mean(abs(B(p_e_pos-p_e_neg)-B(p_prev_pos-p_prev_neg)))` 计算T，B取1/.5/0，正/负变化单列；early仅e2–10。组非加性 `R=D_all-sum_bD_b` 与两个相对F残差均保存，并检验差分恒等式。

## 7. 两coder共享数据与Python接口（冻结）

两个coder均只新增本包及各自实现报告，不修改源文件，不撤销他人改动。Core负责protocol/panels/shadows/readouts/observers/worker/state/run及core测试；Analysis负责analysis/evaluation及对应测试。报告生成函数放analysis，避免第三个框架。以下接口兼容计划05，并明确并行边界。

### 7.1 必须直接消费的epoch记录

持久epoch记录路径为 `out/runs/{job_id}/epochs/E{epoch:03d}.json`；JSON仅允许标量、短列表/字典与预测receipt，不含参数或随机恢复payload。完整epoch一份记录，analysis可遍历这些路径；运行时汇总CSV可在采集或分析时从记录确定性生成。公共元数据放immutable身份表，记录用identity/hash引用以节省2GiB额度。

顶层最少字段：`schema_version,job_id,dataset,objective,seed,epoch,global_step,commit_index,init_hash,order_hash,layout_hash,group_hash,identity_id,validation,model_hash,pulse_readouts,pulse_metrics,innovation_stats,predictions,cost,complete`。

- `validation`：`physical_readout_id -> 源metrics字典`，字典保留`d0t0_auc`/`d0t0_logloss`等全部原键及macro/traffic/pooled原键。`model_hash`同一readout映射。F/H2 alias在PREPARED中冻结，analysis先解析再取数，不重复物理评价。
- `pulse_readouts`：每条为`{panel_id,readout_id,physical_readout_id,group_id,variant,domain,task,functional_support,auc,logloss,rank_risk,actual_count,state_hash,missing_reason}`。panel_id为P_A/P_B/V_mech/V_A/V_B；variant为F/fast/slow/all_fast/all_slow，group_id对F/all为null。P_A/P_B的auc/rank_risk来自rank，logloss来自natural；V类auc/logloss来自natural，rank_risk可null。至少V_A/V_B的准确绝对指标须采集或从同一NPZ派生，未来标签不使用完整V_mech代替。group的48结构零保留。
- `pulse_metrics`：每条按`domain,task,group_id`对应一个c,b；保留`functional_support,panel_id`及以下manifest精确feature名：`gap_rms,pulse_fastslow_rms,gap_pulse_cos,cos_missing_flag`，零范数flag；`P_A_natural_logit_rms_fastF,P_A_natural_logit_rms_slowF,P_A_natural_logit_rms_fastslow`；`P_A_full_rank_gradient_group_rms`；`P_A_rank_six_cell_macro_auc_allfastF,P_A_rank_six_cell_macro_auc_allslowF`；`P_A_gamma_full_fastF,P_A_gamma_full_slowF,P_A_gamma_source_detach_fastF,P_A_gamma_source_detach_slowF`；`P_A_rank_exact_auc_fastF,P_A_rank_exact_auc_slowF,P_A_rank_risk_reduction_fastF,P_A_rank_risk_reduction_slowF`。fastslow分数可由两列相减但须在输出表提供。P_B对应M1/M3复测用同规则改前缀P_B；source_detach梯度RMS、导数语义、计数与missing保留供诊断。全局偏好两项为P_A rank六格等权AUC差，不能用natural/pooled替换。
- `innovation_stats`：12组的epoch统计和必要anchor末步量，含每量计数/missing；不写每step序列。
- `predictions`：列表`{job_id,epoch,readout_id,physical_readout_id,panel_id,path,sha256,panel_hash,pair_hash,state_hash,arrays}`。path相对out；NPZ train键固定`rank_p,rank_z,natural_p,natural_z`，V键固定`p,z`（只需p时省z）；身份NPZ中的row/role/pair/label映射由PROBE_MANIFEST引用。analysis以receipt查找，不猜文件名。常驻alias只出现一次。
- `cost`：计划07规定的分项时间、计数、allocated/reserved/RSS、磁盘字节。非anchor时pulse两个列表为空。

此处长字段可以存在一次immutable metadata及带schema的列式块中，解析后语义必须等价。若core为实际源指标键增加兼容别名，须保留源原键，不改变指标。

### 7.2 共享函数与guardian读取

`protocol.resolve_study(manifest, paths=None)`返回对象至少提供`.manifest`（原解析字典）、`.jobs`（解析后20job）、`.configs`（dataset配置）、`.manifest_hash`与`.identity`；路径由PREPARED冻结，analysis无需另解旧resolver。所有analysis入口接受该对象与out。

`analysis.analyze_stage(study,out,stage)`、`analysis.fit_predictive_folds(study,out)`、`analysis.run_controls(study,out,decision)`、`analysis.generate_report(study,out)`；`evaluation.freeze_endpoints(study,out)`、`evaluation.evaluate_frozen(study,out)`。run仅调用这些接口，report调用generate_report。

`state.request(out, command, **payload)`为内存guardian IPC公共入口：`summaries`返回`{summaries:{job_id:summary}}`；`get,key=job_id`返回`{state:snapshot}`。summary提供job、epoch/global_step/complete、init/layout/group hash、cost及各物理readout的selected/E10/E50元信息，删掉model_state。snapshot提供计划05恢复字段，包含`online,optimizer,bank,pulse,loader,rng,snapshots,retained_pulse_states`，全部CPU RAM；snapshots沿用源`model_state,epoch,model_hash,validation`字段，retained_pulse_states按E20/E40含F/fast/slow与共同buffer。这些对象仅经RAM IPC交付C_BG和独立Test进程，不能写JSON。core负责其他注册/commit/release协议；analysis不修改state模块。

最小实现可在analysis读取epoch JSON/NPZ，在C_BG/冻结Test才读取RAM状态。两个coder如需澄清现有源字典结构，应先消息协调；不得各自创建第二套预测schema或wrapper框架。

## 8. 分析、预测、控制和Test

S0复现归档四格、版本去重和当前缺失观测清单，保留来源commit/blob/hash。缺旧标量记录`MISSING_ARCHIVED_ARTIFACT`并先恢复，不为表格重训。A/B同父online去重，KR旧G-D2不充当前G-D0。归档数值与当前重放按实际hash桥接。

S1每seed先聚合early1–10/middle11–30/late31–50、同刻收益、独立selected收益、online/F selected共同epoch、选点±1邻域；有限年龄延迟为 `age=rho*(1-rho**k)/((1-rho)*S)`，`j=e-floor(age+.5)`，不按结果调j。H偏好在e10/20/30/40/50报样本内与LOSO+V行交叉评价；tie顺序F,H1,H2,H4并先去alias。

S2形成pulse对比、全六格/组矩阵、翻转/时间T、非加性与M0–M3当期对应；S3未来预测只用24个冻结支持关系。

八类ridge：C_design、M0、M1、C_M01、C_simple、M2_full、M2_detach、M3_direct，feature集合严格由manifest predictive.feature_blocks/06定义。固定目标 `sum squared errors + 1*sum squared coefficients`，截距不罚；连续标准化仅训练折，one-hot48个冻结condition×cell×group类别加epoch/50。cos缺失报告NA，拟合用0+missing/zero flag。

每dataset五LOSO×两个V行方向：其他4seed、两condition、origins5/10/15、拟合侧V_A未来标签（epoch10/15/20）共576行；留出seedorigins25/30/35/40/45、另一侧V_B未来标签（30/35/40/45/50）共240行；再交换。Y两列为未来新pulse `[fast-F,slow-F]`，D=两列差。每seed先平均两个行折，两个dataset共160ridge fit。P_A特征不可使用未来/验证信息；P_B只做稳定性复测，不另选更好panel。

报告MAE_D/MAE_Y、相对全部5简单参照skill（0分母NA）、精确非零目标方向率、动作净益/regret及分解。诊断动作argmax(0,fastF,slowF)，tie F/fast/slow；同时四个raw无拟合M2_full/M2_detach/M3_auc/M3_rank。UNIFORM_ACTION仅用拟合折早期标签在相同24单组机会等权平均选动作并冻结，同留出机会评分；不与全参数all-fast混减。

C_CAL严格按e20/e40五seed AUC差与原LL差同号触发；T∈[.25,4]、最多100迭代/tol1e-8、V_A拟合V_B后交换、online/F全部六格两epoch；源p温度映射，T1原样、p0/1极限，AUC沿原p只重算LL并记饱和。最多960fits。

C_BG按e20/e40五seed均值 `q=|D_all-sumD_b|/(|D_all|+sum|D_b|)`，q>.5且max(|D_all|,sum|D_b|)>=.0002触发，0分母q0。按触发strata上mean(abs(D_b))选最多2组，tie源组序；两epoch/五seed/全六格、其余全fast及全slow背景，最多160新增状态，仅标量和state hash落盘。控制各一次，无新增训练，无触发写NOT_TRIGGERED。

freeze需20job、全部1..50完整/五seed、候选/比较/分析hash冻结、selected/E10/E50 RAM hash可用，生成SELECTION_FROZEN与相互绑定FREEZE_RECEIPT。独立evaluate进程核验后才加载Test，只regular online/H/F selected/E10/E50，逻辑上界270，alias/相同state去重；旧Test仅满足data/metric/seed/state全部一致可复用并标archived。pulse/温度/逐cell条件最优不进Test。

## 9. 资源、恢复与工程验收

最多2worker；每卡独立准入，显存和功耗同时低于85%且整机RAM<90%时可增加worker，但总数不超过2。当前GPU0空闲、GPU1有他人任务约22.9GiB/24GiB及高功耗，初始只用GPU0，不能结束他人进程。用两dataset各seed2022工程smoke测真实40960batch/尾批/一个完整pulse时钟与观察，覆盖clean/G；工程总Adam≤400，显式ENGINEERING_ONLY，不进入科学曲线。不得降batch/epoch/seed通过准入。

完整正常预算：4500全V模型读出、200pulse窗口、5400状态含F（新增5200）、最多16200state-panel评价含F、2400M2归约。预测原始payload1857945600bytes+256MiB元数据预留<2GiB；guardian参数payload约3.124GB，硬上限4GiB；cache单列。报告累计worker时间、墙钟、等待/IPC/分析、峰值和实际磁盘，不把它们混称GPU小时。

RAM-only含online/Adam/shadow/pulse/完整梯度/端点/锚点/恢复RNG。epoch唯一键(job_id,epoch,commit_index)，RAM先提交后原子发布，失败回滚RAM，一致重复ALREADY_COMMITTED、冲突不覆盖。worker从最近RAM边界恢复；guardian丢失同job从theta0确定性重放并逐epoch核验init/order/model hash，每逻辑最多一次额外重放，物理成本另计、不增加seed。最多40物理/2000epoch/69000Adam为故障上界。只能操作本任务登记并核验启动身份的进程。

reviewer独立核对规格/实际代码/CODER_IMPLEMENTATION_REPORT；PASS或PASS WITH FIXES且阻塞项已修正后才运行。必要验证限计划05九类：source clean/G loss/grad/一次Adam/F/尾批；observer纯度与RNG；EMA与alias时钟；pulse逐组等价/深拷贝；组完整/full有限差分/source前向相等/hook清理；概率AUC/ties/源LL/missing；面板/未来fold隔离；freeze/NPZ/RAM磁盘cap；RAM恢复/幂等/冲突。各用最小有意义测试，不追加科学确认实验。

## 10. 交付与停止

必须实现audit/preflight/validate/smoke/train/analyze(1–4)/controls/freeze/evaluate/report/release全部CLI及`--help`；公共参数`--manifest --out`，其余按计划05。退出0成功、2来源配置冲突、3工程数值、4资源、5预算/依赖未完成，工程失败不改写为科学阴性。

产物名称以manifest artifacts.required_names为完整清单：身份与PREPARED/RUN、九类验收/smoke/resource收据、归档复用四格、VALIDATION_CURVES/INNOVATION/PULSE_READOUTS/PULSE_METRICS、HISTORY_PREFERENCES/SELECTION_DIAGNOSTICS、RANK_FLIPS/TEMPORAL_RANK_STABILITY/GROUP_INTERACTIONS、PREDICTIVE_ROWS/OOF/SUMMARY、PAIRED_SEED/SUMMARY、STAGE1–4_DECISION/CONTROL_RESULTS、SELECTION_FROZEN/FREEZE_RECEIPT/ENDPOINTS、COSTS/REPLAYS/epoch ledger、MECHANISM_DECISIONS、FINAL_REPORT与PNG/PDF图。

每项结论带question/hypothesis/claim_level/status/scope/effect/n_seeds/source/竞争解释处理/下一决定；按实际证据允许READY_FOR_TARGETED_DESIGN、DESCRIPTIVE_ONLY、UNIFORM_HISTORY_SUFFICIENT_IN_SCOPE及具体SPECIFIC_QUESTION_UNRESOLVED。ACTION_USEFUL须五seed留出平均同时胜F和统一动作；复杂独立必要性才需胜全部简单参照。平均正效应<.0002仍报实际效应并标LOW_PRIORITY_SMALL_EFFECT；机制不套算法AUC≥.003硬门，不要求全seed同向或CI排除0。

统计mean、ddof1 SD、逐seed配对及95% CI `mean±2.776445*SD/sqrt(5)`；六格/AUC/LL、macro/traffic/pooled、split/selected/fixed/common严格区分。最终报告回答Q1–Q5，绑定plan/source/data/implementation/analysis hash及缺陷修订。全局台账/上下文只给更新提案，旧记录保持原样。

完成20条、S1–S4、冻结Test、报告和产物hash后自动release本任务RAM并提交代码/报告到新分支；不因局部问题继续追加H/seed/指标/控制。所跳过工作仅限未授权新算法、旧基线重训及与当前结论无关的工程扩展。

## 11. 修订 R1：V_mech 的自然人口上限与固定容器表示

修订结论：**APPROVED — 实现表示修正，科学配置不变。** preflight 发现 KuaiRand 完整 Validation 各域人数为 `[2692,246232,1843]`，因此其 V_mech 实际有效人数应为 `[2692,4096,1843]`。

具体依据是 manifest `panels.validation_row_cap_rule="min(4096, available domain population)"`，04 第5.1节“4096行或该域全体可用行”，06 第3.1节“最多4096行”。05 第6节配额检查中的 `PANEL_POPULATION_MISMATCH` 适用于不能满足固定训练rank/natural配额或身份错误，不把合法的Validation min规则当作配额失败。05形状表的 `[3,4096,2]` 是有界存储容器上界，不要求每域恰有4096个不同自然样本。本规格第5节“V_mech每域4096”的简写按此修订解释；不补采、不有放回重复、不改变源数据或split。

实现固定为以下规则：

1. Validation的natural索引/row ID/label/side仍用固定域优先容器 `[3,4096,...]`，在内存由已有natural `indices>=0` 派生同布局bool `valid_mask:[3,4096]`（不新增NPZ字段），每域 `valid_count=min(4096,N_d)`。只在前valid_count个槽位放真实自然样本；无效槽的索引、row ID、label、side均设为-1。p/z的无效槽为固定0，仅为未使用的存储槽；有效概率检查仅针对valid槽，mask是解析这些数组的必要条件。已有indices及sentinel唯一确定mask与计数，NPZ receipt引用原panel hash，不按epoch/readout重复保存；无效x槽为0。满额ML的已有NPZ字段、字节及hash完全不变。
2. 原forward、完整V提取到V_mech、pulse读出以及一切tensor/NumPy gather必须先选valid槽，禁止用-1索引访问原数组最后一行或让无效label进入任何公式。得到真实输出后仅散射回valid槽。训练loader、原完整V指标/选点和Test人口完全不受这个表示影响。
3. 精确V_mech/V_A/V_B AUC与LL、H偏好交叉评价、pulse标签/非加性、S3未来label、C_CAL及C_BG的Validation评价全部使用 `valid_mask`，行侧再与 `side==0/1` 取交集；实际分母使用有效真实人数。缺类别继续NA，不重新分配侧或填造指标。P_A/P_B固定训练配额保持原规则。
4. V fixed pair只从valid槽中真实正负样本构造，索引必须指向valid槽；不足4096对使用全部可用pair，缺类NA。若pair也采用固定上界容器，另存pair_valid_mask/pair_count，未用pair索引为-1；T/flip只除有效pair数。此规则沿用03第7.3节，不增加pair或forward预算。
5. `PROBE_MANIFEST`记录派生规则 `valid_mask=(indices>=0)`、每域actual_count及澄清版本 `validation_masked_cap_v1`；既有indices/row_id/side数组hash已唯一绑定mask，保持原panel hash计算规则。尚未成功生成的KR panel使用新sentinel表示形成其首次合法hash；既有ML panel的NPZ字段、内容、identity及hash全部保持不变。PREPARED/job观察身份引用各自实际panel hash。两condition同seed仍共用同一panel；所有epoch/readout使用固定mask，不能按输出改变有效性。文件白名单允许这些bool固定索引元数据，仍禁object/pickle/参数。
6. 当前尚无工程或科学Adam且PREPARED未发布；同attempt继续使用已保存的ML panel身份，不能重新生成或改变其字节/hash。仅完成此前失败的KR panel并在preflight收据绑定本修订；不能在正式epoch记录产生后静默改已有panel。修订、实际人数及新hash写入preflight/CODER报告和最终数据说明。

最小验收在已有panel/指标测试中加入一个人数不等的小型真实索引例子：检查min人数、无效索引从不被gather、掩码数组指标等于只取真实行的指标、V_A/V_B及pair均不包含无效槽。原 `[3,4096,2]` 预测上界、2GiB预算、20轨迹/五seed/50epoch及全部统计折保持不变。
