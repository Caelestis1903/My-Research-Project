# DESIGN SPECIFICATION：四候选五 seed 补全

日期：2026-11-03；规范来源：main `18d18bad6afdf84bc7c13aa8f303cc3f32019b12` 的三份入口，保存于 [source_main](source_main/)。研究入口为根目录 `start.md` → `buchong.md`。本规格完成 DESIGN，允许 coder 按本文实现编排与汇总；训练在独立 reviewer PASS 或清除阻塞后的 PASS WITH FIXES 后开始。

## 1. 本次工作及边界

完成 MovieLens/KuaiRand × C_MIX/H_MIX/P01/M × seed2022–2026 共40条逻辑轨迹，复用10条、补跑30条。两数据分别评价，任何1–4 seed性能结果不改变矩阵。保持原冻结数学定义与配置，不搜索参数、不新增机制、不重复训练已合法存在的同seed轨迹。第二阶段交付统一正式模型比较的可执行方案；本轮 GPU 工作为上述30条补全，不自动扩为模型搜索或新机制矩阵。

复用现有训练 worker、数据读取、优化器、评价和RAM guardian。新代码限于薄编排、历史数值接入及汇总。原包、vendor、旧设计、旧决策与历史报告保持原样。允许新增 `experiments/mdmt_five_seed_completion_20261103/`；新产物放 `artifacts/mdmt_five_seed_completion_20261103/attempt1/` 和 `research_reports/2026-11-03_mdmt_five_seed_completion/`。实现报告、独立审查记录置于新报告目录。主agent负责最终科学解释、台账更新和GitHub main发布。

## 2. 已完成审计与精确矩阵

[PHASE1_MATRIX.csv](PHASE1_MATRIX.csv) 是40行逐seed冻结矩阵，含action、源result路径/哈希和family。[DESIGN_REUSE_AUDIT.json](DESIGN_REUSE_AUDIT.json) 保存控制轨迹桥接证据。

| 方法 | 每个数据可复用seed | 每个数据补跑seed | 两数据新增 |
|---|---|---|---:|
| C_MIX | 2022 | 2023/2024/2025/2026 | 8 |
| H_MIX | 2022 | 2023/2024/2025/2026 | 8 |
| M | 2022 | 2023/2024/2025/2026 | 8 |
| P01 λ=.05 | 2022/2024 | 2023/2025/2026 | 6 |

现有科学seed数分别为1/1/1/2；旧D0/S1停止不能充当五seed否决。四方法均已有足够研究理由进入本矩阵，MovieLens也必须完成。

历史新增数据来源：

- C_MIX/H_MIX/M：`artifacts/mdmt_history_information_20261103/attempt1/runs/*/result.json` 与其 `final_evaluation.json`、epoch JSON、各attempt数据manifest；正式报告 `research_reports/2026-11-03_mdmt_history_information/REPORT.md`。
- P01：`artifacts/conditional_information_momentum_20261102_v3/supplement_20261103/`；正式报告 `research_reports/2026-11-02_mdmt_information_momentum_v3/REPORT.md` 的11月3日补充，端点表位于 `supplement/endpoints.csv`。仅λ=.05，无λ=.2外推。
- 普通训练和固定EMA控制、Gaussian＋EMA候选：`artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720/` 中 `resolved_config.json`、`test_metrics.csv`、`validation_metrics.csv`、`validation_epoch_metrics.csv`、`state_summary.json` 与各run配置。两数据均有五seed selected/E10/E50。ML固定EMA是H2，ρ=.9828205985452511；KR是ema995，ρ=.995。不能误用ML GAM01的ρ=.995。
- GATS B2仅作为已有五seed背景复用，不新增训练；来源为 `research_reports/2026-11-02_gradient_aug_time_select/REPORT.md` 及其tables。KR B2相对A0固定EMA test宏AUC +.001481447、LL −.001777377，相对A1 +.000005156/−.000113784。上述是不同比较对象，不抹去B2可能相对普通训练的整体收益。

已核对当前历史训练包完整source hash为 `7f16a451f5f0e41d9082d18bf548ad29d1d31f6c350175c5fb9f06ddc8813685`；PAIR训练包为 `5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a`，均与历史冻结一致。AIGM01与新source的数据raw hash、六数组、行号、encoder/scaler、人口完全相同；两数据×2022/2024×online/fixedEMA×selected/E10/E50 共24个模型hash逐一完全一致。控制可合法跨这条已核实链条复用，控制新增训练数为0。

同一seed历史重跑、online与EMA、三个epoch视图均只计一个独立seed。实现需按 `(dataset, arm, seed, estimator, endpoint, role)` 去重，不能把AIGM重复历史引用追加成新观察。

## 3. 固定训练与方法身份

fixed1 MDMTRec/M3oE：3域2任务，embedding16，4共享专家，FCN[512,256,256,64]，固定融合1。自然原全局行序80/10/10，KR切分后筛tab0/1/4；只在train拟合预处理，OOV=0。ML任务rating>3/rating>4，KR任务is_click/long_view。原人口不得改比例、重采样或造替代数据。

CSV路径：ML `/home_new/xzf/datasets/ml-1m/ml-1m.csv`；KR `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。原始SHA256分别 `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`、`a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。现有loader已经校验数据，不需另写验证层。

每条50epoch、batch40960含自然尾批，无早停；ML1000步、KR2450步。FP32，AMP/TF32关闭；Adam lr=.01，β=(.9,.999)，eps1e−8，coupled wd ML1e−4/KR1e−5。source函数返回完整配置，必须原样调用。EMA从θ0开始，buffer copy、不反馈在线参数；ρ见上。选点为每个估计器自身validation六格macro AUC最早严格最大值，E0排除。

| 方法 | 调用 | 冻结参数/关键语义 |
|---|---|---|
| C_MIX | history.protocol.job(dataset,'C_MIX',seed,stage) | τ=.1、κ=1；条件probe约束当前梯度，真实loss |
| H_MIX | history.protocol.job(dataset,'H_MIX',seed,stage) | τ=.1、κ=1、agreement cosine=.2；只修旧历史，真实loss |
| M | history.protocol.job(dataset,'M',seed,stage) | λ=.1、τ=.1；条件标签嵌入插值，原Adam |
| P01 | pair.run.job(dataset,'P01',seed,stage,lambda_pair=.05) | 静态PAIR+ECCM；保留2048 pairs、q/w、共享参数分组与时钟 |

`history` 为 `experiments.mdmt_history_information_20261103`；`pair` 为 `experiments.conditional_information_momentum_20261102_v3`。原worker中的stage只标注身份，可传新stage `FIVE_SEED`；不调用原 `decisions.stage_jobs/decide`，也不调用旧schedule的开发种子门。原worker通过 `python -m <family>.run _worker --attempt <new family attempt> --key <key> --device cuda:N` 启动，读取已写入 `jobs/<key>.json`；按原source hash校验。新编排本身再绑定新source与review记录，原算法源码不改。

## 4. 最小执行编排

1. prepare：读取冻结矩阵，核对30个RUN与10个REUSE；生成30个原生job，按history/pair分开attempt目录，以兼容各自guardian。保存配置、source hash、矩阵与复用记录。历史端点从其正式标量文件接入，不重新前向历史模型。
2. 调度：两个family共用一个GPU队列；每张卡显存比例和功耗比例均<85%且可用显存足够时启动新训练，不要求另一张卡同时可用。复用现有gpu_status/choose_device，逐进程登记及首批allocated回执后再追加，避免同时启动未分配进程造成显存误判。保留原batch，OOM先按原batch处理资源并同身份重放，不能改配置。失败记录成本，未完成job仍留在矩阵中。
3. 内存：调用各family现有ensure_guardian/request；模型、optimizer、选择状态及E10状态只驻RAM，不调用torch.save/pickle.dump写磁盘。允许hash、指标、配置、JSONL、CSV与日志落盘。进程终止但guardian有完整epoch时按原worker恢复；恢复次数不是新seed。
4. 冻结：全部30条完整训练完成后，一次冻结本轮全部新配置与各job的online/fixedEMA × selected/E10/E50身份。生成原评价函数需要的冻结文件：history用 `CONFIGURATION_FROZEN.json`+`SELECTION_FROZEN.json`；pair用 `selection_frozen.json`。按source hash、epoch、model_hash构造；不执行旧性能决策。全局完成/冻结之前不开启新增test评价。
5. 测试：复用对应family.run.final_evaluate；每条新轨迹六个端点，共180个新test端点。评价成功后调用原release，结束时shutdown guardian。历史40个候选逻辑端点 ×2估计器×3checkpoint=240行test记录（180新、60复用），另纳入控制和Gaussian历史记录。validation对应端点与全部50epoch曲线同样保留。
6. 汇总：输出统一wide端点CSV（六格AUC/LL、macro、traffic、pooled、worst、domain/task、epoch/hash、原始来源、复用标志）、逐seed配对差、五seed均值/样本SD/配对t95%区间、按比较对象分开的裁定JSON。CI仅描述，不参与合格判定。逐seed原值始终可追溯。

建议新模块提供 prepare/run/summarize（或功能等价的直接命令）；具体CLI由coder按最少代码确定并写明调用。运行命令显式用 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`，不使用base。无需新依赖。

## 5. 冻结评价及保留规则

整体收益：method fixedEMA selected − 同seed普通A0 online selected。组件增量：method fixedEMA selected − 同seed A0 fixedEMA selected。源实验以固定EMA为对照的主比较保留为组件分析；候选整体是否进入正式论文比较按整体收益评价。两项各自完整给出裁定和比较对象，不能以增量不达标取消总收益。

对每一数据集、每一比较，必须恰有2022–2026五seed完整配对，分别计算：

- 宏收益 `mean(delta_macro_auc) >= .003`（含等号）；逐格 `mean(delta_dXtY_auc) >= -.001`（含等号）。两项均满足标记“AUC合格且满足逐格约束”。宏达标但有格越线，标记“宏AUC达标、逐格约束未满足”并逐格列代价。
- LL下降为 `mean(delta_macro_logloss) < 0`，独立于AUC裁定。
- 不添加4/5、5/5、单seed保护、CI全正、validation/test同向、traffic必须提升等淘汰门。seed方向数、traffic/pooled与共同epoch仅作解释。
- 未满五seed状态为INCOMPLETE；工程故障写工程阻塞，数值故障写数值阻塞，不返回性能失败。所有授权矩阵完成后才可形成最终方法裁定。

进入正式比较的建议规则：总宏AUC达标的固定配置保留，包括逐格越线者；后者以“收益伴随逐格代价”进入比较附表，不能宣称满足逐格约束。总AUC未达标但LL下降者保留其LL成绩，当前AUC论文主候选不优先；不追加调参来补过门。一个数据集达标则可列迁移边界，不能据另一个数据集的阴性删除已见收益。Gaussian＋EMA按相同规则分析，优先级不能由算法复杂程度决定。

Gaussian＋EMA已完成五seed：ML总ΔAUC+.004166933/ΔLL+.000931348，六格AUC全部正；KR总ΔAUC+.006092690/ΔLL−.002888935，d0t0为−.001271286，须标明逐格代价。增量相对A0固定EMA为ML+.000318088/−.000317296，KR+.001476291/−.001663593。由此ML已可进入总收益正式比较，KR可作为带逐格代价的候选；无需重复五seed训练。

## 6. 最小验证与职责

Coder只实现本规格与基本检查，不分析科研结果；reviewer须独立对照本文、实现和CODER IMPLEMENTATION REPORT。原算法source hash和原有CPU/S0证据保持有效，无需重复做机制证明。新编排做一个小CPU自检覆盖：40/10/30矩阵计数与seed集合、job原配置一致、历史去重、五seed才裁定、阈值等号及LL独立裁定、负seed和跨零CI不额外否决。冻结/原评价格式作实际对象构造检查；首次真实训练沿原worker首batch接线继续完整50epoch，不把smoke充当性能门。

Runner仅记录原始结果、运行失败/恢复、资源、计数、成本，不作研究结论。主agent在ANALYZE完成AUC/LL、六格/任务、seed一致性、域规模/基线预测难度、模型和阶段分析，并明确STRONG/MEDIUM/WEAK、矛盾、竞争解释、大小是否主解释及小域/大域交换。已经回答的问题不追加实验。

验收：30条RUN均50epoch，10条复用对应合法配置；8个dataset×candidate均5seed；180个新test端点及历史候选端点完整；source与freeze可追溯；未写模型文件；完整raw/差值/汇总/成本/分析报告；main指标台账回填；方案与结果发布main。历史30条worker成本按相同方法外推约14–15 worker小时，实际墙钟取决于共享GPU并发，记录实测值。

## 7. 第二阶段统一协议入口

详细方案见 [PHASE2_PUBLISHED_MODEL_PROTOCOL.md](PHASE2_PUBLISHED_MODEL_PROTOCOL.md)。正式模型仅 M3oE（SIGIR2024）、HyperGate（WWW2025 Companion）、AutoIFS（WSDM2026）；EMA/增强/动量组件均为候选或内部控制。10/29训练人口T0∪F、预处理T0、V按LL选点、预算/批量与近期80/10/10全train、按AUC选点均有实质差异，旧模型绝对值不得与本次候选横排。统一协议采用近期自然人口和AUC选择，保留各发表模型必要训练阶段，明确预算，不把不同优化过程称为严格架构因果比较。第二阶段作为独立执行任务在冻结模型适配、设计、实现、审查完成后运行。
