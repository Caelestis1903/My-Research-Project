# 固定来源、证据层次与复用登记

## 1. 来源快照及读取规则

本次审计仓库为 `Caelestis1903/My-Research-Project`，main固定到 **875215c1e509ab55c1b6632d4f5d13e8b94782da**。下列链接均指定不可变提交，区分main归档与来源提交中的原表/实现。当前规划将作为新的main提交发布；该新提交不改变本表所指历史证据。

| 标识 | 固定 Git commit | 用途与来源范围 |
|---|---|---|
| M | `875215c1e509ab55c1b6632d4f5d13e8b94782da` | 本轮main研究上下文、已归档报告、v3数值内核、近期工程实现 |
| P | `a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288` | 原 `26-11-02_v3` 的PAIR/ECCM两seed原表、实现清单及补充冻结记录 |
| F5 | `652ee0235fec2d0d4cfc67b4a6ef9977519619bf` | 11/03五seed补全原发布来源；main已归档的完整数值优先用M |
| PUBBASE | `c8675ae2923c15b6965884b140b5b646690cd95e` | 11/04统一发表模型、CP及Gaussian域覆盖的原端点与配置 |
| ADAPT | `1391affa5f08556c79b4cb2bfad15228a8b9129f` | 11/09 EMA_ADAPTIVE原报告；main只有整合结论 |
| PUB | `a5f461de714232a416a2e95ca6fdc57fa7ce7ae8` | 11/09 PUB-MDMT报告及候选梯度计算实现 |
| B | `fb120946c50519ea3993bbd6a2532a2c457731fc` | 11/10因果归因、11/11反馈/历史报告与坐标registry等依赖 |

Git commit、Git blob SHA与文件SHA256/算法source hash是不同身份，不得互相替代。执行前对本轮实际读取文件生成 `SOURCE_MANIFEST.json`，包含仓库、commit、path、blob SHA、文件SHA256、用途、是否main可读及新的适配器版本。新编排自身的source hash单列；不能修改旧内核后继续宣称旧PAIR source hash不变。

**本地存在不代表main存在。** 已现场确认：M上的v3 `protocol.py/boundary_sampler.py/objectives.py/optimizers.py`及五seed `run.py`可读；P的两份原两seed CSV、`IMPLEMENTATION_SOURCE_MANIFEST.json`在M缺失；`aug_history_20261105/group_registry.py`在M缺失而B可读。四份11/09–11/11原B报告也留在来源提交。只将本计划文件提交main，不因规划顺便合入所有缓存源码、旧报告或工件。

## 2. 主上下文及执行规范

以下六个主入口均以M读取，研究结论与工程状态联合使用。旧段落记录的四seed、无限循环、人工停止后才开放Test等规则，与本次明确任务的适用关系由 [00](00_EXECUTION_RUNBOOK.md) 固定。

| 文件 | 本轮提取的信息 | 读取限制 |
|---|---|---|
| [EXP_AI_CONTEXT.md](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/EXP_AI_CONTEXT.md) | 当前完整五seed结果、工程状态丢失、研究交接 | 某些较新C05细节以C05原报告补足，不能把旧owner存活写成当前事实 |
| [Experimental_Global_Summary.md](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/Experimental_Global_Summary.md) | 跨分支总结果、独立增量与总收益、原件来源 | 汇总结论不意味着原报告/代码都合入main |
| [Experimental_Stable_Findings.md](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/Experimental_Stable_Findings.md) | 可直接接受的固定EMA/历史/参数组经验事实 | 不能因机制尚待区分而归零已有可靠现象 |
| [Experimental_Open_Questions.md](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/Experimental_Open_Questions.md) | 尚缺的任务身份、真实更新与持续训练证据 | 历史建议不自动成为本轮新增任务 |
| [MDMT_ALGORITHM_METRICS_LEDGER.md](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/MDMT_ALGORITHM_METRICS_LEDGER.md) | 六格绝对指标、选点及合法发表模型比较 | 原始均值用逐seed原表核对；不能跨旧人口/选点排名 |
| [AGENTS.md](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/AGENTS.md) | 原前台监督、不同coder/reviewer、资源门、RAM_ONLY、C05后停止 | 本次仅规划并提交main；后续执行本目录有限矩阵，不恢复旧循环 |

补充读取：[AGENT.md](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/AGENT.md)、[PROJECT_ARCHITECTURE.md](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/PROJECT_ARCHITECTURE.md)、[start.md](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/start.md)。早期架构文档中的单seed动机、旧可学习fusion或cell索引描述不能覆盖当前fixed1真实实现。**小域不是困难域**，历史文件里围绕规模/困难度的假设性研究建议不进入本计划。

## 3. 方向 A：精确原表、实现与身份桥接

### 3.1 报告及数值输入

| ID | 精确文件 | 状态与本轮作用 |
|---|---|---|
| A-R0 | [原PAIR/ECCM主报告，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-02_mdmt_information_momentum_v3/REPORT.md) | 主报告归档含两seed补充背景；不能当五seed独立历史判定 |
| A-R1 | [原始tables/endpoints.csv，P](https://github.com/Caelestis1903/My-Research-Project/blob/a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288/research_reports/2026-11-02_mdmt_information_momentum_v3/tables/endpoints.csv) | 288端点行，含原S1；主控补充身份之一，M缺失 |
| A-R2 | [supplement/endpoints.csv，P](https://github.com/Caelestis1903/My-Research-Project/blob/a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288/research_reports/2026-11-02_mdmt_information_momentum_v3/supplement/endpoints.csv) | 240端点行，P00/P01/P10/P11 λ=.05的两seed组件表；M缺失 |
| A-R3 | [补充selection_frozen.json，P](https://github.com/Caelestis1903/My-Research-Project/blob/a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288/research_reports/2026-11-02_mdmt_information_momentum_v3/supplement/selection_frozen.json)；[NUMERICAL_APPENDIX，P](https://github.com/Caelestis1903/My-Research-Project/blob/a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) | 选点epoch/model hash与逐seed比较；旧Test已暴露 |
| A-R4 | [五seed完整REPORT，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md)；[endpoints.csv，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_mdmt_five_seed_completion/endpoints.csv) | 660端点行；A0/P01、C/H/M等五seed结果，online/EMA与多epoch不增n |
| A-R5 | [RUN_MANIFEST.json，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_mdmt_five_seed_completion/RUN_MANIFEST.json)；[RAW_ARTIFACTS.md，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_mdmt_five_seed_completion/RAW_ARTIFACTS.md) | 数据/source/config/选点来源；只保留标量工件，旧模型/optimizer已释放 |
| A-R6 | [五seed数值附录，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_mdmt_five_seed_completion/NUMERICAL_APPENDIX.md)；[stage_effects.csv](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_mdmt_five_seed_completion/stage_effects.csv)；[costs.csv](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_mdmt_five_seed_completion/costs.csv) | 共同阶段与训练成本；不能用E50救援selected主结果 |
| A-R7 | [原补全DESIGN_SPECIFICATION，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_plans/26-11-03_v2/DESIGN_SPECIFICATION.md)；[PHASE1_MATRIX.csv](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_plans/26-11-03_v2/PHASE1_MATRIX.csv)；[DESIGN_REUSE_AUDIT.json](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_plans/26-11-03_v2/DESIGN_REUSE_AUDIT.json) | 旧2022/2024复用及24模型hash的历史桥接；旧矩阵不直接作为本轮12条任务清单 |

已复算P的supplement与M五seed表之间A0/P01的 **96个端点身份**：两数据×两个旧seed×online/fixed_ema×selected/E10/E50×Validation/Test。模型hash与宏/六格AUC逐值相同，Macro LogLoss最大浮点差5.551115123125783e−17。该桥接支持旧数值复用；不把重复表行当作额外科学重复。

PAIR原执行source hash为 `5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a`，由 [IMPLEMENTATION_SOURCE_MANIFEST.json，P](https://github.com/Caelestis1903/My-Research-Project/blob/a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288/research_reports/2026-11-02_mdmt_information_momentum_v3/IMPLEMENTATION_SOURCE_MANIFEST.json) 固定；五seed薄编排source hash为 `db8d3385f3acabb11e2f10207e6adafb88cc949b51b7b9553a5117434366b7d6`。新编排只接入固定数值内核，不冒用旧review、dispatch或source receipt。

### 3.2 真实 v3 API 与复用边界

以下路径以 `experiments/conditional_information_momentum_20261102_v3/` 为前缀；链接直接到M中已核实内核。P的实现清单用于逐文件复核历史身份。

| 文件 / 真实接口 | 本轮复用内容 | 必须保留的边界 |
|---|---|---|
| [protocol.py](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/conditional_information_momentum_20261102_v3/protocol.py)：`load_protocol()`、`config(arm,dataset,lambda_pair=0.)`、`load_dataset(dataset,split_id='original',p=None)`、`make_optimizer(model,c)` | frozen配置、数据、模型与EMA工具 | M/P的protocol Git blob均为243d979676b20130d089b8c7edab8bc154c7162d；不将旧硬编码运行根直接当新artifact root |
| [boundary_sampler.py](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/conditional_information_momentum_20261102_v3/boundary_sampler.py)：`rng(context,stream)`、`boundaries(domain,y,demand,dynamic)`、`sample(domain,y,demand,dynamic,context,row_ids)`、`forward_logits(model,x)` | 最多24联合标签边界、2048带放回pair、独立流、目标质量 | dataset/seed/epoch/global_step/stream固定键；不加入arm/lambda改变匹配流；不能用NDP视图替代 |
| [objectives.py](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/conditional_information_momentum_20261102_v3/objectives.py)：`train_step(model,optimizer,ema,x,y,row_ids,config,context,diagnostic=False)` | 原生BCE与PAIR梯度、一次Adam/EMA顺序 | 原PAIR无RMS归一化；不能移植Gaussian缩放后仍称单因素对照 |
| [optimizers.py](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/conditional_information_momentum_20261102_v3/optimizers.py)：`shared_name(name)`、`PairOptimizer(model,config)`、`step(pair=None,r=None)`、`state_dict()/load_state_dict()` | 普通Adam、ECCM与collapsed历史数值参照 | dense历史仅shared_weight/shared_bias/skip_conn./star_mlp./expert.，不包含embedding；PCOLL源默认dynamic=True，本轮A21须显式静态 |
| [run.py](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/conditional_information_momentum_20261102_v3/run.py)：`job(dataset,arm,seed,stage,split_id='original',lambda_pair=0.)` | 旧job身份及配置构造参照 | 旧CLI的manifest/initial/stage0/schedule/freeze/evaluate/status/_worker/_guardian/shutdown真实存在；旧两seed调度不是本轮入口 |
| [vendor/ema02/protocol.py](https://github.com/Caelestis1903/My-Research-Project/blob/a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288/experiments/conditional_information_momentum_20261102_v3/vendor/ema02/protocol.py)及其GA31 vendor模型/视图依赖 | make_model/setup/update_ema/evaluate/state_digest、原fixed1图 | gate输入detach、embedding名字、融合常数及加载顺序固定；source closure包含真实依赖 |
| [mdmt_five_seed_completion_20261103/run.py，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/mdmt_five_seed_completion_20261103/run.py) | `prepare/run/summarize`原补全生命周期参考 | 原40条矩阵不能直接启动为本轮12条；仅复用原结果与必要工具 |

本轮新增包唯一规划路径为 `experiments/conditional_history_shared_update_20261122/`，所有新API/CLI在 [06](06_IMPLEMENTATION_SPEC.md) 明确标为待实现。本次没有实现或运行它们。

## 4. 方向 B：历史机制、状态与工程来源

### 4.1 机制原件

| ID | 精确文件与位置 | 可接受结论 / 不可外推之处 |
|---|---|---|
| B-R0 | [EMA_MECHANISM完整报告，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md) | 四条件五seed20轨迹，固定EMA有排序收益且超过晚期年龄匹配；不必重证 |
| B-R1 | [EMA_ADAPTIVE，ADAPT](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md) | 复用20轨迹，无新增训练/Adam/Test；A0未来动作未形成独立增量，A1/B/C未触发，不是完整算法Test失败 |
| B-R2 | [PUB-MDMT，PUB](https://github.com/Caelestis1903/My-Research-Project/blob/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md) | 当前原尺度coupled-WD迁移五seed失败；对退化G的改善不是超越正常A/F |
| B-R3 | [EMA因果归因，B](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)；[FINDINGS_CLARIFICATIONS](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/research_reports/2026-11-10_ema_causal_attribution/FINDINGS_CLARIFICATIONS.md) | S/EMBED贡献与recent/old互补已成立；同权预测组合及current-preconditioned单步是竞争解释/局部现象 |
| B-R4 | [EMA_NEXT，B](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/research_reports/2026-11-11_ema_next/FINAL_REPORT.md) | A_MATCH六epoch持续反馈两数据下降；B有ML LL与KR小幅诊断信号；未运行阶段保留NOT_TRIGGERED |
| B-R5 | [11/10 PROVENANCE，B](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/research_reports/2026-11-10_ema_causal_attribution/PROVENANCE.md)；[11/11 PROVENANCE，B](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/research_reports/2026-11-11_ema_next/PROVENANCE.md) | 模型、Adam、EMA、历史仅驻RAM且报告/Test后释放；原表/hash不能恢复状态 |
| B-R6 | [AUG_HISTORY，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-05_aug_history/REPORT.md) | 方向、分组时间尺度与ML漂移协方差均完成五seed；KR漂移为DEFERRED_NOT_RUN；guardian已释放 |

B-R1至B-R5的原文件不在审计M；main总上下文只整合结论。上述来源提交才是这些路径的有效引用，不把缓存相对路径写成main原件链接。

### 4.2 可复用科学模块与真实限制

| 真实路径 / API | 来源 | 本轮用途及限制 |
|---|---|---|
| [completion_weight_attribution_20261117/core.py](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/completion_weight_attribution_20261117/core.py)：`train_step(...,row_ids=None,diagnose=False)`、`parameter_activity(named)`、`adam_step_counters(named,optimizer)` | M，当前精确clone模板 | 内部nested `virtual_steps()`保存before_optimizer、deepcopy模型、原optimizer load_state_dict、保留None，再step；它不是可直接import的公共API，须按06适配。已有JOINT虚拟/实际model与optimizer hash一致检查 |
| [auxiliary_objective_attribution_20261115/protocol.py](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/auxiliary_objective_attribution_20261115/protocol.py)：`optimizer(model,c)`、`dataset_cache()` | M | 原生Adam foreach=False/fused=False/amsgrad=False；数据raw/array/row/encoder/scaler身份检查；其模型/数据来自v3，不因绝对AUC不同猜测人口变化 |
| [task_view_gradient_residual_momentum_20261116/core.py](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/task_view_gradient_residual_momentum_20261116/core.py)：`source_clean_step(model,optimizer,ema,x,y,config)` | M | 普通BCE backward→Adam→EMA参考；γ=0不耗view/RNG。只复用clean路径，不复活TV残差规则 |
| [aug_history_20261105/group_registry.py](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/experiments/aug_history_20261105/group_registry.py)：`build_path_groups(model,bundle)`、`GroupRegistry.indices(device)/as_dict()` | B；M缺失 | 12互斥坐标组、domain embedding行归D；实际依赖`.registry.ParameterLayout`及`overnight_ab_20261103.state.digest`，提取须连同source closure或显式适配，不假装main可直接import |
| [ema_causal_20261110/interventions.py](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/experiments/ema_causal_20261110/interventions.py)：`group_masks()` | B | EMBED、STAR_SKIP、EXPERT、D/T/DT分组；本轮主矩阵规范名为EMBED、S_DENSE、N，分别映射旧E、R/SHARED_OTHER=S\\E、NONSHARED |
| [ema_causal_20261110/pairs.py](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/experiments/ema_causal_20261110/pairs.py)：`exact_transition_table()`、`transition_metrics()`、`sample_pairs()`、`panel_pair_statistics()` | B | 固定pair、平局半贡献、纠正破坏AUC闭合；若复用C++核同时纳入pairs_kernel.cpp，不展开全量pair |
| [ema_causal_20261110/trajectory.py](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/experiments/ema_causal_20261110/trajectory.py)；[ema_mechanism_20261106/shadows.py](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/experiments/ema_mechanism_20261106/shadows.py)：`FixedShadowBank.after_adam()` | B | 原生训练顺序、observer RNG中立、FP32非反馈EMA；与main update_ema选一套，不能更新两次 |
| [ema_next_20261111/history_readout.py](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/experiments/ema_next_20261111/history_readout.py)：`split_history()` | B | FP64构造recent/old重心后转FP32及误差检查；仅短段同权函数组合，不重搜H |
| [pub_mdmt_20261109/optimizer.py](https://github.com/Caelestis1903/My-Research-Project/blob/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8/experiments/pub_mdmt_20261109/optimizer.py)：`losses()`、`gradients()`、`candidate()` | PUB | 六格自然质量和历史候选的数学参考；原gradients展开None为0、flat step，不能直接作为本轮per-parameter真实Adam反事实引擎 |

CWA现存诊断仅含梯度/update几何及 `deltaEMA=(1−rho)*deltaTheta` 代数记录，没有本轮任务源→六格排序→同位移历史背景试验，所以复用代码不等于重复B科学实验。源`ModelAdapter`部分诊断会解除gate输入detach；真实训练梯度必须使用原生概率forward及原生detach图，不得借用full-functional-graph梯度。

## 5. 近期七组完整五seed证据

以下均为两自然数据集、M3oE fixed1、50epoch、batch40960、五seed正式训练；主端点为各自Validation选择的fixed EMA，**Test均未访问**。CLEAN/NDP的反复引用属于同轨迹数值复用。表中Δ均为相对matched CLEAN的selected Validation Macro AUC，特殊列另标。

| 研究及精确报告（M） | 已完成新训练 | 核心证据 | 本轮决定与保留信号 |
|---|---:|---|---|
| [辅助目标/NDP](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/auxiliary_objective_attribution_20261115/REPORT.md) | Stage1 40；NDP-A 10 | 零扰动Z控制重现主要退化；NDP-A自然质量修正后ML−.000101256、KR−.005775319 | 旧质量缺陷成立，修正后当前NDP仍无EMA之上排序增量；ML online+.000359628保留。NDP-B/HPO未触发不写成负结果 |
| [TV-GRM](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/task_view_gradient_residual_momentum_20261116/REPORT.md) | 50 | γ=.10 ML−.000062272、KR−.000166450；SHUFFLE/INSTANT未建立额外残差历史必要性 | ML γ=.20 +.000153992及LL五seed下降保留；对NDP的KR改善不等于胜CLEAN |
| [AXIS-EVEN C01](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/task_axis_even_views_20261116/REPORT.md) | 30 | even ML−.000237520、KR−.003540710；even−IID2两数据均值负 | KR IID2+.001269552、2/5正、CI跨0保留；不能将even局部loss响应当完整算法收益 |
| [NQR C02](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/natural_quadratic_ranking_20261116/REPORT.md) | 30 | NQR ML−.000343517、KR−.000328734；Brier/mean-gap未建立排序优势 | ML E10 +.001151311、5/5正是阶段现象，不救援selected；这已是完整五seed结果 |
| [Temporal C03](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/nqr_temporal_exposure_20261116/REPORT.md) | 20 | EARLY ML−.000340045、KR−.000068613；LATE ML选中点在干预前 | EARLY相对constant的ML LL−.003069812五seed下降保留；不再重复开关时序来证明同一结论 |
| [Completion C04](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/completion_weight_attribution_20261117/REPORT.md) | 20有效；退役275epoch另计 | W ML−.000154621、KR+.002555086；KR d0t0−.003141373越保护线 | MIXED；KR LL−.003942334且5/5降、W−S的task1 LL收益保留，不说全负；selected状态后来丢失 |
| [Cell mass C05](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/cell_mass_bce_20261117/REPORT.md) | 10 | U ML−.000221160、KR−.000235705，KR 4/5正但2024−.009762547使均值负；两数据LL均值升 | 当前U无继续信号，不能删seed或改中位数；C06未授权。旧selected-only ACK不提供Adam |

近期工程细节还读取：[C04 METHODS](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/completion_weight_attribution_20261117/METHODS.md)、[C05 METHODS](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/cell_mass_bce_20261117/METHODS.md)、[C05 EXECUTION_HISTORY](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/cell_mass_bce_20261117/EXECUTION_HISTORY.md)、[C05 DATA_AVAILABILITY](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/cell_mass_bce_20261117/DATA_AVAILABILITY.md)与[COST_ACCOUNTING](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/cell_mass_bce_20261117/COST_ACCOUNTING.json)。

## 6. 正式发表模型及Gaussian内部控制

| 原件 | 本轮用途与身份 |
|---|---|
| [统一AUC选点/CP报告，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md)；[ENDPOINTS.csv，PUBBASE](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv) | `AUC_SELECTED_V1/FULLTRAIN_NATURAL80_10_10`，M3oE/HyperGate/AutoIFS均已有两数据五seed；30个发表模型训练身份默认复用，新增训练0 |
| [Gaussian覆盖报告，M](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md)；[ML来源补充，PUBBASE](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/MOVIELENS_SUPPLEMENT.md) | ML G-D0是旧Gaussian别名；KR G-D0/G-D2/G-D02已有五seed，不能重复为“全新覆盖搜索” |
| [Gaussian ENDPOINTS，PUBBASE](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/ENDPOINTS.csv)；[RUN_MANIFEST，PUBBASE](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/RUN_MANIFEST.json) | 最终候选Gaussian D0补充的对照身份输入，冻结噪声强度、域覆盖、目标质量及选点 |

合法发表模型selected Test Macro AUC：ML M3oE .775190、HyperGate .771412、AutoIFS .772425；KR .671907/.682113/.678597。旧10/29使用不同训练人口与V LogLoss选点的表不混入本轮主表。HyperGate部分超参有`UNVERIFIED_PROPOSAL`，AutoIFS包含原生search/rewind/ticket/reuse阶段；这三者是经审查工程适配，不能写成完全相同训练预算或严格原论文复现。

固定EMA、Gaussian、PAIR与本轮候选均为内部控制/研究方法。复用旧发表模型须满足人口/预处理hash、seed、模型配置、合法selected epoch/model hash与相同Test人口；模型各自native optimizer和预算保持原冻结值。身份一致就直接比较，不能因旧报告曾写“HG/AI尚未运行”而再新增20条。只有确实改变了协议、导致现有端点不再可比时，才具体重定必要控制与预算。

## 7. 跨研究证据层次及缺项清单

| 对象 | 已有统计与证据层次 | 已支持结论 | 本轮缺项 / 执行动作 |
|---|---|---|---|
| A20=P01 λ=.05 vs A00+F | 两数据各5seed，完整50epoch、历史selected Test | KR总方法相对F Macro +.004085725；d1t0/t1代价越线；ML Test−.000623736 | 复用；不再训练P01证明相同总收益 |
| A20 vs A10=P00 | 两数据各2seed，完整训练 | 条件性组件线索；KR Test约+.012099而Validation约−.000101 | 补A10缺2023/2025/2026；五seed后判断历史包独立作用 |
| A30=P10 vs A10 | 两数据各2seed，完整训练 | 动态抽样效果与读出/人口有关 | 补A30缺3seed；与A10配对，不先假定动态更好 |
| A21静态collapsed/A22单历史长Adam | 原PCOLL/PCV仅smoke；无本轮正式五seed | 代码路径存在，不存在已有性能否决 | 只在A历史独立增量门通过后各10条 |
| P11、旧λ=.20、EMA02部分配置 | 主要2seed开发、部分未触发 | 已测条件性结果保留 | 不默认补全所有旧臂；不写成五seed普遍失败 |
| C_MIX/H_MIX/M与CP | 后续正式五seed；CP完整2×2 | 总收益不能替代组件收益，CP相对P01两数据负 | 结果复用；不重做简单CP叠加 |
| EMA_MECHANISM / S与EMBED / recent与old | 四条件5seed完整训练+冻结状态面板 | EMA有效、共享修正主导、历史互补 | 接受作为基础；不重复20条来重证 |
| EMA_ADAPTIVE、未来规则 | 五seed原轨迹面板，没有新完整算法训练 | 具体训练信号/动作规则无增量 | 不复活同网格；未触发A1/B/C不标负 |
| current-preconditioned / A_MATCH | 单步、五seed短反馈 | 局部正作用可成立，已测持续反馈为负 | B新增任务来源真实Adam反事实；不重做A_MATCH |
| NDP/TV/C01–C05 | 各主臂5seed完整50epoch、Validation筛选 | 当前版本的有效边界已明确，部分条件性正信号保留 | 不改名重跑；新B问题不等于这些增强/重权问题 |
| B所需真实Adam前态 | 历史状态已释放；当前C05最多selected-only | 标量有效，不能恢复m/v | B_CLEAN同源重建10条，40锚点；非新seed |
| 正式发表模型 | 3模型×2数据×5seed合法端点 | 统一比较已完成 | 默认0新训练；只做身份审计 |

所有正式统计单位固定为五个训练seed2022–2026。epoch、六格、anchor、虚拟分支、两个readout、同seed重放、两个随机partition和同轨迹别名均不扩大n。已有Test值作为历史探索背景保留；本次阶段选择仅读Validation，不把反复使用同一Test人口描述成全新独立确认。

## 8. 实际复用登记契约

执行S0输出 `REUSE_MANIFEST.json` 及 `reuse_inventory.csv`。一条唯一训练身份至少由dataset/split、raw/array/row/preprocessing hashes、seed、模型/初始化、完整训练顺序、目标与sampling流、optimizer/history/WD、预算/source数值内核构成；`readout/selected_epoch/model_hash`是其端点身份。online/EMA或E10/E50不创建新trajectory_id。

每行登记以下字段：

~~~text
logical_role, trajectory_id, endpoint_id, dataset, seed, arm, lambda, eta,
source_repo, source_commit, source_path, source_file_sha256, numerical_source_hash,
data_manifest_hash, model_config_hash, init_hash, order_hash, optimizer_config_hash,
readout, split, checkpoint_rule, selected_epoch, model_hash,
evidence_level, test_exposure, reuse_status, physical_parent_id,
ram_owner, owner_generation, current_ram_verified_at, required_state_fields,
missing_reason, next_action, cost_role
~~~

| `reuse_status` | 准入条件 | 允许用途 |
|---|---|---|
| `REUSE_NUMERIC` | 端点及科学身份匹配，原表与来源可核验 | 既有性能比较、配对统计；没有状态权利推断 |
| `REUSE_CURRENT_FULL_RAM` | 当前直接get、全状态字段、hash/generation/事务匹配 | 后续训练或合法锚点；只适用于实际保存的阶段 |
| `REUSE_CURRENT_SELECTED_RAM` | 当前selected状态及选点hash核验 | 冻结selected评价；不得续训或构造历史Adam |
| `NEW_TRAIN` | 正式五seed缺项或阶段门触发的新配置 | 唯一科学新训练 |
| `STATE_REPLAY` | 有数值、无必需状态；明确重建目的 | 从同源初态重放；不增加n |
| `IDENTITY_MISMATCH` | 存在具体人口/配置/选点/内核差异 | 暂停相应复用，记录差异与最小修复；不猜匹配 |
| `NOT_TRIGGERED / COST_BLOCKED / DEFERRED_NOT_RUN` | 原来源确认未执行 | 保留原状态；不得记成实测失败 |

原始CSV：ML `/home_new/xzf/datasets/ml-1m/ml-1m.csv` SHA256 `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`；KR `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv` SHA256 `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。冻结loader使用自然原行序80/10/10、train-only编码/缩放，KR先切分再筛tab0/1/4。数据数组、row ID、encoder/scaler和split身份必须与A-R5等源manifest逐项核对；不能只看raw CSV hash或改绝对路径。

旧prepared目录迁移后缺失属于工程准备问题；可从已核验原CSV按源loader重建并比对，不改变自然人口。不使用旧`/home/xzf/data`默认路径。缺数据、缺依赖、状态不在RAM或遥测失败记TECHNICAL_BLOCKED/PAUSED，不记科学NO_GO。当前状态验收及恢复规则见 [08](08_RESOURCES_AND_SCHEDULER.md)。

## 9. 明确不重复的工作

1. **不重复A0/P01五seed完整训练来重证总收益。** 缺的是P00/P10三seed及条件触发的简单历史对照；状态重放另立目的。
2. **不重跑所有旧P11/λ=.20/EMA02配置。** 未满五seed不等于必须全部补齐；只有影响本轮A主要决策的臂获得预算。
3. **不再证明固定EMA是否有效。** Clean/Gaussian四条件已有五seed，online/F同轨迹、年龄匹配与阶段证据均明确。
4. **不重做S/EMBED删除、recent/old删除或完整H菜单。** 已有作用位置和互补；本轮新增的是任务来源的真实Adam更新及其排序效用。
5. **不重新启动AUG_HISTORY方向/分组ρ/漂移协方差网格。** KR漂移未运行保留事实，不因其未完成自动获得本轮预算。
6. **不复活EMA_ADAPTIVE的同类未来收益预测器和四cell规则。** 过去弱动作价值已记录；B只检验有精确动作定义的新任务来源信息。
7. **不重复原A_MATCH六epoch持续反馈。** 单步正值与持续负值同时保留；本轮脉冲与新持续算法分层裁定。
8. **不重复原尺度coupled-WD的PUB迁移。** 不把对明显退化G的改善包装成算法收益，WD=0建议也非本轮必要缺项。
9. **不再做C_MIX+P01的原CP叠加。** 联合只在A/B各自有增量且互补时启动一套冻结配置。
10. **不重做Gaussian域覆盖或把别名算新实验。** 最终Gaussian补充只回答已晋级组件在冻结D0背景中的独立增量。
11. **不把NDP、TV、AXIS、NQR、EARLY/LATE、W/S/U重新命名后再训练。** 七组已完整五seed；LL、局部阶段或个别数据正值保留，主版本的停止结论不归零。
12. **不重复已完成的M3oE/HyperGate/AutoIFS统一基线训练。** 同身份直接复用；旧LL选点表不充当新表。
13. **不扩新seed、模型、数据集或为理论绝对严谨无限加诊断。** 本轮只解决能改变算法选择的实际缺口。
14. **不围绕“小域不是困难域”提出证明、否定或假设性实验。** 域样本量不能作任务困难度代理。

此清单关闭的是有明确来源和条件的具体重复任务。可靠条件性正结果仍保留；局部机制与完整训练资格分开，未执行与实测负结果分开，不把任何一项旧停止结论扩写成整个优化方向无价值。


## 组合路由补充源码核验

已再次读取 main固定快照下 v3/vendor/ema02/objectives.py 的 losses() 与其 vendor/gradient_aug_20261031_v2/objectives.py 的 losses()，确认Gaussian是目标域原loss部分替换，mix_lambda=.5，返回source pred/real/augmented；06第11节据此冻结组合通道。文件链接：[EMA02目标](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/conditional_information_momentum_20261102_v3/vendor/ema02/objectives.py)、[源替换目标](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/conditional_information_momentum_20261102_v3/vendor/ema02/vendor/gradient_aug_20261031_v2/objectives.py)。
