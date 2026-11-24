# 11 — 来源、协议身份与复用登记

## 1. 证据版本与读取范围

本规划的仓库证据固定于 **`358d3a465747eda6af1d55acc937759fa651a10b`**（2026-11-24读取的main）。后续方案提交产生的新main SHA不改变这份证据基点。已读取的三个上下文入口分别承担索引、设计和数值核对；原始代码及数表按其明确链接补读。`mdmt_research_context/04_HUMAN_RESEARCH_REVIEW.md`未被获取，本研究执行阶段也不读取该文件。

来源应同时登记 `repository / path / commit / git_blob / sha256 / retrieval_date / role`。Git提交用于定位版本，内容SHA256用于核对执行环境真实字节；文件路径相同不等于内容身份相同。历史清单中的旧hash用于追溯旧实验，不能代替本轮实际导入代码的hash。

本次已核对文件及派生文件见 [evidence/source_manifest.json](evidence/source_manifest.json)。它记录**规划期间的来源**；未来执行的 `artifacts/history_information_20261124/SOURCE_MANIFEST.json`还须锁定实际依赖闭包、环境和数据。当前方案没有训练状态可供恢复，也不将旧报告中的RAM收据当作现存状态。

## 2. 主来源表

下表中M表示上述main证据提交。所有相对链接指向仓库路径；严格复核时用M固定读取，避免未来main变化影响身份。

| 编号 | 来源及版本 | 本轮用途 | 复用边界 |
| --- | --- | --- | --- |
| R01 | M：[01索引](../../mdmt_research_context/01_AI_EXPERIMENT_INDEX.md)、[02设计详情](../../mdmt_research_context/02_AI_EXPERIMENT_DETAILS.md)、[03实验结果](../../mdmt_research_context/03_EXPERIMENT_RESULTS.md) | EXP001–042、055–081的设计与证据关系 | 早期四seed、旧人口/预算/选点保持各自身份 |
| R02 | M：[AGENTS.md](../../AGENTS.md)、[start.md](../../start.md) | 资源、当前会话监督、RAM状态和工作流约束 | 采用当前有效要求，不恢复已停止自循环 |
| R03 | M：[EXP080总报告](../../research_reports/conditional_history_shared_update_20261122/FINAL_REPORT.md)、[设计](../../research_reports/conditional_history_shared_update_20261122/DESIGN_SPECIFICATION.md) | A21、A20/A22、选择器关闭、Gaussian身份问题 | 报告是已完成事实，不自动授予本轮Test权限 |
| R04 | M：[A_HISTORY](../../research_reports/conditional_history_shared_update_20261122/A_HISTORY_REPORT.md)、[A_CORE](../../research_reports/conditional_history_shared_update_20261122/A_CORE_REPORT.md)、[A_DOSE](../../research_reports/conditional_history_shared_update_20261122/A_DOSE_REPORT.md) | H0及匹配历史/剂量对照 | 不把已测对照重新列为新增科学轨迹 |
| R05 | M：[S5主表](../../artifacts/conditional_history_shared_update_20261122/S5/main_table.csv)、[配对表](../../artifacts/conditional_history_shared_update_20261122/S5/paired_summary.csv) | 五seed历史Test参照、六格损害、发表模型比较 | 主表均值/SD不替代逐seed新配对统计 |
| R06 | M：`artifacts/conditional_history_shared_update_20261122/tables/A_HISTORY/endpoints.csv`；blob **`d420cdf0043f88b66a6bb886e0c73d72175023a8`** | 6,480条端点的online/EMA、selected/E10/E50验证重分析 | 读取Git blob实际内容；接口返回空content不代表空数表 |
| R07 | M：[history_controls.py](../../experiments/conditional_history_shared_update_20261122/history_controls.py)、[history_runner.py](../../experiments/conditional_history_shared_update_20261122/history_runner.py)、[protocol.py](../../experiments/conditional_history_shared_update_20261122/protocol.py) | 原A21 adapter、无反馈历史shadow、静态PAIR与运行入口 | H0/候选零点使用原adapter；不能根据同名旧类默认值推断动态采样关闭 |
| R08 | M：`experiments/conditional_information_momentum_20261102_v3/`下的[optimizers.py](../../experiments/conditional_information_momentum_20261102_v3/optimizers.py)、[objectives.py](../../experiments/conditional_information_momentum_20261102_v3/objectives.py)、[boundary_sampler.py](../../experiments/conditional_information_momentum_20261102_v3/boundary_sampler.py)、[protocol.py](../../experiments/conditional_information_momentum_20261102_v3/protocol.py) | BCE/PAIR原计算图、A21减法数值路径、事件时钟、静态采样 | 原helper的完整依赖需要由07的闭包检查恢复 |
| R09 | M：同v3下 `vendor/ema02/data.py`、`evaluate.py`、`protocol.py` | 自然行序、train-only编码、概率AUC、完整评估 | 不因诊断logit margin改正式AUC的概率排序与ties |
| R10 | M：[A_CORE_SOURCE_MANIFEST.json](../../artifacts/conditional_history_shared_update_20261122/A_CORE_SOURCE_MANIFEST.json)、[reuse.py](../../experiments/conditional_history_shared_update_20261122/reuse.py)、[state.py](../../experiments/conditional_history_shared_update_20261122/state.py) | 历史源码谱系与状态/数值身份检查 | 其历史v3引用`875215c1e509ab55c1b6632d4f5d13e8b94782da`，并非本轮所有current-main文件的hash |
| R11 | M：[统一发表模型报告](../../research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md) | M3oE/HyperGate/AutoIFS有效自然人口比较与模型原生配置 | main只归档报告，支撑代码/CSV在原来源提交；见第3节 |
| R12 | M：[前一日多机制规划](../../research_plans/26-11-23_multi_mechanism_diagnostics/02_MECHANISM_SELECTION_AND_HYPOTHESES.md) | 防止把相邻规划或候选误写为已完成实验 | 这是方案，不是新增性能证据；不启动其宽矩阵 |

R06已产出 [validation_interaction_seed.csv](evidence/validation_interaction_seed.csv) 和 [validation_interaction_summary.csv](evidence/validation_interaction_summary.csv)，分别1,050和210行；完整定义与结论见01。没有为这次重分析启动训练。复制的S5参考数据见 [confirmed_reference_metrics.csv](evidence/confirmed_reference_metrics.csv)，保持原列与比较身份。

## 3. 发表模型的固定来源与原生配置

### 3.1 来源提交与代码入口

统一发表模型的原分支为 `research-plan/26-11-03-overnight-ab-auc`。本方案使用固定提交 **`c8675ae2923c15b6965884b140b5b646690cd95e`**，不把可移动分支名作为执行锁。规划期间已核读下列源码：

| 路径（相对仓库根） | 固定Git blob | 作用 |
| --- | --- | --- |
| `research_plans/26-11-03_overnight_AB/protocol.json` | `44a5238ecbf7fad0970e9f605cdac1824cb727f7` | 模型、自然数据、选点、原始哈希及阶段配置 |
| `experiments/overnight_ab_20261103/config.py` | `377a928b5f1e6d94a9bb445bf2db3581c3dda8f3` | 按家族闭包身份、job配置 |
| `experiments/overnight_ab_20261103/published_models.py` | `552795bc375456c409d79c163f496c2a5001c060` | HyperGate/AutoIFS真实adapter构造 |
| `experiments/overnight_ab_20261103/published_training.py` | `746cfd22831273ade12066364a620b27c9662b36` | 原生objective、Adam组、AutoIFS阶段事务 |
| `experiments/overnight_ab_20261103/data_bridge.py` | `3a0cb9934e8fb092ed447dac4b177d4ebec6d8a1` | 同自然bundle、行ID、只读cache桥 |
| `experiments/overnight_ab_20261103/vendor/sota2526/SOURCE_MANIFEST.json` | `4e2c06e36da5e95877783157e0260f6be98a456c` | vendored adapter和外部上游文件hash |

固定入口：[原protocol](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_plans/26-11-03_overnight_AB/protocol.json)、[published_training](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/experiments/overnight_ab_20261103/published_training.py)、[外部依赖清单](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/experiments/overnight_ab_20261103/vendor/sota2526/SOURCE_MANIFEST.json)。正式复用还需从同提交核对 `research_reports/2026-11-03_overnight_AB/A_REUSE_AUDIT.csv`、`ENDPOINTS.csv`和对应选点身份，不能只截取报告均值。

该源码闭包依赖 `experiments.mdmt_history_information_20261103`及其vendor，以及原v3的运行家族；只下载`overnight_ab_20261103`目录不能视为完整恢复。外部RecStudio/AutoIFS目录不存在时，按清单固定提交和文件hash恢复，不改成最新版本。

### 3.2 模型原生身份

| 模型 | 论文与来源 | 本轮保留的已冻结配置 |
| --- | --- | --- |
| M3oE | SIGIR 2024，[DOI](https://doi.org/10.1145/3626772.3657686) | fixed1，embedding16、4expert、FCN[512,256,256,64]；batch40960、lr=.01、50epoch；数据别coupled WD；本轮B0即匹配骨干 |
| HyperGate | WWW 2025 Companion，[DOI](https://doi.org/10.1145/3701716.3715236)；上游`fdc82e3f3feee23694a8123f91092615f977900e`，RecStudio `f08cca607e998e8be4c5e915a393346baffa71f6` | embedding16，4expert，expert[256,256]、gate[256]、tower[256,128]；dropout.3、无BN、三门开启、augmented embedding；原domain/task辅助系数均.1；batch1024、eval batch512、lr=.001、WD0、50epoch |
| AutoIFS | WSDM 2026，[DOI](https://doi.org/10.1145/3773966.3777992)；上游`65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e` | embedding16、MLP[1024,512,256]、无dropout/BN，rank ML2/KR16；batch/eval4096，search10、reuse50；search base lr=.0003/WD3e−5、gate lr比.01/WD0；reuse fresh Adam lr=.0001/WD3e−4；hard ticket、冻结gate与原修正接口 |

三模型保留原生目标与训练阶段；共同β为(.9,.999)、eps=1e−8、FP32，AMP/TF32关闭。HyperGate超参来源在原报告中标记为 `UNVERIFIED_PROPOSAL`：本轮将其作为**仓库已冻结的发表架构实现**比较，不改写成作者最优调参认证。

HyperGate的base来源是原 `model.objective(x,y,mask)['loss']` 的梯度，含原生辅助目标；迁移PAIR只添加一个明确新因素。AutoIFS保留最后search epoch前的rewind、最终selector移植、hard ticket、gate冻结与reuse的新optimizer；E10/E50指reuse阶段。对应完整代价由06/10按模型实际batch和阶段计数。

**不能整包沿用旧protocol的运行政策。** 旧文件的固定worker数、预测显存余量和旧Test流程属于其原实验；本轮采用00/08/10。`data_bridge.cached_bundle(root,dataset,include_test=True)`有默认Test入口，新训练必须显式`include_test=False`；其`prepare_cache`原来准备全部split，新实现需将Test准备/载入放在独立权限门内。旧Gaussian组合的KR target为d2，也不能覆盖本轮规定的GA31-R3 D0。

HyperGate迁移中的PAIR历史核与EMA使用06规定的每步参数直接迁移：βP=.99（被选S代表为.9），EMA采用对应数据的固定ρ，不再引入样本时钟映射搜索。不同batch下的有效样本跨度单列，四个HyperGate臂相互匹配。

### 3.3 GA31-R3 D0组合的可恢复来源

本轮主动选择既有批级GA31-R3协议，区别于EXP080未能配齐的EMA02逐行键版本。已核读[原D0覆盖报告](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md)、[G_CONTROL独立身份审查](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/research_reports/conditional_history_shared_update_20261122/G_CONTROL_REVIEW.md)和[SOURCE_G_CONTROL_AUDIT.json](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/artifacts/conditional_history_shared_update_20261122/SOURCE_G_CONTROL_AUDIT.json)。审查给出原始噪声fixture、人口字段与固定源码SHA；可用于本轮身份恢复，不继承原研究队列与权限。

已进一步核读以下`c8675ae…`代码：

| 路径 | Git blob | 已确认内容 |
| --- | --- | --- |
| `experiments/gaussian_domain_coverage_20261104/objectives.py` | `06c3183de96ec08c58510ddce04b94600779d4c2` | KR D0覆盖、原全batch损失分母；其dataset硬编码只适用KR |
| `experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/objectives.py` | `c4b58d0e230c03bfe69f8822ddac818d332d5ed1` | 通用dataset入口、一个Gaussian view、.5替换权重 |
| 上述同目录`views.py` | `ecc354eba1c1d1e8b21694e8dbfb3c2b7221d254` | 本地torch.Generator逐字段draw、detached逐行字段RMS、domain排除 |
| 上述同目录`protocol.py` | `7e59458c3473717f9c07e48dca984e2c19380bd7` | GA31-R3键、前8bytes小端seed与63-bit mask；只复用该数值helper |

精确数学、字段顺序和同clean图组合接口已写入07§7.2。历史首批噪声可以拒绝错误版本；无需再次完整50epoch证明两种噪声流不同。新四臂若缺状态则按本轮授权矩阵恢复，并保持GA31-R3同一身份。

## 4. 数据身份

下列CSV路径及SHA256来自已冻结自然协议。执行S0在实际文件上计算SHA256并核对；不能以路径存在代替校验，也不将“旧prepared目录缺失”认定为原CSV缺失。

| 数据 | 路径 | 原始CSV SHA256 | train域计数 |
| --- | --- | --- | --- |
| MovieLens | `/home_new/xzf/datasets/ml-1m/ml-1m.csv` | `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4` | 155949 / 318599 / 325619 |
| KuaiRand | `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv` | `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f` | 172782 / 1743898 / 89452 |

同一CSV哈希之外，还登记行序/split-row-ID hash、train-only类别字典与OOV策略、域映射、任务标签、特征顺序、batch/尾批、shuffle流、输出概率转换和metric版本。数据源真有变化时创建明确的新人口身份，不与本方案冻结人口的旧数值直接配对。

## 5. 复用决策与逐项登记

每项复用记录写入 `REUSE_REGISTER.csv`，至少包含：

```text
reference_id,canonical_trajectory_id,attempt_id,dataset,seed,arm,readout,
endpoint,split,epoch,source_commit,source_hash,config_hash,data_hash,
preprocess_hash,row_population_hash,selection_protocol,model_hash,
numeric_source,numeric_sha256,numeric_available,predictions_available,
ram_receipt_id,ram_get_verified,optimizer_available,shadow_complete,
reuse_status,rebuild_reason,logical_trajectory_count,physical_run_count
```

| 当前持有的证据 | 可执行工作 | 不可从中推得的状态 | 后续动作 |
| --- | --- | --- | --- |
| 匹配的逐seed端点与完整身份 | 历史差值、CI、六格统计、正式数值参照 | 当前模型/optimizer仍在内存 | 直接复用数值，`NUMERIC_REUSE` |
| 原预测及其模型/人口hash | 同预测人口的纠正/破坏、支持重分析 | 任意新输入或未保存epoch的预测 | 只分析已有预测，登记端点覆盖 |
| 真实可读回并hash一致的selected RAM | 冻结后获授权的相同模型Test、读出检查 | 缺失的历史optimizer和D2影子态 | 对应功能复用，`LIVE_STATE_REUSE` |
| 完整模型+optimizer+RNG+loader+影子态RAM | 同状态诊断、精确续训 | 不同轨迹的无偏替代状态 | 核对07合同后复用，`FULL_STATE_REUSE` |
| 只有旧RAM收据或最终v/Q | 记录旧实验已做事实 | 状态存活、过去D_s或缺失U/vsep | `STATE_UNAVAILABLE`；按需一次父轨迹重建 |
| 人口/选点/源码关键身份不匹配 | 历史背景与边界说明 | 本轮正式匹配参照 | `HISTORICAL_NONMATCHED`，从主比较剔除并说明 |

一条轨迹由 `dataset+seed+model/source+objective+algorithm/config+data/preprocess+random_stream_protocol`唯一确定。技术重放另记attempt，不增加独立seed；重复收集online/EMA和多个epoch也不增加轨迹。需要重建时从同一冻结初始化及全部随机流重放，不择优挑attempt。

B0/B1/H0先查原五seed数值和真实可用状态。父状态缺失时最多按06重建各一次，并在同一次重建收集所需曲线、shadow、selected和共同epoch。H20/H22只做历史复用。正式HG/AutoIFS数值若身份完整则不为“再确认”重训；确实不匹配时，仅重建受影响的发表基线槽位并计入基线恢复预算。

## 6. 已关闭重复路线

| 已有回答 | 本轮直接采用的结论 | 不再新增的同目的工作 |
| --- | --- | --- |
| 固定参数EMA、年龄匹配、Embedding/共享组、近期/较早窗口 | 历史参数信息具有实用排序价值和阶段互补 | 重复证明EMA有效、重复组/窗口归因 |
| A20/A22、低PAIR剂量、旧条件历史 | 条件数/普通长核不自动改善原A21 | 再以这些旧消融作为C1/C2前置门 |
| TV-GRM raw残差历史、shuffle/instant、动态源与剂量 | 这些具体定义未给强自然训练稳定增量 | 改名重跑原机制、以局部正向续命 |
| 旧采样、无动量SGD与SCALE映射 | Adam历史并非所有采样效应的必要原因；尺度含WD/eps作用 | 重新寻求普遍必要性、重复同一尺度恒等式 |
| EMA反馈、已测KD/函数组合、复杂预测器 | 已测形式不足以形成稳定额外排序收益 | 在本轮无新定义地重开teacher/动态ρ宽搜索 |

这些关闭规则针对已测定义；C1的二阶来源耦合和C2的历时预条件组织改变了实际运算，因此按06完成新完整训练。LaProp、AdaTask等相似思想作为明确对照，原始文献及迁移关系详见02，不要求绝对原创性。

## 7. 技术前提未就绪时的唯一处理

| 未就绪项 | 影响范围 | 应完成的恢复 | 其余工作 |
| --- | --- | --- | --- |
| 原自然CSV/预处理身份 | 对应数据的正式比较 | 恢复并核对相同原CSV及train-only转换 | 另一数据与文献/代码工作继续 |
| v3/vendor闭包缺文件 | 依赖该原生入口的臂 | 沿已登记imports和固定清单补齐 | 不替换成近似重写或最新外部代码 |
| HG/AutoIFS上游目录缺失 | 该基线/迁移模块 | 从manifest固定上游恢复并验证文件hash | 主M3oE核心继续 |
| 原RAM消失 | 需要该状态的诊断/评价 | 登记缺失并在冻结预算内重建 | 数值复用与不依赖状态的分析继续 |
| GA31-R3精确噪声/覆盖身份不可恢复 | Gaussian四臂 | 定位原版本与首/尾批噪声身份 | 先完成clean全部结论，模块记`TECHNICAL_PAUSED` |

技术暂停不写成科学负结果；未触发模块写`NOT_TRIGGERED`并给08中的具体规则。恢复不自行改变科学问题、seed、数据分布或评价门槛。
