# 第二阶段：已发表模型统一比较执行方案

本阶段在第一阶段完整五seed汇总后，冻结候选名单再实施。当前交付为执行方案；本轮 GPU 矩阵不包含本阶段新增模型训练。

## 1. 模型资格、现有实现及来源

| 正式模型 | 发表来源 | 复用实现 |
|---|---|---|
| M3oE固定融合适配 | SIGIR 2024，[DOI](https://doi.org/10.1145/3626772.3657686) | 近期source中的fixed1 MDMTRec；所有fusion系数1，不启用DARTS |
| HyperGate | WWW 2025 Companion，[DOI](https://doi.org/10.1145/3701716.3715236) | `experiments/mdmt_sota_baselines_2526/hypergate.py`；上游fdc82e3f3feee23694a8123f91092615f977900e，RecStudio gitlink f08cca607e998e8be4c5e915a393346baffa71f6 |
| AutoIFS | WSDM 2026，[DOI](https://doi.org/10.1145/3773966.3777992) | `experiments/mdmt_sota_baselines_2526/autoifs.py`；上游65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e |

官方来源本机 `/tmp/mdmt_sota_sources_2526/`，执行前按现有 `sources.py` 核对commit。采用10/29已审查的核心模型适配、必要bias修正与mechanism tests；不重写网络。HyperGate现有部分超参来源为UNVERIFIED_PROPOSAL，冻结使用仓库适配配置并如实标注，不能称作者原始参数。M/H/C_MIX/P01、固定EMA、Gaussian变体等均列候选或内部控制，不列正式论文baseline名单。

## 2. 为什么不能直接用10/29绝对值排名

发表论文全名分别为 *M3oE: Multi-Domain Multi-Task Mixture-of Experts Recommendation Framework*、*HyperGate: Hierarchical Perceptive Gating Network for Multi-domain Multi-task Recommendation*、*Automated Information Flow Selection for Multi-scenario Multi-task Recommendation*。发表记录见上表 DOI；AutoIFS 另有[作者机构出版记录](https://scholars.cityu.edu.hk/en/publications/automated-information-flow-selection-for-multi-scenario-multi-tas/)。

| 项目 | 10/29正式比较 | 本次补全及拟统一协议 |
|---|---|---|
| train人口 | T0∪F；ML各域134116/273995/280032，KR148591/1499752/76928 | 完整自然train；ML155949/318599/325619，KR172782/1743898/89452 |
| 预处理拟合 | 只T0 | 只完整train |
| 选点 | V macro LogLoss最小；M3仅epoch12–24可选 | validation六格macro AUC最早严格最大，E0排除 |
| 测试 | 历史E | 同近期原始80/10/10的test；原行号可能与E一致，但编码和train人口不同 |
| M3训练 | lr=.001、24epoch、wd=1e−5 | lr=.01、50epoch、wd ML1e−4/KR1e−5 |
| HyperGate | batch1024、max1000、patience10 | batch1024、固定50epoch；按AUC选点 |
| AutoIFS | batch4096、search10、reuse最多50且patience1 | search10保留，固定reuse50；按reuse validation AUC选点 |
| 输出/指标 | logits，稳定softplus LL | 用近期概率评价函数统一六格、macro、traffic、pooled；模型输出logits时先sigmoid |
| 状态 | 旧training.py会torch.save | 新执行状态只驻RAM |

历史模型比较已经真实完成五seed，科学结果保留。由于人口、编码、训练和选点有实质差异，不能把旧HyperGate .766825/旧M3 .759446与新候选约.78横排宣布胜出。新协议下HyperGate/AutoIFS均需新训，旧30条不能替代。

## 3. 统一人口、选点、评价与预算

所有模型与候选使用第一阶段同一source loader返回的 `train_x/y`、`val_x/y`、`test_x/y`、schema及原始row_ids。不得调用旧baseline `prepare_owned()` 重新生成T0∪F人口，也不需要恢复遗失的mdmt_next prepared目录。特征值、领域、任务、自然比例、OOV和scaler由这一份bundle决定。向HyperGate/AutoIFS转换schema/张量布局时不重新编码，不删除原有输入字段或添加标签特征；domain字段用于模型原本的域条件输入。

主指标与第一阶段一致：每个模型合法validation-selected test六格等权macro AUC；同时报告六格AUC/LL、macro、traffic、pooled、worst及domain/task。HyperGate/AutoIFS训练目标继续使用其原生logits目标，评价统一通过近期 `metrics/evaluate` 语义；不得混用raw logits AUC与另一个模型经过sigmoid产生的饱和tie。选点状态仅由validation决定，全模型/全seed配置与端点冻结后统一test。

所有模型FP32，AMP/TF32关闭，同一项目Python环境。自然无放回shuffle，尾批不丢弃，seed只用2022–2026。M3使用近期固定训练；HyperGate/AutoIFS保留现有model config中的lr、Adam、wd、batch、模型辅助目标，不强行把50个epoch等同为相同更新数。

- M3oE fixed1：batch40960、50epoch、lr=.01、coupled wd ML1e−4/KR1e−5，普通online，无新增EMA。可复用经第一阶段身份审计的A0 online五seed。
- HyperGate：10/29 config的embedding16、4专家、三类门控、增强网络、detach、SENet、专家bias、辅助loss及Xavier不改；batch1024、lr=.001、wd0，50epoch完整预算，无early stop。选点窗口epoch1–50。
- AutoIFS：10/29 config的embedding16、rank ML2/KR16、共享MLP/LoRA/selector不改；batch4096；search10，search_lr=.0003/search_wd=.00003，gate_lr_ratio=.01，temperature1→100；零基epoch9进入前rewind；最终selector及gate embedding移植；hard-ticket/reuse；reuse_lr=.0001/reuse_wd=.0003，完整reuse50，无early stop。主端点仅reuse1–50选择，search validation仅日志，不用于另选结构。总训练人口暴露为60epoch；search两个Adam调用分别计数。

共同阶段报告：M3/HyperGate E10/E50；AutoIFS对应reuse E10/E50并单列已付出的search10。AutoIFS search epoch10不冒充主训练E10。训练量、optimizer调用、标签暴露、参数量、显存/功耗和worker秒完整列出。该表比较统一数据和统一选点下保留必要训练流程的适配方案，更新预算差异是具体成本信息。

当前不安排超参搜索。若后续要比较调优上限，需另立固定验证预算，所有方法同等获得权限，不从本次test决定参数。

## 4. 候选入选与实际矩阵

以五seed完整结果冻结 `(dataset, method, config)`，按同seed普通A0 online的总收益裁定。总宏AUC≥.003的候选进入正式比较；逐格各均差≥−.001者标记满足约束，越线者保留“宏AUC达标、逐格约束未满足”，在表中逐格列出代价。组件增量是否达到.003不另作候选硬门；LL单独裁定。不能要求所有seed同向，不能以CI跨零拒绝。

Gaussian＋EMA至少已在ML总收益达标且逐格满足，应列候选；KR总收益达标但d0t0越线，应列带代价候选。C_MIX/H_MIX/P01/M按第一阶段实测结果追加符合条件的固定配置，不另搜参数。未达总体AUC标准但LL下降的配置保留为LL结果，当前AUC主比较不优先。固定EMA本身可作为内部参照行附表，不能冠以正式发表模型baseline名称。

若两个数据都有入选候选：正式模型逻辑矩阵为2数据×3模型×5seed=30条，M3 online已可复用10条，新增HyperGate/AutoIFS20条。第一阶段完成的候选五seed符合相同统一人口/预算/选点者直接复用，新增候选训练0。若未来新增适配改变了任何候选训练或评价身份，则另列协议，不能把旧指标移植至新模型。

发表模型之间及每个候选相对三模型都输出同seed差。主研究判断以普通M3配对总收益为准，同时报告相对HyperGate/AutoIFS的比较；是否三者均满足AUC标准必须逐项说明，不能把只胜M3称为全面优胜。没有新的4/5或CI门。

## 5. 给后续Codex的直接实现任务

第一阶段已完成：30条新增训练、10条历史复用、180个新增test端点，零失败。按固定EMA selected主端点，冻结以下实际名单：

| 数据 | 总AUC与逐格约束均满足 | 宏AUC达标但有逐格代价 | 当前AUC主比较不优先 |
|---|---|---|---|
| MovieLens | C_MIX、H_MIX、P01、Gaussian＋EMA | — | M：总ΔAUC+.002956435，LL改善保留 |
| KuaiRand | P01、H_MIX、C_MIX、GATS B2＋EMA | Gaussian＋EMA：d0t0−.001271286 | M：总ΔAUC+.002534206，d0t0−.002767112，LL改善保留 |

KuaiRand P01相对固定EMA的组件宏AUC+.004085725，但d1t0/t1为−.001333993/−.001047387；整体方法相对普通训练六格均提升。正式模型比较使用完整配置的总收益资格，组件比较保留“宏AUC达标、逐格约束未满足”。完整结果见[正式报告](../../research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md)和[机器裁定](../../research_reports/2026-11-03_mdmt_five_seed_completion/DECISIONS.json)。本阶段待新增HyperGate/AutoIFS共20条；M3及候选直接复用。

五seed历史结果按新规则重算还应纳入 **KuaiRand GATS B2＋固定EMA**：相对普通A0 online，总ΔMacro AUC **+.006097847**、ΔMacro LogLoss **−.003002719**，六格平均ΔAUC均为正（+.002766/+.014624/+.005044/+.005339/+.002483/+.006331），满足总体收益与逐格约束。相对A0固定EMA的增量为+.001481447/−.001777377；相对Gaussian＋EMA为+.000005156/−.000113784。该配置具备进入正式比较的整体收益，投影组件相对Gaussian的排序增量接近零。五seed普通/固定EMA控制的selected模型hash与AIGM01逐一一致，逐seed重算见 `research_reports/2026-11-03_mdmt_five_seed_completion/GATS_B2_HISTORICAL_PAIRED.csv`；来源是[GATS冻结报告与原数表](https://github.com/Caelestis1903/My-Research-Project/blob/e59efa2fd954c4d2231b679a742406cfbe2a1077/research_reports/2026-11-02_gradient_aug_time_select/REPORT.md)。复用已完成五seed，不增加训练。其跨数据集覆盖记录为ML不足五seed，不补列ML正式收益。

1. 将本协议和第一阶段入选名单冻结为一个新实验config，不修改历史default_config（旧config.py固定T0/F人口和V选择，无法只改路径获得新协议）。
2. 复用 `models.build_model` 中HyperGate/AutoIFS adapter及官方source装载，增加与近期bundle对应的schema适配。M3优先直接沿用第一阶段source构建器，使可复用训练身份保持一致。
3. 复用旧 `training.py` 的HyperGate训练循环及AutoIFS `_fit_autoifs` 算法步骤；只调整数据接口、validation AUC比较、固定预算和RAM状态交接。旧 `_write_checkpoint` 明确写磁盘，不能调用；用第一阶段guardian内存方式承载rewind/selected/E10/final。
4. 保留AutoIFS rewind、selector transplant、hard-ticket及gate/base optimizer归属已有检查。新增只测schema等值、真实一batch backward、同bundle评价公式和内存选择格式。无须重做已核验网络机制证明。
5. coder提交实现报告，由独立reviewer检查核心公式未变、人口/预处理/选点一致、测试不参与选择、禁存权重。通过后runner完成两个模型全部10seed训练，保存标量、阶段预算、运行失败/恢复、资源与成本，最后全局freeze→test→release。
6. 主agent按seed汇总原值/SD/配对t区间、六格和total/component结果，生成正式模型比较表和独立内部参照表，更新main台账的新协议分组，保留旧协议记录与来源链接。

本阶段可独立停止于全部冻结矩阵完成及结果交付，不以中途1–4seed成绩取消其余训练。若发现影响数学、训练或评价的具体错误，则修复对应错误并按同身份恢复/重跑；不增加无关ablation。
