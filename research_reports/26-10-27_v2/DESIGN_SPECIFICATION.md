# DESIGN SPECIFICATION — MDMT Relation-Basis v2

日期：2026-10-27。状态：DESIGN_COMPLETE；实现与训练须在独立 coder / reviewer 流程后执行。本文只定义当前实验，不记录尚未发生的测试或结果。

## 1. 权威与实施解析

已读取根目录 AGENTS.md、AGENT.md、GB18030 编码的 start.md，以及 research_plans/26-10-27_v2/ 全部 11 个文件。算法、接口、决策、默认数值分别依从该目录 03、04、05、07。审计发现 v2 未把共同前缀预算、角色的具体旧分割映射、强直接对照架构和容量匹配写全；以下解析在训练之前一次锁定，不能随结果改动。

共同前缀、源表示和数据角色继承远端已锁定 v1 规范；本地可追溯参考为 V1_EXECUTION_CONFIG_REFERENCE.yaml、V1_SOURCE_AUDIT_REFERENCE.json，远端路径 research_plans/2026-10-27/{02_核心算法数学与机制规范.md,03_Codex实现技术规范.md,07_EXECUTION_CONFIG.yaml,09_SOURCE_AUDIT.json}。v1 的动态分裂、约束投影、额外方法与旧 B offset 不继承。v2 20 epoch / 2048 batch 的统一适配预算覆盖 v1 的 reference 特有预算。

用户最新指令固定 seed 为 2022、2023、2024、2025。它覆盖规划 07 的 310x / 610x：开发为 2 数据集 × 6 方法 × 4 seed = 48 条适配轨迹，共同前缀为 8 个；不使用其他训练 seed。原“两个开发 seed 均一致”的局部条件提升为全部四 seed 一致。正式模型若与开发同 seed、同完整方法身份及训练协议，则复用实际完成的轨迹，记录为冻结后的同轨迹历史测试复评，不重复计作独立训练重复。阶段 2 的新配置若触发，仍使用这四个 seed。

start.md 第 14 条禁止本地保存模型、optimizer、checkpoint 文件，优先于示例目录和远端必要 checkpoint 建议。模型、优化器和完整恢复状态只存在 RAM；可以保存数据缓存、逐行预测、指标、日志、JSON 身份摘要与报告。不得把权重改后缀、写临时文件、写 RAM 文件系统，或作为 JSON 数组绕过此限制。

## 2. 最小复用路径与修改范围

只新增 experiments/mdmt_relation_basis_v2/、artifacts/mdmt_relation_basis_v2/ 和 research_reports/26-10-27_v2/。规划原字节保留；DELIVERY_RECEIPT 只可追加实际执行状态。历史实验、已有源包和原始数据不修改。

本地可复用的实际路径：

| 现有来源 | 可复用部分 | 不直接复用部分 |
|---|---|---|
| experiments/mdmt_dfd_20261026/native_adapter.py | fixed1 MDMTRec 机械适配、原生 logits、四专家 LayerNorm 后拼接 h256 | 不把 shared mean64 当作 h |
| experiments/mdmt_dfd_20261026/functions.py | ResidualMLP / PilotBank 的六个 owner-only 256→32 ReLU→1 源监督函数 | 旧 FunctionBank 的 target-intent 复制与形成目标 |
| experiments/mdmt_dfd_20261026/vendor/strict_data/data_roles.py | 已登记 raw SHA、自然行顺序、特征解析、fit-only 编码和 assign_roles / prepare | 旧路径默认值和旧用途权限 |
| experiments/mdmt_dfd_20261026/vendor/pinned_fixed/ | 固定融合实际网络与依赖闭包 | learned fusion / AutoML 更新 |
| experiments/mdmt_rcf_20261026/state.py | 完整 RAM 快照的 capture / restore 模式 | 无磁盘模型持久化 |
| experiments/mdmt_rcf_20261026/prefix.py | native / pilot 训练顺序、源隐藏标准化、RAM 生命周期模式 | 整个 build_prefix 会多训 local pilots、构造旧 B，不直接调用 |
| experiments/mdmt_dfd_20261026/plans.py | SeedSequence 按 dataset / component / epoch 分流的既有模式 | 旧开发 seed、旧矩阵和运行根目录 |

代码复用须由 orchestrator 的 SOURCE_MANIFEST 固定实际 commit、远端路径、Git blob SHA、版权/许可证和依赖；本地目录没有 .git，因此本地文件不能自动视为当前 main。源码身份不匹配先核对实际差异，不能伪填来源。必要的 vendor 保持相对包层级，strict_data/artifacts.py 的父 state 依赖也要包含。无须复制旧 CLI、旧 pipeline 或整条历史分支。

现有 DFD metrics.py 使用概率 clip 后 LL；它与本次稳定 logit LL 不同，不能作为主指标直接调用。保留其输出字段惯例，主指标按第 7 节实现。

影响面包括新 config/data/native/source_functions/relation/model/losses/trainer/diagnostics/report/cli 及 tests；训练入口、数据准备、RAM selected/final 状态、freeze/evaluate 调用者都必须使用同一有效配置。无新增依赖，使用当前项目环境中已安装的 torch / numpy / pandas / scipy / sklearn / PyYAML。

## 3. 数据身份与角色

raw 路径使用 AGENTS.md 指定迁移后的绝对路径：

| 数据 | raw SHA-256 | 行数 | 标签 |
|---|---|---:|---|
| MovieLens | f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4 | 1000209 | rating>3、rating>4 |
| KuaiRand | a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f | 2507666 | 实存 is_click、long_view |

两任务保持同一交互 row_id，不把 long_view 乘 click。domain 映射、特征列表与数据类型严格采用已登记 strict_data 源码；不改样本比例、不增强、不重采样。原始 CSV 0-based 行号为 row_id。自然前 80% 外层行中按既有 20261024 / 1009 域步长计划产生 fit / probe / query；validation 为自然 80–90%，test 为最后 10%。P_label 按 DFD 既有 20261025+60000+1009*d 源内排列等分 P_form / P_read。该精确计划不是重新抽分割。

| v2 角色 | 旧实际行集合 | 权限 |
|---|---|---|
| T0 | fit | native / source pilot / 编码器 / 源标准化器拟合 |
| F | P_form ∪ P_read，即原 P_label | 新适配模型自然标签 BCE |
| S | P_cal ∪ P_update | 来源保持与结构诊断，不选 checkpoint |
| Q | Q_explore | 开发判定、固定状态功能诊断，禁止 optimizer |
| V | validation | 全模型 checkpoint 的 macro LL 选择 |
| E | test_exposed，即旧 test | 全局冻结后历史测试复评 |
| legacy | Q_legacy_confirm_exposed，即旧 Q_confirm | 记录身份和历史暴露，不加入本轮选择/训练 |

预期各域计数继承 V1_EXECUTION_CONFIG_REFERENCE.yaml 的 data.*.counts；所有角色记录排序 row_id 哈希、每域任务 n / positive / negative。角色必须互斥且覆盖全部行（含 legacy）。编码器只在 T0 拟合，OOV=0；年龄缩放只用 T0；所有任务 mask 当前为 true，模型仍支持缺失 mask。mask=false 的标签值不参与 loss，也不进入有效标签分母。

prepared 数据缺失可按核验后的 pinned strict_data.prepare 从原始 CSV 重建，数据文件写本次 artifact root 的 data 子目录，不修改 /home_new 数据。重新核验实际 raw / schema / row-plan / encoded-array / preprocessor 身份后进入训练。缺少 raw 或 SHA 不符是阶段 0 阻塞；仅缺可确定重建的 prepared 缓存不是停止理由。

## 4. 共同 native 与来源函数重建

每 dataset × seed 一份共同前缀，供六臂共享。native 为固定 exp_d=exp_t=bal_d=bal_t=1 的 fixed1，三域两任务、4 shared experts、fcn_dims=[512,256,256,64]。按既有 original 初始化再复制非 _weight_ 参数的方式产生初始 backbone；不恢复 learned fusion、AutoML 或额外 optimizer。训练 T0 自然两任务平均 BCE，24 完整 epoch、batch40960、保留尾批，单 Adam(lr=.001, betas=.9/.999, eps=1e-8, coupled weight_decay=1e-5)，FP32，无 AMP/TF32。V macro logit LL 在 epoch12–24 选完整状态，1e-12 内取最早。完整 24 epoch 的成本都记录。

native encode 输出 z0[B,2] 和 h[B,256]；h 是四个64维 shared expert 各自无仿射 LayerNorm(eps=1e-5) 后顺序拼接，保留真实域路径。冻结 native 后用 h/z0 拟合六个来源/task PilotBank 头256→32 ReLU→1，输出层零初始化；每头只用自己的 T0_s，训练 z0_t+pilot_residual 的稳定 BCE。12 完整 pass、batch40960、Adam 同 native 数值，固定 pass12，不用 Q/V/E 选 source pilot。

source s 的 phi64 为两个 task pilot 的 ReLU 隐藏32按 task0、task1拼接，再在 T0_s 计算逐维 population mean/std，下限 .001；不 clip。六个 source 隐藏层共49344参数，原 output 教师共198参数。按 v1 代数把 source output 权重/偏置转换到标准化 phi 坐标，保留完整教师 z0_t+c_st+v_st^T phi_s；在固定 T0 样本验证转换前后 max abs logit diff≤1e-6。教师只用于来源诊断，主共享项不用其标签或输出层。

不训练旧 reference 作为共同 B offset：v2 主预测明确是 z0+g+source。B_REFERENCE 是下节单独适配臂。无需训练两个额外 local pilot bank，也不构建旧 RCF/DFD 方法。所有共同前缀成本按实际唯一身份记录一次，各完整方法部署/训练成本另列含前缀视图。

## 5. 六个方法的精确实施身份

统一输入为冻结 h256、z0 两logit、真实 domain、phi[B,3,64]。所有残差末层 / 读取向量零初始化，初始最终预测精确为 z0。g_dt 为六个独立头，只有实际域对应头参与该行两任务输出。

| 方法 | 精确定义 | 新增可训练参数预期 |
|---|---|---:|
| B_REFERENCE | 继承 v1 强 reference 架构：六个258→32 ReLU→1，残差加 z0；用本次统一预算重训 | 49926 |
| DIRECT | v2 g_dt：六个258→32 GELU→1，残差加 z0，不读来源 | 49926 |
| READ | DIRECT 的 g，加冻结 phi 根函数；不训练 U/V、不需要请求编码器；foreign source/basis 的常数偏置 b_dtsj 可学习，NULL=0；w 每来源跨两个基底共享 | 50718 |
| TARGET_ONLY | 六个258→m GELU→1，无来源路径；m 在训练前由整数宽度最小化与 LOWRANK 新增可训练参数绝对差，平局较小宽度；本合同 m=40 | 62406 |
| LOWRANK_RELATION | 第 03 数学合同，J=2 / residual rank4 / relation rank2，g 与 DIRECT 相同 | 62166 |
| FULL_RELATION | 相同 g、请求编码器、函数库和 w；每个 foreign d,t,s,j 使用独立请求线性系数32与偏置 | 62846 |

TARGET_ONLY 参数匹配对象是本次 LOWRANK 模型，不能继承 v1 的 shared hidden47=12456 后称匹配。实际相差240，即约0.386%；报告实际计数。若 coder 的计数与此表不符，先解决结构差异，不能随意调宽主方案。

请求编码 r=GELU(Linear([h,z0],32))，不增加 one-hot 拼接或目标标准化器。关系低秩是 u_dtsj=b_dtsj+sum_r a_dtr*k_sjr*(C*r)_r；本域位置物理不列入候选，foreign 4项与固定 NULL logit0 softmax。FULL 的 u=b+W_dtsj*r。NULL 固定0是 softmax 可辨识基准，既不删除拒绝读取选项，也不增加未规定的 NULL 网络。r、a、k、C 和 FULL 请求系数采用确定性具名 seed 下 PyTorch 对应 Linear / Embedding 的默认初始化；关系 b 初始0，不额外调初始温度或增补多样性正则。U=0，V 每基底独立 N(0,.02²)，无 bias；w=0。g 隐藏层用同一最近现有模式 Xavier 权重/bias0，末层0，B_REFERENCE 用同模式保持 ReLU。初始化规则在配置/报告固定，实际抽样流与批次计划分离。

计数推导：六头g=49926；request=258*32+32=8288；6基底U/V=6*(64*4+4*64)=3072；12 foreign读取向量=768；低秩关系 a12+k12+C64+b24=112；完整关系 W768+b24=792。关系参数节省680，来源函数执行没有减少。本域 w 和 b 不注册为假可训练参数。READ 的3072个 U/V不执行且不作为可训练计数。冻结 source / native 参数与教师审计参数单列。

本轮不新增 SOURCE_K2。03 的 SOURCE_K2 提及承接旧版来源监督概念，当前 05/07 唯一开发矩阵明确六臂；已有 T0 source pilots 就是来源监督基础。新增第七开发臂会改变预算，当前无必要。

## 6. 训练、状态与冻结生命周期

适配统一 Adam lr=.001、betas=(.9,.999)、eps=1e-8、weight_decay=0；FP32、无 AMP/TF32/dropout/scheduler；batch2048、自然行排列、保留尾批；总20 epoch，epoch1是 warmup，不能再加第21轮。LOWRANK / FULL warmup 只更新 g/w，冻结关系、请求编码器及 U/V；READ warmup 冻结关系偏置。其余方法完成同一完整 F pass 的普通头训练。epoch2–20按方法权限联合训练；同一个 Adam 持续使用，不继承 v1 的额外 warmup 后重启。lambda_src=0，不接 S/T0 梯度；每步投影 w 至L2≤2，不投影 U、不加入 KL/均衡/多样性 loss。

F 的 loss=sum(mask*BCE)/(sum mask)。mask false 在数值计算前排除，避免 NaN 标签经零乘进入 loss。空有效任务 batch不更新参数，单测说明。六方法同 dataset/seed 使用相同 row-order 计划；不把缺域强行补齐。冻结参数 grad=None，不以造零梯度替代冻结。

每 epoch 记录 train loss、V六格指标及成本；V 候选仅5–20，最小macro LL，差≤1e-12保留最早。selected 保存完整 RAM state，final单独保留epoch20。不能因为selected较早而停止后续epoch。完整 state包括模型、Adam、scheduler=null、epoch/global step、Python/NumPy/Torch/CUDA RNG、row cursor/plan identity、effective config identity、源码commit/blob清单、data-role身份、preprocessor/source/native身份。

每个dataset/seed worker保持共同前缀、六方法selected与final及所需恢复快照常驻 RAM。独立方法CLI进程不能训练完就退出并丢弃用于最终评估的模型。可由单个 campaign 进程实现，也可由每prefix常驻worker通过只含指标/身份的文件握手等待全局冻结；不得序列化权重跨进程。Q阶段决策及后续触发阶段完成后，锁定方法清单/全部阈值/配置/selected身份，才统一解锁 E。模型销毁须在合法 selected/final E预测、诊断和raw-X一致性收据全部导出之后。

恢复验收用内存snapshot：同一后续 batch 比较 logits 及至少一次更新后的模型/Adam状态。先核验所有身份字段，错身份拒绝恢复。中断后无磁盘resume；必要时同身份确定性重放，另记实际额外成本。未做过的重放不能记作成功恢复。

## 7. 指标、诊断和成本

LL用FP64 logits的 logaddexp(0,z)-y*z，不clip概率；AUC以原logit排序，单类/无样本为NA。主macro是六格合法LL/AUC算术平均，不以pooled代替；natural是所有有效任务请求的均值；pooled分别按task报告。每格导出 n/positive/negative，selected/final epoch与状态hash。可保存logits与sigmoid概率以复算。报告逐seed、均值、sample SD、配对差及seed级95% t区间；n=4时自由度3，不把epoch、任务、行bootstrap当训练重复。

选点后预注册诊断：

1. 在同一固定 Q panel 全行/两任务比较 FULL 与 ALL_ROOT，仅将 f_sj替为phi，其他参数与归一化保持；报告逐格/宏/自然风险差。至少一个seed上 ALL_ROOT−FULL 的macro LL超过既有数值平局容差1e-12且实际 U 非零，才标记有可观测形成证据；同时完整报告其余seed和差值大小。该布尔条件只执行05的“至少一项形成证据”，不替代PRACTICAL。
2. 在 F 无标签前向求每d,t的alpha均值，固定此均值在Q替代请求关系；不再适配reader，不再选checkpoint。FULL / LOWRANK均导出原关系与常量关系风险差。
3. Q固定行面板报告每source两基底输出RMS距离、U范数、各域/任务关系熵、foreign质量与NULL质量。源保持在S对应真实source上比较教师风险与各基底替代后的风险，phi标准化仍只来自T0。函数差异小于实现数值分辨率记FUNCTION_COLLAPSE；风险干预无变化而路由变化记ROUTING_WITHOUT_FUNCTIONAL_VALUE，保留原始数值。
4. 源路径贡献与直接/target-only对照一起解释。门控质量、U范数或输出差异不单独证明跨域收益。

记录适配秒、共同前缀秒、总成本、峰值显存、实际source/function调用次数、新增/冻结/总部署参数；给线性层MAC等可复现算子计数，不能把关系rank少说成稀疏执行。raw-X端到端时延采用v1已锁定batch1/256、50warmup/200测量、CUDA同步计时，报告P50/P95与设备；缓存h/phi的适配时延另列。正常推理按实际域需要的foreign源执行，不伪称NULL带来调用跳过。

## 8. 阶段判定与停止

两个数据集分别决策。C*为四个简单对照 B_REFERENCE / DIRECT / READ / TARGET_ONLY，在四seed selected Q macro LL均值最小者；继承v1比较器的1e-6平局容差，平局用实际适配时间再method ID。V只选checkpoint，Q只选开发对照与是否推进，E完全不参与。

对主方案−C*的四seed平均：GUARD要求natural LL≤+.001、macro AUC≥−.0005、每格mean LL≤+.003；PRACTICAL需GUARD且macro LL≤−.0005，或AUC≥+.0005且macro LL≤+.0002。六个固定域×任务格逐一预注册报告；局部证据要求至少同一格在四个seed分别LL改善≥.0005，其余格仍过GUARD，不能事后任选两个seed。

LOWRANK通过PRACTICAL并有形成证据才考虑阶段2；总体未过但有上述局部证据且GUARD通过时，只允许一次有限诊断。否则停止主线结构扩展并保留最佳简单方法。FULL更稳而低秩实际成本无优势、target-only解释增益、AUC/弱格保护越界等按05停止条件执行，不归咎成算法实现失败。

阶段2不预先实施。只有实际来源风险/RMS退化且目标收益成立，才用报告中明确的退化事实进入lambda_src∈{.01,.1}的补充设计；保持风险具体公式在实施前另行设计审查，不临场新增。只有四seed稳定互补、已达实用价值并能辨识固定容量不足才设计受控split pilot。未触发记NOT_TRIGGERED，不记阴性。不得启动gain predictor、K/rank网格或其他额外实验。

所有48条预注册适配轨迹完成（或有可复核的具体资源/数据阻塞）才算开发矩阵处理完毕。若停止条件成立，完成固定诊断、冻结后的selected/final历史E复评、最终报告和代码上传后结束；不为形式复评把同一四seed同配置重训一遍。若阶段2改变方法，则同四seed完整运行新冻结方法，成本另账。

## 9. 最小验收与运行准入

coder至少提供以下小CPU断言/测试，独立reviewer检查设计、实际实现与coder报告后须PASS或全部blocking fixes落实的PASS WITH FIXES：

- 唯一YAML拒绝重复键、未知方法、缺项/非法维度；audit/plan只读且不训练、不建checkpoint。
- 实际native.encode h256/z2与原生预测一致；源教师转换≤1e-6、U0初始化 f=phi≤1e-7。
- 三域本域屏蔽，foreign+NULL总和1±1e-6；J1形状与强制NULL退化正确。
- warmup权限和联合BCE梯度张量可达关系/w/g/U/V；先让w获得更新后检查非零路径，不把零w第一步零梯度当实现错误。
- mask排除及有效分母、batch1/缺域前向与单batch backward、w半径投影。
- 实际参数计数对应第5节；低秩关系节省仅112对792，不混入函数参数。
- 人工已知预测的稳定LL/AUC与单类NA；不经过概率clip主指标。
- 完整RAM状态下一步恢复≤1e-6及身份不一致拒绝；所有训练步只访问许可角色。
- 新包独立audit/plan/import不调用旧prefix或训练CLI。E在冻结前拒绝进入评估/选择/optimizer。

阶段0的原始数据身份/当前main来源审计由orchestrator提供；prepared缺失先按锁定自然流水线恢复，GPU资源按设备各自85%显存和功耗条件调度，不终止他人进程，不改batch以追求并发。正式运行只能在上述接口验收与独立审查完成后开始。

## 10. 交付与未决边界

代码、有效配置、命令/环境快照、SOURCE_MANIFEST、逐seed逐格selected/final结果、角色/预测身份、完整成本账、功能/关系诊断、门槛决策与NOT_TRIGGERED记录都写本次路径。最终主agent分析AUC/LL、域/任务、seed、域规模/基线难度、训练阶段、相互矛盾结果和替代解释，证据标STRONG/MEDIUM/WEAK，给出主方案/简单法/停止的实际决策。

当前设计无待凭结果选择的主线超参数。仍须执行的准入事实是实际源码身份核验、自然prepared重建、CPU验收与独立review，不能由本设计报告代替。阶段2具体保持loss或split仅在其预注册条件触发时才是必要工作。
