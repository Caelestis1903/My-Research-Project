# 01｜Research Context & Evidence Ledger

## 1. 最终研究问题

M3oE 中共享表示变化会同时影响六个域任务预测路径，但这些路径并非六个独立模型。共享专家、域专家、任务专家和 cell-specific Gate/Tower 以不同粒度共同构成预测。当前同步联合训练没有显式区分共享表示更新与各域任务路径在更新后适配的作用。

研究目标：检验共享表示更新后，专有预测路径是否存在可恢复的排序响应失配，并判断把专有路径更新放在共享表示更新之后，能否产生超出普通联合 Adam、梯度累积和反向更新顺序的增量。

## 2. 研究问题的边界

本轮不把“头部适配滞后”当成已知事实。更准确的工作定义是：
- **共享表示更新**：U 参数组在一次真实自然训练更新中的参数变化。
- **专有路径**：M3oE 中域专家、任务专家、每个域任务对 Gate 和 Tower 的组合路径。
- **响应**：在共享表示更新前后，同一域任务输出的 logit、pairwise margin、AUC 与 LogLoss 变化。
- **适配**：固定 U 后，只更新选定专有参数组，观察输出排序是否恢复或改善。
- **协调更新**：以 U→P 的顺序执行参数块更新，并与 P→U、JOINT-ACC、原始 JOINT 对照。

本研究不声称 M3oE 的六个 cell 有六套完全独立 head；Gate/Tower 是 cell-specific，domain_expert 和 task_expert 分别沿域轴/任务轴共享。

## 3. 可接受的历史事实

### 固定 EMA

固定 EMA 在既定 MovieLens/KuaiRand 与 M3oE fixed1 设置中有稳定排序收益；11-10 参数归因显示共享参数删除会损失主要排序收益，非共享参数删除影响较小。该事实已接受。固定 EMA 仅作为内部强参照和读出，不重新证明，不将其改造成本轮算法。

### EMA 自适应、梯度和历史

参数分组时间尺度、梯度漂移、收益预测、PUB 迁移、PAIR/ECCM、NDP-A、TV-GRM、AXIS-EVEN、NQR 等研究已覆盖大量“从梯度/历史代理预测更新价值”的变体。多个候选未形成稳定的跨数据集独立 AUC 增量。不得再以更复杂 EMA 或通用梯度亲和度作为主方法。

### 来源收益与共享更新

26-11-23 分支的 B_CORE 在 40 个真实 Adam 锚点上核验虚拟联合更新与真实更新身份；报告显示共享 EMBED 一步来源更新的跨格影响并不均匀，但均值较小。B_PULSE 的动态选源未转化为持续训练收益，B_SEARCH 未触发。该结果说明局部一步收益不是可部署训练策略，也没有回答共享更新之后专有路径适配是否能恢复排序。

### 最新条件历史 A 方向

A20 相对同监督普通 Adam 的 validation Macro AUC 均值增量仅 MovieLens +0.000161、KuaiRand +0.000172；A21 的验证/测试方向反转，未取得稳定独立算法资格；低剂量历史也未修复。该结果降低了继续研究条件历史的优先级。

## 4. M3oE 真实计算图

必须以当前 main 中的 `m3oe/MDMTRec/models/multi_domain/model.py` 和 fixed1 canonical adapter 为最终依据。计划中的初始结构描述：

1. Input Embedding。
2. STAR：`shared_weight/shared_bias` 与 `slot_weight[d]/slot_bias[d]` 组合；再经过 `star_mlp`，并与 `skip_conn` 输出相加，形成 domain-conditioned shared representation `emb`。
3. Shared experts：`expert[n]`，由每个 cell 的 `gate[3*t+d]` 对 shared expert outputs 加权。
4. Domain experts：`domain_expert[d]`，沿域轴复用。
5. Task experts：`task_expert[t]`，沿任务轴复用。
6. Cell-specific prediction：每个 cell 有 `gate[3*t+d]` 和 `tower[3*t+d]`；最终输出按真实 domain mask 选取。
7. fixed1 融合系数按历史固定模型定义，不允许 Codex 自行将其改为可学习 gate。

原始实现中的 Gate 输入为 `emb.detach()`。因此上游 embedding 更新会改变 Gate 前向输入值，但 gate loss 不沿该输入反传到 emb；这条梯度路径必须保留。

## 5. 参数分组工作定义

分组由真实 `named_parameters()`、forward 支持和梯度路径审计构建，不允许只靠名字包含 shared/private 推断。

| 组 | 初始内容 | 作用 |
|---|---|---|
| U：Upstream/shared representation | embedding、STAR shared 与 domain slot 参数、skip_conn、star_mlp、shared experts | 生成被多个域任务路径共同消费的表示/专家内容 |
| P_CELL | 六个 cell-specific gate 和 tower | 严格 cell-specific 预测路径 |
| P_AXIS | domain_expert[0:3] 与 task_expert[0:2] | 沿域轴/任务轴共享的专有专家 |
| P_FULL | P_CELL ∪ P_AXIS | 全部非 U 的预测专有路径 |
| FROZEN/OTHER | fixed1 非训练参数、buffer 或经审计的非参与参数 | 保持源语义，不进入候选更新 |

若发现其他可训练参数（例如固定融合标量意外进入 optimizer），必须先判定 canonical fixed1 身份。不得静默放入 U 或 P。所有参数必须互斥且完备；参数的梯度支持必须用真实 cell batch 检验。

## 6. 文献边界

- STGU（ICLR 2025，通用 MTL）：研究任务组更新顺序和共享编码器/任务头的近端亲和度；它是最接近的更新协调工作，但不是当前 M3oE 的域×任务共享专家—域专家—任务专家—cell Tower 结构。
- NTKMTL（NeurIPS 2025，通用 MTL）：提供输出对参数响应速度不同的视角；其 NTK/任务权重公式不能未经验证直接迁移到稀疏推荐 BCE 与 M3oE。
- MDL（KDD 2026，严格 MDMT）：将场景/任务条件信息注入多层特征交互，研究表示提取，不是共享更新后的专有路径适配。
- ORCA/PTDG/SAF：分别涉及条件错误依赖、稀疏任务信号侵蚀、表示依赖/样本分布，不等同于 D2。
- CDC/ETAP/SDSP：来源收益、任务组收益预测或域选择，与本轮的更新顺序和适配路径不同。

文献细节以用户上传的 68 篇报告为准，不在本轮重新做大规模文献检索。

## 7. 本轮不研究的对象

不做 D1 条件错误依赖校正、D3 高阶来源组合网络、D4 新数据增强；不做新 EMA 系数、梯度冲突消除、样本权重或域规模驱动控制器；不叠加 Gaussian/PAIR/ECCM/NQR。若 D2 NO-GO，只报告失败边界，不自动切换方向。
