# R1 Architecture Audit

日期：2026-10-11（Asia/Shanghai）。范围：仅 R1。主Agent整合 GPT-6.1 Sol · Medium 代码审计与 GPT-6.1 Sol · xHigh 独立研究设计审核。证据明细：[代码审计报告](work/R1_Code_Audit_Report.md)、[工件/版本清单](work/R1_Code_Audit_Inventory.json)。未实施算法、未训练、未正式预测评估。

## 1. 审计结论

**结构操纵对象成立（STRONG）；性能提升空间尚无直接证据（WEAK）；执行准备度 NOT READY。** 当前强 Sample baseline 的两个他域专家确实等权融合，没有独立的源域级或任务条件化域专家相对权重。共享 gate 无法直接调这两个域专家的相对系数，但共享分支与非线性塔仍可间接补偿；不能从结构缺口推导 AUC 收益。

本次只有六份 R0 正式报告可读；引用的三份 work 报告在可访问树未找到。历史四seed checkpoint 的存在性核查为失败，未开展 baseline 数值复现。R1文档与设计工作可交付，九份输入阅读与正式实验启动条件未全部满足。

## 2. 输入来源与限制

已读取 `promt/stage_r1.md`，以下六份 `research_context/r0/` 正式报告：R0_Experimental_Evidence_Audit、R0_AO_Hypothesis_Matrix、R0_Contradictions_and_Corrections、R0_Revised_Opportunity_Ranking、R0_Minimum_Pilot_Questions、R0_Global_Conclusion。

缺少其引用的 `research_R0/work/lit_opportunity_review.md`、`exp_evidence_review.md`、`skeptic_review.md`。精确文件名在当前可访问项目树检索无命中；没有把当前审查报告或主Agent重建替代这三份原报告。审计树内两处 S3 archive 目录拒绝访问，详见工件清单，不能据此宣称全磁盘不存在任何材料。

研究上下文交叉核对包括 Phase2B RedTeam、实验 Stable Findings/Global Summary/Open Questions/Hypothesis History/Experiment Index，以及现有 S1 analysis/run report/final review 和 E4 dataset audit。R0声明“原始阶段材料不可见”是旧工作区限制；当前已有历史报告与原始指标可读。性能结论采用最终修正口径，不沿用早期 S1 review FAIL；该文件最终第7轮为PASS（715–773行）。S6-R2仅记录材料身份/协议风险，不作为 AO-04 支持或反证。

项目根不是 Git 仓库，没有可验证 commit。版本以审计 Inventory 中逐文件SHA256、绝对路径与大小锁定，不能虚构代码revision。

## 3. 实际主干、专家、门与塔

依据构造器 `m3oe/MDMTRec/models/multi_domain/model.py:75–136` 与 canonical `experiments/stage_S1_models.py:134–216`。

输入 embedding：KuaiRand 34个稀疏特征×16=544维；MovieLens 7个稀疏×16+age dense=113维。STAR以共享矩阵和实际域 slot 矩阵元素乘形成变换，域bias+共享bias，512→256 MLP；另有输入→256 skip。MLP使用 Linear/LayerNorm/ReLU，无BatchNorm、无Dropout。

正式 joint 配置 D=3/T=2/N=4：4共享专家+3域专家+2任务专家，共9个专家；每个256→64。6个共享 gate，Linear(256,4)+softmax；6个预测塔，64→64+LayerNorm/ReLU→1+sigmoid。专家是 N+D+T 加法结构，gate/塔才按D×T复制，不是D×T个完整backbone。

所有域专家与任务专家对同一目标样本的STAR表示dense执行。称某个域专家为“源域专家”是模型索引名称，不表示它只用源域数据训练或只携带该域信息；通过他域回收也接收其他域梯度。学习比例的含义首先是内部表示偏好/co-adaptation，不能直接称真实跨域因果亲和。

## 4. 精确计算图与参数含义

实际forward见 `stage_S1_models.py:155–181`；原始forward对应 `model.py:160–194`。

```
aD = sigmoid(_weight_exp_d.deep_weights)
aT = sigmoid(_weight_exp_t.deep_weights)
bD = sigmoid(_weight_bal_d.deep_weights)
bT = sigmoid(_weight_bal_t.deep_weights)
R_d = bD*domain_expert[d] + (1-bD)/(D-1)*sum(other domain_experts)
Q_t = bT*task_expert[t]   + (1-bT)/(T-1)*sum(other task_experts)
F_dt = sum_n gate_dt,n(detach(emb))*shared_expert[n] + aD*R_d + aT*Q_t
prediction_dt = sigmoid(tower_dt(F_dt))
```

四个 raw 标量都只有一个元素，跨域/任务/样本全局共享，不是每域β或每任务β。raw初值1，实际值sigmoid(1)=0.7310586。own域系数 aD·bD；每个other系数 aD·(1−bD)/2；other总系数 aD·(1−bD)。aD/aT控制模块加法尺度，bD/bT控制本专家与其他专家的分配。共享项系数为1；三个模块不构成和为1的convex mixture。

索引task-major，i=t*3+d：t0d0/t0d1/t0d2/t1d0/t1d1/t1d2。最终根据实际样本域输出每任务概率，形状[B,2]。任务轴T=2时他任务只有一个，没有可研究的“其他两个任务相对比例”。

gate只对N共享专家softmax，其参数可学习、输出随样本与(d,t)变化；emb.detach阻断gate输入分支回到STAR的梯度，并未阻断gate参数自身或专家路径。它不能直接单独乘域专家E_j/E_k，但STAR、共享表示、任务专家和塔存在间接代偿可能。

Weights.forward会递减tau且同forward多次调用；当前softmax_type=3是与tau无关的sigmoid，因此tau递减不改变有效融合。tau不是state_dict参数/buffer。改softmax_type会变更baseline，不能作为本设计隐含选项。

## 5. 逐模块参数账

代码审计Agent仅在CPU从历史schema构造模型、清点numel；与历史parameter_manifest逐tensor一致。没有加载训练权重、没有forward实验。

| 模块 | KuaiRand | MovieLens |
|---|---:|---:|
| Embedding | 44,752 | 196,992 |
| STAR shared weight/bias | 279,040 | 58,368 |
| 3个slot weight/bias | 837,120 | 175,104 |
| skip connection | 140,032 | 29,696 |
| STAR MLP | 131,840 | 131,840 |
| 4个共享专家 | 66,304 | 66,304 |
| 3个域专家 | 49,728 | 49,728 |
| 2个任务专家 | 33,152 | 33,152 |
| 6个共享gate | 6,168 | 6,168 |
| 6个预测塔 | 26,118 | 26,118 |
| 4个融合标量 | 4 | 4 |
| **总计** | **1,614,258** | **773,474** |

每个专家16,576；每gate1,028；每tower4,353。未来设计B1/Bcap各增加6个真实参数，B2增加3个，不能以2logit softmax存储又只报告6个参数；原始参数量和冗余须分别报告。Reviewer指出Bcap温度可吸收到原gate Linear的W,b，故新增6个活跃参数不增加6个函数类自由度；它是优化重参数化对照，不是严格容量匹配。具体新增量与优化器状态成本均须报告。

## 6. Objective与干预残留审计

S1 Sample直接对样本×任务平均BCE（`stage_S1_run.py:114–142,657–680`）。S2 Equal用固定全训练p_d和1/(3p_d)样本权重，再对样本×任务平均（`stage_S2_objectives.py:58–98,178–220`）；未重新域采样、未按batch重新归一。数学定义及历史尺度差异详见 Baseline Protocol。

Legacy `ctr_trainer.py:116–124`为六格BCE之和；其第50行以~bool筛main参数，-1/-2都truthy，四scalar进入主optimizer后又由架构optimizer更新。S1 canonical runner明确分离两组（618–633行）。本项目“原始M3OE”必须定义为S1设备安全适配器+修正优化器+Sample objective，不能复用examples trainer含混称原始。

S3 target-only改共享梯度访问；S4–S6 HT/DH改共享Adam历史/当前梯度来源；S8是continuation。B0不得引用T/HT/DH或续训。原生Adam状态不等于历史额外动量算法。S6 F的FULL/FULL可作身份核查，不能在没有完整权重/环境匹配时替代S1 B0。

## 7. R0修正与主Agent研究判断

| R0主张/隐含前提 | R1修正 | 强度 |
|---|---|---|
| 均匀他域回收存在 | 当前代码确实成立 | STRONG |
| α/β或gate具有两级域×任务门 | 四global scalar；每格gate仅共享专家 | STRONG |
| MovieLens N=1 | 当前examples default及S1/S6ML均N=4；原文配置不能覆盖当前代码 | STRONG |
| 既有四seedcheckpoint可复用 | 16核查run所声明selected均缺失；不能扫描或声称复现 | STRONG（所审计工件范围） |
| 删除损失证明均匀非最优 | 不识别守恒重分配或训练权重效果 | STRONG（逻辑边界） |
| 删除无损证明训练权重无效 | 不成立；co-adaptation/尺度/代偿均不同 | STRONG（逻辑边界） |
| 结构缺口证明非均匀有收益 | 性能机会仍未测 | WEAK |

稳定历史现象：Sample自然比例下无稳定practical residual NT；Equal局部rescue随objective/数据/选点改变。矛盾与竞争解释：部分late收益不形成selected-best收益；域大小与难度、loss权重、曝光、时间shift和模型同时变化。不能预设curriculum/momentum有效、M3OE必败于难域或小域=难域。历史结论强度与边界沿用SF001/002/006/007/009/010，均不直接支持AO-04性能H1。

现有材料没有识别size主因，没有AO-04小域收益/大域代价的实测结果。最有判别力的未来实验是固定预算的B0/B1/B2/B3/Bcap配对对照，以及独立解释的冻结守恒扫描；优先排除简单解释，再讨论完整亲和算法。

## 8. 结论及范围核验

AO-04保持OPEN-TESTABLE，不能作算法性能GO/STOP。三种干预的estimand已拆分；正式设计须独立Reviewer审核。baseline readiness仍为NOT READY，具体阻塞见 Baseline Protocol。全部新增内容位于research_R1_R3；未覆盖S0–S8文件，未执行R2/R3。
