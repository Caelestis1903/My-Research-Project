Verdict: PASS

# PUB-MDMT 独立实现审查

日期：2026-11-09。审查者为独立 reviewer agent，与 coder 分离。审查对象为冻结的 `DESIGN_SPECIFICATION.md`、`SOURCE_AUDIT.md`、`CODER_IMPLEMENTATION_REPORT.md` 及新增包的实际文件。

通过的实现身份：

`0eb54f049abef69e735b90af2dbca4fb88dd876687cea3ab2719472730069e4d`

**阻塞问题：0。允许进入 RUN。**

## 核对结果

| 责任 | 独立审查证据 | 判定 |
|---|---|---|
| 自然目标与真实模型 | `optimizer.py:12` 按实际 batch 的 `2N` 分解六格 BCE，P0按任务聚合；`data.py` 使用冻结模型与原始自然缓存；两数据实际图/空域/支持检查已通过 | PASS |
| 参数坐标与拓扑 | `group_registry.py` 按源注册表划分 S/D/T/DT，强制计数和完整互斥覆盖；P0的T、DT和零/单活动分支普通Adam；P1直接取S系数子集；P2局部求解 | PASS |
| 候选与状态 | `optimizer.py:25` 仅从旧 m/v 计算候选，`t=step+1`，偏置修正与eps位置符合设计；所有组计算后才提交参数、m/v和step，每物理坐标一次 | PASS |
| Coupled WD | `optimizer.py:88` 的公共差分为 `U(gbar+λθ)-U(gbar)`，状态只以 `gbar+λθ` 更新；普通Adam分支沿用single-tensor运算；WD=0/1e-4/1e-5多步等价及手算检查均通过 | PASS |
| 论文CCP与物理尺度 | `solver.py` 的线性化目标为 `qᵀa`，约束为负双log非正；Gram按trace数值缩放后以 `a/sqrt(s)` 恢复物理alpha，无simplex或候选归一化 | PASS |
| 数值接受与回退 | 对新解检查regularized及原始残差、正alpha和原始utility；失败返回None并执行自然普通Adam；P1的S失败传播到继承组；refresh从step1开始，失败下一步重试 | PASS |
| 真实同alpha控制 | `training.py:80` 从对应C读取逐step trace；G使用自身梯度和状态，不调用solver；核对活动集合、step、完整系数hash、初始化和loader order | PASS |
| 选点与Test | `training.py:30` 严格最大值保留最早epoch；pilot只加载train/val；正式训练先写 `SELECTION_FROZEN.json`，再在同进程RAM状态上读取Test；组合门只读Validation冻结文件 | PASS |
| 有限矩阵 | `run.py` 的18组solver校准、40条完整2epoch频率比较、70条完整50epoch正式轨迹与五seed一致；C2/G2在前阶段完成后执行；组合按预设五seedValidation门触发 | PASS |
| RNG与数据顺序 | 独立generator的seed+11000、完整randperm与尾batch保持；评估和只读短步探针保存/恢复RNG；源模型无Dropout/BatchNorm；loader/optimizer/cache恢复检查通过 | PASS |
| 存储与资源 | 选点、E10/E50、optimizer与恢复状态仅在RAM；磁盘写入点为标量JSON/JSONL/CSV及自然数据数组；逐GPU独立检查显存/功耗<85%、host RAM<90%，无固定两worker上限 | PASS |
| 评价与数表 | 复用冻结概率指标；六格、domain/task、Macro、Traffic/Pooled、selected/E10/E50分别保留；配对统计强制2022–2026，以seed为单位，n=5的SD/t区间及AUC/LL分开裁定 | PASS |

## 来源及检查身份

独立对照 `SOURCE_MANIFEST.json` 中全部 **13个来源文件**：读取历史commit `1391affa5f08556c79b4cb2bfad15228a8b9129f` 的Git对象，重新计算source/destination SHA256并检查实际diff。11个文件逐字节相同；`registry.py`、`group_registry.py` 的唯一差异为本包 `digest` import。新包的运行import不依赖相邻checkout。

重新计算当前实现hash，与以下已有收据一致：

- `CHECKS.json` / `CHECKS.log`：PASS，16项检查通过，exit0。
- `AUDIT.json`：原CSV、数组内容、行身份、train-only预处理、划分和domain人数匹配。
- `SMOKE_movielens.json` / `SMOKE_kuairand.json`：两个数据集各P0/P1/P2、batch40960，六次PASS；独立读取收据确认全部零solver fallback，peak reserved最大1,644,167,168 bytes。

审查复用上述与当前实现绑定的有效检查，未重复完整GPU smoke或启动科学训练。TF32旧设置API的deprecation warning不改变当前关闭TF32的配置。

## 修订与交接

本轮独立审查无需算法修订。coder已记录的候选broadcast修复和toy fixture修订均发生在本次通过的收据之前；没有科研结果需要因此重跑。

Runner可按实现报告执行 `calibrate`、`pilot`、`train`、`report`。原始数值和决策输入保留在本轮artifact目录；科学解释由orchestrator在完整实验后完成。

## RUN期间补充：短步机制诊断覆盖（Relevant）

运行中只读复核发现一处相对设计§10的诊断覆盖差异：`training.py:101` 按 `step%10==1` 请求DIAGNOSTICS，每epoch首batch则单独写SHORT_STEP。冻结频率为10时，正常refresh也位于相同的全局step序列；因此KR的每epoch短步BCE并非都有同step的S组Gram/utility。

每epoch首batch的全局step为 `b*(epoch-1)+1`，其中ML的b=20，KR的b=49。由此得到：

| 数据 | 完整50epoch的SHORT_STEP窗/seed | 正常调度下可与S组utility精确配对的epoch | 配对窗/seed |
|---|---:|---|---:|
| MovieLens | 50 | 1–50 | 50 |
| KuaiRand | 50 | 1、11、21、31、41 | 5 |

抽查运行中已有日志：KR C0五seed已写出的前12epoch均只配对epoch1和11；ML C0两条已写完整50epoch的日志均有50个同step配对。回退后的提前重试可能额外生成DIAGNOSTICS，最终分析须以真实 `(epoch,step)` 内连接得到的配对清单为准。

**影响与处理：** 该开关仅控制`optimizer.py:104`之后的诊断字典、范数和utility输出；参数、m/v、step和alpha已经按相同公式计算，训练更新、RNG、loader、Validation选点及Test入口均不依赖是否写出该字典。本项限制KR机制分析的时间覆盖，不改变训练或性能指标比较。继续保留当前有效训练，不为增加记录而重跑矩阵。

最终报告的utility–BCE分析只使用真实同step配对。KR其余45个epoch仍有短步BCE，但缺少正常调度下同step的utility，不能补写、跨step配对或计入utility关联分析。正常调度下每拓扑五seed对应ML250个窗、KR25个窗；这些窗和候选任务均属于seed内重复观测，统计独立单位仍为5个seed。完整覆盖按最终实际日志重新计数。

判定维持 **PASS**；该Relevant差异按实际配对范围记录，不要求算法修订或追加训练。

## 报告支持脚本独立审查

脚本：`analyze_diagnostics.py`；实现交接：`ANALYSIS_TOOL_REPORT.md`。

**脚本判定：PASS。** SHA256：

`9ed157020205b71d43ffdd39473484248a86d2ebbd09075940cbe1e407d6effc`

独立审查读取上述两文件并执行一次 `--self-check`，7组fixture返回PASS、exit0：精确连接、六格rank、ties/zero/missing、二任务向量、真实solver残差、非对角cosine、阶段边界。实验包身份重新计算仍为本报告通过的 `0eb54f04…`。未执行真实产物汇总或读取Test终点。

核对结果：

- 外层保留dataset/arm/seed，内部按group与epoch阶段1–10、11–30、31–50分组；window及候选分量仍为seed内记录。
- S组DIAGNOSTICS与SHORT_STEP combined按完整 `(epoch,step)` 内连接，拒绝重复键，排除附近step；输出真实配对和缺配清单。上述ML50/KR5的正常范围由实际日志连接得到，不强行补齐。
- C0保留两任务utility和六格BCE变化，rank/sign标记为二任务向量；C1/C2先确认六个真实成员，对utility与负BCE变化计算Spearman。ties使用平均rank，常量rank置空；sign agreement只比较两侧均非零的cell，零值数量和实际分母另存。缺失utility或六格loss时保留原值并给出缺失状态。
- actual utility与data utility分别保留、分别计算关系。`refresh_solve_count`只数非空真实solve记录，P1 D/T继承系数不计为局部solve；`raw_residual`只取真实solve内字段，余弦仅取上三角非对角项。
- 真实汇总入口要求 `EXPERIMENT_COMPLETE.json` 的status为COMPLETE，读取范围为C0/C1/C2训练侧日志，输出到tables，不写模型或训练状态。

使用范围：`DIAGNOSTIC_SEEDS.csv`的均值和比例描述已记录诊断，不代表未记录训练step；`MECHANISM_PAIRS.csv`保留window级关系，跨seed结论应先在每seed/阶段内汇总，再以5个seed比较。C0的两任务utility不进入六cell rank分析；KR缺配epoch沿用上节的实际覆盖记录。

无需修改脚本或训练；原有性能与训练收据继续有效。

最终判定：**PASS**。
