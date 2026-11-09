# PUB-MDMT 来源与语义审计

审计日期2026-11-09；设计阶段已完成源码/图/公式审计。本文与 DESIGN_SPECIFICATION.md 在科学训练前形成。

## Git与计划

原 `start.md` 指向 `research_plans/26-11-09_pub_mdmt`，远端没有同名分支；该目录实际位于 main `40c8afafedab422bdf36c02b38aff19114d95a9b`。00–08文件已从该commit水合，来源状态为规划完成；本次用户指令授权继续完整实验。

历史26-11-09=`1391affa5f08556c79b4cb2bfad15228a8b9129f`；共同祖先=`652ee0235fec2d0d4cfc67b4a6ef9977519619bf`；`git rev-list --left-right --count main...26-11-09`=92/18，即历史分支相对当前main ahead18/behind92。只读历史实现与报告，不合并该实验分支。

已读main `EXP_AI_CONTEXT.md`、`MDMT_ALGORITHM_METRICS_LEDGER.md`当前强制规则及历史 `experiments/Experimental_Global_Summary.md`；已读历史11/09 FINAL_REPORT、IMPLEMENTATION_REVIEW、NUMERICAL_TABLES及A0_DECISION。11/09为A0 `NO_ACTIONABLE_SIGNAL`：新增持续性预测未提供超过简单统一动作的稳定增量，后续A1/B/C未触发；它研究EMA而非PUB，不作为PUB性能判据。11/06固定EMA的五seed排序收益和条件性LL代价作为已知参照，不重设为待验证假设。

## 作者论文与源码

论文 [A Parameter Update Balancing Algorithm for Multi-task Ranking Models in Recommendation Systems](https://arxiv.org/html/2410.05806v2)，arXiv2410.05806v2。固定作者库commit：[e234c149a54d0f84748d5b9de94f45ba541e1e0a](https://github.com/yjdy/Multitask-Optimization-Recommendation-Library/tree/e234c149a54d0f84748d5b9de94f45ba541e1e0a)，原始源码位于本目录author_source（含MIT LICENSE）。主要审计文件：`mto_methods/parameter_balancing/{pub.py,adam_multitask.py,clippy_adam.py}`、`mto_methods/utils.py`。

| 项目 | 论文/实际源码 | 本轮明确选择 |
|---|---|---|
| 候选与moment | Algorithm1分别构造任务候选，最终相同alpha合成原梯度更新m/v；源码step中候选用同一旧m/v | 保留；纯候选一次读取，所有候选后才提交状态 |
| Bias correction | 论文伪代码未列；AdamMultiTask.step用新增step，utils候选probe读旧step | 所有候选和实际提交均使用k+1，保留当前PyTorch修正式和eps位置 |
| 导入 | pub.py导入不存在的`AadmMultiTask`，定义实际为`AdamMultiTask` | 不直接导入作者训练模块 |
| 求解器 | `cp.QSOP`不存在，bare except吞异常并沿用旧alpha | 真实cvxpy conic solver，异常/状态/残差显式记录，失败明确普通Adam回退 |
| CCP目标 | 论文式(4)线性化sum phi；源码额外加入sum(Galpha)并缩放约束 | 实现论文式(4)，记录为论文一致的修正 |
| 迭代返回 | 源码stop在alpha_t赋新值前发生，残差也读旧alpha；可能返回滞后一轮结果 | 对新解验残差后返回；alpha停滞不充当求解成功 |
| 频率 | 源码`step%frequency==1`使frequency1永不求解 | step1以及1+nf求解，活动集改变时重解 |
| Gram支持 | utils遍历全部optimizer.param_groups而非显式S组 | 按实际S/D/T共享坐标和活动loss构造Gram |
| WD | 源码共享参数task循环中重复`p -= lr*wd*p`，为重复decoupled项；源项目是coupled Adam | 数据候选与公共coupled-WD差分分开，m/v仅加入一次λθ；公式与标准退化测试见设计 |
| 原始alpha | 正alpha，无simplex；Galpha=1/alpha含绝对尺度 | 不归一化候选/alpha；数值Gram缩放后恢复物理alpha，记录步长/范数 |
| 专有参数 | 作者专有参数普通Adam方向，源码WD为decoupled | 本模型DT及单任务组用源coupled普通Adam |

作者源文件取得且关键语义可核验。上述缺陷通过独立新包修正，原作者快照不改。论文没有规定本模型自然样本分解、12组拓扑、公共coupled-WD差分或真实同alpha重放；这些均为本项目迁移与对照定义，不能归称作者已实现的行为。

## 实际fixed1图、自然loss和autograd

来源为历史26-11-09的 `experiments/aug_history_20261105/{group_registry.py,registry.py,model_adapter.py}` 与下游冻结 `vendor_ga31/reference.py`、`model.py`、`gradient_aug_20261031_v2/objectives.py`。源码保留gate输入 `emb.detach()`，不能为了拓扑解释开启`probe_input_full_graph`。最终两列task输出按原始domain mask选择tower `3*t+d`。无Dropout/BatchNorm，LayerNorm保留。

用项目环境在真实缓存中每域取前8行（仅24行正确性探针，不进入训练），seed2022、CPU，执行原始模型及六个Ldt的autograd：

| 数据 | 输出 | 参数数 | 原loss | 六格sum | 标量绝对误差 | 梯度最大差 | 非所属组梯度泄漏 |
|---|---|---:|---:|---:|---:|---:|---:|
| ML | [24,2] | 773470 | .6832705140113831 | .6832704544067383 | 5.960464477539063e-8 | **0（逐元素相同）** | 全六格0 |
| KR | [24,2] | 1614254 | .7188008427619934 | .7188009023666382 | 5.960464477539063e-8 | **0（逐元素相同）** | 全六格0 |

自然loss为BCE矩阵先mean(task)再mean(row)。定义 `Ldt=BCE[domain=d,task=t].sum()/(2N)`精确对应其数学目标；不同FP32归约顺序产生上表舍入。不能改成按六个cell均值等权求和。

实际12组计数：

| 组 | ML | KR | cell支持 |
|---|---:|---:|---|
| S | 483152 | 661920 | 全六格 |
| 每Dd | 74960 | 295632 | domain d两task |
| 每Tt | 16576 | 16576 | task t三domain |
| 每DTdt | 5381 | 5381 | 单cell |
| INACTIVE | 0 | 0 | 无 |

domain embedding按row0/1/2分配Dd；registry覆盖全部坐标、没有重叠。六格非所属组的数值梯度都为0，支持关系不仅来自名称。正式实现后还须由其自身必要测试重复验证，以防vendor装配改变行为。

Gaussian历史目标也已检查：只在指定目标域，将clean损失固定比例替换为同自然行的embedding噪声view损失；其目标与clean不同。本轮主问题没有要求新增Gaussian矩阵，全部PUB/梯度控制使用clean自然目标。

## Adam、环境与数据可达性

已检查历史 `ema_mechanism_20261106/worker.py`、protocol、只读 `adam_counterfactual.py`，并在当前环境读取 `torch.optim.adam._single_tensor_adam`。当前Python3.13.5，PyTorch2.9.0+cu126，CUDA可见两张RTX3090。source optimizer是torch.optim.Adam，lr=.01、betas(.9,.999)、eps1e-8；ML/KR WD=1e-4/1e-5，decoupled_weight_decay=False，foreach/fused/amsgrad=False，step在计算bias correction前增加一次。PyTorch对grad先加λθ，再更新m/v；eps位于sqrt(vhat)外。

原始CSV位于start.md/AGENTS指定的`/home_new/xzf/datasets`。自然cache的ML/KR目录确实存在于`/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache`，其manifest记录了raw SHA、数组、row fingerprint、train-only encoder/scaler和人口身份。正式准备阶段核验所有实际文件后复用；本轮不使用缺失`mdmt_next_20261024_strict_v1`。

环境中scipy/pytest已安装，cvxpy/clarabel/scs/ecos初始未安装。实现论文CCP需安装cvxpy及conic solver，并保存实际新增依赖及版本。所有项目Python显式调用项目环境的bin/python；不使用Conda base。

## 资源与测试结论

同一模型FP32静态payload、历史实际batch40960成本和本轮强制smoke见DESIGN_SPECIFICATION §10。历史两epochclean训练ML4.366秒/KR8.163秒、峰GPU约1.024/1.267GB，是初估来源；PUB的多梯度、候选和solver成本须本轮实测。禁止通过缩batch、减少seed或少epoch规避预算。

源码/图/自然目标/原Adam关键语义已取得并明确，可进入独立coder实现。实现后必须完成规范§8的必要测试并经独立review通过，再开始科学训练；该要求针对可能改变研究结论的loss、梯度、更新、状态和评估语义。
