# 06｜Codex 实现合同与必要验收

返回 [执行入口](00_EXECUTION_RUNBOOK.md)。本次只交付实现规范；未来新增包为 `experiments/ema_next_20261111/`。

## 1. 源码复用位置

固定底座：`5ef45b18b973ed11be6fbb5f654562349998de60`。其旧模型实现底座`ffa3de067fcda5a1293af83a465d741e069493f1`，不通过修改旧源码改变历史实验身份。

| 实际路径 | 复用接口/语义 | 新包处理 |
|---|---|---|
| `experiments/aug_history_20261105/model_adapter.py` | build_model、forward_logits、raw tower accessor | 原训练图；scratch观察器 |
| `experiments/aug_history_20261105/registry.py` | ParameterLayout、flatten/copy/hash | 建唯一name/shape/alias布局 |
| `experiments/aug_history_20261105/group_registry.py` | build_path_groups | S/D/T/DT真实支持，不按名字误纳domain embedding |
| `experiments/ema_causal_20261110/interventions.py` | group_masks、source_direction_states | A单步定义参照，不把虚拟状态直接当训练器 |
| `experiments/ema_mechanism_20261106/worker.py` | create_session/train_batch的源训练顺序 | adapter在Adam后/F前插入A；不覆盖源函数 |
| `experiments/ema_mechanism_20261106/shadows.py` | FixedShadowBank.after_adam | 原FP32 EMA递推 |
| `experiments/ema_mechanism_20261106/readouts.py` | full-V/面板有效mask及概率/logit | 全人口、同dtype、同LL与tie |
| `experiments/ema_mechanism_20261106/panels.py` | build_panels、save_panels | P_A/P_B/V_mech身份 |
| `experiments/ema_causal_20261110/history.py` | 稀疏capture/consume引用计数设计 | 新bank保存F，不是原R；禁止偷换对象 |
| `experiments/ema_causal_20261110/pairs.py` | 精确排序迁移/固定pair缓存 | 只做注册端点必要比较，不重做所有归因 |
| `experiments/overnight_ab_20261103/data_bridge.py` | prepare_cache/cached_bundle | train-only处理、include_test=False |
| `experiments/overnight_ab_20261103/selection.py` | 六格Macro及最早严格最大 | 每个完整预测对象独立选点 |
| `experiments/ema_causal_20261110/state.py`、`ema_mechanism_20261106/state.py` | guardian/内存IPC/事务思路 | 新job/fork/consumer管理，维持RAM禁落盘 |
| `experiments/ema_causal_scheduler_20261110.py` | 独立每卡准入、PID/start/owner、physical与commit双预算 | 小范围复用，不能直接套旧20-job完成判据 |

源loss具体路径：`experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/objectives.py`；真实模型位于其 `vendor_ga31/reference.py`。Gaussian源target统一0，嵌套旧元数据不改实际loss。

## 2. 应新增文件及公开接口

| 文件 | 必须接口 |
|---|---|
| `protocol.py` | `resolve_plan(plan_dir)->Study`; `expand_jobs(study)->DAG`; `audit_source`; `prepare` |
| `trajectory.py` | `create_parent`; `train_one_step`; `fork_at_epoch`; `continue_to_epoch` |
| `optimizer_direction.py` | `capture_embed_inputs`; `apply_matched_direction`; `set_embed_beta1_zero` |
| `history_readout.py` | `LaggedFStore.capture/consume`; `split_history`; `predict_split`; `shared_only` |
| `conditional_readout.py` | `build_unique_pairs`; `fit_from_PA`; `apply_lagged_policy`; `evaluate_parameter_match` |
| `distillation.py` | `freeze_teacher`; `cache_train_targets`; `create_student`; `student_step` |
| `evaluation.py` | `evaluate_full_population`; `update_earliest_best`; `freeze_endpoints`; `test_frozen` |
| `decision.py` | 09各门的纯函数，输入完整五seed记录，输出带证据的动作 |
| `state.py` | `put_immutable`; `commit_epoch`; `fork_snapshot`; `release_consumed`; `recover` |
| `scheduler.py` | 单owner DAG、资源预留、五seed barrier、恢复/自动后继 |
| `report.py` | 六格/宏/配对差/LL/成本/决定/来源表 |
| `run.py` | audit/prepare/smoke/execute/status/resume/analyze/freeze/evaluate/report/release |

可合并文件以减少重复，但这些科学职责与接口语义不得缺失。不能另建模型、数据、metrics或loss框架。

## 3. 必须定义的数据结构

`Study`：source/plan/code/environment/data/layout哈希、配置菜单、五seed、全DAG及门、resource_caps、comparison_registry。

`LiveSession`：θ/optimizer/F、epoch/step、独立RNG/loader、B历史、C已拟合未应用/已应用系数、selected对象、source parent/fork身份。

`PredictorPacket`：参数或两态+N/buffer、mode/scope/H/h/a/alpha、fitting/application epoch、parent_F_at_same_epoch、score/population/hash。`parent_F_at_same_epoch`必须是真实FP32父F，不能从两态cast后重构冒充原初始化。

`EpochCommit`：新标量和有限预测hash、前一commit、RAM对象引用、optimizer真实步数、花费、状态owner。序列化参数只用于进程内或socket内存IPC，不写入JSON/NPZ文件。

`DecisionRecord`：门名、输入record IDs、完整五seed标志、公式版本、数值、通过/未通过/阻塞、后继job IDs、reason_code。

## 4. 主循环伪代码

```text
for dataset in [movielens, kuairand]:
    run five Clean parents through E50 with all registered readouts
    assert five complete; freeze parent B configuration; compute C records
    fork two anchors x two A_MATCH alphas x five seeds, six real epochs each
    if G_A: run A_RAW short; continue frozen MATCH and qualifying RAW to E50
    retain AB only for qualifying A and frozen parent B
    decide B/C and one teacher using five-seed Validation only
    if teacher gate: run KD grid + LABEL controls, freeze hyperparams, run Fteacher
    freeze dataset Clean primary and necessary controls; release other consumers
    if G_G: run five Gaussian parents and exactly one inherited candidate family
assert all triggered tasks complete
freeze global experiment/comparison/endpoint list
open Test; evaluate only frozen endpoints; pair against published-model records
write reports and decisions; release owned RAM; verify no model files persisted
```

B/C发生在父训练时，科学晋级在五seed barrier；不能先释放所有模型再只从CSV尝试前向。A/B/C详细步骤由12–14决定。

## 5. 实现正确性测试：只覆盖会改变结论的风险

| 检查 | 通过条件 | 原因 |
|---|---|---|
| 无动作等价 | A alpha0走native no-op，B/C观察开关不改变θ/m/v/F/loader/RNG；source固定batch短序列相同 | 避免控制被污染 |
| A单步与真实反馈 | alpha1对应source_direction_states同mask/q/vhat；第二步用修改θ的新gradient；WD一次；F在patch后 | 区分虚拟probe与真实算法 |
| 参数组 | G=S∩EMBED、排除domain行；非G单步与native一致 | 作用组错误会推翻归因 |
| B代数/时钟 | 初值、h=min(H,t)、KR277、a、两态参数中心、sourceFP32、buffer一致 | 不能将不同历史权重当q近似 |
| C时间与动作 | e拟合仅e+1用；e1固定a；从PA而非V/PB拟合；same-alpha PARAM按真实task索引 | 防未来泄漏与对照错位 |
| KD目标/预算 | teacher stopgrad/eval，train row ID对齐；teacher=y退化源loss；LABEL/Fteacher同初值与预算 | 避免loss尺度与额外训练混淆 |
| 选点/人口 | 全V完整六格，无batch AUC平均；严格>最早；A含前缀、KD含e0；Test loader冻结前不可达 | 公平模型选择 |
| RAM fork/recovery | 不共享可变storage；中断边界恢复后下一步与不中断一致；snapshot齐备 | 原状态不可落盘 |
| 资源DAG | 独立单卡可准入；重复命令不重复派发；未满五seed不能判性能；完整五seed对象图含E20各B前缀、稀疏历史及旧/新commit，预测峰值先通过预算；未触发状态可计算 | 自动队列正确 |

数学检查使用小合成向量即可；GPU smoke仅工程seed2022，在真实batch上分别完成源步、A修改、B/C读出和学生步，并测一完整epoch/完整V。不能用smoke AUC进行候选淘汰。不扩展与当前动作无关的测试框架或回归范围。

## 6. 实现与独立审阅角色

依据仓库start.md，coder与reviewer由不同子agent承担，除机械run外模型与主agent一致。reviewer检查公式/时钟/对照/状态与数据身份，给出PASS或列关键修正；修正后核对一次即可，不无限增加门。本轮规划审阅不等于未来实现已经验收。

未来实现只修改新实验包及其测试/执行收据，源模型和历史结果只读；必要工程adapter有明确source hash。控制器不得以生成一份预期报告代替真实任务完成。
