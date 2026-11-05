# CODER IMPLEMENTATION REPORT C

状态：C_IMPLEMENTED，等待独立 REVIEW。依据 `DESIGN_SPECIFICATION_C.md`、route_C 四份冻结规范和原 manifest，仅实施 MovieLens C_FULL。B/A 原始产物、冻结身份和已执行数学路径保持不变；没有新增依赖、KR 科学 job、额外搜索或 C0 科学训练。

## 实施范围

- `model_adapter` 新增绝对 δ embedding hook，以及仅 probe 启用的 skip/star 捕获和 gate pre-hook。δ 同时作用于各模型自己的 embedding；正式概率前向及 A 参数探针仍保留源 gate.detach。raw tower logits、所有临时 hooks、参数和 `.grad` 保持只读语义。
- `drift_probe`：独立 train d0 自然均匀、无放回最多128行，排除当前 raw batch IDs；两任务共享行和同一 δ，每任务一次输入求导。只有 δ 有图，online/teacher 权重 functional detach。
- `covariance`：FP64 单位漂移外积、含零样本的实际 n 分母、PSD 数值检查；有符号矩阵指数根保证 trace16，A=0/root=I。MERGED 使用自身两任务均值；ROTATED 使用独立固定每字段正交 R，保存自身 R/A/root。
- `conditional_views/objective`：原 Gaussian key 和字段顺序的一次基础 ξ，两任务以 root 变换；源 apply_view 只乘一次 κ 与 detached RMS。不同任务视图只监督各自任务、实际2N分母；相同 root 共享增强前向；γ0直接走源单视图 loss。
- `conditional_state/worker`：每条独立 online/Adam 轨迹只维护一个固定EMA教师。刷新在1+K、任何训练 forward 前读取旧 θ/φ，正式 Adam 后教师更新一次。当前 teacher 与验证 scratch 分开，快照包含 teacher/A/root/R/γ/K/时钟/probe身份与原 online/Adam/loader/RNG/端点。
- `c_search`：在既有 scheduler/guardian 上增加四阶段数值依赖层。初始C1/C2共10条job；W12五seed冻结后才解析C3，W123后才解析C4，C*冻结后才解析两个匹配控制。每配置先写不可覆盖的数值解析收据；job hash绑定γ/K/mode及依赖身份。没有nullγ的可运行job或单seed预筛。
- C 的最终 select 审计全部四阶段、六候选共30条ML轨迹、各50次完整 online/EMA验证、依赖链与五seed；保留全部HPO/曲线。test前冻结C*、两个控制及各自online端点；validation决策固定STOP_AND_REPORT。输出逐seed、六格/总体、配对SD/95%CI及实际成本，科学解释由主代理完成。

## 身份、范围与公开入口

使用新 `artifacts/aug_history_20261105/attempt_C1`。`preflight --route C --datasets movielens` 将 `scope_datasets=[movielens]` 写入 PREPARED/identity/RUN_MANIFEST、数值解析、HPO与冻结。CLI范围冲突报错；review/engineering smoke只检查此范围，不要求KR smoke。

跨阶段 `continue` 按A自身PREPARED/selection身份核验ML的RUN_C动作，只在C输出写不可覆盖的 `ENTRY_CONTINUATION.json`；B/A冻结不回写。γ0父F/online五seedselected/E10/E50引用B冻结，核验同数据/预处理/父参数/源码，保存旧source/trajectory/epoch/model/hash；中间HPO只读父validation。最终评价再读取已冻结的父端点源表，新增父训练为0。

独立收据为 `REVIEW_ACCEPTANCE_C.json`，绑定最终新source hash。正式流程：preflight→validate/review+GPU smoke→跨阶段continue→train内部完整四阶段→select→decide(validation)→evaluate→report→continue(STOP_AND_REPORT)。现有RAM失败恢复、幂等提交和重放直接复用。

GPU smoke实现检查两个γ0原batch更新及EMA逐位等价，再执行到step21的真实非零漂移/条件视图路径并记峰值；只判工程正确性。Coder未运行GPU或正式C训练。

## 必要检查

全部项目Python命令显式使用 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

```text
python -m pytest experiments/aug_history_20261105/tests/test_c.py -q
7 passed, 1 warning in 10.72s
python -m pytest experiments/aug_history_20261105/tests/test_b.py -q
11 passed, 2 warnings in 17.33s
python -m experiments.aug_history_20261105.run --help
exit 0
```

覆盖：FP64有限差分验证完整gate δ导数和各自embedding、θ=φ零漂移、hook/梯度/RNG纯性；PSD/root/trace/γ符号、旋转谱、共享ξ及纯合成期望能量；γ0源loss/grad/Adam/EMA逐位等价；任务视图与尾批2N；K20/K40旧教师刷新与RAM恢复；五seed依赖、数值job identity、冻结不可覆盖、ML范围和test门。

共享/B的11项回归检查和CLI help通过。C正式预检、独立审查、原batch GPU smoke和完整30条训练由后续RUN阶段执行。
