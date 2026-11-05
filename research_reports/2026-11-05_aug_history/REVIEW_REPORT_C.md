# INDEPENDENT REVIEW REPORT C

Verdict：**PASS**。Blocking issues：**无**。批准范围：**MovieLens C_FULL**；KuaiRand C 为 DEFERRED_NOT_RUN。

独立审查了 `DESIGN_SPECIFICATION_C.md`、`CODER_IMPLEMENTATION_REPORT_C.md`、冻结 C1–C4/公共合同及实际新增与修改源码。计划为 `2645f1f724f52f5ed5dea4b479335c0dc12fabcd`，骨干为 `c8675ae2923c15b6965884b140b5b646690cd95e`。B/A 既有报告、审查记录和原始冻结产物保持原身份。

## 实现与合同核对

| 范围 | 独立核对结果 |
|---|---|
| δ 与 raw logits | 绝对 δ 仅加到6个非路由字段；online、teacher 各自 lookup，共用 δ 叶。raw tower 按 `3*t+d` 拼接；probe 专用 gate pre-hook 接回 live STAR+skip，保持前向值，正式训练和 A 仍走源 gate.detach；hooks 在 finally 清理 |
| 探针隔离 | 仅 train d0、排除当前 batch raw IDs、自然均匀无放回最多128行，两任务共用行。权重 functional detach，只求 δ 的两个任务梯度；局部 SHA256-key RNG 不消费 loader/Gaussian/全局流 |
| 统计与能量 | FP64 单位漂移外积，实际样本分母包含零向量；覆盖式刷新、PSD 数值检查和小负值截0；矩阵指数按谱归一化为 trace16，root 对称，A0/rootI 退化；不增加 jitter 或改 γ |
| 共享噪声与控制 | 源 key/字段顺序的一次 ξ，任务 root 只作线性变换，κ/RMS 由源 apply_view 各乘一次。MERGED/ROTATED 使用各自 θ/φ 的统计，固定字段旋转独立抽取、留 RAM 并恢复 |
| 正式监督 | 两视图分别只监督各自任务，原 clean 矩阵和实际2N分母；相同 root 共用单前向；γ0直接调用源单视图 loss，保留 loss/grad/Adam/EMA 次序 |
| 教师次序与 RAM | 1+K 时钟在任何训练 forward 前读旧 θ/φ；唯一 Adam 后教师只更新一次。当前 teacher 与验证 scratch 分开；RAM 保存 teacher/A/root/R/config、尝试/有效刷新步、probe 身份及 online/Adam/loader/RNG/选点端点；延续既有 epoch/owner 恢复事务 |
| 独立科学单位 | 每数值配置×seed独立 online/Adam/teacher，不共用候选历史。计划新增ML六配置×五seed=30条、每条50epoch；γ0复用父F/online，新增科学训练0条 |
| 四阶段搜索 | C1/C2各五seed完成后冻结 W12，再解析 C3；前三候选完成冻结 W123，再解析 C4；四搜索候选冻结 C_WINNER 后派发两个 γ/K 匹配控制。解析收据与 job hash 绑定数值 γ/K/mode 和依赖身份，无 nullγ job |
| 完整性与冻结 | 各候选五seed、50轮 online/EMA 验证和三个 RAM 端点齐全才选型。HPO 用固定EMA validation、相对父F逐格保护及同一并列规则；保留全部六候选 HPO/曲线，最终冻结 C*、两控制及对应 online |
| 跨阶段与范围 | 新C source/prepared/acceptance；scope 仅ML贯穿 preflight、job、smoke、train、HPO、冻结与报告。旧 B 父F/online和旧 A 授权各按自身冻结身份读取，源文件 hash 记录，不回写旧产物 |
| 后续与报告 | 新 test 前冻结 validation 决策；MERGED可按既定收益/成本规则简化。完整/组件/控制/online比较分别生成逐seed、六格、总指标、SD/CI与成本，最终 continue 为 STOP_AND_REPORT |

## 独立验证

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/aug_history_20261105/tests/test_c.py -q
7 passed, 1 warning in 14.93s
```

覆盖 FP64有限差分的完整gate δ导数与各自embedding、前向值/梯度/RNG/hook纯性、零漂移、PSD/root/trace/γ符号与旋转谱、共享ξ和纯合成期望能量、γ0源逐位退化、任务视图/尾批分母、K20/K40旧教师与RAM恢复、五seed依赖/数值身份/范围/冻结门。唯一 warning 为父源码 TF32 setter 的弃用提示。复用 coder 已通过的共享/B 11项检查与 CLI help 证据，不重跑无关范围。

另以实际指标字段的 `ENGINEERING_ONLY` 标量夹具串联真实 `train_stages→select_C→decide_C→report_C→continue`：

- 阶段顺序为 `[C1_NEG,C2_POS]→[C3_WEAK]→[C4_CADENCE]→[C_MERGED,C_ROTATED]`。
- 生成30个互异job ID、各候选五seed及50轮标量记录；3000条 online/EMA验证记录、6个HPO行、90个冻结端点身份。
- C3符号/弱强度、C4刷新间隔、两个控制的 γ/K 均由既有依赖选择准确解析；重复进入时 W12/W123/C_WINNER 的文件 hash 不变。
- 最终可选验证更优的控制，决策和 continue 均为 STOP_AND_REPORT；完整数表串联得到330行端点、3366行配对汇总。
- 使用真实旧 A/B 的授权、父F/online和发表模型来源；检查前后七个旧阶段准备/冻结/决策/端点文件的 SHA-256 均不变。

上述标量夹具使用临时目录，未生成正式训练结果或修改实验源码。首个夹具缺少完整指标列，补齐源指标 schema 后完成了报告串联；实现无需修订。

## Acceptance 与 RUN

- 当前 source hash：`d78a938330e5eb8334213c7f0beed66926956c35e1cff90c180260d12fd331e2`。
- manifest SHA-256：`76345194a8d09c8cadf3544e39c35174a452abb3827f071250da7f259c45749d`。
- `REVIEW_ACCEPTANCE_C.json` 绑定上述 source、manifest 和 ML scope。

可进入 RUN：先记录 implementation commit，执行新 C attempt preflight 与 C 工程 gate，再运行 batch40960 的 γ0源等价及非零旧教师漂移/条件视图 GPU smoke。相同 source/scope 的现有 gate 通过后，启动完整30条ML矩阵。GPU smoke、正式 C 训练和科学分析由后续阶段完成。
