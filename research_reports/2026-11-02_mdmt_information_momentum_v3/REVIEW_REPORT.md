# Independent implementation review

Verdict: **PASS**. No blocking implementation defects were found. The reviewed source hash is `5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a` (52 Python/JSON files). Real-data S0 remains the next required stage before formal training.

The review compared DESIGN_SPECIFICATION, CODER_IMPLEMENTATION_REPORT and the actual new package, including the runner and RAM guardian. All 40 vendored files match the pinned EMA02V2 Git blobs; the implementation source hash independently matches its recorded manifest. No algorithm files were changed during review.

## Scientific and execution semantics

- Boundaries use observed joint labels and within-domain positive/negative rows. Target weights retain missing-cell mass, probabilities normalize on active conditions, and dynamic demand is read before its post-step update. Pair streams exclude arm/lambda and do not consume the loader stream.
- Tower hooks retain the training graph and leave the source gate detach intact. BCE uses actual batch/task mean; pair loss is additive. Lambda zero and A0/A1 call the original numerical path. A1 retains the original EMA02V2 context, and initialization/loader seed and order follow the source.
- Shared history membership follows outer module names, including shared experts whose inner layer names contain domain_specific. Other parameters update through ordinary Adam. Each parameter is updated once.
- ECCM applies frequency-corrected selected history refresh, active-condition clocks, source-channel first moment and one coupled decay contribution to the current second moment. CV reads old tables before correction and writes the selected table after Adam. PCOLL has one independently named collapsed table and pair clock. Empty conditions suppress history injection; ordinary None/explicit-zero semantics are preserved.
- Parameters, EMA, optimizer/history/demand, RNG, loader and selection state remain in RAM. Epoch snapshots retain E10, each estimator's earliest strictly best validation state and E50. The source model has no dropout or batch-normalization state that would be changed by the hook check.
- The runner requires a matching independent-review source hash and S0 PASS before dispatch. S1 contains 24 unique trajectories. Later stages require explicit orchestrator dispatch, use only permitted seeds and enforce the registered development/confirmation caps. Scientific promotion remains the orchestrator's responsibility.
- Test evaluation is reached only after final decision and selection freezing, checks each model hash, evaluates all six selected/E10/E50 online/EMA endpoints, and releases RAM after completed scalar output. Source evaluation retains six-cell, macro, pooled, traffic, task/domain and loss metrics.

## Independent checks

`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/conditional_information_momentum_20261102_v3/tests` completed with **14 passed in 2.47 s**. Coverage includes finite-system expectations, observed support, source lambda-zero equivalence, eight-arm RAM serialization/next-step equivalence, hand-calculated coupled decay/history updates, active versus selected clocks, empty conditions, None/zero and CV old-table ordering. The reviewer also recomputed source hashes, checked all 40 vendor blobs and constructed the initial 24-job dispatch.

Real CUDA/full-batch data identity, lambda-zero parity and P11/CV/PCOLL smoke checks are implemented in S0 and must pass before the formal queue. This review did not run GPU training or inspect research outcomes. It did not add unrelated validation or change the experimental matrix.


## 2026-11-03：P01/P10独立补充执行审查

Verdict: **PASS**；无阻塞项。审查对象为 buchong.md、DESIGN_SPECIFICATION 第11节、coder补充实施说明、补充attempt的四份配置/来源JSON，以及既有训练与RAM评价接口。训练源码仍为 `5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a`，未修改算法。

独立构造并逐项比对8个job：两自然数据×P01/P10×2022/2024，固定lambda=.05、50epoch、batch40960；P01为静态ECCM，P10为动态普通Adam。配置标记本次为停止后的独立补充，S2仅为既有runner接口标签，原停止决定不变，无搜索或S3授权。

原S0记录在新attempt中逐字节一致；106个被配置/身份记录引用的文件哈希已核对，12条A0/P00 .05/P11 .05逻辑对照与原冻结身份关联，复用coder记录的7,344项精确标量核对。原14项CPU及真实S0证据在同一source hash下继续有效，本次不重复运行。旧INDEPENDENT_REVIEW_RECEIPT文件哈希是补充字段追加前的历史快照；该原哈希保存在本次receipt子对象中。

既有schedule接受该独立attempt和已完成S0证据；freeze要求任务全部COMPLETE和协调者最终决定，评价使用selected/E10/E50内存快照及冻结hash，新48测试端点完成后释放RAM。旧12条对照只复用数值，不重训、不声称新前向。

`build_tables.py::supplement` 按同dataset/seed/role/checkpoint/estimator配齐A0与四PAIR臂，四简单效应P01−P00、P11−P10、P10−P00、P11−P01及交互P11−P10−P01+P00正确；交互先逐seed计算再汇总，t区间按n=2、df=1。新旧端点均保留origin标识，旧表不覆盖。最终解释须按AUC与LL各自方向理解符号，并保留selected效应包含选点策略的限定。

未运行GPU训练或重复CPU/S0；本审查允许固定八条补充任务启动。补充结果与实证结论仍待完成后检查。
