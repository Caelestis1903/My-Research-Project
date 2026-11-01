# GATS02 独立实现审查

审查日期：2026-11-02。审查代理与 coder 不同，沿用协调者模型。输入为已批准的 DESIGN_SPECIFICATION、CODER IMPLEMENTATION REPORT、实际新包，以及规划02/03/05/06和所复用的GA31/GAM/AIGM接口。

**最终 verdict：PASS。已发现的一个 Blocking 已修正，无未解决 Blocking。可以冻结此版本，交 runner 执行两数据集真实 CUDA Stage0；Stage0 PASS 且源码 hash/执行 commit 一致后才能正式训练。**

## 对照结果

| 范围 | 代码与设计核对 |
|---|---|
| 数据与底座 | 复用冻结自然预处理、六数组/行号/encoder/scaler检查、fixed1模型和真实metrics。原batch40960、50epoch、FP32、Sample目标、单标准Adam和coupled wd保留；ML κ=.05、KR κ=.1，ML固定rho使用原H2精确值。 |
| A0/A1与关闭分支 | 直接调用原Gaussian scalar并单次backward；λ0/κ0/空目标域不抽视图。B1/B2/B3主开关关闭返回源A1路径，真实Stage0有参数、view hash及梯度调用数对照。 |
| B1/R1 | detached双任务平均BCE的average-tie rank控制字段Bernoulli；mask和Gaussian使用独立keyed generator。R1是真实完整训练，按B1每step/field数量均匀无放回选择，再以detached尺度精确匹配扰动平方能量；不读验证难度。概率/row_id数值NPZ和mask/key/能量CSV提供原始证据。 |
| B2/R2 | 同theta/batch先求full-real，再分别求两个task的clean和delta，单task除原B，投影后再均值一次，共5遍梯度。E/P整个参数组FP64聚合内积/范数，系数detached；U差分须为None或精确零。wd仅由标准Adam处理，未进入delta/参考g。R2按两个开发seed全部训练step/task的retained/raw范数和之比逐组冻结，真实独立训练。 |
| B3/R3 | 延迟5epoch、其余λ=25/45；随机5个关闭epoch用1729预先冻结；累计epoch/step/样本权重均记账。附加delay10配套自己的等剂量随机10-off控制，属于批准的同配置匹配预算。 |
| 时间平均 | A0固定/θ5初始化的延迟EMA/SWA，SWA只在E31末起有效，E10明确N/A；A1/B固定/adaptive/replay。CV读取wd前实际optimizer输入梯度范数、100step population std/mean并按注册范围映射rho。完整rho值序列独立1729排列后确定性重算，逐steponline hash、最终Adam、行序及view一致才合并独立replay端点；所有重算成本计入。C2固定锚定A1。 |
| RAM与test | 原始快照/IPC只在RAM，包含模型、Adam、shadow/controller、RNG/loader和选择状态。selected/E10/E50保留至全体训练及规则冻结；统一test前写完整端点身份，之后没有训练派发。历史test表只在统一test阶段导入，不进入本轮选择。 |
| 历史配对 | 历史数据/冻结超参/rho核验后，逐一比较当前A0/A1开发online/fixed的selected/E10/E50模型hash与epoch；不匹配则补真实同seed基线，补跑计入确认及全局预算。历史数值复用保留来源，不恢复旧权重或增加seed数。 |
| 队列与门 | 基础20条全部完整50epoch；控制/附加≤8，R2/R1/R3优先，未匹配不晋级。默认已能裁决则优先确认；探索η/delay附加需同时容纳自己的两seed控制。τ2没有猜测公式，真正需要时保持RAM等待已审查resolution。最多一个B进入C1，另有独立C2，最多两候选。确认新增基线+候选≤12、全局≤40。 |
| 资源与成本 | 逐卡显存/功耗均低于85%及容量/RAM条件才并发，首次完整A0校准，保留外部任务；OOM只降并发原配置重试，最多两次可恢复重试。每pass记录曝光、前反向/优化器/shadow更新、验证及选点、峰值、训练秒和RAM字节；test单列成本。 |

## Blocking与修订历史

**R1：确认预算缺少独立12条上限，已修正。** 初版 `run.py` 确认排程只限制全局40条；历史基线不匹配时，单B候选跨两数据集可能新增12条A0/A1基线及6条候选，合计18条确认、全局38条仍被接受。路径为“未复用历史基线 → 多派发6条确认 → 超出注册预算并改变候选保留决策”。

现版 `run.py:246` 的 `confirmation_budget_reason` 同时限制确认≤12与全局≤40；`run.py:544` 在每数据集完整基线+候选组派发前应用，容量不足整组取消并记录原因；`run.py:557` 在实际派发前再次断言两个上限。新增小测试覆盖两数据均需9条完整配对组时只接受第一组，以及全局40条约束。实现报告已记录R1。

## 必要验证与交接

独立运行 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/gradient_aug_time_select_20261102/tests`：**2 passed，1.36s**。涵盖已有公式/RNG/数量能量/参数平均/RAM恢复检查及本次确认预算修复。唯一warning来自冻结源TF32设置API，不改变FP32设置。已有syntax、CLI和20-job身份检查收据复用。

本次未运行CUDA或正式训练。真实Stage0负责核验实际U差分、原batch资源、关闭退化和真实模型RAM恢复；其源码manifest与执行commit门已在all入口落实。审查到此完成，不追加候选、性能试验或无关测试。
