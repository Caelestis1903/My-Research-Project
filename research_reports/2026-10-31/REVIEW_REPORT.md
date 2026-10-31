# GA31-v2 独立实现审查

审查日期：2026-10-31。审查对象为 [设计规格](DESIGN_SPECIFICATION.md)、[实施报告](CODER_IMPLEMENTATION_REPORT.md) 与 `experiments/gradient_aug_20261031/` 实际实现。

**最终判定：PASS WITH FIXES；所有 blocking 项已修正，可进入 RUN。**

## 修正

冻结清单最初将 selected_epoch 与 E50 的 global_step 配在一起。coder 已将 `freeze_selection` 改为读取 `states[selected_epoch]['global_step']`，审查已复核并用40个合成run的不同 selected/final step 验证。模型选择、test评估使用的快照本来正确；本次修改修正身份记录。

## 实现与合同一致性

- fixed1 参考源码与冻结来源直接 diff，仅有必要的包内导入调整；自然数据审计副本也仅修改包内导入。新模型无额外可训练参数，保留 fixed1、LayerNorm、gate detach 与单 Adam。
- 真实目标使用原 batch 的 B×2 分母，目标域增强替换权重 λ=.5；扰动位于 Embedding 后、STAR/skip 前，按实际 sparse-then-dense 字段切片，保护 domain embedding 与 age dense。RMS尺度 stopgrad，Embedding保留梯度。κ0/λ0与关闭增强折叠为真实目标。
- 每步独立设备 noise generator；每个 run 相同 seed 重建同一初始化，独立 CPU train generator 自然无放回 shuffle。正式协调器遍历40个任务，固定50epoch、原batch40960；严格 validation macro AUC `>` 选模，全部选点与 P0/P1 工作尺度冻结后才统一test。
- model/Adam/RNG/loader快照由CPU深拷贝保留；Pipe/BytesIO IPC没有模型文件。正式父进程不初始化CUDA；调度按各卡显存和功耗均低于85%新增worker，并等首次完整batch分配后才继续判断。OOM重试保留原batch和预算。
- 探针由源快照恢复模型与Adam，每个分支联合deepcopy并核对参数绑定；raw real/extended_aug/target梯度共用原始支持与B×2分母，三个分支各执行一次真实Adam.step，保留None语义。源快照和所有已捕获RNG在探针后直接比较相等。
- 数值分析以seed为单位输出A1/A0.1/S0/S1/I，五seed主分析使用df4和4/5门槛，原四seed敏感性使用df3和3/4。全部六格、domain/task、macro、worst、pooled与流量加权指标分列。

## 验证证据

1. 独立重跑 `python -m pytest -q experiments/gradient_aug_20261031/tests/test_semantics.py`：**3 passed**。覆盖参考预测/loss/gradient/Adam、零噪声与关闭路径、非均衡/单例/缺域、field保护、RNG、RAM恢复和None。
2. 复核 coder 的 `artifacts/gradient_aug_20261031/GA31-v2/implementation_check1/checks.json`：两个数据集全部raw/六数组/split/encoder/scaler身份匹配，8个原batch40960 CUDA检查通过，未评价test。
3. 独立CPU探针端到端：合成数据、真实固定主干和已建立Adam状态，两个去重锚点共4个batch、16次额外backward、22次forward；源快照与RNG不变。证据为 [探针记录](review_checks/probe_cpu/result.json)。
4. 独立真实开发run：MovieLens/P2/seed2022，fork调度完整1epoch、20steps，Pipe/BytesIO收回9,302,308字节tensor状态，无blocked、无test；训练峰值974,205,952字节。证据为 `artifacts/gradient_aug_20261031/GA31-v2/review_dev1/`。本项使用1/8开发短run预算，未据结果调参。
5. 合成40run×3checkpoint表验证 selected step、工作尺度、五个效应与决策门；每split120行。证据为 [合成验收](review_checks/analysis_synthetic/check_result.json)，数值仅用于程序验收。
6. 检查本轮工件中不存在 `.pt/.pth/.pkl/.ckpt` 模型状态文件；只保留代码、日志与数值记录。

本次审查没有启动正式训练或解释科研效果。正式运行仍须完成40×50epoch、全局冻结后test、全部规定锚点探针与seed/域/任务/阶段分析。
