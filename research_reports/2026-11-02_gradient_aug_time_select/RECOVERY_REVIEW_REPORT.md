# GATS02 恢复补丁 R2 独立审查

**Verdict：PASS。无未解决 Blocking。** 允许冻结本次恢复源码，随后重新执行与新commit/manifest一致的两数据集真实CUDA Stage0；Stage0通过后才启动fresh完整重算。

本次由独立于coder的reviewer完成，输入为已批准的RECOVERY_DESIGN_ADDENDUM、实际run/recovery/tests/README、RECOVERY_IMPLEMENTATION_REPORT及 `recovery_cpu_preflight_final/` 实际收据。旧REVIEW_REPORT与旧attempt均保留原样。

## 修复与恢复合同核对

| 项目 | 实际核对 |
|---|---|
| 已触发schema故障 | `recovery.py:24` 共享核心从历史manifest的 `features.encoder_hash/scaler_hash` 读取，对照protocol顶层hash；保留raw SHA、六数组、行号/指纹、域人口、training/model/optimizer/rng与逐臂冻结超参/rho。缺失值使用独立MISSING标记，缺失或None不因两侧相同而通过。 |
| 真实48端点 | 旧COMPLETE ledger和完整state_summary提供A0/A1×两数据×两开发seed×online/fixed×三个端点，共48条。每数据必须有24个唯一键，历史查找必须唯一且epoch/model hash一致。实际CPU预检及后续RAM wrapper调用同一个核心；RAM wrapper仍由checkpoint_rows生成真实端点，没有构造模型状态替身。 |
| 无test读取 | 共享核心仅打开validation/development/frozen config/data manifest/resolved config五种历史数值文件；预检不加载权重、不读取test CSV。test导入仍在统一冻结后执行。 |
| 时间不重置 | `task_clock` 从传入的原始UNIX起点计算全部deadline；recovery_context核对它等于旧launch ledger首launch。main把原起点传入，all_stages的虚拟perf_counter起点与schedule绝对deadline一致，wait/附加/确认限制亦沿用它。 |
| 旧成本与原预测 | 旧20条COMPLETE实际runtime计入36个物理pass、21367.9183922371 worker秒；原17870.690975639038秒预测保留，fresh校准另列。超预估比使用旧+新worker秒，最终execution_cost_continuity分列旧、新、合计，不把恢复成本清零。 |
| 跨attempt身份与阻断 | `verify_recomputed_trajectory` 检查有效完整config、初始化、数据身份、每step online hash、epoch行序/view、最终Adam、selected epoch/validation，以及全部保留端点model/optimizer hash和指标。基础20条完成后、控制/晋级前调用；不匹配写MISMATCH并抛出阻断，不能选择较好结果继续。预算提前停止和最终统一test前也检查已重算轨迹。 |
| 科学与执行计数 | context保留旧20条与旧run/source/commit；成功映射后恢复20条仍为同20条科学轨迹，旧新pass/成本相加。原确认预算函数保持，补跑baseline也计入确认≤12，附加≤8、科学总量≤40不变。 |
| RAM与源码边界 | recovery只读取旧JSON/CSV数值、hash和日志；旧权重没有恢复。fresh模型/Adam/shadow/RNG仍使用原RAM snapshot/IPC，最终test使用fresh RAM模型。新Stage0门继续同时核对PASS、source manifest和执行commit。诊断energy缺口没有借恢复修改。 |

## 独立必要验证

1. 读取最终真实CLI预检：两数据集verified=true，48/48端点匹配，48项固定身份检查均匹配；无test读取/权重加载。最终源码manifest为 `95c5d28204849e2e815ff33753339a1ef115fdbb2401886923435b6e9511299d`。
2. 独立执行指定环境 `python -m pytest -q experiments/gradient_aug_time_select_20261102/tests`：**4 passed，1.68s**。实际旧48端点通过；删除真实嵌套encoder身份时对应数据集拒绝；改模型hash时只47/48匹配且拒绝；原始10/12h边界与确认12/全局40回归通过。唯一warning仍为冻结源TF32设置API。
3. 原始起点 `1793549242.1426222`，10h截止 `1793585242.1426222`、12h截止 `1793592442.1426222`；更晚的recovery_start不会延长它们。实际最终恢复收据记录相同数值。
4. 对照旧执行 `ac2ec5cc70be7dcb880a7dde1015e4634726fd96`：objectives.py与shadows.py字节SHA不变；run中的counters/train_pass/train_one/worker/make_jobs/confirmation_budget_reason AST完全一致。恢复改动没有改训练数学、原pass执行或预算函数。

此前故障发生在20条基础轨迹完成后、控制和统一test之前；本补丁修复接口并延续同一次任务的预算。旧attempt的RAM已经丢失，恢复仍需完整重算。尚未执行本次新CUDA Stage0或恢复训练；这两步由runner在新不可变执行身份下完成。
