# GATS02 恢复实现报告 R2

依据已批准 [RECOVERY_DESIGN_ADDENDUM.md](RECOVERY_DESIGN_ADDENDUM.md) 实现。此次修改限于本轮实验包历史接口、恢复计时/身份/成本衔接、必要回归与README。`objectives.py` 和 `shadows.py` 未改动，研究公式、默认配置、模型、数据、seed、50 epoch、batch40960、FP32、shadow、选点和晋级门保持原样。未运行恢复正式训练或新CUDA Stage0。

## 已修复的实际故障

旧协调器在20条基础轨迹完成后读取顶层encoder/scaler字段抛出KeyError。现在统一从真实 `data_manifest[dataset].features.encoder_hash/scaler_hash` 读取；协议对照仍使用 `p.datasets[dataset].encoder_hash/scaler_hash`。同时核对原始CSV SHA、六数组、split行号/指纹和按d0/d1/d2排序的train人口，以及历史resolved与当前协议的training/model/optimizer/rng四字典。原逐臂lr/wd/betas/eps/lambda、A1 κ、固定rho仍逐项核验；未生效的A0 κ不参与身份判断。study/source/commit等执行标识分别保留。

共享核心 `recovery.historical_endpoint_core` 接收扁平数值端点。CPU预检从旧完成ledger和完整state_summary提取48个真实A0/A1 online/fixed selected/E10/E50记录；实际队列完成重算后，RAM wrapper仍调用原checkpoint_rows生成记录，再调用同一核心。两种适配没有伪造model/optimizer状态。历史映射保持A0/A0_EMA/A1_EMA和S1 C、S2 ML G1/KR G2回退；每个端点必须唯一命中且epoch/model hash相同。

缺文件、必需identity、冻结配置、端点列或端点hash时，返回对应数据集verified=false并写具体原因；不把缺失值彼此相等视为通过。历史身份拒绝仍进入原新增配对baseline预算路径。CPU预检和共享核心均不读取test_metrics.csv。

## 原始时间、成本与跨attempt身份

CLI新增 `--mode preflight`、`--recovery-attempt` 和 `--task-start-unix`。恢复必须显式传入原始首launch时间 `1793549242.1426222`，并与旧launch ledger首行核对。派发统一使用原始10h/12h绝对deadline；原相对elapsed逻辑采用同一起点的虚拟perf_counter起点，不能重启计时。

代码从旧20条COMPLETE运行的真实runtime_stats计算旧成本，读取旧throughput_calibration中的原预测。旧36个完整物理pass、21367.9183922371 worker秒和原17870.690975639038秒预测继续计入超预算判断；fresh calibration观测另列，不覆盖原预测。最终成本收据分列旧、新、合计曝光、view、forward、梯度遍历、optimizer/shadow更新、validation、GPU event及worker秒。科学去重按跨attempt身份成立后计算，不把旧20+新20称为40个独立重复。

fresh基础队列完整重算后，在控制/晋级前逐轨迹核对有效完整config、init hash、每step online hash、epoch行序、view hash、最终Adam、selected epoch/validation，以及每个selected/E10/E50端点model/optimizer hash和指标。原、新run/source/commit分别记录，成功映射写cross_attempt_identity收据；不一致即阻断晋级并要求独立review。全体最终test仍只能使用fresh attempt实际RAM模型。预算提前停止时同样先核对已经完成的恢复轨迹，再冻结/test/汇总；原attempt始终不动。

## 实际必要验证

- 完整CPU测试：`python -m pytest -q experiments/gradient_aug_time_select_20261102/tests`，**4 passed，1.74s**。新增回归使用实际历史目录和旧attempt的48个真实数值端点；删除真实嵌套features.encoder_hash后ML复用被拒、KR保持通过；改一个端点hash后对应数据集被拒且47/48匹配。原始10h/12h数值边界不随恢复起点改变，原确认新增baseline计入12及全局40检查继续通过。
- 真实CPU CLI预检：两数据集均verified=true，**48/48**模型hash/epoch匹配，所需四字典、冻结配置和数据身份均通过。最终源码收据位于本目录 `recovery_cpu_preflight_final/historical_preflight.json` 与 `recovery_execution_receipt.json`；初次同结果预检保留在 `recovery_cpu_preflight/`。未读取test、未加载权重。
- `python -m compileall -q experiments/gradient_aug_time_select_20261102`：通过。

实际预检命令：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_time_select_20261102.run --mode preflight --recovery-attempt artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005 --task-start-unix 1793549242.1426222 --run-dir research_reports/2026-11-02_gradient_aug_time_select/recovery_cpu_preflight_final
```

独立review通过后，以新不可变commit执行新Stage0，再以新空attempt执行all；两者都传相同旧attempt与原始起点。旧Stage0 source hash已不同，不能沿用。CLI命令与RAM生命周期见包README。
