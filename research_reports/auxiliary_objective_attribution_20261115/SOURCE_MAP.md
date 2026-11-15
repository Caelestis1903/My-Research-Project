# 固定源码与责任映射

main固定提交：`e5738e52294f064c23a9b953fe74abb84de713df`。来源pilot固定提交：`52b48c9aab066d56fa91a060864c7c9abf12e50c`；训练实现提交 `bf36e454adc2d0c569218a7993834f6d56cfc97f`。

令 `P=experiments/conditional_information_momentum_20261102_v3`，`V=P/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31`。

| 责任 | main中可复用位置 | 已核对语义 |
|---|---|---|
| 协议、数据入口 | P/protocol.py | 已迁移CSV路径、natural population、rho/WD、固定source identity |
| fixed1 factory | V/model.py::make_model（P/protocol导出） | embedding16、FCN512/256/256/64、4expert、四系数1；无BN/Dropout/learned fusion |
| 实际forward | V/reference.py::MDMTRec.forward | `[features,B]→[B,2]`；gate输入emb.detach；tower序task*3+domain |
| clean loss | P/vendor/ema02/objectives.py及V/resolved_protocol.json | 所有自然row和两个task的概率BCE均值 |
| 原始预处理 | V/audited_data.py；P/vendor/ema02/data.py | natural80/10/10、train-only encoder、raw/array/rowID/split hashes |
| Adam/EMA | P/vendor/ema02/protocol.py | single coupled Adam；一次after-Adam EMA；buffer复制 |
| 初始化/RNG/状态复制 | P/protocol.py导出的setup/cpu_copy/rng_state/restore_rng/state_digest | FP32/确定性配置与来源一致；状态可RAM复制 |
| validation metrics | P/vendor/ema02/evaluate.py | unperturbed六格、Macro、Traffic(sample_weighted_cell)、Pooled(task pooled mean) AUC/LL |
| RAM guardian | P/guardian.py | BytesIO IPC存model/optimizer/EMA/loader/RNG/selected状态，磁盘仅元数据/日志 |
| 来源采样和perturbation | pilot experiments/heterogeneity_aware_20261115/core.py | 已审计的键控sample、sensitivity、B pair cap、C BCE方向；新代码在独立目录实现必要adapter |
| 来源worker | pilot experiments/heterogeneity_aware_20261115/run.py | 50epoch、40960、seed+11000独立loader、每epochval、earliest strict max、RAM状态 |

52个P目录中的`.py/.json`与来源收据逐字节匹配；具体文件hash写在STAGE0_AUDIT.json。来源B/C的batch汇总没有行级support，故新匹配重跑需要完整trace。

来源可审阅链接：[pilot代码](https://github.com/Caelestis1903/My-Research-Project/tree/52b48c9aab066d56fa91a060864c7c9abf12e50c/experiments/heterogeneity_aware_20261115)、[pilot报告](https://github.com/Caelestis1903/My-Research-Project/blob/52b48c9aab066d56fa91a060864c7c9abf12e50c/research_reports/2026-11-15_heterogeneity_aware/REPORT.md)、[原始输出](https://github.com/Caelestis1903/My-Research-Project/tree/52b48c9aab066d56fa91a060864c7c9abf12e50c/artifacts/heterogeneity_aware_20261115/attempt1)。本轮new implementation不能把这些链接/本机pilot绝对路径作为运行时源码import。
