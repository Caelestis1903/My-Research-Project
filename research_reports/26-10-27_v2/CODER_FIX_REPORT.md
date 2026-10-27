# CODER FIX REPORT — FP32 跨 batch 验收

日期：2026-10-27。状态：IMPLEMENTED_PENDING_INDEPENDENT_REVIEW。

首轮MovieLens seed2022的READ完成20轮后，`diagnostics.latency`因跨batch的纯绝对1e-6检查退出。最初异常缺少实际误差值，RAM状态随进程退出丢失。该阻塞已按批准的 `DESIGN_RUNTIME_NUMERICS_ADDENDUM.md` 修正比较和异常证据。

## 根因与直接证据

使用同seed2022、原训练配置和已验证数据重放native24/pilot12/READ20；跳过无关B_REFERENCE/DIRECT及其时延测量。重放的native、source、READ selected模型hash全部与失败run精确一致：

- native：`18ed1687115098e2ebffb8c81f57c56622bb6a8972ca8d660ae8efc34e370866`
- source：`0e35e42c7db9891031f0a05a5e28b837dcda159ff2f9d2c60c5666475d53da28`
- READ：`a2e7f758a46e1139b7555c98d2eace6984bc426147dc8dbc2a1b5bb9b1a31838`

| 检查 | 最大绝对差 |
|---|---:|
| batch1 final logits | 2.384185791015625e-7 |
| batch256 final logits | 2.384185791015625e-6 |
| 相同4096上下文重算native h、z | 0、0 |
| 相同40960上下文重算source phi | 0 |
| 跨形状native h、z | 1.6689300537109375e-6、9.5367431640625e-7 |
| 跨形状foreign phi | 4.1961669921875e-5 |
| 完整Q40008行，raw4096与缓存最终logits | 0 |

完整Q六格、macro、natural LogLoss和六格/macro AUC差均为0。batch256最大误差点row205/task1的cached值为−4.291538238525391，raw值为−4.291540622711182；相对差约5.56e-7。原始source hidden跨形状差约1e-6，经T0标准化放大；三个来源分别有11/11/16个std=.001下限维度。同形状上下文的逐位复现说明当前路径的失败来自FP32不同GEMM批形状的舍入差。

直接数据位于 `artifacts/mdmt_relation_basis_v2/diagnostic_repair/seed2022_v2/REPRODUCTION.json`、`point_differences.json` 和 `Q_logit_comparison.npz`，完整命令脚本与stdout在其父目录。E未读取。成功重放整体40.946934444秒；之前一次诊断输出脚本遇NumPy int64 JSON序列化错误，其已测native/source/READ组件耗时合计39.780905704秒，日志和全部产物保留。成本账为 `diagnostic_repair/DIAGNOSTIC_COST_LEDGER.json`。

## 最小修复

新增 `compare_raw_logits`，仅供latency的batch1/256最终logits比较：

`abs(raw-cached) <= 1e-6 + 1.3e-6 * abs(cached)`

先检查shape一致和两端finite，明确cached为参考端。通过记录包含batch、row-id身份/范围、atol/rtol、finite/passed、最大/平均绝对差、最大差值位置、两端值及该位置允许误差；失败将同一记录编码进入ValueError消息，由既有worker FAILED.json保留。

teacher转换、RAM恢复和同上下文的原绝对1e-6检查保持原值。模型、训练、缓存chunk、dtype、采样、优化器、预算、选点、LL/AUC和50warmup/200时延计数均未改动。没有新增逐run完整Q重算。

## 验收与修订身份

针对性CPU验收：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_relation_basis_v2/tests/test_runtime_numerics.py experiments/mdmt_relation_basis_v2/tests/test_contract.py::test_ram_restore_next_update_and_identity experiments/mdmt_relation_basis_v2/tests/test_contract.py::test_native_and_source_teacher -q
```

结果：**3 passed**，见 `CPU_FIX_ACCEPTANCE_STDOUT.txt`。新增测试覆盖实际失败点通过、近零超过atol拒绝、显著误差拒绝、NaN/Inf拒绝、shape错误拒绝，并验证异常证据可解析；原RAM下一步恢复和教师转换通过。

更新后的逐文件哈希、运行源码身份和补充设计hash记录在 `IMPLEMENTATION_SOURCE_MANIFEST.json` 与 `SPEC_RESOLUTION.json`。原4个失败/主动中止worker目录和诊断目录保留。主agent指定正式重启目录为 `artifacts/mdmt_relation_basis_v2/revision1`；须独立reviewer复审通过后再运行。

未重新启动完整实验矩阵；本次GPU工作仅为直接相关重放诊断。RAM模型未写磁盘。
