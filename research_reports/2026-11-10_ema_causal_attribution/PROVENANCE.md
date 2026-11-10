# 来源、环境、复现与发布清单

交付日期：2026-11-11（Asia/Shanghai）。20 条正式轨迹、四块归因及 100 个逻辑 Test 状态已完成；新发布分支为 26-11-11。科学结论见 FINAL_REPORT.md；本文件记录来源与复现方法。

## 来源与身份

交付工作目录为 `/home/xzf/Code/phase0-3_ema_causal_20261110`，基础仓库为 `/home/xzf/Code/phase0-3`。该分支从下表的 source commit 建立，发布 commit 包含本轮新增实验、数表与报告。**基础 source commit 不等于包含新增实验代码的发布版本**，复现需使用包含这些交付文件的 checkout，并核对实现摘要。

| 身份 | 值 | 依据 |
| --- | --- | --- |
| source commit | `ffa3de067fcda5a1293af83a465d741e069493f1` | `protocol.py:SOURCE_REF`、两个 attempt 的 SOURCE_RECEIPT/PREPARED；clone HEAD |
| plan commit（冻结声明） | `5cf92af0d3a8ed77688762616248166b9b68266d` | `protocol.py:PLAN_REF`、SOURCE_RECEIPT/PREPARED |
| attempt1 implementation | `575fb1c9bc7f87e900bf56007e3fb21728c92e4e95756c68965313a5818d8b36` | attempt1 PREPARED、执行中止记录 |
| attempt2/current implementation | `4411eec43005a8e432f0de81d426c08a30f09960e84f7b1302e92b29ae4212e5` | attempt2 PREPARED；本次按 `implementation_identity()` 的文件规则和 JSON 摘要规则重新计算一致 |
| attempt2 frozen plan hash | `f2adcddb5535c84fd09d5923991d8a08e6bce051a58ca91921700cf3beaa7a51` | attempt2 PREPARED 的 `identity.plan_hash`；具体冻结文件 hash 见 `identity.plan_files` |
| attempt2 PREPARED SHA-256 | `741df1a95597f516add9ea505006dd1de72947bb24b87d859b8223d1f692a82e` | 文件摘要 |
| current external scheduler SHA-256 | `ec28e47176bae0cb534f9555e75e357b14837a0bf5a588dd78a6fc731156a07c` | [external supervisor](../../experiments/ema_causal_scheduler_20261110.py) |
| report exporter SHA-256 | `2faaee4a193e185a6a17539554ae3673035adb147a2688a0559f64c742bed1b7` | [export_report_tables.py](export_report_tables.py)、EXPORT_REVIEW、导出 manifest |
| environment lock SHA-256 | `5ca1c78d5a1f58636eb3a23b51fd2dbd550c9b2619b9fd918882d622070ebb13` | [requirements-phase0-3.lock](../../requirements-phase0-3.lock) |

科学实现摘要覆盖 `experiments/ema_causal_20261110/` 下全部 `.py/.cpp/.sh` 文件及相对路径，包括测试；不包括 pycache、外部调度器或报告导出器。后两者分别记录文件 SHA-256。不要用当前计划目录全部文件重新计算的值冒充历史 PREPARED 的冻结 plan hash；后续性能/调度修订有单独的设计、review 和执行记录。

当前调度策略的外部 physical-tensor 预算为 7 GiB；原 guardian 的内部 commit predicate 仍为 4 GiB，host RAM 为 90%，每卡显存和功耗分别为 85%。这些计数口径及修订证据见 [EXECUTION_SCHEDULER_AMENDMENT.md](../../research_plans/26-11-10_ema_causal_attribution/EXECUTION_SCHEDULER_AMENDMENT.md)、attempt2 的 `SCHEDULER_REVISIONS.jsonl` 和 `SCHEDULER_DECISIONS.jsonl`。调度器修订不改变上述科学实现摘要。

## 环境与数据

项目解释器：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。SOURCE_RECEIPT 记录 Python 3.13.5、PyTorch 2.9.0+cu126、NumPy 2.1.3、Linux x86_64，执行审计时 CUDA 可见。是否可用 GPU 应在复现机器重新检查，不能沿用这项历史状态。

clone 根目录已有 1,335-byte 环境锁，与原目录 `/home/xzf/Code/phase0-3/requirements-phase0-3.lock` 逐字节摘要相同，因此本次没有复制或修改锁文件。其环境重建方式为：

```bash
conda create --name phase0-3-env python=3.13.5 pip
conda run --no-capture-output -n phase0-3-env python -m pip install -r requirements-phase0-3.lock
```

这是依赖快照，不包含 NVIDIA 驱动、原始数据或 RAM 中的训练状态。已有项目环境无需重新安装。

| 输入 | 本机路径 |
| --- | --- |
| MovieLens CSV | `/home_new/xzf/datasets/ml-1m/ml-1m.csv` |
| KuaiRand processed CSV | `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv` |
| 此实验复用并校验的数据 cache | `/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache` |

数据身份以各 attempt 的 `DATA_MANIFEST.json`、`PROBE_MANIFEST.json`、`PANEL_MANIFEST.json`、`SOURCE_RECEIPT.json` 为准。上述 cache 在本次检查时存在。不要把已失效的 `/home/xzf/data` 路径或未恢复的 `mdmt_next_20261024_strict_v1` 当作本实验的有效输入。异机可替换输入路径，但仍须通过源数组/编码/hash 核对。

## attempt1、attempt2 与报告快照

- [attempt1](../../artifacts/ema_causal_20261110/attempt1/)：`ABORTED_ENGINEERING_PERFORMANCE`。五个 ML-C 轨迹完成，共 250 epochs / 5,000 committed Adam updates，随后固定 pair 重复采样使归因停滞；没有完成的科学 block 表或 Test。详见 `EXECUTION_ABORT_RECORD.md`、`RUN_STATUS_ABORTED.json`、epoch ledger 和日志。这些成本不充当 attempt2 的完成量，结果不拼入 attempt2。
- [attempt2](../../artifacts/ema_causal_20261110/attempt2/)：修复确定性 pair 缓存后，以新实现摘要重跑完整 20 jobs，seeds 2022–2026，每 job 50 epochs。科学矩阵目标为 1,000 epochs / 34,500 Adam updates。已完成四个 BLOCK_COMPLETE、STAGE1–4_COMPLETE、MECHANISM_FREEZE 和 TEST_RECEIPT；最终状态生命周期由 RELEASE_RECEIPT 记录。
- [engineering](../../artifacts/ema_causal_20261110/engineering/)：工程 smoke、fault-injection 和面板准备证据；不同历史 hash 的 receipts 分别保留原身份，不能视为同一正式实验或重复计为新 seed。
- [results/EXPORT_MANIFEST.json](results/EXPORT_MANIFEST.json)：最终导出覆盖四个条件及 Test 五态，提供七图族×四条件、逐 seed/逐 cell 压缩 CSV，以及独立 selected 和 F-selected 同刻两套端点比较。可选来源诊断未触发的条件在 SKIPPED.txt 中保留原因。
- [QUANTITATIVE_APPENDIX.md](QUANTITATIVE_APPENDIX.md) 与 [EVIDENCE_SUMMARY.json](EVIDENCE_SUMMARY.json)：完整 1,604 条统计记录、46 个源文件，missing=0；附录生成器 SHA-256 为 4dba249bdd0e5a673a330e5cb8a3a96c82d80fd2ffa708c059a4451b76a194f9，审阅见 APPENDIX_REVIEW.md。
- [TRAINING_CURVES.csv.gz](TRAINING_CURVES.csv.gz) 与 [training_loss.png](training_loss.png)：全部 1,000 epoch 原标量及训练/验证 loss 图；生成器 plot_training_curves.py 的 SHA-256 为 a0b8df305c9ccddd22ccc10a0b70542c0cb5ad17c662577d3272665e06443b38。

原始科学表只归档一份，位于 `attempt2/blocks/<condition>/*.json.gz`，其摘要由同目录 `ARTIFACT_INDEX.json` 记录。报告 manifest 链接和校验这些原始文件；派生表为 `.csv.gz`，保留原始 ID、3×3 pair counts 和 pair denominator。没有另复制展开的 `*_raw.json`。历史清单中的绝对路径记录原执行位置；异机按相同相对目录定位文件并校验内容 hash，不应改写历史清单以伪装原路径。

## 复现命令与执行门槛

以下为命令记录，不表示本次已重新执行。运行目录应是包含交付代码的 checkout；新实验必须使用**新 attempt 目录**，不能覆盖 attempt1/attempt2。

```bash
EMA_CAUSAL_PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
EMA_CAUSAL_PLAN=research_plans/26-11-10_ema_causal_attribution
EMA_CAUSAL_OUT=artifacts/ema_causal_20261110/attempt_reproduction
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$EMA_CAUSAL_PY" -m experiments.ema_causal_20261110.run audit --plan-dir "$EMA_CAUSAL_PLAN" --out "$EMA_CAUSAL_OUT"
"$EMA_CAUSAL_PY" -m experiments.ema_causal_20261110.run prepare --plan-dir "$EMA_CAUSAL_PLAN" --out "$EMA_CAUSAL_OUT" --ml-csv /home_new/xzf/datasets/ml-1m/ml-1m.csv --kr-csv /home_new/xzf/datasets/kuairand/processed/kuai_rand.csv --cache-root /home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache
"$EMA_CAUSAL_PY" -m experiments.ema_causal_20261110.run smoke --out "$EMA_CAUSAL_OUT" --dataset movielens --device cuda:1
"$EMA_CAUSAL_PY" -m experiments.ema_causal_20261110.run smoke --out "$EMA_CAUSAL_OUT" --dataset kuairand --device cuda:1
```

正式执行前须有绑定该实现的独立 `IMPLEMENTATION_REVIEW.md` 与两份 PASS smoke receipts；设备号按本机可用设备选择。原 controller 命令是：

```bash
"$EMA_CAUSAL_PY" -m experiments.ema_causal_20261110.run run --plan-dir "$EMA_CAUSAL_PLAN" --out "$EMA_CAUSAL_OUT" --devices 0 1 --max-workers 5
```

实际 attempt2 后续采用外部 supervisor 在 ML-C 安全边界接管并允许资源准入的训练/归因重叠。接管必须保留同一 live guardian；不得同时运行两个 controller。只读预检与执行命令模板如下，`EMA_CAUSAL_CONTROLLER_PID` 和 `EMA_CAUSAL_CONTROLLER_START` 应取自当次原 controller 身份及 dry 输出，不能复用历史 PID：

```bash
"$EMA_CAUSAL_PY" -m experiments.ema_causal_scheduler_20261110 dry --out "$EMA_CAUSAL_OUT" --devices 0 1 --controller-pid "$EMA_CAUSAL_CONTROLLER_PID"
# 独立 SCHEDULER_REVIEW.json 必须包含 PASS 和当前 supervisor_sha256。
"$EMA_CAUSAL_PY" -m experiments.ema_causal_scheduler_20261110 run --out "$EMA_CAUSAL_OUT" --devices 0 1 --controller-pid "$EMA_CAUSAL_CONTROLLER_PID" --controller-start "$EMA_CAUSAL_CONTROLLER_START"
```

原 `run_all.sh` 保留最初同步 controller 的执行顺序；它不是外部 supervisor 的并发策略重放器。已有 live supervisor 时不得另启原 `run/resume`。supervisor 重启/收养 worker 的实际身份、review 和修订记录保留在 attempt2 的 `SCHEDULER_*` receipts 中。

所有 block 完成后，原 CLI 的后续顺序是 `analyze --stage 1` 至 `4`，由主代理完成科学 findings，再 `freeze → evaluate → report → release`。`freeze` 依赖完整的科学冻结文件；不能通过自动填造 findings 跳过。`evaluate` 才读取 Test，`release` 仅在最终报告/Test 完成后释放 guardian 参数。

已审阅的纯数表导出命令（可在 block 完成后部分执行，也可最终重跑）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python \
  research_reports/2026-11-10_ema_causal_attribution/export_report_tables.py \
  --out artifacts/ema_causal_20261110/attempt2 \
  --dest research_reports/2026-11-10_ema_causal_attribution/results
```

导出器不加载模型、连接 guardian、拟合桥接模型或触发训练。缺失 block/Test 写入 manifest 和 `SKIPPED.txt`；原 bridge summaries 直接转出。统计口径、五 seed、t4 常数和阶段定义见导出 manifest 与 [EXPORT_REVIEW.md](EXPORT_REVIEW.md)。

## 无 checkpoint 的 RAM 生命周期

在线模型、Adam、EMA、参数历史 R、梯度、anchor packets、恢复 snapshot 和 selected endpoints 留在 worker/guardian RAM。`torch.save` 用于内存 IPC 序列化不等于写磁盘 checkpoint。epoch 事务只落盘标量/身份/成本记录；guardian 中保存可恢复状态。

worker 失败时由仍存活的 guardian 恢复最后完整 epoch，重放未提交部分，单 job 最多一次 worker restart。guardian 丢失不能从 CSV/JSON 恢复参数，只能遵循既定同 seed 确定性重放预算并校验已有记录。一个五-seed block 的归因完成后释放该 block 的重型 anchors，留下 Test 所需 selected endpoints；最终报告/Test 完成后才执行 release。attempt1 中止销毁的 RAM 状态没有伪装为 release 成功，也没有在 attempt2 续接参数。

## 发布排除清单与只读核查

检查范围：`experiments/ema_causal_20261110/`、外部 supervisor、`research_plans/26-11-10_ema_causal_attribution/`、本报告目录、`artifacts/ema_causal_20261110/`。**未发现大于 100 MiB 的单文件**。未发现 `.pt/.pth/.ckpt/.safetensors/.pkl/.pickle/.bin` 模型文件；所见 90 个 `.npz` 的成员均为面板行索引、row IDs、标签、验证分侧或 pair 索引（两种已知布局），不是参数 checkpoint。发布时保留标量表和数据面板身份，排除运行认证文件及缓存。

| 路径或模式 | 发布建议 | 理由 |
| --- | --- | --- |
| `artifacts/ema_causal_20261110/attempt1/guardian.json` | **排除** | 含 `authkey` 认证字段及运行时 socket 定位；不得公开其内容 |
| `artifacts/ema_causal_20261110/attempt2/guardian.json` | **排除** | 同上；当前运行文件保留原样，不能原地删改 |
| `artifacts/ema_causal_20261110/**/scheduler.lock`、`**/scheduler_attribution.lock` | 排除 | 进程锁文件不是科学证据，也不是可迁移恢复状态 |
| `**/__pycache__/`、`*.pyc`、`.pytest_cache/`、临时编译 `.so` | 排除 | 解释器/测试/编译缓存；已有多个相关目录实际存在，外部调度器也有 `.pyc` |
| 任何后续出现的 `*.pt/*.pth/*.ckpt/*.safetensors` 或参数/optimizer snapshot | 排除并报告 | 本实验没有磁盘参数持久化的交付物；本次未发现 |
| 原始 CSV、既有训练 data cache、`/tmp/ema_report_preview*` | 不纳入本次代码/报告发布 | 使用输入 manifest 指定身份；旧展开 preview 曾有超过100MiB的派生 CSV，不是正式 results |
| `attempt2/blocks/*/ARTIFACT_INDEX.json`、`*.json.gz`、完成 receipts | 保留 | 原始科学表与完整性链；只归档一次，不再展开复制 |
| `**/panels/**/*.npz` | 不发布输入面板数组；保留相关 manifest | 这些文件是行号/标签布局，prepare 可按冻结身份重建；本分支发布代码、观测标量与报告 |
| `SCHEDULER_TAKEOVER/REVIEW/REVISIONS/DECISIONS/FINALIZER/LAUNCHES` 记录 | 保留为执行审计，完成后取一致快照 | 含历史 PID/start 等身份，不能当作异机运行控制文件；本次 JSON 认证字段扫描未在这些记录中发现 `authkey` |
| attempt1 abort/cost/epoch/log、engineering smoke/fault evidence | 单列保留 | 失败与工程成本的来源，不拼入正式科学表、不重复计算 seed 或 smoke 调用 |
| 运行中的日志/ledger/尚未完成 block | 待一致快照后再纳入最终发布 | 本文未删除、改写或冻结这些文件；旧时间点的成本表不能冒充最终总量 |

`guardian.json` 的秘密值未写入本文或工具报告。发布时建议按上述允许的源代码、审阅文件、冻结数表、报告和环境锁显式选取路径，不对整个 `artifacts/` 或工作目录无差别暂存。派生 CSV 使用 gzip 压缩，发布前核对单文件大小和允许的文件类型。

完整图表和数表可从本分支所含原始标量重新生成；模型参数已完成 RAM 释放，收据见 attempt2/RELEASE_RECEIPT.json；重训应使用新 attempt。
