# REVIEW REPORT

Verdict: PASS WITH FIXES
Blocking issues: 0
Reviewed source inventory SHA-256: 29b35c01c9b9b4d2a65dfb0bbf610ba5690701551ac71090db3fbfe4f639db65

## 1. Verdict

**PASS WITH FIXES.** Stage S3 Amendment A1 的 coefficient-hash 修复符合设计。审查中新发现的 finalization source binding 与 receipt-bound legacy collector 两个阻塞项已作局部修复并通过临时 queue 演练。当前 blocking issues 为 0。

本轮没有启动训练、正式 collector 或科研分析，没有修改正式 queue、canonical artifacts 或 archive。

## 2. Requirement-by-Requirement Verification

1. **PASS — 三层哈希语义。** `artifact_hashes[objective_coefficients.json]` 只绑定文件 bytes；table 内声明和 manifest `coefficient_table_hash` 必须等于从原始 JSON table 重算的 canonical content hash；跨 FULL/target-only pairing 只比较经过许可的 normalized table-content hash。三者不再互相替代。
2. **PASS — per-artifact fail-closed。** 文件缺失/字节哈希不符、声明 hash 缺失、manifest/raw hash 不一致、canonicalization 失败、字段/键/计数/系数/有效系数异常、非有限值、objective name/manifest 不一致、formula 不等于冻结公式、sampling 或 double-weighting 异常都会失败。
3. **PASS — legacy normalization scope。** 只允许 objective 为 Sample 且仅缺固定 metadata `domain_num`/`task_num` 时，在内存 copy 中补 `3`/`2`。Equal 不允许该兼容；其他缺失字段不推断。artifact bytes、manifest、raw hash 和 declared hash 均不写回。
4. **PASS — 24 objective artifacts。** 24/24 的文件 SHA、declared/raw canonical hash、schema 与 coefficient invariants 通过。唯一发生 schema normalization 的工件是 `target_only_shared|sample|kuairand|natural|m3oe|2022`。
5. **PASS — 8 strict blocks。** `2 settings × 4 seeds` 共 8/8 blocks 均为 `strict_source_identity=true`、`pairing_valid=true`。
6. **PASS — recovered seed-2022 pair。** 两份早期 target-only artifacts 继续只由既有 hash-bound recovery receipt、canonical tree digest、forward-model hash 和 artifact/prediction checks 接纳；没有把当前 validator 源码倒填成其训练来源。
7. **PASS — ordinary fresh runs。** 其余 22 artifacts 的 code inventories 必须匹配 queue 冻结的 producer source inventory。当前 validator/collector inventory 单独由本报告绑定，不能替换 producer provenance。
8. **PASS — no artifact mutation。** 修复前后正式 queue SHA 均为 `1f614dee34d3b5c29752bdc539d3a9bb44e6036b1a6dc4b6266eea335cf35efa`。以 `digest(sorted((formal_run_key, canonical_tree_digest)))` 明确定义的 24-tree combined digest 前后均为 `5b47fb65136afead0ab543f90a92d186116aa64e2112fa15aa8c7df688aaf645`，且 24 个 observed tree digests 均逐项等于 queue records。
9. **PASS — finalization only。** queue 已有 24/24 records、24 formal launches、26 attempts、remaining runs/groups 均为 0。恢复后只需 reconciliation、8-block final validation 和 collector，不得再启动训练。

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- Sample 仍要求 `w_d=1`、effective coefficient=`p_d`；Equal 仍要求 `w_d=1/(3p_d)`、effective coefficient=`1/3`，cell coefficient 再除以固定 `T=2`。
- normalization 只补不影响数值的固定 D/T metadata；任何 coefficient 数值差异仍产生 normalized hash mismatch 或 semantic failure。
- objective formula 现在必须精确等于冻结的 Sample/Equal 公式，不能用任意非空字符串通过。

### Gradient flow & Loss participation

本轮没有修改 forward、loss、gradient routing、optimizer 或 checkpoint。修复只发生在 artifact identity、pair validation、queue finalization gate 与 collector input validation。

### Controlled variables & Baseline preservation

- seeds、matrix、data/split/view、initialization、batch order、exposure、optimizer、checkpoint、metrics 和两进程规则未改变。
- S1/S2 sources/artifacts、24 canonical S3 artifacts 和 archives 均未写入。
- producer source inventory 与 post-training validation inventory 分离记录，防止把 validator 修复伪装成训练源码一致。

### Fail-fast enforcement

- rebind 要求显式 flag、无活动 S3 进程、精确 queue SHA/state/history、精确 prior source inventory、变更文件集合、reviewer-bound current digest、24 records/tree digests、两份 recovery receipts、22 producer-source validations、24 coefficient validations及 8 strict blocks全部通过。
- 任一异常都在 queue write 之前停止；不会 promotion、retrain、repair artifact 或 silent fallback。

## 4. Over-Engineering & Code Bloat Audit

修改局限于 S3 orchestrator/report 和对应 tests。一次性 rebind 是当前已完成训练 queue 的精确 transition，不是通用 migration system；没有新增算法、配置框架、registry 或监控层。

## 5. Bugs / Deviations Found

### CRITICAL — resolved

- **coefficient validator source drift 无合法 finalization path。** 原 queue 仍绑定训练时 inventory；直接更新它会丢失 artifact producer provenance，完全不更新则恢复入口拒绝。现分别保存 immutable artifact source inventory 与 reviewer-bound current validation inventory。
- **正式 collector 会错误拒绝两份 receipt-bound seed-2022 artifacts。** 它原先对全部 fresh rows统一要求当前 formal source schema；两份恢复工件按设计属于既有 recovery receipt 兼容路径。collector 现只对这两个精确 records 使用 receipt-bound validation，并仍检查 coefficient artifact 与 canonical tree。

### MAJOR — resolved by coder and independently verified

- strict pairing 原来混淆 objective JSON bytes、raw table hash 与 normalized semantic table hash，造成语义相同的 legacy/fresh Sample pair 假失败。

### MAJOR — resolved during review

- coefficient semantic validation原先允许从 manifest fallback 推断 objective kind，且只要求 formula 非空。现 table objective 必须可识别并与 manifest 一致，formula 必须等于冻结公式。

### MINOR — documentation limitation

- implementation report 给出 combined digest `a07cf3...`，但未记录 aggregation serialization。独立复审因此使用上述明确的 keyed-tree digest，并同时逐项核对全部 24 个 queue-record digests；不影响 artifacts 完整性结论。

## 6. Corrections Made

- 增加一次性 `--allow-reviewed-coefficient-rebind` 强绑定门。
- queue rebind receipt 分别记录 artifact producer inventory、current validation inventory、24-tree digest、review report hash 与 `NO_RETRAIN_NO_PROMOTION` disposition。
- `validate_fresh_run` 增加显式 expected producer inventory，不再把 post-training validator bytes 当作训练来源。
- final collector 对两个精确 recovery-receipt records 使用既有恢复验证，其余 22 个仍执行 producer-source formal validation。
- 强化 objective name/formula fail-closed checks。
- 新增并扩展 coefficient/rebind tests；过时的 seed-2023 incident fixture 仅在其原始 queue SHA 存在时运行。

## 7. Remaining Risks

- 正式 final validation 和 collector 尚未执行；磁盘或进程状态在 runner 接手前变化会触发 rebind/tree checks 并停止。
- collector 将写入结构化 CSV/manifest，但本 REVIEW 未验证其正式输出，因为 reviewer 不得运行正式 collector。
- 本报告不解释 AUC、LogLoss、domain/task effects 或机制结论。

## 8. Whether Experiments Are Safe to Run

**Yes.** 只需最终 validation 和 collector，不需要也不允许重跑训练。

先执行精确的 queue finalization 命令：

```powershell
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments/stage_S3_orchestrator.py `
  --execute `
  --queue-manifest experiments/stage_S3_queue_manifest_A1.json `
  --legacy-queue-manifest experiments/stage_S3_raw_results/.stage_S3_archives/s3_kuairand_natural_seed2022/parent_abort_79c649abb1994ee81d49_wave_1/queue_snapshot.json `
  --review-report experiments/reports/stage_S3_amendment_review.md `
  --raw-root experiments/stage_S3_raw_results `
  --s1-raw-root experiments/stage_S1_raw_results `
  --s2-raw-root experiments/stage_S2_raw_results `
  --python C:\Users\86199\.conda\envs\EDDAenv\python.exe `
  --device cuda:0 `
  --max-workers 2 `
  --allow-reviewed-coefficient-rebind
```

预期行为必须是：一次性 reviewed rebind → 24/24 read-only reconciliation → 8/8 strict-block final validation → queue `COMPLETE`。任何 child training launch 都是不符合准入条件的异常，必须停止。

queue 完成后才执行正式 collector：

```powershell
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments/stage_S3_report.py `
  --queue-manifest experiments/stage_S3_queue_manifest_A1.json `
  --raw-root experiments/stage_S3_raw_results `
  --s1-raw-root experiments/stage_S1_raw_results `
  --s2-raw-root experiments/stage_S2_raw_results `
  --output-root experiments
```

正式 runner 不得同时提供旧的 `--allow-reviewed-validator-rebind`，不得修改 queue/artifacts，不得调用 `stage_S3_run.py`。
