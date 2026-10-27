# 独立复审：FP32 跨 batch 部署一致性修复

日期：2026-10-27。Reviewer：`/root/reviewer`；Coder：`/root/coder`。

**Verdict：PASS。Blocking findings：0。**

批准按修订源码身份 `72f3e825dac422784fd575bd24450e7858e99ce138f6c8261b3ae7636b8b13fc`，在独立目录 `artifacts/mdmt_relation_basis_v2/revision1` 恢复原四 seed、两数据集、六臂的 48 条适配轨迹。原失败/中止记录及诊断重放成本保留，不能并入修订矩阵作为完成轨迹。

## 依据与影响范围

已读取 `DESIGN_RUNTIME_NUMERICS_ADDENDUM.md`、`CODER_FIX_REPORT.md`、`CPU_FIX_ACCEPTANCE_STDOUT.txt`、更新的 `SPEC_RESOLUTION.json` / `IMPLEMENTATION_SOURCE_MANIFEST.json`，并独立检查 `diagnostics.py` 与新增针对性测试。

将原 seed2022 worker 的 `source_identity.json` 与修订 manifest 比较：已有文件中只有 `diagnostics.py` 的身份改变；唯一新增 Python 文件是 `tests/test_runtime_numerics.py`。模型、数据、native/source/适配训练、缓存生成、指标、配置、选点和 RAM 恢复文件身份保持一致。有效训练配置 hash 仍为 `c08a1ffded211086145c300cffe1807ec014246ef443ce399fd59e11177d0b73`。

## 失败证据核对

已检查 `diagnostic_repair/seed2022_v2/REPRODUCTION.json` 和成本账：native、source、READ selected 三个模型 hash 均匹配原失败轨迹；相同 native4096 / source40960 上下文重算误差为0；READ batch256 跨形状最终 logit 最大绝对差为 `2.384185791015625e-6`，最大差值坐标 cached=`-4.291538238525391`、raw=`-4.291540622711182`。

完整 Q 的 40,008 行重算结果中，raw4096 与既有缓存 logits 最大差为0，`Q_cached` 与 `Q_raw` 的全部六格、macro、natural LL/AUC及pooled AUC记录相同。证据支持此次修复仅调整跨 batch FP32 部署验收；当前研究比较继续使用既定缓存与 FP64 logit 指标。

## 实现核对

| 必须保留的合同 | 实际实现 | 结果 |
|---|---|---|
| 唯一跨 batch 阈值 | `diagnostics.py:128` 的 helper 使用 `abs(raw-cached) <= 1e-6 + 1.3e-6*abs(cached)`，cached为固定参考端，差值与允许量在FP64计算 | PASS |
| 拒绝shape和非有限值 | 先比较shape，再检查两端全部finite；NaN/Inf不进入数值容差分支 | PASS |
| 保存可复核误差 | 记录batch、row-id hash/范围、shape、atol/rtol、参考端、finite/passed、max/mean绝对差、最大差坐标、两端值与该点允许量 | PASS |
| 失败信息持久化路径 | 未通过时将记录以严格JSON写入ValueError消息；既有worker捕获该消息进入FAILED.json后重新抛出 | PASS |
| 固定范围 | helper仅由latency的batch1/256最终logit比较调用；没有用于h/phi、教师转换、RAM恢复、训练或选点 | PASS |
| 原部署成本定义 | raw-X调用及cached路径未改；batch1/256、50 warmup、200次测量和MAC统计保留，无padding | PASS |
| 原恢复/教师标准 | 对应源码及测试中的绝对1e-6未变，未因部署舍入问题放宽 | PASS |

## 验收与运行准入

已检查针对性测试确实覆盖：实测失败数值通过；近零超过原atol拒绝；明显误差拒绝；NaN/Inf与shape差异拒绝；异常JSON可解析且保留固定容差。`CPU_FIX_ACCEPTANCE_STDOUT.txt` 为 **3 passed，1.43秒**，包括原RAM下一次更新恢复和教师转换测试。复用该有效验收证据，本复审未新增训练或扩大测试范围。

修订前的4个失败/主动中止attempt及两次相关诊断重放须单独记入成本和修订历史。修订矩阵保持2022–2025四seed、原训练预算与全局E冻结流程；本次PASS不改变预注册推进门槛或允许额外算法。

未修改实现、未运行GPU训练或额外研究实验。
