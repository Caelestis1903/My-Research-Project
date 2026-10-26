# CODER IMPLEMENTATION REPORT — four-seed campaign recovery

状态：IMPLEMENTATION_COMPLETE，待独立 reviewer 验收。只做实现与基本数值/边界检查，未启动正式训练，未分析任何真实实验效果。

依据：`DESIGN_FOUR_SEED_RECOVERY_ADDENDUM.md`。本 coder 独占 `campaign.py`、`cli.py`、`config_four_seed.json`、`verify_campaign.py`；Coder A 并行拥有数据 mmap / protocol / prefix / baselines / readers / formation，未覆盖其更改。

实现：

- 新配置开发 seeds 5201–5204，锁定 6201–6204；原计划 09 文件不修改；新 artifact 为 `artifacts/mdmt_rcf_20261026/v2_four_seed`。记录父配置、设计 hash、变更原因和已有 SCALE 八条 receipt 来源。
- `_compare` 和 `decide_stage` 都拒绝少于四个独立 seed；所有 seed 的 ΔLL 必须为负，原均值/AUC/cell 阈值保留。群体匹配交集自然扩至四 seed。
- KuaiRand K1_WIDE / SOURCE_INIT_RANDOM 是既有实验证据补全必跑条件，与四 seed S2 的新触发集合取并集。四 seed 中任一缺该 control 则 S3 BLOCKED，不可 freeze。
- S1/S2/S3/locked 在同驻留模型内按已完成对象跳过训练。形成完成但 reader/diagnostics 尚未完成时续后续步骤，不重新训练形成器。每个 method 完成写 `TRAINING_COMPLETE.json`，明确独立 seed 与 prefix/selection 身份。
- 丢失 worker 的 EOF 记录明确 RAM 状态失联，不冒称 checkpoint 恢复。`prefix --resume` 可移除已死的自有 worker、清旧 gate/summary，然后同 seed 重放。5201/5202 对旧 v1 的 native selected/epoch24/pilot hash 和 selected epoch 核对；所有 seed 对本 root 的已有完成 receipt 同样核对。数据 mmap 引入 source/config 新身份，未称整体旧身份一致。
- 旧 seed 的共同 Q 指标逐方法精确比较并记 replay receipt；存在差异时停止并要求复核，不能当新结果静默接纳。这里只实施相等性检查，不分析真实指标方向。
- 原 GPU 单卡显存/功耗双 <85% 准入继续生效；主机 RAM 另保留至少 8GiB 并扣除 active prefix 尚未实现的工作集预留。首个数据集 prefix 的真实 peak RSS 返回后，以实测峰值加 15% 与保守初值较大者作为后续准入；测得首个 prefix 前不再并发同数据集 prefix。资源不足且没有可等待 active prefix 时抛明确可恢复错误，避免无期限挂起或隐降 batch。
- 通过 Coder A 的明确 row-data 接口在 simple predictor 和每 method / RPC 边界映射/回收数据页，生成 `ROW_DATA_CACHE.json`。不遍历或保存模型、optimizer、RNG、formation training_state。
- freeze 重新核四 seed S2/S3，再逐 method×dataset×cohort 查实际训练 receipt / prefix 完成与相同配置身份，输出 `FOUR_SEED_ACCEPTANCE.json`；SCALE 只核 v1 八条完成 receipt 的 hash 和独立 seed 数，不再训练。缺 receipt 拒绝 outer freeze。
- export / raw-X / 推理 / bootstrap 流程保留。finalize 先重新验 export，读当前 v2 的 `ROW_DATA_CACHE.json` 形成精确删除白名单，待 worker 正常结束后才删除其中 prefix/row_data_cache 文件；最终 runs 配对预测、candidate/function NPZ、旧 v1 和其他实验不在删除范围。生成逐文件/字节 cleanup receipt。

基本检查：

1. 项目 Python `-m py_compile` 三个修改的 Python 文件：PASS。
2. `verify_four_seed_recovery`：四 seed 合法阈值、第四 seed 反向、3/4 缺 seed、旧两 seed 配置、KR 必跑 controls 并集及缺 control 阻断：PASS（仅合成指标）。
3. 空实际训练 inventory：BLOCKED_INCOMPLETE，SCALE 引用两个数据集各四个 hash 匹配完成 receipt；没有预填主实验 PASS。
4. 新配置中的设计 SHA 与实际文件匹配。

收据：`RECOVERY_CAMPAIGN_BASIC_CHECKS.json`。首个真实 prefix 峰值、同 seed 模型 hash 重放、完整方法四 seed coverage 及最终导出验收须在独立审查 PASS 后由 runner 执行，不在这里预填完成。

有效配置 hash：`cefb6d934c774bd78a438e4fba5c560e934f4c56ab09fbb3dfe389f4f4842fd6`。

修改文件 SHA256：

```json
{
  "experiments/mdmt_rcf_20261026/campaign.py": "5172e509b2bf2b02822364c8d330ca86df5e166925adfecf40530e2a29bbf1da",
  "experiments/mdmt_rcf_20261026/cli.py": "4d5ad9b532b2f3f82692a8951e35c3c980728a31861c96a1db14a46cc85a86d1",
  "experiments/mdmt_rcf_20261026/config_four_seed.json": "830848363cb730b899616087bba5b952932fddbbbc802f611b397ed6539884af",
  "experiments/mdmt_rcf_20261026/verify_campaign.py": "052efe47a6728906988e03673d5fe702dfadaab9cd14de6e6f63cf1775ae0437"
}
```
