# CODER IMPLEMENTATION REPORT

状态：B_IMPLEMENTED，等待独立 REVIEW。依据 `DESIGN_SPECIFICATION.md` 与冻结的 26-11-5 manifest；源码父提交 `c8675ae2923c15b6965884b140b5b646690cd95e`，计划提交 `2645f1f724f52f5ed5dea4b479335c0dc12fabcd`。

## 实施范围

仅新增 `experiments/aug_history_20261105/` 与本实施报告。父算法、旧报告和原始 CSV 未改。未添加依赖。

- `protocol`：两数据显式解析固定设置；核验 raw/cache bytes、源数组/行号/有序人口、train-only encoder/scaler、域人口，固定发表模型 CSV blob 和 V/test 端点配对。
- `model_adapter/objectives/registry`：保留源概率前向与源 scalar 运算顺序；实际 dataset 的 Gaussian key；同图两个增强 cell 标量；完整 named-parameter 布局、原始 tower logits 只读入口。
- `adam_counterfactual/responses/subspace`：用 PyTorch 普通 Adam 的 single-tensor functional kernel 对独立克隆的参数/m/v/step 做四次只读预测；保留 None/zero 和逐参数 step；实际 FP32 参数位移；FP64 归一化/Gram/秩；实时全梯度反事实与唯一真实 Adam 更新逐位核对。
- `directed_ema/shadow_bank`：15 个独立 φ，从 θ0 初始化；K10/K20 的 1+K 时钟、固定独立坐标符号、旧 innovation 投影、空空间退化 F，各 batch 一次更新。
- `worker/guardian`：源 online 全预算，逐 epoch 完整评价 online+15影子；selected/E10/E50 只保留 RAM。RAM 先接收完整 epoch 状态再发布标量；重复提交幂等、重放核对、冲突拒绝；跨 CLI 的 RAM guardian。
- `selection/report`：完整五 seed 门、每 seed 最早严格最大 AUC 选点、数据集级家族超参选择、六格保护与无约束赢家、冻结 family/匹配控制/online 端点。validation 决策包含可核验的 KR 历史 E50 online 代价来源。test 冻结门、原始逐 seed 六格/总指标、配对均值/样本 SD/95% seed CI、成本数表。
- `run`：入口要求的全部 CLI；独立审查与工程/GPU smoke 门；单卡显存和功耗低于85%、RAM使用低于90%且能容纳原 batch 时追加并发。RAM 状态丢失从 θ0 重放并记录物理成本；`continue` 输出冻结 A/C 的已解析命令。

公开函数的输入/输出形状在对应 docstring、LossTerms 和参数布局中记录。A/C 控制器、路径分组、全输入导数与 probe 在对应路线触发时实现；all validation 将其明确标为 `DESIGN_COMPLETE_NOT_IMPLEMENTED`。B 使用全参数空间，不把未实现的 A 分组当已验收分组。

## 已执行的必要检查

环境：全部通过 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python` 执行。

```text
python -m pytest experiments/aug_history_20261105/tests -q
10 passed, 2 warnings in 20.10s（含源 F 逐位一致断言）
python -m experiments.aug_history_20261105.run --help
exit 0
python -m compileall -q experiments/aug_history_20261105
exit 0
```

检查覆盖两数据参数量、源概率/loss/正式 grad/真实 Adam 更新、完整15影子的在线 hash 不变、源 F 递推、实际2N增强项、空 d0、raw-logit sigmoid 等价与 hook 清理；Adam 初始/已有历史、不同 shape/逐参数 step、None/zero、两 weight decay；FP64 梯度分解与 Q/S/EMA 代数；K 时钟、独立 RNG、RAM 导出/恢复；最早严格最大值、五 seed/test 门和标量幂等/冲突。

PyTorch 发出旧 TF32 setter 的未来弃用提示；沿用父训练设置。测试中的 requires-grad tensor 转 float 提示不改变训练代码。

## 运行与审查接口

独立 reviewer 审查本报告、DESIGN 和实际源码后，写 `REVIEW_ACCEPTANCE.json`：`status` 为 PASS/PASS WITH FIXES、`blocking_issues` 为空、`source_hash` 匹配 preflight 的身份。正式运行拒绝另一实现的审查收据。root 应先提交最终实现再执行正式 preflight，以记录 implementation commit。

Runner 在审查后执行原 batch GPU smoke：

```text
python -m experiments.aug_history_20261105.run validate --manifest research_plans/26-11-5/experiment_manifest.json --suite B --gpu-smoke --output artifacts/aug_history_20261105/attempt1 --devices 0 1
```

随后按 07 入口的 train→select→decide→evaluate→continue→report 顺序执行。训练输出在 `artifacts/aug_history_20261105/attempt1`；数学状态仅 RAM。原始 A/C 方案保持完整，触发后交 coder 实施并经独立审查。

尚未执行正式训练、性能分析与原 batch GPU smoke；这些由 REVIEW 后的 RUN/ANALYZE 完成。当前代码复用已存在的核验缓存路径；缓存缺失时 preflight 给出具体工程错误。

## 独立审查后生命周期修订

Reviewer 首轮指出三个会阻塞准确恢复的 Critical 问题，已按要求修复；B 的算法、候选与训练数学路径未变。

1. `--resume` 传到调度器：明确续跑时，每个失败/死亡 worker 作业在该次调用最多重新开放一次，保留最后已提交 RAM 状态，写 `RETRIES.jsonl`；再次失败结束该次调用。
2. coordinator 重启时跳过 guardian 内仍存活的 worker；注册失败的进程只写自己的注册失败日志，不报告另一 worker 失败。guardian 只接受当前已注册 owner 的 failure 请求。
3. RAM epoch 先接收、标量后发布；标量写入或重放冲突被拒绝时恢复原 RAM snapshot、commit_index 与记录 hash，拒绝的状态不会成为续跑起点。

新增一个聚焦生命周期测试覆盖显式 resume、每次一次恢复、存活 owner 排除、注册失败隔离、foreign failure 拒绝、冲突事务回滚及合法幂等提交。复测命令与结果：

```text
python -m pytest experiments/aug_history_20261105/tests/test_b.py -q -k 'resume_ownership or ram_export_restore'
2 passed, 9 deselected, 1 warning in 8.83s
```

数学路径的既有10项检查保留。等待 reviewer 对上述三个修复重新裁定；正式 GPU/训练尚未运行。
