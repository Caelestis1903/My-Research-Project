# D2 共享表示与域任务专有路径顺序更新

**最终状态 `COMPLETE_NO_GO_S1`。** 共享更新引起了可测的局部排序变化。验证集 V_A 选中的 `P_AXIS` 在独立 V_B 上只保留了相对反向顺序的局部优势，未通过联合累积对照和跨锚点复现门。按冻结方案，本轮在 S1 结束。 [完整实验分析](S1_REPORT.md) · [预注册裁定](../../artifacts/mdmt_shared_private_coordination_20261123/SCOPE_DECISION.json)

| 阶段 | 状态 | 交付 |
| --- | --- | --- |
| S0 来源、实现、CUDA 与 RAM | PASS | 两套数据原始身份，真实首批与自然尾批，原生 Adam/EMA、梯度支持、恢复与独立审查 |
| S1 受控诊断 | COMPLETE · NO_GO | 两套数据 × 五种子 × 50 epoch，30 个锚点，570 次诊断 fork |
| S2 持续训练比较 | 未触发 | V_B 无合格数据集 |
| S3 最终评价 | 未触发 | Test 评价次数为 0 |

KuaiRand 的 V_B online Macro AUC 相对反向顺序为 +0.000053，配对 95% CI 为 [+0.000004, +0.000102]，4/5 个种子为正。相对 J_ACC 为 −0.000082，仅 1/5 个种子为正，两个对照同时为正的锚点只有 E25。MovieLens 相对 J_ACC 的均值为 −0.000001，E25、E40 均为五个种子全负。两套数据都通过了 −0.001 的六格均值保护线。 [逐 seed、逐样本格和训练阶段](S1_NUMERIC_READOUT.md)

S1 所有轨迹首次完成，双卡从首个准入到生成完成收据用时 30.1 分钟，基线与诊断分别调用原生 Adam 17,250 次和 990 次。首轮 S0 的 RAM 服务随命令会话退出，在持续打开的前台会话重做烟测并通过活体验收后才启动 S1。科研复核通过后释放全部 40 个训练与锚点状态，并确认本轮 RAM 服务关闭。 [执行与资源报告](RUNNER_REPORT.md) · [S0 独立验收](S0_SMOKE_REVIEW.md) · [RAM 关闭收据](../../artifacts/mdmt_shared_private_coordination_20261123/RAM_OWNER_CLOSURE.json)

## 复核入口

- [本轮独立代码包](../../experiments/mdmt_shared_private_coordination_20261123/) · [执行入口](CODER_RUNTIME.md) · [实现审查](IMPLEMENTATION_REVIEW.md) · [独立科研复核](S1_REVIEW.md)
- [数据、配置和训练身份](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN_MANIFEST.json) 与 [正式完成收据](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN.json)
- [基线结果](../../artifacts/mdmt_shared_private_coordination_20261123/BASELINE_ENDPOINTS.csv) · [诊断结果](../../artifacts/mdmt_shared_private_coordination_20261123/FORK_ENDPOINTS.csv) · [配对统计](../../artifacts/mdmt_shared_private_coordination_20261123/PAIRED_SEEDS.csv) · [成本](../../artifacts/mdmt_shared_private_coordination_20261123/COSTS.csv)
