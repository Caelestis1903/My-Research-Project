# Stage2D 复现及产物入口

研究身份为 `STAGE2D_KR_IID_20261016`，冻结训练代码 commit `66c4cb3`（完整 hash 见各 run 的 config）。在原工作目录 `/home/xzf/Code/phase0-3` 使用 `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`。这是一组新 IID 研究，不是原 S1/S5/S6 的 exact replay。

## 源码和数据

`experiments/stage2d_20261016.py` 复用 config 中 source_hashes 列出的原模型、数据和评价函数。KuaiRand 为 `/home/xzf/data/kuairand/processed/kuai_rand.csv`，SHA256 `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。原数据、checkpoint 和预测不上传 Git；run_manifest 记录其 hash，全部保留在本机 run 目录。

运行前读取 design_specification.md、review.md、engineering_verification.json、cuda_profile.json、analysis_specification.md。CPU 工程检查和 CUDA profile 的小规模输出不计作正式训练。

## 原始运行与恢复边界

正式目录为 `runs/seed{2022,2023,2024,2025}/{SN,SB,EN,EB}/attempt1/`。每个目录的 config、progress、run_manifest、selection_frozen、metrics、epoch_observations、optimizer_observations 和 batch_trace 分别记录配置、状态、完整产物 hash、选点冻结、指标、曝光、状态范数及批次指纹。

队列持久运行，启动身份见 queue_launch.json；实际命令、GPU、PID、退出码和时间见 queue_events.jsonl 与 queue_status.json。不要在队列活动期间重复启动，也不要把某次已归档的 RUNNING 快照当成最新状态。每卡最多 5 个本研究进程，独立检查显存 90% 上限。

每个 run 的 `last_resume.pt` 保存完整 model、tau、两个 Adam 状态/参数名、Torch/CUDA/NumPy/Python RNG、两个 sampler RNG、累计 unique 曝光、步数和 validation history。selected 及 common epoch 的同类 state 也保留。当前 worker **没有已验证的断点恢复 CLI**；若进程失败，应保留失败 attempt，诊断后建立明确的新 attempt/恢复实现，不能重跑队列覆盖旧目录或宣称已验证无缝恢复。

需要全新独立复现时，先准备未占用的新输出目录，并在可用 GPU 上调用 worker 的 `--mode train --arm ARM --seed SEED --run-dir NEW_DIR --code-commit COMMIT`；具体完整启动参数以 queue_events 为准。新 attempt 与原 attempt 分开保存，不依据 test 改配置或挑重跑。

## 完成后的重算

在全部 16 个 run 真正 COMPLETE 后依次运行：

```bash
/home/xzf/miniconda3/envs/phase0-3-env/bin/python experiments/stage2d_analyze_20261016.py
/home/xzf/miniconda3/envs/phase0-3-env/bin/python experiments/stage2d_readout_20261016.py
/home/xzf/miniconda3/envs/phase0-3-env/bin/python experiments/stage2d_bootstrap_20261016.py
```

第一个脚本核验源码、配对、全部 run_manifest 产物 hash，并用原 CSV 的 row ID/label/domain 绑定 selected val/test 预测重算指标；第二个产生选点时序、common epoch、曝光和跨轨迹 state 描述表；第三个只估计给定四组已训练模型的逐 cell 条件 row bootstrap，不生成具有训练/选择不确定性含义的 macro CI。科学停止和机制结论由主代理在报告中逐项判断。

`--preflight` 仅为运行中的配对/初始化/源码检查，不能替代正式结果验证。已有 test 的所有读数只作为 forensic，不能用来确认新方法。
