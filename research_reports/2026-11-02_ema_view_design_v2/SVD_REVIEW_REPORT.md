# 独立增量审查：FP64 SVD 执行设备

**判定：PASS。** 本次只审查已批准的 SVD 放置改动，承接原独立审查。依据为 DESIGN_ADDENDUM_SVD_PLACEMENT、CODER_SVD_REPORT、SVD_MICROCHECK 及实际源码差异。

- 设计附录 SHA256：`bdd471732067d48ec42843680ca25c9f54f251ea1ee678639755414c91af0bfb`。
- 原已审源码：`e524699526ae6e8eaf38950a0255eb359c7ca7705a64dab4f60bf9aa1ea93307`。
- 新已审源码：`f2658c2163902e6e69d0ea562b0aded9ba0c635477c19edb140d6353bcb15e82`，40 文件中仅 `views_ema_field.py` 改变。

独立逐字比较确认：变化仅为注释、对已归一化 FP64 J 调用 `torch.linalg.svd(j.cpu(), full_matrices=False)`，以及将保留的 basis/rank 返回原设备。归一化、SVD 截断、投影、回退、cap、最终 dtype、随机键与计数逻辑保持原样。该设备选择对两数据集及 F1/FR/F_PR 固定使用。

同一真实 KuaiRand Jacobian 的尺寸为 `[1843,2,33]`。CUDA null-basis 耗时 2.693277 秒，CPU 含传输 0.005798 秒；这些是记录硬件负载下的工程测量。全部行的秩一致；FP64 投影最大绝对差 `2.220446049250313e-15`，联合残差最大值 `3.711471586074082e-15`，满足批准容差。同一初始真实 model/EMA 状态下，F1、FR、F_PR 的 FP32 δ 最大绝对差均为 0，fallback/rank/cap 及工作计数相同，状态和 RNG 哈希未改变。

独立执行原有 focused F 检查：

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/ema_view_design_20261102_v2/tests/test_core.py::test_joint_projection_cap_degeneracy_and_fr_energy
1 passed in 1.30s
```

微测 53.324207 秒已作为 `SVD_PLACEMENT_CALIBRATION` 加入 attempt1 的 Stage0 物理成本，0 optimizer step；初始 GPU-SVD Stage0 与 forecast 文件另存保留。

下一步按附录重复 ML/KR 的 F1 两个真实完整 batch Stage0，保留已有其他方法测量的来源；通过后再冻结新预测与 S1 派发预算。本次审查没有运行 GPU 或科学训练。
