# GA31-v2 实施报告

实现依据：[DESIGN_SPECIFICATION.md](DESIGN_SPECIFICATION.md)。实施阶段已完成，尚未启动正式训练。独立审查通过后，由 runner 执行完整 40-run 协调入口。

## 实现与来源

新代码目录为 `experiments/gradient_aug_20261031/`。`reference.py`、`basic/`、`utils/data.py` 来自冻结 fixed1 源；`audited_data.py`、`source_protocol.py`、`source_data.py` 来自冻结的自然数据审计源。只调整包内导入，来源 commit/blob 见 `SOURCE_FETCH_RECEIPT.json`。历史源码、历史报告未修改。

- `model.py`：继承固定底座，在 Embedding 输出处按实际 sparse-then-dense slice 加相对高斯视图；无新参数、无 learned fusion/DARTS。
- `objectives.py`：同一原 batch 的 Sample 和归一化视图替换；目标域子 batch 额外 forward、一次 backward/单 Adam.step；独立设备 generator。
- `protocol.py`、`resolved_protocol.json`：五种子、40次、4/5门槛、df4；CPU 深拷贝快照与 BytesIO/Pipe IPC；没有模型状态文件。
- `data.py`：按冻结 raw/六数组/split/encoder/scaler 身份逐项验收，输出 split 行号与数据 manifest。
- `run.py`：audit/check/dev/train/evaluate 接口；Linux fork 继承只读数据，父协调进程不调用 CUDA 初始化；每卡显存和功耗均低于85%才新增 worker，等首个完整 batch 分配完成后再判断下一任务。保留所有快照于父进程 RAM，40个 selected 与工作尺度冻结后统一 test。
- `metrics.py`：六格、域、任务、macro、worst、pooled、流量加权指标；概率 BCE 使用 PyTorch 对数下限 -100，不添加概率 epsilon。
- `probe.py`：E1/E10/selected 去重，固定行与独立噪声；实际联合 deepcopy(model,optimizer) 三个单步分支，记录各参数块 raw/target 梯度几何、更新、m/v 差与真实 validation BCE。前后直接比较源状态和 RNG。
- `analyze.py`：从最终数值表生成五效应逐seed/均值/样本标准差/t区间、五seed主规则和四seed敏感性表。科研解释由主 agent 完成。
- `tests/test_semantics.py`、`checks.py`：本次比较所需的语义和原 batch 容量检查。

## 已执行检查

运行环境：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`，PyTorch 2.9.0+cu126。

1. `python -m compileall -q experiments/gradient_aug_20261031` 通过。
2. `python -m pytest -q experiments/gradient_aug_20261031/tests/test_semantics.py`：3 passed。覆盖固定参考预测/loss/gradient/Adam、κ0与关闭增强、λ0、非均衡/单例/缺目标batch、domain/dense保护、独立RNG、RAM epoch边界恢复后下一步和 None 语义。
3. `python -m experiments.gradient_aug_20261031.run --mode check --run-dir artifacts/gradient_aug_20261031/GA31-v2/implementation_check1 --device cuda:1` 通过：两个数据集全部冻结身份匹配；CUDA参考前向/梯度/Adam一致；各四臂各执行一个40960行训练batch，没有test指标。ML峰值0.98–0.995GB，KR峰值1.10–1.12GB；完整检查记录见该目录 `checks.json`。模型 state 字节分别3,093,880与6,457,016。

首次零噪声测试中，重复 forward 的梯度浮点累加使少量 Adam 坐标超出1e-6容差（最大约3.98e-6）。实现将 κ=0 的相同视图精确折叠为真实损失，复验全部通过；正式 κ=.1 路径不变。没有放宽容差。

上述工程检查没有评价 test，也没有选择超参数。单epoch开发训练、完整probe端到端、GPU多进程调度及正式40-run尚待独立review后由runner执行。

## 运行命令与状态边界

完整正式入口：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_20261031.run \
  --mode train --devices 0,1 \
  --run-dir artifacts/gradient_aug_20261031/GA31-v2/attempt_<timestamp>
```

`--mode dev` 可选单dataset/arm，固定seed2022、一个epoch；不会访问test。`--mode evaluate` 冷启动会报告缺少RAM状态；正式test由长期协调进程在冻结后调用。进程退出或机器重启后，受影响run须从原seed重训。快照和初始化权重从不写盘；只有代码、数据记录与日志落盘。

目前没有已知数据身份或原batch容量阻塞。独立review需要审查调度、全局freeze、纯内存IPC、probe与结果表的实际实现。

## 审查修订

独立review发现 `selection_frozen.json` 的selected记录误用了E50终点的global_step。现改为读取 `states[selected_epoch]['global_step']`，与selected_epoch保持同一快照身份；训练、选点和指标公式不变。修订后模块语法检查通过。
