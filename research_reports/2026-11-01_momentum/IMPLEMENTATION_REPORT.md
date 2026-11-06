# GAM01 实施报告

实现依据已批准的 [DESIGN_SPECIFICATION.md](DESIGN_SPECIFICATION.md)。仅新增 `experiments/gradient_aug_momentum_20261101/`，未修改 GA31-R3 包、历史报告和其他实验。来源核对复用协调者提供的 `SOURCE_FETCH_RECEIPT.json`；发布时须同时包含未改动的 R3 依赖目录。

## 实现

- `protocol.py` / `resolved_config.json`：四臂、源数据/模型身份、组件开关、逐参数优化器分支、shadow、RAM snapshot/restore、独立严格选点和 checkpoint 哈希。配置继承 R3 vendor 已冻结的自然数据身份，CSV 使用迁移后的绝对路径。
- `optimizer.py`：分源 `m_real/q_delta`、当前混合梯度的单个 `v`、共享逐参数 step、coupled decay、bias correction。`beta_delta=0` 与关闭 SAD 使用普通混合梯度 Adam；lambda0/增强关闭不激活增强路径。A2=.9 保留为等价控制，A3=.99 改变差分时间尺度。
- `objectives.py`：直接调用 R3 标量目标和 Gaussian 视图，A0 保持原两层均值求和次序，A1 保持原增强标量求和次序；SAD 分别计算全 batch 真实梯度和 target-aug 减 target-clean 梯度。target 分母为原始 `B×2`，同一个参数状态下完成所有梯度后只更新一次。SAD 每个有效增强 batch 实际调用两次 autograd.grad，普通 Adam 一次 backward，分别记成本。
- `run.py`：audit/check/all；四十条正式轨迹、每条五十轮、逐卡显存/功耗低于85%时继续并发、原 batch OOM 降并发重试、失败 ledger；每 epoch 完整 online/shadow validation，独立 selected 及共同 E10/E50，模型/Adam/q/shadow/RNG/loader 状态只存 RAM。D1只测每 epoch 最后一个实际 batch，局部标准 Adam 方向是相同映射状态的一步对照，不是 A1 整条轨迹。
- `analyze.py`：仅做配对数值、均值/样本SD/df4区间、保护指标和预设门，不撰写机制结论。全体 Stage1 完成后先用 validation/D1 冻结 P2 门；未触发则冻结全体选点后统一 test。触发则保留 coordinator RAM，等待本 attempt 的 `phase2_ready.json` 和独立审查后的 `phase2.execute`，本包没有实现 P2。
- `checks.py` / `tests/test_semantics.py`：直接影响公式与训练结论的最小核验。README 包含可执行入口及 P2 续接接口。

## 实际核验

Python 为 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

1. `python -m compileall -q experiments/gradient_aug_momentum_20261101`：通过。
2. `python -m pytest -q experiments/gradient_aug_momentum_20261101/tests/test_semantics.py`：**3 passed**。FP32/FP64连续三步 lambda0、beta0、beta=.9 等价及 beta=.99 手算；None/单侧连接/显式零与逐参数 step；target 差分与源标量梯度；关闭增强/λ0/κ0/缺目标不抽视图；shadow 公式和 buffer 复制、不干扰 online/RNG/loader；独立选点；纯 RAM 保存/恢复后下一步、q/shadow/optimizer/RNG/loader和视图哈希一致。
3. CPU 数值收据写入 `formula_checks.json`、`semantic_checks.json`、`checkpoint_restore_checks.json`；这些收据中的恢复检查使用小张量模型。真实模型 CUDA Stage0 由 runner 在独立 review 通过后执行，当前未运行。

真实 Stage0 入口已实现：两个数据集 seed2022，原 batch40960；A0 与冻结 R3 的预测/loss/梯度/实际 Adam 一步比较，A1/A2/A3 显存与一次真实更新；shadow on/off 三步轨迹与 RAM 恢复后的下一步。短开发窗口在重新生成下一自然 permutation 的边界检查 loader 恢复，每数据集共12次实际参数更新（ML上限20、KR上限49），不评价 test；不声称完成一个完整训练 epoch。失败检查写 `checks_failure.json`，已通过数据集的数值证据写进度收据。检查尚未执行，所以没有 CUDA PASS 或显存结果。

正式 `all` 要求 `--code-commit <reviewed implementation commit>`；本地无 Git 元数据，不把来源 SHA 当本轮实现 commit。运行 manifest 另记录本轮/全部 R3 依赖文件 SHA256、环境、数据身份和模型键；执行 commit 由协调者冻结并传入。

未启动正式训练，未查看本轮研究性能，未分析研究结论。下一阶段为独立 reviewer 检查设计、实际实现和本报告。

## 审查修订

独立审查指出最终自动裁决遗漏 validation 开发支持及 validation 保护项。按协调者批准规则，仅修改 `analyze.py` 的最终裁决：P0 必须同时通过 validation/test 原增益门；P1 的两项 validation 配对均值均须为正，且两项 test 均通过原增益门；两类方法都要求双方保护项为空。任一 split 触发保护项保留 TRADEOFF，并在 `guard_splits` 标明来源。原 val/test 数值与每项增益门原样保留，新增 `development_gate_passed` 便于复算。

新增一个合成裁决检查：validation 反向、test 过门仍为 NO_PROMOTION；validation 保护项为 TRADEOFF 并保留 split；P0完整门和P1正向开发条件各自验收。训练公式、矩阵、选点和P2触发未改。

修订后运行同一 `test_semantics.py`：**4 passed，1.44s**；CUDA仍未执行。
