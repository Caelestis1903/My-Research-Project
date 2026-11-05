# 09 共同工程合同（待 Codex 实现）

## 1. 源码入口与最小改动

新增包合同：`experiments/aug_history_20261105/`，报告：`research_reports/2026-11-05_aug_history/`。本轮规划提交不创建该Python包。

源提交固定 `c8675ae2923c15b6965884b140b5b646690cd95e`：
- `experiments/gaussian_domain_coverage_20261104/objectives.py`：监督替换和标准EMA。
- 同目录 `worker.py`：标准Adam、训练次序、RAM导入导出。
- `experiments/mdmt_history_information_20261103/protocol.py`：数据/配置身份；注意旧A1在KR默认d2，本轮必须显式覆盖d0。
- `experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/`：`views.py`、`objectives.py`、`protocol.py`；其下 `vendor_ga31/model.py`、`reference.py`、`basic/layers.py` 为已核对骨干。
- `experiments/overnight_ab_20261103/evaluation.py`、`selection.py`、`state.py`：评价、最早严格最大值和状态管理，Codex复用前逐项做接口适配。

不得整仓库重构，不改旧报告/原始CSV，不恢复AutoML/DBO/learned fusion。推荐新执行worktree基于该完整源提交，再覆盖本次计划和main的三个最新上下文文件；避免仅复制一个worker而漏掉vendor相对导入。详细git命令见总入口。

## 2. 新公共模块和接口

| 模块 | 必须提供的接口 |
|---|---|
| `protocol.py` | `resolve(manifest, dataset)->ResolvedProtocol`；数据checksum、train-only拟合与人口检查 |
| `model_adapter.py` | `build_model(bundle,config)`；`forward_logits(x, params=None, delta=None, probe_input_full_graph=False)->[N,2]`；源概率前向保持不变 |
| `objectives.py` | `loss_terms(model,x:[F,N],y:[N,2],view)->LossTerms(total,clean,aug_by_cell,metadata)` |
| `registry.py` | `ParameterLayout.from_model()->names,shapes,offsets,coordinate_groups,hash` |
| `probes.py` | 独立train行抽样、只读logit响应、排序探针；无val/test接口 |
| `shadow_bank.py` | `init(theta0)`, `after_step(context)`, `state_dict`, `load_state_dict`；每估计器独立φ与选择器 |
| `worker.py` | `create_session`, `advance_epoch`, `export_session`, `restore_session` |
| `selection.py` | per-seed earliest strict max；五seed HPO aggregate与冻结 |
| `run.py` | 本文件定义的CLI子命令 |
| `report.py` | 生成统一数表、决策与完整报告 |

LossTerms中每个aug_by_cell都是有计算图的标量、含λ/(2N)。原训练用 `total.backward()`；B辅助求导使用autograd.grad且不写`.grad`。A探针只对q求导；C只对δ求导。所有方向与协方差用于正式训练前detach，不建跨步计算图。

## 3. 随机性与同轨迹配对

原loader使用seed+11000独立Generator，原Gaussian完整沿用源`view_key`/`derive_seed`。绝不把candidate_id、h、η或γ加入基础Gaussian键；这些变化不应重抽不同基础噪声。新探针使用单独SHA256键：`AH1105|dataset|seed|global_step|stream|domain|task`。S的符号向量、C的正交旋转分别使用固定流，不能消耗loader、模型初始化或Gaussian全局RNG。

B多个K共用每10步的响应计算，K20只读取其指定时钟的响应；不得用未来响应。A相同刷新时刻不同候选共用探针行，但模型历史与q独立。C各独立轨迹保持相同初始参数、loader序列和基础ξ键；轨迹变化不能伪称完全相同参数状态。

验证/探针前后保存并恢复训练/eval模式、requires_grad、必要RNG和buffers；temporary hooks必须try/finally移除。读影子只用独立tensor/functional调用，不向在线参数写入。正式Adam每batch恰好一次；每影子每batch恰好更新一次。

## 4. 资源和状态

继承源协议：模型、Adam、影子、Q/q/C、选中checkpoint及E10/E50参数状态只存RAM，由源RAM manager/IPC或等价内存guardian管理；**不得写磁盘参数checkpoint，包括优化器和shadow。**磁盘只存标量、CSV、JSON、hash、代码和日志。

序列化快照必须包含：schema、source/manifest/data/layout hash、dataset/seed/trajectory_id、online参数与buffer、Adam完整state和param_groups、所有影子及其控制器状态、各K的最近刷新步、模型/loader/独立流RNG、global_step/epoch/batch位置、选择器及selected/E10/E50快照、累计成本、commit_index。

epoch事务：完成训练与验证→生成RAM快照→提交标量索引→递增commit_index。重复(job,epoch,commit_index)不可重复计训练。冲突hash报错，不能覆盖旧结果。

RAM丢失则从θ0确定性重放该seed的整个必要历史，校验既有顺序/hash并追加attempt记录；不能从磁盘标量恢复模型，也不能在终点新建EMA。RAM保存完整时可精确续跑。五seed聚合/选择完成前不得释放候选所需selected/E10/E50状态；C控制待赢家冻结后派发。

## 5. 验收测试

公共：模型参数量与父配置一致；源F路径logits/loss/grad/update兼容；2N尾批归一化；噪声不触动domain/dense/label；train-only人口checksum；每批一步Adam；同seed新增非反馈影子不改变online hash；validation选点精确并列最早；不满五seed禁止HPO选择；test入口在冻结前拒绝；resume与连续运行一致；RNG隔离；RAM状态不落盘；日志不会把同轨迹估计器当seed。

数值容差是工程阈值，不是性能阈值：FP32比较默认atol=1e-7、rtol=1e-5（源完全同算子路径优先逐位相等）；FP64投影/矩阵identity用1e-10。容差无法满足时定位算子差异，不自动放宽。完整模型CPU检查可用缩小合成batch，不输出性能结论。

资源：使用原batch GPU smoke测peak；若OOM先减少并驻shadow/顺序验证，保持数学与batch不变。不能偷偷改batch、精度或删低分候选。实际峰值、GPU秒、验证秒和探针秒分别记账。工作机器不存在预设环境路径时列为执行环境阻塞，不假装训练成功。

## 6. CLI命令合同

以下模块尚未实现；Codex需先实现并通过--help/parse-only测试：

```bash
python -m experiments.aug_history_20261105.run preflight --manifest PLAN --output RUN
python -m experiments.aug_history_20261105.run validate --manifest PLAN --suite all --output RUN
python -m experiments.aug_history_20261105.run train --manifest PLAN --route B --datasets movielens kuairand --seeds 2022 2023 2024 2025 2026 --output RUN --resume
python -m experiments.aug_history_20261105.run select --manifest PLAN --route B --split validation --output RUN
python -m experiments.aug_history_20261105.run decide --manifest PLAN --route B --basis validation --output RUN
python -m experiments.aug_history_20261105.run evaluate --manifest PLAN --route B --frozen RUN/B/selection_frozen.json --output RUN
python -m experiments.aug_history_20261105.run continue --manifest PLAN --decision RUN/B/DECISION.json --output RUN --resume
python -m experiments.aug_history_20261105.run report --manifest PLAN --output RUN
```

A/C替换route参数；C的train在内部按四候选依赖顺序自动五seed调度，再执行两控制，不允许单seed逐项抢先选择。continue只执行决策指定路线/数据集，不由runner临时改算法。输出路径按route/dataset隔离；已DONE+hash一致则跳过，部分结果补齐。所有命令的exit code：0完成，2配置/数学不一致，3工程阻塞，4成本阻塞，5等待五seed或依赖；后四者均非科学阴性。
