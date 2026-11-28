# 04 — Codex 工程实现规格

## 1. 代码基准和禁止修改范围

已核实的 main 路径：
- M3oE architecture reference：`m3oe/MDMTRec/models/multi_domain/model.py`
- canonical experiment model factory：`experiments/conditional_information_momentum_20261102_v3/protocol.py::make_model`，其实际使用 `vendor/ema02/vendor/gradient_aug_20261031_v2/views.py::make_model`（支持 GA31 `view,kappa` forward）
- 原始 trainer/data reference：`m3oe/MDMTRec/trainers/ctr_trainer.py`、`m3oe/MDMTRec/utils/data.py`、`m3oe/examples/run_movielens.py`、`m3oe/examples/run_kuaishou.py`；本轮训练应复用 v3 canonical protocol，而不是另行实例化旧 trainer
- PAIR optimizer / objective：`experiments/conditional_information_momentum_20261102_v3/optimizers.py`、`objectives.py`、`boundary_sampler.py`
- A21 runner：`experiments/conditional_history_shared_update_20261122/history_runner.py`、`history_controls.py`、`protocol.py`、`runner.py`
- GA31-R3 Gaussian 实现的精确历史来源见 `research_plans/26-11-24_history_information/11_SOURCE_AND_REUSE_REGISTER.md` 与 `experiment_manifest.json`，执行前需核对对应 commit 和原始路径是否在 main 可访问。

不修改原始模型定义以加入新模块；新算法放在独立实验包，使用 adapter 包装已有 model/optimizer/objective。原始 baseline 必须保持精确行为。不得改写历史报告和历史 CSV。

## 2. 新建模块

建议新建：
`experiments/three_directions_20261128/`
- `protocol.py`：冻结 dataset/seed/config/source identity、RunSpec、hash 和读写。
- `macro_pair_sampler.py`：从 Train-global row-ID pools 建立 24 个 pair conditions，按六格 macro target weights 计算固定 w/q；每步采样一个 condition 和 2048 pairs，并通过 row-ID gather 取回 endpoint features。B 仍调用原 `boundary_sampler.sample`。
- `ranking_objectives.py`：clean pairwise、Gaussian pairwise、历史 margin memory loss。
- `gaussian_view.py`：base BCE view 调用原始 `draw_view` + `source_losses(explicit_views=...)`，保持 GA31-R3 key；pairwise Gaussian view 使用相同 draw implementation、相同 κ/字段范围，但独立 key `GA31-R3|{dataset}|{seed}|train|{global_step}|gaussian_pairwise|1`。不得走 row-keyed `normal()` 默认分支。
- `freshness_optimizer.py`：A21 wrapper 与同 batch 梯度校正状态。
- `memory_bank.py`：Train-only pair memory 构建、冻结、抽样和 hash。
- `runner.py`：50 epoch training、online/EMA Validation、selected RAM state。
- `analysis.py`：paired endpoints、six-cell guard、CI、阶段 decision。
- `scheduler.py`：依赖队列、资源状态、pause/resume。
- `report.py`：CSV/JSON/Markdown 报告。
- `tests/`：pair sampler、loss、noise identity、optimizer state、resume、selection、no leakage。

## 3. 与已有接口的连接

### 3.1 Model forward

canonical model 由 `experiments/conditional_information_momentum_20261102_v3/protocol.py::make_model` 创建，实际实现来自 `vendor/ema02/vendor/gradient_aug_20261031_v2/views.py::make_model`，输入为按字段转置的 feature tensor，返回各任务预测概率。PAIR 需要 logit margin；应从模型 tower 最后线性输出建立只读 logits adapter，不能对 sigmoid probability 直接当 logits 使用，也不能反复调用 model.forward 造成不同随机状态。在 `gradient_aug_20261031_v2/objectives.py::losses` 增加可选 logits-capture callback：在原 clean `model(x)` 与 D0 `model(x[:,mask],view=view,kappa)` forward 内复用 `boundary_sampler.forward_logits` 的 tower hooks，捕获 full-batch clean logits 和 D0-local view logits，不得为了取 logits再跑第二次 forward。A/C 的 Train-global pair endpoints 单独 gather 为 `pair_x`；clean pair forward 调用 `model(pair_x)`，GP0/A2 的 D0 pair view 调用 `model(pair_x,view=view,kappa)` 并通过相同 tower hooks 捕获 logits。保留原 `losses` 的 scalar 计算顺序和 sample budget，新增 pairwise 项只在返回 scalar 上叠加。以原 forward probabilities 逐值对齐测试。

模型 forward 内部包含 embedding、shared star、domain/task experts、gate 和 tower。不能通过仅替换最后 tower 而漏掉模型其余结构。使用 hook 或受控子类，必须证明 probability 与原 model.forward 的误差在 FP32 允许范围内。

### 3.2 Dataset row identity

从 canonical data bundle 读取稳定 row IDs、domain ID 和两个 task labels。不得把 DataLoader batch index 当成永久 row identity。A/C 使用 `macro_pair_sampler.py::sample_macro6` 从 Train-global pools 取 2048 pairs，六格 macro weights 固定；B 使用原 `boundary_sampler.py::sample` 从当前 batch 取 natural 24-condition、w/q 和 context RNG。A/C pair endpoints 通过 stable row IDs gather feature rows。A1 arms 共享完全相同的 pair IDs；B0/B1/B2 共享相同 A21 natural pair IDs；C1/C2/C3/C4 共享 macro6 selected conditions 和 pair exposure，但 C2/C3/C4 的 memory pair identities 按各自控制定义不同。全局训练行 ID 进入 row hash。若当前数据 bundle 不暴露 row IDs，先扩展 dataset adapter 并做与 canonical split hash 一致性验证。

### 3.3 PAIR optimizer

复用 `experiments/conditional_history_shared_update_20261122/history_controls.py::history_config('A21',dataset)` 与 `HistoryOptimizer` 的 collapsed mode，以及 `experiments/conditional_information_momentum_20261102_v3/optimizers.py::PairOptimizer`。不要重写 A21 公式。S_DENSE scope 必须完全由现有 `shared_name(name)`/`parameter_manifest` 决定（当前 helper 匹配 `shared_weight`、`shared_bias`、`skip_conn.*`、`star_mlp.*`、`expert.*`）；不得自行重新分类。新 freshness correction 只包装 PAIR source gradient 形成步骤，保留 native BCE gradient、weight decay、Adam state、pair event clock、natural w/q ratio 和参数顺序。实现前导出当前 HistoryOptimizer/PairOptimizer 的完整 state transition 并逐项对齐。

### 3.4 Gaussian

Gaussian view 复用 GA31-R3 原实现的精确字段列表、scale、覆盖、噪声键、生成顺序。main 中已记录 GA31-R3 与 EMA02 的 noise identity 曾不一致，因此不能从相似函数名推断兼容。A0 必须验证：
- 同 seed、同 global step、同原始 batch row order 复现 GA31-R3 archived noise hash；
- 使用 device-local torch.Generator 并按 field 顺序调用 torch.randn；
- 不使用 EMA02V2 row-keyed noise（main manifest 明确禁止替换 GA31-R3 identity）；
- pair row IDs 只用于 pair identity/hash，不重定义 noise RNG；
- field order 与源实现一致；
- domain embedding 不扰动；
- D0 coverage、loss replacement 和分母完全一致。
不满足时只暂停 Gaussian arms，clean pairwise arm可以独立运行。

## 4. B freshness 实现

- 快照只保存 S_DENSE 的上一状态，按参数 manifest 固定排序。
- 旧参数点梯度通过 functional parameter mapping 计算，不能写入 live Parameter 后忘记恢复。
- 旧梯度计算时保持当前 private params、buffers、dropout mask/RNG、pair IDs、importance weights 不变。
- native BCE gradients 在旧参数 probe 前克隆；恢复后校验无变化。
- optimizer、EMA、loader、sampler、model buffers、RNG 哈希前后守恒。
- pair gradients 与旧状态 gradients 都是 FP32；校正项不做额外 clipping 或 normalize。
- 先对 BCE objective backward；再向 S_DENSE 的 `p.grad` 显式加上 `lambda_pair * (w/q) * g_tilde`，再调用 collapsed optimizer `step(pair, r=g_tilde)`。现有 optimizer 会扣除同一当前 PAIR 项并将 g_tilde 写入历史；若不显式加回，optimizer 将错误地从 BCE gradient 中扣除 PAIR 梯度。
- 校正后的 PAIR history 按 A21 原始 β 和 λ 更新；不能让校正梯度进入 BCE 的 Adam 二阶状态之外的另一个未定义 optimizer。
- 任何 parameter scope mismatch、missing gradient、non-finite value 或恢复 hash 不匹配均暂停 job。

## 5. C memory implementation

- Memory bank 只包含 Train row IDs、cell、历史 margin、epoch、snapshot hash。
- 不写入用户数据内容或 Test rows；memory index 与数据 hash 绑定。
- 每个 active condition 最多 2048 unique memory pairs（最多 24 pools）；训练时由 macro6 sampler 选择 condition，保持 macro6 w/q 和每 batch 2048 pair exposure。
- 每 epoch 固定曝光预算；random memory、historical memory 和 current pairwise exposure 对齐。
- C2/C3/C4 bank capacity per condition is matched to C3 correct-pair count; C4 uses exact C3 pair identities; C2 samples the same count from the full candidate pool.\n- bank 建立后冻结，不能按 Validation/Test 表现筛 pair。
- 训练报告需提供 bank size、pair uniqueness、exposure count、current sampler overlap、margin histogram。

## 6. 选点和训练状态

- 每个 epoch 分别评价 online 与 EMA。
- selected = strict maximum Validation six-cell Macro AUC；精确 tie 取 earliest epoch。
- LogLoss、六格、Traffic/Pooled 绑定到同一 selected checkpoint。
- 选点模型和 optimizer 状态只按当前仓库的受监督 RAM state contract 管理；不得绕过 RAM guardian 把模型/optimizer/history snapshot 保存到磁盘。
- 允许磁盘保存的内容：代码、RunSpec、hash、数值指标、合法预测统计、报告、manifest 和状态收据。
- 断点恢复必须恢复模型、optimizer、EMA、PAIR history、上一共享参数快照、sampler state、RNG、loader permutation/cursor、selected metadata；恢复后做 state hash readback。
- 不得引入独立后台服务、headless agent、tmux/nohup/systemd 或自主 driver。

## 7. 测试清单

### Pair sampler
- macro6 sampler 的 pair 不跨域/任务格，六格目标权重各为 1/6；Train-global pair pools 在每个 domain-task cell 内覆盖所有可用正负联合标签状态组合；B 的 A21 natural sampler 保持原 24-condition proposal；
- positive/negative 标签身份正确；
- empty positive / empty negative；
- pair count 上限；
- deterministic replay；
- identical original batch replay preserves pair row-id hash；
- no validation/test access; Train-global pool row IDs 均属于 Train split；macro6 与 A21 natural 两种 target/proposal weights 分别通过独立单元测试。

### Ranking loss
- logits margin 正负方向；
- tau=1；
- macro6: per-cell target weight sums to 1/6, proposal q sums to 1, and w_macro/q_macro ≤ 2；
- A21 natural sampler remains byte-for-byte behaviorally identical；
- zero pair returns finite zero;
- gradient finite and correct sign.

### Gaussian
- archived GA31-R3 first-draw hash and field order under the original batch order;
- noise key reproducibility;
- source implementation equivalence;
- clean BCE unaffected;
- D0-only perturbation.

### Freshness correction
- γ=0 equals A21;
- identical current/previous shared params gives corrected gradient equals current gradient;
- same batch and pair indices used twice;
- private params unchanged;
- optimizer/EMA/RNG/loader/sampler unchanged by probe;
- parameter snapshot advances only after optimizer step;
- no pair / zero pair / None gradient;
- finite-difference gradient delta matches independently computed reference.

### Memory bank
- only Train rows;
- C3/C4 only contain historically correct pairs; C2 random control has no correctness filter; all banks are Train-only;
- margin clipping;
- fixed capacity;
- equal exposure;
- frozen bank identity;
- no test leakage.

### End-to-end
- seed2022 small smoke test on both datasets (not counted as science trajectory);
- first/last batch checks;
- selected tie chooses earliest;
- LL from selected AUC checkpoint;
- resume equals uninterrupted state;
- raw baseline arm exact equivalence.

