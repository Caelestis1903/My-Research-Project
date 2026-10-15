# REVIEW REPORT

## 1. Verdict: PASS

这是 Stage E5 第 1 次独立 REVIEW（最多三次）。实现忠实连接了预注册的五条 alpha trajectory、E3 coefficient 公式、真实 main backward/shared observer、固定 18-epoch 训练预算以及 20-run 配对矩阵；未发现阻止 RUN 的 CRITICAL 或 MAJOR 问题。

本次只审查 `stage_E5_design.md`、`stage_E5_implementation.md`、四个 E5 实现模块、两个 E5 focused test 与被 E5 直接复用的 E3 算法/observer/oracle 路径。没有读取任何 prompt 文件，没有读取或审查 E6/E7，没有修改实现代码，没有启动正式 20-run 矩阵，也没有分析实验结果。

## 2. Requirement-by-Requirement Verification

| Requirement | Status | Evidence |
|---|---|---|
| 五个预注册 arms 精确定义 | PASS | Registry 只包含 `static_sample=1.0`、`static_interior=0.75`、`static_matched=0.875`、`forward`、`reverse`；非法 arm/epoch 显式失败。 |
| 18-epoch alpha 序列 | PASS | epoch 0/8/9/17：Forward=`0.75/0.8676470588/0.8823529412/1.0`，Reverse=`1.0/0.8823529412/0.8676470588/0.75`；两者离散均值均为 `0.875`，逐 epoch 和为 `1.75`。 |
| Alpha-to-coefficient 公式与 cell 顺序 | PASS | 使用实际 audited shares 计算 `D*p_d^alpha/sum(p^alpha)`；`np.tile(per_domain, task_num)` 产生 task-major `[t0d0,t0d1,t0d2,t1d0,t1d1,t1d2]`；每 epoch 六 cell 总和为 6。 |
| Exposure matching | PASS | Forward/Reverse 使用相同 alpha multiset、相同 18 epochs 和固定每 epoch step 数；独立数值检查的 coefficient integral 最大差 `7.11e-15`，step-weighted integral 最大差 `5.68e-14`。Static `0.875` 仅匹配 alpha exposure，代码也单独记录非线性 coefficient integral。 |
| Main loss 真实参与 backward | PASS | 每 epoch 开始生成 scheduled coefficients，写入 `actual_coefficients`；E3 的六 masked-mean BCE 通过显式 dot product形成 loss，并进入一次真实 `backward()` 与 main Adam step。不是断开的日志分支。 |
| Architecture objective / sampler 保持不变 | PASS | Architecture 仍为 E3 unmasked BCE、同 cadence/optimizer；train loader 仍为自然 `shuffle=True,num_workers=0,drop_last=False`。Schedule coefficients 不进入 architecture loss 或 sampler。 |
| 唯一实验操纵量 | PASS | 五臂仅改变 main objective 的 alpha trajectory；model、role mapping、data split、batch size、optimizer、instrument cadence、checkpoint rule 和 fixed budget 固定。未增加 adaptive/EMA/momentum/router/warmup/grid。 |
| 固定 18 epochs，无 early-stop 差异 | PASS | E5 覆盖 `fit()` 并执行完整 configured epoch loop，不读取 patience 决定停止；formal protocol 强制 18 epochs，运行错误/非 finite artifact 不能通过 formal validation。 |
| 同 seed 五臂身份与 batch 配对 | PASS | Comparator 核对 initialization hash、dataset/device、split rows/counts/positive rates、partition/model、oracle、完整 ordered train/architecture batch support、fingerprint、batch size 和 counts；真实 mismatch focused test 可使 block `INVALID`。 |
| 20-run formal matrix | PASS | Formal queue 精确为 5 arms × seeds `{2022,2023,2024,2025}` × canonical `d0_large`，共 20 runs，无额外 setting/grid。 |
| E3 frozen oracle identity | PASS | Resolver 要求 E3 formal-valid artifact 与本次 dataset/device/seed/setting/physical-domain/remapped split-row fingerprints 全部一致；dry-run 对 canonical `d0_large` 验证 `12/12 reused`，缺失时队列不启动。 |
| Shared observer 正确性 | PASS | 复用 E3 strict shared partition；per-cell raw gradient 通过 `autograd.grad` 读取而不写 `.grad`，scheduled coefficient 用于 weighted contribution/reconstruction，随后同一 loss 做真实 backward/step；无额外 optimizer update。 |
| Checkpoint / test / common support | PASS | Validation 六-cell macro AUC 选择 best，严格 `>` 保证 tie 取最早 epoch；test 不参与 schedule 或选择且 CPU/CUDA RNG 被恢复；fixed-final epoch 17 与 selected view 均保留。Formal block 要求五臂完整相同 support。 |
| 指标与 summary 可分析性 | PASS | Per-epoch raw files保留 train/validation/test AUC、exact sample-mean LogLoss、domain/task/role/size/share、oracle/transfer、schedule phase/exposure 和 shared contribution trajectory；summary 固定枚举 20 paths并保留 seed×arm×view×cell 行、overall/worst-domain、difficulty oracle 与 baseline arm identity，足以进行规定的 stage/domain/task/seed/size/difficulty/baseline 分析。 |
| Implementation report 真实性 | PASS | 报告如实说明仅做 syntax/focused tests/单个短 smoke、未运行正式矩阵、未作科研推断，并明确 selected checkpoint 与 fixed-final 的保存语义。 |

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- Alpha trajectory、边界、中点、互补关系和离散积分均符合规格；Forward 与 Reverse 的 coefficient exposure 也因轨迹互为逆序而精确匹配。
- Domain coefficients 从实际训练 counts/shares 计算，没有把 physical-domain id 当作 size coefficient；两 task 正确复制同一 domain coefficient，tensor 长度严格为 6。
- Toy-loss test 验证 scheduled weighted sum，并确认梯度回到参与的 main tensor；实际 runner 的 loss path使用同一个 `weighted_loss_from_terms`。
- `static_matched` 被正确标记为 alpha-exposure control，而非错误宣称 coefficient-exact matched。

### Gradient flow & Loss participation

- Scheduled coefficient tensor直接进入 main loss dot product，loss 进入实际 backward 和 optimizer step。
- Shared observer 的 `.detach()` 只用于记录 raw vectors/统计；真实 loss graph 保留到 backward，observer 不覆盖 `.grad`。
- Raw/weighted norm、variance/SNR、magnitude share、signed contribution、mismatch、actual-update alignment 和 reconstruction evidence 均由真实 batch/backward path产生。

### Controlled variables & Baseline preservation

- 全部五臂共同使用 E3 data/model/optimizer/architecture/observer 代码；E5 实现局限于新 E5 模块和 tests，没有为了本实验改写核心模型或 trainer。
- 同 seed 初始状态、split rows、完整 train/architecture batch order 和 step support由 block comparator 锁定；18-epoch预算无合法 censored/early-stop 例外。
- E3 oracle 是只读 frozen reference，identity 包括 seed、canonical setting、physical domain、两 tasks、dataset/device 与实际 remapped split rows。

### Fail-fast enforcement

- 未注册 arm/epoch、错误 coefficient、错误 setting/role counts、缺失或非正式 oracle、非完整 18 epochs、non-finite artifact、缺失 selected cells、跨臂 identity/batch mismatch 均使 formal run 或 block 无效。
- 不存在用 Markdown 均值、joint baseline、跨 seed aggregate 或错误 oracle 静默替代 frozen oracle 的路径。
- E5 单进程会把异常转换为非零 return code；orchestrator 遇到 child/formal/block failure停止队列并标记失败。

## 4. Over-Engineering & Code Bloat Audit

PASS。新增代码围绕 schedule、单 run、固定 matrix 与 raw report 四个职责局部封装，直接复用 E3 已验证路径；没有 factory、controller、通用配置框架或范围外兼容层。身份/完整性检查均直接保护实验变量、oracle 或可分析性，没有发现需要删除的纯防御性机制。

## 5. Bugs / Deviations Found

### CRITICAL

无。

### MAJOR

无。

### MINOR

- Focused pytest 在工作区无法写 `.pytest_cache`，产生 permission warning；7 个测试均通过，不影响算法或正式输出。

## 6. Corrections Made

Reviewer 未修改任何实现代码；仅新增本 REVIEW 报告。

独立验证：

- E5 focused tests：`7 passed`。
- 四个 E5 模块 `py_compile` 通过。
- Schedule 数值核验：五臂 epoch 0/8/9/17、均值、六-cell总和、Forward/Reverse coefficient 和 step-weighted exposure 全部符合规格。
- Orchestrator dry-run：E3 frozen oracle `12/12 reused`；固定矩阵 `5×4=20`，当前 `20 pending`；使用 `--max_runs 0`，未启动任何 child run。

## 7. Remaining Risks

- PASS 只证明实现按设计执行，不表示 Curriculum 假设成立。正式数据仍可能显示 Forward 不优于 Reverse/static controls；该结论必须由 main analysis按预注册 GO/STOP gate 判断。
- 只有四个 seeds 与一个 canonical role mapping，size 与 physical-domain/difficulty 主效应不可完全识别；分析必须使用 frozen oracle 描述 operational difficulty，不能把 Small 直接解释为 difficult。
- Formal matrix 尚未运行。Runner 必须以 20/20 artifacts、4/4 valid seed blocks、12/12 oracle coverage 为完整性门槛；不得把部分矩阵当作完整结果。

## 8. Whether Experiments Are Safe to Run

**YES。** Stage E5 已通过第 1 次 REVIEW，可按 AGENTS.md 进入 `luna-runner` 阶段。Runner 只应执行固定 20-run formal matrix并记录 raw results/完整性，不得在 RUN 阶段解释科研结论。
