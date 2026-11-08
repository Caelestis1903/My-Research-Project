# 05｜Codex工程实现规范

## 1. 实施边界与源码基底

新包建议：`experiments/ema_adaptive_20261109/`。正式实现前必须在仓库中核验以下文件是否存在于 26-11-07 分支：
- `experiments/ema_mechanism_20261106/`：11/07机制研究的协议、面板、固定EMA影子、读出、观测、RAM状态、分析和冻结评价模块。
- `experiments/aug_history_20261105/model_adapter.py`、`registry.py`、`group_registry.py`：模型前向、参数布局与12组路径映射。
- 11/06规划中的冻结vendor训练loss、数据桥接、源模型初始化与Gaussian RNG实现。

26-11-07 与 main 已分叉。Codex必须记录两分支SHA、merge-base、ahead/behind；以 26-11-07 的完整训练实现作为只读基底，建立隔离worktree或独立分支，不把实验分支合并到main，也不在main上直接训练。所有依赖以实际存在的blob SHA为准；不得仅凭历史文档推断文件已经合入main。

## 2. 新增模块建议

```
experiments/ema_adaptive_20261109/
  __init__.py
  protocol.py              # 来源、配置、seed、数据与实现身份
  panels.py                # P_A/P_B与V_A/V_B身份复用/审计
  state.py                 # EMA状态、checkpoint、原子恢复
  shadows.py               # A/B/C状态与每epoch fast/slow pulse
  task_router.py           # 真实task_id前向路由
  reward.py                # P_A收益、六格/任务Macro与跨面板指标
  controllers.py           # A/B/C与静态偏好控制器
  worker.py                # 训练轨迹与并行shadow
  analysis.py              # A0/A1分析、seed级统计
  evaluation.py            # Validation冻结与独立Test
  run.py                   # audit/analyze/train/freeze/evaluate/report/release
  tests/
```

已有源文件保持不变；优先直接复用11/07的data bridge、固定面板、源loss、ModelAdapter、ParameterLayout和group registry。若接口不兼容，只在新包写适配器，不复制或修改历史vendor。

## 3. 参数组与真实task路由\n\n11/07冻结 group_registry 的 S 坐标数为 MovieLens 483,152、KuaiRand 661,920。FP32下单份S状态理论字节分别为1,932,608与2,647,680；B额外增加一份同尺寸S历史状态。该值仅为参数payload，不含allocator、临时pulse、scratch模型和优化器开销，最终仍以运行时实测为准。

从实际 `build_path_groups` 与模型计算图读取 S 的坐标索引。必须验证：
1. S坐标集合与ParameterLayout完全一致；
2. 任务0/1的真实tower/输出路径都调用同一共享S模块；
3. task_id由真实batch/model route提供，不从参数名或domain id猜测；
4. domain d的三个域格在task t前向时均使用同一task历史S状态；
5. 非S参数在A/B/C之间保持完全相同的读出来源。

B路由实现：构造完整参数映射时，按task_id选择 `F_S^(t)`，其余参数使用固定F读出。输入x本身不含task_id，ModelAdapter.forward_logits一次返回[N,2]两列任务输出；任务t的domain tower索引为3*t+d，domain由x[-1]选择。参考实现必须对同一x做两次functional forward：S替换为F_S^(0)时取输出列0，S替换为F_S^(1)时取输出列1，再stack回原始[N,2]。禁止假设可按输入task mask拆batch。可实现只计算目标task tower的优化前向，但必须与双forward参考实现逐值一致。若源模型已存在task-specific tower route，应在其正确位置替换S参数，而不是复制整个模型或更改gate语义。

C路由：读取单一F_S与当前online θ_S，计算 `F_S + λ_t(θ_S-F_S)`；其余参数同样使用固定F。λ_t只依task，不依domain。

另一个低成本备选C-F在函数空间混合两路raw logits：

\[
z_t=(1-\lambda_t)z_{F,t}+\lambda_t z_{online,t}.
\]

它保留一份共享参数F和online参数，不创建task-specific参数副本；属于输出混合而非参数EMA。必须单独命名、测试和报告。

## 4. EMA状态管理与精确更新顺序

所有EMA从同一online初始参数θ0初始化。每次真实Adam完成后立即更新：
```python
optimizer.step()
with no_grad:
    for state in shadow_states:
        state.mul_(rho_for_current_epoch).add_(theta_online, alpha=1-rho_for_current_epoch)
```
- A：一份S状态；rho按全局动作。
- B：两份S状态；rho按task动作分别更新。
- C：一份S状态始终按rho_F更新；λ在epoch边界更新，不改EMA递推。
- Fixed F：全参数F或与源11/07一致的完整固定EMA状态，作为主内部对照。
- Fast/Slow pulse：从当前控制器S状态复制临时状态，完整覆盖一个epoch；rho分别rho_F²与sqrt(rho_F)。pulse完成后丢弃，不写回正式控制器状态。

**执行次序：**
1. 原始online forward/loss/backward/Adam；
2. Adam后更新F与A/B/C影子；
3. epoch结束后计算P_A训练面板上的候选pulse/λ收益；
4. 用截至epoch e的数据更新收益平滑器并决定epoch e+1动作；
5. 固化动作记录与state hash；
6. 执行完整Validation读出并记录，不得将V结果送入controller；
7. 提交epoch事务和恢复状态。

若控制信号仅在epoch末获得，严禁让它回溯控制刚结束的epoch。

## 5. 收益估计与控制器输入

主控制信号为训练P_A上的AUC。每个任务收益按该任务三个domain cell等权平均；全局收益为六格等权平均。LogLoss单独记录，不混入AUC动作打分。P_B用于交叉审计，不在事后择优；V_A/V_B仅作独立迁移审计。

每个epoch pulse必须保持：
- 同一训练轨迹、相同在线快照和非S参数；
- 同一面板样本、标签、pair索引和forward模式；
- Fast/Fixed/Slow只改变S历史；
- A/B/C及固定F承担相同观测频率和可比计算开销；
- 训练Loader不删除、重采样或重排任何样本。

训练面板AUC如果存在正负类别缺失、有效pair不足或面板身份不一致，当前窗口标记missing，不可填0；控制器保持上次动作或回退Fixed，并写明原因。

## 6. Checkpoint与恢复

每个epoch原子提交：
- online模型参数；
- Adam m/v/step及optimizer param-group；
- A/B/C所有EMA状态；
- 临时pulse起止状态（如epoch中断）；
- controller平滑状态、上一窗口收益、当前动作；
- Python/NumPy/Torch CPU/CUDA RNG；
- loader generator与当前epoch/step；
- P_A/P_B/V_A/V_B面板hash、数据hash、代码hash、manifest hash；
- 当前epoch的指标、预测hash和动作日志。

恢复时先核对所有身份；重复epoch commit若hash一致则幂等，不一致则拒绝并保留原状态。worker失败不能清除其他存活worker owner。不得把Test数据放入训练bundle或checkpoint。

## 7. 测试要求

至少包括：
- A/B/C初始化、单步更新与手算数值一致；
- rho切换发生在epoch边界；
- 同一online轨迹下Adam参数、m/v、step与无控制器运行逐位一致；
- A/B/C不改变训练随机数、loader顺序或Gaussian噪声；
- B task0/task1前向分别读取正确S状态，domain映射正确；
- B当两rho相同时，两任务状态逐位一致；
- C λ=0与固定F完全一致；λ=1与online S相同；
- C λ候选不越界，不能声称模拟Slow；
- S之外参数读出hash一致；
- P_A/P_B/V_A/V_B无重叠与角色泄漏；
- validation/test不能影响controller动作；
- 断点恢复、事务冲突、重复提交、worker owner和随机流可复现；
- 选点为Validation六格Macro AUC最早严格最大值。

## 8. 资源预算

沿用11/07机制研究的安全边界：最多2个并发worker；主机RAM占用达到90%停止新worker；GPU显存/功耗按设备资源门准入；观测预测产物默认上限2 GiB，RAM guardian默认上限4 GiB，除非实际源manifest更严格。每个新增S状态的字节数必须从真实参数layout计算并写入RESOURCE_ESTIMATE.json，不得猜测。记录GPU allocated/reserved峰值、worker/guardian RSS、磁盘、CPU/GPU时间和推理吞吐。超限应标记RESOURCE_BLOCK，不得悄悄缩小训练人口或观测面板。

## 9. CLI与退出码

```bash
python -m experiments.ema_adaptive_20261109.run audit   --manifest research_plans/26-11-09/experiment_manifest.json --out <run_dir>
python -m experiments.ema_adaptive_20261109.run analyze --stage A0 --manifest ... --out <run_dir>
python -m experiments.ema_adaptive_20261109.run train   --stage A1 --devices <ids> --manifest ... --out <run_dir> [--resume]
python -m experiments.ema_adaptive_20261109.run analyze --stage A1 --manifest ... --out <run_dir>
python -m experiments.ema_adaptive_20261109.run train   --stage B --devices <ids> --manifest ... --out <run_dir> [--resume]
python -m experiments.ema_adaptive_20261109.run freeze  --manifest ... --out <run_dir>
python -m experiments.ema_adaptive_20261109.run evaluate --manifest ... --out <run_dir>
python -m experiments.ema_adaptive_20261109.run report  --manifest ... --out <run_dir>
python -m experiments.ema_adaptive_20261109.run release --manifest ... --out <run_dir>
```

退出码：0成功/阶段未触发；2来源或配置hash冲突；3数值/实现错误；4资源阻塞；5五seed或依赖不完整。所有未触发、缺失和工程失败状态分开记录。
