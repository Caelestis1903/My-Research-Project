# CODER IMPLEMENTATION REPORT — A0

日期：2026-11-09。实现依据：`DESIGN_SPECIFICATION.md`。本阶段实现完成，等待独立审查；未执行来源 audit 或科学 analyze 命令。

## 修改范围

全部修改位于新包 `experiments/ema_adaptive_20261109/`：

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 包入口 |
| `audit.py` | Git/内容身份、来源修订链、逐标量归档核对、面板隔离、S坐标计数与资源估算 |
| `analysis.py` | S-only配对、五epoch间隔持续性、训练面板LOSO前瞻拟合、任务/全局动作、静态参照、五seed阶段门和完整数表 |
| `run.py` | audit / analyze A0 / report数值附录 / release，成本与退出状态 |
| `tests/test_a0.py` | 8个手算正确性检查 |

历史训练代码、历史报告、规划字节均未修改。未增加依赖、训练器或尚未触发的A/B/C控制器。

## 规范对应

- 来源：使用固定 main/source commit 与 Git blob 对照，而非只登记本次自算hash。采集实现与后续分析修订逐文件区分；原始JSON归档流式读取，不解包参数文件。若两张pulse源CSV缺失，先恢复该来源commit中的原始CSV字节到本轮目录，再逐行与原始归档核对。
- 面板：复用源 `load_panels` 的SHA/数组hash校验，新增P_A/P_B、训练/验证、V_A/V_B行交集和有效掩码检查。读取范围为观测面板身份与标量日志，不加载模型或Test。
- S-only读出：F、fast_S、slow_S唯一配对；保留绝对AUC/LL、state hash、样本人数和轨迹身份。LL效用为LL_F−LL_action，正值代表改善。metrics的actual_count只作参数坐标计数。
- 直接持续性：同侧lag-1和跨面板e→e+5；分别输出早/晚/全窗，逐seed算符号率、相关与有效数。缺失窗口不跨接。符号描述排除任一侧为0的pair，并单列含零统计。
- 新预测：复用源 `feature_matrices`、`ridge_predict`、`seed_statistics` 和数表工具。每dataset五个LOSO折，各144行早期拟合、60行晚期评分；主特征与拟合标签均为P_A。P_B与V只评分。反向P_B只拟合design/persist/carry三种可用信号，不伪造梯度。
- 任务/全局：先平均真实及预测的fast/slow收益，再算MAE(D)、方向和argmax；不会平均cell最优动作。预测平局优先F。
- **预测方向约定（主agent明确确认、设计已同步）：** 所有预测器使用相同的未来目标D≠0分母，预测D=0计错误，包括raw carry；独立的描述性lag符号率继续使用nonzero-both规则。
- 静态任务/统一参照：共享其他四seed早期训练面板信息，留出seed晚期评分；保留正反面板方向，禁止事后择优替代主P_A。
- 阶段门：KR Gaussian、S的GLOBAL/TASK0/TASK1固定范围；均值>0及所有leave-one-seed-out均值≥0；CI只用于报告。动态、静态、无信号和来源缺失独立记录，保留逐seed门输入。
- 审查修订R1：静态GLOBAL保留正反方向门；TASK资格只使用前向P_A→P_B/V均值的稳定增量，反向TASK数值保留为诊断。修正了初稿将反向TASK诊断额外作为否决门的问题，并增加手算回归检查。
- 历史Validation拟合OOF和嵌套scope动作表单独命名，不混作新P_A-only预测。
- CLI report只产生数值附录与收据；最终科学叙述由主agent撰写。release在最终报告存在后记录最终代码/数据hash、零模型参数文件和零新训练/Test访问。

## 基础检查

均使用项目环境 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

```text
python -m py_compile experiments/ema_adaptive_20261109/*.py
python -m experiments.ema_adaptive_20261109.run --help
python -m pytest -q experiments/ema_adaptive_20261109/tests
8 passed
```

实际最终pytest输出见 `IMPLEMENTATION_BASIC_CHECKS.log`；执行测试仅使用手工小表，未用研究数据选择门或阈值。

检查覆盖：唯一配对及冲突、AUC/LL符号、三/六格聚合、缺窗不跨接、零值与常数相关、seed/time隔离、训练折标准化、V改动不影响预测/动作、预测平局方向错误、四类科学门和五seedCI。

## 执行入口

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_adaptive_20261109.run audit --manifest research_plans/26-11-09/experiment_manifest.json --out artifacts/ema_adaptive_20261109/attempt1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_adaptive_20261109.run analyze --stage A0 --manifest research_plans/26-11-09/experiment_manifest.json --out artifacts/ema_adaptive_20261109/attempt1
```

首条科学执行应为audit；以上命令尚未执行。A0无需GPU；参数状态、Adam步和Test访问新增量均为0。后续A1如被门触发，仍需先完成设计中约定的seed确认修订与独立审查。
