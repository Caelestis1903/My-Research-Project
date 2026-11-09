# Independent implementation review — A0

日期：2026-11-09。Reviewer 与 coder 为不同代理。审查对象为冻结的 DESIGN_SPECIFICATION、实际新包与 CODER_IMPLEMENTATION_REPORT。

**Verdict: PASS WITH FIXES。全部 blocking 修正已闭合，可执行来源 audit，随后运行 A0 analyze。**

## 修正记录

| ID | 级别 | 实际问题及影响 | 修正与证据 |
| --- | --- | --- | --- |
| R1 | Critical（已修正） | 原静态门同时要求 TASK 范围的反向面板结果为正，将诊断额外变成否决条件，可能错误阻断 STATIC_PREFERENCE_ONLY。 | 设计已明确 GLOBAL 与 TASK 范围；`analysis.py` 的 TASK pass 仅采用四项正向检查，GLOBAL 保留正向稳定及反向正均值要求。反向 TASK 数值完整保留并标为 diagnostic_only。新增手算例在 GLOBAL 反向通过、TASK 正向通过但 TASK 反向为负时得到 STATIC_PREFERENCE_ONLY。 |

本次修订及设计澄清发生在科学 audit/analyze 执行前。未读取本轮科学结果调整阶段门。

## 独立核对

| 范围 | 核对结果 |
| --- | --- |
| 来源与归档 | 对实际源码核对 Git blob/SHA 审计、采集与分析修订链、原始 JSON 归档逐标量核对及 CSV 恢复路径。32 个源依赖文件与 manifest 必需表均存在；实际来源 hash 验证由下一步 audit 执行。 |
| 面板与状态 | 复用源面板 SHA/数组身份核验；P_A/P_B 与训练/验证交集、V 有效掩码和行侧检查存在。新路径只处理标量与面板身份，没有模型参数或 checkpoint 读写。 |
| S-only 配对 | 按物理读出键核对重复；唯一 F、fast_S、slow_S 配对，保留状态与轨迹身份。AUC 差及 LL_F−LL_action 符号正确，未使用 all_fast/all_slow 作为 S 动作真值。 |
| 时间与 seed | 显式 e→e+5 merge 不跨缺失窗口；LOSO 的其他四 seed origins 5/10/15 为拟合、留出 seed origins 25/30/35/40/45 为评分，144/60 行检查正确。 |
| 拟合信息 | P_A 主拟合特征和未来标签均来自 P_A。P_B 与 V 只评分；P_B 反向拟合单列，模型集合只含其可构造信号。源 feature_matrices 仅用拟合折 ddof=0 均值/SD，ridge alpha=1、截距不罚。 |
| 任务与全局 | 三格或六格预测和真值先等权聚合，再计算 MAE、方向及 argmax；平局优先 F。静态任务策略和统一策略共用同一训练信息。V 值改变不改变预测和动作。 |
| 零值与统计 | 描述性符号率采用 nonzero-both 并列零率/有效数；预测方向采用共同 target-D≠0 分母，预测为零计错。常数相关为 NA。先在 seed 内聚合，五 seed 样本 SD/配对区间正确。 |
| 阶段门 | KR Gaussian、S 的 GLOBAL/TASK0/TASK1 范围与两类迁移侧固定；均值与 leave-one-seed-out 规则正确，允许单个负 seed，CI 不作额外否决。R1 后静态门与设计一致；来源不完整不会落入科学阴性。 |
| CLI 与产物 | audit/analyze A0/report/release 路径、设计/实现身份绑定、数值附录、成本与释放收据已核对；历史 Validation-fit OOF 独立命名。未实现未触发的训练控制器。 |

## 验证

使用项目 Python 环境独立执行手算检查，最终输出为 `8 passed in 2.50s`。另以人工小表确认 predictive_frame 将 origin 5 的当前收益连接到 target 10 的标签，并核对 source ridge alpha=1。检查日志见 `IMPLEMENTATION_REVIEW_CHECKS.log`。

最终设计 SHA-256：`a28ff4babf98dbbfb0f621c4cb2e84c9bfc7e673ab1a5bca7cd849f11c6d7288`。

| 文件 | 最终 SHA-256 |
| --- | --- |
| analysis.py | e20691334955534f7bcc4b3d7cc91a6f708713460d40e643530b8b4c64cf8fb8 |
| audit.py | c62f23b3dacd098d526e48d7c6ec5e04ac1eaacf9585a5d439d7d82108ad3bd0 |
| run.py | c3d68dc2acf4778701096fb92643758b45f4409e3d394fb3a9e6a9969513ffd0 |
| tests/test_a0.py | 5bf19754d02194da9e8f0cb002c817b631d8da082777c056a0d82e2dfcfa73c3 |

没有未闭合的 Critical 或 Relevant 实现项。本审查未执行科学 analyze；下一阶段由 runner 保存原始结果，科学解释由 orchestrator 完成。
