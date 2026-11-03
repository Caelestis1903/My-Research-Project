# INDEPENDENT REVIEW REPORT

Verdict: **PASS WITH FIXES**。本次发现的阻塞问题已修正，`blocking_issues=[]`。允许进入RUN阶段的prepare/CPU/S0；正式轨迹仍须S0接线与成本验收通过。

审查者：独立子agent `/root/reviewer`。未参与设计或编码，未修改实验实现。审查对象为冻结03/04/05、experiment_queue.yaml、DESIGN_SPECIFICATION、CODER_REPORT、CODER_ORCHESTRATION_REPORT与新包实际代码及其vendor依赖。

最终实现source hash：`7f16a451f5f0e41d9082d18bf548ad29d1d31f6c350175c5fb9f06ddc8813685`。

## 核对结果

- **信息与目标：PASS。** 独立CPU key、域内自然anchor、同目标/另一任务优先donor及回退、非域embedding插值、双端梯度、真实人口权重与λ/τ零旁路均与冻结定义一致。A1复用原目标和EMA02V2随机流。
- **H与控制：PASS。** S按外层参数名选取，probe仅求S且不覆盖当前`.grad`；D来自当前未修复梯度及一次coupled L2；双证据法向量、FP64小Gram活跃集KKT、旧moment修复、C_MIX位置与H_NORM同状态范数语义一致。
- **T与控制：PASS。** 同一当前batch和索引在完整旧参数/buffer处functional求值，旧点仅S求导；旧点L2、全S裁剪、质量因子、二阶矩和更新前快照符合规范。T_NORM保留普通moment方向。每步互斥参数更新与EMA各一次。
- **状态与评价：PASS。** 源RAM guardian保留selected/E10/E50，epoch状态含optimizer、T旧点和RNG；完成后释放optimizer但保留端点。训练只读validation选点，严格最大且最早；全局配置和选点冻结后才进入test，历史test亦在此时导入。
- **决策与成本：PASS。** D0两数据独立、D1固定四配置、2024不重选配置、D2匹配比较与简单性规则、确认seed和身份去重均实现。成本阻塞以具体默认配置处理；单配置终态故障不吞掉其他已授权请求。未完成或待基础设施重放不能进入决策/冻结。
- **来源与输出：PASS。** 独立核对37个vendor文件的Git blob、SHA256、字节数，均匹配固定v3源索引。数据加载入口核对原始及处理后manifest。标量导出覆盖六格、域/任务、macro/traffic/pooled/worst、online/EMA、selected/E10/E50、逐seed差、样本SD、95%配对t区间与开发/非开发子集；统计不把cell/epoch当重复。

## 发现与修订

| 级别 | 问题及对本轮的影响 | 最终处理 |
| --- | --- | --- |
| Critical | 初版将H_MIX/TM成本阻塞扩展为家族不进入D1，会改变科学候选选择。 | 只屏蔽已测具体默认配置，保留固定网格中其他配置及简单近邻；定向决策测试通过。 |
| Critical | 初版单worker失败会中断整个队列，且后续决策要求失败配置也有完整结果。 | 数值故障和耗尽允许重放的基础设施故障显式记录，其他请求继续；缺失D2控制不得支持联合主张，简单候选仍按原门判断。基础设施重放不自动触发，原因与次数仍受限制。 |
| Relevant | H_NORM对非零但≤1e−12的历史D范数未执行c=1分支。 | 仅修正NORM分支，小向量assert通过，H投影定义不变。 |
| Relevant | 只有合并支持率，无法分别检查两个任务的donor支持。 | 现有标量诊断增加两任务各级支持率，未改变采样或目标。 |
| Relevant | 失败重放的成功行采用已commit累计秒数，再加失败物理秒会重复计账。 | 当前成功physical attempt和已commit逻辑累计分别列出；失败physical秒/已完成step计数单列，失败step未完成计数明确留缺。 |

## 验证证据

独立执行：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/mdmt_history_information_20261103/tests
```

最终结果：**11 passed in 14.32s，exit code 0**。其中核心测试包括冻结参考13项CPU代数、H/T公式、源旁路、完整旧点、双端embedding梯度、内存恢复、None/零时钟与控制语义；编排测试包括数据独立阶段门、固定网格/复现、具体成本阻塞、D2缺控制时的简单替代及冻结前拒绝test。另独立核对37个vendor原字节文件，全部一致。

本审查未执行正式GPU训练。真实自然数据的完整40960 batch旁路、GPU计时和峰值显存由已实现S0执行；S0失败时按实际接线问题定点修复后复核对应路径。
