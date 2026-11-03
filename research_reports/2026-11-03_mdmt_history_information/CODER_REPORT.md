# CODER IMPLEMENTATION REPORT

核心实现负责人：独立子agent `/root/luna_coder`。实现依据为已批准DESIGN_SPECIFICATION及冻结03/04/05；编排模块由另一个实现者负责，见CODER_ORCHESTRATION_REPORT。两份实现报告共同交独立reviewer检查。

已实现新protocol、mixing、objectives、HistoryOptimizer、stage0与核心tests。包含A0f/A1/SCALE、M/U_MIX、H_REAL/H_MIX/H_U/H_NORM/C_MIX、T0/TM/T_U/T_NORM；H/T数学、参数范围、状态顺序和控制按冻结定义落实。没有新依赖、额外候选、搜索配置或训练seed。

复用固定v3提交 `a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288` 的37个原字节文件，逐文件source路径、Git blob、SHA256、字节数见SOURCE_MANIFEST。新协议显式加载本轮queue并给原data loader传p；未调用vendor内部旧计划加载入口。源A1保留原损失、噪声及EMA02V2随机key。

参数/optimizer/梯度只驻RAM。T仅保留一份完整前一步更新前参数和buffer，存于optimizer state；guardian完成轨迹后移除optimizer，而保留各估计器selected/E10/E50直到统一test。S0的恢复检查通过源BytesIO IPC roundtrip，不写模型文件。

核心检查命令：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/mdmt_history_information_20261103/tests/test_core.py
```

结果：6 passed（9.29秒）。覆盖冻结参考13项代数、实现H参考一致性、T质量因子、三级donor与独立随机流、M双端embedding梯度、源A0/A1及Hκ0/T0κ0/Mλ0/Mτ0旁路、TMκ0=M、完整旧点与非S参数替换、更新前快照、H/T/NORM/C的内存恢复、probe不改.grad、None/显式零时钟、coupled L2一次及NORM同状态范数。

一次测试fixture修正：原非S替换检查选中了LayerNorm前的隐藏bias，整体平移被LayerNorm消去，改为输出层bias以明确观察完整参数替换；未改变算法实现或实验协议。

真实数据完整batch S0脚本已完成，包含源旁路loss/gradient/Adam/EMA/恢复下一步比较、数据manifest检查和完整成本计数。尚未运行S0或正式轨迹；这些属于审查后的RUN阶段。本报告不解释研究性能。

整包CPU测试与编排验证结果由CPU_TESTS.json和CODER_ORCHESTRATION_REPORT记录。核心与编排共同通过独立审查后，runner可以按CLI顺序执行。

独立review后的定向修订：H_NORM的历史D范数分母≤1e−12时明确使用c=1；此分支只作用于NORM控制，不改变H投影。加入小向量分支assert。现有标量diagnostics新增两个任务各自的三级支持率，保留原整体支持率。受影响的控制和历史恢复测试2项通过（9.39秒）；新增任务诊断字段断言单独复核1项通过（7.58秒）。
