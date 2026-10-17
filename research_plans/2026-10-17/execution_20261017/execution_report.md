# 执行报告：IN PROGRESS

本轮尚未完成。已完成N0、训练前N6原文差异表、冻结N1矩阵和必要检查；32条H100 pilot已全部完成且核验通过，32条独立confirmation已持久启动。未把启动、完整训练、读数验证混同。

## 来源与工作区

计划取自origin/main@a56639ab，核心目录research_plans/2026-10-17；执行底座为v3@2d04c6de。原目录无HEAD/remote且所有文件未跟踪，全部保留；隔离checkout为/home/xzf/Code/phase0-3-20261017-git，远程分支codex/execute-2026-10-17-run。原start是GB18030，原件、UTF8文本、SHA及未占用种子检查已归档。

start明确10-17入口优先于末尾10-15旧入口。其§11要求非机械任务只由主agent完成，优先于此前AGENTS的designer/reviewer派发要求；设计、实现和自审依次保留，independent_review=false。RUN阶段luna-runner仅机械监控现有队列及收集原始状态，不作科学分析或选择。

## 实际训练协议

|项目|实际值|
|---|---|
|版本/模型|2026-10-17方案，沿用V3 fixed M3OE|
|构造/forward|experiments/v2_20261016.py::fixed_model；stage_S1_models.py::StageS1M3OEAdapter.forward(x,test_flag=False)|
|fusion / DARTS|四融合值固定sigmoid(1)≈.7310586；DARTS关闭|
|optimizer|单Adam(.9,.999,eps1e-8)或单无动量SGD；coupled WD1e-5|
|loss|mean_i[(p_d/q_d)*mean_t BCE]；Sample经验目标|
|domain/task|3域×2任务，训练p加权；主评价为六cell AUC宏均值|
|DBO / balancing / augmentation|均关闭|
|routing|emb.detach()保留|
|数据/物理batch|固定KR人口2006132训练行；B40960、49步/epoch、尾批40052|
|种子|tuning2030/2031，confirmation2032–2039；二者不混用|
|LR网格|Adam .0003/.001/.003/.01；SGD .03/.1/.3/1.0；两q完全对称|
|预算/选轮|H100；selection半集宏AUC1..100最优、最早平局；test解封延后|

Python3.13.5、PyTorch2.9.0+cu126、CUDA12.6、2×3090，复用phase0-3-env及原数据。KR原文件SHA核对通过。GPU原有另一项目进程保留；队列逐设备计入外部计算进程、每卡总数≤5、按显存<90%准入。当前可容纳两卡各4条本轮训练。原产物盘/home_new约845GB空闲；大型产物保留/home_new/xzf_phase0_3_n17_artifacts_20261017，仓库runs为本地忽略链接。

## 已完成的最小验证

CPU三步pipeline：Adam/SGD的loss身份残差5.96e−8，一步参数更新与原表达式完全相同；同seed数据/初始化/采样流及完整state恢复通过。完整GPU工程epoch实测49更新/2006132行、约8.7秒、峰值预留1192MiB；不是独占硬件吞吐估计。

N2 quota的完整/尾batch计数和目标、同图梯度、RNG恢复通过；小人口完整坐标四臂矩流水线完成。N3一draw全部局部替换小流程完成，明确ENGINEERING。首次局部smoke的配置关键字序列化错误已在正式机制实验前修复；未影响N1。

正式确认需完整test AULC，额外纯模型逐epoch留存层已在confirmation前补齐并逐tensor验证；pilot已有全selection/diagnostic预测用于调参，不用pilot test调参。详见CONFIRMATION_RETENTION_ADDENDUM.md。

## 任务与依赖

|节点|状态|依赖/范围|
|---|---|---|
|N0.1–4|COMPLETE|4068重算读数、672单调校准、人口及ML用户簇条件区间|
|N6训练前原文表|COMPLETE|八篇原文；最终方法/机制主张待实测|
|N1 pilot|COMPLETE_VALIDATED|32×H100；全部4900步/200613200样本，无失败|
|N1 confirmation|RUNNING|两优化器共同/各自赢家重合，新增32×H100，PID736337|
|N1 H200|CONDITIONAL|仅训练J的原文rho门槛，主H100保留|
|N2固定点|WAIT_N1|必须16锚×4臂×512draw|
|N2完整quota|CONDITIONAL|N1.6诊断残差触发，通常新增16轨迹|
|N3局部/独立确认|WAIT_N1|16锚32draw，选一子块/阶段后另4seed确认；最多64draw|
|N4拓扑|CONDITIONAL|N1.6及N3.6同时满足才启动|
|N5各分支|CONDITIONAL|按原文分别判定；ML在KR关键独立确认实用效应后触发|
|新模型test|SEALED|所有定义/模型/选择规则冻结后统一读取|
|最终分析/问题终态/发布|PENDING|须有真实完成与核验产物|

此前里程碑提交9f273ba（冻结/实现）与3e227ca（N0/启动）已推送origin并回读确认；本报告所处阶段不是最终交付。

进展补充：已完成的首8条pilot原始配置/曲线/指标/状态SHA在run_metadata及completed_artifact_index.json；其余仍在运行或等待，不从不完整响应面提前选LR。已归档历史672个cell/seed/阶段的人口与baseline上下文。N5-C固定ML人口适配已完成必要身份检查：N187500、每epoch5步、尾23660，跨optimizer seed保持同一行列表；是否正式执行仍等待N1门槛。最终七保护界的两例合成算术/JSON检查单独标ENGINEERING，不作为研究结果。分析脚本的NumPy整数JSON序列化问题已最小修正，不影响训练或原始数值，无须重训。

N1 pilot终态及完整源数已归档；N1_pilot_validation.json为32/32完成核验，N1_winners_frozen.json为选择依据。Adam共同/各自eta=.01，SGD均1.0，均为预设网格上边界。N1_PILOT_ANALYSIS.md与N1_pilot_surface保留128行horizon响应和图。3ae4fca冻结后启动confirmation，config逐条记录该commit；H100原始model每轮留存，test仍封存。原始逐batch trace本地保留SHA，Git新增每epoch实际域曝光汇总epoch_domain_exposure.csv。
