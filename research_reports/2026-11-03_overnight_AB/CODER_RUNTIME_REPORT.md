# CODER RUNTIME IMPLEMENTATION REPORT

实施位置：`/home/xzf/Code/phase0-3_overnight_ab/experiments/overnight_ab_20261103/`。历史只读目录：`/home/xzf/Code/phase0-3`。本报告记录实现与工程验证，科研结论交主agent分析。

## 实现

- config读取冻结protocol/plan；按family生成源码闭包，实际核对原history/pair固定hash及HG/AI外部依赖文件hash，source锁不冒用旧包hash。
- data_bridge调用原自然完整人口loader；训练view没有test数组。只读.npy缓存保存原数据数组及manifest，用于epoch量子恢复；缓存验证原raw hash和缓存文件hash，不保存任何模型/optimizer张量。
- inventory读取历史补评420条本轮相关端点和5500条完整V记录，核对每来源checksum、合法50epoch argmax及V/Test同model hash。旧A30逐轨迹采用独立LEGACY_A_RESELECTION_AUDIT.csv；A/B分别导出60/40槽位审计。
- selection统一六格等权严格最大/精确并列最早；evaluation对发表adapter只做一次FP32 sigmoid，统一使用history metrics。
- state与独立AF_UNIX guardian仅RAM保存全部Session；正常epoch register不限次数，与失败预算分离；epoch commit具连续索引/身份/标量ACK。只有完整DONE checksum后release。
- run提供prepare/smoke/run/run-job/status/pause/resume/retry/report/_guardian。正式run-job必须有活跃coordinator claim、review PASS、正确source闭包及trajectory锁。
- queue以完整epoch公平轮换A/B；每卡独立按显存/功耗85%、实测峰值×1.2和RAM预留准入，多worker并发。GPU派发检查与smoke共用按host/UUID锁；未知计算PID保留设备。由主agent明确允许共享的三个flir任务，按PID/start/boot/cmdline写入RUNTIME_OVERRIDES；纯MPS server不算训练占有者。
- 每条轨迹训练完成从RAM commit ledger重建完整合法50epoch V窗口，核验selected argmax，再冻结/统一评价selected/E10/E50、写标量DONE、释放RAM。训练后评价进程失效可恢复complete Session直接补评价。
- reporting自动产出原始端点、绝对/配对/2×2表、合法V曲线、物理成本、进度、决策、报告和台账增量；均以seed为单位，阈值/LL独立，少于五seed INCOMPLETE。
- CG计算入口与预定义矩阵保持默认关闭；主队列只派冻结核心集合。本轮尚无主组合工程失败，未激活替代。

## 必要修订

独立review指出跨卡恢复CUDA RNG按旧设备列表索引会改变HG dropout/AI随机门。经主agent授权，统一修改A/B调用点显式传session实际device：capture保存活动CUDA流，restore映射至新目标GPU；create先set_device。两卡最小随机张量检查通过。

另按独立review修复直接CLI绕过claim、未知GPU计算owner、逐卡首次分配门、正常yield的None owner窗口，以及TRAIN_COMPLETE评价进程失效恢复路径。旧包、旧训练和历史报告均未修改。

## 验证

使用指定Python：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

- `-m pytest experiments/overnight_ab_20261103/tests -q`：15 passed，含A/B原CPU机制与恢复fixture、严格选点/统计边界、epoch公平/资源门、guardian三次正常量子不耗重试、两卡CUDA RNG流映射。
- `-m compileall -q experiments/overnight_ab_20261103`：通过。
- `research_plans/26-11-03_overnight_AB/plan.py self-test`：PASS，80总身份/60核心/30预期新增/A30/B40。
- 真实历史inventory：420相关端点、5500完整V行；旧A30含13条RESELECT_ONLY_LEGACY、17条MISSING_SELECTED_STATE_LEGACY；未将其混入主表。
- 实际原生闭包hash核验通过；GPU可见、CUDA RNG双卡断言通过。
- 真实prepare入口完成：80条job；A_REUSE_AUDIT 60行，B_REUSE_AUDIT 40行；数据缓存、科学配置冻结、身份和首轮报告均已生成，无训练。

尚未运行正式训练或完整原batch GPU smoke；独立review通过后由runner执行smoke与正式固定矩阵。

