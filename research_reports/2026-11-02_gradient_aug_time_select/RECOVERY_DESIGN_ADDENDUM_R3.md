# GATS02 恢复设计附录 R3：已观测的 native worker -11

状态：DESIGN完成，待协调者批准交coder。仅处理 `attempt_20261102_recovery_0315` 已发生的 worker `exitcode=-11`，不更改科研算法或扩大故障类型。原DESIGN及[R2附录](RECOVERY_DESIGN_ADDENDUM.md)继续适用。

## 1. 事实与最小调度修复

KuaiRand/B2/seed2022 的GPU0 worker在replay E42之后无Python错误消息退出，父进程记录 `worker exited -11 without RAM state`。现有schedule把它当不可恢复；同批其他worker结束后协调器退出，RAM再次丢失，test未打开。SIGSEGV原因尚未确定，不把并发当成已证实原因。

coder仅做下列修复：

1. 对Pipe EOF或worker已经退出但未收到完整RAM结果的路径，reap/join后读取最终exitcode，避免把尚未稳定的None当分类依据。只有已确认 `exitcode == -11` 的此类事件新增为 `recoverable=true`，保留 `oom=false`、signal/exitcode、job、GPU、attempt及时间。其他native退出仍为不可恢复。
2. 沿用每run最多两次原配置完整重试（初始执行加最多两次retry），恢复时不得借协调器重启抹去同科学job已观测的native失败。首个-11已耗用初始失败；新的fresh执行对此job是第一次恢复重试，最多再允许一次同类失败后的完整重试。其他job因RAM丢失重建不冒充新增seed，也不因别的job失败扣其自身重试额度。失败与retry计数分开列清楚。
3. -11重试放回原队列，dataset/seed/config/data/RNG/batch40960/50epoch/FP32完全不变，从初始化重算完整primary与replay，不从partial数值日志恢复。使用新attempt子目录，绝不覆盖失败日志。
4. -11重试采用现有serial思路的真正独占执行：等其他active结束再启动；该retry运行期间不再派发其他job。完成后才恢复正常资源调度。不要终止其他正在运行的任务，不缩batch/模型，不改变随机流。达到原始hard deadline不得新派发retry；已有任务按原合同完成。
5. 原Python算子错误、公式/身份/逐step一致性断言及其他非recoverable异常保持失败，不用广泛except或“重试所有退出”吞掉它们。既有OOM/OSError分类不扩展。
6. 在worker打开自身stderr文件后、开始训练前，以该文件为目标启用 `faulthandler` 的all_threads native栈诊断，文件生命周期内保持有效；正常退出前清理。只写诊断栈，不引入模型/optimizer/RNG磁盘快照或core dump方案。

## 2. 第三次fresh执行与成本连续

新执行仍必须DESIGN→coder→独立review PASS→冻结新source/commit→该source的两数据集真实CUDA Stage0 PASS→RUN。再用全新空目录重算20条基础轨迹重建RAM，继续原控制/确认/冻结统一test。不得读取旧test、放宽门槛或新增研究候选。

身份基准继续使用首个 `attempt_20261102_0005` 完整20条；R2的48端点历史预检、新RAM复核及逐step跨attempt身份核验不变。不能把仅11条完整轨迹的0315目录当作完整身份基准。

成本来源与身份来源分离：保持0005作identity source，另接受0315作**额外成本来源**（可使用一个明确重复参数/列表，不做递归目录猜测）。按原始绝对路径去重；分别枚举每来源实际完整run/pass与未完成pass，避免combined runtime_stats和其primary/replay子目录双计，也不能把0315 receipt内已经携带的0005累计成本再加一次。

- 0005的36个完整物理pass全部计入。
- 0315的11条完整科学job按实际合并runtime_stats计入；失败B2 job已完成primary的runtime_stats另计一次，未完成replay按现存batch_trace/epoch日志记可证明的**部分成本下界**。已提交trace的rows/gradient_calls/更新步、完整epoch_seconds等可按字段精确求和；缺失的GPU event/最后未提交步时间标N/A，不能凭E42线性外推为精确成本，不能把partial记为50epoch或完整pass。
- 新attempt发生的失败retry同样保留：最后成功结果的costs.csv不能代表整个任务物理成本，失败partial另列并汇总可测下界。
- 原始forecast和原始任务起点 `1793549242.1426222` 不变；全部已完成worker秒与部分worker秒下界进入保守成本记录/原阈值判定。10h=`2026-11-02 02:07:22.142622 UTC`，12h=`04:07:22.142622 UTC`，禁止重置。
- 科学唯一轨迹按已核验的dataset/config/seed去重，仍≤40；附加≤8、确认含新基线≤12。重算、partial和重试均不是新独立seed。

## 3. 必要验收

只加针对该故障的CPU调度注入测试，不启动研究训练：

1. 模拟完整调度事件：首次无RAM退出-11，下一原配置retry成功；应返回一个成功科学结果、保留失败记录、run_id/config不变、attempt/retry增加，协调器不因首次-11退出。
2. 模拟连续-11达到两次retry后耗尽；必须停止并保留最后失败，不能无限排队；带入历史已有一次-11时验证剩余额度不被重置。
3. 模拟非recoverable assertion/非-11 native退出；不得按新规则retry。
4. 用派发事件证明serial retry等待已有active完成，且运行期间无新job启动；原deadline仍能阻止其派发。保持已有CPU公式、历史预检、预算回归通过。
5. 真实来源成本receipt须列0005、0315各自完整/partial项及不双计说明；完整核算字段与下界/N/A字段分开。reviewer检查实际实现、coder报告和CPU结果；新Stage0必须与新source身份一致。

最终报告据实际状态写明两次协调器RAM丢失、已知-11、重试与重算成本、native原因未定及是否统一test完成；不保留“零训练故障”的旧表述。不增加额外硬化、运行方向或未注册对照。
