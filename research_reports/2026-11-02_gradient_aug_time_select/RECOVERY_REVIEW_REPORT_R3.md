# GATS02 恢复补丁 R3 独立审查

**Verdict：PASS。无未解决 Blocking。** 可以冻结当前恢复源码，执行该source/commit下的新两数据集CUDA Stage0；通过后再派发新的完整RAM重建队列。

审查输入为RECOVERY_DESIGN_ADDENDUM_R3、实际run/recovery及必要测试、RECOVERY_IMPLEMENTATION_REPORT_R3、R3_CPU_TEST_RESULTS.xml与 `recovery_r3_cpu_preflight_final/`。审查者与coder不同。此次未运行科研训练或native stress，未修改此前审查历史。

## 已触发故障的限定处理

- `run.py:189` 在Pipe EOF或已退出但缺完整RAM结果的路径先join，再读取稳定exitcode；只有-11新增为recoverable，保留oom=false、signal/exitcode。收到完整成功结果的路径不套用此分类。
- 既有初始执行+最多两次retry仍有效。`native_failure_history` 从显式旧来源的原始failure与FAILED trial记录匹配完整科学job身份；0315的KuaiRand/B2/2022已失败一次，新首次派发为attempt2/retry_index1，若再次失败仅剩attempt3。其他job额度独立。
- `next_pending_index` 使serial retry等待已有active结束，且serial retry在active期间禁止新job派发。失败仍从初始化重算原配置的完整primary/replay，使用新attempt目录；外部任务不终止。
- Python assertion/身份错误与非-11 native退出继续失败；原OOM/OSError分类没有扩大。pending retry同样受原hard deadline限制。
- worker仅在自身stderr有效生命周期内启用all_threads faulthandler，并在正常Python退出时关闭；没有磁盘模型、optimizer、RNG快照或core dump实现。SIGSEGV根因尚未确定，串行重试不被写成已证明的根因修复。

## 身份、预算与物理成本

身份基准仍为0005完整20条；0315只作为显式额外原始成本来源。来源按resolve后路径去重，不递归读取继承成本收据。合并runtime_stats替代其primary/replay子统计；未合并job的完整primary及partial replay分别计量。当前fresh attempt的全部attempt子目录同样进入物理账本，因此成功结果不会覆盖失败retry成本。

真实最终CPU预检核算：0005为36完整pass、21367.9183922371 worker秒；0315为19完整pass及1 partial、worker秒下界9862.585071863257；合计55完整pass加1 partial、**worker秒下界31230.50346410036**。0315 partial replay保留42个完整epoch、2086个已提交更新step及对应trace计数；缺training GPU event/shadow更新的部分列为N/A，未按进度比例外推。精确提交trace数与整段物理成本下界在账本中区分。

原起点1793549242.1426222、10h截止1793585242.1426222、12h截止1793592442.1426222及原forecast17870.690975639038保持。已有完整成本与partial时间下界进入原预设超成本取消顺序。科学去重、跨attempt逐step/端点一致性阻断、附加8、确认含新基线12、科学总40及统一test门保持原合同。

## 独立必要验证

独立执行指定环境 `python -m pytest -q experiments/gradient_aug_time_select_20261102/tests`：**12 passed，1.73s**。CPU事件注入直接经过真实schedule，覆盖EOF/已退出-11的join后分类、同配置retry成功、最多attempt1/2/3、带旧失败仅attempt2/3、serial排他、非-11/assertion不retry及原deadline取消retry。真实成本测试覆盖重复来源去重、原始完整/partial核算和实际KR/B2/2022旧失败额度。原历史schema/48端点、10/12h、确认12/科学40回归继续通过。

读取最终真实preflight：两数据集verified=true，48/48端点匹配，未读取test或权重；source manifest为 `7f7a0eecb052ca0be1f5f7ffc07433e85d5e2d73bcbfd7e2f6f0ff4efac328a1`。objectives.py及shadows.py与0005原执行SHA逐字节一致。此次审查没有新增训练、研究候选或未触发故障类型。
