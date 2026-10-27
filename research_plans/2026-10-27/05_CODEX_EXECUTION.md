# Codex执行任务书

你在Caelestis1903/My-Research-Project执行MDMT-AP-v1。先读取AGENT.md以及本目录00–07。任务是按已经确定的数学与工程合同实现F0，运行关键开发对照，再依据04自动判断是否需要F1、一次有限机制修订及正式复评。无需让用户选择算法或逐阶段批准。不要重新开展无边界文献探索，不将这里的未测预测写成结果。

## 一、执行前保护与静态验证

检查git status、当前branch/HEAD及main远端。保护用户未提交改动，独立新实验分支实现，不覆盖归档、旧研究方案、旧数据或全局结论。记录规划commit及config SHA。先执行：

```bash
python -B research_plans/2026-10-27/08_VALIDATE_PLAN.py
```

它只验证方案文件，不是模型验收。若默认GPU/数据路径不存在，进行只读定位，按精确hash匹配；找不到原始数据或合法GPU就报告BLOCKED，不生成虚假运行结果。不运行旧DFD cli的dry-run，不能默认已清理checkpoint还存在。

## 二、按依赖顺序实施

1. 实现contracts/config验证、源码来源记录、无副作用CLI audit/plan，随后prepare及新数据权限overlay。模块导入不得创建训练/缓存。
2. vendor固定native必要依赖；实现新NativeFeatures wrapper、前缀训练、完整teacher仿射转换、强B、缓存身份和raw-X完整恢复。只复制所需实现，不把整个history当Python包。
3. 实现F0/READ/SOURCE_K2/WIDE/TARGET_ONLY/DIRECT、同计划初始化和源约束。先通过06核心语义测试，再做明确SMOKE（结果不入研究统计）。
4. 跑stage1六臂×两seed×两自然数据，共24条上限；自动保存V选点及固定终态，生成固定功能干预及Q决策。报告训练角色、实际batch、dtype和模型身份。
5. 仅按已写STAGE_DECISIONS运行来源归因和全campaign一次机制桥。桥失败不能换第二类型。保留原结果，不用新版本覆盖原阴性。
6. 获准数据实施并测试F1事务/需求编译，再运行stage2六臂。先F1取得post14 profile，再从头训练匹配静态控制；NO_GROW必须独立完整轨迹。根据第04节进行等时间继续训练。
7. stage2完成后仅在唯一桥尚未使用且条件成立时运行MASS-inspired触发桥。若未满足，状态为NOT_TRIGGERED，不补满轨迹。
8. 在任何E评分前固化FORMAL_LOCK，按CORE或DYNAMIC实际七臂映射运行四新seed。预算不足和科学停止分开。正式前缀从其合法T0重新训练或严格身份复用。
9. 仅用F构建正式干预群体及refit状态，写EVAL_LOCK绑定完整模型。然后评分E与legacy，所有数据集（包括失败/未推进）进入最终报告。
10. 生成实现说明、代码与配置hash、实际矩阵/NOT_TRIGGERED、聚合结果、成本、机制判断及下一步选择。提交新代码/测试/配置/聚合报告到实验分支；本任务不授权删除旧记录或擅自把新结果写回旧报告。

## 三、实现后CLI调用合同

下列命令目前是需要实现的接口，不应在实现前当成已有runner。audit/plan必须只读，其余命令只执行已登记身份；可自动连续执行，不因未触发阶段等待用户。

```bash
CFG=research_plans/2026-10-27/07_EXECUTION_CONFIG.yaml
python -B -m experiments.mdmt_adaptive_prototypes.cli audit --config "$CFG"
python -B -m experiments.mdmt_adaptive_prototypes.cli plan --config "$CFG"
python -m experiments.mdmt_adaptive_prototypes.cli prepare --config "$CFG"
python -m experiments.mdmt_adaptive_prototypes.cli prefix --config "$CFG" --phase development
python -m experiments.mdmt_adaptive_prototypes.cli run-stage --config "$CFG" --stage stage1 --resume
python -m experiments.mdmt_adaptive_prototypes.cli analyze --config "$CFG" --stage stage1 --role Q
python -m experiments.mdmt_adaptive_prototypes.cli run-stage --config "$CFG" --stage attribution --resume
python -m experiments.mdmt_adaptive_prototypes.cli run-stage --config "$CFG" --stage bridge --resume
python -m experiments.mdmt_adaptive_prototypes.cli analyze --config "$CFG" --stage bridge --role Q
python -m experiments.mdmt_adaptive_prototypes.cli run-stage --config "$CFG" --stage attribution --resume
python -m experiments.mdmt_adaptive_prototypes.cli run-stage --config "$CFG" --stage stage2 --resume
python -m experiments.mdmt_adaptive_prototypes.cli analyze --config "$CFG" --stage stage2 --role Q
python -m experiments.mdmt_adaptive_prototypes.cli run-stage --config "$CFG" --stage bridge --resume
python -m experiments.mdmt_adaptive_prototypes.cli analyze --config "$CFG" --stage bridge --role Q
python -m experiments.mdmt_adaptive_prototypes.cli freeze --config "$CFG" --phase formal
python -m experiments.mdmt_adaptive_prototypes.cli prefix --config "$CFG" --phase formal
python -m experiments.mdmt_adaptive_prototypes.cli run-stage --config "$CFG" --stage formal --resume
python -m experiments.mdmt_adaptive_prototypes.cli freeze --config "$CFG" --phase evaluation
python -m experiments.mdmt_adaptive_prototypes.cli analyze --config "$CFG" --stage formal --role E
python -m experiments.mdmt_adaptive_prototypes.cli analyze --config "$CFG" --stage formal --role legacy
python -m experiments.mdmt_adaptive_prototypes.cli report --config "$CFG" --stage all
```

这是依赖驱动的操作顺序，不是无条件shell串：命令从已有decision选择可运行项。bridge未注册返回NOT_TRIGGERED，重复调用只恢复同一桥不新增；归因桥后可新注册但总数不超过4。没有formal时，evaluation锁定实际完成的stage1/stage2模型，使用对应--stage评分E/legacy，不能对空formal报完成。formal run结束内部先完成F定义诊断状态，freeze evaluation验证它们存在才允许评分。

V选点在各run内部按协议发生，analyze不能从Q/E重新挑checkpoint。--dataset/seed/arms只能缩小冻结网格；少对照时只能INCOMPLETE。多GPU只并行互相独立的run，source-profile依赖、同一run窗口和文件写入不可乱序。

## 四、必交产物

服务器保留完整selected predictor及其native/B/root/normalizer依赖、pass20、profile/事件/行计划、优化器恢复状态和四类干预状态，直到raw-X再评分及hash核对完成。不要像旧清理一样在归因/恢复完成前删除模型。

仓库新增报告至少包含：IMPLEMENTATION_REPORT.md、EXECUTION_STATUS.json、DATA_CONTRACT摘要、RUN_MATRIX.csv、RESULTS_LONG.csv、FUNCTIONAL_EFFECTS.csv、STRUCTURE_EVENTS.csv、COST_REPORT.csv、STAGE_DECISIONS.json、FINAL_REPORT.md。没有启动的行明确NOT_TRIGGERED/BLOCKED，不能用0替代未测。逐行预测、CSV原数据和checkpoint不上传Git。

FINAL_REPORT必须回答：F0是否超过READ/WIDE/target-only；来源追加监督是否有增量；局部互补/跨任务复用是否成立；F1是否超过固定profile/宽函数/NO_GROW且费用合理；应该保留哪个简单方案。把开发与正式、选点与终态、同曝光与同时间、数学推导与实际结果分开。

## 五、停止、恢复与自主推进

资源缺失、原始数据hash不符、冻层发生变化、gamma0不恢复教师、完整状态不能恢复属于硬阻塞；先修实现或保留BLOCKED，不用改科学目标解决。低效果、零增长、低支持是合法结果，按04停止或退回简单路线，不要求用户重新批准。

崩溃恢复不能换S片、候选、seed或清空费用；只继续同事件。已有成功文件及hash一致则复用，不重复训练。发现合同内矛盾应记录CONFIG_CONTRACT_ERROR并作一致性修复；不得把算法改动伪装工程修复。达到总预算停止新注册轨迹并保存状态，输出已完成证据、实际缺口和允许的研究判断。
