# REVIEW REPORT：attempt4 bounded isolated qualification

1. **Verdict：FAIL，限“完整 qualification acceptance 是否闭合”的判定。**

四组raw运行结果全部真实通过：**1 passed /559.19s；1 passed /591.47s；16 passed /503.14s；7 passed /7.66s，均exit0**。本review接受这些观测结果，没有发现本轮生产写入或进程超限证据。它们不足以闭合v7的全部pending requirements和DESIGN §§8/10；Q1–Q4保持blocking。此前code-precheck PASS不撤销；不能把未执行的验收条件写成通过。

绑定run source仅为attempt4 clone test SHA `c2365231af4deebc8bdbeff48f359c8b7a7a61646ba882ccc36f38f4b8f1cd76`，源码副本 `/tmp/f0-v15-attempt4-review-20261014-a/archive/attempt4_test_source.py`。审核期间working copy正在加入后续rendezvous修复；那些变化不是attempt4来源、不纳入本结论。Approved DESIGN v2 SHA `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316`；v7 report/evidence `01675098e0e55f936690f49cc7d207ac625428f3cdb8a8a7b7ab2e1aca359449` / `4cff0b50a22712fe056e23673aa16927e4cf86279b134c99fedc40609216edf5`。

**B6继续OPEN/FAIL；本review不批准incident closure、staged graph GO、principal release、生产或科学RUN。**

2. **Requirement-by-requirement verification**

| v7 pending requirement | attempt4证据及结果 | 状态 |
|---|---|---|
| P1：两边界真实changed-member peer authentication/continuity + supervisor lifecycle | 两组真实supervisor→child→runner，receipt/eventSHA一致，认证trace到runtime:1744 true，peer CLAIMED、三锁held、worker/child nonce与PID/startticks join、heartbeat增长，deferred exit75/norespawn/nopeerstop；original marker顺序归档被后续capture覆盖 | PARTIAL，Q1 |
| P2：受控>60秒exact PREPARED、immutable audit/identity/COMMITTED保护 | caller stale observation触发实时刷新，950/1000、900/1000阈值defer、后续eligible重入、UUID/source/hold拒绝；终态ledger/observation字节检查通过。但无>60秒受控中断/虚拟时间、无长重入前后audit/PREPARED字节证据 | NOT CLOSED，Q2 |
| P3：全部syscall/fault/restart/later-hold | independent PREPARED clone的14 named fault hooks+PREPARED、每点later-hold保存拒绝、final rename存活及还原pending模拟不存活、一次COMMITTED、held-lock fence已验证；未覆盖每次file/dir fsync分离、终态different transaction nonce与所有字节/禁止telemetry-query断言 | PARTIAL，Q3 |
| P4：当前相关regressions | successor文件selected13+parameterization→16项，identity7项；没有worker-control/launcher/queue/run/migration/v14相关suite raw | PARTIAL，Q4 |

DESIGN §8逐项：amendment正常synthetic图及四种digest拒绝 **PARTIAL**；初始per-device/mirror/both/neither/non-target-unreadable/utilization100矩阵 **PARTIAL**；target集合稳定/duplicate worker/GPU/slot负例 **PARTIAL**；B2真实两边界行为及部分fraud负例 **PARTIAL**；B3 **NOT CLOSED**；F140 wrong-full-digest/outside/touched/C97 monotonic **PARTIAL**（未跑broken-symlink/hidden-history/UNCLAIMED/duplicate/death-nonce全矩阵）；old-authority/flags/queue/pointer/asset absence mutation **PARTIAL**；real dispatch/release/peer claim成功 **PARTIAL**（selected-boundary early-return跳过直接claim尾部，former hash-only preclaim rejection→corrected all-layer predicate未完整证明）。详细mapping写入evidence。

研究公式、tensor/mask/routing/weighting或gradient未在本控制qualification改变；这组fake/sentinel执行没有科学效益结论。

3. **Blocking bugs/deviations / exact evidence gaps**

**Q1 MAJOR：两份归档race_order混合不同capture世代。** Run-bound fixture:2130–2152 `capture_then_signal`每次调用重写snapshot/resume；:2761在deferral完成后重读并覆盖原rendezvous `resume_meta`；:2794–2796将新resume和旧snapshot/child一起归档。child-preclaim saved child time394941.664930508 < resume394960.548373022；serialized saved child395580.142278851 < resume395589.55627747。两组resume snapshotSHA均不等于original snapshotSHA。Manifest虽正确绑定现有字节，却不能恢复被覆盖的原始resume事件。原helper:81–85严格顺序assert曾通过，所以不能认定实际运行逆序；但original-order raw证明丢失，不能将此archive验收写为完整。认证、三锁和heartbeat证据不因此作废。后续源码修复应独立review并用新归档证明。

**Q2 MAJOR：B3长重入未验。** Run-bound :1998–2038立即PREPARED→resourceNO-GO→reentry→terminalACK；:1397–1570覆盖aged caller和identity/resource负例，没有受控>60 interruption、时钟推进或长重入前后immutable audit/PREPARED raw比较。整体559/591秒包含大量静态校验与子进程等待，不能替代指定PREPARED中断时长证明。Terminal ledger/observation保护子案例通过，但其不能闭合此项。

**Q3 MAJOR：named fault matrix不等于全部syscall/terminal acceptance。** Run-bound :1648–1746的14 hooks都跑到，并逐点注入later active hold再验证保持，final rename存活/不存活在两个focused函数中分别覆盖。Runtime `durable_bytes_create:604–621`、`_atomic_replace_bytes:624–643` 与 `_append_jsonl_durable:559–575` 包含file fsync/parent fsync；现有hooks未逐个区分这些syscall中间状态和archive/preimage创建边界。比如`RELEASE_ARTIFACT_RENAME`是在已完成file+dir fsync的durable immutable helper之后。实际终态release different-nonce不是receipt wrong-nonce测试；terminalACK目前只比较ledger/observation/committed digest，并未证明all terminal artifact bytes和不query telemetry。不能把聚合成功扩大为全部§10通过。

**Q4 MAJOR：相关suite与完整policy/negative matrix未跑。** archive `run_control_regressions.sh:18–30`、`run_identity_freshness_regressions.sh:17–23`只选successor测试；25个collected PASSED item全部来自同一个文件。run-bound :2862–2875 requested_boundary earlyreturn跳过claim尾部。没有core worker-control/launcher/queue/run/migration/v14 suites的current raw；前述§8negative与one-GPU full selection矩阵也不在attempt4执行节点。已有旧报告的历史pass不能代替本轮要求的current qualification。

这些主要是qualification证据/覆盖缺口，除Q1 fixture归档错误外，不等同已证runtime算法bug。没有新增CRITICAL实现finding。B6历史缺项独立继续OPEN/FAIL。

4. **Corrections made / independent verification**

未修改实施源码、测试或protected production state；未运行测试、训练、E2E、CUDA数学或模型执行。仅用项目Python/task-slot wrapper读回与hash现有证据，新增本report/evidence；review源引用固定到attempt4 clone及外置源码副本。

Archive `/tmp/stageF0v15qualification-20261014-attempt4/archive`；实际manifest文件 **final_artifact_hashes.json**，SHA `12a7f49451edcf4a6f802e2adc0aa83d71a45e9e393cedd8326f472fb258abcd`。**217项payload的SHA和size全部独立一致、无self-entry/重复path**。Allowlist manifest **1348项现有clone文件SHA/size一致**，排除dataset/unrelated checkpoint/symlink；必要冻结H49 checkpoint只用于历史SHA，不反序列化。两个peer manifest payload全部核验，prehistory原bytes/hash/inode metadata保留。此clone log初始化不repair生产malformed历史日志。

运行raw四组exit0；完整PASSED nodeids、command脚本、stdout/stderr SHA、start/end、process samples写入evidence。Regression command.txt写`<reviewed-nodeids>`，实际节点由已hash的launch脚本和raw pytest列举可恢复。没有另存独立runner报告文件，parent所述结果由这些raw直接验证。

保护：run的before/child-after/serialized-after/all-after **9项记录逐字相同**，独立current SHA一致；另外**19项v7 protected artifact SHA**一致。生产hold active、F140无status/canonical-lock/run历史新增、successor release/observation ledger absent。不能从此恢复B6缺失pre-edit inventory/source bytes。

5. **Remaining risks / process cap**

两组真实并行actor为pytest+2supervisor+2child，上界5；raw /proc项目Python采样峰值 **5/5/1/1**，overlimit标志全部0。观测支持本次不超限；采样不能证明每个瞬时全局状态。顶层pytest与fixture各有外置scheduler namespace，不能将各自5-slot设计推论成跨namespace自动全局限制；本次actor上界与全局采样共同提供约束证据。各组顺序执行、唯一basetemp和外置archive。

两个peer heartbeat **1429→1617**、**1894→1988**；同一child pid/startticks保持，held canonical/worker/GPU锁、真实CHILD_STARTED单次、deferral event automatic_respawn=false、peer_stop=false与receiptSHA吻合。受控peer最终由fixture清理kill不表示因resource deferral停止peer；continuity检测时它仍alive/progress。MAC只能由持有secret的真实supervisor校验，archive保留receipt/event/hash，review不会恢复pipe secret。

Fake telemetry与sentinel loader只返回hash metadata；peer在真实claim后被hook保持并心跳，phase无training_before；deferred child在claim前返回。没有模型构造/forward/真实样本或checkpoint加载证据。仍需对Q1–Q4按既定设计补足reviewable evidence；本report不提出新算法或新研究hypothesis。

6. **Whether experiments are safe to run**

**attempt4已执行raw子案例通过；完整资格：FAIL/NOT CLOSED。** 允许工作流返回必要fixture实现/独立REVIEW并按已授权非训练qualification边界补足未验项；不授权重跑科学任务或改变生产hold/gate。**B6 OPEN/FAIL；staged graph GO、principal release、生产/科学RUN均NO-GO。**

Evidence `experiments/reports/stage_F0_v15_qualification_attempt4_review_v1_evidence.json`，SHA `eb75fbe7e27e8c983858160106176e02ebf24e271efd4281cd7369b9f9d09b56`。Independent audit `/tmp/f0-v15-attempt4-review-20261014-a/archive/audit.json`，SHA `f83973f151fae18b3832ed4ca03b29b90f2c58aaac071342936bdb5812471bc2`。
