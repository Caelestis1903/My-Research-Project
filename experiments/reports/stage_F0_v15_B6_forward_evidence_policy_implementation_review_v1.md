# REVIEW REPORT：B6 forward evidence policy tool implementation v1

1. **Separate verdicts：政策设计 PASS；工具实现 FAIL。**

已批准policy design SHA `4084a9ec3054398e6b78ff2a1b1f8bcf55d1556b40fdff0a4c12eeae2941ff5a`，design review SHA `5d37146a57737f510469f77283255ab260e6e50adefba193bf404cece01544d6` 保持PASS：历史缺口不冒充恢复，baseline/qualification/forward/execution分离，execution始终NO_GO。该PASS不接受任何forward disposition或历史要求修订。

工具checkpoint独立审核实际module SHA `7cde4df2f06c24ecacce852953f061d1091e92466c93e7ad7ad58723c50881b7`、test SHA `357657470e1bd3a1ee5de47dd52f109112e321ec8ac422731b5819accd4a26e5`、coder report SHA `906c44a0409772f63e1f13155a6c8e2ff3f057d430a7aa53797bdcd85019bb44`。正常synthetic与13个coder tests通过，但四个精确反例表明完整coverage/quiescence/归档provenance校验不足，B6-T1–T4均MAJOR。**当前不允许启动实际生产只读baseline collection；先修复并独立复审。**

Baseline：NOT COLLECTED/NOT REVIEWED。Qualification coverage：NOT ACCEPTED。Forward disposition：NOT_GRANTED。B6 OPEN/FAIL；execution NO_GO。并发进行的B3/successor harness修改不属于此review，仅固定新B6 module/test/coder报告及policy输入。

2. **Requirement-by-requirement verification**

| Policy requirement | actual implementation/evidence | Result |
|---|---|---|
| no auto authority/history rewriting | collector:483–505固定historical OPEN_FAIL、forward NOT_GRANTED、execution NO_GO及threefalse；validator检查NO_GO/historyfalse；无runtime gate集成 | PASS；不是release许可 |
| raw malformed logs preserved | 实际synthetic history含非JSONline，按bytes归档，不清理；coder positive test通过 | PASS |
| no model/GPU/flock/claim/release | module仅标准库读/新archive写，无相关imports或API | PASS |
| roots/intermediate symlink拒绝，leaf symlink不跟随 | lstat组件、O_NOFOLLOW regular读取、readlink target单独ordinaryarchive、protected symlink拒绝 | 正常静态/synthetic PASS；ancestor并发swap未测 |
| no hardlink/shared source/archive | collector源nlink1与inode重复拒绝，archive exclusive/fsync/readback/nlink1/sourceid不等 | 创建时PASS；persisted source-inode provenance T4 FAIL |
| double inventory stability | 两轮sorted path/kind/bytes/stat/children投影比较，drift保留两档并REJECTED_UNSTABLE | PASS |
| presence/absence | unexpected absent、expected absent exists、broken symlink区别与错误拒绝 | 已覆盖subcases PASS |
| complete九类coverage | category键集合相同，但各类value非空未要求 | T1 FAIL |
| 四类quiescence actual evidence | value可重复指向expected absence，未要求present regular/独立正确绑定；persisted validator不检查结构 | T2/T3 FAIL |
| immutable external evidence/writebounds | exclusive新baselineID、archive/scope/forbidden overlap拒绝、directories/file fsync/readback；writes只指向指定new archive | 正常controlled路径PASS；实际目标plan尚未评审 |
| mandatory历史缺项/状态与bound metadata | collector记录policy/tool hash、scope/exclusions、声明与caller history_missing；validator对quiescence mandatory schema有缺项 | PARTIAL；本review不接受实际baseline |

无研究算法、公式、tensor/mask/weighting/gradient改动；工具新增可能改变future完整source inventory，不能自动更新生产授权SHA。

3. **Blocking findings / exact independent counterexamples**

**B6-T1 MAJOR（§5.3完整scope / §10 missing evidence）：** module:391–401只要求九类category键与union满足covered/required，未拒绝八类empty lists。独立真实collector调用只选择四个F140 expected-absent路径和quiescence目录，其余八类`scope_coverage=[]`，返回 **COLLECTED_UNREVIEWED、inventory_complete=true**，真实validator也接受。遗漏类别精确为current_gate_hold_release、v14_historical_controls、incident_failure_stream_evidence、audit_logs_and_incident_slice、C97_H51_state_runs_checkpoints、queue_and_search_gate、source_runtime_control_and_tests、frozen_data_split_manifests。这不是已完整canonical production baseline。

**B6-T2 MAJOR（§5.3四类窗口快照）：** module:387–390只检查category键，:407–409只检查scope required关系，不要求quiescence value present/regular/distinct。独立collector将`processes/claims/slots/proc_locks`全指向`prod:__b6_coverage/F140/attempt`（declared-absent），仍返回COLLECTED_UNREVIEWED，manifest四类kind都absent，validator接受。F140的legitimate expected absence不能替代四个必需quiescence raw snapshots。

**B6-T3 MAJOR（§5.3必需schema / §10持久读回）：** 从正常完整synthetic manifest中仅删除`quiescence_observations`，raw archive和inventory digests保持；`validate_postincident_manifest:519–609`仍返回成功。其没有对该字段四category、path_ids/inventory row绑定或lock_acquired/claim_or_slot_mutation=false作校验。保持NO_GO不等于manifest required evidence有效。

**B6-T4 MAJOR（§5.3独立archive / §8 alias反例）：** writer:190–218比较source inode，persisted validator:589–605仅检查archive nlink1、archive之间inode不重复及dataSHA，未比较recorded source inode。独立synthetic-only provenance-negative把原source gate文件`os.replace`移入observation1 raw payload，archive保持nlink1、bytesSHA正确；其device/inode **66309/19023277**等于manifest recorded source device/inode，validator仍成功。此例不是两个active路径共享hardlink：synthetic source原路径已absent，使用的是original source itself而非独立copy。它准确证明持久provenance约束缺失，不能报告为已证live production hardlink事件。

所有反例authorization仍为NO_GO；没有授权绕过或生产mutation。修正范围只限非空实际coverage、四类required present ordinary snapshot与schema/inventory绑定、persisted archive/source identity不等的校验及相应synthetic反例，不改变policy/algorithm/运行gate。

4. **Corrections made / evidence**

Reviewer无实施/生产修改。仅普通复制newmodule/test到 `/tmp/f0-B6-policy-review-20261014-a/clone`，所有真实collector调用限制在新建`/tmp` synthetic source/archive；这不是实际production baseline collection。没有E2E、claim/release、模型/样本/CUDA/训练或生产flock；项目Python经task wrapper，峰值一个Python进程。

Exact调用字段与返回值在 `/tmp/f0-B6-policy-review-20261014-a/archive/probe.json`，脚本 `/tmp/f0-B6-policy-review-20261014-a/probe.py`、raw `/tmp/f0-B6-policy-review-20261014-a/archive/probe.stdout.txt`。独立inode脚本 `/tmp/f0-B6-policy-review-20261014-a/inode_probe.py`，raw/JSON同archive目录。所有hash写入evidence；source/test immutable copies绑定上述7cde…/3576…。

5. **Coder checks / non-blocking risks**

Coder final raw **13 passed in2.50s**，stdout SHA `b5b558ca9b355ec9af7fe8ec6b3fc70b1864b0051a70ff091a18340fa05cfdfb`；manifest `dbe483d4cee5477b3f4229c3090daaac883594ac09f15c348a61648cd8597f1c`，六payload逐项已独立核验，module/test/report/design/review与raw全吻合。13项并不覆盖上述四个负例。Coder明确披露中间失败stdout没有独立保留；这些具体初期失败不能由本review独立复核，只能归为其报告，不构造缺失raw/hash。

19项v7 protected SHA当前一致，包括active hold/gate及两历史audit。本review没有据此关闭B6旧pre-edit inventory/source身份缺口，也没有复审正在并发修改的B3 harness。设计PASS与正常writer行为均不证明actual production scope计划完整、quiescence新鲜或无窗口间瞬时副作用。

Static lstat + leaf O_NOFOLLOW与预/后stat提供normal path拒绝；本次未做ancestor concurrently swapped的adversarial证明，不声称race-resistant全部路径隔离。Actual collection的exact roots/required/protected/absence/九类canonical覆盖、quiescence捕获时间与完整临时资格raw持久复制/`SYNTHETIC_FRESH_PEER_LOG`标签需另行review。不得只依赖category名称或13passed。

6. **Whether bounded production read-only baseline may proceed**

**当前：不允许。政策设计PASS保留，工具实现FAIL，返回Coder修复B6-T1–T4后独立REVIEW。** 之后即使工具PASS，也只可按明确reviewed scope另行bounded non-training/read-only collection，baseline结果仍COLLECTED_UNREVIEWED、execution NO_GO，需独立baseline/qualification/forward裁定。

B6 OPEN/FAIL、active hold保持；不批准历史closure、forward revision、staged GO、release、科学/生产RUN。原会话continuation授权不需要重复确认，但不能替代当前实现blocking校验。

Evidence `experiments/reports/stage_F0_v15_B6_forward_evidence_policy_implementation_review_v1_evidence.json`，SHA `4886f4fbab1b83544b9fcb04bb6635b2fe841eaa0362523773dcaf04dc13dfeb`。
