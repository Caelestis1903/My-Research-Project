# REVIEW REPORT：B6 evidence tool implementation v2

1. **Verdicts：政策设计PASS保持；工具实现FAIL。**

Module `76451b1f6f4933464c7a28b07bea6215531b2b67c10bccdf8d42d693c6b68ea4`、test `bf495141763c6b2e65062f18c8d971aeed61e81492f805eab536ec44a385d674`、coder v2 report `d4da0034e49491d9f8ba55035778acb9c7481c4f6282146ad4a7542191ed5109`按普通副本独立绑定。原B6-T1–T4四个反例已全部真实重放并fail closed；但新F140 path-shape规则不符合实际全部140成员的coverage要求，新增MAJOR blocker B6-V2-F140。**当前不允许实际生产只读baseline collection。**

2. **Requirement-by-requirement**

九类coverage非空/disjoint/exactrequired，四类quiescence distinct/presentregular/hash-stat绑定，mandatory section及observedrow等于inventory，archive inode与recorded source inode不等且current source检查均实现。Independent原empty8category、四quiescence同absent路径、删除quiescence段、移原source inode到rawarchive四probe全部拒绝，错误原因与修正目标一致；normal synthetic仍COLLECTED_UNREVIEWED/NO_GO。Static symlink-target不follow、protected/intermediate symlink与hardlink拒绝、double inventory/readback/fsync保持；未改生产validator/model/science/claim/release。新增F140roles启发式FAIL，详下。

3. **Blocking finding**

**B6-V2-F140 MAJOR（approved DESIGN §5.3全F140状态/锁/run/attempt不存在性）。** `_validate_coverage_shape:61–74`按leaf推role：只有literal`status`/`status.json`算status，含lock/run/attempt才分别匹配；任何role出现第二次即拒绝，最终roles必须恰四种。因此整个F140 category最多四个toy路径，不能表示140成员各自的status/lock/run/attempt路径。

实际canonical形状例如`queue_status/00e9bcaf117d178eaa3e.json`、`canonical_locks/00e9bcaf117d178eaa3e.lock`、`00e9bcaf117d178eaa3e`、`00e9bcaf117d178eaa3e/attempts/attempt-00`被新helper拒绝：runid basename不包含自定义role标记。即使人为改成每成员`F140/a/status.json,lock,run,attempt`，第二成员`F140/b/...`也因global role重复被拒绝。两种实际调用形状都已synthetic重放、同error“distinct status/lock/run/attempt absences”。不允许通过只声明一个toy成员或修改canonical path来满足工具：那会静默漏掉其余F140。

最窄返工是让scope/allowlist检验接受真实canonical、多个F140成员的完整显式路径集合，并保持每条required/expectedabsence绑定和非空完整coverage；不能用basename子串或global四role唯一性替代approved exact scope。完整140-member plan仍须独立review，而不是由synthetic名称自动推断。

4. **Corrections / evidence**

Reviewer未改实施/测试/保护文件。只普通copy新module/test/report到 `/tmp/f0-B6-policy-v2-review-20261014-a/clone`与archive，在新`/tmp` synthetic roots执行真实collector/validator调用和path-shape helper，不执行production collection/E2E/claim/release/model/CUDA/training。Raw `/tmp/f0-B6-policy-v2-review-20261014-a/archive/probe.stdout.txt`，JSON `/tmp/f0-B6-policy-v2-review-20261014-a/archive/probe.json`，项目Pythonwrapper峰值1。

Normal synthetic NO_GO；四旧probe拒绝；两新canonical/multiple-member例被错误拒绝。反例中的rawsource移动只发生于synthetic目录；不是production inode事故。19项v7protectedSHA一致。

5. **Coder checks / remaining risks**

Coder声称18 passed/3.37s，明确fullv15E2E未运行。没有单独rawstdout/manifest路径或可核验文件，本review仅将18数字归属coder报告；自己的六个negative probe和normal结果均有raw/hash，不能把inline数字转换成已独立核验的pytest结果。Newsource现存metadata match也意味着validator兼做current-source freshness检查；long-term original `/tmp` evidence消失或source后来变化时raw archive验证有依赖，须在actual baseline plan里明确，不宣称已实现source-independent archive验证。它不能让后来的source变化抹去历史采集窗口事实，亦不能把旧bounded archive当成当前授权。

没有automaticGO：collector/正常manifest execution NO_GO、historical OPEN_FAIL、forward NOT_GRANTED；status不接入productiongate。B6历史缺口和actual baseline/qualification/forward审查都未闭合。本review与并行B3修改分离。

6. **Production baseline collection / safe to run**

**目前不允许实际bounded readonly baseline RUN：工具FAIL，先修B6-V2-F140并独立review。** 政策设计PASS保持；baseline NOTCOLLECTED/NOTREVIEWED、qualification NOTACCEPTED、forward NOTGRANTED。B6OPEN/FAIL、activehold保持；staged GO、release、科学/生产RUN NO-GO。

Evidence `experiments/reports/stage_F0_v15_B6_forward_evidence_policy_implementation_review_v2_evidence.json`，SHA `3e802ed12c087718b3ee0cfa9a8bf92b3753da4f714c40eeb438bbbd5592654b`。
