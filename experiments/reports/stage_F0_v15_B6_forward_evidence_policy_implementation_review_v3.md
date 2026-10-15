# REVIEW REPORT：B6 evidence policy tool implementation v3

1. **Separate verdicts：政策设计PASS；工具实现PASS，仅工具实现范围。**

Module SHA `f76433a63a9fafa9332752f837a058c470473405b3596bfbba6e9e599da5b404`、test `d0bf4b46c08e800620d237788be8ef93c46b98ce2b38c522ca8fea34a148674d`、coder v3 report `ff36e940f1bfe9ac3cd66eb2abd90c64989bc1e6a3e2853ee47e8c894c8f0fb2`以普通source副本固定；approved design/review `4084a9ec3054398e6b78ff2a1b1f8bcf55d1556b40fdff0a4c12eeae2941ff5a` / `5d37146a57737f510469f77283255ab260e6e50adefba193bf404cece01544d6`。旧T1–T4和v2-F140 blocker已独立关闭，无本checkpoint剩余blocking finding。

**允许下一步按已批准设计执行有界、非训练、生产只读baseline collection。** 这不是已经采集/接受baseline或forward处置，不改变B6历史OPEN/FAIL、executionNO_GO、activehold，不批准staged GO/release/claim或科学RUN。

2. **Requirement-by-requirement**

| Contract | actual independent evidence | Result |
|---|---|---|
| full F140 role/path shape | explicit140 sorted unique20hex成员，每member四role绑定canonical status/lock/run/attempt；跨member锁可重复role但路径不重复 | PASS；actual frozen140 IDs helper接受560paths/140locks |
| malformed role/identity拒绝 | missing role、status→lock misassignment、duplicate ID、missing member全部真实probe拒绝 | PASS |
| coverage非空/disjoint/required | 原empty8category probe拒绝；20tests包含cross-category重复拒绝 | PASS |
| 四quiescence actual files/schema | 原sameabsent路径probe、删除整个段均拒绝；regular/hash/beforeafter绑定保留 | PASS；actualfresh-window内容未采集 |
| archive/source noalias | old moved-original source inode/nlink1 probe拒绝；suite source/archive hardlink与readback/corruption覆盖 | PASS |
| symlink/absence/double inventory | 不followleaf target、protected/intermediate拒绝、broken不是absence、两次drift保留REJECTED_UNSTABLE | copied-source20tests PASS |
| noauto authority/history rewrite | normalCOLLECTED_UNREVIEWED、NO_GO；GO mutation拒绝；historic三false与声明保留 | PASS；不是forward acceptance |
| no production/model动作 | pure module仅std库；所有collector运行只在synthetictemp roots；19项protectedSHA一致 | PASS |

没有数学、scientific/model/tensor/gradient或productionvalidator改动。F140 explicit member schema是路径证据修正；实际生产成员仍需Runner绑定冻结集合，不能拿synthetic140IDs当真实范围。

3. **Bugs/deviations found**

无剩余blocking implementation问题。V2按basename推role、global role唯一性造成无法表示真实F140的规则已替换为每成员显式role/canonical path集合。旧四probe在新schema下按同等语义重放，均正确fail closed。

非blocking限制：generic tool验证140-ID数量/形状，不自行证明该集合等于reviewed frozen F140；actualcaller必须使用已绑定source snapshot真实成员。Persisted validator亦核current source stat，不能将其等同完全source-independent archive verifier；长期源路径/临时证据消失后的验证与currentness要求须在baseline方案和独立裁定中分开说明。Ancestor并发swap未实测，不声称覆盖所有adversarial path race。

4. **Corrections / independent tests**

Reviewer未改implementation/tests/protected production。普通clone `/tmp/f0-B6-policy-v3-review-20261014-a/clone`，独立unique basetemp `/tmp/f0-B6-policy-v3-review-20261014-a/basetemp-final`，rawexternalarchive `/tmp/f0-B6-policy-v3-review-20261014-a/archive`。使用项目Python与requiredslotwrapper，峰值1。**独立实际pytest：20 passed in8.14s、exit0**，raw `/tmp/f0-B6-policy-v3-review-20261014-a/archive/independent-pytest.stdout.txt`。首次exit4发生于clone缺失conftest的source-only `stage_F0_data`依赖、测试未运行；已保留`independent-pytest-incomplete-clone.stdout.txt`，补源码依赖后用新basetemp，未删除旧证据。依赖导入没有调用data/model/training。

Exact independent probes `/tmp/f0-B6-policy-v3-review-20261014-a/probe.py`、raw/JSON在archive：normal140-member synthetic有560declaredabsences并NO_GO；四旧反例拒绝；**只读冻结snapshot SHA717ea4…4533的actual140 IDs纯schema检查**接受全部560canonicalpaths、140 distinctlocks，未inventory/collector生产；missing role/misassigned/duplicate/missing member和GO均拒绝。Original-source移动仅在disposable synthetic目录。

5. **Raw bindings / remaining gates**

Reviewer raw manifest `/tmp/f0-B6-policy-v3-review-20261014-a/archive/REVIEW_RAW_MANIFEST.json`，SHA `07b46ba399884f8905773dbdeb052cf0a36bfc04429f2f8a8e4f2d59cf5293da`，绑定8payload（module/test/coderreport、probe/raw/JSON、两次pytestraw），无self-hash。Coder20passed/5.88s只见inline report，没有独立rawstdout/manifest提供；不能伪造其rawSHA。本report主要结果由独立20test+probe支撑。

19项v7protectedSHA（含hold/gate/twoaudit）一致。Productionbaseline尚未采集，不能作baseline ACCEPTED、qualification全部PASS或forward acceptance。B3v2另外scoped implementationPASS，fullscenario待RUN；attempt4其它未闭合项保持。新增tool/test必须进入future真实完整sourceinventory，不更新生产授权hash。

实际baseline RUN限approved §5.3/5.4：完整current gate/hold/release、v14/incident流、两原audit与九事件slice、C97/H51树与checkpoint rawSHA、actualF140560absences、queue/search、当前scientific/runtime/control/test、frozenmanifest/splithash；精确scope/required/protected/absence/排除及fresh四quiescence时间要绑定。必要checkpoint只hash raw，不反序列化；不读取samples、不构造模型、不获取productionflock/创建absentlock/修改scheduler或生产状态。保存所有临时资格raw的独立ordinarycopies与完整manifest，标SYNTHETIC_FRESH_PEER_LOG，保留所有失败/不稳定partialarchive；不覆盖旧报告。Use boundedtimeout/五项目Python上限与指定新externalarchive，collection结果仍待Reviewer验收。

6. **Safe to proceed**

**工具v3PASS允许上述有界非训练、生产只读baseline采集。** 不批准baseline ACCEPTED/forward disposition，不关闭B6，不批准任何staged GO、release、claim、科学/生产执行。会话已有continuation授权无需重新询问；采集后独立审核实际raw、完整coverage与所有剩余前提。

Evidence `experiments/reports/stage_F0_v15_B6_forward_evidence_policy_implementation_review_v3_evidence.json`，SHA `0e1ee567f413ff1598f8b93522776d4ba890eadf8d7977d16f609df7b7659c61`。
