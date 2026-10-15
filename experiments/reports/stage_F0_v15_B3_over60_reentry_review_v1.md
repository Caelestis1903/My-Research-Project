# REVIEW REPORT：B3 >60秒PREPARED reentry test implementation v1

1. **Verdict：FAIL，仅implementation readiness。**

固定test SHA `58df59fc80b6500bc40552655e0e18cb75118bf7995db0b836b1223e731084f1`、coder report SHA `e4daf9781fb851cc434b9ab3280f2b53c4629fd134e50ed2d3a35287678df236`，与DESIGN v2 §7.2、§8第149行及terminal contract独立核对。受控61.25秒与same-UUID refresh思路正确，但B3-I1/I2阻止此checkpoint进入完整scenario资格RUN。后续v2修改单独审核，不纳入v1结论。

2. **Requirement-by-requirement**

Controlled monotonic +61.25使真实`observation_is_fresh`由true转false：独立短contract unit PASS。First PREPARED row/prelaunch/source raw byte与SHA捕获、same target UUID pairs、aged-caller fresh-query计数、defer后eligible commit与终态querycount不增：静态实现存在，未跑fullscenario。Immutable PREPARED row应保持而整个ledger允许合法追加；v1错误要求ledger全体不变。Terminal ledger/observation/digest/querycount比对存在，all terminal artifact exact-byte比对缺失。没有数学/模型/tensor/gradient或runtime源码变化。

3. **Blocking findings**

**B3-I1 MAJOR：test:2199–2200要求PRE_HOLD_RELEASE_DECISION NO-GO后ledger与PREPARED-only全体bytes/hash相同，与设计及实际runtime矛盾。** Runtime:2407–2442依次append RELEASE_ARTIFACT_DURABLE、PENDING_GATE_DURABLE、RELEASE_POINTER_DURABLE，再在:2452进行PRE_HOLD决策。DESIGN §7.2步骤2/3明确允许这些非授权动作、pending gate和pointer；此时active hold保持且HOLD_RELEASED/COMMITTED不应出现。按指定fixture memory序列先GO再pre-hold NO-GO，四个合法ledger event必须存在，故fulltest必在该错误assert失败。Coder报告也写入这个错误要求。最窄修正应验证原PREPARED首行byte-identical、正确非授权前三阶段append和仍active hold/禁止HOLD_RELEASED/COMMITTED，而不是冻结整个ledger。

**B3-I2 MAJOR：test:2233–2244 terminal ACK未覆盖exact terminal graph bytes。** Ledger、observation、committed digest与telemetry counter有断言；release artifact、pending archive、release pointer、final immutable archive、hold、canonical gate没有before/after raw bytes集合比较。按本任务明确验收范围补这些byte checks，保留noquery计数。

4. **Corrections made / independent evidence**

未修改repo source/test/protected production或执行fullscenario。实际v1函数短contract通过AST从外置SHA绑定副本提取执行，真实runtime freshness函数也仅AST提取；没有release/claim/E2E/训练。Probe `/tmp/f0-v15-B3-review-20261014-a/contract_probe.py`、raw `/tmp/f0-v15-B3-review-20261014-a/archive/contract_probe.stdout.txt`、JSON `/tmp/f0-v15-B3-review-20261014-a/archive/contract_probe.json`。项目Python任务wrapper，峰值1。

v1首次读取SHA与源码后，workingcopy已被并发v2更改。外置review copy仅逆转那两处已识别successor改动并恢复原digest变量，最终SHA精确匹配原58df…4f1；记录在restore脚本/stdout，repo没有逆转。Actual code checkpoint binding精确，不能混用后续f619…ee94。

5. **Remaining risks / test claims**

Coder只声明短clock test1 passed/0.09s，fullreal-validator/E2E未运行；没有提供或找到独立raw stdout/manifest文件，故该pytest数字按报告归属，不能另造rawSHA。Reviewer独立短contract验证真实freshness函数的+61.25行为，但没有验证事务scenario或生产路径。19项v7protectedSHA一致，B6保持OPEN/FAIL。此report不评价并行B6新工具。

6. **Safe to run**

**本v1 implementation readiness FAIL；修B3-I1/I2并独立review后才进入single bounded isolated nontraining scenario RUN。** Fullqualification未闭合；B6OPEN/FAIL，staged GO、release、科学/生产RUN NO-GO。

Evidence `experiments/reports/stage_F0_v15_B3_over60_reentry_review_v1_evidence.json`，SHA `0c4e1a2701c70d4d287c6479d93c9472ed7e47e3b40adb225159c4ce20216b7f`。
