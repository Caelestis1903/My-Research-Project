# REVIEW REPORT：clone-only peer audit-log fixture remediation

1. **Verdict：PASS，限隔离 fixture remediation 与 qualification precheck。**

无剩余 blocking finding。当前 test SHA `c2365231af4deebc8bdbeff48f359c8b7a7a61646ba882ccc36f38f4b8f1cd76`；coder implementation report SHA `9b2b033d27ed926014226c328566fd1631f83fd96182e325d91060cdb00cf525`。实际 diff、严格 parser、helper 与条件调用、focused raw 和真实临时产物已独立核对。不把干净 synthetic fixture 等同于修复生产日志或已通过真实 qualification。

2. **逐项核验**

| 要求 | Actual source/evidence | 结论 |
|---|---|---|
| 仅 inner peer-continuity log | target fixture:1932–1943 在 peer flag 后调用；参数为源 `v15.RUNS/...worker_1` 与 inner `repo/...worker_1`；非 peer 路径无调用 | 符合 |
| COW 后、graph/进程前初始化 | source:1928 detach 后、1944 build graph 前；source_log_hashes 保留未重置 | 符合 |
| alias拒绝 | helper:1047–1051 通过samestat拒绝源/clone alias并校验bytes相同；unit test 首先构造hardlink并期望拒绝 | 符合 |
| preserve原bytes/hash/inode | 外置peer evidence root；helper归档bytes并readback；metadata含source/original/fresh SHA、size、device/inode | 正常成功路径符合 |
| new empty ordinary inode原子替换 | helper:1060–1084 exclusive new file、file fsync、old mode、replace、parent fsync、new inode/empty/regular/sourceunchanged、真实strict parser [] | 符合 |
| 真实新peer事件保留原认证流程 | 未改parser/auth或supervisor；原真实peer append与prefix capture保留 | 静态符合；E2E待验 |
| runtime/science baseline | 与上次source archive比较226项Python源，仅本test drift；runtime/runner未变 | 符合；没有公式/tensor/gradient改动 |
| production/source state | 19项v7 protected SHA全部一致，两生产audit保持冻结SHA | 符合 |

3. **Bugs/deviations found**

没有 blocking implementation defect。MINOR报告偏差：coder报告声称metadata记录mode，但实际JSON没有mode字段；代码读取旧mode并应用到新file。MINOR证据限制：metadata在reset完成后写，archive bytes/metadata未fsync，不能声称中途崩溃仍有完整inode metadata的durable archive。本范围只要求正常初始化的clone-only证据，不扩大为该helper自身的crash qualification。

严格 `_read_jsonl_exact` 的 malformed-history fail-closed 行为保持。此初始化只为新的synthetic peer invocation建立clean ledger；生产历史warning行和B6 incident未改变、未被清理。原外层clone及其allowlist/source history不由helper写入。

4. **Corrections made**

Reviewer未修改源码、测试或生产；保留已有focused产物的普通副本到 `/tmp/f0-v15-peer-log-review-20261014-a/archive`，新增review/evidence。未重新跑pytest或helper，没有long E2E、训练、release/migration。

5. **Evidence / remaining risks**

Coder raw **2 passed in 0.14s**（exit0），stdout SHA `0a97b591ab7ce9644866a9aea8e19c37ff6647585418c546d0481e0cb92bfbc7`；manifest SHA `e1969d4fe7adb76cedce6d9d045d1bc20442ec049791d719da9c97dc160042e5`，单一payload逐项核验。Focused覆盖synthetic hardlink alias拒绝与clonefreshening，以及原wrapper compile/order静态测试；它不运行target integration或真实supervisor→child→runner。

独立读回unit artifact：source device/inode `66309/19014130`，old clone `66309/19014131`，fresh clone `66309/19014134`、size0；source原bytes与archivedbytes SHA `3f84ba80d702f0a62b077d5b9a50026c637b7a8a53c0f8ac5708cd776955c60b`一致。原bytes、metadata与当前test source已外置保留，避免pytest retention清理损失。

尚待qualification证明clean log下真实peer身份/锁/nonce/startticks、heartbeat连续性及两个deferral边界。旧attempt3拒绝在读取malformed第2行时已返回false，没有验证后续identity，因此本修复不能预判下次E2E结果。旧日志95行与inner archive97行均有结尾newline；直接原因是非JSON第2/3/7/8/25行，不是truncated-tail。

6. **Whether experiments are safe to run**

**该最窄fixture修复PASS，允许按既定单次/独立clone/freshscheduler与basetemp/外置archive/最多5项目Python边界继续非训练qualification。完整资格未闭合；B6 OPEN/FAIL，科学/生产RUN、principal release仍NO-GO。**

Evidence `experiments/reports/stage_F0_v15_peer_log_fixture_remediation_review_v1_evidence.json`，SHA `5632cb06da765160e5c2266cca8dadc8eab07737b381de2f9a8235132faa284b`。
