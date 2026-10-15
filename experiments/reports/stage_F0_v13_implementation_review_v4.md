# F0 v1.3 independent implementation review v4

日期：2026-10-13（Asia/Shanghai）。独立 REVIEW，针对 MAJOR-08 修复后的实际冻结快照。

## 1. Verdict：PASS

MAJOR-01–08 全部关闭，其中 MAJOR-05 包含此前独立记录的 status/hold rename-fsync retry durability 子项。未发现新的 CRITICAL、MAJOR 或需要修改的 MINOR。此 PASS 批准实施进入受控行政迁移 RUN 阶段；**不直接授权 search 启动、hold release、held-key recovery/retry、formal 或 test**。

本轮只读检查实际源码、测试、设计、Coder 报告及生产工件，用 shell/Perl 独立计算文件内容与集合哈希。没有执行 Python、pytest、迁移、训练、信号、status/hold/gate 写入，也未改源码。唯一新增工件为本报告。

## 2. 独立重算的当前身份

| 对象 | SHA-256 |
|---|---|
| DESIGN | `163f2fbc181712e0186cdb651f8738bb72a27f7a65edf92e4e554e677aac62e7` |
| Coder implementation report | `f26944611f85f9380081968609105de83d9683fdcfe80f5a9173712ea0ae8aeb` |
| Pending execution gate | `3a53b490cf860a901b3c1e4a7515b6dda7b9e0b201bedc4c05a547acde517ad0` |
| Migration module | `02ee3b12203ddfa9184e45a2d8109a13ac48141dd750c91c029a305ad99f1d3d` |
| Migration tests | `54e65a8dbbe80a8193e054b00bcf65f1a37ba661252df489139dc6ee615ca51d` |
| 16 control files，逐文件核验 | `e98c6ef9d23974420bebcb053f3a791b5efddc09b055fb3bdc6867b2fd572fa6` |
| 28 runtime files，逐文件核验 | `edf85d7bee7e3b7396b191a9cc896007a35c454734c30f7b35920ab6c54dc32f` |
| 独立 H49 集合 | `60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e` |
| 原 52 completed key 集合 | `3d1bb138c30d7dd0ae07d7d3e5730e29d27248c6ecc12ee72afc8a2df6fa7a7a` |
| 原 52 completed trees，573 文件全内容扫描 | `94039db9a66bfe20e7a4bb2d96b9249c145aa2fcf23fb53ca71bc225ffc718ae` |
| 27 incident 文件全内容扫描 | `10fc3b31d3f462a89f9204db14f7e97a7bce2ad733279bf2724d3311e3e6e15e` |
| 原生产 active hold，字节未变 | `e9461efcc99e49bef4b813a3f3cb1c67720bdcd172470df2a73a9e7caca400b9` |
| 原 search gate，字节未变 | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |

H49 仍由原 hold 的 45 项与原 v3 两项 canonical claim 独立构成 H47，再加入原 v4 的两项事故 key，实际推导为精确 49 项并与源码/gate 相符。没有信任新 caller/gate 自报列表。

## 3. 按要求与旧缺陷逐项核验

| 项目 | 结论与证据 |
|---|---|
| MAJOR-01 动态队列与 >52 完成 | **关闭。** 固定原 52 completed keys/asset inventory 与 H49 actual BLOCKED；其余 fresh 状态可继续推进。测试两项 serialized claim 和第 53 COMPLETE 后原 asset hash 不变，永久固定的是原资产集合而非 live 完成计数。 |
| MAJOR-02 active/released 状态 | **关闭。** gate 可验证 active 或 released runtime；active hold 仍阻止 claim，inactive hold 必须具备完整 identity-bound release。没有 active-only gate 与 released-only claim 的矛盾。 |
| MAJOR-03 原 hold raw | **关闭。** PREPARED 持久保存 raw base64、实际 SHA、完整 projection；runtime/retry 从 raw 核验冻结来源，hold extension 只允许指定单个事件/字段追加。 |
| MAJOR-04 27 incident 资产 | **关闭。** 两项 attempt-00 完整树、manifest/checkpoint/metrics 与 native logs/receipt 均 hash；peer durable failure 从原 hold 的唯一 canonical-ID/worker/nonce/path/SHA 事件核验并纳入所有 inventory 绑定，SIGSEGV receipt 保持不存在。peer failure 内容漂移测试要求拒绝且 status/hold SHA 未变。 |
| MAJOR-05 full status 与 durability | **关闭。** 原 full projection/canonical fullhash 与 deterministic migrated projection 校验；COMMITTED full projection/SHA 与实盘一致。status/hold rename 后中止的 retry 对现存文件及父目录 fsync 再补 durable event/COMMITTED，测试记录实际 fd/fsync 调用顺序；原 drift 负例保留。 |
| MAJOR-06 correction 来源图 | **关闭。** 原 v4 raw/hash、错误原值、核验值、来源 paths/hash/field/value 和原 peer status/fullhash 全部重建比对，caller 提供重新 hash 的伪造 correction 仍拒绝；更正 append-only，不覆盖原 audit。 |
| MAJOR-07 current review 图 | **关闭。** `_frozen_review_identity` 要求 gate 与 protection 共同冻结当前实际 PASS review path/SHA；release 必须逐项相同。读取 gate 而不递归 validator，避免引用循环。真实旧 v1.2 review 路径+正确 SHA 成对替换负例仍拒绝。非空 design/implementation/source/queue/correction/migration/asset/H49/scope 绑定保持。 |
| MAJOR-08 caller correction freeze | **关闭。** `stage_F0_migration.py:1231` 现在实际比较已定义 `corr_sha`。`test_stage_F0_migration.py:355` 两个 parameterized 实例直接调用 `apply_migration`，相符 SHA 形成 COMMITTED 并绑定该 SHA；不符 SHA 得明确 PermissionError、ledger 不存在、两项 status SHA 与 hold raw 不变、既有 correction SHA 未变。 |
| PREPARED → status → hold → COMMITTED | **符合。** execution_state/worker/GPU/两项 canonical locks 序列化，锁内重新读取来源；PREPARED durable 后才改两项 status，再更新 active hold，完整实盘核验后 COMMITTED。fault/retry 幂等，不自动恢复，不新建 attempt，不改 checkpoint/manifest/receipt/metrics。 |
| Gate/caller/claim/release/prelaunch | **符合。** 精确独立 H49、全部 actual BLOCKED 与完整来源图核验；serialized claim 执行 hold+exact-set+fresh-only eligibility；supervisor 和 runner 在数据/claim 前检查当前 gate。 |
| v1.2 pidfd/receipt/进程控制 | **未回归。** 相同冻结 supervisor/child/F0Runner/wrapper/scheduler 实现保留已核验的 pidfd INT/TERM/HUP、peer supervisor 单次 TERM 转发、durable-before-kill、相同 PGID group STOP/CONT、PDEATHSIG race defense、重复信号与锁内 status/manifest/receipt commit、共享最多 5 Python 与最多 2 GPU worker。此轮未再运行信号 fixture。 |
| 科学行为与 baseline | 本轮只改行政控制变量名和新增两个 API 分支测试；28 runtime hash 未变，没有改算法、数学、tensor/mask/梯度或研究 baseline，没有解读研究结果。 |

无回归证据不仅来自 Coder 描述：本 Reviewer 在内存中把修正的 `corr_sha` 单行反向替换为上一快照错误文本，所得 SHA 恰为上一已审 module 的 `033138d02b7ea83eea0dd8c7bc93f276c11b49833ed345e3b30b583dcebc3e3d`；移除唯一新增 matching/mismatched 测试 block 后 SHA 恰为上一已审 tests 的 `a5853427ed3e53c59077e3ebf54f687fc6ec657807e48a1654860047115f9cca`。这证明除本次修复/两个测试外，MAJOR-01–07 的已审源码和测试字节未变化。没有对磁盘执行反向替换。

## 4. Corrections made、测试证据与限界

Reviewer 未修改实现。以上关闭项是对 Coder 实际修正版的独立核验，不是重新设计实验。

Coder 报告 reserved wrapper + 项目 env 合并 focused suite **103 passed**，比上一 101 项增加两个 freeze 实例；Reviewer 阅读测试实现并独立确认代码身份，但没有运行或独立复现结果。匹配/不匹配 tests 使用已创建且独立复制来源的 correction 工件；代码仍先 ensure correction 后比较 caller freeze。首次错误 freeze 可能创建合法 append-only correction，但在比较拒绝前没有进入迁移 ledger/status/hold 提交，这不是部分迁移 commit；本报告不声称 mismatch 必然零新增辅助工件。

完成树 fixture 以真实只读来源初始全内容 hash、inode/size/mtime 守护和 cached inventory 避免复制 4.72 GB；不能代表任意 completed-tree mutation/addition 已被 fixture 动态测试。本轮独立再次全内容 hash 573 文件、固定 key 集合与 27 incident 文件，生产 validator 确实逐树枚举并全内容 hash，该 read-only 限界可接受。没有反序列化研究 checkpoint 或读取 test 数据进行模型评估。

既有 native SIGSEGV 根因仍未知，控制 PASS 不代表 native 稳定性已解决。manual first-epoch 独立入口不在 16/28 inventory 内；本报告不批准使用它，若使用须单独冻结/审查或明确停用。

## 5. 生产事实与允许的下一阶段

独立实盘核验生产仍为 **52 VALIDATION_COMPLETE / 47 RECOVERY_BLOCKED / 187 UNCLAIMED / 1 RUNNING / 1 INTERRUPTED**。新增两项 H49 未迁移；hold active 且原 raw SHA 未变；correction 与 v5 migration ledger 均不存在。gate reviewer_verdict=PENDING，launch/formal-or-test/recovery/automatic-retry 均 false。源码对必要 evidence/hash 缺失 fail closed。本轮未运行实时 preflight 或检查当前锁空闲，不能把 Coder 过去的“locks free”当下一次实时结论。

**实施 REVIEW 门通过，无需再次修源码。** 主控可将本 PASS 交给 Runner，按冻结实现只做行政迁移：先重新确认无进程及 worker/GPU/canonical 锁空闲、所有来源/资产摘要精确，再形成 correction、PREPARED、两项 BLOCKED、active H49 hold、COMMITTED；检查原 52 keys/573 文件与 27 incident 文件不变、全部 H49 实际 BLOCKED。

之后才绑定本报告与实际 correction/migration/current-hold hashes，完成当前 gate、prelaunch、主控显式 identity-bound fresh-only release 的无循环图。search 只能 claim fresh UNCLAIMED keys；任何 held-key retry/recovery、formal 或 test 仍不在本 PASS 授权范围内。以上是尚待执行的外部 RUN 门，不是未修复的实施缺陷。
