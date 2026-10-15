# F0 v1.3 independent implementation review v3

日期：2026-10-13（Asia/Shanghai）。独立 REVIEW；审查当前修正版实际源码、测试、设计、Coder 报告与生产工件。

## 1. Verdict：FAIL

此前 MAJOR-01–07（含 MAJOR-05 durability 子项）已全部关闭，尤其 peer failure inventory 和 release/current-review 交叉绑定已修复。最终完整数据流检查发现另一真实错误：显式 correction SHA 冻结 API 引用未定义变量，记为 **MAJOR-08（本轮新编号，与先前“七项加 durability 子项”的口头八项计数无关）**。修正并经定向复审前不得执行生产迁移或实验。

本轮未运行 Python、测试、训练、信号、迁移或生产 status 写入；未修改源代码、gate、hold 或实验资产。仅用只读 shell/Perl 独立哈希与集合核验，唯一新增文件是本报告。

## 2. 独立重算的冻结身份

| 对象 | SHA-256 |
|---|---|
| DESIGN | `163f2fbc181712e0186cdb651f8738bb72a27f7a65edf92e4e554e677aac62e7` |
| Coder implementation report | `72c5928b00174e326c9cc5493eb54954ba1ec8d1a69d336e4ad63572e687c968` |
| Pending execution gate | `75e3b2458f28cbf1ab30faf14ef5e4217985c7198096d68e9c1b1d0f8697e8d0` |
| 16 control files，全部实际逐文件 hash 核验 | `b19a81254a390585a4f44c38bf0c7eb26b6786a0f836ef3de54b296ca0213d53` |
| 28 runtime files，全部实际逐文件 hash 核验 | `edf85d7bee7e3b7396b191a9cc896007a35c454734c30f7b35920ab6c54dc32f` |
| H49 独立来源集合 | `60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e` |
| 原 52 completed key 集合 | `3d1bb138c30d7dd0ae07d7d3e5730e29d27248c6ecc12ee72afc8a2df6fa7a7a` |
| 原 completed trees，573 文件全内容扫描 | `94039db9a66bfe20e7a4bb2d96b9249c145aa2fcf23fb53ca71bc225ffc718ae` |
| Incident inventory，27 文件全内容扫描 | `10fc3b31d3f462a89f9204db14f7e97a7bce2ad733279bf2724d3311e3e6e15e` |
| 未变生产 active hold | `e9461efcc99e49bef4b813a3f3cb1c67720bdcd172470df2a73a9e7caca400b9` |
| 未变 prior search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |

H49 由原 hold top-level 45 项、原 v3 两项 canonical claim 组成 H47，再加入原 v4 的 `7fd2006bdc8f1817a6f7` 与 `1f04c7fb623d18f940ac`，独立推导为精确 49 项，与 gate/source 固定集合相符。没有用新 gate/caller 列表重新定义来源。

## 3. Requirement-by-requirement verification

| 旧项/要求 | 本轮结论与实际证据 |
|---|---|
| MAJOR-01 动态 queue 与 >52 completion | **关闭、无回归。** `_asset_inventory` 固定原 52 keys；`validate_h49_runtime` 固定其 COMPLETE 与 H49 BLOCKED，但允许其余 fresh 状态推进。测试真实调用两线程 serialized claim 后核验，再添加第 53 COMPLETE 后保留原资产摘要，未把行政迁移计数当永久 live 计数。 |
| MAJOR-02 active/released GO 模型 | **关闭、无回归。** gate 调 runtime 使用 `require_active_hold=False`；active hold 可验证但 claim 拒绝，inactive hold 须通过完整 release graph。没有为通过 gate 而忽略 hold。 |
| MAJOR-03 原 hold raw 保存 | **关闭、无回归。** PREPARED 保存原 raw base64、实际 SHA 和完整 projection；runtime 与 retry 从保存 raw 重算冻结 SHA，hold extension 只允许完整原 projection 加单个指定事件/字段。 |
| MAJOR-04 所有事故资产不可变 | **关闭。** `stage_F0_migration.py:294` 从 hold 的唯一 WORKER_FAILURE_HOLD/canonical ID 取得 peer 原 failure 来源，并核对 worker、nonce、生产树内路径与实际 SHA。该文件进入 PREPARED/COMMITTED、runtime、gate/release inventory；不读取可复用 GPU owner 推断历史身份。两项 attempt-00 完整树、manifest/checkpoint/metrics、原日志与 peer receipt 均覆盖；SIGSEGV receipt 仍必须不存在。测试 `test_stage_F0_migration.py:529` 明确断言 peer failure 在清单，`:539` 修改独立副本内容后要求拒绝并保持 status/hold SHA 不变。 |
| MAJOR-05 status 全 projection / retry 持久性 | **关闭、无回归。** raw/fullhash/canonical projection 校验、确定性的 migrated status 和 COMMITTED full projection/SHA 实盘比较保持；已 rename 的第二 status 及 hold retry 在补 durable receipt/COMMITTED 前 fsync 文件和父目录。测试通过 tracing `/proc/self/fd` 记录实际 fsync 调用顺序；fullhash/canonical drift 负例仍保留。 |
| MAJOR-06 correction provenance | **关闭、无回归。** 从冻结 v4、实际 peer receipt/原 peer status 重建完整 correction，比较原 audit、错误原值、核验值、source paths/hash/field/value 等全部边（仅生成时间除外）。correction edge 被替换且 caller 给更新 SHA 的负例仍拒绝。 |
| MAJOR-07 release/current review 图 | **关闭。** `_frozen_review_identity`（`:756`）读当前 gate，要求 PASS/PASS WITH FIXES、非空合法 reviewer identity、与 protection 相同且实际文件 hash 相符；`_release_graph:839` 要求 release reviewer path/SHA 与该身份逐项相同。仅读取 gate，不递归调用 validator，不形成验证循环。`:475` 测试将 release reviewer 路径/SHA 成对替换为真实旧 v1.2 PASS 工件而保留当前 gate，必须拒绝。其他非空 source/design/implementation/queue/migration/correction/asset/H49/scope 绑定仍保持。 |
| 行政迁移序列化、幂等与失败关闭 | **主体符合，显式 freeze 分支受 MAJOR-08 阻塞。** execution_state、worker、GPU、两项 canonical locks 下重新读来源；PREPARED 持久后才 status 写入，再 hold extension、全实盘核验与 COMMITTED。partial status/hold rename 重试保持 active hold；完成后幂等返回同一 ledger。没有 checkpoint recovery/new attempt/receipt 重写。 |
| Caller/gate/claim/prelaunch H49 | **通过静态审查。** 固定集合与 actual statuses 一致核验；gate/protection、caller、migration、release 都不能以同 count 替换 key；claim 在 execution_state lock 下检查 hold、精确 H49 和 fresh-only eligibility。supervisor 与 runner 在数据/claim 前校验 gate。 |
| v1.2 pidfd/receipt 控制 | **无回归。** supervisor、child、F0Runner、wrapper 和 scheduler/queue 对应冻结 SHA 保持既有控制实施；核对 peer supervisor pidfd 唯一 TERM 路径、operator INT/TERM/HUP 身份验证、单次转发、durable-before-kill、same PGID STOP/CONT、PDEATHSIG、重复信号和锁内 status/manifest/receipt transaction。未在本轮重跑信号 fixtures。 |
| Python ≤5 / GPU workers ≤2 | **无回归。** 双 slot reservation/继承、共享 scheduler accounting、worker/GPU locks 保留；focused fixture 使用自己的 slot。本轮没有 Python 进程。 |
| 算法、数学、tensor/梯度、baseline | 行政控制修复没有引入研究算法改变；28 runtime digest 与上轮一致。此 review 不分析 AUC/loss 或重新推断研究结论。 |

Peer failure 的实际路径为 `experiments/stage_F0_outputs/runs/worker_logs/worker_0_failure_1791819201_c87c8e65bc7e4b73bed880465ff2dffb.json`，实际 SHA `ce1f5d4e8d68f2a304139c790982b34cef58ac971c590a852ae7bcf95bcefb9d`，本轮 27 文件 inventory 已确实包含它。

## 4. 新阻塞：MAJOR-08 显式 correction freeze 引用未定义变量

实际位置：`experiments/stage_F0_migration.py:1231`。

```python
corr_path, corr_sha = ensure_provenance_correction(runs_root)
if correction_sha256 is not None and correction_sha != correction_sha256:
    raise PermissionError("provenance correction hash differs from caller freeze")
```

本函数没有定义 `correction_sha`；实际局部变量是 `corr_sha`。其他函数的同名参数不会进入该函数作用域，也没有 module-level 定义。因此 `apply_migration(runs_root=fixture, correction_sha256=<实际冻结 SHA>)` 无论 SHA 相符或不符都会在该行抛 NameError，而不是执行 freeze 比较。若 correction 尚不存在，该函数之前已创建 correction 工件；尚未写 status/hold。默认 CLI 没有传此可选参数，故不触发错误，但不能据此宣称公开 freeze API 正确。

这是静态可确定反例，本轮未调用该 API 或创建 fixture。现有 101 tests 中的 `correction_sha256=` 调用属于 `validate_h49_runtime`，没有覆盖 `apply_migration` 的显式 caller freeze 分支。

准确修复：使用 `corr_sha` 进行比较；补独立临时 fixture 的相符 freeze 正例和错误 freeze 负例，后者应为明确 PermissionError 且 status/hold 未变，同时核对 API 是否创建 correction 的既有语义。更新 implementation report、control inventory、gate 及 test SHA，再做定向独立复审。无需改变设计、算法、研究假设或原始实验资产。

## 5. Evidence limits、生产状态与 RUN 边界

Coder 报告 reserved wrapper + phase0-3-env 的 focused suite **101 passed**；Reviewer 独立阅读了测试内容与来源绑定，但未运行或独立复现该结果。本轮新发现明确展示未测试分支的局限。缺失/错误路径的 peer failure guard 已存在；本次新增动态 mutation 测试只直接覆盖其内容改变，不声称运行覆盖每一 guard。

完成资产 fixture 没有复制/修改 4.72 GB 的原 52 树，而是初始全内容 hash 加 inode/size/mtime 守护并返回 cached inventory；这不能代表完成树任意 mutation/addition 的动态负例已被测试。证据限界可接受用于本轮 read-only 审查：Reviewer 独立全内容扫描 573 文件并核对固定 key 集合，生产 validator 实际枚举与 hash 全树，不只检查 inode/mtime。没有对研究 checkpoint 做反序列化或模型评估。

生产仍为 **52 VALIDATION_COMPLETE / 47 RECOVERY_BLOCKED / 187 UNCLAIMED / 1 RUNNING / 1 INTERRUPTED**。两项新增 H49 尚为原 RUNNING/INTERRUPTED；hold `active=true` 且原 SHA 未变；correction 与 v5 migration ledger 均不存在。gate reviewer_verdict=PENDING，launch/formal-or-test/recovery/automatic-retry 均 false。实际源码在必要 nonempty evidence/hash 缺失时 fail closed；本轮没有再次运行 production preflight 或检查实时锁空闲。

无 CRITICAL；本轮唯一未关闭 MAJOR 是上面的新 API 错误。既有 native SIGSEGV 原因仍未知；隔离/授权控制通过不证明 native 稳定性。manual first-epoch 独立入口不在 16/28 inventory 内，后续若使用须单独冻结/审查或停用，本报告不批准该入口。

修复并经 REVIEW PASS 后，Runner 才能进行限定行政迁移；实施 PASS 本身仍不是启动 search 的授权。迁移完成须核验 exact49 BLOCKED、raw/source/status/assets 不变与持久 COMMITTED，冻结 correction/migration/current-review hashes，再经当前 gate、prelaunch 和主控 identity-bound fresh-only release。held-key retry/recovery/formal/test 全部保持未授权。
