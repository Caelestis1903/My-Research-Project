# F0 v1.3 独立实施复审 v2

日期：2026-10-13（Asia/Shanghai）。Reviewer：独立 REVIEW agent。

## 1. Verdict：FAIL

当前交付修正版的 MAJOR-01、02、03、05（含 rename/fsync retry durability 子项）、06 已关闭；MAJOR-04、07 仍有阻塞缺口。不得开始生产行政迁移、解除 hold 或运行 search/formal/test/recovery。此结论只针对下面冻结的交付快照；不评价尚未交付的后续修复。

本轮仅阅读源码、测试、设计、报告和原始工件，使用 Perl/SHA 做独立文件内容哈希与集合核验。没有执行 Python、测试、训练、信号、迁移、status 写入或 gate/hold 修改。唯一新增文件为本报告；未作源码修正。

## 2. 实际冻结身份

以下均从实际文件独立重算，匹配本轮交付：

| 对象 | SHA-256 |
|---|---|
| DESIGN `stage_F0_execution_isolation_design.md` | `163f2fbc181712e0186cdb651f8738bb72a27f7a65edf92e4e554e677aac62e7` |
| CODER report `stage_F0_execution_isolation_implementation_report.md` | `aa89f6fe9e19b0ed74b4a43f5b57f8809fb0f9ce993684fc427dce6817160cfc` |
| Pending execution gate | `9e898b9cdd5fb32072d57ccaf71dc946df1b21a1a715bffbdee418b0e04fa9fa` |
| 16 control files，全部逐文件核验 | `7968359d5c41e4be2d67e26eee556dfc2e9ecbfeefa35b984898cd56e7b66fda` |
| 28 runtime files，全部逐文件核验 | `edf85d7bee7e3b7396b191a9cc896007a35c454734c30f7b35920ab6c54dc32f` |
| 独立来源 H49 | `60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e` |
| 原 52 completed key 集合 | `3d1bb138c30d7dd0ae07d7d3e5730e29d27248c6ecc12ee72afc8a2df6fa7a7a` |
| 原 52 completed trees，573 文件全内容扫描 | `94039db9a66bfe20e7a4bb2d96b9249c145aa2fcf23fb53ca71bc225ffc718ae` |
| 当前代码定义的 incident inventory，26 文件 | `b1c9f95d398519d2f1dad48b14a96125bf181bd4510210c40875be171f9de8ed` |
| 生产 active hold，原字节未变 | `e9461efcc99e49bef4b813a3f3cb1c67720bdcd172470df2a73a9e7caca400b9` |

H49 的独立推导是原 hold top-level 45 项 + 原 v3 两项 canonical claim = H47，再加原 v4 两项 `7fd2006bdc8f1817a6f7`、`1f04c7fb623d18f940ac`，共 49；不是从新 caller/gate 提供的集合反向定义。原 52 completed key 集合与源码固定集合摘要相符。

## 3. 按旧缺陷与设计要求核验

| 项目 | 结论与实际证据 |
|---|---|
| MAJOR-01 动态队列与新增完成 | **关闭。** `stage_F0_migration.py:245` 固定原 52 key，`validate_h49_runtime:883` 要求原 52 COMPLETE、H49 BLOCKED，其余 fresh 状态可推进；不把迁移计数永久当 live 计数。`test_stage_F0_migration.py:339` 检查两项并发 claim 与第 53 COMPLETE 后仍保留原资产摘要。 |
| MAJOR-02 active/released hold 矛盾 | **关闭。** `stage_F0_worker_control.py:254` 使用 `require_active_hold=False`；runtime 分 active/released 分支，released 必须有 release 图，claim 仍执行 hold/release 核验。active 状态可以核验但不能 claim；当前 PENDING gate 不能 launch。 |
| MAJOR-03 原 hold raw 保存 | **关闭。** `_make_prepared` 保存 raw base64、SHA 和完整 projection；`stage_F0_migration.py:870` 从 raw 重新校验原 SHA/完整 projection，COMMITTED 引用原来源。 |
| MAJOR-04 全部事故资产不可变 | **未关闭，见第 4 节。** 两项 attempt-00 完整树、SIGSEGV failure、两边 stderr/phase、peer receipt 和存在的 stdout 已覆盖，但遗漏 peer durable failure audit。 |
| MAJOR-05 full status / durability | **关闭。** 原状态 raw/full projection/canonical fullhash 校验、deterministic migrated projection、COMMITTED 的 full projection 与实际 SHA 均比对。`stage_F0_migration.py:1281` retry 先对已 rename 的 status 文件及父目录 fsync 再补 durable event；`:1328` hold retry 同样重建持久性。测试 `:219`、`:261` 检查 fsync 与 event/COMMITTED 的实际调用顺序，status canonical/fullhash drift 负例见 `:517`。 |
| MAJOR-06 correction 完整来源图 | **关闭。** runtime 重建完整 expected correction 并比对（仅排除生成时间）；原 v4 raw/hash、两字段 original/verified/source path/hash/value、原 peer status/fullhash 全部参与。`:544` 测试各 provenance edge 被替换，即使 caller 提供更新后的 correction SHA 也拒绝。 |
| MAJOR-07 完整 gate/release 引用图 | **部分关闭，仍阻塞。** 非空 mandatory hashes、固定来源路径、queue/design/implementation/correction/migration/assets/H49/scope 已验证；release reviewer 与当前 gate reviewer 仍未交叉绑定，见第 4 节。 |
| 无 checkpoint recovery / 资产写入 | **静态通过，受上述缺口限制。** 行政迁移只写两项 queue status、hold、correction/ledger；原 attempt checkpoint/manifest/metrics/receipt 不加载为恢复训练且不重写；original 52 的生产 validator 全内容哈希。 |
| v1.2 控制保护 | **未见回归。** supervisor、child、F0Runner 保持之前 PASS 的 SHA：`2689e7cc…cbab8`、`4343e928…86e5`、`5ad73c89…2599`。实际阅读 pidfd 验证/单次 INT-TERM-HUP 转发、30s durable-before-kill、相同 PGID 与 group STOP/CONT、PDEATHSIG race 检查、锁内 status/manifest/receipt 和重复信号保护路径；本轮没有运行信号 fixture。 |
| 进程/GPU计槽 | **未见回归。** 16-file inventory 包含两个 wrapper、supervisor、child、scheduler/queue 与 focused tests；双 slot reservation、共享最多 5 Python、最多 2 GPU worker 的控制路径保持原实现。本轮没有新增 Python 进程。 |
| PENDING / 未授权 | **通过。** reviewer_verdict=PENDING，launch/formal-or-test/recovery/automatic-retry 均 false。必要 correction/migration/review hash 尚未完整冻结时 gate 源码拒绝；没有解除 hold 的副作用。 |

## 4. 当前阻塞缺陷与准确修复要求

### MAJOR-04：peer 的 durable failure audit 仍未保护

DESIGN §14.4 明确要求 native stderr/phase/failure 日志保持字节不变；§14.7.1 要求核验 durable supervisor failure 与原 hold 事件。`stage_F0_migration.py:264` 的 `_incident_asset_inventory` 在 `:276` 仅为 SIGSEGV worker 添加 `failure_audit_path`，peer 仅添加 stderr/phase/receipt。当前原 v4 peer record 未带 failure 路径不能成为遗漏依据，因为冻结的原 hold 已在 `global_hold.json:522` 记录 peer failure 文件与 nonce。

实际遗漏文件：

`experiments/stage_F0_outputs/runs/worker_logs/worker_0_failure_1791819201_c87c8e65bc7e4b73bed880465ff2dffb.json`

实际 SHA-256：`ce1f5d4e8d68f2a304139c790982b34cef58ac971c590a852ae7bcf95bcefb9d`。

静态反例：只改变该文件字节，保留已覆盖的 26 文件和原 hold/v4 不变，当前 inventory 摘要完全不变；migration/runtime 没有读取此文件的保护核验，故可以接受事故 failure 日志漂移。本轮未实施该修改或执行反例。

修复：从冻结的原 hold 历史事件/nonce/identity 提取并验证 peer durable failure 原始路径与 SHA，将其纳入完整 incident inventory、PREPARED/COMMITTED、gate/release/runtime 全链条，更新基线与依赖摘要；不得从当前已可复用的 GPU owner 文件推断历史身份。补独立 fixture 对 peer failure 内容改变、缺失及错误来源绑定的拒绝测试，确保拒绝发生于任何 status/hold 写入之前。保留旧日志原字节。

### MAJOR-07：release 可引用旧独立 review 而 gate 引用当前 review

DESIGN §14.9 要求无循环的实际引用图：当前设计/实现/证据 → 独立 review → 绑定 review 的 gate → prelaunch → release；§14.10 要求 release 绑定设计/实现/review。旧 v1.2 PASS 明确不能批准 v1.3 实施或 release。

`stage_F0_migration.py:777` 只检查 release reviewer 文件位于 reports、存在并与 release 自报 SHA 相符；没有要求该路径/hash 与当前 execution gate/protection 的 reviewer identity 相等。`stage_F0_worker_control.py:246` 只检查 gate 与 protection 的 reviewer identity 相等；后续 runtime 对比没有返回/比较 release 的 reviewer identity。

静态反例：在其他 v1.3 release 字段均正确的 fixture 中，同时将 release 的 reviewer 路径与 SHA 替换为真实旧 v1.2 PASS 文件 `experiments/reports/stage_F0_signal_cleanup_review_v3.md` / `a5fe1f766ae969bc9d307e135d73afe1d39cefcb2a6cbf94a1e8133efcc42b96`，保留当前 gate/protection 新 review 绑定。当前两个局部 validator 都可以通过，整体 graph 则由不同 review 批准。现有 release 负例只替换错误路径或 SHA，没有覆盖“旧存在报告 + 匹配 SHA”成对替换。本轮未构造/运行该 fixture。

修复：让 released-hold、claim、gate 与 prelaunch 的 validation 共同校验 release reviewer identity 等于冻结的当前 gate/protection reviewer identity，并保持 hash DAG 无循环；不应通过递归调用 gate/runtime validator 引入验证循环。补合法 release 正例以及真实旧 review 路径/SHA 成对替换负例；确保即使其他全部 design/implementation/source/queue/assets/H49 字段正确也拒绝。

## 5. 测试证据、生产状态与边界

已独立阅读 focused tests 及调用边界；Coder 声称使用项目 env 与 reserved wrapper 合并运行 **101 passed**。该结果是 Coder 证据，本 Reviewer 没有运行或独立复现。fixture 确实使用独立复制的 mutable incident/control 输入并检查不同 inode；full status、provenance edge、release field、第二 status/hold rename-fsync retry、动态 claim 与新增完成测试均可见。

52 completed 树 fixture 保留真实 read-only 来源，缓存全内容 inventory 并以 inode/size/mtime 检查，而没有复制/修改 4.72 GB 树。这可接受为本次只读审查的证据限界：本 Reviewer 独立重算了全部 573 文件实际 SHA 与 aggregate，生产 validator 确实全内容扫描；但不能称该 fixture 已证明 completed 树任意 mutation/addition 的动态拒绝。不能以该测试证据覆盖上述 peer failure omission。

本轮独立核对生产实际状态：52 VALIDATION_COMPLETE、47 RECOVERY_BLOCKED、187 UNCLAIMED、1 RUNNING、1 INTERRUPTED；两项新增 H49 仍是原 RUNNING / INTERRUPTED。active hold SHA 未变；correction 与 v5 migration ledger 均不存在。当前仍是生产未迁移状态，不存在本轮 GO 批准。

剩余非阻塞风险：此前真实 native SIGSEGV 根因未解决；本 review 只审控制、隔离、保护与授权。manual first-epoch helper 没有进入 16/28 清单，后续如再启用该独立入口，应另冻结/审查或明确停用。数学、tensor、梯度和研究 baseline 本轮没有改动，不重新宣称已验证其研究效果。

本轮无 CRITICAL 或新增算法偏差；两个残留 MAJOR 阻塞须返 Coder。修复后需以新的 design/report/gate/source/asset hashes 重新独立复审。只有取得 PASS 或阻塞全部关闭的 PASS WITH FIXES 后，Runner 才能执行限定行政迁移；此后仍需持久迁移证据、当前 review/gate/prelaunch 和主控 identity-bound fresh-only release，绝不隐含授权 held-key recovery、formal 或 test。
