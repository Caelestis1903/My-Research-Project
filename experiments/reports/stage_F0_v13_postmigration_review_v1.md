# F0 v1.3 production postmigration independent audit v1

审计日期：2026-10-13（Asia/Shanghai）；只读观测约 2026-10-12 17:21 UTC。这是 release 前行政迁移 GO/NO-GO，不是新的代码或研究结果 review。

## Verdict：GO（仅行政迁移一致性）；search launch 仍 NO-GO

实际迁移与冻结设计/实施/来源一致；没有发现行政事务、状态保护或被保护资产漂移。H49 全部实际 RECOVERY_BLOCKED，原 52 COMPLETE 与原资产不变，hold active，gate 已绑定当前实施 PASS review，但 launch/formal/test/recovery/retry 全 false。此 GO 不能代替主控 release 或 launch 授权，也不允许任何 held-key 恢复。

本轮没有 Python、测试、训练、信号、迁移、生产写入或 release。独立使用 Perl/SHA、/proc 和只读 nvidia-smi 检查。唯一新增工件为本审计报告；源码、行政对象与实验资产均未修改。

## 实际冻结工件哈希

全部从实际字节独立重算，与交付相符：

| 工件 | SHA-256 |
|---|---|
| Corrected provenance | `c70c1068c8703db5dbad88dbd1e508cfac43bfebab3927682d0a4047b67ee7b7` |
| COMMITTED migration ledger | `2608edd595fcfdc46791eeb73b3aa42b1c82d75b265c7d3abb16146884284f08` |
| Current active H49 hold | `a1da4f3d0aa20c7c867b9b01b7c8c509db21408c7cedfed5b924040dfba48b66` |
| Current execution gate | `9355a491c9c076d7279e04a185f194dc225158229dee1608ff93ff0b8aa4e13f` |
| Raw Runner postmigration report | `a3bbd3d0bf238462882a3a5b29e1afb8224639478837c8993ca68726cae76af3` |
| Bound implementation review v4 | `bd8937766fdae4ab84fdbd2fc9d0de2ca487c9ef62932914564be63232bc804f` |
| Current 16 control files | `e98c6ef9d23974420bebcb053f3a791b5efddc09b055fb3bdc6867b2fd572fa6` |
| Current 28 runtime files | `edf85d7bee7e3b7396b191a9cc896007a35c454734c30f7b35920ab6c54dc32f` |

设计 SHA `163f2fbc181712e0186cdb651f8738bb72a27f7a65edf92e4e554e677aac62e7`、实现报告 SHA `f26944611f85f9380081968609105de83d9683fdcfe80f5a9173712ea0ae8aeb` 与 gate 实际引用均核对；控制/运行源码仍为批准的 v4 快照。

## 逐项实际审计

| 要求 | 独立结果 |
|---|---|
| H49 独立来源与 exact set | 原 hold raw 的 45 项 + 原 v3 两项 claim 构成 H47，再加原 v4 两项事故 key，精确 49，无重复；PREPARED、COMMITTED、当前 hold、gate protection 逐项相同。集合 SHA `60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e`。实际 49 status 全 RECOVERY_BLOCKED。 |
| Queue 实际计数 | 固定 288 项实际为 52 VALIDATION_COMPLETE、49 RECOVERY_BLOCKED、187 UNCLAIMED，RUNNING=0、INTERRUPTED=0；原 52 keys 保持 COMPLETE。queue 文件 SHA `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23` 与 PREPARED/COMMITTED/gate 一致。 |
| Ledger 顺序与绑定 | 实际仅四条：PREPARED → 两项 STATUS_TRANSITION_DURABLE → COMMITTED。PREPARED 原 compact JSON 行的实际 SHA `8b6084d3edcf850232a7f3b91b91b38648eb2c2d935e59faec11bdb89f3779bd` 与 COMMITTED/hold event 一致；两条 transition 的 status SHA 与实盘/COMMITTED 一致，file/parent-directory-fsynced markers 均 true。 |
| 原 hold raw 保存 | PREPARED base64 解码后的原始字节 SHA 为 `e9461efcc99e49bef4b813a3f3cb1c67720bdcd172470df2a73a9e7caca400b9`，与原审查 baseline 一致，完整 projection 相符。当前 hold 完整对象等于原 projection 加允许字段更新；全部历史 events 逐项原样保留，仅追加一个指定 migration event，active=true。 |
| Full target status | 从 PREPARED 原 raw/projection 独立重建允许的 migrated 完整对象，与实际 status 和 COMMITTED status_projections_after 逐项相同。科学 canonical identity/fullhash、attempt、recovery 与 resume 选择不变；不是仅检查 status/count。 |
| 原完成资产 | 全内容读取原 52 trees 的 573 文件，逐项路径/size/SHA 与 PREPARED inventory 相同；aggregate=`94039db9a66bfe20e7a4bb2d96b9249c145aa2fcf23fb53ca71bc225ffc718ae`，与冻结 baseline 和 COMMITTED 相符。 |
| Incident 全资产 | 全内容读取 27 文件，包括两项 attempt-00 全树、checkpoint/manifest/metrics、native stdout/stderr/phase、peer receipt 与 hold-event-bound peer failure；逐项与 PREPARED inventory 相同。aggregate=`10fc3b31d3f462a89f9204db14f7e97a7bce2ad733279bf2724d3311e3e6e15e`，与原 baseline/COMMITTED 相符。SIGSEGV child receipt 仍不存在。 |
| Gate/review/scope | reviewer_verdict=PASS，top-level review 和 protection 的 path/SHA 相同且实际 review 文件 hash 相符。current hold/correction/migration/design/implementation 来源摘要与实际文件一致。launch/formal-or-test/recovery/automatic-retry 均 false；旧 global release 不构成本事故 release。 |
| F0 进程与生产锁 | /proc 实际扫描没有 F0 supervisor/child/runner/test/worker-shell 进程。按 device/inode 与 /proc/locks 交叉检查 75 个 F0 production lock 文件（包括 execution_state、两个 worker、gpu0/gpu1、canonical locks），匹配持锁为 0；默认 workspace scheduler 的 5 个 slot lock 同样持锁为 0。未尝试获取/重写锁。 |

目标 status 具体证据：

| Run | 当前 status SHA | 原 raw status SHA |
|---|---|---|
| `7fd2006bdc8f1817a6f7` | `c728f4d974ed153391663b4eca01eca10335e3d266b519e6184302694b7dffe8` | `74cbabc600e19e4e977c3b38d3068419c72c3be0b678cea5857a1867ec66dbbe` |
| `1f04c7fb623d18f940ac` | `93d6c5c35786aece94bf0e4f81e370ab49883d93958f0dfc2cc9c6a391cdd986` | `61921430b4975ba67720efd19fcb24a6ae4debb5f232f56d536e0e568ad41478` |

两项 attempts 目录都仅有 attempt-00，recovery_count=0、resume_checkpoint=null。原 checkpoint SHA 分别为 `2e669175eb2752ff27cc73202687167e98799d36e2495f25e76d4543f6b0c79f` 与 `8843afe2a382b47b7a920986ffbfc0cf996a73b0ab8822bfd138760b2517597c`；实际 manifest SHA 分别为 `e304b73005ff339b63b0b2ad70c4830b7df9550bce2acc2710ff250690ee0197` 与 `2e85d70d0d97b144fd26cba06be61f9ce3d86bb3879839adde9c866c6377f1d6`。这些原字节均被 unchanged 27-file baseline 及 PREPARED 逐项 inventory 证实，无新 attempt/receipt。

## 来源更正与资源限界

原 v4 保持原字节，SHA `a50162f3016841257e8c098cc0040c9ff2be4edb30073c4c2fb42f8ed416f839`；correction 的 original_audit.raw_value 与实际原 audit 完整相同，错误原字段原样记录。真实 peer receipt 独立重算为 `3a3b86cd54b0685ec09506a32706355a26c3465ed41d872c06e641d94bcbacf7`；其 gpu_lock_owner、canonical_lock_owner 和 PREPARED 原 peer status 的 UUID 均为 `2a59fb49-6bef-5320-74ab-ffb24d056908`，与 correction verified_value/各历史 source 一致。没有用当前可复用 GPU lock owner 重写历史身份。

“无进程”仅指 F0 执行进程。只读 /proc 与 nvidia-smi 发现另一用户 pzg 的 Qwen uvicorn 服务 PID 910078 使用 GPU0；观测 GPU0 used/free 为 4222/19914 MiB，GPU1 为 15/24121 MiB。因此不能宣称全机没有 Python 或两 GPU 完全无其他 context。该服务没有持 F0 锁，不是本次迁移残留；本审计不授权停止它。下一次 prelaunch 须重新核验实际资源及并发总量，不能复用本次瞬时读数。

本轮能证明完整保护 inventory 与行政 status/hold projection 的一致，不能宣称整个仓库所有未冻结文件绝无写入。fsync 的历史真实系统调用不可能由事后读取直接证明；durable markers、事务完整字节、批准实现及此前 failure-injection evidence 支持持久性判定。没有查看或分析研究指标的效果，没有测试模型或反序列化 checkpoint。

## 下一步 GO/NO-GO 边界

没有 postmigration blocking defect；行政迁移一致性门通过。本报告不自行批准 release，主控可以另进行显式 fresh-only release 工件审批。release 必须绑定当前 hold、设计/实现/current review、corrected provenance、COMMITTED ledger、queue、573/27 inventories、exact H49 及禁止 recovery/retry/formal/test 的授权范围，并通过既有无循环 validation 图。

当前仍 **NO-GO for search launch**：hold 尚 active、launch 尚 false；须另完成当前 gate/prelaunch 与 identity-bound 主控 release，再重新确认无 F0 进程/持锁及实时 GPU/slot 可用。187 fresh UNCLAIMED 是允许的候选集合，49 held keys 永久排除于本次 fresh search，不以行政迁移恢复其 checkpoint。native SIGSEGV 原因未消除，此 GO 只证明行政迁移保护状态正确。
