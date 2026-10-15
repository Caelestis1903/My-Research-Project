# DESIGN SPECIFICATION：F0 执行控制 v1.4，本次 SIGSEGV 后的 H51 行政保护及 fresh-only release

日期：2026-10-13。范围：execution-control-only。本设计仅规定行政迁移、来源冻结和启动资格验证，不批准训练、retry、recovery、formal 或 test，不修改任何生产工件。需要新版本 v1.4；现有 v1.3 不能直接授权本事故后的重新启动。

## 1. Research question

在 worker0 fresh trajectory 发生 SIGSEGV、worker1 被受控停止后，能否保持原科学身份及全部实验资产不变，将两个新事故 key 永久排除，并通过独立审查重新允许尚未触碰的 fresh UNCLAIMED key？

“控制满足规格”和“native 故障已解决”分别判断；后者目前没有证据。

## 2. Hypothesis being tested

控制假设：固定保护集合、来源绑定、持久行政事务和每次 claim 前的串行验证，可以阻止事故 key、完成 key及任何具有历史执行资产的 key 被重新执行，同时允许独立冻结的 untouched fresh 集合继续搜索。

本次事实边界：

| 项目 | worker0 | worker1 |
|---|---|---|
| run_id | `0d2e6555b049c012c0e5` | `a80a232a3c0068a48a08` |
| nonce | `34de4e7e0970459a84a6457ff8ed490f` | `1787964e065e44bfbc503c26adca3e97` |
| child exit | −11 / SIGSEGV | 143 / TERM |
| 当前 queue status | RUNNING | INTERRUPTED |
| receipt | 不存在 | 已存在 INTERRUPTED receipt |
| 原 status SHA | `809b4dd83609f38df791afb1581f873c801aa3d408b4f51ff1d60c4dc00d4ca0` | `5dc0c4c0d656c7cc3f71772852d7b5508d9d9daa005c4d88c596dc70c5aaf1d5` |
| 原 manifest SHA | `55404e864d28abe43e346a35b308cb8e2a7e566f55179d81b341af0e64a09c63` | `d8377fafbd11ae7c5b5de209f36f30a6613c4fd060a56ede22bdfdc62fda7ee1` |

worker1 receipt SHA 为 `cced71d99bf329a963916a38b13dd48d0f3818c62faaad1e157a00d6f9b45f1d`，其最后完整 checkpoint 是 epoch32，SHA 为 `52a6559c4466a8e19b11c34e0d6ba12f38e09716b8d39e5f9e357a7dbb7076d4`。

本次 supervisor failure SHA 分别为：

- worker0：`51ac190ccfa846b982f5eb0ac0e4a97ee3d144a6f42a1d89b25b72d6c3fa17e2`
- worker1：`11b2735b9fbfd546ab2272b4f267d4a5358f1f51d62299726baa6dd3b4f1be0b`

当前 active global hold SHA 为 `7b2899b8ba587d986d4cdc8c5a7889888d213151f5a70c9393ce14f4a235c4d3`。这些是本次只读观测的冻结候选值；实施前必须从实盘重新核验，漂移则 NO-GO。

SIGSEGV root cause 仍为 UNKNOWN。peer TERM 不作为第二次 native 故障。

## 3. Variables changed

- 当前保护集合由 H49 扩为 H51。
- 仅将两个新事故 queue status 行政归类为 RECOVERY_BLOCKED。
- 新增本次事故来源快照、迁移 ledger、执行 snapshot、独立 review、prelaunch 和 release 工件。
- 更新执行控制验证入口，使当前 launch/claim/release 必须执行 v1.4 保护。

现有 v1.3 的固定 H49、旧 migration ledger、correction、review 和历史 release 保持原义及原字节，不重新解释为 H51。

## 4. Variables held constant

科学 canonical JSON/full SHA、固定 288 项 search queue、code_hash、protocol_hash、数据划分、模型、目标、optimizer、seed、batch、训练预算、选模规则和 test 门均不变。

所有历史报告、审计、receipt、native 日志、metrics、checkpoint、attempt manifest 及已完成资产保持字节不变。不得将 attempt manifest 的 RUNNING/INTERRUPTED 改成 RECOVERY_BLOCKED；外层行政资格与实际执行状态通过 ledger 关联。

不得创建 attempt-01、提高 recovery_count、改变 resume_checkpoint、补造 SIGSEGV receipt、删除历史锁文件或用其他 seed/config 补足事故轨迹。

## 5. Mathematical / algorithmic definition

唯一当前集合：

```text
H51 = H49 ∪ {
    0d2e6555b049c012c0e5,
    a80a232a3c0068a48a08
}
```

H49 独立 SHA：

```text
60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e
```

H51 排序字符串数组、UTF-8 紧凑 JSON、无尾换行 SHA：

```text
e9077aefa1ed48a3237b769b20c0cd9e8ca3f9b0054ee537de1b7a40cd870a86
```

必须保留独立固定 H49 并加入两个固定新增 key；不接受 caller 提供列表作为保护集合来源。gate、caller、hold、ledger、prelaunch、release 必须 exact H51，无缺项、重复或同数量替换。

固定 search queue SHA：

```text
43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23
```

本次 search baseline：

```text
迁移前：97 VALIDATION_COMPLETE
        49 RECOVERY_BLOCKED
       140 UNCLAIMED
         1 RUNNING
         1 INTERRUPTED

迁移后：97 VALIDATION_COMPLETE
        51 RECOVERY_BLOCKED
       140 UNCLAIMED
```

这 288 项之外的 6 项 RESOURCE_PREFLIGHT_COMPLETE 不得计入 search snapshot 或被此次事务修改。

当前 97 completed key 排序紧凑 JSON SHA：

```text
06cba20a42bfc3f8dfc20b4df68ac7dc774c66bb34972ab4158a7227db4d7fce
```

当前 140 fresh 候选 key 对应 SHA：

```text
78e529b19e6fb9cc55cdc32524f3243005c0bf88f5dbe4206c44062a645f585b
```

计数和 key digest 仅为辅助；必须逐项验证科学身份和实际 fresh 条件。fresh 定义沿用生产 claim：不存在 status 文件、attempt/run 目录、checkpoint 或历史 canonical lock。caller 将旧状态改为 UNCLAIMED、删除 status 文件，均不能重新获得资格。

release 后允许执行集合是冻结 F140 中仍满足上述条件的成员，不扩大搜索空间。live counts 可以因合法 claim/completion 推进；不得要求一直保持最初的 97/51/140，也不得修改冻结 baseline 为新 live counts。

## 6. Exact affected modules / files

允许修改执行控制：

- `experiments/stage_F0_worker_control.py`
- `experiments/stage_F0_queue.py`
- supervisor/child/F0Runner 中必要的当前 gate 调用点
- `experiments/stage_F0_outputs/runs/WORKER_CONTROL.md`
- focused control/queue/migration/launcher tests
- 当前 execution gate/hold/release 行政指针

建议新增：

- `experiments/stage_F0_postabnormal_v14.py`
- `experiments/tests/test_stage_F0_postabnormal_v14.py`
- `experiments/reports/stage_F0_postabnormal_design_v14.md`
- 新 v1.4 implementation report 与独立 review
- `execution_control/postabnormal_sigsegv_audit_v6_fresh_search.json`
- `execution_control/postabnormal_v14_source_snapshot.json`
- `execution_control/postabnormal_v14_migration.jsonl`
- `execution_control/postabnormal_v14_execution_snapshot.json`
- `execution_control/prelaunch_audit_v14_fresh_search.json`
- 唯一版本化 release 工件及 append-only release ledger

所有新增文件必须禁止覆盖已有同名不同字节。当前 gate/hold/release 更新前保存 raw bytes/hash 的不可变版本化快照。

不得覆盖 v1.3 design/implementation report；旧 design SHA 已嵌入历史来源图。新的 design/report 路径独立冻结。科学 search queue 和任何 attempt manifest 均不属于“更新 manifest”的范围；更新仅指新增 execution snapshot 和其行政指针。

## 7. Implementation requirements

现有机制的复用边界：

- 可复用 atomic write/fsync、execution_state/canonical/worker/GPU 锁、身份检查、fresh eligibility、slot wrapper、pidfd、锁内 receipt 和 failure/hold 机制。
- 不能原样复用 v1.3 `apply_migration`：其 target、原 hold、52 completed 基线和 ledger 都硬编码旧事故。
- 不能直接调用旧 `validate_h49_runtime` 作为本次完整 runtime 验证：active 分支要求旧 COMMITTED hold hash，且当前 release 图绑定 H49及旧当前 review。
- 不能只替换 gate/prelaunch 中的列表/hash，更不能添加通用 `ignore_hold`、`skip_provenance` 或 `allow_stale` 开关。

v1.4 必须将历史证据验证与当前 launch 验证分开。历史 v1.3 子图读取冻结的旧 PREPARED/COMMITTED、correction、review、旧 release/hold projection 和旧 gate 快照，仍按旧 H49规则核验；此函数不得授予生产启动资格。当前验证另外要求 H51、本次事故图和当前 release。不得把当前 H51 hold 硬塞入历史 H49 validator，也不得通过隐藏忽略参数绕过历史检查。

H49 API/常量保留历史含义；新增 exact H51 API。当前 supervisor、runner、claim、release、failure hold writer 使用 H51，不得残留能够只检查 H49 的生产入口。当前保护常量的扩展不能使 historical H49 比较函数意外改义。

行政事务：

1. 保持 active hold。核验无残留 F0 进程，相关 worker/GPU/canonical 锁空闲。
2. 在锁内重新核验本次 failure/hold/queue/receipt 身份与全字节基线。若本次原始审计发现转录冲突，新增 correction，原审计不改；冲突未解释则停止。
3. 保存 source snapshot：原 hold/gate/release raw bytes、两项 status raw/projection、97 completed exact key 集合和全树 inventory、140 fresh exact 集合、旧 H49 status projections、两项事故全树与 nonce 日志/receipt/failure inventory、历史 v1.3 子图。
4. source snapshot 通过独立 reviewer 冻结后，才允许行政迁移。禁止由 caller 单独指定新的资产 digest 代替冻结来源。
5. 在 execution_state、两 worker/GPU 与两 target canonical 锁保护下，持久写 PREPARED，随后仅更新两项 status，持久写 transition records，再追加 H51 migration hold event，最后重读并提交 COMMITTED。
6. 保留原 hold 所有历史 events，明确 top-level exact H51。只允许预定义 administrative 字段改变，完整 projection 比较；旧 released_at 等历史字段不改造成新 release。
7. 两项 status 从 raw original projection 重建，仅增加预定义行政字段/原因并修改外层 status；所有其他字段逐项不变。
8. 故障、掉电、部分提交后 hold 仍 active；幂等重入重新 fsync 已存在的文件及目录，再补 durable transition/COMMITTED，不重复恢复或迁移其他 key。
9. 完成后验证原97完成树、旧事故资产及本次两事故资产全内容不变；原 H49 full status 及其迁移元数据不变；fresh 集合不变；不存在新 attempt/receipt/recovery。
10. 新 gate 绑定实际当前 control/runtime inventory、独立新 review、historical 子图、本次 snapshot/ledger、active-hold hash、queue hash、H51 和 fresh F140。
11. 保存科学 code hash 与实际 control amendment，禁止重写 canonical key。新 module/test/doc 必须纳入相关执行 inventory。
12. release 只允许 principal 显式 fresh-only F0-B1-ML；resume/retry/new_attempt/checkpoint_load/formal/test 全 false。

hash 图避免循环：

```text
来源快照与设计
→ 实施/测试及 implementation review
→ 行政迁移 COMMITTED
→ 独立 postmigration review
→ pending release gate 与实时 prelaunch
→ principal release 意图
→ 精确确定的 released-hold projection/hash
→ release 工件
→ 指针提交与最终实时核验
```

release 不要求已经冻结的 gate 反向包含 release 自身 SHA。更新过程由 execution_state 锁串行化；任何中间状态不满足完整图就不能 launch/claim。旧 v1.3 release，即使 hold_id 未改变，也不能解锁本次事故，因为完整 source active-hold hash、H51、incident nonce 和当前 review 不匹配。

## 8. Ablations / failure-injection fixtures

仅临时 CPU/Linux fixtures，经计槽 wrapper、项目 Python 环境执行；可并发的独立 fixture 尽量并发，总任务 Python≤5。不得以生产 checkpoint/manifest/receipt/metrics 为测试写入目标。

必须覆盖实际生产 validator 和 claim：

- 漏新任一项、旧 H49、重复、同数量替换、共同伪造 gate/caller/release/snapshot 列表均拒绝。
- H51 正确但实际任一 held 不是 BLOCKED，拒绝。
- 旧 v1.3 release replay、旧 review 成对替换、旧 hold hash、缺新 failure/peer receipt、nonce/fullhash/UUID 矛盾均拒绝。
- 资产变更、额外文件、缺文件、原 report/snapshot 覆盖、历史 v1.3 子图漂移均拒绝。
- RUNNING stale/newly labeled UNCLAIMED/删除 status但保留历史目录或锁，都不可 claim。
- 正确的 untouched fresh fixture 可 claim，attempt-00/recovery_count=0/resume=null；新 release 后合法 completed/live counts 推进不导致把历史 baseline 改写。
- 两项 status之间、status后hold前、rename/fsync/ledger/hold/release提交各边界注入失败：始终 fail closed、无下一 child；行政重入可收敛且不创建新 attempt。
- 新 hold event 出现在 prelaunch 与 release之间、release 与 claim之间时，旧 release立即失效。
- 重复 migration/release请求不能产生两个 COMMITTED 或重新扩大 F140。
- 生产未批准入口不能绕过新版 gate；manual first-epoch 等未冻结入口保持禁用。

保留 v1.2/v1.3 已审的真实信号、pidfd、receipt屏障、PDEATHSIG、STOP/CONT、30秒期限、durable-before-KILL和slot语义。若相关代码改变，必须重跑相应真实 fixture；不能仅以文本承诺无回归。

## 9. Expected logging requirements

每个新增工件包含 schema/version、路径/hash、冻结集合、原始来源、完整身份、禁止授权、操作者/时间、before/after projections和资产 inventory。

必须保存：

- SIGSEGV receipt absence；不得补造。
- peer original receipt status/epoch32/checkpoint与锁内历史 owner快照。
- 固定288 queue与非search对象排除事实。
- H49/H51/F140/C97 exact列表、digest和来源。
- source snapshot、PREPARED、durable transitions、COMMITTED、review、prelaunch、release graph。
- GPU memory used/total、百分比、UUID、观测时间；GPU utilization 和外部进程存在只记录。
- 当前任务 Python数量与 slot身份。

无需诊断或分析 AUC/loss；此次未改变科学算法，没有新的科研消融。

## 10. Acceptance criteria for reviewer and runner preflight

实施 reviewer 必须独立检查设计、真实实现、coder report及负例。只有 PASS，或 PASS WITH FIXES 且全部 blocking已纠正，才可进行纯行政 RUN。旧 v1.3 PASS不代替此门。

行政迁移后需要独立 postmigration GO，仅证明迁移一致性；不自动释放 hold。

Runner启动前必须重新核验：

- 本次审计、source snapshot、97完成与全部事故资产、历史子图实际 hash；
- 全部51实际 BLOCKED，140候选仍 untouched，固定queue无变化；
- 新 independent review、当前 code amendment、gate/prelaunch/release完整引用图；
- 无残留 F0进程/活动 claim，锁/slot可用，磁盘与设备检查通过；
- 使用两 worker，最多4个常驻任务Python（两supervisor加两child），剩余一个共享槽；不得因用户“最多5 Python”增加第三个GPU worker；
- 每张目标GPU `memory_used / memory_total < 0.90` 才允许新的worker/trajectory启动。GPU使用率或外部任务存在不构成拒绝条件，不停止外部任务；读取失败或达到/超过阈值则不启动新的trajectory，记录NO-GO。不得仅为等待显存启动未计槽Python。
- hold只有在 principal新 identity-bound release后才可inactive；supervisor child-spawn前、runner数据/CUDA/claim前、execution_state锁内claim时再次验证当前图和fresh资格。
- 新异常仍执行已审failure/hold/peer-stop流程，无下一child、无自动retry。

满足门禁仅允许F140中剩余fresh搜索。原固定search单位若因held配置缺失而无法公平选模，保持不完整，不补种子、不进入formal。

## 11. Conditions that would falsify the hypothesis

任一 H51成员进入执行、历史key重标fresh后被claim、遗漏来源仍release、旧release重放成功、资产或科学身份变化、checkpoint load/resume/retry/new attempt、部分行政提交后launch、第三GPU worker或超5任务Python、异常后下一child，均否定此控制规格的满足性，立即保持hold并回到Coder/Reviewer。

新的fresh trajectory再次SIGSEGV只说明native稳定性仍有问题，不能直接否定算法，也不能凭控制PASS将根因由UNKNOWN改成已解决。
