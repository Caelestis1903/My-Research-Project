# DESIGN SPECIFICATION：F0 v1.5 B6 后续证据政策 v1

设计范围：证据保全与可审查的后续裁定。本规格仅允许 IMPLEMENT → REVIEW，不授权测试、release、claim、训练或生产文件变更。现有 B6 为 OPEN/FAIL，生产及科学 RUN 为 NO-GO；本报告不替代独立 Reviewer 的裁定。

## 1. 研究问题

在事故前完整 inventory、原 runner preimage/diff 及部分共享锁诊断不可恢复的情况下，能否建立诚实、可验证的事故后基线，使未来隔离行为有明确证据边界，并为独立 Reviewer 判断后续受限推进提供材料？这是执行控制/证据问题，不是模型效果实验。

依据：按 GB18030 解码阅读 `promt/stageF0-4.md`，核对研究合同/基线协议、v1.5 DESIGN v2、implementation review v7、事故裁定及 peer-log fixture review。提示词要求保留失败原始记录，停止受影响分支，不能重新定义目标绕过阻塞。

## 2. 待检验假设

事故后完整、稳定、明确限定范围的原始归档，加上独立复核的未来隔离证据，可以支持“从新基线起的保全”，但不能支持“事故前完整保全”。竞争解释包括归档不完整、运行时共享 inode/锁/路径仍存在、测试 fixture 与实际生产入口不等价、未记录的历史副作用。

政策不预设 B6 可以关闭，也不预设当前运行环境足以训练。

## 3. 改变变量

仅新增证据政策、事故后 baseline 工件及独立裁定字段；由 Coder 按本规格构造只读收集/校验工具。原历史判定不可改写。当前代码语义、生产 validator、门禁与授权路径不变。

## 4. 保持不变

保留生产 hold `active=true`、ID `cd2dbf50-3a50-4b81-b976-38f1b03991a3`，其全部事件及两份审计日志原始字节。保留 worker0 的九条 `CHILD_STARTED` 追加记录及共享 production flock 事故披露；不截断、不删除、不重写、不将旧事件归类为新的合法训练事件。

保留 C97/H51/F140 科学身份、queue、data/split、model/objective/optimizer/selection、种子和失败预算；H51 不恢复，已触碰 key 不转为 fresh。原 v1.4 release 不重用。四正式 seeds 为 2022–2025；冻结搜索队列 development seeds 不被替换。

GPU 判据仍为单卡 `10*used_bytes < 9*total_bytes`，至少一张卡满足即可；不要求双卡或 GPU 空闲。Python 总并发上限 5，独立工作尽量并发。资源可用不能替代科学/实施/证据门禁。

## 5. 政策与算法定义

### 5.1 必须逐字采用的声明

> B6 历史事故及证据缺口永久保留为已披露事实。当前归档是事故后基线，只约束明确列出的路径和采集窗口，不是事故前 preimage，不证明此前完整无副作用，也不证明原科学 runner 的字节身份已恢复。后续隔离测试仅支持各自实际覆盖条件下的前瞻隔离；通过测试不补回缺失历史证据，不自动关闭 B6，不自动允许 release 或科学 RUN。是否允许任何后续受限推进，必须由独立 Reviewer 对显式政策修订、原始证据及确切执行范围作单独裁定。

### 5.2 分离状态，不设自动放行

分别记录：

| 字段 | 本次允许值/含义 |
|---|---|
| `historical_B6_status` | `OPEN_FAIL_IRRECOVERABLE_GAP`；事实缺口不可用后来测试改成 PASS |
| `postincident_baseline_status` | `NOT_COLLECTED` → `COLLECTED_UNREVIEWED` → 独立裁定 `ACCEPTED_BOUNDED` 或 `REJECTED` |
| `prospective_qualification_status` | 原始 run 为 `RECORDED_UNREVIEWED`；独立复核后按具体 case 给 `PASS_BOUNDED`/`FAIL`，不得合成历史 PASS |
| `forward_disposition_status` | `NOT_GRANTED`；后续独立裁定才可为 `ACCEPTED_WITH_RESIDUAL_LIMITATION` 或 `REJECTED`，并列 exact scope、理由、前提、未解决项 |
| `execution_authorization` | 本政策恒为 `NO_GO`；不得从上述 ACCEPTED/PASS 推导授权 |

如 Reviewer 认为 B6 的运行阻塞可在某个新范围被处理，必须单列“前向处置已接受、历史证据缺口未消失”，不得覆盖事故报告。无法满足当前历史验收要求时继续 NO-GO。状态标签是本证据政策的描述，不直接加入运行时 gate，也不被运行时 validator 当作 approval。

### 5.3 事故后 baseline 的精确绑定

新工件 schema `F0-B6-POSTINCIDENT-EVIDENCE-v1`，必须含：

- `baseline_id`、`label=POST_INCIDENT_BASELINE`、UTC 采集起止、采集者、host/repo 真实路径、policy path/SHA、收集工具 path/SHA/命令/环境；独立 review 字节用后续独立工件绑定，避免自哈希循环。
- `scope_roots` 与精确 `scope_paths`、排除项及理由、absence paths、inventory completion 状态。范围包括当前 gate/hold/release pointers，全部 v1.4 历史控制工件、incident/failure/stdout/stderr/phase/receipt 缺失证据，两份 audit 原字节及九事件 slice，C97/H51 statuses/run trees/checkpoints，全部 F140 status/lock/run/attempt 的不存在性，queue/search gate，当前相关 scientific/runtime/control 源与测试及其各自 inventory。checkpoint 只读取原始 bytes 计算 SHA，不加载模型；数据仅绑定已冻结 manifest/split hash，不扩展到未授权数据读取。
- 每个路径的 `relative_path`、kind（regular/directory/symlink/absent）、size、mode、device/inode/nlink、mtime/ctime、regular-file SHA、采集前后 stat、archive relative path/SHA；symlink 记录目标且不跟随，保护控制路径出现 symlink 拒绝。目录列表确定性排序；缺文件及权限失败不得静默省略。
- 用 SHA-256 对确定性 UTF-8、sorted-key、无 NaN、newline 的完整 inventory 序列化绑定 `inventory_sha256`；每个归档是独立 ordinary file，使用 exclusive 新路径、fsync bytes 与目录、readback 验证。任何归档和生产可变路径不得 samestat；不 hardlink。证据目录位于 active production run/gate/queue 树外，选 `research_F/F0/evidence/B6/<baseline_id>/`；不更新既有报告。文件权限称作约定保全，不宣称 OS 防篡改。
- `history_missing` 逐项列未恢复 pre-edit inventory、原 runner preimage/diff、缺失锁诊断；`historical_no_side_effects_proven=false`、`historical_full_equality_proven=false`、`original_runner_byte_identity_proven=false`；附残余风险与结果可用性限制。
- `quiescence_observations`：采集窗口的进程/claim/slot 与 `/proc/locks` 只读快照；不取得生产 flock，不创建 absent canonical lock，不修改 scheduler。两次完整 inventory 的同范围路径/kind/bytes/关键 stat 一致才可宣布稳定；否则保留不稳定归档、标 `REJECTED_UNSTABLE`。双读与无当前 owner 不能证明完整历史或无窗口间瞬时副作用。

对受保护文件 p，未来保全判断仅为 `SHA(p,t_before)=SHA(p,t_after)` 加同范围 path-kind/inode/absence/锁访问证据；所有结论注明 t 和 scope。hash 相同不足以证明无 flock 同步副作用。

### 5.4 当前资格证据怎样使用

attempt4 四组 stdout 原始记录为 1、1、16、7 passed；它们只支持记录的 case，不能称全部 DESIGN §§8/10 已闭合。总 manifest 位于 `/tmp/stageF0v15qualification-20261014-attempt4/archive/final_artifact_hashes.json`，SHA `12a7f49451edcf4a6f802e2adc0aa83d71a45e9e393cedd8326f472fb258abcd`。before/after protected-list 的共同 SHA 为 `3c0b3d13b173bbd99f0f4a6f6d67f31e314c16fb75661e1c239b7f030fc25d69`；这是九个列出路径的该窗口证据，不是完整新 baseline。

收集者必须把临时原始证据普通复制到上述新证据目录，绑定完整 manifest 并逐项核验；缺项显式失败，不能仅引用 `/tmp` 作为持久证据。归档中 synthetic inner peer-log freshening 的原 bytes/metadata、新 inode/空日志与后续事件必须一并保存并标 `SYNTHETIC_FRESH_PEER_LOG`。生产 peer audit 仍有非 JSON 历史行，未被修复；严格 parser fail-closed 保持。不能直接将 fixture reset helper 用于生产。若未来运行需新日志 stream，必须另行设计/实现/审核有连续 provenance 的新 stream，保留旧 raw log。

## 6. 影响文件

本次新增此设计报告。后续 Coder 仅新增 `experiments/stage_F0_B6_evidence.py`（只读收集/校验）、`experiments/reports/stage_F0_v15_B6_forward_evidence_policy_implementation_v1.md`、上述新 evidence 目录；有必要的反例测试限新增 `experiments/tests/test_stage_F0_B6_evidence.py`。这些路径是拟议范围，不代表当前存在或已批准 RUN。

禁止修改 `stage_F0_run.py`、queue/scientific/model 代码、successor validator、生产 audit/hold/gate/release/ledger/lock/status/run tree及旧报告。新工具源码会改变某些全量 source digest，必须如实列出；不得为维持旧 digest 省略工具，也不得自动刷新生产 source hash。后续 staged review 绑定实际最终 source graph。

## 7. 实现要求与训练恢复的确切前提

Coder 只实现只读采集/校验与文档，不导入 training/model，不 torch.load，不运行 GPU/训练，不调用 release/claim API。输出仅新 evidence/report 文件；mutation API 必须不存在。设计通过、实现独立 PASS 且阻塞修正后，Runner 才可另行执行有界非训练收集/反例验证，记录原始证据。

**当前任何该政策不能直接恢复隔离科学 F0-B–E 训练。** 原因：v7 授权仅为非训练 qualification；事故 disposition 为 B6 OPEN/FAIL；当前 hold active、search gate `PAUSED_ABNORMAL_WORKER_EXIT`；旧 gate 的 `launch_authorized:true` 与旧 release 不释放新 hold。v1.5 范围仅 fresh F0-B1-ML search，不覆盖 B–E 正式矩阵；独立 clone 名称和可用 GPU 均不增加授权。

后续受影响训练要恢复，至少依次具备：

1. 本政策独立 design review；如 Reviewer 接受前向处置，明确签署对 DESIGN v2 §§7.1/10 不可恢复历史要求的**前向范围修订**，列出接受理由、替代保全要求、残余不确定性与科学结果适用边界；本规格不自行宣布豁免。
2. 工具 IMPLEMENT → REVIEW → bounded RUN 形成稳定 baseline；独立 Reviewer 核验实际原始归档与全部 qualification coverage，列出未覆盖/反例、原 runner 历史身份限制及 B1–B5 状态，不能仅依据 25 个 passed 数量给 GO。
3. 对确切拟执行范围、当前源码/scientific semantics、数据/split、队列/selection/seeds、真实入口/日志连续性、freshness、slot/GPU 隔离获得独立 implementation 和 staged graph GO。处理生产 malformed peer log 的未来使用策略须另审；synthetic fixture 结论不替代该门。
4. 走已审核的 v1.5 一次性 release 事务并记录会话既有 user continuation authority；不必因既有授权再次询问。仍不能复用 v1.4 release，不能在政策收集中释放 hold。或另由 Designer 提出真正独立的科学执行 namespace/权限范围，Reviewer 明确裁定其独立性及全局 hold 适用边界；未经此设计审核不得复制 gate/绕过 validator 来训练。
5. F0-B1-ML fresh search 以外 B–E/P、formal/testing 等范围分别完成原研究协议规定的子门；当前 `formal/test/resume/retry/new_attempt/checkpoint_load=false` 不能凭本政策更改。当前 GPU 能见性核验与单卡 <90% 只是最后资源门。

独立 Reviewer 不能合理接受上述证据或范围修订时，确切前提仍未满足，应保持受影响训练 NO-GO并继续无依赖的已授权工作。

## 8. 所需对照/反例

无需模型 ablation。只需针对工具的判别性反例：缺失文件不省略；copy/hash/readback 不一致拒绝；采集间 bytes/path-kind/stat 漂移拒绝稳定状态；F140 broken symlink/lock/显式 UNCLAIMED history 不当作 absent；archive alias 拒绝；malformed log 原字节完整归档且不清理；“post-incident baseline”不能编码成 historical PASS 或 execution GO。所有反例用独立 synthetic roots，production 仅只读。

## 9. 日志与证据等级

完整命令、项目 Python 环境、exit/stdout/stderr、采集窗口、范围/排除、错误、before/after、每次复制/校验、archive manifest、工具源码 hash、最多5 Python slot 证据；不输出 AUC/loss 科学结论。

现有事故事实/当前 hold 的直接原始证据：STRONG。已记录隔离 case 的前瞻能力：经范围独立复核后最多 MEDIUM，范围以具体 case 为准。完整历史无副作用/原 runner preimage：WEAK/UNAVAILABLE；永久列明。记录当前 lock inode、无 owner 仅说明观察时刻，不升格历史无同步影响。

## 10. Reviewer 验收

必须逐项核对 DESIGN、实际工具、Coder report与后续原始证据；确认只写新 evidence/report，无 protected mutation/production lock获取/claim；所有缺项明确、归档无 alias、全范围稳定判断诚实、raw log 与事故事件完整；三层状态和 execution NO-GO分离；实测资格原始证据能持久读回且 clean synthetic log 标签明确。

review verdict 分别给政策设计、工具实现、baseline、qualification coverage、forward disposition；不得一个 PASS 覆盖所有门。任何后续 GO 另列 exact scope/evidence/preconditions，保留历史 B6 FAIL事实及旧报告。无额外 justified decision 时 B6 OPEN/FAIL、active hold、科学/生产 RUN NO-GO保持。

## 11. 证伪条件

遗漏/静默删掉缺失证据、推断原 hash、修改历史事件、生产共享 inode/flock、不稳定 inventory 却声称完整、以 clean synthetic log 证明真实生产无风险、把通过 qualification 改写成历史 PASS、把本政策 PASS 当作 release/训练 authority，任一项均否定该前向证据政策的可接受性，FAIL 并返回设计/实现；不得用训练阳性结果弥补。

## 只读设计时的输入绑定

| 输入 | SHA-256 |
|---|---|
| 事故 disposition（含 addendum） | `e87797f710eaa0aa75f4666f07349bd2e5050af9098b4cc28cf60366c66276b5` |
| implementation review v7 | `01675098e0e55f936690f49cc7d207ac625428f3cdb8a8a7b7ab2e1aca359449` |
| peer-log fixture review | `eafb5ab4131d557e2de7adba978edb4aaffcad158db71274bad7db317791b2da` |
| active hold（attempt4所绑定） | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` |
| canonical execution gate（attempt4所绑定） | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` |
| worker0 / worker1（attempt4所绑定） | `b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2` / `1725b85a04446bd3dbf650412c1ec896815b7a1a3924fe3c31e34c653022f960` |

本次 Designer 仅读取文档/现有 raw 与 SHA，未运行 Python、测试、GPU、模型、flock、release/migration；仅新增本设计报告。此表不是本规格要求的完整新 baseline。
