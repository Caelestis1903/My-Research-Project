# REVIEW REPORT：F0 v1.5 B6 后续证据政策设计审核 v1

日期：2026-10-14。独立 Reviewer `/root/charter_review`。

## 1. Verdict

**PASS — 仅针对证据政策设计，允许进入 IMPLEMENT → REVIEW。** 未发现该设计把事故后证据冒充事故前保全，或把政策 PASS、fixture 通过、独立 clone、可用 GPU 当作训练授权。

**本审核不签署 DESIGN v2 §§7.1/10 的前向范围修订，不接受 forward disposition，不批准 baseline/qualification、staged graph、release、claim 或任何科学/生产 RUN。B6 OPEN/FAIL、现有 active hold、受影响执行 NO-GO 保持。** 政策中列出的未来可能条件不是已经满足的事实。

## 2. 逐项核验

| 要求 | 设计核验 |
|---|---|
| 事故裁定及共享生产 flock 披露 | §§4、5.1、5.3 保留九条事件、原日志、hold 及两类事故；明确相同文件 hash 不证明没有同步副作用。PASS |
| DESIGN v2 §7.1 的原始保全要求 | 不宣称缺失 pre-edit inventory、runner preimage/diff 已恢复；事故后 baseline 必须另标范围、窗口、缺项及三个 `false` 历史证明字段。PASS；原缺口仍存在 |
| DESIGN v2 §10 的隔离验收 | §§5.4、7、10 不以 25 个 passed 或九路径 hash 代替全部覆盖；工具实现、baseline、qualification、前向处置分别裁定。PASS；本次未核验原始资格运行 |
| 未来保全证据 | 明确 ordinary-file 归档、拒绝 alias/hardlink、symlink/absence 检查、双次 inventory、原始日志保留、生产 flock 禁用；没有把双读稳定解释为历史完整性。PASS（设计层） |
| 授权分离 | §5.2 `execution_authorization=NO_GO`，状态不能进入 validator 作 approval；§7 要求独立修订、实现及 staged graph、一次性 release，禁止 gate 复制绕过。PASS |
| 科学合同与范围 | 保存科学身份、搜索队列、种子及失败预算；仅 fresh F0-B1-ML 搜索的既有控制范围，不扩为 B–E/P 或正式 test；不产生模型效果或机制结论。PASS |

## 3. Blocking 与 non-blocking findings

**政策设计新增 blocking：无。** 下列是仍阻止执行的既有/待交付门，不因本 PASS 消除：

1. **MAJOR / B6 历史限制：** 缺失事故前完整 inventory、原 runner preimage/diff及部分锁诊断；未有对 §§7.1/10 的明确前向范围修订及独立处置。后来测试不能补回缺口，当前政策不得替代该独立决定。
2. **MAJOR / 新证据未验收：** 本次仅审设计；工具未在本审核中验收，完整稳定事故后 baseline、持久原始归档、全部 qualification coverage及 B1–B5 状态未由本审核判定通过。政策引用 attempt4 的结果/hash不等于 Reviewer 实测验证。
3. **MAJOR / 实际执行图与日志：** 精确目标范围、当前科学/runtime/control 图、真实入口隔离、freshness、slot/设备、独立 implementation/staged graph GO 均需单独核验；生产 malformed peer log 的连续 provenance 方案另审，synthetic freshening 不能用于生产修复。
4. **MAJOR / hold 与 release：** 当前 hold 不由本政策释放；旧 release/gate 不能授权新 hold。恢复既有受影响路径还需已审核的一次性 v1.5 release；任何替代 namespace 必须另行设计和审查全局 hold 适用边界，不能只改目录名。
5. **MAJOR / 科学子门：** B1-ML fresh search 外的 B–E/P、formal/test/resume/retry/new_attempt/checkpoint_load 资格不得随本政策扩大；研究协议、数据/评估独立性及资源门仍按各自适用范围执行。

**MINOR / non-blocking：** §5.2 baseline 枚举仅写 `REJECTED`，§5.3另用 `REJECTED_UNSTABLE`。实现时明确后者是拒绝原因或拒绝子状态，并保持所有拒绝均 NO_GO；这不构成新的放行状态。`IRRECOVERABLE_GAP` 是保留历史限制的政策标签，不证明所有可能外部来源已穷尽；如后来发现真实历史工件，应新增带来源的审核记录，不能覆盖旧事实或推断旧 hash。

## 4. Corrections made

仅新增本报告；未修改政策、代码、gate、hold、日志或旧报告。没有运行 Python、测试、模型、训练、生产 flock、release 或 claim。

## 5. Remaining risks

实现可能遗漏实际保护路径、在采集期间遇到变化，或错误地将 bounded PASS 汇总成 GO，均需对照实际代码及原始证据独立复审。现有事故裁定区分直接原始证据、归属他人记录的 flock 事件及不可读诊断；本政策设计 PASS 不提高这些证据等级，也不证明原科学 runner 历史字节身份。新源码加入后必须按最终实际 source graph 重新绑定，不得自动刷新生产授权。

## 6. 是否可以执行实验及输入绑定

**科学/生产实验：NO-GO。** 下一步限政策所列工具 IMPLEMENT → 独立 REVIEW；有界非训练收集/反例 RUN 必须在其实现审核和适用 blocking 修正后另行进入。任何受限前向处置需单独列 exact scope、证据、前提与残余限制。

本次只读从磁盘计算 SHA-256：

| 文档 | SHA-256 |
|---|---|
| B6 forward evidence policy design v1 | `4084a9ec3054398e6b78ff2a1b1f8bcf55d1556b40fdff0a4c12eeae2941ff5a` |
| Incident disposition v1（含 addendum） | `e87797f710eaa0aa75f4666f07349bd2e5050af9098b4cc28cf60366c66276b5` |
| Successor control DESIGN v2 | `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316` |
| F Master Research Charter | `65fb26eb94a55956d64a68536a512f7faa1a4672bc0f6bafcc31bcc36ee7fff5` |
