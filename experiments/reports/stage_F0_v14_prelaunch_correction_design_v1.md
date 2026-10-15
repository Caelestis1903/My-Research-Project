# DESIGN SPECIFICATION：v1.4 prelaunch 来源更正补充，仅行政工件

日期：2026-10-13。范围：operational-artifact-only。本补充未批准生产路径切换、hold release 或训练。

## 1. Research question

如何保留缺字段的原始 audit，同时让既有 validator 在固定 canonical path 读取完整、实时的更正 audit？

## 2. Hypothesis

保留原始字节、新增独立更正证据并审查一次明确的行政路径切换，可以修正 audit，且不削弱 H51、fresh-only 或 release 门禁。

## 3. Variables changed

仅新增原始 archive、完整 corrected audit、更正 manifest/review，并在批准后切换 canonical audit 的目录项和新版 gate 绑定。运行代码、控制 inventory、原设计文件均不改。

## 4. Variables held constant

hold 保持 active；H51、F140、C97、科学身份、migration ledger、source snapshot及原资产均不变。retry/recovery/formal/test/launch/release 不由此次更正授权。

## 5. Definition and approval boundary

原始 canonical audit SHA 已只读核验：

```text
12c723a4fff4115e68bf03c4283adcca6b225cf739fdf1e5dc0846215ac59c39
```

原文件已有 `disk.free_bytes`，但缺少 validator 必须检查的顶层 `disk_free_bytes`；不能直接从旧观测复制数值并称为实时 preflight。

既有 v1.4 §6 要求“所有新增文件必须禁止覆盖已有同名不同字节”，仅明确允许 gate/hold/release 行政指针更新。因此 **symlink 和 canonical audit 原子替换目前均未获得原设计授权**。本补充必须经主控接受及独立 Reviewer PASS，才能将该固定路径明确归类为“指向版本化 immutable audit 的当前行政入口”；不能先切换再补审查。

## 6. Files

生产行政工件全部位于 `experiments/stage_F0_outputs/runs/execution_control/`：

- 原 archive：`prelaunch_audit_v14_fresh_search_original_12c723a4.json`
- 完整新版：`prelaunch_audit_v14_fresh_search_corrected_v2.json`
- 更正记录：`prelaunch_audit_v14_fresh_search_correction_v2.json`
- 原 canonical：`prelaunch_audit_v14_fresh_search.json`

新增文件使用排他创建，禁止覆盖。更正规格/review使用新的独立报告文件，不追加到被冻结的原设计文件：

- 本补充规格：`experiments/reports/stage_F0_v14_prelaunch_correction_design_v1.md`
- 独立审核报告：`experiments/reports/stage_F0_v14_prelaunch_correction_review_v1.md`

如首轮 review 不通过，后续报告使用新的版本化路径，实际批准的 review 路径及 SHA 明确绑定 correction manifest；旧报告不覆盖。

## 7. Exact operational steps

1. 在 active hold 下，确认无 release transaction 已 PREPARED/COMMITTED、无 F0 worker/child；若已进入 release transaction，停止此路径并重新设计，不能替换它所绑定的 audit。
2. 持有 execution_state 锁，重新核验原 canonical 为普通文件且 SHA 精确匹配。通过排他 hard-link archive 保留原 inode/字节，再 fsync 原文件及 archive 父目录；若 filesystem 不支持 hard-link，允许排他创建完整 raw-byte copy，fsync 并逐字节/SHA核验后继续。不删除、不截断、不重写原字节。
3. 重新测量 GPU used/total/UUID、F0及任务Python进程/slots、active claims、生产锁、输出盘 free/total bytes。验证锁可用时不得创建 F140 尚不存在的 canonical lock。重新核验当前冻结来源和H51/F140/C97；失败写完整 NO-GO版本，不生成 release。
4. 排他创建完整 corrected audit。保留原schema及全部required bindings；加入顶层真实整数 `disk_free_bytes > 0`，与同次观测的 `disk.free_bytes`一致。GPU两目标显存均 `<90%`；记录新时间和完整新观测，不复制旧实时结论。包含原 audit archive path/SHA、更正理由、原字段缺失事实和 correction ID；保留禁止授权。
5. 排他创建 correction manifest，绑定 original path/SHA、archive path/SHA、corrected path/SHA、补充规格/review及变更原因。Reviewer独立核验原字节、完整新版及实时证据，批准具体 directory-entry 切换。
6. 获批后，在 execution_state 锁内重新核验 archive/current/corrected SHA和active hold。从 immutable corrected audit创建同目录临时普通文件，fsync，**原子替换 canonical目录项**，fsync目录；archive仍保留原始inode/字节，新canonical与corrected版本逐字节/SHA相同。记录 durable切换事件。此步骤不使用原文件就地写入或截断。
7. 新版 gate仍填写固定路径 `execution_control/prelaunch_audit_v14_fresh_search.json`，其 `prelaunch_audit_sha256`绑定corrected实际SHA；额外绑定correction manifest/review。保存旧gate，禁止覆盖immutable pending gate。既有 validator必须原样通过，之后才另行进行principal release。

若第5步审批后实时资源发生变化，新增更高版本重新测量及审查；不覆盖v2。

## 8. Symlink versus atomic regular-file replacement

既有 validator 的 `is_file()`/hash读取会跟随 symlink，但“代码能读取”不代表设计批准。symlink引入未冻结目标和额外路径身份；本补充**不批准symlink方案**。

推荐批准上述原子普通文件替换：archive与corrected版本均immutable，canonical仅作为明确登记的行政入口。若要求原canonical目录项也永久不可变，则**在不改runtime的限制下无可执行修正**，必须保持NO-GO。

## 9. Logging

保存原缺字段事实、原archive精确SHA、新audit全SHA、全部重新观测时间/来源、corrected完整内容、更正review、canonical切换before/after SHA及durability事件。原audit中已有的GO文字保留为历史记录，不作为当前GO证据。

## 10. Acceptance criteria

Reviewer必须确认：原archive精确SHA；新audit不是局部补丁；disk/GPU/process/locks实时且相互一致；全部冻结绑定不变；无源码或实验资产变化；canonical为普通文件且等同corrected版本；旧SHA gate仍fail closed，新SHA gate原样通过。Runner重新核验实时显存、进程、锁和完整图后才交主控release。

## 11. Falsification / stop conditions

原字节丢失或改变、未审批切换、symlink使用、复制旧资源结论、缺字段仍放行、旧gate/release重放成功、任何retry/recovery/训练启动，都使本次更正NO-GO。

本补充为设计工件；其保存不表示已创建、切换或批准上述任何生产工件。
