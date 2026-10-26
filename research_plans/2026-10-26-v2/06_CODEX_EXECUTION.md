# Codex完整执行入口

> 本文是下一次“实现→训练→验证→分析→报告”的执行任务。当前2026-10-26-v2发布轮只完成研究和设计，不在发布阶段启动GPU。用户要求执行本方案后，按以下全文自主运行，不需要再请Pro补算法或由用户在候选之间选择。

## 可直接交给Codex的任务

请基于`Caelestis1903/My-Research-Project`当前最新main执行`research_plans/2026-10-26-v2/`方案。首先读AGENT.md和本目录全部九份说明，数学/训练语义以05为准，阶段分支以04为准。09_EXECUTION_CONFIG.yaml是初始锁定配置；出现与05的实质冲突先在实施前修正配置并记录版本，不按结果挑有利定义。

本次目标是验证**强目标预测后的来源增量，以及有限函数数量和来源保留预算内的目标群体功能形成**。主候选RCF；谱分解、普通残差读取、标准锚定微调是强对照/组成部分，不把名称当作创新。旧DFD/v3停止结论保留，不修改历史结果。

### A. 先完成真实工作区与数据核对

1. 记录`git status`、branch、HEAD、远端main、实际源码路径；不覆盖未提交更改。有并行工作则建隔离worktree/checkout；不要直接合并未审查实验分支。
2. 记录当前main是否包含本方案全部文件、v4源码和7个pinned vendor文件。参考v4证据SHA `dc32f60a6aafce98a25838a485e02cead5425fc6`，不把历史HEAD当当前main。发布前观察的main `139fb294b0dd1e024b2fa1e906a4bd5964bb0db0`已撤回v4实验合并；不能假定旧模块可直接import。
   - 缺失时，从上述固定SHA有选择地恢复已审查的v4包内Python源码和7个vendor Python文件到新目录`experiments/mdmt_rcf_20261026/vendor_snapshot/mdmt_dfd_20261026/`。保留包内相对结构、必要的`__init__.py`、原始文件字节与许可证，保存来源路径、Git blob SHA及SHA256；只加入明确的外部包桥接。若旧文件含绝对包路径，桥接/适配另建文件并记录，不偷偷改冻结副本。
   - 只通过新实现调用RoleBundle/NativeAdapter/PilotBank、稳定数值与状态工具，不执行旧pipeline、full_plan或cleanup。原生/角色vendor逐文件核对来源manifest；当前main上同名但不同blob不得静默替代。
   - 可以使用固定SHA的只读worktree或GitHub逐文件读取取得上述源码，但不合并历史实验分支，不把报告、模型或大工件一起引入，不写回旧实验目录。服务器已有对应代码只有hash匹配才直接复用。 main `bbb899ecc137d9782f23e575605bac0dd2cbdab1`已增加`history_experiments/26-10-26/`归档，可优先从该只读路径取得相同blob；仍以固定dc32来源身份校验，不把归档路径当新实验身份。
   - SCALE触发时，同样从`c5618e281785a4b7063ff9bf2a61cf38d4faf1be`取得必要的reproduction_20261021/data_generate/grad源码到新实验的`vendor_snapshot/scale_20261021/`，记录相同hash/路径身份；只通过新入口覆盖输出到新artifact根，不运行其历史默认输出或清理路径。
3. 对原始数据/行角色/编码/标签/自然域数量作05规定审计。natural fit与历史人为采样分开；fixed-fusion与原版M3oE分开；Sample训练与Macro评价分开。
4. 只读盘点原模型/cache/NPZ。已删除的内容不能假装可复用。新建必要共同前缀，禁止为了补旧outer μ重跑整套v4。
5. 所有写入只在新`experiments/mdmt_rcf_20261026/`、`research_reports/2026-10-26_v2/`与新artifact根内。绝不触碰其他`/home_new`目录或其他用户进程。
6. 开发seeds5201/5202，锁定6201–6204。若同方法身份存在，严格resume/核hash，不自动换seed。没有新封存数据就明写暴露基准复评。

### B. 实现最小完整代码，而非只打印计划

按05第9节实现新模块，复用v4的RoleBundle、NativeAdapter、PilotBank、稳定BCE、checkpoint/hash和metrics。不要复用旧stage gate来关闭KR。先实现共同prefix/B、固定内容读出与RCF三臂，标准RRR及其他S3模块可在被触发前完成，但不能把触发后必须的分支留成伪函数。

精确落实以下语义：

- h为4 shared experts归一化后拼接；B输入[h,z]，source pilot是冻结源监督。
- B的OOF只用于需求代理；A和后继已使用P_form标签，不能称全链OOF。
- 需求在真实target群体上计算，剔除一阶bias/temp响应，不外推η_t到source当作真值。
- k不是目标域名字，K最多2，允许K1/无请求/reader OFF。
- δ隐藏复制source pilot，末层0；形成用B_OOF+δ，最终只读δ。
- source owner/native/B/原pilot不接收目标梯度；δ明确接收目标训练。
- S来源风险、R所有foreign目标RMS与KL按05；dual不等于可行证明，终态可行缩放必须执行。
- 每槽query计划、重要权和auxiliary row plan严格按05；不要擅自跳过零查询步骤又声称同一无偏目标。
- 形成后冻结δ，主reader只P_read；P_cal只整组六头选择。
- FULL_SOURCE_ALL/LOCAL_ACTIVE/H_DIRECT_EFFORT保证强简单解释具有充分标签与容量机会；不能把它们人为削弱。

### C. 实施必要验收

08列出具体风险与验收，不要另开泛化大测试清单。必须实际验证：首步delta末层可学/隐藏首步零合理；真实梯度权限；None与optimizer状态；权重均值及域聚合；约束缩放；raw-X/缓存一致；完整resume下一步；政策组选择；outer机制导出。

通过后生成不可变`IMPLEMENTATION_IDENTITY.json`、`RUN_MANIFEST.json`、`DATA_ROLE_MANIFEST.json`和`LABEL_ACCESS_LEDGER.json`。所有未运行项标未运行；不要预填AUC、耗时或“PASS”。

### D. 运行顺序与自动分支

1. **S0**：两数据×两seed四个共同prefix。
2. **S1**：固定内容与读取判别，完整强简单对照。
3. **S2**：RCF/POOLED_TARGET/LOCAL_ACTIVE一次三臂形成；即使S1源读失败，也只允许这一轮有界形成，不无限救方案。
4. **按04决定S3子集**：完整实用增量才全做归因；机制-only只做K1/pooled/profile；普通reader足够即保留简单方法。
5. **S4仅锁定后触发**：通过数据×4seed完整内部联系复评；未通过数据不擅自重开。外部baseline按身份和忠实度验收单独处理，不将其缺失扩成内部任务整体停摆。
6. **SCALE独立小分支**：原10-21真实配置、Adam数值桥、Dconst01_NO_AUG，严格预算；不是恢复动态权重或动量主因叙事。
7. 所有清单冻结后统一评价暴露test/legacy，不用它们作临时算法选择。
8. 完成全部原始配对预测、功能profile、outer需求、成本和可复现身份，随后才能清理临时模型。

若遇可修复代码问题，定位后以相同数据/seed/计划继续；若改变科学语义则先登记新的实现版本并说明受影响运行，不选择性只重跑负结果。实际数据/源码不可得时完成其余任务，列阻塞来源和不能得出的结论，不伪造替代结果。

### E. 最终分析与报告

不得只生成总AUC表。最终报告必须回答：

1. 强目标基准之后，来源增量在哪些数据/任务/群体/预算内成立？
2. 来源收益来自已有内容读出、来源初始化、来源行为约束、群体形成，还是更多目标拟合？
3. K2是否胜重新训练K1，是否有跨seed的功能交叉优势？
4. OOF需求是否迁移到refit B；哪些源/目标支持范围失效？
5. 源约束训练满足是否在P_cal/Q保留；多少δ被缩到0，原因是什么？
6. 相对强简单对照的收益及完整训练/推理成本是否值得？
7. 哪一层主张获得支持，哪一层只机制成立，哪些应降级基线？
8. SCALE到底解释了什么，是否只是已知正则/尺度效应？
9. 暴露test、少量seed、原始user缺失和未完成外部比较的边界是什么？
10. 下一步对应04中的哪条分支，而不是泛泛“再多做实验”。

## CLI合同与建议命令

以下是**本次待实现接口**，不是声称当前仓库已存在。Codex实现后按这些命令执行；参数规范固定在09配置。所有子命令默认读取同一09配置和其中artifact根，显式--config覆盖时必须与已冻结manifest一致。

```bash
python -m experiments.mdmt_rcf_20261026.cli audit --config research_plans/2026-10-26-v2/09_EXECUTION_CONFIG.yaml
python -m experiments.mdmt_rcf_20261026.cli build-plan --config research_plans/2026-10-26-v2/09_EXECUTION_CONFIG.yaml
python -m experiments.mdmt_rcf_20261026.cli verify --scope required
python -m experiments.mdmt_rcf_20261026.cli prefix --datasets movielens kuairand --seeds 5201 5202 --resume
python -m experiments.mdmt_rcf_20261026.cli read-probe --locked-plan
python -m experiments.mdmt_rcf_20261026.cli form --stage s2 --locked-plan
python -m experiments.mdmt_rcf_20261026.cli decide --stage s2
python -m experiments.mdmt_rcf_20261026.cli diagnose --triggered-only
python -m experiments.mdmt_rcf_20261026.cli decide --stage s3
python -m experiments.mdmt_rcf_20261026.cli freeze --triggered-only
python -m experiments.mdmt_rcf_20261026.cli run-locked --triggered-only --resume
python -m experiments.mdmt_rcf_20261026.cli evaluate --locked-only
python -m experiments.mdmt_rcf_20261026.cli finalize --verify-exports --cleanup-owned-temporary
```

`scale-probe`先numeric，后按04/05触发真实训练；CLI不得因缺失CLI子命令而谎称完整任务完成。stage状态与未启动原因全部写JSON。每条命令幂等，已经完成且hash一致则复用；不一致拒绝覆盖。

## 必须生成的研究报告

`research_reports/2026-10-26_v2/`至少包含：

- `FINAL_REPORT.md`：先给科研结论与行动，含全部反例。
- `IMPLEMENTATION_REVIEW.md`：已落实机制/修订/实际验收。
- `DATA_AND_IDENTITY.md`：自然角色、模型、目标、代码/工件身份。
- `SOURCE_INCREMENT_REPORT.md`：固定内容、读取、强直接/本地对照。
- `FUNCTION_FORMATION_REPORT.md`：A、δ、约束、K1、交叉功能、source干预。
- `SCALE_REPORT.md`：独立历史桥及未运行项。
- `COST_REPORT.md`：独立端到端与共享prefix研究成本分开，不重复相加。
- `OPEN_QUESTIONS.md`：真正未区分解释与具体下一行动。
- `tables/`：逐seed/六格/群体/支持/参数/成本CSV及自动核算收据。
- `ARTIFACT_INDEX.json`、`EXPORT_ACCEPTANCE.json`、`CLEANUP_RECEIPT.json`。

代码、报告、配置与小收据可以提交独立实验PR；大模型和大缓存不进Git。报告明确原始预测的实际服务器位置、hash与保留状态。不得因清理授权把其他用户或其他实验目录当作可删范围。

最终应交付可解释结论及完整负结果。中间常规实现/数值修复/可逆操作自主完成，不交用户反复确认；实际权限或不可替代工件阻塞必须明确报告。

