# REVIEW REPORT：v7 qualification 必要只读范围澄清

1. **Verdict：PASS，仅限 `V7_READONLY_SCOPE_CLARIFICATION_ONLY`。**

现有 v7 `QUALIFICATION_PRECHECK_ONLY` PASS 覆盖：必要时流式读取唯一冻结 H49 checkpoint 的字节计算 SHA256，以及在普通源码副本中导入 qualification 所需入口模块。没有实现或算法变更，**无需重新开启 IMPLEMENT/REVIEW 周期**。此说明补充原 v7 报告，不改写其已绑定字节或扩大正式实验许可。

原始 `promt/stageF0-4.md` 是 GB18030 编码。第60行明确授权“必要时进一步读取对应原始日志、模型代码、checkpoint 和其他阶段报告。”第356行描述 Runner 保存 checkpoint 的职责；它本身不解除审核与放行条件。结合用户继续指令，v7 中宽泛的“Do not access real research/test data”应按本范围解释为不读取训练/测试样本、不反序列化或加载 checkpoint、不执行模型；不应禁止必要的 SHA-only 历史完整性校验或源码导入。

2. **逐项核验**

| 要求 | 实际证据 | 结果 |
|---|---|---|
| 历史 checkpoint 读取只有 hash | `experiments/stage_F0_migration.py:470–473` 调用 `sha256_file`；`stage_F0_worker_control.py:171–176` 只读 `rb` 并逐块更新 SHA256 | 符合 |
| 冻结 H49 identity | 唯一 `1f04c7fb623d18f940ac/attempts/attempt-00/resume_epoch_017.pt` 独立流式 SHA256 `8843afe2a382b47b7a920986ffbfc0cf996a73b0ab8822bfd138760b2517597c` | 与校验器要求一致 |
| 导入入口不执行 main | `stage_F0_run.py:2681` main guard；隔离副本实际 `import stage_F0_run` | PASS |
| 模型/数据/加载/CUDA 不执行 | 实际 guarded import 阻断 nn.Module 构造、torch.load/save、pandas 读取、numpy.load、CUDA 初始化/可用性探测；导入完成，零拦截事件，CUDA 未初始化 | PASS |
| 隔离/只读 | 普通复制 `.py`，不复制样本/checkpoint/artifact；审计禁止打开原生产根、样本目录与 PT/CSV/Parquet，禁止外部写入/子进程/网络连接 | PASS |
| 现有审查绑定 | 本次实际导入副本的 runtime/runner/worker-control/migration SHA 与当前源一致，原 v7 report/evidence SHA 再核验 | 符合；详见 evidence |
| 生产保护 | 本次只读重验 19 项 v7 保护 SHA，全部一致；无生产写入 | 符合 |
| B6 / 完整 qualification | 未执行 E2E、训练或最终 qualification；历史 incident 缺项仍在 | OPEN/FAIL；不闭合 |

3. **Bugs/deviations**

未发现此必要只读范围的 implementation blocker。原 v7 数据访问禁语过宽，本补充按用户明确授权作解释性澄清，不修改实现或历史审查结论。普通 torch 导入允许导入依赖与进程内环境设置；这与调用 CUDA、模型训练或 checkpoint 加载有明确边界。

4. **Corrections made**

仅新增本报告及证据。没有修改代码、测试、生产文件或原 v7 报告。独立 probe 位于 `/tmp/f0-v15-readonly-scope-review-20261014-a/scope_probe.py`，raw `/tmp/f0-v15-readonly-scope-review-20261014-a/archive/scope_probe.stdout.txt`，结构化结果 `/tmp/f0-v15-readonly-scope-review-20261014-a/archive/scope_probe.json`。使用项目 Python、任务槽 wrapper，峰值一个项目 Python 进程，exit 0，实际加载22个 repository 模块。未运行 pytest/E2E。

5. **Remaining risks**

这不是完整历史 migration/gate 校验或 fake/sentinel E2E结果。守卫证明此次导入路径没有触发禁止行为，不能替代之后执行 fake runner 时的分支隔离与写入路径审核。剩余 DESIGN §§8/10 qualification 证据仍待单独 RUN；B6 历史 pre-edit inventory/source-preimage 缺失仍保持 OPEN/FAIL。不能将必要历史 checkpoint SHA检查延伸为任意 checkpoint加载、真实数据读取、模型构造或 forward许可。

6. **Whether experiments are safe to run**

**本次必要 SHA-only 历史证据读取与隔离入口导入：允许，由当前 v7 qualification precheck 覆盖。单独 bounded isolated non-training qualification RUN：保持 v7 precheck许可；本次没有执行。科学训练、生产 RUN、principal release：仍 NO-GO；B6 OPEN/FAIL、生产 hold保持。**

证据文件 `experiments/reports/stage_F0_v15_qualification_readonly_scope_review_v1_evidence.json`，SHA256 `6704a51d81032fa11748daf61cfeb866e9ed2857aabd23fe754e39aac805ca8f`。
