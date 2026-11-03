# Coder implementation report

Implemented `experiments/conditional_information_momentum_20261102_v3/` against DESIGN_SPECIFICATION. Eight method arms, source A0/A1 and lambda0 bypass, true tower logits, observed within-domain pair sampling, ECCM/CV/collapsed-history updates, independent pair streams, fixed EMA, RAM restoration, and all selected/E10/E50 endpoints are wired into a staged command-line runner.

The source byte manifest pins the independent vendor to `38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27`. Adaptation code is outside vendor. Original CSV paths and source manifest comparisons are passed explicitly. The runner uses only seeds 2022–2026, development 2022/2024, 50 epochs and batch40960. No checkpoint files are written. Formal stage decisions require orchestrator-authored dispatch/decision JSON; this implementation does not infer scientific promotions.

CPU checks: 14 tests pass; all eight arms reproduce the next update after a BytesIO state roundtrip; lambda0 matches the source model/gradients/EMA through two updates; exact finite-system expectations, support, weights, hand-computed decay/history updates, active/selected clocks, empty conditions, None and zero gradients are tested. `CPU_TESTS.xml` records results. Compileall and the 24 unique initial-job check passed. Real full-batch S0 and formal training await independent review and runner execution.

No changes to historical files and no research-result analysis were made. The main agent owns interpretation, dispatch decisions, CSV aggregation/reporting and publication. A failed worker leaves its latest epoch state in RAM; its failure/logs require inspection before an explicit retry at the same batch and configuration. Loss of the RAM owner requires re-execution from seed.

## Commands

Use `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python` as PY below and set ATTEMPT to an absolute new artifact directory. REVIEW is the independent review JSON, containing verdict `PASS` or `PASS WITH FIXES`, empty `blocking_issues`, and `source_hash` matching `IMPLEMENTATION_SOURCE_MANIFEST.json`.

```sh
$PY -m pytest -q experiments/conditional_information_momentum_20261102_v3/tests
$PY -m experiments.conditional_information_momentum_20261102_v3.run stage0 --attempt "$ATTEMPT" --review "$REVIEW" --device cuda:0
$PY -m experiments.conditional_information_momentum_20261102_v3.run initial --output "$ATTEMPT/S1_DISPATCH.json"
$PY -m experiments.conditional_information_momentum_20261102_v3.run schedule --attempt "$ATTEMPT" --review "$REVIEW" --dispatch "$ATTEMPT/S1_DISPATCH.json" --devices 0,1
```

Subsequent dispatch JSON is `{stage: S2|S2_EXTRA|S3, approved_by: orchestrator, jobs: [...]}`. Construct entries with `run.job(dataset, arm, seed, stage, lambda_pair=value)`. Existing logical identities omit stage so S3 can reuse matching completed development trajectories. Source hashes are part of identity.

After all triggered stages, final decision JSON must include `approved_by: orchestrator`, `training_complete: true`, and the recorded stage choices/reasons. Freeze and test:

```sh
$PY -m experiments.conditional_information_momentum_20261102_v3.run freeze --attempt "$ATTEMPT" --review "$REVIEW" --decision "$ATTEMPT/FINAL_DECISION.json"
$PY -m experiments.conditional_information_momentum_20261102_v3.run evaluate --attempt "$ATTEMPT" --review "$REVIEW" --devices 0,1
$PY -m experiments.conditional_information_momentum_20261102_v3.run shutdown --attempt "$ATTEMPT"
```

`status` returns RAM owner job states. `runs/<key>/result.json` contains selected/E10/E50 validation, epoch/hash identities, costs and counts; `test_endpoints/` contains the six final endpoint metrics per logical trajectory. All epoch and batch traces remain in `runs/<key>/attemptN/`. Stage0 supplies actual data manifests and peak memory admission measurements. Scientific CSV/report generation is based on these scalar records and is owned by the orchestrator.

## 2026-11-03：P01/P10补充执行配置

按 DESIGN_SPECIFICATION 第11节及 `buchong.md` 完成最小补充配置，没有修改任何训练、评价或vendor代码。执行source hash保持 `5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a`，与原独立审查和实现清单一致。原发布分支 `26-11-02_v3`，原发布commit `eccb94e9986bb2ddf877e0808ddae1ffc608588a`。

独立目录为 `artifacts/conditional_information_momentum_20261102_v3/supplement_20261103/`。其中 `S2_DISPATCH.json` 使用原 `run.job` 生成且仅含8个唯一任务：P01/P10 × MovieLens/KuaiRand × seed2022/2024，lambda=.05、50epoch、batch40960。P01映射为静态ECCM，P10映射为动态普通Adam。S2是既有runner接口标签；本次研究角色为原停止后的独立补充，未追溯修改原晋级或停止决定。

启动前 `CONFIGURATION_FROZEN.json` 已冻结完整8个配置、源身份、原分支/attempt、证据来源和执行边界：不重训A0/P00/P11、不搜索、不自动S3；新8条完整训练后再冻结全部新选点，随后统一48个test逻辑端点。未预写 `training_complete` 或选点结果。

原 `attempt1/STAGE0_REPORT.json` 已原字节复制为新目录的同名文件，SHA256 `d0c4a59a5c791cf9f0d47413efed936341d970bfa3cb66a38831370e1462b9cf`。这复用原PASS和峰值准入数据，本次未重新运行S0；原14项CPU检查也直接复用。source setup仍为每worker `torch.set_num_threads(1)`。

`SOURCE_BASELINE_IDENTITY.json` 记录实际核对结果：A0、P00 .05、P11 .05的12条旧逻辑轨迹逐一与原配置冻结、选点冻结、result及test endpoint关联；数据raw/六数组/原行号/encoder/scaler/域计数/分割大小均匹配固定source manifest。源代码身份相同保持初始化、loader、pair流、EMA和选点规则。旧 `tables/endpoints.csv` 的144行validation/test × online/EMA × selected/E10/E50与原数值工件进行7,344个标量逐值精确比较，全部一致；状态hash和epoch也匹配。旧基线不需要本次模型前向，未重放旧轨迹。

本次配置检查通过：8个job无重复且不属于旧S1 job，种子/配置/方法映射正确。dispatch SHA256为 `841253b0b996790b6bdcfaec20da59d4d6d9a53e403076a00ad2deea06ca1295`；启动前配置SHA256为 `4dfd54d45b37ab2032df2b904f9bab078a88217a8197fa13ce48bfe6e066c2d1`。未启动guardian、GPU训练或test。待另一reviewer通过补充审查后，runner使用以下命令：

```sh
PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
ATTEMPT=/home/xzf/Code/phase0-3/artifacts/conditional_information_momentum_20261102_v3/supplement_20261103
REVIEW=/home/xzf/Code/phase0-3/research_reports/2026-11-02_mdmt_information_momentum_v3/INDEPENDENT_REVIEW_RECEIPT.json
$PY -m experiments.conditional_information_momentum_20261102_v3.run schedule --attempt "$ATTEMPT" --review "$REVIEW" --dispatch "$ATTEMPT/S2_DISPATCH.json" --devices 0,1
```

8条全部完成后，协调者才写 `FINAL_DECISION.json`，含 `approved_by=orchestrator`、`training_complete=true`、`study_role=independent_post_stop_supplement`、`S3=NOT_AUTHORIZED`、`hyperparameter_search=false`。再依次执行：

```sh
$PY -m experiments.conditional_information_momentum_20261102_v3.run freeze --attempt "$ATTEMPT" --review "$REVIEW" --decision "$ATTEMPT/FINAL_DECISION.json"
$PY -m experiments.conditional_information_momentum_20261102_v3.run evaluate --attempt "$ATTEMPT" --review "$REVIEW" --devices 0,1
$PY -m experiments.conditional_information_momentum_20261102_v3.run shutdown --attempt "$ATTEMPT"
```

仅追加本实施记录和既有设计映射，没有新建报告Markdown；运行配置和身份JSON均位于独立补充attempt。
