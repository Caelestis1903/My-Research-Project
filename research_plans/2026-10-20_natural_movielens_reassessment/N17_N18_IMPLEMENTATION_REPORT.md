# N17/N18/V2/AO04 IMPLEMENTATION REPORT

2026-10-20。设计依据 `N17_N18_DESIGN_ADDENDUM.md`。IMPLEMENT 分批交付；尚未启动任何正式训练，不作研究结论。

## 第一批：D-V2，可独立 REVIEW

- `experiments/natural_ml_extensions/v2.py` 调用原 `v2_20261016.Runner` 完整50epoch，原 fixed_model、Sampler、step、fit、validation/test顺序全部继承原实现；只注入统一核心自然bundle、source/output metadata。
- 完整16配置：Adam .01 / SGD .1 × SN/SB ×2022–25，WD1e-5，SGD momentum0，B40960，单optimizer，固定fusion。
- `source_archive/inventory.json` 逐文件记录原source_path、Git commit、SHA256；历史source按Git对象归档，未编辑。`bootstrap.sources` 每次核对归档完整性。
- 统一数据接口 `experiments.natural_ml_reassessment.data.load_population('movielens_natural')`。不复制人口生成逻辑。
- 检查：仅 `py_compile` 与无训练manifest生成成功。未调用正式fit；REVIEW通过后交luna-runner。
- 机械命令：`/home/xzf/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_ml_extensions.v2 --index N --device cuda:0`，N=0..15。每命令独立进程，输出在 `/home_new/xzf_natural_movielens_reassessment/V2/`。`python -m experiments.natural_ml_extensions.manifest` 只写队列。

## 后续批次

N17、AO04与N18离线诊断正在实现，不能把本批V2的review覆盖解释为其它组已可运行。

## V2 REVIEW 修正批（R1/R2/R3）

- R1已修：`bootstrap.dependency_inventory('v2')` 固定V2四个wrapper和全部V2归档，再加入core `__init__.py/bootstrap.py/data.py` 与实际core归档 `stage_F0_data.py/stage_S1_data.py/stage_S1_protocol.py`。core归档同时核对inventory SHA和git_blob_SHA，并要求原source_path/source_commit。所选records完整内容的canonical SHA进入source集合；无关N17/AO04/N18新增文件或inventory条目不影响该集合。
- 数据导入/装载前先核验归档；`actual_module_paths` 对已执行core模块的真实 `__file__` 逐项断言，并记录所有实际从V2/core加载的模块路径与SHA。run config和无训练queue manifest记录相同dependency_inventory与actual_module_paths。
- 核心coder已确认data identity metadata修复冻结；本修复未编辑core文件或历史归档。原fit每epoch的 `self.sources==sources()` 完整保留，现覆盖上述实际人口生产者。
- R2补充：原evaluate前后只读观察model/optimizer digest与step计数，test开始必须已有selection_frozen文件；写入每次test的UTC、freeze SHA及zero_updates证据。fit完成补充 `test_access_receipt.json`，不改原算法或历史pre-test snapshot字段。
- R3补充：config/queue/测试receipt标记exploratory_replication，明确历史test已反复使用；实际review文档绝对路径和SHA写入config。最终独立复审如另存文件，runner机械命令附 `--review-file <复审报告绝对路径>`。
- 基本检查：修复后py_compile成功；完整无训练manifest生成成功，core实际import路径断言通过。未创建Runner、optimizer或执行训练。等待独立复查，不据此自行放行。

### V2 receipt生命周期复查修正

原fit中的只读test observer现在只写 `test_access_progress.json`。最后一次test后该progress不再改变，原fit的run_manifest记录其稳定SHA。fit返回后首次新建最终 `test_access_receipt.json`，并首次创建独立 `execution_manifest.json`，引用原run_manifest SHA与最终receipt SHA。最终receipt与execution_manifest不被原manifest列入，因此没有先列入manifest后覆盖的SHA失效。历史fit/step/evaluate算法source不变。仅wrapper语法检查成功，未运行训练。
