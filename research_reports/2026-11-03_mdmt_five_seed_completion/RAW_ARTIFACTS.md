# 原始标量归档与复算

归档包含2468个文件：本轮30条新增训练的逐batch/epoch日志、数据/配置身份、30条result、180个test端点与全局冻结记录；10条历史候选复用所需原记录；AIGM01控制所需八份原始标量表与冻结身份。模型、optimizer与checkpoint状态未写入归档。

GitHub保存同目录 `raw_scalar_records.tar.xz.part000` 至 `.part012`，原文件列表、逐文件SHA256及分块校验值见[RAW_ARCHIVE_MANIFEST.json](RAW_ARCHIVE_MANIFEST.json)。拼接后为26304180字节，SHA256：

```text
b14aca64bd1ec48a3cb3a3488ce5a4918e3302b4a01277da5468f167f4fa7f02
```

在已检出的仓库根目录运行；压缩包内路径均相对仓库根目录：

```sh
cat research_reports/2026-11-03_mdmt_five_seed_completion/raw_scalar_records.tar.xz.part* > research_reports/2026-11-03_mdmt_five_seed_completion/raw_scalar_records.tar.xz
(cd research_reports/2026-11-03_mdmt_five_seed_completion && sha256sum -c raw_scalar_records.sha256)
tar -xJf research_reports/2026-11-03_mdmt_five_seed_completion/raw_scalar_records.tar.xz
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_five_seed_completion_20261103.run summarize
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-03_mdmt_five_seed_completion/analyze.py
```

复算只读取标量，不重新训练。main中已包含新编排包及其原history/pair源码依赖；原冻结计划也已存在。`summarize`输出660行端点、1080行逐seed配对和5500行validation epoch数据；`analyze.py`输出主结果表、selected逐seed六格附录、台账片段及两张图。GATS B2历史重算原值单独保存在[GATS_B2_HISTORICAL_PAIRED.csv](GATS_B2_HISTORICAL_PAIRED.csv)，来源见报告第6节。

直接浏览数据可用[endpoints.csv](endpoints.csv)、[paired_seed.csv](paired_seed.csv)、[paired_summary.csv](paired_summary.csv)、[validation_epochs.csv](validation_epochs.csv)；单条选点的epoch、model hash、配置及来源见端点和[RUN_MANIFEST.json](RUN_MANIFEST.json)。
