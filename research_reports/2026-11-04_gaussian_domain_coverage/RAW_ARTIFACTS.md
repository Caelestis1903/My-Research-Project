# 原始标量与复算

归档raw_training_scalars.tar.gz包含1581个文件、6975676字节：500个正式epoch及逐batch标量、噪声/原行序hash、配置/数据/源码身份、选点与V/Test端点、成本、资源门禁与运行日志，以及实际复用的历史输入。每个文件及压缩包SHA256见[RAW_ARTIFACTS_MANIFEST.json](RAW_ARTIFACTS_MANIFEST.json)。

压缩包SHA256：10f5e19660347fbc774ca7b866f953e226c0650b7e2b089a78df3cb75d8fb4e3。

从本目录核对归档：

    sha256sum -c raw_training_scalars.sha256

解压到独立目录：

    mkdir -p /path/to/restore
    tar -xzf raw_training_scalars.tar.gz -C /path/to/restore

execution/保存新工作树下的相对路径，history/保存旧历史根目录下的实际复用输入。原绝对路径保留在身份记录中；两根目录对应关系在manifest的roots字段。数据数组缓存和guardian认证/claim文件未加入归档；模型、optimizer和EMA状态在运行期间驻RAM，评价后释放。

在仓库根目录用项目Python复算所有统计及报告：

    /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-04_gaussian_domain_coverage/analyze.py
    /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-04_gaussian_domain_coverage/build_report.py

analyze.py读取已发布的ENDPOINTS.csv，核对五seed、全部六格Macro及完整阶段矩阵，生成ANALYSIS_VALUES.json和STAGE_EFFECTS.csv；build_report.py读取冻结统计、配置裁定和成本，生成REPORT.md及LEDGER_APPEND.md。这两条命令直接使用数表。

训练实现与CLI见[实验包](../../experiments/gaussian_domain_coverage_20261104/README.md)，协议与完整身份见[RUN_MANIFEST.json](RUN_MANIFEST.json)，运行计数见[RUNNER_REPORT.md](RUNNER_REPORT.md)。
