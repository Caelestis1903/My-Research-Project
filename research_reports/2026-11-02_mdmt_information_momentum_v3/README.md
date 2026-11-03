# MDMT 条件监督与动量实验

[实验报告](REPORT.md) · [完整数值附录](NUMERICAL_APPENDIX.md) · [证据审阅](EVIDENCE_REVIEW.md)

24条正式轨迹全部完成，144个测试端点评价完成。两个数据集均触发S1停止门，保留A0；S2/S3未触发。

代码位于 `experiments/conditional_information_momentum_20261102_v3/`。运行入口和项目Python环境见 [实施报告](CODER_IMPLEMENTATION_REPORT.md)。所有模型状态只驻RAM，评价后已释放。

## 原始证据恢复

在仓库根目录执行：

```sh
cat research_reports/2026-11-02_mdmt_information_momentum_v3/raw/raw_scalar_records.tar.xz.part* > /tmp/mdmt_information_momentum_v3_raw.tar.xz
tar -xJf /tmp/mdmt_information_momentum_v3_raw.tar.xz
```

归档恢复原 `artifacts/conditional_information_momentum_20261102_v3/attempt1/` 标量、配置和日志路径。SHA256和分片列表见 [RAW_ARCHIVE_RECEIPT.json](RAW_ARCHIVE_RECEIPT.json)。完整配对表以CSV.gz交付，解压后可直接读表。源码、报告和分片的Git blob身份见PUBLICATION_FILES.json；远程提交身份由最终发布回执记录。

重新汇总：

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-02_mdmt_information_momentum_v3/finalize_tables.py
```

研究结论与阶段决定不依赖恢复模型文件。原实验环境为Python3.13.5/PyTorch2.9.0+cu126，真实数据路径和身份见配置及S0记录。

## 2026-11-03补充机制实验

[原报告中的补充章节](REPORT.md#supplement-20261103)：固定λ=.05，新增P01/P10共8条50epoch训练、48个test端点；A0/P00/P11复用原冻结数值。原停止决定保持不变。完整补充表格和冻结身份在supplement/，本次仍提交同一26-11-02_v3分支。

补充原始证据恢复（仓库根目录）：

```sh
cat research_reports/2026-11-02_mdmt_information_momentum_v3/supplement/raw_scalar_records.tar.xz.part* > /tmp/mdmt_v3_supplement.tar.xz
tar -xJf /tmp/mdmt_v3_supplement.tar.xz
```

重新汇总：

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-02_mdmt_information_momentum_v3/build_tables.py --attempt artifacts/conditional_information_momentum_20261102_v3/supplement_20261103 --baseline research_reports/2026-11-02_mdmt_information_momentum_v3/tables/endpoints.csv --output research_reports/2026-11-02_mdmt_information_momentum_v3/supplement
```
