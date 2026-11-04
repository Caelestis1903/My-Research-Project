# KuaiRand Gaussian领域覆盖

新增G-D0和G-D02，各运行2022–2026五seed、50epoch。原GA31-R3 Gaussian抽样、fixed1 M3oE、普通Adam及固定EMA保持；G-D2、固定EMA和三个发表模型直接导入已核对的A+B端点。算法合同见[设计](../../research_reports/2026-11-04_gaussian_domain_coverage/DESIGN_SPECIFICATION.md)。

从仓库根目录，使用项目环境：

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/gaussian_domain_coverage_20261104/tests -q
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gaussian_domain_coverage_20261104.run prepare
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gaussian_domain_coverage_20261104.run smoke --devices 0,1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gaussian_domain_coverage_20261104.run run --devices 0,1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gaussian_domain_coverage_20261104.run status
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gaussian_domain_coverage_20261104.run summarize
```

smoke/run要求报告目录的`INDEPENDENT_REVIEW_RECEIPT.json`含`verdict=PASS`或已解决blocking的`PASS WITH FIXES`、空`blocking_issues`和与`manifest`命令匹配的`source_hash`。默认attempt为`artifacts/gaussian_domain_coverage_20261104/attempt1`。

prepare核对原CSV、六数组/row_ids/encoder/scaler及缓存文件SHA，复用`/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache`只读数据。所有模型/Adam/EMA/loader/RNG和选点状态只经已有guardian在RAM保存，完整epoch提交ACK后可恢复。每卡独立按85%显存/功耗和峰值/RAM余量追加worker；不终止现有GPU任务，不降低batch。`resume`清除正常暂停标记后从RAM恢复；guardian/主机状态丢失时同身份重放并保留成本。

逐条训练完整50epoch后先封存validation选点，再评价selected/E10/E50的V/Test并释放该轨迹RAM。训练标量与日志留在attempt；汇总只读取封存标量，写新报告目录，保留online诊断与fixedEMA主结果。
