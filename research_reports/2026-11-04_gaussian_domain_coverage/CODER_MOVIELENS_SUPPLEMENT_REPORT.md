# CODER MOVIELENS SUPPLEMENT REPORT

状态：IMPLEMENT COMPLETE。按更新后的`buchong.md`直接接入既有结果；新增训练0条、新增seed0个。

## 实现与数量

单个报告目录脚本`supplement_movielens.py`核对MovieLens既有`GAUSSIAN_A1/fixed_ema`为`augmented_domain=0`，接入G-D0五seed及M3oE、HyperGate、AutoIFS各五seed。共20条既有训练轨迹、40个selected V/Test端点；原KuaiRand20条比较轨迹只作已有结果视图，原270端点及10条新增训练计数不变。

输出：`MOVIELENS_ENDPOINTS.csv`、`MOVIELENS_ABSOLUTE_SUMMARY.csv`、`MOVIELENS_PAIRED_SEED.csv`、`MOVIELENS_PAIRED_SUMMARY.csv`、`MOVIELENS_DECISIONS.json`、`MOVIELENS_SOURCE_AUDIT.json`、`G_D0_TWO_DATASETS.csv`和`.md`。所有表均使用seed2022–2026，SD采用ddof1，配对t95%采用df4系数2.776445。Macro/Traffic/Pooled独立导出；AUC等级、六格保护和LogLoss分别计算。

`build_report.py`仅新增补充说明及双数据集表链接，重新生成`REPORT.md`保留该入口。原训练实现、冻结C8源码、原CSV、旧原始记录均未修改。MovieLens中文科学分析和总台账由主agent负责。

## 已核对的身份

- 原自然人口：train800167，validation/test各100021；train域人数155949/318599/325619。原行序80/10/10、train-only预处理，六数组/划分row_id/encoder/scaler身份与A+B数据manifest一致，原MovieLens CSV SHA核对通过。
- fixed1 M3oE，Gaussian κ=.05、λ=.5，Adam lr=.01、coupled wd=1e-4，batch40960，50epoch，FP32；`H2`固定EMAρ=.9828205985452511。原GA31-R3局部Generator规则、loader seed+11000和Gaussian/model关键原语SHA与冻结来源一致。
- 五seed都有完整epoch1–50验证曲线及E50端点；按六格Macro AUC严格最大、精确并列最早重算selected，epochs依次23、20、25、23、22。V/Test同model hash，源frozen checkpoint、原test CSV、trajectory/config/source/selection身份逐项一致。
- 三个发表模型按同一自然人口及AUC选点评价协议接入；HyperGate使用train阶段，AutoIFS使用reuse阶段；各五seed完整50epoch的合法评价窗口保留。源端点文件SHA及原生封存端点数值均核对。
- 原KuaiRand`ENDPOINTS.csv`仍为270行，SHA与原`RUN_MANIFEST.json`一致。所有输入文件SHA、40行精确导入记录、20个selected状态身份保存在`MOVIELENS_SOURCE_AUDIT.json`。

## 必要检查与复算命令

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-04_gaussian_domain_coverage/supplement_movielens.py
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-04_gaussian_domain_coverage/build_report.py
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m py_compile research_reports/2026-11-04_gaussian_domain_coverage/supplement_movielens.py
```

脚本断言覆盖：固定超参/源身份、每组恰五seed、validation六格argmax与V/Test hash一致、原数据/源文件checksum、Macro及Traffic/Pooled聚合、性能分级阈值、配对seed对齐、均值复算和原270端点不变。命令全部通过。源身份重审读取本机已有历史标量及原CSV；已导出的40端点包含全部统计所需数值。

MovieLens G-D0原始五seedMacro AUC均值0.7793566024710291、SD0.0003933437933124143；Macro LL均值0.4958222650185651、SD0.0015500741710398687。双数据集表同时保留两者各自κ、weight_decay和EMAρ，不把KuaiRand超参套用于MovieLens。

未执行GPU训练或研究方案设计；等待独立review核对本补充代码、来源审计及数表。
