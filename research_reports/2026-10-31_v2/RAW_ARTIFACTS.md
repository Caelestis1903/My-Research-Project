# GA31-R3 数值与日志工件

收录632个文件，来自完整正式attempt、原batch工程验收、首启动中断和恢复audit。包含逐epoch/逐cell、batch行序、视图/优化器/时间/多视图/实际Adam反事实、来源/数据身份、冻结选择与决策、split行号、stdout/stderr及失败记录。模型权重、Adam/RNG快照均只在RAM中使用；归档未包含原始CSV数据集。

正式结果：45×50epoch，135条test记录，120个在线状态、60个离线观察。review_checks/中的合成验收另见审查报告。

| 分卷 | 字节 |
| --- | --- |
| [raw/GA31-R3-artifacts.tar.gz.part000](raw/GA31-R3-artifacts.tar.gz.part000) | 2,097,152 |
| [raw/GA31-R3-artifacts.tar.gz.part001](raw/GA31-R3-artifacts.tar.gz.part001) | 2,097,152 |
| [raw/GA31-R3-artifacts.tar.gz.part002](raw/GA31-R3-artifacts.tar.gz.part002) | 2,097,152 |
| [raw/GA31-R3-artifacts.tar.gz.part003](raw/GA31-R3-artifacts.tar.gz.part003) | 2,097,152 |
| [raw/GA31-R3-artifacts.tar.gz.part004](raw/GA31-R3-artifacts.tar.gz.part004) | 2,097,152 |
| [raw/GA31-R3-artifacts.tar.gz.part005](raw/GA31-R3-artifacts.tar.gz.part005) | 2,097,152 |
| [raw/GA31-R3-artifacts.tar.gz.part006](raw/GA31-R3-artifacts.tar.gz.part006) | 849,046 |

拼接后的gzip SHA-256：fad352aee864c02c1208fa4bf01a48be4ecf498c31edcf5657a70f29fd6061cf。

```bash
cat research_reports/2026-10-31_v2/raw/GA31-R3-artifacts.tar.gz.part* > /tmp/GA31-R3-artifacts.tar.gz
sha256sum /tmp/GA31-R3-artifacts.tar.gz
tar -xzf /tmp/GA31-R3-artifacts.tar.gz -C .
```

[RAW_ARTIFACTS_INDEX.json](RAW_ARTIFACTS_INDEX.json)保存每个源文件和分卷的大小及SHA-256。解压后按REPORT.md的命令复算数值与图表。
