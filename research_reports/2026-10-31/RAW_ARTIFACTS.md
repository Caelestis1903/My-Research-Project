# GA31-v2 数值工件

压缩档共7,515,220字节、4片，包含480份正式/验收/开发数据记录和日志。保留完整batch/epoch/cell/augmentation/optimizer/probe记录、数据身份和split row_id；不含原始CSV或模型状态。

| 分片 | 字节 |
| --- | --- |
| [raw/GA31-v2-artifacts.tar.gz.part000](raw/GA31-v2-artifacts.tar.gz.part000) | 2,097,152 |
| [raw/GA31-v2-artifacts.tar.gz.part001](raw/GA31-v2-artifacts.tar.gz.part001) | 2,097,152 |
| [raw/GA31-v2-artifacts.tar.gz.part002](raw/GA31-v2-artifacts.tar.gz.part002) | 2,097,152 |
| [raw/GA31-v2-artifacts.tar.gz.part003](raw/GA31-v2-artifacts.tar.gz.part003) | 1,223,764 |

在仓库根目录恢复数值工件：

```bash
cat research_reports/2026-10-31/raw/GA31-v2-artifacts.tar.gz.part* > /tmp/GA31-v2-artifacts.tar.gz
tar -xzf /tmp/GA31-v2-artifacts.tar.gz -C .
```

文件路径与大小见 [RAW_ARTIFACTS_INDEX.json](RAW_ARTIFACTS_INDEX.json)。正式矩阵为40run×50epoch，test/validation各120行；232个probe batch。报告可直接读取analysis目录，无需解包查看汇总。
