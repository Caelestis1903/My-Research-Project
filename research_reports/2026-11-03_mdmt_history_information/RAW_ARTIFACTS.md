# 原始标量记录

归档包含本轮全部原始配置、逐epoch/诊断记录、日志、冻结清单、test端点，以及重算报告所需的8条固定历史A0/A1原始run目录和原final_evaluation.json。共1335个JSON/JSONL/日志文件，不含模型、optimizer、梯度、checkpoint或guardian认证文件。

完整归档大小3,544,788字节，SHA256：

`f6bb9bdf8cb23d7e467d1f65562adeff26a86dbe8d508c22aa7accce997ae1a1`

仓库存储4个有序分片：[000](raw/scalar_records.tar.xz.part000)、[001](raw/scalar_records.tar.xz.part001)、[002](raw/scalar_records.tar.xz.part002)、[003](raw/scalar_records.tar.xz.part003)。逐分片及逐文件hash见[RAW_ARCHIVE_MANIFEST.json](RAW_ARCHIVE_MANIFEST.json)。

在仓库根目录恢复：

```bash
cat research_reports/2026-11-03_mdmt_history_information/raw/scalar_records.tar.xz.part* > /tmp/mdmt_history_information_scalar_records.tar.xz
sha256sum /tmp/mdmt_history_information_scalar_records.tar.xz
tar -xJf /tmp/mdmt_history_information_scalar_records.tar.xz
```

解压路径以`artifacts/`开始。历史来源内容保持原字节；本轮CSV直接在报告目录，查阅或运行analyze.py不需要解包。要从原始epoch重跑run report，则先恢复归档。原始来源收据保留执行位置`/home/xzf/Code/phase0-3`的绝对路径；异址重算需将provenance中的result_path/epoch_directory映射到恢复后的同一相对文件，不能改变科学身份或指标。

本轮新增13条轨迹650epoch，没有失败重放。导出报告包含8条历史引用，D0实际比较使用其中seed2022的4条。历史引用不计新增训练或独立重复。模型状态已在冻结后的统一test完成后释放；归档用于标量复算，不提供模型恢复。
