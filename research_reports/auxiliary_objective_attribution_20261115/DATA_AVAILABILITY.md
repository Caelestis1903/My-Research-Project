# 数据与日志

仓库保存代码、全部CSV、逐seed/逐epoch结果与身份收据。模型、optimizer和checkpoint没有磁盘副本。

Stage1的逐批支持JSONL.gz共5,774,198,156 bytes，原件位于服务器工作区 `/home/xzf/Code/phase0-3_auxiliary_20261115/artifacts/auxiliary_objective_attribution_20261115/attempt1/runs/`。每epoch JSON及报告训练表保留日志路径和SHA256；2000份原件已经独立核验。重新生成Stage1中依赖逐批日志的诊断时使用这些原件，全部导出数表可直接读取。

Stage2的500份compact逐批日志按轨迹打包为10个tar，payload保持原始gzip字节。[索引](../../artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1/batch_log_archives/INDEX.csv)列出dataset、seed、路径、大小与归档SHA256。500个成员已与运行期记录的日志SHA256逐一匹配。

在仓库根目录还原后，可运行代码README和STAGE2_ANALYSIS.py中的命令：

```bash
for archive in artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1/batch_log_archives/*.tar; do
  tar -xf "$archive" -C artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1
done
```

服务器保留所有原始日志。运行中RPC连接文件、编译缓存和数据cache不属于发布包。收尾释放记录见 [CLEANUP_RECEIPT.json](../../artifacts/auxiliary_objective_attribution_20261115/CLEANUP_RECEIPT.json)。
