原始标量记录、逐轮日志、F收益/fold数组及实测关系表共2235个文件，以有序分片发布。完整逐行预测、Q/E收益数组、标签访问日志和前缀缓存的本地位置见 `PUBLISH_DATA_INDEX.json`。

下载所有 `RAW_RECORDS.tar.gz.part00` 至 `part12` 后，在同一目录执行：

```bash
cat RAW_RECORDS.tar.gz.part* > RAW_RECORDS.tar.gz
tar -xzf RAW_RECORDS.tar.gz
```

合并文件为39,214,150字节；内容路径以 `artifacts/mdmt_basp_v1/` 开头，正式矩阵位于其 `attempt2/`。失败attempt和资源重调度记录各自保留原路径，不计为独立seed。
