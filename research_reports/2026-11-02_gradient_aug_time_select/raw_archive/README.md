# 完整数值日志归档

每组 `.tar.gz.partNNN` 按文件名顺序拼接为一个 gzip tar。所有原始数值文件、配置、hash、stdout/stderr均保留；未放入原始数据CSV或model/optimizer/shadow/RNG权重。B1的NPZ是逐样本row_id、global_step、probability数值日志。

单个大于100 MiB的文件会被GitHub阻止，因此使用标准tar/gzip和不超过80 MiB的分片，不引入LFS或外部下载依赖。[GitHub文件大小规则](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

`*.SHA256SUMS`核验各分片；`*.manifest.json`记录整个压缩包及每个原始文件的SHA-256、大小和恢复路径。归档脚本实际读回全部tar payload核验原始文件hash，不只检查压缩命令退出码。完整压缩包留在发布机`/tmp`，仓库仅保存分片和manifest。

从仓库根目录恢复失败attempt的示例：

```bash
cd research_reports/2026-11-02_gradient_aug_time_select/raw_archive
sha256sum -c failed_0005.SHA256SUMS
cat failed_0005.tar.gz.part* > /tmp/gats02_failed_0005.tar.gz
sha256sum /tmp/gats02_failed_0005.tar.gz
# 应为1466564a90393fe23f894618c1adfb3dbdcc2a66e72ee6ca0fe0c458472a3d85
cd ../../..
tar -xzf /tmp/gats02_failed_0005.tar.gz
```

`failed_0005`包含首次Stage0、基础开发attempt和外层日志：20条科学轨迹完成，但协调器在历史接口处退出，未统一test，RAM模型丢失。首个wrapper错误文件是当时状态消息的转录，原始stderr曾被覆盖，不能作为原始traceback；最终KeyError的原始stderr保留。恢复attempt的归档与收据另列，勿将两次执行当作更多独立seed。
