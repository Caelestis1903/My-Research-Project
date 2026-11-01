# 数据加载中断与恢复记录

首次正式入口 `attempt_20261031_230801` 在MovieLens身份校验通过后、KuaiRand预处理期间出现 `free(): invalid pointer`，进程以134退出。该attempt没有launch ledger，未启动训练worker，正式optimizer更新为0。

最小复核使用同项目环境、同入口和同冻结配置，仅运行data-only audit：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -X faulthandler \
  -m experiments.gradient_aug_20261031_v2.run --mode audit \
  --run-dir artifacts/gradient_aug_20261031_v2/GA31-R3/native_abort_audit1
```

复核exit0，两个数据集全部raw/array/split/encoder/scaler身份通过，stderr为空。日志保存为 `GA31-R3/native_abort_audit1.stdout.log`、`native_abort_audit1.stderr.log`；数据manifest保存在对应audit目录。环境为项目Python3.13.5、NumPy2.1.3、pandas2.2.3；当时可用RAM约49GiB。

目前记录为一次未复现的native allocator中断；这次audit未改代码、环境、数据或算法，未执行训练或test指标。恢复方式为保留中断attempt，以同配置新attempt启动完整RAM生命周期，并启用`-X faulthandler`保留后续native栈。没有为这次中断修改冻结预处理语义。
