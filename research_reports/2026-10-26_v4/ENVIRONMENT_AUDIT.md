# 环境与启动核实

新分支 `26-10-26` 基于当前 main `8b87d3fc2ac393faa34cc870bc3befee0e8693ca`，独立 checkout `/home/xzf/Code/phase0-3-dfd-v4-git`。原 workspace 的 `.git` 缺 HEAD/config，保持原样；未覆盖已有源码、数据或工件。

`start.md` 使用 GB18030；已按用户要求重新读取修订版本。两 raw CSV SHA256 与本轮规范完全一致。Python 3.13 / torch 2.9.0+cu126 / CUDA 可用。GPU0 为当前空闲 RTX3090；GPU1 有既有 vLLM 服务，本轮不修改或终止它。并发按用户 start 第12条，在授权设备上显存与功耗均小于85%时继续入队。

当前 `/home` 约67GiB 可用，`/home_new` 约1.4TiB可用；大状态优先使用独立 `/home_new/xzf_mdmt_dfd_v4_20261026`。后续预算基于本轮保留的真实组件计时。

用户存储修订：旧 `/home_new` 模型目录由用户主动删除。本轮撤销此前的大状态存放建议，所有新临时状态仅位于本轮 artifact root 的 `_temporary`。模型及优化器档仅供当次必要训练/选点/验收，完成依赖评价后即删；最终保留日志、配置、指标、预测和数据信息。其他 `/home_new` 目录不做任何写入或删除。最终不保留模型权重，因而事后从已训模型再次推理需按代码重训；此项遵从用户最新要求，覆盖旧规范的永久 checkpoint 留存要求。
