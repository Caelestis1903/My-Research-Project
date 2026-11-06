# CODER IMPLEMENTATION REPORT — protocol / panels

实现范围为已批准 `DESIGN_SPECIFICATION.md` 的协议、数据身份与固定面板部分。代码位于 `experiments/ema_mechanism_20261106/`；已有训练源码未修改。

## 文件与覆盖

| 文件 | 实现责任 |
|---|---|
| `protocol.py` | 接管 core 初版并完成 `ResolvedStudy`、20-job/五seed/epoch/EMA clock校验、16份源码 SHA256 与 Git blob 核对、计划与实现hash、原始CSV和完整缓存身份、preflight、来源收据和 S0 callback |
| `panels.py` | SHA256独立随机流、互不交训练行池、P_A/P_B排名与自然面板、Validation固定面板与跨seed稳定行侧、无放回固定pairs、身份NPZ保存/核验/回读 |
| `tests/test_protocol.py` | 冻结矩阵拒绝错误seed/rho；NPY流式摘要与源数组、原始行指纹语义相等 |
| `tests/test_panels.py` | RNG不变、P_A/P_B行不交、标签/pair正确与可复现、身份回读/重复幂等/冲突拒绝、缺类与配额不足分支 |

协议直接引用源数据构建器、bundle、布局与组映射。没有调用旧A/B/C resolver。全部20个 job 保留 manifest job_id，补充与 observer 无关的 online trajectory 摘要及源/数据/布局/组/面板/实现身份。

缓存核验逐项覆盖原始CSV SHA、cache字节 SHA、实际 NPY 内容摘要、原始行ID摘要、split有序联合指纹、encoder/scaler与自然域人数。Test仅以文件字节流核对内容和元数据，未创建Test数组或提供Test标签给训练/分析。缓存缺失通过原 `prepare_cache` 重建；未封存或冲突的已有文件不覆盖。

固定面板从正常训练/Validation人口采样，不改变online人口。P_A/P_B先按源row ID哈希分池，所有排名/自然角色都只使用所属池。Validation行侧使用不含seed的 `SHA256(dataset|row_id)` 最低位。训练每cell排名样本、自然行、Validation行数与pairs由冻结manifest提供；缺类的Validation pair以 `pair_counts=0` 明确表示，填充槽位-1不作为有效pair。

## 公共接口

- `resolve_study(manifest, paths=None)` 返回设计规定六字段；`.paths`包含manifest绝对路径。
- `preflight(study,out,cache_root=None,ml_csv=None,kr_csv=None)` 写身份产物与面板，并把manifest/cache_root/out绝对路径写回 `.paths` 和 `PREPARED.json`。后续独立CLI从PREPARED恢复冻结路径/job身份。
- `audit(study,out)` 写来源收据后调用 analysis coder 的 `audit_archives(study,out)` 生成完整归档重算。
- `build_panels(bundle,seed,spec)` 返回 `{P_A,P_B,V_mech}`；`save_panels`只持久保存行/标签/角色/pair身份，feature x仅驻内存。
- `load_panels(out,dataset,seed,bundle=None)` 核验 `PROBE_MANIFEST.json`、文件与数组hash；提供bundle时重建feature x。V在内存同时提供 `indices/row_ids/y/pairs` 和 `natural_* / rank_pairs` 兼容别名。

面板NPZ采用原子替换、禁object；一致重复保留原文件，冲突拒绝覆盖。没有参数、梯度、optimizer或checkpoint落盘代码。

## 基本验证

执行环境：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

```text
python -m pytest experiments/ema_mechanism_20261106/tests/test_protocol.py \
  experiments/ema_mechanism_20261106/tests/test_panels.py -q
4 passed
```

Resolver基本检查实际通过16份冻结source SHA/Git blob核对，生成20个唯一训练身份、34500总Adam预算。实际缓存NPY header检查显示两个数据集均为源float32 C-order及int64行号。

修订：独立测试初次收集因tests包尚未建立而无法使用相对导入，已改为完整模块导入；集成发现的manifest重复keyword已修正。修改后上述4项重新通过。

完整preflight、GPU smoke、20条科学训练与研究分析由后续 REVIEW/RUN/ANALYZE 执行；本 coder 未启动这些阶段。无本分工内未实现功能。
