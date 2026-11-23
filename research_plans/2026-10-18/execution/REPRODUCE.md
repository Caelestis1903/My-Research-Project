# 复现入口与原始依赖

代码位于仓库 `experiments/n18_*.py`。本机 Python 为
`/home/xzf/miniconda3/envs/phase0-3-env/bin/python`；版本和 GPU 见
[`provenance.json`](provenance.json)。`PYTHONDONTWRITEBYTECODE=1` 避免修改仓库历史 pyc。

先准备历史真实缓存与 checkpoint（它们未伪装成已上传 GitHub 的原始数据）：

```text
N18_HISTORY_ROOT/
  local_cache/kuairand_2022.pkl
  local_cache/movielens_2022.pkl
  runs/V4final__kuairand__adam__SN__seed2022/epoch010.pt
  runs/V4final__kuairand__adam__SN__seed2023/epoch010.pt
  runs/V4final__kuairand__adam__SN__seed2024/epoch010.pt
  runs/V4final__kuairand__adam__SN__seed2025/epoch010.pt
```

默认历史根目录是本机 10-16-v3 checkout 的 execution_20261016；其 runs
指向 `/home_new/xzf_phase0_3_v3_artifacts_20261016`。checkpoint 的创建源码
提交见 provenance；KR/ML 缓存保留原有编码器与人口，不能重新生成另一
随机 subset 替代。ML 时间检查还读取缓存 metadata 的原始 ml-1m.csv。
所有原始特征和新头预测已经随本轮分支保存，P3 不必取得原大数据也可重拟合。

已交付的证据可直接读取。若全程重算，指定**全新**输出目录，脚本不覆盖
现有实验。以下为运行顺序，先阅读设计/修订/自审，再执行；主分析不委托
子 agent，只有 head fitting 按用户入口交给 `luna_worker` 机械运行。

```bash
export N18_OUTPUT_ROOT=/tmp/n18-reproduction
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
PYTHON=/home/xzf/miniconda3/envs/phase0-3-env/bin/python
mkdir -p "$N18_OUTPUT_ROOT"
"$PYTHON" experiments/n18_theory_data.py
"$PYTHON" experiments/n18_p5_decision.py
```

四个种子可按每卡最多 5 进程、独立显存不超过 90% 的规定分配，待执行任务
只有四个时无需为占满显卡扩展实验。实际分配 2022/2024→cuda:0，
2023/2025→cuda:1。下面演示串行命令；本轮实际四个测量任务并发执行。

```bash
for seed in 2022 2023 2024 2025; do
  gpu=$((seed % 2))
  "$PYTHON" experiments/n18_probe.py --seed "$seed" --device "cuda:$gpu"
  "$PYTHON" experiments/n18_fit_heads.py --seed "$seed"
  "$PYTHON" experiments/n18_probe.py --seed "$seed" --device "cuda:$gpu" \
    --skip-features --out "$N18_OUTPUT_ROOT/measurements_functional/seed$seed"
  "$PYTHON" experiments/n18_marginal_cost.py --seed "$seed" --device "cuda:$gpu"
done
"$PYTHON" experiments/n18_readout.py
"$PYTHON" experiments/n18_finalize.py
```

当前 probe 已包含完整函数 Jacobian 修订，因此第一次默认输出的梯度也已
修正，P3 特征不变。原执行中修订前版本保存在 archive，正式读数采用
measurements_functional。无需重复旧版本来复现正式结论。单次计时受 GPU
并发和缓存影响，预期数值会不同；它不是稳定吞吐保证。

若只复算 P3，把已发布 `measurements/seed*/features_*.npz` 放到新的输出根
同结构下，再调用 `n18_fit_heads.py`。若只复算表格，复制正式 measurements、
measurements_functional、P3_training 与 theory_data，调用 readout/finalize。
读取旧 data/checkpoint 不会把它们变成本轮新训练。

P1/P2/P5 的理论判据、P3 等价和 P6 IS 公式可直接由独立推导检查，
不依赖某次 GPU 计时。没有运行的 SAF/D3/AutoIFS/DSMTL 等完整基线
不在命令中，也不在本轮性能胜负声明中。
