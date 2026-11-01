# GA31-R3 实施报告

依据已批准的 [DESIGN_SPECIFICATION.md](DESIGN_SPECIFICATION.md) 实现。正式训练尚未启动，待独立审查通过后由runner执行。

## 实际修改与来源

仅新增 `experiments/gradient_aug_20261031_v2/`。`vendor_ga31/` 的17个必要代码/配置文件逐字节复制冻结提交 `ee6e81fdfd37284e6447e8149cb93d04736dba02` 的源包，未修改历史源文件。来源路径、Git blob和SHA256见包内 `SOURCE_MANIFEST.json`；协调者的来源下载收据保持原样。

- `protocol.py`：新R3合同解析，继承原数据与模型身份；A/B臂、SHA256随机键、validation-only基线选择、单独ANTI方差门和35–45run展开。
- `views.py`、`objectives.py`：保持参数键/顺序的fixed1模型，显式Gaussian/单字段零向量输入视图；K视图始终使用原始B×T分母，正式每batch一次backward/Adam.step。IID2与ANTI2首视图配对，ANTI第二视图取负，无额外抽样。
- `probe.py`：真实A轨迹上120个预注册状态的在线只读诊断；所选普通基线的E1/E10/selected离线多视图诊断，14次梯度/5次反事实step每锚点-batch；两不交叠validation panel。源状态/已有梯度/Adam/loader/RNG核对不变，全部向量仅在内存中计算。
- `run.py`：A15→冻结基线→所选基线离线probe→冻结ANTI门与B任务→M1/IID2必做和ANTI条件臂→全体selected冻结→统一test。自然采样、50epoch、原batch40960；复用逐GPU显存/功耗调度及RAM BytesIO/Pipe状态传递。A每epoch最后一个真实batch记录L2/epsilon/更新能量摘要。状态、初始化、顺序和视图身份另存轻量摘要。
- `analyze.py`：seed配对效应、均值/样本SD/df4描述性t区间、保护规则和决策；汇总increment/temporal/counterfactual数值。只处理数字，不作机制研究结论。
- `checks.py`、`tests/test_semantics.py`：与本轮比较直接相关的底座、视图、目标、Adam、RNG、RAM恢复、时间探针和门逻辑检查。

模型实际前向为 `forward(x,test_flag=False,*,view=None,kappa=.1)`。四融合系数均为1；无learned fusion/DARTS、新网络、额外正式优化器或第二阶段候选。ML/KR参数量分别773470/1614254。新入口不调用旧40run parser/selection断言。未新增依赖。

## 必要检查

Python统一为 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

1. `python -m compileall -q experiments/gradient_aug_20261031_v2`：通过。
2. `python -m pytest -q experiments/gradient_aug_20261031_v2/tests/test_semantics.py`：**4 passed**。覆盖同state固定源预测/loss/梯度/实际Adam一步、κ0/λ0/关闭路径、B×T预算、单例/缺目标、domain/dense保护、MASK整段删除、反向噪声、独立RNG、coupled-Adam完整尺度映射及None语义、RAM epoch边界恢复、四步online on/off轨迹、方差恒等式、L2优先、ANTI先平均后取比、缺seed不触发及35/40/45run预算。
3. `python -m experiments.gradient_aug_20261031_v2.run --mode check --run-dir artifacts/gradient_aug_20261031_v2/GA31-R3/implementation_check1 --device cuda:1`：**PASS**。两个数据集raw/六数组/split/encoder/scaler身份全部匹配；原batch40960下固定源预测/梯度/Adam一致；M1/IID2/ANTI2各一次真实CUDA更新可容纳；每数据集四个真实原batch启用/关闭在线探针后模型、Adam、梯度、loader及全局RNG完全相同。未评价test。

CUDA检查峰值：ML M1为993158144字节，IID2/ANTI2为1054036480字节；KR M1为1113453568字节，IID2/ANTI2为1112757248字节。完整结果见 `implementation_check1/checks.json`，时间探针记录在其 `trajectory/`。

工程开发仅seed2022：每数据集累计13次optimizer更新（源/新baseline各1、三视图臂各1、online on/off各4），ML预算上限20、剩余7；KR上限49、剩余36。CPU合成语义检查不使用真实test。审查时可复用这些真实开发证据，不重新消耗已用预算。

验收中发现参考测试最初用单层BCE均值，而Sample底座采用先任务后样本的两层均值。两种FP32求和次序会在Adam epsilon敏感坐标放大差异；已将测试oracle统一为冻结源的 `reduction='none'.mean(1).mean()`，未更改冻结训练目标或放宽容差。正式数据配置未因开发效果修改。

## 接口、工件与运行

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_20261031_v2.run \
  --mode all --protocol research_plans/26-10-31_v2/phase1_protocol.json \
  --run-dir artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_<timestamp> --devices 0,1
```

`--mode audit|check|all`；all必须在同一coordinator生命周期运行。ML基线只读取A阶段selected validation，ANTI门只读取预注册训练probe，M1/IID2不受门影响。全部A/B selected与共同E10/E50冻结后才统一test；中间不因seed效果删臂。

保留配置/来源/数据manifest、split行号、初始化/模型状态摘要、launch/run/failure ledger、epoch/per-cell/batch/view/optimizer尺度、increment/temporal/反事实、baseline/candidate/selection冻结、配对表及DECISIONS。`runtime_stats.json`保留现有run成本字段，`probe`在训练run中记录online计数；独立离线probe成本写入 `probes/` 及根 `probe_ledger.json`。根聚合CSV便于报告复算。

所有model/Adam/RNG/loader状态只保留RAM，BytesIO序列化不写文件；数值表生成后删除状态引用并记录释放。源码导出目录没有本地Git元数据；来源以固定远端commit/blob记录，发布由主协调者完成。

当前未解决阻塞：无。未执行正式all、未查看新研究性能、未编写科学结论。独立reviewer仍须检查设计、实际实现和本报告并给出裁决。
