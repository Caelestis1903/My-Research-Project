# CODER FIX2 REPORT — 标准 FP32 parity 与存活模型保留

日期：2026-10-27。状态：IMPLEMENTED_PENDING_INDEPENDENT_REVIEW。

KuaiRand四个seed在B_REFERENCE完成20轮后，均因跨batch部署parity的近零绝对误差检查退出。MovieLens四个worker仍持有RAM模型；它们继续使用实际已加载的旧诊断版本，不终止、不重写身份。本次按批准补充设计第5–8节只修复工程比较和必要的冻结兼容入口。

## 真实失败与最小修改

KR2022的batch256最大差为1.341104507446289e-6，cached=−0.07308383285999298、raw=−0.07308517396450043，旧允许值1.0950089827179907e-6。KR2023/2024/2025的最大绝对差位置本身可通过旧相对限值，但其他近零位置被逐元素规则拒绝；旧日志只给max_abs位置，不能从该位置反推整批通过。

仅`compare_raw_logits`采用本机PyTorch float32标准：atol=1e-5、rtol=1.3e-6。shape、finite规则和cached参考保持；在原max_abs证据之外记录`max_normalized_error=max(abs(raw-cached)/allowed)`及`worst_index`、该位置的真实差、两端值和允许值。该位置直接说明逐元素拒绝原因。原teacher/RAM恢复绝对1e-6、模型、训练、缓存、指标、预算和选点未改。

CPU分支回归覆盖原ML点、KR2022近零点、KR2024/2025日志位置通过；1e-4近零误差、明显错误、NaN/Inf和shape错误拒绝；另构造max_abs点通过但近零点失败的两元素例子，确认`worst_index`定位实际违规点。结果**1 passed**，完整输出为`CPU_FIX2_ACCEPTANCE_STDOUT.txt`。未启动新的GPU重放。

## 两个实际源码身份

| 身份 | 完整source hash |
|---|---|
| 旧诊断R1，四个存活ML worker | `72f3e825dac422784fd575bd24450e7858e99ce138f6c8261b3ae7636b8b13fc` |
| 新诊断R2，失败KR worker重跑 | `26708434c81106dd424d52b4de258bee8aad7b35640b1d50b8699290694a2a3c` |

两个完整41文件package快照及逐文件清单位于`SOURCE_REVISIONS/72f3e825/`和`SOURCE_REVISIONS/26708434/`。旧快照在改动前按实际worker的原清单逐文件验证并保存；没有将新代码冒充旧代码。

逐文件比较仅`diagnostics.py`和`tests/test_runtime_numerics.py`不同。去掉且仅去掉`diagnostics.py`与`tests/`后的所有文件集合、每文件SHA完全一致，统一training_source_hash为：

`29d7018f037c06af567d58646c6f81d022cab101a0e87c86c99af295e751cd98`

额外AST检查确认diagnostics除`compare_raw_logits`外所有导入和科学诊断/预测/时延函数相同；实际diff保存在等价证据中，供独立reviewer核对。新旧完整source_hash仍分别保留，不把它们改写成training hash。

## 原位归档与重跑

runner先确认对应失败PID已退出，再将每个KR失败seed的`workers/kuairand/seedNNNN`整目录和`runs/kuairand/<六方法>/seedNNNN`中存在的目录移入`revision1/attempts/KR_seedNNNN_<stamp>/`对应层级，保存失败stdout。只移动该失败seed；保留`revision1/data`、全部ML目录和活跃进程。

新KR进程仍使用原`revision1`根目录，增加现有freeze-file参数：

```bash
CUDA_VISIBLE_DEVICES=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.mdmt_relation_basis_v2 --artifact-root artifacts/mdmt_relation_basis_v2/revision1 train --dataset kuairand --seed 2022 --device cuda:0 --freeze-file artifacts/mdmt_relation_basis_v2/revision1/GLOBAL_FREEZE_FP32_STANDARD.json
```

其余KR seed按真实资源和原seed重跑。原4份失败轨迹及额外成本保留，重放不计作新增训练重复。

## 最小冻结兼容工具

`freeze_compatible_revisions.py`位于报告目录，未纳入package source_hash，也未修改campaign/data/CLI/report。它只支持上表两个已审查版本：验证完整快照；验证其共同训练字节；读取8个READY和48个run receipt；要求方法/selected/final完整、config一致、真实source清单精确匹配、每数据的角色/预处理身份一致；保持每worker原identity。

主agent完成阶段决策及独立复审后执行：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/26-10-27_v2/freeze_compatible_revisions.py --artifact-root artifacts/mdmt_relation_basis_v2/revision1 --decision research_reports/26-10-27_v2/STAGE_DECISION.json --review-record research_reports/26-10-27_v2/IMPLEMENTATION_REVIEW_FP32_STANDARD.md
```

工具生成`SOURCE_EQUIVALENCE.json`、旧worker默认`GLOBAL_FREEZE.json`、新worker指定`GLOBAL_FREEZE_FP32_STANDARD.json`和`GLOBAL_FREEZE_CANONICAL.json`。两view含同一8个真实worker、方法/配置/decision/threshold、共同training-source证明和canonical identity，仅顶层source_hash按接收组设置；canonical另保存实际view文件hash和worker映射。现有evaluate release仍从默认view查真实worker条目，因此8个worker都可按原入口逐个释放。

一个toy quorum测试加载两个快照中的真实Data validator，验证old/new各自view均授权、错view拒绝、8缺1拒绝。结果**1 passed**，见`CPU_FREEZE_COMPATIBILITY_STDOUT.txt`。测试未评分真实数据或模型。

未改其他package源码或结束存活worker；正式重跑与E解锁等待独立复审。此次没有新增GPU诊断成本。
