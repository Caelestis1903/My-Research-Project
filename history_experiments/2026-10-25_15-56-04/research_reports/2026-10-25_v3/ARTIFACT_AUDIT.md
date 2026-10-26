# v3 实验前工件独立审查

2026-10-25。本审查仅记录身份、可访问性和确定性复算，未训练模型、未改旧工件、未作新算法效果判断。完整机器收据：`../../artifacts/mdmt_next_v3/v3_0/artifact_manifest.json`。

## 已独立核实

已完整读取 `research_reports/2026-10-25/` 的 FINAL、S1、S2、REPLAY、OPEN_QUESTIONS、IMPLEMENTATION_REVIEW、COST、BASELINE、FORMAL及修订。报告文件SHA256在manifest登记。

`artifacts/mdmt_next_v2/v2_0/source_versions/` 的 cbe015172169a261025614bf2c14bd77d13eb69d0855677df992926068b26af5、f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7、46719334c8adc2b8e79da640b8ac3ef6987b9f5e52be10b45032be65c2cd5bc2 三包全部逐文件重算SHA256，与各自SOURCE_RECEIPT **100%匹配**。它们是源码收据身份，`git_head=null`，不能称Git commit。

48条核心轨迹的primary_best及epoch24共 **96** 个预测包，均从真实y/logits/domain/rows独立重算六格AUC/LL、macro、natural LL、task-pooled AUC；与旧机器表的主聚合及逐格最大绝对差均为 **0**。全部rows hash与Q_explore角色一致。LL用 `logaddexp(0,z)-y*z`；macro是六格等权均值，pooled单独保存。没有由正文数字合成原始数据。

检查了 **84** 个尝试目录的配置、状态及可得selected/last/epoch5/10/24完整状态。另对HIST/RIDGE/NN与STATIC的两个数据×两个seed×selected/epoch24共 **24** 对模型参数和buffer逐tensor比较，全部bitwise相同；此事实不等于所有时刻optimizer/controller/RNG也相同。

## 数据和环境

| 项 | MovieLens | KuaiRand |
|---|---|---|
| raw CSV | `/home/xzf/data/ml-1m/ml-1m.csv` | `/home/xzf/data/kuairand/processed/kuai_rand.csv` |
| raw SHA256 | `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4` | `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f` |
| prepared目录 | `/home/xzf/data/mdmt_next_20261024_strict_v1/movielens` | `/home/xzf/data/mdmt_next_20261024_strict_v1/kuairand` |
| fit d0/d1/d2 | 124759 /254879 /260495 | 138225 /1395118 /71561 |
| 字段数 /native embedding维 | 8 /113 | 34 /544 |
| P_form/P_read行数 | 24004 /24006 | 60183 /60184 |

raw全文件SHA均与固定预期一致。RoleDataset重新核对x/y字节、fit-only vocabulary/scaler/fit row摘要、无交叠完整角色。按原规则重新生成所有role并逐数组比较，均一致。

按文件03的 `PCG64(20261025+60000+1009*d)` 对排序P_label各域置换和前后半切分，独立证明 **P_form=P_structure、P_read=P_value** 逐row相等，可复用旧表并登记新用途。

| 数据 | P_form hash | P_read hash |
|---|---|---|
| ML | `a04fa99676ed28f48066c4c0cdf4cb6773c266ebe0eff542e6e7bae3ca640e5c` | `47395d052f4f872e2028552a384a5cd5672739b35ccbefdedec8ba9cda61d4a6` |
| KR | `1c40c654c13c84ae3f6484fa96caf921f14666b3692e7d8eba85efce78910385` | `34f009c4c14b14fcf129519a276cf416924e638a23f25784ccb0c65429d4941f` |

环境：`/home/xzf/miniconda3/envs/phase0-3-env/bin/python`，Python3.13.5、torch2.9.0+cu126、CUDA=True、numpy2.1.3、pandas2.2.3、sklearn1.6.1，与旧runtime一致。默认Python3.14没有torch。`~/mdmt_repro/sources/`不存在，实际源码在上述持久归档。checkpoint实际持久目录约5.4GB；tmp_link只作旧来源记录，不是唯一副本。

## 可复用接口

- Native：`experiments/data_generate/MDMTRec/models/multi_domain/fixed.py`，SHA256 `43d7591f28081d8e3ea6c247a877695f92232be87354d95deecbed9fe3e754be`。
- `experiments/mdmt_next_v2_20261025/models/fixed1_adapter.py::Fixed1Adapter.encode(x,risk=False)` 接[B,F]，内部转置机械执行native，返回native_logits[B,2]、expert_outputs[B,4,64]、domain和embedding。真实sigmoid前取logit；risk=True会去掉gate-input detach，不能当原训练图。
- strict数据：`experiments/mdmt_next_20261024/data_roles.py::RoleDataset`；旧包装器 `experiments/mdmt_next_v2_20261025/data/roles.py::Data`。v3需记录P_form/P_read新写入权限。

## P0原始输入、schema与缺失

| 项 | 可访问输入 | 能复算与缺失 |
|---|---|---|
| R1拒绝链 | native run下prequential_predictions.jsonl、bootstrap_prequential.jsonl、utility_events.jsonl、utility_H1_*.npz；原history/state源码 | pre-reveal forecasts有actual/hist/raw/mu/radius，状态模型另有gamma、mu_gamma/radius_gamma/raw_gamma。逐层未存标量需按封存事件和原源码重放，不能臆造。 |
| R2简单历史 | 四条S2 BASE控制期28事件，加本轨迹或S1 warmup12事件 | npz有loss_P_update/loss_P_cal[8,3072,2]、rows_[3072]、support_rows[1,40960]、u/var/cal[8,6]。H5仅两个S1 native diagnostic epoch10 `utility_H5_011.npz`，不能称四条H5全程轨迹。 |
| L1树链 | S1 K4 diagnostics/policies.pt；S2 LPS selected/epoch24的actor.local；S1 epoch10 | 保存节点含feature/threshold/depth/n/action/coarse/margin/se及cal_scores。没有逐刷新全部候选分裂表或所有中间快照，不能从cap=0反推全程结构阶段失败。可在已存状态从P_label/P_cal原行机械重建cache。 |
| L2容量/可读性 | S1 concat K1/K4 epoch10；S2 DIRECT K1/K4 selected/epoch24完整native/bank/reader；真实角色数据 | Q cache只有动作logits/loss、z0、demand、foreign_key，没有owner逐块全响应。需在指定保存状态真实前向重算，不以余弦替代风险。 |
| L3 CAL | S1 K1/K4 Q_explore_cache.npz、policies.pt；原Calibration；同状态角色 | cache字段rows/domain/y/z0/logits/loss/demand42/foreign_key，K1动作7、K4动作51。demand42为anchor32+own logits2+foreign geometry8。旧CAL可直接评价；新增MONO/no-key34须指定标签池拟合，现有报告不含该结果。 |

cache根目录 `artifacts/mdmt_next_v2/v2_0/diagnostics/S1_<dataset>_concat_K<K>_3101/`；核心状态根为 `checkpoints/S2_<dataset>_<kind>_<seed>/`。完整schema和路径均登记在manifest。本审查未用test/Q_confirm选方法，保留其历史暴露身份。

## aux及外部基线真实断点

| run | last完整epoch | 实际Adam状态 |
|---|---:|---|
| ML K1/K4 aux0 | 22 /10 | FullState.optimizers；131 /149参数有m/v/step |
| KR K1/K4 aux0 | 9 /9 | 同上158 /176 |
| ML AutoIFS | 13 | objects.workflow内reuse optimizer22状态，step48；search160/reuse48，rewind已存，reuse=True |
| KR AutoIFS | 7 | workflow内prediction48+selector8状态，step280；search280/reuse0，尚未到rewind |
| ML/KR STEM | 13 /7 | FullState.optimizers；122 /148状态，step208 /280 |
| ML/KR STGU | 19 /7 | FullState.optimizers；107 /134状态，step范围304–327 /280–567；objects.stgu已存 |

上述实际Adam状态全部具exp_avg/exp_avg_sq/step，FullState有python/numpy/torch/cuda RNG、模型/梯度/模式/requires_grad和actor/coverage/workflow/stgu。AutoIFS外层self.opt未执行step，所以为空；真正两阶段优化器及rewind在workflow.state_dict，不能据外层空opt判不可续接。

四条aux0 config均fork各自S1 epoch5；旧aux从epoch6才作用，符合未受aux影响起点。其last可直接恢复actor/optimizer/RNG；Plans.main(epoch)从固定PCG前缀纯函数重建行计划。

**Critical续接路径：**旧Trainer.train在begin==11且bank时无条件先refresh。ML K4 aux0 last10已含epoch10 refresh；真实resume若再次调用会多一次头拟合和标签访问，改变后续训练。隔离续接包装器必须区分“新fork初始化”与“已刷新last真实resume”，后者跳过重复refresh，保留旧工件。其余aux断点22/9/9不走该重复入口。

续接按v3移除旧stage_seconds/native等价timeout，保留旧退出及配置，单列新段成本。AutoIFS workflow与STGU对象需完整恢复、保持模型引用；保留ML冻结selector/input embedding，KR后续search→rewind→reuse，不重置phase/Adam。没有新增完整训练结果时，不把这些last冒充epoch24。

## 证据边界

本审查核实本地归档与SOURCE_RECEIPT，不等于重新核实远端历史Git对象。旧正式4101–4104/50epoch、完整旧aux/external24epoch不存在；无全程树候选日志、全部中间状态、新独立确认数据。所有缺失均保留，不能由正文数字补出。科学解释及P0新增数值分析由主agent后续基于这些真实输入完成。
