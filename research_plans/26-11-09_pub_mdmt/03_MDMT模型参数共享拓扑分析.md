# MDMT模型参数共享拓扑分析

## 模型任务
目标为M3oE fixed1，3个domain × 2个task，共六格：D0T0、D0T1、D1T0、D1T1、D2T0、D2T1。

既有参数注册表 experiments/aug_history_20261105/group_registry.py 将参数坐标划分为12个互斥路径组：S、D0、D1、D2、T0、T1、DT00、DT01、DT10、DT11、DT20、DT21。注册表依据named_parameters、张量形状和domain embedding行切片构造，并检查坐标完整覆盖、不重叠。

## 参数参与集合
| 参数组 | 参与域任务格 |
|---|---|
| S | 六格全部 |
| D0/D1/D2 | 对应domain的两个task |
| T0/T1 | 对应task的三个domain |
| DTdt | 对应单一域任务格 |

domain embedding按行分配D0/D1/D2；其他参数按实际模块根名映射S、D、T、DT。不得把同一物理参数坐标在多个PUB子问题中重复更新。

## 自然样本目标分解
现有clean损失实现线索 objectives.py：逐行、逐任务BCE，先对任务取均值，再对batch行取均值。对batch实际行数N、domain d行数n_d：
L_dt=(n_d/(2N))*mean_{i:domain(i)=d} BCE(p_it,y_it)。
则Σ_dΣ_t L_dt=(1/N)Σ_i(BCE_i0+BCE_i1)/2，恢复自然样本目标。

P0：L_t=Σ_d L_dt。P1：六个L_dt。P2：对参数组g只保留C(g)中的L_dt。禁止六格BCE简单等权求和，禁止改变loader、采样比例或引入按域规模先验权重。n_d只用于精确分解原始目标，不是困难度代理或控制信号。

## Codex开工前的计算图审计
在当前可执行源码commit上确认两列输出分别对应task0/1；domain d的task t是否经过正确tower索引3*t+d；S/D/T/DT的实际autograd非零支持是否与注册表source_cells一致；gate detach等语义是否保留；inactive embedding行是否排除。若图与注册表不一致，暂停实现并先修正审计，不得凭参数名推断。
