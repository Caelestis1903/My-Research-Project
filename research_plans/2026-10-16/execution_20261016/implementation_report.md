# IMPLEMENTATION REPORT

主代理，2026-10-16。严格依据design_specification.md；未修改旧训练代码、已有结果或checkpoint。

- `audit_and_selection.py`：按stage/路径/类别索引实际工件、逐项重新检查旧missing路径、验旧B1/C2及最终静态归档hash；读取已审计逐cell表，输出val-test epoch排名、cell排名和选点/轨迹图。无torch或模型调用。
- `state_oracle.py`：CPU上复用旧有限pool/init/坐标，48行逐任务autograd取得task-mean梯度。numpy实现alpha/q估计器、covariance/ESS/exposure和64步SGD/Adam/AdamW状态replay；运行拒绝覆盖case目录，每case保存数值输入、hash、receipt。新脚本不导入训练runner入口，不调用fit，不读test，不创建正常训练轨迹。
- 原始模型是StageS1M3OEAdapter，forward(x,test_flag=False)，模型源码`experiments/stage_S1_models.py`，继承`m3oe/MDMTRec/models/multi_domain/model.py`。这不是V1/V2。现有四个learned fusion scalar确实存在，来源于旧S1独立baseline；本轮未恢复/新增这些参数，也不将其声称fixed-fusion V1/V2。原baseline有main/arch两个Adam和epoch-level架构更新；本oracle不执行该loop，只对C/P与A坐标沿用lr=.01/.001。梯度balancing、DBO weighting和稀缺域增强均关闭。
- loss=`sum_d alpha_d mean_{row|d}(mean_t BCE)`；估计器=`mean_B((alpha_d/q_d)*mean_t BCE)`，域内从16row pool均匀有放回抽样。Equal和Sample是不同alpha，不能称同目标比较。
- state采用fresh step0、betas(.9,.999)、eps1e-8、wd1e-5；无clip。Adam coupled WD取固定theta，AdamW decoupled WD亦取固定theta；数值replay的累计delta不是重新求梯度的真实trajectory。

basic检查：两脚本py_compile通过。64步detached torch.optim逐步状态及delta对照（SGD/Adam/AdamW，wd0/1e-5）最大绝对残差6.189493362285248e-15，见optimizer_formula_check.json。本报告不分析研究结果。
