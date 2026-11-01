# AIGM01 独立实现审查

日期：2026-11-01。审查者与coder为不同agent，继承协调者相同模型。审查输入为本轮 `DESIGN_SPECIFICATION.md`、实际新包、`IMPLEMENTATION_REPORT.md`、规划03/04/05/06及GAM01/GA31-R3依赖。只写本报告，未改实现或历史文件。

**最终 verdict：PASS。无未解决Blocking项。可将已审查源码冻结为本轮不可变执行commit，交runner执行真实CUDA Stage0；Stage0 PASS后进入有限开发与正式阶段。**

## 核对结果

| 范围 | 实际实现与设计核对 |
|---|---|
| 基础目标与数据 | `protocol.py`继承已固定GAM01/GA31-R3数据、模型、RNG和训练配置；自然80/10/10、fixed1、batch40960、50epoch、FP32、五seed不变。实际加载器检查raw/六数组/split行号/encoder/scaler身份与域计数，缺失prepared目录不在当前依赖路径上。 |
| clean/delta与随机流 | `objectives.py`直接复用源目标和拆分：full-real的任务均值再样本均值，target augmented减target clean除原B×2；同θ上先完成g/d再一次更新。A0/A1一次backward，SAD/AIG有效增强batch两次autograd；关闭/κ0/λ0不抽view。沿用独立视图key和loader generator，诊断/EMA不消费训练随机流。 |
| AIG公式 | `optimizer.py`更新前由未校正旧q、当前不含wd的g/d按静态E/P组计算u；c=1、epsilon=1e−12、斜率4、βδ固定.95。组λ同时用于当前 `(g+wd·θ+λd)^2` 的v与新q分子。m含coupled wd，q不含wd；m/q/v按各自β与活动step偏差校正。没有动态β、AdamW或用q替换当前d二阶矩。 |
| 参数分组与活动语义 | 按实际 `MDMTRec.forward` 核对E/P/U规则：全部embedding、共享及target路径、其他域专属参数；模型原参数顺序保持，g/d未因分组重排。U无q且标准Adam运算次序；None双侧跳过、单侧按零、显式零活动；当前d=0不清空旧q。真实Stage0直接检查全部U差分为None或精确零。 |
| 阈值 | `run.py`只收集S95、两个开发seed的E1各训练step更新前u；每数据集E/P分别合并训练观测，以NumPy linear q50/q75冻结。观测数断言为2seed×2组×step/epoch；M1/M2及后续seed共享训练阈值，不读取validation/test拟合门。 |
| 四shadow与状态 | `.995/H2/H8/H20`从θ0维护，H按20/49 step换算；每online step后更新，buffer复制，评价恢复training标志。online与各shadow独立按完整validation六格macro AUC严格提升选epoch，平局最早、E0排除。RAM快照含model/optimizer/shadow/RNG/loader/config/阈值/选点/计数，保留各selected及E10/E50，过时状态释放。IPC沿用BytesIO/Pipe，未发现磁盘权重路径。 |
| 有限开发与选择 | S1四联合lr/wd点、S2三Gaussian点、S3两固定SAD点、S4两AIG点均精确匹配规格；S2后S3/S4仍固定κ=.1并保留G2参考。候选先序列化模板/hash，实际基础配置和阈值解析后写实际配置hash。每trial完整50epoch，没有E10剪枝或扩网格。 |
| 晋级与保护 | 开发主门为相对最强S2增强shadow的mean Δmacro AUC≥.001、2/2正；LL≥+.002、worst或最大域任一任务≤−.002且2/2不利阻止晋级。S4先限制到通过门/保护的选项再按冻结.0005近似平局顺序选择。数据集独立晋级，普通增强未过增量门仍保留正式参考。 |
| 正式阶段与缓存 | S5前冻结配置、shadow种类、代码/数据身份和五seed清单；A0 online/EMA共享online轨迹。cache key包含完整配置、源码commit/manifest、数据、seed、RNG及指标语义，复用时检查完整50epoch实际identity，ledger显式CACHE_REUSED和来源。训练成本只汇总实际unique结果，不重复计缓存；名义64/69/74与实际训练数分别记录。 |
| test与数表 | 全部开发及正式轨迹完成后，先写全部正式模型selected/E10/E50身份，再统一test；不测被拒开发候选，不用test改变ρ/阈值/配置。相同kind/epoch去重forward而保留端点行。数表按seed配对，n=2/n=5分开，sample SD及df1/df4 t区间正确，macro/pooled/traffic/worst与六格原值保留；正式裁决保留开发门与validation/test保护。 |
| 成本与调度 | 分别计自然样本、增强曝光/view draw、forward、梯度遍历、optimizer与shadow更新、各模型validation时间/forward、选择次数、test成本、峰值allocated/reserved及RAM bytes。CUDA-event训练区间和validation纯forward区间与wall定义明确。逐卡显存与功耗均<85%且原batch峰值可容纳时继续并发；OOM仅降低并发原batch重试，失败账本保留，外部进程不受修改。 |
| Stage0门 | `all`实际要求Stage0 status=PASS、源码manifest hash及执行commit全部相同。真实Stage0复用源真实模型/Adam/SAD检查，增加新AIG/U映射/四shadow/RAM恢复，共每数据集19次实际optimizer更新，低于ML20/KR49步的一epoch等效上限，不调用test。 |

03中v的当前d与一处q文字歧义、基础Adam m的wd简写、首版固定β，以及EMA半衰期10倍勘误均已在设计中明确，本实现遵循该批准版本。M1缺少固定λ=.25专属训练对照、开发seed复用及各方法未独立lr/wd搜索已明确为报告解释边界，不以额外实验扩大本轮预算。

## 独立必要检查

1. 实际运行 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/gradient_aug_aig_m_20261101/tests/test_semantics.py`：**3 passed，1.47s**。覆盖FP32/FP64动态λ多步手算、coupled wd/q/v及偏差修正、固定SAD退化、None/零/旧q、U标准Adam逐位比较、组件关闭、四shadow/RAM下一步恢复、开发门/保护和近似平局顺序。
2. 独立构造真实协议的有限候选与job列表，断言S1/S2/S3/S4为 **16/12/8/8，总44**，formal seeds为2022–2026，shadow恰为四种；通过。未运行训练。
3. 复用coder的compileall、完整数表合成流及既有源公式/语义收据；对应源码没有更改，无需追加同类检查。pytest唯一warning为冻结源setup使用的PyTorch TF32设置API弃用提示，不改变本轮FP32设置。

## 交接

runner下一步执行两个真实数据集的原batch CUDA Stage0，实际验证数据/模型、U差分、显存、四shadow和完整RAM恢复。通过后按S1→S2→S3→S4→S5执行并逐阶段汇报；最终科学解释由协调者完成。本次审查未要求代码修订或新增搜索。
