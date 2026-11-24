# 07 — 实现接口、来源闭包与状态事务

## 1. 实现边界

本文件规定后续 Codex 的工程合同。**本次提交只有规划与历史数表重分析；`experiments/history_information_20261124/` 是待创建模块，不是已经可执行的训练代码。** 数学定义以05为准，队列与选择以06、08及机器 manifest 为准；工程实现不得改变这些定义。

实现沿用既有自然数据、M3oE fixed1、PAIR 计算图、原生 Adam 与 RAM 状态管理的数值内核。新增内容是 C1/C2 及有限控制的状态递推、只读诊断、明确的任务实例和报告接口。原研究的自动选源、旧阶段队列、旧 Test 许可和旧长期研究驱动不进入新调度器。

参考来源固定为 main 证据快照 `358d3a465747eda6af1d55acc937759fa651a10b`；发表模型实现按11所列原报告指向的 `c8675ae2923c15b6965884b140b5b646690cd95e` 恢复。运行前再次记录实际实现提交和内容哈希，不能仅用可移动的 branch 名称作身份。

## 2. 复用入口与新模块分工

### 2.1 已有入口

| 现有文件/符号 | 复用内容 | 必须保留或适配的边界 |
| --- | --- | --- |
| `conditional_information_momentum_20261102_v3/protocol.py`：`load_protocol/config/make_optimizer` | 自然数据来源、M3oE、固定优化/EMA参数 | A0/P00 的源配置身份保留；新臂不能塞入旧 `METHODS` 后借旧队列启动 |
| 同包 `boundary_sampler.py`：`CONDITIONS/boundaries/sample/forward_logits` | 条件顺序、同批支持、静态q、PAIR索引、真实tower logits | `dynamic=False`；PAIR键不含arm；原gate detach及完整反向图不变 |
| 同包 `objectives.py`：`train_step` | 同一次forward、raw r、完整loss backward、一次EMA | B0/B1/H0零点走原入口；新目标适配继承同一数值顺序 |
| 同包 `optimizers.py`：`PairOptimizer/shared_name` | ordinary Adam与S_DENSE定义 | coupled WD只一次；原None、step、foreach/fused/amsgrad语义 |
| `conditional_history_shared_update_20261122/history_controls.py` | `history_config('A21',...)`、`make_history_optimizer`、源六格shadow | H0完整状态等价验收保留原A21配置和所有被动shadow |
| 同包 `history_runner.py`：`step_anchor` | post-forward/pre-step锚点的采集次序 | 复用捕获原则；新锚点含本轮Vsep/U等observer，不照抄旧阶段臂清单 |
| 同包 `a_localization.py`：`replay`；`b_virtual.py`：`registry` | 原状态深拷贝、真步哈希核对、参数坐标注册 | 不复用旧52分支/旧选源规则；本轮D2只有04定义的11分支 |
| 同包 `state.py`：`capture/restore/commit/generation` | guardian generation、完整状态、实际get/hash | 旧epoch packet与新锚点packet分别标型；恢复检查增加新态及新配置哈希 |
| `vendor/ema02/protocol.py`：`update_ema/state_digest` | 参数EMA、buffer原样复制、源状态哈希 | EMA参数每真实step更新一次，所有buffer直接copy；不对buffer额外平滑 |
| `vendor/ema02/evaluate.py`：`evaluate/metrics` | 完整六格概率指标和缺支持语义 | 正式AUC用源FP32概率转FP64；不改成logit排序；推理后恢复原mode |

表中的包均位于 `experiments/`。完整固定链接、闭包清单和复用状态见11。[S1–S6]

### 2.2 待实现文件

| 新文件 | 对外职责与关键输入/输出 |
| --- | --- |
| `protocol.py` | 读取plan manifest，产生不可变`RunSpec`；验证dataset/seed/arm/预算/依赖；生成source/data/config/group哈希 |
| `sources.py` | 按11固定路径核对闭包；恢复冻结模型与train-only预处理；输出`SOURCE_MANIFEST.json`与`DATA_MANIFEST.json` |
| `groups.py` | 生成有序参数/坐标清单、scope mask和可达cell元数据；拒绝未映射/重叠/遗漏参数 |
| `objectives.py` | 原PAIR静态采样、原图raw r和完整native grad；分别适配M3oE、HyperGate和可触发的GA31-R3 |
| `optimizers.py` | C1/C2/C12、M/S/P、N的05递推；H0与零点直接委托原adapter；`state_dict`含版本和时钟 |
| `observers.py` | D1标量与同轨迹shadow；D2锚点采集；D4预测日志；所有observer禁止反馈live更新 |
| `diagnostics.py` | 04的有限11分支、固定面板及pair比较；输出逐锚点记录，不选择训练超参 |
| `state.py` | `StatePacketV1`与`AnchorPacketV1`的RAM提交/实际读回/恢复；管理selected状态生命周期 |
| `runner.py` | 原子batch和epoch、两读出选择、成本、恢复；只产原始结果，不自动增设研究臂 |
| `scheduler.py` | 有限依赖队列、停止控制、10的实测准入、当前会话监督；禁止旧自主driver |
| `selection.py` | 06/08的固定规则、同一家族跨数据选择、控制后简化、条件臂实例化与一次freeze |
| `analysis.py` | 09的配对统计、逐格保护、交互、排序关系与成本汇总 |
| `cli.py` | 00的`audit/check/materialize/run/analyze/freeze/evaluate/report`；命令不能隐式跨阶段 |

模块名称是实现交接合同；内部函数可按可维护性组织，但输出字段、科学语义与命令职责不得改变。删除原工具的绝对目录假设时，记录路径适配差异；保留被复用数值文件的原内容哈希。不能调用旧 `protocol.jobs()` 来生成本轮队列。

## 3. 环境、数据与参数组

### 3.1 来源闭包

运行环境为仓库指定 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`，Python 3.13.5、PyTorch 2.9.0+cu126及锁定依赖；记录实际GPU、CUDA可用性、设备UUID、驱动和包锁差异。当前规划环境没有训练就绪的声明，不能用文档中的历史GPU记录替代现场检查。[S7]

旧v3数值闭包以 `IMPLEMENTATION_SOURCE_MANIFEST.json` 的 SHA256 `7034a3bb59e6d08e3296183258e0e7b16ab74d0248ee0a3b75c526a680462cc0` 为来源线索；沿其明确文件列表逐项核对。新代码、模型vendor、数据loader、采样器、指标和新observer均进入运行源码清单。只获取已列出的路径；禁止递归读取人类综述。缺依赖登记具体文件和阻塞模块，不以“已有同名包”替代源码身份。

### 3.2 自然数据身份

| 数据 | CSV及SHA256 | Train域行数 / 总行数 | 主矩阵batch / steps per epoch |
| --- | --- | --- | --- |
| MovieLens | `/home_new/xzf/datasets/ml-1m/ml-1m.csv`；`f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4` | 155949 / 318599 / 325619；合计800167 | 40960；20步，末批21927 |
| KuaiRand | `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`；`a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f` | 172782 / 1743898 / 89452；合计2006132 | 40960；49步，末批40052 |

原行序80/10/10，不重新随机划分；KR先按原CSV切split，再按源代码筛选tab0/1/4，不能先筛后切；编码器、归一化和词表只拟合Train，未知类别按源实现。ML标签为rating>3、rating>4，联合标签01应不存在；KR标签为click、long_view，tab0/1/4映射d0/1/2。冻结并核对：CSV哈希、特征顺序、array hashes、split row IDs/fingerprints、encoder/scaler hash、各split行数、Train域计数及联合标签计数。旧prepared目录缺失时可以通过冻结loader重建这些结果，不改自然人口。[S1,S6]

训练/诊断接口只构建和暴露Train与Validation数组。原 `v3.load_dataset()` 会构建含Test的完整DataBundle，新数据封装采用窄Train/Validation数据桥，显式`include_test=False`；不向训练/诊断构建或暴露Test feature/label arrays。允许流式读取整CSV做字节哈希及按原行序确定边界，预处理拟合仍只用Train；原Train/Validation数值与哈希必须不变。Test数组、预测、标签指标和selected模型评价只由带08权限核验的独立入口构建和执行，历史已登记的Test人口fingerprint只作来源核对。

发表模型数据桥 `cached_bundle(root,dataset,include_test=True)` 的默认值尤其不能沿用：所有训练调用显式 `include_test=False`，Test封存时也不能直接运行会写出全split缓存的旧 `prepare_cache()`。新适配保留同样Train/Validation数组与预处理哈希，Test数组由获准的评价阶段单独装载；仅在读入全部数据后删除属性，不算完成该接口隔离。[S8]

### 3.3 M3oE参数组注册

采用原fixed1、embedding16、4共享expert、FCN[512,256,256,64]，保留原gate输入detach。每个可训练坐标恰有一个scope；精确名称规则如下。[S2,S4]

| 参数位置 | 组 | scope/支持 |
| --- | --- | --- |
| `shared_weight/shared_bias/skip_conn.*/star_mlp.*` | STAR_SKIP | S_DENSE |
| `expert.*`，包括其含`domain_specific`的子名 | EXPERT | S_DENSE |
| 普通`embedding.*`，排除domain_indicator字段 | EMBED | 共享Embedding，不在主历史scope |
| `embedding.embed_dict.{domain_column_index}.weight`形状[3,16] | 第d行归D_d | N；域Embedding不计入共享EMBED |
| `slot_weight.d.* / slot_bias.d.* / domain_expert.d.*` | D_d | N |
| `task_expert.t.*` | T_t | N |
| `tower.i.* / gate.i.*`，i=3t+d | DT_dt | N |

注册器从`features.columns.index('domain_indicator')`得到域Embedding路径，不能硬编码某一列序。按原 `named_parameters()` 顺序登记name、shape、dtype、numel、整参数native步身份、scope与坐标mask；D/T/DT的结构标签不替代实际计算图的可达性检查。某次梯度恰好为0不构成“该cell不可达”的证据。

源fixed1的计数验收：ML总773470，N=290318，共享EMBED+S_DENSE=483152；KR总1614254，N=952334，共享EMBED+S_DENSE=661920。ML每D组74960、KR每D组295632；每T组16576；每DT组5381。域Embedding三行已含在D组，不能重复计数。架构身份相同而数量变化时先修复注册/模型来源，不能将新形状当作既有协议继续训练。[S4]

N扩展按05局部时钟：native None整参数跳过WD和全部状态；native非None而PAIR对该参数unused时，仅辅助通道按零处理。N的域Embedding三行共同构成整参数的N作用范围；若未来适配出现部分张量scope，保留整参数native时钟并用互斥坐标mask合并一次参数提交，不能整张量Adam后再叠加另一次完整自定义更新。

## 4. 随机流与一次真实更新

### 4.1 随机性合同

正式训练seed仅2022–2026。原模型初始化、Python/NumPy/Torch CPU/CUDA状态和loader generator分别保存；M3oE loader沿源 `torch.Generator().manual_seed(seed+11000)`，每epoch一次`randperm`，尾批保留。数据/seed相同的各臂从相同初始化和批序出发；不能把arm名加入初始化或shuffle键。

PAIR条件和pair索引用原 `dataset|seed|epoch|global_step|stream` 的SHA256键，保留原字节顺序和NumPy generator。诊断面板、固定评价pair和可触发Gaussian使用各自明确命名的流；不得消费真实模型dropout、loader或PAIR随机流。Gaussian严格沿已冻结GA31-R3的batch级流，不换成另一个历史实现的逐行键。

GPU迁移恢复时把保存的逻辑训练设备RNG映射到实际worker设备，不将cuda:0的状态错误写到别的作业设备。报告GPU UUID与逻辑编号；无关GPU状态不由该worker覆盖。数值确定性沿原设置；同设备精确重放失败是实现问题，不能用不同seed的总体均值接近代替。

### 4.2 batch事务顺序

1. 从冻结批序取`row_ids/x/y`，`zero_grad(set_to_none=True)`；原forward返回probability和真实tower logits，保留所有原生模型buffer更新。
2. 按原条件采样一个合法边界、2048对；在同一图上对custom scope求raw `r=grad(R)`，detach后保存。主S_DENSE用源严格used语义；N/HG注册的unused路径使用05的显式None规则。
3. 完整loss一次backward，得到native grads。scope只是更新位置，不能切断PAIR在scope外的完整梯度。B0/B1/H0零点直接走源adapter。
4. 读取`theta_pre`，冻结native None mask；`mixed=p.grad+WD*theta_pre`，`b=mixed-(lambda*ratio)*r`保持源减法顺序；custom范围按05更新m/V/Q/U及两个时钟。新C1凸组合用非负平方输入，U在本步V和D完成后更新。
5. observer从本轨迹相同prestate/梯度更新只读shadow；M的两套shadow也只取M自身路径。shadow不执行参数提交。准备自定义scope与ordinary scope的全部更新，检查有限性及互斥覆盖。
6. 每参数仅提交一次；完整模型EMA随后一次。任何失败不得把半更新状态标为成功；恢复最后已验证RAM事务后重做该未完成原子工作，成本登记technical replay。
7. 写标量/来源/成本日志，登记step成功；训练状态仍在RAM。epoch完成后进行完整Validation、两读出选点及RAM事务提交。

对普通Adam范围，可沿源native optimizer提交；为发现更新失败而不污染后续步骤，worker必须持有可恢复epoch事务。不能在非有限方向时静默跳步、降低lr、裁剪梯度或继续下一个batch。主矩阵的固定步数因此具有明确含义。

旁路先按完整`backbone/base_objective/scope/betaP`解析父入口。`lambda_PAIR=0`只移除PAIR/新增历史：主矩阵返回B0，G分支返回保留GA31-R3的G_B0，HG返回保留全部native objective的HG_B0。注册.99候选的`zeta=alpha=0`返回同背景和scope的H0/H0_N/G_H0/HG_H0；只有主矩阵直接委托原A21并保留其config和被动shadow。不能让B1/P因数值零误入H0，也不能把S的.9核旁路回.99。这些旁路是数值身份合同，不经过新实现先算一遍再归零。

## 5. observer、锚点与虚拟分支

### 5.1 shadow生命周期

H0从初始化同步维护vmix、vsep、U0和原六格M9/Q shadow；不能在E20才创建U并称为完整历史。S2轨迹的诊断shadow也由自己的b/h/D递推。C1记录U时使用其自身当步D的只读U；记录cross时在同一路径维护vmix/vsep。C2使用自己的vmix/U，并增加只读vsep。需要哪些shadow必须在job启动时登记；算法活跃态和诊断额外态分别计成本。

M在S完成、家族C/S代表冻结后才启动，其候选shadow使用该代表完整βP/参数。目标为S时H0的Q99与候选Q9不可共用，U也用.9；D2锚点中的M仍匹配各个原始.99候选。共享shadow存储只允许对完全相同的递推与时钟去重；每个共享对象每step只更新一次。`detach/no_grad`阻断历史计算图，禁止observer保留导致显存随epoch增长的autograd graph。每次诊断前后对live模型、optimizer、EMA、RNG、loader和selected metadata做哈希守恒核对。

### 5.2 两种packet不能混用

**`StatePacketV1`** 是epoch边界可恢复状态；loader cursor等于该epoch permutation长度。**`AnchorPacketV1`** 是E5/E20/E35/E50末批真实forward/backward后、step前；其cursor为该批在permutation中的起始offset，并明确`phase=POST_BACKWARD_PRE_STEP`。源capture的cursor默认值不能直接用于锚点恢复。[S3,S5]

锚点额外保留：当前row IDs与batch indices、完整native grads及None mask、raw r、pair结构、pre-forward RNG/buffers、post-forward模型/buffers、pre-stepoptimizer与observer、EMA前态、实际live后态的model/optimizer/EMA哈希。raw r/state张量深拷贝，禁止共享原Adam私有step tensor的可写引用。

D2先从锚点重放H0一次；与随后真实H0后态三哈希同设备精确相同后，才接受其余10分支。新分支使用同一prestate、同一真实梯度、同一post-forward buffer；不重新做训练forward。scope外保持该H0真实一步结果，scope内按05改运算；branch EMA用其自己的完整branch后态更新一次。

D2的M9分子由H0本路径`mB + sum(cell_M9)`重构，并用原native `.9`偏差修正；不是从独立B1加载。C1的候选分母从同一路径vmix/vsep的本步更新构造；C2用本路径U0。每分支读取完即释放虚拟模型，只有数值记录写盘，不保留11份模型文件。

D1每个锚点用独立eval clone的一次Train探针forward、最多六次autograd得到04固定逆概率pair损失的梯度，不重复反传已释放的live图，结果不反馈。Train探针/V_A/V_B面板和固定pair集合按04一次生成并登记hash；评价只在eval/no_grad下进行，结束恢复live mode和RNG。正式AUC输入概率，logit只用于margin解释；不得为了放大局部变化把正式AUC改成logit版。

## 6. RAM状态、恢复与selected模型

每个job的逻辑身份由不可变`RunSpec`生成，至少含study、dataset、seed、arm、完整参数、backbone、scope、预算、source/data/group hash；attempt只标物理恢复，不改变逻辑job key。

| StatePacket必需字段 | 内容 |
| --- | --- |
| `identity` | schema版本、job key、source/data/config/group哈希、设备逻辑身份、owner generation |
| `progress` | epoch/global_step/事务号、permutation/cursor、原始成功步数、物理重放步数 |
| `online` | 参数、buffers、model mode；不只保存parameter向量 |
| `optimizer` | ordinary参数组/私有step；mB、各V、Q/U/T、M shadows、全局k及N适用的k_p、原辅助字段 |
| `observer` | 原六格shadow、新诊断shadow与来源计数；每项shape/dtype及递推版本 |
| `ema` | 参数、buffers、mode及固定rho |
| `rng_loader` | Python/NumPy/Torch CPU/实际worker CUDA RNG，loader generator与批序 |
| `selected` | online/fixed_ema各自最佳epoch、完整Validation值、模型/buffers及hash |
| `receipts` | 已完成端点、锚点、成本、失败/恢复记录和当前授权scope身份 |

序列化训练张量只进入RAM guardian；代码、hash、数值指标、允许的预测记录和报告可以写盘。不得把optimizer、model、EMA、梯度历史、恢复packet或模型压缩包放入磁盘临时文件。guardian提交ack不足以验收：必须实际`get`，重算完整packet hash，并核对owner generation与job identity。[S5,S7]

运行开始检查当前guardian真实status/get；旧发表模型和归档任务的RAM状态已释放，历史socket/PID收据不能表示当前仍可复用。有可用既有guardian时按真实owner generation使用；没有时采用当前受监督前台runner进程内的RAM对象库，保留同样commit/get/hash接口，不新建替代后台guardian、独立AI或自主研究driver，也不以磁盘持久化绕过RAM约束。状态丢失只从同源初始化/已有合格RAM恢复；不挑选最有利attempt，已消耗成本累加。

每个epoch评价online与EMA，只有该读出完整Validation Macro AUC严格大于先前最佳才覆盖其selected RAM；精确并列保留最早epoch。选点同时绑定六格AUC/LL、Traffic/Pooled和模型hash；不按LL另选模型。其他共同epoch预测由D4记录，不另存磁盘权重。

父轨迹全部诊断完成后可释放不再依赖的锚点和observer；参与最终冻结的selected模型保留到本轮允许的Test完成及报告核验。若权限不含Test，按08输出Validation/freeze状态，并明确所需RAM状态是否仍可保持；后续恢复仅恢复同一冻结模型身份，不重新选择更好的checkpoint。

## 7. 条件适配合同

### 7.1 HyperGate

由原 `experiments/overnight_ab_20261103/published_models.py` 构建HyperGateAdapter，遵循 `published_training.py` 的原生`model.objective(x,y,mask)['loss']`；其components全部进入日志和base梯度。新base为完整原生目标梯度，不把原辅助目标丢掉后只留平均BCE。原三门、4专家、expert[256,256]、gate[256]、tower[256,128]、dropout .3及有效原训练配置保持。[S8]

四臂HG_B0/HG_B1/HG_H0/HG_Cstar均采用batch1024、lr .001、WD0、原生Adam .9/.999和eps1e−8；ML782步/epoch，KR1960步/epoch，full50。PAIR仍同batch单边界2048对有放回，条件键在四臂间相同；额外PAIR来源不覆盖native auxiliary components。保持模型原随机流和dropout语义，不能通过重复forward分别求base与PAIR而引入不同mask。

原发表 `_adam` 未显式设置foreach/fused/amsgrad，而由该冻结PyTorch实现采用默认值；保留原adapter参数组和实际默认身份，记录最终resolved配置，不把M3oE主矩阵的显式False旗标无声覆盖HG。HG_B0及lambda0调用原发表优化器；HG_B1的完整mixed目标仍走它；HG_H0/HG_Cstar的ordinary部分保留其策略，自定义共享部分执行05明确FP32递推。HG候选零点委托同一HG_H0适配器，连同真实图、时钟和EMA作等价核对。

HG共享dense组通过原模块调用关系与六格真实梯度可达性登记，不套M3oE名称规则。结构共享而当次梯度为零仍归共享；真实只支持一个cell的参数归相应专属组。主迁移不将共享Embedding加进custom scope。无可用同forward logits接口时，增加只读hook返回原图logits，并验证概率与原forward逐值相同，不重新调用模型。

**时间核按原数值直接迁移：** H0/C1/C2的betaP=.99；若08最终选择S_C1/S_C2简化代表，则Cstar的betaP=.9。betaB=.9、beta2=.999及同dataset固定rho均不做sample-clock换算，每真实optimizerstep后一次EMA。batch变化导致每个历史核对应的样本跨度变化，逐臂记录PAIR事件率、平均实际batch及有效跨度；四臂公平采用同一声明。迁移首要结论用online，EMA作为同轨迹补充，不另做时间核HPO。

Cstar的完整`family/representative_id/zeta/alpha/betaP/readout`来自06/08冻结；不能只复制zeta/alpha而把S代表悄悄还原为.99。P/M覆盖时按08结束相应家族扩展，不自行把它们代替HG_Cstar。

### 7.2 Gaussian

只有06/08触发后接入GA31-R3原D0实现：ML kappa .05、KR .1、lambda_aug .5，保持其原覆盖含义、目标权重和batch级噪声身份。新base为原GA31-R3混合目标，PAIR来自同一个clean原forward的条件与logits。四臂噪声与PAIR索引各自匹配；额外forward/backward次数、有效标签暴露和峰值资源照实记账。[S6]

具体base目标固定为

\[
L_G=\frac{\sum_{i,t}\ell^{clean}_{it}+.5\sum_{i:d_i=0,t}(\ell^{view}_{it}-\ell^{clean}_{it})}{2B}.
\]

每个D0样本都产生一个Gaussian视图，.5是该视图替换clean损失的权重，不是随机抽一半行。非目标域保持clean损失及自然全batch分母。每个非域稀疏字段f的16维embedding使用`e_view=e_clean+kappa*stopgrad(sqrt(mean(e_clean**2)))*noise_f`；scale逐行逐字段计算，domain embedding排除，ML/KR分别6/33个扰动字段，确切顺序由源`perturb_slices`登记。

唯一噪声键为`GA31-R3|{dataset}|{seed}|train|{global_step}|gaussian|1`，global_step从1起、跨epoch连续。种子是SHA256前8 bytes的小端整数再与`(1<<63)-1`按位与；设备本地`torch.Generator`依源字段顺序逐字段调用`torch.randn((target_rows,16))`，再stack到`[target_rows,fields,16]`。键不含新arm，不能一次randn整个三维矩阵替代原调用顺序。

精确代码在`c8675ae…:experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/{objectives.py,views.py,protocol.py}`；D0覆盖适配在同提交`experiments/gaussian_domain_coverage_20261104/objectives.py`。后者原实验只用于KR、内部硬编码`kuairand`；ML必须调用下层通用入口并显式传`movielens`，不能无声沿用KR随机键。新组合适配先用hook取得一次clean预测及logits，再做目标行view前向并组成上式；不在已完成clean前向后直接调用会再次前向的旧`source_losses`。PAIR只读取这一次clean图，view前向的hook结果不覆盖clean logits。b是完整L_G native梯度（加一次WD）减去PAIR项，与05通用来源规则一致。

不能用 `v3.config` 中一个同名kappa字段推断其增强实现与GA31-R3相同。原目标/噪声闭包未恢复时仅暂停Gaussian模块，clean结论照常完成；未运行不记录为负结果。

### 7.3 AutoIFS正式基线

AutoIFS作为发表模型参照，沿原batch4096、search10+reuse50、原搜索分组与学习率、rewind、selector移植/hard ticket、reuse fresh optimizer及已审查bias接口修正。E10/E50指reuse阶段，不把search计入50epoch窗口。只有基线身份不匹配才按06恢复；不给AutoIFS增加C1/C2方法臂。[S8]

## 8. CLI与产物接口

新入口完整命令见00；所有调用先读manifest并验证schema版本。下列产物位于 `artifacts/history_information_20261124/`，报告位于 `research_reports/history_information_20261124/`，除明确RAM对象外只包含允许落盘的元数据/数值。

| 命令 | 必要输出 | 禁止的隐式动作 |
| --- | --- | --- |
| `audit` | SOURCE/DATA/PARAMETER_SCOPE manifests、REUSE_REGISTER、TEST_POLICY、具体BLOCKERS | 不训练，不评价新Test，不默认运行旧全数据cache脚本 |
| `check --scope scientific_contracts` | `IMPLEMENTATION_CHECKS.json`、coder report、独立review输入 | 检查只覆盖第9节合同，不扩展科学网格 |
| `materialize --stage ...` | `QUEUE.json`、RunSpec、依赖与trigger理由 | 不跳过未满足依赖，不自行调参或追加臂 |
| `run --stage ...` | 每epoch/step日志、RAM收据、Validation端点、costs | 不运行未实例化臂，不在训练末尾自动打开Test |
| `analyze --stage ...` | 配对表、选择表、decisions.json、阶段报告 | 不读取未授权Test，不把某seed更优attempt替换原结果 |
| `freeze` | `FINAL_FREEZE.json`及selected RAM实际读回收据 | 依赖/控制未结案时不冻结；不在冻结时重选方法 |
| `evaluate --split test` | 冻结端点Test表、访问收据、冻结hash | 无权限/状态hash不符即拒绝；结果不能反向改队列 |
| `report` | 12定义报告、数值附录、复用/成本/决策表 | 未运行项标清，不能把计划写成完成 |

单条原始记录共同字段：`study_id,job_key,attempt_id,dataset,seed,backbone,arm,parent_arm,representative_id,scope,config_hash,source_hash,data_hash,group_hash,readout,split,checkpoint,epoch,global_step,model_hash,status,cost_class`。不同readout共享job key，attempt只属物理执行。

`steps.jsonl`增加native None mask hash、PAIR事件/条件ID/index hash、w/q、支持计数、两类时钟摘要、只读组诊断及工作计数；`endpoints.csv`增加完整09指标与selected策略；`diagnostics.csv`增加anchor/branch/panel hash及虚拟成本；`costs.csv`区分成功步、物理重放、autograd、推理、RAM拷贝、worker/wall时间、GPU/RAM峰值。

缺失值用JSON null/CSV空字段和`missing_reason`，不写NaN/Infinity，不以0.5补缺AUC。任务终态至少区分`PLANNED,READY,RUNNING,COMPLETE,NOT_TRIGGERED,TECHNICAL_PAUSED,SCIENTIFIC_CLOSED,TEST_PENDING_AUTHORIZATION`，每个条件臂有明确终态而非静默缺行。现行资源和停止控制的错误只暂停依赖任务，不改变实验定义。

## 9. 有限、直接影响结论的实现验收

这些检查在后续IMPLEMENT/REVIEW阶段执行；本次规划不运行模型。优先用固定小张量验证状态代数，再用两数据seed2022的原生首批/尾批核对真实图；检查batch不计作完整科学轨迹，成本另列。只在发现具体失败时扩大相应检查。

| 合同 | 输入与通过条件 |
| --- | --- |
| 来源/人口/组 | 固定源码闭包和Train/Validation身份匹配；参数scope互斥全覆盖；原形状、域Embedding与计数通过；真实tail保留 |
| 原入口等价 | 按当前背景解析lambdaPAIR0父入口，G_B0必须保留Gaussian；主矩阵B0与原clean、H0/.99候选零点与原A21精确相同；HG/N/G的零点与同scope父入口一致，包含所登记shadow |
| 梯度和一次提交 | 同一forward的raw r和完整native grad；PAIR scope外梯度不丢；WD/参数/EMA各一次；无PAIR、raw zero、合法raw None和native None按05逐项验证 |
| 二阶与使用顺序 | 固定含正负cross的序列，C1凸组合和cross闭合；C2固定D相等、时变D满足交换差；不得对U二次预条件 |
| 同轨迹M与近邻控制 | M半径使用自己的shadow、完整拟提交步与两个固定组；zero例外正确；S同时改Q/U核；P_C1时钟/外部lambda、P_C2顺序/WD/eps符合05 |
| 无反馈与锚点重放 | observer前后live/RNG/loader守恒；真实H0锚点重放三哈希精确一致；D2各branch互不共享可写状态；虚拟分支不递增真实step |
| RAM事务和恢复 | 提交后实际get/hash；一次同源中断恢复的后续model/optimizer/EMA/PAIR索引与连续执行相同；selected和owner generation核验 |
| 评价和封存 | 最早严格最大选点、LL绑定、概率tie/缺支持、同轨迹双读出计数；Test入口无合法freeze/授权时拒绝，新Test不能触发新臂 |
| 条件适配 | 仅触发时检查HG原native objective/components/dropout/logits和小批步数；Gaussian首/尾batch噪声及base身份；AutoIFS阶段恢复身份 |

代数关系用FP64小张量误差上限`atol=1e−12, rtol=1e−10`；FP32新式与独立参考计算采用`atol=1e−7, rtol=1e−5`；仅原六格shadow重构沿源`atol=1e−6, rtol=1e−4`。原数值旁路与同设备真实参考重放使用hash精确相同，不能用宽容差掩盖另一条浮点执行路径。不同硬件的可比性另记来源，不把跨硬件bitwise一致作为进入科学训练的新要求。

coder输出具体检查名、输入身份、最大误差/实际hash、PASS/FAIL和修改文件；独立reviewer检查实现与05/本文件的一致性。所有blocking问题修复后才能materialize正式运行。完成这些有限合同后停止重复检查，进入既定完整训练。

## 来源

- [S1：v3协议与原目标入口](https://github.com/Caelestis1903/My-Research-Project/tree/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_information_momentum_20261102_v3)。
- [S2：原shared_name与PairOptimizer](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_information_momentum_20261102_v3/optimizers.py)；[HistoryOptimizer](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_history_shared_update_20261122/history_controls.py)。
- [S3：锚点与真实历史runner](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_history_shared_update_20261122/history_runner.py)；[a_localization.py](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_history_shared_update_20261122/a_localization.py)。
- [S4：参数坐标registry](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_history_shared_update_20261122/b_virtual.py)。
- [S5：RAM capture/restore/commit](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_history_shared_update_20261122/state.py)。
- [S6：EXP080设计、数据哈希、闭包与训练状态规定](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/research_reports/conditional_history_shared_update_20261122/DESIGN_SPECIFICATION.md)。Gaussian原来源的精确实现身份另见11。
- [S7：仓库AGENTS](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/AGENTS.md)；[start.md](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/start.md)。
- [S8：统一发表模型报告](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md)；[固定发表模型实现](https://github.com/Caelestis1903/My-Research-Project/tree/c8675ae2923c15b6965884b140b5b646690cd95e/experiments/overnight_ab_20261103)；[data_bridge.py](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/experiments/overnight_ab_20261103/data_bridge.py)。
