# DESIGN SPECIFICATION — 冻结后raw-X内容重建观察器修订

日期：2026-10-26。状态：DESIGN_COMPLETE，待coder实现及独立reviewer PASS。本修订只处理实际导出一致性失败，不分析训练效果，不授权任何重训、选参或放宽验收。

## 1. 实际缺陷与不变项

四开发种子矩阵已训练完成，FOUR_SEED_ACCEPTANCE及FROZEN_MANIFEST已生成，S4按登记门槛NO_START。首次evaluate在MovieLens seed5201的FULL_SOURCE_ALL完整raw-X核验失败，最大概率差约2.19767594e-6，超过既定1e-6。原始失败日志/文件必须保留，不能标PASS。父编排者已核该context全行native H/Z最大误差0；4096行推理benchmark不能替代全行验证。

只读源码查明重建路径与实际生成内容路径不一致：PilotAccess._hidden先在完整role按40960分块forward，随后才按domain筛行；baselines.raw_features的FULL类/FULL_TARGET/投影分支却先筛h[mask]再forward。FP32 GEMM的批形状可改变数值，完整特征无界系数可能放大误差；这是待实际fixture验证的根因，不能只凭推测宣布已修复。

另需同步原样重建已使用的formed内容：features_from_functions在形成设备上对完整role调用DeltaFunction.values，其**实际默认chunk是16384**，随后将原module移CPU；旧raw_features按target筛行后在现有CPU module上调用values。4096是独立native/benchmark用的另一个chunk，不能混作formed缓存创建chunk。

全部冻结权重、函数scale、预测p/p_terminal、P_cal候选/选择、lambda、normalizer、群体、训练RNG/Adam、四种子gate、FROZEN_MANIFEST原字节和**1e-6容差**保持不变。不修改训练函数，不改export里的p来匹配观察器，不重训。观察器只能从raw-X独立得到的H/Z和冻结参数重建；禁止以原phi/function.outputs/FeatureBank.values或已缓存预测构造其输出。

## 2. Canonical内容重建合同

输入仍为raw_features(bundle, bank, role, independent_h, independent_z, domain)。保留原角色行序和列序，按以下实际创建路径重算：

1. **source/local/initial隐藏特征**：使用对应冻结head的原设备、FP32、原Linear→ReLU图，对完整输入role按40960顺序分块；得到全role隐藏激活后再做原标准化与float32 cast，最后按target mask取行。source/local使用已冻结的训练normalizer，不重新拟合。不得把h先按target筛选再forward。
2. **RANDOM_FEATURE的initial normalizer**：若没有已保存的initial统计，必须从独立raw-X重新编码的fit H按同一完整fit/40960/initial-head设备路径计算激活，再取source fit行，以原FP64 mean/std、floor1e-3得到统计。不得调用PilotAccess.hidden或读取initial隐藏缓存来计算。可在观察器自己的独立数据命名空间缓存这份32维统计，绑定fit rows/native/head状态与重建recipe；不能冒充原训练记录或修改bundle.normalizers。
3. **pilot residual**：canonical原式是rawhidden @ output.weight.reshape(-1) + float(output.bias)，即一维dot；保留CPU NumPy计算和float32结果。原observer _raw_source的N×1二维乘法若有差别，允许作为第二个明确的observer-only白名单修订，改为与实际PilotAccess.residual相同的一维式；不改变任何训练调用。PILOT_LOGIT仍用source residual减target residual，其余数学式不变。
4. **formed delta**：使用冻结module的独立内存副本，放到缓存创建时的形成设备（本轮为bundle.device，需实际验证），对完整role按16384分块执行同一FP32 module图；每块转CPU NumPy后按原DeltaFunction.values位置乘原scale，再拼接、最后筛target行。不能移动原fn.module、改其training mode/grad/device或读fn.outputs。copy不调用随机初始化、不拟合、不改变RNG；严禁在磁盘保存副本。K1复用函数也用其冻结module/scale重建，不按名称假设新训练。
5. **projection分支**：若实际方法有此metadata，先同source标准化完整role特征，再按原target筛行、以原冻结projection矩阵乘法。没有触发的方法不新增训练或性能分析。

允许同一次raw_features调用复用其**自己刚独立重算**的全role激活/函数值以避免重复forward；不从训练缓存取值。大数组可使用受控数据mmap/分块，不生成全模型序列化。观察器额外缓存与所有原训练/预测数据目录明确区分，绑定hash。纯数据页面可释放，模型仍驻RAM。

## 3. 最小实际验证

实际失败的ML5201全行错误记录可直接作为before证据，无需为再次获取trained fixture增加RPC、栈执行入口或任意运行时调用。先独立审查helper和固定reload入口、做必要的小数值fixture，再正常RPC安装；随后正常evaluate的同一ML5201完整raw-X核验可作为实际after RUN：

- 保留旧“先mask再forward”失败路径及其概率误差的原记录；若现有数据可得，记录最大误差所在role/row/domain/task和对应输入/列误差，缺失定位字段如实标未记录。无需重复运行已证实失败的全套导出。不得用测试标签寻找有利样本，误差定位不需要y。
- 对同一role比较“完整role canonical forward后mask”的新内容与实际冻结内容，记录隐藏/标准化/最终logit及p误差。保留旧失败值，新值须在同一个严格1e-6阈值内。保留实际方法的系数幅度以说明误差是否被放大，但不得据此重选/限制系数。
- 至少包含一个实际formed函数，对完整role GPU16384新重建与冻结输出核验，并对照旧CPU/先mask路径；记录真实创建设备/chunk，不把可能根因写成未测量的断言。无需新增训练或全新实验矩阵。
- 验证独立输出构造没有读取原phi/outputs/p：这些只可作为fixture比较oracle及最终差值检查，不能作为新预测计算的输入。RANDOM统计走独立fit H。
- 安装和无标签fixture前后比较全部native/pilot/本地/base/direct/gate/formed模型参数、模式、requires_grad、已有grad、reader系数/selection、normalizer、RNG及预测数据hash；可复用delivery._observation_fingerprint并补selection/p身份。任何训练状态变化均FAIL。

独立代码/最小数值fixture审查PASS可授权正常RPC安装与实际after RUN；只有实际完整parity支持修订后才认定观察器缺陷已修复。若新canonical路径仍超过1e-6，不增大容差、不改p、不选另一batch凑结果；记录新证据，最小定位剩余路径后另行修订。全八context/全部实际方法的完整raw-X验收仍须在正常evaluate中通过，fixture不是最终替代。delivery的'_delivery_observation'角色是已有4096行独立观测包，按该包完整行序执行相同分块上限，不能访问原角色phi；初始fit统计仍来自独立完整fit H，源/local/initial各自原设备保持（本轮initial_bank原本CPU）。

## 4. 正常RPC激活与冻结身份

新增硬编码observer实现/安装模块；baselines.py只允许raw_features及必要时_raw_source两个观察器函数改变/转发，其余定义尤其所有训练函数AST保持不变。readers.py只增加固定安装入口，所有数值reader函数AST保持不变。不reload整个baselines模块以替换无关训练函数对象；安装时仅把审查通过的固定observer callable赋给已加载baselines.raw_features（及经审查需要的_raw_source），不接受任意目标名称、文件路径、eval/exec或任意代码。

通过现有正常 `reload-readers` RPC在worker端激活。入口仅接受精确campaign._worker.__code__正常reload-readers调用栈的typed self状态（bundle/predictions/formations/config）；验证op、冻结状态、service parent PID2088990、八个已登记worker PID、config hash cefb6d934c774bd78a438e4fba5c560e934f4c56ab09fbb3dfe389f4f4842fd6、FROZEN_MANIFEST原SHA、旧新观察器/reader源码hash及独立PASS授权。parent/client/fresh非reload导入均NO_OP，不再次安装/改动scheduler。不得寻找gc对象或更新controller配置。

现有旧预测validator闭包在调用时从baselines模块导入raw_features，因此固定替换可作用到保留的冻结预测；必须通过fixture证实，不假设闭包自动更新。训练函数对象/AST、模型、旧预测全部保持。正常RPC前后的核心模型hash检查继续执行，另补上述完整冻结指纹和观察器安装收据。

**不得重写FROZEN_MANIFEST、config或四种子验收。** 新建 `OBSERVER_RAW_RECONSTRUCTION_AUTHORIZATION.json`、逐worker修订收据及汇总 `OBSERVER_RAW_RECONSTRUCTION_AMENDMENT.json`，记录原冻结SHA/训练源身份、新observer/readers源码SHA、训练与reader数值AST未变、固定函数映射、实际fixture与review PASS路径。全源码review token可为实际observer文件刷新，但不得将新source hash倒填为训练时身份。最终报告同时展示冻结训练身份和后续观察器修订身份。

## 5. 失败导出与重试

保留首次失败日志及其方法/role/error，不把失败数据当正式验收。重试前对当前已生成的导出文件建立path/bytes/SHA清单。仅将会污染下一次递归文件索引的旧 `ARTIFACT_INDEX.json`、`EXPORT_ACCEPTANCE.json` 以及失败验收收据移至prefix运行目录**之外**的owned attempts子目录；不移动TRAINING_COMPLETE、有效预测cache、模型或四seed证据。当前未生成的文件不伪造。

旧index不能被新index当作自己的被校验成员；已失败导出的其他文件可按原确定性export流程重建，旧清单保留其身份。新index文件集必须排除自身、旧acceptance和attempt归档目录，避免“写完index又改变其被索引文件”的自引用。若现有export的首次重试满足该条件，优先用数据归档解决，不扩大修改导出代码。若实际仍有自引用缺陷，只允许独立审查的最小observer/export索引修订，不改指标。

安装及fixture完成后由runner正常重试evaluate；保留全部八模型直到全方法raw-X、指标复算、功能/成本导出及完整验收实际PASS。仅随后执行原finalize。不得为该观察器错误重启、重新训练或释放模型。
