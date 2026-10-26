# 独立审查：冻结后 raw-X 重建观察器

Verdict：**PASS**。Blocking issues：无。仅授权固定观察器修订及正常 evaluate 重试；尚未认定真实训练模型的失败已修复。

已独立审读 DESIGN_RAW_RECONSTRUCTION_OBSERVER_ADDENDUM.md、CODER_IMPLEMENTATION_REPORT_RAW_RECONSTRUCTION_OBSERVER.md、实际最终三文件、原 baselines/readers 源字节快照、原失败路径和实际固定系数收据。Coder 与 reviewer 为不同 agent；reviewer 未连接或修改驻留 service。

原始实际错误必须保留：ML5201 FULL_SOURCE_ALL 的完整 raw-X 概率误差约2.19767594e-6，超过1e-6；原生全行 H/Z 误差0和已有4096行计时不能替代此失败。失败记录未定位的具体 row/role 不作推断。该实际冻结 reader 的最大绝对特征系数为8.004369684959386，因而隐藏内容的微小偏差可能被放大；本审查不调整其系数、选择或预测。

最终实现复现实际缓存的计算顺序：完整 role 按40960在各自原设备执行 pilot Linear/ReLU，再以冻结统计 FP64 标准化、转 FP32、最后筛 target。RANDOM统计来自独立 raw-X fit H，未读训练 hidden cache。pilot residual 使用原一维 NumPy dot。formed 函数只对冻结 module 作 RAM 副本，回到形成设备按完整 role/16384执行，每块转CPU后按原 scale 相乘，最后筛行。原 module 的设备、模式、梯度和训练状态不动。

原 baselines 定义 AST 仅 _raw_source/raw_features 两个观察器改变为固定转发；全部其他原定义及全部 reader 数值定义相同。独立检查使用实际 fit_reader 创建的既有 validator（mock已拟合候选、没有训练），确认其在调用时解析 baselines 当前观察器符号。旧 callable 指纹由原快照以原绝对 co_filename 编译后只取 code object 得到，未执行该快照，不把新磁盘转发函数误作旧驻留函数。

安装仅接受八个指定 worker PID/父PID2088990、精确 config、原 FROZEN SHA、精确正常 _worker reload-readers frame、显式 PrefixBundle/dict 类型及冻结状态。独立授权绑定三个源码 SHA 和旧/新 callable 指纹。只替换已加载 baselines 的两个固定观察器，不 reload 整个 baselines 或训练模块。安装前后完整指纹覆盖 native/source/local/base/direct/gate/formed及terminal模型的参数、模式、设备、requires_grad、已有grad，reader系数/selection、normalizer、全部预测数据、RAM optimizer/训练状态与RNG；训练函数对象身份也必须保持。parent/client/fresh非reload进程 NO_OP。

独立实际数值 fixture 使用40979行跨两种分块边界，source/local在CUDA、initial在CPU。五个固定内容分支与原 PilotAccess oracle 的误差全部0；完整role GPU16384的冻结DeltaFunction副本误差0。对照旧先mask source路径误差5.7220458984375e-6，旧CPU/mask delta误差1.7881393432617188e-7。构造期间训练phi、FeatureBank.values和function.outputs由禁止访问字典保护，初始统计确由独立fit packets取得。模型参数/模式/设备/梯度及CPU/CUDA/Python/NumPy RNG保持。此为数值实现fixture，无正式训练、无真实驻留模型接触，不冒充真实训练case的after。

随后新增的最小末端审计已单独检查：raw_features的全部输出完成后才进入比较函数，返回值不取决于oracle。只有已正常安装、真实指定worker/父PID、fit及原FROZEN SHA匹配时，才对FULL_SOURCE_ALL的一个source/task raw hidden与标准化phi、RCF的一个实际formed输出读取缓存作为比较oracle，并另记录旧CPU/mask误差。比较仅写小JSON，不影响输出、p、policy或模型。真实logit误差明确NOT_MEASURED，由既有全部方法完整raw-p≤1e-6验收覆盖。核心已通过fixture未重复运行；审查只追加末端比较顺序和身份守卫检查。

独立收据：RAW_RECONSTRUCTION_OBSERVER_INDEPENDENT_CHECKS.json。最终源码/新旧callable及原FROZEN字节绑定已核对。正常RPC安装、八worker完整不变指纹、实际ML5201 after、每worker实际formed比较以及全八context全部实际方法完整raw-X和导出验收仍是RUN义务；若仍失败，不放宽1e-6、不改p、不重选或重训。

本次全源码授权为观察器身份，不倒填训练身份。OBSERVER_RAW_RECONSTRUCTION_AMENDMENT.json同时记录原冻结训练身份与本次观察器身份。原FROZEN、配置、四种子验收、训练及预测保持不变。失败导出数据按已批准清单保留；只将可能污染递归索引的旧index/acceptance移至运行目录之外的owned attempts，不移动训练完成证据或模型，不让新index索引自身。
