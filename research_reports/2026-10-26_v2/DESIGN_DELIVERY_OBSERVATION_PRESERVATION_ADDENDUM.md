# DESIGN SPECIFICATION — 推理测量保留与缺项补测

日期：2026-10-26。状态：DESIGN_COMPLETE，待coder实现和独立reviewer PASS。仅修订交付观察器和数据收据；禁止训练、重启、改变模型/预测/选择或放宽容差。四种子训练及FROZEN_MANIFEST不修改。

## 1. 已确认缺陷与最小范围

当前worker export先调用collect_delivery(benchmark=True)，但完成导出后的_summary又调用collect_delivery(benchmark=False)，后者把同路径COSTS_PARAMETERS_AND_INFERENCE.json覆盖成“未请求计时”。因此MovieLens5201真实测量收据已丢失；不能以已有MEASURED口头记录重建不存在的计时数字。

另已实际读取KuaiRand5201仍存在的True收据：RANDOM_FEATURE为NOT_MEASURED，原因是fused/raw reference parity超过不变的1e-6，max_abs_probability_error和retained_prediction_max_abs_error均为5.709476791126988e-6。这是真实拒绝，不能改成MEASURED。它不在delivery.REQUIRED九个必需方法内；本修订**不修编译器/融合公式/设备路径以追求这项延迟**，保留该项不能公平比较latency的边界。所有方法的原始完整raw-X预测验收仍需严格1e-6，与是否测得融合推理时间分开。

本次最低完成目标是两个数据集×四种子×REQUIRED九方法的真实raw-X推理成本及参数收据完整，保留其余方法已成功的实际测量；非关键控制若编译器parity拒绝则明确NOT_MEASURED，不无限修复。九方法固定为NATIVE、B_REFIT、H_DIRECT、H_DIRECT_EFFORT、GROUP_CAL、FULL_SOURCE_ALL、RCF、POOLED_TARGET、LOCAL_ACTIVE，不从效果结果选择。

## 2. 先保留已有证据

在当前evaluate继续执行时，主编排者可以立即将现有benchmark_requested=True收据复制至runs目录之外的owned `attempts/delivery_measurement_snapshots`，写path/bytes/SHA/原mtime/复制时间清单；这些是实验数据文件，允许保留。未来True收据若能在False覆写前捕获，同样完整保留。不能停止/修改正在训练或导出的worker来制作快照，不能从日志片段伪造原JSON。False覆盖文件和已失败RANDOM测量记录也作为缺陷历史保留。

归档快照可成为有效测量来源，但必须核对真实文件SHA、prefix/config/dataset/seed、方法/冻结policy与4096行规则等身份；不能把旧v1两种子收据或其他模型的测量混入新矩阵。没有完整真实快照的丢失测量只能在同一冻结模型上重新测量，标明新的测量时间和“因收据覆盖补测”，不声称找回原计时。

## 3. 持久收据与幂等合并

修改delivery.collect_delivery及最小必要helper，保持原训练成本公式、参数定义、FrozenDeployment、benchmark_predictor的数值实现和3warmup+10测量、自然原序4096行、同步/传输范围、1e-6 parity全部原样。新收据记录方式：

1. 真正benchmark结果先写不可覆盖的版本化测量记录（方法粒度或完整批次均可），包含原始10次时间、row hash/实际row IDs、parity、设备/环境、源SHA、prefix及方法冻结身份、时间及测量recipe。它位于delivery/measurement_records等owned目录，不位于runs导出递归索引内。原始成功/失败记录均不删除。
2. 现有同身份收据或已校验快照中，每方法已有MEASURED并且10次真实计时、parity≤1e-6、4096原行序、3/10 recipe、环境/设备与冻结方法身份匹配时，复用原记录并注明源path/SHA，不重新测量。仅source文件因本次保留逻辑改变时，不能把旧测量冒称新代码刚测得；保留它原测量source，另记当前汇总observer修订。
3. benchmark=False只更新允许的成本/参数摘要，**绝不发起计时，也不能以NOT_MEASURED覆盖同身份已有MEASURED**。按方法合并，以实际测量状态和来源为准。若没有有效测量，才保留明确NOT_MEASURED。方法清单扩大时，只对新增方法缺项，不借prefix同名复用另一policy的数据。
4. benchmark=True也幂等：有有效MEASURED则跳过；缺失/确实丢失的授权方法才调用原benchmark_predictor。原有明确parity失败的非必需方法保留原失败收据，不在每次summary/reload循环反复尝试。
5. 记录当前调用是否请求benchmark与持久测量覆盖两个不同事实。例如current_call_benchmark_requested=False，但inference条目引用真实已有MEASURED；不能用顶层False误导读者认为已测数据不存在。保留所有方法状态，并另给required_methods、required_measurement_status、missing_required_methods。只有所有实际列出方法均测得才能写全体MEASURED；九方法齐备不能掩盖RANDOM_FEATURE缺项。
6. canonical JSON以临时纯数据文件+原子replace写入；先落版本化记录，再更新汇总，避免半写或再次覆盖不可找回的测量。训练成本CSV可以按相同真实成本更新，不补造秒数。

身份至少包括config/prefix native/pilot、dataset/seed、方法prediction identity与selection/reader policy、形成函数身份（适用时）、自然4096row hash、benchmark recipe、设备/环境。旧快照缺少新增字段时必须用其真实prefix/method收据与当前冻结身份交叉核验并记录依据，不猜造缺失身份；不能确认则视为待补测。

## 4. 当前驻留进程可行的补测入口

不修改或新增cached controller命令，不为此重启或重新evaluate/export/bootstrap。现有正常reload-readers RPC在worker中会硬编码reload delivery模块，因此可以更新collect_delivery定义，并在readers末尾增加一个**固定delivery-only观察器入口**，由专用授权决定是否补测。

新固定入口模块仅在精确campaign._worker.__code__正常reload-readers栈、typed bundle/predictions/formations/diagnostics/config且frozen=True时运行；限定service parent PID2088990、当前八个登记worker、config cefb6d934c774bd78a438e4fba5c560e934f4c56ab09fbb3dfe389f4f4842fd6及FROZEN_MANIFEST原SHA。parent/client/其他导入NO_OP，禁止gc对象查找、任意eval/exec或任意command/路径执行。普通reload只是定义更新；**只有独立PASS授权的专用token明确请求fill_missing_required时，才执行有界补测**。

授权 `DELIVERY_OBSERVATION_PRESERVATION_AUTHORIZATION.json` 固定绑定本设计、review PASS、新delivery/readers/helper源码SHA、冻结身份、每worker允许补测的方法清单（REQUIRED九方法的子集）及已验证快照来源。安装入口调用新版collect_delivery的有界“恢复真实测量/仅补缺项”模式；它只做现有4096行forward/计时/数据写入，不调用export_bundle、bootstrap、训练函数、decide、freeze或finalize。同一授权再次调用必须幂等，不重测已经保存的有效条目。

当前evaluate是controller串行命令，不在其busy时插入RPC。等其自然完成或返回后再执行一次正常reload-readers；所有模型仍保留RAM。代码可以并行完成实现与审查。若此前观察器授权也绑定readers SHA，独立review通过后同步刷新该授权的实际readers SHA，并明确原raw重建数值函数不变，避免旧guard误拒绝；不能改变冻结训练身份。

新入口前后用既有完整冻结指纹比较全部模型/optimizer/RNG/mode/grad、reader系数/selection、p/p_terminal和normalizer。delivery已有benchmark immutability检查继续执行。允许新增版本化观察数据和收据，不修改FROZEN_MANIFEST或任何已通过raw-X验收的预测数据。

## 5. 最少实现与验收

Coder只负责delivery收据保留/幂等合并、新固定观察入口、readers末尾hook及实现报告。其他训练模块、数值reader、编译器图、基准选择和四种子门槛不变；reviewer与coder独立。

必要检查使用小mock计时记录，不训练：

- 同身份True MEASURED→False调用：原10次时间及来源完整保留，不触发benchmark；False→False不虚造测量。
- 相同授权重复fill：有效条目不再调用benchmark，确实缺失的授权REQUIRED方法仅调用一次；方法/policy/row/config不匹配不复用。
- 失败RANDOM_FEATURE仍NOT_MEASURED且原reason/error保留；required覆盖判据与全方法状态分开，不伪造全体MEASURED。
- 版本化记录不可覆盖，canonical原子写；快照SHA/身份校验失败不能导入。
- typed hook错误PID/config/frozen SHA/方法白名单/授权拒绝，parent/client NO_OP；不会调用export/bootstrap/训练。
- FrozenDeployment、benchmark_predictor及所有训练/reader数值函数AST保持原样；安装和补测前后冻结指纹一致。

独立PASS后在正常RPC执行实际RUN，逐worker记录复用/补测/仍未测方法及真实记录path/SHA。最终验收对两数据×四seed所有REQUIRED九方法确认完整3+10原始时间和≤1e-6 parity；其余控制的明确缺项在COST_REPORT中限制相应latency比较。已完成的昂贵导出与bootstrap不重做，模型只在全部原验收及本交付修复完成后释放。
