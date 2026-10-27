# Runtime numerics addendum：跨 batch 的 FP32 部署一致性

日期：2026-10-27。状态：DESIGN_COMPLETE。范围仅为 diagnostics.latency 的 raw-X / cached-prefix 验收及失败信息记录；需 coder 实施后独立 reviewer 通过再恢复矩阵。

## 1. 具体失败与实测证据

首轮 MovieLens seed2022 的 READ 完成20epoch后，在 latency 的 batch256 比较触发纯绝对误差1e-6阈值。原异常未保存差值，worker退出，E尚未解锁。为定位问题，coder使用相同seed、固定预算和既有缓存重建native/source/READ，三者selected state hash均与失败run一致；该重放单独计成本，不作为新增训练重复。

已只读检查证据脚本 artifacts/mdmt_relation_basis_v2/diagnostic_repair/reproduce.py 和 seed2022_v2/{REPRODUCTION.json,point_differences.json}。脚本未修改训练函数或容差；用既有 Data、build_native、fit_source、train_method 重现原身份，并比较同一批row的缓存与raw前向。

| 检查 | 实测最大绝对误差 |
|---|---:|
| native相同4096行上下文重算 h / z | 0 / 0 |
| source相同40960行上下文重算 phi | 0 |
| 跨 batch1/256 的 native h | 最高1.6689300537109375e-6 |
| 跨 batch1/256 的 native z | 最高9.5367431640625e-7 |
| 跨 batch 的 foreign phi | 最高4.1961669921875e-5 |
| READ batch1 final logits | 2.384185791015625e-7 |
| READ batch256 final logits | 2.384185791015625e-6 |
| 完整Q40008行，raw4096与既有缓存最终 logits | 0 |
| 完整Q六格/宏/自然 LL及六格/宏 AUC差 | 全部0 |

batch256最大误差点为row205/task1，cached=-4.291538238525391、raw=-4.291540622711182，相对误差约5.56e-7。source原始hidden的跨形状差约1e-6，T0标准化器存在11/11/16个std下限维度（.001）；归一化放大了部分表示误差。相同cached h上重新使用source原40960上下文，phi逐位相同。真实域选择、foreign屏蔽、冻结eval状态及实际模型身份在此次检查中一致。

依据这些结果，当前失败来自把不同GEMM批形状的FP32结果要求为纯绝对1e-6一致；没有发现会改变本次Q科学比较的逻辑错。该结论限定于这条已重放路径；未来超出以下固定验收标准的差异仍应停下定位。

## 2. 唯一允许的验收修改

同形状、同上下文的缓存再现、教师转换和RAM恢复验收继续使用原绝对1e-6，不增加相对容差。它们与跨batch部署测试是不同检查。

仅 diagnostics.latency 内，batch1和batch256的最终 logits raw-X / cached-prefix 比较改为逐元素：

`abs(raw - cached) <= 1e-6 + 1.3e-6 * abs(cached)`

先断言相同shape且两端全部finite；使用cached为明确参考端，equal_nan=false。atol保留原1e-6；rtol=1.3e-6采用本机已安装PyTorch2.9的float32测试默认相对容差。只读依据是 phase0-3-env/lib/python3.13/site-packages/torch/testing/_comparison.py 的 _DTYPE_PRECISIONS：float32默认为(rtol=1.3e-6,atol=1e-5)。本补充的atol比该默认严格十倍，参数在恢复矩阵前一次冻结，不能按不同方法/seed再次调整。

原失败点允许误差约6.58e-6，实际2.384e-6通过；接近零的logit仍主要受原1e-6限制。明显差异、shape错误、NaN/Inf仍阻断。不得只把max_abs阈值统一放大，也不得对h/phi套此最终logit容差后掩盖接口错误。

通过时在现有costs/latency记录中保存batch、row_id范围/身份、atol、rtol、comparison_reference、finite、passed、max_abs、max差值位置及该位置两端值；可同时保存mean_abs。失败时这些必要数值须进入异常消息或失败JSON，再raise，保证worker FAILED.json能复核实际原因。无需保存模型或新增大数组。

本次实测已经对完整Q重算并得到指标差0；不要求为每臂每seed再新增完整Q raw前向。正常batch1/256检查只使用既有固定Q前缀行，不读E，不用面板指标选择模型或修改容差。50warmup/200计时次数、raw-X实际batch、cached时延路径及算子统计保持原定义，不通过padding更改实际部署成本。

## 3. 最小实现验收

coder只改上述比较及记录，补一项小CPU断言覆盖：原失败数值通过；近零超过原atol的差异拒绝；明显大差异拒绝；NaN/Inf和shape差异拒绝。保留已有同上下文/恢复1e-6测试。没有必要重跑完整GPU研究矩阵来验证这个纯比较分支；原重放已提供实际FP32数值与Q指标影响证据。

不改模型、数据/角色/归一化、native/source/适配dtype、训练缓存生成的chunk、训练row计划、预算、seed、优化器、checkpoint候选或LL/AUC计算。此前失败/主动中止run保留记录；恢复矩阵用新的运行根目录/源码身份，失败轨迹不冒充完成。所有必要重放实际成本单独入账。

## 4. 对科学判断的影响

研究比较继续使用同一冻结前缀缓存、同一selected状态与既有FP64 logit指标。相同数学模型的batch1/256部署误差单独量化，不能据此换checkpoint或接受新的研究变量。此次完整Q的logits和指标逐位一致，故没有证据要求改变方法比较或追加研究ablation。

本补充结束后回到原48条适配轨迹及预注册诊断流程；不扩展seed、数据或方法。尚未产生的E和研究结论仍按全局冻结及既定分析流程执行。

## 5. 修订 R2：采用标准 FP32 跨 batch 容差

本节是恢复后出现第二个真实数值失败的必要修订，覆盖第2、3节关于跨batch的atol=1e-6及相应近零拒绝测试；前述原始事实与处理历史保留。

可复核证据：artifacts/mdmt_relation_basis_v2/revision1/workers/kuairand/seed2022/FAILED.json。KuaiRand seed2022 的 B_REFERENCE 完成20epoch后，batch256 两端shape均[256,2]且全部finite；最大绝对logit差为1.341104507446289e-6，mean_abs=3.2409298000857234e-7。最大点[70,0]的cached=-0.07308383285999298、raw=-0.07308517396450043，旧混合容差允许1.0950089827179907e-6，因近零值仍受过严绝对项限制而拒绝。该run实际耗时255.443536965秒，须保留失败及完整已发生成本。

当前唯一跨batch final-logit验收标准修订为本机torch.testing的**完整float32默认容差**：

`abs(raw - cached) <= 1e-5 + 1.3e-6 * abs(cached)`

使用cached为参考，先检查相同shape与全部finite，equal_nan=false。采用本机现成dtype标准，不再另造比其严格十倍的绝对阈值。新点允许约1.00950089827e-5；既有ML和KR异常点均通过。该标准只用于diagnostics.latency中不同batch几何的部署logits；原同上下文缓存、教师转换、RAM恢复的绝对1e-6保持不变。

这是工程parity标准的修订，不涉及模型训练、功能形成判据、科学风险门槛或研究超参数。稳定BCE对logit的一阶导数绝对值≤1，因此本次固定256行面板的natural LL差绝对值≤mean_abs=3.24093e-7；每请求LL差≤1.34111e-6。sigmoid导数≤1/4，最大概率差≤3.35277e-7。这些是该已观测面板的数值上界，不把它写成完整KR测试集的实测指标。AUC不由LL的Lipschitz界推出；本次不追加AUC实验，因为修订比较函数不改变任何缓存预测、排序或既定指标计算。

## 6. R2 的最小实施与测试

coder仅改跨batch比较的atol常数及对应工程测试，保留已有max/mean差值、最大位置及两端值、容差、batch/row身份的记录。小CPU断言纳入这个真实近零点通过、原ML点通过、明显超过新容差的近零/大值差异拒绝、NaN/Inf/shape差异拒绝。所有模型/训练模块、data/cache生成、effective研究配置、seed、预算、优化器、选点、指标和诊断科学公式保持原字节。

不启动新的重现训练，不给每个run增加完整Q raw重算，不追加新的验证或研究矩阵。只恢复本来要求完成的轨迹；独立reviewer核对新比较分支、测试及下节身份等价证据后放行。

## 7. 保留存活worker及诊断修订身份

已通过旧标准的实际比较点自动通过新标准，因为仅atol从1e-6单调增加至1e-5，rtol/shape/finite规则不变。不得为了统一工具revision结束或重跑仍持有RAM模型的worker。它们继续使用实际已加载的旧诊断实现，保留真实完整source_hash。最新实际FAILED且失去RAM的是KR四seed；存活/READY为ML四seed。执行时始终按实际FAILED/READY记录处理，不根据预计失败预先终止进程；只重跑实际失败且RAM状态丢失的组，新run使用R2工具revision，原失败轨迹先归档保留，不覆盖。两工具view按真实source_hash分组，当前与dataset的分组一致并不构成授权规则。

冻结前建立一份最小的修订等价证据，保存两个实际源码版本快照（代码文件允许落盘），以及每版本逐文件SHA清单。快照必须与worker当时source_identity.json的真实清单匹配，不以当前源码覆盖旧版本后再宣称旧快照。将完整来源身份与训练身份分开记录：

- full_source_hash仍是实际全包来源身份，保留每worker原值，不重写。
- training_source_hash为同一文件集合去掉仅diagnostics.py和tests/后的规范化逐文件哈希摘要。两个版本的余下文件集合和每一字节须完全一致，不能只比较部分抽查文件。vendor闭包、config/data/native/source_functions/relation/model/losses/trainer/state/campaign/metrics/report/cli均在此比较中。
- 单独记录diagnostics.py和tests/的diff，独立reviewer确认实际变化限于跨batch工程比较与对应测试，不改变diagnose科学诊断或raw_forward的数学预测。
- 方法/effective研究config身份、dataset角色/预处理身份、既定预算/seed必须一致；各seed实际native/source/selected状态hash仍各自记录，不要求不同seed的权重相同。

不能仅排除整个源码目录或放宽数据/训练身份验证来容纳修订。若实际差异超出上述边界，先查明具体差异，不把它归入诊断等价。

## 8. 含不同诊断版本的全局冻结

global freeze允许同一研究配置和训练字节、不同工程诊断工具revision。冻结必须包含全部8个worker、每worker真实完整身份、共同training_source_hash、诊断版本映射和源码等价证据身份；E仍在这些信息锁定之后统一解锁。

当前已驻留旧Data.authorize_final检查manifest顶层source_hash等于自身full source_hash，不能给它一个被改写成新版本的假身份。最小兼容方式是先冻结同一份canonical聚合内容，再为实际工具revision生成兼容view：各view含相同全部8个真实worker条目、同一method列表、同一effective研究config、同一训练字节等价证明，仅顶层source_hash取该view所服务worker的真实完整source_hash。旧七worker读取旧revision view；新KR2022通过既有freeze_file参数读取新revision view。每worker的entry和EVALUATE release身份都保持精确匹配。

canonical冻结记录各view的实际哈希/路径及worker映射；各E评估结果可记录实际授权view哈希并回链canonical聚合身份。不能伪称两个JSON字节相同，也不能删除既有source_hash比较使任意源码通过。此方式复用现有授权接口，不要求给已驻留worker热替换代码或结束RAM生命周期。

阶段判定继续聚合48条冻结配置轨迹，不把工程revision分成新方法、不将重放计作独立seed、不把不同工具revision自动视为科研协议不同。最终报告保留R1/R2失败原因、容差修订、完整来源与训练身份等价证据及实际额外成本。

## 9. 已加载旧验证器的具体接线

不修改 data.py、campaign.py、cli.py、report.py 或已驻留worker。现有report.freeze_manifest会强制所有worker的full source_hash等于当前源码，因此本次由report目录内一个专用、可审查的冻结生成脚本实现第8节兼容view；它只接受以下两种完整来源身份：

1. 已审查R1：`72f3e825dac422784fd575bd24450e7858e99ce138f6c8261b3ae7636b8b13fc`。
2. R2标准FP32比较修订的实际full source_hash：由coder完成新源码快照和独立reviewer PASS后写入显式白名单；禁止采用“任何训练摘要相同的来源均接受”等开放条件。

两来源必须同时通过第7节逐文件等价证明。生成器仍执行现有freeze的必要合同：全部八个READY身份、同effective config、六方法完整selected/final身份、stage2已关闭、实际decision与thresholds锁定；不能仅因source属于白名单就省略这些检查。

旧revision存活worker沿原路径读取 `root/GLOBAL_FREEZE.json`，其顶层source_hash为上述R1值。新重跑worker启动时通过既有 `--freeze-file root/GLOBAL_FREEZE_FP32_STANDARD.json` 读取R2 view，顶层source_hash为R2真实值。两view的workers、methods、config_hash、thresholds、decision及canonical hash完全相同，每worker条目保留真实source_hash。现有release_evaluation从默认view查询worker identity即可服务两组，因为两view含相同完整身份表；每worker EVALUATE release仍精确匹配它自己的identity。

为避免哈希循环，先生成immutable canonical core（完整workers、methods、config、thresholds、decision、两版本白名单、训练字节等价证明），计算core hash；两view都嵌入此core hash。再生成canonical索引，记录core hash及两个最终view的实际哈希与路径。不可让core本身包含view哈希同时又让view包含该core hash。

R1/R2的通过判定均为逐元素all。最大绝对差位置可能因为该位置reference较大而满足其自身相对容差，其他近零位置仍可能违规；日志里的max_abs/max_index用于定位，不能用“最大差点通过”取代逐元素检查。若保留失败日志可附违规数量或一个违规位置即可，不要求额外GPU重算。

原失败目录由runner先归档、保留真实source/数据/成本与错误，再在原规定worker路径重跑以复用既有8worker清单和release路径。尚存活组不移动、不覆盖、不重跑。最终失败/重跑数量由实际状态记录得出，不把撰写过程中的暂态数量固定为执行规则。
