# GA31-R3 独立实现审查

**裁决：PASS，可进入正式 RUN。** 审查对象为本轮 [设计规格](DESIGN_SPECIFICATION.md)、[实施报告](CODER_IMPLEMENTATION_REPORT.md)、`experiments/gradient_aug_20261031_v2/` 实际实现，以及 R3-r1 的04/05/JSON合同。coder与本reviewer为不同agent。

## 合同与实现

- 核对17个vendor必要文件的实际字节与已固定源文件及来源SHA256一致；沿用协调者对 `ee6e81fdfd37284e6447e8149cb93d04736dba02` 的远端blob核验。新模型保留参数键、顺序、fixed1结构、LayerNorm与gate detach，无新增参数、learned fusion或DARTS。
- 视图在实际非路由sparse字段Embedding切片上执行；M1逐样本均匀删除一个完整字段，无逆缩放；Gaussian使用clean RMS的stopgrad尺度。IID2和ANTI2首视图使用同一key，ANTI第二视图严格取负。原ID、domain Embedding和dense保持；K视图合成始终采用原B×T分母与每原样本总系数1，每真实batch一次Adam更新。
- A全部15run在预定窗口的真实更新前调用只读`autograd.grad`，固定参考行、两独立噪声，保存带符号S/C；参数、已有`.grad`、Adam、loader、模式及全局RNG执行前后逐项核对。120状态/360额外梯度，未添加正式更新。
- 离线probe在最终工作基线的E1/E10/selected去重快照运行；每观察真实full/target加12个target视图，共14次梯度。IID信号估计与反向相关组分开；方差、协方差用FP64，保留负S，校验直接配对均值方差。五个反事实分支各自联合deepcopy模型与Adam，保留参数绑定、None语义、源L2/epsilon及状态计数，两个validation panel互不重叠并使用完整validation人口权重。
- ML基线只用A阶段selected validation：满足实用替代与保护条件则优先L2，否则按validation均值及规定平局规则；KR固定B-KR。ANTI门只取E10/selected去重的有效观察，先分别平均方差再求每seed比值；五seed均有效、mean R≤.8且至少4/5 R<1才触发。M1、IID2两数据集始终必做。
- 单一RAM生命周期完整覆盖A→基线冻结→离线probe→ANTI门冻结→B→有效run及selected/E10/E50冻结→统一test→数值表→状态释放。50epoch无早停、原batch40960、自然无放回shuffle；最终核对同seed全部臂初始化张量与50个epoch行序哈希相同。固定快照和selected均为CPU深拷贝，BytesIO/Pipe传递不写模型文件。
- seed配对表包含基线/L2、M1、IID2以及触发后的ANTI与同K对照；全部六格、域/任务、macro/pooled/traffic/worst指标分列，统计使用五seed样本SD和df4描述性t区间。性能门与保护规则分别保存validation/test结果；未触发臂为NOT_TRIGGERED。

## 独立验收

1. 项目Python独立复跑 `python -m pytest -q experiments/gradient_aug_20261031_v2/tests/test_semantics.py`：**4 passed**。覆盖固定源预测/loss/梯度/实际Adam、视图与权重预算、RNG、None、尺度状态映射、RAM恢复、在线轨迹不干扰及矩阵/门逻辑。
2. 独立CPU离线probe端到端：合成行、实际fixed1主干与已建立的Adam状态，E1/E10两个去重锚点共4个观察，**56次梯度、20次反事实Adam step、96次forward**；源快照和RNG保持一致。17个vendor字节身份同次检查通过。记录见 [probe验收](review_checks/probe_cpu/result.json)。
3. 独立45-run合成数值验收：每split135个selected/E10/E50语义行，核对selected的global_step来自selected快照，全部五seed配对、基线选择、ANTI同K比较及决策表均通过。记录见 [聚合验收](review_checks/analysis_synthetic/check_result.json)。这些数值用于程序验收，不进入正式实验表。
4. 复用coder已完成的真实CUDA证据 `artifacts/gradient_aug_20261031_v2/GA31-R3/implementation_check1/checks.json`：两个数据集全部冻结身份匹配；原batch40960源/新预测、梯度及Adam一步一致；M1/IID2/ANTI2可容纳；两数据各四步online on/off轨迹完全一致且未评价test。审查未新增真实训练更新，开发消耗仍为每数据13步（ML剩7，KR剩36）。
5. 本轮代码、报告及工件未发现`.pt/.pth/.pkl/.pickle/.ckpt`训练状态文件；模型状态只经RAM处理。

没有未修复的blocking项。审查没有修改算法实现、启动正式训练或解释科研效果；正式RUN按冻结的35次必要/45次上限矩阵执行。
