# AIG-M：训练前证据审查与协议身份

读取日期：2026-11-01。入口为工作区 `start.md` 与远端 main 的 `research_plans/26-11-1_v2/06_Codex完整执行提示词.md`，00–06均已完整读取。此文件记录训练前依据，研究结果另在最终报告中分析。

## 实际来源

| 对象 | 实际读取版本 | 提交时间UTC |
|---|---|---|
| 当前main及新计划 | `0973fa5f1a804aec34f88ccb2ffdd8c0231b602b` | 2026-11-01 08:31:39 |
| GAM01发布分支26-11-01_v2 | `ee4cf0512a7723624c9ee6a1ef192f0f7f17686b` | 2026-11-01 07:21:26 |
| GAM01历史执行代码 | `89edb99a389c56fde86460af677a48e14fbcf116` | 历史报告记录 |

`SOURCE_FETCH_RECEIPT.json`保存远端提交、子树和blob；`LOCAL_SOURCE_VERIFICATION.json`逐文件比较Git blob及SHA-256。7份计划、GAM01完整包、GA31-R3完整包与GAM01报告共90份文件全部一致。当前工作区没有Git元数据，本轮代码冻结将通过已连接GitHub的Git对象接口生成新的执行commit。

已读取GAM01 REPORT、DESIGN_SPECIFICATION、IMPLEMENTATION_REPORT、REVIEW_REPORT、全部validation/test指标CSV、diagnostics_summary、decision、phase1_validation_decision、完整momentum源码及resolved_config；模型/数据/视图/指标接口来自已核验GA31-R3依赖。AGENTS.md与start.md是当前本地执行约束，未将远端缺失AGENTS.md解释为取消这些约束。

## 历史逐seed复算

`recompute_history.py`从历史原始CSV重新配对，两split×两数据集×selected/E10/E50，并导出所有AUC/LogLoss指标的原值、差值、sample SD与df4配对95%区间；不使用旧汇总表代替复算。完整输出为 `historical_paired_effects.csv`、`historical_effect_summary.csv`、`historical_recalculation.json`。

| selected test对照 | MovieLens Δmacro AUC | KuaiRand Δmacro AUC | 现有判断 |
|---|---:|---:|---|
| A0 EMA−online | +0.004399046，5/5正 | +0.004616399，4/5正 | EMA排序效应重复；ML有LL代价 |
| A1普通增强−A0 | −0.000060654，2/5正 | +0.002584048，4/5正 | ML未建立增量，KR条件性收益 |
| A3 βδ=.99−A1 | −0.000335598，3/5正 | −0.001553680，2/5正 | 当前长差分记忆未建立独立增量 |
| A3 EMA−A1 EMA | −0.000166620，2/5正 | −0.002354969，3/5正 | 参数平均没有改变当前独立增量判断 |

ML A0 EMA−online ΔLL=+0.007799597，4/5恶化；KR为−0.001225342，3/5改善。ML A1−A0 ΔLL=−0.001247547，**2/5改善**；A2−A1为−0.001263381，**2/5改善**。旧报告表格的两项3/5改善在此勘误，历史文件保留原样。P2为NOT_TRIGGERED，没有RDI运行结果。

历史Polyak依据为已读取的 `PROJECT_ARCHITECTURE.md` §2.3记录（旧ML mean +.004022、六seed，ρ=.99/.995/.998有限比较）。工作区没有对应 `v2_momentum/RESULTS.md` 与 `xs_*.log` 原始记录，故不能复算旧Polyak逐seed或将其数值合并到当前自然协议。旧记录中存在组件配置比较；没有找到GAM01统一lr/wd trial ledger，不能把固定lr=.01宣称为已完成联合调参的最优点。

## 数据与指标

`RAW_DATA_VERIFICATION.json`记录本轮实际读取的CSV绝对路径、大小与SHA-256，均与GAM01相同。MovieLens train域人口155949/318599/325619；KuaiRand为172782/1743898/89452。自然原行序80/10/10，KR先全局划分再保留tab0/1/4；train拟合encoder/scaler、OOV0，ML标签rating>3和rating>4，KR标签is_click和long_view，保持不变。准备后的六数组、行号、encoder/scaler身份在Stage0再次由现有加载接口核验。

模型fixed1、3域×2任务、embedding16、4 shared experts、fcn_dims=[512,256,256,64]；batch40960、50epoch、FP32、loss_scale1、自然shuffle、seed2022–2026。基础Adam coupled wd同时进入m/v，差分q不含wd。selected按各模型完整validation六格macro AUC严格提升、平局最早、排除E0；主端点是冻结后test六格macro AUC。pooled与traffic单列，LogLoss保留未缩放真实输入定义。

训练缓存不能直接复用：历史模型/optimizer/shadow已按RAM-only规则释放，且缺少本轮多时间尺度shadow与完整验证轨迹；旧test已公开参与开发，不重新命名为独立留出。可复用经hash核验的数据预处理身份与现有代码，无权重缓存可供继续训练。新自然划分仅作为最终方法选定后的后续研究建议，本轮不增训练预算。

## 合同歧义与冻结选择

1. 03和06优先于02的探索公式：首版固定βδ=.95，仅动态λ，v使用当前 `(g+wd·θ+λ_t·d_t)^2`，分子使用bias-corrected真实m和历史q；与现有SAD公式一致。
2. 04的ρ=.995半衰期文字大了10倍；按公式为138.2826 step，即ML约6.9141 epoch、KR约2.8221 epoch。H={2,8,20}的ρ按各自20/49 step换算；保留.995锚点。
3. 开发晋级按数据集独立：相对最强A1+EMA的两个开发seed selected validation Δmacro AUC均值≥.001、2/2正且预设保护为空。A0+EMA关系另列；AIG两数据均失败/单数据通过/均通过分别为64/69/74条预算，不为凑run数训练失败候选。
4. 本轮只执行低预算合同，不自动启用AIG-SWA、温度校准、额外局部搜索、更多seed或新数据划分。SAD β=.90的实数等价通过已有公式核验，不另分配正式训练轨迹。

## 当前证据强度与可证伪问题

ML EMA排序重复为STRONG，KR EMA及当前A3未建立增量为MEDIUM，域规模统一解释与创新量能提供有效增量为WEAK。规模不等同难度：ML最大域d2的baseline较低，KR最低AUC域d0并非最小域d2。KR普通增强有小域收益/大域下降的描述性交换，A3相对A1则方向相反；task、目标增强路径、流量权重及标签语义都是竞争解释。

本轮先寻找更强Adam/EMA和增强参考，再检验AIG-M能否超越该参考；若有限搜索仍不过门，则接受停止。固定SAD与AIG的同配置对照用于区分动态注入和普通增强；M1没有λ=.25固定SAD专属对照，平均注入剂量仍是其机制竞争解释，不增加预算补齐。可靠经验效应与未识别机制分别报告。
