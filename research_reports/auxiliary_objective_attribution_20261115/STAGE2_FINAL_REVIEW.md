# 最终证据审查：Stage 2与完整实验

**Verdict: PASS。Blocking issues: 0。** 已审查最终REPORT、Stage2证据/逐seed/汇总/诊断表、STAGE2_DECISION、FINAL_DECISION、COMPLETION及两张Stage2图。完整实验按预设停止条件结束，后续HPO、冻结Test和额外机制消融均NOT_TRIGGERED。

本次仅填三个完成/决策JSON的independent_review块，以及COMPLETION的stage2_evidence_review=PASS；科学字段和Stage1冻结文件未改。下方hash在这些授权metadata写入后计算，审查JSON没有被反向写入受审文件，避免自引用。

| 最终文件 | SHA256 |
|---|---|
| STAGE2_DECISION.json | `6aed83948d70c40b0f048b38712eee082d83dc22eb9eb50e55759ea7c344426b` |
| FINAL_DECISION.json | `c6ca7eb3b1f758570b74cb07cf302bdee3a47ad786252f0900dceec7f1a364c1` |
| COMPLETION.json | `2abd6f0faec7fded4ec9a65883e0a2447112e29f6391686d47062fdfd039b187` |
| REPORT.md | `4100b8ee0f97cf1f66cca888be9f8f1297b5d7b52c676995a418f9ede02a463a` |
| STAGE2_EVIDENCE.json | `14e2fbdcb05d695d87d8bd90ec25a467a95783d5f5418075d133c462c2d63600` |

## 完整性与身份

Stage2有10条新轨迹、500新epochs、各17250次Adam/EMA更新；10条CLEAN复用后共有20条轨迹、1000份epoch记录。40个online/EMA选点及对应state hash、500组dataset/seed/epoch跨臂自然行序、数据manifest和冻结provenance一致。500份Stage2新日志共30292308 bytes的SHA256匹配，失败/重试为0，Test=false。Stage2源码保持`831cba9947738652cc6d4e5abf864e1ab4e32545b304936cb39aba28b227893b`；Stage1保持`a78bce2f713cf62f9af32548b50bc360b134420ff9ed6ac6990dca22d76a2ad0`。

完整任务为40条Stage1新训练+10条Stage2新训练+10条共用CLEAN，共60条唯一轨迹；新增2500epochs、各86250次Adam/EMA更新。CLEAN跨阶段复用没有重复计为独立轨迹或seed，五seed仍是每个可比配置的统计单位。

复用已完成的源码、预算、选点、日志hash和工程审查证据，没有重跑CPU/full-batch检查或重扫Stage1的5.77GB支持日志。Stage1最终decision仍为`5845910e92cd37a97725dca86878c077546e8dca5a465b818f06549a68f652c1`，与Stage2冻结provenance相符。

## 数表与自然目标

792条Stage2配对汇总、3960条seed差和160条endpoint已独立核对；sample SD、五seed配对t区间及方向数正确，最大CSV往返误差1.73e-16。36条phase对比先在seed内聚合epoch，再作五seed配对。证据JSON、原始CSV、运行收据及report hash一致；selection_hash直接匹配运行清单，未再出现R1编码差异。

120条cell-label-seed权重汇总采用实际1000/2450个batch作分母。全部17250个batch保留完整2B支持、总质量1，分层质量最大误差0，view/clean计数和自然标签先验一致；实际zero direction count为0。全量task delta/drift均值按entry加权、maximum为观测最大值；batch分位数均值和每seed50条首batch摘要的口径有明确区分。Stage2没有把未采集的参数gradient norm/cos写成已有观测。

## 科学裁定

主比较NDP-A fixed EMA−CLEAN fixed EMA的selected验证Macro AUC：ML为−0.000101256（1/5正，CI跨0），KR为−0.005775319（0/5正，CI跨0）。两个完整五seed均值都不为正，精确满足冻结Stage2设计第10.1节停止条件。停止不依赖CI排零、全seed正向、单格保护线或Test结果，没有增加计划外否决门。

REPORT完整保留ML online的+0.000360小增量、候选+EMA对clean online的+0.002895总收益与clean EMA既有收益的区别；KR微小LL均值下降、d2局部AUC改善及d0/d1代价也均保留。AUC/LL分开裁定，selected/common阶段和两task反例有对应数表。ML新增排序信号标WEAK、KR退化与局部取舍标MEDIUM、自然质量/实际机制执行事实标STRONG，符合当前证据。

Stage1零扰动整体构造退化、权重并非全部解释、当前BCE任务视图尚未带来主AUC收益三个结论相互区分。标签/尺度、对抗方向、跨task变化和EMA重叠仍为竞争解释，报告没有把其一写成唯一机制。域人口不作困难度代理；KR的局部域取舍与另一小人口域的大幅下降同时列出，没有升级为普遍规模规律。没有发表模型Test验收或通用方法失败的结论。

## 成品与交接

48个本地Markdown目标存在；4个外部GitHub报告/目录链接已核验。Stage1图沿用既有审查；两张Stage2图的dataset/metric/cell标记、图例、零线和95%区间可读，所示方向与数表一致。大体积Stage1原始trace保留服务器、仓库保留数表/epoch证据及hash的交付范围在REPORT中明确。

科学实验与本地报告通过最终审查。主agent可按既定收尾释放本任务两个RAM guardian并发布已审查代码与报告；原pilot guardian不在此收尾范围。没有新增实验或扩大验证矩阵。
