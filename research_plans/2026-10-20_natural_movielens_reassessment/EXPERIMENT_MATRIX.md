# 全量实验矩阵与覆盖状态

2026-10-20设计冻结。**下列状态只表示设计，不表示实现/训练/分析完成。** 实时运行以本次 `RUN_LEDGER.csv` / queue receipts为准。历史实际完成与本次补充状态分别记录于672行历史config索引。所有正式组2022/2023/2024/2025；N17 pilot2030/31另列。

定义：NAT=完整自然base-train800167；ART2=人工d2_large固定subset2022；N187=自然p最大余数等187500基础train。无meta组NAT每epoch20步；N19/CM19共同5%meta后NAT main760161、每epoch19步；历史ART2 main178125、5步/epoch。N187 meta后178126，保留实际取整差1，不伪称普通main N完全相等。

## 最新阶段状态（2026-10-20 15:52 UTC）

293/1060名义训练预算条目已完成：core120、V2 16、N17 88、AO04 12、N19 57。N17的62诊断与72正式test完整，core160诊断无失败且分析/归档/图表完成，AO04 root零更新test完整、504cell独立复算及阶段报告完成。N19当前10 RUNNING、249 QUEUED，CM19 508仍QUEUED；143transfer诊断及各scope冻结test依赖原队列。完整状态见同时间EXPERIMENT_EXECUTION_SNAPSHOT及实时ledger；下列设计列和较早快照保留为过程记录，不能取代实际receipt。

## 较早执行快照（2026-10-20 10:54 UTC）

下表原DESIGNED列保留为设计冻结记录。本次实际core120及D-V2 16全部完成完整预算训练/冻结测试，S5 32与S6 16均已完成四seed两预算性能分析；AC2D的160固定点机制尚待；V9的8任务全轨迹只读已完成、400预测/2400cell复算。N18的24自然离线任务完成，零新增训练。N17的16pilot与64正式H100完整闭环，共同及own双侧LR均.01；H200门控触发8延长。基础1044之外累计触发16配置，当前1060名义配置；H200全部8配置已达到200轮/4000更新/160,033,400累计曝光，前100轮轨迹及模型索引身份核对一致；当前224名义配置完整（含16pilot及8续训）。N2/N3运行中，precision与test仍待闭环。AO04、N19、CM19后续按冻结队列执行。实际attempt失败/重跑以RUN_LEDGER为准，不当作独立正式配置或科研阴性。

## A–C核心（120名义正式配置）

|ID|覆盖问题/实际原算法|人口/干预|预算|配置数|当前状态/依赖|
|---|---|---|---|---:|---|
|AC-2D-NAT|A/B，learned-fusion两Adam Stage2D|NAT，SN/SB/EN/EB|50完整epoch，B40960，main.01/arch.001|16|DESIGNED；数据冻结→实现→独立review|
|AC-2D-ART2|比例/支持对照同算法|ART2，四臂|50epoch，5main+1arch步/epoch|16|DESIGNED；固定subset不随seed|
|AC-2D-N187|等基础N自然比例支持|N187，四臂|50epoch，5main+1arch步/epoch|16|DESIGNED|
|AC-2D-BUDGET|完整支持、匹配曝光/更新|NAT，四臂；187500 draws/macro-epoch|50macro-epoch，250main/50arch步|16|DESIGNED；独立控制非完整自然epoch|
|AC-V9|B/C，无动量SGD.3 fixed model|NAT SN/SB|50epoch、单optimizer、WD1e-5|8|DESIGNED；v3实际源|
|AC-S5|C，Equal历史/当前因子|NAT F/T/HT/DH，物理target d2|50epoch，main.01/arch.001、原loss scale|16|DESIGNED；原overwrite算法|
|AC-S6|C，Sample历史阴性边界|NAT F/DH，target d2|50epoch，main.01/arch.001、原Sample|8|DESIGNED；原overwrite算法|
|AC-S5-MATCH|状态时钟控制|完整NAT支持，S5四臂|187500 draws×50macro-epoch|16|DESIGNED；main/arch都匹配|
|AC-S6-MATCH|同上Sample|完整NAT支持，S6两臂|187500 draws×50macro-epoch|8|DESIGNED|

固定点、loss/梯度oracle、moment/applied delta、full-coordinate covariance、reachability与same-state probes属于离线/训练内测量，不增加独立训练数。完整规范见CORE_DESIGN_SPECIFICATION。

## D：v2/N17/N18/AO04（100基础配置，其中16pilots）

|ID|研究对象|必要比较|预算/数量|状态|
|---|---|---|---|---|
|D-V2|fixed模型优化器桥接|NAT SN/SB×Adam.01/无动量SGD.1×4|16formal，50epoch|DESIGNED；与Stage2D不可按名称alias|
|D-N17-PILOT|对称调参|NAT q×Adam LR .0003/.001/.003/.01×2030/31|16pilot，100epoch|DESIGNED；完整响应面|
|D-N17-GRID|消除历史缺SN.0003格|NAT SN/SB×LR.0003/.001×4|16formal，100epoch|DESIGNED；不以pilot结果删格|
|D-N17-EXPOSURE|区分q/LR/优化时钟|同四格NAT完整支持，187500draws/macro|16formal，100macro-epoch/500updates|DESIGNED|
|D-N17-N187|等N支持|N187同四格×4|16formal，100epoch/500updates|DESIGNED|
|D-N17-QUOTA|域计数波动机制|共同LR，NAT SN-quota/SB-quota×4，IID父复用|8新增formal，100epoch|DESIGNED；pilot冻结共同LR后|
|D-N17-EXTRA-LR|完整赢家比较|若共同/独立winner在.003/.01，训练缺少对应q×4|条件追加，原100epoch|CONDITIONAL；必须读selection定义，不看test|
|D-N17-H200|原训练平台分支|共同LR双q×全4seed|条件延长200epoch，H100原件保留|CONDITIONAL；rho规则在规范冻结|
|D-N17-PROBE|N2/N3、Adam即时映射|四seed×双路径×E10/E40，512batch/固定点；32draw/局部|0独立训练|DESIGNED；不能把draw当训练seed|
|D-AO04|非对称融合自由度|NAT B0/B1/B2×4|12formal，50epoch，原main/arch协议|DESIGNED；不新增未触发B3/B4|
|D-N18|真实schema/嵌套逻辑/审核|自然ML标签与四mask seeds，同信息LP/随机vs分层审核|0端到端训练|DESIGNED；保留普通对照|
|D-N18-KR|P3/P4/P6 KR教师研究|原V4final KR/SN/E10|0新ML训练|HISTORICAL_INDEPENDENT；非遗漏|
|D-N17-N4|历史长期拓扑|原未通过局部门槛|0历史欠账|HISTORICAL_NOT_TRIGGERED；新问题需独立新设计|

## E：N19（236原ML迁移 +80容量控制 =316正式配置）

|模块|原实际配置映射|自然训练数|预算/必要机制|状态|
|---|---|---:|---|---|
|E-NATIVE|18 Adam方法+五容量/固定AdamW方法，各四seed|92|100epoch，native两个任务；original重叠arch保留|DESIGNED|
|E-P1|七M3OE +三MMoE方法，各四seed|40|100epoch；自然支持上的合成Z/Y1，已知shift；task_mask/global_cf/V-REx强/负对照|DESIGNED_CONTROLLED|
|E-P2|14主方法+gamma1四方法，各四seed|72|100epoch；2%审核、axis/ind/nested/joint/profile/clean oracle完整|DESIGNED_CONTROLLED|
|E-P6|八mask/CV方法，各四seed|32|100epoch；pi.55，r端点，EMA与静态控制；all-label父复用|DESIGNED_CONTROLLED|
|E-P3-EXPOSURE|fixed/rank_uniform/rank_protected/domain_codebook/cell_codebook×Adam/AdamW×4|40|完整NAT支持；178125draws×100macro；原rank epoch10剪枝/重置|DESIGNED|
|E-P3-N187|同五方法×两optimizer×4|40|N187派生meta，100epoch；普通main实际178126|DESIGNED|

P4全部domain/axis/full/interaction/reliability，P5utility/regret/coalition/uniform KL，P3两optimizer五方法已含E-NATIVE，不能再重复加总。历史P3选点≥10、P4confidence双投影、P5同tower finite信用和P6残差可见label边界必须保留。全14addenda及逐配置source flag见N19_CM19_DESIGN_ADDENDUM与config索引。

## F：CM19（436原ML迁移 +72匹配控制 =508名义正式配置）

|组|配置|自然训练数|预算/说明|状态|
|---|---|---:|---|---|
|F-G0|10原生/协议模型×两optimizer×4|80|100epoch，STAR/HAMUR各任务单独|DESIGNED|
|F-G0M|STAR/HAMUR两task移植×两optimizer×4|16|100epoch，联合task loss均值|DESIGNED|
|F-G0MC|两独立主干对应控制×两optimizer×4|16|100epoch，成本/容量差明示|DESIGNED|
|F-G1|fixed/STAR0/STAR1×五decay模式×4|60|100epoch；wd0/shared/slot/both/localAdamW|DESIGNED|
|F-G2|三模型×四variant×两optimizer×4|96|100epoch；no_bypass/additive/bypass_scale/no_modulation|DESIGNED|
|F-G3|PEP3、HAMUR6、专家4变体×两optimizer×4|104|100epoch；各模型自己的neutral/重训语义|DESIGNED|
|F-G4|三模型×fraction.25/.5×normal/matched×4|48|100epoch；物理target d2，scale变化后角色可能跨越|DESIGNED_CONTROLLED|
|F-G4LONG|fixed/STAR0×两optimizer×4|16|原max_steps2350完整，单列100epoch以上状态|DESIGNED|
|F-M-ART2|fixed/STAR0/STAR1×两optimizer×4|24|ART2扣meta，100epoch；可经身份核验复用历史|DESIGNED_OR_HISTORICAL_REUSE_PENDING|
|F-M-N187|同三模型×两optimizer×4|24|自然等N扣meta，100epoch|DESIGNED|
|F-M-EXPOSURE|同三模型×两optimizer×4|24|完整NAT支持，178125 draws×100macro|DESIGNED|

原CM19名义AdamW-WD0及capacity replacement别名按原数学身份保留为alias，不能当独立训练。额外跨E/F别名须实际model/init/loss/采样/更新/BN/选择完整相同才成立。N19 original first-batch与CM19 original cycle-iterator不可alias。ML AdamW名义WD1e-5实际系数舍入1须保留，不能暗改为可表示正收缩。

## 计数、执行依赖与完成规则

基础1044名义配置=120core+100D+316E+508F；其中16pilot，1028名义formal，包含可核验的人工历史24参考及可能严格别名。条件新增winner/H200不在基础数。**基础数不是强迫重复等价训练，也不是已运行数**；RUN_LEDGER逐条记 `independent_training / equivalent_to / historical_reuse / planned / pending`，最终同时报告名义与非冗余实际训练。

依赖：数据冻结→各独立算法设计→coder实现报告→独立review→完整训练→配置/选点/development冻结→零更新test→主代理逐问题分析→最终中文报告/GitHub分支PR。E/F内部方法可平行，依赖父baseline和rank暖启动按原逻辑；不得为了早测有利test提前改方法。原始训练、冻结评价、离线诊断分别计数。失败保留attempt，repair后同seed继续，不能把missing当阴性或成功。

最终覆盖H01–H31且A–F全部有证据对应。没有自然结果的行仍pending，不得在FINAL_REPORT中写“全部结论已复核”。理论/自然KR不受ML比例影响的项可以NOT_APPLICABLE_TO_ML_TRAINING；有真实信息阻塞的项可REDESIGN/DATA_BLOCKED并明确原因，但不能用其取消其它可运行组。

### 13:25 UTC阶段补充

N17全部88训练预算条目（16pilot、64正式H100、8条件H200）、48基础诊断和14精度追加均完整闭合，主代理开发及机制分析完成。72正式配置模型身份已冻结并发布零更新test任务；16pilot排除。H200包含与H100共享前缀，故名义预算条目不等于独立从头训练或独立统计重复。AO04仍在完整训练；core160机制、N19/CM19及对应评价继续，整体尚未完成。
