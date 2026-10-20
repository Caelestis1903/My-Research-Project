# 历史实验与训练人口覆盖审计

审计日期2026-10-20；独立designer只读取证。范围为2026-10-15以来计划、执行分支、其引用的早期数据/model/optimizer。**本文件是历史条件与补充范围审计，不是新自然实验结果或最终研究结论。** 历史报告的“COMPLETE”仅描述其原阶段，不能转写为本次自然补充完成。

## 来源恢复与证据层级

当前目录main材料不含全部执行链，已恢复sibling执行worktree。下面HEAD经只读git核对；实验执行时间与提交时间分别记录，目录日期不能当执行时间。

|来源|只读工作目录|HEAD / pin|主要实证时间与材料|
|---|---|---|---|
|10/15链|`/home/xzf/Code/phase0-3-git`|de80055be56682c47fc3cd4aedd1c722c2bdb5d6|10/15–16审计、B1 reconstruction、C2 oracle|
|10/16 Stage2D / v2|`/home/xzf/Code/phase0-3-20261016-git`|3eeb5961f226cab0cac1c2ee2e554f5b47db53e7|Stage2D及v2执行；v2部分终报文本仍RUNNING，不能只据文字判全矩阵未/已完成|
|10/16 v3|`/home/xzf/Code/phase0-3-20261016-v3-git`|2d04c6de6c925eca8563f81b66af3f0e95f99183|v3最终全读数与问题闭环；10/17分析引用此提交|
|N17|`/home/xzf/Code/phase0-3-20261017-git`|5550afbc57f76e0b94c5783db16f4c1796933541|120条H100总轨迹、72唯一formal读取；含ML N5-C|
|N18|`/home/xzf/Code/phase0-3-20261018-git`|14cf09a3865303f5e451636dc6407fd0e2cc1b8c|六方向首轮，数学/固定KR诊断、冻结头拟合|
|N19|`/home/xzf/Code/phase0-3-n19-git`|c440350681a7a3e41ab3f05eb1c4d31f419b989f|10/18–19实际328正式网络，其中ML236|
|CM19|`/home/xzf/Code/phase0-3-20261019-git`|eab817b3de626caee9ee9aeb1db4dd9aa3ef9e4c|10/19 05:14:59–14:40:23 UTC正式训练窗口，10/20收尾；856正式网络，其中ML436|

已全文读取：restart.md；10/15总方案与执行/机制/范围判定；Stage2+结果；Stage2D设计/结果；v2执行报告；v3最终结果、V1、V9、10/17再分析/下一计划；N17最终分析/设计/N5-C；N18设计/最终报告；N19设计、全部14addenda、最终报告、实际ML ledger/config及data/model/objective/training入口；CM19设计/最终报告/manifest/ML ledger/config及data/train/model入口；F0自然数据完整实现与S1角色抽样/feature定义。

本次取证区分：设计L0、报告转述L1、原表/config/ledger来源L2、原prediction重算L3、独立训练复现L4。**以下研究数值均注明为原报告值，本设计者没有重新运行研究或从全部原prediction重新计算AUC**。新报告需由主代理分析实际新数据后替换暂定评估。672条ML原配置已以完整config JSON和逐文件SHA保存 `HISTORICAL_ML_RUN_CONFIG_INDEX.csv`；其中historical_execution_status与new_reassessment_status分列。

## 数据血缘与Natural语义

MovieLens raw `ml-1m.csv`，三域年龄{1,18}/{25}/{35,45,50,56}；native任务rating>3/rating>4。S1/F0从全局行顺序80/10/10切分，encoder/age scaler在全局base-train拟合，unknown=0。基础自然train预期800167行、155949/318599/325619，正式仍核对raw与编码hash。

S1 `role_reversal_sample` 在基础train按 `RandomState(seed*10+d)`逐域无放回取L/M/S=150000/34500/3000。d2_large为34500/3000/150000；d0_large为150000/34500/3000；validation/test通常未改人口。早期v3 ML优化seed同时改变subset；N17/N19/CM19改固定subset_seed2022、优化seed独立。两种随机性范围不能混并解释。

后期N19/CM19普通ML main排除同人口5% train-meta，成为32775/2850/142500，总178125；自然main应约148152/302670/309339，总760161。两者不只是比例不同，还差约4.27倍独立支持与每epochupdates。

历史Natural/SN表示q=p_current；`suite=natural`常只表示没有额外合成标签/噪声。**它们不证明train未经role sampling。** 全自然F0 loader确实存在，但F0训练规划/控制状态或自然加载代码存在不等于已有可复用正式自然结果。此审计未找到可据以免除本次N19/CM19自然补充的完成证据。

KR natural基础train2006132，域172782/1743898/89452；N19/CM19扣meta后164143/1656704/84980；KR原结果与ML人工/自然分别记录。KR全局时间顺序切分导致train d1约86.9%、val/test约98.2%，不能当成ML比例问题，也不能以val/test总域比相近排除域内分布/有限正例误差。

## 全研究问题覆盖表

分类：`NAT_VERIFIED`=有完整自然ML已验证；`ART_ONLY`=ML经验结论仅人工；`CONTROLLED`=本来人为干预有效、保留其受控结论；`INDEPENDENT`=不依赖ML自然比例无需ML重训；`REDESIGN`=信息/模型语义不允许直接当native自然任务迁移。可多标签；未发现完整NAT_VERIFIED实证不等于断言全仓不存在任何自然ML研究。

|ID /研究组|实际人口、方法与预算|历史发现/状态（原报告值）|分类与本次处置|
|---|---|---|---|
|H01 10/15 A0/A1/A2历史恢复|S4/S5/S6/S8、S6-R2与research_R2工件审计；0新训练|部分历史source/init/state缺失；不能exact重放全部旧运行|INDEPENDENT审计；记录缺口，不把新自然训练冒充填回历史状态|
|H02 10/15 B1 reconstruction|KR natural、M3OE Sample F/DH、4seed、50epoch|R0 test宏/非target差+.000363/+.000579，各2/4正；不是旧初始化exact复现|INDEPENDENT；KR结果保留，Sample ML F/DH自然参照见AC-S6|
|H03 10/15 Equal S5引用|KR及ML/d2_large，M3OE，F/T/HT/DH，50epoch，main.01/arch.001|KR HT恢复target救援约99.2%/95.9%，DH约2%；ML只部分复制，不能归为m单独致因|ART_ONLY(ML) +INDEPENDENT(KR)；自然AC-S5四臂及匹配预算|
|H04 C2.1固定theta oracle|KR natural +ML/d2_large，初始化有限pool，两alpha/q、4seed|同目标identity；Balanced对Equal减trace、对Sample增trace；不是selected性能|ART_ONLY(ML)；自然同theta全block身份、MC与实际Adam诊断|
|H05 10/16 Stage2B/2C|两人口，各域16rows、C/P各至多32coords、4seed、4096replays×64步，非训练|相同alpha期望残差≤2.4e-16；Adam累计均值delta差，SGD均值符合MC；选坐标trace可与完整C相反|ART_ONLY(ML数值)；保留数学、自然全坐标/明确有限pool重算，不把oracle当AUC|
|H06 10/16 Stage2D|仅KR natural，learned fusion+两Adam，SN/SB/EN/EB×4，50epoch|selected val SB−SN−.010406、EB−EN−.006547；test−.012674/−.002714|INDEPENDENT历史KR；自然ML四臂作为A/B/结构协议桥接，不称原来已有ML训练|
|H07 v2固定协议主矩阵|KR多臂/ML d2_large；fixed fusion单Adam.01或SGD.1；50epoch|ML两optimizer最小第二环境共16；部分终报仍RUNNING。v3读数提供后续引用，但须核实逐run最终状态|ART_ONLY(ML)；D-V2完整自然16，不用Stage2D learned模型替代|
|H08 v3 V1–V8及V10/V11主环境|KR natural；同Sample q、LR、PERM/ORDER/mask、C局部、AdaTask与状态；完整各原预算|SGD无动量也有采样效应，Adam非必要；beta2=0 floor不证明修复；C高attenuation可伴风险变差|INDEPENDENT(KR)；原正/阴性全保留，不机械铺所有ML消融；新自然核心/N17对应机制诊断|
|H09 v3 V9 ML|人工d2_large，subset随2022–25改变；fixed M3OE SGD.3，SN/SB，50epoch，5steps/epoch|test early AULC1..16+.003015四正；E50−.000986四负；selected+.000539仅1/4正|ART_ONLY+CONTROLLED；AC-V9自然8及N/step解耦；不以跨环境相反均值唯一归因N|
|H10 S6-R2/ML与Sample负边界|ML/d2_large，F/DH，50epoch原预算；与AO04不同研究|selected test近0；KR旧test ΔN+.006757四正、val−.003851仅1正|ART_ONLY(ML)；AC-S6自然8、匹配8；历史state缺失不得伪复算|
|H11 AO04 B0/B1/B2|ML/d0_large，learned/asymmetric融合，main.01/arch.001、50epoch，旧2seed；KR B1部分未完成|ML B1−B0 val约+.000005765、B1−B2−.000006842；KR2022 val+.001494/test−.008385，2023INCOMPLETE|ART_ONLY+CONTROLLED；D-AO04自然12；B3/B4未执行不当否证|
|H12 N17 N0评价分解|旧v3预测、固定权重插值、校准；0新训练|KR calAdam E50宏−.009074而test-p加权+.003108，过零alpha约.7449|INDEPENDENT固定预测算术；自然新模型同权重/校准读数，不改主指标|
|H13 N17 N1 KR|KR固定自然，100epoch、对称LR、8formal seed|selected Adam−.005203、7/8负但主t/符号Holm未确认；SGD+.000837、5/8正|INDEPENDENT；保持KR原范围，不能说人工ML使KR数值无效|
|H14 N17 N2/N3/N4|KR，16双路径锚，quota完整训练；C局部/独立确认|计数项仅.001297%–.105936% trace；embedding局部约3e−5；N4未触发|INDEPENDENT；自然N17 quota/固定点用于推广；N4 NOT_TRIGGERED不是历史欠账|
|H15 N17 N5-C ML|人工固定2022，Adam四LR双侧pilot16；common SN/SB.001、own SB.0003，8seed，100epoch|test selected common−.010392/own−.010875、全8负；own E100+.021145八正但含LR变化|ART_ONLY；D-N17自然q×LR完整四格+对称pilot/匹配N/曝光/quota；新四seed与旧八seed分开|
|H16 N18 P1/P2理论及ML标签|支持LP、同观测世界、nested逻辑、MAR/MNAR及审核；ML固定2022人工支持|普通LP解释缩界，真实桥接/合法环境未识别；DATA-BLOCKED非算法失败|CONTROLLED数学 +ART_ONLY有限ML数值+REDESIGN自然环境；自然标签/审核离线重算，不伪造环境或直接宣称真实MNAR|
|H17 N18 P3/P4/P6冻结teacher|**KR** V4final SN Adam E10四seed，非ML；P3每seed8解析头求解，P4/P6Jacobian/IS|P3Δmacro−.000559，2正2负，全cell保护0/4；方向方差额外降不足1%；修复等价普通头|INDEPENDENT；无需ML端到端重训；teacher固定KR与ML人工无关|
|H18 N18 P5 /结构等价|D3产品gate对称、同动作风险构造、理论|预测不变而动作风险可反号；普通utility覆盖；没测忠实D3训练收益|INDEPENDENT/CONTROLLED；保留构造，不新增伪D3自然方法胜负表|
|H19 N19 native强基线|ML人工扣meta，fixed/original/original_sample/MMoE/PLE，4seed，100epoch；KR自然50|ML selected test fixed .733599/.533980(AUC/LL)，original .706324/.565083；协议多变量不同|ART_ONLY(ML)；全部相同source自然重训，不能由baseline差推出纯架构因果|
|H20 N19 P1|ML派生合成Z/Y1与已知shift，M3OE七臂、MMoE三臂，40ML网络|DRO−ERM test+.188437；−mask+.004101四正CI跨0；MMoE任务/LL冲突|CONTROLLED+ART_ONLY支持+REDESIGN任务；自然支持上同模拟器40，native结论另列|
|H21 N19 P2|ML派生噪声，2%审核，14主方法及gamma1四方法，72网络，100epoch|joint−nested_axis AUC−.002586，LL−.003804；非加性joint axis−ind joint AUC−.003919四负/LL+.014394四差|CONTROLLED+ART_ONLY；自然支持完整72及审计实际类数；不声称自然噪声识别|
|H22 N19 P3|native人口，低秩48预算/SVD10epoch、域/任务字典、两optimizer；两数据总72专属|KR Adam路径失活使uniform/protected逐seed相同；ML AdamW选中差+.000055近零；字典selected/final矛盾|ART_ONLY(ML)+INDEPENDENT(KR)；自然全部匹配capacity family+80 N/预算控制，不以路径活跃代替有效容量性能|
|H23 N19 P4|native路由domain/axis/full/interaction/reliability，40两数据专属|ML门控−fixed selected+.000458，3/4正；KR selected加权负而final交互AUC+.013137四正|ART_ONLY(ML)；自然所有原控制，无故不删后期阳性/选中阴性|
|H24 N19 P5|nativeutility/finite regret/coalition/uniformKL，32两数据专属，full_mix复用|ML selected regret−uniformKL−.000197，2/4；final+.007960四正但LL冲突|ART_ONLY(ML)；自然原四机制/强控制与same-tower信用诊断|
|H25 N19 P6|ML派生每访问mask，pi.55/r端点、EMA-CV/残差，32网络|residual−antitheticCV selected+.000896，−indCV−.000412；97.7%–99.8%回静态端点；后期正则|CONTROLLED+ART_ONLY+REDESIGN成本；自然支持完整32，nativefull参考；不能承诺真实标注节省|
|H26 CM19 G0/G0M/G0MC|ML人工+KR自然，忠实/移植/独立控制；ML112网络，100epoch|KR fixed selected24/24slot neutral Δp=0；独立STAR/PEP/HAMUR/MMoE/PLE活跃反例|ART_ONLY(ML)；自然ML112全部，不把single-task当MDMT|
|H27 CM19 G1|固定/STAR两任务，五WD局部/全局控制；ML60网络|KR两factor去L2/局部AdamW恢复；shared-only/slot-only弱；恢复selected test增益弱|ART_ONLY(ML)；自然完整G1与同state单步，不宣称动量m单独根因|
|H28 CM19 G2/G3|旁路/加法/移除、PEP/HAMUR/expert各控制；ML200网络|ML STAR即时module-off降AUC但no-modulation重训提升；弱HAMUR移除收益不超强fixed|ART_ONLY(ML)；自然全部200、删除vs冻结干预vs重训分开|
|H29 CM19 G4/G4LONG|固定/STAR物理d2 fraction25/50，normal/matched updates；ML48+16网络|old ML2350updates仍不KR式精确失活；long selected不变、final更差；支持减少损害风险|CONTROLLED+ART_ONLY；自然原完整64，加必要人口/曝光控制；不说规模无关或单独充分|
|H30 CM19浮点AdamW边界|ML lr.001、WD1e-5、float32/foreach=False|1−lrWD实际乘法系数舍入1，12对global AdamW/Adam-WD0 final模型SHA相同；KR实际系数.9999999403953552|ART_ONLY数值+INDEPENDENT浮点原理；自然同实现实测，不把名义正WD说成有效收缩，不暗改精度|
|H31 F0 natural入口|完整自然loader与Q容量/50epoch多模型/SDMT计划，控制/实现材料|代码确实全自然；现未取到足以免除N19/CM19补充的完成自然run证据|实现可复用，不是NAT_VERIFIED性能；只用数据loader，不把F0模型替代旧算法|

## 对人工协议影响的暂定判断

已可确定的**元数据事实**：v2/v3 ML边界、N17 N5-C、N19全部ML经验组和CM19全部ML经验组，训练人口不是完整自然ML。故其“跨自然真实数据”外推未经这些ML结果支持；原人工条件下的数值/受控结论并不因此失效。AO04还用了不同角色d0_large，不能直接和d2_large混合。

**尚不能由审计判断**哪些机制/性能会在自然ML保持、反转或失效：需要新实际训练与分析。人口增大同时改变p、特征/标签支持、每epochsteps和曝光；仅把新旧AUC相减不能归因“人工比例导致”。

**不受影响的内容**：自然KR具体实证、数学identity/标准理论、N18固定KR教师结果、原计划未触发项的状态。它们仍各有原证据范围；没有因为ML问题就一起否定，也不因此获得自然ML外部验证。

## 工件缺失与后续验收

本地历史source/config/ledger/聚合表可读，672个N19/CM19 ML原config均成功索引并SHA绑定。主代理发现旧 `/home_new/xzf_cm19_artifacts`当前不存在，历史重型checkpoint/预测的可加载性不能据旧索引存在而推定。历史S5/S6部分source/init/optimizer lineage早在10/15已列缺失。依赖这些bytes的exact diagnostics标INCOMPLETE；新自然独立初始化不冒充历史精确复现。

最终主代理需对需要引用的旧数值从实际CSV逐seed/cell重算，记录有/无prediction层复核；对新自然结果完成test/哈希/预算验收后，逐H01–H31给出保留、缩限、修改、反证或未识别。不能把本设计审计表、原历史完成状态或数据loader检查替代最终科研报告。
