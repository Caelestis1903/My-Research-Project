# N17完整pilot与学习率冻结（selection only）

16/16 pilot均完整100epoch、2000主更新、80,016,700实际训练曝光；seed2030/2031独立于正式2022–2025。主代理核对完整轨迹100点有限、600个selection cell/运行的聚合、earliest max选中轮次、receipt SHA与冻结结果。未评价pilot test；此文件不代表N17正式性能或机制闭环。

|Adam LR|SN selected AUC / LL|SB selected AUC / LL|共同选择分数|SN / SB final AUC|
|---|---:|---:|---:|---:|
|.0003|.749455 / .530257|.749390 / .532614|.749423|.721647 / .719207|
|.001|.749490 / .542407|.749012 / .533807|.749251|.702111 / .703697|
|.003|.750134 / .548399|.750296 / .535202|.750215|.692727 / .695650|
|.01|.752876 / .545592|.752992 / .540677|.752934|.719007 / .714967|

共同选择分数是每个q/seed自身最大selection宏AUC的四项均值，不是同一个epoch均值，也不是将全部q权重最优值按test重选。own按每q两seed最大值均值。共同与SN/SB各自赢家均.01，平局偏小LR规则已核对。源响应CSV SHA `eb1114bf586b34b0ca76a2e8af05f2e5daf565273ff725b3916a13d0dcd2bcd9`；冻结文件SHA `b2003cbe92ae22024adb16817464cdd31e987b7b51fa64a5faeb0e66a6226b9f`。

**STRONG有限执行/选择事实**：预登记四档内选中.01，原缺SN低LR格的历史问题在新设计中没有删格。原基础formal .0003/.001、matchedN与matched-exposure各四格全部保留；按条件增8个.01 formal，并生成8个共同LR quota。N17本轮现有16pilot+64formal=80配置，H200是否触发仍待共同LR正式H100闭环。全任务基础1044之外目前触发8新formal，不能把8个quota再当额外条件增加。

**WEAK正式效果证据**：pilot只有两个优化种子，评价总体固定；.01位于预设网格上边界，只能称当前网格的赢家，不能称全局最优。各LR final均低于selected，且相应LL明显上升；selected与final排名不同，.01胜出不意味着末轮最优。旧人工N17共同.001、own SN.001/SB.0003与此不同，但人口/支持/时钟共同变化，不能唯一归因比例。正式common与own将复用同一.01模型，不能当独立重复确认；matched controls中冻结LR为.0003/.001，与.01赢家比较时需明确LR不匹配。

首个SN/.0003/2030 attempt在epoch0因共享split临时文件竞争失败，原stderr和FAILED台账保留；同seed attempt02完整完成。失败不当科研阴性，也不增加独立正式样本数；未改研究producer或训练公式。

机器可读表及冻结原件：[analysis/n17_pilots](analysis/n17_pilots)。没有因为本pilot赢家位于边界而自动新增搜索；继续执行已规定训练、quota、平台门控及N2/N3精度门控。
