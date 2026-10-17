# 最终统计实现细则（新test解封前）

原设计主配对seed双侧95%t区间、精确符号随机化和seed bootstrap不变；两优化器共同eta H100 selected主对比为预设Holm两项检验族。H200次级组须单列命名，不与H100同名覆盖或混合。AULC为原历史协议的逐epoch指标离散均值，明确为归一化读数，不称额外独立样本。

数值收益保护：比较文件中的SB字段永远是提前指定candidate，SN字段是提前指定baseline；N1表示真实q臂，其余比较可用这两个字段承载明确记录的模型角色。不得解封后交换方向挑选“方法”。宏AUC平均差≥.003且双侧95%t下界>0；六cell差各给单侧下界mean−t_(1−.05/7,n−1)*SE，需≥−.003。加权LL先在每seed算(SB−SN)/SN，再给单侧上界mean+t_(1−.05/7,n−1)*SE，需≤.01。七保护界Bonferroni同时控制的近似依赖t假设。

n17_method_guards.py输出所有读数，不以某个有利阶段替换预先声明的主要读数。保护不通过、不精确或没有实际收益均分开记录。numerical_benefit_and_protections_pass仅是数值条件；新颖性、预先冻结候选/主读数、独立确认、成本缺一不能宣称GO-METHOD。未触发新候选时记录没有候选方法收益证据，而非所有未来方法无效。

训练耗时来自共享GPU资源，不伪称独占吞吐。resume后的elapsed_seconds原字段是当前段时钟；最终处理必须区分重启/延长段与累计计算时间，不能将H200末段时钟当全部200轮成本。samples与steps用原累计字段，KR49/ML5步每epoch分开解释。

主agent自审：公式、方向、七项族及ratio定义逐项对照原规划，无新test数据参与；PASS_SELF_REVIEW。实际最终清单/字段仍在统一冻结前核对；尚未运行最终test读数或收益判定。

最终图实现n17_figures.py：仅接受已完成performance receipt与同SHA比较清单，再绘N1两主要共同eta的六cell、H100 AUC/完整train J曲线及预设alpha权重敏感性。八训练seed逐点t区间，不作为全局onset或同时保护界；六cell图也明确pointwise，七项Bonferroni保护另表。沿用原宏主指标，不从图挑权重或选点。主agent逐式读数/条件门及语法核对PASS_SELF_REVIEW；尚未运行最终图或新test。
