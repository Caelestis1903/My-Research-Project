# Stage 1 最终证据与推进审查

**Verdict: PASS。Blocking issues: 0。** 已核对最终 `STAGE1_ANALYSIS.md`、`decision.json`、`EVIDENCE.json`、逐seed/汇总/支持/权重CSV和两张图。支持按冻结计划进入一次NDP-A目标检验；当前B/C及零扰动构造仍低于CLEAN，没有算法成功或Test验收结论。

最终decision SHA256：`5845910e92cd37a97725dca86878c077546e8dca5a465b818f06549a68f652c1`。科学报告SHA256：`b5bdfe9387a3d0068daa1c6eca4f39236aa0474ebf4bdfc1404cb374220d2568`。证据SHA256：`1687fdc42543bff98e35d71e8554f5346611391c29536d5b16c430d8939da4db`。训练源码保持`a78bce2f713cf62f9af32548b50bc360b134420ff9ed6ac6990dca22d76a2ad0`。本次只补齐decision的independent_review块，完整数值和科学决策字段保持原值。

## 完成量、身份与统计

40条新轨迹、2000epochs、各69000次Adam/EMA更新完整；加10条CLEAN后共有2500份epoch记录。100个readout的最早严格最大选点、对应状态hash、40份data manifest及500组dataset/seed/epoch跨臂自然行序均一致。2000份支持gzip共5774198156 bytes的逐文件SHA256匹配，失败attempt为0，Test=false。这些已完成核对的证据直接复用，没有重复工程测试或支持日志导出。

4752条配对汇总和23760条seed差逐项核对通过；五seed、sample SD、mean±2.776445×SD/√5及正向seed数正确，CSV往返最大误差1.67e-16。216个phase对比先在各seed内对相应epoch求均值，再对五seed配对，epoch未充当独立重复数。全部端点分清online/fixed EMA、selected/common epoch及Macro/Traffic/Pooled。

480条cell-label-seed权重汇总使用实际1000或2450个batch作质量/贡献分母；总B/M明确为sum(B)/sum(M)。2000条完整epoch训练记录及2000条首batch诊断一致；每seed恰好50条首batch，梯度范数/cosine没有缺失值，先seed内等权均值再五seed汇总。日志口径没有被写成全batch梯度诊断。

## 支持与科学解释

34500个匹配batch的候选身份全部一致。C/Z-EXAMPLE在两数据集所有seed/step的accepted row-task完全一致，Jaccard=1。B/Z-PAIR的对称差entry为ML18、KR874；平均Jaccard分别.99999765625/.9999535247221097，最低值分别.9986979166666666/.9908854166666666。报告保留了pair的自适应支持解释，没有把B/Z写成逐步固定accepted-support实验。

两个Z相对CLEAN在两数据集的Macro AUC均0/5正向、六格均值全负，完整B/C也低于CLEAN。把当前辅助构造的整体退化标为STRONG有证据。报告同时说明额外损失尺度、cell/label构造尚未各自隔离，实测自然权重偏移作为MEDIUM竞争解释，未被直接写成唯一原因。

MovieLens C−Z-EXAMPLE为+0.000352080、4/5正向、5/6格均值正，CI跨零；KuaiRand selected均值为−0.000113225，且存在阶段/选点反号。报告将扰动排序增量标为WEAK，并完整披露这些矛盾，符合证据。AUC与LogLoss独立裁定；没有把EMA总收益、概率损失改善或局部正格升级为新组件普遍成功。

**ADVANCE_NDP_A有计划依据。** 一数据集的五seed正均值、非单格驱动及实测权重偏移支持固定α=.03、λ=.25的单次自然row-task凸混合检验。计划没有CI排零或全seed正的硬门，本审查不增加这些条件。推进范围为新的设计→实现→独立审查→smoke→两数据集五seed运行；HPO、Test、投影和额外机制实验继续等待后续结果门控。

域人口、标签支持和参考表现分别报告，未把域大小作为困难度代理；本轮没有独立规模干预，报告没有据此识别规模为主因。两张图的域/指标标签、arm图例、零参考线和区间可读，曲线与配对图展示的方向与数表一致。

## 已修订问题

R1：decision的selection_hash最初采用原始文本SHA256，run_manifest/config采用相同文本JSON字符串digest。现已统一为`c67a9771fc5980f76f3fd4cfe0afd29a260ab987bbe816865452fbdc69590b89`，并同步科学报告修订记录及analysis_hash。选点规则、训练源、模型输出和指标均未改变；修正后身份字段一致。

本阶段没有新增实验或扩大工程测试矩阵。最终任务审查留待后续阶段完成。
