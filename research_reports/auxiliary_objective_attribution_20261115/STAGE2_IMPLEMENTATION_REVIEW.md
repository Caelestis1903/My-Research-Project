# Stage 2 实现独立审查

**Verdict: PASS。Blocking issues: 0。** 已独立阅读冻结 `STAGE2_DESIGN_SPECIFICATION.md`、实际stage2 core/protocol/runner/tables/tests、最终coder报告以及完整工程收据。允许派发已冻结的10条NDP-A训练，Stage1代码、报告和指标保持原值。

最终Stage2源码hash：`831cba9947738652cc6d4e5abf864e1ab4e32545b304936cb39aba28b227893b`，67份源码/冻结provenance文件；Stage1源码仍为`a78bce2f713cf62f9af32548b50bc360b134420ff9ed6ac6990dca22d76a2ad0`。当前source_manifest、source receipt、CPU/smoke/zero-counterpart收据和run manifest完全一致。源码身份包含实际导入的Stage1/main helpers及冻结触发证据，纯Stage2报告/导出变动不改变训练hash。

## 设计忠实性

- 所有自然B行的两个task均保留。近零敏感度/RMS只将对应delta置零，不删除entry、不进行pair/label/cell采样。最后短batch使用实际2B分母。
- 训练loss为`.75*clean+.25*.5*(view0的task0 BCE+view1的task1 BCE)`；每个row-task总系数1/(2B)，12层自然质量合计1、每task .5。两个view的另一task输出只参与drift诊断。
- delta detach；敏感度沿用已审查的完整数值gate Jacobian，普通参数训练仍保留历史gate detach。一次zero_grad、combined backward、Adam和after-Adam EMA，探针不写参数grad。
- 两套delta全零时返回canonical source clean BCE图，避免不同FP32求和/反传顺序。显式凸混合的loss/梯度检查与canonical路径的精确状态检查职责分开，符合CLEAN复用门。
- 新worker、job集合、attempt和RAM guardian独立；只有两数据集×seeds2022–2026的NDP-A，固定α=.03、λ=.25、50epochs、batch40960。预计500epochs及各17250次Adam/EMA更新。
- Adam/WD、EMA、loader及最早严格最大validation六格Macro AUC选点沿用源协议，online/EMA独立。没有Test、HPO、NDP-B、projection或性能提前停止入口。
- 自然支持以原始row-order hash、冻结loader/data身份和全支持计数记录；没有重写81920-entry数组。自然系数质量、BCE分子/分母、zero/drift分布、同步GPU计时和状态/日志hash齐全。
- 模型/Adam/EMA/RNG/loader/best/selected只在独立RAM guardian中保存；完成后保留selected状态，Stage2没有release或磁盘模型checkpoint路径。调度逐卡显存/功耗<85%、host RAM<90%准入，未知首批峰值时每卡最多一个allocating worker。

## 已核验工程证据

复用相同最终源码的**5项CPU检查PASS，5.75秒**：显式2B凸混合loss/gradient、target-only视图、全支持/短batch/缺域/单标签/近零fallback、canonical zero/source clean exact状态、连续及RAM恢复的model/Adam/EMA/selected身份。未重复运行CPU工程矩阵，也未重扫Stage1的5.77GB支持流。

两数据集均通过真实40960行NDP单步backward及canonical zero/source clean检查；full-batch共有81920 entries，clean/view/total质量=.75/.25/1，每task保留40960项，每次恰好1次Adam/EMA更新。两数据集的zero与直接source clean **model、Adam、EMA三个state hash完全相同**，初始hash匹配相应CLEAN seed2022。10条索引CLEAN result的SHA256均匹配，zero_counterpart收据与CPU/smoke/index hash一致，因此CLEAN可同时承担本轮零delta counterpart。

| 数据集 | 峰值allocated bytes | 峰值reserved bytes | NDP同步step秒 |
|---|---:|---:|---:|
| MovieLens | 2500704256 | 2520776704 | 1.360319 |
| KuaiRand | 2905653248 | 3017801728 | 1.300548 |

实际四行Jacobian/有限差分沿用冻结helper和容差，结果通过；源数值图与Stage1一致。资源准入采用上述实测峰值，没有降低batch或改变更新语义。

本次没有需要修正的阻塞项。科学表现及下一阶段门在完整10条运行结束后评价，当前不作性能结论。
