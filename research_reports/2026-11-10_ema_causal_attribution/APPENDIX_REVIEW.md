# 定量附录生成器独立审阅

日期：2026-11-11。Verdict: **PASS**。无 blocking findings。

审阅代码：`build_quantitative_appendix.py`，SHA-256 `4dba249bdd0e5a673a330e5cb8a3a96c82d80fd2ffa708c059a4451b76a194f9`。审阅对象还包括 coder 实现报告、根代理明确的十项机械附录需求、实际 producer schemas，以及当前 `QUANTITATIVE_APPENDIX.md` / `EVIDENCE_SUMMARY.json`。本审阅不作研究结论。

## 十项范围核对

1. 全 V 的 O/F early E5/E10、middle E20、late E35/E50，以及 late 六格 O 基线和 F−O，均分别输出 AUC/LogLoss。
2. late V 与 V_B 的 F 减 O、B/N、B/S、EMBED/STAR_SKIP/EXPERT 删除态及 C_PRIMARY recent/old 删除态，身份和方向明确。F−C_PRIMARY/recent 是删除损失，不被称作 recent-only。
3. FUNCTIONAL_EFFECTS 的 O_F/X_F、h/fused、late V 四个指定项分别保留 contrast、cut 和原 outcome metric。
4. EMBED 时间 factorial 按 ML h40 / KR h138、O/F 背景和全部六项提取，与 producer 的 C2 身份一致。
5. 准确 pair F−O 的 corrected、broken、delta_auc 直接汇总原有归一化贡献；源码仅用 counts/PN 作一致性检查，不再次归一化输出。
6. q_logit / q_probability 减 F 使用同 block、seed、anchor、V_mech population 和 cell，不误用全 V 或另一时刻的 F。
7. task 融合删除按 O_F/X_F、V/V_B、E35/tF 分开，输出 F−删除态；tF 不并入 late。
8. V_mech/V_B 桥接 utility、delta_static、macro_logloss_delta 直接转录已保存 BRIDGE_SUMMARY，不重新拟合或重算科学估计。
9. Test 五态值、独立 selected F−O、F/S/N 对 OatF/F 的同刻比较，均包括 macro/六格及原/ref epochs；两种比较身份分开，同刻 epoch 不一致时拒绝。
10. Gaussian−Clean 是同 seed 的两个既有 F−O 效应之差，分别处理 late V 与独立 selected Test，不把独立 selected 与同刻干预混为一类。

表、contrast、population、源行号及原表相对链接保留。未运行或未选中的来源诊断不被补造成效应。默认执行要求四个 COMPLETE block 和完整 Test 输入；只有显式 `--partial` 才生成缺失依赖快照。脚本不调用模型、guardian、训练、Test 评价或拟合代码。

## 必要复核

- 独立核对当前 JSON 的代码哈希、全部33个源文件 SHA-256及相对链接。当前677条数值记录均有五个有限 seed 值；逐条重算均值、样本SD、t4区间，均一致。明确 late 记录均使用 E35/E50。
- 已知 ML-C late 全 V F−O 均值 `0.0038398999027555282` 一致。汇总先在同 seed 内按 cell/指定 anchors 平均，再跨 seed 统计；不以 cell 或 anchor 扩大 n。
- 从实际 ML-C 原始表独立复算两种 q−F、corrected/broken/delta_auc 的五 seed 数值，均匹配附录。原 pair 贡献没有二次除以 pair 数。
- 对实际附录中的 ML Gaussian−Clean AUC 交互逐 seed核对，等于对应两条件的 late F−O 差值。
- 用临时标量 CSV 检查尚未产生的 Test 分支：O-selected epoch20 与 F-selected epoch35 可形成独立配对；同刻四态保持 epoch35。得到56条独立 selected记录、336条同刻记录、28条Test Gaussian交互记录，身份/数值正确；人为不一致的同刻 epoch 被拒绝。该检查没有读取真实 Test 数据、执行模型或写入正式结果。

当前附录明确标记 PARTIAL。32条缺失记录均指向尚未完成的 KR-G 或 Test 及其 Gaussian交互依赖；不表示零效应或阴性结论。源文件链接和 JSON 保留完整浮点数，Markdown 仅作可读展示。

## 适用边界

本 PASS 释放机械附录生成器及当前 partial 快照；不宣告整个实验完成。真实 KR-G/Test 尚未出齐，因此这些完整输入的最终运行仍待后续执行。没有重新导出大表、运行实验、拟合或修改研究算法；审阅者只新增本审阅文件，未覆盖其他工作者修改。
