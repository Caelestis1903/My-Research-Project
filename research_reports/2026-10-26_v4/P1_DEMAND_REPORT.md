# P1 需求可预测性：完整阶段报告

P1的两数据×两seed已全部正常完成：四条各24epoch原生前缀、各12pass来源pilot、每前缀36个固定20pass的折内神经策略头及12组train-fold常量拟合。每个P_form行由未见其本阶段标签的头预测，两任务共用固定fold；没有用全池m/τ调OOF动作。该阶段不是最终模型的用户隔离测试。

| 数据 | CONST−DEMAND LL收益5101 /5102 | 平均收益 | 每seed非负cell | LOGIT−DEMAND均值 | 注册决定 |
| --- | --- | ---: | --- | ---: | --- |
| MovieLens | +.000889898 /+.001288802 | +.001089350 | 6/6、6/6 | +.000217679 | PASS→完整P2 |
| KuaiRand | −.000529184 /−.000271951 | −.000400568 | 2/6、1/6 | −.000836168 | FAIL→P2/P3/P4 NO_START |

源码/数据/NPZ/完成收据的身份均已核验，逐行重算macro LL/AUC最大差为0。判定JSON为 `DECISION_P1_FINAL.json`；五项布尔值逐项公开。开发晋级是资源分配规则，不等同统计显著性或真实效应为零。

- [MovieLens全部宏值、seed区间、原始用户bootstrap与域解释](P1_MOVIELENS_ANALYSIS.md)
- [KuaiRand全部宏值、失败条件、row-bootstrap限制与竞争解释](P1_KUAIRAND_ANALYSIS.md)

两数据不合并成一个平均收益。MovieLens需求选读有 **MEDIUM** 开发支持；跨支持的函数形成和完整方法价值须继续由其P2/P3检验。KuaiRand当前DEMAND版本未获支持，证据 **MEDIUM**；其LOGIT相对常量的LL仍两seed改善，禁止把失败扩大为“此数据不存在任何条件需求”。full-h相对LOGIT的独有机制及两数据差异的原因均为 **WEAK**。

种子是独立重复单位，n=2描述性t区间单列；两数据分别进行了每seed1000次配对bootstrap，并重算六格及macro LL/AUC。ML按4545个原始用户共同抽样；KR因processed表缺可信原始用户ID采用规范允许的row退路，未证独立性。bootstrap、fold、六格不作为新增seed，未与训练seed区间合并。

域规模与基线难度并不等价。当前两数据的最小域都不是CONST AUC最低的域；ML相对CONST三个域在两seed的LL/AUC都改善，KR域级方向不稳定。P1没有一个可稳定推广的小域改善/大域下降规律。固定比例下，规模与域身份、标签基率、输入字段、支持与已训练表示混杂，规模作为主解释的因果证据 **WEAK**。

目标P_form的OOF可用性不能证明需求在source fit支持上仍有含义；source需求没有可直接评分的真值。ML本轮已按预定程序触发UNIFORM、GEOM、SHUFFLE、强本地/直接概率等P2对照；KR不生成被关闭的形成结果。所有已触发对照完成前不根据局部胜负作最终方法决定，也不追加新seed或搜索。

完整逐seed、六格、natural LL/pooled AUC、bootstrap原始复制和校验来源保存在本轮 artifact `analysis/p1/{movielens,kuairand}/`。数据身份、checkpoint hash与训练/选择角色分别保留，不将已暴露test或Q重新称为未见确认集。
