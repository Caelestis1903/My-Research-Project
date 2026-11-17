# C05 CELL-MASS-BCE-025：方法与识别边界

本轮承接 C04 已发布的 completion / mass-scale 归因实验，仅增加 U 一个方法。设计先于实现，独立代码复审先于工程运行，资源复审和父智能体 RAM 验收先于正式运行。研究方案与接受记录位于 `research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05` 和同名 artifacts 目录；完整设计文件为 `DESIGN_SPECIFICATION.md`。最终结论由原始前台主智能体在五个 seed 全部完成后分析。

## 比较对象与假设

自然批次有 B 行、两个任务；cell c=(domain,task)，n_c 为该 cell 的行数，w_c=n_c/(2B)。对类别 a，π_ca=n_ca/n_c，b̄_ca 为普通概率空间 BCE 的类别均值。没有样本时保持设计规定的空 cell / 缺失类别分支与 null 标记。λ=.25。

- C：源 CLEAN，自然样本与任务权重。
- W：C04 completion，Σ_c w_c Σ_a (.75+.25π_ca) π_ca b̄_ca。
- m=1−Σ_c w_c·.25·2π_c0π_c1；S=mC，为 C04 总 BCE 质量缩放控制。
- a_c=1−.25·2π_c0π_c1；U=Σ_c (w_c a_c)·mean(BCE_c)，生产实现按 domain-major/task-minor 次序逐 cell 累加。

W 和 U 的 cell、task、全局系数质量相同；U 将 cell 内两类权重统一，从而去除 W 对 cell 内类别的不同加权。U 相对 W 只在相同主干、优化和数据约定下改变这项政策；长期训练之后的差异同时包含 Adam 状态、EMA 和轨迹历史，不是可直接解释为中介效应比例的单步因果分解。λ=0 或所有 cell 都单类时，实际运行委托源 CLEAN，验证精确一致。没有类别平衡、额外归一化或排序图。

## 固定运行协议

两套自然划分数据为 MovieLens 和 KuaiRand；新运行 seed 为 2022、2023、2024、2025、2026，每条 50 epochs，batch40960，FP32。新运行 10 条，另复用 C/W/S 的 30 条已绑定数值轨迹，共 40 条唯一轨迹。配对统计的独立单位仍为五个 seed；40 条不能作为样本量40。原始控制共1530个结果/epoch文件，逐字节绑定；不因本轮比较而重新训练控制。

M3OE 源结构、固定训练门控及 embedding.detach 约定不变；源 Adam 的 lr=.01、betas=(.9,.999)、eps=1e−8，耦合 weight decay 分别为 ML1e−4、KR1e−5。每次真实更新仅执行一次固定 EMA，rho 分别为 ML .9828205985452511、KR .995。不进行 HPO、缩短预算或因疗效提前停止。

ML train800167/validation100021，每 epoch20个批次，尾批21927；KR train2006132/validation250767，每 epoch49个批次，尾批40052。正式合计500个epochs、17250次真实 forward/backward/Adam/EMA、701574750训练行和1403149500任务项。Test 全程封闭。

online 与 fixed_ema 独立按 validation 六 cell Macro AUC 的最早严格最大值选择；LL 属于同一 AUC 选中状态。固定阶段 E1/E10/E30/E50 作为预先指定的阶段比较，不得用其中较好结果替换 selected 主终点。新模型的20个选中状态只在 RAM 中保留，复用历史控制仅保证数值来源，不能声称仍持有其模型。

## 诊断与资源测量

U 每个 seed 在 E1/E10/E30/E50 第一批进行共同状态 C/S/U/W 诊断，共40个点、120次额外 autograd sweep、160次虚拟源 Adam。虚拟更新按顺序克隆，真实训练不增加 forward、真实 Adam/EMA 或算法 RNG。诊断必须验证真实 U 与虚拟 U 更新精确一致、诊断中立性和无参数组活动丢失。工程验证的混合 λ=0 委托分支中 H 为 null/not_computed_delegate；正式观察的 H 不能据此填零。

四个工程用例覆盖每张卡与每套数据，逐用例8次验证调用、2诊断点、6额外sweep、8虚拟Adam、2次 RAM 提交；工程提交不算科学 epoch。失败尝试、重放和未返回操作单列实际成本；未知工作量保留 null。验证通过后仅释放四个工程状态，正式选中状态不释放。

每张卡独立根据当前测得显存<85%、当前功率<85%、主机 RAM<90% 准入；在实际 OOM 后还要求当前占用加实测完整新进程驻留量不超过 CUDA 可用物理容量100%。不使用预测功率、投影85%阈值、固定 worker 数或双卡同时满足条件。新分配后重测；pending 只绑定当前 PID/job/attempt/RUNNING 和真实首个 FULL 分配 ACK。并发量随共用机器的即时负载变化。

## 统计与预设判定

17个配对比较族×2数据×5终点×34指标，形成28900个逐 seed 差值与5780个汇总。均值、sample SD(ddof1)、均值±2.7764451051977987·SD/√5 和方向一致性全部呈现。epoch 无有利方向。区间为相关探索比较的描述性区间，不作多重比较校正后的显著性或独立复制声明。

主比较为 selected fixed_ema 的 U−C。两套数据均要求 Macro AUC 平均增益≥.003，且每个 cell 的平均 AUC 差≥−.001；LL 是否平均下降独立报告。CI 和五 seed 全同号不额外变成门槛。两均值非正→NO_CONTINUATION_SIGNAL；一正一非正→MIXED；两正未达全部门槛→POSITIVE_UNQUALIFIED；全部达到→INTERNAL_SCREEN_QUALIFIED。任何结果都完成报告并按用户约定停止，不启动 C06，不自动开放 Test 或选 champion。

F2 检查 U−S 的 selected EMA task1 LL 是否两套数据都下降；F3 检查 KR U−W 的 d0t0 AUC 上升且 LL 下降。它们用于区分 cell/task 质量分配与 cell 内类别政策，不能替代主终点。STRONG/MEDIUM/WEAK 是本研究内的证据等级，应同时说明重复性、反例和替代解释。

## 规模、难度与外推

domain 大小取实际训练/验证计数；难度结合源 CLEAN 的绝对 AUC、LL、类别率及任务差异，不能把小 domain 定义为困难 domain。检查各 domain / task / seed / model stream / stage 的一致和相反证据；观察到同一轨迹的小域收益与大域下降，也不自动证明规模是主因。C04 中 KR 小域 d0 与 d2 已表现相反，ML 最小域的 baseline AUC 也不最低。历史 C01–C04、自然 NDP、NQR 时段、EMA 与 online 的矛盾结果必须保留。

本轮只识别当前数据、架构和优化协议下的反事实政策比较。若需后续科学检验，优先在同一 domain 内独立操纵类别比例与样本量、预设难度分层并保持配对，以区分规模、类别组成和难度；本次交付不自动执行该后续实验。
