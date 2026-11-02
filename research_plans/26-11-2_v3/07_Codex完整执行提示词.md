# 07 可直接交给 Codex 的执行提示词

你在 Caelestis1903/My-Research-Project 工作。本任务实现并执行 `research_plans/26-11-2_v3`，不是重新规划整个研究方向。先完成实现审查和S0，再按冻结矩阵执行；不得越过科学门自行扩展候选。

## A. 先读上下文，不重开已接受结论

依次读取根目录 EXP_AI_CONTEXT.md、experiments/Experimental_Global_Summary.md 的最新权威段、AGENT.md，然后读取本目录00–06、08与experiment_queue.yaml。小域不是困难域；固定非反馈参数EMA收益、尺度效应及旧版本停线结论直接继承，不安排对应假设性实验。

本轮论文式主张尚未被实验支持。CPU代数PASS不是性能证据。不得写“发现EMA有效”“首次修改动量”“历史所有动量方法无效”等错误结论。

## B. 仓库和依赖

检查git status、当前main HEAD和既有工作，不覆盖他人文件。创建独立实现分支，新增 `experiments/conditional_information_momentum_20261102_v3/` 与 `research_reports/2026-11-02_mdmt_information_momentum_v3/`，不修改历史报告，不恢复learned fusion/AutoML。

本轮方案以main@96d21fccff4e9304589e82527ec7a859e8c3f05c为上下文；运行底座固定为EMA02V2@38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27。从该SHA取得 `experiments/ema_view_design_20261102_v2/` 所需data/evaluate/protocol及vendor依赖，保留原字节source manifest。不要整体merge历史分支到main，不假设当前main根目录存在同一训练实现。

数据沿用既有真实CSV和固定manifest；无法取得所需原始数据时报告执行阻塞，不能用合成数据冒充正式结果。

## C. 明确实现模块

建议文件：`boundary_sampler.py`、`logit_capture.py`、`parameter_groups.py`、`objectives.py`、`optimizers.py`、`state.py`、`protocol.py`、`run.py`、`evaluate.py`、`tests/`。名称可调整，语义不可缺失。

必须提供A0/A1/P00/P10/P01/P11/PCV/PCOLL；静态/动态需求、adam/eccm/cv/collapsed历史模式、lambda、EMA评价均可显式配置。不得在没有依赖定义时直接复制原runner的白名单seed限制；新确认seed2027–2031必须被允许。

真实forward保持gate emb.detach，使用tower hook取得训练图raw logits；不能用冻结参数的input_probe代替。参数分组按05的外层模块规则，不按domain_specific子字符串误分组。

ECCM需实现真正的一阶矩分源重组、alpha/q历史更新和active clocks；CV使用旧表构造当前梯度后再更新表；PCOLL是一份同监督pair历史。三者不能共用一个未区分含义的q变量。参数EMA必须固定且非反馈。

## D. 必须通过的实现检查

1. lambda0旁路与源A0的forward、loss、参数梯度、一个Adam step、EMA以及恢复后下一步一致，允许预先规定的浮点容差；这只是防止接线改变基线，不是重新证明EMA科学收益。
2. hook logits sigmoid与源probability一致；保留gate反向断图，真实BCE仍用源probability函数。
3. ML不制造01；KR实际01不删除；正负配对只在同域，另一任务标签只作为条件，不作为输入特征。
4. q归一、q正、w目标质量不随采样改变；K≤24时alpha/q≤.96、w/q≤4。缺域/缺类别时不填假样本。
5. 精确枚举小系统检验CV条件期望、ECCM频率校正期望和先读后写；对None/显式零、active与selected clock分别测试。
6. 当前mixed梯度含coupled wd仅一次；每参数bias correction正确；一次逻辑step、一次EMA，无双优化器重复更新。
7. H/D不保留计算图；完整checkpoint含历史、时钟、RNG；resume的下一步一致。CPU测试和真实smoke分别记录，不能互相冒充。

## E. 执行与有限搜索

先由独立审查者或明确隔离的审查阶段出具PASS/具体阻塞项，再冻结代码、配置和source hash，执行S0及06所定开发矩阵。不得因为看到旧test小幅正值而补搜索；不得在失败后把beta、rho、wd、学习率、模型结构和输入视图同时改动。

lambda的有限网格与必要历史控制已给出。任何额外训练必须说明它会改变哪个具体算法选择，并受06总预算约束；不临时增加“先验证某个已接受事实”的研究阶段。

已有A0/A1只有数据、初始化、loader、训练语义、seed、epoch、EMA及选点全部匹配时才可复用；报告逻辑端点与实际新增轨迹两种计数。没有checkpoint则不声称完成不存在的前向复核。

## F. 交付

输出SOURCE_MANIFEST、DESIGN_IMPLEMENTATION_MAP、CPU_TESTS、STAGE0、RUN_MANIFEST、DEVELOPMENT_DECISION、CONFIGURATION_FROZEN、SELECTION_FROZEN、逐seed及六格CSV、成本表、REPORT和下一步明确决定。

报告结构采用当前结论→核心证据→适用边界→研究决策。分别回答：新监督有无收益；动态需求选择有无增量；分条件动量是否胜collapsed-history；是否只需CV或普通Adam；两数据/六格的收益与代价。不要把机制未完全解释扩大成所有经验结果不成立。

研究设计文件已经并入main；本任务的实验代码和结果按仓库现行分支工作流提交，不覆盖历史。最终给出真实执行commit、运行身份、未执行阶段及其原因，不编造AUC或资源耗时。
