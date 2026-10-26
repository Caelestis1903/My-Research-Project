# AO04 / N18 实现报告（主代理）

AO04原算法源码完整归档，补充state_writer只核SHA的decision.py，未调用旧科学决策模块。ao04.py原样调用paired_init、_train_val_loop_impl、state_writer、final_test_evaluator。原公开front-door只允许2022/23与cuda:0，本轮四seed/多GPU已预注册，wrapper检查locked config后直调未改的fit body。只注入自然counts/fingerprints、derived参数总数、数据和输出路径。

12配置（三臂×四seed），完整50epoch，1000main步、40008350main行、50arch更新，arch实际曝光另记。每run在独立目录保存同seed确定的B0 init，B1/B2按原严格key加载、q零初始化，收集时核同seed共同tensorhash。完整last_resume、实际optimizer states、selected/final/common模型、逐batch指纹保留。中断attempt保留后同seed完整重跑。

原epoch0..49，新receipt同时记录zero-based和1..50选轮；原每epoch全train/full-validation评价及最早严格最大选模保持，shadow-stop只记不早停。全部12训练/development处置关闭后主代理SHA冻结，再原零优化测试，不偷换N17半val。

ao04_diagnostics.py固定probe为每域前8个validation行，在初始化/selected/final保存E_j−E_k、a_D(1−b_D)、真实tower logit关于q/rho的逐行Jacobian及q±.1预测/风险。B0使用同权重B1 q0局部构造，明确不是训练的B1。单类cell AUC未定义单列，不造0.5；小probe不当完整验证或训练收益。原全epoch expert/gate entropy和fusion轨迹保持。

n18.py将原main的dataset循环限MovieLens，数学/LP/遮蔽/audit算法和随机流未改，注入自然bundle及独立OUT。额外同流重放保存mask/audit rowID SHA，与原n/observed核对。0训练，无test指标。原raw timestamp经rowID/rating核对后按原四分位环境重算，稳定机制仍未识别、曝光政策未观测。

必要检查：新模块语法通过；自然155949/318599/325619，validation/test各100021，原身份函数PASS。seed2024 B1/B2加载完整B0 keys、q全零，6行初态预测与B0最大误差5.96e-8；未backward/optimizer update。正式AO04尚未启动。一次命令误用了系统Python而导入失败，正确环境重新执行；未进入训练/数据计算，不记科研阴性。
