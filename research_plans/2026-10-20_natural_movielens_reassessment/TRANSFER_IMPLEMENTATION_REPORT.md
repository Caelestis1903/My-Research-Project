# N19 / CM19 实现报告

按 start.md，由主代理实现。研究源逐字归档至 experiments/natural_ml_transfer/source_archive，inventory 记录原提交、路径、SHA；没有修改原算法文件。N19 为 c440350681a7a3e41ab3f05eb1c4d31f419b989f，CM19 为 eab817b3de626caee9ee9aeb1db4dd9aa3ef9e4c。

## 配置和训练

N19 原236完整 flags 映射，加入80容量匹配控制，共316；CM19 原436完整 config，加入72匹配控制，共508。824个唯一全局ID，所有原种子保留，没有从阴性方法中删项。人工CM19 24仍从头训练，未使用历史别名减少次数。manifest保存逐条历史config、source hashes、独立人口与路径、机械命令。

原Runner fit/update/optimizer/model/模拟器完整保留。普通100epoch；CM19 G4LONG按原2350steps。匹配曝光只截原完整main排列前178125，保留完整支持与RNG流，每100macroepoch恰500main更新、17812500行；匹配N独立支持main178126，明确近似N匹配。原N19 trajectory按fullN推导的examples外层纠正为实际观察行数；逐batch实际domain/rowID SHA、全行曝光计数、checkpoint SHA另存。N19秩剪枝额外只读保存前后state及有效谱，不改原SVD/重置/训练。

## 测试与机制

所有训练先标 COMPLETE_TRAINING_TEST_DEFERRED。每个study×scope全规定训练及development处置关闭后，由主代理签署closure，冻结配置/选轮/selected/final SHA，再调用原零优化测试。没有提前读test。N19原rank/dynamics/routing/credit与CM19 local/cost转接共143个机械任务；N19 rank/credit仅将原循环的dataset列表限定MovieLens，数值公式不变，不伪造KR人口。CM19成本为每(model,variant)一个代表，不当四seed性能实验。离线common-state局部counterfactual在临时模型上操作，不修改训练checkpoint。

## 最小必要检查

824配置覆盖断言及3个人口缓存生成完成；N19/CM19原模块导入成功，实际自然main760161、meta40006，domain main148152/302670/309339。匹配permutation长度、RNG状态委托已核；未构造新替代模型、未进行额外缩短训练。原fit保持完整，source drift作为运行门槛。中断完整attempt保留，同seed从头重跑，不使用原resume。

本报告是实现证据，不能当824次训练已完成。训练与诊断交给luna_worker，主代理负责后续分析。
