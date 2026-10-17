# 科研机制分析：证据、反例与边界

**BOUNDARY-CONFIRMED；NO_GO。** 本研究确认：同一Sample经验目标及既有stop-gradient图导数字段期望下，q改变有限batch估计器协方差并可改变训练路径和性能；性能效应只在明确的优化器/LR、阶段、选择规则、评价人口和域任务条件下成立（MEDIUM），同图/同均值与采样身份为STRONG。普遍Adam必要、全部selection-only、全部late-only及“小域等于困难域”的解释被反例限制；课程或动量的普遍收益、N主因和统一小域改善/大域损害取舍没有证据。C局部AUC路径为阳性，但联合loss/J不一致，重复/coverage、形状/trace、内容/尾批时钟和状态/进度未独立分离，唯一长期必要中介仍未识别。既有importance sampling、order和optimizer-memory工作部分覆盖一般主题，D3/SAF完整目标不同，固定AdaTask组件残差不代表原论文方法普遍失败。唯一终局分类为BOUNDARY-CONFIRMED，方法决定NO_GO；所有本轮规定且触发的实验、最小控制和冻结测试均已完成，未触发分支与更强未识别主张逐项保留。

## 已确认的观察与证据强度

- **STRONG：** 同前向经验J及同stop-gradient图导数字段期望身份；完整固定θ/批次权重/模型与产物SHA；同q三severity行/顺序/曝光耦合；R已私有/C和E共享/P混合。STRONG身份不等于泛化机制STRONG。
- **MEDIUM：** 明确配置内早期q影响，optimizer/LR交互与顺序阶段敏感性，whole-state映射局部/条件影响，以及C在E10/E40的selection AUC局部路径。test按冻结规则另行确认，见[数值核验](analysis/test_all_cohorts_primary_verification/receipt.json)。
- **WEAK或不可识别：** 唯一长期必要中介、纯content-order、coverage独立于repetition、shape独立于trace、全局等范数下纯方向、N主要因果驱动、普遍困难域规律、小域改善/大域下降取舍，以及新方法全面收益。

## 反事实与竞争解释

[V12终局综合](V12_final_causal_synthesis.md)分别列同J改q、同q改optimizer/sampler、同multiset改order、同θ改块、2B/exact六项。目标/图不一致已反证；泛指所有效应只由Adam、selection、late-overfitting造成均有共同早期反例。不能因此排除速度/clock、参数路径、高阶估计器分布、输入/表示law、coverage/repetition耦合与多通路共享效应。

固定E,N下R=E−U≥max(0,E−N)，SB小域E>N使零重复同资源对照不可行；PERM保留重复并改变依赖结构。ORDER1000臂轮964次移动短尾，不能把纯内容与batch-size-clock隔离。大batch同时改变步数。名义2x/4x别名没有新独立支持。V8 mask改变J_c且只有一个固定嵌套mask，每seed并非独立mask重复。

C的AUC筛选阳性必须保留：10/40轮四seed衰减≥50%；E40 selectionLL和trainJ各2正2负、平均衰减负，原联合一致门槛未满足，条件完整架构训练未触发。局部gap缩小不代表绝对健康；selection与diagnostic也不同。V11固定22的v-only缩gap主要来自SN下降；E50两臂AUC改善而gap交互不稳；β2=0的floor不能当机制被阻断；globalnorm=1仍改变CPRE块幅度。

四seedCI、20kbootstrap与精确sign检验分开，n4双侧最小p=.125；部分固定100只按真实n/seed预算资格描述。历史test暴露、ML训练子集seed混合、onset非FWER、移动θ前6批稀疏lag，以及routing detach均限制外推。没有用宽CI证明等效零来关闭核心，所以额外2026–2029分支未触发。

## Domain/task与模型阶段

KR的中等d0在多baseline中比最小d2更低AUC，N不能代替难度；正例率与train/validation人口不同，loss高低不能独立排序难度。V8目标域SN平均AUC随mask下降，但mild E50 loss略改善、q效应不单调；C/P/R/E稀疏范数也不单调。V5 E50局部小中域SB改善/最大域略损，但早期/最终、V8/ML反例阻止统一取舍；证据WEAK。只有3域，不以伪回归识别N主因。

ML共同16与KR各自37是不同估计量，跨环境数据/输入维度/标签/步钟/采样子集共同变化，不归于唯一dataset/N/difficulty。预设宏观负效应未稳定复现，深层机制复制条件未触发；最终test不反向新增训练。

## 文献与方法判断

九篇原文审计区分目标等价、干预和机制。D3/SAF完整权重一般改J，不能直接覆盖同J问题；固定AdaTask组件残余不能推出最优完整方法失败。一般importance sampling/variance、order与optimizer memory已有先例，没有识别唯一MDMT必要机制就不能宣布创新。详见[文献边界](V10_existing_work_coverage_boundary.md)。

[V13十项终局](V13_final_method_decision.md)：当前没有新方法在真实primary endpoint、四seed稳定且所有域任务无损的完整证据，方法决定为NO_GO。这不等于所有未来机制方法都失败。后续最能区分解释的设计为固定尾部时钟的顺序控制、可行E≤N资源下支持/重复控制、健康/进度匹配状态，以及另行预注册AUC-only C私有化；这些更强新问题不掩盖本轮必做项，后者已全部完成。

全部46问题终态与19新问题六字段见[终局问题表](final_question_closure.md)、[registry](question_registry.json)及[最终报告第13章](FINAL_REPORT.md)。
