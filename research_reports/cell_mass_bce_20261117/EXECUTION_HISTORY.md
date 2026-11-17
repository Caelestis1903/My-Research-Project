# C05 执行历史与实际成本边界

本文件记录已发生的工程事件；正式运行的最终计数以 RUNNER_HANDOFF、父验收、原始成本表和最终报告为准。目录标识为20261117，工程恢复跨入本地2026-11-18。所有 AI 子任务均在原始会话下用 fork_context=false 新建，完成后关闭，不复用上下文。未恢复退役的旧后台 driver。

## 来源与修复

1. fresh Astra 完成六份研究文件，选择 cell 内类别均化 U；designer 形成完整 DESIGN_SPECIFICATION，随后独立 coder 实现 core/runtime。
2. 第一版代码复审发现归档 guard 把模型描述、状态哈希等普通字符串也当作模型状态。专门 coder 只修 guard 和有限复现，distinct reviewer 复审关闭问题。原 FAIL 与修复记录保留，不覆盖。
3. 首次 smoke 控制器在准入记录重复传入 sample_id 时 TypeError，尚无 worker、训练调用或 RAM ACK。专门 coder 只修 admission_row 生产路径与一个有限复现，distinct reviewer R2 通过。旧55文件源码、零状态准备记录和修复来源保留在 retired_prelaunch_01 / SOURCE55_TEXT_PROVENANCE.json；不能把旧源码结果重标为新源码。
4. 当前科学闭包55文件固定为 f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d。此后所有训练与恢复均保持这份源码。原 RAM owner 仅在明确0jobs/0states时完成已审核的来源绑定更新，PID2926581/前台工具会话16855保持同一进程；未创建替代 owner。

## 同源码工程恢复

5. 首次双卡 MovieLens 工程启动出现共享参数字典首次写入竞争：case1在实际训练前失败，case3返回1次 FULL 调用后因暂停退出，尚无 ACK。父编排以 CPU 预先物化仅含名字、shape、dtype、numel 的参数模式；不保存参数值，无训练 forward/Adam。独立恢复复审通过。
6. 恢复控制器最初只解除一个失败 job，现有全局监督检测另一个 FAILED 而安全暂停，未启动新 worker。父编排随后在全部 worker 退出时用两个独立范围的复审凭据依次执行原 recover API，保持原模型状态与来源。原短控制器耗时不当作全进程训练耗时。
7. case1 完成8次调用、两次ACK，λ0、共同状态虚拟 Adam、诊断中立性和精确恢复证明均通过；最终 metadata 汇总却读取到旧零调用 attempt 的空 witness profile。完整 attempt 的实测峰值覆盖旧 profile 全部阶段，五种峰值均不低于旧值。父编排将旧 resource_profile.json 字节不变重分类为 incomplete_resource_profile.json，保留路径映射、SHA、成本与历史凭据；未改源码、模型或测量值。fresh reviewer Ampere PASS WITH FIXES 后，原 ACK2 状态通过现有无新训练分支完成 case1 receipt。
8. case1 metadata 收尾涉及一个 runner 提前返回“就绪”、随后消费已排队 GO、父编排关闭该子智能体的时序。实际 case.json 已产生。同一阶段的 PID2982386 仅有 STARTED，缺失最终 terminal/supervisor 记录；现有凭据不足以精确确认其 controller 父子关系或记录缺失原因，不能仅凭遥测文件名归因。父编排承担关闭时序的调度责任；未知未返回工作量和持续时间保留 null，不倒填为0或伪造清理成功。后续使用全新 Hooke 完成剩余任务，不复用已关闭智能体。
9. case3 在同 owner、ACK0和原源码下重新完成规定8调用；其物理总数9，包含先前未确认的1调用。两套 KuaiRand 用例随后在 GPU0/GPU1 并发完成，各8调用。四个用例的规定工作合计32调用/8诊断点/24额外sweep/32虚拟Adam；实测返回合计33调用/9诊断点/27sweep/36虚拟Adam。额外调用仅计成本，不能增加科学样本量或 epoch。
10. 父编排直接 get 并验证四份完整 RAM 状态、八次 ACK、continuation/scientific identity、不可变 prefix 和 suffix，确认114653192字节，写 PARENT_ENGINEERING_SUPERVISION_ACCEPTANCE.json。工程状态仅4次真实 optimizer/EMA 更新，engineering_commit_ordinal=2；科学 epoch=0。
11. 资源复审发现最后两个 KR 生产者 journal 尚未被汇入统一 witness registry；父编排在所有 worker 已退出时调用原 collect_witnesses，仅追加已有测量，得到13条/22868字节，SHA d8be1c314834cd9cedefe3d727da6a3b37e33665783e04f820e3ed88ec8f3bcb。历史前缀保持不变，无新测量或训练。正式运行必须等待 ENGINEERING_SIGNOFF 和四个工程 key 的父授权释放。
12. 首次资源签署通过、四个工程 key 已释放后，正式 controller2994424/tool69024 启动两个 worker2994508/2994509。worker 在注册前追加 STARTED，随后现有校验发现 signoff 的 terminal_core/supervisor_evidence 指向仍追加的 physical_attempts.jsonl 整文件，因而暂停。两 worker 均退出1，实际0训练调用、0正式注册、0正式ACK，controller区间2.64秒。父编排从原日志提取与旧复审 SHA 完全相同的49773字节完整记录前缀，保存为 ENGINEERING_PHYSICAL_ATTEMPTS_FROZEN.jsonl，SHA67813012fa95ddd634d5666266d3d702358935e04a40ad00fa062c1baf05a3c9；原日志不截断。fresh Boyle 只复审这一不可变凭据绑定及零状态恢复，不改55文件科学源码，也不取消来源检验。旧signoff与release凭据以_01保留。
13. Boyle 复审通过后，新 runner 由 controller2999906/tool20669 运行10条正式轨迹，GPU0/GPU1各5进程。MovieLens五条全部50 epochs；其后父编排 PARENT_PROGRESS.py 在非原子 write_text 更新 AUTONOMOUS_STATE.json 时，停止控制读取碰到短暂空文件，触发 JSONDecodeError 暂停。此为父编排写入错误，不是疗效停止或模型算法失败。父编排改为同目录临时文件写完后 os.replace 原子替换；当前55文件科学源码不改。全部 worker 已安全退出，父 get 重新核验10份状态224702600字节、361次正式epoch提交：ML五条COMPLETE；KR seed2022/2023/2024/2025/2026最后确认21/24/20/23/23。PARENT_FORMAL_PAUSE_ACCEPTANCE_02.json 保留精确 continuation / selected 哈希、最后ACK与暂停字节绑定。后续只从这些确认状态恢复剩余KR；已经返回但未确认的工作若需重放，属于额外物理成本，不增加有用预算或样本量。fresh Lagrange独立复审父写入修正与同源码恢复。

## 计时和可用性约束

返回调用、未知未返回操作、工程重放、CPU有限检查、controller、worker、BytesIO序列化、IPC、诊断和 validation 分别记录。worker wall 与 supervisor process wall 相互重叠；训练调用秒数可能包含诊断，不能把所有组件相加成总耗时。没有 terminal 时，采用 supervisor 已知观察也须保留其时间含义，未知项不能用另一个阶段的均值填补。

C04 已发布完整数值与归档，但其旧 owner 在已确认验收后消失，所选模型当前不可用；C05仅复用其已绑定数值控制，未恢复、重建或冒称持有 C04 模型。C05当前 owner 的可用性由最终阶段重新直接检查，不能由历史验收推出未来持久性。模型/EMA/Adam/RNG/loader/BytesIO只在 RAM，Test 始终封闭。

14. 归档语义检查/完整回读通过后，真实容量门槛差281239635字节。保留677004657字节原归档及SHA，机械接续逐成员复核相同字节、临时转存43个本项目已发布C04分卷，使真实可用3441762304≥3373191251，再执行冻结pack原样剩余语句，得到81分卷。43分卷全部恢复，原有89卷与C04manifest/receipt不变。归档4026成员包含原截断遥测压缩字节的精确hex与1/15/1385完整前缀；原复审类型描述精确UTF8保存。标准pack未重跑，科学55文件未改；详见ARCHIVE_CONTINUATION_04与DATA_AVAILABILITY。
