# DESIGN SPECIFICATION：F0执行控制隔离修复 v1

范围仅为进程隔离、启动门、资源计槽、信号控制和故障审计。未修改代码、未运行测试或实验。本规格已由主控批准作为Coder实施依据；批准设计不替代实施后的独立审核或解除执行hold。

## 1. Research question

将同一Python进程反复执行训练，改为每条canonical轨迹使用新Python进程，能否提供可审查的生命周期隔离，同时保持实验身份、资源限制和可靠停止？

这属于执行控制修复，不验证算法收益。

## 2. Hypothesis being tested

已记录：worker0完成8条轨迹后，在第9次调用未观察到canonical claim之前，以SIGABRT退出，native stderr为`free(): invalid pointer`。

**根因UNKNOWN。重复CUDA/Python复用是WEAK假设。** 源码和时序不足以证明它造成事故。采用新进程是保守隔离措施；后续无事故也不能证明根因已确认。

## 3. Variables changed

- persistent supervisor只管理进程，不执行训练。
- 每次调用启动全新的trajectory Python child，child只调用一次runner。
- STOP/CONT控制单位明确改为整个worker进程组。
- 增加启动前槽位预留、父进程死亡防护、阶段日志与异常退出hold。

## 4. Variables held constant

数据、算法、随机种子、训练/架构更新、50epochs、选模、checkpoint定义、canonical key、GPU/resource/canonical锁路径、每卡独占及原恢复额度均不变。

已完成轨迹不得重跑。此次修复不释放任何formal/test门，不增加科学实验或恢复预算。

## 5. Algorithmic/control definition

### 进程结构

每个worker：

```text
shell wrapper ──exec──> Python supervisor（PID = PGID）
                              └── fresh trajectory Python child（相同PGID）
```

- worker ID仍只有0、1。
- supervisor使用标准库控制逻辑，不导入torch/CUDA或执行数据加载。
- child不得`setsid()`、改变PGID或创建脱离组的训练进程。
- child执行一次既有runner后退出；只有核验成功完成才允许下一child。
- DataLoader workers=0；新增Python子进程全部需要槽位，不允许隐式multiprocessing。

### 信号所有权

- **STOP/CONT必须针对负PGID发送**，暂停/恢复supervisor及child整体。
- 单独SIGSTOP supervisor PID不满足安全暂停，控制文档和命令入口必须明确拒绝该语义。
- wrapper `exec` supervisor，不能留下可被误认为worker的等待shell。
- 日志记录supervisor PID/PGID、child PID、host、启动时间及invocation nonce，供操作者准确控制。

INT/TERM/HUP：

1. supervisor立即设置停止标记，禁止新child/claim；
2. 向当前child转发信号；
3. 等待child退出，保留退出状态与日志；
4. child尽可能通过runner现有异常事务标记INTERRUPTED、释放锁；
5. 超过30秒仍未退出，supervisor对整个worker PGID执行强制终止前写入升级事件；外部控制端随后核验组内无存活进程。

不能保证native调用及时响应Python handler；不得把超时强制退出描述为完整事务清理。

父死亡防护：

- child在导入runner/native训练模块之前设置Linux `PR_SET_PDEATHSIG=SIGKILL`。
- 设置后再次核验PPID、预期supervisor身份及PGID，关闭“父已死亡后才安装防护”的窗口。
- 不使用多线程`preexec_fn`。
- supervisor死亡后child不得继续训练。
- 平台不支持、防护失败、父身份不符均在claim之前失败关闭；不能无防护降级运行。

无法满足上述行为是验收阻塞项。

### process slots

两个worker同时训练时共**4个Python进程**：2 supervisor＋2 child；其他本任务Python最多1个。

- wrapper在启动supervisor Python之前，在现有五槽命名空间原子预留supervisor槽及child槽；不足则不启动Python等待者。
- 两槽可在worker整个生命周期保守保留，避免两supervisor各占槽后无法启动child。
- supervisor采用已预留租约；child采用第二个租约。
- runner不得再额外获取第三个槽；需要审查后的“已预留槽采用”路径，核验FD/inode、owner与启动身份。
- 所有测试、评价、监控Python也必须启动前计槽。
- 采用的共享租约不能由child显式`LOCK_UN`释放supervisor仍持有的锁；child只关闭自身副本，supervisor在child退出后清理。
- 保持原worker-ID、canonical、GPU锁命名空间；禁止建立另一套不互认的槽。

## 6. Exact affected modules/files

- `experiments/stage_F0_outputs/runs/run_search_worker.sh`
- `experiments/stage_F0_outputs/runs/run_search_claim_worker.py`
- 新增最小fresh-child入口，实际路径由Coder登记。
- `experiments/stage_F0_run.py`：只增加槽位采用、阶段事件及执行控制接口。
- `experiments/stage_F0_queue.py`：租约采用、hold/recovery控制；不改科学队列。
- `experiments/stage_F0_outputs/runs/WORKER_CONTROL.md`
- `experiments/tests/test_stage_F0_launcher.py`及必要focused控制测试。
- 新版本执行门与独立审核报告。

## 7. Implementation requirements

### fail-closed launch gate

首次child启动前，冻结并验证：

- 修复设计、supervisor、wrapper、child入口、queue/runner控制修改、focused tests及新Reviewer报告的hash；
- 新launcher-specific verdict为PASS，或PASS WITH FIXES且阻塞项全部纠正；
- 原数据、科学协议、GPU/resource门仍有效；
- 无全局事故hold、非法活动claim或未核实的进程；
- process slots、worker-ID锁及有效worker身份均已取得。

**旧launcher PASS只覆盖旧in-process设计，不能批准新设计。**

执行控制hash与科学身份分开登记；不能改写旧canonical key或旧完成manifest。若现有code-hash校验不能表达修复后的身份，应设计明确、已审查的amendment绑定，不能绕过校验。

### completion与failure

supervisor使用OS级stdout/stderr重定向，保留native stderr；不能只使用Python `redirect_stderr()`。每invocation日志唯一命名，禁止覆盖“current”日志而丢失历史。

继续下一child必须同时满足：

- child exit0；
- child身份绑定的canonical状态为VALIDATION_COMPLETE；
- manifest/checkpoint事务核验通过。

不能只凭日志出现`VALIDATION_COMPLETE`字符串继续。

exit0但QUEUE_EMPTY需结构化确认；其他exit0状态不明也停止。任何非零退出、signal退出、spawn失败、身份不符均：

- durable记录失败和全局hold；
- 停止新child；
- 请求另一worker整体停止；
- 保留现有工件；
- **不自动重试、不自动恢复、不自动重启worker**。

并发启动/claim入口须检查共享hold，并序列化相关状态转换；异常发生时已在运行的peer执行停止清理，不能继续取下一个key。

### interrupted key `59474a37e9d12d956633`

保持hold：

- 核验`resume_epoch_003.pt` hash、完整state、epoch及3行metrics/resource记录；
- 核验canonical、数据、科学协议、随机流与恢复身份；
- 保存原INTERRUPTED状态和事故关联；
- 只有主控明确、可审计的recovery release才能恢复，遵循原一次恢复规则。

仅worker0“第9调用失败”不能把该key认定为worker0失败轨迹；它属于被停止的peer。预claim事故日志的canonical key应为null，不能猜测绑定。

默认claim路径不得重新取得held、异常退出或未经审查的stale RUNNING条目。

## 8. Ablations / focused validation required

无需新研究消融或正式训练。Coder执行以下控制测试：

1. 两个连续成功fixture使用不同fresh child PID，只调用runner一次。
2. 组级STOP/CONT同时暂停/恢复supervisor和child；不宣称PID-only STOP有效。
3. INT/TERM/HUP在child启动前、启动后、运行中均禁止后续claim并完成清理。
4. supervisor SIGKILL后child被PDEATHSIG终止；包括父死亡安装窗口测试。
5. fixture SIGABRT、非零退出、spawn失败、exit0无有效完成均停止并hold，无自动重试。
6. 槽位并发竞争：2 workers＋1测试进程=5；第6个Python在创建前被拒绝。无双计槽/租约提前释放。
7. stale/missing review/hash、hold或父身份异常均在数据/CUDA/claim前拒绝。
8. 完成条目跳过；held interrupted条目不自动恢复；原canonical/GPU锁仍互斥。
9. native stderr、signal、phase、checkpoint与failure回执均保存。

测试使用短生命周期fixture，不启动正式模型训练。每个测试自身也计入五槽上限。

## 9. Expected logging requirements

使用同步写入的结构化phase事件：

- supervisor invocation/start；
- child bootstrap/父防护完成；
- **dataset load/hash之前**及之后；
- **GPU preflight之前**及之后；
- **canonical claim之前**及之后；
- **training之前**及之后；
- completion validation、child exit、hold、signal forwarding、cleanup escalation。

每项包含nonce、PID/PPID/PGID、worker ID、时间、可用时的canonical/attempt/GPU身份。

保留原始native stderr、OS退出码/信号、core是否产生及可用位置/限制，不覆盖事故文件。不将SIGABRT或core本身解释为CUDA复用根因。

checkpoint只接受已完成事务；临时文件和部分epoch不能合并进有效结果。失败后保留最后完整checkpoint及其hash。

## 10. Acceptance criteria for reviewer

独立Reviewer读取设计、实际实现及Coder报告，确认：

- 全部控制测试结果与证据范围明确；
- fresh child、同PGID、父死亡防护与槽位预留无启动竞争漏洞；
- worker≤2、所有本任务Python≤5；
- 无自动恢复held/stale异常条目；
- 科学协议、canonical与既有完成工件未变；
- launch gate绑定新版本且失败关闭；
- 事故和恢复记录完整；
- 所有可能影响训练数学的修改不存在或另行审查。

任何无法保证的进程孤儿、错误计槽、signal窗口或恢复绕过均为blocking。通过审核后仍需主控更新冻结gate，再明确批准解除相应执行hold。

## 11. Conditions that would falsify the mitigation hypothesis

- fresh进程仍发生相同native异常：隔离未解决事故，不自动支持或否定某一底层根因。
- supervisor死亡后child继续训练：父死亡防护失败。
- 组停止后训练仍推进：进程组控制失败。
- 第6个Python被创建、GPU workers超过2或锁重叠：资源控制失败。
- 自动重取异常/held key：停止与恢复控制失败。
- 修复改变checkpoint数值、数据、选模或canonical：超出执行控制范围。

这些情况停止受影响执行、保留证据并返回Coder/Reviewer，不通过继续搜索或自动重试绕过。

## 12. 版本化设计补充：v1.1 重复TERM与清理事务保护

日期：2026-10-12。类型：execution-control-only addendum。适用背景为主控新报告的重复TERM清理失败；本补充定义修复要求，不声称已完成实现、测试或实验。与前文冲突时，以本节的TERM发送和幂等清理规则为准；未修改要求继续有效。历史`free(): invalid pointer`的native根因仍为**UNKNOWN**，本次信号处理问题不解释该native事故。

### 12.1 控制目标与唯一转发者

- peer-stop的TERM只能发送给**peer supervisor PID**，不得使用`killpg(peer_pgid, SIGTERM)`或直接TERM其child。
- 发送前核验已登记的host、worker ID、supervisor PID、PGID及Linux `/proc` startticks；要求当前PID/PGID/startticks与冻结活动身份一致，且supervisor PID=PGID。核验失败、PID消失或身份不匹配须记录未发送及原因，不向猜测PID/PGID发送。
- 尽量使用pidfd绑定被核验的活进程并发送；若使用PID信号，必须在发送前再次核验身份，任何无法关闭的PID复用风险作为Reviewer阻塞项，不虚称身份验证可消除所有竞争窗口。
- **supervisor是唯一的child INT/TERM/HUP转发者**。peer停止请求仅到supervisor；supervisor首个停止请求设置stop latch，禁止新child/claim，并仅向当前child转发一次原始停止信号。后续重复停止请求不得重复转发、改变首个停止原因或重置清理deadline。
- 同一supervisor因收到peer TERM、读取全局hold或收到operator停止请求进入停止状态时，共用同一个stop latch和child-forwarded标记，避免不同控制路径各自发送一次TERM。
- 正常INT/TERM/HUP控制入口同样采用已核验supervisor身份；控制文档不得把组级TERM作为新的常规停止入口。
- **进程组STOP/CONT保持不变**：暂停与恢复仍针对整个worker PGID，不能使用PID-only STOP。父死亡PDEATHSIG及紧急组级KILL规则不变。

### 12.2 child信号幂等与事务一致性

- child的INT/TERM/HUP handler共用持久stop latch。第一个信号先登记首个signum/停止状态，再触发至多一次受控中断（可以是一次KeyboardInterrupt）。
- 一旦停止已启动，所有后续INT/TERM/HUP，包括不同种类的停止信号，在训练栈展开、queue状态写入、manifest更新、锁释放、receipt生成及日志flush期间均不得再次抛出KeyboardInterrupt或其他中断异常。
- supervisor与child均保留第一个停止原因和退出语义；例如首个TERM对应受控退出143，后来的INT不能改为130。重复请求可留统计/事件，但signal handler不能用易失败的日志I/O破坏清理。
- child必须完成原runner的中断事务，写出身份绑定的INTERRUPTED状态与receipt；receipt、queue状态、attempt manifest及最后完整checkpoint的epoch/hash须一致。已完成checkpoint不可改写为新epoch、不得拼接部分epoch，也不因重复信号消耗新的恢复额度。
- 仅在相关事务提交后释放canonical/GPU锁；最终释放child采用的租约副本不能解除仍由supervisor持有的租约。supervisor等待child退出并核验中断记录后结束，不启动下一个child。
- 受控停止不生成VALIDATION_COMPLETE，不转换为可自动claim的状态；现有全局hold与原一次显式恢复规则继续有效。若事务本身失败，登记CLEANUP_FAILED或相应原始错误并保持hold，不能伪造一致的INTERRUPTED成功回执。
- 此规则不屏蔽SIGABRT/native故障，也不把不可捕获的SIGKILL描述为完成了中断事务。

### 12.3 30秒升级保持

清理时限仍从第一次child停止转发开始计算，最多30秒；重复信号不能延长。无响应或清理卡住时，保持原要求：**先durable写入身份绑定的失败audit、全局hold、经核验的peer supervisor TERM请求及升级事件，再向自身整个PGID发送SIGKILL**。外部控制端随后核验组内无存活进程。未完成事务只能标为不完整，不能用KILL替代INTERRUPTED receipt验收。

### 12.4 必需的实际重复信号fixture验收

新增真实Linux进程fixture，不以直接调用handler、纯mock信号或源码阅读替代。使用预留槽位启动真实supervisor和fresh child，遵守Python总数≤5/GPU workers≤2；不进行正式模型训练。

fixture使用实际child handler及实际queue/receipt/锁写入路径，在临时队列和attempt目录中登记一个活动canonical key，预先提交一个有效的完整checkpoint（可用受控CPU状态fixture），以同步屏障明确训练阶段、清理已开始及receipt提交窗口。不得仅按sleep猜测信号落点。

必须执行以下两条路径：

1. **控制转发路径**：向经PID/PGID/startticks核验的supervisor发送TERM；在其已开始停止后再次发送TERM，并覆盖全局hold同时触发的情况。观察supervisor仅转发一次停止信号、deadline不变、没有下一child。
2. **child幂等抗重复路径**：首个TERM通过supervisor转发；在child确认正在清理及准备receipt的屏障处，由测试注入器故意追加真实TERM，并追加INT/HUP验证共同latch。直接向child的信号仅作为测试故障注入，不是生产控制入口。至少让一个重复信号实际在清理处理中被child接收，不能靠所有信号在退出后发送而假称覆盖。

每条路径均须验证：

- queue状态、attempt manifest和身份绑定receipt一致为INTERRUPTED，退出原因保持首个TERM；无第二次KeyboardInterrupt破坏清理的traceback。
- checkpoint文件hash与已提交epoch不变，receipt引用该最后完整checkpoint；无虚构成功端点或半写状态。
- child与supervisor退出并被回收；从独立测试进程实际重新取得canonical、GPU fixture、worker-ID和process-slot锁，证明已释放，而非只检查lock文件存在。
- child adopted租约没有提前解锁supervisor租约；最终worker退出后两槽均可重新取得。
- 全局hold保留，未出现新invocation/child PID/claim，完成工件不重跑，interrupted key不自动恢复。
- phase日志保存第一次停止、重复请求、转发次数及最终receipt身份；记录测试的真实PID/PGID/startticks和信号时序，不将CPU fixture结果扩写为已验证CUDA/native清理时限。

另保留原无响应child的30秒durable-before-KILL测试，检查重复TERM不能延长deadline且升级记录先于组级KILL。实际测试可以使用已审查的缩短fixture deadline，但生产值必须仍为30秒并有独立断言。

### 12.5 新实施门禁

Coder需报告本补充对应的源码修改及上述实际fixture结果；独立Reviewer须检查唯一转发者、stop latch、清理/receipt原子性、身份核验和真实信号覆盖。旧v1/v2/v3实施PASS不能替代本补充的独立审核。新设计/实现/tests/review hash冻结且阻塞修正前，不更新运行许可、不解除事故hold、不启动恢复或正式训练。

## 13. 版本化设计澄清：v1.2 pidfd控制与锁内中断receipt

日期：2026-10-12。类型：execution-control-only clarification。用于关闭新发现的operator信号身份竞争和中断receipt锁外提交问题。本节优先于前文任何允许PID信号回退、或允许释放canonical/GPU锁后才提交中断receipt的表述。未修改的信号幂等、槽位、停止、恢复及30秒durable-before-KILL要求继续有效。native事故根因仍为**UNKNOWN**。本节只定义设计，不代表代码、测试或训练已经完成。

### 13.1 operator INT/TERM/HUP必须使用pidfd

- operator的INT/TERM/HUP只到目标supervisor，必须采用Linux pidfd发送，不允许`kill(pid, signum)`、shell正PID kill或先验证PID后再进行裸PID发送的回退。
- 从已登记的host/worker ID/PID/PGID/startticks取得目标身份，先打开目标pidfd，再核验当前supervisor PID、PGID和startticks与登记一致且PID=PGID；核验全过程须证明所持pidfd对应该被核验进程。最终只经该pidfd发送。PID退出或复用不得使请求转向新进程。
- 核验失败、进程已退出、pidfd不可用或权限不足均失败关闭并保存原因；不向未验证目标发送，也不降级为PID或组级TERM。跨host控制不得把远程身份应用于本机PID。
- peer-stop同样使用已验证supervisor pidfd，supervisor仍是唯一child停止信号转发者。child幂等latch保持第一次停止原因。
- **STOP/CONT仍针对整个已核验worker PGID**。pidfd约束不把暂停/恢复改成supervisor PID-only操作。组级紧急SIGKILL的既有升级协议保持不变。
- 新测试须覆盖PID复用/身份变化、目标在验证后退出、pidfd不可用三种拒绝路径；不能仅证明startticks比较函数返回正确就宣称发送无竞争。

### 13.2 receipt属于F0Runner锁内中断事务

durable INTERRUPTED receipt必须在`F0Runner`的中断事务内构造与提交，且在其`finally`释放canonical锁或GPU锁之前完成。child外层异常处理或supervisor只能读取/核验已提交receipt，不能在锁释放后补造中断receipt作为正常成功清理。

受控中断路径必须保持以下顺序与不变量：

1. 设置幂等stop latch，停止训练推进；此后重复INT/TERM/HUP不得再次中断清理。
2. 在canonical和GPU锁均仍持有时，固定canonical fullhash、attempt、child/supervisor PID/PGID/startticks/nonce、GPU owner及最后完整checkpoint的epoch/hash。不得在锁释放后重新读取可被peer改写的owner作为历史身份。
3. 在同一锁内事务中完成INTERRUPTED queue状态、attempt manifest及身份绑定receipt。receipt引用同一个已提交checkpoint，不增加epoch、不消费额外恢复额度。
4. receipt采用原子文件提交：完整构造、写临时文件并flush/fsync、原子rename，随后fsync包含目录；相关status/manifest亦完成持久化。全部提交成功后才登记事务已提交。
5. 只有在receipt已经durable且status/manifest/checkpoint一致后，才进入释放canonical/GPU锁的`finally`清理。不存在正常中断路径的“status/manifest已提交、两锁已释放、receipt尚未提交”窗口。

文件间不宣称具有数据库式全局原子性；锁与receipt提交标记定义受控中断事务边界。若中途发生进程死亡、存储失败或SIGKILL，则保留部分证据并标记不完整/hold，不能将其当作正常INTERRUPTED事务。尤其receipt构造、fsync或rename失败时，不允许伪造durable成功或锁外补交后宣布通过；按既有失败持久化与升级规则处理，后续恢复仍需显式审核。

成功轨迹的completion receipt锁内事务要求保持不变。本补充不得重写既有完成工件、改变科学checkpoint内容或自动恢复held key。

### 13.3 提交前后真实屏障验收

在v1.1真实Linux进程fixture中新增两个确定性同步屏障，使用实际F0Runner中断事务、实际receipt writer和临时canonical/GPU锁；不用正式训练，不用sleep猜测落点，不用mock锁或mock receipt替代。

**屏障A：receipt提交之前。** 让child已进入中断清理并完成身份/checkpoint快照，但尚未完成receipt原子提交；canonical和GPU锁均必须持有。测试注入器在此屏障追加真实TERM、INT、HUP，验证共同stop latch不抛第二次KeyboardInterrupt。由独立已计槽测试进程尝试非阻塞获取两锁，必须均失败；不得出现已解锁而receipt缺失的状态。屏障可选择status/manifest已有锁内更新的位置，以直接检验原问题窗口，但不得把该中间状态作为外部可消费的完整事务。

**屏障B：receipt提交完成、锁释放之前。** 完成文件及目录fsync后进入屏障，证明receipt已经durable；此时canonical和GPU锁仍必须持有。独立进程读取并核验queue状态、attempt manifest、receipt及checkpoint的canonical/attempt/epoch/hash/owner一致，并确认两锁仍不可获取。解除屏障后，允许正常finally释放锁；随后实际重新获取两锁，验证释放时receipt仍存在且一致，peer复用GPU owner不改变已提交receipt中的历史身份。

两屏障测试共同要求：

- 保留首个停止原因和受控INTERRUPTED退出语义；重复信号不得损坏receipt、status或manifest。
- checkpoint hash与最后完整epoch不变，receipt不引用临时/部分checkpoint。
- 真实进程退出并被回收，无新invocation、无下一child、无新claim；全局hold与显式恢复要求保留。
- 控制入口使用经身份核验的pidfd；直接向child重复信号仅作为v1.1故障注入。
- 对receipt提交失败做独立fixture：不得产生正常提交成功标记，不得由child外层锁外补造成功receipt，不得继续队列。
- 日志明确记录屏障前后、实际信号、锁探测、receipt提交及退出身份。CPU fixture只证明执行事务和Linux信号控制，不证明CUDA/native清理时延。

### 13.4 新事故key的冻结hold覆盖不变量

下一份冻结gate的`preserved_held_run_ids`必须包含原45个held key，加上新事故key `b3a55740aadd0e3377a9`、`cfbf6cc0e541e76cfb92`，构成**47个不重复的key**。必须核验与前一冻结45项的完整集合关系，不能仅核验计数47，也不能用其他key替换这两项。

当前gate关闭且全局hold有效，只能说明现阶段claim受阻，不能替代下一次release的逐key冻结保护。任何后续global-hold release都须绑定全部47项并由supervisor与claim入口一致执行，不得遗漏两个新事故key；gate缺项、重复项、caller/冻结集合不一致或旧hold身份漂移均失败关闭。两项保持held，不能因控制修复通过而自动恢复。若未来明确批准单key恢复，须另走可审计的原恢复规则和版本化gate修订，不隐含于全局release。

测试必须覆盖“47项全部存在但caller遗漏新key”“计数仍为47但替换其中一个新key”“新gate只保留旧45项”等拒绝情形；在拒绝时不得启动新child或claim。保留两个新事故key与前45项的来源、状态和冻结集合hash作为审核证据。

### 13.5 实施审核门

独立Reviewer必须检查operator/peer pidfd的实际身份绑定、receipt writer在F0Runner事务中的调用位置、`finally`释放顺序、文件与目录fsync、真实两屏障测试证据及全部47项冻结hold覆盖。任何裸PID回退、锁外正常中断receipt、提交失败后继续队列、hold集合缺失/漂移或未覆盖真实信号窗口均为blocking。此前PASS不批准这些修改；完成新实施审核和hash冻结前保持执行hold，不运行代码修复验证之外的正式实验或恢复训练。

## 14. DESIGN SPECIFICATION：v1.3 新SIGSEGV后的49项保护与来源更正

日期：2026-10-12。范围：execution-control-only。只设计冻结hold扩展、行政状态归类、证据来源更正和下一次fresh-search门禁；不设计算法修复、不运行Python、测试或训练。本节优先于旧47项要求：旧47项必须完整保留，再加入本轮两个canonical key；不授权恢复、重试、formal或test。

### 14.1 Research question

在fresh trajectory仍发生native SIGSEGV后，如何在不修改实验资产、不恢复checkpoint的条件下，把新事故与受控停止的peer可靠纳入永久保护，并使gate/caller/release/audit无法遗漏或替换保护项？控制正确性与native稳定性分别判断。

### 14.2 Hypothesis being tested与事实边界

v1.2独立实施审核PASS报告为`experiments/reports/stage_F0_signal_cleanup_review_v3.md`，SHA-256为`a5fe1f766ae969bc9d307e135d73afe1d39cefcb2a6cbf94a1e8133efcc42b96`。该PASS不证明native故障被消除，不批准本次新增实施或hold release。

本轮Runner原始审计为`experiments/stage_F0_outputs/runs/execution_control/postabnormal_sigsegv_audit_v4_fresh_search.json`，SHA-256为`a50162f3016841257e8c098cc0040c9ff2be4edb30073c4c2fb42f8ed416f839`。当前active global hold原始快照SHA-256为`e9461efcc99e49bef4b813a3f3cb1c67720bdcd172470df2a73a9e7caca400b9`。

- `7fd2006bdc8f1817a6f7`：worker1 invocation2，fresh child，MMoE/MDMT、seed2019、batch4096；child exit−11/SIGSEGV，queue仍为stale RUNNING，无receipt，最后完整checkpoint为epoch13。
- `1f04c7fb623d18f940ac`：peer worker0，seed2018、batch40960；pidfd TERM后exit143，锁内durable INTERRUPTED receipt，最后完整checkpoint为epoch17。
- 原审计登记无剩余F0进程、worker/GPU锁空闲，未retry/recover/formal/test；下一release必须重新核验，不能把历史核验当实时保证。

**SIGSEGV root cause UNKNOWN。** 无Python traceback和receipt不识别崩溃算子或原因。fresh进程发生故障说明进程隔离不是native稳定性保证；不能据此断言CUDA、数据、batch、模型或硬件是根因，也不能把peer的INTERRUPTED认定为独立native故障。

### 14.3 Variables changed

- 独立冻结保护集合从47扩到49。
- 两个新queue状态由RUNNING/INTERRUPTED行政归类为RECOVERY_BLOCKED。
- 新增append-only provenance correction、状态迁移audit、版本化gate/release/prelaunch绑定及拒绝用例。

不以此修复训练数值或native库；根因定位如需新实验，另走DESIGN→IMPLEMENT→REVIEW→RUN→ANALYZE。

### 14.4 Variables held constant与资产不可变

科学canonical身份、数据、算法、optimizer、随机性、seed、batch、epoch预算、选模和test门不变。原v4审计、native stderr/phase/failure日志、receipt、attempt manifest、metrics、checkpoint及52项完成资产保持字节不变。

允许修改的既有对象仅为明确登记的行政queue-status和当前global-hold/gate；先保存其原始字节与hash，再版本化更新并追加迁移记录。不得改attempt manifest的RUNNING/INTERRUPTED原始事实使其伪装为RECOVERY_BLOCKED，更不得替SIGSEGV补造receipt。新的queue RECOVERY_BLOCKED是外层执行资格，原receipt/manifest保留其实际执行状态；两者通过迁移记录关联，不能强求原资产重写为相同状态。

### 14.5 Exact set / algorithmic definition

令`H47`为v1.2已独立冻结且经旧来源核验的47项集合，不能由caller、新gate或release提供的列表重新定义。新集合唯一为：

```text
H49 = H47 ∪ {7fd2006bdc8f1817a6f7, 1f04c7fb623d18f940ac}
```

要求`|H47|=47`、两个新项原不在H47、`|H49|=49`、无重复。采用排序字符串数组的UTF-8紧凑JSON（无空白、无尾换行）计算集合SHA-256。本次只读shell/Perl重建得到：

- H47：`ccbf64945396b542ee2746f4bab37d7d2fcdaea49b00817f096facec45f7f8bf`。
- H49：`60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e`。

实现以经过审查的固定49项集合和这些独立来源校验为准，不信任caller声称“49项已齐”。gate、caller、release、迁移audit、prelaunch audit均须无重复且与H49逐项相等；count或list hash单独通过不能替代来源及actual status核验。supervisor启动前、runner启动前和串行claim边界均执行保护，不能只在人工release时检查一次。

### 14.6 Exact affected modules/files与来源更正

实施涉及已有`stage_F0_worker_control.py`的固定集合及gate/hold/release验证、`stage_F0_queue.py`的fresh-only资格与行政迁移工具、supervisor/runner对应prelaunch调用、focused tests及版本化控制工件。只能改执行控制；Coder报告具体文件和hash。

必须新增append-only更正工件，建议路径为`experiments/stage_F0_outputs/runs/execution_control/postabnormal_sigsegv_audit_v4_provenance_correction.json`。它绑定原v4审计路径/hash，不覆盖原文件，逐项保留“原值、实测值、原始资产路径/hash、核验方法、核验者/时间、影响范围”：

1. 原v4记录的peer receipt SHA不准确。实际receipt为`worker_logs/trajectory_worker0_2_c87c8e65bc7e4b73bed880465ff2dffb.receipt.json`，实测SHA-256为`3a3b86cd54b0685ec09506a32706355a26c3465ed41d872c06e641d94bcbacf7`。
2. 原v4 peer GPU UUID为`2a59fb49-6bef-5320-74ab-bffb24d056908`；真实receipt的锁内GPU/canonical owner和queue状态为`2a59fb49-6bef-5320-74ab-ffb24d056908`。更正应来自这些历史锁内快照，不能读取现已释放且可重用的GPU lock当前owner代替历史身份。

本轮原始queue状态hash：SIGSEGV项`74cbabc600e19e4e977c3b38d3068419c72c3be0b678cea5857a1867ec66dbbe`；peer项`61921430b4975ba67720efd19fcb24a6ae4debb5f232f56d536e0e568ad41478`。原attempt manifest hash分别为`e304b73005ff339b63b0b2ad70c4830b7df9550bce2acc2710ff250690ee0197`、`2e85d70d0d97b144fd26cba06be61f9ce3d86bb3879839adde9c866c6377f1d6`。

最后完整checkpoint：SIGSEGV的`resume_epoch_013.pt` SHA为`2e669175eb2752ff27cc73202687167e98799d36e2495f25e76d4543f6b0c79f`；peer的`resume_epoch_017.pt` SHA为`8843afe2a382b47b7a920986ffbfc0cf996a73b0ab8822bfd138760b2517597c`。

更正工件不得默默消除冲突或更改实验身份。任何后续绑定必须同时验证原v4审计、更正工件及其实际资产hash/身份；不允许只信caller或复制审计中错误字段。更正缺失、hash不符或冲突未解释均NO-GO。

### 14.7 Implementation requirements：先持久保护，再考虑release

1. 保持global hold active且新gate未授权。验证原47来源、原v4与更正来源、两项canonical fullhash/attempt/nonce、durable supervisor failure及本次global hold事件；核验进程已退出、worker/GPU锁空闲。
2. 在既有execution-state序列化锁及相应canonical锁保护下，将两项行政queue status置为RECOVERY_BLOCKED。SIGSEGV归类依据是durable failure/global hold和进程死亡证据，不是“成功中断receipt”；无receipt保持无receipt。
3. peer的原INTERRUPTED receipt、manifest、epoch17/checkpoint hash及锁内事务身份一致性保持。迁移记录另存original_status、receipt_status/hash、reason、事故来源、before/after状态hash、未授权retry/recovery。
4. 不加载checkpoint继续计算、不调用resume、不创建attempt-01、不增加recovery_count、不改resume选择或last checkpoint。SIGSEGV旧RUNNING状态的行政修复不得借用自动stale recovery路径。
5. 原global hold字节/hash保存后，当前hold追加本事故与状态归类事件，明确top-level `preserved_held_run_ids=H49`及集合hash；fsync文件、rename及父目录。源hold旧hash用于审计，release须绑定更新后的实际active-hold hash，不能继续引用旧hash解锁。
6. 两项状态与hold全部持久后，从磁盘重读核验全部49项均RECOVERY_BLOCKED；保存迁移audit。跨文件不宣称数据库原子性，任何部分迁移/掉电/写入失败都保持active hold，不释放队列；重复行政调用只能在相同已核验事实下幂等完成，不能重复恢复或修改资产。

队列固定288项，纯行政迁移预期：原`52 COMPLETE + 47 BLOCKED + 187 UNCLAIMED + 1 RUNNING + 1 INTERRUPTED`，变为`52 VALIDATION_COMPLETE + 49 RECOVERY_BLOCKED + 187 UNCLAIMED`，RUNNING/INTERRUPTED均0。52项完成key集合与资产必须不变，不能仅验证计数；任何额外状态/key变化阻塞。187 fresh UNCLAIMED是本次release最大可用集合，不授权补足held配置或重搜替代seed。

### 14.8 Ablations / failure-injection fixtures required

仅短生命周期CPU/Linux控制fixture，遵守总Python≤5、GPU workers≤2，不运行正式训练。必须覆盖实际生产验证函数和fresh claim入口：

- gate/caller/release/audit分别遗漏任一新项、仍用旧47、同count替换新项、重复项、改变旧47任一项，全部拒绝；即使四份caller文档共同伪造同一列表，也不能改独立固定集合。
- H49正确但任一实际状态仍RUNNING/INTERRUPTED/缺失、旧47不再BLOCKED、active hold hash或来源漂移，拒绝child启动/claim/release。
- 更正缺失、receipt坏hash、UUID冲突未解释、checkpoint/manifest变化、原v4被覆盖、错误/旧hold release，全部NO-GO。
- 在两个queue迁移之间、status提交后hold提交前、fsync/rename失败处注入异常：active hold保留，无resume/新attempt/下一child；重新审核的行政迁移可幂等收敛，不隐含retry。
- fresh-only入口拒绝持有checkpoint/attempt/历史claim但被改名为UNCLAIMED的key，以及全部49held、52completed、其他stale状态；不能把删status文件当fresh。
- 在活动fixture中真实native SIGSEGV注入或等价不可捕获signal退出，验证durable failure/hold、无伪造receipt、peer supervisor pidfd TERM和停止后无下一child。该注入只验证控制行为，不复现或识别真实SIGSEGV原因。
- 保留v1.2实际重复TERM/INT/HUP及receipt提交前后屏障测试；同时覆盖失败hold与peer停止重复请求，首次原因/锁内receipt/30秒deadline不变。PDEATHSIG、组STOP/CONT、预留槽和durable-before-KILL不得回退。

fixture允许临时queue状态修改；生产checkpoint、receipt、manifest、metrics及52完成资产不能作为写入测试目标。

### 14.9 Expected logging / hash requirements

新增来源更正、行政迁移、exact-set验证、asset inventory、pending gate、独立review、prelaunch、显式release工件。各自保存schema/version、路径/hash、原始来源、old/new状态、H47/H49集合与hash、授权范围及禁止事项。

hash依赖顺序避免循环：冻结设计/实现/test及来源更正/迁移证据→独立review→绑定review和当前hold的新gate→prelaunch→主控显式fresh-only release。release不能把自己的hash循环写入已冻结gate；另保存无循环的release审计并核验实际引用图。每阶段从实际文件重算hash，不沿用v1.2旧code/control/gate摘要；科学canonical hash与实际执行amendment hash分别保留。

### 14.10 Acceptance criteria / prelaunch GO–NO-GO

下一次运行必须同时满足：

- 新设计获主控接受；实施与focused fixtures经新的独立Reviewer PASS或全部blocking纠正后的PASS WITH FIXES。旧v1.2 PASS不审批新实现。
- independent fixed H49、所有49项RECOVERY_BLOCKED、52完成资产不变、原v4及corrected provenance、状态迁移、active hold实际hash完整核验；无法解释的冲突为blocking。
- 当前无遗留进程/活动canonical claim，worker/GPU锁可用；确认全部fresh资格、磁盘/设备门及两worker/五Python限制。状态只由历史审计证明时不足以GO。
- gate/caller/release/audit全部exact H49；主控另给identity-bound显式release，绑定当前hold、设计/实现/review/来源/队列hash，只允许F0-B1-ML search的fresh UNCLAIMED。retry/recovery/formal/test授权均false。
- supervisor在child启动之前、runner在数据/CUDA/claim之前、claim序列化边界均重新验证门与hold来源。受保护项、完成项或有历史attempt项不能进入fresh执行；任一检查失败不启动等待Python、不自动绕过。

满足条件也只代表受保护的fresh-search执行资格，不代表native故障已修复；继续fresh任务是主控单独的有界科研决策。本设计自身不解除global hold。若原固定搜索单位因held轨迹缺失而无法公平选配置，该单位保持不完整，不用其他seed/config替代，不进入formal。

### 14.11 Conditions that falsify the control hypothesis

漏掉/替换任何H49成员、未全部BLOCKED就release、来源更正未核验、行政迁移触发checkpoint恢复、资产变化、非fresh被claim、异常后下一child、重复信号破坏锁内receipt、计槽或信号语义回退，均说明v1.3控制不满足设计，停止受影响执行并返回Coder/Reviewer。

fresh运行再次native崩溃不等于上述保护逻辑必然失败，也不否定算法；保留事故证据并执行停止门。控制PASS、peer正常清理、有限checkpoint或若干成功轨迹都不能把SIGSEGV root cause由UNKNOWN改为已解决。
