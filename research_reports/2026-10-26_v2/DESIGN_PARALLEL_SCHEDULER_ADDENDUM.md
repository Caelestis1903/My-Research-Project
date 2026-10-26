# DESIGN SPECIFICATION — 驻留阶段并发调度补充

日期：2026-10-26。状态：DESIGN_COMPLETE，待coder实现和独立reviewer PASS。仅为遵守start.md第12条GPU并发偏好；不分析实验指标，不改变算法。本文件不修改已冻结的DESIGN_FOUR_SEED_RECOVERY_ADDENDUM.md、有效配置、prefix源码或其绑定hash。

## 1. 审计判断与适用边界

当前Campaign.prefix能根据单卡显存/功耗及主机RAM准入并发；Campaign.phase逐worker发送后立即recv，使S1/S2/S3仅有一个训练worker活跃。它是实际存在的吞吐限制。八个prefix已有或即将有RAM状态，不能为调度改进再次重启、重放或丢弃模型。

允许通过现有 `reload-readers` 正常RPC的**controller第一次导入readers**，执行一个固定、独立审查授权的scheduler安装入口。仅安装已审查函数到当前Campaign类的 `phase` 属性，安装后原instance会使用该方法。该方案不使用gdb/ptrace、任意eval/exec/pickle、任意模块路径或任意属性替换，属于明确授权且可审查的正常Python模块导入与固定类方法更新；原设计禁止的是任意/未经审查的注入。本补充明确授权的范围仅此一个方法，不授权升级controller配置或算法代码。

前置条件固定：工作区 `/home/xzf/Code/phase0-3`，service PID **2088990**，artifact `artifacts/mdmt_rcf_20261026/v2_four_seed`，有效config hash **cefb6d934c774bd78a438e4fba5c560e934f4c56ab09fbb3dfe389f4f4842fd6**。S0继续按原代码完成；只在没有pending命令、prefix完成后的正常RPC边界安装。若parent已导入readers、PID/config/已加载方法身份不符、存在pending训练或无法验证身份，则安装FAIL/CLOSED；不为吞吐重启，不尝试第二条注入路线，保留已存在训练和模型。

## 2. 固定安装入口

新增 `experiments/mdmt_rcf_20261026/phase_scheduler.py`，包含普通Python定义的并发phase函数及明确类型/固定目标的安装函数。readers.py只添加一个固定模块入口调用，所有数值函数AST保持原样。不得reload整个campaign模块或替换 `_worker`、prefix、训练/reader函数；不得遍历gc寻找对象，不得接触模型/Adam/RNG。

安装函数必须：

- 在非指定PID（尤其各spawn worker）返回NO_OP，不访问或更新其训练状态。CLI客户端导入也不安装。
- 读取固定artifact根的 `PHASE_SCHEDULER_AUTHORIZATION.json`，要求独立review PASS/PASS WITH FIXES且无blocking、精确PID/config/root、精确新模块/reader文件hash、旧phase和新phase代码指纹及范围仅 `Campaign.phase`。先验证后赋值；禁止从token执行代码或选择路径/模块/属性。
- 验证已加载campaign.Campaign.phase是预期旧实现，且当前服务身份文件匹配PID/config。安装逻辑不更改self.config、self.identity、self.workers/pipe、summaries、decisions或任何模型。不得借安装把冻结配置换hash。
- 在唯一赋值前后记录固定目标/旧新函数指纹、PID、时间、源文件hash和授权hash到 `PHASE_SCHEDULER_INSTALL_RECEIPT.json`。重复安装相同函数可以返回ALREADY_INSTALLED，其他替换一律拒绝。
- 现有全源码review token和专用reader reload token在独立审查完成后刷新为实际文件hash。现有worker reader reload的前后模型hash检查照常执行；新hook在worker是NO_OP，原数值reader行为不变。

Controller正常RPC的导入机会是一次性条件；不依靠修改已加载模块的内存、debugger或外部进程接口。若fresh import条件失败，记录原因，停止本调度升级，不阻塞原科学矩阵继续完成。

## 3. 并发phase合同

函数签名保持 `phase(self, phase, *, methods_by_dataset=None)`，仅接受S1/S2/S3；参数和每worker消息内容与原方法逐字等价。开发种子集合、S3已登记methods/decision、训练本体和四种子gate均不变。阶段之间仍严格barrier，不能并发S1与S2、不能在四seed完整前决策。此安装不改变locked训练、export、bootstrap、freeze或finalize接口。

执行顺序：

1. `self.require_review()`；检查全部worker存活且没有其他pending，先按原正常 `release-data` RPC释放纯数据映射页。由原worker字典顺序建立该阶段的key队列；S3没有methods的dataset照原规则跳过。
2. 每个resident worker已绑定设备，不创建新worker、不迁移模型/optimizer。每轮为每张实际使用GPU重新查询其显存和功耗，只有二者均<85%才派发该卡一个新的训练命令；没有可靠测量时不视为空闲。GPU1外部VLLM不可终止、修改或挪动。
3. 保留既有8GiB主机RAM余量。对新命令及所有pending命令预留尚未体现到RSS的工作集，不能只看已发生的RSS。可复用当前prefix准入的保守dataset工作集/实际prefix峰值，减去该resident worker当前RSS后仅预留新增部分；阶段实际RSS峰值可向上修正估计。预算数据必须来自资源观测/原冻结资源配置，不从性能指标推导。若主机内存不允许则等待已有任务完成，不改batch、不取消种子。
4. `_send(key, phase, ...)` 后记入pending集合，继续依据新一轮GPU/主机采样派发其他eligible worker；对pending连接用poll收取任何已完成结果，每个key只recv一次。尚有训练时不阻塞在固定队首recv而遗漏后续完成者。单worker始终最多一个命令。
5. 每个完成结果照原 `_remember(key,result)` 绑定到自己的dataset/seed，及时写阶段进度和资源准入收据；返回results按原key队列顺序排列，不按完成时间改变科学清单。全部完成后才写COMPLETE。失败结果/失联worker不得伪装完成，保留失败事实及其余pending句柄，不在异常时擅自重训/关worker。
6. 仅在没有新进展时短轮询等待；持续写可读进度，不形成长时间盲等。允许GPU读数仍低于85%时追加同卡训练；不得增加与当前阶段无关的实验。

所有随机流在不同worker内以各自seed/stage/row-plan生成，controller不生成训练随机数；新的完成顺序不能改变任何worker消息参数、本地计算、选点或指标。训练墙时会受到共享资源竞争，成本表记录真实并发和阶段总墙时，不将并发单run墙时冒充独占设备的标准化速度。端到端推理benchmark仍在训练阶段barrier完成后按原隔离流程执行，不由此并发。

## 4. 最小验收与停止条件

Coder仅拥有新 `phase_scheduler.py`、readers.py末尾固定安装hook及其实现报告。禁止修改已完成prefix、原四seed设计或配置。Independent reviewer必须与coder不同，审读本设计、实际代码与coder report。

必要验收：用轻量mock worker/pipe验证至少两个eligible同卡命令会在首个recv前派发；GPU内存或功耗任一≥85%均不派发；主机预算不足时等待；乱序完成保持各key摘要/最终返回顺序；S3空子集跳过且消息methods/decision与旧实现一致；失联/失败不标COMPLETE。数值reader函数AST与审查前完全一致；父PID专用安装在worker/client NO_OP；错误授权/hash/旧方法身份拒绝；仅一个类方法变更，config/worker句柄与模型/RNG不变。mock只是调度验收，不生成或分析科研效果数据。

独立PASS后父编排者通过正常reload-readers安装，收据确认service PID未变、八个prefix仍在RAM、worker模型hash前后相同。之后自然S1/S2运行日志证明多个eligible worker实际pending/活跃；按既有旧seed重放hash与Q数值一致性检查验证训练未变，不额外展开测试矩阵。任何安装前置条件失败均放弃调度升级、继续原矩阵，严禁为纯吞吐重复八个prefix。
