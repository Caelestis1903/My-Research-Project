# PRIMARY SELF_REVIEW — Stage2D

依据design_specification.md、实际worker/queue及implementation_report.md。遵守最新start.md的主代理职责约定，independent_review=false；不伪造独立PASS，也不再把职责指令冲突作为全局停止理由。

核对损失为`mean_B[(alpha_d/q_d)*mean_task BCE]`，没有按出现域重归一化。Natural q=p_train、Balanced q=1/3，两者均域内均匀有放回；所有alpha/q支持正且初始模型相同。每epoch实际N=2006132、49main step、1arch step，不把重曝光当独立新数据。main/arch均使用相同臂定义，A仍保持原每epoch一batch更新，不把main/arch同时干预解释为C单因果。

代码没有继承/篡改S1的sampler禁用协议；使用新study身份显式允许本轮规划的q因素。旧F0受管入口/hold不调用不改。test完全在50epoch和validation选择冻结之后，任何common/selected/final test结果不进入选点。

同q配对流与初始化检查、真实训练backward/Adam step、工程端到端state/预测保存读回已检查。记录时序问题已修正：test记录该checkpoint的实际step；最终progress先写再hash。峰值校准按正式40960而非小smoke batch。

Critical：最终source的CPU完整流程或CUDA实际batch失败均阻止正式训练，先修正。Relevant：有放回新协议不等于旧S1 shuffle；A/m/v/step调度与q耦合不支持单分量归因；均在设计/报告中限定。Irrelevant：格式/非当前模块边界不追加测试。

通过最终source对应的engineering_verification.json与cuda_profile.json后，接受Stage2D正式16run执行。审查判定的机器字段在review_verdict.json中绑定实际worker SHA，训练前保存。Stage4/4B历史依赖继续分别处理，不由本Stage2D擅自关闭。
