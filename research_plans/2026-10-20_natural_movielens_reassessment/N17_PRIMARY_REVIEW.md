# N17 主代理自审

**PASS；Blocking issues 0。** 按 start.md 第11条，主代理审阅 N17_N18_DESIGN_ADDENDUM、结果前精度补充、六个新增wrapper、原n17_ml/capture/train/common/quota/geometry/local/performance源码以及实现报告。本次是主代理自审，不是独立审查。

原 fixed_model、采样、importance-loss、Adam与路由detach复用。fit仅注入注册曝光预算，普通完整800167行对应20步/epoch，匹配187500为5步，尾批不丢。quota每实际batch重算B*p/n；原row-ID/joint-label selection固定不依赖优化seed。已有同batch单步身份及sampler/quota检查为必要证据，不作科研结论。

72基础配置与额外LR/H200分支完整；pilot2030/2031只作调参，正式2022–25。所有pilot完成后按冻结响应规则选共同/独立LR，不取消四格。H200仅train-J触发，对两q四seed整体延长并保留H100。每epoch模型与完整native状态捕获沿原实现；只读gradient/native moment probes不改实际step。

新结果前补全N3精度门槛：发展集配对C三比较MCSE或未定义触发，双bank最多64、追加流互斥，无方向/显著性门槛。完整N2 512draw、N3全部预定子块不删除。训练test延期，必须所有required/triggered关闭后由主代理冻结全模型SHA，之后零优化读数；selected/final/AULC分别呈现。

元数据已改 independent_review=False，训练/诊断入口核对本门槛的source hashes。中断attempt保留并以同seed全预算重跑；H200是已注册连续训练分支。V2/core运行依赖没有更改。

批准64个可执行初始训练配置。8quota及额外LR等待pilot规则冻结后批准相应实际配置；H200等待平台gate。诊断manifest在commonLR冻结后机械生成。AO04/N18不在此放行范围。
