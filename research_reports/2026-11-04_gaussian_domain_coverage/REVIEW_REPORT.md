# 独立实现审查

结论：**PASS WITH FIXES**。所有阻塞问题已修复并复核，允许进入原batch GPU smoke及全部10条正式训练。

审查日期：2026-11-04。reviewer与coder为不同agent。审查对象为本目录的DESIGN_SPECIFICATION、CODER_IMPLEMENTATION_REPORT，以及新实验包和实际调用的原Gaussian、模型、指标、选点、RAM guardian及资源调度代码。

## 一、公式和训练合同

| 项目 | 独立核查结论 |
|---|---|
| 覆盖范围 | 仅G-D0=[0]和G-D02=[0,2]新增训练；d1不生成增强视图；10job、2022–2026五seed、每条50epoch。 |
| 单域兼容 | 单域直接调用未修改的GA31-R3 source_losses；d0/d2标量、梯度、噪声与源函数精确一致。 |
| 双域预算 | 一次真实forward，分别按域生成视图；两个任务逐元素BCE差值求和，统一除以实际2B；无域均值等权、额外重采样或重复原始权重。 |
| Gaussian与随机流 | 原view_key、SHA256局部种子、逐字段torch.randn及RMS尺度detach保留；D0单/双域和D2源/双域对应噪声一致，路由字段不扰动。 |
| 优化与EMA | fixed1 M3oE、普通Adam lr=.01/wd=1e-5/betas=.9,.999/eps=1e-8，batch40960、FP32、TF32关闭；每batch一次backward、Adam和rho=.995固定EMA。 |
| 初始化和loader | 源setup/model factory，1614254参数；独立loader种子seed+11000，完整自然train每epoch49步，50epoch2450步。 |
| RAM状态 | model/EMA/Adam/loader/Python/NumPy/CPU/CUDA RNG及选点进入RAM IPC；export/restore继续训练一致；无磁盘模型或checkpoint写入路径。 |
| 选点和评价 | 六格validation AUC严格最大，精确并列最早；epoch1–50；完整50epoch后冻结selected/E10/E50及model hash，再评价同一状态的V/Test。 |
| 资源派发 | 每张GPU独立判断显存及功耗均低于85%，并检查1.2倍峰值余量和RAM预留；保留已有用户进程，不设每卡一个worker上限。 |
| 导出裁定 | 逐seed六格、Macro/Traffic/Pooled AUC及未缩放LL；样本SD与df4配对t95%；.003强算法/.002候选算法，六格-.001保护和LL分别裁定。 |

## 二、数据与历史复用

独立读取准备记录核对：train=2006132、validation/test各250767；d0/tab0=172782、d1/tab1=1743898、d2/tab4=89452。原CSV、九个缓存文件、六数组、row_ids、编码器/缩放器和划分身份按来源哈希验证；训练bundle不载入test数组。

150个复用端点为5个配置×5seed×selected/E10/E50×V/Test，分别对应G-D2、固定EMA、M3oE、HyperGate、AutoIFS。原表及来源文件SHA、同epoch/model hash配对完整；G-D2 selected epoch为28,22,36,47,36，固定EMA为33,50,50,50,47。发表模型使用11月4日统一协议结果。

核对原GA31文件SHA256：

- objectives.py：28eaa19444cb650ca46d28c4e22829771612920add6c27188b828d934ea5479c
- views.py：cc83584fa43a57ffd8c9c176e14a668a5dc835e902086e378a8db15e2a75f44b
- protocol.py：1f6e345cf50d417285fde8100c1c99bf4ef3a71bddf4e256c3b3640186247ccd

## 三、发现及修订

1. **Critical（运行恢复，已修复）**：旧实现将guardian状态丢失的job标为STATE_LOST_REPLAY，却保留旧提交链并重用launch_1。标量fixture实际复现旧epoch1与重放1–50混合后，archive拒绝完整窗口。修订后共享ledger保持只追加；archive/summarize共同使用每job最新commit1开始的连续链；旧worker退出后将其旧标量目录移到replay_history，保留旧epoch、launch、失败记录和成本。新正式链50项完整，旧文件不被覆盖，同一seed仍只计一次。
2. **Relevant（完成恢复，已修复）**：DONE写入后、RAM release前worker退出会留下guardian状态并阻塞shutdown。reconcile现在先验证DONE文件哈希，再释放残留RAM，保留已封存结果。定向fixture通过。

修订限定在新run.py、reporting.py及两项标量恢复检查；原Gaussian及旧实验代码未修改。未改变数据、算法配置、种子、训练预算或统计规则。

## 四、独立执行证据与RUN交接

项目环境Python：/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python。

- 修订前独立pytest：12 passed in 1.98s；已用临时纯标量fixture复现上述恢复阻塞。
- 修订后独立pytest：**14 passed in 2.26s**。
- 修订后compileall：通过。
- 独立读取最终prepare：源码hash一致、10job/500epoch、五seed和150个复用端点身份完整。

最终审查源码hash：`c8c76da2cb644559ea1cdf69d56e75fef00ae1fb70e214529708a023884b096d`。receipt绑定该hash。RUN阶段先执行G-D0/G-D02的batch40960真实GPU smoke及RAM状态往返，再执行全部50epoch正式队列；smoke不参与性能筛选。

本审查不包含训练结果分析；GPU smoke和正式训练的实测记录由runner写入。
