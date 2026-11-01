# 05｜Codex完整执行提示词

你负责在仓库 Caelestis1903/My-Research-Project 中实现并执行本研究合同。研究目标是检验“参数轨迹平均”和“增强差分来源分离双时间尺度动量”能否在MDMT推荐中提升validation-selected macro AUC。你不负责重新设计算法，不得扩展研究范围，不得把候选选择留给临场发挥。

## A. 必读文件与来源

先完整阅读：
1. research_plans/26-11-1/README.md
2. research_plans/26-11-1/01_证据审查与研究问题.md
3. research_plans/26-11-1/02_多候选算法探索与替代设计.md
4. research_plans/26-11-1/03_优先方案算法与实现规范.md
5. research_plans/26-11-1/04_实验矩阵与研究决策.md

实验来源：
- GA31-R3报告：research_reports/2026-10-31_v2/REPORT.md
- GA31-R3规格：research_reports/2026-10-31_v2/DESIGN_SPECIFICATION.md
- 实际代码：experiments/gradient_aug_20261031_v2/
- 历史Polyak证据：PROJECT_ARCHITECTURE.md §0/§2.3；research_plans/26-10-31/01_历史证据与研究问题.md §3.5。

先记录main当前HEAD、来源分支和代码SHA。GA31-R3来源发布分支为26-11-01，实验底座固定于ee6e81fdfd37284e6447e8149cb93d04736dba02。不得把旧计划当作实际实现；必须核对实际代码。

## B. 研究范围和禁止事项

只允许研究：
- 普通Adam基线；
- 参数shadow EMA/Polyak，仅用于评价；
- Gaussian单视图增强；
- 增强差分来源分离双时间尺度一阶矩（SAD-Momentum）；
- 只有第一阶段触发时，才实现可靠性调节差分注入（RDI）。

禁止：
- 差异化共享、多原型、跨域读取、域级控制器；
- 恢复SAGU/SAOS旧排序或默认MASK为最终算法；
- 重新加入ANTI2作为必须方法；
- 任务/域独立动量、课程学习、梯度冲突投影；
- 未经合同授权的二阶矩新状态、递归控制变量、超参搜索；
- 使用test标签调参、选seed或决定是否运行P2。

## C. 实现目录和代码约束

新代码放 experiments/gradient_aug_momentum_20261101/。
新报告放 research_reports/2026-11-01_momentum/。
新运行工件放 artifacts/gradient_aug_momentum_20261101/attempt_<timestamp>/。
不要修改GA31-R3旧包、历史报告或其他分支正在进行的工作。

复用GA31-R3 fixed1模型、数据读取、指标、自然分布、split、encoder/scaler、RNG隔离和训练协议。优先通过明确命名空间复用，不得用顶层同名模块污染import。记录数据raw/split/encoder/scaler哈希、模型参数数量、state_dict键、row_id顺序哈希、PyTorch/CUDA/Python版本、设备信息和代码commit。

基线：
- MovieLens：L2-ML，loss scale=1、coupled wd=1e-4；
- KuaiRand：B-KR，loss scale=1、coupled wd=1e-5；
- Adam lr=.01、betas=(.9,.999)、eps=1e-8；
- 50 epoch、batch=40960、FP32；
- seeds 2022–2026；
- 自然数据分布和GA31-R3固定split。
不要把历史Polyak实验的旧人口或旧loss scale带进当前基线。

## D. 必须实现的算法

### D1. P0 shadow EMA

φ0=θ0；每次在线Adam.step后：
φ_t=ρφ_(t−1)+(1−ρ)θ_t。
默认ρ=.995。影子参数只用于验证和保存，不反馈训练。LayerNorm affine参数参与EMA；若有running buffers，默认复制在线buffer而非直接EMA。在线和shadow分别按validation六格macro AUC选点，平局最早、E0排除。

### D2. P1 SAD-Momentum

同一θ和batch计算全batch真实梯度g_real；另分别计算GA31指定目标域上的aug与clean loss梯度，二者均使用原始B×T分母，δ=g_aug_target−g_real_target。绝不能用目标域aug梯度减去全batch真实梯度。
m_real=β1*m_real+(1−β1)*(g_real+wd*θ)
q_delta=β_delta*q_delta+(1−β_delta)*δ
v=β2*v+(1−β2)*(g_real+λδ+wd*θ)^2
mhat=m_real/(1−β1^t)
qhat=q_delta/(1−β_delta^t)
vhat=v/(1−β2^t)
θ←θ−lr*(mhat+λ*qhat)/(sqrt(vhat)+eps)

默认β1=.9、β2=.999、β_delta=.99、λ=.5、Gaussian κ=.1。唯一主敏感性β_delta=.90。λ=0退化普通Adam；β_delta=β1时一阶矩必须与标准混合梯度Adam精确等价；β_delta=0时退化普通增强Adam。weight decay按GA31-R3的coupled语义执行；δ不含decay，因为同状态差分中decay抵消。q_delta、m_real、v全部detach，不保留跨step计算图。每batch只允许一次参数更新。

### D3. P2 RDI（条件执行）

只有阶段1触发才实现。K=2独立视图，估计差分均值、视图方差及有界增益；按03文件的公式和04的进入条件执行。与固定λ=.25/.5/.75对照。若不触发，不实现、不训练，并在报告中写明未触发而不是阴性。

## E. 组件开关

至少实现：
- use_shadow_ema
- use_augmented_view
- use_delta_momentum
- use_reliability_gain
- beta_delta / rho_shadow / lambda_aug / noise_scale

关闭P0不得创建shadow或额外验证。关闭增强不得抽样view或推进view RNG。关闭SAD或β_delta=0时必须与普通增强Adam对应。use_reliability_gain仅允许K=2，非法组合应报错。

必要训练臂：
A0：无增强普通Adam + shadow
A1：Gaussian K=1普通增强Adam（β_delta=0）+ shadow
A2：Gaussian K=1 SAD β_delta=.90 + shadow
A3：Gaussian K=1 SAD β_delta=.99 + shadow

## F. 实现核验与阶段0

先做CPU FP32/FP64小张量代数测试，再做真实模型batch检查。阶段0每数据集seed2022，最多1 epoch。核验：
1. fixed1模型预测、loss、梯度和一次Adam更新与GA31-R3一致；
2. λ=0与普通Adam单步一致；
3. β_delta=β1与标准混合梯度Adam单步一致；
4. β_delta=0与普通增强Adam单步一致；
5. None/unused参数语义、coupled decay、bias correction、step计数一致；
6. shadow开启不改变online参数、Adam状态、RNG和loader顺序；
7. checkpoint恢复后shadow、q_delta、Adam、RNG和后续轨迹一致；
8. view RNG与训练loader/model RNG隔离。

验收失败先修复实现并重新核验。不得以放宽容差掩盖公式或状态错误。阶段0通过后冻结代码commit和resolved config。

## G. 阶段1正式矩阵

两数据集×五seed×四臂=40条正式训练轨迹；每条50epoch，共2000 epoch。A0–A3全部维护shadow，便于在线/平均模型分离比较。A0在线与shadow是P0主比较；A1是增强普通Adam强对照；A2/A3与A1比较差分记忆增量。

每epoch完整validation，online与shadow分别选点。保存selected、E10、E50和必要的optimizer/shadow/q_delta/RNG快照。不能把shadow当成单独训练run；它沿同一online轨迹更新。

主诊断仅两项：
1. q_delta范数、m_real范数、增强注入比、q_delta与当前delta的方向关系；判断SAD是否真实改变更新方向。
2. online-selected与shadow-selected AUC、选点epoch和LogLoss差；判断参数轨迹平均的独立作用。

额外forward/backward、optimizer.step、wall time、显存和额外视图曝光必须据实记录。不得将额外shadow validation算作新训练。

## H. 决策门

主指标为selected test六格macro AUC，validation用于开发选点。每数据集seed配对；报告mean、sample SD、配对区间、5个seed逐项差值、六格/域/任务/pooled/traffic/worst和LogLoss。

继续门：平均ΔAUC≥+.001且至少4/5 seed正向。置信区间跨0不自动判定等效；未过门也不能无限追加实验。

P0：shadow在当前协议是否超过同轨迹online。若无selected收益，停止调ρ，不将历史旧协议结果当作失败或当前成功。
P1：SAD必须同时优于A0和A1，才支持差分记忆的独有增量。只优于A0不优于A1，只能说明增强可能有用，不能归功于动量。只改善E50不改善selected，不晋升。
若仅一个数据集通过，保留条件性结论。
若触发最大域任务、pooled、worst-cell或macro LogLoss保护门，报告TRADEOFF。
阶段1结束后按04的明确规则决定是否进入P2；未触发即停止。

## I. P2条件矩阵

仅当P1出现有实用价值的正向趋势但固定β_delta不稳定，或注入诊断给出可解释的阶段失配时触发。只在触发数据集执行；RDI与固定λ=.25/.5/.75同视图预算对比。每臂五seed，单数据集最多10条（固定规则可共享已有A1/A2结果，新增RDI与必要固定臂按实际执行计数）；两数据集最多20条。若固定规则相当、a_t坍缩或selected无增量，停止RDI。

## J. 交付物

research_reports/2026-11-01_momentum/至少包含：
- README.md、IMPLEMENTATION_REPORT.md、REVIEW_REPORT.md、REPORT.md；
- resolved_config.json、source_manifest.json、data_manifest.json；
- run_ledger、epoch_metrics、per_cell_metrics、selected_metrics、paired_effects；
- optimizer_state_diagnostics、shadow_metrics、decision.json、costs.csv；
- 公式等价测试结果、checkpoint恢复测试结果。

报告区分实际完成、未触发、失败阻塞；不上传原始数据和大权重。保留失败attempt，不删seed、不缩batch、不改split。

## K. 完成后停止并发布

完成阶段1及按条件触发的P2后停止，不自动扩展新算法。写明：
- 当前接受的经验结论；
- 哪些因果解释有直接对照支持；
- 哪些机制仍未区分；
- 哪些数据集/seed/域任务适用；
- 是否值得进入下一阶段，以及唯一最有决策价值的下一实验。

通过正常Git流程将新代码、配置、轻量报告和结果提交到独立分支，创建PR合入main。不得覆盖历史文件。若资源/权限阻塞，交付已完成内容并明确阻塞，不得虚构训练或推送状态。


**梯度差分与None语义补充（强制覆盖任何简写）**：g_real是全batch真实loss梯度；g_real_target和g_aug_target分别是指定目标域clean/aug loss梯度，二者都除以原始B×T。δ=g_aug_target−g_real_target，混合梯度为g_real+λδ。不得用g_aug_target−g_real_full。每参数独立step_p：两路均None时跳过该参数所有状态；仅一侧None时将该侧视为零参与差分，合成梯度非None则更新并递增step_p。所有bias correction按step_p执行。
