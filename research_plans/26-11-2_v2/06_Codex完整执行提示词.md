# 06｜Codex 完整执行提示词（唯一入口）

你在 `Caelestis1903/My-Research-Project` 中执行 EMA02V2 研究。先读本目录00–07及 `experiment_queue.yaml`。本次任务是实现、独立审查、分阶段实验和科学分析；规划本身没有运行结果。已授权在本合同范围内自动执行，无需阶段间询问用户，但不得扩大研究范围。

## A. 来源与有效规则

1. 核对当前main HEAD、git status、目标目录已有内容。基准读取main为`0bf3e970ca47da44b129e67bb4c84a653ba6a20b`；保留后来提交，不reset、不force push、不覆盖无关文件。
2. GATS02固定来源`26-11-02@e59efa2fd954c4d2231b679a742406cfbe2a1077`；AIGM来源`26-11-01_v3@6f5742c2085c1e49d1767b0a7079bb9f28160eda`；GAM来源`26-11-01_v2@ee4cf0512a7723624c9ee6a1ef192f0f7f17686b`。读取01/07所列核心报告、数表和代码，核对分支若有后续更新是否改变合同。不要把报告发布SHA写成实际训练SHA。
3. 读当前根AGENT.md、AGENTS.md、必要PROJECT_ARCHITECTURE；当前损失以GATS实际源码与本包03为准，是原BT的Sample预算，不恢复早期Equal/DBO/learned fusion/AutoML/DARTS。
4. 使用显式解释器`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。ML CSV=`/home_new/xzf/datasets/ml-1m/ml-1m.csv`；KR CSV=`/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。按来源核对raw/六数组/行号/encoder/scaler哈希，不用缺失的mdmt_next prepared目录替换当前loader。
5. main缺少旧依赖时，仅把所需依赖闭包以原blob只读vendor快照放入新包，记录原commit/path/blob与import适配diff；不合并整个历史研究分支，不改历史源码和报告。

输出SOURCE_RECEIPT：实际分支HEAD、读取路径和blob、执行源码manifest、环境和数据身份、当前GPU外部任务、未读/缺失内容。历史RAM状态已释放，不宣称旧权重可恢复。

## B. DESIGN → IMPLEMENT → REVIEW → RUN → ANALYZE

遵循仓库编排：designer产生DESIGN_SPECIFICATION；luna-coder只实现批准合同；另一reviewer对设计/代码/实现报告独立审查；PASS或所有blocking fixes完成后由luna-runner运行；协调者做研究分析。角色不可互相冒名。工具环境确实没有指定子代理时，记录能力阻塞，不能伪造独立review；不得因此跳过独立审查直接跑正式实验。

本轮新增包建议`experiments/ema_view_design_20261102_v2/`，报告`research_reports/2026-11-02_ema_view_design_v2/`，标量artifacts按attempt隔离。只新增本轮文件，不覆盖旧计划/原始CSV/报告/results。

## C. 不得交给 coder 猜测的核心合同

- 主一J1：当前训练目标域batch内，按完整两任务标签同值选最多两个同域供体；排除自身；共同半径`min(κ||h||,.2 min||donor−h||)`；用两任务归一化输入响应的最小值选最大者。对照J0随机选择；J_SUM用任务平均响应。
- 主二F1：教师两个raw logit对无量纲字段增益的完整输入Jacobian构成J；投影学生监督输入响应到`I−Jᵀ(JJᵀ)†J`；按03统一缩放/cap。φ=θ时梯度可代数归零，按合同回退随机零空间，不伪造梯度活跃。F0无教师投影；FR匹配控制自身状态的F1反事实能量；F_PR始终随机零空间。
- 原模型gate使用`emb.detach()`。只读probe必须冻结参数并移除这一输入断图，使其前向与原模型一致、输入导数与有限差分一致；训练路径保留原detach。普通functional_call不能自动修复该问题。
- 最终`δ`整体detach，真实训练用`h+δ`；不让donor或teacher获得监督梯度，不用高阶梯度，不把`h*(1+detach(a))`产生的梯度缩放混进F1。
- 共同损失`Lreal + λ Σ_target,t(ℓaug−ℓclean)/(原B×2)`；λ=.5。TG两个任务各用自己的一个视图，不让两个视图各重复全部任务预算。空域、缺供体、零方向按03确定处理。
- Adam仍是原标准单Adam，一次step；coupled wd只进入Adam，保留None/显式零、参数step与bias correction语义；没有q_delta/βδ/递归状态。
- EMA从θ0初始化，MLrho=.9828205985452511、KR=.995；本步读旧φ构造视图，Adam后更新φ。J/TG的φ不反馈；F1/FR/F_PR的φ反馈，不自动继承旧EMA效用。一个EMA状态兼任教师与平均评价，不伪装成同轨迹独立非反馈对照。
- 所有新机制训练信息只来自train；不读validation梯度/收益在线调整策略，不读test选策略、超参或epoch。

## D. Stage0 与审查验收

先运行本包 `validate_plan.py`；再实现03最后一节的最小语义检查，尤其source A0/A1关闭一致、原BT分母、probe真实数值导数、joint-label donor、等能量、F联合投影/退化、FR能量、TG任务预算、δdetach、None/zero、EMA顺序、随机流隔离与RAM恢复下一步一致。

对两个真实数据集做短CUDA检查：训练联合标签四格及每域计数、ML嵌套断言、KR不预设嵌套、供体覆盖、rank/fallback、实际能量、无监督teachergrad、真实峰值显存与成本。Stage0不打开test，不产生新方法性能结论。未通过只修复阻塞问题，不扩展算法。

## E. 队列与有限调参

按04/YAML：先24条开发完整轨迹A0/A1/J0/J1/F0/F1×两数据×seed2022/2024，全部50epoch。F1满足触发条件才追加FR，最多4条。

随后只选一个全局路径：
- main：每数据集最多一个边界家族、一个κ减半点，J0/J1或F0/F1/FR配对；必要消融仅J_SUM或F_PR。
- backup：只有未启动任何主校准/消融、所有主家族及简单替代无可推进/可校准信号才跑TG/RG/AG。不能主分支失败后再追加backup。
- stop：无可执行条件或预算不足时完成报告，不凑满矩阵。

每数据集最多一个获选机制进入seed2023/2025/2026。基线训练也计预算，F/TG最多5臂、J最多4臂。确认validation有希望者进行W1（原CSV 0–60% train、60–70% select、70–80% confirm），冻结配置不调参，seed2023/2025、三臂。W1不是新外部数据、不是已证明时间顺序。

主路径≤86、备选≤82、全局≤86条新增唯一轨迹；条件不满足就跳过。开发/确认阈值、保护、选简单替代及缺控制状态完全按05，不因一个好seed放宽规则。把每次派发前decision JSON写入标量ledger。

## F. 资源、恢复与评价

外部GPU任务不得终止。保持batch/精度/epoch，只调整并发；遵守现行GPU约束。S0生成实测成本预测，教师forward、输入梯度、shadow validation、调参和重试都计成本。

RAM-only：模型/optimizer/EMA/RNG/loader快照不落磁盘。使用独立RAM guardian及现有IPC，完成接收后标RAM_COMMITTED。协调器失败不重算整个队列；内存全失明确不可恢复。必要job最多一次重试，全部物理成本限冻结派发预算的1.20倍。失去必要模型且无预算则标EVAL_STATE_LOST，禁止补造test。

每条训练的online/EMA独立validation选最早最佳；保留selected和E50。先冻结候选、超参、各split选点及模型hash，再统一原test/confirm_W1。历史test已用过，报告不能称完全未见。

## G. 结果与提交

交付源码、源/数据收据、DESIGN、CODER_REPORT、独立REVIEW、Stage0、配置/YAML、run/trial/decision/cost/failure ledgers、逐seed六格与macro/LL/traffic/pooled/worst指标、配对分析及最终REPORT。分析区分经验收益、机制归因、适用范围和停止决定；普通EMA已有收益不算新方法增量。

只在05门通过时称对应范围内确认；简单替代更好就保留简单方案。全部失败也明确停止当前候选，不回到差异化共享/多原型/样本重加权，不新增未登记门控或优化器。

完成后按用户授权提交本轮源码与报告到独立研究分支；除非本任务另有明确要求，不改历史报告。规划包已在main，运行报告的最终合并遵循仓库当前工作约定；不要将规划状态改写为“实验已成功”。
