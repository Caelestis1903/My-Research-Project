# Codex 完整启动任务：DFD-v4.0


以下正文可以直接交给能够访问实验服务器与本仓库的 Codex。它授权按本轮已锁研究设计连续完成**代码实现 → 必要验证 → 实验执行 → 分析归因 → 最终报告**。这不是要求再提出一份研究规划。规划阶段尚未训练新模型，所有效果与成本需真实运行后报告。

---

## 你的任务与唯一主线


在 `Caelestis1903/My-Research-Project` 上执行 `research_plans/2026-10-26/` 中的 **DFD-v4.0：需求条件来源函数形成**。不得维护旧RHD/LPS结构作为前提，也不得自行替换本轮科学假设、指标、seed、对照或晋级条件。完成已经触发的完整实验包；负结果是正常输出。没有触发的后续阶段按NO_START结案，不为了寻找阳性结果加实验。

先完整阅读仓库 AGENT.md（及当前路径生效的AGENTS.md若有），再读取以下全部文件：

1. [00_README.md](00_README.md)
2. [01_实验事实复盘与机制分析.md](01_实验事实复盘与机制分析.md)
3. [02_相关研究与创新定位.md](02_相关研究与创新定位.md)
4. [03_研究方向重新设计与决策.md](03_研究方向重新设计与决策.md)
5. [04_核心算法与Codex实现规范.md](04_核心算法与Codex实现规范.md)
6. [05_完整实验矩阵与研究决策.md](05_完整实验矩阵与研究决策.md)
7. [07_开放问题与研究风险.md](07_开放问题与研究风险.md)

数学/梯度/模块以04为准，矩阵/门槛/统计/外部适配以05为准。本启动文件完整规定任务边界，但不能替代精确公式。若发现实质矛盾，先列明并修复相同科学语义下的文档错误；不得等看到结果后选有利解释。可以自主解决实现细节、恢复与故障，不需要用户再次选择研究方向。

## 一、首先核实真实工作环境


检查当前仓库、分支、commit与工作区变更，不覆盖既有工作。以当前main创建独立实现分支，例如 `research/dfd-v4-20261026`；不直接在main做实验开发，不整体合并任何旧实验分支。

本轮设计主要证据为 `2026-10-25_15-56-04@8a4b3b296ca5dc7349956fae9eca5e202aefe2c0`。先读其 `research_reports/2026-10-25_v3/REPORT_INDEX.md`、FINAL、DESIGN_SPECIFICATION、IMPLEMENTATION_REVIEW及与本任务直接相关附录。该树主要发布报告，不能默认完整v3训练源码已在远端。`research_plans/26-10-25_v2/` 当时未发现，不能把不可读路径写成已检查依赖。

可读源码基点：

- `c5618e281785a4b7063ff9bf2a61cf38d4faf1be:experiments/data_generate/MDMTRec/models/multi_domain/fixed.py`
- 同commit的 `experiments/reproduction_20261021/protocol.py`
- `f0b1084b7bcbde5a6c6cb6210c4ef7511bcfff8e:experiments/natural_ml_reassessment/controlled.py`

如服务器v3代码可读，核对实际文件与审计身份后使用；否则按04提取pinned fixed包及依赖到本轮vendor，以机械adapter获取原生logits/shared experts。不要恢复被有意关闭的learned fusion或AutoML/DARTS。历史protocol的lr=.01和Equal loss不是本轮配置，不能直接套旧trainer。

报告给出的服务器位置供核实，不能未经查看称其现在存在：

- ML raw `/home/xzf/data/ml-1m/ml-1m.csv`，报告SHA256 `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`；
- KR raw `/home/xzf/data/kuairand/processed/kuai_rand.csv`，报告SHA256 `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`；
- prepared `/home/xzf/data/mdmt_next_20261024_strict_v1/{movielens,kuairand}`；
- 旧v3工件 `/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/`，旧checkpoint可能位于 `/home_new/xzf_mdmt_next_v3_2026-10-25_15-56-04`；
- 旧环境 `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`。

新工件使用独立 `artifacts/mdmt_dfd_v4/v4_0/`。先核原始数据、磁盘、GPU可用性和依赖；不修改旧数据、checkpoint或结论。不杀其他进程、不占用用途未明确的GPU。

## 二、不可更改的数据与训练语义


保持ML/KR自然领域比例，fit行数预期为 ML `[124759,254879,260495]`、KR `[138225,1395118,71561]`。fit-only词表/scaler/OOV；validation/test行身份不动。角色manifest与复建整数规则见04，先验hash不匹配必须解释，不能直接混用。

角色用途：fit训练native/pilot/source函数；P_form训练η、折外需求诊断与所有软reader的共同预拟合；P_read训练最终读取；P_cal选完整六头组；validation选native；Q_explore只开发和晋级；P_update主算法不用；旧Q_confirm/test一律暴露基准，锁定后评价。一个阶段的OOF记录不等于最终模型仍然对该数据折外。

原生配置：fixed fusion `exp_d=exp_t=bal_d=bal_t=1`，4 shared experts，`fcn_dims=[512,256,256,64]`。原生输入历史为 `[F,B]`，新adapter接收`[B,F]`并明确转置一次。真实训练的gate detach保留；不要调用改变梯度图的risk模式。

主训练是自然行两任务BCE平均，逻辑batch40960，drop_last=False，microbatch4096可累积但分母必须为完整batch，Adam只step一次。Adam lr=.001、betas(.9,.999)、eps1e-8、coupled wd1e-5，FP32，无AMP/TF32。数据不重采样、不复制、不加噪声。Equal与Sample是不同目标，不把变更归为简单梯度幅度。

开发seed固定5101/5102，正式固定6101/6102/6103/6104；先检查是否已用于该方法，冲突须记录，不能自动换seed。每个数据集独立决策，所有方法成对。seed、source、data、row plan和stage均独立RNG命名空间，严格按04保存。

## 三、必须实现的完整算法


### A. 原生与pilot


每seed从scratch训练native24epoch，按validation macro LL选epoch12–24，精确平局最早；另存epoch24。永久冻结selected native及所有状态，缓存四shared expert归一化拼接 `h256` 和native两logits；不要误取同维STAR向量。

6个source/task残差pilot `256→32 ReLU→1`，只用来源自身任务fit标签，固定12passes。损失为自然两任务平均，输入h/native冻结。pilot末层初始0、隐藏Xavier，训练后冻结。输出 `p_s,q=sigmoid(z_q+f_s,q(h))`。本域pilot是LOCAL，原生是NATIVE，两者都报告。

### B. 条件概率与需求


每个target/task一个η头 `261→32→1`，输入 `[h256,z0,z1,f_0q,f_1q,f_2q]`。fit目标域特征normalizer冻结；P_form两折，每fold20passes，batch512，Adam .001/wd0；折外记录只由未见该行标签的头生成。两任务共用row fold。该数据集P1通过后，才在全部P_form从规定初始化refit20，不能把refit预测写回OOF文件。

固定比较候选是本地p0与半读取 `pa=.5*p0+.5*p_source`，令a0/a1为其logits；解析需求

`mu=softplus(a0)-softplus(a1)-eta*(a0-a1)`。

严格按04使用FP64数值保护、全池真实差均值m和OOF中心预测SD定义τ。收益差没有提供额外独立标签，不将逐行事后赢家当无噪声真类。

### C. 需求条件形成


每个source/task有两个意图函数，对应另外两个目标域，共12个函数 `256→32→1`。从source pilot复制，末层weight加std1e-3配对微扰；新建Adam状态0，不能继承pilot moments。

在source实际fit行上保持实际domain不变，用两个target需求头外推；中心化需求b后

`w_k=.1+1.8*softmax_k(b_k/tau)`，每行/任务w和=2。

权重一次缓存冻结，不做各prototype batch均值归一。形成loss=`sum_i,q,k w_iqk*BCE(z_iq+F_source,k,q,y_iq)/(4B)`，每个函数有独立监督，固定12passes。梯度只更新实际source对应函数；native/pilot/η都冻结。缺源参数grad=None、不推进Adam状态。最终库固定末次状态，不用P_cal选形成epoch。

target需求到source支持的外推是关键未证假设，必须保留相反解释和支持统计。正权weighted BCE在理想无限容量下不保证不同Bayes函数，不能仅由w不同宣布功能分化。

### D. 普通读取


每target/task读取本地pilot加两foreign source各两个意图函数，共5个概率；读取所有4个foreign函数，不筛成只符合target名称的两个。普通reader `263→32 ReLU→5softmax`，输入h256、native两logits、5个概率，输出概率凸组合。6头共51678参数。

所有软reader先P_form20passes，再P_read10passes，Adam .001/wd0、batch512，状态连续；P_cal从精确OFF、预拟合后pass0和完整pass1–10组六头选择；不逐cell拼头。不在P_form阶段选择最优pass。稳定logsumexp计算混合BCE。库和输入均detach。

DFD新增部署参数200304，η额外50502只训练时需要；真实native参数、激活量、缓存和成本自行测量，不能把这些参数数当速度证据。

## 四、必须实现的对照


核心P2每seed固定10个输出：

- NATIVE、LOCAL：共享前缀导出。
- H_DIRECT：258维完整h+native logits直接概率拟合，P_form20→P_read10，P_cal整组选。
- ETA_DIRECT：复制全池refitη及Adam，P_read10，P_cal含pass0整组选；原DFDη保持冻结。它使用source pilot特征，不能叫无foreign信息。
- UNIFORM：相同12函数，w=1。
- GEOM：source fit上的固定K2软几何责任，具体子集/温度/seed见04。
- SHUFFLE：按source/task、source标签及本地pilot置信度十分位分层，对整个二维责任向量置换一次，12passes固定；保留分层边际和每行总量。
- CONST_UNIFORM：相同UNIFORM库，P_form∪P_read求5维常量simplex凸BCE；P_cal只比完整常量组与OFF。检查可行性与Frank–Wolfe gap，不能以几步弱优化替代。
- LOCAL_CAP：每target/task4个本地函数，只用本地fit标签；总参数更多但每预测函数数相同，每函数loss系数1/(4B)；完整边界见04。
- DFD：唯一主方法。

P3只在P2通过后加入U_BUDGET（UNIFORM的reader P_form60→P_read10）和POOL（pooled pilot初始化的12个pooled函数）。U20/U60均报告；只匹配训练label-pass，不冒称walltime/FLOPs完全相同。POOL具体loss与来源替换范围按04，不借旧source初始化宣称消除了所有来源监督。

## 五、严格执行 P0–P4，不再开放研究设计


实现目录为 `experiments/mdmt_dfd_20261026/`，模块和CLI全部按04建立。先DESIGN已锁核对→IMPLEMENT→独立REVIEW→RUN→ANALYZE。审查只针对会影响科学结论的数值/数据/梯度/状态/选择语义，不扩张成无关测试工程。

P0完成原生parity、数据角色、BCE恒等式、w守恒、权限/None-Adam、resume、simplex近优和whole-policy选择收据。缺旧工件不阻止可验证的新主线，缺原始数据/真实标签则明确阻塞，不能伪造运行。

P1两数据×两seed，固定CONST、LOGIT、DEMAND、DIRECT四折外策略。LOGIT输入是完整候选预测的5维 `[z0,z1,f_0q,f_1q,f_2q]`，不是只给本地两个值。所有策略只在train fold拟合。数据集通过条件：DEMAND相对CONST两seed LL增益均>0且均值≥.0001，每seed≥4/6格非负，平均不差于LOGIT，且无实现阻断。DIRECT匹配不否定需求存在。

P2对通过数据集完成全部10输出。DFD相对其余9个对照，两个seed各自LL收益>0且均值≥.0005；每个对照平均AUC退化≤.0005；`max_cell mean_seed(ΔLL_cell)≤.003`；非退化与实现合格。逐项布尔表公开，不能只比弱对手。失败则NO_START P3/P4，保留科学结论。

P3两seed的U_BUDGET与POOL必须完整完成，继续用同样LL/AUC/逐格门槛。加固定原readerα下的source替换、单函数UNIFORM替换和同source意图交换，每格最多8干预，无新训练。需求组切点只来自P1 OOF。固定状态贡献不是完整训练必要性证明。

P4只对P3通过数据集，锁定6101–6104、12个内部输出和两个外部方法的raw/INFO后，全部从scratch执行。**正式seed不再按它自己的P1/P2表现筛除**。正式主要表为旧test_exposed，legacy/Q/validation均并列；同数据上新seed仍不是未见数据确认。

外部STEM、AutoIFS的作者pin、准确结构和dense字段适配按05，不由你另选默认值。STEM采用README的embedding分块16、shared8、specific1、专家512×3、单层和6head；AutoIFS保持3域2任务、rank2、[1024,512,256]及四种信息流。AutoIFS search10、rewind完成9轮、reuse逻辑10–24共15遍，物理25遍；fresh predictor Adam、冻结search末selector；不得沿用作者训练内test调用。外部INFO用05规定的真实hidden tap和P_form20/P_read10，不能只有裸baseline获得更少目标标签后声称公平。

先建立确切run manifest及预算。全通过最多28条24epoch大模型轨迹，128个正式逻辑预测器多数共享合法前缀/缓存；不要重复计算，也不要把逻辑导出当独立训练seed。按测得组件剩余成本加1.35预留准入完整阶段，资源可用时自动继续。所有已触发实验完成或登记不可恢复失败后再统一评价暴露基准。

## 六、RHD仅限有界诊断


从旧KR3101/3102的R0_BASE完成epoch12/20恢复4个预定完整锚。缺工件就MISSING，不新训旧baseline或改锚。四分支ALL/PRIVATE/SHARED/STOP，H=1/20/40/80同一路径中间态，最多960后续batch；验证复用ALL后最多640新增。STOP只eval父模型，冻结参数grad=None且状态/buffer保持。Q_explore和P_update只评分，不选动作。达到05的重启条件也只记未来研究建议，本轮不自动开第二完整研究线。

## 七、日志、完整状态与交付


保存source/config/roles/encoder/raw hash、所有row/fold/RNG计划、模型与所有optimizer、stage/pass/游标、normalizer、m/τ/w、形成库、候选完整六头状态、选择记录和成本。resume必须从同完整状态继续，不重复P_form预拟合或微扰。共享前缀在实际研究账只计一次，在每方法独立部署账仍计必要费用。

输出原始逐行预测（row_id/domain/y/p/role/checkpoint_hash）、六格和macro/自然总体指标、4seed配对差及描述性t区间、按原始用户的1000次配对cluster bootstrap、局部支持/需求/内容贡献和所有负结果。不可把6格、多个动作或延迟测量重复当成新seed。未见新数据时结论止于暴露基准。

新报告目录 `research_reports/2026-10-26_v4/`，至少包含FINAL_REPORT、REPORT_INDEX、IMPLEMENTATION_REVIEW、DATA_AND_ROLE_AUDIT、P1_DEMAND_REPORT、FULL_PIPELINE_REPORT、ATTRIBUTION_REPORT、COST_REPORT、OPEN_QUESTIONS及逐行工件索引。源码与精简报告保留在独立实现分支；大数据/checkpoint不自动上传，旧成果不覆盖。

遇基础设施中断则同状态续训；发现确定性bug则修复并重跑全部受影响配对，不只重跑失败方法。负结果不重跑，不补替代seed。所有触发阶段正常完成或存在明确外部阻塞后汇总，不能仅交付一段代码、运行命令或新计划。

最终报告逐项说明：H1需求是否可预测；H2跨支持形成是否有证据；H3完整方法是否超越直接概率与其他强简单方法；AUC/逐格代价；全部成本；数据与方法边界；未触发/失败原因；下一项可区分解释的问题。不要把“模块执行了”“权重不相等”或某个局部阳性图当作算法有效的充分证据。

现在从环境/仓库核实开始，连续执行到本矩阵的完整正常终止或明确的外部阻塞。
