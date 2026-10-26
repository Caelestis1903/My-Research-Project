# 数据与实现身份

本轮为 RCF-v1，使用历史暴露基准复评。数据审计已实际完成；两数据各四种子的S0/S1/S2、KuaiRand触发的S3已全部训练完成，共136条方法完成收据；独立SCALE八条训练也已完成。S4按预登记门槛两数据均NO_START。四种子验收及清单已冻结，原始导出、成本交付与最终复核均已PASS，八个RAM上下文已释放，清理COMPLETE；最终训练与选择身份以冻结清单和逐运行收据为准。

工作目录为 `/home/xzf/Code/phase0-3`。这是恢复后的文件树，不包含 `.git`，因此本地 branch/HEAD 不可用，不能将远端提交冒充本地 checkout。任务方案来自 `Caelestis1903/My-Research-Project` 的 `research-plan/2026-10-26-v2-mechanisms@92a73cec187838c2e8baea577620abe32dda31b5`；执行前观察到 main 为 `19620d91f517fce01d4b162eef20ae4749d845e7`。方案目录 12 个文件与远端 Git blob 逐字节一致，main 中方案也与该分支相同。远端身份和本地实际源码哈希分别记录。

| 数据 | 原始文件 | SHA256 | fit 域0/1/2 | 特征数 |
|---|---|---|---|---|
| MovieLens | `/home/xzf/data/ml-1m/ml-1m.csv` | `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4` | 124759 / 254879 / 260495 | 8 |
| KuaiRand | `/home/xzf/data/kuairand/processed/kuai_rand.csv` | `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f` | 138225 / 1395118 / 71561 | 34 |

prepared 数据位于 `/home/xzf/data/mdmt_next_20261024_strict_v1/<dataset>`。独立审查实际运行完整审计，核对 raw SHA、prepared x/y 校验和、原始标签和域、fit-only 词表与 age scaler、OOV 规则、整数行角色及穷尽不交叠。审计退出码为 0，两数据通过；完整逐角色六格数量、正负数及行哈希保存在 `artifacts/mdmt_rcf_20261026/v2_four_seed/DATA_ROLE_MANIFEST.json` 中。审计不是性能复现。

| 角色 | MovieLens | KuaiRand |
|---|---:|---:|
| fit | 640133 | 1604904 |
| P_form | 24004 | 60183 |
| P_read | 24006 | 60184 |
| P_cal | 16003 | 40123 |
| P_update（主线不用） | 16004 | 40123 |
| Q_explore | 40008 | 100307 |
| validation | 100021 | 250767 |
| test_exposed | 100021 | 250767 |
| Q_legacy_confirm_exposed | 40009 | 100308 |

ML 任务为 rating>3、rating>4；KR 为实际 is_click、long_view。ML 原始用户可用于支持分层和用户配对 bootstrap；KR 处理表没有可信原始用户/物品身份，采用行 bootstrap 时必须保留相关性限制。未构造新数据、未改变自然域比例、未重新划分外层角色。新训练 seed 不能消除历史测试暴露。

底层源码选择性恢复自 `dc32f60a6aafce98a25838a485e02cead5425fc6`，26 个纯源码/vendor 文件逐 Git blob 和 SHA256 核对；SCALE 的 22 个所需历史源码文件与 `c5618e281785a4b7063ff9bf2a61cf38d4faf1be` 的 blob 一致。冻结副本放在新实验的 `vendor_snapshot/`，新桥接单独实现，不执行旧 pipeline 或旧阶段门槛。来源清单保留原路径、提交与内容哈希。

执行使用项目环境 `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`，Python 3.13.5、PyTorch 2.9.0+cu126。两张 RTX 3090 可见；GPU1 的既有 VLLM 进程保留。完整环境与包版本见 `ENVIRONMENT.json`。并发依据单卡实际显存、功耗余量，配置不因 OOM 静默缩小 batch。

最新 start.md 禁止本地保存模型或检查点，因此执行采用 `STORAGE_MEMORY_ONLY_V1`：模型、Adam、RNG、位置与选择状态只在内存中保留；仅数据、逐行预测、日志、代码和报告落盘。进程丢失后不能声称磁盘恢复；必要时以相同身份确定性重放并另记成本。`/root` 是协作代理的逻辑命名空间，不是本轮写入路径。


当前有效配置为 `experiments/mdmt_rcf_20261026/config_four_seed.json`，有效配置 hash 为 `cefb6d934c774bd78a438e4fba5c560e934f4c56ab09fbb3dfe389f4f4842fd6`；配置文件自身 SHA 为 `830848363cb730b899616087bba5b952932fddbbbc802f611b397ed6539884af`。原取得的 09 配置保持不改，其父有效 hash 为 `c3ba96191f76e638c734c91ddd6b28f723a5a2bdc4412ea4fd08185024270b43`。用户新增要求覆盖原开发预算：开发训练 seeds 5201–5204，S4 保留独立 seeds 6201–6204，只有既定四开发种子门槛通过才触发。

S0 的 native selected、epoch24、pilot 及 selected_epoch 在原四个5201/5202上下文均与旧记录精确匹配。旧小头汇总并非全部位级相等；独立纯数据审计已保留差值，旧 ETA 的选点未记录，不能补称相同。旧 summary 原字节及SHA全部保留在 `artifacts/mdmt_rcf_20261026/v1/summaries_historical_two_seed/`，归档只取消额外整字典精确相等诊断对继续运行的阻断，未改变新四种子科学门槛。最终比较只用 v2_four_seed 的实际新结果。

旧 durable 启动文档要求 OMP/OPENBLAS/MKL=1，新服务启动初始环境未设置这三项，八个工作进程观测到69线程；worker内的setdefault在模块顶层NumPy导入之后执行。BLAS归约和内存布局是非位级小头差异的具体候选来源，尚未确证唯一原因。本轮没有为追求旧小头位级相等而挑seed或反复重训。详见 `LEGACY_REPLAY_NUMERIC_INDEPENDENT_AUDIT.json` 与独立审查收据。

CPU逐行H/Z/φ、预测、功能输出及标签访问掩码通过明确白名单NPY/mmap外移，模型、Adam与随机状态没有序列化。新服务PID2088990用正常RPC安装只涉及任务派发的已审查并发方法，八个工作进程的模型hash前后不变；其过程收据为 `PHASE_SCHEDULER_INSTALL_RECEIPT.json`。S1幂等重汇总前后104个训练完成收据SHA全部一致，没有新增训练。

代码写入根为 `/home/xzf/Code/phase0-3/experiments/mdmt_rcf_20261026/`，报告根为 `/home/xzf/Code/phase0-3/research_reports/2026-10-26_v2/`，正式本轮工件根为 `/home/xzf/Code/phase0-3/artifacts/mdmt_rcf_20261026/v2_four_seed/`；旧本轮v1保留为过程证据，SCALE八条四种子训练留在其中的 `scale/`。远端main在执行过程中推进，后续观察为 `cc954f86410e887105fe39c6425c3fe93ab60c7a`；不把该动态提交倒写成实验启动身份，不将其未审查源码混入冻结副本。最终发布仍回到原方案分支。

训练阶段prefix身份中的source_hash、后续完整冻结源码身份、观察器补充身份分别记录，不混称一个版本。FROZEN_MANIFEST.json的SHA256为`09def5e2f43786f9c2875e5ae466845a10ec7db47a3c6edd28379827b72e927a`，冻结时source hash为`1b5bd46890e6659fe24eb1c9187854c5ba352b28892b66bb8ba276bc82ba3dc7`。冻结后raw重建观察器身份为`7a1482902a936e83e64a88f7b5daeb8a827e9141dd571da816e0cf34cb094b14`，没有重写冻结清单；全部136训练收据及RAM模型/Adam/RNG/选点hash保持不变。后续若补充交付记录观察器，其身份另存，不倒写为早期训练源码。

测量记录保全补充已独立审查PASS，当前磁盘观察器source hash为`617f1841bf2f4b4208b82000d62779fc3b10ea2e55347501ab3d0ce8419d178b`；已在evaluate自然结束后通过原normal reload-readers安装与执行。原collector原字节也保留，动态磁盘SHA不冒称实际加载身份，见`DELIVERY_SOURCE_PROVENANCE.json`。

## 角色分布差异

真实自然数据不意味着每个既定角色具有相同域占比或标签基率。KR fit/Q的域1占比约86.9%，test为98.2%；域1的click率fit/Q为.37790997/.37763633，validation/test为.20973716/.21051713，long_view率对应.25610378/.25525546与.11152084/.11131458。legacy仍接近Q侧基率。角色保持既定整数行协议，未为本轮改比例或生成样本。

因此Q、validation、test、legacy是不同观测角色，不能将分数差都解释为模型或训练seed。标签先验/输入支持变化可与读取/拟合机制共同影响校准与排序；本轮没有控制干预将其识别为唯一原因。ML test的全局fit支持计数user/item均seen80591、仅item seen19313、仅user seen104、均unseen13；“有数据/无支持”与域大小不等价。KR原始user/item缺失记MISSING，不以feature编码OOV替代原始身份重叠。

逐角色自然域数与六格正负/基率见`tables/DATA_ROLE_domain_counts.csv`、`tables/DATA_ROLE_six_cells.csv`。这些是固定数据审计，域/任务不当成独立训练重复。

## selected与终态的具体含义

主线NATIVE selected按validation MacroLL在epoch12–24整模型选择；terminal为实际epoch24当轮保存的logit/p及state hash。后者没有可下载权重，最终raw-X完整预测器重建针对所选部署，不能冒称用当前selected模型重建epoch24。H/ETA/EFFORT与PILOT_PROB的terminal为PR pass10，selected为P_cal选择的整组六头pass0/5/10（概率gate为5/10）。线性reader已是形成冻结后的完整政策优化，terminal字段按接口保留同一selected p，不是额外训练epoch；LOCAL/B_REFIT也不凭重复字段虚构后续训练。SCALE terminal则是各run实际early-stop末轮，和主线epoch24不是同一时点。完整表保留两字段，阶段解释必须按方法语义分别进行。

ML5201原True测量收据遗失，因此独立追加设计/实现/审查允许且已执行该context固定七个其余已登记方法各一次补测，以确保保留的PILOT_PROB等也有四seed成本；其他worker白名单不扩，已有明确parity失败不重试。此修订不新增训练条件、不改benchmark数学或任何预测/选点。

## 实际模型、优化与选择协议

| 组件 | 实际执行 |
|---|---|
| Native | fixed-fusion M3oE，3域2任务，4个shared experts归一化后拼接h256，fcn[512,256,256,64]，exp_d/exp_t/bal_d/bal_t均固定1；不是STAR或未经限定的原版M3oE |
| Native训练 | 自然fit、两任务BCE总和/(2B)，physical batch40960、保留短末批，24epoch；coupled Adam lr.001、β(.9,.999)、eps1e-8、wd1e-5；FP32，无AMP/TF32 |
| Native选择 | 仅validation六格MacroLL，在12–24整模型选择，最早平局；保存selected与实际24轮输出身份 |
| Source/local pilots | h256→32→1，source own监督与两套local重复各12passes；batch40960，Adam.001/wd1e-5，之后冻结；原始随机bank及隐藏normalizer身份分开 |
| B与direct | [h,z]258→32→1，P_form两折各20passes+refit20，batch512，Adam.001/wd0；H沿refit及Adam接PR10，整policy选0/5/10；ETA261输入匹配扩展；EFFORT从同unit2初态PF60后PR10 |
| 形成 | 每槽δ256→32→1，source隐藏复制+配对N(0,.001)微扰，末层0；12passes，query512、KL source fit4096、S source form256、foreign均方每target256；Adam.001/wd0，source KL权重.1 |
| 形成约束 | S≤.001，两个foreign域等权δ平方均值R≤.25；dual init1/lr10/[0,100]，终态全量验收+40步最大可行缩放；只有δ接收形成梯度 |
| Reader | 冻结δ/内容，PR训练、PC整组六头政策选λ.001/.1或精确B OFF；FP64 L-BFGS-B，原300迭代仅数值失败同起点扩600；alpha[.25,4]、beta[-4,4]，普通w[-1,1]、K1每source界2；FULL内容系数无界，实际合成修正正则+系数ridge1e-6 |

pilot/Native的physical batch与小头/槽query batch是不同组件协议，不因OOM静默改动。FULL_ALL额外使用PF+PR，LOCAL_ACTIVE24头比RCF12头更强，POOLED全foreign监督剂量更大；不能把相同passes写成完全相同标签量或计算。逐seed有效参数、标签访问、原始选择及模型hash在各prefix/方法收据与最终CORE/COST表，不只引用名义配置。

主线所有outer标签统一在全局冻结后用于评价，最终保持6201–6204 NO_START。SCALE使用单独历史80%自然train、六格均值之和、Adam lr.01、valMacroAUC早停，与主线不能直接混比分数。所有数据为真实原始行，没有人工平衡采样或本轮增强；旧Dconst01增强仅引用身份匹配的四seed历史证据。
