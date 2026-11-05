# 07 交付Codex的总执行入口

你接手的是 **已完成A/B/C数学、实现和实验设计的工程任务**。请实现、测试并按阶段运行，而不是重新探索研究方向。本文件及同目录manifest构成执行入口。

## 最高优先级

规划位于main的`research_plans/26-11-5/`。B先运行，A/C依据已有决策树继续；所有路线方案均完整。每个用于超参数优劣比较的候选都必须完成2022–2026五seed全预算，不允许单seed预筛后只补赢家。仅使用validation比较超参数和选checkpoint，test在配置/选点冻结后访问。

小域不是困难域，不安排该结论的假设性检验。正式baseline只使用已发表模型；F/A/B/C及所有变体属于内部方法/消融。按现有.003宏AUC与−.001逐格规则报告，LL独立，不能强制所有seed同向。

## 1. 必读顺序

先读main的EXP_AI_CONTEXT.md、experiments/Experimental_Global_Summary.md、MDMT_ALGORITHM_METRICS_LEDGER.md，然后本目录00/01/02/05/08/09、manifest、B1–B4。A/C的四份独立规范必须在对应实现前完整阅读。source index明确固定源代码和报告，不用会话印象替代。

## 2. 准备完整执行源码

本次计划仅上传文档，不含新Python实现。当前main主要合并研究结论，完整父训练代码在固定提交`c8675ae2923c15b6965884b140b5b646690cd95e`。不要在main上盲目import缺失包，也不要把旧A1默认的KR d2当本轮d0。

在已有repo里先检查`git status --short`，不覆盖任何未提交文件。可用下列命令新建执行worktree（路径/分支已存在则先核对，不强删）：

```bash
git fetch origin
PLAN_REF=$(git rev-parse origin/main)
SOURCE_REF=c8675ae2923c15b6965884b140b5b646690cd95e
git cat-file -e "$PLAN_REF:research_plans/26-11-5/experiment_manifest.json"
git cat-file -e "$SOURCE_REF^{commit}" || git fetch origin "$SOURCE_REF"
git worktree add -b research/26-11-5-aug-history ../My-Research-Project-26-11-5 "$SOURCE_REF"
cd ../My-Research-Project-26-11-5
git restore --source="$PLAN_REF" -- research_plans/26-11-5 EXP_AI_CONTEXT.md experiments/Experimental_Global_Summary.md MDMT_ALGORITHM_METRICS_LEDGER.md
```

这里的execution分支仅用于之后的正式代码与实验，**不改变本次规划已按用户要求放main的事实**。记录PLAN_REF/SOURCE_REF与实际implementation commit；若main后续规划变动，必须冻结所用版本，不在训练中途漂移。

source环境路径曾为`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`，不要假定当前机器一定存在；检测依赖/版本/CUDA及数据路径，填写实际环境。禁止自动升级优化器版本后继续声称逐位同源。数据不可得时报告工程阻塞，不生成模拟性能。

## 3. 实现范围

新增包`experiments/aug_history_20261105/`，实现09公共接口与各路线模块，不能改写旧原始实验结果。先实现B和必要公共功能，A/C按触发实现；设计无需再问。代码须有清晰类型/形状契约、异常处理和单元测试。不得做无关模型重构。

特别检查：source模型输入[F,N]、输出[N,2]；gate index3t+d；Gaussian RMS detach；实际2N分母；普通coupled Adam；EMA后更新；None与零梯度；参数组坐标互斥；C teacher旧状态；独立RNG；RAM-only状态和准确恢复。

## 4. 按合同运行

以下CLI是本次要求你实现的公开接口，当前并非已存在脚本。实现后必须通过--help/参数解析与测试再执行。

```bash
PLAN=research_plans/26-11-5/experiment_manifest.json
RUN=artifacts/aug_history_20261105/attempt1
python -m experiments.aug_history_20261105.run preflight --manifest "$PLAN" --output "$RUN"
python -m experiments.aug_history_20261105.run validate --manifest "$PLAN" --suite all --output "$RUN"
python -m experiments.aug_history_20261105.run train --manifest "$PLAN" --route B --datasets movielens kuairand --seeds 2022 2023 2024 2025 2026 --output "$RUN" --resume
python -m experiments.aug_history_20261105.run select --manifest "$PLAN" --route B --split validation --output "$RUN"
python -m experiments.aug_history_20261105.run decide --manifest "$PLAN" --route B --basis validation --output "$RUN"
python -m experiments.aug_history_20261105.run evaluate --manifest "$PLAN" --route B --frozen "$RUN/B/selection_frozen.json" --output "$RUN"
python -m experiments.aug_history_20261105.run continue --manifest "$PLAN" --decision "$RUN/B/DECISION.json" --output "$RUN" --resume
python -m experiments.aug_history_20261105.run report --manifest "$PLAN" --output "$RUN"
```

若先只实现B，validate的all应报告A/C为DESIGN_COMPLETE_NOT_IMPLEMENTED而非假PASS；B工程全部PASS即可按阶段运行。A/C触发后各自suite必须PASS，才能开始其正式矩阵。自动continue不可用时输出04规定的已解析人工命令和机器决策，不临时发明新配置。

## 5. 调参不是可省略的附录

B：h=.5/2、K10/20，D/M/S同预算；U=.5/1/2。
A：分组/全局自适应η=.05/.2、K20/40及固定profile，完整15影子。
C：先γ−2/+2,K20，再赢家符号的.5，再K40；每一步都五seed后选择；两个机制控制匹配最终γ/K。

全部取较优验证综合值，不声称全局最优。无test后扩网，无早停，无单seed筛选。

## 6. 结束与交付

生成完整源码/配置/数据身份、工程测试和审查报告、每epoch V曲线、所有HPO候选五seed结果、冻结选点、六格与总AUC/LL、成本、decision和中文研究报告。报告明确：已完成、未运行、工程阻塞、成本阻塞各是什么；不要把没有运行的A/C写成失败。

正式运行后更新研究总台账与必要handoff，只写实际结果。全部新代码和标量产物提交到执行分支，保留可复核commit；不将大模型、数据或RAM参数快照写入Git。阶段执行停止于预定义规则，不为了凑满三条路线或追求绝对证明无限追加实验。
