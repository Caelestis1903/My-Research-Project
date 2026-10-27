# CODER IMPLEMENTATION REPORT — Relation-Basis v2

状态：IMPLEMENTATION_READY_FOR_INDEPENDENT_REVIEW。日期：2026-10-27。

运行修订：首轮READ跨batch数值验收阻塞已按批准补充设计修正；实测证据、针对性验收和重启身份见 `CODER_FIX_REPORT.md`。原运行和诊断成本保留。

第二次运行修订：KR近零parity采用标准FP32绝对容差并补实际违规点记录；保留ML常驻模型的双revision冻结方案见 `CODER_FIX2_REPORT.md`。

已按 DESIGN_SPECIFICATION.md 实现新包 `experiments/mdmt_relation_basis_v2/`。未修改历史算法、原始数据或规划文件。运行 seed 固定为用户指定的2022、2023、2024、2025；开发矩阵48条适配轨迹，共8份共同前缀。

## 实现范围

- config：唯一YAML、重复键/缺项/未知方法/非法值拒绝；用户seed与设计解析作为有效配置显式记录。
- data：复用已核验strict_data完整prepare；显式迁移后raw路径；重建后重新核验raw、编码、真实标签、fit-only词典/缩放、精确自然row计划。训练标签权限独立于历史类。
- native/source：固定融合fixed1，original初始化后复制非融合参数，native24epoch、physical batch40960；V在12–24选点；owner-only pilot12pass；两任务隐藏32拼来源64并在T0_s标准化；教师坐标转换检查。
- model/relation：B_REFERENCE、DIRECT、READ、TARGET_ONLY、LOWRANK_RELATION、FULL_RELATION，foreign-only候选与NULL、rank2关系、rank4函数、w半径2投影。
- trainer：自然有效标签BCE、20epoch含第1轮warmup、不重启Adam、六臂共享F顺序计划、V候选5–20、selected/final完整RAM状态。
- diagnostics：Q全行的ALL_ROOT和固定F均值关系干预、输出距离/U范数/关系熵与NULL、S来源教师风险；raw-X和缓存前缀batch1/256的50warmup/200测量；参数及明确范围的MAC账。
- campaign/CLI：每dataset/seed常驻worker训练六臂，CPU offload后写READY；八个worker的配置/源码/selected/final身份全局冻结后，逐worker释放E历史复评。audit/plan只读。失败记录保留实际耗时和错误。
- report：逐seed逐格CSV/JSON，均值/sample SD/seed级t区间、配对差和预注册门槛的数值计算。科学解释由主agent完成。

全部模型、Adam、RNG、selected/final和前缀状态保留在RAM；磁盘仅写配置、来源与状态摘要、行数据缓存、预测、指标、日志和报告。没有磁盘模型恢复入口。进程中断后的重放必须保留原失败目录并另记实际成本。

## CPU 验收

命令：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_relation_basis_v2/tests -q
```

结果：**10 passed**。完整输出见 `CPU_ACCEPTANCE_STDOUT.txt`。测试覆盖：配置拒绝；六臂实际参数；U零初始化/独立V；本域屏蔽与NULL/J1；warmup和联合梯度；w投影与单行/缺域；NaN缺失mask分母；稳定FP64 logit LL与AUC；RAM下一步logits和Adam更新精确恢复/错身份拒绝；native机械前向和教师转换；角色权限与只读CLI；小CPU适配链运行到20轮并保留两状态且E未评分；48条记录汇总和八worker冻结quorum。

新增可训练参数：B_REFERENCE=49926；DIRECT=49926；READ=50718；TARGET_ONLY=62406；LOWRANK_RELATION=62166；FULL_RELATION=62846。关系参数112对792；函数参数不计入关系节省。LOWRANK与TARGET_ONLY相差240。

有一条现用PyTorch TF32设置API的弃用提示；实际设置关闭TF32。未增加依赖。源码逐文件来源、实际哈希及许可证检查见 `IMPLEMENTATION_SOURCE_MANIFEST.json`。

## 审查通过后的命令

以下均从仓库根目录执行；`--artifact-root`必须放在子命令前。

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 audit
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 plan
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 prepare --dataset movielens
CUDA_VISIBLE_DEVICES=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.mdmt_relation_basis_v2 train --dataset movielens --seed 2022 --device cuda:0
```

训练命令常驻等待全局冻结；runner须保留进程，按单卡实际资源启动其余ML seed，随后prepare KuaiRand并使用同一冻结实现运行四seed。每张卡单独核对显存和功耗，维持原physical batch；native/source大阶段结束与READY时均释放可回收CUDA缓存。

每个worker写 `artifacts/mdmt_relation_basis_v2/workers/<dataset>/seed<seed>/READY.json`，其中`identity`是冻结所需的精确对象。48条轨迹和诊断完成后：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 report
```

主agent作出阶段决策后，写JSON文件，形如 `{"status":"FINAL_METHODS_FROZEN","stage2":{"movielens":"NOT_TRIGGERED","kuairand":"NOT_TRIGGERED"},"decision":"主agent填写实际决定"}`。只有真实未触发时才用NOT_TRIGGERED；触发项须完成后用COMPLETED。然后：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 freeze --decision research_reports/26-10-27_v2/STAGE_DECISION.json
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 evaluate --dataset movielens --seed 2022
```

evaluate释放指定常驻worker的E评估；runner按设备资源释放其余worker。`COMPLETE.json`表示该worker六臂的selected/final历史E数据均已输出。最后再次report汇总E。源码在全部worker运行与冻结期间须保持不变。

## 尚待运行的验收事实

真实prepared重建、8份GPU共同前缀、48条真实适配轨迹、真实成本与E复评由reviewer准入后的runner执行。阶段2代码按设计不预先实现；触发时先补充设计和独立审查。

未运行真实GPU训练或研究结果分析。进程中断会丢失RAM模型，需要同身份重放并记录额外成本。
