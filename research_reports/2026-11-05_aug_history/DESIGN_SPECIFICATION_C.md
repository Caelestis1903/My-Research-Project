# DESIGN SPECIFICATION C：MovieLens 历史漂移条件增强

状态：DESIGN_COMPLETE。依据 A 冻结的 validation 决策，仅 MovieLens 执行 C_FULL；KuaiRand 的 C 为 DEFERRED_NOT_RUN。本规格落实既有 C1–C4、04/05/08/09 和 manifest，不改公式或搜索集合。

## 1. 来源、范围与新阶段身份

PLAN_REF=`2645f1f724f52f5ed5dea4b479335c0dc12fabcd`；骨干 SOURCE_REF=`c8675ae2923c15b6965884b140b5b646690cd95e`。A 实施提交=`f00b5ce6b9e6476ca689b781667a5d698742aa70`，source hash=`afecc8dcef7f1687d61383af0ca69176eb7b7ec26d0700eb43836a827b5e2ff9`。A 的 `attempt_A1/A/DECISION.json` 绑定 selection hash=`ba672ccb017266847085f3de6ccd72cbec8115c93489b4c89b84517b21d8a4c2`：ML RUN_C，KR RETAIN_A_OR_B_AND_REPORT。动作在 A 新 test 前冻结，本阶段不读取 A test 改派。

B `attempt1/`、A `attempt_A1/` 及其源码/冻结/标量产物保持旧身份；A 冻结评价与报告完成、guardian 释放前不修改源代码。C 使用新 `artifacts/aug_history_20261105/attempt_C1/`、新 source hash、SOURCE_RECEIPT/PREPARED 和独立 REVIEW_ACCEPTANCE_C.json。记录旧阶段路径、文件 hash、旧 source/selection 身份，不将新增 C 源码标成旧 source hash。

`scope_datasets=['movielens']` 持久写入 C 的 PREPARED/identity、RUN_MANIFEST、CONTINUATION、HPO 与冻结输出；train、smoke、完整性检查和 report 都使用该范围，不沿用当前硬编码两数据集循环。C 不能创建 KR 科学 job，CLI 范围与 prepared 冲突直接报错。

新增 `drift_probe.py`、`covariance.py`、`conditional_views.py`、`conditional_objective.py` 和 C 状态/顺序调度所需最小模块；窄改现有 model_adapter/protocol/worker/run/selection/report 分派。既有 B/A 数学与 vendor 不改，不新增依赖或恢复框架。完成实施报告、独立审查和 C 工程验收后才正式训练。

## 2. 父配置与独立训练单位

ML 完全继承自然80/10/10、train-only预处理、train800167/validation100021/test100021、d0/d1/d2=155949/318599/325619、fixed1 773470参数、原字段/Adam/loader/Gaussian设置。batch40960、50epoch、每epoch20步、FP32、Adam lr=.01/betas(.9,.999)/eps1e-8/coupled wd1e-4；κ=.05、λ=.5、ρ0=.9828205985452511、target d0；AMP/TF32/foreach/fused/amsgrad/clipping关闭。

每个候选改变 online，独立维护 θ、Adam、固定EMA教师 φ、漂移统计和选点，均从同seed θ0初始化。不能把6候选变成一条 online 的6影子。每 job 仅 online+该候选的固定EMA两个估计器；主选点和 HPO 使用固定EMA，online 是诊断。

## 3. 源模型的 δ 与完整输入导数映射

已重新核对 source `reference.py`、`model.py`、`views.py`。embedding输出进入 skip_conn 和 STAR；`emb=star_mlp(STAR_output)+skip_conn(input_emb)`；gate 原输入为 `emb.detach()`，gate/tower 索引 `3*t+d`。源无 BatchNorm/Dropout，样本间无前向混合。ML `perturb_slices` 对应6个非路由 sparse字段、每字段16维；domain embedding与dense不在其中。

`ModelAdapter.forward_logits(x,params=None,delta=None,probe_input_full_graph=False)` 保留现有普通分支。δ形状 `[n,F,16]`、F=6，表示绝对加性坐标，不乘κ/RMS。临时 embedding forward hook 对每个 perturb slice 返回 `own_embedding_slice + delta[:,f,:]`；clone输出，不改 ID、domain/dense坐标或 δ。online与teacher各自lookup/各自后续网络，使用同一个δ叶tensor，禁止把online embedding传给teacher。

最小 full-graph 方案无需复制整个source forward：probe模式临时捕获 `skip_conn` 与 `star_mlp` 的live输出，为6个gate Sequential安装forward_pre_hook，将原detached输入替换成 `captured_star+captured_skip`。该值与源emb逐值一致，同时恢复经gate输入的δ梯度。现有tower hooks继续取得raw logits，再按domain mask和 `3*t+d` 拼成 `[n,2]`。该修正只在 `probe_input_full_graph=True` 安装；正式训练和 A 的参数探针仍走原gate.detach。全部hooks在try/finally移除，捕获tensor不跨调用保留。

probe用独立/functional detached权重冻结θ、φ，只对δ求导；不要在整个probe外包no_grad。模式/参数requires_grad/原梯度/buffers/RNG调用前后恢复。固定教师的当前φ不能被每epoch selected/E10/E50的scratch评价覆盖；scratch与当前教师状态明确区分。

在δ=0，分别构造 `z_online(x;θ,δ)` 与 `z_teacher(x;φ,δ)`。对t=0/1各求 `grad(sum_i(z_online[i,t]-z_teacher[i,t]),delta)`，得到 `[n,2,F,16]` 的u；最多两个输入梯度遍历。样本无混合，因此sum方式给出逐样本导数。不用sigmoid反解logit，不对参数求梯度。

## 4. train probe、统计与协方差

刷新时钟 `k>=1 and (k-1)%K==0`，位于本步任何训练forward/Adam之前，读取θ_(k−1)、φ_(k−1)。从train d0自然均匀无放回取最多128行，排除当前原batch raw row IDs；两任务共享这些行，不按标签/难度筛选。空probe保留最近有效A/root，初始A=0/root=I；分别记录刷新尝试步和最后有效统计步。

probe使用独立key `AH1105|dataset|seed|step|C_PROBE|0|0`，不含γ/K/candidate；同step配对候选抽相同行，但各自θ/φ不同。源loader seed+11000和基础Gaussian流不变。

FP64下每个u按其16维norm归一化，norm>1e-12取单位向量，否则0。`A[t,f]=mean_i(unit_u unit_u.T)`，平均分母含零向量样本；每刷新直接覆盖，无额外统计EMA。对称化后FP64 eigh，按合同容差截极小负特征值；明显负值/NaN报数值错误。

对 eig(A) 的 a,V，计算 `w=exp(γ*a-max(γ*a))`，`c=16*w/sum(w)`，`C=V diag(c) V.T`，`root=V diag(sqrt(c)) V.T`。A PSD、trace≤1；C正定、trace16，root@root.T=C。A=0可精确返回I。root/A在训练前detach，无跨步图。

C_MERGED对**本控制自身**的两任务A求平均，再给两任务相同root。C_ROTATED对自身A使用每字段固定正交R_f：`A_rot[t,f]=R_f A[t,f] R_f.T`，同字段两任务共用R_f。R_f以独立固定流（stream含字段索引，例如C_ROTATION_F0，step0/domain0/task0）CPU FP64高斯矩阵QR生成并保存，整个seed不重抽。检查其自身旋转前后谱/trace相同，不声称不同候选轨迹的谱相同。

## 5. 基础噪声、监督权重与教师次序

用源 `draw_view` 和 `view_key(dataset,seed,'train',step,'gaussian',1)` 一次生成 ξ `[N_target,F,16]`，保持字段顺序，不将candidate/γ/K加入key。每任务每字段noise为 `ξ[:,f,:] @ root[t,f].T`，最终FP32；将该noise传入源 `apply_view`，由其只乘一次κ和detached字段RMS。原ξ/root/输入不原地改写，不逐样本归一化噪声。

trace C=16保证每字段条件期望能量κ²s²16；不要求实际每次norm相等。metadata记录base key/seed/hash、root hash、probe/teacher/online的preupdate step及能量标量，不存参数或完整统计到磁盘。

正式目标：同一次clean BCE矩阵R，对d0两个任务各自视图t仅读取其任务t的BCE；`total=(R.sum()+λ*sum_t(aug_t[:,t].sum()-R[mask,t].sum()))/(2*N_actual)`。t0视图不再计算t1监督，反之亦然；非d0保持完整clean损失。root相同可共享一次增强forward；不同则两个。γ=0硬分支直接调用原源单视图loss，连基础噪声和浮点次序一起退化，不绕成双视图再称等价。

每步：按需刷新A/root→用旧统计和旧φ构造视图→一次正式backward→一次源Adam→按ρ0更新φ一次并复制buffers。φ既是下一步教师又是固定EMA评估器；禁止更新φ后回填本步视图，禁止将probe loss加入正式目标。φ从θ0初始化，ρ0不调。推理为单个冻结online或φ、真实无扰动输入。

## 6. 顺序搜索与最小job解析架构

| 阶段 | 完整五seed候选 | 冻结后才解析下一阶段 |
|---|---|---|
| I | C1_NEG γ−2/K20；C2_POS γ+2/K20，可并行 | W12=两候选validation选择 |
| II | C3_WEAK γ=.5×sign(γ_W12)/K20 | W123=前三候选validation选择 |
| III | C4_CADENCE γ=γ_W123/K40 | C*=四候选validation选择 |
| IV | C_MERGED、C_ROTATED，均匹配C*的γ/K | 完成两控制各五seed，最终冻结与报告 |

30条ML新轨迹=1500epoch=30000次Adam；每epoch online+φ验证，共3000条模型验证记录。γ0父控制合法复用，新训练0条。没有KR job，没有额外C0科学训练或B+C扩展。

当前 `resolve` 的15影子断言、一个seed只有一个job的select逻辑不适用于C。为C单独解析：attempt根PREPARED保存稳定source/manifest/data/scope及候选模板；已知C1/C2可立即生成数值config。依赖候选在 `C/movielens/` 写小型追加式解析收据（每阶段独立JSON即可），包含candidate_id、mode、确定γ/K、五seed列表、依赖选择hash、输入validation文件hash和本阶段身份。已有解析不得被另一值覆盖。

每个确定config×seed生成独立job/trajectory_id，hash包括完整父config、数值γ/K、控制mode、source/data/manifest和依赖收据身份；每job只有该候选一个φ，不强塞15个candidates。worker读prepared+已合法冻结解析收据查job；尚未解析者无可运行job。禁止给C3/C4先造nullγ job、事后改config但保留旧job_id。

`train --route C` 顺序调用现有并发训练调度器完成当前stage，等待其每候选五seed50epoch及保留RAM selected/E10/E50，再根据validation生成W12/W123/C*冻结文件和下阶段解析。重启按已有相同hash收据继续，完成job跳过，缺seed补齐；不重选赢家、重抽R或扩网。只新增这一层依赖编排，不另建训练/RAM框架。

每job的主φ仍可用candidate_id作为snapshot键，明确 estimator=fixed_ema；online单独键。已有guardian epoch事务、失败回收和重放逻辑沿用。C快照额外保存当前teacher、A/root、γ/K/mode、last_refresh/last_valid_step、R、probe键或Generator状态与累计probe成本，以及原online/Adam/loader/RNG/选点/端点/commit身份。恢复不能用selectedφ覆盖currentφ，不能初始化统计或改变刷新时钟。

## 7. γ0复用、选择与最终冻结

主代理已核验B/A所有online/order hash和F血缘一致。C0引用B或A冻结的ML F+online五seed selected/E10/E50，记录原source、trajectory、epoch/model hash及来源文件hash；不改造成新C源码产生的模型，不额外计seed/轨迹。中间HPO只读取父 `selection_frozen.json` 内F的validation字典；父test数值不参与W12/W123/C*。

每job online与φ各按50次validation macro AUC最早严格最大值选点；候选比较只使用φ的五seed selected validation均值。复用05的六格相对F保护优先、unconstrained winner、实测成本、|γ|及candidate_id并列规则；各stage所有必要候选五seed完整才选择。不同seed共用同一阶段解析γ/K。

四搜索候选完成后先冻结C*，两个控制使用其γ/K训练，不让控制结果反向改变C3/C4解析或重扫参数。最终select再次核验全部六候选五seed和依赖链，冻结C*、两个控制、各自online及父F/online的规定selected/E10/E50端点。全部六候选HPO和validation曲线保留。至少评价C*、MERGED、ROTATED及相应online和父参考；如需报告其它搜索候选test，必须同时在最终test访问前加入冻结端点。

`decide --basis validation`在新test前按04/C3记录C*与验证更优简单控制的取舍；merge在既定近似容忍度内且更便宜可简化。默认动作STOP_AND_REPORT，不论正/负γ获胜均保留实际结果，不依test追加网格或另一轮C。最终报告完整方法对三发表模型、组件对F、任务条件对MERGED、真实方向对ROTATED的逐seed与六格/Macro/Traffic/Pooled AUC/LL、selected/E10/E50、SD/CI和实际成本；正式主参考仍ML M3oE fixed1。KR C继续标DEFERRED_NOT_RUN。

## 8. 仅需新增的验收与运行门

- raw logit sigmoid与源概率一致；probe full-graph开关前后forward值一致；小型FP64有限差分确认δ导数包含gate输入路径，正式训练仍detach。分别用各模型自身embedding，θ=φ时u/A为0、root为I；hooks/mode/梯度/Adam/RNG无污染。
- γ0在相同初始/源batch/RNG下loss/正式grad/Adam/EMA逐位退化；所有基础ξ相同，字段/RMS/κ不重复缩放。尾batch 2N、两任务单独监督、空d0/probe分支符合C1。
- A PSD、C正定/trace16、root identity、γ正负特征方向趋势；纯合成Monte Carlo能量自检；MERGED两root相同、固定R正交和旋转谱一致。不扩大性能矩阵。
- 旧教师先于views、Adam后教师一次更新；source loader/Gaussian流隔离；probe仅train/排除当前raw IDs。两种K跨刷新步RAM恢复时θ/Adam/φ/A/root/R/选点与连续运行一致，状态不落盘。
- stage依赖不足五seed拒绝解析、一次冻结不可覆盖、job_id绑定数值config，test冻结门和ML-only范围通过；已有guardian恢复框架直接复用。
- 原batch GPU smoke检查γ0源等价和一个实际非零统计/条件视图路径及峰值，标ENGINEERING_ONLY，不用其AUC选参数。C独立review绑定最终source，scope为ML的STAGE0/SMOKE全部通过后运行。

FP32 atol1e-7/rtol1e-5、FP64矩阵identity1e-10沿用09；完全同算子路径优先逐位。数值问题不静默加jitter/调γ/放宽容差。并发沿用单卡显存和功耗均<85%、系统RAM<90%规则；OOM先减少并发/临时图驻留，保留原batch、probe128与精度。

## 9. 公开命令与交付

为preflight增加可选 `--datasets`（既有B/A默认保持），C持久scope只含ML；validate/review_gate不再硬要求KR smoke。continue验证输入A决策对本scope授权，把跨attempt收据写到C输出，不要求当前C源码等于A旧source。既有准备身份检查继续约束新C输出。

```bash
PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
PLAN=research_plans/26-11-5/experiment_manifest.json
OUT=artifacts/aug_history_20261105/attempt_C1
ADEC=artifacts/aug_history_20261105/attempt_A1/A/DECISION.json
$PY -m experiments.aug_history_20261105.run preflight --manifest "$PLAN" --route C --datasets movielens --output "$OUT"
$PY -m experiments.aug_history_20261105.run validate --manifest "$PLAN" --route C --suite C --gpu-smoke --output "$OUT"
# 独立review通过并绑定本次C source。
$PY -m experiments.aug_history_20261105.run continue --manifest "$PLAN" --route C --decision "$ADEC" --output "$OUT" --resume
$PY -m experiments.aug_history_20261105.run train --manifest "$PLAN" --route C --datasets movielens --seeds 2022 2023 2024 2025 2026 --output "$OUT" --resume
$PY -m experiments.aug_history_20261105.run select --manifest "$PLAN" --route C --split validation --output "$OUT"
$PY -m experiments.aug_history_20261105.run decide --manifest "$PLAN" --route C --basis validation --output "$OUT"
$PY -m experiments.aug_history_20261105.run evaluate --manifest "$PLAN" --route C --frozen "$OUT/C/selection_frozen.json" --output "$OUT"
$PY -m experiments.aug_history_20261105.run report --manifest "$PLAN" --route C --output "$OUT"
$PY -m experiments.aug_history_20261105.run continue --manifest "$PLAN" --route C --decision "$OUT/C/DECISION.json" --output "$OUT" --resume
```

train内部完成四阶段全部30轨迹；continue自动执行完成时train幂等跳过。落盘仅源码、hash、标量/曲线/解析/冻结/决策/日志与报告；teacher/optimizer/A/root/R及端点参数保留RAM。运行到定义矩阵结束后STOP_AND_REPORT，回填实际结果与新阶段身份；缺失阶段按具体工程/成本阻塞记录，不写性能失败。

不运行KR C、不新增C0训练或B+C、不扩γ/K、不重做父训练；本次新增工作只为完成ML冻结的C_FULL及其源码/结果交付。
