# 02 任务A：发表模型统一数据与Validation宏AUC选点

本文件是实现规格，不是训练结果。继承main已制定的`26-11-03_v2/PHASE2_PUBLISHED_MODEL_PROTOCOL.md`，不在本晚重新选择训练预算或调参。

## 1. 精确修改边界

所有新增代码在`experiments/overnight_ab_20261103/`。下表“原位置”均为01登记的固定commit，函数名是定位锚点；不要直接编辑旧文件。

| 新文件/接口 | 原位置/需要移植的部分 | 确定修改 |
|---|---|---|
| `data_bridge.py: load_bundle(dataset)` | history.protocol.load_dataset/load_protocol | 只返回完整自然人口；复用已有哈希检查，不调用旧baseline.Data/prepare_owned |
| `data_bridge.py: schema_of(bundle)` | audited_data.FeatureSchema.as_dict | 直接`bundle.features.as_dict()`；不重编码、不改变columns顺序 |
| `published_models.py: build_published(method,bundle,seed,cfg)` | baseline.models.build_model的HG/AI两分支 | 直接调用固定adapter，复制字段映射；M3走history.make_model |
| `vendor/sota2526/` | baseline.hypergate.py/autoifs.py/sources.py及其实际同目录依赖 | 保持核心源码和已有bias适配；保存来源清单 |
| `published_training.py: train_epoch(...)` | baseline.training._train_epoch/_stage_rng/_objective | train_ids改为bundle.train范围；保留原生目标、shuffle、阶段编号和优化器组 |
| `published_training.py: create_session/advance_epoch` | _fit_standard、_fit_autoifs | 取消LL早停，固定最终50epoch，严格最大AUC选点，按epoch可恢复 |
| `selection.py: consider/freeze` | 旧训练两处score=macro_logloss，以及history.run.worker的strict-max逻辑 | 单一选点实现；不使用test或LL |
| `evaluation.py: evaluate_endpoint` | history.protocol.metrics/evaluate | HG/AI输出sigmoid后使用相同指标函数；输出规范化平表 |
| `state.py`、`guardian.py` | 旧_checkpoint、pair.guardian的RAM IPC机制 | 不调用_write_checkpoint/torch.save；支持search/reuse状态及量子恢复 |

不得以修改旧`default_config.json`实现新协议：旧config.load_config硬锁T0/F/T0-only及旧protocol，会混淆数据身份。新配置只从本目录protocol.json解析。

## 2. 数据与输入接口

```python
bundle, manifest = history.protocol.load_dataset(dataset, 'original')
schema = bundle.features.as_dict()
# bundle.train_x/val_x/test_x: [N,F] float32
# bundle.train_y/val_y/test_y: [N,2] float32 binary
# bundle.*_row_ids: 原始行号；features含列顺序/类别词表/encoder/scaler hash
```

ML读取`/home_new/xzf/datasets/ml-1m/ml-1m.csv`；KR读取`/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。原始SHA256在protocol.json锁定。位置改变但内容hash不变可作路径别名，不能悄悄换数据。loader已经检查完整train/val/test六数组、人口和预处理身份；不新增数据重划分。

输入差异只在边界：M3模型接收`x.T`，HG/AI适配器接收`x`。域列始终是`domain_indicator`、原值0/1/2。HyperGateAdapter内部将域token映射到1/2/3并保留padding；外部不得再次加1。ML age保持既有连续特征；KR保留原33个非域字段及域列，不改为论文其他数据格式。mask为与y同shape的全True；当前数据没有通过删行掩盖缺失标签的许可。

训练阶段只向训练函数交付TrainView（train_x/y/row_ids/schema）；ValidationView独立；TestView仅由冻结后的Evaluator打开。原loader可计算test数组hash，但训练函数不能接收test指标或用test反馈修改配置。

## 3. 模型与优化配置（固定，不再由Codex挑选）

| 参数 | M3OE_FIXED1 online | HyperGate | AutoIFS |
|---|---|---|---|
| embedding | 16 | 16 | 16 |
| 结构 | 4共享专家，FCN[512,256,256,64]，融合全1 | 4专家；expert[256,256]、gate[256]、tower[256,128] | MLP[1024,512,256]，LoRA rank ML2/KR16 |
| batch | 40960 | 1024 | 4096 |
| 最终阶段 | 50epoch | 50epoch | reuse50epoch，另search10 |
| lr | .01 | .001 | search .0003；gate为search×.01；reuse .0001 |
| coupled wd | ML1e−4/KR1e−5 | 0 | search base3e−5/gate0；reuse3e−4 |
| Adam | β=(.9,.999),eps1e−8 | 同β/eps，保留原_make_optimizer语义 | 同β/eps，search两组/ reuse新base optimizer |
| 主端点 | online | online | online hard-ticket reuse |
| 选点 | epoch1–50 | epoch1–50 | reuse1–50 |
| 早停 | 无 | 无 | 无 |
| 精度 | FP32，无AMP/TF32 | 同左 | 同左 |

HyperGate额外固定：relu、dropout=.3、batch_norm=False；embedding/expert_bias/fusion三门全开；gate_input_type=augmented_embedding，aug_dim16、aug_dims[64]、aug_dropout0；lambda_domain=lambda_task=.1。保留SENet、专家偏置、增强网络、detach、辅助loss及Xavier初始化。当前配置继续标记`UNVERIFIED_PROPOSAL`，不要写成已核验作者原始超参。

AutoIFS：dropout0、batch_norm=False、gate_penalties=[0,0]；上游选择器、四条信息路径及已有四处输出bias机械修正保持不变。不得删去search或把多优化器改为单optimizer来强行等预算。

M3使用当前固定融合适配而不是原论文全部AutoML/DARTS。正式论文模型栏必须写出适配身份。M3若复用A0在线曲线，不新增EMA到正式模型定义；并行收集过非反馈EMA的事实另列成本备注。

固定50epoch的目的为消除旧LL早停对AUC候选窗口的截断；这不是宣称所有模型更新次数相同。报告epochs、真实optimizer调用、标签暴露、GPU时间/显存，而非仅用epoch数宣称等预算。今晚不搜索学习率/正则/epoch上限；不能看到test后延长某一个模型。

## 4. 原生目标与shuffle

HG/AI直接调用`adapter.objective(x,y,mask)`，保留返回loss及各辅助分量，不能用外部统一BCE替换原生目标。backward一次；HG一个Adam，AI search的base/gate各更新一次且参数集合互斥完整覆盖；reuse只有可训练base。

保持baseline._stage_rng的确定性规则：NumPy PCG64 SeedSequence `[seed,dataset_index,method_index,formal_phase_index,epoch_argument]`，方法序为M3/HG/AI，dataset为ML/KR，formal_phase_index=1。HG epoch_argument=1..50；Auto search=0..9；reuse=101..150。自然无放回permutation，尾批保留。不要把新运行名称或输出路径加入随机流。

M3/B继续原history worker的torch.Generator(seed+11000)顺序、原初始化及context；不得为“统一shuffle”重写它而破坏已验证复用。不同发表架构保留其登记训练机制；B四格除组件开关外严格一致。

## 5. AutoIFS逐阶段唯一实现顺序

1. 初始化模型，确认base/gate optimizer_groups不重叠且覆盖全部requires_grad参数，建两个search Adam。
2. search零基epoch0温度1；之后每epoch乘`100**(1/9)`，最终温度100。
3. **进入零基epoch9之前**保存rewind模型（即完成前9个search epoch的状态）；继续完整第10个search epoch。不能在最终search后才取rewind，也不能改成初始化权重。
4. 取最终search的`export_search_state()`；重新build一个AutoIFS模型，load rewind，再`import_selector_state(final_selector)`，包括gate embedding。设置ticket=True,temp=100，freeze_gate。
5. 验证selector transplant一致；gate不可训练，所有base按原实现可训练。创建**全新的**reuse Adam，不复用search的一阶矩、二阶矩或step。
6. reuse1–50全部训练；每轮按同一validation宏AUC选点。search验证只作为日志，不能选择另一个search架构或取search checkpoint作主端点。
7. 采集reuse E10/E50为诊断，附带search10成本；不能把search epoch10冒充最终训练E10。

`create_session(job,bundle,device) -> Session`与`advance_epoch(session) -> EpochRecord`是共同接口。Session至少含method、stage、next_stage_epoch、completed_search、completed_final、commit_index、model、optimizers、RNG/sampler、best_score/selected及E10快照。AutoIFS的ticket/temp和requires_grad阶段状态不是纯state_dict就能恢复，必须单列。

search10到reuse_ready的转换作为最后一个search epoch的原子RAM提交的一部分：若转换前进程中断，从search9完整状态重放search10及转换；若已提交reuse_ready，恢复时不得再次rewind、再次随机初始化或继承search optimizer。commit_index跨search/reuse单调递增，共60；不可沿用旧guardian.complete的epoch==50判断。

## 6. AUC选点与评价

`selection.consider(stage,epoch,validation,model_state,model_meta)`：仅合法最终阶段；六格AUC全部有限才允许产生候选；macro为六格等权。best初始化−inf；仅`score > best`替换，精确相等保留最早，不使用1e−12人为改变最大值。LL记录但不读入比较器。

每个epoch记录stage/epoch、六格V指标、macro、model_hash、train_objective分量、实际steps/exposures/成本。选择状态中保存完整model buffers、stage metadata；哈希基于真实张量，不只存配置hash。

`evaluation.py`为M3/B复用history.protocol.evaluate；HG/AI使用只做布局变换和一次sigmoid的wrapper，使history metrics接收同语义概率。HG eval_batch=512，AI=4096；M3/B=40960。评估期间eval模式、无增强、no_grad，之后还原training状态。sigmoid在模型原FP32输出上执行，然后按原评价函数转float64统计；不得为其中一个模型改用raw-logit AUC以改变饱和tie。LL使用同一个source metrics的原始未缩放定义，不使用训练总loss/辅助loss。

输出六格AUC/LL/count/positive/negative、macro、sample_weighted_cell、各domain/task、pooled_task_mean_auc/pooled_logloss。若某格单类导致AUC未定义，保留null，主macro也null，不nanmean成“五格宏”。这是评价工程阻塞，不是算法性能阴性。原数据协议应有完整六格；不得修改人口来修补。

只对冻结的同一checkpoint计算V/Test。E10/E50只描述，不能据其test结果覆盖selected。旧raw-logit/softplus指标仍留旧协议历史附录。

## 7. 精确调用与复现

实现后可单独检查/恢复一个job，但正式矩阵由共享队列调度；不要手工同时跑下列训练命令抢占GPU。

```sh
# 先通过prepare，job-id由plan.py固定给出
$PY -m experiments.overnight_ab_20261103.run run-job \
  --attempt "$ATTEMPT" --job-id kuairand.HYPERGATE.s2022 --device cuda:0 --quantum-epochs 1
$PY -m experiments.overnight_ab_20261103.run run-job \
  --attempt "$ATTEMPT" --job-id movielens.AUTOIFS.s2022 --device cuda:0 --quantum-epochs 1
$PY -m experiments.overnight_ab_20261103.run report --attempt "$ATTEMPT"
```

`run-job`同样必须获得coordinator认领、GPU锁和工程PASS，不提供绕过这些检查的参数。正式五seed矩阵为两个dataset×三个发表模型×2022–2026，M3 job是A0轨迹online端点的alias，不额外生成十条训练。

复现优先级：完整签名标量端点直接接入；仍有RAM selected状态则重复评价；无状态只可按相同数据/代码/配置/RNG完整重放该seed，记录新attempt，不增加n。记录Python/PyTorch/CUDA/GPU、依赖版本和全部源码hash；库版本不一致时报告复现环境差异，不擅自修改算法。

## 8. 必须通过的工程验收（不是科研筛选）

schema值/列顺序保持、domain只映射一次；两个adapter各真实一batch forward/backward且loss有限；原AutoIFS参数归属/rewind/transplant/hard-ticket已有测试复用，仅新增接口与阶段恢复检查；合成V曲线验证argmax/earliest/search排除；同model hash对应V/Test；训练函数未读test；无落盘模型状态。A中某个adapter阻塞不阻断B或另一个已通过模型。
