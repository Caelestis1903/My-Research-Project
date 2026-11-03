# 设计到实现映射

冻结来源：`research_plans/26-11-03/03_ALGORITHM_SPEC.md`、04/05与本目录DESIGN_SPECIFICATION。代码目录为 `experiments/mdmt_history_information_20261103/`。

| 合同 | 实现 |
| --- | --- |
| StepContext、MixBatch、随机key、自然anchor、donor三级回退 | `mixing.py`：generator、build_mix_batches；CPU Generator以冻结SHA256 key播种，两个任务共享anchor，U从第二级开始。 |
| 虚拟嵌入、domain字段不变、donor梯度与旧点 | `mixing.py`：forward_mixed、model_forward；field_slices定位，finally移除hook，strict functional_call传完整参数与buffer。 |
| L0、自然域任务权重L_M、源A1 | `objectives.py`：real_loss、mixed_objective、train_step；A1直接调用原字节source_step并固定EMA02V2 key。 |
| H两份probe、H_REAL/H_U/C_MIX | `objectives.py`：collect_history_probes；只求S，当前.grad保留，domain轮转在train_step。 |
| H的D度量、最多四法向量、KKT、H_NORM | `optimizers.py`：project_history、HistoryOptimizer.step；FP64小Gram/q、活跃集合枚举、失败终止。 |
| T全参数旧点、全S裁剪、质量因子、T_NORM | `objectives.py`及`optimizers.py`：preupdate快照驻opt.previous_state，transport_correction返回(GradList,标量诊断)。 |
| 单一Adam时钟、L2一次、互斥参数组 | `optimizers.py`：HistoryOptimizer；共享参数按冻结名字，普通参数用torch Adam，κ0直接普通Adam。 |
| RAM state_dict/load接口 | `HistoryOptimizer.state_dict/load_state_dict`加原cpu_copy；worker快照同时包含模型/EMA、RNG、loader与选点。 |
| 新协议与科学身份 | `protocol.py`：load_protocol、config、job；显式当前queue+vendor resolved_protocol，source/data/model/随机定义进入key，stage不进入key。 |
| 完整真实batch接线与成本 | `stage0.py`：wiring及stage0；每数据A0f/M/H_MIX/TM各2预热+3计时，超过4倍标具体方案COST_BLOCKED。 |
| 调度、历史复用、阶段门、冻结、统一test、表格 | `run.py`、`guardian.py`、`decisions.py`、`reporting.py`，见CODER_ORCHESTRATION_REPORT。 |

源文件以原字节保存在vendor/v3，37个文件清单见SOURCE_MANIFEST；只保留模型、数据、A1目标与导入依赖、协议、指标、guardian，未复制历史训练代码入口或测试。旧包和旧报告未修改。

CPU小测试是数学和接线检查；真实数据S0与正式50epoch队列均在独立审查通过后由runner执行。
